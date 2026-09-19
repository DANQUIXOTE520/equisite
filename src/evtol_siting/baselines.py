"""经典单目标选址模型基线。

论文价值
--------
本研究的主张是"四阶段集成 + 多目标"优于传统单目标选址。这个主张**必须
用实验证明**，否则只是修辞。本模块实现四个经典模型，在**同一候选集、
同一需求、同一接驳时间矩阵**下求解，与 NSGA-II 的帕累托前沿对比：

============  ==================  ============================
模型           优化维度            对应申报书中的目标
============  ==================  ============================
p-中位 (p-median)  效率            总接驳时间最小
p-中心 (p-center)  公平(minimax)   最差体验最优
最大覆盖 (MCLP)    覆盖            服务人数最多
集合覆盖 (SCP)     成本            起降场总数/成本最小
============  ==================  ============================

预期结论：每个单目标模型的**最优解在其余三个目标上表现很差**，且都位于
NSGA-II 前沿的支配域内（即被前沿中的某个解全面优于）。这为"多目标必要性"
提供了可量化的证据——这是论文的核心论证链条。

所有模型均用 PuLP + **HiGHS** 精确求解，保证基线不是"因为求解器差而表现差"。
（原先用 CBC：同一算例下阶段一的 MCLP 要 67–101 s，HiGHS 约 15 s，而 p-中位的形式
下 CBC 连可行解都拿不到。求解器强度在这个规模上直接决定能否出结果。）
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 算例指纹缓存与并行求解
# ---------------------------------------------------------------------------
#
# 基线解**只依赖算例**：候选集、需求向量、阻抗矩阵、p 取值与求解时限。
# 然而 ``export_3d.py`` 与 ``08_multirun.py`` 会在**完全相同的算例**上各求解
# 一遍（≤13 个 MILP × 2），其中单次求解实测可达 150 s 以上。因此按算例
# 指纹缓存到磁盘：第二次直接命中，不再重解。
#
# ⚠ 任何会改变解的东西（求解时限、目标表述、约束、变量）一旦改动，
#   必须同时抬高 ``_CACHE_VERSION``，否则会静默读到过期结果。
#
# 并行：CBC 是单线程子进程，且 pulp 每次 ``actualSolve`` 都新建独立临时目录
# （``create_tmp_files``），因此同名 LpProblem 并发求解是安全的——已用
# 6 个同名问题并发/串行对照实测，最优值完全一致。

# v4：p-中位改为**不截断邻域 + x 连续**（见 solve_p_median 的说明）。
# v5：求解时限 300 -> 1200 s。
# v6：p-中位改为**两阶段**（最大覆盖 -> 定服务量下最小化总时间），消掉大 M。
# v7：p-中位真正落实两阶段，并**修掉"函数没有 return"**——v6 的 solve_p_median
#     执行到底后直接落出函数体，返回 None，上游 `b.feasible` 抛
#     `'NoneType' object has no attribute 'feasible'`，导致 baseline_vs_front.csv
#     整张表写不出来。v6 缓存里的 p-中位结果全部无效，必须重解。
# v8：p-中位改为**显式未服务变量 u 的软分配**形式（见 solve_p_median），
#     并改用 HiGHS 求解。v7 的两阶段锚定形式在大算例上连可行解都拿不到，
#     其缓存（若有）同样不可用。
_CACHE_VERSION = 11

# **临时性**状态：换个时限或环境就可能解出来，不能当成最终结果固化/跳过。
# 其余状态（Optimal / Feasible / Infeasible）都是确定性结论，可以缓存也可以续算。
_TRANSIENT = frozenset({"Failed", "Not Solved", "Undefined"})


def _instance_fingerprint(demand, candidates, access_time_s, p_values) -> str:
    """算例指纹：候选、需求、阻抗矩阵与 p 取值共同决定基线解。"""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(f"v{_CACHE_VERSION}".encode())
    h.update(np.ascontiguousarray(candidates["cand_id"].to_numpy()).tobytes())
    h.update(np.ascontiguousarray(
        candidates["cost"].to_numpy(dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(
        demand["demand"].to_numpy(dtype=np.float64)).tobytes())
    at = np.asarray(access_time_s, dtype=np.float64)
    h.update(np.ascontiguousarray(np.nan_to_num(at, nan=-1.0)).tobytes())
    h.update(repr(sorted(int(p) for p in p_values)).encode())
    return h.hexdigest()


def _cache_file(cfg, fingerprint: str) -> Path | None:
    try:
        d = Path(cfg.dir("output.data_dir")).parent / "cache"
    except Exception:
        return None
    d.mkdir(parents=True, exist_ok=True)
    return d / f"baselines_{fingerprint}.json"


def _load_cached(path: Path | None) -> tuple[list["BaselineSolution"] | None, bool]:
    """读基线缓存，返回 ``(解列表, 是否完整)``。

    ``完整`` 标记用于**断点续算**：p-中位单点可解 30 分钟以上，一次 13 个模型的
    运行很容易被中断（实测发生过一次——系统内存不足被杀）。每个解一解出来就
    落盘并标记 ``complete=False``，下次运行只补解缺的那几个，而不是推倒重来。
    """
    if path is None or not path.exists():
        return None, False
    try:
        blob = json.loads(path.read_text("utf-8"))
        if blob.get("version") != _CACHE_VERSION:
            return None, False
        sols = [
            BaselineSolution(
                name=s["name"], status=s["status"],
                selected=[int(i) for i in s["selected"]],
                objective=float(s["objective"]), n_sites=int(s["n_sites"]),
                solve_time_s=float(s.get("solve_time_s", 0.0)),
                feasible=bool(s.get("feasible", True)),
                gap=(None if s.get("gap") is None else float(s["gap"])),
            )
            for s in blob["solutions"]
        ]
        # v8 之前的缓存没有 complete 字段，但那时候只有全部成功才落盘，
        # 所以缺省视为完整。
        return (sols or None), bool(blob.get("complete", True))
    except Exception as exc:
        logger.warning("基线缓存读取失败（将重解）: %s", str(exc)[:140])
        return None, False


def _save_cached(path: Path | None, sols: list["BaselineSolution"],
                 complete: bool = True) -> None:
    if path is None or not sols:
        return
    try:
        path.write_text(json.dumps({
            "version": _CACHE_VERSION,
            "complete": bool(complete),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_solutions": len(sols),
            "solutions": [
                {"name": s.name, "status": s.status, "selected": list(s.selected),
                 "objective": s.objective, "n_sites": s.n_sites,
                 "solve_time_s": s.solve_time_s, "feasible": s.feasible,
                 "gap": s.gap}
                for s in sols
            ],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("基线结果已缓存: %s", path.name)
    except Exception as exc:
        logger.warning("基线缓存写入失败: %s", str(exc)[:140])


@dataclass
class BaselineSolution:
    """基线模型求解结果。"""
    name: str
    status: str
    selected: list[int]
    objective: float
    n_sites: int
    solve_time_s: float = 0.0
    feasible: bool = True
    # 求解器**实际达到**的相对 MIP 间隙（0 = 证明最优，None = 未记录/纯 LP）。
    # 论文应报告这个数，而不是笼统地写"求解至最优"——HiGHS 的默认
    # mip_rel_gap 就是 1e-4，绝大多数"最优"其实带 0.01% 的间隙。
    gap: float | None = None


# 基线求解时限。原为 300 s，实测**不够**：p-中位在正确的（不截断）形式下
# 需要约 550 s 才收敛，300 s 会返回截断的当前最好解——而 CBC 的 timeLimit
# 并不可靠（实测 300 s 的设定能跑出 750 s），pulp 又会把截断解标成 Optimal，
# 于是"最优"与"最好解"混为一谈。抬高到 1200 s 留足余量；其余基线本就秒级
# 收敛，不受影响。
DEFAULT_TIME_LIMIT_S = 1200


def _solver(time_limit_s: int = DEFAULT_TIME_LIMIT_S, gap_rel: float | None = None):
    """MILP 求解器。优先用 HiGHS，不可用时回退 CBC。

    ``gap_rel`` 给定相对间隙时，求解器会在**有证书**的情况下提前停止
    （当前最好解与界之差 ≤ gap_rel × |界|），而不是耗满时限后交出一个无证书的
    截断解。对大算例这是唯一能拿到"可辩护的最优性"的途径。

    为什么换成 HiGHS：同一算例下阶段一的 MCLP，CBC 需 67–101 s，HiGHS 约 15 s；
    基线模型规模在十几万至几十万变量，求解器强度直接决定能否出结果。
    HiGHS 也是可引用的开源求解器（Huangfu & Hall, 2018），论文里比 CBC 更站得住。
    """
    import pulp

    if _use_highs():
        try:
            from pulp.apis.highs_api import HiGHS

            kw = {"msg": False, "timeLimit": int(time_limit_s)}
            if gap_rel:
                kw["gapRel"] = float(gap_rel)
                kw["gapAbs"] = 0.0
            return HiGHS(**kw)
        except Exception as exc:                       # 装了但用不了 -> 回退
            logger.warning("HiGHS 不可用，回退 CBC: %s", str(exc)[:120])

    kw = {"msg": 0, "timeLimit": int(time_limit_s)}
    if gap_rel:
        kw["gapRel"] = float(gap_rel)
        kw["gapAbs"] = 0.0
    return pulp.PULP_CBC_CMD(**kw)


_WARNED_NO_HIGHS = False


def _use_highs() -> bool:
    """是否使用 HiGHS。可由 ``EVTOL_BASELINE_SOLVER=cbc`` 关掉（用于对照复现）。

    ⚠ 退回 CBC 必须**出声**。之前这里是静默 `return False`：没装 highspy 的环境
    会一声不响地改用 CBC，而 p-中位在 CBC 下**根本解不出来**，于是 §5.7.1/§5.8 的
    p-中位各行凭空消失，复现者只会看到"结果和论文不一样"却不知为什么。
    """
    global _WARNED_NO_HIGHS
    if os.environ.get("EVTOL_BASELINE_SOLVER", "").strip().lower() == "cbc":
        return False
    try:
        import highspy  # noqa: F401
        from pulp.apis.highs_api import HiGHS  # noqa: F401
        return True
    except Exception as exc:
        if not _WARNED_NO_HIGHS:
            _WARNED_NO_HIGHS = True
            logger.warning(
                "未安装 highspy（或 pulp 过旧），**回退 CBC**：%s。"
                "p-中位在大算例下用 CBC 解不出来，基线结果将不完整、与论文不符。"
                "请 `pip install -r requirements.txt`（highspy 是必需依赖）。",
                str(exc)[:120])
        return False


# 求解状态判据**只有一份实现**，见 solver_status.py。这里保留同名别名是为了
# 不打断本模块内的调用点，但语义完全一致——审计发现判据散落多处时，
# 修了一处漏三处，而漏掉的恰好是正文引用的那几处。
from .solver_status import proven_optimal as _proven_optimal          # noqa: E402
from .solver_status import solver_gap as _solver_gap                  # noqa: E402
from .solver_status import describe as _status_describe               # noqa: E402


def _extract(prob, y, cand, name, t0, status) -> BaselineSolution:
    import pulp

    if pulp.LpStatus[prob.status] != "Optimal":
        logger.warning("[%s] 求解状态 %s", name, pulp.LpStatus[prob.status])
        return BaselineSolution(name, pulp.LpStatus[prob.status], [], np.inf, 0,
                                time.time() - t0, feasible=False)
    sel = [j for j in range(len(y)) if y[j].value() is not None and y[j].value() > 0.5]
    obj = float(pulp.value(prob.objective))
    # 若目标里含未分配罚项，报告时应还原为真实的总接驳时间，
    # 否则数值带上了一个巨大的常数偏移，无法与其他模型比较。
    offset = getattr(prob, "_penalty_offset", None)
    if offset is not None:
        obj = obj + offset
    # "Optimal" 只在**证明**了最优时才写；达到间隙/时限而停下但有可行解的记为
    # "Feasible"，下游与论文据此区分"最优解"与"当前最好解"。
    status = "Optimal" if _proven_optimal(prob) else "Feasible"
    if status == "Optimal":
        logger.info("[%s] 已证明最优", name)
    else:
        g = _solver_gap(prob)
        logger.warning("[%s] 未证明最优（%.2f%% 相对间隙 / 或达到时限），"
                       "返回当前最好解——论文不得称其为「最优」",
                       name, 100.0 * g if g is not None else float("nan"))
    return BaselineSolution(
        name, status, cand["cand_id"].iloc[sel].tolist(),
        obj, len(sel), time.time() - t0, gap=_solver_gap(prob),
    )


def _core_mask(demand_vals: np.ndarray, cfg=None) -> np.ndarray:
    """核心需求单元掩码。口径与 stage3_ip 完全一致（同一函数）。

    覆盖类模型（集合覆盖、p-中心）的约束只施加于核心单元：否则需求≈0 的
    边缘单元也会要求被覆盖，站数显著虚高。阈值取
    ``stage3_ip.min_core_demand``（默认 0.60，即需求最高 40% 的单元）。
    """
    from .stage3_ip import core_demand_mask

    thr = 0.6
    if cfg is not None:
        try:
            thr = float(cfg.get("stage3_ip.min_core_demand", 0.6))
        except Exception:
            thr = 0.6
    return core_demand_mask(demand_vals, thr)


# ---------------------------------------------------------------------------
# p-中位：最小化需求加权的总接驳时间（效率）
# ---------------------------------------------------------------------------

def solve_p_median(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    p: int,
    cfg,
    name: str = "p_median",
) -> BaselineSolution:
    """p-中位模型。

    .. math::

        \\min \\sum_i \\sum_j d_i \\, t_{ij} \\, x_{ij}
        \\quad \\text{s.t.} \\quad
        \\sum_j y_j = p,\\; \\sum_j x_{ij} = 1,\\; x_{ij} \\le y_j

    即"总出行时间最小"，只关心效率，完全不顾及公平与成本差异。
    """
    import pulp

    t0 = time.time()
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)

    d = dem["demand"].to_numpy(dtype=np.float64)
    # ⚠️ 不可达一律保留为 inf，不要替换成"一个大数"。
    # 早期写法 `np.where(isfinite, x, 1e9)/60` 把不可达变成 1.67e7 分钟，
    # 而下游的可达性判据写成 `at < 1e8`——那个阈值是按**秒**的量级定的，
    # 于是所有不可达配对都被误判为可达，并带着 1.67e7 分钟的"行程时间"
    # 进入目标函数（实测 p-中位目标值虚高到 1.02e10）。判据与量纲必须一致。
    at = np.where(np.isfinite(access_time_s), access_time_s, np.inf) / 60.0

    # ------------------------------------------------------------------
    # 变量规模：完整模型的 n_dem × n_cand（成都约 2519 × 1324 ≈ 333 万）无法
    # 直接交给 CBC。但**不需要人为截断**——接驳时间预算已经把每个需求单元的
    # 可达候选天然限制住了，可达集本身就是有界的。
    # ------------------------------------------------------------------
    # ⚠ 邻域**不再按固定 k 截断**，改为直接用**可达集**。
    #
    # 原来取 ``evaluation.p_median_k_nearest``（默认 25）。实测该值远小于
    # 实际可达数：成都算例下每个需求单元在 15 min 预算内**中位有 127 个**
    # 可达候选（p90 = 232，按需求加权平均 112.3），k=25 一次性截掉了
    # **78 % 的可达配对**。
    #
    # 后果不是"模型变小"，而是**模型被系统性削弱**：选中站若排在某个单元的
    # 第 26 名之后（哪怕它就在 10 分钟车程内），模型收不到任何收益，于是目标
    # 函数看到的是一个失真的可服务性景观。而评估指标用的是**完整矩阵**，
    # 两者的落差表现为"模型认为很好的解，实测覆盖率只有 32 %"——且不同运行
    # 间极不稳定（同一算例另一次给出 99.8 %）。
    #
    # 不必截断：接驳时间预算已经把每个单元的可达集天然限制住了（成都最大
    # 272 个），所以"可达集"本身就是有界的，不需要再人为加一个 k。
    # 若确有需要，把 ``p_median_k_nearest`` 设为一个足够大的数即可回退到
    # 旧行为（设为 <=0 或留空表示不截断）。
    k_cfg = cfg.get("evaluation.p_median_k_nearest", None)
    k = n_cand if k_cfg in (None, 0) else max(1, min(int(k_cfg), n_cand))

    order = np.argsort(at, axis=1, kind="stable")[:, :k]      # (n_dem, <=k)
    # 过滤掉完全不可达（1e9）的邻居
    reachable = np.isfinite(at[np.arange(n_dem)[:, None], order])

    n_neigh = [np.flatnonzero(reachable[i]) for i in range(n_dem)]
    neigh = [[int(j) for j in order[i][n_neigh[i]]] for i in range(n_dem)]
    p_eff = min(p, n_cand)

    # ------------------------------------------------------------------
    # 未服务单元怎么表达——三个失败过的写法都记在这里，避免再走回去
    #
    # ① 硬分配 `Σ_{j∈N(i)} x_ij == 1`
    #    p 远小于候选总数时（p=8 而候选 1324 个），选中的 8 个站很可能一个都
    #    不落在某个单元的可达集里，"必须分配"于是无解，整个模型不可行
    #    （实测 p=10、p=11 全部 Infeasible）。
    #
    # ② 大 M 罚项  min Σ (d_i t_ij − M) x_ij，配 `Σ_j x_ij <= 1`
    #    允许不分配，用 −M 奖励"多分配"。可行，但 **M 毁掉 LP 松弛的界**：每项
    #    都带 −M，松弛解靠多开 x 即可无限压低目标，界与真值差得极远，分支定界
    #    失效。实测 p=8 跑满 2 160 s 仍 `Not Solved`，p=10 跑到 2 977 s。
    #
    # ③ 两阶段锚定  min Σ d_i t_ij x_ij  s.t.  Σ d_i Σ_j x_ij >= S*
    #    （S* = 最大可服务量）。界干净，**但可行域窄到不可行**：要求 8 个站服务
    #    95.28 % 的需求，等价于要求站点布局已接近最优，连一个可行整数解都难找。
    #    实测 CBC 跑 1 762 s、HiGHS 跑 30 min，**都未给出任何可行解**。
    #
    # 正解是**显式的未服务变量 u_i**：
    #
    #     min Σ d_i t_ij x_ij + P Σ d_i u_i
    #     s.t.  Σ_{j∈N(i)} x_ij + u_i = 1,   x_ij <= y_j,   Σ y_j = p
    #           x_ij, u_i ∈ [0, 1]
    #
    # 取 P > max(t_ij)，则"能服务就一定服务"成立：服务一个单位省下 P·d_i，
    # 而最多只花 t_ij·d_i < P·d_i。相比 ② 的三点关键区别：
    #   · u_i 由**自身**上界 1 限制，而非被 Σ_j y_j 限制，所以 LP 松弛里
    #     "免费服务"的额度不会随候选数膨胀；
    #   · 罚项 P 与目标里的 t **同量级**（约 20 分钟 vs 0–15 分钟），不是 ② 里
    #     那种碾压一切的巨数，界因此仍然可用；
    #   · **x ≡ 0、u ≡ 1 天然可行**，求解器立刻拿到可行解，不必先猜中一个接近
    #     最优的站点布局——正是 ③ 卡死的地方。
    #
    # 语义与 ②③ 完全一致（先最大化被服务需求，再最小化总接驳时间），
    # 只是换了一个等价且数值上可解的表述。
    # ------------------------------------------------------------------
    at_fin = at[np.isfinite(at)]
    # P 必须严格大于任何可达行程时间，否则"不服务"会变成更划算的选择。
    P = float(at_fin.max()) * 1.01 + 1e-6 if at_fin.size else 1.0

    prob = pulp.LpProblem("p_median", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_cand)]
    x_vars: dict[tuple[int, int], pulp.LpVariable] = {}
    u_vars: dict[int, pulp.LpVariable] = {}
    obj_terms = []
    for i in range(n_dem):
        # 完全不可达的单元：任何方案都服务不到，u 恒为 1 是个常数，
        # 既不进变量也不进目标（否则只是给目标加一个常数）。
        if not neigh[i]:
            continue
        u = pulp.LpVariable(f"u_{i}", lowBound=0.0, upBound=1.0)
        u_vars[i] = u
        obj_terms.append(P * d[i] * u)
        for j in neigh[i]:
            # x 用**连续**变量（0 ≤ x ≤ 1），只留 y 是 0-1。给定已开站集合时，
            # 每单元的最优分配就是"分给最近的已开站"，关于 x 的 LP 本身取整数解，
            # 不损失精确性；声明成 0-1 会让变量数膨胀到十几万个，CBC 搜不动
            # （实测同一算例由此从 99.4 % 覆盖率掉到 32 %）。
            v = pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, upBound=1.0)
            x_vars[(i, j)] = v
            obj_terms.append(d[i] * at[i, j] * v)

    prob += pulp.lpSum(obj_terms)
    prob += pulp.lpSum(y) == p_eff, "exactly_p"
    for i in range(n_dem):
        if not neigh[i]:
            continue
        prob += (pulp.lpSum(x_vars[(i, j)] for j in neigh[i]) + u_vars[i] == 1,
                 f"assign_{i}")
        for j in neigh[i]:
            prob += x_vars[(i, j)] <= y[j], f"link_{i}_{j}"

    # 变量规模达十几万至几十万个，小 p 时最难。给一个较宽的时限，并允许设一个
    # 相对间隙：达到间隙即停，交出的解**带最优性证书**（与界的相对差 ≤ 该值），
    # 比"耗满时限再返回一个无证书的最好解"更站得住，也比把截断解称作"最优"诚实。
    # 两者都可在配置里调（evaluation.p_median_*）。
    tl, gap = DEFAULT_TIME_LIMIT_S, None
    try:
        tl = int(cfg.get("evaluation.p_median_time_limit_s", tl) or tl)
        g = cfg.get("evaluation.p_median_gap_rel", None)
        gap = float(g) if g else None
    except Exception:
        pass

    logger.info("[%s] p=%d: %d 个需求单元 × %d 可达候选 = %d 个分配变量 + "
                "%d 个未服务变量（P=%.2f），时限 %ds，相对间隙 %s",
                name, p, n_dem, k, len(x_vars), len(u_vars), P, tl,
                gap or "无（求解到证明最优）")
    prob.solve(_solver(tl, gap))

    # 目标里含未服务罚项 P·Σ d_i u_i，报告时必须扣掉，否则数值带一个常数偏移，
    # 无法与 p-中心 / 最大覆盖等模型的"总接驳时间"比较。u 只是为了让模型可行而
    # 引入的松弛变量，不属于 p-中位的效率目标。
    try:
        penalty = P * float(sum(d[i] * (u_vars[i].value() or 0.0)
                                for i in u_vars))
    except Exception:
        penalty = 0.0
    prob._penalty_offset = -penalty

    sol = _extract(prob, y, cand, name, t0, prob.status)
    if not sol.feasible:
        # 求解失败时不要伪装成"0 个站、目标 0"——那会在下游被当成一个真实解。
        logger.warning("[%s] p=%d 未解出（状态 %s），耗时 %.1fs",
                       name, p, sol.status, sol.solve_time_s)
        return sol
    n_unserved = sum(1 for v in u_vars.values()
                     if v.value() is not None and v.value() > 0.5)
    logger.info("[%s] p=%d -> %d 个设施，总接驳 %.1f，未服务 %d 个单元，耗时 %.1fs",
                name, p, sol.n_sites, sol.objective, n_unserved, sol.solve_time_s)
    return sol


# ---------------------------------------------------------------------------
# p-中心：最小化最差接驳时间（公平 / minimax）
# ---------------------------------------------------------------------------

def solve_p_center(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    p: int,
    cfg,
    name: str = "p_center",
) -> BaselineSolution:
    """p-中心模型（minimax）。

    最小化**最差**单元到最近已开站的接驳时间 R，站点数不超过 p：

        min R   s.t.  sum_j y_j <= p,
                       sum_{j: t_ij <= R} y_j >= 1   对每个可达单元 i

    R 只可能取有限个接驳时间值，故用**对 R 的二分**求解，每一步是一次集合
    覆盖可行性判定。在最大可达半径下仍无法覆盖全部可达单元、或所需站数超过
    ``p`` 时报告 ``Infeasible``——那是模型的真实结论（实测 p=8 即如此）。
    """
    import pulp

    t0 = time.time()
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)

    at = np.where(np.isfinite(access_time_s), access_time_s, np.inf) / 60.0
    reach = np.isfinite(at)
    # 与 set_covering 同口径：只要求覆盖**核心需求单元**（见那里的说明）。
    coverable = _core_mask(dem["demand"].to_numpy(dtype=np.float64), cfg) & reach.any(axis=1)

    def _infeasible() -> BaselineSolution:
        return BaselineSolution(name, "Infeasible", [], np.inf, 0,
                                time.time() - t0, feasible=False)

    if not coverable.any() or not reach.any():
        return _infeasible()

    finite = np.unique(at[reach])

    # ⚠ 二分法的"最优"不是自动成立的：它依赖**每一步可行性判定都精确**。
    #   而 cover_within 里求解失败（达到时限/间隙）与"真的不可行"都返回 None，
    #   两者被混为一谈。若某一步其实是求解器没跑完，二分就会收敛到一个
    #   **偏大的半径**，而外层却把它标成 "Optimal"——这正是审计里
    #   「硬编码 Optimal」那一类的缺陷。这里把证明状态记下来并向上传递。
    proof = {"all_proven": True}

    def cover_within(radius: float):
        """在半径 radius 内用不超过 p 个站覆盖全部可达单元；不可行返回 None。"""
        within = reach & (at <= radius)
        if not within[coverable].any(axis=1).all():
            return None
        # 目标就是 min Σy，故**不加** `Σy <= p` 这条冗余约束：它不改变最优值，
        # 却让 CBC 难以证明最优——实测加了它返回 18 个站，去掉后同一 R 下正确
        # 返回 11 个（与参考解一致）。站数上限改为解完后判定。
        prob = pulp.LpProblem("p_center_feas", pulp.LpMinimize)
        y = [pulp.LpVariable("y_%d" % j, cat="Binary") for j in range(n_cand)]
        prob += pulp.lpSum(y)
        for i in range(n_dem):
            if not coverable[i]:
                continue
            ne = np.flatnonzero(within[i])
            if ne.size == 0:
                return None
            prob += pulp.lpSum(y[int(j)] for j in ne) >= 1, "cover_%d" % i
        prob.solve(_solver())
        if pulp.LpStatus[prob.status] != "Optimal":
            proof["all_proven"] = False
            return None
        if not _proven_optimal(prob):
            proof["all_proven"] = False
        sel = [j for j in range(n_cand)
               if y[j].value() is not None and y[j].value() > 0.5]
        # 最少站数超过 p —— 该 R 在给定站数上限下不可行
        return sel if len(sel) <= min(p, n_cand) else None

    lo, hi, best, best_r = 0, int(finite.size) - 1, None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        sel = cover_within(float(finite[mid]))
        if sel is not None:
            best, best_r, hi = sel, float(finite[mid]), mid - 1
        else:
            lo = mid + 1

    if best is None:
        logger.warning("[%s] p=%d 在最大可达半径下仍不可行", name, p)
        return _infeasible()
    # 二分解只有在每一步可行性判定都精确时才等于 p-中心最优值；否则只是
    # "找到的最好可行解"。如实区分，不硬编码 "Optimal"。
    if not proof["all_proven"]:
        logger.warning(
            "[%s] p=%d 二分中的可行性子问题**未全部证明最优**（达到时限或间隙），"
            "因此该解标记为 Feasible 而非 Optimal——论文不得称其为精确最优。",
            name, p,
        )
    status = "Optimal" if proof["all_proven"] else "Feasible"
    logger.info("[%s] p=%d -> %d 个设施，最差接驳 %.4f min，耗时 %.1fs",
                name, p, len(best), best_r, time.time() - t0)
    return BaselineSolution(name, status, cand["cand_id"].iloc[best].tolist(),
                            float(best_r), len(best), time.time() - t0)


# ---------------------------------------------------------------------------
# 最大覆盖：用不超过 p 个站最大化被服务的需求（覆盖）
# ---------------------------------------------------------------------------

def solve_max_coverage(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    p: int,
    cfg,
    name: str = "max_coverage",
) -> BaselineSolution:
    """最大覆盖模型 (MCLP)。

        max sum_i d_i z_i   s.t.  z_i <= sum_{j in N(i)} y_j,  sum_j y_j <= p

    用紧凑形式（``z`` 表示"单元是否被服务"，不需要 x 矩阵）。目标值是**被服务
    的需求量**（次/日），上限为总需求。
    """
    import pulp

    t0 = time.time()
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)
    d = dem["demand"].to_numpy(dtype=np.float64)
    reach = np.isfinite(access_time_s)

    prob = pulp.LpProblem("max_coverage", pulp.LpMinimize)
    y = [pulp.LpVariable("y_%d" % j, cat="Binary") for j in range(n_cand)]
    z = [pulp.LpVariable("z_%d" % i, lowBound=0.0, upBound=1.0)
         for i in range(n_dem)]
    prob += pulp.lpSum(y) <= min(p, n_cand), "card"
    for i in range(n_dem):
        ne = np.flatnonzero(reach[i])
        if ne.size == 0:
            prob += z[i] == 0, "unserv_%d" % i
            continue
        prob += z[i] <= pulp.lpSum(y[int(j)] for j in ne), "cover_%d" % i
    prob += pulp.lpSum(-d[i] * z[i] for i in range(n_dem))

    prob.solve(_solver())
    sel = [j for j in range(n_cand)
           if y[j].value() is not None and y[j].value() > 0.5]
    if pulp.LpStatus[prob.status] != "Optimal":
        logger.warning("[%s] 求解状态 %s", name, pulp.LpStatus[prob.status])
        return BaselineSolution(name, pulp.LpStatus[prob.status], [], np.inf, 0,
                                time.time() - t0, feasible=False)
    covered = float(sum(d[i] * (z[i].value() or 0.0) for i in range(n_dem)))
    # ⚠ 这里原本**硬编码** "Optimal"，而 LpStatus 在两个求解器下都不可靠
    #   （CBC 会把"时限用尽但有解"改写成 Optimal，HiGHS 达间隙而停也报 Optimal）。
    #   与 A3 的 `"feasible": True` 是同一类缺陷：状态是写上去的，不是判出来的。
    tag, gap = _status_describe(prob, label=name)
    logger.info("[%s] p=%d -> %d 个设施，覆盖需求 %.1f，耗时 %.1fs",
                name, p, len(sel), covered, time.time() - t0)
    return BaselineSolution(name, tag, cand["cand_id"].iloc[sel].tolist(),
                            covered, len(sel), time.time() - t0,
                            gap=None if gap is None else float(gap))


# ---------------------------------------------------------------------------
# 集合覆盖：用最少的站覆盖全部可达需求单元（成本 / 网络规模）
# ---------------------------------------------------------------------------

def solve_set_covering(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
    name: str = "set_covering",
) -> BaselineSolution:
    """集合覆盖模型 (SCP)。

        min sum_j y_j   s.t.  sum_{j in N(i)} y_j >= 1   对每个可达单元 i

    目标值是站点数。**无可达候选的单元不参与约束**——它们任何方案都服务不到，
    把它们写进约束会让模型恒不可行。
    """
    import pulp

    t0 = time.time()
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)
    d = dem["demand"].to_numpy(dtype=np.float64)
    reach = np.isfinite(access_time_s)

    # ⚠ 覆盖约束只施加于**核心需求单元**，与 stage3_ip 的口径一致
    # （``stage3_ip.min_core_demand``，分位数 0.60 = 需求量最高的 40%）。
    # 若对**全部**可达单元施加约束，需求≈0 的边缘单元也要覆盖，站数会显著虚高
    # （实测 15 站 vs 正确的 10 站）。max_coverage 不受影响，因为零需求单元
    # 本来就不进它的目标——这也是当初只有它一个能对上的原因。
    core = _core_mask(d, cfg)
    prob = pulp.LpProblem("set_covering", pulp.LpMinimize)
    y = [pulp.LpVariable("y_%d" % j, cat="Binary") for j in range(n_cand)]
    prob += pulp.lpSum(y)
    n_con = 0
    for i in range(n_dem):
        if not core[i]:
            continue
        ne = np.flatnonzero(reach[i])
        if ne.size == 0:
            continue
        prob += pulp.lpSum(y[int(j)] for j in ne) >= 1, "cover_%d" % i
        n_con += 1
    prob.solve(_solver())
    sol = _extract(prob, y, cand, name, t0, prob.status)
    logger.info("[%s] -> %d 个设施（覆盖 %d 个可达单元），耗时 %.1fs",
                name, sol.n_sites, n_con, time.time() - t0)
    return sol


# ---------------------------------------------------------------------------
# 编排：求解全部基线（算例指纹缓存 + 并行）
# ---------------------------------------------------------------------------

def run_all_baselines(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
    p_values=(10, 11, 20, 30),
) -> list[BaselineSolution]:
    """在**同一算例**上求解全部单目标基线。

    任务构成：集合覆盖 1 个 + p-中位 / p-中心 / 最大覆盖 各 ``len(p_values)`` 个。
    结果按算例指纹缓存（见本模块开头）；**只有全部成功才落盘**，避免把一次失败
    固化成永久缓存。并行度取 ``baselines.max_workers``（默认 6）。
    """
    # ⚠ 潜在耦合：``BaselineSolution.selected`` 存的是 **cand_id**，而下游
    # （``08_multirun.py`` 的 `x[[int(i) for i in b.selected]] = 1.0`、
    # ``solution_metrics``）按**位置**索引用它。两者只有当 cand_id 恰好是
    # 0..n-1 时才一致。当前两城都满足，但一旦候选集被过滤/重排就会静默算错，
    # 所以这里显式检查一次，宁可吵也不要不声不响地出错。
    _cid = candidates["cand_id"].to_numpy()
    if not np.array_equal(_cid, np.arange(len(_cid))):
        logger.warning("cand_id 不是 0..n-1（前几个：%s）；下游按位置索引 selected，"
                       "覆盖率等指标可能算错。", _cid[:3])

    ps = sorted({int(v) for v in p_values})
    fingerprint = _instance_fingerprint(demand, candidates, access_time_s, ps)
    cache = _cache_file(cfg, fingerprint)

    all_tasks = [("set_covering", 0)]
    for p in ps:
        all_tasks += [("p_median", p), ("p_center", p), ("max_coverage", p)]

    def _task_name(item):
        kind, p = item
        return {"p_median": "p_median_p%d" % p, "p_center": "p_center_p%d" % p,
                "max_coverage": "max_coverage_p%d" % p}.get(kind, kind)

    cached, complete = _load_cached(cache)
    if complete and cached:
        logger.info("基线命中缓存（%s，%d 个解）", fingerprint[:12], len(cached))
        return cached

    # 断点续算：只把**非临时性**的结果算作已完成——临时性失败（超时/异常）
    # 换个时限或环境就可能解出来，必须重试。
    done = {s.name: s for s in (cached or []) if s.status not in _TRANSIENT}
    if done:
        logger.info("基线断点续算：缓存已有 %d/%d 个解，本次只补解其余 %d 个",
                    len(done), len(all_tasks), len(all_tasks) - len(done))
    tasks = [t for t in all_tasks if _task_name(t) not in done]

    def _ordered(mapping):
        return [mapping[n] for n in (_task_name(t) for t in all_tasks)
                if n in mapping]

    def _run(item):
        kind, p = item
        nm = _task_name(item)
        try:
            if kind == "p_median":
                s = solve_p_median(demand, candidates, access_time_s, p, cfg,
                                   name=nm)
            elif kind == "p_center":
                s = solve_p_center(demand, candidates, access_time_s, p, cfg,
                                   name=nm)
            elif kind == "max_coverage":
                s = solve_max_coverage(demand, candidates, access_time_s, p, cfg,
                                       name=nm)
            else:
                s = solve_set_covering(demand, candidates, access_time_s, cfg)
        except Exception as exc:                      # 单点失败不拖垮整组
            logger.warning("[%s] 求解异常: %s", nm, str(exc)[:160])
            s = None
        # ⚠ 任何一条路径返回 None 都会在下游变成
        #   `'NoneType' object has no attribute 'feasible'`，让整张对比表写不出来。
        # 这里统一兜底成"显式不可行"，下游据此跳过该点即可。
        if s is None:
            s = BaselineSolution(nm, "Failed", [], float("inf"), 0, 0.0,
                                 feasible=False)
        return s

    # ⚠ 并行度不能太高。单个 p-中位模型有 32 万个变量，6 个并发求解实测
    # 把机器内存吃光、整个进程被系统杀掉（21:36 那次）。降到 3 个并发，
    # 代价是墙钟时间略长，但换来跑得完。
    workers = int(cfg.get("evaluation.baselines_max_workers", 3) or 3)
    workers = max(1, min(workers, len(tasks)))
    logger.info("求解单目标基线：本次需解 %d/%d 个模型（p 取值 %s），并行度 %d",
                len(tasks), len(all_tasks), ps, workers)

    if tasks:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # ⚠ 必须用 as_completed，**不能用 ex.map**。
            #
            # map 是**按提交顺序**产出结果的：它会把后面的结果全部阻塞，
            # 直到前面那个跑完。而 p_median_p8 排在第二位、要解 30 分钟，
            # 于是排在它后面的结果一个都存不下去——实测出现过"7 个模型已经
            # 解完，缓存里只有 1 个（set_covering）"，增量落盘完全失效，
            # 一被打断照样全部重来。
            #
            # as_completed 才是真正的"解出一个存一个"。
            futs = [ex.submit(_run, t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    s = fut.result()
                except Exception as exc:              # _run 内部已兜底，这里再保一层
                    logger.warning("基线任务异常: %s", str(exc)[:160])
                    continue
                done[s.name] = s
                if not s.feasible:
                    logger.warning("  %s 不可行/未解出（%s）", s.name, s.status)
                _save_cached(cache, _ordered(done), complete=False)

    sols = _ordered(done)
    # ⚠ 缓存判据不能用 `all(s.feasible)`。"不可行"是**确定性结果**（求解器证明了
    # 模型无解，如 p-中心 p=8 在最大可达半径下仍覆盖不全），把它当失败排除，
    # 会让整组基线**永远写不进缓存**，每次全流程都要重解 40 分钟。真正不该固化
    # 的只有**临时性失败**——超时未解出、异常、返回 None。
    bad = [s.name for s in sols if s.status in _TRANSIENT]
    if bad:
        logger.warning("基线有 %d 个临时性失败（%s），本次不标记为完整，留待续算",
                       len(bad), ", ".join(bad))
        _save_cached(cache, sols, complete=False)
    else:
        _save_cached(cache, sols, complete=True)
    return sols
