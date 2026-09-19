"""固定规模的选址方案族，与保规模的方案杂交算子。

设计意图
--------
与 :mod:`stage3_ip` 的"设施数浮动 + 集合覆盖"不同，本模块处理的是另一种
规划问题：

    **在起降场总数固定为 N 的前提下，不同的优化目标会给出不同的布局方案。**

例如给定 N = 50，可以分别求：

* 总接驳时间最小 —— "距离最近"
* 总建设成本最小 —— "成本最低"
* 被服务需求量最大 —— "覆盖最大"
* 最差人群接驳时间最小 —— "体验最均衡"

每个目标解出一套方案，构成一个**方案族**。规划者据此可以说"要最近就选
这套、要最省就选那套"。

两种范式的区别（论文中必须讲清楚）
----------------------------------
固定 N 与浮动 N 回答的是**不同的问题**：

* **浮动 N（集合覆盖）**回答"最少要建几个站才能达到覆盖要求"。
  实测成都在 4 km 机场缓冲下，11 个站即可覆盖 97.1% 的需求。
* **固定 N** 回答"既然规划了 N 个站，它们应该摆在哪里"。
  N 远大于覆盖所需时（如 N = 50），覆盖目标已经饱和，方案之间的
  差异主要体现在**可达性的空间均衡**与**成本**上。

两者不可相互替代，论文中应并列报告。

局限（必须声明）
----------------
方案族给出的是**单目标最优解**。它无法回答"为了把最差人群的接驳时间
再压低一分钟，成本要增加多少"——那需要帕累托前沿。因此方案族的正确用法
是作为多目标优化的**初始种群**与**对照基准**，而非最终推荐。
本模块的 :func:`crossbreed_schemes` 正是为此设计：把方案族杂交，
在保持规模的前提下探索它们之间的折中。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Scheme:
    """一套固定规模的选址方案。"""
    name: str
    objective_key: str
    n_sites: int
    selected: list[int]          # 候选点的 cand_id
    metrics: dict                # 该方案在四个目标上的取值
    solve_time_s: float = 0.0
    status: str = "Optimal"
    #: 求解器**实际达到**的相对 MIP 间隙；读不到为 ``None``。
    #: ``status="Feasible"`` 时论文必须引用这个数，写成"求解至相对间隙 X%"，
    #: 而不是"最优"——两者在 CBC 下仅凭 ``LpStatus`` 无法区分。
    solver_gap: float | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "objective": self.objective_key,
             "n_sites": self.n_sites, "status": self.status,
             "solver_gap": self.solver_gap,
             "solve_time_s": round(self.solve_time_s, 2)}
        d.update({k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in self.metrics.items()})
        return d


# ---------------------------------------------------------------------------
# 单目标求解（固定 N）
# ---------------------------------------------------------------------------

def solve_fixed_n(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    n_sites: int,
    objective: str,
    cfg,
    name: str | None = None,
) -> Scheme:
    """固定设施数 N 的单目标选址。

    Parameters
    ----------
    objective
        * ``"min_total_time"``  —— 需求加权的总接驳时间最小（p-中位口径，"距离最近"）
        * ``"min_cost"``        —— 总建设成本最小
        * ``"max_coverage"``    —— 接驳预算内被服务的需求量最大（MCLP 口径）
        * ``"min_worst_case"``  —— 最差人群组的平均接驳时间最小（公平口径）

    Returns
    -------
    :class:`Scheme`。若求解失败，``status`` 会反映出来，不抛异常——
    便于批量生成方案族。
    """
    import pulp
    import time as _time

    t0 = _time.time()
    name = name or objective
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)
    n_sites = int(min(n_sites, n_cand))

    at = np.where(np.isfinite(access_time_s), access_time_s, np.inf) / 60.0  # 分钟
    d = dem["demand"].to_numpy(dtype=np.float64)
    cost = cand["cost"].to_numpy(dtype=np.float64)

    # 接驳时间预算（可达定义），与阶段三保持一致
    ac = cfg.get("stage3_ip", {})
    budget_min = ac.get("access_time_budget_min")
    if budget_min is None:
        radius = float(ac.get("access_radius_m", 3000.0))
        v = max((float(m["speed_kmh"]) for m in cfg.get("access.modes", [])), default=25.0)
        budget_min = radius / (v * 1000 / 3600) / 60.0
    reach = np.isfinite(at) & (at <= float(budget_min))

    prob = pulp.LpProblem(f"fixed_n_{objective}", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_cand)]
    prob += pulp.lpSum(y) == n_sites, "exactly_n"

    # ------------------------------------------------------------------
    # 共同的服务底线：核心需求必须被覆盖。
    #
    # 这一约束对所有目标**都施加**，是本模块设计的关键。若不加，
    # "成本最低"会退化为"挑 11 个最便宜的站点"——实测只覆盖 50% 的
    # 需求，这不是一个可用的规划方案，而是"建 11 个便宜但没用的站点"。
    #
    # 加上共同底线后，四套方案都是**可行规划**，区别在于侧重点：
    #   成本最低   -> 在满足服务底线的前提下最省
    #   距离最近   -> 在满足底线的前提下总接驳时间最短
    #   覆盖最大   -> 在满足底线的前提下尽可能多覆盖（会溢出到非核心区）
    #   体验最均衡 -> 在满足底线的前提下最差人群的体验最好
    #
    # 这四种方案才构成论文中可对照的"方案族"。
    # ------------------------------------------------------------------
    from .stage3_ip import core_demand_mask

    d_min = float(ac.get("min_core_demand", 0.6))
    core_mask = core_demand_mask(d, d_min)
    core_idx = np.flatnonzero(core_mask)
    n_unreachable = 0
    for i in core_idx:
        js = np.flatnonzero(reach[i])
        if len(js) == 0:
            n_unreachable += 1
            continue
        prob += pulp.lpSum(y[j] for j in js) >= 1, f"core_cover_{i}"
    if n_unreachable:
        logger.warning(
            "[%s] %d/%d 个核心需求单元在接驳预算内无候选可达，已跳过其覆盖约束",
            name, n_unreachable, len(core_idx),
        )

    # 站点数固定后，每个目标都只需 y 变量（分配用贪心还原），
    # 除非目标本身含分配决策（max_coverage / min_total_time）。
    if objective == "min_cost":
        prob += pulp.lpSum(cost[j] * y[j] for j in range(n_cand))

    elif objective == "max_coverage":
        # MCLP：在核心覆盖底线之上，最大化被覆盖的**全部**需求
        # （因此这一方案会溢出到非核心单元，覆盖率通常高于其他三个）。
        z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n_dem)]
        prob += -pulp.lpSum(d[i] * z[i] for i in range(n_dem))
        for i in range(n_dem):
            js = np.flatnonzero(reach[i])
            if len(js) == 0:
                prob += z[i] == 0, f"nocov_{i}"
            else:
                prob += z[i] <= pulp.lpSum(y[j] for j in js), f"cov_{i}"

    elif objective == "min_total_time":
        # 说明：本目标的时间只统计**核心需求单元**，见下方 assign_cells 的注释。
        # p-中位：为每个需求单元建分配变量。
        #
        # ⚠️ 邻域宽度 k 必须**显著大于** p-中位基线用的 25。固定规模模式下
        # 全局只选 N=10 个站，若某单元的 25 近邻里一个选中站都没有，它就
        # 无法被分配，求解器于是转而优化"可分配性"而不是"总接驳时间"。
        #
        # 当前算例的实测（`scripts/27_probe_k25_diagnostic.py`，N=10）：
        #
        #   k=25  min_total_time  总接驳 452 529  成本 5.06 亿  最差 10.09  覆盖 95.8%
        #   k=120 min_total_time  总接驳 453 744  成本 4.53 亿  最差  9.75  覆盖 96.2%
        #
        # ⚠ 旧注释写的是"k=25 时该方案的成本与总时间**双双劣于**成本最低方案
        #   （6.20 亿 / 453 955 对 3.48 亿 / 450 715），即在它自己要优化的目标
        #   上输了"。**该结论在当前算例上不再成立**：k=25 反而在总接驳时间上
        #   略胜（452 529 < 453 744，约 0.3%），但代价是成本高 12%、最差人群
        #   差 0.34 分钟、覆盖率低 0.4 个百分点。
        #
        #   所以取宽邻域 k=120 的理由应据此改写：不是"窄邻域使目标不可达"，
        #   而是**窄邻域让求解器靠牺牲其余三个目标去赢得它自己那一个**——
        #   同一种退化，只是方向相反。k=120 仍是对的选择，但依据不同了。
        #
        # 这里取 k = 120（约覆盖候选总数的 9%），使任意合理的 10 站布局
        # 都能被表达。变量数 2519×120 ≈ 30 万，CBC 数分钟可解。
        k = int(cfg.get("stage3_ip.fixed_n_k_nearest", 120))
        k = max(1, min(k, n_cand))
        order = np.argsort(at, axis=1, kind="stable")[:, :k]
        ok = np.isfinite(at[np.arange(n_dem)[:, None], order])

        # ------------------------------------------------------------------
        # 时间目标只在**核心需求单元**上统计——这一点是可比性的关键。
        #
        # 若对全部单元建分配变量并用未分配惩罚，求解器会为了减少罚项而
        # 把偏远单元也一并分配，**总接驳时间反而被拉高**：实测 k=120 时
        # "距离最近"方案的总时间（477,615）显著劣于"最低成本"方案
        # （450,715），即在自己要优化的目标上输给了别人。
        #
        # 限制到核心集合后，四个方案都在**同一个需求集合**上计算时间，
        # 比较才有意义。这也与规划语义相符：核心集合正是所有方案都必须
        # 服务的那个需求集合（见共同服务底线约束）。
        # ------------------------------------------------------------------
        assign_cells = core_idx
        x = {}
        terms = []
        # ⚠️ 必须迭代 order[i][ok[i]]（有限近邻的**候选编号**），
        # 不能写 `for j in ok[i]`——ok[i] 是布尔数组，迭代出来的是
        # True/False，int(True)=1 会取到 at[i,1]，若该值恰为 inf，
        # d[i]*inf = inf 使 M 变成 inf，PuLP 报
        # "Cannot multiply variables with NaN/inf values"。
        finite = [d[i] * at[i, int(j)] for i in assign_cells
                  for j in order[i][ok[i]]]
        M = (max(finite) * 1.2 + 1.0) if finite else 1e9
        for i in assign_cells:
            for jj in order[i][ok[i]]:
                j = int(jj)
                v = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
                x[(i, j)] = v
                terms.append((d[i] * at[i, j] - M) * v)
        prob += pulp.lpSum(terms)
        for i in assign_cells:
            neigh = [int(j) for j in order[i][ok[i]]]
            if not neigh:
                continue
            prob += pulp.lpSum(x[(i, j)] for j in neigh) <= 1, f"assign_{i}"
            for j in neigh:
                prob += x[(i, j)] <= y[j], f"link_{i}_{j}"

    elif objective == "min_worst_case":
        # 公平口径：最小化**最差人群组**的需求加权平均接驳时间。
        #
        # 表述为：以辅助变量 T 为组均值的上界，目标为最小化 T；
        # 服务约束用未分配惩罚软约束（保证可行且尽量服务）。
        # 这一口径与经典 p-中心的单用户极小极大不同——它关心的是
        # "哪一类人群被服务得最差"，而不是"哪一个最偏远的单元"。
        if "cluster" in dem.columns:
            groups = [np.flatnonzero((dem["cluster"] == c).to_numpy())
                      for c in sorted(dem["cluster"].unique())]
        else:
            logger.warning("需求表无 cluster 列，min_worst_case 退化为全体单一组")
            groups = [np.arange(n_dem)]

        # 本分支的邻域必须比 min_total_time 更宽。原因：这里对**每组必须
        # 服务的需求份额**施加了硬约束，若某单元的可达候选里一个选中站都
        # 没有，该约束无法满足、整个模型 Infeasible。实测 k=120、alpha=0.95
        # 时 11 个站落在某单元 120 近邻内的概率仅约 65%，约 35% 的单元
        # 无法分配，模型不可行。放宽到 300 并降低门槛后可行。
        k = int(cfg.get("stage3_ip.fixed_n_worstcase_k_nearest", 300))
        k = max(1, min(k, n_cand))
        order = np.argsort(at, axis=1, kind="stable")[:, :k]
        ok = np.isfinite(at[np.arange(n_dem)[:, None], order])

        x = {}
        # 同上：迭代 order[i][ok[i]] 而非布尔数组 ok[i]
        finite = [d[i] * at[i, int(j)] for i in range(n_dem)
                  for j in order[i][ok[i]]]
        M = (max(finite) * 1.2 + 1.0) if finite else 1e9

        # ------------------------------------------------------------------
        # 表述：min T，使 T 界住**各人群组的已服务需求加权平均接驳时间**。
        #
        # 为什么不能像初版那样写
        # ----------------------
        # 初版用 (a) 目标里加 `-M·已分配数` 来防止全 0 解，以及
        # (b) 约束 `组总时间 ≤ T · 组**总**需求`。两者叠加产生**反向激励**：
        #   · (a) 奖励"多分配"，求解器把偏远单元也服务了，拉高最差组均值；
        #   · (b) 的分母是**全部**需求，于是"少服务"能让组总时间更小、
        #     从而让 T 更小——求解器有动机不服务。
        # 实测结果：专门优化公平性的方案最差人群接驳 10.469 分钟，反而**劣于**
        # 成本最小方案的 9.748 分钟，即在该方案自己的目标上输了。
        #
        # 正确表述：先要求每组服务不少于其需求的 α（保证分母有下界、可比），
        # 再用 `组总时间 ≤ T · α · 组总需求` 界住均值（因已服务需求 ≥ α·总需求，
        # 该式是 T ≥ 实际组均值的**充分**线性条件）。目标只含 T，不含分配激励。
        # ------------------------------------------------------------------
        alpha = float(cfg.get("stage3_ip.fixed_n_min_group_service", 0.95))
        Tvar = pulp.LpVariable("T", lowBound=0.0)
        prob += Tvar
        # 分配变量只建在**核心单元**上：核心集合是所有方案共同的服务底线，
        # 在它上面比较组均值才有意义；同时把变量数从 2519×300 降到
        # 1008×300，建模时间随之减半。
        assign_cells = core_idx
        for i in assign_cells:
            for jj in order[i][ok[i]]:
                j = int(jj)
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in assign_cells:
            neigh = [int(j) for j in order[i][ok[i]]]
            if not neigh:
                continue
            prob += pulp.lpSum(x[(i, j)] for j in neigh) <= 1, f"assign_{i}"
            for j in neigh:
                prob += x[(i, j)] <= y[j], f"link_{i}_{j}"

        for gi, g in enumerate(groups):
            # ⚠️ 必须用 order[i][ok[i]] 取**候选编号**，不能用
            # np.flatnonzero(ok[i])（那给的是位置）——x 的键是候选编号。
            # 这个错误在本文件里已经犯过两次，重写时又写回来一次。
            # ⚠️ 必须只在**核心单元**上取分配变量。x 只为核心单元建立，
            # 若这里遍历整个组 g（含非核心单元），x[(i, j)] 会抛 KeyError，
            # 而且报错信息是一个裸元组（如 "(3, 1032)"），完全看不出是
            # KeyError——实测为此白查了一轮。
            g_core = np.intersect1d(g, assign_cells, assume_unique=False)
            dtot = float(d[g_core].sum()) if len(g_core) else 0.0
            if dtot <= 0:
                continue
            js_all = [(i, int(j)) for i in g_core for j in order[i][ok[i]]]
            if not js_all:
                continue
            served_w = pulp.lpSum(d[i] * x[(i, j)] for i, j in js_all)
            tt = pulp.lpSum(d[i] * at[i, j] * x[(i, j)] for i, j in js_all)
            # 每组至少服务 α 份额，保证组均值的分母有下界
            prob += served_w >= alpha * dtot, f"min_service_{gi}"
            prob += tt <= Tvar * alpha * dtot, f"worst_group_{gi}"

    else:
        raise ValueError(f"未知目标 '{objective}'")

    # 停止判据：**墙钟时限**。这是本模块唯一一处会让结果依赖机器的地方，
    # 必须写清楚为什么不用别的判据。
    #
    # 为什么不用间隙判据（``gapRel``）
    # ------------------------------
    # 实测（PuLP 3.3.2 + CBC），三种停止方式的状态映射：
    #     gapRel=0.5            -> LpStatus=Optimal  sol_status=1  ← **谎报**
    #     maxNodes=1            -> LpStatus=Optimal  sol_status=2  ← 诚实
    #     timeLimit 截断（带解） -> LpStatus=Optimal  sol_status=2  ← 诚实
    # 因**间隙**而停时 `sol_status` 是 `LpSolutionOptimal`，与真正证明最优
    # **无法区分**，于是 `solver_status.describe` 会把"达到 1% 间隙"记成
    # "已证明最优"。用 `gapRel` 等于把假声明引进来，比现状更糟。
    # `maxNodes` 虽然诚实，但无法标定出一个"既能限时又保住解"的节点数
    # （实测小算例 maxNodes=5 就直接解到最优了）。
    #
    # 代价：停止点依赖墙钟，**跑本求解时机器必须是空的**。实测空载下连续两次
    # 求解给出逐位相同的解与指标（`审计记录/probe_wc_2.log`），故结果在本算例上可复现。
    # 论文引用该行时须写成"求解至 N 秒时限、未证明最优"，而非"最优"。
    _tl = cfg.get("stage3_ip.fixed_n_time_limit_s", 600)
    solver = pulp.PULP_CBC_CMD(
        msg=0, timeLimit=(None if not _tl else float(_tl)))
    try:
        prob.solve(solver)
    except Exception as exc:
        logger.error("[%s] 求解器异常: %s", name, str(exc)[:160])
        return Scheme(name, objective, 0, [], {}, _time.time() - t0, "solver_error")

    status = pulp.LpStatus[prob.status]
    dt = _time.time() - t0
    if status != "Optimal":
        logger.warning("[%s] 求解状态: %s", name, status)
        return Scheme(name, objective, 0, [], {}, dt, status)

    # "Optimal" 只在**证明**最优时才写（CBC 看 sol_status，HiGHS 读真实 mip_gap）。
    # 早期这里直接用 LpStatus，会把"达到时限/间隙而停下且有可行解"记成最优。
    from .solver_status import describe as _describe_status

    status, _gap = _describe_status(prob, label=name)

    sel = [j for j in range(n_cand) if y[j].value() and y[j].value() > 0.5]
    sel_ids = [int(c) for c in cand["cand_id"].iloc[sel].tolist()]

    metrics = _evaluate_scheme(sel, dem, cand, at, reach, d, cost,
                               core_min_demand=float(
                                   ac.get("min_core_demand", 0.6)))
    logger.info(
        "[%s] N=%d 固定规模: %d 站 | 总接驳 %.0f 分钟·次/日 | 最差人群 %.2f 分钟 "
        "| 覆盖 %.1f%% | 成本 %.2f 亿元 | 耗时 %.1fs",
        name, n_sites, len(sel_ids), metrics["total_access_time"],
        metrics["worst_group_access_time_min"], metrics["demand_coverage_pct"],
        metrics["total_cost_cny"] / 1e8, dt,
    )
    return Scheme(name, objective, len(sel_ids), sel_ids, metrics, dt, status,
                  _gap)


def _evaluate_scheme(sel, dem, cand, at, reach, d, cost,
                     core_min_demand: float = 0.6) -> dict:
    """计算一套方案在各目标上的取值（用于方案族表格与论文对照）。

    ⚠ **两个口径都要落盘。** 模型的优化定义域与主指标的口径**不是同一个集合**：

    * ``max_coverage`` 显式在**全部**需求单元上优化（它本就该溢出到非核心区）；
    * ``min_total_time`` 与 ``min_worst_case`` 只在**核心需求集合**上建分配变量
      （``solve_fixed_n`` 里的 ``assign_cells = core_idx``，为把变量数压到可解规模）。

    而下面主指标的 ``demand_coverage_pct`` / ``worst_group_access_time_min`` 等
    一律在**全部**单元上算。二者不一致会产生一个可见的荒谬结果：**专门优化
    最差人群接驳的方案，在"最差人群接驳"那一列上输给别的方案**——实测确实如此
    （全集合 10.09 分钟，劣于"最短接驳"的 9.75 分钟），而它在模型真正优化的
    核心集合上是 9.34 分钟、为四套中最低。§5.6 必须把这件事如实写出来，
    因此这里把 ``*_core`` 一组也落盘，使正文引用的两个数都可追溯。
    """
    if not sel:
        return {}
    # ⚠ 不用 np.nanmin：当某个需求单元在 sel 内**全不可达**时，整片切片是 NaN，
    #   nanmin 只发一条 RuntimeWarning 然后返回 NaN 就过去了——日志里仅一行
    #   "All-NaN slice encountered"，不抛异常、不改结果，因此极易被当成噪声放过。
    #   改用 inf 做下界再取 min，语义相同且无警告：全不可达 → inf → served=False。
    sub = np.where(np.isfinite(at[:, sel]), at[:, sel], np.inf)
    best = sub.min(axis=1)
    served = np.isfinite(best)
    sel_cost = float(cost[sel].sum())

    out = {
        "total_access_time": float(np.nansum(d[served] * best[served])),
        "unserved_demand": float(d[~served].sum()),
        "demand_coverage_pct": float(100 * d[served].sum() / max(d.sum(), 1e-9)),
        "total_cost_cny": sel_cost,
        "mean_access_time_min": float(np.nanmean(best[served])) if served.any() else np.nan,
    }
    # 最差人群组（按需求加权的组均值取最大）。**两个口径都算**（见函数文档）：
    #   full —— 全部需求单元（主指标，与全文其他表格一致）
    #   core —— 核心需求集合（模型真正优化的定义域）
    if "cluster" in dem.columns:
        from .stage3_ip import core_demand_mask

        core_mask = core_demand_mask(d, float(core_min_demand))
        core_idx = np.flatnonzero(core_mask)

        def _worst(restrict) -> float:
            w_ = 0.0
            for c in sorted(dem["cluster"].unique()):
                g = np.flatnonzero((dem["cluster"] == c).to_numpy())
                if restrict is not None:
                    g = np.intersect1d(g, restrict, assume_unique=False)
                if not len(g):
                    continue
                ok = served[g]
                if not ok.any():
                    return np.inf
                idx = g[ok]
                ww = d[idx]
                w_ = max(w_, float(np.sum(ww * best[idx]) / max(ww.sum(), 1e-9)))
            return w_

        out["worst_group_access_time_min"] = _worst(None)
        out["worst_group_access_time_core_min"] = _worst(core_idx)

        # 核心集合上的覆盖面，使"模型优化口径"可独立核对
        sc = served & core_mask
        out["core_n_cells"] = int(core_mask.sum())
        out["core_demand_coverage_pct"] = float(
            100 * d[sc].sum() / max(d[core_mask].sum(), 1e-9))
        out["core_total_access_time"] = float(np.nansum(d[sc] * best[sc]))
    return out


def build_scheme_family(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
    n_sites: int | None = None,
    objectives: list[str] | None = None,
) -> list[Scheme]:
    """在固定规模 N 下，按不同目标生成方案族。

    默认 N 取 ``stage3_ip.fixed_n``（配置项），目标取四个经典维度。
    """
    n_sites = int(n_sites if n_sites is not None
                  else cfg.get("stage3_ip.fixed_n", 50))
    objectives = objectives or [
        "min_total_time", "min_cost", "max_coverage", "min_worst_case",
    ]
    labels = {
        "min_total_time": "最短接驳（距离最近）",
        "min_cost": "最低成本",
        "max_coverage": "最大覆盖",
        "min_worst_case": "最均衡体验",
    }
    logger.info("生成固定规模方案族: N=%d，%d 个目标", n_sites, len(objectives))
    out: list[Scheme] = []
    failed: list[str] = []
    for obj in objectives:
        label = labels.get(obj, obj)
        try:
            s = solve_fixed_n(demand, candidates, access_time_s,
                              n_sites, obj, cfg, name=label)
        except Exception as exc:
            logger.error("方案 '%s' 求解失败: %s: %s", obj, type(exc).__name__, str(exc)[:200])
            failed.append(label)
            continue
        # ⚠ 不可行时 solve_fixed_n **不抛异常**，而是返回一个 n_sites=0、
        # selected 为空、status="Infeasible" 的对象。早期这里无条件 append，
        # 于是四个方案**全部不可行**时仍打印"4/4 成功"——日志看上去一切正常，
        # 下游却拿到一个空方案族（实测某次 N=11 时如此，白跑 99 分钟）。
        # 必须按 status 与 selected 判定成功，而不是按"有没有返回对象"。
        #
        # ⚠⚠ 判据是"**有没有解**"，不是"有没有**证明**最优"。
        #   这里曾经写成 `s.status != "Optimal"`，于是 status="Feasible"
        #   （**有可行解**、只是没在时限内证明最优）的方案被当成不可行丢掉。
        #   后果极隐蔽：scheme_family.csv 少一行 → all_schemes_comparison.csv
        #   从 27 行变 26 行（日志 05:25 起如此）→ 表 24 重生成时**静默少一行**，
        #   而少一行不会有人注意到。那句"不可行（N 可能过小）"更把人引向错误
        #   方向：方案本身没问题，N=10 也不小，只是没解到证明最优。
        if not s.selected or s.status not in ("Optimal", "Feasible"):
            logger.warning("方案 '%s' 求解失败（status=%s，N=%d 可能过小），已剔除",
                           label, s.status, n_sites)
            failed.append(label)
            continue
        if s.status == "Feasible":
            logger.warning(
                "方案 '%s' **保留**但未证明最优（相对间隙 %s）——引用该行时"
                "须写成「求解至相对间隙 X%%」而非「最优」。",
                label,
                "不可读" if s.solver_gap is None else f"{100 * s.solver_gap:.4f}%")
        out.append(s)
    logger.info("方案族生成完成: %d/%d 成功%s", len(out), len(objectives),
                f"；失败: {'、'.join(failed)}" if failed else "")
    return out


# ---------------------------------------------------------------------------
# 保规模杂交算子
# ---------------------------------------------------------------------------

def make_fixed_cardinality_repair(prob, n_sites: int):
    """构造**保规模**的可行性修复算子。

    为什么需要它
    ------------
    通用修复算子（``stage4_nsga2.make_repair``）在覆盖不足时**加点**。
    在固定规模模式下这会破坏"所有方案恰好含 N 个站"这一前提——实测
    以 11 站的方案族初始化、并用保规模交叉，最终前沿的解仍然不是 11 站，
    因为解码阶段就被加点了。

    本算子改用**换点**：覆盖不足时，加入一个能新覆盖最多未覆盖核心单元的
    候选，同时移出一个"最不关键"的已选站点（移出后失去覆盖的核心单元最少、
    成本最高者优先）。站点总数因此恒定不变。

    Parameters
    ----------
    prob
        :class:`evtol_siting.stage4_nsga2.SitingProblem`。
    n_sites
        目标站点数 N。

    Returns
    -------
    ``repair(x) -> x'``，``x'`` 恒含 ``n_sites`` 个站。
    """
    cov_core = prob.cov[prob.core_mask]
    n_core = len(cov_core)
    n_cand = prob.n_cand
    cost_norm = prob.cost / max(prob.cost.max(), 1.0)

    def repair(x: np.ndarray) -> np.ndarray:
        x = (np.asarray(x) > 0.5).copy()
        sel = np.flatnonzero(x)

        # 先归一到恰好 N 个站（初始解可能因交叉而偏离）
        if len(sel) > n_sites:
            drop = sel[np.argsort(cost_norm[sel])[-(len(sel) - n_sites):]]
            x[drop] = False
        elif len(sel) < n_sites:
            rest = np.flatnonzero(~x)
            need = min(n_sites - len(sel), len(rest))
            if need > 0:
                x[rest[:need]] = True

        if n_core == 0:
            return x.astype(np.float64)

        # 贪心换点：每次迭代修一个"覆盖缺口最大"的核心单元
        for _ in range(n_sites * 2):
            sel = np.flatnonzero(x)
            covered = cov_core[:, sel].any(axis=1)
            uncovered = ~covered
            if not uncovered.any():
                break
            gains = cov_core[uncovered].sum(axis=0).astype(np.float64)
            gains[x] = -1                       # 已选中的不考虑加入
            j_add = int(np.argmax(gains - 1e-6 * cost_norm))
            if gains[j_add] <= 0:
                break

            # 选一个最不关键的站点移出：移出后失去覆盖的核心单元最少
            loss = np.array([
                int((cov_core[:, j] & ~cov_core[:, [k for k in sel if k != j]].any(axis=1)).sum())
                if len(sel) > 1 else 10**6
                for j in sel
            ])
            # 并列时优先移出成本高的
            order = np.lexsort((-cost_norm[sel], loss))
            j_drop = int(sel[order[0]])
            if j_drop == j_add or len(sel) <= 1:
                break
            x[j_drop] = False
            x[j_add] = True

        return x.astype(np.float64)

    return repair


def crossbreed_masks(
    a: np.ndarray,
    b: np.ndarray,
    n_sites: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """保规模的方案杂交：从两个父代方案的并集中抽取 N 个站点。

    算法
    ----
    1. **公共站点直接继承**（两个父代都选的，必进子代）——这是"共识"部分；
    2. 其余名额从**差异站点**（父代之一选的）中随机抽取，抽中概率
       正比于该站点在两个父代中出现的次数（两个都选已归入第 1 步，
       故这里都是 1 次）；
    3. 若名额仍不足（差异站点太少），从全局未选站点随机补齐。

    这样得到的子代**恰好含 N 个站**，符合"所有方案规模相同"的设定，
    同时真正混合了两个父代的布局——而不是像自由掩码交叉那样，
    子代规模会随机漂移。

    为什么必须保规模
    ----------------
    若父代都是 N 个站而交叉后子代规模自由变化，那么"杂交两个方案"
    这个语义就丢失了：子代会退化成一次随机撒点。保规模才能让搜索
    真正在**方案空间**上做组合，而不是在"要不要建站"上做取舍。
    """
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    n_var = len(a)
    common = a & b
    diff = a ^ b

    child = common.copy()
    n_have = int(child.sum())

    n_from_diff = min(n_sites - n_have, int(diff.sum()))
    if n_from_diff > 0:
        idx = np.flatnonzero(diff)
        pick = rng.choice(idx, size=n_from_diff, replace=False)
        child[pick] = True
        n_have += n_from_diff

    if n_have < n_sites:
        rest = np.flatnonzero(~child)
        need = min(n_sites - n_have, len(rest))
        if need > 0:
            child[rng.choice(rest, size=need, replace=False)] = True
    elif n_have > n_sites:
        sel = np.flatnonzero(child)
        drop = rng.choice(sel, size=n_have - n_sites, replace=False)
        child[drop] = False

    return child.astype(np.float64)


def swap_mutate(
    x: np.ndarray,
    n_sites: int,
    rng: np.random.Generator,
    n_swaps: int = 1,
) -> np.ndarray:
    """保规模的交换变异：删掉一个已选站点、加入一个未选站点。

    与自由掩码下的位翻转变异不同，交换变异**不改变站点总数**，
    因此与固定规模设定相容。
    """
    x = (np.asarray(x) > 0.5).astype(np.float64)
    sel = np.flatnonzero(x)
    unsel = np.flatnonzero(x < 0.5)
    if len(sel) == 0 or len(unsel) == 0:
        return x
    k = min(n_swaps, len(sel), len(unsel))
    drop = rng.choice(sel, size=k, replace=False)
    add = rng.choice(unsel, size=k, replace=False)
    x[drop] = 0.0
    x[add] = 1.0
    return x
