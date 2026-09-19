"""阶段三：0-1 整数规划选址（集合覆盖型精确求解）。

对应申报书"阶段三：0-1 整数规划选址"与四大要求约束。

数学模型
--------
决策变量::

    y_j ∈ {0, 1}        j ∈ M   —— 是否在候选点 j 建设起降场
    x_ij ∈ {0, 1}       i ∈ D, j ∈ M —— 需求单元 i 是否由起降场 j 服务

目标（两种情景，对应申报书的"两个最优选址方案"）::

    (A) 成本优先:   min  Σ_j c_j · y_j
    (B) 覆盖优先:   min  Σ_j y_j          （等价于设施数最少）

约束
----
1. **需求全覆盖**（硬约束，仅对需求量 >= d_min 的核心单元强制）::

       Σ_{j : d_ij <= R} y_j >= 1        ∀ i ∈ D_core

   对非核心单元（``demand < min_core_demand``）不强制覆盖——这是申报书
   "最小核心需求"约束的含义：把有限预算集中在真正的需求热点上，避免为
   极稀疏的郊区需求建设昂贵设施。

2. **最大接驳半径**  R —— 通过邻接集合 ``N(i) = { j : d_ij <= R }`` 体现。

3. **起降场总数上限**（对应"起降场总数最小 / 投入设施最低"）::

       Σ_j y_j <= N_max

4. **容量约束**（本研究新增，见下）::

       Σ_i demand_i · x_ij <= Q_j · y_j        ∀ j

关于容量约束
------------
申报书的四大约束中没有容量约束，但一个起降场的实际吞吐能力有限
（FATO 数量 × 小时架次 × 运营时长）。**不加容量约束的选址模型会给出
"少数几个巨型起降场覆盖全城"的不现实方案**，使结果失去工程意义。
因此本实现在 IP 中加入容量约束，并把它作为相对于申报书的一处方法改进。

集合覆盖问题是 NP-hard，但成都案例 ``|M|`` 约 100–300、``|D|`` 约 2400，
CBC 分支定界可在秒级求出**全局最优**——这正是本阶段的价值：为
NSGA-II 提供精确解作为参照下界，也构成论文中的单目标基线。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IPSolution:
    """整数规划求解结果。"""
    name: str
    status: str
    objective: float
    selected: list[int]                       # 选中的 cand_id
    n_sites: int
    total_cost: float
    coverage: dict = field(default_factory=dict)
    assignment: np.ndarray | None = None       # 每个需求单元被分配到哪个候选（列下标）
    solve_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "n_sites": self.n_sites,
            "objective": round(self.objective, 2),
            "total_cost_cny": round(self.total_cost, 0),
            "solve_time_s": round(self.solve_time_s, 2),
            **{f"coverage_{k}": round(v, 4) if isinstance(v, float) else v
               for k, v in self.coverage.items()},
        }


# ---------------------------------------------------------------------------
# 邻接结构
# ---------------------------------------------------------------------------

def core_demand_mask(demand_vals: np.ndarray, threshold: float) -> np.ndarray:
    """判定"核心需求单元"的布尔掩码。

    阈值语义由数值大小自动判别（全项目统一，避免各处实现不一致）：

    * ``0 < threshold < 1`` —— **分位数**。``0.6`` 表示取需求量最高的
      40% 单元作为核心（即 ``demand >= quantile(0.6)``）。
    * ``threshold >= 1``    —— **绝对量**（次/日）。

    分位数口径是推荐默认值：绝对阈值与出行生成率的标定强耦合，
    换城市或换一组出行率就会让核心单元变成空集（覆盖约束静默失效）
    或全集（等于没有区分优先级）。

    Parameters
    ----------
    demand_vals
        各需求单元的需求量（次/日）。
    threshold
        阈值，见上。

    Returns
    -------
    与 ``demand_vals`` 等长的布尔数组。
    """
    v = np.asarray(demand_vals, dtype=np.float64)
    if 0.0 < threshold < 1.0:
        if v.size == 0:
            return np.zeros(0, dtype=bool)
        return v >= float(np.quantile(v, threshold))
    return v >= float(threshold)


def build_coverage_sets(
    demand_xy: np.ndarray,
    cand_xy: np.ndarray,
    access_time: np.ndarray,
    time_budget_s: float,
) -> list[np.ndarray]:
    """构造每个需求单元的**可达候选集合** ``N(i)``。

    判据用**接驳时间**而非直线距离：``t_ij <= time_budget``。
    由于接驳时间本身是由最大接驳距离定义的（见 ``geo.travel_time_matrix``），
    两种口径等价，但时间口径便于在论文中直接解释为"15 分钟生活圈"。

    Returns
    -------
    长度 ``len(demand_xy)`` 的列表，第 i 项是 ``N(i)`` 中候选点的列下标数组。
    """
    within = access_time <= time_budget_s
    return [np.flatnonzero(within[i]) for i in range(within.shape[0])]


# ---------------------------------------------------------------------------
# 求解
# ---------------------------------------------------------------------------

def solve_ip(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time: np.ndarray,
    cfg,
    name: str = "ip",
    access_radius_m: float | None = None,
    max_sites: int | None = None,
    objective: str = "cost",
    enforce_capacity: bool = True,
) -> IPSolution:
    """求解 0-1 整数规划选址模型。

    Parameters
    ----------
    demand
        需求单元表，须含 ``cell_id``、``centroid_x/y``、``demand``。
    candidates
        候选表，须含 ``cand_id``、``x``、``y``、``cost``、``capacity``。
    access_time
        ``(n_demand, n_cand)`` 接驳时间矩阵（秒）。
    objective
        ``"cost"`` —— 最小化总建设成本；``"sites"`` —— 最小化设施数量。
    enforce_capacity
        是否施加容量约束。关闭后即为经典集合覆盖模型，可用于对照。

    Returns
    -------
    :class:`IPSolution`。若问题不可行（例如 ``max_sites`` 过小而无法覆盖），
    ``status`` 会反映出来，且不抛异常——便于敏感性分析批量运行。
    """
    import pulp
    import time as _time

    t0 = _time.time()

    ac = cfg.get("stage3_ip", {})
    radius = float(access_radius_m if access_radius_m is not None
                   else ac.get("access_radius_m", 3000.0))
    n_max = int(max_sites if max_sites is not None else ac.get("max_sites", 40))
    d_min = float(ac.get("min_core_demand", 500.0))
    budget_min = ac.get("access_time_budget_min", None)

    # 时间预算：配置给了分钟数就用它，否则由半径与最快方式推算
    if budget_min is not None:
        time_budget_s = float(budget_min) * 60.0
    else:
        v = max((float(m["speed_kmh"]) for m in cfg.get("access.modes", [])), default=25.0)
        time_budget_s = radius / (v * 1000 / 3600)

    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)

    if n_cand == 0:
        return IPSolution(name, "no_candidates", np.inf, [], 0, 0.0, {"feasible": False})
    if access_time.shape != (n_dem, n_cand):
        raise ValueError(
            f"access_time 形状 {access_time.shape} 与 (n_demand={n_dem}, n_cand={n_cand}) 不匹配"
        )

    coverage_sets = build_coverage_sets(
        dem[["centroid_x", "centroid_y"]].to_numpy(),
        cand[["x", "y"]].to_numpy(),
        access_time, time_budget_s,
    )

    # 核心需求单元：需求量达到阈值的单元才强制覆盖。
    #
    # 阈值支持两种口径，由数值大小自动判别：
    #   0 < v < 1  ->  **分位数**（如 0.6 表示需求量最高的 40% 单元）
    #   v >= 1     ->  **绝对量**（如 500 表示 >= 500 次/日）
    #
    # 分位数口径是更稳健的默认：绝对阈值与出行生成率的标定强耦合，
    # 换一个城市或换一组出行率就会让"核心单元"变成空集或全集。
    # 实测中 `min_core_demand: 500` 配上约 8 次/日/单元的合成需求，
    # 会得到 0 个核心单元，覆盖约束全部失效——这类错误不会报错，
    # 只会让模型悄悄退化为"无约束选址"。
    demand_vals = dem["demand"].to_numpy(dtype=np.float64)
    if 0.0 < d_min < 1.0:
        cutoff = float(np.quantile(demand_vals, d_min))
        mode_desc = f"分位数 {d_min:.2f}（>={cutoff:.1f} 次/日）"
    else:
        cutoff = float(d_min)
        mode_desc = f"绝对量（>={cutoff:.1f} 次/日）"

    core = demand_vals >= cutoff
    core_idx = np.flatnonzero(core)
    if len(core_idx) == 0:
        logger.warning(
            "[%s] **核心需求单元为 0**：阈值 %s 高于所有单元的需求量"
            "（最大 %.1f）。覆盖约束将完全失效，模型退化为无约束选址。"
            "请调低 stage3_ip.min_core_demand，或改用分位数口径（0–1 之间）。",
            name, mode_desc, float(demand_vals.max()) if len(demand_vals) else 0.0,
        )
    logger.info(
        "[%s] 需求单元 %d 个（核心 %d 个，阈值 %s），候选 %d 个，"
        "接驳时间预算 %.1f 分钟，设施上限 %d",
        name, n_dem, len(core_idx), mode_desc, n_cand, time_budget_s / 60.0, n_max,
    )

    # 无候选可达的核心单元 -> 该单元的覆盖约束**无法施加**。
    #
    # ⚠ 早期版本在这里只打一条 warning，然后把该约束 `continue` 跳过，却仍然
    # 在返回值里写死 `"feasible": True`。后果是：模型的"核心需求全覆盖"这条
    # 硬约束被**静默摘掉了一部分**，而调用方（以及论文附录 D）看到的是一个
    # 标着 feasible 的解——实际 `core_coverage_pct` 已经不是 100 %。
    # 现在把"被摘掉的约束数"显式记录并向上暴露，让调用方自己决定是否可用。
    unreachable = [i for i in core_idx if len(coverage_sets[i]) == 0]
    if unreachable:
        logger.warning(
            "[%s] 有 %d/%d 个核心需求单元在接驳预算内无任何候选起降场可达，"
            "其覆盖约束**无法施加**（已从模型中剔除）。"
            "本解的核心覆盖率不可能是 100 %%，不得当作「核心需求全覆盖」引用。"
            "需要放宽接驳半径或增加候选点。",
            name, len(unreachable), len(core_idx),
        )

    prob = pulp.LpProblem(f"vertiport_{name}", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_cand)]

    # -- 目标 --------------------------------------------------------------
    if objective == "sites":
        prob += pulp.lpSum(y)
    else:
        cost = cand["cost"].to_numpy(dtype=np.float64)
        prob += pulp.lpSum(cost[j] * y[j] for j in range(n_cand))

    # -- 约束 1: 核心需求全覆盖 -------------------------------------------
    for i in core_idx:
        js = coverage_sets[i]
        if len(js) == 0:
            continue          # 已在上面警告；跳过以免整个模型 infeasible
        prob += pulp.lpSum(y[j] for j in js) >= 1, f"cover_{i}"

    # -- 约束 2: 设施总数上限 ---------------------------------------------
    prob += pulp.lpSum(y) <= n_max, "max_sites"

    # -- 约束 3: 容量 ------------------------------------------------------
    if enforce_capacity:
        # 单位换算：demand 是"次/日"，capacity 是"架次/小时"。
        # 必须乘上日运营小时数才能比较——否则 load(几十次/日) 永远大于
        # cap(十几架次/小时)，模型会把**每一个候选都判为不可选**，
        # 导致整个问题不可行。
        op_hours = float(cfg.get("stage3_ip.capacity_hours_per_day", 14.0))
        cap_daily = cand["capacity"].to_numpy(dtype=np.float64) * op_hours

        # 负载估计：把每个需求单元的需求**在其可达候选之间均分**。
        #
        # 早期版本用"就近分配"（需求全部压给最近的候选）。那是负载的
        # **上界**，且过分保守：现实中用户会在可达的多个起降场之间分散，
        # 而模型却假设所有人都挤向同一个站。实测该口径下大量候选的负载
        # 超过容量而被强制置 0，最终使覆盖约束无法满足、整个问题不可行
        # （成都案例：1008 个核心单元中 535 个因此失去可达性）。
        #
        # 均分口径给出的是**期望**负载，既不过分保守也不会低估，是容量
        # 筛查更合适的依据。注意这仍是筛查（site-admissibility screen），
        # 不是负载再分配——真正的再分配需要引入 x_ij 变量，见 §6 局限。
        reach = np.isfinite(access_time) & (access_time <= time_budget_s)
        n_reach = reach.sum(axis=1)                       # 每个单元的可达候选数
        load = np.zeros(n_cand, dtype=np.float64)
        ok_rows = n_reach > 0
        if ok_rows.any():
            share = np.zeros_like(access_time, dtype=np.float64)
            share[ok_rows] = (demand_vals[ok_rows] / n_reach[ok_rows])[:, None]
            share[~reach] = 0.0
            load = share.sum(axis=0)

        n_blocked = int((load > cap_daily).sum()) if n_cand else 0
        if n_blocked:
            logger.info(
                "[%s] 容量约束: %d/%d 个候选的期望负载超过日容量，将被强制置 0"
                "（负载中位数 %.1f，容量中位数 %.1f）",
                name, n_blocked, n_cand, float(np.median(load)), float(np.median(cap_daily)),
            )
            if n_blocked > 0.5 * n_cand:
                logger.error(
                    "[%s] **超过半数候选因容量被排除** —— 容量模型与需求规模不匹配。"
                    "请检查 costs.*.capacity 或 stage3_ip.capacity_hours_per_day "
                    "是否与需求量的量级相称。",
                    name,
                )

        for j in range(n_cand):
            if load[j] > 1e-9:
                # y_j = 1 时要求 load_j <= cap_daily_j；y_j = 0 时自动满足。
                prob += load[j] * y[j] <= cap_daily[j], f"cap_{j}"

    # -- 求解 --------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)
    try:
        prob.solve(solver)
    except Exception as exc:
        logger.error("[%s] 求解器异常: %s", name, str(exc)[:200])
        return IPSolution(name, "solver_error", np.inf, [], 0, 0.0,
                          {"feasible": False}, solve_time_s=_time.time() - t0)

    status = pulp.LpStatus[prob.status]
    dt = _time.time() - t0

    # ⚠ pulp 会把"时限用尽但有可行解"也记成 LpStatusOptimal（见
    # pulp/apis/coin_api.py::get_status：解文件首行以 `Stopped` 开头、第 5 个词是
    # `objective` 时状态被改写）。**仅凭 status 无法区分"证明最优"与"截断的最好解"**。
    # 判据统一走 solver_status 模块（CBC 看 sol_status，HiGHS 读真实 mip_gap），
    # 以免本文件与 baselines.py 各用一套口径、修了一处漏一处。
    from .solver_status import describe as _describe_status

    if status != "Optimal":
        logger.warning("[%s] 求解状态: %s（耗时 %.2fs）", name, status, dt)
        return IPSolution(name, status, np.inf, [], 0, 0.0,
                          {"feasible": False}, solve_time_s=dt)

    proven_tag, gap = _describe_status(prob, label=name)

    selected = [j for j in range(n_cand) if y[j].value() is not None and y[j].value() > 0.5]
    sel_cand_ids = cand["cand_id"].iloc[selected].tolist()
    total_cost = float(cand["cost"].iloc[selected].sum())

    # -- 覆盖率统计 --------------------------------------------------------
    covered = np.zeros(n_dem, dtype=bool)
    assign = np.full(n_dem, -1, dtype=int)
    for i in range(n_dem):
        js = [j for j in coverage_sets[i] if j in set(selected)]
        if js:
            covered[i] = True
            # 分配到接驳时间最短的已选起降场
            assign[i] = min(js, key=lambda j: access_time[i, j])

    pop = dem["population"].to_numpy(dtype=np.float64) if "population" in dem.columns else demand_vals
    pop_total = float(pop.sum())
    coverage = {
        "cells_covered": int(covered.sum()),
        "cells_total": int(n_dem),
        "cell_coverage_pct": float(100 * covered.sum() / max(n_dem, 1)),
        "demand_covered": float(demand_vals[covered].sum()),
        "demand_total": float(demand_vals.sum()),
        "demand_coverage_pct": float(100 * demand_vals[covered].sum() / max(demand_vals.sum(), 1e-9)),
        "population_covered": float(pop[covered].sum()),
        "population_total": pop_total,
        # 注意：population_coverage_pct 必须与 population_covered 一并给出。
        # 早期版本遗漏了该键，导致日志行抛出 KeyError 而中断整个求解流程。
        "population_coverage_pct": float(100 * pop[covered].sum() / max(pop_total, 1e-9)),
        "core_coverage_pct": float(100 * covered[core_idx].sum() / max(len(core_idx), 1)),
        "n_core_cells": int(len(core_idx)),
        # 被静默摘掉覆盖约束的核心单元数。>0 时 "核心需求全覆盖" 这一说法不成立，
        # 调用方与论文必须据此降级表述。
        "n_core_unreachable": int(len(unreachable)),
        "core_coverage_enforced": bool(len(unreachable) == 0),
        # 求解质量：proven=True 才是真证明最优；否则给出实际达到的相对间隙。
        "proven_optimal": bool(proven_tag == "Optimal"),
        "solver_gap": (None if gap is None else float(gap)),
        "feasible": True,
    }

    logger.info(
        "[%s] 解: %d 个起降场，总成本 %.2f 亿元，需求覆盖率 %.1f%%，"
        "人口覆盖率 %.1f%%，核心覆盖率 %.1f%%（耗时 %.2fs，%s）",
        name, len(selected), total_cost / 1e8, coverage["demand_coverage_pct"],
        coverage["population_coverage_pct"], coverage["core_coverage_pct"],
        dt, ("已证明最优" if coverage["proven_optimal"]
             else "未证明最优，gap=%.4f%%" % (100.0 * gap if gap is not None else float("nan"))),
    )

    return IPSolution(
        name=name, status=proven_tag,
        objective=float(pulp.value(prob.objective)),
        selected=sel_cand_ids, n_sites=len(selected), total_cost=total_cost,
        coverage=coverage, assignment=assign, solve_time_s=dt,
    )


# ---------------------------------------------------------------------------
# 情景批量求解
# ---------------------------------------------------------------------------

def solve_scenarios(demand, candidates, access_time_fn, cfg) -> list[IPSolution]:
    """按配置中的 ``stage3_ip.scenarios`` 批量求解，产出"两个最优选址方案"。

    Parameters
    ----------
    access_time_fn
        可调用对象 ``f(radius_m) -> (n_demand, n_cand) 时间矩阵``。
        不同情景的接驳半径不同，时间矩阵也不同，因此以函数形式传入。
    """
    scenarios = cfg.get("stage3_ip.scenarios", [])
    if not scenarios:
        scenarios = [{"name": "default", "max_sites": None, "access_radius_m": None}]

    solutions = []
    for sc in scenarios:
        name = sc.get("name", "scenario")
        radius = sc.get("access_radius_m")
        n_max = sc.get("max_sites")
        at = access_time_fn(radius)
        # 成本优先情景用成本目标；覆盖优先情景用设施数目标
        obj = "sites" if "coverage" in str(name) else "cost"
        sol = solve_ip(
            demand, candidates, at, cfg,
            name=name, access_radius_m=radius, max_sites=n_max, objective=obj,
        )
        solutions.append(sol)

    logger.info(
        "阶段三完成: %d 个情景求解成功（%s）",
        sum(s.status == "Optimal" for s in solutions),
        ", ".join(f"{s.name}={s.status}" for s in solutions),
    )
    return solutions
