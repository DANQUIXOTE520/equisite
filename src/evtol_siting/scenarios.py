"""四约束情景方案族。

与 :mod:`schemes` 的区别
------------------------
``schemes`` 里的四套方案是**同一组约束、四个不同目标**（成本最低／距离最近／
覆盖最大／最均衡体验）。

本模块的四套方案由**四种不同的规划约束**定义——每一种对应规划实践中一个
真实的问题：

============  ==================================  ====================
情景           它回答的规划问题                     特征约束
============  ==================================  ====================
S1 预算约束型   "钱就这么多，最多能服务多少人？"      Σ c_j y_j ≤ B
S2 服务标准型   "要达到更严的接驳标准，至少花多少？"  接驳预算收紧到 R′
S3 规模约束型   "只能批这么多站，能覆盖多少？"        Σ y_j ≤ N
S4 公平约束型   "不能让任何一类人群被落下，代价多少？" 每组覆盖率 ≥ α
============  ==================================  ====================

四者都是**线性模型**（0-1 整数或混合整数），可精确求解。

为什么不做 LP / NLP 版本
------------------------
连续松弛（LP）会解出"建 3.7 个起降场"这类不可实施的答案——选址问题的
决策变量本质上是二值的。非线性（NLP）在本研究的表述下也有实质障碍：
公平性目标含"组均值"这一分式，而分组约束的分母若用组**总**需求，
求解器会通过"少服务"来降低目标值（见 §6.5 缺陷 D8 的详细记录）。

因此本模块用**四约束**而非"四模型类型"：在保留规划语义的同时，
让每个模型都保持良态、可精确求解、结果可解释。
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import pandas as pd

from .schemes import Scheme, _evaluate_scheme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 情景定义
# ---------------------------------------------------------------------------

CONSTRAINT_SCENARIOS: list[dict] = [
    {
        "key": "budget",
        "param_key": "budget_cny",
        "default": 500_000_000.0,
        "name_zh": "预算约束型",
        "name_en": "Budget-constrained",
        "objective": "max_coverage",
        "desc_zh": "给定资本预算上限，在预算内最大化被服务的出行需求",
        "desc_en": "Given a capital budget cap, maximise served travel demand",
    },
    {
        "key": "access_standard",
        "param_key": "radius_min",
        "default": 10.0,
        "name_zh": "服务标准约束型",
        "name_en": "Access-standard-constrained",
        "objective": "min_cost",
        "desc_zh": "给定更严格的接驳时间标准，求满足该标准的最低成本方案",
        "desc_en": "Given a stricter access-time standard, minimise cost subject to it",
    },
    {
        "key": "facility_cap",
        "param_key": "n_max",
        "default": 15,
        "name_zh": "设施规模约束型",
        "name_en": "Facility-capped",
        "objective": "max_coverage",
        "desc_zh": "在规划批准的站点数上限内，最大化被服务的出行需求",
        "desc_en": "Within an approved cap on site count, maximise served demand",
    },
    {
        "key": "equity_floor",
        "param_key": "alpha",
        "default": 0.95,
        "name_zh": "公平约束型",
        "name_en": "Equity-constrained",
        "objective": "min_cost",
        "desc_zh": "要求每一类收入人群的覆盖率均不低于 α，求最低成本方案",
        "desc_en": "Require every income group to reach coverage α, then minimise cost",
    },
]


# ---------------------------------------------------------------------------
# 求解
# ---------------------------------------------------------------------------

def solve_constraint_scenario(
    spec: dict,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
) -> Scheme:
    """求解单个约束情景。

    共用结构：选址变量 ``y_j ∈ {0,1}``；覆盖指示 ``z_i`` 满足
    ``z_i ≤ Σ_{j∈N(i)} y_j``，其中 ``N(i)`` 是接驳预算内可达的候选集。
    各情景只在**目标**与**特征约束**上不同。
    """
    import pulp

    t0 = _time.time()
    key = spec["key"]
    name = spec["name_zh"]

    sc_cfg = cfg.get("stage3_ip.scenario_params", {}) or {}
    param = float(sc_cfg.get(spec["param_key"], spec["default"]))

    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    n_dem, n_cand = len(dem), len(cand)

    at = np.where(np.isfinite(access_time_s), access_time_s, np.inf) / 60.0   # 分钟
    d = np.nan_to_num(dem["demand"].to_numpy(dtype=np.float64), nan=0.0)
    cost = cand["cost"].to_numpy(dtype=np.float64)

    ac = cfg.get("stage3_ip", {})
    budget_min = ac.get("access_time_budget_min")
    if budget_min is None:
        radius = float(ac.get("access_radius_m", 3000.0))
        v = max((float(m["speed_kmh"]) for m in cfg.get("access.modes", [])), default=25.0)
        budget_min = radius / (v * 1000 / 3600) / 60.0
    budget_min = float(budget_min)

    # 服务标准情景把接驳预算**收紧**到 R′；其余情景用基准预算
    reach_min = param if key == "access_standard" else budget_min
    cov = np.isfinite(at) & (at <= reach_min)

    from .stage3_ip import core_demand_mask

    core_mask = core_demand_mask(d, float(ac.get("min_core_demand", 0.6)))

    prob = pulp.LpProblem(f"scenario_{key}", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_cand)]
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n_dem)]

    for i in range(n_dem):
        js = np.flatnonzero(cov[i])
        if len(js) == 0:
            prob += z[i] == 0, f"nocov_{i}"
        else:
            prob += z[i] <= pulp.lpSum(y[j] for j in js), f"cov_{i}"

    if spec["objective"] == "max_coverage":
        prob += -pulp.lpSum(d[i] * z[i] for i in range(n_dem))
    else:
        prob += pulp.lpSum(cost[j] * y[j] for j in range(n_cand))

    # -- 特征约束 ----------------------------------------------------------
    if key == "budget":
        prob += pulp.lpSum(cost[j] * y[j] for j in range(n_cand)) <= param, "budget_cap"

    elif key == "access_standard":
        # 目标：最小化成本。约束：核心需求在**收紧后的** R′ 内可达。
        # 注意本情景**不要求全部单元覆盖**——这正是它与集合覆盖的区别：
        # 它问的是"要达到更严的标准，最低要花多少"，而非"最少建几个站"。
        n_unreach = 0
        for i in np.flatnonzero(core_mask):
            js = np.flatnonzero(cov[i])
            if len(js) == 0:
                n_unreach += 1
                continue
            prob += pulp.lpSum(y[j] for j in js) >= 1, f"standard_{i}"
        if n_unreach:
            logger.warning(
                "[%s] %d/%d 个核心单元在 %.0f 分钟内无候选可达，已跳过其约束",
                name, n_unreach, int(core_mask.sum()), reach_min,
            )

    elif key == "facility_cap":
        prob += pulp.lpSum(y) <= int(param), "facility_cap"

    elif key == "equity_floor":
        if "cluster" not in dem.columns:
            raise KeyError("公平约束情景需要 demand 表含 cluster 列")
        # 每组覆盖率 ≥ α。用 z_i 的加权和表达，**线性**且不引入分式，
        # 因此不会出现"少服务以降低目标值"的反向激励。
        for c in sorted(dem["cluster"].unique()):
            g = np.flatnonzero((dem["cluster"] == c).to_numpy())
            dtot = float(d[g].sum())
            if dtot <= 0:
                continue
            prob += pulp.lpSum(d[i] * z[i] for i in g) >= param * dtot, f"equity_{int(c)}"
        # 基础服务底线：核心需求仍须覆盖
        for i in np.flatnonzero(core_mask):
            js = np.flatnonzero(cov[i])
            if len(js) == 0:
                continue
            prob += pulp.lpSum(y[j] for j in js) >= 1, f"core_{i}"

    else:
        raise ValueError(f"未知约束情景: {key}")

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=600)
    try:
        prob.solve(solver)
    except Exception as exc:
        logger.error("[%s] 求解器异常: %s: %s", name, type(exc).__name__, str(exc)[:160])
        return Scheme(name, key, 0, [], {}, _time.time() - t0, "solver_error")

    status = pulp.LpStatus[prob.status]
    dt = _time.time() - t0
    if status != "Optimal":
        logger.warning("[%s] 求解状态: %s（%s=%.4g）", name, status, spec["param_key"], param)
        return Scheme(name, key, 0, [], {}, dt, status)

    # "Optimal" 只在**证明**最优时才写。判据与 baselines / stage3_ip 同一份实现。
    from .solver_status import describe as _describe_status

    status, _gap = _describe_status(prob, label=name)

    sel = [j for j in range(n_cand) if y[j].value() and y[j].value() > 0.5]
    sel_ids = [int(c) for c in cand["cand_id"].iloc[sel].tolist()]
    metrics = _evaluate_scheme(sel, dem, cand, at, cov, d, cost)
    metrics["scenario_param"] = param
    metrics["scenario_param_key"] = spec["param_key"]

    logger.info(
        "[%s] %s=%.4g -> %d 站 | 成本 %.2f 亿 | 覆盖 %.1f%% | 最差人群 %.2f 分钟 | %.1fs",
        name, spec["param_key"], param, len(sel_ids),
        metrics.get("total_cost_cny", 0.0) / 1e8,
        metrics.get("demand_coverage_pct", 0.0),
        metrics.get("worst_group_access_time_min", float("nan")), dt,
    )
    return Scheme(name, key, len(sel_ids), sel_ids, metrics, dt, status)


def build_constraint_scenarios(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
    scenarios: list[dict] | None = None,
) -> list[Scheme]:
    """生成四约束情景方案族，返回成功求解的 :class:`Scheme` 列表。"""
    scenarios = scenarios or CONSTRAINT_SCENARIOS
    logger.info("生成四约束情景方案族: %d 个情景", len(scenarios))
    out: list[Scheme] = []
    for i, spec in enumerate(scenarios, 1):
        # 求解前先落一行日志：场景数增大后单次求解可能长达数十分钟，没有这行
        # 就无法判断是"卡在某个情景"还是"整体没启动"。求解完还会再报一行结果。
        logger.info("[%d/%d] 开始求解 %s …", i, len(scenarios), spec.get("name_zh", "?"))
        try:
            out.append(
                solve_constraint_scenario(spec, demand, candidates, access_time_s, cfg)
            )
        except Exception as exc:
            logger.error(
                "情景 '%s' 求解失败: %s: %s",
                spec.get("name_zh", "?"), type(exc).__name__, str(exc)[:200],
            )
    ok = sum(1 for s in out if s.status == "Optimal")
    logger.info("四约束方案族完成: %d/%d 成功", ok, len(scenarios))
    return out
