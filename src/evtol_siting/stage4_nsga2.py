"""阶段四：NSGA-II 多目标选址优化。

对应申报书"难点三：四个目标的帕累托最优权衡"。

决策变量与编码
--------------
申报书表述为"以起降场编号为基因进行整数编码"。本实现采用**等价的
0/1 掩码编码**：个体 ``x ∈ {0,1}^|M|``，``x_j = 1`` 表示在候选点 ``j``
建设起降场。两种编码在解空间上一一对应（掩码中 1 的位置即所选编号），
但掩码编码：

* 维度固定，可直接使用 NSGA-II 的标准算子（SBX / 多项式变异）；
* 天然避免整数编码下"重复基因""变长交叉"带来的冗余与非法解。

论文中应说明这一编码等价性，并可由 ``encoding: "integer"`` 配置给出
变长整数编码的对照实验。

四个目标（均为最小化）
----------------------
1. ``unserved_demand``   未服务需求 = Σ_i d_i · [ i 未被覆盖 ]
2. ``total_access_time`` 总接驳时间 = Σ_i d_i · min_j t_ij     （单位：分钟·次/日）
3. ``inequity``          最差体验 = max_g ( mean_{i∈g} t_i )   （单位：分钟）
4. ``total_cost``        总建设成本 = Σ_j c_j · x_j            （单位：元）

**目标 3 的定义是本研究的核心方法贡献。** 申报书中"最差体验"的含义
需要精确化，存在两种可能：

* (a) ``max_i t_i`` —— 最差单个用户的接驳时间，即 min-max 公平（Rawlsian）。
  但这一口径与经典 p-center 模型完全重合，不构成新意；
* (b) ``max_g (mean_{i∈g} t_i)`` —— **最差人群组的平均接驳时间**。

本实现默认采用 (b)：它把"人群"作为公平性的分析单元，直接回应"UAM 是否
只服务于高收入人群"这一政策质疑。因为 UAM 长期被批评为精英化交通方式，
以收入分层人群为单元度量的可达性公平，比单用户 min-max 更具决策意义。
配置项 ``stage4_nsga2.equity_measure`` 可在 ``group_mean``（默认）与
``individual_max`` 之间切换，便于做对照实验。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 问题实例
# ---------------------------------------------------------------------------

@dataclass
class SitingProblem:
    """选址问题的**预计算缓存**。

    把所有与个体无关的量（接驳时间矩阵、覆盖矩阵、人群分组、需求量）
    预先算好并转为 numpy 数组，使适应度评估只做数组切片与归约——
    这是让 NSGA-II 在 6 万次评估量级下仍能在分钟级跑完的关键。
    """

    at_min: np.ndarray          # (n_dem, n_cand) 接驳时间（分钟），不可达为 inf
    cov: np.ndarray             # (n_dem, n_cand) bool，是否在接驳时间预算内
    demand: np.ndarray          # (n_dem,) 需求量（次/日）
    population: np.ndarray      # (n_dem,) 人口
    cost: np.ndarray            # (n_cand,) 建设成本
    capacity: np.ndarray        # (n_cand,) 容量（架次/日）
    core_mask: np.ndarray       # (n_dem,) 是否为核心需求单元（强制覆盖）
    groups: list[np.ndarray]    # 各人群包含的需求单元下标
    facility_is_rooftop: np.ndarray  # (n_cand,) bool
    equity_measure: str = "group_mean"
    min_coverage_ratio: float = 1.0  # 核心需求要求的最低覆盖率

    @property
    def n_dem(self) -> int:
        return len(self.demand)

    @property
    def n_cand(self) -> int:
        return len(self.cost)

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())


def build_problem(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access_time_s: np.ndarray,
    cfg,
) -> SitingProblem:
    """从阶段二/三的输出构建 NSGA-II 问题实例。

    Parameters
    ----------
    access_time_s
        ``(n_demand, n_cand)`` 接驳时间矩阵（**秒**）。
    """
    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)

    ac = cfg.get("stage3_ip", {})
    budget_min = ac.get("access_time_budget_min")
    if budget_min is None:
        radius = float(ac.get("access_radius_m", 3000.0))
        v = max((float(m["speed_kmh"]) for m in cfg.get("access.modes", [])), default=25.0)
        budget_min = radius / (v * 1000 / 3600) / 60.0

    at_min = np.where(np.isfinite(access_time_s), access_time_s, np.inf) / 60.0
    cov = at_min <= float(budget_min)

    demand_vals = dem["demand"].to_numpy(dtype=np.float64)
    pop_vals = (
        dem["population"].to_numpy(dtype=np.float64)
        if "population" in dem.columns else demand_vals.copy()
    )

    from .stage3_ip import core_demand_mask

    d_min = float(ac.get("min_core_demand", 0.6))
    core_mask = core_demand_mask(demand_vals, d_min)

    if "cluster" in dem.columns:
        groups = [
            np.flatnonzero((dem["cluster"] == c).to_numpy())
            for c in sorted(dem["cluster"].unique())
        ]
    else:
        logger.warning("需求表缺少 cluster 列，公平性目标退化为全体用户")
        groups = [np.arange(len(dem))]

    prob = SitingProblem(
        at_min=at_min.astype(np.float32),
        cov=cov,
        demand=demand_vals,
        population=pop_vals,
        cost=cand["cost"].to_numpy(dtype=np.float64),
        capacity=cand["capacity"].to_numpy(dtype=np.float64),
        core_mask=core_mask,
        groups=groups,
        facility_is_rooftop=(cand["facility_type"] == "rooftop").to_numpy(),
        equity_measure=str(cfg.get("stage4_nsga2.equity_measure", "group_mean")),
        min_coverage_ratio=float(cfg.get("stage4_nsga2.min_coverage_ratio", 1.0)),
    )

    # -- 启动自检 ----------------------------------------------------------
    # 区分三种失败情形——它们的修法完全不同，笼统报"覆盖不了"会误导排查：
    #   (a) 核心单元为空集  -> 改 min_core_demand（大概率是阈值口径用错）
    #   (b) 核心单元非空但无候选可达 -> 放宽接驳预算或增加候选
    #   (c) 正常
    n_core = int(core_mask.sum())
    if n_core == 0:
        logger.error(
            "**核心需求单元为空集**（阈值 %.4g）。这通常是把 min_core_demand "
            "当绝对量（次/日）用，而实际每单元需求量远小于该值（最大 %.1f）。"
            "请改用分位数口径（0–1 之间，如 0.6 表示需求量最高的 40%% 单元）。"
            "注意：核心单元为空时覆盖约束完全失效，模型会退化为无约束选址。",
            d_min, float(demand_vals.max()) if len(demand_vals) else 0.0,
        )
    else:
        union = prob.cov[core_mask].any(axis=0)
        if not union.any():
            logger.error(
                "核心需求单元（%d 个）中**没有任何一个**能被任何候选起降场覆盖。"
                "接驳时间预算 %.1f 分钟过紧，或候选集分布与需求热点错位。"
                "请检查 stage3_ip.access_time_budget_min / access.modes "
                "与阶段一的候选筛选阈值。",
                n_core, float(budget_min),
            )
        else:
            always = prob.cov[core_mask].sum(axis=0)
            logger.info(
                "问题实例: %d 个需求单元（核心 %d），%d 个候选；"
                "接驳预算 %.1f 分钟；单点最大可覆盖 %d 个核心单元（占核心 %.1f%%）",
                prob.n_dem, n_core, prob.n_cand, float(budget_min),
                int(always.max()), 100.0 * always.max() / max(n_core, 1),
            )
    return prob


# ---------------------------------------------------------------------------
# 适应度评估
# ---------------------------------------------------------------------------

def evaluate_solution(x: np.ndarray, prob: SitingProblem) -> np.ndarray:
    """计算单个选址方案的四个目标值（全部最小化）。

    Parameters
    ----------
    x
        ``{0,1}^{n_cand}`` 掩码。
    """
    sel = np.flatnonzero(x > 0.5)

    # 空集：其它目标是"无穷差"，但成本为 0。用有限大数而非 inf，
    # 避免非支配排序中出现 inf - inf = nan。
    if len(sel) == 0:
        big = 1e12
        return np.array([
            prob.total_demand,                      # 全部未服务
            big,                                    # 总接驳时间无意义
            big,                                    # 公平性无意义
            0.0,                                    # 成本为 0
        ])

    # -- 逐需求单元的最短接驳时间 -----------------------------------------
    sub = prob.at_min[:, sel]                       # (n_dem, k)
    best = sub.min(axis=1)                          # (n_dem,)
    reachable = np.isfinite(best)

    # -- 目标 1: 未服务需求 ------------------------------------------------
    served = prob.demand[reachable].sum()
    f_unserved = prob.total_demand - served

    # -- 目标 2: 总接驳时间（需求加权，单位 分钟·次/日）-------------------
    f_time = float(np.sum(prob.demand[reachable] * best[reachable]))

    # -- 目标 3: 公平性 / 最差体验 ----------------------------------------
    if prob.equity_measure == "individual_max":
        f_equity = float(best[reachable].max()) if reachable.any() else 1e12
    else:
        # 人群组的平均接驳时间取最大。用**需求量加权**的组均值，
        # 使组均值反映"该人群典型成员的体验"，而非"该人群所在格网的平均"。
        worst = 0.0
        for g in prob.groups:
            mask = reachable[g]
            if not mask.any():
                worst = 1e12
                break
            idx = g[mask]
            w = prob.demand[idx]
            tot = w.sum()
            worst = max(worst, float(np.sum(w * best[idx]) / tot) if tot > 0
                        else float(best[idx].mean()))
        f_equity = worst

    # -- 目标 4: 总建设成本 ------------------------------------------------
    f_cost = float(prob.cost[sel].sum())

    return np.array([f_unserved, f_time, f_equity, f_cost])


def make_evaluator(prob: SitingProblem, cache: dict | None = None):
    """构造批量评估函数，带**记忆化**。

    NSGA-II 经过修复算子后会产生大量重复个体（尤其在收敛后期），
    记忆化可显著减少重复计算。用掩码的 ``tobytes()`` 作为键，
    既精确又比字符串快。
    """
    memo = cache if cache is not None else {}

    def evaluate(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty((len(X), 4), dtype=np.float64)
        for i, x in enumerate(X):
            key = (x > 0.5).astype(np.uint8).tobytes()
            val = memo.get(key)
            if val is None:
                val = evaluate_solution(x, prob)
                memo[key] = val
            out[i] = val
        return out

    evaluate.memo = memo        # 便于外部查看命中率
    return evaluate


# ---------------------------------------------------------------------------
# 可行性修复
# ---------------------------------------------------------------------------

def make_repair(prob: SitingProblem, prune: bool = False):
    """构造解码算子：贪心补足未覆盖的核心需求，并剪除冗余站点。

    **修复（补点）**
    覆盖约束是本研究最硬的约束。不做修复的话，随机初始种群中绝大多数
    个体都严重不满足覆盖要求，NSGA-II 需要消耗大量代数才能"学会"覆盖——
    而覆盖恰恰是选址问题里最容易用贪心逼近的部分。修复算子把这个先验
    直接注入，使搜索立刻聚焦于**覆盖之上的权衡**（成本、时间、公平）。

    **关于剪枝：为什么默认关闭，以及它曾经造成的两个相反错误**

    剪枝的规则是：逐个考察已选站点，若移除后不增加未覆盖的核心需求，
    则移除。

    它解决过一个真实问题——修复算子只能**加点**，若初始种群稠密而搜索
    无法删点，解会被锁在高成本区。实测（当前 1315 候选算例、初始密度
    50%，见 `scripts/25_probe_dense_decoder_defect.py`）：NSGA-II 的整个前沿
    成本落在 329–375 亿元且一百代内超体积一次也没有改进，而整数规划用
    11 个站、3.14 亿元即达到 96.13 % 覆盖率，相差百倍以上。

    但把剪枝放进**解码器**会造成一个更隐蔽的相反错误：任何稠密解一旦
    进入评价就被剪回最小覆盖集（一个 400 站的解会只剩约 15 站），
    于是搜索**永远无法探索高站点数区段**。实测：以 18–432 站的区间
    初始化种群，第 0 代的有效站点数就只剩 13–26——区间初始化形同虚设。

    因此剪枝从解码器移出，改为独立的局部搜索算子
    （``make_pruner``），由 :func:`nsga2` 按概率作用于部分子代。解码器
    只负责补点（保证覆盖）。

    Parameters
    ----------
    prune
        是否在解码时剪枝。**默认 False**："解码"应当只保证可行性，
        改善解的质量是搜索算子的职责，混在一起会让种群多样性被解码器
        单方面抹平。
    """
    cov_core = prob.cov[prob.core_mask]         # (n_core, n_cand)
    n_cand = prob.n_cand
    n_core = len(cov_core)

    # 同时施加成本惩罚偏好：同等覆盖增益时优先选便宜的候选
    cost_norm = prob.cost / max(prob.cost.max(), 1.0)

    def repair(x: np.ndarray) -> np.ndarray:
        x = (np.asarray(x) > 0.5).copy()
        if n_core == 0:
            return x.astype(np.float64)

        covered = cov_core[:, x].any(axis=1) if x.any() else np.zeros(n_core, dtype=bool)

        # ---- 补点：贪心覆盖未覆盖的核心需求 ------------------------------
        for _ in range(n_cand):
            uncovered = ~covered
            if not uncovered.any():
                break
            gains = cov_core[uncovered].sum(axis=0).astype(np.float64)
            gains[x] = -1                       # 已选中的不再考虑
            score = gains - 1e-6 * cost_norm
            j = int(np.argmax(score))
            if gains[j] <= 0:
                break                           # 无法进一步改进
            x[j] = True
            covered |= cov_core[:, j]

        # ---- 剪枝：移除对覆盖无贡献的站点，优先移除最贵的 ---------------
        if prune:
            # 每个核心单元被多少个已选站点覆盖；为 1 时该站点不可移除
            sel = np.flatnonzero(x)
            if len(sel) > 1:
                cov_count = cov_core[:, sel].sum(axis=1)
                # 只被一个站点覆盖的核心单元，其站点是"关键"的
                critical = np.zeros(len(sel), dtype=bool)
                single = cov_count == 1
                if single.any():
                    owner = np.argmax(cov_core[single][:, sel], axis=1)
                    critical[np.unique(owner)] = True

                # 非关键站点按成本降序尝试移除（先删贵的）
                removable = sel[~critical]
                if len(removable):
                    order = removable[np.argsort(-prob.cost[removable])]
                    drop = []
                    for j in order:
                        # 移除后是否仍有核心单元失去唯一覆盖？
                        others = [k for k in sel if k != j and x[k] and k not in drop]
                        if not others:
                            break
                        # 该站点覆盖的核心单元是否都被其他站点覆盖
                        covered_by_j = cov_core[:, j]
                        if covered_by_j.any():
                            rest = cov_core[:, others].any(axis=1)
                            if np.all(rest[covered_by_j]):
                                drop.append(j)
                    if drop:
                        x[drop] = False

        return x.astype(np.float64)

    return repair


def make_pruner(prob: SitingProblem):
    """构造**局部搜索**算子：剪除对覆盖无贡献的站点。

    与 :func:`make_repair` 的区别在于用途：``make_repair`` 是解码器
    （保证可行性，只加点），本函数是搜索算子（改善质量，只删点）。
    把二者分开是必要的——混在解码器里会让任何稠密解在进入评价的瞬间
    就被剪回最小覆盖集，高站点数区段因此无法被探索。

    规则：按成本降序考察已选站点，若移除后**不增加未覆盖的核心需求**，
    则移除。复杂度 ``O(n_selected × n_core)``，对数百站的解也在毫秒级。
    """
    cov_core = prob.cov[prob.core_mask]
    n_core = len(cov_core)

    def prune(x: np.ndarray) -> np.ndarray:
        x = (np.asarray(x) > 0.5).copy()
        sel = np.flatnonzero(x)
        if n_core == 0 or len(sel) <= 1:
            return x.astype(np.float64)

        cov_count = cov_core[:, sel].sum(axis=1)
        critical = np.zeros(len(sel), dtype=bool)
        single = cov_count == 1
        if single.any():
            owner = np.argmax(cov_core[single][:, sel], axis=1)
            critical[np.unique(owner)] = True

        removable = sel[~critical]
        if len(removable) == 0:
            return x.astype(np.float64)

        order = removable[np.argsort(-prob.cost[removable])]
        keep = list(sel)
        for j in order:
            others = [k for k in keep if k != j]
            if not others:
                break
            covered_by_j = cov_core[:, j]
            if covered_by_j.any() and np.all(cov_core[:, others].any(axis=1)[covered_by_j]):
                keep.remove(j)
                x[j] = False
        return x.astype(np.float64)

    return prune


def initial_genotypes(
    n_pop: int,
    n_var: int,
    expected_sites: int,
    rng: np.random.Generator,
    jitter: float = 0.35,
    exact: bool = False,
    site_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """生成**稀疏**初始种群。

    为什么不能直接用 ``rng.random()``
    ---------------------------------
    对每个基因独立取均匀 [0,1] 再按 0.5 二值化，期望密度是 **50%**。
    在 1315 个候选的算例上这意味着每个个体开出约 657 个起降场——比最优解
    高两个数量级。而修复算子只加点不删点（见 :func:`make_repair`），
    搜索因此被永久锁在高成本区：实测该配置下前沿成本落在 **329–375 亿元**，
    且一百代内超体积一次也没有改进（`scripts/25_probe_dense_decoder_defect.py`）。

    本函数直接按目标规模生成初始解：随机挑选 ``expected_sites`` 个基因
    取靠近 1 的连续值，其余取靠近 0 的值。这样既保持了连续基因型
    （SBX 与多项式变异仍然有效），又让种群从合理的规模出发。

    Parameters
    ----------
    expected_sites
        期望的初始站点数。默认按整数规划解的量级推断（见调用处）。
    jitter
        取值区间宽度。0.35 表示选中基因取 U(0.60, 0.95)、未选中取
        U(0.05, 0.40)，与 0.5 阈值留出余量，使变异能在两个状态间切换。
    """
    # exact=True 用于**固定规模模式**：每个个体的站点数恰好为 expected_sites。
    # 这与"以某概率独立选取每个基因"不同——后者个体间规模会波动，
    # 与保规模交叉算子（子代恰含 N 个站）的前提不符，会让杂交语义失效。
    if site_range is not None and not exact:
        # **站点数区间初始化**：个体规模在 [lo, hi] 上按对数均匀抽取。
        #
        # 为什么必须这样
        # --------------
        # 修复算子只能加点、不能删点。若所有个体都从同一个稀疏规模出发
        # （例如 sqrt(n_var)），搜索就**永远到不了高站点数区域**——实测
        # 我的前沿全部落在 20–30 站、成本 4.9–9.2 亿元，而 pymoo 用均匀
        # 随机初始化（密度 50%）能覆盖 400+ 站、成本 226–350 亿元的区域。
        # 两者其实是帕累托前沿的两个不同区段，但前沿质量指标（超体积）
        # 奖励覆盖面，于是窄区间的初始化被判为"差 3 倍"。
        #
        # 按对数均匀抽取使初始种群的**规模谱**覆盖全区间，搜索得以在两个
        # 区段之间铺开。取对数是因为站点数的合理范围跨越一个数量级
        # （十几站到几百站），线性抽样会把绝大多数个体堆在高数端。
        lo, hi = int(site_range[0]), int(site_range[1])
        lo = max(1, min(lo, n_var))
        hi = max(lo, min(hi, n_var))
        ks = np.exp(rng.uniform(np.log(lo), np.log(hi + 1), size=n_pop))
        ks = np.clip(np.round(ks).astype(int), lo, hi)
        X = rng.uniform(0.05, 0.40, size=(n_pop, n_var))
        for i in range(n_pop):
            k = int(ks[i])
            idx = rng.choice(n_var, size=k, replace=False)
            X[i, idx] = rng.uniform(0.60, 0.95, size=k)
        return X

    k = int(np.clip(expected_sites, 1, n_var))
    X = rng.uniform(0.05, 0.40, size=(n_pop, n_var))
    for i in range(n_pop):
        idx = rng.choice(n_var, size=k, replace=False)
        X[i, idx] = rng.uniform(0.60, 0.95, size=k)
    return X


# ---------------------------------------------------------------------------
# 目标函数包装（供 pymoo 对照使用）
# ---------------------------------------------------------------------------

def problem_for_pymoo(prob: SitingProblem):
    """把问题包装成 pymoo 的 ``Problem``，用于算法实现的交叉验证。

    pymoo 的 NSGA-II 与本项目自实现版本在相同算例上应给出统计上可比的
    前沿质量。若两者差异显著，说明自实现存在缺陷——这是论文附录中值得
    报告的一个验证步骤。
    """
    try:
        from pymoo.core.problem import Problem
    except ImportError:
        return None

    evaluator = make_evaluator(prob)

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=prob.n_cand, n_obj=4, xl=0.0, xu=1.0, vtype=float)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = evaluator(X)

    return _P()


# ---------------------------------------------------------------------------
# 决策辅助：从帕累托前沿选出推荐方案
# ---------------------------------------------------------------------------

def knee_point(F: np.ndarray) -> int:
    """用**凸包超平面法**从帕累托前沿中选出推荐方案（拐点）。

    ⚠️ 一个会静默失效的朴素做法
    ----------------------------
    常见的简化实现是"归一化后取到『理想点—最低点』对角线的最大垂距"。
    **这个做法在目标数 ≥ 3 时会退化为选角点**：理想点取全 0、最低点取
    全 1 时，前沿的**端点**（如 (0,0,0,1)）到该对角线的垂距恒为
    ``sqrt(m-1)/sqrt(m)``——对 4 目标是 0.87——而中部解只有 0.3 左右。
    于是"找拐点"变成"找最贵的那个解"。实测成都算例中该方法选出了
    成本最高的端点解（191.65 亿元 / 362 站），显然不是折中。

    正确的定义
    ----------
    拐点是**前沿上"边际收益递减"转折最大**的解。标准做法是：取每个目标
    方向上的极值解（共 ``m`` 个），由它们张成一个超平面；前沿中距该超平面
    最远（凸包外方向）的解即拐点。二维下这退化为"距两端连线最远的点"，
    与经典做法一致。

    Returns
    -------
    前沿内的行索引。
    """
    F = np.atleast_2d(F)
    n, m = F.shape
    if n == 0:
        raise ValueError("帕累托前沿为空")
    if n <= m:
        return 0

    lo, hi = F.min(axis=0), F.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Fn = (F - lo) / span

    # 各目标方向上的极值解：构成超平面的 m 个支撑点
    ext = np.array([Fn[np.argmin(Fn[:, k])] for k in range(m)])

    # 超平面法向：由极值点张成的仿射子空间的正交补
    A = ext[1:] - ext[0]                      # (m-1, m)
    try:
        _u, _sv, vt = np.linalg.svd(A)
        normal = vt[-1]                       # 最小奇异向量 = 法向
    except np.linalg.LinAlgError:
        return 0

    ref = ext.mean(axis=0)
    signed = (Fn - ref) @ normal

    # 拐点 = 偏离"极值点所张超平面"**最远**的解，**不分侧**。
    #
    # 为什么不能按"理想点在哪一侧"来定符号：前沿是凸还是凹决定了
    # 它偏向哪一侧，而两种情形都合法，符号约定因此不稳健。自检暴露过：
    # 凹前沿 y=(1-x)^0.35 上，按理想点定符号会选出索引 0（端点）；
    # 凸前沿 y=(1-x)^3 上同样规则的取值方向又相反。
    #
    # 取绝对值则统一了两种情形，与二维下"距两端连线最远的点"这一
    # 经典拐点定义一致。
    return int(np.argmax(np.abs(signed)))


def topsis_ranking(F: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """TOPSIS 多准则排序（论文中的决策方法对照）。

    与拐点法互补：拐点法给出"折中最好"的单个方案，TOPSIS 在给定权重
    偏好下对全部前沿解排序。两种方法结论一致时可增强推荐方案的说服力。

    Parameters
    ----------
    weights
        各目标权重（越小越重要的目标应给越大权重）。默认等权。
    """
    F = np.atleast_2d(F).astype(np.float64)
    m, n = F.shape
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # 归一化 -> 加权 -> 理想解/负理想解（全部最小化）
    norm = np.sqrt((F ** 2).sum(axis=0))
    norm = np.where(norm > 1e-12, norm, 1.0)
    V = (F / norm) * weights

    ideal = V.min(axis=0)
    nadir = V.max(axis=0)
    d_pos = np.linalg.norm(V - ideal, axis=1)
    d_neg = np.linalg.norm(V - nadir, axis=1)
    cc = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    return np.argsort(-cc)      # 贴近度越大越优


def objective_table(F: np.ndarray, cfg) -> pd.DataFrame:
    """把目标矩阵转成带列名的表，便于写 CSV 与制表。"""
    specs = cfg.get("stage4_nsga2.objectives", [])
    cols = [o.get("key", f"f{i}") for i, o in enumerate(specs)] or \
           [f"f{i}" for i in range(F.shape[1])]
    df = pd.DataFrame(F, columns=cols)
    if "total_cost" in df.columns:
        df["total_cost_cny"] = df["total_cost"].round(0)
    return df
