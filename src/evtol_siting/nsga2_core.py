"""NSGA-II 算法核心（从零实现）。

对应申报书"负责 NSGA-II 算法的 Python 编码与调试"。

为什么自己实现而不直接用 pymoo
------------------------------
1. 申报书明确把 NSGA-II 编码列为项目任务，从零实现可完整掌握算法细节，
   在答辩中可以解释每一个算子；
2. 本研究需要**带可行性修复的约束处理**（覆盖约束），pymoo 的
   约束处理框架对这类"结构性约束"支持不够直接；
3. 便于把本实现与 pymoo 的 NSGA-II 在相同算例上对照，作为**算法实现的
   正确性验证**（见 ``scripts/06_nsga2.py --validate``），这本身是论文中
   一个增强可信度的细节。

实现依据
--------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
*IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

三个核心部件
------------
* :func:`fast_non_dominated_sort` —— 快速非支配排序，``O(MN²)``
* :func:`crowding_distance`       —— 拥挤度距离（保持解集多样性）
* :func:`sbx_crossover` / :func:`polynomial_mutation` —— 实数编码算子
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 非支配排序
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """判断解 ``a`` 是否支配 ``b``（所有目标为最小化）。

    ``a`` 支配 ``b`` 当且仅当 ``a`` 在所有目标上不劣于 ``b``，且至少在一个
    目标上严格优于 ``b``。
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """快速非支配排序。

    Parameters
    ----------
    F
        ``(N, M)`` 目标值矩阵，**所有目标均为最小化**。

    Returns
    -------
    前沿列表，``fronts[0]`` 为第一层（帕累托前沿），依此类推。
    每个元素是该层解的索引数组。
    """
    n = len(F)
    if n == 0:
        return []

    # domination count 与支配集
    S: list[list[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        # 向量化：p 是否支配每个 q
        le = np.all(F[p] <= F, axis=1)
        lt = np.any(F[p] < F, axis=1)
        p_dominates = le & lt          # p 支配 q
        q_dominates_p = np.all(F <= F[p], axis=1) & np.any(F < F[p], axis=1)

        n_dom[p] = int(q_dominates_p.sum())
        S[p] = np.flatnonzero(p_dominates).tolist()

        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()   # 最后一个空层
    return [np.asarray(f, dtype=int) for f in fronts]


def crowding_distance(F: np.ndarray, front: np.ndarray) -> np.ndarray:
    """计算一个前沿内各解的拥挤度距离。

    边界解（每个目标的极值）赋予无穷大距离，保证其总被保留。
    物理含义：该解在目标空间中与最近邻解的平均间距，越大表示越"孤立"，
    越应当保留以维持帕累托前沿的分布均匀性。
    """
    n = len(front)
    dist = np.zeros(n, dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf)

    f = F[front]
    for m in range(f.shape[1]):
        order = np.argsort(f[:, m], kind="stable")
        vals = f[order, m]
        rng = vals[-1] - vals[0]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if rng <= 1e-12:
            continue
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / rng
    return dist


# ---------------------------------------------------------------------------
# 遗传算子
# ---------------------------------------------------------------------------

def sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator,
    prob: float = 1.0, bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（Simulated Binary Crossover, SBX）。

    SBX 在实数编码下模拟单点交叉的"父代基因平均传递"性质，是 NSGA-II 的
    标准交叉算子。``eta`` 控制子代靠近父代的倾向：``eta`` 越大，子代越接近
    父代（探索性越弱）。Deb 建议 ``eta_c = 20``。
    """
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2

    lo, hi = bounds
    u = rng.random(len(p1))
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def polynomial_mutation(
    x: np.ndarray, eta: float, rng: np.random.Generator,
    prob: float = 1.0, bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """多项式变异（Polynomial Mutation）。

    相比高斯变异，多项式变异的扰动幅度随变量接近边界而自适应收缩，
    不会产生越界解，是实数编码 NSGA-II 的配套算子。
    """
    lo, hi = bounds
    y = x.copy()
    n = len(y)
    if n == 0:
        return y

    mutate = rng.random(n) <= prob
    if not mutate.any():
        return y

    idx = np.flatnonzero(mutate)
    u = rng.random(len(idx))
    delta = np.where(
        u < 0.5,
        (2 * u) ** (1.0 / (eta + 1.0)) - 1.0,
        1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0)),
    )
    span = hi - lo
    y[idx] = y[idx] + delta * span
    return np.clip(y, lo, hi)


# 保规模算子：从 schemes 模块导入。放在这里是为了让 nsga2() 的主体不必
# 关心算子实现在哪——固定规模模式与自由规模模式对调用方是透明的。
from .schemes import crossbreed_masks as _crossbreed, swap_mutate as _swap_mutate  # noqa: E402


def binary_tournament(
    rank: np.ndarray, crowd: np.ndarray, rng: np.random.Generator, k: int = 2
) -> int:
    """二元锦标赛选择。

    先比非支配层级（越小越好），层级相同再比拥挤度（越大越好）——
    这正是 NSGA-II "优劣 + 多样性"双重选择压力的体现。
    """
    idx = rng.integers(0, len(rank), size=k)
    best = idx[0]
    for i in idx[1:]:
        if rank[i] < rank[best] or (rank[i] == rank[best] and crowd[i] > crowd[best]):
            best = i
    return int(best)


# ---------------------------------------------------------------------------
# 主算法
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Result:
    """NSGA-II 运行结果。"""
    X: np.ndarray                       # 最终种群的决策变量
    F: np.ndarray                       # 对应的目标值
    pareto_X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pareto_F: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    history: dict = field(default_factory=dict)

    @property
    def n_pareto(self) -> int:
        return len(self.pareto_F)


def nsga2(
    evaluate: Callable[[np.ndarray], np.ndarray],
    n_var: int,
    cfg,
    pop_size: int | None = None,
    n_gen: int | None = None,
    repair: Callable[[np.ndarray], np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    verbose: bool = True,
    init_sites: int | None = None,
    n_sites_fixed: int | None = None,
    local_search: Callable[[np.ndarray], np.ndarray] | None = None,
    local_search_prob: float = 0.15,
) -> NSGA2Result:
    """运行 NSGA-II。

    Parameters
    ----------
    evaluate
        ``f(X) -> F``，``X`` 形状 ``(n_var,)`` 或 ``(N, n_var)``，
        ``F`` 形状 ``(N, n_obj)``，**全部最小化**。
    n_var
        决策变量维度（候选起降场数量，采用 0/1 掩码编码）。
    repair
        可选的可行性修复算子 ``g(x) -> x'``。覆盖约束是"结构性"的：
        一个不满足全覆盖的解并非不可行到无法使用，而是表达能力差。
        修复（贪心补点）能显著加快收敛，是本实现的关键工程优化。
    x0
        可选的初始种群（形状 ``(N, n_var)``）。用于注入启发式解
        （如 IP 最优解）以加速收敛。

    Returns
    -------
    :class:`NSGA2Result`，``pareto_F`` 为第一层非支配前沿。
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    ng = cfg.get("stage4_nsga2", {})
    pop_size = int(pop_size or ng.get("pop_size", 200))
    n_gen = int(n_gen or ng.get("n_gen", 300))
    eta_c = float(ng.get("eta_c", 20.0))
    eta_m = float(ng.get("eta_m", 20.0))
    p_cross = float(ng.get("crossover_prob", 0.9))

    m_prob = ng.get("mutation_prob", "auto")
    p_mut = 1.0 / n_var if str(m_prob) == "auto" else float(m_prob)

    # -- 初始种群 ----------------------------------------------------------
    # ⚠️ 初始稀疏度至关重要，且不能用 `rng.random()`。
    # 对每个基因独立取均匀 [0,1] 再按 0.5 二值化，期望密度是 50%——在
    # 1315 个候选的成都算例上即每个个体开出约 657 个起降场，比最优解高
    # 两个数量级。由于修复算子只加点不删点，搜索会被永久锁在高成本区。
    #
    # 实测（`scripts/25_probe_dense_decoder_defect.py`，当前 1315 候选算例）：
    # 初始每体 613–706 站，一百代内超体积**一次也没有改进**
    # （HV 3.871e+06 -> 3.871e+06），整个帕累托前沿的成本落在
    # **329–375 亿元**，而整数规划用 11 站、3.14 亿元就达到 96.13% 覆盖率。
    #
    # ⚠ 该注释此前写的是"1332 个候选……316–384 亿元 / 11 站 4.56 亿元 / 97%"，
    #   那是 A6 图层修复与坡度判据生效**之前**的算例。论文正文当时引用了
    #   这半句，却把整数规划那半句更新成了新口径，于是一句话里两个数字来自
    #   两个算例。现按当前算例重跑并同步。
    # 因此按目标规模**稀疏**初始化，默认取 sqrt(n_var)（对 1315 个候选
    # 约 36 个站），并在调用方可用 init_sites 覆盖。
    # ``n_sites_fixed`` 非空时进入**固定规模模式**：所有个体恰好含 N 个站，
    # 交叉与变异均为保规模算子。此时 init_sites 必须等于 N。
    cardinality_fixed = n_sites_fixed is not None
    if cardinality_fixed:
        init_sites = int(n_sites_fixed)
    elif init_sites is None:
        init_sites = max(5, int(np.sqrt(n_var)))

    from .stage4_nsga2 import initial_genotypes

    # 自由规模模式下用**区间初始化**（见 initial_genotypes 的说明）：
    # 若初始种群规模单一，而修复算子只加点不删点，搜索会被锁在
    # 一个窄的站点数区间里，永远到不了另一个区段。
    site_range = None
    if not cardinality_fixed:
        ng2 = cfg.get("stage4_nsga2", {}) or {}
        sr = ng2.get("init_site_range")
        if sr:
            site_range = (int(sr[0]), int(sr[1]))
        else:
            site_range = (max(5, init_sites // 2), max(20, init_sites * 12))

    X = initial_genotypes(pop_size, n_var, init_sites, rng,
                          exact=cardinality_fixed, site_range=site_range)
    logger.info(
        "初始种群: %d 个个体 × %d 维，站点数 %s（%s）",
        pop_size, n_var,
        f"恒为 {init_sites}" if cardinality_fixed else f"区间 {site_range}",
        "固定规模模式，保规模算子" if cardinality_fixed else "区间初始化，自由规模",
    )
    if x0 is not None:
        x0 = np.atleast_2d(np.asarray(x0, dtype=np.float64))
        n_inject = min(len(x0), pop_size)
        X[:n_inject] = x0[:n_inject]

    def _binarize(Z: np.ndarray) -> np.ndarray:
        return (Z > 0.5).astype(np.float64)

    def _phenotype(Z: np.ndarray) -> np.ndarray:
        """连续基因型 -> 二进制表型（含可行性修复 + 可选局部搜索）。

        这是"解码"步骤，**只用于求值**，不写回基因型。两者的分离是
        必要的：SBX 与多项式变异是为连续空间设计的算子，若把二值化后的
        表型存回种群，连续基因型每代都被抹掉，两个算子退化为在 {0,1} 上
        的低效扰动，搜索能力大幅下降（表现为帕累托前沿从一开始就不再改进）。

        ``local_search``（剪枝）**按概率**作用于子代，而不是无条件执行。
        无条件执行会把任何稠密解瞬间剪回最小覆盖集，使高站点数区段
        无法被探索——这与"修复算子只加点"恰好构成一对相反的错误。
        按概率作用则在保留该区段的同时仍能改善解的质量。
        """
        Zb = _binarize(Z)
        if repair is not None:
            Zb = np.array([repair(z) for z in Zb])
        # 固定规模模式**必须忽略**剪枝：它只删点，会破坏"恰含 N 个站"的
        # 前提。这里加一道防线，避免调用方忘记关闭而静默产生错误结果。
        if (local_search is not None and local_search_prob > 0
                and not cardinality_fixed):
            hit = rng.random(len(Zb)) < local_search_prob
            for i in np.flatnonzero(hit):
                Zb[i] = local_search(Zb[i])
        return Zb

    history = {"hypervolume": [], "n_pareto": [], "best": []}
    evaluated = 0

    def _evaluate_pop(Z: np.ndarray):
        """求值：返回 (去重后的表型, 目标值)，**不修改基因型**。"""
        nonlocal evaluated
        Zb = _phenotype(Z)
        Zb, uniq_idx = np.unique(Zb, axis=0, return_index=True)
        Fv = np.atleast_2d(evaluate(Zb))
        evaluated += len(Zb)
        return Zb, Fv, uniq_idx

    # -- 初代 --------------------------------------------------------------
    # 三个数组保持**行对齐**：
    #   P  —— 连续基因型（SBX/变异作用的对象）
    #   Pp —— 二进制表型（修复后的实际选址方案，求值对象）
    #   Fp —— 对应的目标值
    # 基因型与表型分离是本实现的关键：修复算子会改写表型，但不应污染
    # 基因型，否则多种可行方案会被压缩成同一个基因点，多样性提前枯竭。
    P = X.copy()
    Pp = _phenotype(P)
    Fp = np.atleast_2d(evaluate(Pp))
    evaluated += len(Pp)

    # 去重后若不足 pop_size，补随机个体（保持基因型连续）
    while len(P) < pop_size:
        need = pop_size - len(P)
        extra_geno = rng.random((need, n_var))
        extra_pheno = _phenotype(extra_geno)
        P = np.vstack([P, extra_geno])
        Pp = np.vstack([Pp, extra_pheno])
        Fp = np.vstack([Fp, np.atleast_2d(evaluate(extra_pheno))])
        evaluated += len(extra_pheno)
    P, Pp, Fp = P[:pop_size], Pp[:pop_size], Fp[:pop_size]

    from .metrics import hypervolume_2d_proxy

    base_Fp = Fp.copy()          # 初代前沿基线，用于诊断改进幅度

    # 收敛监控用**固定参考点**：在第 0 代确定后全程不变。
    # 若每代都用当前前沿的最大值作参考，前沿改善时参考点会收缩，
    # 算出的超体积反而下降，曲线非单调——看上去像算法在退化。
    hv_ref = (Fp[:, :2].max(axis=0) * 1.1 + 1e-9) if Fp.shape[1] >= 2 else None

    for gen in range(n_gen):
        fronts = fast_non_dominated_sort(Fp)
        rank = np.empty(len(P), dtype=int)
        crowd = np.empty(len(P), dtype=np.float64)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding_distance(Fp, fr)

        # -- 生成子代（在**连续基因型**上做交叉与变异）----------------------
        n_pop = len(P)
        if n_pop == 0:
            logger.error("种群为空，终止迭代")
            break

        Q = np.empty_like(P)
        for i in range(0, n_pop, 2):
            a = binary_tournament(rank, crowd, rng)
            b = binary_tournament(rank, crowd, rng)

            if cardinality_fixed:
                # 固定规模模式：用**方案杂交**算子。
                # 这里不再用 SBX——它是为连续空间设计的，作用在掩码上会
                # 让子代的站点数随机漂移，从而丢失"杂交两个规模相同的
                # 布局方案"这层语义。改用从父代并集中抽取 N 个站点。
                c1 = _crossbreed(P[a], P[b], init_sites, rng)
                c2 = _crossbreed(P[a], P[b], init_sites, rng)
                if rng.random() < p_cross:
                    c1 = _swap_mutate(c1, init_sites, rng,
                                      n_swaps=max(1, int(0.02 * init_sites)))
                if rng.random() < p_cross:
                    c2 = _swap_mutate(c2, init_sites, rng,
                                      n_swaps=max(1, int(0.02 * init_sites)))
            else:
                c1, c2 = sbx_crossover(P[a], P[b], eta_c, rng, p_cross)
                c1 = polynomial_mutation(c1, eta_m, rng, p_mut)
                c2 = polynomial_mutation(c2, eta_m, rng, p_mut)

            Q[i] = c1
            if i + 1 < n_pop:
                Q[i + 1] = c2

        Qp = _phenotype(Q)
        # 表型去重：不同基因型可能修复成同一方案，只保留一份以免浪费评估
        Qp, uniq = np.unique(Qp, axis=0, return_index=True)
        Q = Q[uniq]
        Fq = np.atleast_2d(evaluate(Qp))
        evaluated += len(Qp)

        # -- 精英保留：父代 + 子代合并后截断 --------------------------------
        R_geno = np.vstack([P, Q])
        R_pheno = np.vstack([Pp, Qp])
        Fr = np.vstack([Fp, Fq])
        fronts = fast_non_dominated_sort(Fr)

        new_idx: list[int] = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= pop_size:
                new_idx.extend(fr.tolist())
            else:
                cd = crowding_distance(Fr, fr)
                order = np.argsort(-cd, kind="stable")
                need = pop_size - len(new_idx)
                new_idx.extend(fr[order[:need]].tolist())
                break

        P, Pp, Fp = R_geno[new_idx], R_pheno[new_idx], Fr[new_idx]

        # -- 记录收敛过程 --------------------------------------------------
        pf = Fp[fast_non_dominated_sort(Fp)[0]]
        hv = hypervolume_2d_proxy(pf, ref=hv_ref)     # ← 固定参考点
        history["hypervolume"].append(hv)
        history["n_pareto"].append(len(pf))
        history["best"].append(float(np.min(Fp[:, 0])))
        if verbose and (gen % max(1, n_gen // 10) == 0 or gen == n_gen - 1):
            logger.info(
                "  第 %3d 代: 前沿 %3d 个解, HV=%.4g, 最优目标1=%.4g, 累计评估 %d",
                gen, len(pf), hv, np.min(Fp[:, 0]), evaluated,
            )

    fronts = fast_non_dominated_sort(Fp)
    pf_idx = fronts[0]
    # 前沿内按第一目标排序，便于制表与绘图
    order = np.argsort(Fp[pf_idx, 0], kind="stable")

    history["n_evaluations"] = evaluated
    history["hv_initial"] = float(hypervolume_2d_proxy(
        base_Fp[fast_non_dominated_sort(base_Fp)[0]], ref=hv_ref))
    history["hv_final"] = float(history["hypervolume"][-1]) if history["hypervolume"] else None

    hv0, hv1 = history["hv_initial"], history.get("hv_final") or 0.0
    if hv1 <= hv0 * 1.0001:
        logger.warning(
            "帕累托前沿在 %d 代内**没有改进**（HV %.4g -> %.4g）。"
            "可能原因：(1) 初始种群已被 IP 解锚定在最优附近；"
            "(2) 修复算子过强，压制了搜索空间；(3) 目标间无实质权衡。"
            "建议检查目标值分布是否退化。",
            n_gen, hv0, hv1,
        )

    # 决策变量返回**表型**（即实际选址方案）——下游要用它解码选中的起降场
    return NSGA2Result(
        X=Pp, F=Fp,
        pareto_X=Pp[pf_idx[order]], pareto_F=Fp[pf_idx[order]],
        history=history,
    )
