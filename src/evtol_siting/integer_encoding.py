"""变长整数编码（基因 = 起降场编号）的 NSGA-II 对照实现。

为什么需要它
------------
申报书把编码表述为"以起降场编号为基因进行整数编码"，而本实现采用的是
**等价的 0/1 掩码编码**（``stage4_nsga2``）。代码注释与论文 §4.5.1 都写了
"整数编码作为配置开关保留以供对照"——但审计发现**这个开关根本不存在**：
没有任何代码读取 ``encoding`` 配置项，也没有任何一次整数编码的运行。

于是一句"二者在解空间上等价"的断言，既没有实现支撑、也没有实验支撑。
本模块补上实现，使这句断言可以被**检验**而不是被声明。

两种编码的关系
--------------
设候选集 ``M``、设施上限 ``K``。掩码编码的解是 ``x ∈ {0,1}^|M|``；
整数编码的解是**去重的编号序列** ``g = (g_1, …, g_k)``，``g_i ∈ M``，``k ≤ K``。

二者张成的是**同一个解集合**（"选中哪些站"），差别只在：

* 掩码编码的搜索算子（SBX + 多项式变异）作用在连续松弛上，会改变选中的
  站点数；整数编码的算子直接在集合上做交/并/替换，**站点数变化是显式的**。
* 整数编码天然避免"同一编号出现两次"的冗余基因，但需要去重与定长填充，
  也就是变长编码的经典代价。

因此"等价"指的是**解空间**等价，不是**搜索行为**等价——后者正是本对照要
测的东西：同一评估预算下，两种编码到达的前沿质量是否有实质差别。

实现说明
--------
基因型用**定长数组** ``(pop, K)``（``-1`` 表示空槽）而非 Python 列表，
以便复用 :mod:`nsga2_core` 的非支配排序、拥挤距离与锦标赛选择——
这些选择机制与编码无关，必须完全一致，否则比出来的差异分不清是编码
造成的还是选择压力造成的。
"""

from __future__ import annotations

import logging

import numpy as np

from .nsga2_core import (
    binary_tournament,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
)

logger = logging.getLogger(__name__)


def decode(genes: np.ndarray, n_cand: int) -> np.ndarray:
    """整数基因型 -> 0/1 掩码。重复编号自动去重。"""
    x = np.zeros(n_cand, dtype=np.float64)
    idx = genes[genes >= 0]
    if idx.size:
        x[np.unique(idx)] = 1.0
    return x


def decode_pop(G: np.ndarray, n_cand: int) -> np.ndarray:
    return np.array([decode(g, n_cand) for g in G])


def _normalize(genes: np.ndarray, max_sites: int, n_cand: int,
               rng: np.random.Generator) -> np.ndarray:
    """整理基因型：去重、截断到 ``max_sites``、右补 ``-1``。

    去重是整数编码的必需步骤——交叉的并集与变异的替换都可能产生重复编号，
    而重复编号在表型上是同一个解，不去重会白白浪费评估预算并让种群退化。
    """
    out = np.full(max_sites, -1, dtype=np.int64)
    seen = set()
    w = 0
    for g in genes:
        gi = int(g)
        if gi < 0 or gi >= n_cand or gi in seen:
            continue
        seen.add(gi)
        out[w] = gi
        w += 1
        if w >= max_sites:
            break
    return out


def random_individual(n_cand: int, max_sites: int, k: int,
                      rng: np.random.Generator) -> np.ndarray:
    k = max(1, min(k, max_sites, n_cand))
    return _normalize(rng.choice(n_cand, size=k, replace=False),
                      max_sites, n_cand, rng)


def set_crossover(pa: np.ndarray, pb: np.ndarray, max_sites: int,
                  n_cand: int, rng: np.random.Generator) -> np.ndarray:
    """集合交叉：取父代并集，再随机下采样到父代规模之间。

    这是变长集合编码的标准算子（"union then reduce"）。它保持了
    "子代是两个布局方案的组合"这层语义，且站点数的变化幅度受父代约束，
    不会像 SBX 作用在掩码上那样随机漂移。
    """
    ua = set(int(g) for g in pa if g >= 0)
    ub = set(int(g) for g in pb if g >= 0)
    union = sorted(ua | ub)
    if not union:
        return random_individual(n_cand, max_sites, 1, rng)
    lo = max(1, min(len(ua), len(ub)) if ua and ub else 1)
    hi = min(max_sites, len(union), max(len(ua), len(ub)) * 2)
    hi = max(hi, lo)
    k = int(rng.integers(lo, hi + 1))
    pick = rng.choice(len(union), size=min(k, len(union)), replace=False)
    return _normalize(np.array([union[i] for i in pick], dtype=np.int64),
                      max_sites, n_cand, rng)


def set_mutation(genes: np.ndarray, n_cand: int, max_sites: int,
                 rng: np.random.Generator, p: float) -> np.ndarray:
    """集合变异：以概率 ``p`` 对某个基因做"换成未选站点"或"加入/删除"。"""
    g = genes.copy()
    if rng.random() >= p:
        return g
    sel = set(int(v) for v in g if v >= 0)
    nonsel = np.setdiff1d(np.arange(n_cand), np.array(sorted(sel), dtype=np.int64),
                          assume_unique=False)
    if rng.random() < 0.5 and len(sel) > 1:
        # 删除一个已选站点
        drop = int(rng.choice(np.array(sorted(sel))))
        g[g == drop] = -1
    elif len(nonsel) and len(sel) < max_sites:
        # 加入一个未选站点
        new = int(rng.choice(nonsel))
        slots = np.flatnonzero(g < 0)
        if slots.size:
            g[slots[0]] = new
        else:
            g[int(rng.integers(0, len(g)))] = new
    elif len(nonsel):
        # 替换一个已选站点
        slots = np.flatnonzero(g >= 0)
        if slots.size:
            g[slots[int(rng.integers(0, len(slots)))]] = int(rng.choice(nonsel))
    return _normalize(g, max_sites, n_cand, rng)


def integer_nsga2(
    evaluate_mask,
    n_cand: int,
    max_sites: int,
    cfg,
    pop_size: int = 100,
    n_gen: int = 100,
    seed: int | None = None,
    verbose: bool = False,
    ref: np.ndarray | None = None,
):
    """整数编码的 NSGA-II。返回对象与 :func:`nsga2_core.nsga2` 同构。

    选择机制（非支配排序 + 拥挤距离 + 二元锦标赛）与掩码编码**完全一致**，
    只有变异/交叉算子不同——否则测的就不是编码的差别。
    """
    from .metrics import hypervolume_2d_proxy
    from .nsga2_core import NSGA2Result

    ng = cfg.get("stage4_nsga2", {}) if hasattr(cfg, "get") else {}
    p_cross = float(ng.get("crossover_prob", 0.9))
    # ⚠ `mutation_prob` 在配置里是字符串 "auto"（意为 1/n_var），不能直接 float()。
    #   nsga2_core 里有这段处理，这里当初漏了——于是编码消融一跑就
    #   `ValueError: could not convert string to float: 'auto'`。
    #   两臂的变异率必须取**同一规则**，否则比的就不是编码而是变异强度。
    m_prob = ng.get("mutation_prob", "auto")
    p_mut = 1.0 / n_cand if str(m_prob) == "auto" else float(m_prob)
    rng = np.random.default_rng(
        seed if seed is not None else cfg.get("project.random_seed", 42))

    init_k = max(2, int(np.sqrt(n_cand)))
    G = np.array([random_individual(n_cand, max_sites, init_k, rng)
                  for _ in range(pop_size)], dtype=np.int64)
    X = decode_pop(G, n_cand)
    F = np.atleast_2d(evaluate_mask(X))
    evaluated = len(X)

    hv_ref = ref if ref is not None else (F.max(axis=0) * 1.1 + 1e-9)
    history = {"hypervolume": [], "n_pareto": [], "best": []}

    for gen in range(n_gen):
        fronts = fast_non_dominated_sort(F)
        rank = np.empty(len(F), dtype=int)
        crowd = np.zeros(len(F), dtype=np.float64)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding_distance(F, fr)

        Q = np.empty_like(G)
        for i in range(0, pop_size, 2):
            a = binary_tournament(rank, crowd, rng)
            b = binary_tournament(rank, crowd, rng)
            c1 = (set_crossover(G[a], G[b], max_sites, n_cand, rng)
                  if rng.random() < p_cross else G[a].copy())
            c2 = (set_crossover(G[a], G[b], max_sites, n_cand, rng)
                  if rng.random() < p_cross else G[b].copy())
            c1 = set_mutation(c1, n_cand, max_sites, rng, p_mut)
            c2 = set_mutation(c2, n_cand, max_sites, rng, p_mut)
            Q[i] = c1
            if i + 1 < pop_size:
                Q[i + 1] = c2

        Qx = decode_pop(Q, n_cand)
        # 表型去重：不同基因型可能解码成同一方案，只保留一份
        Qx, uniq = np.unique(Qx, axis=0, return_index=True)
        Q = Q[uniq]
        Fq = np.atleast_2d(evaluate_mask(Qx))
        evaluated += len(Qx)

        Rg = np.vstack([G, Q])
        Rx = np.vstack([X, Qx])
        Fr = np.vstack([F, Fq])
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
        G, X, F = Rg[new_idx], Rx[new_idx], Fr[new_idx]

        pf = F[fast_non_dominated_sort(F)[0]]
        history["hypervolume"].append(hypervolume_2d_proxy(pf, ref=hv_ref))
        history["n_pareto"].append(len(pf))
        history["best"].append(float(np.min(F[:, 0])))
        if verbose and (gen % max(1, n_gen // 10) == 0 or gen == n_gen - 1):
            logger.info("  [整数编码] 第 %3d 代: 前沿 %3d 个解, HV=%.4g",
                        gen, len(pf), history["hypervolume"][-1])

    fronts = fast_non_dominated_sort(F)
    pf_idx = fronts[0]
    history["n_evaluations"] = evaluated
    # NSGA2Result 的前两个字段是**最终种群**（X, F），pareto_* 才是前沿切片。
    # 只传 pareto_* 会抛 TypeError: missing 2 required positional arguments。
    return NSGA2Result(
        X=X, F=F, pareto_X=X[pf_idx], pareto_F=F[pf_idx], history=history,
    )
