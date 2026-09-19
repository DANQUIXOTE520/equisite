"""帕累托前沿质量指标与方案评价指标。

论文中用于回答两个不同层面的问题：

**1. 算法层面**——"NSGA-II 求得的解集好不好？"
   * :func:`hypervolume`  超体积 HV：综合收敛性与多样性，越大越好
   * :func:`spacing`      间距 SP：解集分布的均匀性，越小越好
   * :func:`spread`       分布广度 Δ：覆盖范围的完整度，越小越好
   * :func:`igd`          反世代距离 IGD：与参考前沿的逼近程度，越小越好

**2. 决策层面**——"这个选址方案好不好？"
   * :func:`solution_metrics`：覆盖率、公平性、成本、可达性
   * :func:`gini_coefficient`：人群间可达性差异的基尼系数

关于 HV 的实现
--------------
HV 的精确计算随目标数指数增长。本研究为 4 目标，精确算法耗时不可忽略，
因此采取分层策略：2 目标精确计算、3 目标切片精确计算、4 目标及以上用
**固定种子的蒙特卡洛估计**（样本量可配置，默认 2×10⁶）。固定种子保证
同一实验的 HV 可复现——这是论文可复现性要求的一部分。
若环境中有 ``pymoo``，优先使用其经过验证的 HV 实现做交叉校验。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:          # 仅用于类型标注，避免运行时硬依赖 pandas
    import pandas as pd

logger = logging.getLogger(__name__)

# 某一类人群**完全未被服务**时，其"接驳时间"取该哨兵值。
#
# 为什么必须是哨兵值而不是把该组丢掉：优化器（``stage4_nsga2._evaluate``）
# 在同样情形下给 f_equity = 1e12，而评价器早期版本会把整组从表里删掉、
# 返回"已服务组里最差的那个"。同一个解在优化里是 1e12、在评价里可能是
# 12.5 分钟，支配关系可被直接翻转。两者必须同口径——见 _evaluate_solution。
UNSERVED_ACCESS_MIN = 1e12


# ---------------------------------------------------------------------------
# 超体积
# ---------------------------------------------------------------------------

def hypervolume_2d_proxy(F: np.ndarray, ref: np.ndarray | None = None) -> float:
    """2 目标投影下的 HV 快速估算，用于 NSGA-II 训练过程中监控收敛。

    取前两个目标（未服务需求、总接驳时间）做 2D 计算。这是一个**代理指标**，
    只用于观察收敛趋势，不能作为论文中的最终 HV 报告值。

    ⚠️ 参考点必须**跨代固定**
    -------------------------
    若每次都用当前前沿自己的最大值作参考点（``ref=None`` 的旧行为），
    前沿一变好、参考点就随之收缩，算出的 HV 反而下降——收敛曲线因此
    非单调，看上去像算法在退化。实测出现过“第 0 代 HV=3.60e8、
    第 99 代 HV=3.57e8”这种误导性记录。

    正确做法是在**第 0 代**确定参考点后全程固定（``nsga2`` 已如此调用）。
    传入 ``ref`` 即使用固定参考点。
    """
    if len(F) == 0:
        return 0.0
    f = F[:, :2] if F.shape[1] >= 2 else np.column_stack([F[:, 0], F[:, 0]])
    if ref is None:
        ref = f.max(axis=0) * 1.1 + 1e-9
    else:
        ref = np.asarray(ref, dtype=np.float64)[:2]
    pts = f[np.all(f <= ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def hypervolume(F: np.ndarray, ref: np.ndarray | None = None, n_samples: int = 2_000_000) -> float:
    """超体积（HV）：解集与参考点围成的目标空间体积。

    Parameters
    ----------
    F
        ``(N, M)`` 目标值矩阵（最小化）。
    ref
        参考点。``None`` 时自动取各目标最大值上浮 10%。
        论文中报告 HV 时必须**固定参考点**，否则不同算法的 HV 不可比。
    n_samples
        4 目标及以上时的蒙特卡洛样本量。

    Returns
    -------
    HV 值。越大越好。

    Notes
    -----
    所有目标都是最小化，参考点必须**支配**所有解（即 ``ref > max(F)``）。
    """
    if len(F) == 0:
        return 0.0
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    F = F[np.all(np.isfinite(F), axis=1)]
    if len(F) == 0:
        return 0.0

    if ref is None:
        ref = F.max(axis=0) * 1.1 + 1e-9
    ref = np.asarray(ref, dtype=np.float64)
    if np.any(ref <= F.min(axis=0)):
        logger.warning("参考点未支配全部解，HV 可能被低估")

    # 只保留被参考点支配且互不支配的解
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    from .nsga2_core import fast_non_dominated_sort

    F = F[fast_non_dominated_sort(F)[0]]

    m = F.shape[1]
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        return _hv_2d(F, ref)
    if m == 3:
        return _hv_3d(F, ref)
    return _hv_monte_carlo(F, ref, n_samples)


def _hv_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """2 目标 HV 精确解（沿第一目标扫描）。"""
    f = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in f:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def _hv_3d(F: np.ndarray, ref: np.ndarray) -> float:
    """3 目标 HV 精确解（沿第三目标切片，每片内做 2D HV）。"""
    hv = 0.0
    zs = np.unique(F[:, 2])
    zs = np.sort(zs)
    prev_z = ref[2]
    for z in zs:
        layer = F[F[:, 2] <= z][:, :2]
        if len(layer) == 0:
            continue
        from .nsga2_core import fast_non_dominated_sort

        layer = layer[fast_non_dominated_sort(layer)[0]]
        hv += _hv_2d(layer, ref[:2]) * (prev_z - z)
        prev_z = z
        # 只保留 z 以上的点，减少重复计算
        F = F[F[:, 2] > z]
        if len(F) == 0:
            break
    return float(hv)


def _hv_monte_carlo(F: np.ndarray, ref: np.ndarray, n_samples: int) -> float:
    """蒙特卡洛 HV 估计（4 目标及以上）。

    在参考点围成的超矩形内均匀采样，统计落入解集支配域的样本比例，
    乘以超矩形体积。固定种子保证可复现。

    收敛性：对 4 目标、约 100 个前沿解的情形，2×10⁶ 样本的相对标准误
    约在 0.1% 量级，足以支撑算法间的比较。
    """
    lo = F.min(axis=0)
    lo = np.minimum(lo, ref) - 1e-9
    span = ref - lo
    if np.any(span <= 0):
        return 0.0

    rng = np.random.default_rng(20240501)   # 固定种子 -> 可复现
    pts = lo + rng.random((n_samples, F.shape[1])) * span

    # 逐块判定支配，控制内存
    chunk = 200_000
    inside = 0
    for s in range(0, n_samples, chunk):
        p = pts[s:s + chunk]
        # 点被 F 中任一解支配：min_j (p >= F_j) 全维成立
        dom = np.any(np.all(p[:, None, :] >= F[None, :, :], axis=2), axis=1)
        inside += int(dom.sum())

    return float(inside / n_samples * np.prod(span))


def hypervolume_pymoo(F: np.ndarray, ref: np.ndarray) -> float | None:
    """用 pymoo 计算 HV，作为自实现版本的交叉校验。

    返回 ``None`` 表示 pymoo 不可用。论文中可报告"两种独立实现结果一致"
    以增强结果可信度。
    """
    try:
        from pymoo.indicators.hv import HV

        return float(HV(ref_point=np.asarray(ref, dtype=float))(np.asarray(F, dtype=float)))
    except Exception as exc:
        logger.debug("pymoo HV 不可用: %s", str(exc)[:120])
        return None


# ---------------------------------------------------------------------------
# 分布性指标
# ---------------------------------------------------------------------------

def spacing(F: np.ndarray) -> float:
    """间距指标 SP (Schott, 1995)。

    ``SP = sqrt( 1/(n-1) * Σ (d̄ - d_i)² )``，其中 ``d_i`` 是第 i 个解到
    其它解的最小曼哈顿距离。``SP = 0`` 表示解在目标空间中完全均匀分布。
    越小越好。

    Notes
    -----
    计算前对目标做归一化，否则量纲大的目标（如成本，10⁸ 量级）会完全
    主导距离——这是很多文献报告 SP 时的常见错误。
    """
    F = np.atleast_2d(F)
    if len(F) < 2:
        return 0.0
    Fn = _normalize(F)
    d = _pairwise_manhattan(Fn)
    np.fill_diagonal(d, np.inf)
    dmin = d.min(axis=1)
    return float(np.sqrt(np.sum((dmin.mean() - dmin) ** 2) / (len(F) - 1)))


def spread(F: np.ndarray) -> float:
    """分布广度 Δ (Deb et al., 2002)。``Δ = 0`` 表示完全均匀分布，越小越好。

    ``Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1) d̄)``

    其中 ``d_i`` 是按第一目标**排序后相邻解之间的距离**，``d_f`` / ``d_l``
    是两端极值解到真实前沿边界的距离。

    ⚠️ 两个常见的实现错误（本实现都犯过，自检时才暴露）
    --------------------------------------------------
    **错误一：把 ``d_f``/``d_l`` 算成"极值解到理想点的距离"。**
    Deb 的定义是到**真实前沿边界**的距离。当参考前沿就是解集自身时，
    两端极值解**就是**边界，故 ``d_f = d_l = 0``。用"到理想点的距离"
    会让一个完美均匀的两目标前沿算出 Δ = 0.586（而不是 0）——自检即知。

    **错误二：用"到最近邻的欧氏距离"当作 ``d_i``。** Deb 的 Δ 用的是
    排序后**相邻**解的距离。对聚簇型前沿（一半解挤在一处、另一半挤在
    另一处），最近邻距离恒等于簇内间距，**完全检测不出簇间的大空隙**，
    Δ 会错误地接近 0。按第一目标排序取相邻距离则能正确反映空隙。
    """
    F = np.atleast_2d(F)
    n = len(F)
    if n < 3:
        return 0.0

    Fn = _normalize(F)
    order = np.argsort(Fn[:, 0], kind="stable")
    Fs = Fn[order]
    d = np.linalg.norm(np.diff(Fs, axis=0), axis=1)      # 相邻解距离
    if len(d) == 0:
        return 0.0
    dbar = d.mean()
    if dbar <= 1e-12:
        return 0.0

    # d_f = d_l = 0：参考边界即解集自身的两端
    return float(np.sum(np.abs(d - dbar)) / (len(d) * dbar))


def igd(F: np.ndarray, reference_front: np.ndarray) -> float:
    """反世代距离 IGD (Van Veldhuizen & Lamont, 1998)。

    参考前沿上每个点到解集的最小距离的平均值。同时反映**收敛性**与
    **多样性**：解集越接近参考前沿且覆盖越广，IGD 越小。

    Parameters
    ----------
    reference_front
        所有算法解集的合并非支配前沿（``evaluation.reference_front``
        设为 ``"auto"`` 时的行为）。
    """
    F = np.atleast_2d(F)
    R = np.atleast_2d(reference_front)
    if len(F) == 0 or len(R) == 0:
        return np.inf

    # 用参考前沿的范围归一化，保证多目标量纲可比
    lo, hi = R.min(axis=0), R.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Fn, Rn = (F - lo) / span, (R - lo) / span

    d = _pairwise_euclidean(Rn, Fn)
    return float(d.min(axis=1).mean())


# ---------------------------------------------------------------------------
# 距离工具
# ---------------------------------------------------------------------------

def _pairwise_euclidean(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    B = A if B is None else B
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    sq = np.maximum(aa + bb - 2 * A @ B.T, 0.0)
    return np.sqrt(sq)


def _pairwise_manhattan(A: np.ndarray) -> np.ndarray:
    return np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)


def _normalize(F: np.ndarray) -> np.ndarray:
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (F - lo) / span


# ---------------------------------------------------------------------------
# 方案层面的评价指标
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """基尼系数。用于度量人群间可达性的不平等程度。

    ``G = Σ_i Σ_j |x_i - x_j| / (2 n² x̄)``，取值 ``[0, 1]``：
    0 表示完全平等，1 表示完全不平等。

    在本研究中，输入为各类人群的平均接驳时间——基尼系数越大，说明
    不同收入人群享受到的起降场可达性差距越大。
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 or x.mean() <= 1e-12:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2 * n * n * x.mean()))


def solution_metrics(
    selected: Sequence[int] | np.ndarray,
    demand: "pd.DataFrame",
    candidates: "pd.DataFrame",
    access_time: np.ndarray,
    cfg,
    label: str = "",
) -> dict:
    """计算单个选址方案的全套决策层指标。

    Returns
    -------
    dict，含覆盖率、成本、可达性与公平性指标：

    ``n_sites`` / ``n_rooftop`` / ``n_ground``
        设施数量与类型构成。
    ``total_cost_cny``
        总建设成本。
    ``demand_coverage_pct`` / ``population_coverage_pct``
        需求与人口覆盖率。
    ``mean_access_time_min`` / ``median_access_time_min`` / ``p95_access_time_min``
        可达性分布（仅统计已覆盖单元）。
    ``max_access_time_min``
        最差可达性（p-center 视角）。
    ``worst_group_access_time_min``
        最差人群的平均接驳时间（**公平性目标**）。
    ``equity_gini``
        人群间可达性基尼系数。
    ``pct_within_10min``
        10 分钟内可达的人口比例（政策常用口径）。
    """
    import numpy as np
    import pandas as pd

    dem = demand.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    sel = np.asarray(sorted(selected), dtype=int)

    n_dem, n_cand = len(dem), len(cand)
    if access_time.shape != (n_dem, n_cand):
        raise ValueError(f"access_time 形状 {access_time.shape} != ({n_dem}, {n_cand})")

    if len(sel) == 0:
        return {
            "label": label, "n_sites": 0, "total_cost_cny": 0.0,
            "demand_coverage_pct": 0.0, "population_coverage_pct": 0.0,
            "mean_access_time_min": np.inf, "worst_group_access_time_min": np.inf,
            "equity_gini": np.nan,
        }

    sub = access_time[:, sel]
    sub = np.where(np.isfinite(sub), sub, np.inf)
    best = sub.min(axis=1)                       # 每个需求单元的最短接驳时间
    covered = np.isfinite(best)

    demand_vals = dem["demand"].to_numpy(dtype=np.float64)
    pop_vals = (
        dem["population"].to_numpy(dtype=np.float64)
        if "population" in dem.columns else demand_vals
    )

    t_min = best[covered] / 60.0
    cost = float(cand["cost"].iloc[sel].sum())
    types = cand["facility_type"].iloc[sel]

    # -- 公平性：按人群类别的平均接驳时间 ---------------------------------
    #
    # ⚠ 必须是**按需求加权**的组均值，与 §4.4.2 的 f_3 和 NSGA-II 目标函数
    #   （stage4_nsga2）保持同一口径。公平性的单元是"人"而不是"格网"：用
    #   未加权的格网均值，会让占据大片低密度区域的群体被系统性高估权重，
    #   从而使本函数算出的"最差人群"与论文自己声明的定义不是同一个量
    #   （实测同一方案可差 0.1–0.5 分钟，足以翻转支配关系的判定）。
    #
    # ⚠⚠ 另一处更严重的口径不一致（已修）：**未服务的需求单元曾被整批丢弃**。
    #   早期代码先 ``dem.loc[covered]`` 再分组，于是"某一类人群完全未被服务"
    #   时该组**从表里消失**，``grp.max()`` 返回的是"已服务人群里最差的那个"。
    #   而优化器在同样情形下给 f_equity = 1e12（见 stage4_nsga2._evaluate：
    #   整组不可达即 break 并置 1e12）。结果是**同一个解在优化里是 1e12、
    #   在评价里是 12.5 分钟**——支配关系可被这一差异直接翻转，而论文的
    #   公平性结论正是建立在评价器上。
    #
    #   现在的口径：组均值对**该组全部需求**计算；组内若存在未服务单元，
    #   该组标记为未服务并取 UNSERVED_ACCESS_MIN。同时单独给出
    #   ``*_served_min`` 作为描述性指标，供正文在确知无未服务组时引用。
    equity = {}
    if "cluster" in dem.columns:
        cov_dem = dem.loc[covered].copy()
        cov_dem["_t"] = t_min
        cov_dem["_w"] = demand_vals[covered]
        if len(cov_dem):
            _w = cov_dem["_w"].to_numpy(dtype=np.float64)
            _t = cov_dem["_t"].to_numpy(dtype=np.float64)
            _c = cov_dem["cluster"].to_numpy()
            _num = pd.Series(_w * _t).groupby(_c).sum()
            _den = pd.Series(_w).groupby(_c).sum()
            grp = (_num / _den).replace([np.inf, -np.inf], np.nan).dropna()
        else:
            grp = pd.Series(dtype=float)

        # 组均值的口径必须与优化器**逐字一致**（stage4_nsga2._evaluate）：
        # 分子分母都只取**该组已服务（可达）的单元**。因为"未服务"由 f1 单独
        # 承担，f3 回答的是"被服务的人体验如何"。
        #
        # ⚠ 这里我曾经修错过一次，值得记下来：我当时的做法是把**全组需求**当分母、
        #   把未服务单元的"接驳时间"记成哨兵值 $10^{12}$ 计入分子。结果是
        #   **只要组里有任何一个未服务单元，整组均值就被 $10^{12}$ 吞掉**——实测
        #   p-中心的组 0 有 830/916 个单元可达，却被报成 $10^{12}$。
        #   正确的规则只有一条：**整组无人被服务**时才置哨兵。
        all_groups = pd.unique(dem["cluster"].dropna()) if "cluster" in dem else []
        served_groups = set(grp.index.tolist())
        unserved_groups = [g for g in all_groups if g not in served_groups]

        equity["n_groups"] = int(len(all_groups))
        equity["n_groups_unserved"] = len(unserved_groups)
        equity["groups_all_served"] = bool(len(unserved_groups) == 0)
        # 与优化器 f_equity 同口径：任一组**整组**未服务 -> 哨兵值
        equity["worst_group_access_time_min"] = (
            float(grp.max()) if len(grp) and not unserved_groups
            else (UNSERVED_ACCESS_MIN if unserved_groups else np.inf)
        )
        equity["best_group_access_time_min"] = (
            float(grp.min()) if len(grp) else np.inf
        )
        equity["group_gap_min"] = (
            float(grp.max() - grp.min()) if len(grp) and not unserved_groups
            else (UNSERVED_ACCESS_MIN if unserved_groups else np.inf)
        )
        # 描述性：仅在**已服务**人群间比较。当无整组未服务时与上面同值；
        # 有整组未服务时它给出"仍被服务的人群之间的差距"，是补充信息而非替代。
        equity["worst_group_access_time_served_min"] = (
            float(grp.max()) if len(grp) else np.inf
        )
        equity["group_gap_served_min"] = (
            float(grp.max() - grp.min()) if len(grp) else np.inf
        )
        equity["equity_gini"] = gini_coefficient(grp.to_numpy())
        for cid, val in grp.items():
            equity[f"group_{int(cid)}_access_time_min"] = float(val)
        for cid in unserved_groups:
            equity[f"group_{int(cid)}_access_time_min"] = UNSERVED_ACCESS_MIN

    # -- 公平性：按收入分位（不依赖聚类标签，更稳健）---------------------
    if "income" in dem.columns and covered.any():
        inc = dem.loc[covered, "income"].to_numpy(dtype=np.float64)
        if np.isfinite(inc).sum() >= 10:
            try:
                q = pd.qcut(inc, 5, labels=False, duplicates="drop")
                qt = pd.Series(t_min).groupby(q).mean()
                equity["income_quintile_gap_min"] = float(qt.max() - qt.min())
                equity["income_quintile_gini"] = gini_coefficient(qt.to_numpy())
            except ValueError:
                pass

    out = {
        "label": label,
        "n_sites": int(len(sel)),
        "n_rooftop": int((types == "rooftop").sum()),
        "n_ground": int((types == "ground").sum()),
        "total_cost_cny": cost,
        "mean_cost_per_site_cny": cost / len(sel),
        "demand_covered": float(demand_vals[covered].sum()),
        "demand_coverage_pct": float(100 * demand_vals[covered].sum() / max(demand_vals.sum(), 1e-9)),
        "population_covered": float(pop_vals[covered].sum()),
        "population_coverage_pct": float(100 * pop_vals[covered].sum() / max(pop_vals.sum(), 1e-9)),
        "mean_access_time_min": float(t_min.mean()) if len(t_min) else np.inf,
        "median_access_time_min": float(np.median(t_min)) if len(t_min) else np.inf,
        "p95_access_time_min": float(np.percentile(t_min, 95)) if len(t_min) else np.inf,
        "max_access_time_min": float(t_min.max()) if len(t_min) else np.inf,
        "pct_within_10min": float(
            100 * pop_vals[covered][t_min <= 10].sum() / max(pop_vals.sum(), 1e-9)
        ),
        "pct_within_15min": float(
            100 * pop_vals[covered][t_min <= 15].sum() / max(pop_vals.sum(), 1e-9)
        ),
    }
    out.update(equity)
    return out


def compare_solutions(metrics_list: list[dict], sort_by: str = "demand_coverage_pct") -> "pd.DataFrame":
    """把多个方案的指标汇总成对比表（论文 Table 5）。"""
    import pandas as pd

    df = pd.DataFrame(metrics_list)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df
