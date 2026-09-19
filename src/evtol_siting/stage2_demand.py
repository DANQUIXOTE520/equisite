"""阶段二：K-means 人群分类与需求空间化。

对应申报书"难点二：大规模人群数据的分类与需求空间化表达"。

核心思想
--------
仅按人口总量配置起降场会忽略**人群构成的差异**——高收入人群的 eVTOL
出行生成率显著高于平均，而他们的空间分布与总人口分布并不重合。因此
本研究以 ``(收入, POI 密度)`` 为二维特征对需求单元聚类，得到 K 类
人群，再为每类赋予差异化的出行生成率，从而把"需求"从人口总量细化到
"人群 × 出行率"。

论文定位提示
------------
K-means 本身不是新方法（申报书已引 Jeong J. 等的首尔案例）。本步骤在
论文中的价值不在于算法，而在于：

1. 把聚类结果**显式转化为分层的需求权重**，供后续公平性目标使用；
2. 让"最差体验"这一目标有明确的**人群维度**——这正是本研究相对
   既有单目标选址模型的增量贡献。

因此论文中应把 K-means 定位为"需求分层工具"而非"方法创新"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DemandModel:
    """需求模型结果。

    Attributes
    ----------
    grid
        需求单元表，含 ``cell_id`` / ``x`` / ``y`` / ``population`` /
        ``income`` / ``poi_density`` / ``cluster`` / ``trip_rate`` /
        ``demand`` 等列。
    kmeans
        拟合后的 ``sklearn`` KMeans 对象（仅供诊断，不参与下游计算）。
    cluster_profile
        各类人群的画像表（论文 Table 3）。
    """
    grid: pd.DataFrame
    kmeans: object
    cluster_profile: pd.DataFrame

    @property
    def n_clusters(self) -> int:
        return int(self.grid["cluster"].nunique())

    def demand_by_cluster(self) -> pd.DataFrame:
        g = self.grid.groupby("cluster")
        return pd.DataFrame(
            {
                "n_cells": g.size(),
                "population": g["population"].sum().round(0),
                "demand_trips_per_day": g["demand"].sum().round(1),
                "income_mean": g["income"].mean().round(3),
                "poi_density_mean": g["poi_density"].mean().round(1),
                "trip_rate": g["trip_rate"].first().round(4),
            }
        ).reset_index()


# ---------------------------------------------------------------------------
# 特征构建
# ---------------------------------------------------------------------------

def build_features(
    grid,
    cfg,
    income_points=None,
    population_raster=None,
    poi=None,
    buildings=None,
) -> pd.DataFrame:
    """为每个需求单元构建 ``(收入, POI 密度)`` 二维特征。

    Parameters
    ----------
    grid
        需求单元 GeoDataFrame（含 ``cell_id``、``centroid_x/y``、``geometry``）。
    income_points
        外部收入/房价点矢量。用**反距离加权**插值到单元质心——外部数据的
        网格通常比需求单元粗，最近邻采样会产生明显块状伪影。
    population_raster
        WorldPop 100 m 人口栅格。
    poi
        POI 点矢量，按单元计数后除以面积得到密度。
    buildings
        建筑轮廓（含融合后高度）。**提供时启用体积权重的 dasymetric 人口
        重分配**，把人口从粗栅格按建筑体积摊到 500 m 需求格网，而不是
        假设人口在格网内均匀分布。见
        :mod:`evtol_siting.data.dasymetric` 的方法说明。

    Returns
    -------
    含 ``population`` / ``income`` / ``poi_count`` / ``poi_density`` 列的
    DataFrame（与 ``grid`` 行序对齐）。
    """
    out = pd.DataFrame({"cell_id": grid["cell_id"].values})
    cx = grid["centroid_x"].to_numpy(dtype=np.float64)
    cy = grid["centroid_y"].to_numpy(dtype=np.float64)
    cell_area_km2 = (float(cfg.get("stage1_candidates.grid_size_m", 500.0)) / 1000.0) ** 2

    # -- 人口 --------------------------------------------------------------
    #
    # 来源与量级由配置控制，供敏感性分析使用（§5.8 的人口参量扫描）：
    #
    #   ``stage2_demand.population_source``
    #       ``"worldpop"``（默认）| ``"building_volume"``。取后者时忽略
    #       WorldPop 栅格，改由建筑体量估计人口——这是一条**独立于人口栅格**
    #       的路径，用于检验结论是否依赖人口层本身。§3.5 记录了该路径的
    #       循环依赖隐患（它与屋顶型候选共享同一份 OSM 建筑图层），因此它
    #       只作为敏感性对照，不作主结果。
    #   ``stage2_demand.population_scale``
    #       乘性缩放（默认 1.0），用于检验需求量级的影响。
    pop_source = str(cfg.get("stage2_demand.population_source", "worldpop")).lower()
    pop_scale = float(cfg.get("stage2_demand.population_scale", 1.0) or 1.0)
    if pop_source == "building_volume":
        if population_raster is not None:
            logger.info("人口来源配置为 building_volume：忽略 WorldPop 栅格")
        population_raster = None

    if population_raster is not None:
        use_dasy = bool(cfg.get("stage2_demand.dasymetric_population", True))
        pop = None
        if use_dasy and buildings is not None and len(buildings):
            try:
                from .data.dasymetric import (
                    compare_population_models,
                    dasymetric_population,
                )
                from .data.rasters import zonal_sum_to_grid

                pop, _method = dasymetric_population(
                    grid, buildings, cfg, population_raster,
                    use_weights=bool(cfg.get("stage2_demand.dasymetric_use_weights", False)),
                    return_method=True,
                )
                # 如实记录实际走的方法：源比目标细时 dasymetric 会退化为
                # 分区求和，此时把它标注成 dasymetric 是数据溯源上的错误陈述。
                out["population_source"] = _method

                # 同时算一份均匀分配版，用于论文中论证 dasymetric 的必要性
                try:
                    pop_uni = zonal_sum_to_grid(population_raster, grid, cfg, stat="sum")
                    cmp_df = compare_population_models(grid, pop, pop_uni)
                    logger.info(
                        "人口分配方式对比（论文 Table 用）:\n%s",
                        cmp_df.to_string(index=False),
                    )
                    out.attrs["population_comparison"] = cmp_df
                except Exception as exc:
                    logger.debug("均匀分配对照计算失败: %s", str(exc)[:120])

            except Exception as exc:
                logger.warning("dasymetric 重分配失败，回退到分区求和: %s", str(exc)[:200])

        if pop is None:
            from .data.rasters import zonal_sum_to_grid

            pop = zonal_sum_to_grid(population_raster, grid, cfg, stat="sum")
            out["population_source"] = "zonal_sum_uniform"

        out["population"] = pop
        logger.info(
            "人口: 合计 %.0f 人，单元中位数 %.0f 人（来源: %s）",
            float(np.nansum(pop)), float(np.nanmedian(pop)),
            out["population_source"].iloc[0],
        )
    elif buildings is not None and len(buildings):
        # 人口栅格不可得时的替代路径：由建筑体量估计。
        # 优先于"合成人口"——合成人口与真实城市无关，结果毫无意义；
        # 建筑体量估计至少有物理依据，且分辨率等于建筑轮廓本身。
        # 见 data/dasymetric.py::population_from_buildings 的方法与标定说明。
        try:
            from .data.dasymetric import population_from_buildings

            pop = population_from_buildings(grid, buildings, cfg)
            out["population"] = pop
            out["population_source"] = "building_volume_estimate"
            logger.warning(
                "**人口栅格不可得，改用建筑体量估计人口**（合计 %.0f 人）。"
                "该估计量有人均住房面积的标定假设，论文须报告标定依据与"
                "隐含总量，并在敏感性分析中检验。",
                float(np.nansum(pop)),
            )
        except Exception as exc:
            logger.error("建筑体量人口估计失败: %s", str(exc)[:200])
            out["population"] = np.nan
            out["population_source"] = "unavailable"
    else:
        out["population"] = np.nan
        out["population_source"] = "unavailable"
        logger.warning("既无人口栅格也无建筑数据，人口列置空（将由调用方回退）")

    # 量级缩放（敏感性分析用）。缩放后仍标注原来源，另附缩放因子，
    # 避免把"人为缩放"误读成"来自另一个数据源"。
    if pop_scale != 1.0:
        out["population"] = out["population"].astype(float) * pop_scale
        out["population_source"] = out["population_source"].astype(str) + f"_x{pop_scale:g}"
        logger.info("人口按配置缩放 ×%g（合计 %.0f）", pop_scale,
                    float(np.nansum(out["population"])))

    # -- 收入（外部数据：夜间灯光 / 统计年鉴 / 房价；反距离加权插值）------
    if income_points is not None and len(income_points) > 0:
        from scipy.spatial import cKDTree

        ip = income_points
        if str(ip.crs) != str(grid.crs):
            ip = ip.to_crs(grid.crs)
        rwi_col = next((c for c in ip.columns if c.lower() in ("rwi", "value", "wealth")), None)
        if rwi_col is None:
            raise KeyError(f"收入点数据缺少 rwi 列，现有列: {list(ip.columns)}")

        src = np.column_stack([ip.geometry.x.values, ip.geometry.y.values])
        vals = ip[rwi_col].to_numpy(dtype=np.float64)
        tree = cKDTree(src)

        # 取最近 4 个点做反距离加权（power=2），兼顾平滑与计算量
        k = min(4, len(src))
        dist, idx = tree.query(np.column_stack([cx, cy]), k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** 2
        out["income"] = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
        logger.info(
            "收入代理(RWI): 范围 [%.2f, %.2f]，单元均值 %.3f",
            float(np.nanmin(out["income"])), float(np.nanmax(out["income"])),
            float(np.nanmean(out["income"])),
        )
    else:
        out["income"] = np.nan
        logger.info(
            "未提供外部收入数据，将使用 OSM POI 构成的经济活动指数作为代理"
            "（见 data/rasters.py::build_poi_economic_index 的说明与局限）"
        )

    # -- POI 密度 ----------------------------------------------------------
    if poi is not None and len(poi) > 0:
        import geopandas as gpd

        p = poi
        if str(p.crs) != str(grid.crs):
            p = p.to_crs(grid.crs)
        pts = p[["geometry"]].copy()
        pts["geometry"] = pts.geometry.centroid
        joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="inner", predicate="within")
        counts = joined.groupby("cell_id").size()
        out["poi_count"] = out["cell_id"].map(counts).fillna(0).astype(int)
        out["poi_density"] = out["poi_count"] / cell_area_km2   # 个/km²
        logger.info(
            "POI: 研究区内 %d 个，单元密度中位数 %.0f 个/km²",
            int(out["poi_count"].sum()), float(out["poi_density"].median()),
        )
    else:
        out["poi_count"] = 0
        out["poi_density"] = 0.0
        logger.warning("未提供 POI 数据，POI 密度置 0")

    # -- 收入代理回退：POI 经济活动指数 ------------------------------------
    # 当没有外部收入/房价/夜间灯光数据时，用 POI 构成的经济活动指数代理。
    # 这是**代理变量**，不是收入本身；论文中必须如此表述（见 rasters.py）。
    if out["income"].isna().all() and float(out["poi_density"].sum()) > 0:
        from .data.rasters import build_poi_economic_index

        out["income"] = build_poi_economic_index(poi, grid, cfg)
        out["income_source"] = "poi_economic_index"
    elif income_points is not None:
        out["income_source"] = "external"
    else:
        out["income_source"] = "unavailable"

    return out


# ---------------------------------------------------------------------------
# 回退特征（无外部数据时）
# ---------------------------------------------------------------------------

def synthesise_features(grid, cfg) -> pd.DataFrame:
    """在缺少真实数据时生成**结构与量级合理**的合成特征。

    仅用于打通流程与单元测试，**绝不可用于论文实证结论**。
    合成逻辑：成都呈现"单中心 + 南向延伸"格局，因此用天府广场为峰值的
    双中心核密度生成人口与 POI，再叠加由中心距离决定收入梯度与噪声。

    返回列与 :func:`build_features` 一致。
    """
    logger.warning(
        "使用**合成**需求特征（人口/收入/POI）。仅用于流程验证，"
        "结果不可用于论文。请配置真实数据源后重跑。"
    )
    rng = np.random.default_rng(cfg.seed)
    cx = grid["centroid_x"].to_numpy(dtype=np.float64)
    cy = grid["centroid_y"].to_numpy(dtype=np.float64)

    # 两个中心：天府广场、天府新区（南向）
    import geopandas as gpd

    def _to_xy(lon, lat):
        pt = gpd.GeoSeries(
            gpd.points_from_xy([lon], [lat]), crs=cfg.crs_geo
        ).to_crs(cfg.crs_proj)
        return float(pt.x.iloc[0]), float(pt.y.iloc[0])

    c1 = _to_xy(104.0648, 30.6570)   # 天府广场
    c2 = _to_xy(104.0700, 30.4800)   # 天府新区核心区

    d1 = np.hypot(cx - c1[0], cy - c1[1]) / 1000.0
    d2 = np.hypot(cx - c2[0], cy - c2[1]) / 1000.0

    # 人口：双中心高斯核。峰值取 35 000 人/km²——成都锦江区、青羊区等
    # 核心城区的实际常住人口密度约在 20 000–35 000 人/km² 量级。
    # （早期版本误用 250 000 人/km²，使研究区总人口达到 3300 万，
    #   远超成都市实际约 2100 万的全市常住人口，属明显的量级错误。）
    pop_km2 = 35_000 * np.exp(-(d1 ** 2) / (2 * 4.0 ** 2)) + 9_000 * np.exp(-(d2 ** 2) / (2 * 6.0 ** 2))
    pop_km2 *= np.exp(rng.normal(0, 0.25, size=len(cx)))          # 空间异质性
    cell_area_km2 = (float(cfg.get("stage1_candidates.grid_size_m", 500.0)) / 1000.0) ** 2
    population = np.maximum(pop_km2 * cell_area_km2, 0.0)

    # POI 密度与人口强相关但更集中
    poi_density = 4000 * np.exp(-(d1 ** 2) / (2 * 2.5 ** 2)) + 900 * np.exp(-(d2 ** 2) / (2 * 3.5 ** 2))
    poi_density *= np.exp(rng.normal(0, 0.35, size=len(cx)))

    # 收入：中心高、外围低（成都实际为"中心-近郊"梯度），叠加噪声
    income = 1.2 * np.exp(-d1 / 12.0) + 0.4 * np.exp(-d2 / 15.0) - 0.6
    income += rng.normal(0, 0.28, size=len(cx))

    return pd.DataFrame(
        {
            "cell_id": grid["cell_id"].values,
            "population": population,
            "income": income,
            "poi_count": (poi_density * cell_area_km2).round().astype(int),
            "poi_density": poi_density,
        }
    )


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------

def _select_k(X: np.ndarray, cfg, seed: int) -> tuple[int, dict]:
    """选择聚类数 K。

    ``silhouette``：在 ``k_range`` 内取轮廓系数最大的 K（默认）。
    ``elbow``：取惯性下降率拐点。
    ``fixed``：直接使用 ``n_clusters``。

    轮廓系数在 ~2400 个样本上对每个 K 需 O(n²) 距离计算，k_range 为
    [2,10] 时总计约 9 次聚类，实测 < 20 s，可接受。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    method = str(cfg.get("stage2_demand.k_selection.method", "silhouette")).lower()
    # 注意：轮廓系数在小 K 处系统性偏高（K=2 几乎总占优），这是该指标的
    # 已知偏置。若 k_range 下界为 2，自动选择几乎必然退化为"二分"。
    # 因此默认下界取 3，并在论文中报告各 K 的得分供读者判断（见日志与
    # outputs/data/table_clusters.csv）。
    if method == "fixed":
        k = int(cfg.get("stage2_demand.n_clusters", 5))
        return k, {"method": "fixed", "k": k}

    lo, hi = cfg.get("stage2_demand.k_selection.k_range", [2, 10])
    lo, hi = int(lo), min(int(hi), len(X) - 1)
    if hi < lo:
        lo = hi = max(2, min(2, len(X) - 1))

    diag: dict = {"method": method, "scores": {}}
    best_k, best_score = lo, -np.inf

    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = km.labels_
        if method == "silhouette":
            n_unique = len(np.unique(labels))
            score = (
                float(silhouette_score(X, labels))
                if 1 < n_unique < len(X)
                else -np.inf
            )
        else:  # elbow：用惯性下降的相对变化率
            score = -float(km.inertia_) / k
        diag["scores"][k] = round(float(score), 4)
        if score > best_score:
            best_score, best_k = score, k

    diag["k"] = best_k
    diag["best_score"] = round(float(best_score), 4)
    logger.info("K 选择 (%s): K=%d，得分 %.4f，各 K 得分 %s",
                method, best_k, best_score, diag["scores"])
    return best_k, diag


def _assign_trip_rates(profile: pd.DataFrame, cfg, k: int) -> np.ndarray:
    """为各类人群分配 eVTOL 出行生成率（次/人/日）。

    默认规则：出行率随收入单调递增。这基于两条依据：

    1. UAM 的初期目标客群以时间价值高的商务/高收入人群为主，这是
       多数 UAM 需求研究的共识；
    2. 收入是航空出行生成最强的单一预测变量。

    具体取值在论文中属于**待标定参数**，因此本函数给出的绝对值并不
    重要，重要的是**类间的相对比例**。默认把最高收入类的出行率设为
    最低收入类的 5 倍，在两者之间按收入秩线性插值。该比例必须在敏感性
    分析中做检验（见 ``scripts/07_sensitivity.py``）。
    """
    configured = cfg.get("stage2_demand.trip_rate_by_cluster", "auto")
    if isinstance(configured, (list, tuple)) and len(configured) == k:
        return np.asarray(configured, dtype=np.float64)

    # 最高收入类 / 最低收入类之比。可经配置覆盖，供 §5.9.2 的证伪检验扫描。
    #
    # ⚠ 早期版本要求调用方**显式给出长度等于 K 的列表**，而调用方必须事先知道 K
    #   ——这迫使证伪检验把 K 钉死（曾硬编码为 7），于是控制臂与主分析的聚类
    #   不一致，两臂根本不可比。改为在此处读比值、用**实际的 K** 生成阶梯后，
    #   K 由轮廓系数照常决定，两臂的聚类完全相同，唯一差别只剩需求模型。
    ratio = float(cfg.get("stage2_demand.trip_rate_ratio", 5.0))
    base = 0.0015   # 最低收入类的日出行率（次/人/日），UAM 早期市场量级

    inc = profile["income_mean"].to_numpy(dtype=np.float64)
    if np.ptp(inc) < 1e-9:            # 所有类收入相同（退化情形）
        return np.full(k, base, dtype=np.float64)

    # 按收入秩映射到 [0, 1]，避免受异常值影响
    ranks = pd.Series(inc).rank(method="average").to_numpy()
    t = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    return base * (1.0 + (ratio - 1.0) * t)


def cluster_demand(
    grid,
    features: pd.DataFrame,
    cfg,
    k: int | None = None,
) -> DemandModel:
    """对需求单元执行 K-means 聚类并生成需求。

    标准流程：z-score 标准化 -> K-means -> 质心还原到原始量纲 ->
    按收入秩分配出行率 -> 需求 = 人口 × 出行率。

    Parameters
    ----------
    grid
        需求单元表（须含 ``cell_id``）。
    features
        :func:`build_features` 的输出。
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feat_names = cfg.get("stage2_demand.features", ["income", "poi_density"])
    missing = [f for f in feat_names if f not in features.columns]
    if missing:
        raise KeyError(f"特征列缺失: {missing}；现有列 {list(features.columns)}")

    df = grid[["cell_id", "centroid_x", "centroid_y"]].merge(
        features, on="cell_id", how="left"
    )

    # 人口缺失回退：用 POI 密度线性外推（POI 与人口强相关）
    if df["population"].isna().any():
        logger.warning(
            "人口列存在 %d 个缺失值，用 POI 密度线性外推填补",
            int(df["population"].isna().sum()),
        )
        pd_ = df["poi_density"].to_numpy(dtype=np.float64)
        pop_ = df["population"].to_numpy(dtype=np.float64)
        ok = np.isfinite(pop_) & np.isfinite(pd_)
        if ok.sum() >= 10:
            a, b = np.polyfit(pd_[ok], pop_[ok], 1)
            fill = a * pd_ + b
        else:
            fill = np.full(len(df), float(np.nanmedian(pop_)) if np.isfinite(pop_).any() else 0.0)
        df["population"] = np.where(np.isfinite(pop_), pop_, np.maximum(fill, 0.0))

    df["population"] = df["population"].fillna(0.0).clip(lower=0.0)

    X_raw = df[feat_names].to_numpy(dtype=np.float64)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    if cfg.get("stage2_demand.standardize", True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        scaler, X = None, X_raw

    k_final = int(k) if k is not None else None
    k_diag: dict = {}
    if k_final is None:
        k_final, k_diag = _select_k(X, cfg, cfg.seed)
    df["cluster"] = KMeans(
        n_clusters=k_final, n_init=20, random_state=cfg.seed
    ).fit_predict(X)

    # -- 类画像 ------------------------------------------------------------
    profile = (
        df.groupby("cluster")
        .agg(
            n_cells=("cell_id", "size"),
            population=("population", "sum"),
            income_mean=("income", "mean") if "income" in df.columns else ("population", "mean"),
            poi_density_mean=("poi_density", "mean") if "poi_density" in df.columns else ("population", "mean"),
        )
        .reset_index()
        .sort_values("income_mean" if "income" in df.columns else "population")
        .reset_index(drop=True)
    )

    # -- 出行生成率：离散选择模型（优先）或外生假设（退路）----------------
    #
    # 默认走离散选择模型：由票价、时间节省与各收入群体的时间价值**推导**
    # 采用概率。这样可达性不平等来自行为而非研究者的设定，避免了
    # "假设不平等、再论证不平等"的循环。
    #
    # 外生假设（``_assign_trip_rates``：出行率随收入单调递增）保留为退路，
    # 用于在没有收入标定或需要对照实验时使用。
    use_choice = bool(cfg.get("choice_model.enabled", True))
    choice_table = None
    if use_choice:
        try:
            from .choice_model import EVTOLChoiceModel

            model = EVTOLChoiceModel(cfg)
            choice_table = model.cluster_demand_rates(profile)
            rate_map = dict(
                zip(choice_table["cluster"].astype(int), choice_table["demand_rate"])
            )
            # 把收入与时间价值一并带回单元表，供公平性分析使用
            for col in ("monthly_income_cny", "vot_cny_per_h", "p_evtol"):
                m = dict(zip(choice_table["cluster"].astype(int), choice_table[col]))
                df[col] = df["cluster"].map(m).astype(np.float64)
        except Exception as exc:
            logger.error(
                "离散选择模型失败（%s），回退到外生出行率假设", str(exc)[:180]
            )
            use_choice = False

    if not use_choice:
        rates = _assign_trip_rates(profile, cfg, k_final)
        rate_map = {int(c): float(r) for c, r in zip(profile["cluster"], rates)}

    df["trip_rate"] = df["cluster"].map(rate_map).astype(np.float64)
    df["demand"] = df["population"] * df["trip_rate"]      # 次/日

    profile["trip_rate"] = profile["cluster"].map(rate_map)
    if choice_table is not None:
        for col in ("monthly_income_cny", "vot_cny_per_h", "p_evtol"):
            profile[col] = profile["cluster"].map(
                dict(zip(choice_table["cluster"].astype(int), choice_table[col]))
            )
    profile["demand_trips_per_day"] = (
        df.groupby("cluster")["demand"].sum().reindex(profile["cluster"]).values
    )
    profile["k_selection"] = str(k_diag.get("method", "fixed"))
    profile.attrs["demand_model"] = (
        "logit_choice" if use_choice else "exogenous_trip_rate"
    )

    logger.info(
        "K-means 聚类完成: K=%d，总需求 %.0f 次/日，总人口 %.0f 人",
        k_final, float(df["demand"].sum()), float(df["population"].sum()),
    )
    logger.info("类画像:\n%s", profile.to_string(index=False))

    kmeans_obj = KMeans(n_clusters=k_final, n_init=20, random_state=cfg.seed).fit(X)
    return DemandModel(grid=df, kmeans=kmeans_obj, cluster_profile=profile)


def demand_model_from_cfg(cfg, grid, **kwargs) -> DemandModel:
    """一站式构建需求模型：构建特征（缺失则合成）-> 聚类。"""
    feats = build_features(grid, cfg, **kwargs)
    if feats["population"].isna().all() or (feats["income"].isna().all() and feats["poi_density"].sum() == 0):
        if not cfg.get("data.allow_synthetic_fallback", True):
            raise RuntimeError("需求特征全部缺失，且未启用合成回退")
        feats = synthesise_features(grid, cfg)
    return cluster_demand(grid, feats, cfg)
