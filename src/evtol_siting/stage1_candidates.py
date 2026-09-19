"""阶段一：基于 GIS 的候选起降场筛选。

对应申报书"难点一：复杂城市环境中候选点的筛选与密度控制"。

流程
----
1. **空间栅格化** —— 研究区划分为 500 m 单元（需求聚合单元，见 ``geo.make_grid``）
2. **屋顶型筛选** —— ``h >= 20 m`` 且屋顶可用面积 ``A >= 1600 m²``
3. **地面型筛选** —— 排除水域/山体/生态用地，坡度 ``<= 15°``，地块 ``>= 1600 m²``
4. **安全排除** —— 机场净空区、既有直升机场缓冲区
5. **密度稀释** —— 候选点间最小间距 ``c_min = 500 m``，
   每个栅格最多保留 ``max_candidates_per_grid`` 个

设计取舍说明
------------
* 屋顶型的 ``A >= 1600 m²`` 意味着约 40 m × 40 m 的屋顶。这是 FATO
  （Final Approach and Take-off Area）+ 安全区的基本量级，与 EASA
  PTS-VPT-DSN 及 FAA EB-105 的典型尺寸假设一致，但在论文中应做
  敏感性分析（1400–2500 m²）。
* 稀释采用**贪心最大间距**而非随机剔除：按"候选质量分"排序后依次保留
  与已选点距离大于阈值的点，保证结果确定可复现（无随机性）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import distance_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 候选点数据结构
# ---------------------------------------------------------------------------

@dataclass
class CandidateSet:
    """候选起降场集合。

    Attributes
    ----------
    gdf
        GeoDataFrame，每行一个候选点。关键列：

        ``cand_id``        0..M-1 的连续编号（下游用作数组下标）
        ``facility_type``  ``"rooftop"`` 或 ``"ground"``
        ``x`` / ``y``      投影坐标下的位置
        ``height_m``       屋顶型为建筑高度；地面型为 0
        ``capacity``       设计容量（同时服务架次/小时），用于容量约束
        ``cost``           建设成本（元）
        ``quality``        候选质量分，用于稀释排序
    """
    gdf: "pd.DataFrame"

    def __len__(self) -> int:
        return len(self.gdf)

    @property
    def xy(self) -> np.ndarray:
        return self.gdf[["x", "y"]].to_numpy(dtype=np.float64)

    @property
    def ids(self) -> np.ndarray:
        return self.gdf["cand_id"].to_numpy()

    def by_type(self, facility_type: str):
        return self.gdf[self.gdf["facility_type"] == facility_type]

    def summary(self) -> pd.DataFrame:
        """按设施类型汇总（论文 Table 2）。"""
        g = self.gdf.groupby("facility_type")
        return pd.DataFrame(
            {
                "n": g.size(),
                "height_mean_m": g["height_m"].mean().round(1),
                "cost_mean_cny": g["cost"].mean().round(0),
                "cost_total_cny": g["cost"].sum().round(0),
            }
        ).reset_index()


# ---------------------------------------------------------------------------
# 成本模型
# ---------------------------------------------------------------------------

def _rooftop_cost(row, cfg) -> float:
    """屋顶型起降场建设成本：固定费 + 面积相关费。

    屋顶型不需要征地，但需要结构加固、升降机改造、消防与供电增容，
    因此固定成本高而面积成本低。这与地面型的成本结构相反。
    """
    c = cfg.get("costs.rooftop", {})
    area = float(row.get("roof_usable_m2", 1600.0))
    return float(c.get("fixed", 12e6)) + float(c.get("per_m2", 8e3)) * area


def _ground_cost(row, cfg) -> float:
    """地面型起降场建设成本：固定费 + 征地费 + 土建费。

    注意：征地与建设费用按**起降场实际占地**（``site_area_m2``，默认
    4000 m² = FATO + 停机位 + 航站与围界）计算，**不是**按整个栅格单元的
    面积（250 000 m²）计算。后者会把单站成本高估约 60 倍，是建模中很容易
    踩的坑——一个 500 m 格网里有空地与征用整个格网是两回事。
    """
    c = cfg.get("costs.ground", {})
    area = float(row.get("site_area_m2", c.get("site_area_m2", 4000.0)))
    return (
        float(c.get("fixed", 25e6))
        + float(c.get("land_per_m2", 3e3)) * area
        + float(c.get("per_m2", 5e3)) * area
    )


# ---------------------------------------------------------------------------
# 屋顶型候选
# ---------------------------------------------------------------------------

def build_rooftop_candidates(
    buildings, cfg, usable_roof_ratio: float | None = None
) -> pd.DataFrame:
    """从建筑数据中筛选屋顶型候选起降场。

    判据（申报书式）：``M_R = { g_xy | h_xy >= 20 m, A_xy >= 1600 m² }``

    Parameters
    ----------
    buildings
        已经过 :func:`evtol_siting.data.heights.fuse_building_heights` 与
        :func:`estimate_roof_area` 处理的建筑 GeoDataFrame。
    """
    from .data.heights import estimate_roof_area

    if buildings is None or len(buildings) == 0:
        logger.warning("建筑数据为空，屋顶型候选为空")
        return pd.DataFrame()

    rc = cfg.get("stage1_candidates.rooftop", {})
    if not rc.get("enabled", True):
        logger.info("配置已禁用屋顶型候选")
        return pd.DataFrame()

    min_h = float(rc.get("min_height_m", 20.0))
    min_a = float(rc.get("min_roof_area_m2", 1600.0))
    ratio = float(
        usable_roof_ratio
        if usable_roof_ratio is not None
        else rc.get("usable_roof_ratio", 0.70)
    )

    b = buildings
    if "roof_usable_m2" not in b.columns:
        b = estimate_roof_area(b, ratio)
    else:
        b = b.copy()

    mask = (b["height_m"] >= min_h) & (b["roof_usable_m2"] >= min_a)
    sel = b.loc[mask].copy()
    logger.info(
        "屋顶型筛选: %d / %d 栋建筑满足 h>=%.0fm 且 A_roof>=%.0fm² (%.2f%%)",
        len(sel), len(b), min_h, min_a, 100 * len(sel) / max(len(b), 1),
    )

    if sel.empty:
        return pd.DataFrame()

    cent = sel.geometry.centroid
    out = pd.DataFrame(
        {
            "facility_type": "rooftop",
            "x": cent.x.values,
            "y": cent.y.values,
            "height_m": sel["height_m"].values,
            "roof_area_m2": sel["roof_area_m2"].values,
            "roof_usable_m2": sel["roof_usable_m2"].values,
            "height_source": sel.get("height_source", pd.Series("unknown", index=sel.index)).values,
            "osm_id": sel.get("osm_id", pd.Series(-1, index=sel.index)).values,
            "building_type": sel.get("building", pd.Series("yes", index=sel.index)).values,
        }
    )

    # 质量分：屋顶越大、越高、高度越可信 -> 越优先保留
    # 归一化后加权，用于后续密度稀释的排序
    def _norm(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    source_weight = {
        "osm_height": 1.0, "osm_levels": 0.8, "raster_height": 0.6, "imputed": 0.3,
    }
    out["quality"] = (
        0.45 * _norm(out["roof_usable_m2"])
        + 0.35 * _norm(out["height_m"])
        + 0.20 * out["height_source"].map(source_weight).fillna(0.3)
    )
    out["cost"] = [_rooftop_cost(r, cfg) for _, r in out.iterrows()]

    # 容量：屋顶面积每 800 m² 支持 1 个 FATO，上限 4。
    # 单个 FATO 的小时吞吐取 25 架次：按 2–2.5 分钟的起降循环（进近→着陆→
    # 移出→下一架）计，这是当前 vertiport 概念研究中常见的量级。
    # 早期取 12 架次/小时，比现实保守一倍，与需求规模不匹配，会使大量候选
    # 因"负载超容量"被排除并使覆盖问题不可行。
    out["n_fato"] = np.clip(np.floor(out["roof_usable_m2"] / 800.0), 1, 4).astype(int)
    out["capacity"] = out["n_fato"] * float(cfg.get("costs.fato_throughput_per_hour", 25.0))

    out["geometry"] = sel.geometry.values
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 地面型候选
# ---------------------------------------------------------------------------

def build_ground_candidates(
    grid,
    cfg,
    landuse=None,
    water=None,
    slope_raster=None,
    buildings=None,
) -> pd.DataFrame:
    """筛选地面型候选起降场。

    判据（申报书式 + 本研究补充）::

        M_G = { g_xy | u_xy ∉ {山脉, 水域, 生态红线}, slope_xy <= 15°,
                building_coverage_xy <= beta }

    候选位置取**需求栅格单元的中心**，并以单元为地块代理（500 m × 500 m
    = 250 000 m²，远大于 1600 m² 最小地块要求）。

    关于建筑覆盖率判据 ``beta``
    ---------------------------
    申报书的判据只排除了自然地理意义上的不可建区域（山、水、生态红线）。
    但在成都这样的平原城市，这个判据几乎不排除任何东西——绕城高速以内
    的坡度普遍小于 5°，水域占比很小，结果是**几乎每个 500 m 格网都会
    成为地面型候选**，候选集约 2300 个。这显然不符合实际：已经建满的
    老城区根本没有 1600 m² 的空地可征。

    因此本实现补充一条判据：单元内的**建筑基底覆盖率**超过阈值
    ``max_building_coverage`` 时，视为已建成区，不再作为地面型候选。
    该指标直接从建筑轮廓数据算出，是"可开发空地"的一个可计算代理。

    这一补充是相对申报书的方法改进，论文中应作为候选筛选流程的一部分
    加以说明，并对其阈值做敏感性分析。

    Parameters
    ----------
    landuse, water
        投影坐标系下的 GeoDataFrame，用于空间排除。
    slope_raster
        坡度栅格路径（度）。为 ``None`` 时跳过坡度判据并在日志中警告。
    buildings
        建筑轮廓（投影坐标系）。提供时施加建筑覆盖率判据。
    """
    gc = cfg.get("stage1_candidates.ground", {})
    if not gc.get("enabled", True):
        logger.info("配置已禁用地面型候选")
        return pd.DataFrame()

    cand = grid.copy()
    min_a = float(gc.get("min_parcel_area_m2", 1600.0))
    max_slope = float(gc.get("max_slope_deg", 15.0))
    max_cov = gc.get("max_building_coverage", 0.55)
    n_before = len(cand)

    # -- 用地排除 ----------------------------------------------------------
    if landuse is not None and len(landuse) > 0:
        exclude_lu = set(gc.get("exclude_landuse", []))
        lu = landuse[landuse.get("landuse", pd.Series(dtype=str)).isin(exclude_lu)]
        if len(lu) > 0:
            cand = _spatial_exclude(cand, lu, label="用地类型")
    if water is not None and len(water) > 0:
        cand = _spatial_exclude(cand, water, label="水域")

    if len(cand) == 0:
        logger.warning("地面型候选在用地排除后为空")
        return pd.DataFrame()

    # -- 建筑覆盖率判据（已建成区不可能是空地）----------------------------
    if buildings is not None and len(buildings) > 0 and max_cov is not None:
        cand = _exclude_built_up(cand, buildings, float(max_cov))
        if len(cand) == 0:
            logger.warning("地面型候选在建筑覆盖率判据后为空")
            return pd.DataFrame()

    # -- 坡度判据 ----------------------------------------------------------
    if slope_raster is not None:
        from .geo import sample_raster_at_points

        cx = cand.geometry.centroid.x.values
        cy = cand.geometry.centroid.y.values
        try:
            slope = sample_raster_at_points(slope_raster, cx, cy, dst_crs=str(cand.crs))
            ok = np.isfinite(slope) & (slope <= max_slope)
            logger.info(
                "坡度判据: %d / %d 单元满足 slope<=%.0f° (剔除 %d)",
                int(ok.sum()), len(cand), max_slope, int((~ok).sum()),
            )
            cand = cand.loc[ok].copy()
        except Exception as exc:
            logger.warning("坡度采样失败，跳过坡度判据: %s", str(exc)[:160])
    else:
        logger.warning(
            "未提供坡度栅格：地面型候选**未施加坡度判据**。"
            "成都平原坡度普遍 <5°，影响有限，但论文中必须说明。"
        )

    if len(cand) == 0:
        return pd.DataFrame()

    # -- 构造输出 ----------------------------------------------------------
    cent = cand.geometry.centroid
    cell_area = float(cfg.get("stage1_candidates.grid_size_m", 500.0)) ** 2
    site_area = float(gc.get("site_area_m2", 4000.0))
    out = pd.DataFrame(
        {
            "facility_type": "ground",
            "x": cent.x.values,
            "y": cent.y.values,
            "height_m": 0.0,
            "roof_area_m2": np.nan,
            "roof_usable_m2": np.nan,
            # 单元总面积（用于"是否有足够空地"的判据）
            "cell_area_m2": cell_area,
            # 起降场实际占地（用于成本计算）
            "site_area_m2": site_area,
            "parcel_area_m2": cell_area,
            "cell_id": cand["cell_id"].values,
            "height_source": "n/a",
            "osm_id": -1,
            "building_type": "ground",
        }
    )
    # 要求单元内可提供的空地不小于起降场实际占地
    out = out[out["parcel_area_m2"] >= max(min_a, site_area)]

    # 质量分：地面型在平原区无自然地物差异，用"到路网的距离"更合适，
    # 此处用占位质量分（0.5），待路网可用时由 stage1 主流程覆盖。
    out["quality"] = 0.5
    out["cost"] = [_ground_cost(r, cfg) for _, r in out.iterrows()]
    # 地面型：默认 2 个 FATO（FATO + 停机位 + 航站的典型占地配置）
    out["n_fato"] = 2
    out["capacity"] = 2 * float(cfg.get("costs.fato_throughput_per_hour", 25.0))
    out["geometry"] = cand.geometry.values[: len(out)]

    logger.info(
        "地面型筛选: %d 个候选（用地排除 -%d，坡度排除后）",
        len(out), n_before - len(cand) if len(cand) else n_before,
    )
    return out.reset_index(drop=True)


def _exclude_built_up(cand, buildings, max_coverage: float):
    """剔除建筑覆盖率过高的单元（已建成区，无可用空地）。

    用 ``sjoin`` 把建筑质心归入栅格单元，再按建筑占地面积之和除以单元
    面积得到覆盖率。用质心归属而非几何相交，是为了让一栋跨单元的大建筑
    只计入一个单元，避免覆盖率被人为放大到 >1 的无意义值。

    覆盖率高于 ``max_coverage`` 的单元意味着空地不足 45%，
    难以提供 1600 m² 的连续可用地面。
    """
    import geopandas as gpd

    try:
        b = buildings
        if str(b.crs) != str(cand.crs):
            b = b.to_crs(cand.crs)

        bpts = gpd.GeoDataFrame(
            {"_area": b.geometry.area.to_numpy(dtype=np.float64)},
            geometry=b.geometry.centroid, crs=cand.crs,
        )
        joined = gpd.sjoin(bpts, cand[["cell_id", "geometry"]], how="inner", predicate="within")
        if len(joined) == 0:
            logger.warning("  建筑覆盖率判据：没有建筑落入任何单元，跳过该判据")
            return cand

        area_sum = joined.groupby("cell_id")["_area"].sum()
        cell_area = cand.geometry.area.iloc[0] if len(cand) else 1.0
        coverage = (area_sum / cell_area).clip(upper=1.0)
        cov_col = cand["cell_id"].map(coverage).fillna(0.0)

        keep = cov_col <= max_coverage
        out = cand.loc[keep].copy()
        out["building_coverage"] = cov_col[keep]
        logger.info(
            "  建筑覆盖率判据 (max=%.2f): 剔除 %d 个单元（保留 %d）；"
            "覆盖率中位数 %.2f",
            max_coverage, int((~keep).sum()), len(out), float(cov_col.median()),
        )
        return out
    except Exception as exc:
        logger.warning("  建筑覆盖率判据失败，跳过: %s", str(exc)[:180])
        return cand


def _spatial_exclude(cand, exclude_gdf, label: str = ""):
    """剔除与 ``exclude_gdf`` 相交的候选单元。"""
    try:
        a = cand[["geometry"]].copy()
        if str(a.crs) != str(exclude_gdf.crs):
            exclude_gdf = exclude_gdf.to_crs(a.crs)
        # 用质心判定而非几何相交：避免边界单元被大面积水域"误伤"
        pts = a.copy()
        pts["geometry"] = a.geometry.centroid
        joined = pts.sjoin(
            exclude_gdf[["geometry"]], how="inner", predicate="within"
        )
        drop_idx = joined.index.unique()
        out = cand.drop(index=drop_idx)
        logger.info("  排除 %s: %d 个单元（%.1f%%）", label, len(drop_idx),
                    100 * len(drop_idx) / max(len(cand), 1))
        return out
    except Exception as exc:
        logger.warning("  空间排除(%s)失败，跳过: %s", label, str(exc)[:160])
        return cand


# ---------------------------------------------------------------------------
# 安全排除
# ---------------------------------------------------------------------------

def apply_safety_exclusions(candidates: pd.DataFrame, airports, cfg) -> pd.DataFrame:
    """剔除落在机场净空区 / 既有直升机场缓冲区内的候选点。

    起降场不能建在运输机场的净空保护区内，也应与既有直升机场保持
    运行间隔。这是**硬约束**（可行性），不是优化目标。
    """
    if candidates is None or len(candidates) == 0:
        return candidates

    sc = cfg.get("stage1_candidates.safety", {})
    r_airport = float(sc.get("airport_exclusion_radius_m", 8000.0))
    r_heliport = float(sc.get("existing_heliport_exclusion_m", 1000.0))

    if airports is None or len(airports) == 0:
        logger.warning("未提供机场数据，**未施加净空区约束**")
        return candidates

    import geopandas as gpd
    from shapely.geometry import Point

    a = airports.copy()
    if str(a.crs) != str(candidates.crs):
        a = a.to_crs(candidates.crs)

    # 净空区的判定口径由配置项
    # ``stage1_candidates.safety.airport_exclusion_mode`` 决定：
    #
    #   ``"all_aeroway"``（**默认，本文采用**）
    #       一切非 ``helipad``/``heliport`` 的 ``aeroway`` 要素都按运输机场
    #       处理，统一施加 ``r_airport`` 净空区。
    #   ``"military_only"``
    #       只排除军用机场（``landuse=military`` 或 ``military=airfield``）。
    #
    # 为什么默认取保守口径：本提取包中民用机场（双流）**没有** ``aerodrome``
    # 标签，只以航站楼、滑行道、停机位等内部要素出现。若改用
    # ``"military_only"``，双流周边的候选会全部进入可行集——实测候选数由
    # 1 332 增至 1 552，并使"预算约束情景"等 MILP 的求解时间由数秒膨胀到
    # 无法在可接受时间内求解（结构性原因是 1 201 个地面候选成本完全一致，
    # 造成对称性爆炸；详见 RUN_STATE「零之五」）。
    #
    # 此外，"eVTOL 可否设站于民航机场"本身是一个**尚未有定论的政策问题**，
    # 在安全约束上取保守口径，优于先行替政策作答。该口径的代价在 §6.4 作为
    # 局限说明。
    mode = str(cfg.get("stage1_candidates.safety.airport_exclusion_mode",
                       "all_aeroway")).strip().lower()

    aeroway = a.get("aeroway", pd.Series("aerodrome", index=a.index)).astype(str)
    is_heliport = aeroway.isin(["helipad", "heliport"])

    if mode == "military_only":
        military = a.get("military", pd.Series(pd.NA, index=a.index)).astype(str)
        landuse = a.get("landuse", pd.Series(pd.NA, index=a.index)).astype(str)
        is_military = military.eq("airfield") | landuse.eq("military")
        is_airport = (~is_heliport) & is_military
        n_civil = int(((~is_heliport) & (~is_military)).sum())
        logger.info("  口径 military_only：民用机场要素 %d 个**不排除**", n_civil)
        if int(is_airport.sum()) == 0:
            logger.warning(
                "  未识别到军用机场（landuse=military / military=airfield）——"
                "净空区约束实际未生效，候选集会静默变大"
            )
    else:
        is_airport = ~is_heliport
        logger.info("  口径 all_aeroway：%d 个非直升要素全部按运输机场处理",
                    int(is_airport.sum()))

    pts = gpd.GeoDataFrame(
        candidates.copy(),
        geometry=gpd.points_from_xy(candidates["x"], candidates["y"]),
        crs=candidates.crs,
    )

    keep = np.ones(len(pts), dtype=bool)
    airport_label = ("军用机场净空区" if mode == "military_only"
                     else "机场净空区（含民用）")
    for label, sub, radius in (
        (airport_label, a[is_airport], r_airport),
        ("既有直升机场", a[is_heliport], r_heliport),
    ):
        if len(sub) == 0:
            continue
        if radius <= 0:
            # 半径为 0 表示显式关闭该约束（敏感性分析用）。
            # 不能直接 buffer(0)——那会得到空几何，虽然结果同为"不排除"，
            # 但日志里看不出约束是被有意关闭还是数据缺失，容易误读。
            logger.info("  %s: 约束已关闭（r=0）", label)
            continue
        buf = sub.copy()
        buf["geometry"] = buf.geometry.centroid.buffer(radius)
        hit = pts.sjoin(buf[["geometry"]], how="inner", predicate="within")
        dropped = hit.index.unique()
        keep[dropped] = False
        logger.info(
            "  %s (r=%.0fm): 剔除 %d 个候选（占 %.1f%%）",
            label, radius, len(dropped), 100 * len(dropped) / max(len(pts), 1),
        )

    out = candidates.loc[keep].reset_index(drop=True)
    logger.info("安全排除后剩余 %d 个候选（原有 %d）", len(out), len(candidates))
    return out


# ---------------------------------------------------------------------------
# 密度稀释
# ---------------------------------------------------------------------------

def _greedy_min_distance(df: pd.DataFrame, c_min: float, score_col: str = "quality") -> pd.DataFrame:
    """贪心最小间距筛选（确定性）。

    按 ``score_col`` 降序依次保留与所有已保留点距离均 ``>= c_min`` 的候选。
    等价于在候选图上求极大独立集的一个确定性近似，复杂度 ``O(M²)``。
    """
    if len(df) == 0:
        return df
    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    order = np.argsort(-df[score_col].to_numpy(), kind="stable")
    keep: list[int] = []
    kept: list[np.ndarray] = []
    for idx in order:
        p = pts[idx]
        if kept and np.linalg.norm(np.asarray(kept) - p, axis=1).min() < c_min:
            continue
        keep.append(int(idx))
        kept.append(p)
    return df.iloc[sorted(keep)].reset_index(drop=True)


def dilute_candidates(
    candidates: pd.DataFrame,
    cfg,
    min_distance_m: float | None = None,
    max_per_grid: int | None = None,
) -> pd.DataFrame:
    """按最小间距约束稀释候选点：``d(p, q) >= c_min``。

    为什么必须**分类别**稀释
    ------------------------
    屋顶型与地面型是两类性质不同的设施：屋顶型不需要征地、单站成本低但
    容量受限；地面型需要征地、成本高但容量大。如果把它们放进同一个池子
    按同一质量分排序稀释，会出现系统性的淘汰偏差——地面型候选本质上是
    "每个 500 m 栅格的中心"，天然密集成网，且其质量分由路网可达性给出
    （容易接近满分）；屋顶型候选稀疏且质量分来自屋顶面积与高度。实测中
    这会导致**屋顶型候选 100% 被淘汰**，使"屋顶/地面双类别"这一建模前提
    在候选阶段就失效。

    因此本实现采用三阶段稀释：

    1. **类别内配额**——按栅格限额，避免核心区候选过度集中；
    2. **类别内最小间距**——在每个类别内部保证 ``c_min`` 间距，使同类
       候选不冗余；
    3. **跨类别冲突裁决**——不同类别的候选若相距小于 ``c_min``，保留
       **单位容量成本更低**的一个（屋顶型通常胜出），因为二者服务能力
       重叠时，经济性应成为裁决依据。

    这样既保证了最小间距约束在最终候选集上成立，又让两个类别都能进入
    优化阶段——这正是本研究"设施类别异质性"贡献得以成立的前提。

    Parameters
    ----------
    min_distance_m
        最小间距阈值 ``c_min``；默认取 ``stage1_candidates.dilution.min_distance_m``。
    """
    if candidates is None or len(candidates) == 0:
        return candidates

    dc = cfg.get("stage1_candidates.dilution", {})
    c_min = float(min_distance_m if min_distance_m is not None else dc.get("min_distance_m", 500.0))
    quota = int(max_per_grid if max_per_grid is not None else
                cfg.get("stage1_candidates.max_candidates_per_grid", 3))
    n0 = len(candidates)

    df = candidates.copy()

    # -- 步骤 1：类别内配额 ------------------------------------------------
    if "cell_id" not in df.columns:
        df["_bucket_x"] = (df["x"] // c_min).astype(int)
        df["_bucket_y"] = (df["y"] // c_min).astype(int)
        buckets = ["_bucket_x", "_bucket_y"]
    else:
        buckets = ["cell_id"]

    # 防呆：分桶列一旦含 NaN，pandas 的 groupby 会**静默丢弃**整行。
    # 这曾导致全部屋顶型候选被无声删除（其 cell_id 来自建筑而非栅格，
    # 若上游未补齐即为 NaN）。这里显式检查并修补，绝不让它再次静默发生。
    for col in buckets:
        if df[col].isna().any():
            n_nan = int(df[col].isna().sum())
            logger.error(
                "分桶列 '%s' 含 %d 个 NaN——groupby 会静默丢弃这些行！"
                "已用坐标分桶修补，但请检查上游的 cell_id 赋值。",
                col, n_nan,
            )
            cell = float(min_distance_m or dc.get("min_distance_m", 500.0))
            if col == "cell_id":
                bx = (df["x"] // cell).astype(int).astype(str)
                by = (df["y"] // cell).astype(int).astype(str)
                fill = pd.Series("bucket_" + bx + "_" + by, index=df.index)
                # 先转 object，否则向 float64 列填字符串会抛 TypeError
                df[col] = df[col].astype(object).where(df[col].notna(), fill)

    df = df.sort_values("quality", ascending=False)
    df["_rank"] = df.groupby(["facility_type"] + buckets, observed=True).cumcount()
    df = df[df["_rank"] < quota].drop(columns=["_rank"] + [b for b in buckets if b.startswith("_bucket")])
    n_quota = len(df)
    if n_quota == 0:
        logger.warning("配额过滤后候选为空")
        return candidates.iloc[0:0]

    # -- 步骤 2：类别内最小间距 --------------------------------------------
    parts = []
    for ftype, sub in df.groupby("facility_type", observed=True):
        kept = _greedy_min_distance(sub, c_min)
        logger.info("  [%s] 类别内稀释: %d -> %d", ftype, len(sub), len(kept))
        parts.append(kept)
    df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    n_intra = len(df)
    if n_intra == 0:
        return candidates.iloc[0:0]

    # -- 步骤 3：跨类别冲突裁决（保留单位容量成本更低者）------------------
    df = df.copy()
    df["_cost_per_capacity"] = df["cost"] / df["capacity"].replace(0, np.nan)
    df["_cost_per_capacity"] = df["_cost_per_capacity"].fillna(df["cost"])
    # 值越小越好 -> 转成可排序的"得分"（越大越好）
    cpc = df["_cost_per_capacity"].to_numpy()
    rng_ = np.ptp(cpc)
    df["_rank_score"] = 1.0 - (cpc - cpc.min()) / rng_ if rng_ > 0 else 1.0

    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    order = np.argsort(-df["_rank_score"].to_numpy(), kind="stable")
    keep: list[int] = []
    kept: list[np.ndarray] = []
    kept_type: list[str] = []
    dropped_cross = 0
    for idx in order:
        p = pts[idx]
        typ = df["facility_type"].iloc[idx]
        if kept:
            d = np.linalg.norm(np.asarray(kept) - p, axis=1)
            near = np.flatnonzero(d < c_min)
            # 只与"异类"近邻冲突时才需要裁决；同类冲突已在步骤 2 消除
            if any(kept_type[k] != typ for k in near):
                dropped_cross += 1
                continue
        keep.append(int(idx))
        kept.append(p)
        kept_type.append(typ)

    out = df.iloc[sorted(keep)].drop(
        columns=[c for c in df.columns if c.startswith("_")], errors="ignore"
    ).reset_index(drop=True)

    by_type = out["facility_type"].value_counts().to_dict()
    logger.info(
        "密度稀释 (c_min=%.0fm, 每栅格<=%d): %d -> %d 个候选 "
        "（配额 -%d，类别内间距 -%d，跨类别冲突 -%d）；构成: %s",
        c_min, quota, n0, len(out), n0 - n_quota, n_quota - n_intra,
        dropped_cross, by_type,
    )
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_candidate_set(
    cfg,
    buildings=None,
    grid=None,
    landuse=None,
    water=None,
    slope_raster=None,
    airports=None,
    roads=None,
) -> CandidateSet:
    """执行阶段一全流程，返回 :class:`CandidateSet`。

    Parameters
    ----------
    buildings
        已融合高度的建筑数据（``None`` 时跳过屋顶型）。
    grid
        需求栅格（``None`` 时自动用 :func:`geo.make_grid` 生成）。
    roads
        路网数据；提供时用"到路网的距离"作为地面型候选的质量分——
        交通可达性好的地面地块更适合建起降场。
    """
    from .geo import make_grid

    parts: list[pd.DataFrame] = []

    if buildings is not None and len(buildings) > 0:
        roof = build_rooftop_candidates(buildings, cfg)
        if len(roof):
            parts.append(roof)

    if grid is None:
        grid = make_grid(cfg)
    ground = build_ground_candidates(
        grid, cfg, landuse=landuse, water=water, slope_raster=slope_raster,
        buildings=buildings,
    )
    if len(ground):
        # 用路网距离改进地面型质量分
        if roads is not None and len(roads) > 0:
            ground = _score_by_road_access(ground, roads, grid)
        parts.append(ground)

    if not parts:
        raise RuntimeError(
            "阶段一未产生任何候选点。请检查：建筑数据是否为空、"
            "高度阈值是否过严、用地排除是否过度。"
        )

    cand = pd.concat(parts, ignore_index=True)

    # pd.concat 会把 GeoDataFrame 降级为普通 DataFrame，这里显式还原，
    # 否则下游所有依赖 .crs / sjoin 的空间操作都会失败。
    import geopandas as gpd

    if not isinstance(cand, gpd.GeoDataFrame):
        cand = gpd.GeoDataFrame(cand, geometry="geometry", crs=cfg.crs_proj)
    if cand.crs is None:
        cand = cand.set_crs(cfg.crs_proj, allow_override=True)

    # ------------------------------------------------------------------
    # 为**所有**候选补齐 cell_id（所属需求栅格单元）。
    #
    # 这一步是必需的，不是可有可无的整洁处理：屋顶型候选来自建筑数据，
    # 本身没有 cell_id（那是栅格单元才有的字段），concat 后该列在屋顶型
    # 行上全是 NaN。而下游密度稀释用 groupby(["facility_type", "cell_id"])
    # 做配额，**pandas 的 groupby 默认丢弃键为 NaN 的行**——结果是全部
    # 屋顶型候选被静默删除，候选集退化为纯地面型，使"屋顶/地面双类别"
    # 的建模前提在候选阶段就失效。
    #
    # 用空间连接把每个候选点归入其所在的 500 m 单元，从根上消除 NaN。
    # ------------------------------------------------------------------
    if cand["cell_id"].isna().any():
        n_missing = int(cand["cell_id"].isna().sum())
        try:
            pts = cand[["geometry"]].copy()
            pts["geometry"] = cand.geometry.centroid
            joined = gpd.sjoin(
                pts, grid[["cell_id", "geometry"]], how="left", predicate="within"
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            cand["cell_id"] = cand["cell_id"].fillna(joined["cell_id"])
            n_filled = int(cand["cell_id"].notna().sum())
            logger.info(
                "已为 %d 个缺少 cell_id 的候选（屋顶型）补齐所属栅格单元"
                "（补齐后 %d/%d 有 cell_id）",
                n_missing, n_filled, len(cand),
            )
        except Exception as exc:
            logger.warning("cell_id 空间补齐失败: %s", str(exc)[:180])

        # 兜底：仍未归入任何单元的（落在研究区边界外的），用坐标分桶，
        # 保证绝不留下 NaN——NaN 会再次触发 groupby 静默丢行。
        still = cand["cell_id"].isna()
        if still.any():
            cell = float(cfg.get("stage1_candidates.grid_size_m", 500.0))
            # 必须先把列转成 object：该列因含 NaN 而是 float64，
            # 直接写入字符串会抛 TypeError（Invalid value ... for dtype 'float64'）。
            cand["cell_id"] = cand["cell_id"].astype(object)
            bx = (cand.loc[still, "x"] // cell).astype(int).astype(str)
            by = (cand.loc[still, "y"] // cell).astype(int).astype(str)
            cand.loc[still, "cell_id"] = ("bucket_" + bx + "_" + by).values
            logger.info("  %d 个候选落在栅格之外，改用坐标分桶作为 cell_id", int(still.sum()))

    # 安全排除必须在稀释之前：先剔除不可行点，再控制密度，
    # 否则被剔除点会"占用"间距名额，造成候选无谓稀疏。
    cand = apply_safety_exclusions(cand, airports, cfg)

    cand = dilute_candidates(cand, cfg)
    cand = cand.reset_index(drop=True)
    cand["cand_id"] = np.arange(len(cand), dtype=np.int32)

    keep_cols = [
        "cand_id", "facility_type", "x", "y", "height_m", "roof_usable_m2",
        "parcel_area_m2", "height_source", "osm_id", "building_type",
        "n_fato", "capacity", "cost", "quality", "cell_id",
    ]
    for c in keep_cols:
        if c not in cand.columns:
            cand[c] = np.nan
    out = cand[keep_cols + ["geometry"]].copy()

    cs = CandidateSet(gdf=out)
    logger.info(
        "阶段一完成: %d 个候选起降场（屋顶型 %d，地面型 %d）",
        len(cs), (out["facility_type"] == "rooftop").sum(),
        (out["facility_type"] == "ground").sum(),
    )
    return cs


def _score_by_road_access(ground: pd.DataFrame, roads, grid) -> pd.DataFrame:
    """用"到最近主干路的距离"改进地面型候选的质量分。

    起降场需要良好的地面接驳条件。距主干路越近，接驳成本越低。
    这里用简化规则：距主干路 <200 m 得满分，>1000 m 得 0 分，线性插值。
    """
    import geopandas as gpd

    try:
        major = roads[roads.get("highway", pd.Series(dtype=str)).isin(
            ["motorway", "trunk", "primary", "secondary", "tertiary"]
        )]
        if len(major) == 0:
            major = roads
        major = major.to_crs(grid.crs) if str(major.crs) != str(grid.crs) else major

        pts = gpd.GeoDataFrame(
            ground.copy(),
            geometry=gpd.points_from_xy(ground["x"], ground["y"]),
            crs=grid.crs,
        )
        near = gpd.sjoin_nearest(
            pts, major[["geometry"]], how="left", distance_col="dist_road_m"
        )
        near = near[~near.index.duplicated(keep="first")]
        d = near["dist_road_m"].fillna(5000.0).to_numpy()
        score = np.clip(1.0 - (d - 200.0) / 800.0, 0.0, 1.0)
        ground = ground.copy()
        ground["dist_road_m"] = d
        ground["quality"] = np.where(np.isfinite(ground["quality"]), 0.5 * ground["quality"] + 0.5 * score, score)
        logger.info("地面型候选已按路网可达性重新评分（中位距离 %.0f m）", float(np.median(d)))
    except Exception as exc:
        logger.warning("路网可达性评分失败，保留原质量分: %s", str(exc)[:160])
    return ground
