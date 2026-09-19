"""建筑高度多源融合与屋顶可用面积估计。

问题背景（这是本研究一个实质性的数据方法贡献）
----------------------------------------------
屋顶型起降场的可行性判据是 ``h >= 20 m`` 且 ``A_roof >= 1600 m²``，
因此**建筑高度数据质量直接决定候选集的可信度**。实测成都市中心的
OSM 数据：

===============  ==========  ==========
区域             height 标签  levels 标签
===============  ==========  ==========
天府广场片区      5.4%        19.7%
城南某 2km 瓦片   1.5%        8.5%
===============  ==========  ==========

仅用 OSM 标签会漏掉 80% 以上的高层建筑，使候选集系统性偏少。
本模块因此实施**四级融合**，并为每栋建筑记录高度来源
（``height_source``），以便在论文中报告各来源的贡献占比与不确定性。

融合优先级
----------
1. ``osm_height``   —— OSM ``height`` 标签（最可信，实测占比低）
2. ``osm_levels``   —— ``building:levels`` × 层高（按建筑类型取 3.0–4.5 m）
3. ``raster_height``—— 建筑高度栅格（CNBH-10m / GlobalBuildingAtlas）
                       在建筑质心处采样；对无标签建筑提供独立信息
4. ``imputed``      —— 基于建筑面积的分位数回归填补（最后的兜底）

对第 3、4 级来源，同时给出 ``height_sigma``（不确定度），供敏感性分析
与论文中的误差讨论使用。
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 不同建筑类型的层高（米）。取值依据《民用建筑设计统一标准》GB 50352
# 与常见工程实践：住宅层高约 2.8–3.0 m，办公 3.3–3.9 m，商业/酒店更大。
LEVEL_HEIGHT_M: dict[str, float] = {
    "residential": 3.0,
    "apartments": 3.0,
    "house": 3.0,
    "detached": 3.0,
    "terrace": 3.0,
    "dormitory": 3.0,
    "office": 3.6,
    "commercial": 4.2,
    "retail": 4.2,
    "industrial": 4.5,
    "warehouse": 4.5,
    "hotel": 3.6,
    "hospital": 3.9,
    "school": 3.9,
    "university": 3.9,
    "civic": 3.9,
    "public": 3.9,
    "government": 3.9,
    "church": 6.0,
    "stadium": 8.0,
    "parking": 3.0,
    "garage": 3.0,
    "garages": 3.0,
    "shed": 3.0,
    "roof": 3.0,
    "yes": 3.3,      # 未标注类型——取住宅与办公的折中
}
DEFAULT_LEVEL_HEIGHT_M = 3.3

# 屋顶附属物折减：电梯机房、水箱、空调机组、女儿墙、光伏等占用
# 该系数在配置中可覆盖（stage1_candidates.rooftop.usable_roof_ratio）
DEFAULT_USABLE_ROOF_RATIO = 0.70


# ---------------------------------------------------------------------------
# 标签解析
# ---------------------------------------------------------------------------

# OSM height 标签的常见写法: "20", "20 m", "20m", "65'", "20;25"
_HEIGHT_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_height_tag(value) -> float | None:
    """解析 OSM ``height`` 标签为米。

    支持 ``"20"`` / ``"20 m"`` / ``"20m"`` / ``"65'"``（英尺）/ 分号分隔的
    多值（取最大值）。无法解析时返回 ``None``（而不是 0——0 会被误认为
    "平地建筑"，从而错误地排除合法候选）。
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "unknown", "?"}:
        return None

    # 多值取最大（如 "20;25" 表示分段高度）
    if ";" in s:
        parts = [parse_height_tag(p) for p in s.split(";")]
        vals = [p for p in parts if p is not None]
        return max(vals) if vals else None

    is_feet = "'" in s or "ft" in s
    m = _HEIGHT_NUM_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None

    if is_feet:
        v *= 0.3048
    # 合理性检查：单栋建筑 0–800 m。超出则视为数据错误
    if not (0.0 < v <= 800.0):
        return None
    return v


def parse_levels_tag(value) -> float | None:
    """解析 ``building:levels`` 标签为层数。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "unknown"}:
        return None
    if ";" in s:
        parts = [parse_levels_tag(p) for p in s.split(";")]
        vals = [p for p in parts if p is not None]
        return max(vals) if vals else None
    m = _HEIGHT_NUM_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    # 合理性检查：地面以上层数 1–200
    if not (0.5 <= v <= 200.0):
        return None
    return v


def _level_height_for(building_type: str | None) -> float:
    """按建筑类型取层高（米）。"""
    if not building_type:
        return DEFAULT_LEVEL_HEIGHT_M
    key = str(building_type).strip().lower()
    if key in LEVEL_HEIGHT_M:
        return LEVEL_HEIGHT_M[key]
    # 处理 "yes"、"commercial;office" 等复合值
    for part in re.split(r"[;,]", key):
        part = part.strip()
        if part in LEVEL_HEIGHT_M:
            return LEVEL_HEIGHT_M[part]
    return DEFAULT_LEVEL_HEIGHT_M


# ---------------------------------------------------------------------------
# 主融合流程
# ---------------------------------------------------------------------------

def fuse_building_heights(
    buildings,
    cfg,
    height_raster: str | None = None,
    raster_crs: str | None = None,
) -> "object":
    """对建筑 GeoDataFrame 执行四级高度融合。

    Parameters
    ----------
    buildings
        含 ``geometry``（投影坐标系下的多边形）与可选 ``height`` /
        ``building:levels`` / ``building`` 列的 GeoDataFrame。
    height_raster
        建筑高度栅格路径；``None`` 时跳过第 3 级。

    Returns
    -------
    原 GeoDataFrame 的副本，新增列：

    ``footprint_area_m2``  建筑占地面积
    ``n_levels``           解析出的层数
    ``height_m``           融合后的建筑高度（米）
    ``height_source``      来源：osm_height / osm_levels / raster_height / imputed
    ``height_sigma_m``     高度不确定度（米），用于敏感性分析
    """
    import geopandas as gpd

    if buildings is None or len(buildings) == 0:
        logger.warning("建筑数据为空，跳过高度融合")
        return buildings

    gdf = buildings.copy()
    n_total = len(gdf)

    # -- 几何派生量 --------------------------------------------------------
    geom = gdf.geometry
    if not geom.is_valid.all():
        gdf["geometry"] = geom.buffer(0)
        geom = gdf.geometry
    gdf["footprint_area_m2"] = geom.area.astype(np.float64)

    # -- 第 1 级：OSM height ----------------------------------------------
    h_col = next((c for c in ("height", "building:height") if c in gdf.columns), None)
    h_osm = (
        gdf[h_col].map(parse_height_tag).astype(float)
        if h_col
        else pd.Series(np.nan, index=gdf.index)
    )

    # -- 第 2 级：OSM building:levels -------------------------------------
    lvl_col = next(
        (c for c in ("building:levels", "levels", "building:level") if c in gdf.columns),
        None,
    )
    n_levels = (
        gdf[lvl_col].map(parse_levels_tag).astype(float)
        if lvl_col
        else pd.Series(np.nan, index=gdf.index)
    )
    gdf["n_levels"] = n_levels

    btype = gdf["building"] if "building" in gdf.columns else pd.Series(None, index=gdf.index)
    lvl_h = btype.map(_level_height_for).astype(float)
    h_from_levels = n_levels * lvl_h

    # 地面层附加高度（首层通常更高）
    h_from_levels = np.where(n_levels.notna(), h_from_levels + 0.6, np.nan)

    height = np.full(n_total, np.nan, dtype=np.float64)
    source = np.array(["none"] * n_total, dtype=object)
    sigma = np.full(n_total, np.nan, dtype=np.float64)

    m1 = h_osm.notna().values
    height[m1] = h_osm.values[m1]
    source[m1] = "osm_height"
    sigma[m1] = 0.5                       # 实测标签，误差约 ±0.5 m

    m2 = (~m1) & pd.notna(h_from_levels)
    height[m2] = np.asarray(h_from_levels)[m2]
    source[m2] = "osm_levels"
    sigma[m2] = 1.5                       # 层高假设带来的误差

    # -- 第 3 级：高度栅格 ------------------------------------------------
    n_before_raster = int((source != "none").sum())
    if height_raster:
        from ..geo import sample_raster_at_points

        need = source == "none"
        if need.any():
            cx = gdf.geometry.centroid.x.values
            cy = gdf.geometry.centroid.y.values
            try:
                vals = sample_raster_at_points(
                    height_raster,
                    cx[need],
                    cy[need],
                    src_crs=raster_crs,
                    dst_crs=cfg.crs_proj,
                )
                # 栅格中的 0 通常表示"无建筑"，不可当作高度 0
                valid = np.isfinite(vals) & (vals > 0.5)
                idx = np.flatnonzero(need)
                sel = idx[valid]
                height[sel] = vals[valid]
                source[sel] = "raster_height"
                sigma[sel] = 3.0          # 10 m 栅格的不确定度
                logger.info(
                    "高度栅格提供了 %d 栋建筑的高度（覆盖 %.1f%%）",
                    len(sel), 100 * len(sel) / n_total,
                )
            except Exception as exc:
                logger.warning("高度栅格采样失败，跳过第 3 级: %s", str(exc)[:200])

    n_after_raster = int((source != "none").sum())
    # -- 第 4 级：基于占地面积的分位数回归填补 -----------------------------
    # 说明：只用第 3 级之后的缺口，前两级已有标签的建筑不再改动。
    # 核心假设：占地面积越大的建筑越可能是高层（大型楼宇多为商业/公共建筑）。
    # 用**已有标签**的建筑拟合 log(面积) -> 高度的分位数映射，再对外推。
    need = source == "none"
    if need.any():
        known = (source != "none") & np.isfinite(height)
        if known.sum() >= 30:
            area_known = np.log1p(gdf["footprint_area_m2"].values[known])
            h_known = height[known]
            # 分位数分箱：按面积分成若干箱，取箱内高度中位数
            nbins = min(20, max(5, known.sum() // 50))
            try:
                bins = pd.qcut(area_known, q=nbins, duplicates="drop")
            except ValueError:
                bins = pd.cut(area_known, bins=nbins)
            med = pd.Series(h_known).groupby(bins, observed=True).median()
            med.index = med.index.astype(object)

            area_need = np.log1p(gdf["footprint_area_m2"].values[need])
            try:
                b_need = pd.cut(
                    area_need,
                    bins=[iv.left for iv in med.index] + [med.index[-1].right],
                    include_lowest=True,
                )
                pred = b_need.map(med).astype(float).values
            except Exception:
                pred = np.full(need.sum(), np.nanmedian(h_known))

            idx = np.flatnonzero(need)
            fill = np.where(np.isfinite(pred), pred, np.nanmedian(h_known))
            height[idx] = fill
            source[idx] = "imputed"
            sigma[idx] = 6.0          # 填补值不确定度最大
            logger.info(
                "分位数回归填补了 %d 栋建筑的高度（中位数 %.1f m）",
                len(idx), float(np.nanmedian(fill)),
            )
        else:
            # 标记数据太少，无法建立面积-高度关系：用全局先验
            idx = np.flatnonzero(need)
            height[idx] = 12.0
            source[idx] = "imputed"
            sigma[idx] = 10.0
            logger.warning(
                "带高度标签的建筑仅 %d 栋，不足 30 栋，%.0f 栋使用全局先验 12 m",
                int(known.sum()), len(idx),
            )

    gdf["height_m"] = height
    gdf["height_source"] = source
    gdf["height_sigma_m"] = sigma

    # -- 来源构成报告（论文表 1 直接可用）---------------------------------
    #
    # ⚠ 审计：论文 §3.3 引用了三个数——「带标签 CNBH 中位数 23.0 m（n = 5 304），
    #   无标签 18.7 m（n = 49 169）」——但**这三个数在全部日志与 outputs/ 中
    #   都没有来源**，而且 5 304 + 49 169 = 54 473 ≠ 总建筑数。原因就是这里
    #   只打比例、不打**每个来源的栋数与中位数**，论文作者只能凭印象写。
    #   现在把 (来源, 栋数, 占比, 中位数, 均值, 标准差) 一并打出，
    #   使正文的每个数都可追溯、且能被机械核对。
    share = pd.Series(source).value_counts(normalize=True).sort_values(ascending=False)
    logger.info(
        "高度融合结果（共 %d 栋）:\n%s",
        n_total,
        "\n".join(f"    {k:16s} {v*100:5.1f}%" for k, v in share.items()),
    )

    # 逐来源的栋数与高度分布——论文 §3.3 的数字出处
    hs = pd.Series(np.asarray(height, dtype=float))
    src = pd.Series(np.asarray(source))
    tbl = pd.DataFrame({"height_m": hs, "src": src}).dropna(subset=["height_m"])
    by_src = (tbl.groupby("src")["height_m"]
              .agg(n="size", median="median", mean="mean", std="std")
              .sort_values("n", ascending=False))
    logger.info(
        "各来源的栋数与高度分布（论文 §3.3 引用这些数）:\n%s",
        by_src.to_string(float_format=lambda x: f"{x:,.2f}"),
    )
    logger.info(
        "    合计 %d 栋（逐来源栋数之和必须等于此数）", int(by_src["n"].sum()),
    )

    # -- §3.3 的选择偏差诊断：**同一把尺子**量带标签与无标签建筑 -------------
    #
    # 论文 §3.3 的核心论点是"OSM 高度标签偏向更高、测绘更充分的建筑"，
    # 用来支撑它的数是「带标签建筑的 **CNBH 实测**中位数 vs 无标签的」。
    #
    # ⚠ 这里必须**在两个子集上都用 CNBH 栅格**，而不能拿"带标签建筑的高度"
    #   （那是 OSM 标签值或层高换算值）去和"无标签建筑的栅格值"比——那样比的是
    #   **两个不同的测量口径**，得到的差异里混着口径差，证不了选择偏差。
    #   早期论文引用的「23.0 m（n = 5 304）」正是这么一个来源不明的数：
    #   它不在任何日志里，5 304 + 49 169 也凑不出总建筑数。
    #
    # 因此这里**额外采样一次栅格**（不参与融合，只用于诊断），使正文的
    # 每个数都有出处，且两个子集严格同口径。
    tagged = np.isin(np.asarray(source), ["osm_height", "osm_levels"])
    if height_raster and tagged.any() and (~tagged).any():
        try:
            from ..geo import sample_raster_at_points

            cx = gdf.geometry.centroid.x.values
            cy = gdf.geometry.centroid.y.values
            cnbh = sample_raster_at_points(
                height_raster, cx, cy, src_crs=raster_crs, dst_crs=cfg.crs_proj,
            )
            ok = np.isfinite(cnbh) & (cnbh > 0.5)
            for label, mask in (("带 OSM 高度标签", tagged),
                                ("无 OSM 高度标签", ~tagged)):
                m = mask & ok
                if m.sum():
                    logger.info(
                        "§3.3 诊断（同口径，均为 CNBH 栅格实测）: %s %d 栋，"
                        "中位数 %.1f m，均值 %.1f m",
                        label, int(m.sum()), float(np.median(cnbh[m])),
                        float(np.mean(cnbh[m])),
                    )
                else:
                    logger.warning("§3.3 诊断: %s 子集在栅格上无有效值", label)
            logger.info(
                "    （两个子集的中位数之差即 §3.3 所述的选择偏差；"
                "差值方向应为正值——标签偏向更高建筑）"
            )
        except Exception as exc:
            logger.warning("§3.3 选择偏差诊断失败: %s", str(exc)[:200])
    elif not height_raster:
        logger.warning("未提供高度栅格，无法做 §3.3 的同口径选择偏差诊断")

    logger.info(
        "    有标签覆盖率: %.1f%% -> 融合后覆盖率: 100.0%%",
        100 * n_before_raster / n_total,
    )

    return gdf


def estimate_roof_area(
    gdf,
    usable_ratio: float = DEFAULT_USABLE_ROOF_RATIO,
) -> "object":
    """估计屋顶可用面积。

    简化假设：建筑水平投影面积 ≈ 屋顶面积（对多层退台建筑会高估）。
    再乘以 ``usable_ratio`` 扣除电梯机房、水箱、空调机组、女儿墙等占用。

    对 OSM ``building:part`` 建模的建筑（成都极少），理论上可用各部件的
    最大高度面重建真实屋顶轮廓；本实现未采用，作为论文中的方法局限说明。
    """
    out = gdf.copy()
    out["roof_area_m2"] = out["footprint_area_m2"].astype(float)
    out["roof_usable_m2"] = out["roof_area_m2"] * float(usable_ratio)
    return out


def height_quality_report(gdf) -> pd.DataFrame:
    """生成高度数据质量汇总表（论文 Table 1）。

    返回列为 ``height_source`` 的计数、占比、高度均值/中位数/标准差。
    """
    if "height_source" not in gdf.columns:
        raise KeyError("请先调用 fuse_building_heights()")
    g = gdf.groupby("height_source")["height_m"]
    rep = pd.DataFrame(
        {
            "count": g.size(),
            "share_pct": (g.size() / len(gdf) * 100).round(2),
            "height_mean_m": g.mean().round(2),
            "height_median_m": g.median().round(2),
            "height_std_m": g.std().round(2),
        }
    ).sort_values("count", ascending=False)
    return rep.reset_index()
