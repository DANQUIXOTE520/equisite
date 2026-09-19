"""合成城市数据生成器（仅用于离线测试与流程验证）。

**警告**：本模块生成的数据是**虚构的**，只能用于：
1. 在没有网络/数据的环境下验证四阶段流程能否端到端跑通；
2. 单元测试与调试；
3. 教学演示。

**绝不可用于论文的实证结论。** 所有由合成数据产生的图表都必须在
文件名与图注中标注 ``SYNTHETIC``，避免误用。

生成逻辑刻意模仿成都的形态特征，使调试时的中间结果在量级上接近真实：
* 单中心（天府广场）+ 南向次中心（天府新区）的密度梯度；
* 建筑高度呈重尾分布——绝大多数为 6–30 m 的多层住宅，少数核心区
  建筑达 100–250 m；
* 建筑高度标签的**缺失模式**刻意复刻 OSM 实测情况（仅约 5% 有
  ``height``、约 15% 有 ``building:levels``），以便真实检验
  :mod:`evtol_siting.data.heights` 的融合逻辑。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _lonlat_to_proj(cfg, lon, lat):
    """单点经纬度 -> 投影坐标。"""
    import geopandas as gpd

    gs = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=cfg.crs_geo).to_crs(cfg.crs_proj)
    return float(gs.x.iloc[0]), float(gs.y.iloc[0])


def synthetic_buildings(cfg, n_buildings: int = 60000, seed: int | None = None):
    """生成合成建筑轮廓（投影坐标系）。

    返回的列与真实 OSM 下载 + 解析后的结构一致：
    ``osm_id`` / ``building`` / ``height`` / ``building:levels`` / ``geometry``。

    高度标签的**缺失率**按实测设置，用于验证高度融合模块。

    数量与尺寸标定
    --------------
    研究区约 630 km²，其中建成区约占一半。成都建成区建筑密度取
    1500–2500 栋/km²（含住宅楼栋），故 60 000 栋对应约 150 km² 的高密度
    城区，其余为低密度外围——这与"单中心 + 南向延伸"的撒点权重一致。
    占地面积取对数正态（中位数约 380 m²，即 20 m × 19 m 的典型住宅楼），
    并带重尾：约 4% 的建筑超过 2000 m²（商场、医院、厂房、高校），
    使屋顶型候选（需 >= 2286 m² 基底）在量级上接近真实情形。
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    minx, miny, maxx, maxy = _bbox_proj(cfg)

    # 两个中心（天府广场 / 天府新区）
    c1 = _lonlat_to_proj(cfg, 104.0648, 30.6570)
    c2 = _lonlat_to_proj(cfg, 104.0700, 30.4800)

    # 直接在投影坐标下撒点，密度按到双中心的距离衰减
    pts, tries = [], 0
    while len(pts) < n_buildings and tries < n_buildings * 30:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        d1 = np.hypot(x - c1[0], y - c1[1]) / 1000.0
        d2 = np.hypot(x - c2[0], y - c2[1]) / 1000.0
        p = 0.85 * np.exp(-(d1 ** 2) / (2 * 5.0 ** 2)) + 0.25 * np.exp(-(d2 ** 2) / (2 * 7.0 ** 2))
        if rng.random() < p:
            pts.append((x, y, min(d1, d2 + 8.0)))

    lonlat = gpd.GeoSeries(
        gpd.points_from_xy([p[0] for p in pts], [p[1] for p in pts]), crs=cfg.crs_proj
    ).to_crs(cfg.crs_geo)
    lon = lonlat.x.to_numpy()
    lat = lonlat.y.to_numpy()

    # -- 建筑高度：重尾分布，核心区更高 -----------------------------------
    d_center = np.array([p[2] for p in pts])
    # 基础高度：对数正态，中位数约 15 m（5–6 层住宅）
    base = rng.lognormal(mean=np.log(14.0), sigma=0.55, size=len(pts))
    # 核心区加成：距中心 3 km 内显著拔高
    boost = 1.0 + 2.6 * np.exp(-(d_center ** 2) / (2 * 2.2 ** 2))
    height = base * boost
    # 少量超高层地标
    n_tall = max(1, int(0.004 * len(pts)))
    tall_idx = rng.choice(len(pts), size=n_tall, replace=False)
    height[tall_idx] = rng.uniform(100, 250, size=n_tall)
    height = np.clip(height, 3.0, 280.0)

    # 占地面积：中位数约 380 m²（20 m × 19 m 典型住宅楼），带重尾。
    # 用两个对数正态混合：主体为住宅，约 6% 为大型公共/工业建筑。
    is_large = rng.random(len(pts)) < 0.06
    edge_small = rng.lognormal(mean=np.log(20.0), sigma=0.42, size=len(pts))
    edge_large = rng.lognormal(mean=np.log(58.0), sigma=0.45, size=len(pts))
    edge = np.where(is_large, edge_large, edge_small)
    # 高层建筑占地偏小（容积率高），但地标塔楼仍可能有大底盘
    edge = edge * (1.0 - 0.22 * np.clip(height / 150.0, 0, 1))
    edge = np.clip(edge, 7.0, 190.0)

    geoms, types, h_tag, lvl_tag = [], [], [], []
    btype_choices = ["yes", "residential", "commercial", "office", "retail", "industrial", "hotel"]

    for i in range(len(pts)):
        x, y = pts[i][0], pts[i][1]
        w, d = edge[i], edge[i] * rng.uniform(0.6, 1.6)
        # 轻微旋转，避免完全正交的网格感
        th = rng.uniform(0, np.pi / 2)
        dx, dy = w / 2, d / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners = corners @ rot.T + np.array([x, y])
        geoms.append(Polygon(corners))

        h = height[i]
        btype = "residential" if h < 30 else rng.choice(btype_choices)
        types.append(btype)

        # -- 复刻 OSM 的标签缺失模式（实测：height 约 5%，levels 约 15%）--
        # 关键：让标签的"是否有值"与建筑真实高度弱相关（真实 OSM 中
        # 高知名度建筑更可能被标注），从而检验融合模块的无偏性
        p_height = 0.05 + 0.10 * (h > 60)
        p_levels = 0.15 + 0.20 * (h > 30)
        if rng.random() < p_height:
            h_tag.append(f"{h:.0f}")
        else:
            h_tag.append(None)
        if rng.random() < p_levels:
            lvl_tag.append(str(max(1, int(round(h / 3.3)))))
        else:
            lvl_tag.append(None)

    gdf = gpd.GeoDataFrame(
        {
            "osm_id": np.arange(1, len(pts) + 1),
            "osm_type": "way",
            "building": types,
            "height": h_tag,
            "building:levels": lvl_tag,
            # 供核对：真实高度（模拟中已知，实际数据里不存在）
            "true_height_m": height,
        },
        geometry=geoms,
        crs=cfg.crs_proj,
    )
    logger.info(
        "生成合成建筑 %d 栋（height 标签 %.1f%%，levels 标签 %.1f%%）",
        len(gdf), 100 * gdf["height"].notna().mean(), 100 * gdf["building:levels"].notna().mean(),
    )
    return gdf


def _bbox_proj(cfg):
    from ..geo import bbox_projected

    return bbox_projected(cfg)


def synthetic_poi(cfg, n: int = 20000, seed: int | None = None):
    """生成合成 POI 点（与 OSM POI 解析后的结构一致）。"""
    import geopandas as gpd

    rng = np.random.default_rng((seed if seed is not None else cfg.seed) + 7)
    minx, miny, maxx, maxy = _bbox_proj(cfg)
    c1 = _lonlat_to_proj(cfg, 104.0648, 30.6570)
    c2 = _lonlat_to_proj(cfg, 104.0700, 30.4800)

    pts = []
    while len(pts) < n:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        d1 = np.hypot(x - c1[0], y - c1[1]) / 1000.0
        d2 = np.hypot(x - c2[0], y - c2[1]) / 1000.0
        p = 0.95 * np.exp(-(d1 ** 2) / (2 * 2.8 ** 2)) + 0.3 * np.exp(-(d2 ** 2) / (2 * 4.0 ** 2))
        if rng.random() < p:
            pts.append((x, y))

    cats = ["amenity", "shop", "office", "leisure", "tourism"]
    return gpd.GeoDataFrame(
        {
            "osm_id": np.arange(1, len(pts) + 1),
            "osm_type": "node",
            "poi_category": rng.choice(cats, size=len(pts)),
        },
        geometry=gpd.points_from_xy([p[0] for p in pts], [p[1] for p in pts]),
        crs=cfg.crs_proj,
    )


def synthetic_roads(cfg, seed: int | None = None):
    """生成合成路网（主/次干路网格 + 环路），用于路网可达性评分。

    ⚠️ 本函数生成的是**示意性**路网，**不能用于基于路网的接驳时间计算**。

    原因是它按构造就是**拓扑断裂**的：每条线独立生成，交叉处并不共享
    顶点（真实 OSM 路网在路口共享同一节点，因此是连通的）。实测这样生成
    的路网最大连通分量仅占全部节点的约 2%，用它算最短路会得到大量
    "不可达"，结果没有意义。

    需要路网口径的接驳时间时，必须用真实路网数据（PBF 通道的
    ``roads`` 图层，见 ``data/pbf.py``）。本函数生成的线仅可用于：
    地面型候选的"到路网距离"评分（只用到距离，不涉及连通性）。
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    rng = np.random.default_rng((seed if seed is not None else cfg.seed) + 11)
    minx, miny, maxx, maxy = _bbox_proj(cfg)
    step = 1200.0
    lines, hw = [], []

    for x in np.arange(minx, maxx, step):
        # 加入轻微弯曲，避免纯粹正交网格
        ys = np.arange(miny, maxy, step)
        pts = [(x + rng.normal(0, 40), y) for y in ys]
        if len(pts) >= 2:
            lines.append(LineString(pts))
            hw.append("primary" if rng.random() < 0.25 else "residential")
    for y in np.arange(miny, maxy, step):
        xs = np.arange(minx, maxx, step)
        pts = [(x, y + rng.normal(0, 40)) for x in xs]
        if len(pts) >= 2:
            lines.append(LineString(pts))
            hw.append("secondary" if rng.random() < 0.4 else "residential")

    logger.info("生成合成路网 %d 条", len(lines))
    return gpd.GeoDataFrame(
        {"highway": hw, "osm_id": np.arange(1, len(lines) + 1)},
        geometry=lines, crs=cfg.crs_proj,
    )


def synthetic_landuse(cfg, water_fraction: float = 0.06, seed: int | None = None):
    """生成合成水域与绿地，用于验证用地排除逻辑。"""
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng((seed if seed is not None else cfg.seed) + 13)
    minx, miny, maxx, maxy = _bbox_proj(cfg)
    w, h = maxx - minx, maxy - miny

    polys, kinds = [], []
    n_water = 14
    for _ in range(n_water):
        cx = rng.uniform(minx + 0.1 * w, maxx - 0.1 * w)
        cy = rng.uniform(miny + 0.1 * h, maxy - 0.1 * h)
        rx, ry = rng.uniform(300, 1400), rng.uniform(200, 900)
        ang = np.linspace(0, 2 * np.pi, 24)
        r = 1.0 + 0.35 * np.sin(3 * ang + rng.uniform(0, 6))
        polys.append(
            __import__("shapely.geometry", fromlist=["Polygon"]).Polygon(
                np.column_stack([cx + rx * r * np.cos(ang), cy + ry * r * np.sin(ang)])
            )
        )
        kinds.append(("natural", "water"))

    n_green = 20
    for _ in range(n_green):
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        polys.append(Point(x, y).buffer(rng.uniform(400, 1800), resolution=8))
        kinds.append(("landuse", "forest"))

    return gpd.GeoDataFrame(
        {
            "tag_key": [k[0] for k in kinds],
            "landuse": [k[1] if k[0] == "landuse" else None for k in kinds],
            "natural": [k[1] if k[0] == "natural" else None for k in kinds],
        },
        geometry=polys, crs=cfg.crs_proj,
    )


def synthetic_airports(cfg):
    """生成两个运输机场（对应成都双流 CTU 与天府 TFU）用于净空区约束。

    坐标使用真实机场位置——这是公开信息，不属于虚构数据。
    """
    import geopandas as gpd

    coords = [(103.9471, 30.5785, "aerodrome"),   # 成都双流国际机场 CTU
              (104.4419, 30.3125, "aerodrome")]   # 成都天府国际机场 TFU
    xs, ys = [], []
    for lon, lat, _ in coords:
        x, y = _lonlat_to_proj(cfg, lon, lat)
        xs.append(x)
        ys.append(y)

    return gpd.GeoDataFrame(
        {"aeroway": [c[2] for c in coords], "name": ["Chengdu Shuangliu (CTU)", "Chengdu Tianfu (TFU)"]},
        geometry=gpd.points_from_xy(xs, ys), crs=cfg.crs_proj,
    )
