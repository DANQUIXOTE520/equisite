"""基于路网的接驳时间计算。

为什么必须做这一步
------------------
默认的 :func:`evtol_siting.geo.travel_time_matrix` 用**直线距离**除以方式
速度估算接驳时间。这对成都这种"环线 + 放射"且被河流、铁路、封闭园区分割的
城市会系统性低估——两点直线 1 km，绕行可能要走 2.5 km。

文献调研显示这是本方法相对 *Computers, Environment and Urban Systems*、
*Transportation Research* 一类期刊的主要短板：那里的审稿人几乎一定会要求
路网口径。因此本模块把 OSM 路网建成图，用最短路计算真实接驳时间。

方法
----
1. 把 ``roads`` 图层（OSM ``highway`` 标签）建成有向图，节点为道路端点；
2. 按道路等级赋速度，边长除以速度得到通行时间；
3. 对每个需求单元做**半径截断**的最短路（``cutoff``），避免全局搜索。

第 3 步的截断是关键的性能优化：本研究只需知道"接驳半径内有哪些起降场
可达"，不需要全城最短路。实测未截断时单源 Dijkstra 会遍历全图（约 10 万
节点），2519 个源需要数十分钟；截断到 5 km 后每个源只访问局部子图，
总耗时降到分钟级。

与自由流欧氏口径的关系
----------------------
本模块**不替换**欧氏口径，而是与之并存：论文中应同时报告两种口径的结果
作为稳健性检验。若结论在两种口径下一致，说明选址方案对可达性度量方式不
敏感——这本身就是有价值的稳健性证据。
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 道路等级 -> 城市道路通行速度（km/h）。
# 取值参考《城市综合交通体系规划标准》GB/T 51328 与常见交通工程实践，
# 反映**城市内部**的实际平均行程速度（含信号交叉口延误），而非设计速度。
SPEED_BY_HIGHWAY: dict[str, float] = {
    "motorway": 70.0, "motorway_link": 45.0,
    "trunk": 55.0, "trunk_link": 35.0,
    "primary": 45.0, "primary_link": 30.0,
    "secondary": 38.0, "secondary_link": 28.0,
    "tertiary": 32.0, "tertiary_link": 25.0,
    "unclassified": 28.0,
    "residential": 22.0,
    "living_street": 12.0,
    "service": 15.0,
    "pedestrian": 5.0,
    "footway": 4.8, "path": 4.0, "steps": 2.0, "cycleway": 12.0,
}
DEFAULT_SPEED_KMH = 20.0

# 步行专用道路：短驳车不可通行，只能步行
FOOT_ONLY = {"footway", "path", "steps", "pedestrian", "cycleway"}


def build_road_graph(roads, cfg, speeds: dict[str, float] | None = None):
    """把 OSM 路网 GeoDataFrame 建成 ``networkx`` 有向图。

    Parameters
    ----------
    roads
        投影坐标系下的道路 GeoDataFrame，含 ``highway`` 列。
    speeds
        道路等级 -> 速度（km/h）映射；默认用 :data:`SPEED_BY_HIGHWAY`。

    Returns
    -------
    ``networkx.DiGraph``，节点属性 ``x`` / ``y``（投影坐标），
    边属性 ``length``（米）与 ``travel_time``（秒）。

    Notes
    -----
    * 用**坐标舍入到 0.1 m** 作为节点键，把 OSM 中因绘制精度造成的
      "同一点不同节点 ID"合并，否则路网会因微小坐标差而断裂——这是
      路网建模中最常见的坑，表现为大量点"不可达"。
    * 双向道路建两条有向边；``oneway`` 标签为 ``yes/-1`` 时只建一条。
    """
    import networkx as nx

    speeds = speeds or SPEED_BY_HIGHWAY
    if roads is None or len(roads) == 0:
        raise ValueError("道路数据为空，无法构建路网")

    gdf = roads
    if str(gdf.crs) != str(cfg.crs_proj):
        gdf = gdf.to_crs(cfg.crs_proj)

    hw_col = gdf["highway"] if "highway" in gdf.columns else pd.Series("unclassified", index=gdf.index)
    oneway_col = gdf["oneway"] if "oneway" in gdf.columns else pd.Series(None, index=gdf.index)

    G = nx.DiGraph()
    n_edges = 0
    n_skipped = 0

    for geom, hw, ow in zip(gdf.geometry.values, hw_col.values, oneway_col.values):
        if geom is None or geom.is_empty:
            continue
        gtype = geom.geom_type
        if gtype == "LineString":
            lines = [geom]
        elif gtype == "MultiLineString":
            lines = list(geom.geoms)
        else:
            n_skipped += 1
            continue

        hw_key = str(hw).split(";")[0].strip().lower() if hw else "unclassified"
        v_kmh = speeds.get(hw_key, DEFAULT_SPEED_KMH)
        v_ms = v_kmh * 1000.0 / 3600.0
        foot_ok = hw_key in FOOT_ONLY

        for line in lines:
            coords = np.asarray(line.coords)
            if len(coords) < 2:
                continue

            for a, b in zip(coords[:-1], coords[1:]):
                # 0.1 m 舍入合并同一物理节点
                ka = (round(float(a[0]), 1), round(float(a[1]), 1))
                kb = (round(float(b[0]), 1), round(float(b[1]), 1))
                if ka == kb:
                    continue

                seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                if seg_len <= 0:
                    continue
                t = seg_len / v_ms

                if ka not in G:
                    G.add_node(ka, x=ka[0], y=ka[1])
                if kb not in G:
                    G.add_node(kb, x=kb[0], y=kb[1])

                # 反向通行：只对非单行且非步行专用道建反向边
                ow_s = str(ow).lower() if ow is not None and not pd.isna(ow) else ""
                fwd = ow_s != "-1"
                bwd = (ow_s not in ("yes", "1", "true")) and not foot_ok

                if fwd:
                    _add_edge(G, ka, kb, seg_len, t, hw_key)
                    n_edges += 1
                if bwd:
                    _add_edge(G, kb, ka, seg_len, t, hw_key)
                    n_edges += 1

    logger.info(
        "路网构建完成: %d 个节点，%d 条有向边（跳过 %d 个非线要素）",
        G.number_of_nodes(), n_edges, n_skipped,
    )

    if G.number_of_nodes() == 0:
        raise RuntimeError("路网为空——请检查道路数据的几何类型与 CRS")

    # 连通性诊断：最大弱连通分量占比。比例过低说明路网因坐标精度而断裂。
    try:
        comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        frac = len(comps[0]) / G.number_of_nodes()
        logger.info(
            "路网连通性: 最大连通分量占 %.1f%%（%d 个分量）；"
            "低于 90%% 通常意味着存在断裂，需检查节点合并阈值",
            100 * frac, len(comps),
        )
        if frac < 0.9:
            logger.warning("路网连通性偏低，部分区域间可能不可达")
    except Exception:
        pass

    return G


def _add_edge(G, u, v, length, t, hw_key):
    """加入一条有向边；重复边取更短时间（等价于保留最快路径）。"""
    if G.has_edge(u, v):
        if t < G[u][v]["travel_time"]:
            G[u][v]["travel_time"] = t
            G[u][v]["length"] = length
            G[u][v]["highway"] = hw_key
    else:
        G.add_edge(u, v, length=length, travel_time=t, highway=hw_key)


def snap_points_to_graph(
    xy: np.ndarray, G, max_snap_m: float = 1200.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把点吸附到最近的路网节点。

    城市里起降场/需求点未必正好落在路上，需要吸附。吸附距离过大说明该点
    远离路网（如园区内部），此时接驳时间会被低估，因此记录吸附距离供
    后续使用。

    Returns
    -------
    ``(node_keys, snap_dist_m, ok)``；``ok`` 为 ``False`` 表示超出吸附阈值。

    Notes
    -----
    ``node_keys`` 返回的是**普通 Python 列表**而非 numpy 数组。图节点是
    ``(x, y)`` 元组，若用 ``np.array(list(G.nodes()))`` 会得到形状
    ``(N, 2)`` 的二维数组，索引后得到的是数组而非元组——数组不可哈希，
    用作字典键会抛 ``TypeError: unhashable type: 'numpy.ndarray'``。
    """
    from scipy.spatial import cKDTree

    keys = list(G.nodes())
    nodes = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)])
    if len(nodes) == 0:
        raise RuntimeError("路网无节点")

    tree = cKDTree(nodes)
    dist, idx = tree.query(np.asarray(xy, dtype=np.float64), k=1)
    ok = dist <= max_snap_m
    snapped = [keys[int(i)] for i in idx]
    return snapped, dist, ok


def network_travel_time_matrix(
    from_xy: np.ndarray,
    to_xy: np.ndarray,
    G,
    cfg,
    cutoff_m: float = 6000.0,
    snap_m: float = 1200.0,
) -> np.ndarray:
    """基于路网的最短路接驳时间矩阵（秒，多方式取最快）。

    Parameters
    ----------
    cutoff_m
        搜索截断半径（米）。超过该路网距离的 OD 对记为不可达。这个截断是
        性能关键：无截断时单源 Dijkstra 遍历全图，2519 个需求单元需要
        数十分钟；截断到 6 km 后只访问局部子图，总耗时降到分钟级。
    snap_m
        点到路网的最大吸附距离。

    Returns
    -------
    ``(len(from_xy), len(to_xy))`` 的接驳时间矩阵（秒），不可达为 ``inf``。

    口径与 :func:`evtol_siting.geo.travel_time_matrix` 一致：**多方式取最小，
    且每种方式的距离上限是硬约束**（超出即该方式不可用）。两者的唯一区别是
    距离的度量方式——这里用路网距离，那里用直线距离。因此两者可直接对比，
    这正是把它作为稳健性检验的前提。

    .. warning::
       本函数曾在实现中**丢失方式距离上限**：短驳车只在 5 km 内"取更快者"，
       超出 5 km 不作废，而是保留网络步行时间。后果是可达率由 9.8 % 虚高到
       91.0 %、时间上限由 15 min 变成 89 min——**换了模型而不是换了距离口径**，
       与欧氏口径不可比。上限现已施加在网络距离上。
    """
    import networkx as nx

    from_xy = np.asarray(from_xy, dtype=np.float64)
    to_xy = np.asarray(to_xy, dtype=np.float64)

    src_keys, src_d, src_ok = snap_points_to_graph(from_xy, G, snap_m)
    dst_keys, dst_d, dst_ok = snap_points_to_graph(to_xy, G, snap_m)

    logger.info(
        "点吸附路网: 需求点 %d/%d 成功（中位距离 %.0f m），候选点 %d/%d 成功",
        int(src_ok.sum()), len(src_ok), float(np.median(src_d)),
        int(dst_ok.sum()), len(dst_ok),
    )

    # 目标节点 -> 列索引
    dst_lookup: dict = {}
    for j, (k, ok) in enumerate(zip(dst_keys, dst_ok)):
        if ok:
            dst_lookup.setdefault(k, []).append(j)

    modes = cfg.get("access.modes", [])
    if not modes:
        raise ValueError("access.modes 为空，无法计算接驳时间")

    # 每种方式：(速度 m/s, 距离上限 m, 附加时间 s)。距离上限是**硬约束**，
    # 超出即该方式不可用——这与 geo.travel_time_matrix 的语义完全一致，
    # 唯一区别是这里度量的是**路网距离**而不是直线距离。
    mode_specs = [
        (float(m["speed_kmh"]) * 1000.0 / 3600.0,
         float(m.get("max_distance_m", np.inf)),
         float(m.get("fixed_wait_min", 0.0)) * 60.0)
        for m in modes
    ]

    out = np.full((len(from_xy), len(to_xy)), np.inf, dtype=np.float64)
    dst_d_arr = np.asarray(dst_d, dtype=np.float64)

    for i, (k, ok) in enumerate(zip(src_keys, src_ok)):
        if not ok:
            continue
        try:
            # 按 **length（米）** 求最短路——需要的是距离，不是时间：
            # 各方式的距离上限必须施加在网络距离上。
            dist = nx.single_source_dijkstra_path_length(
                G, k, cutoff=cutoff_m, weight="length"
            )
        except Exception:
            continue

        for node, L in dist.items():
            cols = dst_lookup.get(node)
            if not cols:
                continue
            # 起讫点到路网的接入段（米）计入总距离
            total_m = L + src_d[i] + dst_d_arr[cols]
            best = np.full(len(cols), np.inf)
            for v_ms, d_max, wait_s in mode_specs:
                t = total_m / v_ms + wait_s
                best = np.minimum(best, np.where(total_m <= d_max, t, np.inf))
            out[i, cols] = np.minimum(out[i, cols], best)

        if (i + 1) % 500 == 0:
            logger.info("  路网最短路进度 %d/%d", i + 1, len(from_xy))

    reach = float(np.isfinite(out).mean() * 100)
    logger.info(
        "路网接驳时间矩阵: %s，可达率 %.1f%%（图截断 %.0f m；"
        "方式距离上限 %s）",
        out.shape, reach, cutoff_m,
        "、".join(f"{d/1000:.1f} km" if np.isfinite(d) else "∞"
                 for _, d, _ in mode_specs),
    )
    return out.astype(np.float32)
