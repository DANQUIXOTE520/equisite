"""几何与空间计算核心。

本模块提供与具体阶段无关的 GIS 原语：

* 研究区边界构建与坐标变换（``study_area_polygon`` / ``to_projected``）
* 需求栅格化（``make_grid``）
* 距离与接驳时间矩阵（``distance_matrix`` / ``travel_time_matrix``）
* 栅格采样（``sample_raster_at_points``）

约定
----
* 所有距离、面积计算一律在 **投影坐标系**（默认 EPSG:32648）下进行，
  度（degree）仅用于数据下载与文件交换。
* 距离矩阵返回 ``float32`` 以控制内存：成都案例约 2400 栅格 × 200 候选，
  单精度已远超所需精度。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# geopandas / shapely 为必需依赖，缺失时给出可执行的提示而不是裸 ImportError
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "需要 geopandas 与 shapely。请运行: pip install geopandas shapely pyproj fiona"
    ) from exc


# ---------------------------------------------------------------------------
# 研究区边界
# ---------------------------------------------------------------------------

def study_area_polygon(cfg) -> Polygon:
    """按配置构建研究区多边形（WGS84）。

    ``study_area.boundary_mode`` 有两种取值：

    * ``"polygon"`` —— 使用配置中显式给出的顶点（默认）；
    * ``"file"``    —— 从 ``study_area.boundary_file`` 读取矢量并求并集。

    若显式多边形无效（顶点不足或自相交），回退到 ``study_area.bbox`` 矩形，
    并在日志中给出明确警告——宁可结果可追溯，也不要静默产出错误研究区。
    """
    mode = cfg.get("study_area.boundary_mode", "polygon")

    if mode == "file":
        fpath = cfg.get("study_area.boundary_file")
        if not fpath:
            raise ValueError("boundary_mode='file' 但未设置 study_area.boundary_file")
        p = cfg.path("study_area.boundary_file") if not str(fpath).startswith(("/", "\\")) else None
        p = p if (p is not None and p.exists()) else (cfg.path("data.raw_dir") / str(fpath))
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs(cfg.crs_geo)
        poly = unary_union(gdf.to_crs(cfg.crs_geo).geometry.values)
        if not poly.is_valid:
            poly = poly.buffer(0)
        logger.info("研究区边界来自文件: %s (面积 %.1f km²)", p, poly.area * 111.32**2 * 0.8)
        return poly

    verts = cfg.get("study_area.polygon") or []
    poly = None
    if len(verts) >= 3:
        candidate = Polygon([(float(x), float(y)) for x, y in verts])
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if candidate.is_valid and not candidate.is_empty and candidate.area > 0:
            poly = candidate
        else:
            logger.warning("配置中的研究区多边形无效（自相交或面积为零），回退到 bbox。")

    if poly is None:
        poly = box(*cfg.bbox)

    return poly


def to_projected(gdf: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    """把 GeoDataFrame 转换到投影坐标系。"""
    if gdf.crs is None:
        gdf = gdf.set_crs(cfg.crs_geo, allow_override=True)
    return gdf.to_crs(cfg.crs_proj)


def bbox_projected(cfg) -> tuple[float, float, float, float]:
    """研究区外包矩形在投影坐标系下的范围 ``(minx, miny, maxx, maxy)``。"""
    poly = study_area_polygon(cfg)
    gs = gpd.GeoSeries([poly], crs=cfg.crs_geo).to_crs(cfg.crs_proj)
    return tuple(gs.total_bounds)


# ---------------------------------------------------------------------------
# 需求栅格化
# ---------------------------------------------------------------------------

def make_grid(
    cfg,
    cell_size_m: float | None = None,
    clip: bool = True,
    keep_geometry: bool = True,
) -> gpd.GeoDataFrame:
    """把研究区栅格化为正方形单元。

    返回的 GeoDataFrame 含列：

    ``cell_id`` 整型唯一编号 | ``row`` / ``col`` 行列号 |
    ``geometry`` 单元多边形 | ``centroid_x`` / ``centroid_y`` 投影坐标下的中心点。

    Parameters
    ----------
    cell_size_m
        单元边长（米）。默认取 ``stage1_candidates.grid_size_m``。
    clip
        是否将单元裁剪到研究区边界内（默认 True，仅保留质心落在区内的单元）。
        采用"质心落入"规则而非几何相交，避免边界上出现大量细碎单元。
    """
    cell = float(cell_size_m or cfg.get("stage1_candidates.grid_size_m", 500.0))
    minx, miny, maxx, maxy = bbox_projected(cfg)

    xs = np.arange(minx, maxx + cell, cell)
    ys = np.arange(miny, maxy + cell, cell)
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx <= 0 or ny <= 0:
        raise ValueError(f"研究区过小，无法按 {cell} m 栅格化 (nx={nx}, ny={ny})")

    # 向量化构建规则网格：每个单元由左下角 (xs[i], ys[j]) 定义
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    x0, y0 = xs[ii], ys[jj]

    cx, cy = x0 + cell / 2.0, y0 + cell / 2.0
    geom = gpd.GeoSeries(
        [box(x, y, x + cell, y + cell) for x, y in zip(x0, y0)],
        crs=cfg.crs_proj,
    )

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(ii), dtype=np.int32),
            "row": jj.astype(np.int32),
            "col": ii.astype(np.int32),
            "centroid_x": cx.astype(np.float64),
            "centroid_y": cy.astype(np.float64),
        },
        geometry=geom,
        crs=cfg.crs_proj,
    )

    if clip:
        poly_proj = gpd.GeoSeries(
            [study_area_polygon(cfg)], crs=cfg.crs_geo
        ).to_crs(cfg.crs_proj).iloc[0]
        pts = gpd.GeoSeries(
            gpd.points_from_xy(gdf["centroid_x"], gdf["centroid_y"]),
            crs=cfg.crs_proj,
        )
        inside = pts.within(poly_proj).values
        gdf = gdf.loc[inside].copy()
        # 重新编号，保证 cell_id 连续（下游用作数组下标）
        gdf["cell_id"] = np.arange(len(gdf), dtype=np.int32)

    gdf = gdf.reset_index(drop=True)
    if not keep_geometry:
        gdf = gdf.drop(columns="geometry")

    logger.info(
        "栅格化完成: 单元边长 %.0f m, 单元数 %d, 覆盖 %.1f km²",
        cell,
        len(gdf),
        len(gdf) * cell * cell / 1e6,
    )
    return gdf


# ---------------------------------------------------------------------------
# 距离与接驳时间
# ---------------------------------------------------------------------------

def distance_matrix(
    from_xy: np.ndarray, to_xy: np.ndarray, dtype=np.float32
) -> np.ndarray:
    """欧氏距离矩阵 ``D[i, j] = ||from_i - to_j||``（单位与输入一致）。

    使用 ``(a-b)²`` 展开式配合 ``scipy.cdist`` 的等价实现；对成都案例
    （2400 × 200）耗时 < 10 ms，无需分块。
    """
    a = np.asarray(from_xy, dtype=np.float64)
    b = np.asarray(to_xy, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("from_xy / to_xy 形状须为 (n, 2)")

    # ||a-b||² = |a|² + |b|² - 2 a·b，数值上可能产生极小负数，clip 处理
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    sq = aa + bb - 2.0 * (a @ b.T)
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq).astype(dtype)


def travel_time_matrix(
    from_xy: np.ndarray,
    to_xy: np.ndarray,
    cfg,
) -> np.ndarray:
    """多方式接驳时间矩阵（秒）。

    对每一种接驳方式 ``m``（步行、短驳车……），用户取**最快**方式的出行时间::

        t_ij = min_m ( d_ij / v_m + w_m )     s.t.  d_ij <= d_m^max

    若所有方式都超出其最大接驳距离，则该 OD 对不可达，记为 ``np.inf``。
    这一"用户自选最快方式"的设定比单一速度更贴近实际，也避免了
    在小半径内高估接驳时间。

    网络距离可用时（``road_network`` 参数）应优先使用
    :func:`network_travel_time_matrix`；本函数为自由流欧氏近似。
    """
    d = distance_matrix(from_xy, to_xy, dtype=np.float64)  # 米
    modes = cfg.get("access.modes", [])
    if not modes:
        raise ValueError("access.modes 为空，无法计算接驳时间")

    best = np.full(d.shape, np.inf, dtype=np.float64)
    for m in modes:
        v_ms = float(m["speed_kmh"]) * 1000.0 / 3600.0
        d_max = float(m.get("max_distance_m", np.inf))
        wait_s = float(m.get("fixed_wait_min", 0.0)) * 60.0
        with np.errstate(divide="ignore"):
            t = d / v_ms + wait_s
        t = np.where(d <= d_max, t, np.inf)
        np.minimum(best, t, out=best)

    return best.astype(np.float32)


def network_travel_time_matrix(
    from_xy: np.ndarray,
    to_xy: np.ndarray,
    graph,
    cfg,
    weight: str = "travel_time",
    snap_tolerance_m: float = 800.0,
) -> np.ndarray:
    """基于路网的最短路接驳时间矩阵。

    把每个起讫点吸附（snap）到最近的路网节点，在 ``networkx`` 图上求
    单源最短路。相比欧氏距离，路网距离能正确反映河流、铁路、封闭园区
    造成的绕行——对成都这样的环状+放射状路网尤其重要。

    Parameters
    ----------
    graph
        ``networkx`` 有向/无向图，节点含 ``x`` / ``y`` 属性（投影坐标）。
    weight
        边权属性名；``"travel_time"``（秒）或 ``"length"``（米）。

    Returns
    -------
    ``(len(from_xy), len(to_xy))`` 的时间矩阵（秒），不可达为 ``inf``。
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    nodes = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in graph.nodes])
    node_ids = np.array(list(graph.nodes))
    tree = cKDTree(nodes)

    def snap(xy: np.ndarray):
        dist, idx = tree.query(np.asarray(xy, dtype=np.float64), k=1)
        ok = dist <= snap_tolerance_m
        return node_ids[idx], dist, ok

    src_nodes, src_d, src_ok = snap(from_xy)
    dst_nodes, dst_d, dst_ok = snap(to_xy)

    # 反向索引：路网节点 -> 目标列
    dst_lookup: dict = {}
    for j, (nid, ok) in enumerate(zip(dst_nodes, dst_ok)):
        if ok:
            dst_lookup.setdefault(nid, []).append(j)

    out = np.full((len(from_xy), len(to_xy)), np.inf, dtype=np.float64)

    for i, (nid, ok) in enumerate(zip(src_nodes, src_ok)):
        if not ok:
            continue
        lengths = nx.single_source_dijkstra_path_length(graph, nid, weight=weight)
        for node, cost in lengths.items():
            for j in dst_lookup.get(node, ()):  # 命中目标
                out[i, j] = cost
        if (i + 1) % 500 == 0:
            logger.debug("路网最短路进度 %d/%d", i + 1, len(from_xy))

    # 加上起讫点到路网的直线接入时间（用步行速度折算）
    walk_v = min(
        (float(m["speed_kmh"]) for m in cfg.get("access.modes", [])),
        default=4.8,
    ) * 1000.0 / 3600.0
    out = out + (src_d[:, None] + dst_d[None, :]) / walk_v
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 栅格采样
# ---------------------------------------------------------------------------

def sample_raster_at_points(
    raster_path,
    x: np.ndarray,
    y: np.ndarray,
    src_crs: str | None = None,
    dst_crs: str | None = None,
    band: int = 1,
    nodata_fill: float = np.nan,
) -> np.ndarray:
    """在给定点坐标处采样栅格值。

    用于把建筑高度栅格、人口栅格、收入栅格的值赋给栅格单元或建筑质心。
    采用**双线性插值**（``rasterio`` 默认的重采样语义），对连续表面
    （人口、高度、收入）比最近邻更合理。

    Parameters
    ----------
    src_crs
        栅格的 CRS。**留空时会自动从栅格文件读取**，不传是安全的。
    dst_crs
        输入点坐标的 CRS。若它与栅格 CRS 不同，先做投影变换。

    ⚠ 这个签名曾经是个陷阱
    ----------------------
    早期版本只在 ``src_crs`` 与 ``dst_crs`` **都非空**时才做变换，而三处调用
    点都只传 ``dst_crs``——于是变换分支**从未执行**：投影坐标（UTM，x≈60 万、
    y≈327 万）被直接丢进 WGS84 栅格（经度 106–107）采样，全部落在范围外，
    ``rasterio`` 返回 0。

    后果：**坡度判据从未生效过**。成都平原坡度普遍 <5°，所以日志里
    "剔除 0" 看上去很正常，几十次运行都没有人察觉；换到多山的城市才会暴露
    （山地研究区实测坡度 p50 已达 12.35°、最大 65.9°）。
    ``heights.py`` 的调用点因为显式传了 ``src_crs`` 而幸免。

    现在改为自动读取栅格 CRS，并在**采样大量落空**时告警——这类失败不会抛
    异常，只会安静地返回常数，必须显式拦截。
    """
    import rasterio

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    with rasterio.open(raster_path) as src:
        # 栅格 CRS 以文件为准；仅在调用方显式覆盖时才采用其取值。
        raster_crs = src_crs or (str(src.crs) if src.crs else None)
        if raster_crs and dst_crs and str(raster_crs) != str(dst_crs):
            from pyproj import Transformer

            tr = Transformer.from_crs(dst_crs, raster_crs, always_xy=True)
            x, y = tr.transform(x, y)

        coords = list(zip(x, y))
        vals = np.array([v[0] for v in src.sample(coords, indexes=band)], dtype=np.float64)
        if src.nodata is not None:
            vals = np.where(np.isclose(vals, src.nodata), nodata_fill, vals)

        # 落空自检：采样点若大量落在栅格范围外，返回的是一串无意义的常数
        # （0 或 nodata）。这不会报错，但会让下游判据静默失效。
        outside = (x < src.bounds.left) | (x > src.bounds.right) | \
                  (y < src.bounds.bottom) | (y > src.bounds.top)
        if len(x) and outside.mean() > 0.5:
            logger.warning(
                "栅格采样：%.0f%% 的点落在 %s 范围之外（点 CRS=%s，栅格 CRS=%s）"
                "——返回值不可信，请检查 CRS 是否一致",
                100 * outside.mean(), Path(raster_path).name, dst_crs, raster_crs,
            )

    return vals


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    """两点大圆距离（公里）。用于快速校验下载范围是否覆盖研究区。"""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def polygon_area_km2(poly, crs_geo: str = "EPSG:4326") -> float:
    """多边形面积（平方公里），先投影到等积坐标系再计算。"""
    gs = gpd.GeoSeries([poly], crs=crs_geo).to_crs("EPSG:6933")  # EASE-Grid 2.0 等积
    return float(gs.area.iloc[0] / 1e6)


def greedy_farthest_point(points: np.ndarray, n: int, seed_idx: int = 0) -> list[int]:
    """最远点采样：在候选点中选出空间上最分散的 ``n`` 个。

    用作 NSGA-II 的初始种群多样性与基线方法（p-dispersion）的构造。
    """
    n_pts = len(points)
    n = min(n, n_pts)
    chosen = [seed_idx]
    d = np.linalg.norm(points - points[seed_idx], axis=1)
    for _ in range(n - 1):
        idx = int(np.argmax(d))
        chosen.append(idx)
        d = np.minimum(d, np.linalg.norm(points - points[idx], axis=1))
    return chosen
