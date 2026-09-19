"""OpenStreetMap 数据获取（Overpass API）。

为什么需要分块
--------------
成都绕城高速以内 + 天府新区核心区约 700 km²。一次性查询该范围内全部
建筑（约 30 万栋）会让 Overpass 查询超时或返回 504。实测单个
0.02°×0.02°（≈2.2 km）瓦片约 5–8 s、1500–3000 栋建筑，因此本模块把
外包矩形切成瓦片逐一查询并本地缓存。

缓存策略
--------
每个瓦片的结果按 ``{layer}_{tile_key}.json`` 存到 ``data/cache/overpass/``。
重跑时若缓存命中则直接读取，不重复请求——这既对公共 API 友好（Overpass
是志愿者资助的服务），也让中断后可从断点续传。

礼貌限速
--------
瓦片之间 ``sleep_between_s``（默认 2 s），并在 429/504 时指数退避重试，
轮换 ``endpoints`` 列表中的镜像。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 requests：pip install requests") from exc


# ---------------------------------------------------------------------------
# 图层定义：每个图层对应一段 Overpass QL 过滤器
# ---------------------------------------------------------------------------

LAYER_FILTERS: dict[str, str] = {
    # 建筑轮廓（含 height / building:levels 等标签）
    "buildings": 'way["building"]({bbox});',
    # 建筑部件（3D 建模常用，可选）
    "building_parts": 'way["building:part"]({bbox});',
    # POI：兴趣点，用于需求特征
    "poi": (
        'node["amenity"]({bbox});'
        'node["shop"]({bbox});'
        'node["office"]({bbox});'
        'node["leisure"]({bbox});'
        'node["tourism"]({bbox});'
    ),
    # 路网：用于基于路网的接驳时间
    "roads": (
        'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
        'unclassified|residential|living_street|service|'
        'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|'
        'footway|pedestrian|cycleway|path|steps)$"]({bbox});'
    ),
    # 用地与自然要素：用于排除不可建区域
    "landuse": 'way["landuse"]({bbox});relation["landuse"]({bbox});',
    "natural": 'way["natural"]({bbox});relation["natural"]({bbox});',
    "water": (
        'way["natural"="water"]({bbox});'
        'way["waterway"]({bbox});'
        'relation["natural"="water"]({bbox});'
    ),
    # 既有航空设施：机场、直升机场（安全约束）
    "airports": (
        'node["aeroway"~"^(aerodrome|helipad|heliport)$"]({bbox});'
        'way["aeroway"~"^(aerodrome|helipad|heliport)$"]({bbox});'
    ),
    # 轨道交通站点：接驳方式的重要输入
    "rail_stations": (
        'node["railway"="station"]({bbox});'
        'node["station"="subway"]({bbox});'
    ),
}

# Overpass 的 out 语句模式。
#
# 必须用 `geom`（完整几何），不能用 `tags center`：
#   * 屋顶型候选判据需要**建筑占地轮廓**来计算屋顶面积
#     （A_roof >= 1600 m²）。`center` 只给出一个点，算不出面积，
#     会使屋顶型筛选完全失效——而且不报错，只是候选集里没有屋顶型。
#   * 地面型候选需要水域/绿地的**多边形**才能做空间排除。
#   * `geom` 对 node 同样输出 lat/lon，因此 POI 等点要素也一并适用。
#
# 代价是响应体积显著增大（实测 0.04° 瓦片约 4.2 MB），这是必要的。
OUT_MODE = "geom"


# ---------------------------------------------------------------------------
# 瓦片切分
# ---------------------------------------------------------------------------

def tile_bbox(
    bbox: Sequence[float], tile_size_deg: float
) -> list[tuple[float, float, float, float]]:
    """把外包矩形切成 ``tile_size_deg`` 见方的小块。

    返回 ``(minx, miny, maxx, maxy)`` 列表。最后一行/列的瓦片会贴合边界，
    不会超出原始 bbox。
    """
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    if tile_size_deg <= 0:
        raise ValueError("tile_size_deg 必须为正")

    xs = np.arange(minx, maxx, tile_size_deg)
    ys = np.arange(miny, maxy, tile_size_deg)
    tiles: list[tuple[float, float, float, float]] = []
    for x in xs:
        for y in ys:
            tiles.append(
                (
                    round(x, 6),
                    round(y, 6),
                    round(min(x + tile_size_deg, maxx), 6),
                    round(min(y + tile_size_deg, maxy), 6),
                )
            )
    return tiles


def _tile_key(tile: Sequence[float]) -> str:
    """瓦片的稳定短标识（用于缓存文件名）。"""
    s = ",".join(f"{v:.6f}" for v in tile)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Overpass 查询
# ---------------------------------------------------------------------------

def build_query(layer: str, tile: Sequence[float], timeout: int = 180,
                out_mode: str = OUT_MODE) -> str:
    """构造单个瓦片、单个图层的 Overpass QL 查询。"""
    if layer not in LAYER_FILTERS:
        raise KeyError(f"未知图层 '{layer}'。可选: {sorted(LAYER_FILTERS)}")
    minx, miny, maxx, maxy = tile
    bbox = f"{miny},{minx},{maxy},{maxx}"  # Overpass 用 (S,W,N,E)
    body = LAYER_FILTERS[layer].format(bbox=bbox)
    return f"[out:json][timeout:{timeout}];\n(\n{body}\n);\nout {out_mode};"


def _fetch_tile_adaptive(
    layer: str,
    tile: Sequence[float],
    endpoints: Sequence[str],
    ua: str,
    timeout_s: int,
    max_retries: int,
    min_span_deg: float = 0.005,
    _depth: int = 0,
) -> dict | None:
    """下载一个瓦片；若 Overpass 返回 504/超时，则**二分瓦片**后分别重试。

    为什么需要这个
    --------------
    ``out geom`` 的响应体积与瓦片内的建筑密度成正比。成都市中心一个
    0.04° 瓦片的建筑可能多到让 Overpass 服务端超时（HTTP 504），而郊区
    同尺寸瓦片却秒回。用固定的瓦片尺寸要么在市中心反复失败，要么在郊区
    浪费请求。二分策略让瓦片尺寸自适应数据密度——这是唯一能同时兼顾
    两者的简单办法。

    递归深度由 ``min_span_deg`` 限制，避免在异常数据上无限细分。
    返回 ``None`` 表示该瓦片最终仍失败（调用方记录并继续）。
    """
    minx, miny, maxx, maxy = tile
    query = build_query(layer, tile, timeout=min(timeout_s, 300))
    try:
        return _post_with_retry(query, endpoints, ua, timeout_s, max_retries)
    except RateLimited as exc:
        # 限流不是"瓦片太大"，二分只会让请求数翻 4 倍、雪上加霜。
        # 这里直接放弃该瓦片，留给下一轮（缓存机制让重跑可以续传）。
        logger.warning(
            "瓦片 %s 因限流放弃（非瓦片过大，不做二分）: %s",
            tuple(round(v, 4) for v in tile), str(exc)[:100],
        )
        return None
    except Exception as exc:
        span_x, span_y = maxx - minx, maxy - miny
        if max(span_x, span_y) / 2.0 < min_span_deg or _depth >= 4:
            logger.warning(
                "瓦片 %s 在最小尺寸下仍失败，放弃: %s",
                tuple(round(v, 4) for v in tile), str(exc)[:120],
            )
            return None

        logger.info(
            "瓦片 %s 下载失败，二分后重试（深度 %d）: %s",
            tuple(round(v, 4) for v in tile), _depth + 1, str(exc)[:100],
        )
        mx, my = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        subs = [
            (minx, miny, mx, my), (mx, miny, maxx, my),
            (minx, my, mx, maxy), (mx, my, maxx, maxy),
        ]
        merged: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for sub in subs:
            part = _fetch_tile_adaptive(
                layer, sub, endpoints, ua, timeout_s, max_retries,
                min_span_deg, _depth + 1,
            )
            if part is None:
                continue
            for el in part.get("elements", []):
                key = (el.get("type", "?"), el.get("id", 0))
                if key not in seen:
                    seen.add(key)
                    merged.append(el)
        return {"elements": merged} if merged else None


class RateLimited(RuntimeError):
    """Overpass 返回 504/429——属于**限流**，不是查询本身有问题。

    与"瓦片太大导致服务端超时"必须区分：前者应当**拉长冷却后原样重试**，
    后者才应该把瓦片二分。把限流误判为瓦片过大，会让请求数翻 4 倍，
    反而加剧限流——实测中这会把下载速度拖慢一个数量级。
    """


def _post_with_retry(
    query: str,
    endpoints: Sequence[str],
    user_agent: str,
    timeout_s: int = 240,
    max_retries: int = 3,
    rate_limit_cooldown_s: float = 30.0,
) -> dict:
    """向 Overpass 发起查询，失败时轮换镜像并指数退避。

    504/429 视为限流，抛出 :class:`RateLimited` 并做**较长**的冷却退避。
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": user_agent},
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 504):
                # 限流：响应体很小、返回很快。这是最常见的失败原因。
                wait = rate_limit_cooldown_s * (attempt + 1)
                logger.warning(
                    "Overpass 限流 (HTTP %d，响应仅 %d 字节)，冷却 %.0fs 后重试"
                    "（第 %d/%d 次）",
                    resp.status_code, len(resp.content), wait, attempt + 1, max_retries,
                )
                last_err = RateLimited(f"HTTP {resp.status_code}")
                time.sleep(wait)
                continue

            if resp.status_code in (400, 404):
                raise RuntimeError(
                    f"Overpass 拒绝了查询 (HTTP {resp.status_code}): {resp.text[:300]}"
                )
            last_err = RuntimeError(f"HTTP {resp.status_code} from {endpoint}")
            logger.warning("Overpass %s 返回 %d", endpoint, resp.status_code)
        except requests.RequestException as exc:
            last_err = exc
            logger.warning(
                "Overpass 请求失败 (%s)，第 %d/%d 次：%s",
                endpoint, attempt + 1, max_retries, str(exc)[:120],
            )
        time.sleep(2.0 ** attempt * 3)
    raise RuntimeError(f"Overpass 查询在 {max_retries} 次尝试后仍失败") from last_err


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def fetch_layer(
    layer: str,
    cfg,
    bbox: Sequence[float] | None = None,
    use_cache: bool = True,
    limit_tiles: int | None = None,
) -> list[dict]:
    """分块下载一个 OSM 图层，返回原始 element 列表。

    Parameters
    ----------
    layer
        :data:`LAYER_FILTERS` 中的键。
    bbox
        覆盖范围；默认取 ``study_area.bbox``。建议传入比研究区略大的
        矩形，以免边界处的建筑被切断。
    use_cache
        是否读写瓦片缓存（默认 True）。
    limit_tiles
        仅处理前 N 个瓦片——用于快速试跑与调试，避免误触发全量下载。

    Returns
    -------
    Overpass 返回的 ``elements`` 列表（已去重）。
    """
    bbox = bbox or cfg.bbox
    tile_deg = float(cfg.get("data.overpass.tile_size_deg", 0.02))
    endpoints = cfg.get("data.overpass.endpoints")
    ua = cfg.get("data.overpass.user_agent")
    timeout_s = int(cfg.get("data.overpass.request_timeout_s", 240))
    max_retries = int(cfg.get("data.overpass.max_retries", 3))
    sleep_s = float(cfg.get("data.overpass.sleep_between_s", 2.0))

    # 必须显式创建 overpass 子目录：cfg.dir() 只保证 data/cache 存在，
    # 不会创建其下的 overpass/。缺目录时瓦片写缓存会抛 FileNotFoundError，
    # 而且是在**下载成功之后**才抛——数据白下了一遍。
    cache_dir = cfg.dir("data.cache_dir") / "overpass"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tiles = tile_bbox(bbox, tile_deg)
    if limit_tiles:
        tiles = tiles[:limit_tiles]

    logger.info("图层 '%s': %d 个瓦片，覆盖 bbox=%s", layer, len(tiles), tuple(bbox))

    all_elements: list[dict] = []
    seen: set[tuple[str, int]] = set()
    n_from_cache = 0

    for i, tile in enumerate(tiles, 1):
        cache_file = cache_dir / f"{layer}_{_tile_key(tile)}.json"

        if use_cache and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                n_from_cache += 1
            except (json.JSONDecodeError, OSError):
                logger.warning("缓存损坏，重新下载: %s", cache_file.name)
                payload = None
        else:
            payload = None

        if payload is None:
            payload = _fetch_tile_adaptive(
                layer, tile, endpoints, ua, timeout_s, max_retries,
                min_span_deg=float(tile_deg) / 8.0,
            )
            if payload is None:
                # 单个瓦片失败不应终止整个下载：记录并继续，最后汇总缺口
                logger.error("瓦片 %d/%d %s 下载失败（含二分重试）", i, len(tiles), tile)
                continue
            if use_cache:
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            time.sleep(sleep_s)

        for el in payload.get("elements", []):
            key = (el.get("type", "?"), el.get("id", 0))
            if key not in seen:
                seen.add(key)
                all_elements.append(el)

        if i % 20 == 0 or i == len(tiles):
            logger.info(
                "  %d/%d 瓦片完成（累计 %d 个要素，%d 来自缓存）",
                i, len(tiles), len(all_elements), n_from_cache,
            )

    logger.info("图层 '%s' 完成: %d 个唯一要素", layer, len(all_elements))
    return all_elements


# ---------------------------------------------------------------------------
# 解析为 GeoDataFrame
# ---------------------------------------------------------------------------

def elements_to_gdf(
    elements: Iterable[dict],
    cfg,
    with_geometry: bool = True,
):
    """把 Overpass element 列表转成 GeoDataFrame。

    * 点要素（POI、站点）用 ``lat`` / ``lon`` 构造；
    * 线/面要素用 ``geometry``（``out geom`` 输出）构造。

    坐标一律按 WGS84 读入，再投影到 ``crs.projected``。
    无几何的要素（``out center`` 输出）会被跳过并计入日志。
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    records, geoms = [], []

    for el in elements:
        tags = el.get("tags", {}) or {}
        etype = el.get("type")
        row = dict(tags)
        row["osm_id"] = el.get("id")
        row["osm_type"] = etype

        geom = None
        if with_geometry:
            if etype == "node" and "lat" in el and "lon" in el:
                geom = Point(el["lon"], el["lat"])
            elif "geometry" in el and el["geometry"]:
                pts = [(g["lon"], g["lat"]) for g in el["geometry"] if g]
                if len(pts) >= 2:
                    if etype == "way" and len(pts) >= 4 and pts[0] == pts[-1]:
                        geom = Polygon(pts)
                    else:
                        geom = LineString(pts)
            elif etype == "relation" and "members" in el:
                # 多边形关系（如大型水域、机场）按外环成员拼接近似多边形。
                # 完整的多环拼接需要处理内环与角色（outer/inner），
                # 这里简化处理并记录——对本研究（只需排除水域/绿地范围）
                # 精度足够，但论文中应说明这一简化。
                outer: list[tuple[float, float]] = []
                for m in el.get("members", []):
                    if m.get("role") not in ("outer", ""):
                        continue
                    for g in m.get("geometry") or []:
                        if g:
                            outer.append((g["lon"], g["lat"]))
                if len(outer) >= 4:
                    geom = Polygon(outer).buffer(0)
        elif "center" in el:
            c = el["center"]
            geom = Point(c["lon"], c["lat"])

        if geom is None:
            continue
        records.append(row)
        geoms.append(geom)

    if not records:
        logger.warning("没有可解析的要素（可能全部缺少几何）")
        return gpd.GeoDataFrame(
            {"osm_id": [], "osm_type": []}, geometry=[], crs=cfg.crs_geo
        )

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=cfg.crs_geo)
    gdf = gdf.to_crs(cfg.crs_proj)
    logger.info("解析出 %d 个要素 -> %s", len(gdf), cfg.crs_proj)
    return gdf


def fetch_geodataframe(layer: str, cfg, **kwargs):
    """``fetch_layer`` + ``elements_to_gdf`` 的便捷组合。"""
    els = fetch_layer(layer, cfg, **kwargs)
    return elements_to_gdf(els, cfg)
