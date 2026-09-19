"""OpenStreetMap 数据获取（PBF 批量提取，推荐通道）。

为什么有了 Overpass 还要做 PBF
------------------------------
``data/osm.py`` 用 Overpass API 分块下载。实测在受限网络环境下，Overpass
对同一 IP **强限流**——首个请求返回 200，紧接着的全部返回 504（响应体仅
695 字节，是限流特征而非查询问题）。成都研究区 300 个瓦片按此速率需要
**8 小时以上**，实际不可用。

PBF 通道改为一次下载省级提取包：

============  =====================================  ==========  ==========
通道           来源                                   体积        实测耗时
============  =====================================  ==========  ==========
Overpass      overpass-api.de（分块 ×300）             ~500 MB     8 小时+
PBF（推荐）    OSM.fr 四川省提取                       116 MB      **18 秒**
============  =====================================  ==========  ==========

PBF 通道还有一个方法论优势：它给出的是**某一时刻的完整快照**而非逐块
抓取的拼接，因此不存在瓦片边界处要素被截断的问题，结果的时点一致性也
更强（论文中可声明"数据快照日期 = OSM.fr 提取包日期"）。

数据源
------
OSM.fr 提供按国家/省份切分的提取包（``download.openstreetmap.fr/extracts/``）。
四川省包覆盖整个四川，含成都研究区全部要素。若该源失效，可退回 Geofabrik
全国包（``asia/china-latest.osm.pbf``，1.59 GB），本模块同样支持。

解析
----
用 ``pyosmium``（C++ 实现的 Python 绑定）流式解析，内存占用与文件大小
无关；边界框过滤下推到 C++ 层（``BoundingBoxFilter``），只把研究区内的
要素交给 Python，因此 116 MB 的省级包在几十秒内解析完毕。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 各图层对应的 OSM 标签键（与 osm.py 的 LAYER_FILTERS 语义保持一致）。
# 这里只用于 C++ 层的 KeyFilter 预筛——**键命中不等于属于该图层**，
# 真正的归属判定见 LAYER_VALUES / _matches_layer。
LAYER_KEYS: dict[str, tuple[str, ...]] = {
    "buildings": ("building",),
    "building_parts": ("building:part",),
    "poi": ("amenity", "shop", "office", "leisure", "tourism"),
    "roads": ("highway",),
    "landuse": ("landuse",),
    "natural": ("natural",),
    "water": ("natural", "waterway", "landuse"),
    "airports": ("aeroway",),
    "rail_stations": ("railway", "station"),
}

# 取值白名单：**键命中之后，值还必须属于该图层**。
#
# ⚠ 这是一个真实发生过的严重缺陷。早期版本只做键匹配
# （``any(k in tags for k in keys)``），而 ``water`` 图层的键是
# ``("natural", "waterway")``——于是**任何**带 ``natural`` 标签的要素都被
# 当成水域。在成都研究区内实测，被判为"水域"的 4 531 个 way 里：
#
#   ====================  ======  ==========================
#   标签                   数量    实际是什么
#   ====================  ======  ==========================
#   waterway=*            2 489   河流（线状，见下）
#   natural=water         1 031   **真水域**
#   natural=wood            479   **林地**
#   natural=tree_row        253   **行道树**
#   natural=scrub           138   **灌丛**
#   natural=sand            102   沙地
#   natural=wetland          29   湿地
#   natural=grassland         5   草地
#   natural=cliff/rock/mud    5   崖壁/岩石/泥滩
#   ====================  ======  ==========================
#
# 即 **21.7 % 的"水域"其实是林地、灌丛、行道树**。这些要素被送进
# ``_spatial_exclude(..., label="水域")`` 做质心排除，把本可以是候选的
# 地面单元整片剔掉，且日志里显示的原因还是"水域"——一个不会报错、
# 只会悄悄改变候选集的错误。
#
# ``None`` 表示该键的任意值都算（如 ``building=*``）。
LAYER_VALUES: dict[str, dict[str, set[str] | None]] = {
    "buildings": {"building": None},
    "building_parts": {"building:part": None},
    "poi": {"amenity": None, "shop": None, "office": None,
            "leisure": None, "tourism": None},
    "roads": {"highway": None},
    "landuse": {"landuse": None},
    "natural": {"natural": None},
    # 水域 = 水面 + 湿地 + 河流/运河/沟渠 + 水库类用地。
    # 注意 wetland 计入：湿地同样不可建设，排除它符合"不可建区域"的语义。
    "water": {
        "natural": {"water", "wetland", "bay", "strait", "spring"},
        "waterway": None,
        "landuse": {"reservoir", "basin", "salt_pond"},
    },
    "airports": {"aeroway": None},
    "rail_stations": {"railway": None, "station": None},
}

# 每个图层的几何类型偏好：多边形 / 线 / 点 / 任意
LAYER_GEOM: dict[str, str] = {
    "buildings": "polygon",
    "building_parts": "polygon",
    "poi": "point",
    "roads": "line",
    "landuse": "polygon",
    "natural": "polygon",
    # 水域**两种几何都要**：面（湖、库、湿地的 multipolygon relation）
    # 与线（河流、运河、沟渠）。早期版本只留多边形，把 2 489 条
    # ``waterway=*`` 线整批丢掉；而恰恰是大型水域多为 relation，
    # 只解析 way 又会把整片水库漏掉（见 parse_all_layers 的说明）。
    "water": "any",
    "airports": "any",
    "rail_stations": "point",
}

# OSM.fr 省级提取包的 URL 模板
OSMFR_EXTRACT_URL = (
    "https://download.openstreetmap.fr/extracts/asia/china/{province}-latest.osm.pbf"
)
# 备选：Geofabrik 全国包
GEOFABRIK_CHINA_URL = "https://download.geofabrik.de/asia/china-latest.osm.pbf"


def default_pbf_path(cfg) -> Path:
    """本项目中 PBF 提取包的默认落盘位置。

    ⚠ 省名必须取自 ``data.pbf_province``。早期这里写死 "sichuan"，在多城市
    场景下会静默出错：换了城市的配置，本函数仍返回 ``sichuan-latest.osm.pbf``，
    而该文件通常**已存在**，于是 ``run_stage1`` 里的 ``if not pbf_path.exists()``
    直接跳过下载，**拿着另一个省的数据把本城验证跑完且不报任何错**。
    """
    province = str(cfg.get("data.pbf_province", "sichuan"))
    return cfg.dir("data.raw_dir") / "osm_pbf" / f"{province}-latest.osm.pbf"


def fetch_pbf(cfg, province: str = "sichuan", source: str = "osmfr") -> Path | None:
    """下载省级 OSM PBF 提取包（已存在则跳过）。

    Parameters
    ----------
    province
        OSM.fr 的省份标识（小写英文），如 ``"sichuan"``。
    source
        ``"osmfr"``（省级，推荐）或 ``"geofabrik"``（全国，1.59 GB，备选）。

    Returns
    -------
    PBF 文件路径；下载失败返回 ``None``。
    """
    from .rasters import download_file

    raw_dir = cfg.dir("data.raw_dir") / "osm_pbf"
    ua = cfg.get("data.overpass.user_agent")

    if source == "geofabrik":
        url = GEOFABRIK_CHINA_URL
        dest = raw_dir / "china-latest.osm.pbf"
    else:
        url = OSMFR_EXTRACT_URL.format(province=province)
        dest = raw_dir / f"{province}-latest.osm.pbf"

    try:
        return download_file(url, dest, ua, timeout_s=3600)
    except Exception as exc:
        logger.error(
            "PBF 下载失败 (%s): %s\n"
            "可手动下载后放入 %s：\n  %s",
            source, str(exc)[:200], raw_dir, url,
        )
        return None


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _geometry_from_way(way) -> object | None:
    """把 pyosmium 的 way 转成 shapely 几何。

    闭合的 way（首尾节点相同且 >= 4 个节点）视为多边形，否则为线。
    节点坐标由 ``with_locations()`` 提供。
    """
    from shapely.geometry import LineString, Polygon

    try:
        coords = [(n.lon, n.lat) for n in way.nodes]
    except Exception:
        # 缺少位置信息（节点在提取包边界外）——跳过该要素
        return None

    if len(coords) < 2:
        return None
    if len(coords) >= 4 and coords[0] == coords[-1]:
        try:
            return Polygon(coords)
        except Exception:
            return None
    return LineString(coords)


def _tags_to_dict(tags: Iterable) -> dict:
    """pyosmium 的 TagList 转普通 dict。"""
    out: dict = {}
    for t in tags:
        out[t.k] = t.v
    return out


def _matches_layer(tags: dict, layer: str) -> bool:
    """判断要素标签是否属于目标图层 —— **键与值都要匹配**。

    ⚠ 早期版本只看键（``any(k in tags for k in keys)``），使
    ``natural=wood`` 被当成水域。详见 :data:`LAYER_VALUES` 的说明。
    """
    spec = LAYER_VALUES.get(layer)
    if spec is None:
        raise KeyError(f"未知图层 '{layer}'。可选: {sorted(LAYER_VALUES)}")
    for k, allowed in spec.items():
        if k in tags and (allowed is None or tags[k] in allowed):
            return True
    return False


def _area_source(area) -> str:
    """``Area`` 是来自闭合 way 还是 multipolygon relation。

    ⚠ ``Area.is_multipolygon`` 与 ``from_relation`` 在 pyosmium 里都是
    **方法而不是属性**。写成 ``if obj.is_multipolygon:`` 拿到的是一个
    *绑定方法对象*，它**恒为真**——于是每一个面要素都会被标成 relation
    （实测 69 991/70 006 栋建筑被标成 relation）。必须调用。
    """
    for name, tag in (("from_relation", "relation"), ("is_multipolygon", "relation"),
                      ("from_way", "way")):
        fn = getattr(area, name, None)
        if fn is None:
            continue
        try:
            if fn():
                return tag
        except Exception:
            continue
    return "area"


def _xy(node) -> tuple[float, float]:
    """取一个环成员的经纬度（度）。

    ⚠ ``Area.outer_rings()`` 的元素是 ``osmium.osm.NodeRef``，**不是坐标元组**。
    它有两个容易搞混的访问器：

    * ``.lon`` / ``.lat`` —— 十进制度（**要的是这个**）
    * ``.x`` / ``.y``   —— 定点整数（度 × 1e7），如 ``1020660476``

    早期实现写的是 ``for x, y in ring``，对 ``NodeRef`` 直接解包会抛
    ``TypeError``；异常被 ``except`` 吞掉后函数返回 ``None``，于是**每一个
    面要素都被当成"不在研究区内"丢弃**——实测 343 714 个面全部被丢，
    建筑要素从 69 675 掉到 15，而日志里只会看到一个偏小的数字。
    这个坑不报错、不崩溃，只让结果静默变空。
    """
    try:
        return float(node.lon), float(node.lat)
    except AttributeError:                      # 兼容 (lon, lat) 元组
        return float(node[0]), float(node[1])


def _area_to_geometry(area):
    """把 pyosmium 的 ``Area`` 转成 shapely 多边形。

    用**全部外环**构造 MultiPolygon；内环（孔洞）忽略——本项目中这些
    图层只用于"质心是否落在其中"的空间排除，孔洞相对于 500 m 网格
    可忽略，而保留孔洞会让几何更容易非法。

    Area 的几何可能自交（OSM 数据质量所致）。非法几何先尝试
    ``make_valid`` 修复，仍失败则丢弃该要素并计数。
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    polys = []
    try:
        rings = list(area.outer_rings())
    except Exception:
        return None
    for ring in rings:
        coords = []
        for nd in ring:
            try:
                coords.append(_xy(nd))
            except Exception:
                continue
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            p = Polygon(coords)
        except Exception:
            continue
        if not p.is_valid:
            try:
                p = make_valid(p)
            except Exception:
                continue
        if p.is_empty:
            continue
        polys.append(p)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    try:
        return MultiPolygon(polys)
    except Exception:
        return polys[0]


def _ring_centroid(area) -> tuple[float, float] | None:
    """用外环坐标均值近似 Area 的中心，仅用于 bbox 判断。

    与 :func:`_way_centroid` 同样的取舍：边界处失真的只有贴着研究区
    外缘的一小圈要素，而研究区外还有缓冲。
    """
    try:
        for ring in area.outer_rings():
            n = 0
            s_lon = s_lat = 0.0
            for nd in ring:
                try:
                    lon, lat = _xy(nd)
                except Exception:
                    continue
                s_lon += lon
                s_lat += lat
                n += 1
            if n:
                return s_lon / n, s_lat / n
    except Exception:
        return None
    return None


def _in_bbox(lon: float, lat: float, bbox: Sequence[float]) -> bool:
    """点是否落在 WGS84 包围盒内。"""
    minx, miny, maxx, maxy = bbox
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


def libosmium_path(p: Path) -> str:
    """返回 libosmium 能成功打开的路径字符串。

    为什么需要这个
    --------------
    libosmium（pyosmium 的 C++ 底层）在 Windows 上**无法打开含非 ASCII
    字符的绝对路径**，会抛 ``RuntimeError: Open failed ... unknown error``。
    本项目的目录名含中文（``E:\\项目\\基于 GIS 的 eVTOL 起降场选址研究``），
    因此直接传绝对路径必然失败——而且报错信息完全看不出是路径编码问题。

    规避策略（按代价从低到高）：

    1. 路径本身是纯 ASCII —— 直接用；
    2. **相对当前工作目录的路径**是纯 ASCII —— 用相对路径。操作系统会
       按 CWD 解析，libosmium 只看到 ASCII 字符串。这是最省事的方式，
       因为 CWD 里有没有中文不影响；
    3. 以上都不行 —— 复制到系统临时目录（保证 ASCII 路径）。

    返回的路径在调用后应**立即**使用，避免 CWD 变化导致失效。
    """
    import os
    import shutil
    import tempfile

    # 1. 绝对路径本身是 ASCII
    ap = str(p.resolve())
    if ap.isascii():
        return ap

    # 2. 相对 CWD 的路径是 ASCII
    try:
        rel = os.path.relpath(ap)
        if rel.isascii() and not rel.startswith(".."):
            return rel
    except ValueError:
        pass

    # 3. 复制到临时目录
    tmp_dir = Path(tempfile.gettempdir()) / "evtol_osm"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / p.name
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        logger.warning(
            "PBF 路径含非 ASCII 字符且无法表示为相对路径，"
            "正在复制到临时目录: %s", dst,
        )
        shutil.copy2(p, dst)
    return str(dst)


def _processors(pbf_path: Path, bbox: Sequence[float], keys: Sequence[str]):
    """构造 pyosmium 的 FileProcessor（**含面积装配**）。

    过滤策略分两层：

    1. **C++ 层**用 ``KeyFilter`` 按标签键预筛——只把含目标键的要素交给
       Python。这一步是性能关键：四川省包里有数千万个节点，绝大多数与
       本研究无关。
    2. **Python 层**再做 bbox 判断与**取值**判定（``_matches_layer``）。

    ⚠ 关于 ``with_areas()``
    ----------------------
    早期版本用 ``osmium.osm.NODE | osmium.osm.WAY`` 作为实体掩码，
    **完全不读 relation**。后果不是"少几个要素"，而是**整类地物缺失**：
    大型水库、湖泊、机场在 OSM 里普遍是 ``type=multipolygon`` 的
    relation（标签挂在 relation 上，成员 way 本身无标签，因此即使被
    扫到也会被 ``KeyFilter`` 丢掉）。两城的水域/机场图层因此系统性缺了
    最大的一片，而日志只会显示一个偏小的要素数——不会报错。

    正确做法是 ``.with_areas(filter)``：pyosmium 会把**闭合 way 与
    multipolygon relation 一并装配成** ``Area`` 返回。注意实体掩码此时
    不能再限制（`FileProcessor` 文档明确说明：使用位置/面积处理器时
    通常不应限制实体），否则面积装配拿不到成员节点。

    传给 ``with_areas`` 的 KeyFilter 只作用于**第一遍**的 relation 候选
    筛选，不作为输出过滤。

    ⚠ **``with_filter`` 不能省。** 它和 ``with_areas`` 是两件事：前者是
    C++ 层的输出过滤（把无关要素挡在 Python 之外），后者开启面积装配。
    只留 ``with_areas`` 时，全省数千万个节点会**逐个进入 Python 循环**，
    实测解析从约 2 分钟劣化到 10 分钟以上仍不结束。面积装配后的 ``Area``
    对象同样会经过 ``with_filter``（见 ``FileProcessor.__iter__`` 里
    ``BufferIterator(*self._filters)``），所以这一个过滤两头都管。
    """
    import osmium

    kf = osmium.filter.KeyFilter(*keys)
    fp = (
        osmium.FileProcessor(libosmium_path(pbf_path), osmium.osm.ALL)
        .with_areas(kf)
        .with_filter(osmium.filter.KeyFilter(*keys))
    )
    return fp


def _way_centroid(way) -> tuple[float, float] | None:
    """way 的近似中心（节点坐标均值），用于 bbox 判断。

    对建筑轮廓这类要素，用中心点判 bbox 与用整体相交判 bbox 的差异仅
    存在于边界处的一小圈要素，而研究区外还有一圈缓冲，影响可忽略。
    这样做的收益是不必构造完整几何再做相交测试，速度差别明显。
    """
    try:
        n = len(way.nodes)
        if n == 0:
            return None
        s_lon = s_lat = 0.0
        for nd in way.nodes:
            s_lon += nd.lon
            s_lat += nd.lat
        return s_lon / n, s_lat / n
    except Exception:
        return None


def parse_layer(
    pbf_path: Path,
    layer: str,
    cfg,
    bbox: Sequence[float] | None = None,
) -> "pd.DataFrame":
    """从 PBF 中解析指定图层为 GeoDataFrame（投影坐标系）。

    Parameters
    ----------
    bbox
        WGS84 过滤框 ``(minx, miny, maxx, maxy)``。默认取 ``study_area.bbox``。

    Returns
    -------
    与 :func:`evtol_siting.data.osm.elements_to_gdf` **结构一致**的
    GeoDataFrame（含 OSM 标签列 + ``osm_id`` / ``osm_type`` / ``geometry``），
    因此可作为 Overpass 通道的**直接替代**。
    """
    return parse_all_layers(pbf_path, cfg, layers=[layer], bbox=bbox)[layer]


def parse_all_layers(
    pbf_path: Path,
    cfg,
    layers: Sequence[str] | None = None,
    bbox: Sequence[float] | None = None,
) -> dict:
    """解析一个或多个图层，返回 ``{图层名: GeoDataFrame}``。

    用**单次遍历**同时收集所有请求的图层。PBF 是顺序流式格式，每调用
    一次解析就要完整读一遍文件；逐个图层解析会把 116 MB 的文件读 9 遍，
    单次遍历则只需一遍（实测快约 5 倍）。

    三类几何的来源**互不重叠**，以免同一地物被记两次：

    * **面**（buildings / landuse / natural / water 的面部分 / airports）
      一律取自 ``Area``，即 pyosmium 装配后的闭合 way **与 multipolygon
      relation**。这是 relation 得以进入结果集的唯一通道。
    * **线**（roads / water 的线部分）取自**非闭合** way。闭合 way 已由
      ``Area`` 覆盖，若此处再取会重复计数。
    * **点**（poi / rail_stations）取自 node。

    早期版本只取 way，导致三层损失：大型水域与机场（relation）整体缺失、
    ``natural=wood`` 等被误判为水域、河流线被几何偏好挡掉。
    """
    import geopandas as gpd

    layers = list(layers or LAYER_KEYS.keys())
    bbox = tuple(bbox or cfg.bbox)
    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF 文件不存在: {pbf_path}")

    # 所有图层涉及的标签键并集，交给 C++ 层做预筛
    all_keys: list[str] = []
    for lyr in layers:
        for k in LAYER_KEYS[lyr]:
            if k not in all_keys:
                all_keys.append(k)

    buckets: dict[str, tuple[list, list]] = {lyr: ([], []) for lyr in layers}
    n_area_kept = 0
    n_area_bad = 0

    n = 0
    for obj in _processors(pbf_path, bbox, all_keys):
        n += 1
        if n % 200_000 == 0:
            logger.info("  已扫描 %d 个候选要素…", n)

        if obj.is_area():
            # 闭合 way + multipolygon relation（含大型水域/机场）
            tags = _tags_to_dict(obj.tags)
            target = [l for l in layers if _matches_layer(tags, l)
                      and LAYER_GEOM.get(l) in ("polygon", "any")]
            if not target:
                continue
            c = _ring_centroid(obj)
            if c is None or not _in_bbox(c[0], c[1], bbox):
                continue
            geom = _area_to_geometry(obj)
            if geom is None:
                n_area_bad += 1
                continue
            n_area_kept += 1
            area_type = _area_source(obj)
            for lyr in target:
                recs, gs = buckets[lyr]
                recs.append({**tags, "osm_id": obj.orig_id(),
                             "osm_type": area_type})
                gs.append(geom)
            continue

        if obj.is_node():
            if not _in_bbox(obj.lon, obj.lat, bbox):
                continue
            tags = _tags_to_dict(obj.tags)
            for lyr in layers:
                if LAYER_GEOM.get(lyr) == "line":
                    continue          # 线图层不取点
                if _matches_layer(tags, lyr):
                    from shapely.geometry import Point

                    recs, gs = buckets[lyr]
                    recs.append({**tags, "osm_id": obj.id, "osm_type": "node"})
                    gs.append(Point(obj.lon, obj.lat))
            continue

        if obj.is_way():
            tags = _tags_to_dict(obj.tags)
            # 只服务"线"图层；面已由 Area 分支处理，避免闭合 way 被记两次。
            target = [l for l in layers
                      if LAYER_GEOM.get(l) in ("line", "any")
                      and _matches_layer(tags, l)]
            if not target:
                continue
            c = _way_centroid(obj)
            if c is None or not _in_bbox(c[0], c[1], bbox):
                continue
            geom = _geometry_from_way(obj)
            if geom is None:
                continue
            # 线图层只接受非闭合 way：闭合 way 已由 Area 分支收纳，
            # 在这里再取一次会让同一地物重复计入。
            if geom.geom_type != "LineString":
                continue
            for lyr in target:
                recs, gs = buckets[lyr]
                recs.append({**tags, "osm_id": obj.id, "osm_type": "way"})
                gs.append(geom)

    out: dict = {}
    for lyr in layers:
        recs, gs = buckets[lyr]
        if recs:
            gdf = gpd.GeoDataFrame(recs, geometry=gs, crs=cfg.crs_geo).to_crs(cfg.crs_proj)
        else:
            logger.warning("图层 '%s' 在 bbox=%s 内无要素", lyr, bbox)
            gdf = gpd.GeoDataFrame(
                {"osm_id": [], "osm_type": []}, geometry=[], crs=cfg.crs_geo
            )
        out[lyr] = gdf
        logger.info("  %-16s %7d 个要素", lyr, len(gdf))

    logger.info(
        "PBF 解析完成（扫描 %d 个候选要素，装配并保留 %d 个面要素，"
        "丢弃 %d 个非法几何，输出 %d 个图层）",
        n, n_area_kept, n_area_bad, len(layers),
    )
    return out
