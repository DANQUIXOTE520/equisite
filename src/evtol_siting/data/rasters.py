"""栅格数据获取：人口、收入代理变量、高程。

数据源与选型理由
----------------
============  ==========================================  ==========
变量          数据源                                       分辨率
============  ==========================================  ==========
人口          WorldPop Unconstrained Individual Countries  100 m
              (China, R2021A / 2020)
收入          Meta Relative Wealth Index (RWI)             2.4 km
              备用：NPP-VIIRS 夜间灯光（500 m）
高程          Copernicus DEM GLO-30                        30 m
              （用于坡度计算，进而判定地面型候选点）
============  ==========================================  ==========

关于"月收入"的说明
------------------
申报书中的"月收入"在国内没有公开的格网尺度数据。本研究采用
**Meta Relative Wealth Index**（相对财富指数）作为收入的空间代理变量：
它是基于连接性、卫星影像与调查数据估计的 2.4 km 网格相对财富指标，
在同类研究中被广泛使用且可公开获取。RWI 是**相对**指标（均值 0、
标准差 1 的标准化值），用于 K-means 聚类时只关心其相对排序，
不影响聚类结果的解释。

论文中必须显式说明这一替换及其局限——不能声称使用的是真实月收入。
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 下载工具
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, user_agent: str, timeout_s: int = 600) -> Path:
    """带断点保护的流式下载（已存在且非空则跳过）。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("已存在，跳过下载: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    logger.info("下载 %s -> %s", url, dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (20 << 20) < (1 << 20):
                    logger.info("  %.1f%%  (%.0f/%.0f MB)", 100 * done / total, done / 1e6, total / 1e6)
    tmp.replace(dest)
    logger.info("完成: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def city_slug(cfg) -> str:
    """派生产物（裁剪后的人口/收入/高度/DEM）文件名里的城市标识。

    为什么需要这个
    --------------
    这四个文件名早期把 ``"chengdu"`` **写死**在代码里。单城市时看不出问题，
    多城市时是静默失败的：换了城市的配置，``chengdu_population_100m.tif``
    等文件**已经存在**，于是下载与裁剪被整体跳过，流水线拿 A 城的人口、
    收入、建筑高度和 DEM 去跑 B 城的选址，而日志只打印一行"已存在"，
    看不出异常。（实测：错配城市时屋顶型候选因此只剩 16 个，正常应为数百个。）

    取配置文件的主干名（``chengdu.yaml`` → ``chengdu``），成都既有缓存的
    文件名因此**保持不变**，不需要迁移。
    """
    return cfg.source_path.stem


def unzip_if_needed(path: Path, extract_to: Path | None = None) -> list[Path]:
    """若是 zip 则解压，返回其中的 .tif/.csv 文件列表。"""
    if path.suffix.lower() != ".zip":
        return [path]
    target = extract_to or path.parent
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        zf.extractall(target)
    return [target / n for n in names if Path(n).suffix.lower() in {".tif", ".tiff", ".csv", ".asc"}]


# ---------------------------------------------------------------------------
# 人口
# ---------------------------------------------------------------------------

def fetch_population(cfg, bbox: Sequence[float] | None = None) -> Path | None:
    """获取 WorldPop 100 m 人口栅格（裁剪到研究区）。

    WorldPop 的中国数据集按 1°×1° 瓦片发布。本函数下载覆盖研究区的
    瓦片并做马赛克裁剪；若下载失败且配置允许，返回 ``None`` 并由调用方
    回退到合成数据。

    Returns
    -------
    裁剪后的人口 GeoTIFF 路径；失败返回 ``None``。
    """
    bbox = bbox or cfg.bbox
    user_agent = cfg.get("data.overpass.user_agent")
    raw_dir = cfg.dir("data.raw_dir") / "population"
    out = raw_dir / f"{city_slug(cfg)}_population_100m.tif"

    if out.exists() and out.stat().st_size > 0:
        logger.info("人口栅格已存在: %s", out)
        return out

    # WorldPop 中国 2020 约束版 100 m。URL 已于 2026-09 实测验证（HTTP 200，
    # Content-Length 920,085,083，image/tiff）。注意 WorldPop 会调整发布路径，
    # 若此 URL 失效，可在 https://www.worldpop.org/ 检索 "China 100m constrained"。
    #
    # 注意：该服务器**忽略 Range 请求头**，不支持断点续传，必须一次性下载
    # 完整的 920 MB。未约束版约 4 GB，不建议使用。
    base = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2020/CHN/v1/100m/constrained"
    url = f"{base}/chn_pop_2020_CN_100m_R2024B_v1.tif"

    try:
        tif = download_file(url, raw_dir / "chn_ppp_2020_constrained.tif", user_agent)
    except Exception as exc:
        logger.warning("WorldPop 下载失败: %s", str(exc)[:200])
        return _population_fallback(cfg, bbox)

    return _clip_raster_to_bbox(tif, bbox, out, band=1)


def _population_fallback(cfg, bbox):
    """人口数据不可得时的回退路径。"""
    if not cfg.get("data.allow_synthetic_fallback", True):
        raise RuntimeError("人口数据下载失败且未启用合成回退（data.allow_synthetic_fallback=false）")
    logger.warning(
        "人口数据不可得：将使用由 OSM 建筑体量估计的合成人口分布。"
        "该结果**不可用于论文实证结论**，仅用于验证流程连通性。"
    )
    return None


def _clip_raster_to_bbox(src_path: Path, bbox, out_path: Path, band: int = 1) -> Path | None:
    """按 WGS84 bbox 裁剪栅格并写出。

    ⚠️ ``rio_mask`` 需要的是**几何对象列表**（GeoJSON 几何 dict 或 shapely
    几何），不是 GeoDataFrame 的 ``to_json()`` 输出——后者是
    ``FeatureCollection``，直接传进去不会报错，但会产出尺寸与 transform
    都错乱的栅格（实测得到一个 2×1 像素、落在哈萨克斯坦的结果）。
    这里显式做坐标变换后传 shapely 几何，避免该陷阱。
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from pyproj import Transformer
    from shapely.geometry import box as shp_box
    from shapely.ops import transform as shp_transform

    try:
        with rasterio.open(src_path) as src:
            poly = shp_box(*bbox)
            if str(src.crs) not in ("EPSG:4326", "OGC:CRS84", "") and src.crs is not None:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                poly = shp_transform(tr.transform, poly)

            out_image, out_transform = rio_mask(
                src, [poly], crop=True, nodata=src.nodata, all_touched=False
            )
            profile = src.profile.copy()
            profile.update(
                height=out_image.shape[1], width=out_image.shape[2],
                transform=out_transform, compress="lzw",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_image)

            # 裁剪结果自检：分辨率为 0 或像元数过少都说明裁剪失败。
            # 这类失败不会抛异常，只会让下游静默拿到全 0 数据，
            # 因此必须显式拦截。
            res_x = abs(out_transform.a)
            if res_x <= 0 or out_image.shape[1] < 10 or out_image.shape[2] < 10:
                logger.error(
                    "裁剪结果异常（分辨率 %.6g，尺寸 %s）——保留原文件，请检查 CRS",
                    res_x, out_image.shape,
                )
                out_path.unlink(missing_ok=True)
                return src_path

        logger.info(
            "栅格已裁剪: %s（%d×%d 像元，分辨率 %.1f m）",
            out_path.name, out_image.shape[2], out_image.shape[1],
            abs(out_transform.a) * (111320 if (src.crs and src.crs.is_geographic) else 1),
        )
        return out_path
    except Exception as exc:
        logger.warning("栅格裁剪失败 (%s)，保留原文件", str(exc)[:200])
        return src_path


# ---------------------------------------------------------------------------
# 收入 / 财富代理
# ---------------------------------------------------------------------------

def fetch_income(cfg, bbox: Sequence[float] | None = None) -> Path | None:
    """获取收入代理变量。

    ⚠️ 重要更正：Meta RWI **不覆盖中国**
    --------------------------------------
    最初设计采用 Meta Relative Wealth Index。经实测核验（2026-09，
    ``package_show?id=relative-wealth-index`` 返回 112 个资源），
    **RWI 仅覆盖 93 个中低收入国家，中国被明确排除在设计范围之外**。
    任何"用 RWI 表示成都市收入"的写法都是错误的，必须在论文中避免。

    因此本函数改为**策略链**，按可获得性依次尝试：

    ==================  ====================================================
    策略                 说明
    ==================  ====================================================
    ``viirs``            NPP-VIIRS 夜间灯光年合成。**需在 EOG 注册免费
                         账号后手动下载**（eogdata.mines.edu 走 OAuth，
                         无法自动抓取）。下载后放到
                         ``data/raw/income/`` 并在配置中指定文件名。
    ``poi_index``        基于 OSM POI 构成的经济活动指数（**全自动**）。
                         见 :func:`build_poi_economic_index`。
    ``local_file``       用户自备数据（房价、统计年鉴人均可支配收入等）。
    ``synthetic``        合成梯度，仅用于离线测试。
    ==================  ====================================================

    论文中的表述要求
    ----------------
    使用 ``poi_index`` 时，收入维度必须描述为**"基于 POI 构成的经济活动
    指数"**，而不是"收入"。二者不等价：前者度量的是商业活跃度与业态
    档次，与居民收入高度相关但并非同一概念。第 3.5 节（数据）应说明这一
    代理关系及其局限，敏感性分析应检验结论对该代理的依赖程度。

    Returns
    -------
    含 ``rwi`` 列（此处为经济指数，列名保持兼容）的点矢量路径；
    全部策略失败时返回 ``None``。
    """
    bbox = bbox or cfg.bbox
    strategy = str(cfg.get("data.sources.income", "poi_index"))
    raw_dir = cfg.dir("data.raw_dir") / "income"
    out = raw_dir / f"{city_slug(cfg)}_income_proxy.gpkg"

    if out.exists() and out.stat().st_size > 0:
        logger.info("收入代理数据已存在: %s", out)
        return out

    if strategy == "viirs":
        got = _fetch_viirs(cfg, bbox, raw_dir, out)
        if got is not None:
            return got
        logger.warning("VIIRS 策略未取得数据，回退到 POI 经济指数")
        strategy = "poi_index"

    if strategy == "local_file":
        return _fetch_income_local(cfg, out)

    if strategy == "synthetic":
        return None      # 交由 stage2 的 synthesise_features 处理

    # 默认：POI 经济指数（由 stage2 在构建特征时计算，此处返回标记路径）
    logger.info(
        "收入策略 = poi_index：经济活动指数将在阶段二由 OSM POI 构成计算"
        "（见 stage2_demand.build_poi_economic_index）"
    )
    return None


def _fetch_income_local(cfg, out: Path) -> Path | None:
    """读取用户自备的收入/房价数据文件。"""
    src = cfg.get("data.local_income_file")
    if not src:
        logger.warning("data.sources.income='local_file' 但未设置 data.local_income_file")
        return None
    p = Path(src)
    if not p.is_absolute():
        p = cfg.path("data.raw_dir") / src
    if not p.exists():
        logger.warning("自备收入数据不存在: %s", p)
        return None
    logger.info("使用自备收入数据: %s", p)
    return p


def _fetch_viirs(cfg, bbox, raw_dir: Path, out: Path) -> Path | None:
    """尝试载入本地已下载的 VIIRS 夜间灯光栅格并裁剪到研究区。

    EOG 数据门户需要 OAuth 登录，**无法自动下载**。用户需自行注册
    （免费）后下载年度合成产品，例如::

        VNL_v22_npp-j01_2020_global_vcmslcfg_v2_c20210315.average_masked.dat.tif.gz

    放到 ``data/raw/income/``，本函数会自动发现并裁剪。
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(raw_dir.glob("*VNL*.tif")) + sorted(raw_dir.glob("*VNL*.tif.gz"))
    if not candidates:
        logger.info(
            "未在 %s 找到 VIIRS 夜间灯光文件。"
            "如需使用，请在 https://eogdata.mines.edu/products/vnl/ 注册免费账号"
            "下载年度合成产品放入该目录。",
            raw_dir,
        )
        return None

    tif = candidates[-1]
    logger.info("发现 VIIRS 夜间灯光: %s", tif.name)
    if tif.suffix == ".gz":
        import gzip
        import shutil

        plain = tif.with_suffix("")
        if not plain.exists():
            logger.info("解压 %s ...", tif.name)
            with gzip.open(tif, "rb") as fi, open(plain, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        tif = plain

    clipped = _clip_raster_to_bbox(tif, bbox, out.with_suffix(".tif"), band=1)
    if clipped is None:
        return None
    # ⚠ fetch_income 的契约是**返回含 rwi 列的点矢量**（build_features 会对它做
    #   反距离加权插值），而裁剪产出的是栅格。此处必须做一次栅格→点的转换，
    #   否则调用方拿到 Path 会抛 `TypeError: object of type 'WindowsPath'
    #   has no len()`。此前该分支从未被真正跑过，这个不匹配一直没暴露。
    return _raster_to_points(clipped, out)


def _raster_to_points(tif: Path, out_gpkg: Path) -> Path | None:
    """把栅格转成质心点矢量，栅格值写入 ``rwi`` 列。

    每个有效像元变成一个点，坐标为像元中心；``rwi`` 为该像元的观测值。
    零点保留——在夜光产品中"无光"是**真实观测**（经济活动弱），而不是缺失。
    """
    import numpy as np
    import rasterio

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:
        logger.warning("缺少 geopandas/shapely，无法把栅格转为点矢量: %s", str(exc)[:120])
        return None

    with rasterio.open(tif) as r:
        a = r.read(1).astype(np.float64)
        if r.nodata is not None:
            a = np.where(a == r.nodata, np.nan, a)
        rows, cols = np.nonzero(np.isfinite(a))
        xs, ys = rasterio.transform.xy(r.transform, rows, cols)
        vals = a[rows, cols]
        crs = r.crs

    gdf = gpd.GeoDataFrame(
        {"rwi": vals.astype(np.float64)},
        geometry=[Point(float(x), float(y)) for x, y in zip(xs, ys)],
        crs=crs,
    )
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    logger.info(
        "VIIRS 已转为点矢量: %d 个点（rwi 范围 %.3f–%.3f）-> %s",
        len(gdf), float(vals.min()) if len(vals) else float("nan"),
        float(vals.max()) if len(vals) else float("nan"), out_gpkg.name,
    )
    return out_gpkg


def build_poi_economic_index(poi, grid, cfg) -> np.ndarray:
    """由 OSM POI 构成构造**经济活动指数**（收入代理，全自动）。

    构造思路
    --------
    夜间灯光与 POI 密度都是经济活跃度的经典代理变量。单纯的 POI **计数**
    会退化为"商业密度"，无法区分业态档次。本函数因此对 POI 按类别赋权，
    权重反映该业态与高收入人群/高价值地段的关联强度：

    ====================================  ======   ==========================
    POI 类别                               权重     依据
    ====================================  ======   ==========================
    银行/金融 (bank, atm, bureau_de_change)  3.0     金融网点密度是城市地价
                                                     与收入水平的强预测变量
    写字楼 (office)                          2.5     高薪就业岗位的空间分布
    高档零售 (department_store, mall,
              jewelry, boutique)             2.5     消费档次
    酒店 (hotel)                             2.0     商务活动强度
    餐饮 (restaurant, cafe, bar)             1.0     基准商业活跃度
    生活服务、超市等                          0.6     普遍性业态，区分度低
    ====================================  ======   ==========================

    指数定义为单元内加权 POI 数除以单元面积（个/km²），再取对数以压缩
    长尾。取对数后的分布在核心区与外围之间更接近线性梯度，有利于
    K-means 形成有解释力的聚类。

    Parameters
    ----------
    poi
        OSM POI 点矢量（投影坐标系）。
    grid
        需求栅格。

    Returns
    -------
    长度等于 ``len(grid)`` 的指数数组。
    """
    import geopandas as gpd

    # 权重按 OSM 标签值映射；未列出的类别取默认权重
    WEIGHTS: dict[str, float] = {
        # 金融
        "bank": 3.0, "atm": 3.0, "bureau_de_change": 3.0, "financial": 3.0,
        # 办公
        "office": 2.5, "company": 2.5, "government": 2.5, "embassy": 2.5,
        # 高档零售
        "department_store": 2.5, "mall": 2.5, "jewelry": 2.5,
        "boutique": 2.5, "watches": 2.5, "car": 2.0,
        # 酒店
        "hotel": 2.0, "resort": 2.0,
        # 餐饮娱乐
        "restaurant": 1.0, "cafe": 1.0, "bar": 1.0, "pub": 1.0,
        "nightclub": 1.0, "fast_food": 0.8,
        # 生活服务
        "supermarket": 0.7, "convenience": 0.5, "bakery": 0.5,
        "hairdresser": 0.5, "laundry": 0.5, "pharmacy": 0.6,
        "school": 0.6, "kindergarten": 0.5, "hospital": 0.8,
        "clinic": 0.6, "place_of_worship": 0.4, "parking": 0.3,
    }
    DEFAULT_W = 0.7

    if poi is None or len(poi) == 0:
        logger.warning("POI 数据为空，经济活动指数置 0")
        return np.zeros(len(grid), dtype=np.float64)

    p = poi
    if str(p.crs) != str(grid.crs):
        p = p.to_crs(grid.crs)

    # 逐行取第一个命中的标签值作为类别
    tag_cols = [c for c in ("amenity", "shop", "office", "leisure", "tourism",
                            "poi_category") if c in p.columns]
    if not tag_cols:
        logger.warning("POI 缺少可用标签列（%s），指数置 0", list(p.columns))
        return np.zeros(len(grid), dtype=np.float64)

    def _w(row):
        for c in tag_cols:
            v = row.get(c)
            if isinstance(v, str) and v:
                return WEIGHTS.get(v.lower(), DEFAULT_W)
        return DEFAULT_W

    weights = p.apply(_w, axis=1).to_numpy(dtype=np.float64)

    pts = gpd.GeoDataFrame({"w": weights}, geometry=p.geometry.centroid, crs=grid.crs)
    joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    wsum = joined.groupby("cell_id")["w"].sum()

    cell_area_km2 = (float(cfg.get("stage1_candidates.grid_size_m", 500.0)) / 1000.0) ** 2
    vals = grid["cell_id"].map(wsum).fillna(0.0).to_numpy(dtype=np.float64) / cell_area_km2

    # log1p 压缩长尾；结果非负
    idx = np.log1p(vals)
    logger.info(
        "POI 经济活动指数: 范围 [%.3f, %.3f]，中位数 %.3f（加权 POI 总数 %.0f）",
        float(idx.min()), float(idx.max()), float(np.median(idx)), float(weights.sum()),
    )
    return idx


# ---------------------------------------------------------------------------
# 建筑高度栅格 (CNBH-10m)
# ---------------------------------------------------------------------------

# CNBH-10m 的瓦片命名规则（**官方未文档化，以下为实测解码结果**）：
#   瓦片为 2°×2°，文件名中的 X/Y 是瓦片的**中心**坐标，而非左下角。
#   例：CNBH10m_X103Y31 覆盖 lon 102–104, lat 29.97–32.03。
# ⚠️ 若按常规的 `lon // 2` 角点索引推算瓦片名，会取到错误瓦片，
#    导致研究区恰好落在瓦片间的空隙里而得不到任何高度数据。
CNBH_ZENODO_RECORD = "7923866"
CNBH_BASE_URL = f"https://zenodo.org/records/{CNBH_ZENODO_RECORD}/files"


def cnbh_center(v: float) -> int:
    """把经纬度映射到覆盖它的 CNBH 瓦片**中心坐标**。

    瓦片以中心命名，覆盖 ``[c-1, c+1]``，故中心坐标为奇数：
    103, 105, …。实测：lon=104.06 -> 105，lon=103.95 -> 103。
    """
    return int(np.floor(v / 2.0) * 2 + 1)


def cnbh_tile_name(lon: float, lat: float) -> str:
    """由经纬度推算覆盖该点的 CNBH-10m 瓦片名（按**中心坐标**命名）。

    实测：成都 (104.06, 30.66) -> ``CNBH10m_X105Y31``。
    """
    return f"CNBH10m_X{cnbh_center(lon)}Y{cnbh_center(lat)}"


def cnbh_tiles_for_bbox(bbox: Sequence[float]) -> list[str]:
    """列出覆盖给定 bbox 所需的全部 CNBH 瓦片名。

    做法：枚举 bbox 跨度内的**每一个整数度**（以及两端点），逐个映射到
    其所在瓦片的中心坐标。由于瓦片边界落在偶数整数上，枚举全部整数
    保证不会漏掉任何跨越边界的瓦片——例如 bbox 为 103.88–104.29 时，
    必须同时取 X103 与 X105 两个瓦片，而不是看上去对应的一个。

    实测：成都研究区 bbox (103.88, 30.36, 104.28, 30.82) 恰好得到
    ``[X103Y31, X105Y31]`` 两个瓦片，与人工核验一致。
    """
    minx, miny, maxx, maxy = (float(v) for v in bbox)

    def _centers(lo: float, hi: float) -> set[int]:
        pts = {lo, hi}
        pts.update(float(k) for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
        return {cnbh_center(p) for p in pts}

    lon_cs = _centers(minx, maxx)
    lat_cs = _centers(miny, maxy)
    return sorted(
        f"CNBH10m_X{lc}Y{tc}" for tc in sorted(lat_cs) for lc in sorted(lon_cs)
    )


def fetch_building_height_raster(cfg, bbox: Sequence[float] | None = None) -> Path | None:
    """下载并镶嵌 CNBH-10m 中国建筑高度栅格。

    CNBH-10m（Wu et al., 2023, Zenodo ``10.5281/zenodo.7923866``）是
    目前公开可得的中国建筑高度产品中分辨率最高的（10 m，2020 年基准），
    精度报告为 RMSE 6.1 m / MAE 5.2 m / R=0.77。

    本函数相对 :func:`evtol_siting.data.heights.fuse_building_heights`
    的意义：OSM 的高度标签在成都市中心的覆盖率仅 1.5–5.4%，CNBH 为
    **无标签建筑**提供了独立的高度信息，是高度融合第 3 级的数据来源。

    Returns
    -------
    镶嵌后的高度栅格路径；失败返回 ``None``（此时融合会自动跳到第 4 级
    的统计填补，并在日志中告警）。
    """
    bbox = bbox or cfg.bbox
    user_agent = cfg.get("data.overpass.user_agent")
    raw_dir = cfg.dir("data.raw_dir") / "height"
    out = raw_dir / f"{city_slug(cfg)}_building_height_10m.tif"

    if out.exists() and out.stat().st_size > 0:
        logger.info("建筑高度栅格已存在: %s", out)
        return out

    tile_names = cnbh_tiles_for_bbox(bbox)
    logger.info(
        "CNBH 需要 %d 个瓦片覆盖 bbox=%s: %s",
        len(tile_names), tuple(bbox), ", ".join(tile_names),
    )

    tiles: list[Path] = []
    for name in tile_names:
        url = f"{CNBH_BASE_URL}/{name}.tif?download=1"
        dest = raw_dir / f"{name}.tif"
        try:
            tiles.append(download_file(url, dest, user_agent, timeout_s=900))
        except Exception as exc:
            logger.warning("CNBH 瓦片 %s 下载失败: %s", name, str(exc)[:140])

    if not tiles:
        logger.warning(
            "未能获取任何 CNBH 瓦片。请在 https://zenodo.org/records/%s "
            "手动下载并放入 %s", CNBH_ZENODO_RECORD, raw_dir,
        )
        return None

    try:
        import rasterio
        from rasterio.merge import merge

        srcs = [rasterio.open(t) for t in tiles]
        arr, transform = merge(srcs, nodata=0.0)
        profile = srcs[0].profile.copy()
        profile.update(height=arr.shape[1], width=arr.shape[2],
                       transform=transform, nodata=0.0, compress="lzw")
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr)
        for s in srcs:
            s.close()
        logger.info(
            "CNBH 高度栅格镶嵌完成: %s（%d 个瓦片，%s，%s）",
            out, len(tiles), srcs[0].crs if srcs else "?", arr.shape,
        )
        return out
    except Exception as exc:
        logger.warning("CNBH 镶嵌失败: %s", str(exc)[:200])
        return None


# ---------------------------------------------------------------------------
# 高程与坡度
# ---------------------------------------------------------------------------

# Copernicus DEM GLO-30 在 AWS 开放数据桶上的路径规则
_COPERNICUS_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)


def copernicus_tile_url(lat: int, lon: int) -> str:
    """构造 Copernicus DEM GLO-30 的瓦片 URL（按整度取瓦片）。"""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return _COPERNICUS_URL.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))


def fetch_dem(cfg, bbox: Sequence[float] | None = None) -> Path | None:
    """下载覆盖研究区的 Copernicus DEM GLO-30 瓦片并镶嵌。"""
    bbox = bbox or cfg.bbox
    user_agent = cfg.get("data.overpass.user_agent")
    raw_dir = cfg.dir("data.raw_dir") / "dem"
    out = raw_dir / f"{city_slug(cfg)}_dem_30m.tif"

    if out.exists() and out.stat().st_size > 0:
        logger.info("DEM 已存在: %s", out)
        return out

    minx, miny, maxx, maxy = bbox
    lats = range(int(np.floor(miny)), int(np.floor(maxy)) + 1)
    lons = range(int(np.floor(minx)), int(np.floor(maxx)) + 1)

    tiles: list[Path] = []
    for lat in lats:
        for lon in lons:
            url = copernicus_tile_url(lat, lon)
            dest = raw_dir / Path(url).name
            try:
                tiles.append(download_file(url, dest, user_agent))
            except Exception as exc:
                logger.warning("DEM 瓦片 %s 下载失败: %s", Path(url).name, str(exc)[:120])

    if not tiles:
        logger.warning("未能获取任何 DEM 瓦片，坡度计算将不可用")
        return None

    try:
        import rasterio
        from rasterio.merge import merge

        srcs = [rasterio.open(t) for t in tiles]
        arr, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=arr.shape[1], width=arr.shape[2], transform=transform, compress="lzw"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr)
        for s in srcs:
            s.close()
        logger.info("DEM 镶嵌完成: %s", out)
        return out
    except Exception as exc:
        logger.warning("DEM 镶嵌失败: %s", str(exc)[:200])
        return None


def compute_slope(dem_path: Path, out_path: Path, units: str = "degrees") -> Path | None:
    """由 DEM 计算坡度栅格。

    使用 Horn (1981) 三阶反距离平方权差分算子，与 ArcGIS 的
    ``Slope`` 工具一致，便于与既有研究对照。结果用于阶段一判定
    地面型候选点是否满足 ``slope <= 15°``。
    """
    import rasterio
    import numpy as np

    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            nodata = src.nodata
            transform = src.transform
            profile = src.profile.copy()

        if nodata is not None:
            dem = np.where(np.isclose(dem, nodata), np.nan, dem)

        # 像元尺寸（DEM 为地理坐标时需按纬度换算经度方向的实际距离）
        dy = abs(transform.e)                       # 纬度方向（米/度 -> 需换算）
        dx = abs(transform.a)
        if src.crs and src.crs.is_geographic:
            lat_mid = transform.f + transform.e * dem.shape[0] / 2
            m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(lat_mid)) + 1.175 * np.cos(4 * np.radians(lat_mid))
            m_per_deg_lon = 111412.84 * np.cos(np.radians(lat_mid)) - 93.5 * np.cos(3 * np.radians(lat_mid))
            dx_m, dy_m = dx * m_per_deg_lon, dy * m_per_deg_lat
        else:
            dx_m, dy_m = dx, dy

        # Horn 算子
        z = np.pad(dem, 1, mode="edge")
        dzdx = (
            (z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:])
            - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])
        ) / (8 * dx_m)
        dzdy = (
            (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
            - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
        ) / (8 * dy_m)

        slope_rad = np.arctan(np.hypot(dzdx, dzdy))
        slope = np.degrees(slope_rad) if units == "degrees" else slope_rad
        slope = np.where(np.isfinite(slope), slope, 0.0)

        profile.update(dtype="float32", compress="lzw")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(slope.astype(np.float32), 1)
        logger.info("坡度栅格已生成: %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("坡度计算失败: %s", str(exc)[:200])
        return None


# ---------------------------------------------------------------------------
# 栅格聚合到需求单元
# ---------------------------------------------------------------------------

def zonal_sum_to_grid(raster_path: Path, grid, cfg, stat: str = "sum") -> np.ndarray:
    """把栅格聚合到需求栅格单元（分区统计）。

    人口栅格（100 m）比需求单元（500 m）细，因此需要按单元求和；
    若栅格比单元粗（如 RWI 的 2.4 km），则退化为按单元质心取值。

    实现上用 ``rasterio.features.geometry_mask`` 逐单元统计——单元数
    ~2400，直接循环可接受（< 5 s），且避免了 ``rasterstats`` 的额外依赖。
    """
    import rasterio
    from rasterio.features import geometry_mask

    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(np.isclose(arr, nodata), np.nan, arr)
        transform = src.transform
        raster_crs = src.crs

    grid_r = grid.to_crs(raster_crs) if str(grid.crs) != str(raster_crs) else grid
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    # 栅格尺度粗于需求单元 -> 用质心采样
    cell_size = float(cfg.get("stage1_candidates.grid_size_m", 500.0))
    if res_x > cell_size:
        from ..geo import sample_raster_at_points

        cx = grid.geometry.centroid.x.values
        cy = grid.geometry.centroid.y.values
        vals = sample_raster_at_points(raster_path, cx, cy, dst_crs=str(grid.crs))
        return np.nan_to_num(vals, nan=0.0)

    out = np.zeros(len(grid_r), dtype=np.float64)
    for i, geom in enumerate(grid_r.geometry.values):
        try:
            mask = geometry_mask(
                [geom], out_shape=arr.shape, transform=transform,
                invert=True, all_touched=False,
            )
            vals = arr[mask]
            if vals.size == 0:
                out[i] = 0.0
            elif stat == "sum":
                out[i] = np.nansum(vals)
            elif stat == "mean":
                out[i] = np.nanmean(vals) if np.isfinite(vals).any() else 0.0
            else:
                raise ValueError(f"不支持的统计量: {stat}")
        except Exception:
            out[i] = 0.0
    return out
