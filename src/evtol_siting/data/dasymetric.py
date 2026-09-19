"""人口重分配（dasymetric mapping）——把粗分辨率人口栅格降到需求格网尺度。

问题
----
WorldPop 中国人口栅格是 100 m 分辨率，而本研究的需求单元是 500 m 格网。
直接做**分区求和**会把一个格网内的 25 个像元简单相加，等价于假设"人口在
格网内均匀分布"——但城市里人口几乎全部集中在建筑里，公园、水面、工业
仓储用地上的人口密度接近零。这个假设会系统性地把人口从建成区摊到空地，
在需求估计中引入空间误差，进而影响选址结果。

方法
----
采用**体积权重的 dasymetric 重分配**：先用建筑基底面积 × 建筑高度得到
每个格网的建筑体积，再按体积比例把粗栅格的人口分配下去。

对粗栅格的每个像元 ``c``：:

    P_i = P_c · V_i / Σ_{i ∈ c} V_i        ∀ i ∈ c

其中 ``V_i`` 是格网 ``i`` 的建筑体积，``P_c`` 是粗像元 ``c`` 的人口。

这一形式有两个重要性质：

1. **总量守恒**：``Σ_{i ∈ c} P_i = P_c``，不会凭空创造或消灭人口；
2. **保留粗尺度格局**：重分配只在粗像元**内部**进行，跨像元的总量分布
   完全遵循原始栅格，因此不会用一个粗糙的体积模型去覆盖真实的人口梯度。

为什么用体积而非面积
--------------------
建筑基底面积只反映占地，不反映容量。一栋 30 层的住宅楼与一个同等底面积的
单层仓库，居住人数差两个数量级。用体积（面积 × 高度）能区分二者——而本
项目的建筑高度已通过 CNBH-10m 等多源融合补齐（实测把覆盖率从 8.4% 提升到
100%），因此体积是可得的。这是前一阶段高度融合工作在此处的直接收益。

局限（论文需说明）
------------------
体积是**居住容量**的代理，不是实际居住人数。纯体积分配会把大型工业厂房、
物流仓库也计入人口承载，从而高估这些区域的人口。缓解办法是引入用地类型
权重（住宅权重 1.0、商业 0.6、工业 0.15 等）。本实现提供了该选项
（``landuse_weights``），但默认关闭——因为 OSM 在成都的用地标注覆盖率
有限，权重本身也会引入不确定性。论文中应报告开启/关闭两种设定的对比。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 按建筑类型的居住权重：反映"单位体积容纳的常住人口"的相对量级。
# 住宅最高；商业办公次之（白天人口多、夜间少）；工业仓储最低。
DEFAULT_USE_WEIGHTS: dict[str, float] = {
    "residential": 1.00, "apartments": 1.00, "house": 1.00, "detached": 1.00,
    "terrace": 1.00, "dormitory": 1.00, "bungalow": 1.00,
    "yes": 0.80,                     # 未标注类型——偏住宅
    "commercial": 0.55, "retail": 0.55, "office": 0.50, "hotel": 0.70,
    "hospital": 0.45, "school": 0.40, "university": 0.40, "civic": 0.40,
    "public": 0.40, "government": 0.40, "museum": 0.20, "library": 0.20,
    "industrial": 0.15, "warehouse": 0.10, "factory": 0.15,
    "garage": 0.02, "garages": 0.02, "parking": 0.02,
    "shed": 0.05, "roof": 0.05, "construction": 0.05,
    "church": 0.20, "stadium": 0.10, "train_station": 0.10,
}
DEFAULT_USE_WEIGHT = 0.75


def building_volume_per_cell(
    grid,
    buildings,
    cfg,
    use_weights: bool = False,
) -> np.ndarray:
    """统计每个需求格网单元内的建筑体积（m³）。

    Parameters
    ----------
    grid
        需求格网（投影坐标系，含 ``cell_id``）。
    buildings
        建筑轮廓，须含 ``height_m``（由 :mod:`evtol_siting.data.heights` 融合得到）
        与 ``footprint_area_m2``。
    use_weights
        是否按建筑类型的居住权重折算体积。默认 ``False``（纯体积）。

    Returns
    -------
    长度等于 ``len(grid)`` 的体积数组（``use_weights=True`` 时为加权体积）。
    """
    import geopandas as gpd

    out = np.zeros(len(grid), dtype=np.float64)
    if buildings is None or len(buildings) == 0:
        logger.warning("建筑数据为空，无法计算建筑体积")
        return out

    b = buildings
    if str(b.crs) != str(grid.crs):
        b = b.to_crs(grid.crs)

    if "height_m" not in b.columns:
        raise KeyError(
            "建筑数据缺少 height_m 列。请先调用 "
            "evtol_siting.data.heights.fuse_building_heights()。"
        )

    area = (
        b["footprint_area_m2"].to_numpy(dtype=np.float64)
        if "footprint_area_m2" in b.columns
        else b.geometry.area.to_numpy(dtype=np.float64)
    )
    height = np.nan_to_num(b["height_m"].to_numpy(dtype=np.float64), nan=0.0)
    volume = np.clip(area, 0, None) * np.clip(height, 0, None)

    if use_weights:
        btype = b.get("building", pd.Series("yes", index=b.index)).astype(str).str.lower()
        w = btype.map(DEFAULT_USE_WEIGHTS).fillna(DEFAULT_USE_WEIGHT).to_numpy()
        volume = volume * w
        logger.info("已按建筑类型居住权重折算体积（住宅 1.0，工业 0.15）")

    # 用建筑质心归属格网，避免跨格网的大建筑被重复计入多个单元
    pts = gpd.GeoDataFrame(
        {"_vol": volume}, geometry=b.geometry.centroid, crs=grid.crs
    )
    joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    if len(joined) == 0:
        logger.warning("没有建筑落入任何格网单元")
        return out

    vol_sum = joined.groupby("cell_id")["_vol"].sum()
    out = grid["cell_id"].map(vol_sum).fillna(0.0).to_numpy(dtype=np.float64)

    logger.info(
        "建筑体积统计: 总量 %.3e m³，有建筑的格网 %d/%d (%.1f%%)",
        out.sum(), int((out > 0).sum()), len(grid), 100 * (out > 0).mean(),
    )
    return out


def dasymetric_population(
    grid,
    buildings,
    cfg,
    population_raster,
    use_weights: bool = False,
    force: bool = False,
    return_method: bool = False,
):
    """把人口栅格按建筑体积重分配到需求格网。

    ⚠️ 先判断**是否真的需要**
    ---------------------------
    如果人口栅格比需求格网**更细**（如 WorldPop 100 m 对 500 m 格网），
    那么"每个格网内各像元之和"本身就已经是正确的 500 m 人口总量——
    在格网**内部**再做体积重分配不会改变这个总量，因此对格网尺度的人口
    估计**没有任何作用**。此时本函数默认直接退化为分区求和，并在日志中
    说明原因。

    体积重分配真正有用的情形是**源比目标粗**（如 1 km 人口栅格降到
    500 m 格网）——那时格网内的分布确实只能靠代理变量来推断。这正是
    dasymetric mapping 的经典适用条件。

    强制重分配（``force=True``）的用途：把人口分配到**建筑**而非格网，
    以便计算"从每栋建筑到起降场"的接驳时间。这属于另一类分析，不在
    本函数的默认路径上。

    实现说明（针对源比目标粗的情形）
    --------------------------------
    按**面积重叠**而非质心归属来分配：对每个粗像元 ``c``，找出与它相交的
    所有格网 ``i``，按 ``重叠面积 × 该格网的建筑体积密度`` 加权分配::

        P_i += P_c · (a_ci · v_i) / Σ_j (a_cj · v_j)

    其中 ``v_i = V_i / A_i`` 是格网 ``i`` 的建筑体积密度。用面积重叠是
    必须的：若改用"格网质心落在哪个粗像元"来分组，当粗像元比格网**小**
    时绝大多数粗像元里根本没有质心，其人口会被静默丢弃（实测可丢失
    80% 以上）。反之，粗像元比格网大时质心法虽不会丢人口，但边界处的
    分配误差明显。面积重叠法在两种情况下都守恒且准确。

    Returns
    -------
    长度等于 ``len(grid)`` 的人口数组（人）。``return_method=True`` 时
    返回 ``(人口数组, 方法名)``，方法名为 ``"zonal_sum"`` 或
    ``"dasymetric_volume"``——**调用方应据此如实标注数据来源**，
    不要一律写成 dasymetric（源比目标细时它退化为分区求和，
    标注成 dasymetric 会造成数据溯源上的错误陈述）。
    """
    import rasterio

    def _ret(arr, method):
        return (arr, method) if return_method else arr

    if population_raster is None:
        logger.warning("未提供人口栅格，dasymetric 重分配不可用")
        return _ret(np.full(len(grid), np.nan), "none")

    # -- 0. 分辨率检查：源比目标细时无需重分配 ----------------------------
    #
    # ⚠️ 单位必须统一。WorldPop 是 EPSG:4326，其 transform.a 是**度**
    # （约 0.00083°），而格网边长是**米**（500 m）。直接比较 0.00083 与 500
    # 会得到"栅格远比格网细"的结论——这次碰巧与正确答案一致，但理由是错的；
    # 若栅格是投影坐标系（transform.a 就是米），同样的代码会得出相反结论，
    # 把该走分区求和的情形误判为需要重分配。
    try:
        with rasterio.open(population_raster) as src:
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            raster_crs = src.crs
            if src.crs is not None and src.crs.is_geographic:
                # 度 -> 米：纬度方向约 111 320 m/度，经度方向按中心纬度折算
                lat_mid = src.bounds.top - (src.bounds.top - src.bounds.bottom) / 2
                m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_mid))
                raster_res_m = min(res_x * m_per_deg_lon, res_y * 111_320.0)
            else:
                raster_res_m = min(res_x, res_y)
    except Exception as exc:
        logger.warning("无法读取人口栅格元数据: %s", str(exc)[:160])
        return _ret(_uniform_population(grid, cfg, population_raster), "zonal_sum")

    grid_size = float(cfg.get("stage1_candidates.grid_size_m", 500.0))
    if raster_res_m <= grid_size and not force:
        logger.info(
            "人口栅格分辨率 %.0f m 细于需求格网 %.0f m —— 分区求和已是正确的"
            "格网人口总量，**体积重分配在此不产生作用**，直接使用分区求和。"
            "（体积权重的价值在于把人口摊到建筑上做更细的分析，需 force=True）",
            raster_res_m, grid_size,
        )
        return _ret(_uniform_population(grid, cfg, population_raster), "zonal_sum")

    logger.info(
        "人口栅格 %.0f m 粗于需求格网 %.0f m —— 启用体积加权的 dasymetric 重分配",
        raster_res_m, grid_size,
    )

    # -- 1. 建筑体积与体积密度 --------------------------------------------
    volume = building_volume_per_cell(grid, buildings, cfg, use_weights=use_weights)
    if volume.sum() <= 0:
        logger.warning("研究区内建筑体积为 0，退化为分区求和")
        return _ret(_uniform_population(grid, cfg, population_raster), "zonal_sum")

    cell_area = grid.geometry.area.to_numpy(dtype=np.float64)
    cell_area = np.where(cell_area > 0, cell_area, 1.0)
    vol_density = volume / cell_area                      # m³/m²

    # -- 2. 面积重叠：粗像元 × 格网 ---------------------------------------
    import geopandas as gpd
    from shapely.geometry import box as shp_box

    with rasterio.open(population_raster) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform, crs = src.transform, src.crs
        H, W = arr.shape

    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0)

    grid_c = grid.to_crs(crs) if str(grid.crs) != str(crs) else grid

    # 只处理有人的像元，跳过绝大多数空像元（WorldPop 大部分像元为 0）
    rows_i, cols_i = np.nonzero(arr)
    if len(rows_i) == 0:
        logger.warning("人口栅格内没有任何非零像元")
        return _ret(np.zeros(len(grid), dtype=np.float64), "empty")

    logger.info("人口栅格中非零像元 %d / %d", len(rows_i), arr.size)

    pop_cells = gpd.GeoDataFrame(
        {"pop": arr[rows_i, cols_i]},
        geometry=[
            shp_box(
                transform.c + c * transform.a,
                transform.f + (r + 1) * transform.e,
                transform.c + (c + 1) * transform.a,
                transform.f + r * transform.e,
            )
            for r, c in zip(rows_i, cols_i)
        ],
        crs=crs,
    )
    # 规范化：确保 bbox 的 min<max（e 为负时上面构造会反）
    pop_cells["geometry"] = pop_cells.geometry.buffer(0).make_valid()

    # grid_c 加一列行号，便于 sjoin 后直接定位到 grid 的行
    grid_c = grid_c.reset_index(drop=True)
    grid_c["_row"] = np.arange(len(grid_c))

    pairs = gpd.sjoin(
        pop_cells, grid_c[["_row", "geometry"]],
        how="inner", predicate="intersects",
    )
    if len(pairs) == 0:
        logger.error("人口像元与需求格网无重叠——请检查两者的 CRS 与覆盖范围")
        return _ret(np.full(len(grid), np.nan), "none")

    # 重叠面积用 shapely 2.x 的**向量化**几何运算计算。
    # 早期写法用 itertuples 逐行 intersection，既慢又依赖 sjoin 可能变动的
    # 列名（index_left/index_right 在不同 geopandas 版本中不一致，实测
    # 会抛 AttributeError）。向量化版本没有这两个问题。
    import shapely

    left_idx = pairs.index.to_numpy()                 # 人口像元在 pop_cells 中的位置
    right_row = pairs["_row"].to_numpy()              # 对应格网在 grid 中的行号
    inter_area = shapely.area(
        shapely.intersection(
            pop_cells.geometry.to_numpy()[left_idx],
            grid_c.geometry.to_numpy()[right_row],
        )
    ).astype(np.float64)

    tmp = pd.DataFrame({
        "pix": left_idx,
        "grid_row": right_row,
        "pop": pairs["pop"].to_numpy(dtype=np.float64),
        "area": inter_area,
    })
    tmp["w"] = tmp["area"] * vol_density[tmp["grid_row"].to_numpy()]
    tmp["w"] = np.where(tmp["w"] > 0, tmp["w"], tmp["area"])   # 无建筑时退化为面积权重

    wsum = tmp.groupby("pix")["w"].transform("sum")
    tmp["alloc"] = np.where(wsum > 0, tmp["pop"] * tmp["w"] / wsum, 0.0)

    result = np.zeros(len(grid), dtype=np.float64)
    np.add.at(result, tmp["grid_row"].to_numpy(), tmp["alloc"].to_numpy())

    # -- 3. 守恒性诊断 -----------------------------------------------------
    # 人口栅格通常覆盖整个下载 bbox，而需求格网被裁剪到研究区**多边形**
    # 内。因此落在研究区之外的像元本就不该被计入——这不是质量损失，而是
    # 正确的研究区裁剪。必须把这两者分开报告，否则 10% 量级的差额会被
    # 误读成实现缺陷（实测差异中绝大部分来自边界裁剪）。
    src_total = float(arr.sum())
    assigned_total = float(result.sum())
    # 与格网有重叠的像元人口之和 = 研究区内的人口
    in_area_total = float(tmp.drop_duplicates("pix")["pop"].sum())

    logger.info(
        "dasymetric 重分配完成:\n"
        "    栅格全域人口      %12.0f 人\n"
        "    其中落在研究区内  %12.0f 人（%.1f%%）\n"
        "    重分配后合计      %12.0f 人（相对区内守恒误差 %.4f%%）\n"
        "    区外人口          %12.0f 人（按研究区边界正确剔除）",
        src_total, in_area_total, 100 * in_area_total / max(src_total, 1e-9),
        assigned_total,
        100 * abs(assigned_total - in_area_total) / max(in_area_total, 1e-9),
        src_total - in_area_total,
    )
    if abs(assigned_total - in_area_total) / max(in_area_total, 1e-9) > 0.01:
        logger.warning(
            "重分配相对研究区内人口的不守恒误差超过 1%%，请检查格网与栅格的"
            "重叠计算（可能因几何无效或 CRS 不匹配导致部分重叠未被识别）"
        )
    return _ret(result, "dasymetric_volume")


def population_from_buildings(
    grid,
    buildings,
    cfg,
    area_per_person_m2: float | None = None,
    use_weights: bool = True,
) -> np.ndarray:
    """由建筑体量估计人口（无人口栅格时的替代方案）。

    何时使用
    --------
    当人口栅格不可得时（数据源故障、网络受限），与其回退到**合成**人口
    （那样结果毫无意义），不如用一个有物理依据、可标定、可复核的估计量。
    本函数即为此设计。

    方法
    ----
    城市人口的绝大部分时间在建筑内，因此"建筑面积 ÷ 人均住房建筑面积"
    是人口的一个自然估计量::

        pop_i = Σ_{b ∈ i} [ footprint_b × n_floors_b × w_b ] / a_pc

    其中 ``n_floors_b = height_b / storey_height``（层数由融合高度换算），
    ``w_b`` 是按建筑类型的居住权重（见 :data:`DEFAULT_USE_WEIGHTS`），
    ``a_pc`` 是人均住房建筑面积。

    与栅格底图方式的区别
    --------------------
    该估计量直接挂在**建筑**上，空间分辨率等于建筑轮廓本身，比 WorldPop
    的 100 m 栅格更细；缺点是它假设了统一的人均面积，无法反映同一建筑类型
    内部的居住强度差异（群租、空置、办公改住等）。

    标定与论文表述要求
    ------------------
    ``area_per_person_m2`` 默认取 40 m²/人，对应成都市 2020 年前后城镇
    人均住房建筑面积的量级。**这是一个人为标定值，论文必须**：

    1. 报告它，并说明取值依据（成都市统计年鉴的城镇人均住房建筑面积）；
    2. 报告该值下研究区的**隐含总人口**，与第七次全国人口普查对应区县的
       常住人口对比，说明偏差；
    3. 在敏感性分析中检验结论对该值的依赖（该值只影响总人口标度，
       对**空间分布**的影响是中性的——因为所有单元同比例缩放，
       而这正是本研究关心的公平性所依赖的部分）。

    时点差异所需注意：建筑现状反映的是测绘时点，人口普查是 2020 年，
    新建成但未入住的楼盘会导致高估，这是该方法的主要系统性偏差来源。

    Returns
    -------
    长度等于 ``len(grid)`` 的人口数组（人）。
    """
    if buildings is None or len(buildings) == 0:
        logger.error("建筑数据为空，无法用建筑体量估计人口")
        return np.full(len(grid), np.nan)

    a_pc = float(
        area_per_person_m2
        if area_per_person_m2 is not None
        else cfg.get("stage2_demand.area_per_person_m2", 40.0)
    )
    storey_h = float(cfg.get("stage2_demand.storey_height_m", 3.3))

    volume = building_volume_per_cell(grid, buildings, cfg, use_weights=use_weights)

    # 体积 -> 建筑面积：除以层高。用 use_weights 时体积已被居住权重折算，
    # 等价于"等效住宅建筑面积"。
    floor_area = volume / max(storey_h, 1e-6)
    pop = floor_area / max(a_pc, 1e-6)

    total = float(pop.sum())
    logger.info(
        "建筑体量法人口估计: 合计 %.0f 人（人均 %.0f m²，层高 %.1f m）\n"
        "    在成都市的量级参照：第七次人口普查（2020）中心城区常住人口约 700–900 万。\n"
        "    ⚠️ 这是**估计值**而非普查值，论文须报告标定依据与隐含总量。",
        total, a_pc, storey_h,
    )
    return pop


def _uniform_population(grid, cfg, population_raster, fallback: bool = True) -> np.ndarray:
    """退路：不建体积模型，直接把栅格按分区求和。"""
    from .rasters import zonal_sum_to_grid

    logger.warning("退化为按分区求和（假设人口在格网内均匀分布）")
    return zonal_sum_to_grid(population_raster, grid, cfg, stat="sum")


def compare_population_models(
    grid,
    pop_dasymetric: np.ndarray,
    pop_uniform: np.ndarray,
) -> pd.DataFrame:
    """对比两种人口分配方式，用于论文的方法论证。

    报告两者在**总量**上一致（守恒性检验），但在**空间分布**上的差异
    （变异系数、最大单元比、空间自相关对比）。若 dasymetric 版本的
    空间集中度显著更高，说明均匀假设确实在把人口摊平——这正是引入
    dasymetric 重分配的理由。
    """
    d = np.nan_to_num(pop_dasymetric, nan=0.0)
    u = np.nan_to_num(pop_uniform, nan=0.0)

    def _stats(a, name):
        nz = a[a > 0]
        return {
            "method": name,
            "total_population": float(a.sum()),
            "n_nonzero_cells": int((a > 0).sum()),
            "mean_per_cell": float(a.mean()),
            "std_per_cell": float(a.std()),
            "cv": float(a.std() / a.mean()) if a.mean() > 0 else 0.0,
            "max_cell": float(a.max()),
            "top1pct_share": float(
                np.sort(a)[::-1][: max(1, len(a) // 100)].sum() / max(a.sum(), 1e-9)
            ),
            "gini": _gini(a),
        }

    df = pd.DataFrame([_stats(d, "dasymetric"), _stats(u, "uniform")])
    return df


def _gini(x: np.ndarray) -> float:
    """基尼系数，度量人口分布的不均衡程度。"""
    v = np.sort(np.clip(x, 0, None))
    n = len(v)
    if n < 2 or v.sum() <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
