"""三维可视化数据导出。

把四阶段流水线的结果导出成一份自包含的 JSON 数据包，供
``viz3d/chengdu_evtol_3d.html``（CesiumJS 三维地球）加载。

为什么用 JSON 数据包而不是让网页直接读 GeoJSON 文件
------------------------------------------------------
三维页面要能以 ``file://`` 协议直接双击打开（评审人/答辩现场不会去
起本地服务器）。``file://`` 下浏览器会因同源策略拒绝 ``fetch()`` 本地
文件。把数据内联成 ``<script src="data.js">`` 形式的 JS 变量，
或由页面以 ``<script>`` 标签加载，就能绕开这一限制且不需要服务器。

坐标系
------
Cesium 使用 WGS84 经纬度（EPSG:4326）。流水线内部一律用投影坐标
（EPSG:32648）计算，因此导出时**必须**转换回经纬度——这是最容易
出错的一步：直接把 UTM 坐标当经纬度喂给 Cesium，物体会出现在
地球外的随机位置。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_lonlat(gdf_or_xy, cfg, crs=None):
    """把投影坐标点转换为经纬度。

    Parameters
    ----------
    gdf_or_xy
        ``(N, 2)`` 数组或含 ``x`` / ``y`` 列的 DataFrame。
    """
    import geopandas as gpd

    if hasattr(gdf_or_xy, "columns"):
        x = gdf_or_xy["x"].to_numpy(dtype=np.float64)
        y = gdf_or_xy["y"].to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(gdf_or_xy, dtype=np.float64)
        x, y = arr[:, 0], arr[:, 1]

    gs = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=crs or cfg.crs_proj).to_crs(cfg.crs_geo)
    return np.asarray(gs.x), np.asarray(gs.y)


def _round_list(arr, ndigits: int = 6) -> list[float]:
    """转成 Python float 列表并保留小数位（控制文件体积）。"""
    return [round(float(v), ndigits) for v in np.asarray(arr)]


def build_bundle(
    cfg,
    candidates=None,
    demand=None,
    ip_solutions=None,
    nsga2_result=None,
    selected_solution=None,
    max_demand_points: int = 6000,
    max_candidate_points: int = 4000,
) -> dict:
    """组装三维可视化的数据包。

    Parameters
    ----------
    candidates
        :class:`evtol_siting.stage1_candidates.CandidateSet`。
    demand
        :class:`evtol_siting.stage2_demand.DemandModel`。
    ip_solutions
        :class:`evtol_siting.stage3_ip.IPSolution` 列表。
    nsga2_result
        NSGA-II 结果（含帕累托前沿）。
    selected_solution
        要在三维视图中高亮的方案：``{"label": str, "cand_ids": [int, ...]}``。
        通常传入 IP 最优解或 NSGA-II 拐点解。
    max_demand_points / max_candidate_points
        点数上限。Cesium 逐实体渲染的帧率对实体数敏感，超过约 1 万会明显
        卡顿。超出时按需求量 / 质量分降序抽样，保留最重要的部分。

    Returns
    -------
    可直接 ``json.dumps`` 的嵌套字典。
    """
    from .geo import study_area_polygon

    bundle: dict = {
        "meta": {
            "city": cfg.get("project.case_city"),
            "crs_source": cfg.crs_proj,
            "crs_target": cfg.crs_geo,
            "note": "由 evtol_siting.viz3d.build_bundle 导出",
        },
        "study_area": None,
        "candidates": {"rooftop": [], "ground": []},
        "demand": [],
        "clusters": [],
        "solutions": [],
        "pareto": None,
    }

    # -- 研究区边界 --------------------------------------------------------
    try:
        poly = study_area_polygon(cfg)
        geom = poly.__geo_interface__
        bundle["study_area"] = geom
        logger.info("研究区边界已导出")
    except Exception as exc:
        logger.warning("研究区边界导出失败: %s", str(exc)[:160])

    # -- 候选起降场 --------------------------------------------------------
    if candidates is not None and len(candidates):
        g = candidates.gdf
        lon, lat = _to_lonlat(g, cfg)
        n_cand = len(g)

        # 抽样（若超出上限）——按 quality 降序，保留最重要的候选
        keep = np.arange(n_cand)
        if n_cand > max_candidate_points:
            order = np.argsort(-g["quality"].to_numpy())
            keep = np.sort(order[:max_candidate_points])
            logger.info("候选点 %d -> 抽样 %d 个用于三维展示", n_cand, len(keep))

        for i in keep:
            row = g.iloc[i]
            item = {
                "id": int(row.get("cand_id", i)),
                "lon": round(float(lon[i]), 6),
                "lat": round(float(lat[i]), 6),
                "cost": float(row.get("cost", 0.0)),
                "capacity": float(row.get("capacity", 0.0)),
                "quality": round(float(row.get("quality", 0.0)), 4),
            }
            if row["facility_type"] == "rooftop":
                item["height_m"] = round(float(row.get("height_m", 0.0)), 1)
                item["roof_m2"] = round(float(row.get("roof_usable_m2", 0.0)), 0)
                item["height_source"] = str(row.get("height_source", "?"))
                bundle["candidates"]["rooftop"].append(item)
            else:
                bundle["candidates"]["ground"].append(item)

        logger.info(
            "候选点已导出: 屋顶型 %d，地面型 %d",
            len(bundle["candidates"]["rooftop"]), len(bundle["candidates"]["ground"]),
        )

    # -- 需求栅格与人群类 --------------------------------------------------
    if demand is not None and len(demand.grid):
        d = demand.grid
        lon, lat = _to_lonlat(d.rename(columns={"centroid_x": "x", "centroid_y": "y"}), cfg)

        vals = d["demand"].to_numpy(dtype=np.float64)
        keep = np.arange(len(d))
        if len(d) > max_demand_points:
            order = np.argsort(-vals)
            keep = np.sort(order[:max_demand_points])
            logger.info("需求单元 %d -> 抽样 %d 个用于三维展示", len(d), len(keep))

        for i in keep:
            row = d.iloc[i]
            bundle["demand"].append({
                "lon": round(float(lon[i]), 6),
                "lat": round(float(lat[i]), 6),
                "demand": round(float(vals[i]), 2),
                "pop": round(float(row.get("population", 0.0)), 0),
                "cluster": int(row.get("cluster", -1)),
            })

        prof = demand.cluster_profile
        for _, r in prof.iterrows():
            bundle["clusters"].append({
                "cluster": int(r["cluster"]),
                "n_cells": int(r["n_cells"]),
                "population": round(float(r["population"]), 0),
                "trip_rate": round(float(r["trip_rate"]), 6),
                # 供三维视图按人群着色与图例显示（与论文图 5 同一配色）
                "monthly_income_cny": round(float(r.get("monthly_income_cny", 0.0)), 0),
                "p_evtol": round(float(r.get("p_evtol", 0.0)), 6),
                "demand_trips_per_day": round(
                    float(r.get("demand_trips_per_day", 0.0)), 1),
            })
        logger.info("需求单元 %d 个、人群类 %d 个已导出", len(bundle["demand"]), len(bundle["clusters"]))

    # -- 选址方案 ----------------------------------------------------------
    if ip_solutions:
        for s in ip_solutions:
            if getattr(s, "status", "") != "Optimal" or not s.selected:
                continue
            bundle["solutions"].append({
                "label": f"IP · {s.name}",
                "cand_ids": [int(i) for i in s.selected],
                "n_sites": int(s.n_sites),
                "total_cost": float(s.total_cost),
                "coverage": {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in s.coverage.items()},
            })

    if nsga2_result is not None and len(getattr(nsga2_result, "pareto_F", [])):
        F = np.asarray(nsga2_result.pareto_F, dtype=np.float64)
        X = np.asarray(nsga2_result.pareto_X, dtype=np.float64)
        keys = ["unserved_demand", "total_access_time", "inequity", "total_cost"]
        bundle["pareto"] = {
            "keys": keys[: F.shape[1]],
            "points": [[round(float(v), 4) for v in row] for row in F],
            "solutions": [
                {"row": int(i), "cand_ids": [int(j) for j in np.flatnonzero(x > 0.5)]}
                for i, x in enumerate(X)
            ],
        }
        logger.info("帕累托前沿 %d 个解已导出", len(F))

    if selected_solution:
        bundle["solutions"].insert(0, {
            "label": selected_solution.get("label", "selected"),
            "cand_ids": [int(i) for i in selected_solution.get("cand_ids", [])],
            "n_sites": len(selected_solution.get("cand_ids", [])),
            "total_cost": float(selected_solution.get("total_cost", 0.0)),
            "coverage": selected_solution.get("coverage", {}),
            "highlight": True,
        })

    return bundle


def export_bundle(bundle: dict, out_js: Path, var_name: str = "EVTOL_DATA") -> Path:
    """把数据包写成 JS 文件（``window.<var_name> = {...}``）。

    写成 ``.js`` 而非 ``.json`` 是为了让页面能用 ``<script src="...">``
    加载——``file://`` 下 ``fetch()`` 会被同源策略拒绝，而 ``<script>``
    标签不受此限制。这样三维页面双击即可打开，无需本地服务器。
    """
    out_js = Path(out_js)
    out_js.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    out_js.write_text(
        f"// 由 evtol_siting.viz3d.export_bundle 自动生成 —— 请勿手工编辑\n"
        f"window.{var_name} = {payload};\n",
        encoding="utf-8",
    )
    size_mb = out_js.stat().st_size / 1e6
    logger.info("三维数据包已导出: %s (%.2f MB)", out_js, size_mb)
    if size_mb > 20:
        logger.warning(
            "数据包 %.1f MB 偏大，浏览器加载会变慢。"
            "可调低 build_bundle 的 max_candidate_points / max_demand_points。",
            size_mb,
        )
    return out_js
