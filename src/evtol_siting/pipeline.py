"""四阶段流水线总装与结果持久化。

本模块把四个阶段串成可重复执行的管道，并在每一步之间落盘中间结果
（``outputs/data/`` 与 ``data/interim/``）。这样做的理由：

* **可调试**——某一步出错时无需从头重跑（数据下载可能耗时 30 分钟）；
* **可复现**——中间结果与代码版本、配置一起构成完整的实验记录；
* **可审计**——论文审稿人若要求补充材料，中间产物即为证据链。

用法::

    from evtol_siting.pipeline import Pipeline
    from evtol_siting.config import load_config

    cfg = load_config()
    pipe = Pipeline(cfg, data_mode="auto")   # auto | real | synthetic
    results = pipe.run_all()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .geo import distance_matrix, make_grid, travel_time_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class PipelineResults:
    """流水线各阶段产物的集合。"""
    cfg: Config
    data_mode: str = "unknown"

    grid: pd.DataFrame | None = None
    buildings: object | None = None
    height_report: pd.DataFrame | None = None
    candidates: object | None = None
    demand: object | None = None
    access_time_s: np.ndarray | None = None
    ip_solutions: list = field(default_factory=list)
    nsga2_result: object | None = None
    baselines: list = field(default_factory=list)
    solution_metrics: pd.DataFrame | None = None
    diagnostics: dict = field(default_factory=dict)

    def save(self, out_dir: Path) -> None:
        """把结果落盘（表格 CSV、数组 NPZ、诊断 JSON）。

        调用方传入的就是目标目录本身，此处**不再猜测是否要拼一层 ``data/``**
        ——那个猜测与 ``Pipeline.save`` 取 ``.parent`` 的写法叠加后，会把落盘
        位置钉死在 ``outputs/data``，使数据目录的覆盖失效。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        d = out_dir

        if self.grid is not None:
            _to_csv(self.grid, d / "grid.csv")
        if self.height_report is not None:
            self.height_report.to_csv(d / "table_height_sources.csv", index=False)
        if self.candidates is not None:
            cand = self.candidates.gdf.drop(columns=["geometry"], errors="ignore")
            _to_csv(cand, d / "candidates.csv")
        if self.demand is not None:
            self.demand.grid.to_csv(d / "demand_grid.csv", index=False)
            self.demand.cluster_profile.to_csv(d / "table_clusters.csv", index=False)
        if self.access_time_s is not None:
            np.savez_compressed(d / "access_time_s.npz", at=self.access_time_s)
        if self.ip_solutions:
            pd.DataFrame([s.to_dict() for s in self.ip_solutions]).to_csv(
                d / "table_ip_scenarios.csv", index=False
            )
        if self.nsga2_result is not None:
            from .stage4_nsga2 import objective_table

            objective_table(self.nsga2_result.pareto_F, self.cfg).to_csv(
                d / "pareto_front.csv", index=False
            )
            pd.DataFrame(
                {"generation": np.arange(len(self.nsga2_result.history.get("hypervolume", []))),
                 "hypervolume": self.nsga2_result.history.get("hypervolume", []),
                 "n_pareto": self.nsga2_result.history.get("n_pareto", [])}
            ).to_csv(d / "convergence.csv", index=False)
        if self.solution_metrics is not None:
            self.solution_metrics.to_csv(d / "table_solution_metrics.csv", index=False)
        if self.baselines:
            pd.DataFrame(
                [
                    {"name": b.name, "status": b.status, "n_sites": b.n_sites,
                     "objective": b.objective, "solve_time_s": b.solve_time_s}
                    for b in self.baselines
                ]
            ).to_csv(d / "table_baselines.csv", index=False)

        (d / "diagnostics.json").write_text(
            json.dumps(_jsonable(self.diagnostics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("结果已保存到 %s", d)


def _to_csv(df: pd.DataFrame, path: Path) -> None:
    """写 CSV，遇到几何/列表列时先转成字符串。"""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
    out.to_csv(path, index=False)


def _jsonable(obj, depth: int = 0):
    """把 numpy / pandas 类型递归转成可 JSON 序列化的结构。"""
    if depth > 6:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist(), depth + 1)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------

class Pipeline:
    """四阶段选址流水线。

    Parameters
    ----------
    cfg
        配置对象。
    data_mode
        * ``"real"``      —— 强制使用真实数据，缺失即报错；
        * ``"synthetic"`` —— 强制使用合成数据（离线测试）；
        * ``"auto"``      —— 优先真实数据，失败时回退合成并**显著告警**。
    limit_overpass_tiles
        限制 OSM 下载瓦片数，用于快速试跑。``None`` 表示全量。
    """

    def __init__(
        self,
        cfg: Config,
        data_mode: str = "auto",
        limit_overpass_tiles: int | None = None,
        cache: dict | None = None,
    ):
        self.cfg = cfg
        self.data_mode = data_mode
        self.limit_overpass_tiles = limit_overpass_tiles

        # 缓存分两层，**不可混用**：
        #   shared_cache —— 跨实例共享，**只放原始数据**（OSM 图层、栅格路径）。
        #                   这些与配置无关，复用安全，是敏感性分析省时的关键。
        #   self.cache   —— 实例内，放**派生结果**（候选集、需求、接驳时间矩阵）。
        #                   它们依赖配置，跨参数点复用会得到错误结果。
        #
        # 早期版本只有一层缓存并被跨参数点共享，导致切换参数后
        # compute_access_time 命中旧键、返回上一个参数点的接驳时间矩阵，
        # 报出"access_time 形状与 (n_demand, n_cand) 不匹配"。这类错误在
        # 矩阵形状恰好相同的情况下**不会报错**，只会静默给出错误结果。
        self.shared_cache: dict = cache if cache is not None else {}
        self.cache: dict = {}
        cfg.seed_everything()
        self.res = PipelineResults(cfg=cfg, data_mode=data_mode)

    # -- 阶段一前置：数据 --------------------------------------------------

    def load_data(self) -> dict:
        """载入（或合成）阶段一所需的全部输入数据。"""
        if "data" in self.shared_cache:
            return self.shared_cache["data"]

        from .data import synthetic as syn

        mode = self.data_mode
        data: dict = {}
        used_synthetic = False

        if mode in ("auto", "real"):
            try:
                data = self._load_real_data()
            except Exception as exc:
                if mode == "real":
                    raise
                logger.warning(
                    "真实数据加载失败（%s），回退到**合成数据**。"
                    "合成结果不可用于论文实证结论。",
                    str(exc)[:200],
                )
                used_synthetic = True
        if mode == "synthetic" or (mode == "auto" and not data):
            used_synthetic = True

        if used_synthetic:
            logger.warning("=" * 70)
            logger.warning("  正在使用 SYNTHETIC 合成数据 —— 结果不可用于论文")
            logger.warning("=" * 70)
            data = {
                "buildings": syn.synthetic_buildings(self.cfg),
                "poi": syn.synthetic_poi(self.cfg),
                "roads": syn.synthetic_roads(self.cfg),
                "landuse": syn.synthetic_landuse(self.cfg),
                "water": None,
                "airports": syn.synthetic_airports(self.cfg),
                "population_raster": None,
                "income_points": None,
                "slope_raster": None,
            }
            self.res.data_mode = "synthetic"
        else:
            self.res.data_mode = "real"

        self.shared_cache["data"] = data
        return data

    def _load_real_data(self) -> dict:
        """从 OSM 与公开栅格数据源载入真实数据。

        两条 OSM 通道，由 ``data.sources.osm_backend`` 选择：

        * ``"pbf"``（默认，推荐）—— 一次下载省级提取包后本地解析。实测
          四川省包 116 MB / 18 秒下载 + 41 秒解析；无速率限制。
        * ``"overpass"`` —— 按瓦片调用 Overpass API。在受限网络下会被
          强限流（首个请求 200、后续全部 504），300 个瓦片需 8 小时以上。

        栅格数据（CNBH 建筑高度、人口、DEM、坡度的下载与失败降级见下方）。
        """
        from .data import rasters
        from .data.heights import estimate_roof_area, fuse_building_heights

        cfg, lim = self.cfg, self.limit_overpass_tiles
        backend = str(cfg.get("data.sources.osm_backend", "pbf")).lower()

        needed = ["buildings", "poi", "roads", "landuse", "water", "airports"]

        if backend == "pbf":
            from .data import pbf

            pbf_path = cfg.get("data.pbf_file")
            pbf_path = (
                Path(pbf_path) if pbf_path and Path(pbf_path).is_absolute()
                else pbf.default_pbf_path(cfg)
            )
            if not pbf_path.exists():
                logger.info("未找到 PBF 提取包，开始下载…")
                got = pbf.fetch_pbf(
                    cfg,
                    province=str(cfg.get("data.pbf_province", "sichuan")),
                    source=str(cfg.get("data.pbf_source", "osmfr")),
                )
                if got is None:
                    raise RuntimeError(
                        f"PBF 下载失败。可手动下载 "
                        f"{cfg.get('data.pbf_province', 'sichuan')} 提取包放入 "
                        f"{pbf_path.parent}，或改用 Overpass 通道"
                        "（data.sources.osm_backend: 'overpass'）"
                    )
                pbf_path = got

            logger.info("解析 PBF 提取包: %s", pbf_path.name)
            layers = pbf.parse_all_layers(pbf_path, cfg, layers=needed)
            buildings = layers["buildings"]
            poi = layers["poi"]
            roads = layers["roads"]
            landuse = layers["landuse"]
            water = layers["water"]
            airports = layers["airports"]
        else:
            from .data import osm

            logger.info("通过 Overpass API 分块下载 OSM 数据（可能很慢）…")
            buildings = osm.fetch_geodataframe("buildings", cfg, limit_tiles=lim)
            poi = osm.fetch_geodataframe("poi", cfg, limit_tiles=lim)
            roads = osm.fetch_geodataframe("roads", cfg, limit_tiles=lim)
            landuse = osm.fetch_geodataframe("landuse", cfg, limit_tiles=lim)
            water = osm.fetch_geodataframe("water", cfg, limit_tiles=lim)
            airports = osm.fetch_geodataframe("airports", cfg, limit_tiles=lim)

        # 建筑高度栅格（CNBH-10m）——高度融合的第 3 级数据来源。
        # 下载失败不是致命错误：融合会退到第 4 级的统计填补，但精度下降，
        # 因此这里显式告警而不是静默跳过。
        height_raster = None
        if str(cfg.get("data.sources.building_height_raster", "cnbh10m")).lower() not in ("none", ""):
            try:
                height_raster = rasters.fetch_building_height_raster(cfg)
            except Exception as exc:
                logger.warning("建筑高度栅格获取失败: %s", str(exc)[:200])
        if height_raster is None:
            logger.warning(
                "**无建筑高度栅格**：高度融合将主要依赖 OSM 标签与统计填补。"
                "鉴于 OSM 高度标签覆盖率仅 1.5–5.4%%，候选集可能有系统偏差。"
            )
        self.cache["height_raster"] = height_raster

        # 建筑高度融合（真实数据的核心处理步骤）
        buildings = fuse_building_heights(buildings, cfg, height_raster=height_raster)
        buildings = estimate_roof_area(
            buildings, float(cfg.get("stage1_candidates.rooftop.usable_roof_ratio", 0.70))
        )
        from .data.heights import height_quality_report

        self.res.height_report = height_quality_report(buildings)

        # 栅格数据（失败不致命，回退到合成特征）
        pop = inc = dem = slope = None
        try:
            pop = rasters.fetch_population(cfg)
        except Exception as exc:
            logger.warning("人口数据失败: %s", str(exc)[:160])
        try:
            inc = rasters.fetch_income(cfg)
            # fetch_income 返回的是**文件路径**，而 build_features 需要点矢量本身。
            # 这一步此前缺失：POI 策略返回 None，所以从未暴露；但任何返回路径的
            # 策略（viirs / local_file）都会在 build_features 里抛
            # ``TypeError: object of type 'WindowsPath' has no len()``。
            # 载入后统一为 GeoDataFrame，后续调用方不必再关心来源是文件还是内存。
            if inc is not None and not hasattr(inc, "geometry"):
                import geopandas as gpd

                inc = gpd.read_file(inc)
                logger.info("收入代理已载入: %d 个点（%s）",
                            len(inc), str(getattr(inc, "crs", "?")))
        except Exception as exc:
            logger.warning("收入代理数据失败: %s", str(exc)[:160])
        try:
            dem = rasters.fetch_dem(cfg)
            if dem:
                # 坡度文件名须带城市标识：早期写成共用的 `slope.tif`，多城市下
                # 第二个城市会**命中第一个城市留下的坡度栅格**（compute_slope
                # 见文件存在即跳过），于是候选筛选用的是另一个城市的坡度判定。
                slope = rasters.compute_slope(
                    dem,
                    cfg.dir("data.interim_dir") / f"{rasters.city_slug(cfg)}_slope.tif",
                )
        except Exception as exc:
            logger.warning("DEM/坡度失败: %s", str(exc)[:160])

        return {
            "buildings": buildings, "poi": poi, "roads": roads,
            "landuse": landuse, "water": water, "airports": airports,
            "population_raster": pop, "income_points": inc, "slope_raster": slope,
        }

    # -- 阶段一 ------------------------------------------------------------

    def run_stage1(self) -> "object":
        """候选起降场筛选。"""
        from .stage1_candidates import build_candidate_set
        from .data import heights as H

        logger.info("=" * 60)
        logger.info("阶段一：基于 GIS 的候选起降场筛选")
        logger.info("=" * 60)

        data = self.load_data()
        grid = make_grid(self.cfg)
        self.res.grid = grid

        buildings = data.get("buildings")
        if buildings is not None and "height_m" not in getattr(buildings, "columns", []):
            buildings = H.fuse_building_heights(buildings, self.cfg)
            buildings = H.estimate_roof_area(
                buildings, float(self.cfg.get("stage1_candidates.rooftop.usable_roof_ratio", 0.70))
            )
            self.res.height_report = H.height_quality_report(buildings)
        self.res.buildings = buildings

        cs = build_candidate_set(
            self.cfg,
            buildings=buildings,
            grid=grid,
            landuse=data.get("landuse"),
            water=data.get("water"),
            slope_raster=data.get("slope_raster"),
            airports=data.get("airports"),
            roads=data.get("roads"),
        )
        self.res.candidates = cs
        self.res.diagnostics["stage1"] = {
            "n_candidates": len(cs),
            "n_rooftop": int((cs.gdf["facility_type"] == "rooftop").sum()),
            "n_ground": int((cs.gdf["facility_type"] == "ground").sum()),
            "mean_cost_cny": float(cs.gdf["cost"].mean()),
            "candidate_summary": cs.summary().to_dict(orient="records"),
        }
        return cs

    # -- 阶段二 ------------------------------------------------------------

    def run_stage2(self) -> "object":
        """K-means 人群分类与需求空间化。"""
        from .stage2_demand import build_features, cluster_demand

        logger.info("=" * 60)
        logger.info("阶段二：K-means 人群分类与需求空间化")
        logger.info("=" * 60)

        if self.res.grid is None:
            self.run_stage1()
        data = self.load_data()

        feats = build_features(
            self.res.grid, self.cfg,
            income_points=data.get("income_points"),
            population_raster=data.get("population_raster"),
            poi=data.get("poi"),
            buildings=self.res.buildings,
        )
        # 记录 dasymetric 对比结果（若已计算），供论文 Table 使用
        if "population_comparison" in getattr(feats, "attrs", {}):
            self.res.diagnostics["population_comparison"] = (
                feats.attrs["population_comparison"].to_dict(orient="records")
            )

        # 任一关键特征整体缺失 -> 合成
        # 只有当**人口**与**收入/POI 特征**同时缺失时才退到合成数据。
        # 人口单独缺失不再触发合成：阶段二会改用建筑体量估计
        # （见 stage2_demand.build_features 与 data/dasymetric.py）。
        pop_missing = feats["population"].isna().all()
        feat_missing = (
            feats["income"].isna().all() and float(feats["poi_density"].sum()) == 0.0
        )
        need_synth = pop_missing and feat_missing
        if pop_missing and not feat_missing:
            logger.warning(
                "人口列整体缺失，但收入/POI 特征可用 —— 使用建筑体量估计人口，"
                "**不退化为合成数据**（详见 data/dasymetric.py 的标定说明）"
            )
        if need_synth:
            if not self.cfg.get("data.allow_synthetic_fallback", True):
                raise RuntimeError("需求特征全部缺失且未启用合成回退")
            from .stage2_demand import synthesise_features

            feats = synthesise_features(self.res.grid, self.cfg)
            self.res.data_mode = "synthetic" if self.res.data_mode != "real" else "mixed"

        dm = cluster_demand(self.res.grid, feats, self.cfg)
        self.res.demand = dm
        self.res.diagnostics["stage2"] = {
            "n_clusters": dm.n_clusters,
            "total_population": float(dm.grid["population"].sum()),
            "total_demand_per_day": float(dm.grid["demand"].sum()),
            "cluster_profile": dm.cluster_profile.to_dict(orient="records"),
        }
        return dm

    # -- 阶段三 ------------------------------------------------------------

    def compute_access_time(self, radius_m: float | None = None) -> np.ndarray:
        """计算需求单元 -> 候选起降场的接驳时间矩阵（秒）。

        ``access.time_basis`` 决定口径：

        * ``"euclidean"``（默认）—— 直线距离 / 方式速度。快，但会低估
          被河流、铁路、封闭园区分割的城市里的实际绕行。
        * ``"network"``     —— 基于 OSM 路网的最短路。慢（分钟级），
          但是高影响因子期刊的通行要求。见 :mod:`data.network`。

        ``radius_m`` 只影响缓存键（时间矩阵本身由 ``access.modes`` 的最大
        接驳距离决定）；保留该参数是为了让情景调用签名统一。
        """
        if self.res.demand is None:
            self.run_stage2()
        grid = self.res.demand.grid
        cand = self.res.candidates.gdf

        basis = str(self.cfg.get("access.time_basis", "euclidean")).lower()
        # 键里必须含候选集规模：半径相同但候选集不同的两次运行（敏感性分析
        # 的常见情形）否则会共用同一个矩阵。
        key = f"at_{basis}_{radius_m}_{len(cand)}"
        if key in self.cache:
            return self.cache[key]

        dem_xy = grid[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        cand_xy = cand[["x", "y"]].to_numpy(dtype=np.float64)

        if basis == "network":
            from .data.network import build_road_graph, network_travel_time_matrix

            roads = self.cache.get("roads")
            if roads is None:
                roads = self.load_data().get("roads")
            if roads is None or len(roads) == 0:
                logger.warning("无路网数据，回退到欧氏口径")
                at = travel_time_matrix(dem_xy, cand_xy, self.cfg)
            else:
                G = self.cache.get("road_graph")
                if G is None:
                    G = build_road_graph(roads, self.cfg)
                    self.cache["road_graph"] = G
                at = network_travel_time_matrix(
                    dem_xy, cand_xy, G, self.cfg,
                    cutoff_m=float(self.cfg.get("access.network_cutoff_m", 6000.0)),
                    snap_m=float(self.cfg.get("access.network_snap_m", 1200.0)),
                )
        else:
            at = travel_time_matrix(dem_xy, cand_xy, self.cfg)

        self.cache[key] = at
        return at

    def run_stage3(self) -> list:
        """0-1 整数规划，求解两个情景方案。"""
        from .stage3_ip import solve_scenarios

        logger.info("=" * 60)
        logger.info("阶段三：0-1 整数规划选址")
        logger.info("=" * 60)

        if self.res.candidates is None:
            self.run_stage1()
        if self.res.demand is None:
            self.run_stage2()

        # 主接驳时间矩阵（默认半径），供阶段四使用
        self.res.access_time_s = self.compute_access_time(None)

        sols = solve_scenarios(
            self.res.demand.grid, self.res.candidates.gdf,
            self.compute_access_time, self.cfg,
        )
        self.res.ip_solutions = sols
        self.res.diagnostics["stage3"] = {
            "scenarios": [s.to_dict() for s in sols],
        }
        return sols

    # -- 阶段四 ------------------------------------------------------------

    def run_stage4(self, n_gen: int | None = None, pop_size: int | None = None,
                   validate: bool = False, verbose: bool = True):
        """NSGA-II 多目标优化。"""
        from .nsga2_core import nsga2
        from .stage4_nsga2 import build_problem, make_evaluator, make_repair

        logger.info("=" * 60)
        logger.info("阶段四：NSGA-II 多目标优化")
        logger.info("=" * 60)

        if self.res.access_time_s is None:
            self.run_stage3()

        prob = build_problem(
            self.res.demand.grid, self.res.candidates.gdf, self.res.access_time_s, self.cfg
        )
        evaluate = make_evaluator(prob)
        repair = make_repair(prob) if self.cfg.get("stage4_nsga2.repair", True) else None
        # 剪枝作为**局部搜索算子**按概率作用于子代（见 make_pruner 的说明：
        # 放进解码器会让稠密解被瞬间剪回最小覆盖集，高站点数区段无法探索）。
        pruner = None
        if self.cfg.get("stage4_nsga2.local_search", True):
            from .stage4_nsga2 import make_pruner

            pruner = make_pruner(prob)

        # 用 IP 最优解作为初始种子的种子，加速收敛
        x0 = None
        for s in self.res.ip_solutions:
            if getattr(s, "status", "") == "Optimal" and s.selected:
                v = np.zeros(prob.n_cand)
                v[[int(i) for i in s.selected]] = 1.0
                x0 = v[None, :] if x0 is None else np.vstack([x0, v])
        if x0 is not None:
            logger.info("已注入 %d 个 IP 最优解作为初始种群种子", len(x0))

        # 初始站点规模锚定到 IP 解的量级：IP 给出的是**可行且近最优**的
        # 规模，用它（乘 2 留出探索余量）作为 NSGA-II 的出发点，比按
        # sqrt(n_cand) 的通用启发式更贴合本问题。
        init_sites = None
        ip_sizes = [s.n_sites for s in self.res.ip_solutions
                    if getattr(s, "status", "") == "Optimal" and s.n_sites > 0]
        if ip_sizes:
            init_sites = int(max(8, min(ip_sizes) * 2))

        res = nsga2(
            evaluate, n_var=prob.n_cand, cfg=self.cfg,
            pop_size=pop_size, n_gen=n_gen, repair=repair, x0=x0,
            verbose=verbose, init_sites=init_sites,
            local_search=pruner,
            local_search_prob=float(self.cfg.get("stage4_nsga2.local_search_prob", 0.15)),
        )
        self.res.nsga2_result = res
        self.res.diagnostics["stage4"] = {
            "n_pareto": res.n_pareto,
            "n_evaluations": res.history.get("n_evaluations"),
            "final_hypervolume": (
                float(res.history["hypervolume"][-1]) if res.history.get("hypervolume") else None
            ),
            "memo_size": len(getattr(evaluate, "memo", {})),
        }

        if validate:
            self._validate_against_pymoo(prob, res)

        return res

    def _validate_against_pymoo(self, prob, res, n_gen: int = 100) -> None:
        """用 pymoo 的 NSGA-II 在相同算例上跑一遍，交叉验证自实现。"""
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
        except ImportError:
            logger.warning("pymoo 不可用，跳过交叉验证")
            return

        from .metrics import hypervolume
        from .stage4_nsga2 import problem_for_pymoo

        p = problem_for_pymoo(prob)
        if p is None:
            return
        try:
            out = minimize(
                p, NSGA2(pop_size=len(res.X)), ("n_gen", n_gen),
                seed=self.cfg.seed, verbose=False,
            )
            ref = np.vstack([res.pareto_F, out.F]).max(axis=0) * 1.1
            hv_self = hypervolume(res.pareto_F, ref)
            hv_pymoo = hypervolume(out.F, ref)
            logger.info(
                "交叉验证: 自实现 HV=%.4g vs pymoo HV=%.4g（比值 %.3f）",
                hv_self, hv_pymoo, hv_self / max(hv_pymoo, 1e-12),
            )
            self.res.diagnostics["validation"] = {
                "hv_self_implemented": float(hv_self),
                "hv_pymoo": float(hv_pymoo),
                "ratio": float(hv_self / max(hv_pymoo, 1e-12)),
                "note": "比值应接近 1；显著偏离说明自实现存在缺陷",
            }
        except Exception as exc:
            logger.warning("pymoo 交叉验证失败: %s", str(exc)[:200])

    # -- 评价 --------------------------------------------------------------

    def run_evaluation(self, run_baselines: bool = True) -> pd.DataFrame:
        """计算各方案指标并与单目标基线对比。"""
        from .baselines import run_all_baselines
        from .metrics import compare_solutions, solution_metrics
        from .stage4_nsga2 import knee_point, topsis_ranking

        logger.info("=" * 60)
        logger.info("评价：方案指标与基线对比")
        logger.info("=" * 60)

        if self.res.nsga2_result is None:
            self.run_stage4()

        dem = self.res.demand.grid
        cand = self.res.candidates.gdf
        at = self.res.access_time_s
        rows = []

        # -- IP 情景方案 ---------------------------------------------------
        for s in self.res.ip_solutions:
            if getattr(s, "status", "") != "Optimal" or not s.selected:
                continue
            rows.append(solution_metrics(s.selected, dem, cand, at, self.cfg, label=f"IP:{s.name}"))

        # -- NSGA-II 前沿 --------------------------------------------------
        F = self.res.nsga2_result.pareto_F
        X = self.res.nsga2_result.pareto_X
        if len(F):
            ki = knee_point(F)
            rows.append(
                solution_metrics(np.flatnonzero(X[ki]), dem, cand, at, self.cfg,
                                 label="NSGA-II:knee")
            )
            order = topsis_ranking(F)
            rows.append(
                solution_metrics(np.flatnonzero(X[order[0]]), dem, cand, at, self.cfg,
                                 label="NSGA-II:TOPSIS")
            )
            # 前沿极值解
            for j, key in enumerate(["unserved_demand", "total_access_time", "inequity", "total_cost"]):
                if j < F.shape[1]:
                    idx = int(np.argmin(F[:, j]))
                    rows.append(
                        solution_metrics(np.flatnonzero(X[idx]), dem, cand, at, self.cfg,
                                         label=f"NSGA-II:best_{key}")
                    )
            # 前沿解的指标分布（用于报告范围）
            all_met = [
                solution_metrics(np.flatnonzero(x), dem, cand, at, self.cfg, label=f"pareto_{i}")
                for i, x in enumerate(X)
            ]
            df_all = pd.DataFrame(all_met)
            self.res.diagnostics["pareto_metrics_range"] = {
                c: [float(df_all[c].min()), float(df_all[c].max())]
                for c in ["demand_coverage_pct", "total_cost_cny", "mean_access_time_min",
                          "worst_group_access_time_min", "equity_gini"]
                if c in df_all.columns
            }

        # -- 单目标基线 ----------------------------------------------------
        if run_baselines:
            try:
                p_values = _baseline_p_values(self.cfg, self.res)
                bls = run_all_baselines(dem, cand, at, self.cfg, p_values=p_values)
                self.res.baselines = bls
                for b in bls:
                    if b.feasible and b.selected:
                        rows.append(
                            solution_metrics(b.selected, dem, cand, at, self.cfg,
                                             label=f"Baseline:{b.name}")
                        )
            except Exception as exc:
                logger.error("基线求解失败: %s", str(exc)[:300])

        df = compare_solutions(rows, sort_by="demand_coverage_pct")
        self.res.solution_metrics = df
        logger.info("方案对比表:\n%s", df.to_string(index=False, max_cols=12))
        return df

    # -- 全流程 ------------------------------------------------------------

    def run_all(self, n_gen: int | None = None, pop_size: int | None = None,
                run_baselines: bool = True, save: bool = True):
        """依次执行四个阶段与评价。"""
        self.run_stage1()
        self.run_stage2()
        self.run_stage3()
        self.run_stage4(n_gen=n_gen, pop_size=pop_size)
        self.run_evaluation(run_baselines=run_baselines)
        if save:
            self.save()
        return self.res

    def save(self) -> Path:
        # 直接用配置里的 data_dir 本身。**不要取 .parent**：父目录恒为
        # ``outputs/``，会使 ``--set output.data_dir=...`` 完全失效，
        # 结果静默写回 ``outputs/data``，覆盖上一次实验的产物
        # （对照实验就是这么把主结果覆盖过两次的）。
        out = self.cfg.dir("output.data_dir")
        self.res.save(out)
        return out


def _baseline_p_values(cfg, res) -> list[int]:
    """为基线模型选取设施数量 p，使其与 NSGA-II 前沿可比。"""
    p_default = int(cfg.get("stage3_ip.max_sites", 40))
    ps = {max(3, p_default // 4), max(5, p_default // 2), max(8, int(p_default * 0.75)), p_default}
    if res is not None and getattr(res, "ip_solutions", None):
        for s in res.ip_solutions:
            if getattr(s, "n_sites", 0) > 0:
                ps.add(int(s.n_sites))
    # 控制规模：每个 p 要跑 3 个 MILP，取不超过 4 个
    return sorted(ps)[:4]
