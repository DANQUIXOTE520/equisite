#!/usr/bin/env python
"""敏感性分析与稳健性检验。

论文价值
--------
选址模型的结论是否依赖于**任意选定的参数**，是审稿人最常质疑的一点。
本脚本对关键参数做单因素扫描（OFAT），并回答三类问题：

1. **结论稳健吗**——方案规模、成本、覆盖率随参数如何变化？
2. **新意成立吗**——把出行生成率的收入比 ρ 调到 1（即需求与收入无关）
   之后，公平性差距是否还存在？这是一次**证伪检验**：若差距消失，
   说明公平性问题只是需求模型的产物而非空间事实，论文必须如实报告。
3. **哪些参数真正重要**——按对结果的影响幅度排序，为后续研究指出重点。

设计要点
--------
* 数据只加载一次（``Pipeline`` 的 cache 复用），后续每个参数点只重跑
  受影响的阶段。这比逐点完整重跑快一个数量级。
* 每个参数点独立记录，输出可直接用于论文的敏感性表格与图件。
* 支持断点续传：结果逐点追加写入 CSV，中断后重跑会跳过已完成的点。

用法
----
::

    python scripts/07_sensitivity.py --data-mode real
    python scripts/07_sensitivity.py --data-mode real --only roof_area,access_radius
    python scripts/07_sensitivity.py --data-mode synthetic --fast   # 快速验证
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import PROJECT_ROOT, load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("sensitivity")


def _run_fingerprint(cfg, data_mode: str, pop_size: int, n_gen: int) -> str:
    """当前基线的指纹：配置内容 + 数据模式 + 搜索预算。

    用于判断 ``sensitivity.csv`` 里的旧点是否还有效。**必须包含配置文件的
    内容**（而不是路径或修改时间）——上游判据改动往往只动代码、不动配置，
    所以指纹还要叠加候选集规模；但候选集规模要跑完 stage1 才知道，
    因此这里先取配置内容摘要，候选集规模在 stage2 完成后并入（见下）。
    """
    import hashlib

    h = hashlib.sha256()
    try:
        h.update(Path(cfg.source_path).read_bytes())
    except Exception:
        h.update(str(cfg.source_path).encode())
    h.update(json.dumps(cfg.overrides, sort_keys=True, ensure_ascii=False).encode())
    h.update(f"{data_mode}|{pop_size}|{n_gen}".encode())
    return h.hexdigest()[:16]


def trip_distance_override(v) -> dict:
    """出行距离结构的配置覆盖。

    ``"fixedNN"`` → 关闭分布、用固定单点 NN km（复现原口径）；
    数值 → 启用对数正态距离分布并取该值作中位数。
    """
    if isinstance(v, str) and v.startswith("fixed"):
        return {"choice_model.trip_distance_dist.enabled": False,
                "choice_model.trip_distance_km": float(v[5:])}
    return {"choice_model.trip_distance_dist.enabled": True,
            "choice_model.trip_distance_dist.median_km": float(v)}


def population_override(v) -> dict:
    """人口扫描的配置覆盖。

    数值 → 缩放倍数（WorldPop 为主源）；``"building"`` → 换用建筑体量估计。
    """
    if isinstance(v, str) and v == "building":
        return {"stage2_demand.population_source": "building_volume",
                "stage2_demand.population_scale": 1.0}
    return {"stage2_demand.population_source": "worldpop",
            "stage2_demand.population_scale": float(v)}


# ---------------------------------------------------------------------------
# 扫描设计
# ---------------------------------------------------------------------------
# 每一项: (名称, 配置覆盖模板, 取值列表, 受影响的阶段, 论文中的说明)
#   受影响阶段用于跳过多余计算：
#   "stage1" 之后所有阶段都要重跑；"stage2" 之后重跑 2/3/4；以此类推。
SWEEPS: dict[str, dict] = {
    "roof_area": {
        "values": [1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2500.0],
        "override": lambda v: {"stage1_candidates.rooftop.min_roof_area_m2": v},
        "affects": "stage1",
        "desc": "屋顶型起降场最小可用屋顶面积 (m²)。基准 1600（≈40m×40m）。",
    },
    "roof_height": {
        "values": [15.0, 20.0, 25.0, 30.0, 40.0],
        "override": lambda v: {"stage1_candidates.rooftop.min_height_m": v},
        "affects": "stage1",
        "desc": "屋顶型起降场最小建筑高度 (m)。基准 20。",
    },
    "building_coverage": {
        "values": [0.40, 0.50, 0.55, 0.60, 0.70],
        "override": lambda v: {"stage1_candidates.ground.max_building_coverage": v},
        "affects": "stage1",
        "desc": "地面型候选的建筑基底覆盖率上限。基准 0.55。",
    },
    "dilution": {
        "values": [300.0, 400.0, 500.0, 700.0, 1000.0],
        "override": lambda v: {"stage1_candidates.dilution.min_distance_m": v},
        "affects": "stage1",
        "desc": "候选起降场最小间距 c_min (m)。基准 500。",
    },
    "airport_buffer": {
        # ★ 本参数的作用域在 2026-09-16 发生了实质变化，解读时务必注意。
        #
        # 原判据把**一切非直升的 aeroway 要素**都当机场排除，因而该半径实际
        # 作用于双流的航站楼/滑行道/停机位等 1 298 个内部要素——双流距市中心
        # 仅约 10 km，故 8 km 半径曾剔除 3 860 个候选中的 2 776 个（72%）。
        #
        # 现判据只对**军用**机场（landuse=military / military=airfield，共 3 处：
        # 黄田坝、太平寺、凤凰山）施加该缓冲；民用运输机场及其内部设施不再
        # 排除（eVTOL 可在民航机场设站中转）。因此该扫描现在度量的是**军用
        # 机场净空区**的影响，量级远小于从前。
        #
        # 真实的空域约束是机场障碍物限制面（OLS，一个有坡度的方向性曲面），
        # 而非各向同性的圆柱体；圆柱体仍是保守近似。论文必须报告本参数，
        # 否则会被质疑候选集是被"削"出来的。
        "values": [0.0, 2000.0, 4000.0, 6000.0, 8000.0, 12000.0],
        "override": lambda v: {"stage1_candidates.safety.airport_exclusion_radius_m": v},
        "affects": "stage1",
        "desc": "**军用**机场净空区排除半径 (m)。基准 4000。"
                "0 表示不施加该约束。民用机场不参与该约束。",
    },
    "n_clusters": {
        "values": [3, 4, 5, 6, 7, 8],
        "override": lambda v: {
            "stage2_demand.n_clusters": v,
            "stage2_demand.k_selection.method": "fixed",
        },
        "affects": "stage2",
        "desc": "K-means 人群类别数 K。基准由轮廓系数自动选择。",
    },
    # ⚠ 原 "access_radius" 扫描项已删除（2026-09-16）。
    #
    # 它覆盖的是 ``stage3_ip.access_radius_m``，而该参数**在主路径上不起作用**：
    # 一旦配置里给了 ``access_time_budget_min``（本研究即 15 min），距离就只
    # 在时间预算缺省时才作为回退。``pipeline.compute_access_time`` 的文档也
    # 写明该参数"只影响缓存键"。
    #
    # 实测后果：原扫描的 5 个取值（2000–5000 m）给出**逐列完全相同**的结果
    # （候选、站数、公平性、成本），即该扫描是空的。详见 RUN_STATE「零之六」。
    #
    # 改扫真正起作用的量——接驳时间预算。注意其**上限由方式本身决定**：
    # 短驳车 25 km/h、上限 5000 m、含 3 min 候车 ⇒ 可能出现的最大接驳时间
    # 恰为 15.00 min。因此取值必须跨到 15 以下才会咬合；列入 18 是为了
    # 用"无变化"直接证明预算在该点之上是非约束的。
    "access_budget": {
        "values": [8.0, 10.0, 12.0, 15.0, 18.0],
        "override": lambda v: {"stage3_ip.access_time_budget_min": v},
        "affects": "stage3",
        "desc": "接驳时间预算 (min)。基准 15。**这才是接驳标准真正起作用的量**；"
                "距离口径（access_radius_m）在主路径上不生效。"
                "18 min 预期与 15 min 完全相同——用于证明该预算在 15 以上非约束。",
    },
    "core_demand": {
        "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "override": lambda v: {"stage3_ip.min_core_demand": v},
        "affects": "stage3",
        "desc": "核心需求分位数（需求量最高的 (1-q) 比例须被覆盖）。基准 0.60。",
    },
    "cost_scale": {
        "values": [0.5, 0.75, 1.0, 1.25, 1.5],
        "override": lambda v: {
            "costs.rooftop.fixed": 12_000_000.0 * v,
            "costs.rooftop.per_m2": 8_000.0 * v,
            "costs.ground.fixed": 25_000_000.0 * v,
            "costs.ground.land_per_m2": 3_000.0 * v,
            "costs.ground.per_m2": 5_000.0 * v,
        },
        "affects": "stage3",
        "desc": "建设成本整体缩放系数。基准 1.0（±50% 检验）。",
    },
    "trip_rate_ratio": {
        # ★ 这是证伪检验，不是普通稳健性检查
        "values": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
        "override": None,      # 特殊处理：见 _apply_trip_ratio
        "affects": "stage2",
        "desc": "出行生成率的最高收入类/最低收入类比值 ρ。基准 5.0。"
                "ρ=1 表示需求与收入完全无关 —— 若此时公平性差距消失，"
                "说明该差距是需求模型的产物而非空间事实。**证伪检验**。",
    },
    "equity_measure": {
        "values": ["group_mean", "individual_max"],
        "override": lambda v: {"stage4_nsga2.equity_measure": v},
        "affects": "stage4",
        "desc": "公平性目标的定义：人群组均值最大（本研究新意）vs "
                "单用户最小最大（等价于经典 p-中心）。用于证明前者不是后者的重复。",
    },
    "trip_distance": {
        # 原实现只用**一个固定距离**（15 km）代表全部长距离出行。而 UAM 的
        # 竞争力高度依赖距离：短途被接驳与候机开销压垮，长途被票价压垮。
        # 该扫描既检验"距离结构是否影响选址结论"，也把 15 km 这个取值本身
        # 纳入检验——它此前是未经验证的单点假设。
        "values": ["fixed15", 8.0, 12.0, 15.0, 20.0, 30.0],
        "override": trip_distance_override,
        "affects": "stage2",
        "desc": "出行距离结构。fixed15 = 现行固定 15 km 单点；数值 = 启用对数"
                "正态距离分布并取该值作**中位数**（km，σ=0.7、范围 3–60 km）。",
    },
    "income_median": {
        # 基准取**本城真实统计值**（成都 4 049 元/月，出自 2020 年统计公报；
        # 换城市须连同此值一并换掉）。故扫描点必须**相对本城标定值**给出，否则
        # 换城市后会去扫描另一个城市的绝对取值。倍数在 main() 里按配置解析
        # （见 `_relative_to_baseline`）。
        "values": [0.8, 1.0, 1.25, 1.5],
        "relative_to": "choice_model.income_median_cny",
        "override": lambda v: {"choice_model.income_median_cny": v},
        "affects": "stage2",
        "desc": "收入阶梯中位数（元/月）。基准取本城城镇居民人均可支配收入的"
                "官方统计值；扫描点为该基准的 0.8／1.0／1.25／1.5 倍。",
    },
    "income_sigma": {
        "values": [0.45, 0.60, 0.75],
        "override": lambda v: {"choice_model.income_sigma": v},
        "affects": "stage2",
        "desc": "收入对数正态的离散度 σ。基准 0.60（中国城市居民收入离散度量级）。",
    },
    "population": {
        # ★ 人口的量级与来源此前**从未被扰动过**，而它同时进入需求模型
        #   （需求 = 人口 × 出行率 × 采用概率）与全部覆盖统计量，研究区总量
        #   771 万又是全文头条之一。两方向：
        #     * 缩放 —— 检验需求量级的影响；
        #     * 换源 —— 用建筑体量估计取代 WorldPop 栅格。该路径独立于人口
        #       栅格，但与屋顶型候选共享同一份 OSM 建筑图层（§3.5 记录的
        #       循环依赖），故只作对照、不作主结果。
        #   本参数 affects="stage2"，复用 stage1，故比 stage1 类参数快。
        "values": [1.0, 0.8, 1.2, 0.5, 1.5, "building"],
        "override": population_override,
        "affects": "stage2",
        "desc": "人口来源与量级。1.0 = WorldPop 基准；数值为缩放倍数；"
                "\"building\" = 改用建筑体量估计人口。",
    },
}


# ---------------------------------------------------------------------------
# 各扫描项的**基准取值**
# ---------------------------------------------------------------------------
# 单因素扫描的每个参数都必有一个取值等于该参数的默认值。在那一行上，覆盖后的
# 整份配置就等于全局基准配置；又因为各组用同一组种子，所以 `anchor=True` 的
# 那些行的每一个指标都应当**逐位相同**——这是一条不需要额外信息、只用产物
# 自身就能证伪"协议不统一"的硬检验（见
# ``scripts/22_check_sensitivity_consistency.py``）。
#
# 为什么必须把它写成数据
# ----------------------
# 这些基准值此前**只写在上面各条 `desc` 的散文里**，机器读不到。于是校验脚本
# 只能靠配置键去猜，而配置键大多不存在，就退化成"取组内众数"——它把
# airport_buffer 的基准猜成 0、roof_area 猜成 1400、dilution 猜成 1000，
# 15 个参数里猜错了 9 个。**比对的根本不是该比的那一行**，检验因此形同虚设。
#
# `anchor=False` 的两项是**按构造**就无法与全局基准重合的，必须排除在
# "逐位相同"的断言之外，否则检验器会误报：
#   * `n_clusters`——扫描强制 `k_selection.method="fixed"`，而基准是轮廓系数
#     自动选择；K=5 只保证**选出的 K** 相同，不保证配置相同。
#   * `trip_rate_ratio`——扫描强制 `choice_model.enabled=False` 走外生阶梯，
#     而基准走 Logit 选择模型（见 `apply_trip_ratio` 的注释）。
BASELINES: dict[str, dict] = {
    "roof_area":         {"value": 1600.0,        "anchor": True},
    "roof_height":       {"value": 20.0,          "anchor": True},
    "building_coverage": {"value": 0.55,          "anchor": True},
    "dilution":          {"value": 500.0,         "anchor": True},
    "airport_buffer":    {"value": 4000.0,        "anchor": True},
    "access_budget":     {"value": 15.0,          "anchor": True},
    "core_demand":       {"value": 0.60,          "anchor": True},
    "cost_scale":        {"value": 1.0,           "anchor": True},
    "equity_measure":    {"value": "group_mean",  "anchor": True},
    "income_median":     {"value": 1.0,           "anchor": True},   # 相对基准的倍数
    "income_sigma":      {"value": 0.60,          "anchor": True},
    "population":        {"value": 1.0,           "anchor": True},
    "trip_distance":     {"value": "fixed15",     "anchor": True},
    "n_clusters":        {"value": 5,             "anchor": False,
                          "why": "扫描强制 k_selection.method=fixed，基准为轮廓系数自动选择"},
    "trip_rate_ratio":   {"value": 5.0,           "anchor": False,
                          "why": "扫描强制 choice_model.enabled=False（外生阶梯），基准走 Logit"},
}


def apply_trip_ratio(cfg_overrides: dict, ratio: float) -> dict:
    """把 ρ 写进配置：显式给出各类人群的出行率。

    实现方式：先按收入的秩把 K 类排开，最低类取基准出行率，
    最高类取 ``基准 × ρ``，中间线性插值。这与 ``stage2_demand`` 中
    ``_assign_trip_rates`` 的逻辑一致，只是把固定的 5 倍改为可扫描的 ρ。
    """
    ov = dict(cfg_overrides)

    # ★ 必须同时关闭行为模型。``choice_model.enabled = true``（默认）时出行率
    #   由 Logit 选择模型导出，外生阶梯会被**静默忽略**——实测 ρ=1 与 ρ=10
    #   给出逐位相同的结果，扫描形同虚设。外生阶梯只在 ``enabled = false``
    #   时才被使用（见 stage2_demand.cluster_demand 的两条分支）。
    ov["choice_model.enabled"] = False

    # ★ **不要钉死 K**。早期实现把 n_clusters 与 k_selection 一并固定（曾为 7），
    #   使控制臂的聚类与主分析不同，两臂不可比。现在只把 ρ 交给
    #   ``_assign_trip_rates``，由它按**实际的 K** 生成阶梯——K 仍由轮廓系数
    #   照常决定，两臂唯一的差别只剩需求模型。
    ov["stage2_demand.trip_rate_ratio"] = float(ratio)
    return ov


# ---------------------------------------------------------------------------
# 单点评估
# ---------------------------------------------------------------------------

def evaluate_point(
    base_cfg,
    overrides: dict,
    data_mode: str,
    pop_size: int,
    n_gen: int,
    pipe_cache: dict,
) -> dict:
    """在给定参数下跑一遍流水线，返回汇总指标。

    复用 ``pipe_cache`` 中的已加载数据，避免每个参数点都重新读盘/下载。
    """
    cfg = base_cfg.override(overrides)
    cfg.seed_everything()

    from evtol_siting.pipeline import Pipeline

    pipe = Pipeline(cfg, data_mode=data_mode, cache=pipe_cache)
    t0 = time.time()

    cs = pipe.run_stage1()
    dm = pipe.run_stage2()
    sols = pipe.run_stage3()
    res = pipe.run_stage4(n_gen=n_gen, pop_size=pop_size, verbose=False)

    F = np.asarray(res.pareto_F, dtype=np.float64)
    out: dict = {
        "n_candidates": len(cs),
        "n_rooftop": int((cs.gdf["facility_type"] == "rooftop").sum()),
        "n_ground": int((cs.gdf["facility_type"] == "ground").sum()),
        "n_clusters": dm.n_clusters,
        "total_population": float(dm.grid["population"].sum()),
        "total_demand": float(dm.grid["demand"].sum()),
        # ⚠ 审计 #24：原先只记 ``[s.n_sites for s in sols if status == "Optimal"]``，
        #   于是某个参数点的结果是 ``[]`` 时，**分不清是"无可行解"还是"脚本没记"**——
        #   正文把它渲染成"无可行解"，但没有证据。这里把每个情景的**原始状态**
        #   与站数一并落盘，使该歧义在产物层面就被消除。
        "ip_status": {s.name: str(getattr(s, "status", "?")) for s in sols},
        "ip_n_sites_all": {s.name: int(getattr(s, "n_sites", 0)) for s in sols},
        # 只有**证明最优**的解才计入（status 由 solver_status 统一判定）
        "ip_n_sites": [s.n_sites for s in sols if getattr(s, "status", "") == "Optimal"],
        "ip_cost": [float(s.total_cost) for s in sols if getattr(s, "status", "") == "Optimal"],
        # 被静默摘掉覆盖约束的核心单元数（A3）；>0 时该点的"核心全覆盖"不成立
        "ip_n_core_unreachable": {
            s.name: int(s.coverage.get("n_core_unreachable", 0))
            for s in sols if getattr(s, "coverage", None)
        },
        "pareto_size": int(len(F)),
        "elapsed_s": round(time.time() - t0, 1),
        # 报告口径：pop/gen 必须与结果一起落盘，否则正文无从声明
        "pop_size": int(pop_size),
        "n_gen": int(n_gen),
        "seed": int(cfg.get("project.random_seed", 42)),
    }

    if len(F):
        # 目标 1..4 = 未服务需求、总时间、公平性、成本
        out["pareto_unserved_min"] = float(F[:, 0].min())
        out["pareto_unserved_max"] = float(F[:, 0].max())
        out["pareto_time_min"] = float(F[:, 1].min())
        out["pareto_inequity_min"] = float(F[:, 2].min())
        out["pareto_inequity_max"] = float(F[:, 2].max())
        out["pareto_cost_min"] = float(F[:, 3].min())
        out["pareto_cost_max"] = float(F[:, 3].max())
        out["equity_gap_ratio"] = float(F[:, 2].max() / max(F[:, 2].min(), 1e-9))

        # 决策层：拐点解的指标
        from evtol_siting.metrics import solution_metrics
        from evtol_siting.stage4_nsga2 import knee_point

        k = knee_point(F)
        sel = np.flatnonzero(res.pareto_X[k] > 0.5)
        # ⚠ 保存拐点解的**站点索引**，而不只是它的指标。
        #
        #   2026-09-19 的教训：`sensitivity.csv` 里与人群组相关的几列
        #   （worst_group / group_gap / n_groups_unserved）事后**无法复现**——
        #   用脚本自己的 `evaluate_point`、同样的配置与种子重跑，得到的是
        #   另一组数（覆盖率 97–99%、n_unserved=0），而 CSV 记的是
        #   n_unserved=2.8、worst=1e12，且后者与 99 % 覆盖率在算术上不可能并存。
        #   由于当时**没有保存解本身**，无法判定是指标算错还是记录错，
        #   只能整轮重跑（3.1 小时）。
        #   存下 `sel` 之后，将来若再怀疑指标口径，可以直接对同一批解重算指标，
        #   把"重跑搜索"与"重算指标"解耦。
        out["knee_sel"] = [int(j) for j in sel]
        m = solution_metrics(
            sel, pipe.res.demand.grid, pipe.res.candidates.gdf,
            pipe.res.access_time_s, cfg, label="knee",
        )
        for key in ("n_sites", "total_cost_cny", "demand_coverage_pct",                    "population_coverage_pct", "mean_access_time_min",
                    "worst_group_access_time_min", "equity_gini",
                    # ⚠ 审计 #19：收入组间差距才是公平性证伪检验要测的量。
                    #   原先只记 pareto_inequity_min/max（前沿在 f3 上的**跨度**），
                    #   那不是组间差距，测不出任何东西。这里补上。
                    "group_gap_min", "group_gap_served_min",
                    "n_groups_unserved", "groups_all_served",
                    "worst_group_access_time_served_min"):
            if key in m:
                out[f"knee_{key}"] = m[key]

    return out


def _reps_of(r) -> int:
    """一行结果是在几次重复下算出来的。``n_reps`` 只在 reps>1 时写入，缺列/NaN 即 1。"""
    v = getattr(r, "n_reps", None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 1
    return 1 if f != f else int(f)              # NaN -> 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="敏感性分析与稳健性检验")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--only", default=None,
                    help="只跑指定项（逗号分隔），如 roof_area,access_radius")
    ap.add_argument("--pop-size", type=int, default=80)
    ap.add_argument("--n-gen", type=int, default=60)
    ap.add_argument("--reps", type=int, default=1,
                    help="每个参数点的独立重复次数。NSGA-II 是随机算法，"
                         "凡是要**据其排序或下结论**的参数点都必须 >=10，"
                         "否则点估计落在自身噪声之内（审计 #18）。"
                         "确定性量（候选数、IP 站数）在重复间不变，不受影响。")
    ap.add_argument("--reps-only", default=None,
                    help="只对列出的参数项用 --reps，其余仍为单次。"
                         "全部 76 个点都跑 10 次需要约 5 小时；而正文只对"
                         "其中少数几项下结论（ρ 的证伪检验、聚类数排序、"
                         "接驳半径、屋顶面积阈值）。默认（不指定）= 全部用 --reps。")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    ap.add_argument("--fast", action="store_true",
                    help="每个参数只取 3 个点（快速验证脚本可运行）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "sensitivity.log")

    base = load_config()
    names = [n.strip() for n in args.only.split(",")] if args.only else list(SWEEPS)
    bad = [n for n in names if n not in SWEEPS]
    if bad:
        raise SystemExit(f"未知的扫描项: {bad}\n可选: {list(SWEEPS)}")

    reps_only = ([s.strip() for s in args.reps_only.split(",")]
                 if args.reps_only else None)
    if reps_only:
        bad_r = [n for n in reps_only if n not in SWEEPS]
        if bad_r:
            raise SystemExit(f"--reps-only 中的未知扫描项: {bad_r}")
        log.info("仅以下参数项使用 reps=%d：%s（其余为单次）",
                 args.reps, ", ".join(reps_only))

    out_path = Path(args.out) if args.out else (
        Path(base.dir("output.tables_dir")) / "sensitivity.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- 断点续传：读入已完成的结果 ---------------------------------------
    #
    # ⚠ 续传必须以**数据指纹**为界。审计发现：上游改过判据（坡度、机场排除、
    #   PBF 图层匹配）之后候选集从 1877 变到 1315，但 ``sensitivity.csv`` 里
    #   76 个旧点仍被当成"已完成"而**全部跳过**——脚本安静地跑完、什么都不产出，
    #   日志还显示一切正常。这正是"改了上游、下游拿着旧数继续用"的典型路径。
    #
    #   现在把基线的指纹写进每一行；续传时只接受指纹相同的行，其余丢弃并告警。
    fp = _run_fingerprint(base, args.data_mode, args.pop_size, args.n_gen)
    log.info("基线指纹: %s", fp)
    done = pd.DataFrame()
    if out_path.exists():
        try:
            done = pd.read_csv(out_path)
            if "run_fingerprint" in done.columns:
                n_all = len(done)
                done = done[done["run_fingerprint"].astype(str) == fp]
                if len(done) < n_all:
                    log.warning(
                        "已有 %d 条结果中只有 %d 条与当前基线指纹一致，"
                        "其余 %d 条**视为过期并重跑**（上游判据或配置已变）。",
                        n_all, len(done), n_all - len(done),
                    )
            elif len(done):
                log.warning(
                    "已有 %d 条结果**没有指纹列**，无法判断是否与当前基线一致——"
                    "为安全起见全部重跑（不续传）。", len(done),
                )
                done = pd.DataFrame()
            # 同一 (参数, 取值) 出现多行时，只留**重复次数最高**的那一行。
            # 这种重复是历史续传留下的：旧版按 (parameter, value) 判重、写回时
            # 又不删旧行，于是重算过的点会留下两份。不清掉的话，下游按参数取
            # 第一行会取到协议不同的旧行（第五批就因此被误判为"协议未统一"）。
            if len(done) and {"parameter", "value"} <= set(done.columns):
                before = len(done)
                done = (done.assign(_r=[_reps_of(r) for r in done.itertuples()])
                            .sort_values("_r", kind="stable")
                            .drop_duplicates(subset=["parameter", "value"],
                                             keep="last")
                            .drop(columns="_r")
                            .reset_index(drop=True))
                if len(done) < before:
                    log.warning(
                        "已有结果中有**同一个 (参数, 取值) 的多行**（%d → %d 行）："
                        "保留 reps 最高的一行，其余丢弃。",
                        before, len(done))

            if len(done):
                log.info("已有 %d 条同基线结果，将跳过这些参数点", len(done))
        except Exception as exc:
            log.warning("读取既有结果失败（将全量重跑）: %s", str(exc)[:160])
            done = pd.DataFrame()
    done_keys = set()
    if len(done) and {"parameter", "value"} <= set(done.columns):
        # ⚠ 键必须含**重复次数**。此前只按 (parameter, value) 判重，于是把
        #   "用 1 次算出来的行"也当成"已完成"：用 `--reps 10` 重跑时，76 个
        #   取值点会**全部被跳过**，输出把旧表原样写回——跑完 0 秒，而"协议
        #   不统一"的问题分毫未动。本轮整表重跑险些就这样被静默吞掉。
        #   `n_reps` 列只在 reps>1 时写入（见下方），缺列或为空即 1 次。
        done_keys = {(str(r.parameter), str(r.value), _reps_of(r))
                     for r in done.itertuples()}

    # 数据只加载一次，所有参数点复用
    pipe_cache: dict = {}
    rows: list[dict] = []

    log.info("=" * 74)
    log.info(" 敏感性分析：%d 个参数项，data_mode=%s", len(names), args.data_mode)
    log.info("=" * 74)

    for name in names:
        spec = SWEEPS[name]
        values = list(spec["values"])
        # 相对取值：把"基准的倍数"解析成本城的绝对取值。
        # 收入阶梯的基准取**本城真实统计值**（成都 4 049 元/月），
        # 若扫描点仍写绝对数，换城市后就会去扫描另一个城市的取值。
        rel = spec.get("relative_to")
        if rel:
            base_v = float(base.get(rel))
            values = [round(base_v * float(m), 1) for m in values]
        if args.fast and len(values) > 3:
            idx = np.linspace(0, len(values) - 1, 3).round().astype(int)
            values = [values[i] for i in sorted(set(idx))]

        log.info("")
        log.info("── %s ── %s", name, spec["desc"])
        log.info("   取值: %s", values)

        # ⚠ 重复次数必须在**跳过判定之前**算出来——续传的键要带上 reps，
        #   否则 1 次算出的行会挡住 10 次的重算（见 done_keys 处的说明）。
        reps = max(1, int(args.reps))
        if reps_only is not None and name not in reps_only:
            reps = 1

        for v in values:
            if (name, str(v), reps) in done_keys:
                log.info("   %-10s 已完成（%d 次），跳过", v, reps)
                continue

            # ⚠ 同一个 (参数, 取值) 在表里**只允许留一行**。
            #
            #   此前不删旧行，只做 `concat([done, rows])`。于是凡是因 reps 不同而
            #   被重算的点，旧行仍留在 `done` 里、新行再追加一份——跑完得到一张
            #   "一个参数点两行、分属两套协议"的表（第五批实测 123 行 = 76 新 + 47 旧）。
            #   危害不在行数，而在**下游取错行**：`22_check_sensitivity_consistency.py`
            #   按参数取第一行做基准断言，取到的正是旧行，于是 15 个基准行"不一致"，
            #   把一次**已经统一**的运行误判成协议没统一——差一点就据此把表删掉重跑。
            #
            #   语义是**替换**而非丢弃：旧行由本轮结果顶掉。
            if len(done):
                same = np.array(
                    [str(r.parameter) == name and str(r.value) == str(v)
                     for r in done.itertuples()], dtype=bool)
                if same.any():
                    log.warning(
                        "   %-10s 旧结果中已有同键行（n_reps=%s），"
                        "由本轮（%d 次）**替换**——不保留两份。",
                        v, sorted({_reps_of(r) for r in done[same].itertuples()}),
                        reps,
                    )
                    done = done[~same]

            ov = spec["override"](v) if spec["override"] else {}
            if name == "trip_rate_ratio":
                ov = apply_trip_ratio(ov, float(v))

            try:
                # ⚠ 审计 #18：NSGA-II 是随机算法，单次运行的点估计不能用来排序。
                #   §5.10 自报同一指标的 10 次运行间跨度是 0.42 分钟，而 §5.9.1
                #   拿它去排"聚类数 5% = 0.31 分钟"的差异——**小于自身噪声**。
                #   这里按 --reps 独立重复，把**确定性量**（候选数、IP 站数、
                #   成本下界）与**随机量**（前沿指标、拐点指标）分开报告：
                #   前者取首次数值（逐次相同），后者给 均值 ± 标准差。
                runs = []
                for rep in range(reps):
                    ov_rep = dict(ov)
                    ov_rep["project.random_seed"] = int(
                        base.get("project.random_seed", 42)) + rep
                    runs.append(evaluate_point(
                        base, ov_rep, args.data_mode, args.pop_size,
                        args.n_gen, pipe_cache))
                m = dict(runs[0])
                if reps > 1:
                    # ⚠ ``det`` = **已实测为确定性**的量，不写标准差列。
                    #   2026-09-19 用 ``scripts/26_probe_seed_demand.py`` 在 seed=42/43/44 上
                    #   实测（只跑 stage1+stage2）：n_candidates / n_rooftop /
                    #   n_clusters / total_population 三次**逐位相同**（极差 0），
                    #   故留在 det 里。
                    #   ``total_demand`` 则**随种子变化**（三次取值 50 334.94 /
                    #   50 345.86 / …，极差/均值 ≈ 2.2e-4）：K-means 的
                    #   ``random_state=cfg.seed`` 换了簇归属，类画像随之改变，
                    #   选择模型给出的采用率就不同。此前它被误判为确定性量，
                    #   于是表里只有 10 次均值、没有标准差——读者无从知道
                    #   "需求量"这一列本身带着运行间噪声。已移出 det。
                    det = {"n_candidates", "n_rooftop", "n_ground", "n_clusters",
                           "total_population", "pareto_size",
                           "ip_n_sites", "ip_cost",
                           # 运行口径本身不是"结果"，不该有标准差列
                           "pop_size", "n_gen", "seed"}
                    for k in list(runs[0].keys()):
                        vals = [r.get(k) for r in runs
                                if isinstance(r.get(k), (int, float))
                                and not isinstance(r.get(k), bool)]
                        if len(vals) < 2:
                            continue
                        mu = float(np.mean(vals))
                        sd = float(np.std(vals, ddof=1))
                        m[k] = mu
                        if k not in det:
                            m[f"{k}_sd"] = sd
                            m[f"{k}_n"] = len(vals)
                    m["n_reps"] = reps
                log.info(
                    "   %-10s 候选 %5d（屋顶 %4d）| 前沿 %3d | 组间差距 %.3f±%.3f min "
                    "| 成本下界 %.2f 亿 | %.0fs",
                    v, m["n_candidates"], m["n_rooftop"], m["pareto_size"],
                    m.get("knee_group_gap_min", np.nan),
                    m.get("knee_group_gap_min_sd", 0.0),
                    m.get("pareto_cost_min", np.nan) / 1e8, m["elapsed_s"],
                )
                m.update({"parameter": name, "value": v, "ok": True,
                          "run_fingerprint": fp})
            except Exception as exc:
                log.error("   %-10s 失败: %s", v, str(exc)[:200])
                m = {"parameter": name, "value": v, "ok": False,
                     "error": str(exc)[:300], "run_fingerprint": fp}

            rows.append(m)
            # 逐点落盘，保证中断不丢结果
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                out_path, index=False
            )

    final = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    final.to_csv(out_path, index=False)

    # -- 汇总 --------------------------------------------------------------
    log.info("")
    log.info("=" * 74)
    log.info(" 完成。结果: %s（%d 条）", out_path, len(final))

    if len(final) and "ok" in final.columns:
        ok = final[final["ok"] == True]  # noqa: E712
        if len(ok):
            log.info("")
            log.info("── 关键参数的相对影响幅度（用于论文敏感性讨论）──")
            for name in names:
                sub = ok[ok["parameter"] == name]
                if len(sub) < 2:
                    continue
                for col in ("n_candidates", "pareto_inequity_min", "pareto_cost_min"):
                    if col not in sub.columns:
                        continue
                    v = sub[col].astype(float)
                    if v.notna().sum() < 2 or v.mean() == 0:
                        continue
                    log.info(
                        "   %-18s %-22s 变化幅度 %.1f%%（%.4g → %.4g）",
                        name, col, 100 * (v.max() - v.min()) / abs(v.mean()),
                        v.min(), v.max(),
                    )

            # 证伪检验的结论
            #
            # ⚠ 审计 #19：这里原先用的是 ``pareto_inequity_min/max``，即**前沿在
            #   f3 目标上的跨度**，而 ρ 的证伪检验要问的是"**收入组之间的可达性
            #   差距**是否随 ρ→1 而消失"。前沿跨度是"同一个解集合内部最好与最差
            #   解的距离"，与组间差距不是一回事；而且前沿跨度对 ρ 不敏感本就是
            #   意料之中（ρ 只改变需求权重，不改变空间格局），于是这个检验
            #   **证不了任何东西**，却撑着一条结论。
            #
            #   正确的量是拐点解的 ``group_gap_min``（最差组均值 − 最好组均值），
            #   它在 metrics.solution_metrics 里早就算好了，只是敏感性脚本没用。
            fals = ok[ok["parameter"] == "trip_rate_ratio"].sort_values("value")
            gap_col = ("knee_group_gap_min"
                       if "knee_group_gap_min" in fals.columns else None)
            if len(fals) >= 2 and gap_col:
                log.info("")
                log.info("── 公平性证伪检验（ρ = 出行率收入比，口径 = 收入组间差距）──")
                for _, r in fals.iterrows():
                    # ⚠ r["value"] 从 CSV 读回后**可能是字符串**（如 "8.0"），
                    # 直接喂给 %-5.1f 会在 logging 内部抛 TypeError。该异常被
                    # logging 自己吞掉，**只丢掉这一行日志、不影响计算**，因此
                    # 极难发现——`choice_model.py` 有过同款问题。这里先归一到文本。
                    try:
                        rho_txt = f"{float(r['value']):.1f}"
                    except (TypeError, ValueError):
                        rho_txt = str(r["value"])
                    log.info(
                        "   ρ=%-5s  组间差距 %.3f min（最差组 %.3f，覆盖 %.1f%%，"
                        "组全服务=%s）",
                        rho_txt,
                        r.get(gap_col, np.nan),
                        r.get("knee_worst_group_access_time_min", np.nan),
                        r.get("knee_demand_coverage_pct", np.nan),
                        r.get("knee_groups_all_served", "-"),
                    )
                lo = fals.iloc[0]
                g_lo = float(lo.get(gap_col, np.nan))
                g_hi = float(fals.iloc[-1].get(gap_col, np.nan))
                log.info(
                    "   判读：ρ=1 表示出行率与收入无关。若此时组间差距仍显著 >0，"
                    "说明可达性不平等主要来自**空间分布**而非收入出行率的设定，"
                    "新意成立；若差距随 ρ 下降而消失，则公平性问题主要由需求模型"
                    "驱动，论文必须如实报告。"
                )
                log.info("   实测：ρ 最大时差距 %.3f min，ρ 最小时 %.3f min，"
                         "变化 %.1f%%", g_hi, g_lo,
                         100 * (g_hi - g_lo) / max(abs(g_hi), 1e-9))
            elif len(fals) >= 2:
                log.warning(
                    "   未找到 knee_group_gap_min 列，无法做组间差距的证伪检验。"
                    "此时 §5.9.2 的支持性结论**必须删除**，不得沿用旧的前沿跨度口径。"
                )
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
