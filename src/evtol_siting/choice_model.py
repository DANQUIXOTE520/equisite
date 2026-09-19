"""离散选择模型：eVTOL 采用概率与需求生成。

为什么需要这个模块
------------------
默认的需求生成方式（:func:`stage2_demand._assign_trip_rates`）把"出行率随
收入单调递增"当作**外生假设**硬编码进去：最高收入类的出行率是最低类的
ρ 倍，中间线性插值。这个假设很方便，但它是**假设的**——论文要论证"UAM
可能加剧不平等"，而如果需求本身就是按不平等的方式设定的，那就成了循环
论证。

本模块改为**推导**需求：给定票价、时间节省与各收入群体的时间价值，用离散
选择模型计算每个群体选择 eVTOL 的概率，再由概率生成需求:

    需求_i = 人口_i × 长距离出行率 × P(选择 eVTOL | 收入组 k(i))

这样，可达性不平等就来自**行为**（不同收入的人对同样的票价和时间节省做
不同的权衡），而不是来自研究者的设定。

**这同时补上了文献中的一个空白。** 系统调研显示，UAM 选址文献普遍使用
"聚合代理变量"作为需求，明确缺乏行为基础；Lu et al. (2025) 把这一点列为
自身局限。引入 logit 选择模型使本研究在需求侧具备了行为微观基础。

模型
----
二元 Logit（eVTOL vs. 地面交通）。旅客比较两种方式的**广义费用**
（generalized cost，货币单位）::

    GC_evtol  = 票价_evtol  + VOT_k · T_evtol
    GC_ground = 票价_ground + VOT_k · T_ground

选择概率::

    P(evtol) = 1 / (1 + exp( (GC_evtol - GC_ground) / μ ))

其中 ``VOT_k`` 是第 k 类人群的时间价值（元/小时），``μ`` 是 Logit 的尺度
参数（元），反映未观测到的效用差异。

时间价值由收入导出
------------------
::

    VOT_k = (月收入_k / (月工作天数 × 日工作小时)) × β_vot

``β_vot`` 是时间价值占工资率的比例。通勤时间价值通常取工资率的 40%–60%；
本文默认 0.5，并在敏感性分析中检验。

**收入阶梯的标定问题（论文必须说明）**
研究中实际可得的是**经济活动指数**（见 :mod:`data.rasters`），不是收入。
本模块把聚类的收入排名映射到一条对数正态收入阶梯上，其参数由成都市统计
数据标定。这是一个**标定假设**，不是观测值：它决定了各收入组的绝对水平，
但不决定它们的**相对顺序**（顺序由经济活动指数给出）。由于选择概率主要由
收入组之间的**相对**差异驱动，结论对该标定的敏感性低于其对收入排序的
敏感性——这一点应在论文中明确，并用敏感性分析佐证。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

@dataclass
class ChoiceParameters:
    """选择模型的全部参数。默认值来自成都市统计量与 UAM 运营成本文献。"""

    # -- 收入阶梯标定 ------------------------------------------------------
    # 成都城镇人均可支配收入（2023 年约 5.4 万元/年 ≈ 4 500 元/月）。
    # 建模用**收入中位数**而非均值：对数正态的收入中位数接近城市统计的
    # "人均可支配收入"，且不受高收入尾部影响。
    income_median_cny: float = 4_500.0
    # 对数正态的 σ（收入离散度）。中国城市居民收入的 log-σ 约 0.55–0.70；
    # 取 0.60 使最高收入组落在月收入 3–5 万元量级，符合 UAM 目标客群。
    income_sigma: float = 0.60
    # 各类人群在收入分布中的代表分位点（由聚类排名自动生成，此处仅存默认）
    cluster_percentiles: list[float] = field(default_factory=list)

    # -- 时间价值 ----------------------------------------------------------
    vot_share: float = 0.50          # VOT 占工资率的比例
    working_days_per_month: float = 21.75
    working_hours_per_day: float = 8.0

    # -- 票价 --------------------------------------------------------------
    evtol_fare_fixed_cny: float = 40.0      # 起降与场站使用费
    evtol_fare_per_km_cny: float = 6.0      # 元/km（早期商业运营量级）
    ground_fare_fixed_cny: float = 10.0     # 网约车/出租车起步价
    ground_fare_per_km_cny: float = 2.3

    # -- 出行时间 ----------------------------------------------------------
    trip_distance_km: float = 15.0          # 代表性出行距离（直线）
    ground_detour_factor: float = 1.35      # 路网绕行系数
    ground_speed_kmh: float = 22.0          # 城市道路含拥堵的平均行程速度
    evtol_cruise_kmh: float = 200.0
    evtol_access_min: float = 10.0          # 到起降场的接驳时间
    evtol_egress_min: float = 4.0           # 起降场到目的地的接驳时间
    evtol_wait_min: float = 6.0             # 候机与登机

    # -- Logit 尺度 --------------------------------------------------------
    # μ 反映未观测效用差异的尺度。取 25 元意味着约 27 元的广义费用差
    # 对应 3:1 的选择优势（概率 75%），是交通方式选择研究中的常见量级。
    logit_scale_cny: float = 25.0

    # -- 出行生成 ----------------------------------------------------------
    # 每人每日**长距离出行**（量级 15 km）次数。并非所有出行都在 UAM 的
    # 适用范围内，因此先在出行总量上做一次筛选，再由选择模型决定采用率。
    long_trip_rate_per_day: float = 0.10

    # -- 区域内收入异质性 --------------------------------------------------
    # **这是建模正确性的关键一步。** 选择模型是个体层面的决策，而聚类给出
    # 的是**区域平均**收入。用区域平均收入直接代入 logit，等价于假设区域内
    # 所有人的收入完全相同——这会系统性低估采用率，也会压平收入群体之间的
    # 差异：UAM 的实际客群是各区域内**收入分布的右尾**，而不是区域均值。
    #
    # 因此在每个区域的收入分布上积分：
    #     P_area = ∫ P(选择 | 收入 y) · f(y | 区域均值 I_k) dy
    # 其中区域内分布取对数正态，σ_within 反映区域内部差异。
    #
    # σ_within 的取值：区域内部收入差异显著小于区域之间，中国城市数据的
    # 量级约 0.3–0.5。取 0.40。该参数直接影响采用率的绝对水平，是敏感性
    # 分析的必要项。
    within_area_income_sigma: float = 0.40
    n_income_quadrature: int = 24     # 积分节点数（Gauss-Hermite 等价的网格法）

    @classmethod
    def from_config(cls, cfg) -> "ChoiceParameters":
        """从 YAML 配置构造参数对象（未配置的项用默认值）。"""
        d = cfg.get("choice_model", {}) or {}
        p = cls()
        for k, v in d.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
        return p


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class EVTOLChoiceModel:
    """eVTOL 采用概率与需求生成。

    Parameters
    ----------
    cfg
        配置对象；参数从 ``choice_model`` 段读取。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.p = ChoiceParameters.from_config(cfg)

    # -- 收入阶梯 ----------------------------------------------------------

    def income_ladder(self, n_clusters: int, ranks: np.ndarray | None = None) -> np.ndarray:
        """把聚类映射到月收入（元/月）。

        Parameters
        ----------
        n_clusters
            人群类别数 K。
        ranks
            各聚类的收入排名（0 = 最低）。``None`` 时按 0..K-1 等序处理。

        Returns
        -------
        长度 K 的月收入数组（元/月）。

        Notes
        -----
        分位点取各类别的秩中点 ``(i + 0.5) / K``，再经对数正态分位数函数
        映射为收入。用秩中点而非端点，避免最高类被映射到分布的极端尾部
        （那会让 VOT 出现不合理的量级）。
        """
        if ranks is None:
            ranks = np.arange(n_clusters)
        ranks = np.asarray(ranks, dtype=np.float64)
        order = np.argsort(ranks, kind="stable")
        pct = (np.arange(n_clusters, dtype=np.float64) + 0.5) / max(n_clusters, 1)

        from scipy.stats import lognorm

        # 对数正态参数化：median = income_median，shape = sigma
        dist = lognorm(s=self.p.income_sigma, scale=self.p.income_median_cny)
        inc_sorted = dist.ppf(pct)

        # 还原到原始类别顺序
        inc = np.empty(n_clusters, dtype=np.float64)
        inc[order] = inc_sorted
        logger.info(
            "收入阶梯（K=%d）: %s",
            n_clusters,
            "  ".join(f"{v:,.0f}" for v in inc),
        )
        return inc

    # -- 时间价值 ----------------------------------------------------------

    def value_of_time(self, monthly_income: np.ndarray) -> np.ndarray:
        """月收入 -> 时间价值（元/小时）。

        ``VOT = 月收入 / (月工作天数 × 日工作小时) × β_vot``
        """
        wage_rate = monthly_income / (
            self.p.working_days_per_month * self.p.working_hours_per_day
        )
        vot = wage_rate * self.p.vot_share
        logger.info(
            "时间价值: %s 元/小时（对应月收入 %s）",
            "  ".join(f"{v:.1f}" for v in vot),
            "  ".join(f"{v:,.0f}" for v in monthly_income),
        )
        return vot

    # -- 广义费用 ----------------------------------------------------------

    def travel_times(self) -> dict[str, float]:
        """代表性出行的两种方式耗时（分钟）。"""
        d = self.p.trip_distance_km
        d_ground = d * self.p.ground_detour_factor
        t_ground = d_ground / self.p.ground_speed_kmh * 60.0
        t_evtol = (
            self.p.evtol_access_min
            + self.p.evtol_wait_min
            + d / self.p.evtol_cruise_kmh * 60.0
            + self.p.evtol_egress_min
        )
        return {"t_ground_min": t_ground, "t_evtol_min": t_evtol,
                "ground_distance_km": d_ground, "saving_min": t_ground - t_evtol}

    def fares(self, evtol_fare_multiplier: float = 1.0) -> dict[str, float]:
        """两种方式的票价（元）。"""
        d = self.p.trip_distance_km
        d_ground = d * self.p.ground_detour_factor
        return {
            "fare_evtol": (
                self.p.evtol_fare_fixed_cny + self.p.evtol_fare_per_km_cny * d
            ) * evtol_fare_multiplier,
            "fare_ground": (
                self.p.ground_fare_fixed_cny
                + self.p.ground_fare_per_km_cny * d_ground
            ),
        }

    def trip_distance_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """返回出行距离的离散节点与权重（权重和为 1）。

        固定模式下退化为单点（原行为，保证与既有结果可比）。启用分布后，
        在 ``trip_distance_dist`` 指定的对数正态上取等概率分位点作为节点。
        """
        cfg = self.cfg.get("choice_model.trip_distance_dist", None) or {}
        if not cfg.get("enabled", False):
            return np.asarray([self.p.trip_distance_km], dtype=np.float64), \
                   np.asarray([1.0], dtype=np.float64)

        from scipy.stats import lognorm

        med = float(cfg.get("median_km", 8.0))
        sigma = float(cfg.get("sigma", 0.7))
        lo, hi = (float(x) for x in cfg.get("range_km", [3.0, 60.0]))
        n = int(cfg.get("n_nodes", 24))

        # lognorm 以 scale=中位数参数化：s=sigma, scale=med
        dist = lognorm(s=sigma, scale=med)
        cdf_lo, cdf_hi = dist.cdf(lo), dist.cdf(hi)
        qs = np.linspace(cdf_lo, cdf_hi, n + 1)
        edges = dist.ppf(qs)
        edges = np.clip(edges, lo, hi)
        nodes = 0.5 * (edges[:-1] + edges[1:])          # 分位中点
        weights = np.diff(qs)
        weights = weights / weights.sum()
        return nodes.astype(np.float64), weights.astype(np.float64)

    def choice_probability(
        self,
        vot: np.ndarray,
        evtol_fare_multiplier: float = 1.0,
    ) -> np.ndarray:
        """各收入群体选择 eVTOL 的概率（二元 Logit）。

        ``trip_distance_dist.enabled`` 为真时，在各距离节点上分别求概率再按
        出行距离分布加权平均——即把"固定 15 km 的代表性出行"换成**出行距离
        分布上的期望采用概率**。UAM 的竞争力高度依赖距离（短途被接驳与候机
        开销压垮、长途被票价压垮），单点距离无法表达这一点。

        Returns
        -------
        长度与 ``vot`` 相同的概率数组，取值 ``(0, 1)``。
        """
        nodes, weights = self.trip_distance_nodes()
        p = np.zeros_like(np.asarray(vot, dtype=np.float64))
        base_d = self.p.trip_distance_km          # 循环内临时改写，结束后还原
        try:
            for d_km, w in zip(nodes, weights):
                self.p.trip_distance_km = float(d_km)
                t = self.travel_times()
                f = self.fares(evtol_fare_multiplier)
                gc_evtol = f["fare_evtol"] + vot * (t["t_evtol_min"] / 60.0)
                gc_ground = f["fare_ground"] + vot * (t["t_ground_min"] / 60.0)
                delta = gc_evtol - gc_ground
                p += w * (1.0 / (1.0 + np.exp(
                    np.clip(delta / self.p.logit_scale_cny, -50, 50))))
        finally:
            self.p.trip_distance_km = base_d
        return p

    # -- 需求生成 ----------------------------------------------------------

    def cluster_demand_rates(
        self,
        cluster_profile: pd.DataFrame,
        evtol_fare_multiplier: float = 1.0,
        integrate_within_area: bool | None = None,
    ) -> pd.DataFrame:
        """为每个聚类计算 eVTOL 采用概率与出行生成率。

        Parameters
        ----------
        cluster_profile
            阶段二输出的类画像，须含 ``cluster`` 与收入列
            （``income_mean`` 或与之等价的排序依据）。
        integrate_within_area
            是否在区域内部收入分布上积分（默认按配置
            ``choice_model.integrate_within_area``，缺省为 True）。
            设为 ``False`` 可退化为"用区域平均收入直接代入"的口径，
            用于在论文中对比两种处理方式的差异。

        Returns
        -------
        含 ``cluster`` / ``monthly_income_cny`` / ``vot_cny_per_h`` /
        ``p_evtol`` / ``demand_rate`` 的表（论文 Table 用）。

        Notes
        -----
        ``demand_rate`` 是**每人每日 eVTOL 出行次数**，等于
        ``长距离出行率 × P(evtol)``。下游需求 = 人口 × demand_rate。

        积分口径下 ``p_evtol`` 不再对应某个具体收入水平，而是该区域全体
        居民的平均采用概率；``monthly_income_cny`` 仍是区域均值，用于刻画
        该区域的收入水平。
        """
        prof = cluster_profile.copy()
        inc_col = next(
            (c for c in ("income_mean", "income", "income_index") if c in prof.columns),
            None,
        )
        if inc_col is None:
            logger.warning("类画像缺少收入列，按类号顺序假定收入递增")
            ranks = np.arange(len(prof), dtype=np.float64)
        else:
            ranks = prof[inc_col].to_numpy(dtype=np.float64)

        inc = self.income_ladder(len(prof), ranks)
        vot_mean = self.value_of_time(inc)

        if integrate_within_area is None:
            integrate_within_area = bool(
                self.cfg.get("choice_model.integrate_within_area", True)
            )

        if integrate_within_area:
            p = self._probability_integrated(inc, evtol_fare_multiplier)
            mode_note = f"区域内积分（σ_within={self.p.within_area_income_sigma}）"
        else:
            p = self.choice_probability(vot_mean, evtol_fare_multiplier)
            mode_note = "区域均值直接代入"

        # 距离口径说明。**不能打印 self.p.trip_distance_km**——积分模式下它在
        # 循环后被还原成基准值，日志会显示"出行 15 km"而实际已按分布积分，
        # 造成"覆盖没生效"的误判（本次验证中确实误判过一次）。
        _nodes, _w = self.trip_distance_nodes()
        if len(_nodes) > 1:
            _dist_note = (f"出行距离分布 {_nodes.min():.0f}–{_nodes.max():.0f} km"
                          f"（{len(_nodes)} 节点，加权均值 "
                          f"{float((_nodes * _w).sum()):.1f} km）")
        else:
            _dist_note = f"出行 {self.p.trip_distance_km:g} km（单点）"

        out = prof[["cluster"]].copy()
        out["monthly_income_cny"] = np.round(inc, 0)
        out["vot_cny_per_h"] = np.round(vot_mean, 2)
        out["p_evtol"] = np.round(p, 5)
        out["demand_rate"] = np.round(p * self.p.long_trip_rate_per_day, 6)

        t, f = self.travel_times(), self.fares(evtol_fare_multiplier)
        logger.info(
            "选择模型 [%s]: %s，地面 %.0f 分钟 / %.0f 元，"
            "eVTOL %.0f 分钟 / %.0f 元，节省 %.0f 分钟；"
            "采用概率 %.4f–%.4f（类间比 %.2f）",
            mode_note, _dist_note, t["t_ground_min"], f["fare_ground"],
            t["t_evtol_min"], f["fare_evtol"], t["saving_min"],
            float(p.min()), float(p.max()), float(p.max() / max(p.min(), 1e-9)),
        )
        logger.info("各类人群采用概率:\n%s", out.to_string(index=False))
        return out

    def _probability_integrated(
        self, area_mean_income: np.ndarray, evtol_fare_multiplier: float
    ) -> np.ndarray:
        """在区域内收入分布上积分得到平均采用概率。

        对区域 ``k``，其居民收入服从对数正态，**中位数**取该区域的平均收入
        （用中位数而非均值作为分布中心，避免均值被自身尾部拉偏）::

            P_k = ∫ P(选择 y) · LN(y; median=I_k, σ_w) dy

        积分用对数收入轴上的均匀网格做数值求积：对数正态在对数轴上是对称
        的，网格法比高斯-埃尔米特在尾部收敛更稳，且实现简单。

        Returns
        -------
        长度等于 ``area_mean_income`` 的平均采用概率数组。
        """
        sigma_w = float(self.p.within_area_income_sigma)
        n_q = int(self.p.n_income_quadrature)
        if sigma_w <= 1e-6 or n_q < 3:
            # 退化为无区域内部异质
            return self.choice_probability(
                self.value_of_time(area_mean_income), evtol_fare_multiplier
            )

        # 对数收入网格：覆盖 ±4σ
        z = np.linspace(-4.0, 4.0, n_q)
        # 对数正态的密度权重（标准对数正态在 z 轴上的密度）
        w = np.exp(-0.5 * z ** 2)
        w = w / w.sum()

        out = np.empty(len(area_mean_income), dtype=np.float64)
        for k, inc_k in enumerate(area_mean_income):
            y = inc_k * np.exp(sigma_w * z)          # 该区域内各收入水平
            vot = self.value_of_time(y)
            p_y = self.choice_probability(vot, evtol_fare_multiplier)
            out[k] = float(np.sum(w * p_y))
        return out

    def break_even_income(self, evtol_fare_multiplier: float = 1.0) -> float:
        """选择概率恰为 50% 的月收入（元/月）。

        这是本模型最有政策含义的一个输出：它回答"票价定在多少，
        什么收入水平的人才会开始使用"。按当前默认参数，它通常落在城市
        收入分布的顶部若干个百分点——这正是 UAM "精英化"批评的量化形式。
        """
        t = self.travel_times()
        f = self.fares(evtol_fare_multiplier)
        # 广义费用相等 => VOT* = (fare_evtol - fare_ground) / 时间节省(小时)
        saving_h = (t["t_ground_min"] - t["t_evtol_min"]) / 60.0
        if saving_h <= 0:
            return float("inf")
        vot_star = (f["fare_evtol"] - f["fare_ground"]) / saving_h
        wage = vot_star / max(self.p.vot_share, 1e-9)
        return float(wage * self.p.working_days_per_month * self.p.working_hours_per_day)

    # -- 政策情景 ----------------------------------------------------------

    def fare_scenarios(
        self,
        cluster_profile: pd.DataFrame,
        multipliers: list[float] | None = None,
    ) -> pd.DataFrame:
        """票价情景扫描：票价如何改变各收入群体的采用率与公平性。

        这是选择模型相对"外生出行率"假设的**主要增量**：票价是可调的政策
        变量，而外生假设下根本不存在这个杠杆。

        Returns
        -------
        每个票价倍数一行的汇总表，含各类人群采用概率、最低/最高收入类
        概率之比（采用公平性），以及总体加权采用率。
        """
        mults = multipliers or [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        rows = []
        for m in mults:
            df = self.cluster_demand_rates(cluster_profile, evtol_fare_multiplier=m)
            p = df["p_evtol"].to_numpy()
            w = cluster_profile.set_index("cluster")["population"]
            w = w.reindex(df["cluster"]).to_numpy(dtype=np.float64)
            rows.append({
                "fare_multiplier": m,
                "p_min": float(p.min()),
                "p_max": float(p.max()),
                "p_ratio_max_min": float(p.max() / max(p.min(), 1e-9)),
                "p_population_weighted": float(np.average(p, weights=w)),
                "break_even_income_cny": self.break_even_income(m),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 与阶段二的衔接
# ---------------------------------------------------------------------------

def apply_choice_model(
    grid: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cfg,
    evtol_fare_multiplier: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用选择模型替换外生出行率，生成需求。

    Parameters
    ----------
    grid
        需求单元表，须含 ``cluster`` 与 ``population``。
    cluster_profile
        阶段二的类画像。

    Returns
    -------
    ``(grid_with_demand, choice_table)``。``grid`` 新增列
    ``monthly_income_cny`` / ``vot_cny_per_h`` / ``p_evtol`` / ``demand``。
    """
    m = (
        float(evtol_fare_multiplier)
        if evtol_fare_multiplier is not None
        else float(cfg.get("choice_model.evtol_fare_multiplier", 1.0))
    )
    model = EVTOLChoiceModel(cfg)
    table = model.cluster_demand_rates(cluster_profile, evtol_fare_multiplier=m)

    out = grid.copy()
    out = out.merge(
        table[["cluster", "monthly_income_cny", "vot_cny_per_h", "p_evtol", "demand_rate"]],
        on="cluster", how="left",
    )
    out["demand"] = out["population"].to_numpy(dtype=np.float64) * out[
        "demand_rate"
    ].fillna(0.0).to_numpy(dtype=np.float64)
    out["trip_rate"] = out["demand_rate"]

    logger.info(
        "选择模型生成需求: 合计 %.0f 次/日（总体采用率 %.4f%%）",
        float(out["demand"].sum()),
        100 * float(out["demand"].sum()) / max(float(out["population"].sum()) * model.p.long_trip_rate_per_day, 1e-9),
    )
    return out, table
