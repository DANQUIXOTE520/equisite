"""出版级图件生成。

设计约定
--------
* **英文标注**——稿件是英文投稿，图内文字与正文语言一致。
* 色盲友好配色：使用 Okabe–Ito 调色板（对三类色觉障碍均可区分），
  且各颜色在灰度打印下的明度差异足够。
* 默认同时输出 PDF（矢量，投稿用）、PNG（600 dpi，预览/审稿用）与 SVG。
* 所有图件函数只接收数据、返回文件路径，不做 I/O 之外的事情，
  便于在脚本里批量调用与重跑。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Okabe–Ito 色盲友好调色板
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#999999",
}


def _setup_style():
    """统一的 matplotlib 样式（字体、线宽、网格）。"""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _save(fig, out_dir: Path, name: str, dpi: int = 600,
          formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """保存图件为多种格式，返回路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi if fmt == "png" else None, format=fmt)
        paths.append(p)
    logger.info("图件已保存: %s (%s)", name, ", ".join(f.upper() for f in formats))
    return paths


def _airport_exclusion_geometry(airports, cfg):
    """运输机场／直升机场两类净空区的**并集**几何（已投影）。

    判据与 :func:`evtol_siting.stage1_candidates.apply_safety_exclusions`
    严格一致：只对**军用**机场（``landuse=military`` / ``military=airfield``）
    施加净空区缓冲，民用运输机场及其内部设施**不排除**（eVTOL 可在民航机场
    设站中转）。直升机场按运行间隔单独处理。

    返回 ``(军用机场净空区, 直升机场净空区)``，任一为空则返回 ``None``。
    """
    if airports is None or len(airports) == 0:
        return None, None
    try:
        from shapely.ops import unary_union
    except Exception:
        return None, None

    a = airports
    try:
        if cfg is not None and str(a.crs) != str(cfg.crs_proj):
            a = a.to_crs(cfg.crs_proj)
    except Exception:
        pass

    r_ap = 4000.0
    r_hel = 1000.0
    if cfg is not None:
        r_ap = float(cfg.get("stage1_candidates.safety.airport_exclusion_radius_m", r_ap))
        r_hel = float(cfg.get("stage1_candidates.safety.existing_heliport_exclusion_m", r_hel))

    if "aeroway" in a.columns:
        hel = a["aeroway"].astype(str).isin(["helipad", "heliport"])
    else:
        hel = pd.Series(False, index=a.index)

    # 与 apply_safety_exclusions 保持同一口径（配置项
    # stage1_candidates.safety.airport_exclusion_mode，默认 all_aeroway）
    mode = "all_aeroway"
    if cfg is not None:
        mode = str(cfg.get("stage1_candidates.safety.airport_exclusion_mode",
                           "all_aeroway")).strip().lower()
    if mode == "military_only":
        mil = a.get("military", pd.Series(pd.NA, index=a.index)).astype(str).eq("airfield")
        lu = a.get("landuse", pd.Series(pd.NA, index=a.index)).astype(str).eq("military")
        airport = (~hel) & (mil | lu)
    else:
        airport = ~hel

    out = []
    for mask, r in ((airport, r_ap), (hel, r_hel)):
        sub = a[mask]
        if len(sub) == 0:
            out.append(None)
            continue
        try:
            geoms = [g.buffer(r) for g in sub.geometry
                     if g is not None and not g.is_empty]
            out.append(unary_union(geoms) if geoms else None)
        except Exception as exc:
            logger.debug("净空区并集计算失败: %s", str(exc)[:100])
            out.append(None)
    return out[0], out[1]


def _plot_zone(ax, geom, label=None, alpha=0.13, zorder=0, lw=0.7):
    """把一个（多）多边形净空区画成半透明底 + 虚线边界。"""
    if geom is None:
        return
    polys = getattr(geom, "geoms", None)
    polys = list(polys) if polys is not None else [geom]
    labelled = False
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        xs, ys = p.exterior.xy
        ax.fill([x / 1000 for x in xs], [y / 1000 for y in ys],
                facecolor=OKABE_ITO["grey"], alpha=alpha,
                edgecolor="none", zorder=zorder)
        ax.plot([x / 1000 for x in xs], [y / 1000 for y in ys],
                color=OKABE_ITO["grey"], lw=lw, ls="--", zorder=zorder + 1,
                label=None if labelled else label)
        labelled = True


# ---------------------------------------------------------------------------
# 图 2：建筑高度融合
# ---------------------------------------------------------------------------

def fig_height_fusion(height_report: pd.DataFrame, out_dir: Path,
                      tag_only: dict | None = None):
    """高度来源构成与覆盖率的提升。

    左图：各来源的栋数与占比（堆叠条）；右图：仅标签 vs 融合后的屋顶候选数。
    右图的对比是论文的关键论证——标签法会漏掉 84% 的合格屋顶。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # -- 左：来源构成 ------------------------------------------------------
    ax = axes[0]
    labels_en = {
        "raster_height": "CNBH-10 m raster",
        "imputed": "Quantile regression",
        "osm_levels": "OSM building:levels",
        "osm_height": "OSM height tag",
    }
    df = height_report.copy()
    df["label"] = df["height_source"].map(labels_en).fillna(df["height_source"])
    df = df.sort_values("count", ascending=True)
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"],
              OKABE_ITO["green"], OKABE_ITO["vermillion"]][: len(df)]

    ax.barh(df["label"], df["share_pct"], color=colors, height=0.62)
    for y, (share, cnt) in enumerate(zip(df["share_pct"], df["count"])):
        ax.text(share + 1.5, y, f"{share:.1f}%  (n={cnt:,})",
                va="center", fontsize=7.5, color="#333333")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of buildings (%)")
    ax.set_title("(a) Height source after fusion", loc="left")
    ax.xaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # -- 右：标签法 vs 融合 -------------------------------------------------
    ax = axes[1]
    if tag_only:
        # ⚠ 这里**不能**给默认值。原先写的是 `.get("tag_only", 227)`，那意味着
        #   一旦调用方忘了传，图就会用一对**写死的旧数**画出来，而正文、摘要与
        #   表 3 都引用它——重跑流水线也不会更新。真实值由
        #   `scripts/20_height_source_comparison.py` 产出并落盘到
        #   `outputs/tables/height_source_comparison.json`；缺失时应当直接报错，
        #   而不是悄悄退回旧值。
        missing = [k for k in ("tag_only", "fused") if k not in tag_only]
        if missing:
            raise KeyError(
                f"fig_height_fusion 缺少 tag_only 的键 {missing}。"
                "请先运行 scripts/20_height_source_comparison.py，"
                "并传入其输出的 height_source_comparison.json。"
                "**不允许退回硬编码默认值**——那正是本图曾经失真的原因。")
        vals = [tag_only["tag_only"], tag_only["fused"]]
        bars = ax.bar(["OSM tags\nonly", "Multi-source\nfusion"], vals,
                      color=[OKABE_ITO["grey"], OKABE_ITO["green"]], width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                    f"{v:,}", ha="center", fontsize=9, fontweight="bold")
        ratio = vals[1] / max(vals[0], 1)
        ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                    arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillion"], lw=1.2))
        ax.text(0.5, (vals[0] + vals[1]) / 2 * 1.05, f"{ratio:.1f}×",
                ha="center", fontsize=11, fontweight="bold",
                color=OKABE_ITO["vermillion"])
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_ylabel("Rooftop candidates")
        ax.set_title("(b) Candidates recovered by fusion", loc="left")
        ax.yaxis.grid(True, alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, out_dir, "fig02_height_fusion")


# ---------------------------------------------------------------------------
# 图 3：人群分类与选择模型
# ---------------------------------------------------------------------------

def fig_demand_clusters(cluster_table: pd.DataFrame, out_dir: Path,
                        fare_table: pd.DataFrame | None = None):
    """人群分层的收入—采用概率关系，以及票价情景。

    左图：各人群类的月收入 vs eVTOL 采用概率（气泡大小 = 人口）。
    右图：票价倍数对采用概率的影响——这是选择模型相对外生出行率假设的
    主要增量，因为外生假设下根本不存在这个政策杠杆。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # -- 左：收入 -> 采用概率 ----------------------------------------------
    ax = axes[0]
    d = cluster_table.sort_values("monthly_income_cny")
    sizes = 30 + 400 * (d["population"] / d["population"].max())
    ax.scatter(d["monthly_income_cny"] / 1000, d["p_evtol"] * 100,
               s=sizes, c=OKABE_ITO["blue"], alpha=0.72,
               edgecolors="white", linewidths=0.8, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f"C{int(r['cluster'])}",
                    (r["monthly_income_cny"] / 1000, r["p_evtol"] * 100),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Monthly income (thousand CNY)")
    ax.set_ylabel("P(choose eVTOL)  (%)")
    ax.set_title("(a) Adoption rises with income", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # -- 右：票价情景 ------------------------------------------------------
    ax = axes[1]
    if fare_table is not None and len(fare_table):
        ax.plot(fare_table["fare_multiplier"], fare_table["p_max"] * 100,
                "o-", color=OKABE_ITO["vermillion"], lw=1.6, ms=4,
                label="Highest-income cluster")
        ax.plot(fare_table["fare_multiplier"], fare_table["p_min"] * 100,
                "s-", color=OKABE_ITO["blue"], lw=1.6, ms=4,
                label="Lowest-income cluster")
        ax.fill_between(fare_table["fare_multiplier"],
                        fare_table["p_min"] * 100, fare_table["p_max"] * 100,
                        color=OKABE_ITO["grey"], alpha=0.16)
        ax.set_xlabel("eVTOL fare multiplier  (1.0 = base case)")
        ax.set_ylabel("P(choose eVTOL)  (%)")
        ax.set_title("(b) Fare is an equity lever", loc="left")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)

        # 标注基准票价与类间比
        base = fare_table[np.isclose(fare_table["fare_multiplier"], 1.0)]
        if len(base):
            b = base.iloc[0]
            ax.axvline(1.0, color="#888888", ls=":", lw=0.9)
            ax.text(1.02, ax.get_ylim()[1] * 0.5,
                    f"base case\nratio {b['p_ratio_max_min']:.2f}×",
                    fontsize=7.5, color="#555555")

    fig.tight_layout()
    return _save(fig, out_dir, "fig03_demand_clusters")


# ---------------------------------------------------------------------------
# 图 4：帕累托前沿
# ---------------------------------------------------------------------------

def fig_pareto(pareto: pd.DataFrame, out_dir: Path, knee_idx: int | None = None):
    """四目标的帕累托前沿（两两投影）。

    四目标无法直接画，因此用 2×2 的两两投影，并在每格标出该目标对的
    相关系数——这能直观显示哪些目标之间存在真实权衡（负相关）而哪些
    近乎冗余（正相关）。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    cols = [c for c in ("unserved_demand", "total_access_time", "inequity", "total_cost")
            if c in pareto.columns]
    labels = {
        "unserved_demand": "Unserved demand\n(trips/day)",
        "total_access_time": "Total access time\n(min·trips/day)",
        "inequity": "Inequity\n(min)",
        "total_cost": "Total cost\n(10⁸ CNY)",
    }
    scale = {"total_cost": 1e8, "total_access_time": 1e0}

    # 布局：上三角 n(n-1)/2 个面板。用 GridSpec 只在实际使用的位置建轴，
    # 避免 3×3 网格里空出 4 格、左下留一大片白（初版就是这样）。
    n = len(cols)
    npairs = n * (n - 1) // 2
    ncol = n - 1
    nrow = int(np.ceil(npairs / ncol))
    fig = plt.figure(figsize=(2.75 * ncol + 0.8, 2.55 * nrow + 0.6))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.62, wspace=0.55)

    axes: dict[tuple[int, int], object] = {}
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # GridSpec 必须用元组索引 gs[r, c]；链式 gs[r][c] 会抛
            # "SubplotSpec object is not subscriptable"。
            axes[(i, j)] = fig.add_subplot(gs[k // ncol, k % ncol])
            k += 1

    for i in range(n - 1):
        for j in range(i + 1, n):
            ax = axes[(i, j)]
            x = pareto[cols[j]].to_numpy(dtype=float) / scale.get(cols[j], 1.0)
            y = pareto[cols[i]].to_numpy(dtype=float) / scale.get(cols[i], 1.0)
            ax.scatter(x, y, s=9, c=OKABE_ITO["blue"], alpha=0.55,
                       edgecolors="none", zorder=3)
            if knee_idx is not None and 0 <= knee_idx < len(x):
                ax.scatter(x[knee_idx], y[knee_idx], s=60, marker="*",
                           c=OKABE_ITO["vermillion"], zorder=5,
                           edgecolors="white", linewidths=0.6,
                           label="knee point" if (i == 0 and j == 1) else None)
            r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
            ax.text(0.04, 0.93, f"r = {r:+.2f}", transform=ax.transAxes,
                    fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#CCCCCC", lw=0.5))
            ax.set_xlabel(labels.get(cols[j], cols[j]), fontsize=7.5)
            ax.set_ylabel(labels.get(cols[i], cols[i]), fontsize=7.5)
            ax.grid(True, alpha=0.45)
            ax.set_axisbelow(True)

    axes[(0, 1)].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Pareto front: pairwise objective projections", fontsize=10, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig04_pareto_front")


# ---------------------------------------------------------------------------
# 图 5：收敛曲线
# ---------------------------------------------------------------------------

def fig_convergence(conv: pd.DataFrame, out_dir: Path):
    """超体积随代数的变化，用于证明算法确实在收敛（而非开局即停滞）。"""
    import matplotlib.pyplot as plt

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    ax = axes[0]
    hv = conv["hypervolume"].to_numpy(dtype=float)
    ax.plot(conv["generation"], hv / max(hv.max(), 1e-12),
            color=OKABE_ITO["blue"], lw=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalised hypervolume")
    ax.set_title("(a) Convergence", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(conv["generation"], conv["n_pareto"],
            color=OKABE_ITO["green"], lw=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Non-dominated solutions")
    ax.set_title("(b) Front size", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, out_dir, "fig05_convergence")


# ---------------------------------------------------------------------------
# 图 6：机场缓冲敏感性（关键图）
# ---------------------------------------------------------------------------

def fig_airport_sensitivity(sens: pd.DataFrame, out_dir: Path):
    """机场净空区半径对候选集、覆盖率与公平性的影响。

    这是全文最有分量的一张敏感性图：它显示一个通常被当作"约定"的参数
    如何几乎单独决定了结论。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    # ⚠ 必须**先按参数过滤**。调用方传进来的是整张 sensitivity.csv（15 个参数、
    # 76 行），不过滤的话 14 个参数的数据会全部画进这张"机场半径"图：x 轴
    # value/1000 会混入 core_demand=0.3、building_coverage=0.40、roof_area=1400
    # 等不同量纲的取值，前十几个点全挤在 x≈0，曲线呈锯齿状——**整张图是废的**，
    # 而图注写的却是正确的机场半径结论（正文对、图错）。
    if "parameter" in sens.columns:
        d = sens[sens["parameter"].astype(str) == "airport_buffer"].copy()
    else:
        d = sens.copy()
    d = d[d.get("ok", True) == True]  # noqa: E712
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"]).sort_values("value")

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))

    # (a) 候选集
    ax = axes[0]
    ax.plot(d["value"] / 1000, d["n_candidates"], "o-", color=OKABE_ITO["blue"],
            lw=1.6, ms=4, label="All")
    ax.plot(d["value"] / 1000, d["n_rooftop"], "s-", color=OKABE_ITO["green"],
            lw=1.6, ms=4, label="Rooftop")
    ax.set_xlabel("Airport exclusion radius (km)")
    ax.set_ylabel("Candidates")
    ax.set_title("(a) Candidate supply", loc="left")
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # (b) 覆盖率
    ax = axes[1]
    if "knee_demand_coverage_pct" in d.columns:
        ax.plot(d["value"] / 1000, d["knee_demand_coverage_pct"], "o-",
                color=OKABE_ITO["vermillion"], lw=1.6, ms=4)
        # ⚠ 基准是 **4 km**（config/chengdu.yaml），不是 8 km。
        # 原来画在 8 km 处还标 "base case"，与图注"基准取值 4 km"直接矛盾。
        ax.axvline(4.0, color="#888888", ls=":", lw=0.9)
        ax.text(4.15, 30, "base case", fontsize=7.5, color="#555555")
    ax.set_xlabel("Airport exclusion radius (km)")
    ax.set_ylabel("Demand coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Coverage", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    # (c) 公平性
    ax = axes[2]
    if "knee_worst_group_access_time_min" in d.columns:
        y = pd.to_numeric(d["knee_worst_group_access_time_min"], errors="coerce")
        y = y.where(y < 1e6)
        ax.plot(d["value"] / 1000, y, "o-", color=OKABE_ITO["purple"],
                lw=1.6, ms=4)
        ax.axvline(8.0, color="#888888", ls=":", lw=0.9)
    ax.set_xlabel("Airport exclusion radius (km)")
    ax.set_ylabel("Worst-group access (min)")
    ax.set_title("(c) Equity", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, out_dir, "fig06_airport_buffer_sensitivity")


# ---------------------------------------------------------------------------
# 图 7：方案对比
# ---------------------------------------------------------------------------

def fig_solution_comparison(metrics: pd.DataFrame, out_dir: Path):
    """各方案在覆盖率、成本、公平性三个维度上的对比。

    用散点而非柱状图：三个量纲不同，散点能同时表达"谁在哪个维度更好"，
    而柱状图需要归一化才能并列，容易误导。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    d = metrics.copy()
    d = d[d["n_sites"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    is_nsga = d["label"].str.startswith("NSGA-II")
    is_base = d["label"].str.startswith("Baseline")
    is_ip = d["label"].str.startswith("IP:")

    def _scatter(ax, xcol, ycol, xlab, ylab):
        for mask, color, marker, name in (
            (is_ip, OKABE_ITO["orange"], "D", "IP scenarios"),
            (is_nsga, OKABE_ITO["blue"], "o", "NSGA-II"),
            (is_base, OKABE_ITO["grey"], "^", "Single-objective baselines"),
        ):
            sub = d[mask]
            if len(sub):
                ax.scatter(sub[xcol], sub[ycol], s=46, c=color, marker=marker,
                           alpha=0.8, edgecolors="white", linewidths=0.7,
                           label=name, zorder=3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)

    if {"total_cost_cny", "demand_coverage_pct"} <= set(d.columns):
        _scatter(axes[0], "total_cost_cny", "demand_coverage_pct",
                 "Total cost (10⁸ CNY)", "Demand coverage (%)")
        axes[0].set_xlim(left=0)
        axes[0].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e8:.1f}"))
        axes[0].set_title("(a) Cost vs coverage", loc="left")

    if {"worst_group_access_time_min", "demand_coverage_pct"} <= set(d.columns):
        _scatter(axes[1], "worst_group_access_time_min", "demand_coverage_pct",
                 "Worst-group access time (min)", "Demand coverage (%)")
        axes[1].set_title("(b) Equity vs coverage", loc="left")

    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "fig07_solution_comparison")


# ---------------------------------------------------------------------------
# 图 8：接驳模型
# ---------------------------------------------------------------------------

def fig_choice_model_mechanism(cfg, out_dir: Path):
    """选择模型的机制图：广义费用随收入的变化与盈亏平衡点。

    这张图解释"为什么"——两条广义费用曲线在何处相交，交点即盈亏平衡
    收入，交点右侧的人群才会选择 eVTOL。
    """
    import matplotlib.pyplot as plt

    from ..choice_model import EVTOLChoiceModel

    _setup_style()
    model = EVTOLChoiceModel(cfg)
    t = model.travel_times()
    f = model.fares()

    incomes = np.linspace(1_000, 60_000, 300)
    vot = model.value_of_time(incomes)
    gc_evtol = f["fare_evtol"] + vot * (t["t_evtol_min"] / 60.0)
    gc_ground = f["fare_ground"] + vot * (t["t_ground_min"] / 60.0)
    p = model.choice_probability(vot)
    be = model.break_even_income()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    ax.plot(incomes / 1000, gc_evtol, color=OKABE_ITO["vermillion"], lw=1.6,
            label="eVTOL")
    ax.plot(incomes / 1000, gc_ground, color=OKABE_ITO["blue"], lw=1.6,
            label="Ground transport")
    if np.isfinite(be) and be < incomes[-1]:
        ax.axvline(be / 1000, color="#888888", ls=":", lw=1.0)
        ax.text(be / 1000 + 0.6, ax.get_ylim()[0] + 0.28 * np.ptp(ax.get_ylim()),
                f"break-even\n{be/1000:.1f}k CNY/month", fontsize=7.5, color="#555555")
    ax.set_xlabel("Monthly income (thousand CNY)")
    ax.set_ylabel("Generalised cost (CNY)")
    ax.set_title("(a) Two cost curves cross", loc="left")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(incomes / 1000, p * 100, color=OKABE_ITO["green"], lw=1.8)
    ax.axhline(50, color="#AAAAAA", ls="--", lw=0.8)
    ax.set_xlabel("Monthly income (thousand CNY)")
    ax.set_ylabel("P(choose eVTOL)  (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Resulting choice probability", loc="left")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, out_dir, "fig08_choice_model")


# ---------------------------------------------------------------------------
# 图 1：研究区与候选分布（地图）
# ---------------------------------------------------------------------------

def fig_study_area(cfg, candidates: pd.DataFrame, grid: pd.DataFrame,
                   airports=None, selected_ids=None, out_dir: Path = None):
    """研究区、候选起降场与选中方案的平面图。"""
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(5.4, 6.2))

    # 需求格网（底色表示密度）
    if grid is not None and "population" in grid.columns:
        ax.scatter(grid["centroid_x"] / 1000, grid["centroid_y"] / 1000,
                   s=1.2, c="#EEEEEE", marker="s", linewidths=0, zorder=1)

    # 机场净空区：按 apply_safety_exclusions 的同一判据取并集后再画。
    # 旧实现是逐要素画圆——该图层有 1 364 个要素（多为滑行道、停机位、
    # 登机口），叠出上千个半透明大圆，会把底下的候选点整片盖住。
    zone_ap, zone_hel = _airport_exclusion_geometry(airports, cfg)
    _plot_zone(ax, zone_ap, label="Airport clear zone", alpha=0.12, zorder=2)
    _plot_zone(ax, zone_hel, label="Heliport buffer", alpha=0.16, zorder=2)

    # 候选
    if candidates is not None and len(candidates):
        roof = candidates[candidates["facility_type"] == "rooftop"]
        gnd = candidates[candidates["facility_type"] == "ground"]
        ax.scatter(gnd["x"] / 1000, gnd["y"] / 1000, s=5, c=OKABE_ITO["sky"],
                   alpha=0.75, linewidths=0, label=f"Ground ({len(gnd)})", zorder=3)
        ax.scatter(roof["x"] / 1000, roof["y"] / 1000, s=9, c=OKABE_ITO["green"],
                   alpha=0.9, linewidths=0, label=f"Rooftop ({len(roof)})", zorder=4)

    # 选中方案
    if selected_ids is not None and candidates is not None:
        sel = candidates[candidates["cand_id"].isin(selected_ids)]
        ax.scatter(sel["x"] / 1000, sel["y"] / 1000, s=78, marker="*",
                   c=OKABE_ITO["vermillion"], edgecolors="white", linewidths=0.7,
                   label=f"Selected ({len(sel)})", zorder=6)

    ax.set_xlabel("Easting (km, UTM 48N)")
    ax.set_ylabel("Northing (km, UTM 48N)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_dir, "fig01_study_area")


def fig_accessibility(cfg, grid: pd.DataFrame, candidates: pd.DataFrame,
                      access_time_s: np.ndarray, selected_ids, out_dir: Path):
    """选中方案下的接驳时间空间分布。

    这是论文中最能说明"公平性"的图：把每个需求单元到最近起降场的接驳
    时间画成热力图，高值区即服务盲区。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    sel_pos = candidates.reset_index(drop=True)
    sel_mask = sel_pos["cand_id"].isin(selected_ids).to_numpy()
    if sel_mask.sum() == 0:
        return []

    sub = access_time_s[:, sel_mask]
    best = np.nanmin(np.where(np.isfinite(sub), sub, np.nan), axis=1) / 60.0
    best = np.where(np.isfinite(best), best, np.nan)

    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    m = ax.scatter(grid["centroid_x"] / 1000, grid["centroid_y"] / 1000,
                   c=best, s=6, cmap="viridis_r", vmin=0,
                   vmax=np.nanpercentile(best, 95), linewidths=0)
    cb = fig.colorbar(m, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Access time to nearest vertiport (min)", fontsize=8)

    sel = sel_pos[sel_mask]
    ax.scatter(sel["x"] / 1000, sel["y"] / 1000, s=62, marker="*",
               c=OKABE_ITO["vermillion"], edgecolors="white", linewidths=0.7,
               label="Selected vertiport", zorder=5)

    n_unreach = int(np.isnan(best).sum())
    ax.text(0.02, 0.02, f"Unreachable cells: {n_unreach} / {len(grid)}",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.5))
    ax.set_xlabel("Easting (km, UTM 48N)")
    ax.set_ylabel("Northing (km, UTM 48N)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "fig09_accessibility")


# ---------------------------------------------------------------------------
# 图：需求按收入人群类的空间分布
# ---------------------------------------------------------------------------

# 按收入**升序**分配，故最后一个（vermillion）恒为最高收入类。
# 相邻两色在 Okabe–Ito 调色板中色相间隔尽可能大，灰度和三类色觉障碍下均可区分。
_GROUP_COLORS = ["#56B4E9", "#0072B2", "#009E73", "#F0E442",
                 "#E69F00", "#CC79A7", "#D55E00"]


def fig_demand_by_group(grid: pd.DataFrame, out_dir: Path, cfg=None,
                        airports=None):
    """需求按收入人群类的空间分布，以及同一批单元的需求强度。

    论文 §5.2 的核心事实是"空间集中度极高"：收入最高的一类只占 **13 个单元**
    （研究区的 0.5 %），却拥有最高的 eVTOL 采用概率；而收入最低的一类横跨
    **1 391 个单元**。该事实支撑全文的公平性主张，因此值得单独成图，而不是
    只以表格呈现。

    左图按人群类着色（**不同颜色代表不同人群**），图例同时给出该类的月收入与
    采用概率；右图给出同一批单元的需求强度。两图对照可见：需求图由广阔的
    低采用率外围主导，而**可服务需求是一个小而高度集中的核心**——这正是
    公平目标与效率目标发生分歧的配置。

    参数
    ----
    grid
        需求格网，需含 ``centroid_x`` / ``centroid_y`` / ``cluster`` /
        ``demand`` / ``monthly_income_cny`` / ``p_evtol``。
    airports
        可选；机场点数据，用于叠加净空区。
    """
    import matplotlib.pyplot as plt

    _setup_style()
    d = grid.copy()
    for col in ("centroid_x", "centroid_y", "cluster", "demand"):
        if col not in d.columns:
            logger.warning("需求人群图缺少列 %s，跳过", col)
            return []

    # 人群类按收入升序，保证配色与"低收入→高收入"一致
    prof = (d.groupby("cluster")
              .agg(income=("monthly_income_cny", "median"),
                   p=("p_evtol", "median"),
                   n=("cluster", "size"))
              .sort_values("income"))
    order = list(prof.index)
    colors = {c: _GROUP_COLORS[i % len(_GROUP_COLORS)]
              for i, c in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.6))

    # -- (a) 按人群类着色 --------------------------------------------------
    ax = axes[0]
    for c in order:
        s = d[d["cluster"] == c]
        ax.scatter(s["centroid_x"] / 1000, s["centroid_y"] / 1000,
                   s=7, c=colors[c], linewidths=0, alpha=0.9,
                   label=f"C{int(c)}  ¥{prof.loc[c, 'income'] / 1000:.1f}k"
                         f"  ({prof.loc[c, 'p'] * 100:.2f} %)")
    # -- (b) 需求强度 ------------------------------------------------------
    ax = axes[1]
    dm = d["demand"].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(dm, 95)) if np.isfinite(dm).any() else None
    m = ax.scatter(d["centroid_x"] / 1000, d["centroid_y"] / 1000,
                   s=7, c=dm, cmap="viridis", vmin=0, vmax=vmax, linewidths=0)
    cb = fig.colorbar(m, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("eVTOL demand (trips/day per cell)", fontsize=7.5)

    # -- 两者共用的机场净空区叠加（并集，与模型同一判据） -----------------
    zone_ap, zone_hel = _airport_exclusion_geometry(airports, cfg)
    for i, a in enumerate(axes):
        _plot_zone(a, zone_hel, alpha=0.16, zorder=0)
        _plot_zone(a, zone_ap, alpha=0.13, zorder=0,
                   label="Airport clear zone" if i == 1 else None)

    # (a) 的图例：人群类（放在叠加之后，避免被净空区遮住标题行）
    axes[0].legend(title="Income group (monthly, adoption)", loc="upper left",
                   fontsize=6.4, title_fontsize=6.6, markerscale=1.6,
                   labelspacing=0.32, handletextpad=0.4)
    # 净空区图例放在 (b) 的右下留白处
    h, l = axes[1].get_legend_handles_labels()
    if h:
        axes[1].legend(h, l, loc="lower right", fontsize=7)

    for a, t in zip(axes, ("(a) Demand cells by income group",
                           "(b) eVTOL demand intensity")):
        a.set_title(t, loc="left")
        a.set_xlabel("Easting (km, UTM 48N)")
        a.set_aspect("equal")
        a.grid(True, alpha=0.4)
        a.set_axisbelow(True)
    axes[0].set_ylabel("Northing (km, UTM 48N)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig11_demand_by_group")
