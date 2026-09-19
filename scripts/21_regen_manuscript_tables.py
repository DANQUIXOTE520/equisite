#!/usr/bin/env python
"""从产物直接重生成正文中的 Markdown 表格。

为什么需要它
------------
论文正文的表格是**手抄**产物的。这一路审计出的问题几乎都由这条路径产生：
`1324` 这个候选数出现在 65 处、改一次要动 130 个位置；前沿极值改了之后，
"两条选点规则选中同一个解"那句脚注还在；IP 情景的站数从 10 变 11 之后，
"同样的 10 个站"那句论断还在。**手抄必然漂移**。

本脚本把"表 N ← 产物 CSV"的映射写成代码，使表格可一键重生成、且与产物逐位一致。

设计
----
* 只替换**表格体**（连续的 `|` 行），不碰表题、不碰正文。
* `--dry-run` 打印逐行差异，供人工确认后再写。
* 若产物缺列或行数变化，**报错而不是静默生成一张残缺的表**。

用法::

    python scripts/21_regen_manuscript_tables.py --dry-run
    python scripts/21_regen_manuscript_tables.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from evtol_siting.config import load_config                      # noqa: E402

MANU = Path(__file__).resolve().parents[2] / "论文稿件"

# 表号 -> 规格。caption 用于定位；columns 是 (产物列, 表头文本) 的有序映射。
# 参数键（sensitivity.csv 里的 `parameter` 列）→ 两稿的显示名。
# 表 26 与表 34 共用一份，避免两处各写一份、改一处漏一处。
PARAM_ZH = {
    "roof_area": "屋顶面积阈值 (m²)", "roof_height": "屋顶高度阈值 (m)",
    "building_coverage": "地面建筑覆盖率上限", "dilution": "候选最小间距 (m)",
    "airport_buffer": "机场排除半径 (m)", "n_clusters": "聚类数 K",
    "access_budget": "接驳时间预算 (min)", "core_demand": "核心需求分位数",
    "cost_scale": "成本缩放系数", "trip_rate_ratio": "出行率比 ρ",
    "trip_distance": "出行距离结构", "income_median": "收入阶梯中位数 (元/月)",
    "income_sigma": "收入离散度 σ", "population": "人口来源与量级",
    "equity_measure": "公平性度量（定义）",
}
PARAM_EN = {
    "roof_area": "Roof-area threshold (m²)",
    "roof_height": "Roof-height threshold (m)",
    "building_coverage": "Ground building-coverage cap",
    "dilution": "Minimum candidate spacing (m)",
    "airport_buffer": "Airport exclusion radius (m)",
    "n_clusters": "Cluster count K",
    "access_budget": "Access-time budget (min)",
    "core_demand": "Core-demand quantile",
    "cost_scale": "Cost-scaling factor",
    "trip_rate_ratio": "Trip-rate ratio rho",
    "trip_distance": "Trip-distance structure",
    "income_median": "Income-ladder median (CNY/month)",
    "income_sigma": "Income dispersion sigma",
    "population": "Population source and scale",
    "equity_measure": "Equity measure (definition)",
}


def _fmt_int(v) -> str:
    """本项目的数字排版约定：千位用**空格**分隔（`1 865`），不是逗号。"""
    return f"{float(v):,.0f}".replace(",", " ")


def build_table26(_cfg) -> pd.DataFrame:
    """表 26 的源：把单因素扫描**按参数汇总**。

    此前这张表是**手抄**的，于是只有一部分行跟着上游更新过——实测 15 行里
    有 7 行还是旧口径（如"公平性度量 6.30 → 14.82（135 %）"实为
    "6.00 → 14.85（147 %）"）。这类"部分更新"正是本项目最难发现的一类缺陷：
    表看着是新的，逐行查才发现一半是旧的。故改为从 `sensitivity.csv` 汇总。
    """
    import ast as _ast
    d = pd.read_csv(Path(_cfg.dir("output.tables_dir")) / "sensitivity.csv")
    reps = d["n_reps"].fillna(1)
    d = d[reps == reps.max()]          # 只用重复次数最高的那一批，避免混协议
    rows = []
    for p, g in d.groupby("parameter"):
        lo = float(g["pareto_inequity_min"].min())
        hi = float(g["pareto_inequity_min"].max())
        ratio = float(g["total_demand"].max()) / float(g["total_demand"].min())
        bad: list[str] = []
        for r in g.itertuples():
            try:
                st = _ast.literal_eval(str(r.ip_status))
            except (ValueError, SyntaxError):
                continue
            if isinstance(st, dict) and any(v != "Optimal" for v in st.values()):
                bad.append(_norm_key(r.value))
        bad = sorted(set(bad), key=lambda s: float(s) if s.replace(".", "").isdigit() else 0.0)
        pct = 100 * (hi - lo) / lo if lo else float("nan")
        # 百分比取整后为 0 的行不写成"6.00 → 6.00（0 %）"——那看着像排版错误。
        if pct < 0.5:
            mv_zh, mv_en = f"**{hi:.2f}（0 %）**", f"**{hi:.2f} (0 %)**"
        else:
            mv_zh = f"{lo:.2f} → **{hi:.2f}**（**{pct:.0f} %**）"
            mv_en = f"{lo:.2f} → **{hi:.2f}** min (**{pct:.0f} %**)"
        rows.append({
            "param_zh": PARAM_ZH.get(p, p), "param_en": PARAM_EN.get(p, p),
            "n_values": len(g), "_pct": pct,
            "movement_zh": mv_zh, "movement_en": mv_en,
            "demand_zh": f"**{ratio:.2f} 倍**" if ratio > 1.001 else "—",
            "demand_en": f"**{ratio:.2f}×**" if ratio > 1.001 else "—",
            "feas_zh": (f"**是**（{'、'.join(bad)} 时 IP 无解）" if bad else "否"),
            "feas_en": (f"**yes** (no IP solution at {'/'.join(bad)})" if bad
                        else "no"),
        })
    return (pd.DataFrame(rows).sort_values("_pct", ascending=False)
            .reset_index(drop=True))


def build_table30(_cfg) -> pd.DataFrame:
    """表 30 的源：十次独立运行的算法指标与决策指标。

    此前这张表是**手抄**的，实测八行里七行与产物对不上——超体积
    5.203×10¹⁹（产物 4.618×10¹⁹）、间距 0.0509（产物 0.0517）、
    最优解成本 4.05–5.14 亿（产物 4.41–5.05 亿）等。整张表看起来
    "就是一组统计量"，最容易被原样留下不查，故纳入重生成。

    源有两个，出自 ``08_multirun.py`` 的**同一次 10 次运行**：

    * ``multirun_summary.json`` —— 算法指标（由 ``multirun_metrics.csv``
      汇总；超体积的参考点跨运行固定，见该脚本的 ``ref_point``）；
    * ``multirun_runs.csv`` —— 每次运行的**前沿最优**决策指标
      （``best_unserved`` / ``best_cost`` / ``best_inequity``）。

    ⚠ 后三行取的是"该次运行前沿上的最小值"，**不是拐点解**——拐点解的
    跨运行分布由 ``multirun_knee.csv`` 给出，量级完全不同（成本 4.67–61.16
    亿元），两者不可混用。
    """
    tab = Path(_cfg.dir("output.tables_dir"))
    m = json.loads((tab / "multirun_summary.json").read_text(encoding="utf-8"))["metrics"]
    runs = pd.read_csv(tab / "multirun_runs.csv")
    E = 1e19

    def _ms(key: str, scale: float, dec: int, tail: str = "") -> str:
        return (f"{m[key]['mean'] / scale:.{dec}f} ± "
                f"{m[key]['std'] / scale:.{dec}f}{tail}")

    def _rng(key: str, scale: float, dec: int, tail: str = "") -> str:
        return (f"{m[key]['min'] / scale:.{dec}f} – "
                f"{m[key]['max'] / scale:.{dec}f}{tail}")

    def _cv_pct(key: str) -> float:
        return 100 * m[key]["std"] / m[key]["mean"]

    n_zero = int((runs["best_unserved"] == 0).sum())
    worst_un = float(runs["best_unserved"].max())
    c_lo, c_hi = runs["best_cost"].min() / 1e8, runs["best_cost"].max() / 1e8
    c_gap = (c_hi - c_lo) / c_lo * 100
    i_lo, i_hi = runs["best_inequity"].min(), runs["best_inequity"].max()

    # (键, 中文名, 英文名, 量纲, ms 位数, range 位数, 方向)
    METRICS = [
        ("hypervolume", "超体积", "Hypervolume", E, 3, 3, "越大越好", "larger"),
        ("spacing", "间距 (SP)", "Spacing (SP)", 1.0, 4, 4, "越小越好", "smaller"),
        ("spread", "分布度 (Δ)", "Spread (Δ)", 1.0, 3, 3, "越小越好", "smaller"),
        ("igd", "IGD", "IGD", 1.0, 4, 4, "越小越好", "smaller"),
    ]
    # 加粗规则：**只加粗变异系数最小的那一行**——正文的论点是"前沿的
    # *质量*很稳定"，靠的就是这个对比（超体积 1.6 % vs 间距 12.3 %）。
    # 旧表只加粗了超体积，但那是手写的；这里让规则自维护，取值变了也不会错位。
    best_cv_key = min((row[0] for row in METRICS), key=_cv_pct)

    rows: list[dict] = []
    for key, zh, en, sc, d_ms, d_rg, dir_zh, dir_en in METRICS:
        tail = " × 10¹⁹" if sc == E else ""
        cv_txt = f"{_cv_pct(key):.1f} %"
        if key == best_cv_key:
            cv_txt = f"**{cv_txt}**"
        rows.append({
            "metric_zh": zh, "metric_en": en,
            "ms_zh": _ms(key, sc, d_ms, tail), "ms_en": _ms(key, sc, d_ms, tail),
            "range_zh": _rng(key, sc, d_rg, tail), "range_en": _rng(key, sc, d_rg, tail),
            "cv_zh": cv_txt, "cv_en": cv_txt,
            "dir_zh": dir_zh, "dir_en": dir_en,
        })
    rows.append({
        "metric_zh": "前沿规模", "metric_en": "Front size",
        "ms_zh": "100 ± 0", "ms_en": "100 ± 0",
        "range_zh": "100 – 100", "range_en": "100 – 100",
        "cv_zh": "—", "cv_en": "—", "dir_zh": "—", "dir_en": "—",
    })
    rows.append({
        "metric_zh": "最优未服务需求（次/日）",
        "metric_en": "Best unserved demand (trips/day)",
        "ms_zh": "—", "ms_en": "—",
        "range_zh": f"**十次中 {n_zero} 次为 0.0**，最差 {worst_un:.2f}",
        "range_en": f"**0.0** in {n_zero} of 10 runs; worst {worst_un:.2f}",
        "cv_zh": "—", "cv_en": "—",
        "dir_zh": "越小越好", "dir_en": "smaller",
    })
    rows.append({
        # 中文用亿元、英文用百万元——与两稿既有的单位习惯一致（表 33 同此）。
        "metric_zh": "最优解成本（亿元）", "metric_en": "Best solution cost (M CNY)",
        "ms_zh": "—", "ms_en": "—",
        "range_zh": f"{c_lo:.2f} – {c_hi:.2f}（相差 **{c_gap:.1f} %**）",
        "range_en": f"{c_lo * 100:.0f} – {c_hi * 100:.0f} (**{c_gap:.1f} %** apart)",
        "cv_zh": "—", "cv_en": "—",
        "dir_zh": "越小越好", "dir_en": "smaller",
    })
    rows.append({
        "metric_zh": "最优最差人群接驳（分钟）",
        "metric_en": "Best worst-group access (min)",
        "ms_zh": "—", "ms_en": "—",
        "range_zh": f"{i_lo:.2f} – {i_hi:.2f}", "range_en": f"{i_lo:.2f} – {i_hi:.2f}",
        "cv_zh": "—", "cv_en": "—",
        "dir_zh": "越小越好", "dir_en": "smaller",
    })
    return pd.DataFrame(rows)


def build_table33(_cfg) -> pd.DataFrame:
    """附录 C 的成本分位表：由 `pareto_front.csv` 直接算分位，不再手抄。

    旧表（0 % 行 312.8 M / 469 162 / 10.53）与当前前沿对不上——那是更早口径的
    数值。分位表看起来"就是个分位表"，最容易整张留下来不查，故也纳入重生成。
    """
    d = pd.read_csv(Path(_cfg.dir("output.data_dir")) / "pareto_front.csv")
    d = d.sort_values("total_cost_cny").reset_index(drop=True)
    n = len(d)
    rows = []
    for q in (0, 10, 25, 50, 75, 100):
        r = d.iloc[min(int(round(q / 100 * (n - 1))), n - 1)]
        rows.append({"q": q,
                     "cost": float(r["total_cost_cny"]) / 1e6,
                     "unserved": float(r["unserved_demand"]),
                     "time": float(r["total_access_time"]),
                     "inequity": float(r["inequity"])})
    return pd.DataFrame(rows)


def _sp(v, dec: int = 0) -> str:
    """千位用**空格**分隔（本项目的排版约定是 `1 865`，不是 `1,865`）。"""
    return f"{float(v):,.{dec}f}".replace(",", " ")


def _tab(cfg) -> Path:
    return Path(cfg.dir("output.tables_dir"))


def _dat(cfg) -> Path:
    return Path(cfg.dir("output.data_dir"))


def build_table16(cfg) -> pd.DataFrame:
    """表 16：四约束情景 vs 自由规模杂交。

    两行三个量分别来自**两个**产物：情景侧取 `scenario_family.csv` 的列最小值，
    杂交侧取 `scenario_ga_runs.csv`（5 次运行）的最小值，"解数"则取
    `scenario_ga_summary.json` 的**并集**非支配解数（`front_merged_size`，
    不是单次运行的 100）。三者不同源，最容易只更新一处。
    """
    sf = pd.read_csv(_tab(cfg) / "scenario_family.csv")
    ga = pd.read_csv(_tab(cfg) / "scenario_ga_runs.csv")
    sm = json.loads((_tab(cfg) / "scenario_ga_summary.json").read_text(encoding="utf-8"))
    w_min = float(ga["min_inequity"].min())
    rows = [
        ("四约束情景", "Four constraint scenarios", len(sf),
         float(sf["total_cost_cny"].min()), float(sf["unserved_demand"].min()),
         float(sf["worst_group_access_time_min"].min()), False),
        ("自由规模杂交（以四情景为初始种群）", "Free-cardinality hybridisation",
         int(sm["front_merged_size"]),
         float(ga["min_cost_1e8"].min()) * 1e8, float(ga["min_unserved"].min()),
         w_min, True),
    ]
    out = []
    for zh, en, n, cost, uns, w, bold_w in rows:
        # 加粗只落在杂交行的最差人群上——正文的论点正是"杂交在公平性维度上外推"。
        # 成本与未服务两列两行持平，加粗任何一方都会读成"更优"。
        wtxt = f"**{w:.2f}**" if bold_w else f"{w:.2f}"
        out.append({
            "m_zh": zh, "m_en": en, "n": n,
            "cost_zh": f"{cost / 1e8:.2f}", "cost_en": f"{cost / 1e6:.0f}",
            "uns": f"{uns:.1f}",
            "w_zh": wtxt, "w_en": wtxt,
        })
    return pd.DataFrame(out)


def build_table17(cfg) -> pd.DataFrame:
    """表 17：S2 服务标准约束型 vs TOPSIS 等权推荐。

    两行来自**不同产物**：S2 行取自 `scenario_family.csv`；TOPSIS 行取自
    `scenario_ga_summary.json` 的 `recommended_objectives`
    （顺序 = 未服务、总接驳、最差人群、成本）——`scenario_comparison.csv`
    **没有**未服务与总接驳两列，从那里取会 KeyError。两者同属 §5.4，
    加粗规则是**逐列取优**（成本与总接驳 S2 好，覆盖率、未服务、最差人群
    TOPSIS 好）——这正是"两者互不支配"的可视化，不能只加粗一行的最优。
    """
    sf = pd.read_csv(_tab(cfg) / "scenario_family.csv")
    sm = json.loads((_tab(cfg) / "scenario_ga_summary.json").read_text(encoding="utf-8"))
    sc = pd.read_csv(_tab(cfg) / "scenario_comparison.csv")
    s2 = sf[sf["name"] == "服务标准约束型"].iloc[0]
    tp = sc[sc["label"] == "TOPSIS 推荐方案"].iloc[0]
    uns, tat, ineq, cost = sm["recommended_objectives"]

    def row(zh, en, n, cost_v, cov, uns_v, tat_v, w, bold):
        # ⚠ 加粗必须**逐列**判定：早前写成 `b = lambda v: ... if bold else v`
        #   再套到整行，于是 `bold` 集里没有的列也被加粗了。
        def b(key, text):
            return f"**{text}**" if key in bold else text
        return {
            "p_zh": zh, "p_en": en, "n": int(n),
            "cost_zh": b("cost", f"{cost_v / 1e8:.2f}"),
            "cost_en": b("cost", _sp(cost_v / 1e6)),
            "cov": b("cov", f"{cov:.2f} %"), "uns": b("uns", f"{uns_v:.1f}"),
            "tat": b("tat", _sp(tat_v)), "w": b("w", f"{w:.2f}"),
        }
    return pd.DataFrame([
        row("S2 服务标准约束型", "S2 Access-standard-constrained", s2["n_sites"],
            s2["total_cost_cny"], s2["demand_coverage_pct"], s2["unserved_demand"],
            s2["total_access_time"], s2["worst_group_access_time_min"],
            {"cost", "tat"}),
        row("TOPSIS 等权推荐", "TOPSIS, equal weights", tp["n_sites"],
            float(cost), tp["demand_coverage_pct"], float(uns), float(tat),
            float(ineq), {"cov", "uns", "w"}),
    ])


def build_table18(cfg) -> pd.DataFrame:
    """表 18：三种前沿选点规则。源 `table_solution_metrics.csv`。

    ⚠ 三条规则里"成本最小"与"TOPSIS"分别等同 §5.3 的精确 IP 解与 §5.5 的
    TOPSIS 解，因此本表与表 24 的对应行必须同进同退。
    """
    d = pd.read_csv(_dat(cfg) / "table_solution_metrics.csv").set_index("label")
    picks = [
        ("成本最小（精确 IP，§5.3）", "Cost-minimal (exact IP, §5.3)",
         "NSGA-II:best_total_cost", {"cost", "w"}),
        ("拐点（最大偏离）", "Knee point (maximum deviation)", "NSGA-II:knee", set()),
        ("TOPSIS（等权）", "TOPSIS (equal weights)", "NSGA-II:TOPSIS",
         {"cov", "w"}),
    ]
    rows = []
    for zh, en, key, bold in picks:
        r = d.loc[key]

        def b(k, text):
            return f"**{text}**" if k in bold else text

        rows.append({
            "r_zh": zh, "r_en": en, "n": int(r["n_sites"]),
            # ⚠ 成本列用**百万元、一位小数**：产物 6.739167e8 → 673.9（旧表写 674.0）、
            #   2.738039e9 → 2 738.0（旧表写 2 737.5）——两个都是旧口径残留。
            "cost": b("cost", _sp(float(r["total_cost_cny"]) / 1e6, 1)),
            "cov": b("cov", f"{float(r['demand_coverage_pct']):.2f} %"),
            "w": b("w", f"{float(r['worst_group_access_time_min']):.2f}"),
        })
    return pd.DataFrame(rows)


def build_table20(cfg) -> pd.DataFrame:
    """表 20：方案族 / 保规模杂交 / 自由规模。源 `scheme_family_comparison.csv`。"""
    d = pd.read_csv(_tab(cfg) / "scheme_family_comparison.csv")
    label = {
        "Scheme family (N=10)": ("方案族", "Scheme family", set()),
        "Fixed-N GA breeding": ("**保规模杂交（N = 10）**",
                                "**Fixed-N breeding (N = 10)**",
                                {"hv", "cost", "w"}),
        "Free-cardinality NSGA-II": ("自由规模 NSGA-II",
                                     "Free-cardinality NSGA-II", {"uns", "w"}),
    }
    def _hv(v: float) -> str:
        """超体积的尾数统一到 10¹⁹ 或 10²⁰——两者混排时读者会误比数量级。"""
        if v / 1e19 >= 10:
            return f"{v / 1e20:.3f} × 10²⁰"
        return f"{v / 1e19:.3f} × 10¹⁹"

    rows = []
    for _, r in d.iterrows():
        zh, en, bold = label[str(r["approach"])]

        def b(k, text):
            return f"**{text}**" if k in bold else text

        rows.append({
            "a_zh": zh, "a_en": en, "n": int(r["n_solutions"]),
            "hv": b("hv", _hv(float(r["hypervolume"]))),
            # `min_cost_1e8` 列**存的是亿元**（列名省略了除数），表里要百万元，故 ×100。
            "cost": b("cost", _sp(float(r["min_cost_1e8"]) * 100, 1)),
            "uns": b("uns", f"{float(r['min_unserved']):.1f}"),
            "w": b("w", f"{float(r['min_inequity']):.2f}"),
        })
    return pd.DataFrame(rows)


def build_table21(cfg) -> pd.DataFrame:
    """表 21：四个经典单目标基线。源 `baseline_vs_front.csv`。

    两处**不是**简单映射，必须显式编码：

    * p-中心 p = 8 不可行、p = 12/18/25 给出**同一个解**——所以三行合成一行，
      并另立一个"不可行"行。旧表把 p = 8 写成一行数据、把 12/18/25 写成三行，
      两种情况都会读错。
    * 支配率 0 % 的行不加粗；非零才加粗（那是"被前沿改进"的信号）。
    """
    d = pd.read_csv(_tab(cfg) / "baseline_vs_front.csv").set_index("baseline")
    pretty = {"p_center": ("p-中心", "p-centre"), "p_median": ("p-中位", "p-median"),
              "max_coverage": ("最大覆盖", "Maximal covering"),
              "set_covering": ("集合覆盖", "Set covering")}
    order = ["p_center_p12", "set_covering", "p_median_p8", "p_median_p12",
             "p_median_p18", "p_median_p25", "max_coverage_p8", "max_coverage_p12",
             "max_coverage_p18", "max_coverage_p25"]
    rows = [{
        "m_zh": "p-中心", "m_en": "p-centre", "p_zh": "8", "p_en": "8",
        "n": "—", "cov": "**不可行**（需 ≥ 10 站）" if True else "",
        "cov_en": "**Infeasible** (needs ≥ 10 sites)",
        "w": "—", "dom": "—",
    }]
    for base in order:
        r = d.loc[base]
        if base == "set_covering":
            # ⚠ 它的名字里**没有** `_p<k>` 后缀，`rpartition("_")` 会切出
            #   head="set"、pv="covering"——旧写法在这里 KeyError。
            zh, en = pretty["set_covering"]
            p_zh = p_en = "—"
        else:
            head, _, pv = base.rpartition("_")
            zh, en = pretty[head]
            p_zh = p_en = pv[1:]
            if base == "p_center_p12":
                # p-中心在 p ≥ 12 后给出**同一个解**，正文把三行合成一行。
                p_zh, p_en = "12、18、25", "12, 18, 25"
        nd = float(r["dominated_by_front_pct"])
        # 空集覆盖的 n_sites 是 10、最大覆盖 p=18/25 实际解出 16/24 站——都照产物写。
        rows.append({
            "m_zh": zh, "m_en": en, "p_zh": p_zh, "p_en": p_en,
            "n": f"**{int(r['n_sites'])}**" if base in
                 ("max_coverage_p18", "max_coverage_p25") else str(int(r["n_sites"])),
            "cov": f"**{float(r['demand_coverage_pct']):.0f} %**" if base in
                   ("max_coverage_p18", "max_coverage_p25")
                   else f"{float(r['demand_coverage_pct']):.4f} %",
            "cov_en": f"**{float(r['demand_coverage_pct']):.0f} %**" if base in
                      ("max_coverage_p18", "max_coverage_p25")
                      else f"{float(r['demand_coverage_pct']):.4f} %",
            "w": f"{float(r['worst_group_access_time_min']):.4f}",
            "dom": f"**{nd:.4f} %**" if nd > 0 else "0 %",
        })
    return pd.DataFrame(rows)


def build_table22(cfg) -> pd.DataFrame:
    """表 22：与 pymoo 的超体积对照。源 `pymoo_validation.json`。"""
    d = json.loads((_tab(cfg) / "pymoo_validation.json").read_text(encoding="utf-8"))
    E = 1e19
    return pd.DataFrame([
        {"q_zh": "自实现 NSGA-II 超体积", "q_en": "Self-implemented NSGA-II, hypervolume",
         "v": f"{d['hv_self_mean'] / E:.3f} ± {d['hv_self_std'] / E:.3f} × 10¹⁹"},
        {"q_zh": "`pymoo` NSGA-II 超体积", "q_en": "`pymoo` NSGA-II, hypervolume",
         "v": f"{d['hv_pymoo_mean'] / E:.3f} ± {d['hv_pymoo_std'] / E:.3f} × 10¹⁹"},
        {"q_zh": "**比值**", "q_en": "**Ratio**", "v": f"**{d['ratio']:.3f}**"},
        {"q_zh": "Mann–Whitney U 检验 p 值", "q_en": "Mann–Whitney U, p",
         "v": f"{d['mannwhitney_p']:.2e}".replace("e-04", " × 10⁻⁴")
              .replace("e-05", " × 10⁻⁵")},
        {"q_zh": "Cliff's δ", "q_en": "Cliff's δ",
         "v": f"**{d['cliffs_delta']:+.2f}**"},
    ])


def build_table23(cfg) -> pd.DataFrame:
    """表 23：两种公平性度量前沿的相互支配。源 `equity_measure_summary.json`。

    ⚠ 产物的 `direction` 字段本身是**中文短语**（"group_mean 被 individual_max
    支配的解数"），两稿正文都把它改写成了带反引号、"前沿"的正式表述。故此处
    必须显式映射到两稿各自的文字，不能直接照抄产物字段。
    """
    d = json.loads((_tab(cfg) / "equity_measure_summary.json").read_text(encoding="utf-8"))
    lab = {
        "group_mean 被 individual_max 支配的解数": (
            "`group_mean` 前沿被 `individual_max` 前沿支配",
            "`group_mean` front dominated by `individual_max` front"),
        "individual_max 被 group_mean 支配的解数": (
            "`individual_max` 前沿被 `group_mean` 前沿支配",
            "`individual_max` front dominated by `group_mean` front"),
    }
    out = []
    for r in d["comparison_unified"]:
        key = str(r["direction"])
        if key not in lab:
            raise KeyError(f"表 23 未收录的 direction：{key!r}——"
                           "产物新增了方向，必须同时补两稿的表述")
        zh, en = lab[key]
        cnt, tot = int(r["count"]), int(r["total"])
        out.append({"d_zh": zh, "d_en": en, "c": f"{cnt} / {tot}",
                    "s": f"**{100 * cnt / tot:.1f} %**"})
    return pd.DataFrame(out)


def build_table27(cfg) -> pd.DataFrame:
    """表 27：ρ 证伪检验。源 `sensitivity.csv` 的 `trip_rate_ratio` 行。

    "最差／最好"两列由拐点解反推：最差 = `knee_worst_group_access_time_min`，
    最好 = 最差 − `knee_group_gap_min`。⚠ 这与表 26 的"最差人群接驳"
    （`pareto_inequity_min`，前沿最优）**不是同一个量**，列题必须写明"拐点解的"。
    """
    d = pd.read_csv(_tab(cfg) / "sensitivity.csv")
    s = d[d["parameter"] == "trip_rate_ratio"].copy()
    s["_v"] = pd.to_numeric(s["value"], errors="coerce")
    s = s.sort_values("_v")
    note = {1.0: ("（需求与收入无关）", " (demand independent of income)"),
            5.0: ("（外生阶梯的默认口径）", " (default of the exogenous ladder)")}
    rows = []
    for _, r in s.iterrows():
        v = float(r["_v"])
        w = float(r["knee_worst_group_access_time_min"])
        gap = float(r["knee_group_gap_min"])
        sd = float(r.get("knee_group_gap_min_sd", float("nan")))
        nz, ne = note.get(v, ("", ""))
        rows.append({
            "rho_zh": f"{v:g}{nz}", "rho_en": f"{v:g}{ne}",
            "dem": _sp(r["total_demand"]),
            "wb_zh": f"{w:.2f} ／ {w - gap:.2f}", "wb_en": f"{w:.2f} / {w - gap:.2f}",
            "gap": f"{gap:.2f} ± {sd:.2f}",
        })
    return pd.DataFrame(rows)


def build_table28(cfg) -> pd.DataFrame:
    """表 28：票价情景。源 `fare_scenarios.csv`，**全部 8 档**。

    ⚠ 加粗两处：0.30（类间差距最小的一端）与 1.00（基准票价）——正文同时
    引用这两行。把 1.00 标成"基准"是因为读者要靠它对齐其他章节的口径。
    """
    d = pd.read_csv(_tab(cfg) / "fare_scenarios.csv")
    d["_f"] = pd.to_numeric(d["fare_multiplier"], errors="coerce")
    d = d.sort_values("_f")
    # 加粗逐行不同：0.30 只加粗**比值**（正文引"类间比 1.12"），1.00 三格全加粗
    # （基准行，含"只有 6.53 % 会选择 eVTOL"这一被反复引用的数）。写成显式集合，
    # 不用"首行加粗"之类的位置规则——位置规则在行序变化时会静默错位。
    bold_cols = {0.3: {"ratio"}, 1.0: {"ratio", "pw"}}
    label = {0.3: ("**0.30**", "**0.30**"),
             1.0: ("**1.00（基准）**", "**1.00 (base)**")}
    rows = []
    for _, r in d.iterrows():
        f = float(r["_f"])
        fb, fe = label.get(round(f, 2), (f"{f:.2f}", f"{f:.2f}"))
        bs = bold_cols.get(round(f, 2), set())

        def b(col, text):
            return f"**{text}**" if col in bs else text

        rows.append({
            "f_zh": fb, "f_en": fe,
            "ratio": b("ratio", f"{float(r['p_ratio_max_min']):.4f}"),
            "pw": b("pw", f"{100 * float(r['p_population_weighted']):.2f} %"),
        })
    return pd.DataFrame(rows)


def build_table29(cfg) -> pd.DataFrame:
    """表 29：欧氏 vs 路网阻抗。源 `network_basis_comparison.csv`（逐指标）
    与 `network_basis_summary.json`（Jaccard）。

    ⚠ 本表逐行**手工编码**而非套循环：单位（元 → 亿元 / M CNY）、小数位、
    加粗对象、减号字符（正文用 U+2212 `−`，不是连字符）逐行都不同，
    套一层"通用格式"反而会比手写更容易错。
    """
    m = pd.read_csv(_tab(cfg) / "network_basis_comparison.csv").set_index("metric")
    sm = json.loads((_tab(cfg) / "network_basis_summary.json").read_text(encoding="utf-8"))
    MINUS = "−"

    def v(k: str, col: str) -> float:
        return float(m.loc[k, col])

    def pct(k: str) -> str:
        d = v(k, "delta_pct")
        s = f"{abs(d):.0f} %"
        return (f"{MINUS}{s}" if d < 0 else f"+{s}")

    def money(k: str) -> tuple[str, str]:
        return (f"{v(k, 'euclidean') / 1e8:.2f} 亿元",
                f"{v(k, 'network') / 1e8:.2f} 亿元")

    def money_en(k: str, dec: int = 0) -> tuple[str, str]:
        return (_sp(v(k, "euclidean") / 1e6, dec),
                _sp(v(k, "network") / 1e6, dec))

    def pair(k: str, f) -> tuple[str, str]:
        return f(v(k, "euclidean")), f(v(k, "network"))

    ROWS = [
        # (中文名, 英文名, 欧氏, 路网, 变化, 加粗欧氏?, 加粗路网?, 加粗变化?)
        ("需求单元平均可达候选数", "Mean reachable candidates per demand cell",
         *pair("mean_reachable_cand_per_demand", lambda x: f"{x:.1f}"),
         pct("mean_reachable_cand_per_demand"), False, True, True),
        ("拐点解站数", "Knee-solution site count",
         *pair("knee_n_sites", lambda x: f"{x:.1f}"),
         pct("knee_n_sites"), False, True, True),
        ("拐点解成本", "Knee-solution cost",
         *money("knee_cost_cny"), pct("knee_cost_cny"), False, True, True),
        ("前沿最低成本", "Front minimum cost",
         *money("pareto_cost_min"), pct("pareto_cost_min"), False, True, True),
        ("前沿最小未服务需求", "Front minimum unserved demand",
         f"{v('pareto_unserved_min', 'euclidean'):.0f}",
         f"{_sp(v('pareto_unserved_min', 'network'))} 次/日",
         "—", True, True, False),
        ("拐点解需求覆盖率", "Knee demand coverage",
         *pair("knee_demand_coverage_pct", lambda x: f"{x:.2f} %"),
         f"{MINUS}{abs(v('knee_demand_coverage_pct', 'delta_pct')):.1f} pp",
         False, False, False),
        ("拐点解最差人群接驳", "Knee worst-group access",
         *pair("knee_worst_group_min", lambda x: f"{x:.2f} min"),
         f"{MINUS}{abs(v('knee_worst_group_min', 'delta_pct')):.1f} %",
         False, False, False),
    ]
    out = []
    for zh, en, ez, nz, dz, be, bn, bd in ROWS:
        b = lambda flag, t: f"**{t}**" if flag else t  # noqa: E731
        # 英文侧：成本行换成 M CNY（与表 24/30 的单位习惯一致），未服务行带单位。
        if en == "Knee-solution cost":
            e_en, n_en = (f"{t} M CNY" for t in money_en("knee_cost_cny"))
        elif en == "Front minimum cost":
            e_en, n_en = (f"{t} M CNY" for t in money_en("pareto_cost_min"))
        elif en == "Front minimum unserved demand":
            e_en, n_en = (f"{v('pareto_unserved_min', 'euclidean'):.0f}",
                          f"{_sp(v('pareto_unserved_min', 'network'))} trips/day")
        else:
            e_en, n_en = ez, nz
        out.append({"i_zh": zh, "i_en": en,
                    "e_zh": b(be, ez), "e_en": b(be, e_en),
                    "n_zh": b(bn, nz), "n_en": b(bn, n_en),
                    "d_zh": b(bd, dz), "d_en": b(bd, dz)})
    ji, jk = sm["jaccard_ip_mean"], sm["jaccard_knee_mean"]
    out.append({"i_zh": "**站点集合重合度（Jaccard）**",
                "i_en": "**Site-set overlap (Jaccard)**",
                "e_zh": "—", "e_en": "—", "n_zh": "—", "n_en": "—",
                "d_zh": f"**{ji:.3f}（IP）／{jk:.3f}（拐点）**",
                "d_en": f"**{ji:.3f} (IP) / {jk:.3f} (knee)**"})
    return pd.DataFrame(out)


SPECS = {
    16: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 16.**", "en_caption": "**Table 16.**",
        "builder": build_table16,
        "zh_cols": ["m_zh", "n", "cost_zh", "uns", "w_zh"],
        "en_cols": ["m_en", "n", "cost_en", "uns", "w_en"],
        "header_zh": ["方法", "解数", "最低成本（亿元）", "最小未服务（次/日）",
                      "最小最差人群（分钟）"],
        "header_en": ["Method", "Solutions", "Min cost (M CNY)",
                      "Min unserved (trips/day)", "Min worst-group (min)"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 5,
    },
    17: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 17.**", "en_caption": "**Table 17.**",
        "builder": build_table17,
        "zh_cols": ["p_zh", "n", "cost_zh", "cov", "uns", "tat", "w"],
        "en_cols": ["p_en", "n", "cost_en", "cov", "uns", "tat", "w"],
        "header_zh": ["方案", "站数", "成本（亿元）", "需求覆盖率", "未服务（次/日）",
                      "总接驳时间（分钟·次/日）", "最差人群（分钟）"],
        "header_en": ["Plan", "Sites", "Cost (M CNY)", "Demand coverage",
                      "Unserved (trips/day)", "Total access time (min·trips/day)",
                      "Worst-group (min)"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 7,
    },
    18: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 18.**", "en_caption": "**Table 18.**",
        "builder": build_table18,
        "zh_cols": ["r_zh", "n", "cost", "cov", "w"],
        "en_cols": ["r_en", "n", "cost", "cov", "w"],
        "header_zh": ["选取规则", "站数", "成本（百万元）", "需求覆盖率",
                      "最差人群接驳（分钟）"],
        "header_en": ["Selection rule", "Sites", "Cost (M CNY)", "Demand coverage",
                      "Worst-group access (min)"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 5,
    },
    20: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 20.**", "en_caption": "**Table 20.**",
        "builder": build_table20,
        "zh_cols": ["a_zh", "n", "hv", "cost", "uns", "w"],
        "en_cols": ["a_en", "n", "hv", "cost", "uns", "w"],
        "header_zh": ["方法", "解数", "超体积", "最低成本（百万元）",
                      "最小未服务（次/日）", "最小最差组（分钟）"],
        "header_en": ["Approach", "Solutions", "Hypervolume", "Min cost (M CNY)",
                      "Min unserved (trips/day)", "Min worst-group (min)"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 6,
    },
    21: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 21.**", "en_caption": "**Table 21.**",
        "builder": build_table21,
        "zh_cols": ["m_zh", "p_zh", "n", "cov", "w", "dom"],
        "en_cols": ["m_en", "p_en", "n", "cov_en", "w", "dom"],
        "header_zh": ["基线模型", "p", "站数", "需求覆盖率", "最差人群接驳（分钟）",
                      "被前沿支配"],
        "header_en": ["Baseline model", "p", "Sites", "Demand coverage",
                      "Worst-group access (min)", "Dominated by the front"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 6,
    },
    22: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 22.**", "en_caption": "**Table 22.**",
        "builder": build_table22,
        "zh_cols": ["q_zh", "v"], "en_cols": ["q_en", "v"],
        "header_zh": ["量", "值"], "header_en": ["Quantity", "Value"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 2,
    },
    23: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 23.**", "en_caption": "**Table 23.**",
        "builder": build_table23,
        "zh_cols": ["d_zh", "c", "s"], "en_cols": ["d_en", "c", "s"],
        "header_zh": ["方向", "被支配解数", "占比"],
        "header_en": ["Direction", "Solutions dominated", "Share"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 3,
    },
    27: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 27.**", "en_caption": "**Table 27.**",
        "builder": build_table27,
        "zh_cols": ["rho_zh", "dem", "wb_zh", "gap"],
        "en_cols": ["rho_en", "dem", "wb_en", "gap"],
        "header_zh": ["$\\rho$", "总需求（次/日）",
                      "拐点解的 最差／最好人群接驳（分钟）",
                      "组间差距（分钟，10 次运行）"],
        "header_en": ["$\\rho$", "Total demand (trips/day)",
                      "Worst / best group access, knee solution (min)",
                      "Between-group gap (min, 10 runs)"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 4,
    },
    28: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 28.**", "en_caption": "**Table 28.**",
        "builder": build_table28,
        "zh_cols": ["f_zh", "ratio", "pw"], "en_cols": ["f_en", "ratio", "pw"],
        "header_zh": ["票价倍数", "类间采用率之比", "总体加权采用率"],
        "header_en": ["Fare multiplier", "Cross-class adoption ratio",
                      "Population-weighted adoption"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 3,
    },
    29: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 29.**", "en_caption": "**Table 29.**",
        "builder": build_table29,
        "zh_cols": ["i_zh", "e_zh", "n_zh", "d_zh"],
        "en_cols": ["i_en", "e_en", "n_en", "d_en"],
        "header_zh": ["指标", "欧氏（基准）", "路网", "变化"],
        "header_en": ["Metric", "Euclidean (base)", "Network", "Change"],
        "fmt": {}, "en_fmt": {}, "align": ["---"] * 4,
    },
    30: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 30.**", "en_caption": "**Table 30.**",
        "builder": build_table30,
        "zh_cols": ["metric_zh", "ms_zh", "range_zh", "cv_zh", "dir_zh"],
        "en_cols": ["metric_en", "ms_en", "range_en", "cv_en", "dir_en"],
        "header_zh": ["指标", "均值 ± 标准差", "取值范围", "变异系数", "方向"],
        "header_en": ["Metric", "Mean ± SD", "Range", "CV", "Better"],
        # builder 已把两稿的显示文本都生成好了，这里不再做任何格式化或换算。
        "fmt": {}, "en_fmt": {},
        "align": ["---"] * 5,
    },
    33: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 33.**", "en_caption": "**Table 33.**",
        "builder": build_table33,
        "cols": ["q", "cost", "unserved", "time", "inequity"],
        "header_zh": ["成本分位", "成本（百万元）", "未服务需求（次/日）",
                      "总接驳时间（分钟·次/日）", "最差人群接驳（分钟）"],
        "header_en": ["Cost percentile", "Cost (M CNY)", "Unserved demand (trips/day)",
                      "Total access time (min·trips/day)", "Worst-group access (min)"],
        # 只加粗两端的成本与最差人群——与正文"两端"的论述对应。
        "value_map_zh": {"q": {"0": "0 %", "10": "10 %", "25": "25 %",
                               "50": "50 %", "75": "75 %", "100": "100 %"}},
        "value_map_en": {"q": {"0": "0 %", "10": "10 %", "25": "25 %",
                               "50": "50 %", "75": "75 %", "100": "100 %"}},
        "fmt": {
            "cost": lambda v: f"{float(v):,.1f}".replace(",", " "),
            "unserved": lambda v: f"{float(v):,.1f}".replace(",", " "),
            "time": _fmt_int,
            "inequity": lambda v: f"{float(v):.2f}",
        },
        "bold_rows": {"key": "q", "values": [0, 100],
                      "cols": ["cost", "inequity"]},
        "align": ["---"] * 5,
    },
    25: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 25.**", "en_caption": "**Table 25.**",
        "csv": "sensitivity.csv",
        "filter": ("parameter", "airport_buffer"),
        "sort_by": "value",
        "cols": ["value", "n_candidates", "n_rooftop", "pareto_unserved_min",
                 "pareto_inequity_min", "knee_demand_coverage_pct"],
        "header_zh": ["缓冲半径 (m)", "候选数", "屋顶型", "未服务需求（次/日）†",
                      "最差人群接驳（分钟）†", "需求覆盖率 ‡"],
        "header_en": ["Buffer radius (m)", "Candidates", "Rooftop",
                      "Unserved demand (trips/day) †",
                      "Worst-group access (min) †", "Demand coverage ‡"],
        # 只加粗两处：半径 0 与基准行——**加粗是排版约定，不是"最优值"标注**。
        # 旧英文稿把每行每列的最优值都加粗了，读者会误以为在比大小；
        # 而本表**列间不可比**（见脚注 †‡），加粗"最优"恰恰是误导。
        "value_map_zh": {"value": {"0": "**0**", "4000": "4 000（基准）"}},
        "value_map_en": {"value": {"0": "**0**", "4000": "**4 000 (base case)**"}},
        "fmt": {
            "value": _fmt_int,
            "n_candidates": _fmt_int, "n_rooftop": _fmt_int,
            "pareto_unserved_min": _fmt_int,
            "pareto_inequity_min": lambda v: f"{float(v):.2f}",
            "knee_demand_coverage_pct": lambda v: f"{float(v):.2f} %",
        },
        "align": ["---"] * 6,
    },
    26: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 26.**", "en_caption": "**Table 26.**",
        "builder": build_table26,
        "zh_cols": ["param_zh", "n_values", "movement_zh", "demand_zh", "feas_zh"],
        "en_cols": ["param_en", "n_values", "movement_en", "demand_en", "feas_en"],
        "header_zh": ["参数", "取值数", "最差人群接驳的变动†", "需求量的变动",
                      "是否改变可行性"],
        "header_en": ["Parameter", "Values", "Movement in worst-group access",
                      "Movement in demand", "Changes feasibility?"],
        # builder 已把两稿的显示文本都生成好了，这里不再做任何格式化或换算。
        "fmt": {}, "en_fmt": {},
        "align": ["---"] * 5,
    },
    19: {
        # 表 19 此前是**手工维护**的。手工维护正是这一路漂移的根源：本轮审计
        # 发现它的四行**全部**过期（成本 312.8→321.1、524.1→453.0 等），
        # 而那组数是 A6/坡度修复**之前**的第二批产物。
        #
        # 加粗规则原先也是乱的：同一列里 `457 821` 与 `449 347` **同时加粗**，
        # 而 449 347 更小；最差人群列加粗 `9.90`，而 `9.86` 更小。那是
        # "每套方案标自己的目标值"与"标该列最优"两套规则的残留混合。
        # 这里把规则定死为**该列最优**，由生成器计算，不再手写。
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 19.**", "en_caption": "**Table 19.**",
        "csv": "scheme_family.csv",
        "cols": ["name", "n_sites", "total_cost_cny", "total_access_time",
                 "demand_coverage_pct", "worst_group_access_time_min"],
        # 表 19 的正文标签比产物短：产物叫「最短接驳（距离最近）」，正文叫
        # 「距离最近」（表 24 才用全称）。不做这个映射会把排版改坏。
        "label_zh": {"最短接驳（距离最近）": "距离最近"},
        "label_en": {
            "最低成本": "Lowest cost",
            "最短接驳（距离最近）": "Shortest access",
            "最大覆盖": "Maximum coverage",
            "最均衡体验": "Most balanced",
        },
        "header_zh": ["方案", "站数", "成本（百万元）", "总接驳时间（分钟·次/日）",
                      "需求覆盖率", "最差人群接驳（分钟）"],
        "header_en": ["Scheme", "Sites", "Cost (M CNY)",
                      "Total access time (min·trips/day)", "Demand coverage",
                      "Worst-group access (min)"],
        "fmt": {
            "n_sites": lambda v: f"{int(v)}",
            "total_cost_cny": lambda v: f"{float(v) / 1e6:,.1f}",
            # 与正文既有写法一致：千位用**细空格**而非逗号（469 162）
            "total_access_time": lambda v: f"{float(v):,.0f}".replace(",", " "),
            "demand_coverage_pct": lambda v: f"{float(v):.2f} %",
            "worst_group_access_time_min": lambda v: f"{float(v):.2f}",
        },
        # 加粗 = 该列最优（成本/总接驳/最差人群取最小，覆盖率取最大）
        "bold_best": {
            "total_cost_cny": "min",
            "total_access_time": "min",
            "demand_coverage_pct": "max",
            "worst_group_access_time_min": "min",
        },
        "align": ["---"] * 6,
    },
    24: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 24.**", "en_caption": "**Table 24.**",
        "csv": "all_schemes_comparison.csv",
        # ⚠ 产物 CSV 的列名与标签都是**中文**，与语言无关；两稿只差表头文字。
        #   我最初给英文稿单独写了一份英文列名，直接 KeyError——CSV 里没有那些列。
        "cols": ["组", "方案", "站数", "成本(亿元)", "需求覆盖率%", "最差人群(min)"],
        # ⚠ 正文的**行标签是手工改写过的**（"p center p12" → "p-centre, p=12"），
        #   所以纯按 CSV 重生成会把排版改坏。这里保留一份标签映射，
        #   使重生成只更新**数字**、不动排版。
        "label_zh": {
            "p center p12": "p-中心（p=12）", "p center p18": "p-中心（p=18）",
            "p center p25": "p-中心（p=25）", "set covering": "集合覆盖",
            "max coverage p8": "最大覆盖（p=8）", "max coverage p12": "最大覆盖（p=12）",
            "max coverage p18": "最大覆盖（p=18）", "max coverage p25": "最大覆盖（p=25）",
            "p median p8": "p-中位（p=8）", "p median p12": "p-中位（p=12）",
            "p median p18": "p-中位（p=18）", "p median p25": "p-中位（p=25）",
        },
        "scheme_en": {
            "整数规划 · 成本优先": "IP · cost priority",
            "整数规划 · 覆盖优先": "IP · coverage priority",
            "约束情景 · 预算": "Constraint · budget",
            "约束情景 · 服务标准": "Constraint · access standard",
            "约束情景 · 设施规模": "Constraint · facility cap",
            "约束情景 · 公平": "Constraint · equity floor",
            "固定 N=10 · 最低成本": "Fixed N=10 · lowest cost",
            "固定 N=10 · 最短接驳（距离最近）": "Fixed N=10 · shortest access",
            "固定 N=10 · 最大覆盖": "Fixed N=10 · max coverage",
            "固定 N=10 · 最均衡体验": "Fixed N=10 · most balanced",
            "前沿 · 成本最小": "Front · cost-minimal",
            "前沿 · 拐点": "Front · knee point",
            "前沿 · TOPSIS 等权": "Front · TOPSIS, equal weights",
            "前沿 · 公平最优": "Front · best equity",
            "前沿 · 未服务最小": "Front · best unserved demand",
        },
        "label_en": {
            "p center p12": "p-centre, p=12", "p center p18": "p-centre, p=18",
            "p center p25": "p-centre, p=25", "set covering": "Set covering",
            "max coverage p8": "Maximal covering, p=8",
            "max coverage p12": "Maximal covering, p=12",
            "max coverage p18": "Maximal covering, p=18",
            "max coverage p25": "Maximal covering, p=25",
            "p median p8": "p-median, p=8", "p median p12": "p-median, p=12",
            "p median p18": "p-median, p=18", "p median p25": "p-median, p=25",
        },
        "zh_cols": ["组", "方案", "站数", "成本(亿元)", "需求覆盖率%", "最差人群(min)"],
        "en_cols": ["组", "方案", "站数", "成本(亿元)", "需求覆盖率%", "最差人群(min)"],
        # 组名两稿不同（中文"本文方案"/"前沿选点"/"单目标基线"，英文三组英文名）
        "group_zh": {"本文方案": "本文方案", "前沿选点": "前沿选点",
                     "单目标基线": "单目标基线"},
        "group_en": {"本文方案": "This paper", "前沿选点": "Front selection",
                     "单目标基线": "Baseline"},
        # ⚠ 单位不同：产物用**亿元**，英文稿用 **M CNY**（百万人民币）。
        #   1 亿元 = 100 M CNY。英文稿若直接印 4.82 会被读成 482 万，差两个数量级。
        "en_scale": {"成本(亿元)": 100.0},
        # ⚠ 这里**只格式化、不再换算**：换算由 en_scale 完成。
        #   先前两处都乘了 100，313 M CNY 被印成 31,300。
        "en_fmt": {
            "成本(亿元)": lambda v: f"{float(v):,.0f}",
            "需求覆盖率%": lambda v: f"{float(v):.2f} %",
            "最差人群(min)": lambda v: ("—" if pd.isna(v) or float(v) > 1e6
                                        else f"{float(v):.2f}"),
            "站数": lambda v: f"{int(v):,}",
        },
        "header_zh": ["组", "方案", "站数", "成本（亿元）", "需求覆盖率", "最差人群接驳（分钟）"],
        "header_en": ["Group", "Scheme", "Sites", "Cost (M CNY)",
                      "Demand coverage", "Worst-group access (min)"],
        "fmt": {
            "站数": lambda v: f"{int(v):,}",
            "成本(亿元)": lambda v: f"{float(v):,.2f}",
            "需求覆盖率%": lambda v: f"{float(v):.2f} %",
            "最差人群(min)": lambda v: ("—" if pd.isna(v) or float(v) > 1e6
                                        else f"{float(v):.2f}"),
        },
        # 一律用左对齐，**与正文既有写法一致**：重生成只应改数字，不改排版约定。
        "align": ["---"] * 6,
    },
    34: {
        "zh_file": "中文稿.md", "en_file": "manuscript_draft.md",
        "zh_caption": "**表 34.**", "en_caption": "**Table 34.**",
        "csv": "sensitivity_full.csv",
        "cols": ["参数", "取值", "候选数", "IP 站数", "最小未服务",
                 "最差人群(min)", "成本下界(亿)"],
        # 参数名两稿语言不同
        "param_en": {
            "屋顶面积阈值 (m²)": "Roof-area threshold (m²)",
            "屋顶高度阈值 (m)": "Roof-height threshold (m)",
            "地面建筑覆盖率上限": "Ground building-coverage cap",
            "候选最小间距 (m)": "Minimum candidate spacing (m)",
            "机场排除半径 (m)": "Airport exclusion radius (m)",
            "聚类数 K": "Cluster count K",
            "接驳时间预算 (min)": "Access-time budget (min)",
            "核心需求分位数": "Core-demand quantile",
            "成本缩放系数": "Cost-scaling factor",
            "出行率比 ρ": "Trip-rate ratio rho",
            "出行距离结构": "Trip-distance structure",
            "收入阶梯中位数 (元/月)": "Income-ladder median (CNY/month)",
            "收入离散度 σ": "Income dispersion sigma",
            "人口来源与量级": "Population source and scale",
            "公平性度量（定义）": "Equity measure (definition)",
        },
        "header_zh": ["参数", "取值", "候选数", "IP 站数", "最小未服务（次/日）",
                      "最差人群接驳（分钟）", "成本下界（亿元）"],
        "header_en": ["Parameter", "Value", "Candidates", "IP sites",
                      "Min unserved (trips/day)", "Worst-group access (min)",
                      "Cost lower bound (M CNY)"],
        # 附录 D 的"成本下界"列英文稿同样用 M CNY（原表即为 M CNY 量级）
        "en_scale": {"成本下界(亿)": 100.0},
        "en_fmt": {
            "取值": lambda v: str(v),
            "候选数": lambda v: f"{int(v):,}",
            "最小未服务": lambda v: f"{float(v):,.0f}",
            "最差人群(min)": lambda v: f"{float(v):.2f}",
            "成本下界(亿)": lambda v: f"{float(v):,.0f}",
        },
        "fmt": {
            "候选数": lambda v: f"{int(v):,}",
            "最小未服务": lambda v: f"{float(v):,.0f}",
            "最差人群(min)": lambda v: f"{float(v):.2f}",
            "成本下界(亿)": lambda v: f"{float(v):.2f}",
        },
        "align": ["---"] * 7,
    },
}


def _norm_key(v) -> str:
    """把取值规整成字典键（4000.0 与 4000 视为同一个）。"""
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except (TypeError, ValueError):
        return s


def build_table(spec: dict, lang: str) -> list[str]:
    cfg = load_config()
    if spec.get("builder"):
        df = spec["builder"](cfg)
    else:
        csv_path = Path(cfg.dir("output.tables_dir")) / spec["csv"]
        df = pd.read_csv(csv_path)
    flt = spec.get("filter")
    if flt:
        col, val = flt
        df = df[df[col].astype(str) == str(val)]
    if spec.get("sort_by"):
        # ⚠ `value` 列在 sensitivity.csv 里是**混合类型**（多数是数字，`equity_measure`
        #   那几行是 `group_mean` 这样的字符串），pandas 于是整列读成 object。
        #   按字符串排序会把 12 000 排到 2 000 前面（表 25 实测如此）。
        #   一律先转数值再排；转不动的（非数值取值）排在最后。
        k = pd.to_numeric(df[spec["sort_by"]], errors="coerce")
        df = df.assign(_k=k).sort_values("_k", kind="stable",
                                        na_position="last").drop(columns="_k")
    # 规格可以只给一份 `cols`（列名与语言无关时），也可分 zh_cols/en_cols
    cols = spec.get("zh_cols" if lang == "zh" else "en_cols") or spec["cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{spec['csv']} 缺少列 {missing}；实际列 {list(df.columns)}")
    hdr = spec["header_zh"] if lang == "zh" else spec["header_en"]
    lmap = spec.get("label_zh" if lang == "zh" else "label_en", {})
    gmap = spec.get("group_zh" if lang == "zh" else "group_en", {})
    efmt = spec.get("en_fmt", {}) if lang == "en" else {}
    escale = spec.get("en_scale", {}) if lang == "en" else {}

    # 加粗：先按**原始数值**算出各列极值，再在格式化之后套上 `**`。
    # 必须用原始值比较——`en_scale` 会改写 v，而极值判定与单位无关。
    bold = spec.get("bold_best", {})
    best_val: dict = {}
    for c, how in bold.items():
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals):
            best_val[c] = float(vals.min() if how == "min" else vals.max())

    brow = spec.get("bold_rows", {})
    brow_key = brow.get("key", "q")
    brow_cols = set(brow.get("cols", []))
    brow_vals = list(brow.get("values", []))

    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(spec["align"]) + "|"]
    for _, r in df.iterrows():
        row_bold = bool(brow_vals) and r.get(brow_key) in brow_vals
        cells = []
        for c in cols:
            v = r[c]
            if c == "组" and str(v) in gmap:
                cells.append(gmap[str(v)])
                continue
            if c == "参数" and lang == "en":
                pmap = spec.get("param_en", {})
                if str(v) in pmap:
                    cells.append(pmap[str(v)]); continue
            # 方案名列：表 24 叫「方案」，表 19 叫「name」
            if c in ("方案", "name"):
                smap = spec.get("scheme_en", {}) if lang == "en" else {}
                if str(v) in smap:
                    cells.append(smap[str(v)]); continue
                if str(v) in lmap:
                    cells.append(lmap[str(v)]); continue
            vmap = spec.get("value_map_" + lang, spec.get("value_map", {})).get(c)
            if vmap and _norm_key(v) in vmap:
                cells.append(vmap[_norm_key(v)]); continue
            f = efmt.get(c) or spec["fmt"].get(c)
            if c in escale:
                v = float(v) * escale[c]
            txt = str(f(v)) if f else str(v)
            if c in best_val and pd.notna(r[c]) and float(r[c]) == best_val[c]:
                txt = f"**{txt}**"
            elif row_bold and c in brow_cols:
                txt = f"**{txt}**"
            cells.append(txt)
        out.append("| " + " | ".join(cells) + " |")
    return out


def replace_block(lines: list[str], caption: str, new_rows: list[str]) -> tuple[list[str], int, int]:
    """把 caption 之后的第一个 `|` 连续块整体换成 new_rows。"""
    ci = next((i for i, l in enumerate(lines) if l.startswith(caption)), None)
    if ci is None:
        raise KeyError(f"找不到表题 {caption}")
    s = next((i for i in range(ci + 1, len(lines)) if lines[i].lstrip().startswith("|")), None)
    if s is None:
        raise KeyError(f"{caption} 之后找不到表格")
    e = s
    while e < len(lines) and lines[e].lstrip().startswith("|"):
        e += 1
    return lines[:s] + new_rows + lines[e:], e - s, len(new_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="从产物重生成正文表格")
    ap.add_argument("--tables", default=None,
                    help="要重生成的表号，逗号分隔。**默认为 SPECS 里的全部表**——"
                         "早前默认是单个 '24'，于是 `--dry-run` 看着跑成功了，"
                         "实际只核对了一张表，其余手抄表照样漂移（这正是本项目"
                         "反复出现的『把沉默当通过』）。要查全部就必须明确："
                         "不传 --tables 即查全部。")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-row-change", action="store_true",
                    help="允许重生成后表格行数变化。默认拒绝——行数变了通常意味着"
                         "上游产物流失，而不是表格本身该变。")
    args = ap.parse_args()

    if args.tables:
        nums = [int(x) for x in args.tables.split(",") if x.strip()]
    else:
        nums = sorted(SPECS)
        print(f"未指定 --tables：核对 SPECS 中的全部 {len(nums)} 张表 {nums}\n")
    rc = 0
    for n in nums:
        spec = SPECS.get(n)
        if spec is None:
            print(f"⚠ 表 {n} 还没有规格定义，跳过"); rc = 1; continue
        for lang, key in (("zh", "zh_file"), ("en", "en_file")):
            cap = spec["zh_caption"] if lang == "zh" else spec["en_caption"]
            path = MANU / spec[key]
            lines = path.read_text(encoding="utf-8").split("\n")
            new_rows = build_table(spec, lang)
            try:
                out, old_n, new_n = replace_block(lines, cap, new_rows)
            except KeyError as exc:
                print(f"  ✗ {path.name} 表 {n}: {exc}"); rc = 1; continue
            # ⚠ 行数守卫。这一类缺陷在本项目已出现五次，形态都一样：
            #   上游静默丢掉一行 → 产物少一行 → 这里照着写回去 → **表里少一行
            #   而没人会注意到**（数字错还有机会被发现，少一行不会）。
            #   实测：`min_worst_case` 方案因 status="Feasible" 被上游剔除，
            #   all_schemes_comparison.csv 由 27 行变 26 行，表 24 差一点
            #   就被静默重生成成 26 行。故行数一变即**拒绝写入**。
            # 只拦**写入**。--dry-run 不落盘，让它照常显示差异更有用——
            # 差异本身正是判断"该不该改"的依据。
            if old_n != new_n and not args.allow_row_change and not args.dry_run:
                print(f"  ✗ {path.name} 表 {n}: 行数由 {old_n} 变为 {new_n}，"
                      f"**拒绝写入**——避免静默增删行。请先查清上游产物为何少/多行；"
                      f"确认无误后加 --allow-row-change 重跑。")
                rc = 1
                continue
            flag = "" if old_n == new_n else f"  ⚠ **行数变化 {old_n} → {new_n}**"
            print(f"  {path.name} 表 {n}: {old_n} 行 -> {new_n} 行{flag}")
            if args.dry_run:
                s = next(i for i, l in enumerate(lines) if l.startswith(cap))
                s = next(i for i in range(s + 1, len(lines)) if lines[i].lstrip().startswith("|"))
                e = s
                while e < len(lines) and lines[e].lstrip().startswith("|"):
                    e += 1
                for a, b in zip(lines[s:e], new_rows):
                    mark = "  " if a == b else "≠ "
                    print(f"    {mark}旧: {a[:100]}")
                    if a != b:
                        print(f"     新: {b[:100]}")
            else:
                path.write_text("\n".join(out), encoding="utf-8")
    if args.dry_run:
        print("\n--dry-run：未写回。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
