"""生成论文正文引用的两张汇总表。

## 为什么需要这个脚本

`all_schemes_comparison.csv`（§5.8 的全部方案总表）与 `sensitivity_full.csv`
（附录 D 的完整敏感性表）此前**在仓库里找不到任何生成脚本**——它们是早前用
一次性代码产出的，因而不可复现，也不会随重跑更新。2026-09-17 成都改口径重跑时，
这两个文件仍停留在旧值，正文与附录因此与实际产物不符。

本脚本把两者的口径**显式写成代码**，使它们可复现、可复核、随重跑自动更新。

## 两张表的构造规则（论文须与此一致）

### all_schemes_comparison.csv

三组，共 27 行：

1. **本文方案（10 行）**
   - `整数规划 · 成本优先／覆盖优先`  ← `outputs/data/table_ip_scenarios.csv`
   - `约束情景 · 预算／服务标准／设施规模／公平` ← `outputs/tables/scenario_family.csv`
   - `固定 N · 最低成本／最短接驳／最大覆盖／最均衡体验` ← `outputs/tables/scheme_family.csv`

2. **前沿选点（5 行）** ← `outputs/data/table_solution_metrics.csv` 的 NSGA-II 行，
   标签一一对应：

   | 表中方案 | table_solution_metrics 的 label |
   |---|---|
   | 前沿 · 成本最小 | `NSGA-II:best_total_cost` |
   | 前沿 · 拐点 | `NSGA-II:knee` |
   | 前沿 · TOPSIS 等权 | `NSGA-II:TOPSIS` |
   | 前沿 · 公平最优 | `NSGA-II:best_inequity` |
   | 前沿 · 未服务最小 | `NSGA-II:best_unserved_demand` |

   （`NSGA-II:best_total_access_time` **不入选**，与既有表保持一致。）

3. **单目标基线（12 行）** ← `outputs/tables/baseline_vs_front.csv`，
   即 §5.7 的 p8/12/18/25 那一组（**不是** export_3d 内部用的 p10/11/20/30 组）。

> ⚠ 全表的"最差人群接驳"一律取 `worst_group_access_time_min`（**按需求加权**的
> 组均值，与 §4.4.2 的 $f_3$ 同口径）。**不要**改用 `scenario_comparison.csv`
> 的同名列——那是未加权口径，数值不同。

### sensitivity_full.csv

由 `outputs/tables/sensitivity.csv` 逐行映射而来（中文列名供正文表格直接使用）：

| 输出列 | 来源列 |
|---|---|
| 参数 | `parameter`（映射为中文名） |
| 取值 | `value` |
| 候选数 | `n_candidates` |
| IP 站数 | `ip_n_sites`（`[11, 10]` → `11／10`） |
| 最小未服务 | `pareto_unserved_min`（四舍五入到整数） |
| 最差人群(min) | `pareto_inequity_min`（保留 2 位） |
| 成本下界(亿) | `pareto_cost_min / 1e8`（保留 2 位） |

用法::

    python scripts/12_paper_tables.py
    EVTOL_CONFIG=config/<城市>.yaml python scripts/12_paper_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evtol_siting.config import load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402
import logging  # noqa: E402

log = logging.getLogger("paper_tables")

# 前沿选点的 (表中方案名, table_solution_metrics 的 label)
FRONT_PICKS = [
    ("前沿 · 成本最小", "NSGA-II:best_total_cost"),
    ("前沿 · 拐点", "NSGA-II:knee"),
    ("前沿 · TOPSIS 等权", "NSGA-II:TOPSIS"),
    ("前沿 · 公平最优", "NSGA-II:best_inequity"),
    ("前沿 · 未服务最小", "NSGA-II:best_unserved_demand"),
]

# 敏感性参数的中文名
PARAM_ZH = {
    "roof_area": "屋顶面积阈值 (m²)",
    "roof_height": "屋顶高度阈值 (m)",
    "building_coverage": "地面建筑覆盖率上限",
    "dilution": "候选最小间距 (m)",
    "airport_buffer": "机场排除半径 (m)",
    "n_clusters": "聚类数 K",
    "access_budget": "接驳时间预算 (min)",
    "core_demand": "核心需求分位数",
    "cost_scale": "成本缩放系数",
    "trip_rate_ratio": "出行率比 ρ",
    "equity_measure": "公平性度量",
    "trip_distance": "出行距离结构",
    "income_median": "收入阶梯中位数 (元/月)",
    "income_sigma": "收入离散度 σ",
    "population": "人口来源与量级",
}
# 参数在正文表格中的出现顺序（首行 = 机场排除半径之后按影响幅度排序的既有序，
# 与附录 D 保持一致）
PARAM_ORDER = list(PARAM_ZH)


def _row(group: str, name: str, n_sites, cost_cny, cov_pct, worst_min) -> dict:
    return {
        "组": group,
        "方案": name,
        "站数": int(n_sites),
        "成本(亿元)": round(float(cost_cny) / 1e8, 2),
        "需求覆盖率%": round(float(cov_pct), 2),
        "最差人群(min)": round(float(worst_min), 2),
    }


def build_all_schemes(cfg) -> pd.DataFrame:
    out = cfg.dir("output.data_dir")
    tab = cfg.dir("output.tables_dir")
    rows: list[dict] = []

    # --- 1a. 整数规划两种情景 ---
    ip = pd.read_csv(out / "table_ip_scenarios.csv")
    zh = {"cost_priority": "整数规划 · 成本优先", "coverage_priority": "整数规划 · 覆盖优先"}
    for key, label in zh.items():
        r = ip[ip["name"] == key]
        if len(r):
            r = r.iloc[0]
            rows.append(_row("本文方案", label, r["n_sites"], r["total_cost_cny"],
                             r["coverage_demand_coverage_pct"],
                             # IP 情景没有分组指标，用其自身的平均接驳时间不具可比性，
                             # 故从 table_solution_metrics 的同名标签取（与旧表一致）。
                             _worst_from_metrics(out, f"IP:{key}")))

    # --- 1b. 四约束情景 ---
    sf = tab / "scenario_family.csv"
    if sf.exists():
        for _, r in pd.read_csv(sf).iterrows():
            rows.append(_row("本文方案", f"约束情景 · {r['name'].replace('约束型','')}",
                             r["n_sites"], r["total_cost_cny"],
                             r["demand_coverage_pct"], r["worst_group_access_time_min"]))

    # --- 1c. 固定规模方案族 ---
    scf = tab / "scheme_family.csv"
    if scf.exists():
        n_fixed = int(cfg.get("stage3_ip.fixed_n", 11))
        label_map = {
            "最低成本": f"固定 N={n_fixed} · 最低成本",
            "最短接驳（距离最近）": f"固定 N={n_fixed} · 最短接驳（距离最近）",
            "最大覆盖": f"固定 N={n_fixed} · 最大覆盖",
            "最均衡体验": f"固定 N={n_fixed} · 最均衡体验",
        }
        for _, r in pd.read_csv(scf).iterrows():
            nm = str(r.get("name", ""))
            rows.append(_row("本文方案", label_map.get(nm, f"固定 N={n_fixed} · {nm}"),
                             r["n_sites"], r["total_cost_cny"],
                             r["demand_coverage_pct"], r["worst_group_access_time_min"]))

    # --- 2. 前沿选点 ---
    tsm = pd.read_csv(out / "table_solution_metrics.csv")
    for label, key in FRONT_PICKS:
        r = tsm[tsm["label"] == key]
        if not len(r):
            log.warning("table_solution_metrics 缺少 %s，跳过「%s」", key, label)
            continue
        r = r.iloc[0]
        rows.append(_row("前沿选点", label, r["n_sites"], r["total_cost_cny"],
                         r["demand_coverage_pct"], r["worst_group_access_time_min"]))

    # --- 3. 单目标基线 ---
    bvf = tab / "baseline_vs_front.csv"
    if bvf.exists():
        pretty = {
            "p_center": "p center", "p_median": "p median",
            "max_coverage": "max coverage", "set_covering": "set covering",
        }
        for _, r in pd.read_csv(bvf).iterrows():
            base = str(r["baseline"])
            if base == "set_covering":
                nm = "set covering"
            else:
                head, _, p = base.rpartition("_")
                nm = f"{pretty.get(head, head)} {p}"
            rows.append(_row("单目标基线", nm, r["n_sites"], r["total_cost"],
                             r["demand_coverage_pct"], r["worst_group_access_time_min"]))

    return pd.DataFrame(rows)


def _worst_from_metrics(out_dir: Path, label: str) -> float:
    """从 table_solution_metrics 取某个标签的最差人群接驳（IP 情景用）。"""
    tsm = pd.read_csv(out_dir / "table_solution_metrics.csv")
    r = tsm[tsm["label"] == label]
    return float(r.iloc[0]["worst_group_access_time_min"]) if len(r) else float("nan")


def build_sensitivity_full(cfg) -> pd.DataFrame:
    src = cfg.dir("output.tables_dir") / "sensitivity.csv"
    if not src.exists():
        raise FileNotFoundError(f"未找到 {src}；请先运行 scripts/07_sensitivity.py")
    s = pd.read_csv(src)

    recs = []
    for p in PARAM_ORDER:
        sub = s[s["parameter"] == p]
        if not len(sub):
            continue
        for _, r in sub.iterrows():
            # ip_n_sites 形如 "[11, 10]"；IP 不可行时是空列表 "[]"，
            # 表中记作"无可行解"（机场排除半径 ≥6 km 等点会出现）。
            sites = r.get("ip_n_sites", "")
            try:
                lst = eval(str(sites))  # noqa: S307 —— 来源是本项目自己写的 CSV
                sites_txt = "／".join(str(int(v)) for v in lst) if lst else "无可行解"
            except Exception:
                sites_txt = str(sites)

            # ⚠ **部分**失败此前是静默的：两个 IP 情景中只有一个解出来时，
            #   这一格只印那个数（如接驳预算 10 min 的 "27"），读者无从知道
            #   另一个情景其实不可行——与"两个情景都解出来"长得一模一样。
            #   凡出现这种情况就加标记 †，并在附录 D 的读表说明里解释。
            try:
                st = eval(str(r.get("ip_status", "")))  # noqa: S307
                if isinstance(st, dict) and st:
                    infeas = [k for k, v in st.items() if v != "Optimal"]
                    if infeas and len(infeas) < len(st):
                        sites_txt = f"{sites_txt}†"
            except Exception:
                pass
            recs.append({
                "参数": PARAM_ZH[p],
                "取值": r["value"],
                "候选数": int(r["n_candidates"]),
                "IP 站数": sites_txt,
                "最小未服务": int(round(float(r["pareto_unserved_min"]))),
                "最差人群(min)": round(float(r["pareto_inequity_min"]), 2),
                "成本下界(亿)": round(float(r["pareto_cost_min"]) / 1e8, 2),
            })
    return pd.DataFrame(recs)


def main() -> int:
    setup_logging(False, PROJECT_ROOT / "outputs" / "logs" / "paper_tables.log")
    cfg = load_config()
    tab = cfg.dir("output.tables_dir")
    tab.mkdir(parents=True, exist_ok=True)
    log.info("城市配置: %s   输出目录: %s", cfg.source_path.name, tab)

    df = build_all_schemes(cfg)
    p1 = tab / "all_schemes_comparison.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    log.info("已写出 %s（%d 行，%d 组）", p1.name, len(df), df["组"].nunique())

    try:
        sf = build_sensitivity_full(cfg)
        p2 = tab / "sensitivity_full.csv"
        sf.to_csv(p2, index=False, encoding="utf-8-sig")
        log.info("已写出 %s（%d 行，%d 个参数）", p2.name, len(sf), sf["参数"].nunique())
    except FileNotFoundError as exc:
        log.warning("跳过 sensitivity_full.csv：%s", exc)

    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
