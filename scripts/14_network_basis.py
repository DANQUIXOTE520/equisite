#!/usr/bin/env python
"""阻抗口径稳健性检验：欧氏直线 vs OSM 路网最短路。

为什么必须单独成脚本
--------------------
论文 §4.3 与 §4.6 **两次**声称"已在 §5.9 报告两种阻抗口径各自的选址结果"，
并写明"这一对照不是可选项"。但审计发现：

* 两城配置的 ``access.time_basis`` 都是 ``euclidean``；
* ``outputs/`` 里没有任何路网口径的选址产物；
* 仅有的 ``审计记录/probe_network_basis.py`` 是**试点脚本**，自述"从未跑过路网口径"，
  且它为了不污染欧氏结果而**刻意不落盘**。

更麻烦的是那个试点脚本留下的两个输出**互相矛盾**：一次报路网可达对
3 054 451（占 91.0 %）、另一次报 138 586（占 4.1 %）。两者相差 22 倍，
说明中途改过截断/预算口径却都没记录。这种数字若进了论文，审稿人一复现
就会发现对不上。

本脚本把该对照做成一等产物：**同一套代码、同一份候选集与需求、只换阻抗口径**，
把两边的 IP 选址与 NSGA-II 前沿放在一起比。要回答的问题只有一个：

    选址结论是不是"欧氏距离"这一假设的产物？

判据是**站点集合的重合度**（Jaccard）与关键指标的差值，而不是"目标值是否
完全相同"——路网口径下接驳时间系统性变长，目标值本就应当不同。

用法::

    python scripts/14_network_basis.py
    python scripts/14_network_basis.py --reps 3 --n-gen 60

输出：
    outputs/tables/network_basis_comparison.csv   两口径逐项对照
    outputs/tables/network_basis_sites.csv        两口径各自选中的站点
    outputs/logs/network_basis.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("network_basis")


def _run_one(cfg, cache, n_gen, pop_size, seed):
    """在给定阻抗口径下跑完整四阶段，返回结果字典。"""
    from evtol_siting.metrics import solution_metrics
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.stage4_nsga2 import knee_point

    cfg = cfg.override({"project.random_seed": seed})
    cfg.seed_everything()

    pipe = Pipeline(cfg, data_mode="real", cache=cache)
    t0 = time.time()
    cs = pipe.run_stage1()
    pipe.run_stage2()
    sols = pipe.run_stage3()
    res = pipe.run_stage4(n_gen=n_gen, pop_size=pop_size, verbose=False)
    dt = time.time() - t0

    F = np.asarray(res.pareto_F, dtype=np.float64)
    out: dict = {
        "n_candidates": len(cs),
        "n_sites_ip": [s.n_sites for s in sols],
        "cost_ip": [float(s.total_cost) for s in sols],
        "ip_site_idx": [sorted(int(i) for i in s.selected) for s in sols],
        "pareto_size": int(len(F)),
        "elapsed_s": round(dt, 1),
    }
    if len(F):
        from evtol_siting.stage4_nsga2 import build_problem

        prob = build_problem(pipe.res.demand.grid, pipe.res.candidates.gdf,
                             pipe.res.access_time_s, cfg)
        k = knee_point(F)
        sel = np.flatnonzero(res.pareto_X[k] > 0.5)
        m = solution_metrics(sel, pipe.res.demand.grid, pipe.res.candidates.gdf,
                             pipe.res.access_time_s, cfg, label="knee")
        out.update({
            "knee_site_idx": sorted(int(i) for i in sel),
            "knee_n_sites": m["n_sites"],
            "knee_cost_cny": m["total_cost_cny"],
            "knee_demand_coverage_pct": m["demand_coverage_pct"],
            "knee_mean_access_min": m["mean_access_time_min"],
            "knee_worst_group_min": m.get("worst_group_access_time_min"),
            "knee_group_gap_min": m.get("group_gap_min"),
            "knee_groups_all_served": m.get("groups_all_served"),
            "pareto_cost_min": float(F[:, 3].min()),
            "pareto_cost_max": float(F[:, 3].max()),
            "pareto_unserved_min": float(F[:, 0].min()),
        })
        # 需求单元的平均可达候选数——路网口径下应显著下降，
        # 这是"路网更保守"的直接体现，也是覆盖率变化的机制解释。
        at = np.asarray(pipe.res.access_time_s, dtype=np.float64)
        budget = float(cfg.get("stage3_ip.access_time_budget_min", 15.0)) * 60.0
        out["mean_reachable_cand_per_demand"] = float(
            (np.isfinite(at) & (at <= budget)).sum(axis=1).mean()
        )
    return out


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="阻抗口径稳健性检验")
    ap.add_argument("--reps", type=int, default=1,
                    help="每个口径独立重复次数（默认 1；>=3 可给出跨种子波动）")
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    base = load_config()
    out_dir = (Path(args.out_dir) if args.out_dir
               else base.dir("output.tables_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose,
                  log_file=Path(base.dir("output.data_dir")).parent / "logs" / "network_basis.log")

    log.info("=" * 70)
    log.info("阻抗口径稳健性检验：euclidean vs network")
    log.info("  reps=%d  n_gen=%d  pop=%d  seed=%d",
             args.reps, args.n_gen, args.pop_size, args.seed)
    log.info("=" * 70)

    # 两个口径共用同一份数据缓存：候选集与需求必须完全一致，
    # 否则比出来的差异分不清是口径造成的还是数据造成的。
    cache: dict = {}
    results: dict[str, list[dict]] = {}
    for basis in ("euclidean", "network"):
        cfg = base.override({"access.time_basis": basis})
        if basis == "network":
            cfg = cfg.override({"access.time_basis": "network"})
        results[basis] = []
        for rep in range(args.reps):
            log.info("")
            log.info("── 口径 %s，第 %d/%d 次 ──", basis, rep + 1, args.reps)
            try:
                r = _run_one(cfg, cache, args.n_gen, args.pop_size,
                             args.seed + rep)
                r["rep"] = rep
                results[basis].append(r)
                log.info("   完成：IP %s 站 / 拐点 %s 站 / 覆盖率 %.2f%% / 耗时 %.0fs",
                         r["n_sites_ip"], r.get("knee_n_sites"),
                         r.get("knee_demand_coverage_pct", float("nan")),
                         r["elapsed_s"])
            except Exception as exc:
                log.error("   口径 %s 第 %d 次失败: %s: %s",
                          basis, rep + 1, type(exc).__name__, str(exc)[:300])
                raise

    # -- 汇总对照 ---------------------------------------------------------
    import pandas as pd

    rows = []
    keys = ["n_candidates", "knee_n_sites", "knee_cost_cny",
            "knee_demand_coverage_pct", "knee_mean_access_min",
            "knee_worst_group_min", "knee_group_gap_min",
            "pareto_cost_min", "pareto_cost_max", "pareto_unserved_min",
            "mean_reachable_cand_per_demand"]
    for k in keys:
        vals = {}
        for basis in ("euclidean", "network"):
            v = [r.get(k) for r in results[basis] if r.get(k) is not None]
            vals[basis] = float(np.mean(v)) if v else np.nan
            vals[basis + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rel = ((vals["network"] - vals["euclidean"])
               / abs(vals["euclidean"]) * 100.0
               if vals["euclidean"] not in (0.0, np.nan)
               and np.isfinite(vals["euclidean"]) else np.nan)
        rows.append({"metric": k, **vals, "delta_pct": rel})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "network_basis_comparison.csv", index=False)
    log.info("")
    log.info(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # -- 站点集合重合度（核心判据）----------------------------------------
    site_rows = []
    for basis in ("euclidean", "network"):
        r = results[basis][0]
        for i, s in enumerate(r["ip_site_idx"]):
            site_rows.append({"basis": basis, "solution": f"ip_{i}",
                              "n_sites": len(s)})
    pd.DataFrame(site_rows).to_csv(out_dir / "network_basis_sites.csv",
                                   index=False)

    jac_ip, jac_knee = [], []
    for rep in range(args.reps):
        a, b = results["euclidean"][rep], results["network"][rep]
        for i in range(min(len(a["ip_site_idx"]), len(b["ip_site_idx"]))):
            jac_ip.append(jaccard(a["ip_site_idx"][i], b["ip_site_idx"][i]))
        if "knee_site_idx" in a and "knee_site_idx" in b:
            jac_knee.append(jaccard(a["knee_site_idx"], b["knee_site_idx"]))

    summary = {
        "n_reps": args.reps,
        "jaccard_ip_mean": float(np.mean(jac_ip)) if jac_ip else None,
        "jaccard_knee_mean": float(np.mean(jac_knee)) if jac_knee else None,
        "jaccard_knee_all": [float(x) for x in jac_knee],
        "basis": {"euclidean": results["euclidean"], "network": results["network"]},
    }
    with open(out_dir / "network_basis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("")
    log.info("── 站点集合重合度（Jaccard，1.0 = 完全相同）──")
    log.info("   IP 情景解:   %.3f", summary["jaccard_ip_mean"] or float("nan"))
    log.info("   拐点解:      %.3f", summary["jaccard_knee_mean"] or float("nan"))
    log.info("   需求单元平均可达候选数: 欧氏 %.1f → 路网 %.1f",
             df.loc[df.metric == "mean_reachable_cand_per_demand",
                    "euclidean"].iloc[0],
             df.loc[df.metric == "mean_reachable_cand_per_demand",
                    "network"].iloc[0])
    log.info("")
    log.info("判读：Jaccard 高（>0.6）说明选址结论**不是**欧氏假设的产物，"
             "可在论文中如实报告为稳健性证据；Jaccard 低则必须报告差异"
             "并把路网口径的结果作为主结果之一，而不是只留欧氏。")
    log.info("已写出 %s", out_dir / "network_basis_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
