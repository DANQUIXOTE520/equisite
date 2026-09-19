#!/usr/bin/env python
"""四约束情景 → 遗传杂交 → 综合最优方案。

完整链条
--------
本脚本实现研究设计中的后两个阶段：

1. **生成四约束情景方案**（``src/evtol_siting/scenarios.py``）——四种规划
   约束各解出一套布局：预算约束型、服务标准约束型、设施规模约束型、
   公平约束型。
2. **以四套方案为初始种群做多目标杂交**，搜索它们之间的折中解，
   再按 **TOPSIS 等权**选出一套"综合最平衡"的推荐方案。

为什么这里用**自由规模**杂交，而不是方案族那节的保规模杂交
------------------------------------------------------------
四套约束情景的站点数**本来就不同**（预算型 15 站、服务标准型 31 站、
规模型 15 站、公平型 12 站）——因为它们由不同的约束驱动，规模正是约束的
结果而非前提。此时"保规模杂交"没有定义：子代该保谁的数量？因此本节用
自由规模杂交，让站点数也参与搜索。

（``scripts/10_scheme_family.py`` 里那套**保规模**杂交仍然有效，它服务于
另一个问题：**站点数固定为 N 时**如何在不同目标间权衡。两者互补，论文中
应分别说明。)

用法
----
::

    python scripts/11_scenarios_ga.py --data-mode real
    python scripts/11_scenarios_ga.py --data-mode real --runs 5 --pop-size 100 --n-gen 100
"""

from __future__ import annotations

import argparse
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

log = logging.getLogger("scenarios_ga")


def _mask_from_ids(cand: pd.DataFrame, ids) -> np.ndarray:
    x = np.zeros(len(cand), dtype=np.float64)
    pos = {int(c): i for i, c in enumerate(cand["cand_id"].to_numpy())}
    for i in ids:
        if int(i) in pos:
            x[pos[int(i)]] = 1.0
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="四约束情景 + 遗传杂交 + 综合最优")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "scenarios_ga.log")
    if args.fast:
        args.pop_size = min(args.pop_size, 50)
        args.n_gen = min(args.n_gen, 40)
        args.runs = min(args.runs, 2)

    # WARNING: 输出目录必须取自配置。早期这里写死 PROJECT_ROOT/outputs/tables，
    # 多城市下第二个城市的表格会**直接覆盖第一个城市的**（后跑的城市覆盖先跑的，
    # 实测发生过一次，成都的 multirun/baseline/scenario/sensitivity 全部被冲掉）。
    out_dir = Path(load_config().dir("output.tables_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from evtol_siting.metrics import hypervolume, solution_metrics
    from evtol_siting.nsga2_core import dominates, nsga2
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.scenarios import CONSTRAINT_SCENARIOS, build_constraint_scenarios
    from evtol_siting.stage4_nsga2 import (
        build_problem, knee_point, make_evaluator, make_pruner, make_repair,
    )

    cfg = load_config()
    log.info("=" * 78)
    log.info(" 四约束情景 → 遗传杂交 → 综合最优")
    log.info("=" * 78)

    # ------------------------------------------------------------------
    # 数据与四约束情景
    # ------------------------------------------------------------------
    pipe = Pipeline(cfg, data_mode=args.data_mode)
    pipe.run_stage1()
    pipe.run_stage2()
    pipe.run_stage3()

    dem = pipe.res.demand.grid
    cand = pipe.res.candidates.gdf
    at = pipe.res.access_time_s

    log.info("")
    log.info("── 步骤 1：四约束情景方案 ──")
    family = build_constraint_scenarios(dem, cand, at, cfg)
    ok = [s for s in family if s.status == "Optimal" and s.selected]
    if not ok:
        log.error("没有情景求解成功，终止")
        return 1

    rows = []
    for s in ok:
        d = s.to_dict()
        d["_mask"] = _mask_from_ids(cand, s.selected)
        rows.append(d)

    scen_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_mask"} for r in rows])
    scen_df.to_csv(out_dir / "scenario_family.csv", index=False)

    log.info("")
    log.info("四约束情景对照（论文表）:")
    log.info(
        "%-16s %5s %10s %9s %13s %11s",
        "情景", "站数", "成本(亿)", "覆盖率%", "最差人群(min)", "总时间",
    )
    for r in rows:
        log.info(
            "%-16s %5d %10.2f %9.1f %13.2f %11.0f",
            r["name"], r["n_sites"], r["total_cost_cny"] / 1e8,
            r["demand_coverage_pct"], r["worst_group_access_time_min"],
            r["total_access_time"],
        )

    # 情景间支配关系
    prob = build_problem(dem, cand, at, cfg)
    evaluate = make_evaluator(prob)
    repair = make_repair(prob)
    pruner = make_pruner(prob)
    ls_prob = float(cfg.get("stage4_nsga2.local_search_prob", 0.15))

    F_scen = np.vstack([evaluate(r["_mask"][None, :])[0] for r in rows])
    log.info("")
    log.info("情景间支配关系:")
    for i, r in enumerate(rows):
        beaten = [rows[j]["name"] for j in range(len(rows))
                  if i != j and dominates(F_scen[i], F_scen[j])]
        log.info("   %-16s 支配: %s", r["name"], "、".join(beaten) if beaten else "（无）")

    # ------------------------------------------------------------------
    # 步骤 2：以四情景为初始种群做自由规模杂交
    # ------------------------------------------------------------------
    x0 = np.vstack([r["_mask"] for r in rows])
    log.info("")
    log.info("── 步骤 2：自由规模杂交（以 %d 套情景为初始种群）──", len(x0))
    log.info(
        "说明：四情景站点数分别为 %s，规模本身是约束的结果，"
        "因此不适用保规模杂交。",
        "、".join(str(r["n_sites"]) for r in rows),
    )

    fronts: list[np.ndarray] = []
    results: list = []          # 保存各次运行的完整结果，用于还原推荐解的掩码
    run_rows: list[dict] = []
    for r in range(args.runs):
        t0 = time.time()
        res = nsga2(
            evaluate, n_var=prob.n_cand, cfg=cfg,
            pop_size=args.pop_size, n_gen=args.n_gen,
            repair=repair, x0=x0, seed=8000 + r, verbose=False,
            local_search=pruner, local_search_prob=ls_prob,
        )
        F = np.asarray(res.pareto_F, dtype=np.float64)
        sizes = np.asarray(res.pareto_X > 0.5).sum(axis=1)
        fronts.append(F)
        results.append(res)
        run_rows.append({
            "run": r, "pareto_size": len(F),
            "n_sites_min": int(sizes.min()), "n_sites_max": int(sizes.max()),
            "min_unserved": float(F[:, 0].min()),
            "min_cost_1e8": float(F[:, 3].min()) / 1e8,
            "min_inequity": float(F[:, 2].min()),
            "elapsed_s": round(time.time() - t0, 1),
        })
        log.info(
            "   第 %d/%d 次: 前沿 %3d 解，站点数 %d–%d，耗时 %.0fs",
            r + 1, args.runs, len(F), sizes.min(), sizes.max(), time.time() - t0,
        )
    pd.DataFrame(run_rows).to_csv(out_dir / "scenario_ga_runs.csv", index=False)

    # ------------------------------------------------------------------
    # 步骤 3：TOPSIS 等权选出"综合最平衡"的推荐方案
    # ------------------------------------------------------------------
    from evtol_siting.stage4_nsga2 import topsis_ranking

    best_front = None
    best_hv = -np.inf
    allF = np.vstack([f for f in fronts if len(f)])
    ref = allF.max(axis=0) * 1.1 + 1e-9
    for i, F in enumerate(fronts):
        if not len(F):
            continue
        hv = hypervolume(F, ref)
        if hv > best_hv:
            best_hv, best_front = hv, i

    merged = np.vstack([f for f in fronts if len(f)])
    merged = merged[_nondominated(merged)]
    rank = topsis_ranking(merged, weights=np.ones(merged.shape[1]))
    rec_F = merged[rank[0]]

    # 从各次运行中找回该解对应的选址掩码。合并前沿里的解一定来自某一次
    # 运行，按目标向量精确匹配即可——不用重新评估，也不会取错解。
    rec_ids: list[int] = []
    for res_i in results:
        F = np.asarray(res_i.pareto_F, dtype=np.float64)
        if not len(F):
            continue
        dist = np.abs(F - rec_F[None, :]).sum(axis=1)
        j = int(np.argmin(dist))
        if dist[j] < 1e-6:
            mask = res_i.pareto_X[j] > 0.5
            rec_ids = [int(c) for c in cand["cand_id"].iloc[np.flatnonzero(mask)]]
            break

    log.info("")
    log.info("── 步骤 3：TOPSIS 等权选出的推荐方案 ──")
    log.info(
        "   目标值: 未服务 %.2f | 总时间 %.0f | 公平性 %.3f | 成本 %.2f 亿",
        rec_F[0], rec_F[1], rec_F[2], rec_F[3] / 1e8,
    )
    if not rec_ids:
        log.warning("未能从运行结果中还原推荐解的掩码（目标值匹配失败）")

    all_cmp = []
    for r in rows:
        m = solution_metrics(
            [int(c) for c in cand["cand_id"].iloc[np.flatnonzero(r["_mask"] > 0.5)]],
            dem, cand, at, cfg, label=r["name"],
        )
        all_cmp.append(m)
    if rec_ids:
        all_cmp.append(solution_metrics(rec_ids, dem, cand, at, cfg,
                                        label="TOPSIS 推荐方案"))
    cmp_df = pd.DataFrame(all_cmp)
    cmp_df.to_csv(out_dir / "scenario_comparison.csv", index=False)

    log.info("")
    log.info("全部方案对照（论文表）:")
    keep = [c for c in ("label", "n_sites", "total_cost_cny", "demand_coverage_pct",
                        "mean_access_time_min", "worst_group_access_time_min",
                        "equity_gini") if c in cmp_df.columns]
    disp = cmp_df[keep].copy()
    if "total_cost_cny" in disp:
        disp["total_cost_cny"] = (disp["total_cost_cny"] / 1e8).round(2)
    log.info("\n%s", disp.to_string(index=False))

    # 四情景中被杂交结果支配的比例
    n_dom = sum(any(dominates(g, F_scen[i]) for g in merged) for i in range(len(F_scen)))
    log.info("")
    log.info(
        "四情景中被杂交结果支配的: %d/%d（%.0f%%）",
        n_dom, len(F_scen), 100 * n_dom / max(len(F_scen), 1),
    )

    with open(out_dir / "scenario_ga_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "data_mode": pipe.res.data_mode,
            "n_scenarios_ok": len(ok),
            "scenarios": [r["name"] for r in rows],
            "front_merged_size": int(len(merged)),
            "n_scenarios_dominated": int(n_dom),
            "recommended_objectives": [float(v) for v in rec_F],
            "recommended_ids": rec_ids,
        }, fh, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 78)
    log.info(" 完成。输出: scenario_family.csv / scenario_ga_runs.csv / "
             "scenario_comparison.csv")
    log.info("=" * 78)
    return 0


def _nondominated(F: np.ndarray) -> np.ndarray:
    from evtol_siting.nsga2_core import fast_non_dominated_sort

    if len(F) == 0:
        return np.zeros(0, dtype=bool)
    m = np.zeros(len(F), dtype=bool)
    m[fast_non_dominated_sort(F)[0]] = True
    return m


if __name__ == "__main__":
    raise SystemExit(main())
