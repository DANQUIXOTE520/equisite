#!/usr/bin/env python
"""固定规模方案族 + 保规模方案杂交。

与 ``08_multirun.py`` 的区别
-----------------------------
``08_multirun.py`` 做的是**自由规模**的多目标优化：起降场数量由模型自己决定，
成本目标会把数量压下来。

本脚本做的是**固定规模**问题：起降场总数锁定为 N（默认 11），在这个前提下：

1. 按四个不同目标分别求解，得到**方案族**——成本最低、距离最近、覆盖最大、
   体验最均衡四套布局；
2. 以方案族为初始种群，用**保规模的方案杂交**（子代恰含 N 个站）搜索更优
   折中解；
3. 三者对照：单目标方案族 vs 杂交后的解集 vs 自由规模 NSGA-II。

为什么"杂交"必须保规模
-----------------------
若父代各含 N 个站而交叉后子代规模自由漂移，那"杂交两个布局方案"这层语义
就丢失了——子代退化为一次随机撒点。保规模才能让搜索真正在**方案空间**上做
组合。算子实现见 ``schemes.crossbreed_masks`` / ``schemes.swap_mutate``。

用法
----
::

    python scripts/10_scheme_family.py --data-mode real
    python scripts/10_scheme_family.py --data-mode real --n-sites 20 --fast
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

log = logging.getLogger("scheme_family")


def _mask_from_ids(cand: pd.DataFrame, ids) -> np.ndarray:
    """把 cand_id 列表转成掩码。"""
    x = np.zeros(len(cand), dtype=np.float64)
    pos = {int(c): i for i, c in enumerate(cand["cand_id"].to_numpy())}
    for i in ids:
        if int(i) in pos:
            x[pos[int(i)]] = 1.0
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="固定规模方案族与保规模杂交")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--n-sites", type=int, default=None,
                    help="固定起降场数量 N（默认取配置 stage3_ip.fixed_n）")
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--runs", type=int, default=5, help="保规模 GA 的独立重复次数")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "scheme_family.log")
    if args.fast:
        args.n_gen = min(args.n_gen, 40)
        args.runs = min(args.runs, 2)

    # WARNING: 输出目录必须取自配置。早期这里写死 PROJECT_ROOT/outputs/tables，
    # 多城市下第二个城市的表格会**直接覆盖第一个城市的**（后跑的城市覆盖先跑的，
    # 实测发生过一次，成都的 multirun/baseline/scenario/sensitivity 全部被冲掉）。
    out_dir = Path(load_config().dir("output.tables_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from evtol_siting.metrics import hypervolume, solution_metrics
    from evtol_siting.nsga2_core import nsga2
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.schemes import build_scheme_family, make_fixed_cardinality_repair
    from evtol_siting.stage4_nsga2 import (
        build_problem, make_evaluator, make_pruner, make_repair, objective_table,
    )

    cfg = load_config()
    n_sites = int(args.n_sites if args.n_sites is not None
                  else cfg.get("stage3_ip.fixed_n", 11))

    log.info("=" * 78)
    log.info(" 固定规模方案族与保规模杂交     N = %d，data_mode = %s",
             n_sites, args.data_mode)
    log.info("=" * 78)

    # ------------------------------------------------------------------
    # 数据与问题实例
    # ------------------------------------------------------------------
    pipe = Pipeline(cfg, data_mode=args.data_mode)
    pipe.run_stage1()
    pipe.run_stage2()
    pipe.run_stage3()

    dem = pipe.res.demand.grid
    cand = pipe.res.candidates.gdf
    at = pipe.res.access_time_s

    prob = build_problem(dem, cand, at, cfg)
    evaluate = make_evaluator(prob)
    repair_free = make_repair(prob)
    # 固定规模模式必须用**保规模**修复算子：通用修复在覆盖不足时加点，
    # 会破坏"每个方案恰含 N 个站"的前提（实测保规模交叉因此完全失效）。
    repair_fixed = make_fixed_cardinality_repair(prob, n_sites)
    pruner = make_pruner(prob)
    LS_PROB = float(cfg.get("stage4_nsga2.local_search_prob", 0.15))

    log.info("候选 %d 个，需求单元 %d 个，总需求 %.0f 次/日",
             prob.n_cand, prob.n_dem, prob.total_demand)

    # ------------------------------------------------------------------
    # 1. 固定规模方案族
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 步骤 1：固定 N=%d 的方案族（不同目标 → 不同布局）──", n_sites)
    objectives = cfg.get("stage3_ip.scheme_objectives") or [
        "min_cost", "min_total_time", "max_coverage", "min_worst_case",
    ]
    family = build_scheme_family(dem, cand, at, cfg, n_sites=n_sites,
                                 objectives=objectives)

    fam_rows = []
    for s in family:
        # ⚠ 与 schemes.build_scheme_family 同一判据：按"**有没有解**"判定，
        #   不是"有没有**证明**最优"。这里曾写成 status != "Optimal"，于是
        #   status="Feasible"（有可行解、只是 CBC 没在时限内证明最优）的方案
        #   被丢掉——本 CSV 由 4 行变 3 行，而正文表 19/24 都报着那第四套
        #   方案（"最均衡体验"），重生成表 24 时会**静默少一行**。
        if not s.selected or s.status not in ("Optimal", "Feasible"):
            log.warning("方案族剔除 '%s'（status=%s，无可用解）", s.name, s.status)
            continue
        row = s.to_dict()
        row["_mask"] = _mask_from_ids(cand, s.selected)
        fam_rows.append(row)

    if not fam_rows:
        log.error("方案族为空，无法继续")
        return 1

    fam_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_mask"}
                           for r in fam_rows])
    fam_df.to_csv(out_dir / "scheme_family.csv", index=False)

    # 站点清单另落一份**长表**（一行 = 一套方案的一个站）。
    # ⚠ 审计：此前只落指标、不落站点集合，正文写着"四套方案"却**无从查起**
    #   ——审稿人若要核查某些站为什么入选，产物里没有答案。`selected` 本来是
    #   有的（`Scheme.selected`），只是在写 CSV 时被当作 `_mask` 的原料丢掉了。
    site_rows = [{"scheme": s.name, "objective": s.objective_key,
                  "cand_id": int(c)}
                 for s in family for c in (s.selected or [])]
    if site_rows:
        pd.DataFrame(site_rows).to_csv(
            out_dir / "scheme_family_sites.csv", index=False)
        log.info("   站点清单已写出: scheme_family_sites.csv（%d 行，%d 套方案）",
                 len(site_rows), len({r["scheme"] for r in site_rows}))

    log.info("")
    log.info("方案族对照（论文表）:")
    show = [c for c in ("name", "n_sites", "total_cost_cny", "total_access_time",
                        "demand_coverage_pct", "worst_group_access_time_min")
            if c in fam_df.columns]
    disp = fam_df[show].copy()
    if "total_cost_cny" in disp:
        disp["total_cost_cny"] = (disp["total_cost_cny"] / 1e8).round(3)
        disp = disp.rename(columns={"total_cost_cny": "cost_1e8"})
    log.info("\n%s", disp.to_string(index=False))

    # 方案族之间的支配关系：谁支配谁
    F_fam = np.vstack([evaluate(r["_mask"][None, :])[0] for r in fam_rows])
    from evtol_siting.nsga2_core import dominates

    log.info("")
    log.info("方案族内部支配关系（行是否支配列）:")
    names = [r["name"] for r in fam_rows]
    dom_mat = np.zeros((len(F_fam), len(F_fam)), dtype=bool)
    for i in range(len(F_fam)):
        for j in range(len(F_fam)):
            if i != j:
                dom_mat[i, j] = dominates(F_fam[i], F_fam[j])
    for i, n in enumerate(names):
        beaten = [names[j] for j in range(len(F_fam)) if dom_mat[i, j]]
        log.info("   %-16s 支配: %s", n, ", ".join(beaten) if beaten else "（无）")

    # ------------------------------------------------------------------
    # 2. 保规模方案杂交
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 步骤 2：保规模杂交（子代恒含 %d 站，以方案族为初始种群）──", n_sites)

    x0 = np.vstack([r["_mask"] for r in fam_rows])
    log.info("注入 %d 个方案族个体作为初始种子", len(x0))

    fixed_fronts: list[np.ndarray] = []
    fixed_rows: list[dict] = []
    for r in range(args.runs):
        t0 = time.time()
        res = nsga2(
            evaluate, n_var=prob.n_cand, cfg=cfg,
            pop_size=args.pop_size, n_gen=args.n_gen,
            repair=repair_fixed, x0=x0,
            seed=5000 + r, verbose=False,
            n_sites_fixed=n_sites,          # ← 固定规模模式
            # 固定规模下**禁用剪枝局部搜索**：它只删点，同样会破坏
            # "每个方案恰含 N 个站"的前提。保规模模式下唯一的算子组合是
            # 保规模交叉 + 交换变异 + 保规模修复。
            local_search=None, local_search_prob=0.0,
        )
        F = np.asarray(res.pareto_F, dtype=np.float64)
        # 校验：所有解是否恰好含 N 个站
        sizes = np.asarray(res.pareto_X > 0.5).sum(axis=1)
        ok_fixed = bool((sizes == n_sites).all())
        fixed_fronts.append(F)
        fixed_rows.append({
            "run": r, "pareto_size": len(F),
            "all_fixed_cardinality": ok_fixed,
            "min_cost": float(F[:, 3].min()) if len(F) else np.nan,
            "min_unserved": float(F[:, 0].min()) if len(F) else np.nan,
            "min_inequity": float(F[:, 2].min()) if len(F) else np.nan,
            "min_total_time": float(F[:, 1].min()) if len(F) else np.nan,
            "elapsed_s": round(time.time() - t0, 1),
        })
        log.info("   第 %d/%d 次: 前沿 %3d 解，站点数恒为 %d: %s，耗时 %.0fs",
                 r + 1, args.runs, len(F), n_sites,
                 "是" if ok_fixed else "**否**", time.time() - t0)

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(out_dir / "scheme_family_ga_runs.csv", index=False)

    len_ok = bool(fixed_df["all_fixed_cardinality"].all())
    if len_ok:
        log.info("   ✓ 全部运行的解都恰好含 %d 个站——保规模算子有效", n_sites)
    else:
        log.error("   ✗ 存在站点数不等于 %d 的解，保规模算子未生效！", n_sites)

    # ------------------------------------------------------------------
    # 3. 对照：自由规模 NSGA-II
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 步骤 3：对照（自由规模 NSGA-II，设施数由模型自定）──")
    free_fronts: list[np.ndarray] = []
    free_rows: list[dict] = []
    for r in range(args.runs):
        res = nsga2(
            evaluate, n_var=prob.n_cand, cfg=cfg,
            pop_size=args.pop_size, n_gen=args.n_gen,
            repair=repair_free, x0=x0, seed=6000 + r, verbose=False,
            local_search=pruner, local_search_prob=LS_PROB,
        )
        F = np.asarray(res.pareto_F, dtype=np.float64)
        sizes = np.asarray(res.pareto_X > 0.5).sum(axis=1)
        free_fronts.append(F)
        free_rows.append({
            "run": r, "pareto_size": len(F),
            "n_sites_min": int(sizes.min()) if len(sizes) else 0,
            "n_sites_max": int(sizes.max()) if len(sizes) else 0,
            "min_cost": float(F[:, 3].min()) if len(F) else np.nan,
            "min_unserved": float(F[:, 0].min()) if len(F) else np.nan,
            "min_inequity": float(F[:, 2].min()) if len(F) else np.nan,
            "min_total_time": float(F[:, 1].min()) if len(F) else np.nan,
        })
        log.info("   第 %d/%d 次: 前沿 %3d 解，站点数 %d–%d",
                 r + 1, args.runs, len(F),
                 free_rows[-1]["n_sites_min"], free_rows[-1]["n_sites_max"])

    free_df = pd.DataFrame(free_rows)
    free_df.to_csv(out_dir / "scheme_family_free_runs.csv", index=False)

    # ------------------------------------------------------------------
    # 4. 统一参考点下的对照
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 步骤 4：统一参考点下的对照 ──")
    all_fronts = [f for f in fixed_fronts + free_fronts if len(f)]
    all_fam = F_fam
    merged = np.vstack([np.vstack(all_fronts), all_fam])
    ref = merged.max(axis=0) * 1.1 + 1e-9

    cmp_rows = []
    # 方案族整体
    cmp_rows.append({
        "approach": f"Scheme family (N={n_sites})",
        "n_solutions": len(F_fam),
        "hypervolume": hypervolume(F_fam[~_dominated(F_fam)], ref),
        "min_cost_1e8": float(F_fam[:, 3].min()) / 1e8,
        "min_unserved": float(F_fam[:, 0].min()),
        "min_inequity": float(F_fam[:, 2].min()),
    })
    for label, fronts in (("Fixed-N GA breeding", fixed_fronts),
                          ("Free-cardinality NSGA-II", free_fronts)):
        fs = [f for f in fronts if len(f)]
        if not fs:
            continue
        cat = np.vstack(fs)
        cat = cat[~_dominated(cat)]
        cmp_rows.append({
            "approach": label,
            "n_solutions": len(cat),
            "hypervolume": hypervolume(cat, ref),
            "min_cost_1e8": float(cat[:, 3].min()) / 1e8,
            "min_unserved": float(cat[:, 0].min()),
            "min_inequity": float(cat[:, 2].min()),
        })
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / "scheme_family_comparison.csv", index=False)
    log.info("\n%s", cmp_df.to_string(index=False))

    # 方案族是否被杂交结果支配
    #
    # ⚠ 审计：这个比例此前**只打日志、不落盘**，而论文 §5.6 引用了「1/4 被支配」。
    #   结果审稿人按图索骥时，`scheme_family_summary.json` 里**没有这个数**——
    #   正文的数字没有可追溯产物。现在一并写进 JSON。
    dom_stat: dict = {"n_family": int(len(F_fam))}
    ga_cat = np.vstack([f for f in fixed_fronts if len(f)]) if any(
        len(f) for f in fixed_fronts) else np.empty((0, 4))
    if len(ga_cat):
        ga_cat = ga_cat[~_dominated(ga_cat)]
        n_dom = sum(
            any(dominates(g, F_fam[i]) for g in ga_cat) for i in range(len(F_fam))
        )
        dom_stat.update({
            "n_family_dominated_by_hybrid": int(n_dom),
            "pct_family_dominated_by_hybrid": float(100 * n_dom / max(len(F_fam), 1)),
            "n_hybrid_front_points": int(len(ga_cat)),
        })
        log.info("")
        log.info(
            "方案族中被**杂交结果支配**的：%d/%d（%.0f%%）",
            n_dom, len(F_fam), 100 * n_dom / max(len(F_fam), 1),
        )
        log.info(
            "判读：若比例高，说明『杂交不同方案』确实产出了优于任一单目标"
            "方案的折中解——这正是该设计的价值所在；若为 0，说明方案族本身"
            "已在前沿上，杂交没有带来改进，应如实报告。"
        )
    else:
        dom_stat["n_family_dominated_by_hybrid"] = None
        log.warning("无杂交前沿可比，方案族支配统计不可得——论文不得引用该比例。")

    # ------------------------------------------------------------------
    with open(out_dir / "scheme_family_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n_sites": n_sites,
            "data_mode": pipe.res.data_mode,
            "n_candidates": int(prob.n_cand),
            "n_demand_cells": int(prob.n_dem),
            "family_size": len(F_fam),
            "fixed_cardinality_verified": len_ok,
            "domination": dom_stat,
            "comparison": cmp_df.to_dict(orient="records"),
        }, fh, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 78)
    log.info(" 完成。输出目录: %s", out_dir)
    log.info(" 文件: scheme_family.csv / scheme_family_ga_runs.csv / "
             "scheme_family_free_runs.csv / scheme_family_comparison.csv")
    log.info("=" * 78)
    return 0


def _dominated(F: np.ndarray) -> np.ndarray:
    """返回被支配解的布尔掩码（True = 被支配）。"""
    from evtol_siting.nsga2_core import fast_non_dominated_sort

    if len(F) == 0:
        return np.zeros(0, dtype=bool)
    m = np.ones(len(F), dtype=bool)
    m[fast_non_dominated_sort(F)[0]] = False
    return m


if __name__ == "__main__":
    raise SystemExit(main())
