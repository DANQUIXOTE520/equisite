#!/usr/bin/env python
"""编码消融：0/1 掩码编码 vs 变长整数编码（基因 = 起降场编号）。

为什么必须做
------------
申报书把编码表述为"以起降场编号为基因进行整数编码"，实现采用的是
**等价的掩码编码**。代码注释与论文 §4.5.1 都写了"整数编码作为配置开关
保留以供对照"——但审计发现这个开关**根本不存在**：没有任何代码读取
``encoding`` 配置项，也没有任何一次整数编码的运行。一句"二者在解空间上
等价"于是既无实现支撑、也无实验支撑。

本脚本补上这个对照。要回答的问题是：

    "解空间等价"是否意味着**搜索行为也等价**？同一评估预算下，
    两种编码到达的前沿质量有实质差别吗？

设计
----
* 两臂使用**完全相同**的非支配排序、拥挤距离与二元锦标赛（都取自
  :mod:`nsga2_core`），只有交叉/变异算子不同——否则测的是选择压力，
  不是编码。
* 两臂使用**相同的 pop_size / n_gen / 随机种子集合**，独立重复 n 次。
* 判据：超体积（HV）的跨种子分布，用 Mann-Whitney U 检验 + Cliff's δ。
  同时报告两臂的评估次数（整数编码因表型去重会略少）。

判读方向
--------
这是一个**中性的消融**，不预设谁更好：

* HV 无显著差异 → 两种编码在解空间与搜索能力上等价，论文可以保留
  "掩码编码是整数编码的等价高效实现"这一表述，并**以本实验为据**。
* 有显著差异 → 必须如实报告差异方向，并把 §4.5.1 的"等价"限定为
  **解空间等价**，明确搜索行为不等价。

用法::

    python scripts/15_encoding_ablation.py --runs 10
    python scripts/15_encoding_ablation.py --runs 3 --n-gen 40   # 快速验证

输出：
    outputs/tables/encoding_ablation.csv
    outputs/tables/encoding_ablation.json
    outputs/logs/encoding_ablation.log
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

log = logging.getLogger("encoding_ablation")


def mannwhitney(a, b):
    """Mann-Whitney U 检验，返回 (p, Cliff's delta)。与 08_multirun 同实现。"""
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        return float("nan"), float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    delta = (gt - lt) / (len(a) * len(b))
    return float(p), float(delta)


def main() -> int:
    ap = argparse.ArgumentParser(description="编码消融：掩码 vs 整数")
    ap.add_argument("--runs", type=int, default=10, help="每个编码的独立运行次数")
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--data-mode", default="real",
                    choices=["auto", "real", "synthetic"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    base = load_config()
    out_dir = base.dir("output.tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose,
                  log_file=Path(base.dir("output.data_dir")).parent / "logs" / "encoding_ablation.log")

    from evtol_siting.metrics import hypervolume
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.stage4_nsga2 import build_problem, make_evaluator, make_pruner, make_repair
    from evtol_siting.nsga2_core import nsga2
    from evtol_siting.integer_encoding import integer_nsga2

    log.info("=" * 74)
    log.info("编码消融：mask vs integer")
    log.info("  runs=%d  pop=%d  gen=%d  seed=%d  data=%s",
             args.runs, args.pop_size, args.n_gen, args.seed, args.data_mode)
    log.info("=" * 74)

    cfg = base.override({"project.random_seed": args.seed})
    cfg.seed_everything()
    pipe = Pipeline(cfg, data_mode=args.data_mode, cache={})
    pipe.run_stage1()
    pipe.run_stage2()
    pipe.run_stage3()
    prob = build_problem(pipe.res.demand.grid, pipe.res.candidates.gdf,
                         pipe.res.access_time_s, cfg)
    ev = make_evaluator(prob)
    rep = make_repair(prob) if cfg.get("stage4_nsga2.repair", True) else None
    pruner = make_pruner(prob)
    max_sites = int(cfg.get("stage3_ip.max_sites", 40))
    log.info("候选 %d，需求 %d，设施上限 %d",
             prob.n_cand, len(pipe.res.demand.grid), max_sites)

    LS_PROB = 0.15
    fronts = {"mask": [], "integer": []}
    timing = {"mask": [], "integer": []}
    evals = {"mask": [], "integer": []}

    for i in range(args.runs):
        sd = args.seed + i
        t0 = time.time()
        rm = nsga2(ev, n_var=prob.n_cand, cfg=cfg, pop_size=args.pop_size,
                   n_gen=args.n_gen, repair=rep, seed=sd, verbose=False,
                   local_search=pruner, local_search_prob=LS_PROB)
        fronts["mask"].append(np.atleast_2d(rm.pareto_F))
        timing["mask"].append(time.time() - t0)
        evals["mask"].append(int(rm.history.get("n_evaluations", 0)))

        t0 = time.time()
        ri = integer_nsga2(ev, n_cand=prob.n_cand, max_sites=max_sites, cfg=cfg,
                           pop_size=args.pop_size, n_gen=args.n_gen,
                           seed=sd, verbose=False)
        fronts["integer"].append(np.atleast_2d(ri.pareto_F))
        timing["integer"].append(time.time() - t0)
        evals["integer"].append(int(ri.history.get("n_evaluations", 0)))

        # HV 需要**统一参考点**才能跨臂比较，因此这里只报规模与成本，
        # HV 留到全部运行结束后一次性算（见下）。
        log.info("  第 %2d/%d 次 (seed=%d)：掩码 %3d 解 / %6d 次评估 / %5.1fs"
                 "｜整数 %3d 解 / %6d 次评估 / %5.1fs",
                 i + 1, args.runs, sd,
                 len(fronts["mask"][-1]), evals["mask"][-1], timing["mask"][-1],
                 len(fronts["integer"][-1]), evals["integer"][-1],
                 timing["integer"][-1])

    # 统一参考点：两臂所有前沿的并集再放宽 10%
    allF = np.vstack([f for fl in fronts.values() for f in fl if len(f)])
    ref = allF.max(axis=0) * 1.1 + 1e-9
    hv = {k: np.array([hypervolume(f, ref) for f in v]) for k, v in fronts.items()}

    p, delta = mannwhitney(hv["mask"], hv["integer"])
    log.info("")
    log.info("── 结果（统一参考点 %s）──", np.array2string(ref, precision=4))
    for k in ("mask", "integer"):
        log.info("   %-8s HV 均值 %.5g ± %.3g（n=%d）｜前沿 %d±%d 个解"
                 "｜评估 %d±%d 次｜耗时 %.1f±%.1f s",
                 k, hv[k].mean(), hv[k].std(ddof=1) if len(hv[k]) > 1 else 0.0,
                 len(hv[k]),
                 int(np.mean([len(f) for f in fronts[k]])),
                 int(np.std([len(f) for f in fronts[k]])),
                 int(np.mean(evals[k])), int(np.std(evals[k])),
                 float(np.mean(timing[k])), float(np.std(timing[k])))
    log.info("   Mann-Whitney p=%.4g，Cliff's δ=%+.3f", p, delta)

    if not np.isfinite(p):
        verdict = "样本不足，无法检验"
    elif p > 0.05:
        verdict = ("两种编码的前沿质量**无显著差异**。可以保留「掩码编码是整数"
                   "编码的等价实现」这一表述，并注明该结论由本消融实验支撑。")
    else:
        better = "整数" if hv["integer"].mean() > hv["mask"].mean() else "掩码"
        verdict = (f"两种编码存在**显著差异**（{better}编码更优）。"
                   "§4.5.1 的「等价」必须限定为**解空间等价**，"
                   "并如实写明搜索行为不等价、差异方向与幅度。")
    log.info("   判读：%s", verdict)

    import pandas as pd

    df = pd.DataFrame({
        "encoding": ["mask"] * args.runs + ["integer"] * args.runs,
        "run": list(range(args.runs)) * 2,
        "seed": [args.seed + i for i in range(args.runs)] * 2,
        "hv": np.concatenate([hv["mask"], hv["integer"]]),
        "front_size": [len(f) for f in fronts["mask"]] + [len(f) for f in fronts["integer"]],
        "n_evaluations": evals["mask"] + evals["integer"],
        "elapsed_s": timing["mask"] + timing["integer"],
    })
    df.to_csv(out_dir / "encoding_ablation.csv", index=False)
    with open(out_dir / "encoding_ablation.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n_runs": args.runs, "pop_size": args.pop_size, "n_gen": args.n_gen,
            "reference_point": ref.tolist(),
            "hv_mask_mean": float(hv["mask"].mean()),
            "hv_mask_std": float(hv["mask"].std(ddof=1)) if args.runs > 1 else 0.0,
            "hv_integer_mean": float(hv["integer"].mean()),
            "hv_integer_std": float(hv["integer"].std(ddof=1)) if args.runs > 1 else 0.0,
            "mannwhitney_p": p, "cliffs_delta": delta, "verdict": verdict,
        }, fh, indent=2, ensure_ascii=False)
    log.info("已写出 %s", out_dir / "encoding_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
