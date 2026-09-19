#!/usr/bin/env python
"""多次独立运行与统计检验 —— 随机算法结果的可信度论证。

为什么必须做
------------
NSGA-II 是**随机**算法：不同随机种子会得到不同的帕累托前沿。只报告单次
运行的结果在方法论上是不成立的——审稿人会问"换一个种子结论还成立吗"。
正确做法是独立重复 n 次，报告指标的**均值 ± 标准差**，并对算法间差异
做统计检验。

本脚本产出论文第 5.5 节所需的全部统计证据：

1. **算法自身稳定性**：n 次运行的 HV / SP / Δ / IGD 的均值与标准差。
2. **与 pymoo 的一致性**：用经过广泛验证的 pymoo 实现跑同一算例，
   检验自实现 NSGA-II 是否正确。若两者前沿质量统计上无显著差异，
   说明自实现无缺陷——这是论文附录中值得报告的一个验证步骤。
3. **与单目标基线的对比**：把每个基线解的目标值放到同一目标空间，
   检验它是否被 NSGA-II 前沿**支配**（这是"多目标必要性"的核心证据）。
4. **公平性新意的显著性**：对比 ``equity_measure`` 两种设定下的
   最差人群接驳时间差异。

统计方法
--------
* 指标对比用 **Mann-Whitney U 检验**（非参数，不假设正态性——目标值分布
  通常是重尾的，用 t 检验不合适）。
* 支配关系用**逐对比较**：统计基线解被前沿中多少个解支配。
* 显著性水平取 0.05，同时报告效应量（Cliff's delta），避免只看 p 值。

用法
----
::

    python scripts/08_multirun.py --data-mode real --runs 10
    python scripts/08_multirun.py --data-mode synthetic --runs 5 --fast
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

log = logging.getLogger("multirun")


def mannwhitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U 检验，返回 ``(p 值, Cliff's delta 效应量)``。

    选它而不选 t 检验：目标值分布通常是重尾且小样本（n=10），正态性假设
    不成立。Cliff's delta 是配套的非参数效应量，取值 ``[-1, 1]``，
    ``|delta|`` 的经验阈值：0.147 小、0.33 中、0.474 大。
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return np.nan, np.nan

    # Cliff's delta：P(a>b) - P(a<b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    delta = (gt - lt) / (len(a) * len(b))
    return float(p), float(delta)


def dominates(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """``a`` 是否支配 ``b``（全部目标最小化）。"""
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多次独立运行与统计检验")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--runs", type=int, default=10, help="独立重复次数")
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--fast", action="store_true", help="减少规模，快速验证脚本")
    ap.add_argument("--no-pymoo", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "multirun.log")
    if args.fast:
        args.runs = min(args.runs, 3)
        args.pop_size = min(args.pop_size, 40)
        args.n_gen = min(args.n_gen, 30)

    # ⚠ 输出目录必须取自配置。早期这里写死 PROJECT_ROOT/outputs/tables，
    # 多城市下第二个城市的表格会**直接覆盖第一个城市的**（实测发生过一次，
    # 先跑那一城的 multirun/baseline/scenario/sensitivity 全部被冲掉）。
    out_dir = Path(load_config().dir("output.tables_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 准备：数据与问题实例只构建一次，所有运行复用
    # ------------------------------------------------------------------
    from evtol_siting.metrics import (
        hypervolume, igd, solution_metrics, spacing, spread,
    )
    from evtol_siting.nsga2_core import nsga2
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.stage4_nsga2 import (
        build_problem, knee_point, make_evaluator, make_pruner, make_repair,
    )

    base_cfg = load_config()
    log.info("=" * 76)
    log.info(" 多次独立运行：%d 次，pop=%d，gen=%d，data_mode=%s",
             args.runs, args.pop_size, args.n_gen, args.data_mode)
    log.info("=" * 76)

    pipe = Pipeline(base_cfg, data_mode=args.data_mode)
    pipe.run_stage1()
    pipe.run_stage2()
    pipe.run_stage3()

    prob = build_problem(
        pipe.res.demand.grid, pipe.res.candidates.gdf, pipe.res.access_time_s, base_cfg
    )
    evaluate = make_evaluator(prob)
    repair = make_repair(prob) if base_cfg.get("stage4_nsga2.repair", True) else None
    # 剪枝作为局部搜索算子（按概率作用于子代）。
    # 注意不能放进解码器：那会把任何稠密解瞬间剪回最小覆盖集，
    # 使高站点数区段无法被探索——实测会让超体积只有 pymoo 的 18%。
    pruner = make_pruner(prob) if base_cfg.get("stage4_nsga2.local_search", True) else None
    LS_PROB = float(base_cfg.get("stage4_nsga2.local_search_prob", 0.15))

    # ------------------------------------------------------------------
    # 1. n 次独立运行
    # ------------------------------------------------------------------
    fronts: list[np.ndarray] = []
    run_rows: list[dict] = []

    for r in range(args.runs):
        cfg_r = base_cfg.override({"project.random_seed": 1000 + r})
        cfg_r.seed_everything()
        t0 = time.time()
        res = nsga2(
            evaluate, n_var=prob.n_cand, cfg=cfg_r,
            pop_size=args.pop_size, n_gen=args.n_gen,
            repair=repair, seed=1000 + r, verbose=False,
            local_search=pruner, local_search_prob=LS_PROB,
        )
        F = np.asarray(res.pareto_F, dtype=np.float64)
        fronts.append(F)
        hv_hist = res.history.get("hypervolume", [])
        run_rows.append({
            "run": r, "seed": 1000 + r,
            "pareto_size": len(F),
            "hypervolume_proxy": float(hv_hist[-1]) if hv_hist else np.nan,
            "hv_initial": res.history.get("hv_initial"),
            "n_evaluations": res.history.get("n_evaluations"),
            "elapsed_s": round(time.time() - t0, 1),
            "best_unserved": float(F[:, 0].min()) if len(F) else np.nan,
            "best_cost": float(F[:, 3].min()) if len(F) else np.nan,
            "best_inequity": float(F[:, 2].min()) if len(F) else np.nan,
        })
        log.info(
            "  第 %2d/%d 次: 前沿 %3d 个解，HV=%.4g，最优未服务=%.1f，耗时 %.1fs",
            r + 1, args.runs, len(F), run_rows[-1]["hypervolume_proxy"],
            run_rows[-1]["best_unserved"], run_rows[-1]["elapsed_s"],
        )

    df_runs = pd.DataFrame(run_rows)
    df_runs.to_csv(out_dir / "multirun_runs.csv", index=False)

    # ------------------------------------------------------------------
    # 2. 合并参考前沿与质量指标
    # ------------------------------------------------------------------
    all_F = np.vstack([f for f in fronts if len(f)])
    ref_front = all_F[_nondominated(all_F)] if len(all_F) else np.empty((0, 4))
    # 固定参考点：必须跨运行固定，否则各次 HV 不可比
    ref_point = all_F.max(axis=0) * 1.1 + 1e-9 if len(all_F) else None

    metric_rows = []
    for i, F in enumerate(fronts):
        if len(F) == 0:
            continue
        metric_rows.append({
            "run": i,
            "n": len(F),
            "hypervolume": hypervolume(F, ref_point),
            "spacing": spacing(F),
            "spread": spread(F),
            "igd": igd(F, ref_front),
        })
    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(out_dir / "multirun_metrics.csv", index=False)

    log.info("")
    log.info("── 算法稳定性（%d 次独立运行）──", len(df_metrics))
    summary = {}
    for col in ("hypervolume", "spacing", "spread", "igd", "n"):
        if col not in df_metrics.columns:
            continue
        v = df_metrics[col].to_numpy(dtype=np.float64)
        summary[col] = {
            "mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
        }
        log.info(
            "   %-14s 均值 %-12.5g 标准差 %-12.5g （%.5g – %.5g）",
            col, summary[col]["mean"], summary[col]["std"],
            summary[col]["min"], summary[col]["max"],
        )
    log.info(
        "   注：HV 越大越好；SP/Δ/IGD 越小越好。论文中报告为 均值 ± 标准差。"
    )

    # ------------------------------------------------------------------
    # 3. 与 pymoo 交叉验证
    # ------------------------------------------------------------------
    if not args.no_pymoo:
        try:
            from evtol_siting.stage4_nsga2 import make_repair, problem_for_pymoo

            p = problem_for_pymoo(prob)
            if p is not None:
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.core.repair import Repair
                from pymoo.optimize import minimize

                # ⚠ 两处必须修的地方（审计 #16）：
                #
                # 1. **pymoo 也必须跑 n 个种子。** 早期版本只跑 seed 20240501
                #    一次，然后把**同一个 HV 值复制 10 份**去和自实现的 10 个
                #    值做 Mann-Whitney——那是在拿一个常数和一个样本比，
                #    p=6.4e-5、δ=+1.00 是构造出来的，不是测出来的。
                # 2. **pymoo 必须带同一套修复算子。** 原先它解的是松弛后的
                #    连续问题（xl=0, xu=1, 无修复），与自实现解的不是同一个
                #    算例；差异里混着"约束处理不同"这一项。这里把修复算子
                #    包装成 pymoo 的 Repair，先二值化再修复，使两者可行域一致。
                rep_fn = (make_repair(prob)
                          if base_cfg.get("stage4_nsga2.repair", True) else None)

                class _MaskRepair(Repair):
                    def _do(self, problem, X, **kw):
                        Xb = (np.asarray(X, dtype=np.float64) > 0.5).astype(np.float64)
                        if rep_fn is None:
                            return Xb
                        return np.array([rep_fn(x) for x in Xb])

                Fpy_list = []
                for si in range(args.runs):
                    sd = 20240501 + si
                    out = minimize(
                        p, NSGA2(pop_size=args.pop_size, repair=_MaskRepair()),
                        ("n_gen", args.n_gen), seed=sd, verbose=False,
                    )
                    Fpy_list.append(np.atleast_2d(out.F))
                    log.info("   pymoo 第 %d/%d 个种子完成（seed=%d，前沿 %d 个解）",
                             si + 1, args.runs, sd, len(Fpy_list[-1]))

                merged = np.vstack([ref_front] + Fpy_list)
                ref2 = merged.max(axis=0) * 1.1 + 1e-9
                hv_self = np.array([hypervolume(f, ref2) for f in fronts if len(f)])
                hv_py = np.array([hypervolume(f, ref2) for f in Fpy_list])
                pval, delta = mannwhitney(hv_self, hv_py)
                log.info("")
                log.info("── 与 pymoo NSGA-II 交叉验证（n 个种子，含同一修复算子）──")
                log.info("   自实现 HV 均值 %.5g ± %.3g (n=%d)",
                         hv_self.mean(), hv_self.std(ddof=1) if len(hv_self) > 1 else 0.0,
                         len(hv_self))
                log.info("   pymoo  HV 均值 %.5g ± %.3g (n=%d)",
                         hv_py.mean(), hv_py.std(ddof=1) if len(hv_py) > 1 else 0.0,
                         len(hv_py))
                log.info("   比值 %.4f，Mann-Whitney p=%.4g，Cliff's δ=%+.3f",
                         hv_self.mean() / max(hv_py.mean(), 1e-12), pval, delta)
                # 判读方向必须写对：这是**正确性验证**，不是性能竞赛。
                if pval > 0.05:
                    verdict = ("两者统计上无显著差异 → 自实现的非支配排序、"
                               "拥挤距离与选择机制与 pymoo 一致，实现通过验证。")
                elif delta > 0:
                    verdict = ("自实现**优于** pymoo，这不是验证通过的证据，"
                               "而是提示两者仍有未对齐之处（最可能是修复算子的"
                               "作用强度不同）。论文应如实报告为「存在差异，"
                               "来源待查」，不得写成正确性验证。")
                else:
                    verdict = ("自实现**劣于** pymoo，说明实现存在缺陷，"
                               "论文必须如实报告并排查。")
                log.info("   判读：%s", verdict)
                with open(out_dir / "pymoo_validation.json", "w", encoding="utf-8") as fh:
                    json.dump({
                        "n_seeds": args.runs,
                        "repair_operator_aligned": rep_fn is not None,
                        "hv_self_mean": float(hv_self.mean()),
                        "hv_self_std": float(hv_self.std(ddof=1)) if len(hv_self) > 1 else 0.0,
                        "hv_pymoo_mean": float(hv_py.mean()),
                        "hv_pymoo_std": float(hv_py.std(ddof=1)) if len(hv_py) > 1 else 0.0,
                        "ratio": float(hv_self.mean() / max(hv_py.mean(), 1e-12)),
                        "mannwhitney_p": pval, "cliffs_delta": delta,
                        "verdict": verdict,
                    }, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            log.warning("pymoo 交叉验证失败: %s", str(exc)[:200])

    # ------------------------------------------------------------------
    # 4. 与单目标基线的支配关系
    # ------------------------------------------------------------------
    if not args.no_baselines:
        try:
            from evtol_siting.baselines import run_all_baselines

            log.info("")
            log.info("── 单目标基线 vs NSGA-II 前沿 ──")
            ps = sorted({8, 12, 18, 25})
            bls = run_all_baselines(
                pipe.res.demand.grid, pipe.res.candidates.gdf,
                pipe.res.access_time_s, base_cfg, p_values=ps,
            )
            rows = []
            for b in bls:
                if b is None or not b.feasible or not b.selected:
                    continue
                x = np.zeros(prob.n_cand)
                x[[int(i) for i in b.selected]] = 1.0
                f = evaluate(x[None, :])[0]
                # 被前沿中多少个解支配
                n_dom = int(sum(dominates(rf, f) for rf in ref_front))
                m = solution_metrics(
                    b.selected, pipe.res.demand.grid, pipe.res.candidates.gdf,
                    pipe.res.access_time_s, base_cfg, label=b.name,
                )
                rows.append({
                    "baseline": b.name, "n_sites": b.n_sites,
                    "unserved_demand": float(f[0]), "total_access_time": float(f[1]),
                    "inequity": float(f[2]), "total_cost": float(f[3]),
                    "n_front_dominating": n_dom,
                    "dominated_by_front_pct": 100 * n_dom / max(len(ref_front), 1),
                    "demand_coverage_pct": m.get("demand_coverage_pct"),
                    "worst_group_access_time_min": m.get("worst_group_access_time_min"),
                    "equity_gini": m.get("equity_gini"),
                })
            skipped = [b.name for b in bls if b is None or not b.feasible or not b.selected]
            if skipped:
                log.warning("   以下基线未解出，已从对比表中剔除：%s", ", ".join(skipped))
            if not rows:
                raise RuntimeError("没有任何基线解出，不覆盖已有的 baseline_vs_front.csv")
            df_bl = pd.DataFrame(rows).sort_values("dominated_by_front_pct", ascending=False)
            df_bl.to_csv(out_dir / "baseline_vs_front.csv", index=False)
            if len(df_bl):
                log.info(df_bl.to_string(index=False))
                log.info(
                    "   判读：dominated_by_front_pct 越高，说明该单目标模型的最优解"
                    "被 NSGA-II 前沿支配得越彻底——这是'多目标必要性'的直接证据。"
                )
        except Exception as exc:
            log.warning("基线对比失败: %s", str(exc)[:200])

    # ------------------------------------------------------------------
    # 5. 加权求和对照（回答"为什么不用加权和"这一头号质疑）
    #
    # 实验设计的关键：**用同一个搜索算法、同一份评估预算**，只改变选择机制
    # （加权和标量化 vs 非支配排序）。这样才能把"多目标处理的必要性"从
    # "搜索力度差异"里隔离出来——否则审稿人可以说"加权和只是跑得少"。
    #
    # 做法：把四目标加权成一个标量，用同一套 SBX/多项式变异/修复算子做
    # 单目标搜索（NSGA-II 在单目标下退化为标准 GA，直接复用即可）。
    # 权重用 Dirichlet 分布采样，覆盖目标空间中的多个偏好方向。
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 加权求和对照（同一算法、同一预算，仅改选择机制）──")
    try:
        rng = np.random.default_rng(20240501)
        n_w = 12
        weights = rng.dirichlet(np.ones(4), size=n_w)      # 每行和为 1
        # 补上四个"极端"偏好，确保覆盖角点
        weights = np.vstack([weights, np.eye(4)])
        # ⚠ 预算必须对齐（审计 #17）。早期版本用
        #     ws_gen = n_evaluations // pop_size
        # 折算，实测得到 62 代，而 NSGA-II 跑的是 100 代——加权和臂被砍掉
        # 近四成搜索预算。在"同一算法、同一预算、只改选择机制"的对照里，
        # 预算不对齐会直接把结论污染成"加权和只是跑得少"。
        # 现在两臂用**相同的 pop_size 与 n_gen**。
        ws_gen = int(args.n_gen)
        budget = int(df_runs["n_evaluations"].max()) if "n_evaluations" in df_runs else None
        log.info("   两臂预算：pop_size=%d，n_gen=%d（NSGA-II 实测最大评估数 %s）",
                 args.pop_size, ws_gen, budget)

        # 参考前沿有两个口径，必须分别报告：
        #   * ref_front      —— n 次运行的**并集**（对加权和臂最不利的比较）
        #   * ref_single     —— **单次**运行的典型前沿（等预算比较）
        # 只拿并集当参考，等于让加权和臂以 1 次运行对抗 n 次运行的并集。
        sizes = [len(f) for f in fronts if len(f)]
        if sizes:
            med = int(np.argsort(sizes)[len(sizes) // 2])
            ref_single = [f for f in fronts if len(f)][med]
        else:
            ref_single = ref_front

        def make_weighted(w):
            def ev(X):
                F = evaluate(X)
                return (F * w[None, :]).sum(axis=1, keepdims=True)
            return ev

        ws_rows = []
        for i, w in enumerate(weights):
            res_w = nsga2(
                make_weighted(w), n_var=prob.n_cand, cfg=base_cfg,
                pop_size=args.pop_size, n_gen=ws_gen, repair=repair,
                seed=7000 + i, verbose=False,
                local_search=pruner, local_search_prob=LS_PROB,
            )
            # 标量化运行的 F 只有 **1 列**（加权和），不能直接与 4 维权重向量
            # 相乘——`cand_F @ w` 会因维度不匹配抛 matmul 错误。
            # 必须用 4 目标评估器把该种群重新解码评价，才能得到可比较的目标值。
            F4 = np.atleast_2d(evaluate(res_w.X))
            best_idx = int(np.argmin(F4 @ w))                  # 加权和最优个体
            best_rel = F4[best_idx]
            n_dom = int(sum(dominates(rf, best_rel) for rf in ref_front))
            n_dom1 = int(sum(dominates(rf, best_rel) for rf in ref_single))
            ws_rows.append({
                "weight_unserved": w[0], "weight_time": w[1],
                "weight_inequity": w[2], "weight_cost": w[3],
                "unserved_demand": float(best_rel[0]),
                "total_access_time": float(best_rel[1]),
                "inequity": float(best_rel[2]),
                "total_cost": float(best_rel[3]),
                # 对 n 次运行并集（对加权和最不利）
                "n_front_dominating": n_dom,
                "dominated_pct": 100 * n_dom / max(len(ref_front), 1),
                # 对单次运行（等预算，公平）
                "n_front_dominating_single": n_dom1,
                "dominated_pct_single": 100 * n_dom1 / max(len(ref_single), 1),
                "n_evaluations": int(res_w.history.get("n_evaluations", 0)),
            })
            if (i + 1) % 4 == 0:
                log.info("   已完成 %d/%d 个权重方向", i + 1, len(weights))

        df_ws = pd.DataFrame(ws_rows)
        df_ws.to_csv(out_dir / "weighted_sum_vs_front.csv", index=False)

        mean_dom = float(df_ws["dominated_pct"].mean())
        frac_any = float((df_ws["n_front_dominating"] > 0).mean())
        mean_dom1 = float(df_ws["dominated_pct_single"].mean())
        frac_any1 = float((df_ws["n_front_dominating_single"] > 0).mean())
        log.info(
            "   对 **n 次运行并集**：%.0f%% 的解被支配，平均被 %.1f%% 的前沿解支配",
            100 * frac_any, mean_dom,
        )
        log.info(
            "   对 **单次运行**（等预算，主判据）：%.0f%% 的解被支配，"
            "平均被 %.1f%% 的前沿解支配",
            100 * frac_any1, mean_dom1,
        )
        log.info(
            "   判读：加权和法的解被前沿支配，说明它**无法到达帕累托前沿的"
            "非凸部分**。这正是多目标处理的必要性所在——若所有加权和解都落在"
            "前沿上，则说明问题目标间无实质权衡，多目标处理不必要。"
        )
        if frac_any1 < 0.5:
            log.warning(
                "   注意：多数加权和解**未被**同等预算的单次前沿支配。"
                "这意味着在本算例中加权和法是有效的，论文不应声称"
                "多目标处理绝对必要，而应改述为「提供更完整的权衡信息」。如实报告。"
            )
        with open(out_dir / "weighted_sum_summary.json", "w", encoding="utf-8") as fh:
            json.dump({
                "n_weight_directions": len(df_ws),
                "pop_size": int(args.pop_size),
                "n_gen_both_arms": ws_gen,
                "nsga2_max_evaluations": budget,
                "ws_mean_evaluations": float(df_ws["n_evaluations"].mean()),
                "budget_aligned": True,
                "vs_union_front": {
                    "mean_dominated_pct": mean_dom,
                    "fraction_with_any_dominating": frac_any,
                    "reference_front_size": int(len(ref_front)),
                },
                "vs_single_run_front": {
                    "mean_dominated_pct": mean_dom1,
                    "fraction_with_any_dominating": frac_any1,
                    "reference_front_size": int(len(ref_single)),
                },
            }, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning("加权求和对照失败: %s", str(exc)[:250])

    # ------------------------------------------------------------------
    # 6. 公平性新意的证成：group_mean vs individual_max
    #
    # 为什么必须做这个实验
    # --------------------
    # 本研究的核心新意是"以**人群组**的可达性作为公平性单元"。审稿人完全
    # 可以说："这不就是经典的 p-中心（最小化最差单个用户）换了个说法吗？"
    # 只靠文字辩解是不够的，必须用实验证明两者给出**不同**的解。
    #
    # 预期结果：individual_max 会为了照顾某个偏远需求点而牺牲整体效率，
    # 其解在 group_mean 口径下表现更差；反之亦然。两个前沿互不支配的部分
    # 越大，说明二者刻画的是不同的公平维度，新意成立。
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 公平性定义对比：人群组均值 vs 单用户最小最大 ──")
    try:
        # ⚠ 审计 #20：两臂的 f3 **定义不同**（individual_max 臂的 f3 是
        #   max_i t_i，恒 ≥ 人群组均值型），直接比较两条前沿的 f3 数值，
        #   34% 的差异至少有一部分是**定义的算术后果**，不是公平性观念的差别。
        #   公平的比较必须把两臂的解放到**同一口径**下重新评价。
        #   代码里早就构造了统一口径的 prob_probe，但从未使用——这里用上。
        fronts_eq: dict[str, list[np.ndarray]] = {}
        popX_eq: dict[str, list[np.ndarray]] = {}
        for measure in ("group_mean", "individual_max"):
            cfg_eq = base_cfg.override({"stage4_nsga2.equity_measure": measure})
            prob_eq = build_problem(
                pipe.res.demand.grid, pipe.res.candidates.gdf,
                pipe.res.access_time_s, cfg_eq,
            )
            ev_eq = make_evaluator(prob_eq)
            rep_eq = make_repair(prob_eq) if cfg_eq.get("stage4_nsga2.repair", True) else None
            reps_F, reps_X = [], []
            for rep in range(args.runs):
                res_eq = nsga2(
                    ev_eq, n_var=prob_eq.n_cand, cfg=cfg_eq,
                    pop_size=args.pop_size, n_gen=args.n_gen,
                    repair=rep_eq, seed=31337 + rep, verbose=False,
                    local_search=make_pruner(prob_eq), local_search_prob=LS_PROB,
                )
                F = np.asarray(res_eq.pareto_F, dtype=np.float64)
                if len(F):
                    reps_F.append(F)
                    reps_X.append(np.asarray(res_eq.pareto_X, dtype=np.float64))
            # 前沿取并集（各自口径下的最好结果），X 保留每个解
            fronts_eq[measure] = (np.vstack(reps_F) if reps_F
                                  else np.empty((0, 4)))
            popX_eq[measure] = (np.vstack(reps_X) if reps_X
                                else np.empty((0, prob_eq.n_cand)))
            log.info("   %-14s 完成 %d 次运行，合并前沿 %d 个解",
                     measure, args.runs, len(fronts_eq[measure]))

        # -- 统一口径重评：两臂的解都放到 group_mean 目标空间里 -------------
        cfg_uni = base_cfg.override({"stage4_nsga2.equity_measure": "group_mean"})
        prob_uni = build_problem(
            pipe.res.demand.grid, pipe.res.candidates.gdf,
            pipe.res.access_time_s, cfg_uni,
        )
        ev_uni = make_evaluator(prob_uni)
        F_uni = {m: np.atleast_2d(ev_uni(popX_eq[m])) for m in fronts_eq
                 if len(popX_eq[m])}

        def _cross(A, B, label_a, label_b):
            n_ab = sum(any(dominates(rb, ra) for rb in B) for ra in A)
            return {
                "direction": f"{label_a} 被 {label_b} 支配的解数",
                "count": n_ab, "total": len(A),
                "pct": 100 * n_ab / max(len(A), 1),
            }

        rows = [
            _cross(F_uni["group_mean"], F_uni["individual_max"],
                   "group_mean", "individual_max"),
            _cross(F_uni["individual_max"], F_uni["group_mean"],
                   "individual_max", "group_mean"),
        ]
        df_eq = pd.DataFrame(rows)
        # 同时给出**未统一**口径下的互支配（旧口径），供对照说明差异来源
        rows_raw = [
            _cross(fronts_eq["group_mean"], fronts_eq["individual_max"],
                   "group_mean", "individual_max"),
            _cross(fronts_eq["individual_max"], fronts_eq["group_mean"],
                   "individual_max", "group_mean"),
        ]
        df_eq["pct_note"] = "统一口径（group_mean 目标空间，n=%d 次运行）" % args.runs
        df_eq.to_csv(out_dir / "equity_measure_comparison.csv", index=False)
        log.info("")
        log.info("   【统一口径】两臂的解都在 group_mean 目标空间重评：")
        log.info(df_eq[["direction", "count", "total", "pct"]].to_string(index=False))
        log.info("   【各自口径】未统一时（旧做法，仅作对照）：")
        for r in rows_raw:
            log.info("      %s: %d/%d (%.1f%%)",
                     r["direction"], r["count"], r["total"], r["pct"])

        # 关键指标：统一口径下两臂能达到的公平性最优值
        gap_uni = {}
        for m in F_uni:
            gap_uni[m] = {
                "f3_min": float(F_uni[m][:, 2].min()),
                "f3_median": float(np.median(F_uni[m][:, 2])),
                "front_size": int(len(F_uni[m])),
            }
            log.info("   %-14s 统一口径 f3 最小 %.4f / 中位 %.4f",
                     m, gap_uni[m]["f3_min"], gap_uni[m]["f3_median"])

        any_cross = sum(r["count"] > 0 for r in rows)
        log.info(
            "   判读：统一口径下若**两个方向都存在互相支配的解**，说明两种公平性"
            "定义刻画的是**不同的**公平维度，「人群组公平」不是 p-中心的重复表述，"
            "新意 (A) 成立。若某一方完全支配另一方，则应如实报告后者不构成独立贡献。"
        )
        with open(out_dir / "equity_measure_summary.json", "w", encoding="utf-8") as fh:
            json.dump({"n_reps_per_arm": args.runs,
                       "comparison_unified": rows,
                       "comparison_raw": rows_raw,
                       "unified_group_mean_objectives": gap_uni,
                       "group_mean_front_size": int(len(fronts_eq["group_mean"])),
                       "individual_max_front_size": int(len(fronts_eq["individual_max"]))},
                      fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning("公平性定义对比失败: %s", str(exc)[:250])

    # ------------------------------------------------------------------
    # 7. 决策层指标的跨运行分布
    # ------------------------------------------------------------------
    log.info("")
    log.info("── 拐点解的指标跨运行分布（论文可报告为 均值 ± 标准差）──")
    dec_rows = []
    for i, F in enumerate(fronts):
        if len(F) == 0:
            continue
        k = knee_point(F)
        # 拐点解的决策变量：由该次运行的 pareto_X 给出；此处用目标值即可
        dec_rows.append({
            "run": i,
            "unserved": float(F[k, 0]), "total_time": float(F[k, 1]),
            "inequity": float(F[k, 2]), "cost": float(F[k, 3]),
        })
    if dec_rows:
        df_dec = pd.DataFrame(dec_rows)
        df_dec.to_csv(out_dir / "multirun_knee.csv", index=False)
        for col in df_dec.columns:
            if col == "run":
                continue
            v = df_dec[col].to_numpy(dtype=np.float64)
            log.info(
                "   %-12s %.4g ± %.4g", col, np.nanmean(v),
                np.nanstd(v, ddof=1) if len(v) > 1 else 0.0,
            )

    # ------------------------------------------------------------------
    with open(out_dir / "multirun_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "runs": args.runs, "pop_size": args.pop_size, "n_gen": args.n_gen,
            "data_mode": pipe.res.data_mode,
            "metrics": summary,
            "reference_front_size": int(len(ref_front)),
        }, fh, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 76)
    log.info(" 完成。输出目录: %s", out_dir)
    log.info(" 文件: multirun_runs.csv / multirun_metrics.csv / "
             "baseline_vs_front.csv / weighted_sum_vs_front.csv / multirun_knee.csv / pymoo_validation.json")
    log.info("=" * 76)
    return 0


def _nondominated(F: np.ndarray) -> np.ndarray:
    """返回非支配解的布尔掩码。"""
    from evtol_siting.nsga2_core import fast_non_dominated_sort

    if len(F) == 0:
        return np.zeros(0, dtype=bool)
    m = np.zeros(len(F), dtype=bool)
    m[fast_non_dominated_sort(F)[0]] = True
    return m


if __name__ == "__main__":
    raise SystemExit(main())
