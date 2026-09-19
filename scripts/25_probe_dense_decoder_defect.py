"""复现"解码器只补点 + 稠密初始种群 → 前沿被锁在高成本区"这一缺陷。

## 为什么要有这个脚本

论文里有一句对**修复前配置**的描述：解码器只补点时，稠密初始种群使搜索
无法删点，*"整个帕累托前沿落在 316–384 亿元"*。这句话的**唯一出处是一段
代码注释**（`src/evtol_siting/nsga2_core.py` 的"初始种群"一节），
而不是任何产物文件——全库检索找不到能复现它的 CSV/JSON。

更麻烦的是，注释里量的是 **1 332 个候选**的旧算例，而正文引用时把
整数规划那半边更新成了当前口径（3.14 亿元 / 96.13 %），前半句却原样留着。
于是一句话里两个数字来自两个不同的算例。

本脚本把那个配置**按当前候选集重新跑一遍**，使这句话重新有产物支撑：

* 初始种群 = 逐基因独立 `U(0,1) < 0.5` 二值化（旧写法，期望密度 50%）
* `local_search=None` —— 即**没有**剪枝算子，只有"只加点"的修复

跑出来的前沿成本区间就是正文该写的数字。若它与 316–384 亿元接近，
说明该结论稳健；若明显不同，正文须改用新值（旧注释也一并更新）。

用法::

    python scripts/25_probe_dense_decoder_defect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import load_config                       # noqa: E402
from evtol_siting.logging_setup import setup_logging              # noqa: E402
from evtol_siting.nsga2_core import nsga2                         # noqa: E402
from evtol_siting.pipeline import Pipeline                        # noqa: E402
from evtol_siting.stage4_nsga2 import (build_problem, make_evaluator,  # noqa: E402
                                       make_repair)

POP, GEN = 100, 100


def main() -> int:
    setup_logging(False, None)
    cfg = load_config()
    pipe = Pipeline(cfg, data_mode="real", cache={})
    pipe.run_stage1()
    pipe.run_stage2()
    pipe.run_stage3()

    prob = build_problem(pipe.res.demand.grid, pipe.res.candidates.gdf,
                         pipe.res.access_time_s, cfg)
    evaluate = make_evaluator(prob)
    repair = make_repair(prob)

    # 旧写法：逐基因独立 U(0,1) < 0.5。期望密度 50%。
    rng = np.random.default_rng(42)
    dense = (rng.random((POP, prob.n_cand)) < 0.5).astype(np.float64)
    print(f"稠密初始种群：每体站点数 {dense.sum(axis=1).min():.0f}–"
          f"{dense.sum(axis=1).max():.0f}（候选 {prob.n_cand} 个，期望 50%）")

    res = nsga2(evaluate, n_var=prob.n_cand, cfg=cfg, pop_size=POP, n_gen=GEN,
                repair=repair, x0=dense, verbose=False,
                local_search=None,          # ← 关键：没有剪枝算子
                local_search_prob=0.0)
    F = np.asarray(res.pareto_F, dtype=np.float64)
    cost = F[:, 3] / 1e8

    print()
    print("=" * 62)
    print(f"前沿解数 {len(F)}")
    print(f"成本区间 **{cost.min():.1f} – {cost.max():.1f} 亿元**")
    print(f"站点数区间 {F[:, 0].size and ''}"
          f"{int(np.asarray(res.pareto_X).sum(axis=1).min())} – "
          f"{int(np.asarray(res.pareto_X).sum(axis=1).max())}")
    print(f"未服务需求最小 {F[:, 0].min():.1f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
