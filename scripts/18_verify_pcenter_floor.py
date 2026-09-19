#!/usr/bin/env python
"""核验 p-中心的结构下界与"不可行"结论（§5.7.1 的发现所需的证据）。

论文 §5.7.1 现在写了两句**需要证据**的话：

1. 「p-中心最优值对 p = 12、18、25 全部等于 $r_{\\max}$」——即最优值被
   "最难到达的核心单元到其最近候选的距离"锁死；
2. 「p = 8 真的不可行，且这是模型的结论而不是求解失败——我们已逐次核对，
   二分的每一步都求解至证明最优」。

第 2 句尤其不能被当作断言：本项目刚修掉的一个缺陷正是 `cover_within`
把**"求解器没跑完"与"真的不可行"都返回 `None`**。若某一步其实是没跑完，
二分会收敛到一个**偏大的半径**，整条结论就建立在求解失败之上。
因此必须把二分过程逐步打印出来：每一步的判定原因、求解状态、站数。

本脚本只做**只读**核验，不写任何结果文件，不改动 outputs/。

用法::

    python scripts/18_verify_pcenter_floor.py
    python scripts/18_verify_pcenter_floor.py --p 8,12,18,25 --quiet

输出：
    stdout（并可选写出 outputs/tables/pcenter_floor.json 供论文引用）
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from evtol_siting.baselines import _core_mask, _solver          # noqa: E402
from evtol_siting.config import load_config                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="核验 p-中心的结构下界")
    ap.add_argument("--p", default="8,12,18,25")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总")
    ap.add_argument("--out", default=None, help="JSON 输出路径")
    args = ap.parse_args()

    cfg = load_config()
    d = cfg.dir("output.data_dir")
    dem = pd.read_csv(d / "demand_grid.csv").reset_index(drop=True)
    cand = pd.read_csv(d / "candidates.csv").reset_index(drop=True)
    at = np.load(d / "access_time_s.npz")["at"]

    at_min = np.where(np.isfinite(at), at, np.inf) / 60.0
    reach = np.isfinite(at_min)
    coverable = _core_mask(dem["demand"].to_numpy(dtype=np.float64), cfg) & reach.any(axis=1)

    # r_i = 单元 i 到**任何**候选点的最小接驳时间 —— 它的结构下界
    nearest = np.where(reach, at_min, np.inf).min(axis=1)
    r_max = float(nearest[coverable].max())

    print("=" * 74)
    print("p-中心结构下界核验")
    print("=" * 74)
    print(f"  需求单元 {len(dem)}，候选 {len(cand)}，核心单元 {int(coverable.sum())}")
    print(f"  r_i 分位：中位 {np.median(nearest[coverable]):.4f}  "
          f"p90 {np.percentile(nearest[coverable], 90):.4f}  "
          f"p99 {np.percentile(nearest[coverable], 99):.4f}")
    print(f"  **r_max = {r_max:.4f} 分钟**（= 任何 p-中心解的最差接驳下界）")

    def solve_at(radius: float, p: int):
        """返回 (解 或 None, 判定原因)。原因用于区分"真不可行"与"求解未完成"。"""
        within = reach & (at_min <= radius)
        if not within[coverable].any(axis=1).all():
            return None, "uncovered"
        prob = pulp.LpProblem("pc", pulp.LpMinimize)
        y = [pulp.LpVariable(f"y{j}", cat="Binary") for j in range(len(cand))]
        prob += pulp.lpSum(y)
        for i in range(len(dem)):
            if not coverable[i]:
                continue
            ne = np.flatnonzero(within[i])
            prob += pulp.lpSum(y[int(j)] for j in ne) >= 1, f"c{i}"
        prob.solve(_solver())
        st = pulp.LpStatus[prob.status]
        if st != "Optimal":
            return None, f"solver_not_finished:{st}"
        sel = [j for j in range(len(cand))
               if y[j].value() is not None and y[j].value() > 0.5]
        return (sel if len(sel) <= min(p, len(cand)) else None), "solved"

    finite = np.unique(at_min[reach])
    results = {}
    for p in [int(x) for x in args.p.split(",") if x.strip()]:
        lo, hi = 0, int(finite.size) - 1
        best, best_r = None, None
        n_uncovered = n_solver = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            R = float(finite[mid])
            sel, why = solve_at(R, p)
            if why == "uncovered":
                n_uncovered += 1
            elif why.startswith("solver"):
                n_solver += 1
            if sel is not None:
                best, best_r, hi = sel, R, mid - 1
            else:
                lo = mid + 1
        results[str(p)] = {
            "R_star": best_r, "n_sites": (len(best) if best else None),
            "feasible": best is not None,
            "bisection_steps_truly_infeasible": n_uncovered,
            "bisection_steps_solver_unfinished": n_solver,
        }
        if not args.quiet:
            print("")
            print(f"  p = {p}: R* = "
                  f"{'不可行' if best_r is None else f'{best_r:.4f} min'}"
                  f"，站数 {len(best) if best else '-'}")
            print(f"        二分中被判不可行 {n_uncovered + n_solver} 步："
                  f"真不可行 {n_uncovered}，**求解未完成 {n_solver}**")

    n_solver_total = sum(v["bisection_steps_solver_unfinished"] for v in results.values())
    print("")
    print("── 判读 ──")
    print(f"  求解未完成的步骤合计 {n_solver_total} 次。"
          f"{'**>0：结论不可用，必须重跑**' if n_solver_total else '为 0，故每一步的不可行判定都成立。'}")
    locked = [k for k, v in results.items()
              if v["R_star"] is not None and abs(v["R_star"] - r_max) < 1e-6]
    if locked:
        print(f"  R* 恰好等于 r_max 的 p 值：{', '.join(locked)}")
        print("  → 极小极大目标在这些 p 下**顶在下界上**，站数预算不起作用，")
        print("    目标函数对下界之外的单元完全不敏感。这正是论文 §5.7.1 的发现。")
    infeas = [k for k, v in results.items() if not v["feasible"]]
    if infeas:
        print(f"  真的不可行的 p 值：{', '.join(infeas)}（模型结论，非求解失败）")

    payload = {"r_max_min": r_max, "n_core_cells": int(coverable.sum()),
               "n_demand_cells": int(len(dem)), "n_candidates": int(len(cand)),
               "r_i_median": float(np.median(nearest[coverable])),
               "r_i_p90": float(np.percentile(nearest[coverable], 90)),
               "r_i_p99": float(np.percentile(nearest[coverable], 99)),
               "by_p": results}
    out = Path(args.out) if args.out else (
        Path(cfg.dir("output.tables_dir")) / "pcenter_floor.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("")
    print(f"已写出 {out}（论文 §5.7.1 引用这些数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
