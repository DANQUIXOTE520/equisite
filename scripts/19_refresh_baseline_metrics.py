#!/usr/bin/env python
"""重建 ``baseline_vs_front.csv`` 中由 ``solution_metrics`` 派生的三列。

为什么需要它
------------
``baseline_vs_front.csv`` 有 11 列，其中 8 列来自**帕累托前沿的目标值评估**
（未服务、总接驳、公平性、成本、被多少前沿解支配……），3 列来自
``metrics.solution_metrics``（需求覆盖率、最差人群接驳、公平性 Gini）。

后者曾因一个**我在修 A8 时引入的口径错误**而全部失真：当时的实现把
"整组需求"当分母、把未服务单元的接驳时间记成哨兵 $10^{12}$ 计入分子，
于是**只要组里有任何一个未服务单元，整组均值就被吞掉**——12 行里有 10 行
的最差人群接驳变成了 $10^{12}$。正确规则是"整组无人被服务"时才置哨兵。

由于受影响的三列**不需要重跑 NSGA-II**（它们只依赖基线解本身、需求与阻抗
矩阵），本脚本从基线缓存直接重算，避免为三列重跑 55 分钟。

用法::

    python scripts/19_refresh_baseline_metrics.py --dry-run
    python scripts/19_refresh_baseline_metrics.py
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evtol_siting.config import load_config                      # noqa: E402
from evtol_siting.metrics import solution_metrics                # noqa: E402

# 这三列由 solution_metrics 派生，其余列保持原样（来自前沿评估，不受影响）
DERIVED = ["demand_coverage_pct", "worst_group_access_time_min", "equity_gini"]
ADDED = ["n_groups_unserved", "groups_all_served", "group_gap_min"]


def main() -> int:
    ap = argparse.ArgumentParser(description="重建基线对照表中的派生指标列")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写回")
    args = ap.parse_args()

    cfg = load_config()
    d = Path(cfg.dir("output.data_dir"))
    tab_dir = Path(cfg.dir("output.tables_dir"))
    csv_path = tab_dir / "baseline_vs_front.csv"
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}")

    dem = pd.read_csv(d / "demand_grid.csv")
    cand = pd.read_csv(d / "candidates.csv")
    at = np_load(d / "access_time_s.npz")

    cache_dir = d.parent / "cache"
    files = sorted(cache_dir.glob("baselines_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("找不到基线缓存")
    blob = json.loads(files[-1].read_text("utf-8"))
    print(f"基线缓存: {files[-1].name}（version {blob.get('version')}）")
    sols = {s["name"]: s for s in blob["solutions"]}

    df = pd.read_csv(csv_path)
    print(f"读取 {csv_path}（{len(df)} 行）")
    before = df[DERIVED].copy()

    for i, row in df.iterrows():
        name = row["baseline"]
        s = sols.get(name)
        if s is None:
            print(f"  [{name}] 缓存中无此解，跳过")
            continue
        m = solution_metrics(s["selected"], dem, cand, at, cfg, label=name)
        for c in DERIVED:
            df.at[i, c] = m.get(c)
        for c in ADDED:
            df.at[i, c] = m.get(c)

    print("")
    print("── 改动前后对比（只列有变化的行）──")
    for i in range(len(df)):
        old = before.iloc[i]
        new = df.iloc[i]
        changed = any(
            (pd.isna(old[c]) != pd.isna(new[c]))
            or (not pd.isna(old[c]) and not pd.isna(new[c])
                and abs(float(old[c]) - float(new[c])) > 1e-9)
            for c in DERIVED
        )
        if changed:
            print(f"  {new['baseline']:<20} "
                  f"最差人群接驳 {float(old['worst_group_access_time_min']):.4g} "
                  f"-> {float(new['worst_group_access_time_min']):.4g}"
                  f"   （整组未服务组数 {new.get('n_groups_unserved')}）")

    if args.dry_run:
        print("")
        print("--dry-run：未写回。")
        return 0

    bak = csv_path.with_suffix(".csv.bak_buggy_metrics")
    shutil.copy2(csv_path, bak)
    df.to_csv(csv_path, index=False)
    print("")
    print(f"已写回 {csv_path}（旧版备份 {bak.name}）")
    return 0


def np_load(p: Path):
    import numpy as np
    return np.load(p)["at"]


if __name__ == "__main__":
    raise SystemExit(main())
