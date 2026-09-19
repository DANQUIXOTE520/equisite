#!/usr/bin/env python
"""票价情景扫描：把"票价是可调的政策杠杆"落成一张可追溯的表。

为什么单独成脚本
----------------
`EVTOLChoiceModel.fare_scenarios()` 此前只被 `09_figures.py` 调用一次，用来给
图 3 画一条曲线——**结果既不落盘也不成表**，而论文摘要却引用了"基准票价下类间
采用率之比 1.46 / 1.53、降至票价 30% 时 1.11 / 1.12"这两个数。审计发现：
outputs/ 里没有任何票价扫描的产物，两稿的数值还互相不一致。本脚本把该扫描
变成一等产物（CSV），使摘要里的每个数都可追溯。

用法::

    python scripts/13_fare_scenarios.py
    python scripts/13_fare_scenarios.py --multipliers 0.3,0.5,0.75,1.0,1.25,1.5

输出：
    outputs/tables/fare_scenarios.csv   每个票价倍数一行的汇总
    outputs/logs/fare_scenarios.log     运行日志
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.choice_model import EVTOLChoiceModel  # noqa: E402
from evtol_siting.config import load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

log = logging.getLogger("fare_scenarios")

# 默认网格刻意包含 0.3（论文摘要引用了"票价降至 30%"这一点，而
# 原先的默认网格 [0.4,0.6,0.8,1.0,1.2,1.5] 里**没有 0.3**，
# 摘要那个数根本无从产生）。
DEFAULT_MULTS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multipliers", default=None,
                    help="逗号分隔的票价倍数，默认 %s" % DEFAULT_MULTS)
    ap.add_argument("--data-mode", default="auto",
                    choices=["auto", "real", "synthetic"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(verbose=args.verbose,
                  log_file=Path(cfg.dir("output.data_dir")).parent / "logs"
                  / "fare_scenarios.log")

    mults = ([float(x) for x in args.multipliers.split(",")]
             if args.multipliers else DEFAULT_MULTS)

    clusters_path = cfg.dir("output.data_dir") / "table_clusters.csv"
    if not clusters_path.exists():
        log.error("未找到 %s；请先跑流水线（scripts/run_all.py）。", clusters_path)
        return 1
    clusters = pd.read_csv(clusters_path)
    log.info("人群分层: %s", clusters_path)
    log.info("票价倍数网格: %s", mults)

    model = EVTOLChoiceModel(cfg)
    tab = model.fare_scenarios(clusters, multipliers=mults)

    out_dir = cfg.dir("output.tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fare_scenarios.csv"
    tab.to_csv(out, index=False)

    base = float(cfg.get("choice_model.evtol_fare_multiplier", 1.0))
    b = tab.loc[(tab["fare_multiplier"] - base).abs().idxmin()]
    log.info("")
    log.info("基准票价倍数 %.2f：类间采用率之比 %.4f", base, b["p_ratio_max_min"])
    for _, r in tab.iterrows():
        log.info("  倍数 %.2f -> 类间比 %.4f，总体加权采用率 %.4f%%",
                 r["fare_multiplier"], r["p_ratio_max_min"],
                 100.0 * r["p_population_weighted"])
    log.info("")
    log.info("已写出 %s（%d 行）", out, len(tab))

    # 摘要要用的一句话，直接算出来，避免再手抄出错
    lo = tab["fare_multiplier"].min()
    r_lo = tab.loc[tab["fare_multiplier"].idxmin()]
    log.info("摘要口径：基准票价下类间比 %.2f，票价降至 %.0f%% 时收窄至 %.2f",
             b["p_ratio_max_min"], 100 * lo, r_lo["p_ratio_max_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
