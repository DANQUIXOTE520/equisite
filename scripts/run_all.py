#!/usr/bin/env python
"""四阶段选址流水线主入口。

用法
----
::

    # 全流程（真实数据；首次运行需下载 OSM 与栅格，约 30–60 分钟）
    python scripts/run_all.py

    # 离线快速验证流程连通性（合成数据，几分钟）
    python scripts/run_all.py --data-mode synthetic --pop-size 40 --n-gen 20

    # 只跑到某个阶段
    python scripts/run_all.py --until stage2

    # 覆盖任意参数（点号路径，可用多次）
    python scripts/run_all.py --set stage1_candidates.rooftop.min_roof_area_m2=1400
    python scripts/run_all.py --set stage3_ip.access_radius_m=2000 --set stage2_demand.n_clusters=4

    # 用 pymoo 交叉验证自实现的 NSGA-II
    python scripts/run_all.py --validate-nsga2

    # 只下载数据（供检查数据可用性，不跑模型）
    python scripts/run_all.py --fetch-only

输出
----
``outputs/data/`` 下的 CSV / NPZ / JSON；日志打印到 stdout。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 允许从仓库根目录直接运行脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import PROJECT_ROOT, load_config  # noqa: E402


def _parse_set(pairs: list[str] | None) -> dict:
    """把 ``--set a.b=1 --set c.d=x`` 解析成覆盖字典。

    值按 YAML 规则解析，因此 ``1400`` 得到 int、``true`` 得到 bool、
    ``[1,2]`` 得到 list——无需为类型单独加参数。
    """
    import yaml

    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set 参数格式应为 key=value，收到: {item!r}")
        key, _, raw = item.partition("=")
        out[key.strip()] = yaml.safe_load(raw.strip())
    return out


from evtol_siting.logging_setup import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="基于 GIS 与多目标优化的 eVTOL 起降场选址 —— 四阶段流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default=None, help="配置文件路径（默认 config/chengdu.yaml）")
    ap.add_argument("--set", dest="sets", action="append", metavar="KEY=VALUE",
                    help="覆盖配置项，点号路径；可重复")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto",
                    help="auto=优先真实数据失败则回退合成（默认）；real=强制真实；synthetic=强制合成")
    ap.add_argument("--until", choices=["data", "stage1", "stage2", "stage3", "stage4", "eval"],
                    default="eval", help="执行到哪个阶段为止")
    ap.add_argument("--pop-size", type=int, default=None, help="NSGA-II 种群规模")
    ap.add_argument("--n-gen", type=int, default=None, help="NSGA-II 迭代代数")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="限制 OSM 下载瓦片数（快速试跑）")
    ap.add_argument("--no-baselines", action="store_true", help="跳过单目标基线求解")
    ap.add_argument("--validate-nsga2", action="store_true",
                    help="用 pymoo 的 NSGA-II 交叉验证自实现版本")
    ap.add_argument("--fetch-only", action="store_true", help="仅下载数据，不跑模型")
    ap.add_argument("--run-tag", default=None,
                    help="本次实验标签，用于输出目录命名（如 sens_radius_2000）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    tag = args.run_tag or time.strftime("run_%Y%m%d_%H%M%S")
    log_file = PROJECT_ROOT / "outputs" / "logs" / f"{tag}.log"
    setup_logging(args.verbose, log_file)

    log = logging.getLogger("run_all")
    log.info("=" * 78)
    log.info(" 基于 GIS 与多目标优化的 eVTOL 起降场选址 —— 四阶段集成方法")
    log.info(" run tag: %s", tag)
    log.info("=" * 78)

    overrides = _parse_set(args.sets)
    cfg = load_config(args.config, overrides=overrides)
    log.info("\n%s", cfg.describe())

    if args.fetch_only:
        from evtol_siting.pipeline import Pipeline

        pipe = Pipeline(cfg, data_mode="real", limit_overpass_tiles=args.limit_tiles)
        data = pipe.load_data()
        for k, v in data.items():
            n = len(v) if hasattr(v, "__len__") else v
            log.info("  %-20s %s", k, n)
        log.info("数据下载完成。")
        return 0

    from evtol_siting.pipeline import Pipeline

    pipe = Pipeline(cfg, data_mode=args.data_mode, limit_overpass_tiles=args.limit_tiles)
    t0 = time.time()

    # 按 --until 分阶段执行，便于中途检查中间结果
    if args.until == "data":
        pipe.load_data()
    else:
        pipe.run_stage1()
        log.info("阶段一候选集:\n%s", pipe.res.candidates.summary().to_string())

        if args.until != "stage1":
            dm = pipe.run_stage2()
            log.info("阶段二人群画像:\n%s", dm.cluster_profile.to_string(index=False))

            if args.until != "stage2":
                sols = pipe.run_stage3()
                for s in sols:
                    log.info("  %-22s %-12s %2d 站  %.2f 亿元  需求覆盖 %.1f%%",
                             s.name, s.status, s.n_sites, s.total_cost / 1e8,
                             s.coverage.get("demand_coverage_pct", float("nan")))

                if args.until != "stage3":
                    pipe.run_stage4(n_gen=args.n_gen, pop_size=args.pop_size,
                                    validate=args.validate_nsga2)

                    if args.until == "eval":
                        pipe.run_evaluation(run_baselines=not args.no_baselines)

    pipe.save()
    log.info("=" * 78)
    log.info(" 完成。耗时 %.1f 分钟。输出目录: %s", (time.time() - t0) / 60,
             cfg.get("output.data_dir"))
    log.info(" 运行日志: %s", log_file)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
