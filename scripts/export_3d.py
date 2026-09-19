#!/usr/bin/env python
"""导出三维可视化所需的两个 JS 文件。

用法
----
::

    # 真实数据：跑完整流水线，同时导出三维数据 + 论文表格（推荐）
    python scripts/export_3d.py --data-mode real --pop-size 100 --n-gen 100

    # 只导三维数据、不做基线对比（更快）
    python scripts/export_3d.py --data-mode real --no-tables

    # 离线用合成数据验证链路
    python scripts/export_3d.py --data-mode synthetic --no-nsga2

    # 只重新生成 token 文件
    python scripts/export_3d.py --token-only

生成的文件（输出到 ``viz3d/``）::

    cesium_token.js   window.CESIUM_ION_TOKEN = "..."
    evtol_data.js     window.EVTOL_DATA = {...}

本脚本会在**同一次运行**内完成：跑流水线 → 生成论文表格与基线对比 →
导出三维数据 → 落盘全部结果。真实数据一次完整运行需要数十分钟，分开跑
两遍纯属浪费，因此合并在一起。

打开三维视图（**不能双击 HTML**，必须经本地 HTTP 服务）::

    python viz3d/start_3d.py

为什么 token 单独放一个文件
---------------------------
三维页面本身是可以分享的（发给老师、贴进答辩材料），而 access token
属于个人凭证。把 token 隔离到独立文件，分享页面时只需剔除这一个文件，
不必担心凭证外泄。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import PROJECT_ROOT, load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

VIZ_DIR = PROJECT_ROOT / "viz3d"
TOKEN_SRC = VIZ_DIR / "cesium_token.local.txt"


def write_token_js(out: Path) -> bool:
    """把本地 token 文件转成浏览器可用的 JS 变量。"""
    if not TOKEN_SRC.exists():
        logging.error(
            "未找到 token 文件: %s\n请创建该文件，内容为一行 Cesium ion access token。",
            TOKEN_SRC,
        )
        return False

    token = ""
    for line in TOKEN_SRC.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            token = line
            break

    if not token:
        logging.error("token 文件存在但没有有效内容: %s", TOKEN_SRC)
        return False

    out.write_text(
        "// 由 scripts/export_3d.py 生成 —— 含个人凭证，请勿公开分享\n"
        f'window.CESIUM_ION_TOKEN = "{token}";\n',
        encoding="utf-8",
    )
    # 只打印前缀，避免把完整凭证写进日志
    logging.info("token 已写入 %s（%s…，%d 字符）", out.name, token[:12], len(token))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="导出三维可视化数据")
    ap.add_argument("--set", dest="sets", action="append", metavar="KEY=VALUE",
                    help="覆盖配置项（点号路径），可重复")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--n-gen", type=int, default=None)
    ap.add_argument("--limit-tiles", type=int, default=None, help="限制 OSM 下载瓦片数")
    ap.add_argument("--token-only", action="store_true", help="只生成 token 文件")
    ap.add_argument("--no-nsga2", action="store_true", help="跳过 NSGA-II（只导出 IP 方案）")
    ap.add_argument("--no-tables", action="store_true",
                    help="跳过论文表格与基线对比（只导三维数据，更快）")
    ap.add_argument("--no-baselines", action="store_true", help="跳过单目标基线求解")
    ap.add_argument("--max-demand-points", type=int, default=6000)
    ap.add_argument("--max-candidate-points", type=int, default=4000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "export_3d.log")
    log = logging.getLogger("export_3d")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    if not write_token_js(VIZ_DIR / "cesium_token.js"):
        log.warning("token 文件未生成，三维页面将无法加载底图（但数据仍会导出）")

    if args.token_only:
        return 0

    import yaml

    overrides = {}
    for item in args.sets or []:
        if "=" not in item:
            raise SystemExit(f"--set 应为 key=value，收到 {item!r}")
        k, _, raw = item.partition("=")
        overrides[k.strip()] = yaml.safe_load(raw.strip())

    cfg = load_config(overrides=overrides)

    # ------------------------------------------------------------------
    # 运行流水线（或复用已有结果）
    # ------------------------------------------------------------------
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.viz3d import build_bundle, export_bundle

    pipe = Pipeline(cfg, data_mode=args.data_mode, limit_overpass_tiles=args.limit_tiles)

    log.info("运行阶段一：候选起降场筛选")
    pipe.run_stage1()
    log.info("运行阶段二：K-Means 人群分类")
    pipe.run_stage2()
    log.info("运行阶段三：0-1 整数规划")
    pipe.run_stage3()

    if not args.no_nsga2:
        log.info("运行阶段四：NSGA-II 多目标优化")
        pipe.run_stage4(n_gen=args.n_gen, pop_size=args.pop_size)

    # 顺带产出论文表格。
    # 真实数据的一次完整运行（阶段一~四）在成都案例下需要数十分钟，若导出
    # 三维图与生成表格各跑一遍，就要等两倍时间。这里在同一次运行内一并完成。
    if not args.no_tables:
        try:
            log.info("计算方案指标与单目标基线对比")
            pipe.run_evaluation(run_baselines=not args.no_baselines)
        except Exception as exc:
            log.warning("评价阶段失败（不影响三维导出）: %s", str(exc)[:200])

    # ------------------------------------------------------------------
    # 选出三维视图默认展示的方案：优先 IP 的 cost_priority 情景
    # ------------------------------------------------------------------
    selected = None
    for s in pipe.res.ip_solutions:
        if getattr(s, "status", "") == "Optimal" and s.selected:
            selected = {
                "label": f"IP · {s.name}",
                "cand_ids": [int(i) for i in s.selected],
                "total_cost": float(s.total_cost),
                "coverage": s.coverage,
            }
            break

    if selected is None and pipe.res.nsga2_result is not None \
            and len(pipe.res.nsga2_result.pareto_F):
        # 退而求其次：用 NSGA-II 的拐点解
        from evtol_siting.stage4_nsga2 import knee_point

        F = pipe.res.nsga2_result.pareto_F
        X = pipe.res.nsga2_result.pareto_X
        k = knee_point(F)
        selected = {
            "label": "NSGA-II · 拐点解",
            "cand_ids": [int(i) for i in __import__("numpy").flatnonzero(X[k] > 0.5)],
            "total_cost": float(F[k, 3]) if F.shape[1] > 3 else 0.0,
            "coverage": {},
        }

    # 把接驳半径带进 meta，供页面绘制覆盖圈
    cfg.raw.setdefault("meta", {})
    bundle = build_bundle(
        cfg,
        candidates=pipe.res.candidates,
        demand=pipe.res.demand,
        ip_solutions=pipe.res.ip_solutions,
        nsga2_result=pipe.res.nsga2_result,
        selected_solution=selected,
        max_demand_points=args.max_demand_points,
        max_candidate_points=args.max_candidate_points,
    )
    bundle["meta"]["access_radius_m"] = float(cfg.get("stage3_ip.access_radius_m", 3000.0))
    bundle["meta"]["data_mode"] = pipe.res.data_mode
    if pipe.res.data_mode != "real":
        bundle["meta"]["warning"] = (
            "本数据包基于 SYNTHETIC 合成数据生成，仅用于演示流程，"
            "不可作为研究结论。"
        )
        log.warning("=" * 70)
        log.warning("  导出的是 **合成数据** 结果 —— 三维图仅供演示，非实证结论")
        log.warning("=" * 70)

    out = export_bundle(bundle, VIZ_DIR / "evtol_data.js")

    # 一并保存流水线的全部产物（表格 / 诊断 / 帕累托前沿）
    try:
        pipe.save()
    except Exception as exc:
        log.warning("保存结果失败: %s", str(exc)[:200])

    log.info("=" * 70)
    log.info(" 完成。打开三维视图：")
    log.info("   %s", VIZ_DIR / "chengdu_evtol_3d.html")
    log.info(" 数据: %s", out.name)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
