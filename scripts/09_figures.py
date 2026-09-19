#!/usr/bin/env python
"""从已保存的结果生成论文图件。

用法
----
::

    python scripts/09_figures.py                 # 生成全部图件
    python scripts/09_figures.py --only pareto,height_fusion

输出去向
--------
默认输出到项目根的 ``论文稿件/图件/``，与论文稿件放在一起。
可用 ``--out`` 指向别处。

依赖
----
只依赖 ``outputs/`` 下已保存的 CSV 与 numpy 数组，**不需要重跑流水线**。
因此可以随时调整图件样式并秒级重新生成。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import PROJECT_ROOT, load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("figures")


def default_out_dir(cfg) -> Path:
    """图件默认输出目录 —— **按城市区分**，避免跨城互相覆盖。

    ⚠ 早期版本写死为 ``论文稿件/图件``，与城市无关。多城市验证时这是
    一个**静默**故障：用 ``EVTOL_CONFIG`` 先后跑两座城市时，后跑的图会把
    先跑的图**同名覆盖**，而两边的日志都显示"图件已生成"。
    论文里若同时引用两城结果，就会出现图文不符且无从察觉。

    命名规则：主案例（成都）沿用 ``图件`` 以保持既有稿件不变；其余城市
    用 ``图件_<配置文件名>``。判据取自 ``cfg.source_path`` 而非硬编码，
    因此新增城市无需改代码。
    """
    slug = Path(cfg.source_path).stem          # 取自配置文件名，如 chengdu
    base = PROJECT_ROOT.parent / "论文稿件"
    return base / "图件" if slug == "chengdu" else base / f"图件_{slug}"


def _load_results(cfg):
    """载入流水线保存的全部结果。"""
    import numpy as np
    import pandas as pd

    d = cfg.dir("output.data_dir")
    r: dict = {}

    def _csv(name):
        p = d / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception as exc:
                log.warning("读取 %s 失败: %s", name, str(exc)[:120])
        else:
            log.warning("缺少结果文件: %s（可先运行 scripts/run_all.py）", name)
        return None

    r["candidates"] = _csv("candidates.csv")
    r["clusters"] = _csv("table_clusters.csv")
    r["height"] = _csv("table_height_sources.csv")
    r["pareto"] = _csv("pareto_front.csv")
    r["convergence"] = _csv("convergence.csv")
    r["ip"] = _csv("table_ip_scenarios.csv")
    r["metrics"] = _csv("table_solution_metrics.csv")
    r["grid"] = _csv("demand_grid.csv")

    at = d / "access_time_s.npz"
    if at.exists():
        try:
            r["access_time"] = np.load(at)["at"]
        except Exception as exc:
            log.warning("读取接驳时间矩阵失败: %s", str(exc)[:120])

    sens = Path(cfg.dir("output.tables_dir")) / "sensitivity.csv"
    if sens.exists():
        r["sensitivity"] = pd.read_csv(sens)

    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成论文图件")
    ap.add_argument("--out", default=None, help="输出目录（默认 论文稿件/图件）")
    ap.add_argument("--only", default=None, help="只生成指定图件，逗号分隔")
    ap.add_argument("--formats", default="pdf,png", help="输出格式，逗号分隔")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose, PROJECT_ROOT / "outputs" / "logs" / "figures.log")

    from evtol_siting.viz import figures as F

    cfg = load_config()
    out_dir = Path(args.out) if args.out else default_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("图件输出目录: %s（配置 %s）", out_dir, cfg.source_path.name)

    r = _load_results(cfg)
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    def want(name: str) -> bool:
        return only is None or name in only

    made: list[Path] = []

    # 图 1：研究区与候选
    if want("study_area") and r.get("candidates") is not None:
        try:
            import geopandas as gpd

            airports = None
            # 机场位置从原始数据取；缺失时用已知坐标兜底
            try:
                from evtol_siting.data import pbf

                pbf_path = pbf.default_pbf_path(cfg)
                if pbf_path.exists():
                    airports = pbf.parse_all_layers(pbf_path, cfg, layers=["airports"])["airports"]
            except Exception as exc:
                log.debug("机场数据读取失败: %s", str(exc)[:100])

            grid = r.get("grid")
            if grid is not None and {"centroid_x", "centroid_y"} <= set(grid.columns):
                pass
            else:
                from evtol_siting.geo import make_grid

                grid = make_grid(cfg)

            # 默认选中方案 = IP 成本优先情景
            sel_ids = None
            if r.get("ip") is not None and len(r["ip"]):
                sel_ids = _selected_ids_from_ip(cfg)

            made += F.fig_study_area(cfg, r["candidates"], grid, airports,
                                     selected_ids=sel_ids, out_dir=out_dir)
        except Exception as exc:
            log.error("图 1 生成失败: %s", str(exc)[:250])

    # 图 2：高度融合
    #
    # ⚠ 这里曾写成 `tag_only={"tag_only": 227, "fused": 1430}` —— 一对**硬编码的
    #   字面量**，没有任何计算产生过它们，而正文 §3.3、摘要、表 3 与表 11 都引用。
    #   实测（scripts/20_height_source_comparison.py）真值是 256 / 1 517，倍数
    #   5.93× 而非图注写的 6.3×。现在改为从产物读取，读不到就报错。
    if want("height_fusion") and r.get("height") is not None:
        try:
            cmp_path = cfg.dir("output.tables_dir") / "height_source_comparison.json"
            if not cmp_path.exists():
                raise FileNotFoundError(
                    f"缺少 {cmp_path}；请先运行 "
                    "`python scripts/20_height_source_comparison.py`")
            hcmp = json.loads(cmp_path.read_text(encoding="utf-8"))
            made += F.fig_height_fusion(
                r["height"], out_dir,
                tag_only={"tag_only": hcmp["tag_only_rooftop"],
                          "fused": hcmp["fused_rooftop"]},
            )
            log.info("  图 2 的标签/融合对照取自 %s（%d vs %d）",
                     cmp_path.name, hcmp["tag_only_rooftop"],
                     hcmp["fused_rooftop"])
        except Exception as exc:
            log.error("图 2 生成失败: %s", str(exc)[:250])

    # 图 3：人群分类与选择模型
    if want("demand_clusters") and r.get("clusters") is not None:
        try:
            from evtol_siting.choice_model import EVTOLChoiceModel

            fare_tab = None
            try:
                m = EVTOLChoiceModel(cfg)
                fare_tab = m.fare_scenarios(r["clusters"])
            except Exception as exc:
                log.debug("票价情景计算失败: %s", str(exc)[:120])
            made += F.fig_demand_clusters(r["clusters"], out_dir, fare_tab)
        except Exception as exc:
            log.error("图 3 生成失败: %s", str(exc)[:250])

    # 图 3b：需求按人群类的空间分布（正文 §5.2）
    if want("demand_by_group") and r.get("grid") is not None:
        try:
            airports = None
            try:
                from evtol_siting.data import pbf

                pbf_path = pbf.default_pbf_path(cfg)
                if pbf_path.exists():
                    airports = pbf.parse_all_layers(
                        pbf_path, cfg, layers=["airports"])["airports"]
            except Exception as exc:
                log.debug("机场数据读取失败: %s", str(exc)[:100])
            made += F.fig_demand_by_group(r["grid"], out_dir, cfg=cfg,
                                          airports=airports)
        except Exception as exc:
            log.error("需求人群分布图生成失败: %s", str(exc)[:250])

    # 图 4：帕累托前沿
    if want("pareto") and r.get("pareto") is not None:
        try:
            from evtol_siting.stage4_nsga2 import knee_point

            ki = None
            try:
                ki = knee_point(r["pareto"].to_numpy(dtype=float))
            except Exception:
                pass
            made += F.fig_pareto(r["pareto"], out_dir, knee_idx=ki)
        except Exception as exc:
            log.error("图 4 生成失败: %s", str(exc)[:250])

    # 图 5：收敛
    if want("convergence") and r.get("convergence") is not None:
        try:
            made += F.fig_convergence(r["convergence"], out_dir)
        except Exception as exc:
            log.error("图 5 生成失败: %s", str(exc)[:250])

    # 图 6：机场缓冲敏感性
    if want("airport_sensitivity") and r.get("sensitivity") is not None:
        try:
            made += F.fig_airport_sensitivity(r["sensitivity"], out_dir)
        except Exception as exc:
            log.error("图 6 生成失败: %s", str(exc)[:250])

    # 图 7：方案对比
    if want("solution_comparison") and r.get("metrics") is not None:
        try:
            made += F.fig_solution_comparison(r["metrics"], out_dir)
        except Exception as exc:
            log.error("图 7 生成失败: %s", str(exc)[:250])

    # 图 8：选择模型机制
    if want("choice_model"):
        try:
            made += F.fig_choice_model_mechanism(cfg, out_dir)
        except Exception as exc:
            log.error("图 8 生成失败: %s", str(exc)[:250])

    # 图 9：接驳可达性
    if want("accessibility") and r.get("access_time") is not None \
            and r.get("candidates") is not None:
        try:
            sel_ids = _selected_ids_from_ip(cfg)
            if sel_ids:
                made += F.fig_accessibility(cfg, r["grid"], r["candidates"],
                                            r["access_time"], sel_ids, out_dir)
        except Exception as exc:
            log.error("图 9 生成失败: %s", str(exc)[:250])

    log.info("=" * 66)
    log.info(" 完成，共生成 %d 个文件", len(made))
    for p in sorted(set(made)):
        log.info("   %s", p.name)
    log.info(" 输出目录: %s", out_dir)
    log.info("=" * 66)
    return 0


def _selected_ids_from_ip(cfg):
    """从 IP 情景结果中取出默认展示方案的候选编号。

    需要重算一次 IP 才能拿到选中集合（结果 CSV 只存了汇总指标）。
    为避免重跑，这里改为从三维数据包读取——它已经存了各方案的
    ``cand_ids``。
    """
    import json

    js = PROJECT_ROOT / "viz3d" / "evtol_data.js"
    if not js.exists():
        return None
    try:
        txt = js.read_text(encoding="utf-8")
        data = json.loads(txt.split("=", 1)[1].rstrip().rstrip(";"))
        sols = data.get("solutions", [])
        for s in sols:
            if s.get("label", "").startswith("IP"):
                return list(s.get("cand_ids", []))
        if sols:
            return list(sols[0].get("cand_ids", []))
    except Exception as exc:
        log.debug("从三维数据包读取方案失败: %s", str(exc)[:120])
    return None


if __name__ == "__main__":
    raise SystemExit(main())
