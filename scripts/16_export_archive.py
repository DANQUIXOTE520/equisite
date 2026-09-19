#!/usr/bin/env python
"""把可复现性材料归档到 ``论文稿件/数据/``。

为什么需要它
------------
论文附录 A 声称「全部代码、配置与中间产物均已归档」，但审计发现
``论文稿件/数据/`` **是一个空目录**——声称与事实不符，而且这是审稿人
按图索骥时第一个会去翻的地方。

本脚本把该声称变成事实，并且**可重跑**：任何一次结果更新后重新执行，
归档内容随之同步，不会出现"归档的是三个月前的旧数"这种更难发现的错。

归档内容
--------
1. **配置快照** —— 主配置 + 本次运行的全部覆盖项（论文每个数字都由它们决定）
2. **表格产物** —— ``outputs/tables/*.csv``（正文与附录的每张表）
3. **关键中间产物** —— 候选集、需求栅格与人群分层、IP 情景解、帕累托前沿
4. **运行清单** —— 每个文件的 SHA-256 与行数，便于核对论文引用的版本
5. **环境清单** —— Python 与关键库版本

刻意**不归档**原始数据（OSM PBF 116 MB、CNBH 栅格、WorldPop）：它们体积大且
有稳定公开来源，``README`` 里给了下载入口。附录 A 应写明这一点，而不是
笼统地说"全部归档"。

用法::

    python scripts/16_export_archive.py
    python scripts/16_export_archive.py --out "论文稿件/数据"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import PROJECT_ROOT, load_config  # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("export_archive")

# 关键中间产物：论文正文/附录直接引用的那份
KEY_ARTIFACTS = [
    "candidates.csv", "demand_grid.csv", "table_clusters.csv",
    "table_solution_metrics.csv", "pareto_front.csv", "convergence.csv",
    "ip_scenarios.csv", "access_time_s.npz",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="归档可复现性材料")
    ap.add_argument("--out", default=None,
                    help="归档目录（默认 论文稿件/数据）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = (Path(args.out) if args.out
           else PROJECT_ROOT.parent / "论文稿件" / "数据")
    setup_logging(verbose=False,
                  log_file=Path(cfg.dir("output.data_dir")).parent / "logs" / "export_archive.log")

    data_dir = cfg.dir("output.data_dir")
    tab_dir = cfg.dir("output.tables_dir")
    out.mkdir(parents=True, exist_ok=True)
    log.info("归档目录: %s", out)

    manifest: dict = {"config_source": str(cfg.source_path),
                      "overrides": cfg.overrides, "files": {}}

    # -- 1. 配置快照 -------------------------------------------------------
    dst_cfg = out / "config_used.yaml"
    shutil.copy2(cfg.source_path, dst_cfg)
    with open(out / "config_overrides.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.overrides, fh, indent=2, ensure_ascii=False)
    log.info("  配置快照: %s（%d 项覆盖）", dst_cfg.name, len(cfg.overrides))

    # -- 2/3. 表格与关键中间产物 ------------------------------------------
    copied = 0
    for src_dir, sub in ((tab_dir, "tables"), (data_dir, "data")):
        if not src_dir.exists():
            log.warning("  目录不存在，跳过: %s", src_dir)
            continue
        target = out / sub
        target.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() in (".bak", ".tmp"):
                continue
            if sub == "data" and f.name not in KEY_ARTIFACTS:
                continue
            d = target / f.name
            shutil.copy2(f, d)
            copied += 1
            try:
                n_lines = (sum(1 for _ in open(d, "r", encoding="utf-8",
                                               errors="replace"))
                           if d.suffix.lower() in (".csv", ".json", ".txt", ".md")
                           else None)
            except Exception:
                n_lines = None
            manifest["files"][str(d.relative_to(out))] = {
                "bytes": d.stat().st_size,
                "sha256": sha256(d),
                "lines": n_lines,
            }
    log.info("  已归档 %d 个产物文件", copied)

    # -- 4. 运行清单 -------------------------------------------------------
    with open(out / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # -- 5. 环境清单 -------------------------------------------------------
    env_lines = [f"python: {sys.version.split()[0]}", f"platform: {sys.platform}"]
    for mod in ("numpy", "pandas", "geopandas", "shapely", "pulp", "highspy",
                "scikit-learn", "pymoo", "rasterio", "matplotlib", "pyosmium"):
        try:
            m = __import__(mod)
            env_lines.append(f"{mod}: {getattr(m, '__version__', 'unknown')}")
        except Exception:
            env_lines.append(f"{mod}: NOT INSTALLED")
    (out / "ENVIRONMENT.txt").write_text("\n".join(env_lines) + "\n",
                                         encoding="utf-8")
    log.info("  环境清单: ENVIRONMENT.txt")

    # -- 说明 --------------------------------------------------------------
    (out / "README.md").write_text(
        "# 归档说明\n\n"
        "本目录由 `scripts/16_export_archive.py` 生成，可随时重跑同步。\n\n"
        "| 子目录/文件 | 内容 |\n|---|---|\n"
        "| `config_used.yaml` | 本次运行的完整配置（唯一参数入口） |\n"
        "| `config_overrides.json` | 相对主配置的全部覆盖项 |\n"
        "| `tables/` | 正文与附录引用的全部表格（CSV） |\n"
        "| `data/` | 关键中间产物（候选集、需求栅格、IP 解、帕累托前沿） |\n"
        "| `MANIFEST.json` | 每个文件的 SHA-256、字节数与行数 |\n"
        "| `ENVIRONMENT.txt` | Python 与关键库版本 |\n\n"
        "## 未归档的内容\n\n"
        "原始数据（OSM PBF、CNBH-10 m 建筑高度栅格、WorldPop 人口、"
        "Copernicus DEM）体积大且来源公开稳定，不随稿提交；"
        "下载入口见仓库 `README.md`。\n\n"
        "## 复现\n\n"
        "```bash\n"
        "python scripts/run_all.py --data-mode real\n"
        "python scripts/28_run_baselines_full.py\n"
        "python scripts/08_multirun.py --runs 10\n"
        "python scripts/07_sensitivity.py --reps 10\n"
        "python scripts/13_fare_scenarios.py\n"
        "python scripts/14_network_basis.py --reps 3\n"
        "python scripts/15_encoding_ablation.py --runs 10\n"
        "python scripts/09_figures.py\n"
        "python scripts/16_export_archive.py\n"
        "```\n",
        encoding="utf-8",
    )
    log.info("  说明文件: README.md")
    log.info("")
    log.info("归档完成：%d 个文件 -> %s", len(manifest["files"]), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
