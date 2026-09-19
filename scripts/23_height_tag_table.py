#!/usr/bin/env python
"""重算表 2（研究区内建筑的高度标签覆盖情况），并落成**产物**。

为什么必须单独做这一步
--------------------
表 2 按"有 `height` / 有 `building:levels` / 两者皆无"三分类，而现有产物
`table_height_sources.csv` 给的是**融合级联之后的来源归属**——两者语义不同：
同时带 `height` 与 `building:levels` 标签的建筑，在表 2 里归入"有 height"，
在表 4 里只会被级联用掉一次。**不能用来源归属去凑表 2。**

此前表 2 一直是**手抄**的，于是 A6 修复改变建筑集（69 675 → 70 006）之后，
它与表 4 就对不上了：表 2 的合计是 70 094、表 4 是 69 675，而当前建筑集是
70 006。本脚本从 PBF 重新统计并落盘，使表 2 与表 4 同源。

输出
----
``outputs/data/table_height_tags.csv``：一行一类，列为
``tag_status,count,share_pct``，口径为**互斥**三分类（`height` 优先）。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from evtol_siting.config import load_config                      # noqa: E402
from evtol_siting.data import pbf                                # noqa: E402

# 与 `data/heights.py` 的检测列保持一致，否则两处的"有标签"口径会不同
HEIGHT_COLS = ("height",)
LEVEL_COLS = ("building:levels", "levels", "building:level")


def _present(gdf, cols) -> pd.Series:
    """任一列非空即视为"有该标签"。"""
    have = pd.Series(False, index=gdf.index)
    for c in cols:
        if c in gdf.columns:
            have |= gdf[c].notna() & (gdf[c].astype(str).str.strip() != "")
    return have


def main() -> int:
    cfg = load_config()
    path = cfg.get("data.pbf_file")
    path = (Path(path) if path and Path(path).is_absolute()
            else pbf.default_pbf_path(cfg))
    if not path.exists():
        raise SystemExit(f"找不到 PBF 提取包: {path}")

    print(f"解析 {path.name} …", flush=True)
    layers = pbf.parse_all_layers(path, cfg, layers=["buildings"])
    b = layers["buildings"]
    n = len(b)
    print(f"建筑总数 {n}")

    has_h = _present(b, HEIGHT_COLS)
    has_l = _present(b, LEVEL_COLS)
    n_h, n_l, n_both = int(has_h.sum()), int(has_l.sum()), int((has_h & has_l).sum())
    print(f"  带 height 标签      : {n_h}")
    print(f"  带 building:levels  : {n_l}")
    print(f"  两者都有            : {n_both}")

    # 互斥三分类：`height` 优先
    rows = [
        ("has_height", n_h),
        ("has_levels_only", n_l - n_both),
        ("neither", n - n_h - n_l + n_both),
    ]
    df = pd.DataFrame(rows, columns=["tag_status", "count"])
    df["share_pct"] = (100 * df["count"] / max(n, 1)).round(2)
    df["n_buildings_total"] = n

    out = Path(cfg.dir("output.data_dir")) / "table_height_tags.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    print(f"\n已写出 {out}")

    tot = int(df["count"].sum())
    print(f"\n三分类合计 {tot}（必须等于建筑总数 {n}）："
          f"{'✓' if tot == n else '✗ 不一致！'}")
    tag_only = 100 * (n_h + n_l - n_both) / max(n, 1)
    print(f"仅标签可得的比例 = ({n_h} + {n_l} - {n_both}) / {n} = {tag_only:.2f} %")
    print("（论文 §3.3 表 2 与 §3.4「8.5 % → 100 %」应引用此值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
