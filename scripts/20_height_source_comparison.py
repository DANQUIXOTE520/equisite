#!/usr/bin/env python
"""计算 §3.3 的核心对照：**仅用 OSM 标签** vs **多源融合** 两种高度来源下的屋顶候选数。

为什么必须把它变成脚本
----------------------
论文 §3.3 的核心论点是"OSM 高度标签严重偏向高耸、测绘完善的建筑，只用标签会丢掉
大量屋顶型候选"，支撑它的那两个数（**227** 与 **1 430**）出现在摘要、§3.3 正文、
表 3、图 2 的图注与表 11 头条结果里——而审计发现它们在**全部日志与 outputs/ 中
都没有来源**。

追查后的真相更糟：这两个数是**硬编码的字面量**，写在

* ``scripts/09_figures.py:160``  —— ``tag_only={"tag_only": 227, "fused": 1430}``
* ``src/evtol_siting/viz/figures.py:195`` —— 作为 ``.get()`` 的**默认值**

也就是说，论文用来说服审稿人的那张图（图 2b），画的是一个**没有被任何计算产生过**
的数；重跑流水线也不会更新它。本脚本把这两个数真正算出来并落盘。

口径
----
* **融合**：全部建筑，用融合后的 ``height_m`` 与 ``roof_usable_m2`` 施加
  ``h >= min_height_m`` 且 ``A_roof >= min_roof_area_m2``（与阶段一完全一致）。
* **仅标签**：只保留 ``height_source ∈ {osm_height, osm_levels}`` 的建筑，
  再施加同一组判据。栅格与统计填补得到的高度**一律不计**——这正是"只用标签"的含义。

用法::

    python scripts/20_height_source_comparison.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evtol_siting.config import load_config                      # noqa: E402

TAG_SOURCES = ("osm_height", "osm_levels")


def main() -> int:
    ap = argparse.ArgumentParser(description="§3.3 高度来源对照")
    ap.add_argument("--data-mode", default="real",
                    choices=["auto", "real", "synthetic"])
    args = ap.parse_args()

    cfg = load_config()
    from evtol_siting.pipeline import Pipeline
    from evtol_siting.data.heights import estimate_roof_area

    pipe = Pipeline(cfg, data_mode=args.data_mode, cache={})
    pipe.run_stage1()                     # 载入数据 + 高度融合 + 屋顶面积
    b = pipe.res.buildings
    if b is None or len(b) == 0:
        raise SystemExit("建筑数据为空，无法计算")

    rc = cfg.get("stage1_candidates.rooftop", {})
    min_h = float(rc.get("min_height_m", 20.0))
    min_a = float(rc.get("min_roof_area_m2", 1600.0))
    ratio = float(rc.get("usable_roof_ratio", 0.70))

    if "roof_usable_m2" not in b.columns:
        b = estimate_roof_area(b, ratio)

    h = b["height_m"].to_numpy(dtype=float)
    a = b["roof_usable_m2"].to_numpy(dtype=float)
    ok = np.isfinite(h) & np.isfinite(a) & (h >= min_h) & (a >= min_a)
    src = b["height_source"].astype(str).to_numpy()
    tagged = np.isin(src, TAG_SOURCES)

    n_fused = int(ok.sum())
    n_tag = int((ok & tagged).sum())
    n_total = len(b)

    print("=" * 70)
    print("§3.3 高度来源对照（判据 h >= %.0f m 且 A_roof >= %.0f m²）" % (min_h, min_a))
    print("=" * 70)
    print(f"  建筑总数            {n_total:,}")
    print(f"  有 OSM 高度标签     {int(tagged.sum()):,}"
          f"（{100*tagged.sum()/n_total:.1f}%）")
    print("")
    print(f"  **仅用标签**的屋顶候选   {n_tag:,}")
    print(f"  **多源融合**的屋顶候选   {n_fused:,}")
    if n_tag:
        print(f"  倍数                    {n_fused / n_tag:.2f}×")
    print("")
    print("  逐来源（该来源单独能否满足判据）：")
    for s in sorted(set(src)):
        m = src == s
        if m.sum():
            print(f"    {s:<16} 建筑 {int(m.sum()):7,}  其中满足判据 {int((ok & m).sum()):6,}")

    payload = {
        "min_height_m": min_h, "min_roof_area_m2": min_a,
        "n_buildings": n_total,
        "n_tagged": int(tagged.sum()),
        "tagged_pct": float(100 * tagged.sum() / n_total),
        "tag_only_rooftop": n_tag,
        "fused_rooftop": n_fused,
        "ratio": (float(n_fused / n_tag) if n_tag else None),
        "tag_sources": list(TAG_SOURCES),
    }
    out = Path(cfg.dir("output.tables_dir")) / "height_source_comparison.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("")
    print(f"已写出 {out}")
    print("⚠ 图 2 与正文 §3.3 必须引用本文件的值，不得再使用脚本里的硬编码字面量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
