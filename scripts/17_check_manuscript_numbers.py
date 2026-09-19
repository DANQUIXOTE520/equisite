#!/usr/bin/env python
"""两稿数值一致性机器核对。

为什么需要它
------------
审计发现两稿有 **6 处同一个量给出了不同的数**——摘要类间比中文 1.46 / 英文 1.53、
0 km 覆盖率中文 100 % / 英文 99.66 %、§5.6 中文「1.8 倍」/ 英文「1.5 倍」等。
这类错误靠人眼比对两篇 2000 行的中英文稿是抓不全的，而且**每次改动数字都可能
新引入一处**。必须机器核对。

本脚本做两件事
--------------
1. **跨语言集合比对**：抽出两稿中所有"带单位或百分号的数值"，归一化后比较
   多重集合。只在一边出现的数，要么是该语言特有的表述，要么就是不一致——
   逐条列出供人工判定。
2. **过期值扫描**：给定一份"已作废的旧值"清单（如 A6 重跑前的候选数 1324），
   报告两稿中仍存在的每一处及其行号。这是回填后的验收检查。

刻意**不**做数值真值比对（把稿里的数与 CSV 里的数自动对上）——那需要理解
每个数字的语义，正则做不到，硬做只会产生大量误报并让人不再看输出。真值核对
仍靠 `_数字回填清单.md` 里的产物出处表逐项进行。

用法::

    python scripts/17_check_manuscript_numbers.py
    python scripts/17_check_manuscript_numbers.py --stale 1324,1430,3.13,4.59
    python scripts/17_check_manuscript_numbers.py --json      # 机器可读输出
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Windows 控制台默认 GBK，中文输出会抛 UnicodeEncodeError 并**静默丢行**。
# 本项目多次踩到这个坑（比价、敏感性日志），这里显式设 UTF-8。
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT.parent / "论文稿件"
ZH = MANU / "中文稿.md"
EN = MANU / "manuscript_draft.md"

# 要抽的"带单位数值"。刻意要求有单位或百分号——裸数字里混着年份、章节号、
# 参考文献编号、公式编号，全抽出来会淹没真正的信号。
NUM = r"[-+]?\d[\d\s,，]*(?:\.\d+)?"
UNIT_PATTERNS = [
    (r"%|％", "pct"),
    (r"亿元", "yi"), (r"百万元", "m"), (r"万元", "wan"), (r"元", "cny"),
    (r"分钟", "min"), (r"小时", "h"), (r"秒", "s"),
    (r"次/日", "trips"), (r"分钟·次/日", "mintrips"),
    (r"m²", "m2"), (r"km²", "km2"), (r"m\b", "m"), (r"km\b", "km"),
    (r"人", "ppl"), (r"个单元", "cells"), (r"个", "ge"),
    (r"栋", "bldg"), (r"站", "sites"), (r"倍", "x"), (r"×", "x"),
    (r"架次/小时", "thr"),
]


def extract(path: Path) -> Counter:
    """抽出 (归一化数值, 单位类别) 的多重集合。"""
    text = path.read_text(encoding="utf-8")
    out: Counter = Counter()
    for unit_re, tag in UNIT_PATTERNS:
        for m in re.finditer(NUM + r"\s*(?:" + unit_re + r")", text):
            raw = m.group(0)
            num = re.match(NUM, raw).group(0)
            num = num.replace(" ", "").replace(",", "").replace("，", "")
            try:
                v = float(num)
            except ValueError:
                continue
            # 保留 4 位有效数字。注意这**不会**把 96.19 与 96.2 合并
            # （前者归一成 "96.19"、后者 "96.2"），因此取整差异会以
            # "只在某稿出现"的形式暴露出来——这正是我们要抓的一类问题。
            key = f"{v:.4g}|{tag}"
            out[key] += 1
    return out


def stale_scan(path: Path, stale: list[str]) -> list[tuple[int, str, str]]:
    """在文中找出所有仍存在的过期值，返回 (行号, 旧值, 该行摘要)。"""
    hits: list[tuple[int, str, str]] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines, 1):
        for s in stale:
            if re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", line):
                hits.append((i, s, line.strip()[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="两稿数值一致性机器核对")
    ap.add_argument("--stale", default="1324,1 324,1430,1 430",
                    help="已作废的旧值，逗号分隔；默认是本轮 A6 重跑前的候选数")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out", default=None, help="JSON 输出路径")
    args = ap.parse_args()

    if not ZH.exists() or not EN.exists():
        raise SystemExit(f"找不到稿件: {ZH} / {EN}")

    zh, en = extract(ZH), extract(EN)
    only_zh = zh - en
    only_en = en - zh
    both = zh & en

    stale = [s for s in args.stale.split(",") if s.strip()]
    zh_stale = stale_scan(ZH, stale)
    en_stale = stale_scan(EN, stale)

    if args.json:
        payload = {
            "n_zh": sum(zh.values()), "n_en": sum(en.values()),
            "n_shared": sum(both.values()),
            "only_zh": dict(only_zh), "only_en": dict(only_en),
            "stale_zh": zh_stale, "stale_en": en_stale,
        }
        txt = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(txt, encoding="utf-8")
            print(f"已写出 {args.out}")
        else:
            print(txt)
        return 0

    print("=" * 72)
    print("两稿数值一致性核对")
    print("=" * 72)
    print(f"  中文稿带单位数值 {sum(zh.values())} 个（去重 {len(zh)} 种）")
    print(f"  英文稿带单位数值 {sum(en.values())} 个（去重 {len(en)} 种）")
    print(f"  两稿共有        {len(both)} 种")

    def _show(title: str, c: Counter, limit: int = 40) -> None:
        print("")
        print(f"── {title}（{len(c)} 种）──")
        if not c:
            print("   （无）")
            return
        for k, n in sorted(c.items(), key=lambda kv: -kv[1])[:limit]:
            v, tag = k.split("|")
            print(f"   {v:>12} {tag:<10} 多出 ×{n}")
        if len(c) > limit:
            print(f"   … 另有 {len(c) - limit} 种")

    # ⚠ 这里用 Counter 相减，得到的是**出现次数的差**，不是"只在一边存在"。
    #   两稿表述习惯不同（同一件事中文写一次、英文写两次），次数差是常态。
    #   所以标题必须写成"多出"，否则会让人以为某个值在另一稿里根本不存在。
    _show("中文稿出现次数多于英文稿", only_zh)
    _show("英文稿出现次数多于中文稿", only_en)

    print("")
    print("── 过期值扫描 ──")
    for label, hits in (("中文稿", zh_stale), ("英文稿", en_stale)):
        print(f"   {label}: {len(hits)} 处")
        for ln, s, txt in hits[:15]:
            print(f"      L{ln:<5} [{s}] {txt}")
        if len(hits) > 15:
            print(f"      … 另有 {len(hits) - 15} 处")

    print("")
    print("判读：")
    print("  * 两张表列的是**出现次数的差**（Counter 相减），不是'某值只在一稿存在'。")
    print("    两稿表述习惯不同（同一件事中文写一次、英文写两次；中文写'1 324 个'、")
    print("    英文写 '1,324'），次数差是常态，**大部分是噪声，不要逐条消灭**。")
    print("  * **真正的信号只有一类**：同一个量两稿写成了不同的值。它会在两张表里")
    print("    各出现一次，数值接近、单位相同（例如中文 '96.19 pct' 与英文 '96.2 pct'，")
    print("    或中文 '1.12 x' 与英文 '1.15 x'）。**只看这一类配对**，其余忽略。")
    print("  * 过期值扫描是回填后的验收项：应降到 0 处（除非该旧值在文中")
    print("    有别的含义，需逐处确认）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
