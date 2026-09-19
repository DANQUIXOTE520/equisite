#!/usr/bin/env python
"""敏感性表的**自洽性**检验：各参数取默认值的那一行必须逐位相同。

原理
----
单因素扫描对每个参数扫若干取值，其中**必有一个取值等于该参数的默认值**。
在这一行上，覆盖后的整份配置就等于全局基准配置；又因为各参数组用的是
**同一组种子**（``base_seed + rep``），所以这些行的每一个指标都应当
**逐位相同**。

若它们不同，只有两种可能：

1. **协议不一致**——例如部分参数跑了 10 次重复、其余只跑了 1 次。
   实测过：第四批用 ``--reps-only`` 只给 5 个参数 10 次，于是
   ``n_reps`` 为 10 的 29 行与 NaN 的 47 行混在一张表里，基准行分成
   ``(3.204773, 6.004299)`` 与 ``(0.0, 6.298707)`` 两拨，
   且分歧与 ``seed`` 列（46.5 = mean(42..51) vs 42.0 = 单次）**完全一致**。
2. **上游有非确定性**——那本身就是必须查清的问题。

这个检验不需要任何额外信息，只用产物自身即可证伪"协议不统一"。

哪些参数**不**参与断言
----------------------
``SWEEPS`` 的 ``BASELINES[...]["anchor"]`` 标记了该项的基准行是否与全局基准
配置重合。``n_clusters``（扫描强制 ``k_selection.method=fixed``，基准是轮廓
系数自动选择）与 ``trip_rate_ratio``（扫描强制 ``choice_model.enabled=False``
走外生阶梯，基准走 Logit）**按构造**就不重合，断言里必须排除，否则误报。
它们仍会被打印出来，只是不计入结论。

用法::

    python scripts/22_check_sensitivity_consistency.py
    python scripts/22_check_sensitivity_consistency.py --csv <路径>
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from evtol_siting.config import load_config                      # noqa: E402

# 参与比对的数值列（确定性量与随机量都算上）
COMPARE = ["n_candidates", "n_rooftop", "total_demand",
           "pareto_unserved_min", "pareto_inequity_min",
           "ip_n_sites", "ip_cost"]


def _load_sweeps():
    """按路径加载 ``07_sensitivity.py``，取出 SWEEPS 与 BASELINES。

    不复制粘贴那两张表——基准值必须**只有一份定义**，否则改了一处、漏了
    另一处，而漏掉的那处正是校验用的锚点。
    """
    p = Path(__file__).resolve().parent / "07_sensitivity.py"
    spec = importlib.util.spec_from_file_location("_sens07", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SWEEPS, mod.BASELINES


def _norm(v) -> str:
    """把取值规整成可比较的字符串（1315.0 与 1315 视为同一个）。"""
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except (TypeError, ValueError):
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description="敏感性表自洽性检验")
    ap.add_argument("--csv", default=None,
                    help="默认取配置的 output.tables_dir/sensitivity.csv")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="数值容差。默认 0 = 要求逐位相同")
    args = ap.parse_args()

    cfg = load_config()
    path = Path(args.csv) if args.csv else (
        Path(cfg.dir("output.tables_dir")) / "sensitivity.csv")
    d = pd.read_csv(path)
    sweeps, baselines = _load_sweeps()

    print(f"读取 {path}")
    print(f"  {len(d)} 行，{d['parameter'].nunique()} 个参数")
    if "n_reps" in d.columns:
        print(f"  n_reps 分布: {d['n_reps'].value_counts(dropna=False).to_dict()}")

    # ---- 重复键守卫 ----------------------------------------------------
    #   同一个 (参数, 取值) 出现多行时，下面 `hit.iloc[0]` 取到哪一行**是不确定的**
    #   （取决于行序）。第五批实测就取到了旧行：表里 76 行是新的 10 次结果、
    #   47 行是上一批留下的单次结果，两者基准行数值不同，于是本检验报
    #   "协议未统一"——而按 reps 最高的行去比，15 个基准行其实**逐位相同**。
    #   差一点就据此把一张已经正确的表删掉重跑四小时。
    #   所以这里不再"尽力而为"：表本身坏了就拒绝给结论。
    dup = d.duplicated(subset=["parameter", "value"], keep=False)
    if dup.any():
        print(f"\n✗ **表里有重复键**：{int(dup.sum())} 行属于 "
              f"{d[dup].groupby(['parameter', 'value']).ngroups} 个重复的 "
              f"(参数, 取值) 组合。")
        for (p, v), g in d[dup].groupby(["parameter", "value"]):
            reps = sorted({str(x) for x in g["n_reps"]})
            print(f"    · {p} @ {v}：{len(g)} 行，n_reps={reps}")
        print("\n  成因：`07_sensitivity.py` 续传时只 `concat([done, rows])`、"
              "\n  **不删同键旧行**，于是重算过的点会留下两份（新旧各一）。"
              "\n  修法：重跑 `07_sensitivity.py`（新版会在载入时按 reps 去重、"
              "重算时替换旧行）。"
              "\n  **本检验不对这种表给结论**——取到哪一行不确定，"
              "结论正负取决于行序。")
        return 1

    # ---- 定位各参数的基准行 ----
    picks, skipped = [], []
    for p, g in d.groupby("parameter"):
        bl = baselines.get(p)
        sw = sweeps.get(p, {})
        if bl is None:
            skipped.append((p, "BASELINES 里没有定义基准值"))
            continue
        want = bl["value"]
        # 相对取值（如收入中位数按倍数扫）要还原成绝对取值
        rel = sw.get("relative_to")
        if rel:
            cur = cfg.get(rel)
            if cur is None:
                skipped.append((p, f"配置缺 {rel}，无法还原绝对基准"))
                continue
            want = float(cur) * float(want)
        hit = g[g["value"].map(_norm) == _norm(want)]
        if not len(hit):
            skipped.append((p, f"表里找不到基准取值 {want}（该组取值："
                               f"{list(g['value']) }）"))
            continue
        r = hit.iloc[0]
        picks.append({"param": p, "value": r["value"],
                      "anchor": bool(bl.get("anchor", True)),
                      **{c: r.get(c) for c in COMPARE if c in d.columns}})

    b = pd.DataFrame(picks)
    print(f"\n=== 各参数的基准行（{len(b)} 行）===")
    show = b.copy()
    show["anchor"] = show["anchor"].map({True: "是", False: "否（按构造不重合）"})
    print(show.to_string(index=False))

    off = b[~b["anchor"]]
    if len(off):
        print("\n以下参数的基准行**不参与**断言（配置与全局基准按构造不同）：")
        for _, r in off.iterrows():
            print(f"  · {r['param']}：{baselines[r['param']].get('why', '')}")

    if skipped:
        print("\n⚠ 未能定位基准行的参数：")
        for p, why in skipped:
            print(f"  · {p}：{why}")

    on = b[b["anchor"]]
    cols = [c for c in COMPARE if c in on.columns]
    print(f"\n=== 断言：{len(on)} 个 anchor 参数，比对列 {cols}，容差 {args.tol} ===")

    bad: list[tuple[str, str]] = []
    for c in cols:
        try:
            nums = [float(v) for v in on[c].tolist()]
        except (TypeError, ValueError):
            # 列表值（`ip_n_sites`、`ip_cost` 落盘成 "[11, 10]" 这样的字符串）。
            # ⚠ 此前这里直接 `continue`，于是 7 个比对列里有 2 个**从未被检验**，
            #   却仍打印"逐位一致"——一个把沉默当通过的缺口。改为逐元素比。
            try:
                seqs = [ast.literal_eval(str(v)) for v in on[c].tolist()]
            except (ValueError, SyntaxError):
                print(f"  – {c:22s} 既非数值也非列表，**未检验**")
                continue
            if all(s == seqs[0] for s in seqs):
                print(f"  ✓ {c:22s} 全部一致 = {seqs[0]}")
            else:
                bad.append((c, f"取值不一：{seqs}"))
            continue
        spread = max(nums) - min(nums)
        if spread > args.tol:
            bad.append((c, f"跨度 {spread:g}：{sorted(set(nums))}"))
        else:
            print(f"  ✓ {c:22s} 全部一致 = {nums[0]:g}")

    print()
    if bad:
        print("✗ **基准行不一致**——协议未统一或上游存在非确定性：")
        for c, msg in bad:
            print(f"    {c}: {msg}")
        print(f"\n  以上 {len(on)} 行的整份配置完全相同，结果必须逐位相同。"
              "\n  若因 reps 不同所致，用 `--reps 10`（**不加** `--reps-only`）"
              "整表重跑；\n  切勿拼接两次运行的结果——那正是本检验要防的形态。")
        return 1
    if len(on) < 2:
        print("⚠ anchor 行不足 2 个，断言无意义。")
        return 2
    print(f"✓ {len(on)} 个 anchor 参数的基准行逐位一致——协议已统一。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
