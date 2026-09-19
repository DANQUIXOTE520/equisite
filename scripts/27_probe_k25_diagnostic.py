"""复核 `schemes.py` 注释里那组"k=25 时 min_total_time 方案在它自己的目标上输了"的数。

该注释是"邻域宽度 k 必须显著大于 p-中位的 25"这一设计决定的**唯一证据**，
但其中的数字（6.20 亿 / 453 955，对照 3.48 亿 / 450 715）是旧算例的。
本脚本在当前算例上把 k=25 与 k=120 两臂各解一次，给出可引用的对照。

⚠ 只跑固定 N 的 `min_total_time` 一个目标，两次 CBC，秒级。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import load_config              # noqa: E402
from evtol_siting.logging_setup import setup_logging     # noqa: E402
from evtol_siting.pipeline import Pipeline               # noqa: E402
from evtol_siting.schemes import solve_fixed_n           # noqa: E402

setup_logging(False, None)
base = load_config()
n = int(base.get("stage3_ip.fixed_n"))
print(f"固定 N = {n}\n")

cache: dict = {}
rows = []
for k in (25, 120):
    cfg = base.override({"stage3_ip.fixed_n_k_nearest": k})
    pipe = Pipeline(cfg, data_mode="real", cache=cache)
    pipe.run_stage1(); pipe.run_stage2(); pipe.run_stage3()
    s = solve_fixed_n(pipe.res.demand.grid, pipe.res.candidates.gdf,
                      pipe.res.access_time_s, n, "min_total_time", cfg,
                      name=f"min_total_time_k{k}")
    rows.append((k, s))
    mt = s.metrics
    print(f"  k={k:3d}  {s.status:9s}  成本 {mt['total_cost_cny']/1e8:.2f} 亿元  "
          f"总接驳 {mt['total_access_time']:,.0f}  "
          f"最差人群 {mt['worst_group_access_time_min']:.2f}  min  "
          f"覆盖 {mt['demand_coverage_pct']:.1f} %".replace(",", " "))

# 顺带取当前固定的 N 下的最低成本方案作为对照基线
cfg = base
pipe = Pipeline(cfg, data_mode="real", cache=cache)
pipe.run_stage1(); pipe.run_stage2(); pipe.run_stage3()
mc = solve_fixed_n(pipe.res.demand.grid, pipe.res.candidates.gdf,
                   pipe.res.access_time_s, n, "min_cost", cfg, name="min_cost")
print(f"\n  对照·成本最低  k=120 成本 {mc.total_cost/1e8:.2f} 亿元  "
      f"总接驳 {mc.total_access_time:,.0f}".replace(",", " "))
