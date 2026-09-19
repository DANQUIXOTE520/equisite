"""跑完整基线组（论文的 p 集合 8/12/18/25），并算出每个解的覆盖率。

这是解锁 §5.7.1 / §5.8 的那一步：只要四个 p-中位点都解出且覆盖率合理，
就可以重导论文表格。
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from evtol_siting.baselines import run_all_baselines
from evtol_siting.config import load_config
from evtol_siting.metrics import solution_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)

cfg = load_config()
cand = pd.read_csv("outputs/data/candidates.csv")
dem = pd.read_csv("outputs/data/demand_grid.csv")
at = np.load("outputs/data/access_time_s.npz")["at"]

PS = (8, 12, 18, 25)
t0 = time.time()
print("=== run_all_baselines p=%s ===" % (PS,), flush=True)
sols = run_all_baselines(dem, cand, at, cfg, p_values=PS)
print("=== 全部完成，wall=%.1fs ===" % (time.time() - t0), flush=True)

rows = []
for s in sols:
    if s is None or not s.feasible or not s.selected:
        rows.append({"name": s.name if s else "None", "status":
                     (s.status if s else "None"), "n_sites": 0,
                     "obj": np.nan, "cov%": np.nan, "worst_min": np.nan,
                     "gap": np.nan, "proven": False,
                     "time_s": (s.solve_time_s if s else np.nan)})
        continue
    m = solution_metrics(s.selected, dem, cand, at, cfg, label=s.name)
    # gap / proven 必须逐解落盘：论文 §5.7.1 与 §4.6 的"求解至最优"这一说法
    # 只能按实际达到的相对间隙来写。此前这个字段根本不存在，正文照旧写
    # "四个 p 值均证明最优"——属于不可追溯的声称（审计 A2）。
    rows.append({
        "name": s.name, "status": s.status, "n_sites": s.n_sites,
        "obj": s.objective, "cov%": m.get("demand_coverage_pct"),
        "worst_min": m.get("worst_group_access_time_min"),
        "n_groups_unserved": m.get("n_groups_unserved"),
        "gap": getattr(s, "gap", None),
        "proven": bool(s.status == "Optimal"),
        "time_s": s.solve_time_s,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
print(df.to_string(index=False, float_format=lambda v: "%.6g" % v), flush=True)
df.to_csv("baselines_full.csv", index=False)

# 明确打印论文要用的那句话，避免手抄出错
_print = [(r["name"], r["status"], r["gap"]) for r in rows]
prov = [r["name"] for r in rows if r.get("proven")]
notprov = [(r["name"], r["gap"]) for r in rows if not r.get("proven") and r.get("gap") is not None]
print("", flush=True)
print("已证明最优（gap <= 1e-9）: %s" % (", ".join(prov) or "（无）"), flush=True)
print("未证明最优（论文须写「求解至相对间隙 X%%」而非「最优」）:", flush=True)
for n, g in notprov:
    print("   %-22s gap = %.6f%%" % (n, 100.0 * g), flush=True)
print("saved baselines_full.csv", flush=True)
