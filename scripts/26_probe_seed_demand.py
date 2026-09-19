"""探测：stage1/stage2 的产物是否随 seed 变化。

`07_sensitivity.py` 的 ``det`` 集合把 total_population/total_demand 当作确定性量，
不给标准差。若实测随种子变化，这个判定就是错的。
只跑 stage1+stage2（不跑 NSGA-II），并且三次复用同一份 pipe_cache。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evtol_siting.config import load_config          # noqa: E402
from evtol_siting.logging_setup import setup_logging  # noqa: E402
from evtol_siting.pipeline import Pipeline            # noqa: E402

setup_logging(False, None)
base = load_config()
cache: dict = {}
rows = []
for seed in (42, 43, 44):
    cfg = base.override({"project.random_seed": seed})
    cfg.seed_everything()
    pipe = Pipeline(cfg, data_mode="real", cache=cache)
    cs = pipe.run_stage1()
    dm = pipe.run_stage2()
    rows.append(dict(
        seed=seed,
        n_candidates=len(cs),
        n_rooftop=int((cs.gdf["facility_type"] == "rooftop").sum()),
        n_clusters=dm.n_clusters,
        total_population=float(dm.grid["population"].sum()),
        total_demand=float(dm.grid["demand"].sum()),
    ))
    print(rows[-1], flush=True)

import pandas as pd
d = pd.DataFrame(rows)
print()
for c in ("n_candidates", "n_rooftop", "n_clusters",
          "total_population", "total_demand"):
    v = d[c].to_numpy(dtype=float)
    nuniq = len(set(v.tolist()))
    print(f"  {c:18s} 取值数={nuniq}  min={v.min():.6f}  max={v.max():.6f}  "
          f"极差/均值={((v.max()-v.min())/v.mean() if v.mean() else 0):.3e}")
