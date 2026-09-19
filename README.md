# EQUISITE — equity-aware vertiport siting

Code, configuration and output artefacts for a study on where to put urban air mobility
vertiports in Chengdu, China. EQUISITE stands for **EQUI**ty-aware **SITE** selection.
The name is *not* an initialism of the four stages.

The problem this code addresses is narrow but awkward. Vertiport siting studies optimise
coverage or cost, and treat equity as a post-hoc check on whatever the optimiser produced.
Here the access time of the worst-off income group is one of the four objectives the search
is actually solving for. Separately, rooftop and ground vertiports have opposite cost
structures — rooftops avoid land acquisition but carry high fixed costs, ground parcels the
reverse — and are modelled as two distinct facility classes inside one optimisation.

Companion paper: *Equity-aware multi-objective siting of urban air mobility vertiports*
(under review).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22848615.svg)](https://doi.org/10.5281/zenodo.22848615)

A Chinese version of this file is at [`README.zh-CN.md`](README.zh-CN.md).

---

## 1. Quick start

```bash
pip install -r requirements.txt

# Full pipeline on real data. First run downloads ~13 GB of third-party inputs.
python scripts/run_all.py

# Flow check on synthetic data — a few minutes, no downloads.
# Its outputs are fabricated and must never reach a paper. The logs say so loudly.
python scripts/run_all.py --data-mode synthetic --pop-size 40 --n-gen 20

# Fetch inputs only, then stop. Useful for checking data availability.
python scripts/run_all.py --fetch-only

# Stop after stage 2.
python scripts/run_all.py --until stage2

# Override any parameter by dotted path — no code edits.
python scripts/run_all.py --set stage1_candidates.rooftop.min_roof_area_m2=1400
python scripts/run_all.py --set stage3_ip.min_core_demand=0.5 --set stage2_demand.n_clusters=4

# Sensitivity scan, ten independent runs, fare scenarios.
python scripts/07_sensitivity.py --data-mode real --reps 10
python scripts/08_multirun.py   --data-mode real --runs 10
python scripts/13_fare_scenarios.py

# Cross-check the hand-written NSGA-II against pymoo.
python scripts/run_all.py --validate-nsga2
```

Everything writes to `outputs/`. Every number in the paper traces to a CSV or a log line
under that directory.

---

## 2. Method

```
┌──────────────────────────┐   ┌──────────────────────────┐
│ Stage 1  GIS screening    │   │ Stage 2  Demand strata    │
│ stage1_candidates.py      │   │ stage2_demand.py          │
│                           │   │                           │
│ · 500 m grid              │   │ · features: income proxy  │
│ · rooftop: h≥20 m,        │   │   + POI density           │
│   usable roof ≥1600 m²    │──▶│ · K by silhouette         │
│ · ground: slope ≤15°,     │   │ · stratified trip rates   │
│   parcel ≥1600 m²,        │   │ · demand = pop × rate     │
│   building coverage ≤0.55 │   │                           │
│ · airport clearance zones │   │                           │
│ · density dilution,       │   │                           │
│   per facility class      │   │                           │
└──────────────────────────┘   └──────────────────────────┘
              │                              │
              ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ Stage 3  0–1 integer prog │   │ Stage 4  NSGA-II          │
│ stage3_ip.py              │   │ stage4_nsga2.py           │
│                           │   │ nsga2_core.py             │
│ min cost / site count     │──▶│ 4 objectives:             │
│ s.t. core demand covered  │   │  ① unserved demand        │
│      access-time budget   │   │  ② total access time      │
│      site cap, capacity   │   │  ③ worst-group access     │
│ exact (HiGHS) → 2 plans   │   │  ④ capital cost           │
└──────────────────────────┘   │ → Pareto front + knee     │
              │                └──────────────────────────┘
              └──────────┬───────────────────┘
                         ▼
              ┌────────────────────────┐
              │ Evaluation              │
              │ baselines.py, metrics.py│
              │ p-median / p-center /   │
              │ MCLP / set covering     │
              │ HV, SP, Δ, IGD          │
              └────────────────────────┘
```

### 2.1 Two things this study claims as new

**(A) Income-stratified equity as an explicit objective.** Reviewing the siting corpus, we
found no study that puts income-stratified accessibility into the objective function. The
closest work (Lu et al. 2025, Shenzhen) handles equity during screening while the
optimisation stays single-objective; Fadhil (2018) uses high income as a *positive* siting
criterion, which sharpens the inequality rather than reducing it. Our equity objective is
the mean access time of the worst-off group, which is deliberately not the single-user
minimax of a classical p-center. The paper tests that distinction empirically rather than
asserting it.

**(B) Rooftop and ground as different facility classes.** Stratified facility hierarchies
are not new (NASA's Vertihub / Vertiport / Vertistop; Guo et al.'s Vertipad / Vertibase /
Vertihub), but nobody has split them along the rooftop/ground axis, where the cost
structures are inverted.

### 2.2 Three departures from the original project proposal

| # | Change | Why |
|---|--------|-----|
| 1 | **Multi-source building-height fusion** (`data/heights.py`) | OSM height tags cover only 1.5–5.4 % of central Chengdu (8.5–19.7 % for `building:levels`). Using tags alone discards more than 80 % of tall buildings and badly under-counts candidates. We fuse four sources: OSM `height` → OSM `building:levels` × storey height → CNBH-10 m raster → quantile regression on footprint area. Each building records its `height_source`, so the paper can report each source's contribution and uncertainty. |
| 2 | **Building-coverage criterion for ground sites** | The proposed ground criterion (exclude mountain/water/ecological red lines, slope ≤ 15°) excludes almost nothing on the Chengdu plain — it yields roughly 2 300 candidates, i.e. nearly every 500 m cell. That is not credible: dense old-city districts have no 1 600 m² of open land. We add "building footprint coverage ≤ β" and run a sensitivity analysis on β. |
| 3 | **Density dilution, per class** | Rooftop and ground sites are different kinds of facility. Diluting them together on a single quality score lets ground sites win systematically — in testing it eliminated 100 % of rooftop candidates, which kills the dual-class premise before the optimiser even runs. We changed to quota-within-class → minimum spacing within class → cross-class conflicts resolved by cost per unit capacity. |

---

## 3. Data

| Variable | Source | Resolution | Status |
|----------|--------|-----------|--------|
| Building footprints / POI / roads / land use | **OSM Sichuan extract (PBF)** | — | works |
| ↑ alternative | OSM Overpass API (tiled) | — | rate-limited; see below |
| Building height | **CNBH-10 m** (Zenodo `10.5281/zenodo.7923866`) | 10 m | works |
| Population | WorldPop China 2020 constrained (primary) | 100 m | server flaky; see below |
| Population (fallback) | building-volume estimate | building | circularity risk; see below |
| Elevation / slope | Copernicus DEM GLO-30 | 30 m | works |
| **Income proxy** | **NPP-VIIRS 2020 night-time lights, `average_masked`** | 500 m | see below |

The active income source is set by `data.sources.income` in `config/chengdu.yaml`, which
ships as `viirs`. `poi_index`, `local_file` and `synthetic` are also implemented.

### 3.1 Why population is the awkward variable

WorldPop's server is unreliable. A 920 MB national file died at 55 % with
`ChunkedEncodingError: IncompleteRead`; the server ignores HTTP `Range` headers, so there is
no resume. We retry and wait rather than degrade automatically.

The fallback path `population_from_buildings()` exists but is not the default, for a
circularity reason. Rooftop candidates come from the same OSM building layer, and their
heights come from the same fusion model. If population is then inferred from building
volume, demand and supply become two functions of one dataset — and the accessibility
inequality the paper measures could be an artefact of that shared source rather than a
property of the city. An independent population surface is worth waiting for.

If you do switch, report the calibration (default 40 m²/person), the implied study-area
total, and how it compares with the census. Note that the calibration only rescales the
*total*; the spatial distribution — which is what the equity result depends on — is
unaffected.

### 3.2 Why PBF and not Overpass

Both backends are implemented, switched by `data.sources.osm_backend`.

| Backend | Source | Size | Measured |
|---------|--------|------|----------|
| Overpass (`data/osm.py`) | overpass-api.de, 300 tiles | ~500 MB | **8 h+** |
| **PBF (`data/pbf.py`)** | OSM.fr Sichuan extract | **116 MB** | **18 s download + 41 s parse** |

Under restricted networks Overpass rate-limits hard by IP: the first request returns 200 and
everything after it returns 504 with a 695-byte body, which is a throttle signature rather
than a query problem. Common mirrors (kumi.systems, private.coffee) are unreachable from
mainland China, so each retry burns a full timeout. The PBF route also has a methodological
advantage: it is a complete snapshot at one instant, so no features are truncated at tile
boundaries and the paper can state a single data vintage.

> Windows note: libosmium cannot open absolute paths containing non-ASCII characters, and
> this project directory has a Chinese name. `pbf.libosmium_path()` falls back to a relative
> path. Without it you get `RuntimeError: Open failed ... unknown error`, which tells you
> nothing about the real cause.

### 3.3 The income variable, and one correction worth carrying forward

**Meta's Relative Wealth Index does not cover China.** It is built for 93 low- and
middle-income countries and China is excluded by design (checked against the HDX API in
2026-09: no China among 112 resources). Any write-up that uses RWI for Chengdu income is
wrong.

We use **NPP-VIIRS night-time lights** as the income proxy, chosen over the OSM POI
economic index that was used earlier. This changed more than data quality:

| | POI economic index (superseded) | **VIIRS night lights (current)** |
|---|---|---|
| Clusters K | 7 | **5** |
| Cells in the top income class | 13 (0.5 %) | **300 (11.9 %)** |
| Total demand | 48 720 trips/day | **50 337 trips/day** |

Night lights measure built-up brightness, whose peak is far flatter than the POI index,
which lights up essentially only the CBD. The consequence is that an earlier headline
finding — "UAM-servable demand is a small, highly concentrated core" — is substantially
weakened. Section 3.6 of the paper reports that trade-off rather than hiding it. POI-era
results are kept under `outputs/data_poi_index/` for comparison.

**This is a proxy for economic activity, not income.** The two correlate but are not the
same thing: one measures commercial intensity and retail mix, the other what people earn.
Section 3.6 of the paper states the proxy relationship and its limits, and the sensitivity
analysis tests how far the conclusions depend on it.

---

## 4. Repository layout

```
evtol_siting/
├── config/chengdu.yaml          # the only parameter entry point; experiments override it
├── src/evtol_siting/
│   ├── config.py                # dotted-path overrides, warnings on unknown paths
│   ├── geo.py                   # rasterisation, distance/access-time matrices, sampling
│   ├── logging_setup.py         # UTF-8-safe logging (GBK consoles on Chinese Windows)
│   ├── pipeline.py              # four-stage assembly and result persistence
│   ├── data/
│   │   ├── pbf.py               # OSM province PBF parsing (recommended route)
│   │   ├── osm.py               # Overpass tiled download (alternative route)
│   │   ├── heights.py           # four-source height fusion, usable-roof estimation
│   │   ├── dasymetric.py        # volume-weighted population redistribution
│   │   ├── network.py           # OSM road-network shortest-path access times
│   │   ├── rasters.py           # population / income / DEM / CNBH rasters
│   │   └── synthetic.py         # synthetic data — offline tests only, never for a paper
│   ├── stage1_candidates.py     # stage 1
│   ├── stage2_demand.py         # stage 2
│   ├── choice_model.py          # demand generation: binary logit mode choice
│   ├── stage3_ip.py             # stage 3
│   ├── schemes.py               # fixed-cardinality scheme family + cardinality-preserving crossover
│   ├── stage4_nsga2.py          # stage 4: problem definition and the four objectives
│   ├── nsga2_core.py            # NSGA-II itself, written from scratch
│   ├── baselines.py             # p-median / p-center / MCLP / set covering
│   ├── metrics.py               # HV / SP / Δ / IGD plus equity measures
│   ├── solver_status.py         # one shared MIP-gap-based optimality test
│   ├── integer_encoding.py      # integer-encoding equivalence
│   ├── viz3d.py                 # 3D visualisation export
│   └── viz/figures.py           # publication figures
├── scripts/                     # pipeline entry points and verification tools
├── viz3d/                       # CesiumJS 3D globe (see §9)
├── outputs/{figures,tables,data,logs}/
└── paper/framework.md           # paper skeleton and reviewer-risk register
```

Script numbering starts at `07`. Numbers `01`–`06` were early single-stage scripts, since
replaced by `run_all.py` and the four `stage*.py` modules; the numbering was left alone
rather than renumbered, because existing logs and documents cite the scripts by name.

### 4.1 The scripts you will actually want

| Script | Purpose |
|--------|---------|
| `run_all.py` | the four-stage pipeline |
| `07_sensitivity.py` | one-factor-at-a-time scans, 12 parameters, 63 values |
| `08_multirun.py` | ten independent runs, statistics, pymoo cross-check |
| `09_figures.py` | paper figures, from saved result CSVs (seconds, no re-run) |
| `12_paper_tables.py` | the §5.8 scheme table and the Appendix D sensitivity table |
| `16_export_archive.py` | reproducibility archive with SHA-256 and row counts per file |
| `17_check_manuscript_numbers.py` | scans the manuscripts for superseded values |
| `21_regen_manuscript_tables.py` | regenerates manuscript tables from outputs; `--dry-run` prints a row-by-row diff |
| `25_probe_dense_decoder_defect.py` | reproduces the decoder defect in ~6 s (see §4.5) |
| `28_run_baselines_full.py` | the full baseline set (p = 8/12/18/25) |

---

## 4.5 Three optimiser design traps

These were not coding mistakes. Each is a design-level error that raises no exception and
simply lets the search converge somewhere wrong. All three were found by comparing against
an independent implementation (pymoo), which is why that cross-check is in the repo.

### 4.5.1 The decoder and the search operators must be separate

**A decoder's job is to restore feasibility, not to improve solution quality.**

* **Mistake one — repair by adding only.** If the repair operator only adds sites and never
  removes them, a dense initial population can never shed sites. Measured: hypervolume never
  improved once in 100 generations, site counts stuck at 613–706, front cost
  **CNY 32.9–37.5 billion** — against an integer program that reaches 96.13 % coverage for
  CNY 314 million. Reproduce with `scripts/25_probe_dense_decoder_defect.py`.
* **Mistake two — pruning inside the decoder.** Fixing mistake one by pruning redundant
  sites during decoding means every dense solution is collapsed back to a minimal cover at
  evaluation time, so the high-site-count region becomes unreachable. Against pymoo,
  hypervolume was **18 %** of the reference.

The working design: the decoder only adds; pruning becomes a local-search operator applied
probabilistically to a subset of offspring. The same implementation then reaches **153 %** of
pymoo's hypervolume.

### 4.5.2 The initial population has to span the plausible site-count range

Because repair only adds, the initial population's size spectrum bounds what the search can
ever reach. Starting every individual from the same cardinality confines the search
permanently. We draw distinct site counts log-uniformly from `[12, 400]`
(`stage4_nsga2.init_site_range`).

### 4.5.3 In fixed-cardinality mode, every operator must preserve cardinality

When N is fixed, "every plan has exactly N sites" is a hard premise, and anything that
changes the count breaks it:

* generic repair (adds on under-coverage) → replaced by `make_fixed_cardinality_repair`, which
  swaps rather than adds
* pruning local search (removes only) → must be disabled; `nsga2_core` has a guard so a
  caller who forgets cannot corrupt the run
* SBX crossover (offspring cardinality drifts) → replaced by `crossbreed_masks`, drawing N
  sites from the parents' union

`scripts/10_scheme_family.py` verifies every solution's site count against N and errors out
on a mismatch instead of passing silently.

---

## 5. Implementation notes

### 5.1 Why NSGA-II is hand-written

The project proposal listed implementing NSGA-II as a deliverable, so it is implemented from
scratch in `nsga2_core.py` rather than imported from pymoo. That also allows the
structure-specific repair this problem needs. pymoo is kept as an independent cross-check
(`--validate-nsga2`), and the paper reports the front-quality ratio between the two
implementations.

### 5.2 Encoding equivalence

The proposal describes "integer encoding with vertiport indices as genes". The
implementation uses an equivalent 0/1 mask over candidates (dimension = candidate count,
`x_j = 1` means build at candidate `j`). Mask positions correspond one-to-one with the
integer encoding's chosen indices, and the mask works directly with standard operators
(SBX, polynomial mutation) without the duplicate-gene and infeasible-solution problems of
variable-length integer encodings.

### 5.3 What the capacity constraint does and does not do

`stage3_ip`'s capacity constraint is `ℓ_j · y_j ≤ q_j`, where `ℓ_j` is a **precomputed
constant** — a load upper bound estimated by nearest-assignment — not a decision variable.
It is therefore a *siting feasibility filter* that excludes candidates whose own load would
exceed capacity. It does **not** let demand be reallocated between sites. Any claim about
load balancing requires extending the model with explicit `x_ij` assignment variables.

### 5.4 NSGA-II has no site-count cap

The IP scenarios include `Σy_j ≤ N_max`; none of the four NSGA-II objectives carries a site
cap (cost limits the count implicitly). Section 5.5 of the paper compares the two, and states
that this is a design choice rather than an oversight.

### 5.5 `min_core_demand` has two readings

Values in `0–1` are read as a **quantile** (recommended; `0.6` means the top 40 % of demand
cells must be covered); values `≥ 1` are read as an **absolute count** (trips/day). The
quantile form is recommended because an absolute threshold couples tightly to the trip-rate
calibration — change city or change trip rates and the core set silently becomes empty (the
coverage constraint quietly stops binding) or becomes everything (losing all
prioritisation). The code logs an explicit warning when the core set is empty.

---

## 6. Limitations

These are stated in the paper, and repeated here so nobody has to read the paper to find
them.

1. **Ground candidates are not real parcels.** Positions are the centres of 500 m demand
   cells, with the cell standing in for a plot (250 000 m², far above the 1 600 m² minimum).
   Real parcel boundaries, ownership and current use are not represented. Parcel-level data
   would drop straight into `build_ground_candidates()`.
2. **Income is a proxy**, not measured income. See §3.3.
3. **Access time is free-flow Euclidean.** `travel_time_matrix` divides straight-line
   distance by mode speed and ignores network detours. A road-network shortest path is
   implemented (`geo.network_travel_time_matrix`) but off by default because it is slow;
   Table 29 of the paper compares the two. This is the method's main weakness relative to a
   journal like *CEUS*, and the first thing to improve.
4. **One case city (n = 1).** Transferability is untested.
5. **Building heights carry uncertainty.** Per-source `height_sigma_m` is recorded, and the
   sensitivity analysis checks how far the candidate set moves under height error.
6. **The airport-exclusion radius is a convention, not a measurement.** The sensitivity scan
   shows it is the only parameter that reverses a conclusion.

---

## 7. Reproducibility

* Global seed: `project.random_seed` (default 42), fixed through `Config.seed_everything()`.
  Not every reported quantity is seed-independent — `total_demand` drifts slightly
  (≈ 2.2 × 10⁻⁴ relative range across seeds) because K-means assignments shift. Which
  quantities are seed-stable is recorded in `scripts/07_sensitivity.py`, and the
  multi-run script reports mean ± SD.
* All parameters live in `config/chengdu.yaml`. Experiments override, never edit code.
* Every run writes a full log to `outputs/logs/`.
* Intermediate results persist to `outputs/data/` and `data/interim/`, giving an auditable
  chain from raw input to each published number.
* Monte Carlo hypervolume estimation uses a fixed seed, so metrics are reproducible.

Two checkers verify a checkout:

```bash
python scripts/17_check_manuscript_numbers.py           # superseded values: expect 0
python scripts/21_regen_manuscript_tables.py --dry-run  # table drift: expect no output
```

Both compare the manuscript against `outputs/`, so they need the manuscript folder sitting
next to this one and will fail in a code-only checkout. For the table check, only rows
marked `≠` are real differences; unmarked rows are printed for context.

---

## 8. Known issues and roadmap

- [ ] Enable road-network access times by default (`network_travel_time_matrix`) to remove
      the Euclidean approximation.
- [ ] Extend the IP with explicit assignment variables so capacity constraints support load
      balancing, not just a feasibility filter.
- [ ] Obtain Qu et al. (2024), *Green Energy and Intelligent Transportation* 3(3):100173, for
      external calibration of the demand model.
- [ ] Replace the income proxy with observed income data if any is obtainable.

---

## 9. 3D visualisation (CesiumJS)

```bash
python scripts/export_3d.py --data-mode real   # writes viz3d/evtol_data.js
python viz3d/start_3d.py                       # serves locally and opens a browser
```

**Do not open `chengdu_evtol_3d.html` by double-clicking it.** Cesium fetches
`evtol_data.js` and `cesium_token.js` over XHR, which the browser's same-origin policy
blocks under `file://`. The symptom is a blank page with nothing useful in the console.
`start_3d.py` starts a localhost-only static server and opens the page.

### You need your own Cesium ion token

Terrain and building basemaps come from Cesium ion (World Terrain + OSM Buildings), which
needs a free access token. **No token is included in this repository** — it is a personal
credential. To set one up:

1. Create a token at <https://ion.cesium.com/tokens> (free).
2. Write it into `viz3d/cesium_token.local.txt` — one line, no quotes, nothing else.
3. Generate the browser-readable JS:
   ```bash
   python scripts/export_3d.py --token-only
   ```

`cesium_token.local.txt` and the generated `cesium_token.js` should never be committed or
shared. Without a token the page shows an explicit message and a link, not a blank screen.

---

## 10. Citation and contact

* Framework name: **EQUISITE** (EQUIty-aware SITE selection)
* Author: Mohan Long (龙墨翰), Civil Aviation Flight University of China, Guanghan 618307,
  Sichuan, China
* Affiliation ROR: <https://ror.org/01xyb1v19>
* ORCID: <https://orcid.org/0009-0007-3291-8296>
* Corresponding author and funding: to be completed
* **DOI (this version, v1.0.0): <https://doi.org/10.5281/zenodo.22848616>**
* Concept DOI, always resolving to the newest version:
  <https://doi.org/10.5281/zenodo.22848615>

Citation metadata appears in two files, and they do different jobs.
[`CITATION.cff`](CITATION.cff) drives GitHub's "Cite this repository" button.
[`.zenodo.json`](.zenodo.json) is what Zenodo actually reads — **if both are present, Zenodo
ignores `CITATION.cff` entirely**, so the deposit record comes from `.zenodo.json`.

If you use this code, please cite the paper:

```bibtex
@article{long2026equisite,
  title   = {Equity-aware multi-objective siting of urban air mobility vertiports:
             an integrated GIS, demand-stratification and NSGA-II framework with
             application to Chengdu, China},
  author  = {Long, Mohan},
  journal = {Computers, Environment and Urban Systems},
  year    = {2026},
  note    = {Under review}
}

@misc{equisite-code,
  title  = {EQUISITE: equity-aware vertiport siting for urban air mobility},
  author = {Long, Mohan},
  year   = {2026},
  version = {1.0.0},
  doi    = {10.5281/zenodo.22848616},
  url    = {https://github.com/DANQUIXOTE520/equisite}
}
```

### Licence

MIT — see [`LICENSE`](LICENSE). In short: use it, modify it, redistribute it, including
commercially, provided the copyright and permission notices travel with it. No warranty.

Two things the MIT licence does **not** cover:

* **Input data.** OSM, WorldPop, CNBH-10 m, Copernicus DEM and NPP-VIIRS each carry their
  own terms. This repository downloads them; it does not redistribute them.
* **Artefacts derived from OSM** (candidate sets, demand grids, road-network distances)
  may carry ODbL obligations. Check before republishing those CSVs elsewhere.
