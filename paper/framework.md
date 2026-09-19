# Paper Framework — Working Document

**Title (working):** *Equity-aware multi-objective siting of urban air mobility vertiports: an integrated GIS, demand-clustering and NSGA-II framework with application to Chengdu, China*

**Document status:** structure and framework only — NOT publication prose. Every number the paper will need is either given as a placeholder or listed in *Evidence needed*.

**Placeholder conventions used throughout**

| Token | Meaning |
|---|---|
| `[RESULT: ...]` | A value the student must produce from the code. Never fabricate. |
| `[CITATION NEEDED: ...]` | A reference whose full bibliographic details are not yet in hand. |
| `[VERIFY: ...]` | A factual/metadata claim to be checked before submission (journal scope, IF, policy document). |

**Naming conventions** (must match `src/evtol_siting/` and `config/chengdu.yaml`)

| Symbol | Meaning | Config key |
|---|---|---|
| $I$ | 500 m demand grid cells | `stage1_candidates.grid_size_m: 500.0` |
| $J$ | candidate vertiport sites, $J = M \cup G$ | — |
| $M$ / $G$ | rooftop / ground candidates | `stage1_candidates.rooftop` / `.ground` |
| $k$ | income-stratified demand cluster | `stage2_demand.n_clusters` |
| $d_i$ | daily eVTOL trips generated at cell $i$ | `stage2_demand.demand_unit` |
| $t_{ij}$ | access time cell $i$ → site $j$ (min) | `access.modes` |
| $c_j$ | construction cost of site $j$ (CNY) | `costs.rooftop` / `costs.ground` |
| $q_j$ | design capacity of site $j$ | — |
| $R$ | access time budget (min) | `stage3_ip.access_time_budget_min: 15.0` |
| $N_{max}$ | facility-count cap | `stage3_ip.max_sites` |
| $d_{min}$ | core-demand threshold (trips/day) | `stage3_ip.min_core_demand: 500.0` |

---

## 0. Target journal — recommendation and staged plan

### 0.1 Where the corpus actually published (this constrains the realistic target)

| Venue | Corpus precedent | Notes |
|---|---|---|
| *Aerospace* (MDPI) | **Lu et al. 2025, 12(8):709** (Shenzhen, closest prior work) | Direct editorial precedent for vertiport siting |
| *Applied Sciences* (MDPI) | **Jeong et al. 2021, 11(12):5729** | Precedent for UAM site selection |
| *Green Energy and Intelligent Transportation* (Elsevier/KeAi) | **Qu, Huang, Li et al. 2024, 3(3):100173** (Chengdu UAM demand) | Open access, young journal |
| *Engineering* (Elsevier/CAE) | **Wu & Zhang 2021, 7(4):473–487** | High-prestige Chinese-led journal |
| *European Journal of Operational Research* | **Xu, Murray, Church & Wei 2023** | Equity in service allocation — methods venue |
| *Transportation Research Part C* | **Gao et al. 2024** | Noise-aware equitable UATM |
| *交通运输工程学报* (CN, EI/中文核心) | 李卓伦/陆建 et al. 2026, 26(3):89–105; 姜雨 et al. 2026 | Chinese-language route |
| *北京航空航天大学学报*, *交通运输工程与信息学报* | 党庆庆 et al.; 尹浩东 et al. 2025, 23(3):88–102 | Chinese-language route |

**No corpus paper is in CEUS, JTG, or TR-A/D.** The corpus has never cleared those venues. That is the honest starting point.

### 0.2 Assessment

| Venue | Fit of this paper | Difficulty for an undergraduate-led team | Verdict |
|---|---|---|---|
| **Journal of Air Transport Management** (Elsevier) | Very high — vertiport siting and UAM policy are core scope; likes case studies with policy implications | Moderate–high. Wants a clear managerial contribution and rigorous, non-toy case evidence. Single-city is acceptable *if* the equity finding is sharp. | **Stretch target (2nd submission)** |
| **Aerospace** (MDPI) | Very high — Lu et al. 2025 sits here; the same editors handle vertiport siting | Low–moderate. Methodological novelty must be stated loudly, because the venue sees many "GIS + AHP + site ranking" papers. | **Recommended first submission** |
| **ISPRS IJGI** (MDPI) | High — the building-height fusion + GIS candidate generation is genuinely a GIScience contribution | Moderate. Reviewers will interrogate the height-fusion validation and the OSM coverage claim. | **Strong alternate / co-first** |
| **Drones** (MDPI) | Moderate–high — UAM/vertiport papers appear here | Moderate. Risk: perceived as an applications venue; needs the optimization to carry the paper. | Alternate |
| **Sustainability** / *Land* / *Applied Sciences* (MDPI) | Moderate | Low. Fast, but weak signal for a methods contribution; equity framing is welcome. | Fallback only |
| **Computers, Environment and Urban Systems** (Elsevier) | Moderate — the GIS/height-fusion half fits; the NSGA-II half is less CEUS-native | **High.** CEUS wants a *generalizable* urban-analytics method and deep validation, not a one-city layout exercise. | Ambitious stretch |
| **Journal of Transport Geography** | Moderate — equity + accessibility is squarely JTG | **High.** JTG will demand real network travel times and a serious engagement with the equity literature; grid-cell impedance will likely be rejected. | Ambitious stretch |
| **Transportation Research Part A / C / D** | Part C has the closest precedent (Gao et al. 2024) | **Very high.** Requires either a behavioral demand model, a real network, or a genuine methodological advance in the optimization itself. | Unrealistic as first submission |
| **IEEE Access** | n/a (not in corpus) | Moderate on rigour, but the UAM-siting readership is thin and the venue no longer carries the signal it once did. | Not recommended |
| **交通运输工程学报** (CN) | Two corpus papers (2026) | Moderate — but requires a full Chinese rewrite and a Chinese-language framing of novelty. | Parallel track if a CN publication is also wanted |

`[VERIFY: 2025/2026 JCR impact factors and current aims-and-scope text for each shortlisted venue before submission]`

### 0.3 Recommended staged plan

1. **Write the paper once, at full rigour, targeting *Aerospace*.** It is the only venue with a direct precedent for exactly this problem, and it will review the equity objective on its merits rather than demanding a network-based impedance model. Submission window: after *Evidence needed* items E1–E12 are complete.
2. **Simultaneously prepare the JATM version.** JATM will want §6 (Discussion) expanded, the policy implications made Chengdu-specific and concrete, and probably one extra experiment showing the layouts are robust to the demand-model assumption. Keep the same figures; rewrite the framing from "method" to "planning decision support".
3. **Only if both are rejected**, downgrade to *Drones* / *Sustainability*, or upgrade to CEUS/JTG **after** replacing grid-cell impedance with a real network and re-running — which is a new research project, not a revision.
4. **Do not submit to TR-A/C/D or CEUS first.** A desk rejection costs weeks and the reviews will not be actionable at the team's current data resolution.

**Framing rule for every submission:** lead with novelty claim **(A) income-stratified equity as an optimization objective**, because it is the claim that survives a change of venue. Claim (B) rooftop-vs-ground classes is the mechanism that makes (A) interesting, not the headline.

---

## Abstract (structured, ~250 words)

**Purpose.** Compress the whole paper into a structured abstract that a Chengdu planner and an operations-research reviewer can both act on. This is the single most-read element; write it last, and only after every `[RESULT:]` is filled.

**Must carry.** (i) The problem and why it is not yet solved; (ii) the four-stage method named explicitly; (iii) the two novelty claims; (iv) the study area; (v) 4–6 headline numbers; (vi) the policy implication in one sentence.

**Structure** (use the journal's headings — MDPI uses a running paragraph; Elsevier TR/JATM accepts "Highlights" separately):

- **Context** — UAM is criticized as an elite mode; vertiport siting decisions determine who actually gets access.
- **Gap** — existing vertiport siting studies either place equity in a pre-screening stage or optimize a single objective; none treat income-stratified accessibility as an explicit objective, and none model rooftop and ground vertiports as distinct facility classes inside one optimization.
- **Method** — four stages: (1) GIS candidate generation over a 500 m grid with multi-source building-height fusion; (2) K-means demand stratification on a relative-wealth proxy and POI density; (3) exact 0-1 integer programming (set-covering with capacity) producing two scenario layouts; (4) NSGA-II over a four-objective vector (unserved demand, demand-weighted access time, inequity, cost), benchmarked against four classical single-objective baselines solved to optimality.
- **Case** — Chengdu, China (within the G4201 Ring Expressway plus the Tianfu New Area core).
- **Findings** — `[RESULT: candidate counts by class]`; `[RESULT: K selected by silhouette and the cluster trip-rate ladder]`; `[RESULT: IP layouts — site counts, cost, coverage for both scenarios]`; `[RESULT: Pareto front size and knee point]`; `[RESULT: the equity gap between best and worst income group, and how much it narrows under NSGA-II vs the single-objective baselines]`.
- **Implication** — one sentence: e.g. "Cost-minimal siting concentrates service in high-income clusters; treating equity as an explicit objective recovers `[RESULT: X%]` of the worst-group access-time gap at `[RESULT: Y%]` additional cost."

**Figures/tables:** none.

---

## 1. Introduction

**Purpose.** Establish that vertiport siting is a live planning problem, that UAM's equity problem is real and acknowledged, and that the two gaps this paper fills are both open. Land the contribution statement by the end of page 2.

**Must carry.**
- UAM/eVTOL as a policy reality, not a concept — cite the Chinese national and Chengdu municipal context. `[CITATION NEEDED: Chengdu low-altitude economy development plan / Sichuan provincial UAM policy document]` `[VERIFY: exact document title and year]`
- The equity critique: UAM is widely described as an elite mode because early fares and network design favour high-income, time-sensitive travellers.
- The siting decision is where that critique is either mitigated or baked in — a cost-minimal siting model serves the demand-dense core, which is also the high-income core.
- Prior work stops short: equity enters as a *screening criterion* or a *post-hoc check*, never as an objective; and rooftop/ground is never a modelled axis.
- Chengdu has **never** been used as a vertiport *siting* case (Qu et al. 2024 model Chengdu UAM **demand**, not siting) — so the case is both novel and non-arbitrary (large, flat-core, high-rise, two-airport city).

**Outline**
- 1.1 Background: urban air mobility and the vertiport siting problem.
- 1.2 The equity problem in UAM — why siting is the lever.
- 1.3 Research gaps (state (A) and (B) verbatim as the two contributions).
- 1.4 Contributions and novelty of this work (numbered list, 4 items).
- 1.5 Paper organization (one short paragraph — omit if the venue does not want it).

**Contribution list to reproduce in 1.4 (order matters — this ordering is the paper's spine):**
1. **Income-stratified equity as an explicit optimization objective** in vertiport siting (novelty A).
2. **Rooftop and ground vertiports modelled as distinct facility classes** with different cost structures and capacities within a single optimization (novelty B).
3. A multi-source building-height fusion procedure that recovers tall buildings that OSM tags alone discard.
4. A rigorous demonstration — against p-median, p-center, MCLP and set-covering solved exactly — that the multi-objective treatment is *necessary*, not merely preferable.

**Figures/tables:** Fig. 1 (study area) may be referenced in 1.1 to orient the reader; Fig. 2 (workflow) is referenced in 1.4.

---

## 2. Literature review

**Purpose.** Not a survey. It is an argument in three movements that terminates in two explicitly unoccupied cells of the literature. Every paragraph must be doing work toward §2.4.

**Must carry.**
- **2.1 Vertiport siting methods.** GIS-based suitability/multi-criteria screening is the dominant paradigm (Jeong et al. 2021, *Applied Sciences*). Note the structural limitation: MCDM ranks candidate sites but does not decide *how many* or *which combination* serves a demand surface. Optimization-based siting is the smaller, more recent strand (Lu et al. 2025, *Aerospace*).
- **2.2 Demand modelling for UAM.** Chengdu-specific demand modelling exists (Qu, Huang, Li et al. 2024) — **flag as must-acquire**; this is the closest thing to a demand anchor for the case city, and the paper must state plainly whether the present demand model is calibrated against it or is an independent construction. Also cite higher-level UAM demand/system studies (Wu & Zhang 2021, *Engineering*) and recent demand-estimation work (Yoon et al. 2025, arXiv:2502.00399). Chinese-language reviews on vertiport planning: 姜雨 et al. 2026; 党庆庆 et al. `[CITATION NEEDED: 北京航空航天大学学报 — year, volume, pages]`; 尹浩东 et al. 2025, 23(3):88–102; 李卓伦/陆建 et al. 2026, 26(3):89–105.
- **2.3 Equity in facility location.** The general theory: p-median (Hakimi 1964) is efficiency-oriented, p-center and MCLP (Church & ReVelle 1974) are minimax/coverage-oriented, and equity in *service allocation* has been formalized by Xu, Murray, Church & Wei 2023 (EJOR). Gao et al. 2024 (TR Part C) bring equity to UATM but on the **noise-exposure** axis, not the income-access axis. Fadhil 2018 uses income **regressively** — high income is scored as *more suitable* — which compounds rather than mitigates the equity problem, and is a useful foil.
- **2.4 The two gaps — build the argument that they are unoccupied.** This is the most important subsection in the paper.

**2.4 must be written as an explicit elimination, not an assertion.** Suggested structure — a short table (Table 1, the gap matrix) whose rows are the closest papers and whose columns are: *uses income-stratified groups*, *equity is an optimization objective (not a screen)*, *multi-objective*, *rooftop vs ground as distinct classes*, *exact/benchmarked optimization*.

Gap (A) argument, step by step:
1. Lu et al. 2025 (Shenzhen) is the closest work in the corpus. It **does** consider equity — but in the **screening** stage, and optimizes a **single** objective. Equity is therefore an input filter, not a traded-off quantity; the model can never report how much equity it costs, or is bought by, the layout.
2. Fadhil 2018 uses income as a positive suitability weight, which actively steers facilities toward high-income areas.
3. Xu et al. 2023 provide the equity-in-allocation machinery, but for generic service facilities, not UAM, and not with an income-stratified demand surface.
4. Gao et al. 2024 optimize equity for UATM, but over noise exposure.
5. **Therefore:** no paper in the corpus makes income-stratified accessibility an explicit objective of a vertiport siting optimization. State this as the claim, and support it with Table 1.

Gap (B) argument:
1. Tiered vertiport concepts exist — NASA's Vertihub / Vertiport / Vertistop `[CITATION NEEDED: NASA UAM vertiport tier definition — technical report or AC]`; Guo et al.'s Vertipad / Vertibase / Vertihub `[CITATION NEEDED: Guo et al. — full citation]`. These tiers are defined by **function and traffic volume**, not by **land ownership and construction type**.
2. The rooftop/ground distinction is therefore not a relabelling of an existing tier scheme: rooftop sites have a different cost structure (high fixed cost for structural reinforcement, lift and fire-safety retrofit; **no land acquisition**) and a different capacity ceiling, versus ground sites (land acquisition plus civil works, larger footprint).
3. **Therefore:** no paper models the rooftop/ground axis as an optimization variable with class-specific cost and capacity. The consequence is that the existing literature cannot answer whether a rooftop-led or ground-led network is the better equity/cost trade — a question Chengdu's built form makes unavoidable.

**Figures/tables:** **Table 1** (gap matrix) is mandatory. No figures.

---

## 3. Study area and data

**Purpose.** Make the case reproducible and the data limitations explicit before any method is described. Reviewers will read §3 to decide whether to trust §4.

**Must carry.**
- **Study area definition and area.** Chengdu, Sichuan. Boundary = the area within the G4201 Ring Expressway plus the Tianfu New Area core. Bounding box `[103.880, 30.360, 104.280, 30.820]` (WGS84). Projected CRS EPSG:32648 (UTM 48N) for all distance and area computation. **629.8 km²**; **2 519** grid cells (computed by `geo.make_grid`).
- **Why Chengdu.** Fast-growing, high-rise, two commercial airports (CTU Shuangliu, TFU Tianfu) whose clear zones carve out large exclusion areas — a non-trivial candidate-elimination geometry. Flat core with mountain edge. `[RESULT: population within the study area]`
- **Administrative/descriptive context:** districts, population, GDP. `[VERIFY: use the latest Chengdu Statistical Yearbook; state the year]`
- **The data stack — present as Table 2 with a provenance column** (source, resolution/scale, date of access, licence):

| Dataset | Source | Resolution / scale | Date accessed |
|---|---|---|---|
| Building footprints + tags | OpenStreetMap (Overpass API) | vector | `[RESULT: date]` |
| Building height raster | CNBH-10m / GlobalBuildingAtlas `[VERIFY: the exact product actually used and its native resolution]` | `[RESULT: resolution]` | `[RESULT]` |
| Population | WorldPop | 100 m | `[RESULT]` |
| Relative wealth | Meta Relative Wealth Index (RWI) | 2.4 km, IDW-interpolated to grid | `[RESULT]` |
| POI | OSM (amenity / shop / office / leisure / tourism) | vector, aggregated per km² | `[RESULT]` |
| Road network | OSM | vector | `[RESULT]` |
| Land use | OSM landuse / natural | vector | `[RESULT]` |
| Terrain | Copernicus DEM GLO-30 | 30 m | `[RESULT]` |

- **3.3 The height-coverage problem — a headline finding, not a footnote.** Measured coverage of OSM height tags across central Chengdu: of **69 675** buildings in the study area, only **799 (1.1 %)** carry a `height` tag and **5 483 (7.9 %)** carry `building:levels`; **63 812 (91.6 %)** carry neither. A pipeline that trusts tags alone therefore **systematically discards more than 80 % of tall buildings** — which would silently shrink the rooftop candidate set by roughly an order of magnitude and bias it toward the small share of buildings that happen to be well-mapped (typically landmark and commercial buildings). Realised figures (see `outputs/tables/table_height_sources.csv`): tag-only coverage 8.4 % → **100 %** after fusion, decomposed as CNBH-10 m raster 70.6 % / quantile-regression imputation 21.0 % / `building:levels` 7.3 % / `height` 1.1 %. Data snapshot: OSM.fr Sichuan extract; state the extract date in the final manuscript.
- **3.4 The fusion solution.** The four-level cascade implemented in `data/heights.py`: (1) OSM `height` tag; (2) OSM `building:levels` × storey height (building-type-dependent, plus a 0.6 m ground-floor allowance); (3) building-height raster sampled at footprint centroid, treating raster 0 as "no building" rather than height 0; (4) quantile-regression imputation from footprint area. Each level carries an uncertainty $\sigma_h$ (0.5 m / 1.5 m / `[RESULT]` / `[RESULT]`), propagated in §5.6.
- **3.5 Ethical and quality notes.** State up front, in §3 not §6: RWI is a **relative wealth proxy**, not income data; no individual-level or household income is used; POI density is an OSM-mapped-activity proxy with known completeness bias toward the urban core.

**Outline**
- 3.1 Study area and rationale for Chengdu.
- 3.2 Data sources and preprocessing (Table 2).
- 3.3 The building-height coverage problem (Fig. 3).
- 3.4 Multi-source height fusion and its uncertainty (Fig. 4).
- 3.5 Data limitations and caveats (the three honest statements above).

**Figures/tables:** Fig. 1, Fig. 3, Fig. 4; Table 2. Possibly Table 3 (OSM tag coverage by district, if Fig. 3 alone is not enough).

---

## 4. Methodology

**Purpose.** Fully specify the four stages so the study is reproducible without reading the code, and so a reviewer cannot find an undefined symbol. Every symbol used in §5 must be defined here. Include a notation table if the venue allows (Table 4).

### 4.1 Stage 1 — GIS candidate generation

**Purpose.** Turn the built environment into a finite, defensible candidate set $J$.

**Content — the three sub-steps, with the actual predicates:**

*Rooftop candidates* $M$ — a building qualifies if
$$h_j \ge 20\ \text{m} \quad \wedge \quad A_j^{roof} \cdot \rho \ge 1600\ \text{m}^2$$
where $\rho$ is the usable-roof ratio (default 0.70, deducting lift shafts, plant rooms and parapets). The 1600 m² threshold corresponds to a 40 m × 40 m FATO-plus-safety-area footprint. `[CITATION NEEDED: the vertiport design standard underlying the 1600 m² and 20 m thresholds — EASA PTS-VPT-DSN, FAA Engineering Brief 105, or the relevant Chinese standard; the paper must cite one and state why]`

*Ground candidates* $G$ — a candidate qualifies if
$$\text{slope}_j \le 15^\circ \quad \wedge \quad \text{landuse}_j \notin E \quad \wedge \quad A_j^{parcel} \ge 1600\ \text{m}^2$$
with $E$ the excluded land-use set (water, reservoir, riverbank, wetland, marsh, wood, forest, scrub, grassland, farmland, orchard, vineyard, quarry, landfill) and the excluded `natural` set (water, wetland, wood, scrub, peak, cliff, ridge, bare_rock, sand, beach). Slope is computed from Copernicus GLO-30.

*Safety exclusions* — remove any candidate within the airport clear-zone buffer (8000 m of CTU and TFU, `[VERIFY: this is a modelling choice — state it as such and test it in §5.6]`) or within 1000 m of an existing heliport.

*Density dilution* — enforce a minimum spacing of $c_{min} = 500$ m between retained candidates, and at most `max_candidates_per_grid = 3` per 500 m grid cell, so that spatial clustering of tall buildings does not translate into a degenerate cluster of near-identical candidates. Retain by a quality score (roof size, height, height-source reliability).

**Critical framing for this subsection.** State explicitly that **ground candidates are grid-cell proxies derived from raster and land-use screening, not surveyed parcel polygons**. They represent "a site of at least 1600 m² exists in this cell", not a specific lot with an owner. This is a deliberate and stated simplification; §6.4 lists what it forecloses.

**Figures/tables:** Fig. 5 (four-panel candidate map), Fig. 6 (candidate characteristics), Table 5 (candidate counts by filter stage), Table 6 (cost and capacity parameters by class).

### 4.2 Stage 2 — K-means demand stratification

**Purpose.** Replace the single scalar demand surface used by essentially all prior vertiport siting work with a **stratified** one, which is a prerequisite for any income-based equity objective.

**Content.**
- Feature vector per grid cell $i$: $z_i = (\text{RWI}_i^{\text{IDW}},\ \text{POIdensity}_i)$, z-score standardized. RWI is bilinearly/IDW interpolated from its native 2.4 km support to the 500 m grid — state plainly that this **upsamples a coarse surface** and that it therefore cannot resolve intra-neighbourhood income variation. `[RESULT: correlation between RWI and POI density, to show the two features are not redundant]`
- K-means on $\{z_i\}$, minimizing $\sum_{k}\sum_{i \in C_k}\|z_i - \mu_k\|^2$, with $K$ selected by **silhouette score** over $K \in [2,10]$.
- **Differentiated trip-generation rates.** Each cluster $k$ receives a rate $\tau_k$ (trips/person/day), monotonically increasing in cluster mean RWI:
$$\tau_k = \tau_{base}\left[1 + (\rho_{up} - 1)\, \tilde{r}_k\right]$$
where $\tilde{r}_k \in [0,1]$ is the normalized income rank of cluster $k$, $\tau_{base} = 0.0015$ trips/person/day for the lowest-income cluster, and $\rho_{up}$ is the high-to-low ratio (default 5, varied 1–10 in §5.6).
- Demand: $d_i = p_i \cdot \tau_{k(i)}$.
- **Be explicit that $\tau_{base}$ and $\rho_{up}$ are uncalibrated, and that only their ratio matters.** The paper's claims must be about the *relative* structure of demand, never about absolute trip volumes. This is the honest formulation and it also defuses a reviewer objection.

**Figures/tables:** Fig. 7 (features + silhouette curve + cluster map), Fig. 8 (demand surface), Table 7 (K-selection diagnostics), Table 8 (cluster profiles: income mean, POI density, population, $\tau_k$, total demand).

### 4.3 Stage 3 — 0-1 integer programming

**Purpose.** Produce two **exactly optimal** reference layouts that the NSGA-II results can be measured against, and which are themselves publishable planning scenarios.

**Formulation.** Let $y_j \in \{0,1\}$ indicate that candidate $j$ is opened, and let $\mathcal{N}(i) = \{j : t_{ij} \le R\}$ be the set of candidates reachable from cell $i$ within the access budget.

$$\min \sum_{j \in J} c_j y_j \qquad \text{(Scenario A: cost priority)}$$
$$\min \sum_{j \in J} y_j \qquad \text{(Scenario B: facility-count priority)}$$

subject to

$$\underbrace{\sum_{j \in \mathcal{N}(i)} y_j \ge 1 \quad \forall i \in I : d_i \ge d_{min}}_{\text{core demand coverage}} \qquad
\underbrace{\sum_{j \in J} y_j \le N_{max}}_{\text{facility-count cap}} \qquad
\underbrace{\ell_j \, y_j \le q_j \quad \forall j}_{\text{capacity}} \qquad
y_j \in \{0,1\}$$

where $\ell_j = \sum_{i : j = \arg\min_{j' \in \mathcal{N}(i)} t_{ij'}} d_i$ is a **precomputed nearest-assignment load upper bound** for site $j$, and $q_j$ is its design capacity.

**Be transparent about the capacity constraint — this is the paper's own acknowledged soft spot.** Written this way, the capacity constraint is a **site-admissibility screen**, not a load-balancing constraint: it excludes candidates whose nearest-assignment load exceeds their own capacity, which is exactly what prevents the unrealistic "few giant vertiports" solutions, but it does **not** redistribute demand between open sites. Say this explicitly in §4.3 (do not let a reviewer discover it), and point forward to the limitation in §6.4 and the mitigation experiment in §5.6 (adding explicit assignment variables $x_{ij}$ for the top-decile demand cells).

Solved exactly with CBC (PuLP), 300 s time limit.

**Two scenarios.** Scenario A: $N_{max}=20$, $R=3000$ m. Scenario B: $N_{max}=35$, $R=2000$ m. `[RESULT: whether both solve to proven optimality within the time limit, and the actual solve times]`

**Figures/tables:** Fig. 9 (two scenario layouts with coverage catchments), Table 9 (IP scenario results: sites, rooftop/ground split, cost, demand coverage, population coverage, worst-group access time).

### 4.4 Stage 4 — NSGA-II formulation

**Purpose.** The core methodological contribution. This subsection must be written with enough rigor that a reviewer cannot accuse the objectives of being ad hoc.

**Encoding.** Binary mask $x \in \{0,1\}^{|J|}$, one bit per candidate. Note the alternative (variable-length integer encoding over site indices) and state that the mask was chosen because it makes the crossover/mutation operators well-defined and the repair operator cheap. `[RESULT: if the integer encoding was run as a comparison, report the comparison]`

**Four objectives, all minimized.** Write each with its units — the units are what make the objectives interpretable and are what a reviewer will check.

**(1) Unserved demand** (trips/day)
$$f_1(x) = \sum_{i \in I} d_i - \sum_{i \in \mathcal{I}^+(x)} d_i$$
where $t_i^*(x) = \min_{j \in S(x)} t_{ij}$ is the access time to the nearest open site, $S(x) = \{j : x_j = 1\}$, and $\mathcal{I}^+(x) = \{i : t_i^*(x) \le R\}$ is the served set.

**(2) Total demand-weighted access time** (trips·min/day)
$$f_2(x) = \sum_{i \in \mathcal{I}^+(x)} d_i \, t_i^*(x)$$

**(3) Inequity — worst income group's demand-weighted mean access time** (min)
$$f_3(x) = \max_{k \in \{1,\dots,K\}} \frac{\sum_{i \in I_k \cap \mathcal{I}^+(x)} d_i \, t_i^*(x)}{\sum_{i \in I_k \cap \mathcal{I}^+(x)} d_i}$$
with $f_3(x) = +\infty$ (implemented as a large finite constant) if any group is entirely unserved. The demand weighting is deliberate and must be justified in the text: an unweighted mean over cells would let a group with a few very remote cells look badly served, whereas the demand-weighted mean reflects the experience of a *typical member* of that group.

**Justify this functional form explicitly, and give the alternatives** — this is where "the equity objective is arbitrary" gets pre-empted:
- rejected: the Gini coefficient over group means (aggregates but is insensitive to *which* group is worst, and is not aligned with the Rawlsian concern that motivates the paper);
- rejected as primary, retained as a reported diagnostic: max-min ratio $\max_k \bar{t}_k / \min_k \bar{t}_k$ (scale-free but unstable when the best-served group has near-zero time);
- **adopted:** max over groups of the demand-weighted mean — a Rawlsian (maximin) criterion, which is the standard formalization of "worst-off group" equity and maps directly onto the policy question.

**(4) Total construction cost** (CNY)
$$f_4(x) = \sum_{j \in S(x)} c_j, \qquad
c_j = \begin{cases} c^{R}_{fix} + c^{R}_{m^2} \cdot A_j^{roof,usable} & j \in M \\[4pt] c^{G}_{fix} + \left(c^{G}_{land} + c^{G}_{m^2}\right) \cdot A_j^{parcel} & j \in G \end{cases}$$

with default parameters $c^R_{fix} = 1.2\times10^7$, $c^R_{m^2} = 8\times10^3$; $c^G_{fix} = 2.5\times10^7$, $c^G_{land} = 3\times10^3$, $c^G_{m^2} = 5\times10^3$ CNY. **This is where novelty (B) physically enters the model**: the two facility classes have different fixed costs and different area-scaling terms, so the optimizer can trade rooftop against ground. State that these are planning-level unit costs from `[CITATION NEEDED: cost basis for vertiport construction — a design consultancy report, a comparable infrastructure cost study, or Chinese civil-works unit costs]` and that all cost conclusions are ratio-based within the model. `[VERIFY: cite the actual source of these unit costs, or reclassify them as illustrative and test them in §5.6]`

**Constraint handling.** Coverage of core demand cells is treated as a hard constraint enforced by a **greedy repair operator**: for an infeasible individual, iteratively add the candidate that newly covers the most currently-uncovered core cells, breaking ties in favour of lower cost. The rationale — state it plainly — is that coverage is the one part of the problem a greedy heuristic approximates very well, so injecting it as a prior lets the evolutionary search spend its budget on the *trade-offs above coverage* (cost, time, equity) instead of re-learning coverage every generation. Report the repair's effect: `[RESULT: fraction of the initial random population that is infeasible before repair]`

**Algorithm settings.** Population 200, generations 300, SBX crossover ($p_c = 0.9$, $\eta_c = 20$), polynomial mutation ($p_m = 1/|J|$, $\eta_m = 20$), binary tournament, 10 independent runs with a fixed global seed for reproducibility. Memoization of the fitness function (exact mask-keyed cache) is an implementation detail worth a sentence, because it is what makes $6\times10^4$ evaluations tractable. `[RESULT: actual wall-clock time per run, and the cache hit rate]`

**Knee point / compromise solution.** Select the knee point of the non-dominated front (and, optionally, TOPSIS ranking) to nominate one layout for policy discussion. State the method used and report the chosen solution's objective vector.

**Figures/tables:** Fig. 10 (Pareto projections), Fig. 11 (knee point + layout map), Table 10 (NSGA-II parameters), Table 11 (Pareto front quality metrics).

### 4.5 Baselines

**Purpose.** Demonstrate that the four-objective formulation is *necessary* — that single-objective classics, even when solved to proven optimality, produce layouts that are worse on the equity metric the paper cares about.

**Content.** Four classical models, each solved exactly with CBC, then **evaluated on all four objectives** so the comparison is apples-to-apples:

| Baseline | Objective | Classical role |
|---|---|---|
| p-median (Hakimi 1964) | minimize demand-weighted access time, $p$ facilities | efficiency |
| p-center | minimize the maximum access time | minimax fairness (spatial) |
| MCLP (Church & ReVelle 1974) | maximize demand covered within $R$ | coverage |
| Set-covering (= Stage 3 IP) | minimize cost subject to full core coverage | cost |

**The argument the paper must make.** p-center is the natural single-objective proxy for "fairness" — but it minimizes the worst *individual cell*, not the worst *income group*. This distinction is the paper's central empirical claim, and Table 12 must show it: a p-center layout can have a small maximum access time while still leaving the worst *income group* badly served, because the worst-served cell may sit in a sparsely populated high-income area. If the runs do not show this, the paper must report that honestly and adjust the claim.

**Figures/tables:** Fig. 12 (baseline comparison bar chart), Table 12 (all solutions evaluated on all metrics — the paper's central quantitative table).

### 4.6 Evaluation metrics

**Purpose.** Define every number reported in §5, in two distinct families, so that algorithmic quality and decision quality are never conflated.

**(a) Pareto front quality** (algorithmic — measures the front, not the plan). Report mean ± standard deviation over 10 independent runs:
- **Hypervolume (HV)** — relative to a reference point; state how the reference point is set (the code uses a merged non-dominated front as reference). `[RESULT: exact reference-point convention actually used, and whether the 2D/3D exact or Monte-Carlo HV routine was used]`
- **Spacing** — uniformity of the front.
- **Spread** — extent plus distribution.
- **IGD** — distance from the true/reference front; state that the reference front is the merged non-dominated set of all algorithms compared, which makes IGD here a *relative* rather than absolute indicator.

**(b) Decision-level metrics** (measures the plan a planner would actually build):
- demand coverage % and population coverage %;
- number of sites, split by rooftop/ground;
- total construction cost;
- mean and median access time;
- **Gini coefficient over income-group mean access times** (report alongside — but not instead of — the maximin objective, with a note on why they are different);
- **worst-group access time** — the decision-level counterpart of objective $f_3$;
- income-quintile gap and quintile Gini, as an independent check that the cluster-based equity measure is not an artefact of the clustering.

**Note for the writer:** keep (a) and (b) in separate tables. Mixing HV with coverage invites the reviewer to ask why a plan with a better front is not a better plan.

**Figures/tables:** Table 11 (front quality), Table 12 (decision metrics), Table 13 (equity decomposition by group).

---

## 5. Results

**Purpose.** Report, in a fixed order: what the data permitted (5.1), what the demand looks like (5.2), what the exact models chose (5.3), what the trade-off surface looks like (5.4), whether the multi-objective treatment was necessary (5.5), and whether any of it survives parameter perturbation (5.6). **Every claim in §6 must be traceable to a numbered result here.**

### 5.1 Candidate set and height-fusion outcomes

- Candidate counts at each filter stage: all buildings → height ≥ 20 m → roof ≥ 1600 m² → safety exclusions → dilution. Report rooftop and ground separately.
- **The headline for this subsection (measured):** a tag-only pipeline yields **227** rooftop candidates; the full fusion cascade yields **1 430** — a **6.3× increase**. **A tag-only pipeline silently discards 84.1 % of qualifying rooftop sites.** The bias is directional and demonstrable: buildings carrying *some* height tag have a CNBH-measured median height of **23.0 m** (n = 5 304) versus **18.7 m** (n = 49 169) for untagged buildings — i.e. OSM tags preferentially describe the taller, better-mapped buildings, so tag-only pipelines are biased toward exactly the landmark buildings they already capture. This is the empirical justification for Stage 1's height-fusion design, and it is a reusable finding for any rooftop-siting study in a data-sparse city.
- Height-source composition: what fraction of the final rooftop candidates came from each of the four cascade levels.
- Sensitivity of the candidate set to $\sigma_h$: does the rooftop set change materially if imputed heights are perturbed?
- **Honesty requirement:** if the imputed (level-4) heights contribute a large share of the final rooftop candidates, say so prominently and treat it as the pipeline's main validity threat, not a footnote.

**Figures/tables:** Fig. 5, Fig. 6, Table 5, Table 6, Table 3.

### 5.2 Demand clusters

- Silhouette curve and selected $K$; report the runner-up $K$ and note that the analysis is repeated for it in §5.6.
- Cluster profile table: mean RWI, POI density, population, $\tau_k$, total demand.
- Map of clusters; map of the resulting demand surface.
- **The finding that sets up the whole paper:** are high-RWI and high-demand cells spatially coincident with the dense core? Quantify the overlap. `[RESULT: correlation between cluster income rank and demand density; spatial concentration of the top-income cluster]`
- Cross-check with Qu et al. 2024 (Chengdu UAM demand): state whether the spatial pattern is consistent, and be explicit that this is a qualitative consistency check, not a calibration.

**Figures/tables:** Fig. 7, Fig. 8, Table 7, Table 8.

### 5.3 IP scenario layouts

- Both scenarios: site counts, rooftop/ground split, total cost, demand coverage, population coverage, core-coverage (must be 100 % where feasible), worst-group access time.
- Maps with access catchments.
- Whether the capacity constraint changed the solution — report the same model with `enforce_capacity = False` and quantify the difference. **This is the direct evidence for the claim that without capacity the model returns unrealistic giant vertiports.** `[RESULT: sites and cost with and without the capacity constraint]`
- Note any infeasible core cells and how they were handled.

**Figures/tables:** Fig. 9, Table 9.

### 5.4 Pareto front and knee point

- Front size and dimensionality (how many of the four objectives are actually in tension — if two objectives are strongly correlated, say so; a four-objective formulation where one objective is redundant is a weakness a reviewer will find).
- Pairwise objective projections (cost vs unserved; cost vs inequity; inequity vs unserved) — the cost-vs-inequity plot is the paper's signature figure.
- **The key quantitative statement of the paper:** `[RESULT: the marginal cost of reducing worst-group access time by one minute — i.e., the slope of the cost–inequity trade-off along the front]`
- Knee point and the nominated compromise layout; objective vector of the knee point vs both IP scenarios.

**Figures/tables:** Fig. 10, Fig. 11, Table 11.

### 5.5 Comparison against single-objective baselines

**This is the subsection that carries secondary contribution 4.** Structure it as an argument, not a list:
1. Present all five solution families (p-median, p-center, MCLP, set-covering, NSGA-II knee) in one table, all evaluated on all decision metrics.
2. Show that each baseline is optimal *for its own objective* — this is why they were solved exactly, and it must be stated so the comparison is not read as an unfair one.
3. **Show where they fail on equity.** Specifically: p-center's minimax objective optimizes the worst *cell*; report the worst *income group* it produces, and compare to the NSGA-II knee. If p-center happens to do well on the group metric, report that and revise the framing — the honest outcome is still publishable and more credible.
4. Show the rooftop/ground composition of each baseline layout — this is the empirical content of novelty (B), and it should reveal whether single-objective models systematically prefer one class.

**Figures/tables:** Fig. 12, Fig. 13, Table 12, Table 13.

### 5.6 Sensitivity analysis

**Purpose.** Establish that the conclusions are properties of the problem, not of the parameter guesses. Report all of the following, each as a one-factor sweep with the *decision metrics* of §4.6(b) as the response:

| Parameter | Range | Why it matters |
|---|---|---|
| Minimum roof area | 1400 – 2500 m² | Directly controls rooftop candidate supply; the 1600 m² threshold is a design-standard choice |
| Access radius / time budget $R$ | 2 – 5 km (or the equivalent minutes) | Drives coverage feasibility and the whole cost–equity trade |
| Number of clusters $K$ | 3 – 8 | Tests whether the equity finding is an artefact of the clustering granularity |
| Trip-rate income ratio $\rho_{up}$ | 1 – 10 | **The most important test.** At $\rho_{up}=1$ demand is income-independent — if the equity gap persists even then, the gap is spatial, not demand-driven; if it vanishes, the equity problem is entirely a demand-model artefact and the paper must say so. |
| Cost parameters | rooftop fixed ±50 %, ground fixed ±50 %, land price ±50 % | Tests whether the rooftop/ground preference is robust to the cost guesses |
| Airport exclusion radius | 6 – 10 km | Tests the largest single geometric exclusion |
| Building-height uncertainty $\sigma_h$ | perturb imputed heights by ±$\sigma_h$ | Tests the height-fusion pipeline's influence on the layout |

- Report as a tornado/line figure plus a table. For each parameter, state whether the *ranking* of the solution families changes, not just the absolute values — ranking stability is the strong claim.
- Add a note on the canonical reviewer follow-up — a **weighted-sum scalarization** compared against NSGA-II — with the outcome. `[RESULT: whether the weighted-sum sweep was run and, if so, whether its solutions are dominated by the NSGA-II front]`

**Figures/tables:** Fig. 14, Table 14.

---

## 6. Discussion

**Purpose.** Convert results into claims that a planner or a reviewer will act on, and pre-empt the limitations a reviewer would otherwise raise as objections.

**Must carry.**
- **6.1 Policy implications for Chengdu.** Concrete: where should the first tranche of vertiports go; is a rooftop-led or ground-led network better for equity; what does the equity–cost trade-off price the worst-served group's time at; what would a purely cost-driven plan cost the lowest-income group. Anchor to Chengdu's actual planning context `[CITATION NEEDED: Chengdu low-altitude economy plan / vertiport pilot policy]`.
- **6.2 The equity finding.** The paper's most transferable claim. If income-stratified demand plus spatially concentrated high-income areas is the mechanism, then the finding generalizes to any city with that structure — state the precondition explicitly, and state that it is a *precondition*, not a universal.
- **6.3 Transferability.** The four-stage pipeline is city-agnostic; what is Chengdu-specific is (a) the two-airport exclusion geometry, (b) the RWI/POI demand structure, (c) the rooftop supply implied by the local building stock. Say which parts are reusable as-is and which need re-derivation.
- **6.4 Limitations — write these as full paragraphs, not a buried list.** At minimum:
  1. **RWI is a relative wealth proxy, not income data** — it cannot support claims about absolute affordability or fare policy.
  2. **Ground candidates are grid-cell proxies**, not surveyed parcels with owners and prices — the ground cost estimates are therefore planning-level.
  3. **Access time is Euclidean/straight-line based, not network-based** — this overstates accessibility and, more importantly, may understate the access penalty in the low-income periphery, which is exactly the effect the paper is measuring. State the direction of the likely bias.
  4. **The capacity constraint is a site-admissibility upper bound**, not a load-balancing constraint (§4.3).
  5. **The demand model is not calibrated to observed vertiport ridership** — no such data exists for Chengdu, since no vertiports operate commercially there.
  6. **Single case study (n = 1)** — the transferability claim (§6.3) is an argument, not an empirical result.
  7. **Trip rates are uncalibrated**; only relative structure is claimed.
- **6.5 Future work** — network-based impedance; real parcel data via land-registry or planning documents; adding the assignment variables $x_{ij}$; extending the equity objective to other axes (age, car ownership, disability); a second city for an actual cross-case test.

**Figures/tables:** reference Fig. 10 and Table 12 (the trade-off evidence); no new figures.

---

## 7. Conclusion

**Purpose.** Four paragraphs, no new material.

- **Paragraph 1 — what was done:** the four-stage framework, the case, the facility classes.
- **Paragraph 2 — what was found:** the headline numbers only, each already reported in §5.
- **Paragraph 3 — why it matters:** the two novelty claims restated as findings, not as promises.
- **Paragraph 4 — what it does not settle:** one sentence pointing to §6.4, plus the single most important next step.

**Figures/tables:** none.

---

## Consolidated Figure list

| # | Short title | Description | Type |
|---|---|---|---|
| **Fig. 1** | Study area | Chengdu study area: G4201 Ring Expressway boundary, Tianfu New Area core, 500 m demand grid, districts, CTU/TFU airports with clear-zone buffers, terrain hillshade | Map |
| **Fig. 2** | Methodological workflow | Four stages (GIS candidate generation → K-means demand stratification → 0-1 IP → NSGA-II) with the data stack feeding in from the left and the evaluation/baseline comparison below | Flow diagram |
| **Fig. 3** | OSM height-tag coverage | Panel (a) % of buildings with `height`; (b) % with `building:levels`; (c) % with neither, per 500 m cell; (d) histogram of coverage across districts | Map (a–c) + bar/histogram (d) |
| **Fig. 4** | Height fusion outcome | Panel (a) fused building-height surface; (b) composition of rooftop candidates by height source; (c) fused height vs footprint area coloured by source | Map + bar + scatter |
| **Fig. 5** | Candidate generation | Four panels showing the sequential filters: all buildings ≥ 20 m → roof ≥ 1600 m² → after safety exclusions → after 500 m dilution | Four-panel map |
| **Fig. 6** | Candidate characteristics | Scatter of building height vs usable roof area for rooftop candidates (coloured by height source); distributions of cost and capacity for rooftop vs ground classes | Scatter + box/violin |
| **Fig. 7** | Demand stratification | Panel (a) RWI surface (note native 2.4 km support); (b) POI density; (c) silhouette score vs $K$; (d) cluster assignment map | Map + line |
| **Fig. 8** | Demand surface | Daily trips per 500 m cell, and stacked demand by cluster (bar), showing spatial concentration of high-income demand | Map + bar |
| **Fig. 9** | IP scenario layouts | Two panels: Scenario A (cost priority, $N_{max}=20$) and Scenario B (facility-count priority, $N_{max}=35$), each showing site locations by class and 15-min access catchments | Two-panel map |
| **Fig. 10** | Pareto front projections | Pairwise objective projections of the non-dominated front, with the cost-vs-inequity panel highlighted as the signature result | Scatter (multi-panel) |
| **Fig. 11** | Knee point | Panel (a) front with knee point and both IP scenarios marked; (b) the knee-point layout mapped | Scatter + map |
| **Fig. 12** | Baseline comparison | Decision metrics (demand coverage, worst-group access time, cost, rooftop/ground split) for p-median, p-center, MCLP, set-covering and the NSGA-II knee | Grouped bar chart |
| **Fig. 13** | Equity decomposition | Access time by income cluster for each solution family — the figure that shows whose time each objective optimizes | Grouped bar / box |
| **Fig. 14** | Sensitivity analysis | One-factor sweeps of the seven parameters in §5.6, response = worst-group access time and total cost; tornado summary panel | Line plots + tornado |

## Consolidated Table list

| # | Short title | Description | Type |
|---|---|---|---|
| **Table 1** | Gap matrix | Closest prior works × five criteria (income-stratified groups / equity as objective / multi-objective / rooftop-vs-ground classes / exact benchmarked optimization) — the evidence for novelty claims (A) and (B) | Table |
| **Table 2** | Data sources | Every dataset: source, native resolution/scale, date accessed, licence, derived product | Table |
| **Table 3** | OSM tag coverage | `height` and `building:levels` coverage by district, with the study-area range | Table |
| **Table 4** | Notation | Symbol, meaning, unit, config key (include only if the venue permits) | Table |
| **Table 5** | Candidate counts | Candidates surviving each filter stage, rooftop and ground separately, with the % dropped at each step | Table |
| **Table 6** | Cost and capacity parameters | Fixed cost, area-scaling cost, land cost, capacity for rooftop vs ground classes, with the source or an "illustrative" label | Table |
| **Table 7** | K-selection diagnostics | Silhouette, within-cluster SSE and cluster sizes for $K = 2\dots10$; selected $K$ marked | Table |
| **Table 8** | Cluster profiles | Per cluster: mean RWI, POI density, population, $\tau_k$, total daily demand, share of study-area demand | Table |
| **Table 9** | IP scenario results | Both scenarios: sites, class split, cost, demand/population coverage, core coverage, worst-group access time, solve time, optimality status; plus a row for `enforce_capacity = False` | Table |
| **Table 10** | NSGA-II configuration | Population, generations, operators and their parameters, runs, seed, wall-clock time, cache hit rate | Table |
| **Table 11** | Pareto front quality | HV, spacing, spread, IGD — mean ± sd over 10 runs; state that IGD is relative to the merged reference front | Table |
| **Table 12** | **Central comparison** | All solution families (4 baselines + NSGA-II knee, plus both IP scenarios) × all decision metrics — the paper's key quantitative table | Table |
| **Table 13** | Equity decomposition | Income-group and income-quintile access times, gaps, and Gini per solution family | Table |
| **Table 14** | Sensitivity results | Parameter × response matrix; a column flagging whether the solution ranking changed | Table |

---

## Evidence needed

Every item below is a concrete deliverable from `evtol_siting/`. The paper cannot be written until these exist. Items marked **BLOCKING** are required before any submission; the rest are required before the stated section can be finalized.

**Data and pipeline**
- **E1 (BLOCKING)** — Study-area statistics: total area (km²) within the boundary, number of 500 m grid cells, population within the study area. → §3.1 placeholder.
- **E2 (BLOCKING)** — Building count in the study area, and the measured OSM tag coverage: % of buildings with `height`, % with `building:levels`, % with neither, over the whole study area **and** per district (as a range, e.g. "1.5–5.4 %"). Must record the OSM extract date. → §3.3, Table 3, Fig. 3.
- **E3 (BLOCKING)** — Height-fusion output: fused height for every building, `height_source` composition, and `height_sigma_m`. Plus the counterfactual: rooftop candidate count under **tag-only** vs **full cascade**. → §5.1, Fig. 4.
- **E4** — Fused height validation: compare cascade heights against a manually checked sample of buildings (a few hundred) or against the raster source where it is independent. Report MAE/bias by source level. → §5.1. **This is the single most valuable additional experiment**, because it is what makes the height-fusion contribution defensible.
- **E5 (BLOCKING)** — Candidate counts at each filter stage, rooftop and ground separately; the % excluded by safety buffers and by dilution; candidate cost and capacity distributions. → §5.1, Table 5, Fig. 5, Fig. 6.

**Demand**
- **E6 (BLOCKING)** — Silhouette / SSE vs $K$ for $K=2\dots10$; the selected $K$; cluster profiles (mean RWI, POI density, population, $\tau_k$, demand); correlation between RWI and POI density. → §5.2, Table 7, Table 8, Fig. 7.
- **E7** — Correlation between cluster income rank and demand density, and the spatial concentration of the top-income cluster (e.g. its share of demand vs its share of area). → §5.2, the finding that motivates the equity objective.

**Optimization — Stage 3**
- **E8 (BLOCKING)** — Both IP scenarios solved: sites, class split, cost, coverage, solve time, optimality status; plus the `enforce_capacity = False` counterfactual showing the giant-vertiport degeneracy. → §5.3, Table 9, Fig. 9.

**Optimization — Stage 4 and evaluation**
- **E9 (BLOCKING)** — 10 NSGA-II runs: Pareto fronts, HV / spacing / spread / IGD as mean ± sd, wall-clock time, cache hit rate. Must state the reference point convention. → §5.4, Table 11.
- **E10 (BLOCKING)** — Knee point (and TOPSIS ranking) selected; the knee-point layout and its four objective values; the cost–inequity trade-off slope along the front. → §5.4, Fig. 10, Fig. 11.
- **E11 (BLOCKING)** — All four baselines solved exactly with optimality status confirmed, then evaluated on **all** decision metrics; the full comparison table including rooftop/ground composition. → §5.5, Table 12, Fig. 12, Fig. 13. **If any baseline fails to solve to optimality, say so and report the gap — do not present a heuristic solution as an exact one.**
- **E12 (BLOCKING)** — Equity decomposition: per-group and per-quintile access times, gaps and Gini for every solution family. → §5.5, Table 13.

**Sensitivity**
- **E13** — Seven one-factor sweeps per §5.6, each reporting decision metrics and whether the solution-family ranking changed. The $\rho_{up} = 1$ run is mandatory (it is the falsification test for the paper's central claim). → §5.6, Table 14, Fig. 14.
- **E14** — Weighted-sum scalarization sweep compared against the NSGA-II front (dominance check). → §5.6. Optional but pre-empts the most likely methodological objection.
- **E15** — Network-based access time using the OSM road network, at least as a robustness check on the Euclidean version. → §6.4 limitation, or §5.6 if it materially changes the equity finding. **The results of this experiment determine whether the CEUS/JTG stretch targets are reachable at all.**

**Reproducibility**
- **E16 (BLOCKING)** — Fixed global random seed, pinned dependency versions, and a statement of the exact OSM extract and RWI/WorldPop/Copernicus versions used. → §3.2, data availability statement.

---

## Reviewer risk register

| # | Likely objection | Severity | Concrete mitigation |
|---|---|---|---|
| **R1** | **"The equity objective is arbitrary — you picked one functional form."** | High | Pre-empt in §4.4: enumerate the alternatives (group-mean Gini, max-min ratio, Rawlsian maximin), justify the maximin choice on policy grounds, and **report the Gini and max-min ratio as diagnostics for every solution** (Table 13) so the reader can check that the conclusions do not depend on the chosen form. If the conclusions *do* depend on it, say so. |
| **R2** | **"Why NSGA-II rather than a scalarized weighted sum? Four objectives do not require a metaheuristic."** | High | Two-part answer, both backed by experiments. (1) Show the weighted-sum sweep (E14) and demonstrate that its solutions are dominated by or interior to the NSGA-II front — the standard non-convexity argument, but demonstrated rather than asserted. (2) Note that the problem is a set-covering problem, hence NP-hard, so an exact multi-objective solver (ε-constraint) does not scale to `[RESULT: |J|]` candidates × 4 objectives; report the ε-constraint attempt and where it fails. |
| **R3** | **"Income data are synthetic/absent — RWI is not income."** | High | Concede immediately and fully in §3.5 and §6.4. Reframe every affected claim as being about **relative wealth**, never income. Then add the falsification test: the $\rho_{up} = 1$ sensitivity run (E13) shows whether the equity gap is demand-driven or purely spatial. This converts the weakness into a robustness result. |
| **R4** | **"Ground candidates are grid-cell centroids, not real parcels — the ground cost is fiction."** | High | State it in §4.1 as a deliberate simplification with a stated meaning ("a site of ≥ 1600 m² exists in this cell"), not as an oversight. Then show that the headline conclusions are robust to the ground cost being varied ±50 % (E13), and that a purely rooftop-constrained rerun produces the same qualitative equity finding. |
| **R5** | **"Euclidean access time, not network distance — the accessibility numbers are wrong, and the bias runs in the direction of your own claim."** | High | This is the sharpest technical objection, because straight-line distance **overstates** accessibility in the low-income periphery and therefore **understates** the inequity the paper reports — i.e. the finding survives, but only if stated. Run E15 (network-based access on the OSM road graph) and report both. If the network version widens the gap, say so and use it; if it narrows it, report that too and revise the magnitude of the claim. Do not leave this to future work if the target is JTG/CEUS. |
| **R6** | **"One case study — the transferability claim is unsupported."** | Medium | Reframe §6.3 from an empirical claim to a stated precondition: identify the structural conditions under which the finding should replicate (spatially concentrated high-wealth demand + a demand model whose rates rise with wealth) and state plainly that this is an argument. Then add at least one **partial** generalization test: rerun the pipeline on a second Chinese city with different built form using the same code path, and report whether the qualitative equity gap appears. Even a coarse second case defuses this substantially. |
| **R7** | **"You never validate the demand model against observed vertiport demand — the whole thing rests on $\tau_{base}$ and $\rho_{up}$."** | High | Concede that no commercial vertiport exists in Chengdu, so no ridership validation is possible — this is a limitation of the field, not of the paper. Then do two things: (a) state that only the *relative* structure of demand enters the conclusions, and show that the rank ordering of solutions is invariant to $\tau_{base}$; (b) compare the spatial demand pattern qualitatively against Qu et al. 2024 (the one existing Chengdu UAM demand model) — **acquire this paper first**. |
| **R8** | **"The capacity constraint is an upper-bound approximation, not a real capacity constraint."** | Medium | The paper already knows this — say it in §4.3 before the reviewer does, and explain precisely what the constraint does (site-admissibility screen that eliminates giant-vertiport degeneracies) and what it does not (load redistribution). Then run the mitigation: add explicit assignment variables $x_{ij}$ for the top-decile demand cells and show the layouts are unchanged, or report how they change. |
| **R9** | **"The 20 m / 1600 m² thresholds are asserted, not derived."** | Medium | Cite the specific vertiport design standard that motivates them `[CITATION NEEDED]` and show the candidate set's sensitivity across the 1400–2500 m² range (E13, Fig. 14). If no standard supports 20 m specifically, say the threshold is a planning assumption and show the layout ranking is stable across 15–30 m. |
| **R10** | **"Rooftop and ground vertiports differ in more than cost — you have not modelled the operational differences (approach paths, obstacle clearance, wind, noise)."** | Medium | Concede the scope limit explicitly in §4.1 and §6.4: this paper models the *siting economics* of the two classes, not their airspace operations. Note that rooftop sites are subject to obstacle-limited approach surfaces that a full study would model, and cite it as the most important extension. Do **not** claim the model captures operational feasibility. |

---

## Key references to cite

Complete where known; `[CITATION NEEDED: …]` marks what must be retrieved before submission. **Bold** = load-bearing for novelty claims.

### Core method
| Ref | Citation | Where used |
|---|---|---|
| Deb et al. 2002 | Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. | §4.4 |
| Hakimi 1964 | Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. `[CITATION NEEDED: confirm journal, volume, pages — standard p-median reference]` | §4.5 |
| Church & ReVelle 1974 | Church, R., ReVelle, C. (1974). The maximal covering location problem. `[CITATION NEEDED: confirm journal, volume, pages — standard MCLP reference]` | §4.5 |

### Equity in facility location
| Ref | Citation | Where used |
|---|---|---|
| Xu, Murray, Church & Wei 2023 | Xu, J., Murray, A. T., Church, R. L., Wei, R. (2023). Service allocation equity in location coverage analytics. *European Journal of Operational Research*. `[CITATION NEEDED: volume, issue, pages]` | §2.3, §4.4 |
| **Gao et al. 2024** | Gao, et al. (2024). Noise-aware equitable urban air traffic management. *Transportation Research Part C: Emerging Technologies*. `[CITATION NEEDED: author list, volume, article number]` | §2.3 — the closest equity-in-UAM work; equity is on the **noise** axis, not income access |

### Vertiport siting and UAM
| Ref | Citation | Where used |
|---|---|---|
| **Lu et al. 2025** | Lu, et al. (2025). *Aerospace*, 12(8), 709. `[CITATION NEEDED: full author list and exact title — Shenzhen vertiport siting]` | §2.1, §2.4 — **the closest prior work**; equity in the screening stage, single-objective optimization |
| Jeong et al. 2021 | Jeong, et al. (2021). *Applied Sciences*, 11(12), 5729. `[CITATION NEEDED: full author list and exact title]` | §2.1 — GIS/MCDM siting paradigm |
| **Fadhil 2018** | Fadhil, D. N. (2018). *A GIS-based analysis for selecting ground infrastructure locations for urban air mobility* (Master's thesis). Technical University of Munich. | §2.4 — income used **regressively** (high income = more suitable); the foil for novelty (A) |
| Yoon et al. 2025 | Yoon, et al. (2025). arXiv:2502.00399. `[CITATION NEEDED: full author list and title]` | §2.2 |
| Wu & Zhang 2021 | Wu, Z., Zhang, Y. (2021). *Engineering*, 7(4), 473–487. `[CITATION NEEDED: confirm exact title]` | §2.2 |
| NASA vertiport tiers | `[CITATION NEEDED: NASA UAM vertiport tier definition — Vertihub / Vertiport / Vertistop — technical report or advisory circular]` | §2.4, gap (B) |
| Guo et al. tiers | `[CITATION NEEDED: Guo et al. — Vertipad / Vertibase / Vertihub — full citation]` | §2.4, gap (B) |
| Vertiport design standard | `[CITATION NEEDED: design standard supporting the 20 m height and 1600 m² roof thresholds — EASA PTS-VPT-DSN, FAA Engineering Brief 105, or the applicable Chinese standard]` | §4.1, R9 |
| Vertiport construction cost basis | `[CITATION NEEDED: source for the cost parameters in §4.4, or reclassify as illustrative]` | §4.4, Table 6, R10 |
| **Qu, Huang, Li et al. 2024** | Qu, Huang, Li, et al. (2024). *Green Energy and Intelligent Transportation*, 3(3), 100173. `[CITATION NEEDED: full author list and exact title]` | §2.2, §5.2 — **the one existing Chengdu UAM demand model. MUST ACQUIRE.** The paper must state whether its demand model is calibrated against this or is an independent construction. |

### Chinese-language literature
| Ref | Citation | Where used |
|---|---|---|
| 李卓伦/陆建 et al. 2026 | 李卓伦, 陆建, et al. (2026). 交通运输工程学报, 26(3), 89–105. `[CITATION NEEDED: exact title]` | §2.1, §2.2 |
| 姜雨 et al. 2026 | 姜雨, et al. (2026). 交通运输工程学报. `[CITATION NEEDED: exact title, volume, issue, pages]` | §2.2 (review) |
| 党庆庆 et al. | 党庆庆, et al. 北京航空航天大学学报 (vertiport review). `[CITATION NEEDED: exact title, year, volume, issue, pages]` | §2.2 |
| 尹浩东 et al. 2025 | 尹浩东, et al. (2025). 交通运输工程与信息学报, 23(3), 88–102. `[CITATION NEEDED: exact title]` | §2.2 |

### Policy and data
| Ref | Citation | Where used |
|---|---|---|
| Chengdu low-altitude policy | `[CITATION NEEDED: Chengdu / Sichuan low-altitude economy development plan or UAM pilot policy document — exact title, issuing body, year]` | §1.1, §6.1 |
| Chengdu statistics | `[CITATION NEEDED: Chengdu Statistical Yearbook, latest edition — state the year used]` | §3.1 |
| Meta RWI | `[CITATION NEEDED: Meta Relative Wealth Index — data description and the paper introducing it]` | §3.2 |
| WorldPop | `[CITATION NEEDED: WorldPop dataset citation for China, with version and access date]` | §3.2 |
| Copernicus DEM | `[CITATION NEEDED: Copernicus GLO-30 DEM product citation]` | §3.2 |
| Building-height raster | `[CITATION NEEDED: CNBH-10m and/or GlobalBuildingAtlas product paper — whichever was actually used]` | §3.2, §3.4 |
| OpenStreetMap | `[CITATION NEEDED: OSM citation, with the extract date and licence (ODbL)]` | §3.2 |

---

*End of framework. Next action: work the Evidence needed checklist top to bottom; E2, E3 and E5 are the cheapest blocking items and they determine whether the height-fusion contribution (secondary contribution 3) is real.*
