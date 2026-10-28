# Maritime Fuel Efficiency Optimization — Prototype

**Indian Coast Guard — Problem Statement 77: "Fuel consumption optimization using
AI-based tools."**

A working prototype that **predicts a ship's fuel-burn rate** from static (vessel
design) + dynamic (voyage) data, then uses a **quantum-inspired optimizer**
(simulated annealing) to recommend the **fuel-minimizing speed that still meets
the arrival deadline** — i.e. data-driven *slow steaming*. CO₂ savings are
reported too, for IMO/environmental compliance.


---

> **Status (2026-10-28).** Phases 1-3 run end to end: fuel-rate prediction,
> QUBO/annealing/gate-model optimizers for cruise speed and weather routing, an
> honest multi-solver benchmark, and a Streamlit dashboard. A fleet-level extension
> (multi-ship mission scheduling: ALNS, a genetic algorithm and a quantum-classical
> hybrid against an exact optimum, including a run on real IBM quantum hardware)
> lives in [`fleet/`](fleet/) — see [`fleet/README.md`](fleet/README.md). Every
> result below is from the full run [`runs/2026-10-28_0231/`](runs/2026-10-28_0231/);
> what changed and why is in [What changed](#what-changed-on-2026-10-28) and
> [`journal.md`](journal.md).

---

## Quick start

```bash
conda create -n quantum-fuel python=3.11
conda activate quantum-fuel
conda env config vars set PYTHONNOUSERSITE=1    # keep ~/.local packages out (re-activate after)
pip install -r requirements.txt
# NVIDIA GPU (optional, Linux): pip uninstall -y qiskit-aer && pip install -r requirements-gpu.txt

python run.py test                 # the smoke test, ~1.5 min, single-threaded: laptop-safe
python run.py demo                 # the whole Phase 1 pipeline + a plot
python run.py dashboard            # the interactive app (Phase 3), opens on port 8501
```

`python run.py` with no argument lists everything that can be run. It is only a
launcher: `python demo.py`, `python -m experiments.qubo_route` and
`cd fleet; python run_fleet.py` still work as before.

```bash
python run.py generate_data --n 10000 --seed 42       # -> data/ship_fuel_data.csv
python run.py train_model                             # -> models/ship_fuel_model.joblib
python run.py optimizer --distance 600 --deadline 80 --waves 4 --wind 30
python run.py qubo_route --rows 5 --cols 7
python run.py run_fleet --seed 0
```

**The full run.** `run_all.sh` regenerates every result in this README and puts
it in a NEW folder, `runs/<date>_<time>/`, so it never overwrites an earlier run.
The folder has the log, a manifest (package versions, CPU/GPU, settings), the time
and exit code of every step, every plot, `results/*.json`, the data and model the
run used, and `RESULTS.md` with every table. The tables below are generated from
one such folder by `tools/make_report.py`; each says which.

```bash
FUEL_JOBS=28 bash run_all.sh                          # cores to use; -1 = all
python run.py report --update-readme                  # refresh the tables below
```

Scripts run on their own write to `outputs/` (and `fleet/outputs/`). The results
of the very first version of the project are kept unchanged in
[`outputs_archive/original_2026-10-06/`](outputs_archive/original_2026-10-06/).

**Hardware.** Training and the benchmarks use `FUEL_JOBS` cores, default **2** --
laptop-safe on purpose. QAOA/VQE simulate on the GPU from 18 qubits when the GPU
build of Aer is installed (`FUEL_QDEVICE=CPU` turns that off); nothing else uses
a GPU.

---

## Why ships are different from cars (the modelling insight)

- **Cubic speed–power law.** A ship's fuel *rate* (tonnes/day) grows with the
  **cube of speed** (`power ~ displacement^(2/3) × speed³`). So minimizing fuel
  alone trivially says "go as slow as possible".
- **The real problem is constrained.** Operationally you must **arrive on time**.
  So we minimize *total voyage fuel* `= rate(speed) × distance / (24 × speed)`
  subject to a **deadline** (`speed ≥ distance / deadline`). Because the ship
  also burns a speed-independent **hotel/auxiliary load**, the total-fuel curve
  is **U-shaped** — there is an "economic speed". This is exactly the real-world
  practice of **slow steaming**.
- **Environment matters.** Sea state (wave height), wind, current
  (speed-through-water vs over-ground), and load condition all change fuel burn.

---

## What's in here

**Main pipeline -- project root, run directly** (`python <file>` or `python run.py <name>`):

| File | Role | Report § |
|------|------|----------|
| `config.py` | Feature schema, ship classes, voyage + emission constants, `FUEL_JOBS`, output folder | §3 |
| `generate_data.py` | Physics-based ship-voyage simulator (cubic law) + the true-physics oracle | §4 |
| `train_model.py` | Linear / Random Forest / Gradient Boosting / XGBoost / LightGBM / grey-box + metrics | §6 |
| `optimizer.py` | Simulated-annealing slow-steaming optimizer (+ CO₂, + decision regret) | §7 |
| `demo.py` | One-command end-to-end demonstration | §12 |
| `run.py` | Launcher for everything below (`python run.py` lists it) | |
| `run_all.sh` | The full run, into `runs/<date>_<time>/` | |

**`core/` -- shared components (imported, never run):**

| File | Role |
|------|------|
| `physics_model.py` | Grey-box model: Admiralty-law form + boosting on the residual |
| `results.py` | Every script's numbers as `results/*.json` next to its plots |
| `stats.py` | Wilson / bootstrap intervals, time-to-solution (TTS99), paired tests |

**`solvers/` -- optimizer algorithms (imported, not run):**

| File | Role |
|------|------|
| `backends.py` | QUBO solver swap point: `neal` / `dwave` (real QPU) / `tabu` / `gsa` / `ga` |
| `gate.py` | Gate-model plumbing: QUBO → Qiskit, Aer sampler on CPU/GPU, CVaR, sample decoding |
| `gsa.py` | **Phase 2d:** Gravitational Search Algorithm (continuous + binary) |
| `qgso.py` | **Phase 2e:** QGSO -- quantum-inspired GSA (qubit rotation-gate update) |
| `ga.py` | Genetic Algorithm (continuous + binary), same interface as `gsa.py` |

**`experiments/` -- optional studies** (`python run.py <name>`):

| File | Role |
|------|------|
| `qubo_speed.py` | **Phase 2a:** speed optimization as a QUBO (pyqubo + neal) |
| `qubo_route.py` | **Phase 2b:** weather routing as a QUBO (the headline); `--presolve` |
| `quantum_gate.py` | **Phase 2c:** the same QUBO on gate-model **QAOA + VQE**, with and without CVaR |
| `speed_compare.py` | **SA vs GA vs GSA** (vs QUBO vs grid) on cruise speed; `--legs N` for a speed per leg |
| `joint_speed_route.py` | **Phase 2+:** speed **and** route in one QUBO; `--rows 1` = multi-segment speed profile |
| `benchmark.py` | **Honest solver benchmark** -- fixed budget, several instances × many seeds, intervals, TTS99; `--gate` for QAOA/VQE |
| `audit_seeds.py` | Full seed-distribution audit (exposes any cherry-picking) |

**Elsewhere:** [`fleet/`](fleet/) (Phase 4-5, its own README) ·
[`dashboard/app.py`](dashboard/app.py) (Phase 3, Streamlit) ·
[`tools/`](tools/) (`make_report.py`: results tables; `manifest.py`: what a run ran on) ·
[`tests/test_smoke.py`](tests/test_smoke.py) (the light check) ·
[`docs/`](docs/) (`concepts.pdf` and its generator; covers Phases 1-2b) ·
[`journal.md`](journal.md) (decision history) ·
`runs/` (full runs) · `outputs/` (single-script runs) ·
[`outputs_archive/`](outputs_archive/) (the original results, untouched).

Ship classes modeled (ICG-style patrol fleet): **Interceptor**, **Fast Patrol**,
**Offshore Patrol Vessel**. Generated at runtime: `data/`, `models/`, `outputs/`, `runs/`.

---

## What changed on 2026-10-28

The first full run, on the Linux workstation, plus what the owner chose from the
list of proposed improvements. Every decision, alternative and measurement is in
[`journal.md`](journal.md).

- **Fleet: the hybrid's headline did not survive scale.** On 4 ships × 12
  missions (36-slot horizon, before the horizon fix below), the original hybrid
  found no feasible plan on 2 of 3 instances. Its pairwise QUBO rates
  overloaded ships *below* the true optimum. v2 adds lazy cuts (minimal
  infeasible bundles forbidden, then resample) and a stronger repair. It is
  significantly better than v1 at every size from 4 × 8. But from 4 × 10 the
  GA beats it, and from 5 × 14 ALNS does too (p < 0.05, 10 instances per size).
  Up to 16 missions the exact DP (at most 25 s) is the planner to use; beyond
  that, ALNS. Numbers in [`fleet/README.md`](fleet/README.md).
- **Fleet: a genetic algorithm** over assignments (`fleet/genetic.py`), scored
  exactly like the hybrid and the random ablation, at equal numbers of exact
  scorings.
- **Fleet: a scale study with exact ground truth to 16 missions.** The subset DP
  is far cheaper than estimated (0.5 s at 4 × 12, about 25 s at 5 × 16), but every
  instance past ~2 missions per ship used to be infeasible because all windows
  were packed into 36 hours; the horizon now grows with missions per ship (the
  default 3 × 6 instances are unchanged).
- **A bug that hid most QAOA/VQE samples.** qiskit-optimization squared the
  probabilities qiskit-algorithms returns and dropped everything below 1e-6 -- so
  only bitstrings measured twice or more were ever candidates (and at ~21 qubits
  none, and it crashed). On the 3 × 4 audit QUBO that is QAOA optimal 1/15 →
  3/15, feasible 9/15 → 13/15 on the same distributions. Every gate-model number
  from before is affected.
- **Gate model on Aer, on the GPU, with CVaR.** QAOA/VQE use Aer's C++
  simulator (GPU from 18 qubits: 22 qubits in 2.6 s instead of 121 s per 20
  iterations), so the benchmark reaches 21 qubits and single runs 26; each also
  runs with a CVaR objective.
- **How sure is each number:** 95% intervals, time-to-solution (TTS99) and
  paired tests on the same instances (`core/stats.py`).
- **Results are generated, not copied:** every script writes JSON, every full run
  gets its own `runs/<date>_<time>/` folder, `tools/make_report.py` builds the
  tables below from it.
- **Layout:** `run.py` runs anything from the root; `physics_model.py` moved to
  `core/`; `tools/` and `dashboard/` added; the duplicated old plots removed from
  `outputs/` (the archive keeps them).
- **Phase 3 dashboard:** `python run.py dashboard`.
- **Fleet table cache** is now keyed by the fuel model too (it would have served
  tables priced by an old model after retraining).

## What changed on 2026-10-07

Full reasoning, alternatives considered and what is still untested:
[`journal.md`](journal.md). In short:

- **The optimizer was optimising a tree model's artefacts.** Tree models are flat
  between splits, so total fuel = rate ÷ speed comes out as a *sawtooth* (visible
  in the original plot below) and the "optimal speed" is whichever tooth is
  lowest. From the generator's own formula, the original relaxed-schedule
  recommendation (8.4 kn, believed to cost 18.3 t) really costs **20.1 t**,
  while the true optimum is **6.67 kn at 18.4 t** — about **9% more fuel than
  necessary**. "SA matches the grid" only ever proved the optimizer agreed with
  the *model*.
- **Fix, at the model:** `physics_model.py` gives the speed response a smooth
  physical form (hotel load + a power law whose exponent is *fitted*) and lets
  boosting learn only the residual. The existing booster also gets a log target
  and monotonic constraints, and XGBoost / LightGBM join the comparison.
- **New metric — decision regret:** because the data is synthetic the true
  physics is known, so every model is now scored on how much fuel the optimizer
  *wastes* by trusting it, next to R². On a 3,000-row test fit the physics form
  recovered a speed exponent of **3.002** with **0.00%** regret; plain gradient
  boosting showed **5.2% mean / 10.6% worst-case**. (Small-sample test numbers,
  not the full 10,000-row run.)
- **Genetic algorithm** (`solvers/ga.py`) as a fourth optimizer family:
  `--backend ga` everywhere, a `ga` column in the benchmark.
- **Phase 2+ extensions:** multi-segment speed profile and joint speed + route
  in one QUBO (`experiments/joint_speed_route.py`), with the deadline handled as
  a *price on time* rather than slack bits, checked against an exact DP.
- **Presolve** (`--presolve`): fix the ports and drop unreachable cells before
  building the route QUBO — 3×4: 12 → 6 variables, 5×7: 35 → 21, 11×25: 275 → 213.
- **Benchmark:** several weather instances per size instead of one, budgets
  stated in evaluations (they were called "equal" but were not equal compute),
  `--jobs` for parallel runs.
- **Fleet:** exact ground truth by subset DP, good to ~12 missions instead of 6.
- **Fixes:** the shipped model could not be loaded by a current scikit-learn
  (now detected and retrained); `decode_route` did not check the ports; Tabu runs
  were unseeded; training data did not cover the slowest speeds the optimizer
  searched.

---

## Results

Every table and figure in this section is generated by `tools/make_report.py`
from one full run, [`runs/2026-10-28_0231/`](runs/2026-10-28_0231/) (its
`RESULTS.md` has them all in one place); the prose was written against that run.
The original, pre-2026-10-07 results are kept, untouched, in
[`outputs_archive/original_2026-10-06/`](outputs_archive/original_2026-10-06/).
Where a finding changed, the text says so.

<!-- results:run_info -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
Run `2026-10-28_0231` — started 2026-10-28T02:31:42 on user (Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 32 cores; GPU: NVIDIA RTX A6000, 49140 MiB, 570.144), FUEL_JOBS=28. Packages: numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.2.0, lightgbm 4.7.0, pyqubo 1.5.0, dwave-neal 0.6.0, dwave-samplers 1.8.0, qiskit 2.5.2, qiskit-aer-gpu 0.18.2, qiskit-algorithms 0.4.0, qiskit-optimization 0.7.0, matplotlib 3.11.2, streamlit 1.66.1. 34 steps, 2.0 h in total; all steps succeeded.
<!-- /results:run_info -->

## Phase 1 — fuel-rate models and slow steaming

**Model comparison** — predicting fuel rate (t/day), 10,000 synthetic voyage legs:

<!-- results:phase1_models -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
| Model | CV R² | Test R² | MAE | RMSE | MAPE | Regret mean | Regret max |
|---|---|---|---|---|---|---|---|
| **Physics + Boosting** | 0.994 | 0.994 | 0.81 | 1.63 | 4.9% | 0.00% | 0.05% |
| Gradient Boosting | 0.986 | 0.989 | 1.21 | 2.31 | 11.9% | 3.50% | 9.62% |
| LightGBM | 0.986 | 0.988 | 1.25 | 2.38 | 12.1% | 4.67% | 12.44% |
| XGBoost | 0.983 | 0.984 | 1.38 | 2.70 | 11.5% | 2.06% | 13.53% |
| Random Forest | 0.977 | 0.981 | 1.48 | 3.00 | 9.2% | 1.47% | 6.10% |
| Gradient Boosting (log, monotone) | 0.979 | 0.979 | 1.44 | 3.16 | 8.7% | 2.70% | 6.51% |
| Linear Regression | 0.630 | 0.638 | 9.12 | 13.00 | 391.9% | 116.65% | 519.10% |

10,000 rows, 20% held-out test + 5-fold CV. Regret = extra TRUE fuel burned at the speed each model recommends, over 12 voyages (`optimizer.regret_panel`). Saved model: **Physics + Boosting** (best test R²).
Fitted physics: hotel = 0.01949, prop_scale = 4.984e-05, speed_exponent = 3.002, hull_exponent = 1.03, wave_coef = 0.04912, wind_coef = 0.004809, laden_coef = 0.1136.
<!-- /results:phase1_models -->

- **The grey box wins on every column** -- test R² 0.994, MAPE 4.9% -- and reads
  the cubic law back out of noisy data: fitted speed exponent **3.002** (true: 3).
  *Caveat, also in `core/physics_model.py`:* the generator has the same "hotel load
  + power law" form, so this shows correct calibration, not discovery. The
  black-box rows are the test of learning the law from scratch.
- **Decision regret is the column that matters, and R² does not predict it.** The
  boosters score R² 0.98-0.99 -- nearly as good as the grey box -- yet the speed
  they recommend burns **1.5-4.7% more true fuel on average and up to 13.5%** on
  one of the 12 test voyages. The ranking differs too: Random Forest has the lowest
  R² of the tree models and the lowest regret; XGBoost has the worst single voyage.
  The cause is the sawtooth: a tree model is flat between splits, so
  rate ÷ speed is a sawtooth and the "optimum" is whichever tooth is lowest.
- The linear baseline fails outright (R² 0.64): it cannot represent the cubic law
  across vessels from ~5 to ~90 t/day.

**Optimization** — Offshore Patrol Vessel, 600 nm leg, baseline = 18 kn service speed:

<!-- results:phase1_speed -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
| Scenario | Optimal speed | Fuel (model) | vs baseline | CO₂ saved | True optimum | Decision regret |
|---|---|---|---|---|---|---|
| Tight schedule (40 h) | 15.00 kn | 40.0 t | -26.3% | 45.9 t | 15.00 kn / 39.9 t | +0.00% |
| Relaxed schedule (90 h) -> slow steaming | 6.67 kn | 18.4 t | -66.1% | 115.2 t | 6.67 kn / 18.4 t | +0.00% |

![Total voyage fuel vs speed](runs/2026-10-28_0231/voyage_fuel_vs_speed.png)
<!-- /results:phase1_speed -->

Both recommendations are the true optimum (regret 0.00%). The relaxed case is
still **deadline-bound** at 6.67 kn: the true economic speed is 6.3 kn, which
needs ~95 h. *Changed:* the original run recommended 8.4 kn here, believed to cost
18.3 t; by the true physics it costs 20.1 t -- the sawtooth artefact, about 9% of
fuel wasted.

## Phase 2 — the same decisions as QUBOs

The optimization is reformulated as a **QUBO** (Quadratic Unconstrained Binary
Optimization) — the exact Ising/binary form a **D-Wave quantum annealer** solves
natively — and solved classically with `neal`. The *same QUBO* could be submitted
to real quantum hardware unchanged; that portability is what makes it
"quantum-inspired". Constraints (pick one option, meet the deadline, no illegal
moves) are encoded as **penalty terms**, the standard QUBO technique.

### Phase 2a — speed as a QUBO (`python run.py qubo_speed`)

<!-- results:speed_qubo -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
| Deadline | Levels | QUBO (neal) | Simulated annealing | Grid (exact) | One-hot valid |
|---|---|---|---|---|---|
| 40 h | 40 | 15.31 kn / 41.39 t | 15.00 kn / 40.04 t | 15.00 kn / 40.04 t | yes |
| 90 h | 40 | 6.69 kn / 18.45 t | 6.67 kn / 18.44 t | 6.67 kn / 18.44 t | yes |
<!-- /results:speed_qubo -->

One-hot over 40 speed levels, 0.53 kn apart. At 40 h the first level above the
15.0 kn deadline floor is 15.31 kn, which costs 3.4% more fuel than the continuous
optimum -- the price of discretisation, not a solver error (at 90 h it is 0.05%).

### Phase 2b — weather routing as a QUBO (`python run.py qubo_route`)

A ship crosses a weather field; each grid cell's fuel cost comes from the **Phase 1
model**. The QUBO picks the minimum-fuel path *around* the storms; an exact DP
(the monotone grid path is a shortest-path problem) gives the true optimum.

<!-- results:route_qubo -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
| Grid | Storms | Seed | Presolve | Vars | Naive (t) | QUBO (t) | Exact DP (t) | Saved vs naive | QUBO vs exact | Decision regret |
|---|---|---|---|---|---|---|---|---|---|---|
| 5x7 | 1 centred | 0 | yes | 21 | 53.01 | 48.28 | 48.28 | +8.9% | +0.00% | +0.00% |
| 5x7 | 1 centred | 0 | no | 35 | 53.01 | 48.28 | 48.28 | +8.9% | +0.00% | +0.00% |
| 7x15 | 3 | 1 | no | 105 | 50.52 | 47.07 | 47.07 | +6.8% | +0.00% | +0.00% |
| 9x19 | 4 | 3 | no | 171 | 48.40 | 46.16 | 45.48 | +4.6% | +1.49% | +1.56% |
| 11x25 | 6 | 5 | yes | 213 | 49.38 | 45.02 | 44.38 | +8.8% | +1.44% | +1.43% |
| 11x25 | 6 | 5 | no | 275 | 49.38 | 44.82 | 44.38 | +9.2% | +0.99% | +0.96% |

![Weather routing QUBO](runs/2026-10-28_0231/route_qubo_5x7_s0_seed0_r2-2.png)
<!-- /results:route_qubo -->

- **Routing saves 5-10% against the straight line** (exact optimum vs naive).
- **One neal run is exact up to 105 variables and 1-1.5% off from 171.** The
  original README called neal "100% reliable" at 35-275 variables; for a single
  run at the default budget that is not true. The benchmark below gives the
  distribution.
- Decision regret of the chosen routes equals their gap to the exact optimum: the
  model's cell costs and the true physics agree.

**Presolve** (`--presolve`) substitutes what is already known before the QUBO is
built: the two port cells are fixed, and a cell more rows from a port than it is
columns away can never be on a legal path. Same Hamiltonian, same optimum
(checked by exhaustive enumeration in the tests), fewer variables: 3×4 from 12 to
6, 5×7 from 35 to 21, 11×25 from 275 to 213. Its effect on reliability is measured
in the benchmark below.

### Swappable solver backends (`solvers/backends.py`)

The same QUBO can be handed to different solvers with `--backend`:

| Backend | What it is | Needs |
|---------|-----------|-------|
| `neal` (default) | Classical simulated annealing | nothing — offline |
| `tabu` | Classical Tabu search | nothing — offline |
| `gsa` | Binary Gravitational Search (swarm) | nothing — offline |
| `ga` | Binary Genetic Algorithm (evolutionary) | nothing — offline |
| `dwave` | **Real D-Wave quantum annealer** via Leap | a Leap API token |

```bash
python run.py qubo_route --rows 5 --cols 7 --backend tabu
python run.py qubo_route --rows 5 --cols 7 --backend ga
python run.py qubo_route --rows 5 --cols 7 --backend dwave   # needs a Leap token
```

> **Note on `dwave`:** D-Wave's free *Trial* plan now allows demos only — it no
> longer issues API tokens for submitting your own problems. Running our QUBO on a
> real annealer needs a paid/commercial or academic Leap plan. The code path is
> ready; it's a one-flag switch if/when access is granted.

### Phase 2c — the same QUBO on gate-model quantum (QAOA + VQE, and CVaR)

`quantum_gate.py` converts the QUBO to an Ising Hamiltonian and solves it with
**QAOA** and **VQE**, the two standard variational algorithms for IBM-style
gate-model machines, on Aer's statevector simulator (the GPU from 18 qubits, so
21-26 qubits are in reach). Each also runs with a **CVaR** objective (Barkoutsos et
al. 2020): the optimiser is scored on the best 25% of shots instead of the mean,
which rewards making the *good* tail likely -- what a sampler we decode by "best
bitstring" needs.

> **A decoding bug, fixed 2026-10-28, affects every gate-model number published
> before.** qiskit-optimization squared the measured probabilities it received
> (it expected amplitudes) and dropped everything below 1e-6, so a bitstring
> measured once in 1,024 shots was never a candidate. Only ~25% of samples were
> used; at ~21 qubits none, and it crashed. On the 15-seed audit QUBO the same
> QAOA distributions decode to 3/15 optimal and 13/15 feasible instead of 1/15 and
> 9/15. Details in [`journal.md`](journal.md).

```bash
python run.py quantum_gate --problem speed --levels 8                                     # 8 qubits
python run.py quantum_gate --problem route --rows 3 --cols 4 --storms 1 --seed 2          # 12 qubits
python run.py quantum_gate --problem route --rows 5 --cols 8 --storms 1 --seed 2 --presolve  # 26 qubits
```

<!-- results:quantum_gate -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
**ROUTE - 3x4 grid, 1 storm(s)** — 12 qubits, Aer on CPU, 5 trials

| Method | Best fuel (t) | Gap | Feasible | Hit optimum | Time/run |
|---|---|---|---|---|---|
| Exact (DP) | 53.97 | – | 1/1 | 1/1 | 0.00 s |
| neal (anneal) | 53.97 | +0.0% | 5/5 | 5/5 | 0.03 s |
| GSA (gravity) | 53.97 | +0.0% | 5/5 | 5/5 | 2.09 s |
| GA (genetic) | 53.97 | +0.0% | 5/5 | 5/5 | 0.36 s |
| QAOA | 53.97 | +0.0% | 4/5 | 1/5 | 2.74 s |
| QAOA-CVaR(0.25) | 53.97 | +0.0% | 5/5 | 1/5 | 2.62 s |
| VQE | 53.97 | +0.0% | 5/5 | 4/5 | 6.68 s |
| VQE-CVaR(0.25) | 53.97 | +0.0% | 5/5 | 5/5 | 4.82 s |

![ROUTE - 3x4 grid, 1 storm(s)](runs/2026-10-28_0231/compare_route_3x4_s1_seed2_q42_t5.png)


**ROUTE - 4x6 grid, 1 storm(s)** — 14 qubits, Aer on CPU, 5 trials

| Method | Best fuel (t) | Gap | Feasible | Hit optimum | Time/run |
|---|---|---|---|---|---|
| Exact (DP) | 45.68 | – | 1/1 | 1/1 | 0.00 s |
| neal (anneal) | 45.68 | +0.0% | 5/5 | 5/5 | 0.03 s |
| GSA (gravity) | 45.68 | +0.0% | 5/5 | 5/5 | 2.17 s |
| GA (genetic) | 45.68 | +0.0% | 5/5 | 5/5 | 0.38 s |
| QAOA | 48.00 | +5.1% | 5/5 | 0/5 | 3.76 s |
| QAOA-CVaR(0.25) | 46.36 | +1.5% | 5/5 | 0/5 | 3.73 s |
| VQE | 45.68 | +0.0% | 5/5 | 1/5 | 7.28 s |
| VQE-CVaR(0.25) | 45.68 | +0.0% | 5/5 | 3/5 | 7.24 s |

![ROUTE - 4x6 grid, 1 storm(s)](runs/2026-10-28_0231/compare_route_4x6_s1_seed2_q42_t5_presolve.png)


**ROUTE - 5x7 grid, 1 storm(s)** — 21 qubits, Aer on GPU, 3 trials

| Method | Best fuel (t) | Gap | Feasible | Hit optimum | Time/run |
|---|---|---|---|---|---|
| Exact (DP) | 45.76 | – | 1/1 | 1/1 | 0.00 s |
| neal (anneal) | 45.76 | +0.0% | 3/3 | 3/3 | 0.05 s |
| GSA (gravity) | 45.76 | +0.0% | 3/3 | 3/3 | 4.04 s |
| GA (genetic) | 45.76 | +0.0% | 3/3 | 3/3 | 0.74 s |
| QAOA | – | – | 0/3 | 0/3 | 4.91 s |
| QAOA-CVaR(0.25) | – | – | 0/3 | 0/3 | 4.75 s |
| VQE | 45.76 | +0.0% | 3/3 | 1/3 | 12.40 s |
| VQE-CVaR(0.25) | 45.76 | +0.0% | 3/3 | 2/3 | 12.48 s |

![ROUTE - 5x7 grid, 1 storm(s)](runs/2026-10-28_0231/compare_route_5x7_s1_seed2_q42_t3_presolve.png)


**ROUTE - 5x8 grid, 1 storm(s)** — 26 qubits, Aer on GPU, 2 trials

| Method | Best fuel (t) | Gap | Feasible | Hit optimum | Time/run |
|---|---|---|---|---|---|
| Exact (DP) | 45.24 | – | 1/1 | 1/1 | 0.00 s |
| neal (anneal) | 45.24 | +0.0% | 2/2 | 1/2 | 0.06 s |
| GSA (gravity) | 46.66 | +3.1% | 2/2 | 0/2 | 12.43 s |
| GA (genetic) | 45.24 | +0.0% | 2/2 | 2/2 | 1.36 s |
| QAOA | – | – | 0/2 | 0/2 | 46.99 s |
| QAOA-CVaR(0.25) | – | – | 0/2 | 0/2 | 52.97 s |
| VQE | 46.84 | +3.5% | 2/2 | 0/2 | 106.23 s |
| VQE-CVaR(0.25) | 45.91 | +1.5% | 2/2 | 0/2 | 105.33 s |

![ROUTE - 5x8 grid, 1 storm(s)](runs/2026-10-28_0231/compare_route_5x8_s1_seed2_q42_t2_presolve.png)


**SPEED - 8 levels, 600 nm, 40 h deadline** — 8 qubits, Aer on CPU, 5 trials

| Method | Best fuel (t) | Gap | Feasible | Hit optimum | Time/run |
|---|---|---|---|---|---|
| Exact (scan) | 44.53 | – | 1/1 | 1/1 | 0.00 s |
| neal (anneal) | 44.53 | +0.0% | 5/5 | 5/5 | 0.03 s |
| GSA (gravity) | 44.53 | +0.0% | 5/5 | 5/5 | 1.61 s |
| GA (genetic) | 44.53 | +0.0% | 5/5 | 5/5 | 0.27 s |
| QAOA | 44.53 | +0.0% | 5/5 | 5/5 | 1.03 s |
| QAOA-CVaR(0.25) | 44.53 | +0.0% | 5/5 | 5/5 | 0.98 s |
| VQE | 44.53 | +0.0% | 5/5 | 5/5 | 4.64 s |
| VQE-CVaR(0.25) | 44.53 | +0.0% | 5/5 | 5/5 | 3.76 s |

![SPEED - 8 levels, 600 nm, 40 h deadline](runs/2026-10-28_0231/compare_speed_L8_q42_t5.png)
<!-- /results:quantum_gate -->

### Solver reliability — the full distribution over 15 seeds

<!-- results:audit -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
3×4 route QUBO, exact optimum 53.97 t, 15 seeds per solver, every run counted. CVaR α = 0.25.

| Solver | Hit optimum | 95% CI | Feasible | Mean gap | Worst gap |
|---|---|---|---|---|---|
| neal | 15/15 | 80%–100% | 15/15 | +0.0% | +0.0% |
| GSA | 15/15 | 80%–100% | 15/15 | +0.0% | +0.0% |
| GA | 15/15 | 80%–100% | 15/15 | +0.0% | +0.0% |
| QAOA | 2/15 | 4%–38% | 12/15 | +7.3% | +12.8% |
| QAOA-CVaR | 6/15 | 20%–64% | 15/15 | +4.4% | +12.5% |
| VQE | 7/15 | 25%–70% | 15/15 | +3.9% | +12.5% |
| VQE-CVaR | 15/15 | 80%–100% | 15/15 | +0.0% | +0.0% |
<!-- /results:audit -->

**Every gate-model claim rests on distributions like this one, not on single
runs.** On this 12-qubit route QUBO the classical solvers are optimal on every
seed. Plain QAOA is optimal 2/15 and VQE 7/15. **CVaR changes the picture for
VQE:** 15/15, the same seeds and starting points as plain VQE. For QAOA it helps
less (6/15, but feasible on every seed).

> **Correction / honesty note (kept on purpose).** Earlier versions of this
> section published a single `--qseed 11` diagram showing QAOA/VQE at "+13.5%".
> That was their **worst of 15 seeds**, cherry-picked to look dramatic, and it
> also gave GSA an unfair one-swarm budget. Both were corrected with
> full-distribution reporting. The lesson, enforced by `benchmark.py`: equal
> budget, many seeds, report the spread.

## The honest benchmark — every solver, many instances, many seeds

`experiments/benchmark.py`: fixed, stated budgets; 3 weather instances × 10 seeds
per cell, **every run counted**; 95% intervals; **TTS99**, the time to hit the
optimum at least once with 99% confidence by repeating the solve. The full
tables, with intervals and budgets, are written to `BENCHMARK.md`,
`BENCHMARK_presolve.md` and `BENCHMARK_gate.md` in the run folder.

<!-- results:benchmark -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
**Study A, raw QUBO** (3 instances × 10 seeds; cell = % optimal / mean gap; intervals and TTS99 in `BENCHMARK.md`)

| Grid | Vars | neal | tabu | gsa | ga | qgso |
|---|---|---|---|---|---|---|
| 3x4 | 12 | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 4x5 | 20 | 100% / +0.0% | 100% / +0.0% | 50% / +1.7% | 100% / +0.0% | 100% / +0.0% |
| 4x6 | 24 | 100% / +0.0% | 100% / +0.0% | 0% / +6.2% | 100% / +0.0% | 33% / +2.2% |
| 5x6 | 30 | 97% / +0.0% | 100% / +0.0% | 10% / +4.1% | 100% / +0.0% | 0% / +3.6% |
| 6x7 | 42 | 70% / +0.2% | 100% / +0.0% | 0% / infeasible | 90% / +0.1% | 0% / infeasible |

![Reliability vs size (raw QUBO)](runs/2026-10-28_0231/bench_size.png)


**Study B, raw QUBO** — penalty weight on 4×6

| Penalty | neal | tabu | gsa | ga | qgso |
|---|---|---|---|---|---|
| 1× | 67% / +0.0% | 67% / +0.0% | 3% / +4.9% | 67% / +0.0% | 0% / +2.4% |
| 2× | 100% / +0.0% | 100% / +0.0% | 3% / +5.3% | 100% / +0.0% | 20% / +2.7% |
| 3× | 100% / +0.0% | 100% / +0.0% | 17% / +5.9% | 100% / +0.0% | 43% / +1.7% |
| 5× | 100% / +0.0% | 100% / +0.0% | 17% / +5.0% | 100% / +0.0% | 43% / +1.7% |
| 10× | 100% / +0.0% | 100% / +0.0% | 0% / +6.2% | 100% / +0.0% | 33% / +2.2% |
| 20× | 97% / +0.1% | 100% / +0.0% | 13% / +4.6% | 100% / +0.0% | 33% / +1.4% |

![Penalty sensitivity (raw QUBO)](runs/2026-10-28_0231/bench_penalty.png)


**Study A, presolved QUBO** (3 instances × 10 seeds; cell = % optimal / mean gap; intervals and TTS99 in `BENCHMARK_presolve.md`)

| Grid | Vars | neal | tabu | gsa | ga | qgso |
|---|---|---|---|---|---|---|
| 3x4 | 6 | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 4x5 | 10 | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 4x6 | 14 | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 5x6 | 16 | 97% / +0.0% | 100% / +0.0% | 90% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 6x7 | 22 | 80% / +0.3% | 100% / +0.0% | 13% / +0.8% | 100% / +0.0% | 97% / +0.0% |

![Reliability vs size (presolved QUBO)](runs/2026-10-28_0231/bench_size_presolve.png)


**Study B, presolved QUBO** — penalty weight on 4×6

| Penalty | neal | tabu | gsa | ga | qgso |
|---|---|---|---|---|---|
| 1× | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 2× | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 3× | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 5× | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 10× | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |
| 20× | 97% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% | 100% / +0.0% |

![Penalty sensitivity (presolved QUBO)](runs/2026-10-28_0231/bench_penalty_presolve.png)


**Study A, gate-model, presolved** (3 instances × 5 seeds; cell = % optimal / mean gap; intervals and TTS99 in `BENCHMARK_gate.md`)

| Grid | Vars | neal | qaoa | qaoa_cvar | vqe | vqe_cvar |
|---|---|---|---|---|---|---|
| 3x4 | 6 | 100% / +0.0% | 93% / +0.3% | 100% / +0.0% | 47% / +2.7% | 100% / +0.0% |
| 4x5 | 10 | 100% / +0.0% | 60% / +1.9% | 73% / +0.8% | 13% / +3.9% | 93% / +0.2% |
| 4x6 | 14 | 100% / +0.0% | 27% / +5.5% | 13% / +4.9% | 13% / +6.8% | 53% / +1.6% |
| 5x6 | 16 | 93% / +0.0% | 0% / +5.7% | 0% / +6.4% | 13% / +2.7% | 27% / +2.5% |
| 5x7 | 21 | 87% / +0.1% | 0% / +10.8% | 0% / +8.8% | 13% / +4.7% | 33% / +1.3% |

![Reliability vs size (gate-model, presolved)](runs/2026-10-28_0231/bench_size_gate.png)
<!-- /results:benchmark -->

What the benchmark says:

- **Tabu, neal and the GA form the reliable tier.** Tabu is optimal on every run
  at every size. neal holds 100% to 24 variables, then drops to 97% at 30 and 70%
  at 42 [95% CI 52-83%]; its mean gap there is only +0.2%. **The textbook binary
  GA is the surprise:** 100% to 30 variables and 90% at 42, as reliable as the
  annealers. (The 2026-10-07 README expected it to share GSA's weakness on
  penalty-heavy QUBOs. It does not.)
- **neal is still the fastest route to the optimum**, by TTS99: 0.03-0.5 s,
  against 0.2-1.6 s for the GA and a flat 4.2 s for Tabu, whose budget is
  wall-clock, not work.
- **The gravity swarms degrade with size.** GSA is 50% optimal at 20 variables,
  0% at 24, 33% feasible at 30 and never feasible at 42. QGSO sits between: 100%
  at 20, 33% at 24, feasible but never optimal at 30, never feasible at 42.
- **QGSO still beats GSA at every penalty weight,** as before: 20-43% optimal at
  +1.4 to +2.7%, against GSA's 0-17% at +4.6 to +6.2%. The quantum-inspired
  rotation gate is a real improvement *within* the swarm tier, not a jump past it.
- **A penalty of 1× the largest cell cost is too weak.** On one of the three 4×6
  instances the QUBO's **true ground state is not a valid route** (checked by
  brute force over all 2²⁴ states). The 67% feasibility of neal, Tabu and the GA
  at P = 1× is therefore correct minimisation of an under-penalised QUBO, not
  solver failure. The earlier one-instance study could not see this. From 2× up,
  neal, Tabu and the GA are 100% optimal.
- **Presolve rescues the swarms.** At 6×7, GSA goes from never feasible to 100%
  feasible, and QGSO from never feasible to 97% optimal. neal goes from 70% to
  80% optimal and the GA from 90% to 100%. Penalty sensitivity disappears: on the
  presolved 4×6 every solver is 100% optimal at every weight, bar one neal run at
  20×. *Honest reading:* most of this is the smaller QUBO (42 → 22 variables); at
  a matched size, raw 4×5 with 20 variables against presolved 6×7 with 22, GSA
  does better on the raw one (50% vs 13% optimal).
- **Gate model, presolved 6-21 qubits** (5 seeds × 3 instances; the third table
  above):
  - **CVaR helps VQE at every size:** 100/93/53/27/33% optimal against
    47/13/13/13/13% for plain VQE. At 21 qubits its mean gap is +1.3%, against
    +4.7%.
  - **QAOA loses feasibility as the problem grows,** CVaR or not: 93% feasible at
    14 qubits, 53-73% at 16, 20-27% at 21. Its standard X mixer wanders out of
    the constrained subspace -- the textbook argument for a constraint-preserving
    (XY) mixer, which is not built.
  - **neal is 87-100% optimal at 0.1 s a run,** against 6-16 s for the gate
    solvers.

### Phase 2d/2e — the gravity swarms (GSA, QGSO)

Every other solver is annealing (SA/neal/D-Wave), gate-model (QAOA/VQE),
evolutionary (GA) or local search (Tabu). **GSA** (Rashedi et al., 2009) is a
**swarm metaheuristic** where each candidate solution is a "mass" and better
solutions attract the swarm (`F = G·M_i·M_j / R`), with `G` shrinking over time
for SA's explore→exploit arc. Continuous GSA (`gsa.gsa_minimize`) solves the
speed problem; binary GSA (`gsa.bgsa_qubo`, Rashedi 2010) maps it to 0/1 through
a `|tanh(v)|` flip probability, so it is a QUBO backend. **QGSO**
(`solvers/qgso.py`) keeps the gravity but makes each agent a register of qubits
(P(1) = sin²θ), measured each iteration and steered by a quantum rotation gate.
It runs at the identical budget, so the only difference measured is the update
rule.

> **GSA vs QGSA?** We implemented **canonical GSA** first because it has one
> agreed definition (Rashedi 2009) and is the baseline any "quantum-inspired GSA"
> must beat. QGSO is a classical algorithmic twist (quantum-inspired position
> updates), not real quantum hardware.

## Phase 2 extras — genetic algorithm, speed profile, joint speed + route

### A fourth optimizer family: the genetic algorithm (`solvers/ga.py`)

Annealing improves one solution; the gravity swarms move a population along
attraction forces; a **GA breeds** one — tournament selection, crossover,
mutation, elitism. It is written in plain NumPy with the same two forms and call
signatures as `gsa.py`, so it enters every comparison at the same budget: `--backend
ga` on any QUBO script, a `ga` column in the benchmark, a `GA` row in
`quantum_gate.py` and `audit_seeds.py`, and the continuous form in
`speed_compare.py`. It is deliberately the textbook algorithm, with no one-hot-aware
operators. It turned out as reliable as the annealers (benchmark above). The fleet
has its own GA over ship assignments ([`fleet/genetic.py`](fleet/genetic.py)).

### SA vs GA vs GSA — and a problem that can tell them apart

```bash
python run.py speed_compare                       # one cruise speed
python run.py speed_compare --legs 8 --storms 2 --deadline 50
```

<!-- results:speed_compare -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
**600 nm, 1 leg(s), deadline 40 h** — 2,000 evaluations per optimizer

| Method | Fuel (t) | Hours | vs ref | Time | Speed per leg (kn) |
|---|---|---|---|---|---|
| SA  (annealing) | 40.04 | 40.0 | +0.0% | 14.5 s | 15.0 |
| GA  (genetic) | 40.04 | 40.0 | +0.0% | 0.8 s | 15.0 |
| GSA (gravitational) | 40.04 | 40.0 | +0.0% | 0.7 s | 15.0 |
| QUBO (neal, 40 levels) | 41.39 | 39.2 | +3.4% | 0.2 s | 15.3 |
| Grid (reference) | 40.04 | 40.0 | +0.0% | 0.0 s | 15.0 |
| Baseline (18 kn) | 54.36 | 33.3 | +35.8% | 0.0 s | 18.0 |

![600 nm, 1 leg(s), deadline 40 h](runs/2026-10-28_0231/speed_compare_600nm_40h_L1.png)


**600 nm, 1 leg(s), deadline 90 h** — 2,000 evaluations per optimizer

| Method | Fuel (t) | Hours | vs ref | Time | Speed per leg (kn) |
|---|---|---|---|---|---|
| SA  (annealing) | 18.44 | 90.0 | +0.0% | 14.5 s | 6.7 |
| GA  (genetic) | 18.44 | 90.0 | +0.0% | 0.8 s | 6.7 |
| GSA (gravitational) | 18.44 | 90.0 | +0.0% | 0.8 s | 6.7 |
| QUBO (neal, 40 levels) | 18.45 | 89.7 | +0.0% | 0.2 s | 6.7 |
| Grid (reference) | 18.44 | 90.0 | +0.0% | 0.0 s | 6.7 |
| Baseline (18 kn) | 54.36 | 33.3 | +194.8% | 0.0 s | 18.0 |

![600 nm, 1 leg(s), deadline 90 h](runs/2026-10-28_0231/speed_compare_600nm_90h_L1.png)


**600 nm, 8 leg(s), deadline 50 h** — 16,000 evaluations per optimizer; reference 38.94 t, lower bound 38.63 t

| Method | Fuel (t) | Hours | vs ref | Time | Speed per leg (kn) |
|---|---|---|---|---|---|
| SA  (annealing) | 39.02 | 49.9 | +0.2% | 116.5 s | 13.7 11.2 11.8 11.4 11.7 12.1 12.2 12.5 |
| GA  (genetic) | 38.65 | 50.0 | -0.8% | 6.1 s | 13.3 12.7 11.5 11.4 11.3 11.3 12.0 12.9 |
| GSA (gravitational) | 38.76 | 50.0 | -0.5% | 6.3 s | 12.6 12.6 11.6 12.3 11.0 11.2 12.4 12.6 |
| Lagrangian (reference) | 38.94 | 49.7 | +0.0% | 0.0 s | 13.1 12.6 11.6 11.4 11.4 11.4 12.6 12.6 |
| Baseline (18 kn) | 77.01 | 33.3 | +97.8% | 0.0 s | 18.0 18.0 18.0 18.0 18.0 18.0 18.0 18.0 |

![600 nm, 8 leg(s), deadline 50 h](runs/2026-10-28_0231/speed_compare_600nm_50h_L8.png)
<!-- /results:speed_compare -->

On a single cruise speed all three land on the same answer: the problem is
one-dimensional, so it validates the solvers but cannot rank them. **The 8-leg
speed profile can rank them.** That is N coupled decisions under one deadline, at
equal evaluation budgets, against a Lagrangian reference that brings its own
lower bound.

- **The GA finds 38.65 t, within 0.02 t of the lower bound** (38.63 t).
- **That beats the reference itself**, whose plan sits 0.8% above the bound --
  the reference's own duality gap.
- GSA comes next at −0.46%, then SA at +0.20%.
- SA's 116 s against ~6 s is implementation, not algorithm: SA makes one model
  call per evaluation, while the population methods evaluate a whole generation
  in one call.

### Speed and route in one QUBO (`experiments/joint_speed_route.py`)

One binary variable per (stage, row, speed level). The arrival deadline is an
inequality, which a QUBO cannot hold directly. Instead of slack bits, time is
given a **price** in tonnes per hour and a classical loop bisects it until the
deadline is just met. Each step is one ordinary QUBO solve. The known cost is a
duality gap: a price can only reach plans on the convex hull of the fuel-vs-time
trade-off. An exact label-setting DP measures it.

<!-- results:joint -->
<!-- generated by tools/make_report.py from runs/2026-10-28_0231 -- do not edit by hand -->
| Grid | Speeds (kn) | Vars | Deadline | Naive (t) | QUBO + price (t) | Exact (t) | Gap | QUBO solves |
|---|---|---|---|---|---|---|---|---|
| 1x8 | 10 13 15 18 | 32 | 42.9 h | 57.76 | 50.92 | 50.68 | +0.47% | 12 |
| 3x5 | 10 14 18 | 45 | 42.9 h | 44.99 | 44.99 | 44.99 | +0.00% | 12 |
| 5x7 | 10 14 18 | 105 | 42.9 h | 44.15 | 48.98 | 41.85 | +17.04% | 12 |

![Joint speed + route](runs/2026-10-28_0231/joint_1x8x4_s2_seed0.png)
<!-- /results:joint -->

**The price on time works on the small problems and fails on the 5×7 grid**: it
lands 17% above the exact optimum, and worse than the naive straight line at one
constant speed (44.15 t). Measured by re-running the same loop with an **exact**
inner solver in place of neal, the method's own gap is **+27%**. With 3 speed
levels the cheap on-time plans are not on the convex hull, so no price reaches
them; neal's sampling noise happened to land somewhere better. Candidate fixes,
not built: keep the cheapest on-time plan among *all* samples, a slack-bit
deadline (the ablation the 2026-10-07 entry deferred), or local search on the
final plan.

## Feasibility study — which solver to use, and why

### What the measurements say

| Solver | Paradigm | Reliability measured here | Practical ceiling |
|--------|----------|---------------------------|-------------------|
| **Exact DP** | Exact (grid structure) | optimal by construction, milliseconds | any grid (route only) |
| **Tabu** | Classical local search | 100% optimal to 42 vars (raw) | hundreds of vars; wall-clock budget |
| **neal** | Classical annealing | 100% to 24 vars, 70% at 42; single runs 1-1.5% off at 171-275 | fastest TTS99; reads/sweeps must grow with size |
| **GA** | Classical evolutionary | 100% to 30 vars, 90% at 42 | as reliable as the annealers at this size |
| GSA / QGSO | Classical swarm | degrade from ~20 vars; presolve rescues them | ~20-30 vars raw |
| VQE-CVaR | Gate-model | 15/15 at 12 qubits; 33% at 21, +1.3% | ~26 qubits on the GPU simulator |
| QAOA (± CVaR) | Gate-model | feasible 20-27% of runs at 21 qubits | constrained QUBOs need a constraint-preserving mixer |

### Decision guide

| If you need… | Use | Why |
|--------------|-----|-----|
| The **production answer** for single-voyage routing | **Exact DP** | the route problem as modelled is a shortest path: exact, instant, any size |
| A **QUBO** answer (the hardware-portable form) | **neal**, presolved, with a penalty ≥ 2× the largest cost | fastest to the optimum; give it more reads/sweeps past ~40 vars |
| A classical **second opinion** on the QUBO | **Tabu** or the **GA** | both as reliable as neal here, different mechanisms |
| To optimize a **continuous** variable or a **speed profile** directly | **GA** (or GSA) | the GA reached the lower bound on the 8-leg profile |
| To **demonstrate quantum portability** (gate-model) | **VQE-CVaR** on a presolved QUBO | the only gate-model solver that stayed feasible and near-optimal to 26 qubits |
| Real **quantum annealing at scale** | D-Wave Advantage (`--backend dwave`) | embeds hundreds of variables -- *needs paid/academic Leap access* |

### Verdict

> **For a single voyage, exact methods win:** the DP solves the route and a scan
> or the grid solves the speed, so no heuristic is needed. The QUBO formulations
> exist so the *same model* runs unchanged on quantum hardware. Among the solvers
> of those QUBOs, **neal, Tabu and the GA are reliable at the sizes we run**;
> presolve and a sane penalty weight matter more than the choice between them.
> **Gate-model quantum is a portability demonstration, not a solver here:** with
> CVaR, VQE is credible to ~20 qubits, while QAOA cannot hold the constraints
> past ~14. The combinatorial problem a designed optimizer must earn its keep on
> is the **fleet** ([`fleet/README.md`](fleet/README.md)).

---

## The story for the demo (30 seconds)

> *"We simulate Coast Guard voyage data from real naval-architecture physics —
> the cubic speed–power law. A physics-informed model reads that law back out of
> noisy data (exponent 3.00, R² 0.99), and — unlike a plain boosted tree that
> scores almost as well — recommends speeds that waste no fuel. The optimizer then
> picks the slowest speed that still meets the mission deadline: 26% less fuel on
> a tight schedule, 66% on a relaxed one, with the CO₂ reported for IMO
> compliance. Every number is checked against the exact optimum, and every
> benchmark counts every run."*

**Honest caveats** (good to state up front): the physics coefficients are
plausible but not calibrated to a specific vessel; the data is synthetic; the
QUBOs are solved on *classical* hardware (neal, and QAOA/VQE on a simulator) --
the formulation is hardware-portable.

---

## Roadmap

- **Phase 1 — DONE:** synthetic ship data → fuel-rate model → slow-steaming
  optimizer with deadline + CO₂ + decision regret.
- **Phase 2 — DONE:** QUBO optimizers for speed, routing and both jointly; six
  solver families; an honest, many-seed benchmark with intervals.
- **Phase 3 — DONE (2026-10-28):** Streamlit dashboard, `python run.py dashboard`.
- **Phase 4-5 — fleet:** see [`fleet/README.md`](fleet/README.md).

**Open, with the evidence for each:**

- Joint speed + route: the price-on-time gap is +27% on 5×7. Keep the cheapest
  on-time sample, or try slack bits.
- QAOA on constrained QUBOs: feasibility collapses past 14 qubits. Try a
  constraint-preserving (XY) mixer started from a valid state.
- neal past ~150 variables: scale reads/sweeps with size, measured by the benchmark.
- Fleet: the assignment QUBO stalls from 4 × 10, because its pairwise costs
  misprice bundles of 3+ missions. Feed the oracle's true bundle costs back, or
  seed the GA with the QUBO's first proposals (good early, while the GA is
  better late).
- Retrain on a real public dataset (e.g. FuelCast). Needs the schema mapped.
- Proposed and not chosen on 2026-10-28 (reasons in the journal): rough-sea
  oversampling, prediction intervals, selecting the model by regret, a
  misspecified-physics test, refreshing `docs/concepts.pdf`.
