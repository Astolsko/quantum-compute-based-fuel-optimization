# Project journal

A running record of what was decided, why, what was tried, what was measured and
what is still unverified. Written for whoever picks this project up next —
including future Claude sessions, which start with no memory of earlier ones.

**How to use it.** Read the "Current state" section first. Add a new dated entry
at the bottom for each working session; do not rewrite old entries (correct them
with a later note instead). Say plainly whether a claim was measured, calculated
by hand, or assumed.

---

## Current state (as of 2026-10-28)

- **Code:** extended on 2026-10-28 (entry at the bottom): fleet GA, hybrid v2,
  growing fleet horizon, Aer CPU/GPU + CVaR, statistics, JSON results, run
  folders, generated README tables, run.py, Streamlit dashboard.
- **Verified:** the first full run, `runs/2026-10-28_0231/` (34 steps, 2.0 h,
  all succeeded), on the workstation. `python run.py test` passes, 14 checks,
  about 90 s.
- **READMEs are current.** Every table between `<!-- results:NAME -->` markers
  is generated from that run by `python run.py report --run
  runs/2026-10-28_0231 --update-readme`. The prose was written against the same
  run. Rerunning and refreshing replaces the numbers but not the prose, so
  reread the prose after a new run.
- **Main findings:** single voyage: exact methods win; the physics grey box
  has ~0 decision regret. Gate model: CVaR helps both QAOA and VQE; QAOA loses
  feasibility past ~14 qubits. Fleet: the exact DP plans up to 16 missions in
  at most 25 s. Beyond that, ALNS (level with the GA, far faster). The QUBO
  hybrid falls behind from 4 ships × 10 missions.
- **Standing rules from the project owner:**
  - On the laptop, run nothing but the tests. It force-shut-down on 2026-10-06
    during a full `demo.py` run. Full runs happen on the owner's Linux
    workstation (NVIDIA A6000).
  - Keep this journal up to date.
  - Propose improvements first and ask; then make every chosen change in one
    batch and run it.

### Things that will trip you up

- Run everything from the project root through `python run.py <target>`
  (`python run.py` alone lists the targets). The old ways still work:
  `python -m experiments.<name>` from the root, `cd fleet; python <script>.py`.
- `bash run_all.sh` writes everything to a new `runs/<date>_<time>/` folder
  (via `FUEL_RUN_DIR`). Scripts run on their own still write to `outputs/` and
  `fleet/outputs/`.
- The conda env `quantum-fuel` sets `PYTHONNOUSERSITE=1`, because the
  workstation's ~/.local Python 3.11 packages otherwise leak in (scipy and
  packaging were imported from there). Outside that env, export it yourself.
- qiskit-aer-gpu replaces qiskit-aer (same import name). `FUEL_QDEVICE`
  (auto|CPU|GPU) picks the device; auto = GPU from 18 qubits if one is there.
  `FUEL_AER_THREADS` caps CPU threads per simulation (`benchmark.py` sets 1
  when it runs solves in parallel).
- The model pickle refers to `core.physics_model.PhysicsResidualRegressor`
  (moved from the root on 2026-10-28). Pickles made before that cannot be
  loaded; `optimizer.load_model()` then retrains.
- Fleet rate tables are cached in `fleet/cache/`, keyed by scenario AND an md5
  of the pickled model. Delete the folder to force a rebuild.
- `FUEL_JOBS` caps cores for training and for `benchmark.py --jobs` (run_all
  used 28). The fleet benchmark is single-process.
- Several tests pass the TRUE physics (`generate_data.TruthModel`) where the
  scripts normally use the trained model. That keeps them light; it also means
  they do not test model quality at full size.

---

## 2026-10-06 / 07 — reorganisation, model fix, GA, Phase 2 extras

Session by Claude (Opus 5.5) with the project owner. Two days because the laptop
shut down mid-session on the first day.

### What was asked

Read everything; create the `quantum-fuel` conda env and install requirements;
keep the existing outputs safe; arrange files into folders with the main
training files at the root; implement the Phase 2 optional extras and a genetic
algorithm; suggest further improvements before changing code. The owner then
approved improvement items 1–10, 13, 16 and 17 from the list that was proposed,
and asked for this journal.

### What was found before changing anything

1. **The optimizer was optimising a tree model's artefacts.** Tree ensembles are
   flat between split points, so `rate(speed) / speed` is a sawtooth and the
   "optimum" is whichever tooth is lowest. Calculated from the generator's own
   constants and later confirmed by `test_truth_and_regret_numbers`: for the
   default 600 nm voyage the true economic speed is 6.29 kn; under the 90 h
   deadline the true optimum is the deadline-bound 6.67 kn at 18.43 t; the
   README's recommendation of 8.4 kn truly costs 20.08 t — about 9% more than
   necessary. The README's "SA matches the grid to 0.01 kn" was true but only
   shows the optimizer agrees with the model.
2. **The shipped model could not be loaded.** It was pickled with scikit-learn
   1.7.0; the fresh env resolved 1.9.1 and loading failed with
   `No module named '_loss'`. `load_model()` only retrained when the file was
   missing, so every script except `demo.py` / `train_model.py` would crash on a
   fresh install.
3. **The benchmark's "equal budget" was fixed but not equal compute**, each grid
   size was a single weather instance (only the solver seed varied), and Tabu was
   unseeded because `backends.sample_qubo` dropped the seed.
4. `decode_route` did not check the start and end ports; only `quantum_gate.py`
   did, in its own wrapper.
5. Training speeds started at 5 kn while the optimizer searched from 4 kn.
6. Docs drift: `BENCHMARK.md` linked but absent (and written to whatever the
   working directory was); `make_concepts_doc.py` wrote to another machine's path
   and needed an unlisted package; the concepts PDF said Laden 1.15× / Ballast
   0.85× where the code uses 1.12 / 1.0; the README linked `route_qubo.png`,
   a filename the current code never produces.
7. Some plots in `outputs/` were made by older code than what was in the repo.
8. `quantum_gate.py` called an exact eigensolver for the speed problem at any
   size (memory doubles per speed level).
9. `fleet/run_fleet.py` read `q["raw_feasible"]`, a key that is absent when the
   QUBO decode fails early — a latent `KeyError`.

### What was tried and did not work

- **Baseline run of the old code on the new library versions.** Started as
  `demo.py` in a scratch copy; the laptop force-shut-down during it. So there is
  NO before/after comparison at full size from this session — only the archived
  plots and the README numbers, which were produced with older library versions.
  The likely trigger is all-core CPU load (the random forest used `n_jobs=-1`
  with 5-fold CV); that is an inference, not something that was diagnosed.

### Decisions, with reasons

**Layout.** Root keeps the main pipeline, runnable directly: `config.py`,
`generate_data.py`, `train_model.py`, `optimizer.py`, `demo.py` (plus
`physics_model.py`, which `train_model.py` imports). Algorithms went
to `solvers/`, optional studies to `experiments/`, the concepts doc to `docs/`,
the test to `tests/`. `fleet/` kept its layout; only two import lines changed.
`base_scenario_for` moved from `qubo_route.py` to `config.py` so the fleet code
no longer imports an experiment (and with it pyqubo and matplotlib) for one
helper. `gsa_speed.py` became `experiments/speed_compare.py`.
*Rejected:* a path-bootstrap line at the top of every script so they could also
be run as files (works, but it is a hack in eight places); an installable
package (more machinery than this project needs).

**Existing outputs.** Copied, not moved, to `outputs_archive/original_2026-10-06/`
(`outputs/`, `fleet_outputs/`, plus the `data/` and `models/` that produced
them), verified byte-identical. README image links now point into the archive.

**Genetic algorithm — plain NumPy, not DEAP.** `solvers/ga.py` mirrors `gsa.py`:
continuous `ga_minimize`, binary `bga_qubo`, and a sampler adapter, so it is
`--backend ga` everywhere and a `ga` column in the benchmark at the same budget
as GSA and QGSO. DEAP was listed as an optional extra, but it evaluates one
individual at a time and would be the slowest thing in a multi-seed benchmark;
about 80 vectorised lines avoid a new dependency. The operators are the textbook
ones on purpose (tournament, uniform/BLX crossover, bit-flip/Gaussian mutation,
elitism): no one-hot-aware repair, so the benchmark measures the GA family
rather than problem knowledge.

**Phase 2 extras — what that was taken to mean.** (a) From `requirements.txt`:
XGBoost and LightGBM, added as model candidates that are skipped if not
installed. (b) From the README's extension list: multi-segment speed profile and
joint speed + route. FuelCast retraining was NOT done — it needs the dataset's
schema checked and mapped first.

**Joint speed + route: a price on time, not slack bits.**
`experiments/joint_speed_route.py` uses one variable per (stage, row, speed
level). The deadline is an inequality; the textbook encoding adds slack qubits
and squares the whole sum, coupling every pair of variables. Instead time gets a
price `lam` in tonnes per hour (a linear term) and a classical loop bisects it.
The known cost is a duality gap — only plans on the convex hull of the
fuel-vs-time trade-off are reachable — so an exact label-setting DP is included
to measure it. With `--rows 1` this is the speed profile; with `--levels 1` it
is the route QUBO. *Not tried:* the slack-bit version, so the two are not
compared.

**SA vs GA vs GSA needed a harder problem.** On one cruise speed every optimizer
gives the same answer, so `speed_compare.py --legs N` adds the N-leg profile
with equal evaluation budgets (2,000 per leg each). Its reference is a
Lagrangian relaxation that returns both a feasible profile and a lower bound.

**The model fix (item 9).** `train_model.py` now trains seven candidates: the
original three unchanged, `Gradient Boosting (log, monotone)`, `XGBoost`,
`LightGBM`, and `Physics + Boosting`. The log-target/monotone booster is a NEW
row rather than a change to the existing one, so the original three stay
comparable with the old README table. The best model is still chosen by test R².

- *Why log target + monotone constraints are not enough on their own:* they tame
  the noise but the model is still piecewise-constant in speed, so the sawtooth
  remains. Kept because it was the approved minimum and is a fair "improved
  black box" row.
- *The grey box* (`physics_model.py`):

  ```text
  rate = D^(2/3) * [ hotel + exp(k + n*ln(stw) + b*ln(hull) + a_w*waves + a_u*wind + a_l*laden) ] * exp(g(x))
  ```

  Seven coefficients fitted by least squares on ln(rate) (the noise is
  multiplicative), started from the textbook cubic at the data's own scale, not
  from the generator's constants. `g` is gradient boosting on the residual with
  early stopping so it stays near zero where the form already fits.
- *Honesty caveat, also in the module docstring:* the generator has the same
  "hotel + power law" structure, so a near-perfect fit shows calibration, not
  discovery. The black-box rows remain the test of learning from scratch.
- *Why its own module:* a class defined in a script that is run as `__main__`
  pickles as `__main__.Class` and cannot be loaded from any other entry point.
- *Rejected:* a pure log-log linear model (smooth, but with no additive hotel
  term the total-fuel curve has no interior minimum, so "economic speed"
  disappears); kernel-smoothing the tree model over speed (model-agnostic but
  biases a convex curve upward and is a patch at the decision layer); smoothing
  only inside the speed optimizer (would not help the route or fleet layers,
  which also query the model).

**Decision regret (item 10).** `generate_data.TruthModel` exposes the noise-free
physics behind `.predict()`, so it can stand in for the trained model anywhere.
`optimizer.regret_panel` scores a model on 12 voyages (3 ship classes × calm /
rough × no deadline / tight deadline): extra true fuel burned at the speed the
model picks, versus the true optimum. Reported as two columns in the model
comparison, per voyage in `demo.py`, and for the chosen route in
`qubo_route.py`. It only exists while the data is synthetic; `train_model.main`
carries a `ponytail:` comment marking where to drop it when real data arrives.

**Robust model loading + pinned versions (item 1).** The model's metadata records
the scikit-learn version; `load_model()` retrains when it differs or the file
will not load. `requirements.txt` pins the versions the tests passed on.

**Laptop safety (item 2).** `FUEL_JOBS` replaces the random forest's
`n_jobs=-1`; training is wrapped in `threadpool_limits` because the boosters
thread through OpenMP regardless of `n_jobs`; the benchmark has `--jobs`. The
eigensolver in `quantum_gate.py` was replaced by a scan of the feasible speed
levels, which is the same optimum for a one-hot QUBO.

**Batched model calls (item 3).** `optimizer.predict_rates` makes one model call
for an array of speeds; the grid search, the plot, `cell_costs` and the speed
QUBO use it. *Correction to what was promised:* simulated annealing's 2,000
evaluations are inherently sequential and remain one call each, so the saving
is the grid's 1,000 calls and the per-cell loops, not "all 3,000".

**Presolve (item 13) is opt-in.** `build_route_qubo(..., presolve=True)`
substitutes constants for the port columns and for cells too far from a port to
be reachable; the Hamiltonian is unchanged, just written over fewer variables
(3×4: 12→6, 5×7: 35→21, 11×25: 275→213). Opt-in so the raw numbers stay
reproducible and so "does presolve make the swarm and gate solvers more
reliable?" can be run as its own experiment. `decode_route` now takes the ports
and checks them (item 4).

**Benchmark (item 16).** `--instances` (default 3) pools several storm seeds per
grid size; `budget_text` states each budget in evaluations and the docstrings
say "fixed", not "equal"; GA is on by default, QGSO still behind `--with-qgso`;
results are keyed by grid label because two grids can presolve to the same size
(3×4 and 4×4 both become 6 variables); `BENCHMARK.md` is written to the project
root. For `--jobs`, the budget is passed to each solve as an argument: worker
processes re-import the module, so a global lookup would have silently ignored
`--fast` and `--gsa-restarts`. Tabu is now seeded (item 5); its per-read limit
is 20 ms of wall-clock (checked in the installed version — an earlier note in
this session assumed 100 ms and was wrong).

**Fleet exact ground truth (item 17).** `matheuristic.exact_dp`: per ship a
Held–Karp DP over mission subsets that carries the whole "fuel to be free by
slot tf" vector (the same min-plus chaining `Evaluator.ship_cost` uses), then a
set-partition DP across ships. Exact to about 12 missions, where enumeration
stops at 6. `benchmark_fleet.py` and `run_fleet.py` use it and accept
`--ships` / `--missions`. The position QUBO's `max_pos` now scales with missions
per ship, otherwise it cannot represent a 12-mission plan.

**Smaller fixes.** Generator speeds start at `PHYSICAL_MIN_SPEED` (item 7; this
changes the dataset). Dead code removed from `op_swap` (item 8). Concepts
generator path, load factor text and file paths fixed; the PDF itself was not
regenerated (item 6). `demo.py` takes `--n` and draws the true-physics curve.

### What was measured

All on the laptop, single thread, by `tests/test_smoke.py` on 2026-10-07. These
are small-sample checks, not results to quote as the project's performance.

| What | Result |
|---|---|
| Whole suite | 10 checks pass, ~34 s |
| White-box fit on 3,000 rows | speed exponent 3.002, hotel 0.0195 (true 0.02 before the ≤3% fuel-energy factor), 0.65% mean error vs the true physics |
| Decision regret on 3,000 rows, mean / worst over the 12-voyage panel | physics form 0.00% / 0.00%; plain gradient boosting 5.17% / 10.58% |
| Route QUBO, raw and presolved (3×4) | both equal the exact DP under exhaustive enumeration of all states |
| Joint speed + route, 2×3 grid × 2 speeds | price-of-time loop = exact optimum, 56.82 t (naive 72.39 t) |
| Speed profile, 3 legs, true physics as model | SA +0.2%, GA −0.2%, GSA +0.1% vs reference 41.46 t (lower bound 41.38 t) |
| Fleet, 2 ships × 4 missions | `exact_dp` = `exhaustive` = 55.16 t |
| Qiskit 2.5.2 | QAOA and VQE run at 6 qubits; fleet QAOA at 8 |
| Parallel benchmark path | two worker processes, including pickling the Qiskit problem |

The 400-row model table from the same test, for orientation only (400 rows is
far too few for the tree models; expect them near R² 0.98 at 10,000):

| Model | Test R² | MAPE | Regret mean / max |
|---|---|---|---|
| Physics + Boosting | 0.993 | 5.5% | 0.18% / 1.53% |
| Gradient Boosting (log, monotone) | 0.946 | 17.2% | 5.50% / 18.09% |
| Random Forest | 0.929 | 20.6% | 2.14% / 4.78% |
| LightGBM | 0.916 | 29.3% | 8.02% / 28.78% |
| Gradient Boosting | 0.901 | 36.4% | 4.88% / 14.08% |
| XGBoost | 0.895 | 30.3% | 11.23% / 29.11% |
| Linear Regression | 0.591 | 226% | 117% / 519% |

### What is NOT verified

- Any full-size run: `demo.py` at 10,000 rows, every experiment with the real
  trained model, both benchmarks, the fleet benchmark, `run_all.sh` itself.
- Which model wins at 10,000 rows (expected: the grey box).
- Whether presolve improves solver reliability, and how GA ranks.
- Whether the fleet findings (hybrid 10/10 optimal, QUBO guidance beating random
  proposals) survive the new fuel model and the M = 12 instances.
- Time and memory of `exact_dp` at M = 12 (estimated: seconds per ship, about
  14 MB per ship; not measured).
- Anything on Linux. All testing was on Windows.

### Consequences to expect on the first full run

- Every fuel figure changes, because the model changed.
- The demo's "relaxed 90 h" case becomes deadline-bound at 6.7 kn instead of
  "8.4 kn economic speed". That is correct: the true economic speed (6.3 kn)
  needs about 95 h. To show the bottom of the U-curve, relax that deadline past
  ~96 h in `demo.py`. Left unchanged because it is a presentation choice.
- Benchmark tables gain a GA column and pool three instances per size, so they
  are not row-for-row comparable with the archived ones. `--instances 1`
  reproduces the old single-instance design.

### Proposed and not done

| Idea | Why not (yet) |
|---|---|
| Sample more rough-sea training data (only ~1.7% of rows have waves above 6 m, yet routing depends on storm-core costs) | Not selected. Still worthwhile; matters most for the tree models. |
| Prediction intervals via quantile boosting | Not selected. |
| Constraint-preserving QAOA (XY mixer from a valid path) | Not selected. The standard remedy for the README's 2/15 QAOA result on the route QUBO; presolve is a cheaper first step. |
| GPU statevector simulation on the A6000 | Not selected. Feasible: the workstation is Linux. Note the code uses Qiskit's NumPy `StatevectorSampler`, with Aer only as the transpile target, so this means switching to Aer's sampler with `device="GPU"`. Would lift the ceiling from ~18 to ~30 qubits. |
| GA over fleet assignments, scored by the same oracle | Not selected. Would be the natural classical rival to the QUBO hybrid. |
| Half-hour time slots in the fleet oracle | Not selected. Fixes the documented one-cell-per-slot speed cap. |
| Streamlit dashboard (README Phase 3) | Not selected. |
| Real-data loader / FuelCast | Not selected; needs the schema first. |
| PyTorch physics-informed network on the GPU | Not proposed for doing: at 10,000 rows a GPU gives nothing. |
| Slack-bit deadline QUBO | Deliberately skipped in favour of the price on time; would make a good ablation. |

### Next steps

1. On the workstation: `conda activate quantum-fuel` then
   `FUEL_JOBS=-1 bash run_all.sh 2>&1 | tee run_all.log`.
2. Update the results tables in `README.md` and `fleet/README.md` from the new
   outputs, repoint the figure links from the archive back to `outputs/`, and
   remove the "read this before quoting a number" banners.
3. Record here what the full run showed — especially the regret columns at
   10,000 rows, raw vs presolved benchmark, and the M = 12 fleet study.

---

## 2026-10-28 — workstation: env, first full run, fleet GA, hybrid v2, GPU, reporting

Session by Claude (Opus 5.5) with the project owner, on the Linux workstation
(Xeon Gold 6226R, 32 cores, 62 GB, RTX A6000 48 GB, Ubuntu 20.04).

### What was asked

The same request as 2026-10-06 (read everything, conda env, keep outputs safe,
arrange files, Phase 2 extras + GA, propose improvements first). Most of it was
already done; that was verified file by file and said so. Of the 21 ideas then
proposed the owner chose 1, 2, 4, 7, 8, 15, 16, 17, 18, 19, 21:
fleet GA (1), fix the hybrid at scale (2), scale the fleet horizon + scale study
(4), Aer/GPU simulator (7), CVaR for QAOA/VQE (8), significance tests + TTS (15),
results as JSON + generated README tables (16), one folder per full run (17),
clear the duplicated outputs (18), run.py + move physics_model (19), Streamlit
dashboard (21). Not chosen: exact per-ship ordering by subset DP (3), parallel
fleet benchmark (5), half-hour slots (6), XY-mixer QAOA (9), IBM rerun (10),
model/data items 11-14, concepts.pdf refresh (20).
Defaults taken where questions went unanswered: GPU build of Aer (bumps 0.17.2 ->
0.18.2), FUEL_JOBS=28 for the full run, READMEs rewritten from the new run.

### Environment (things that will trip the next person up)

- `conda create` died in the channel-notices fetch (JSONDecodeError on a cached
  notices file); `CONDA_NUMBER_CHANNEL_NOTICES=0` for that one command worked.
- ~/.local/lib/python3.11/site-packages (613 packages: torch, tensorflow, scipy
  ...) leaks into every python3.11 env. scipy and packaging were being imported
  from there. The env now sets `PYTHONNOUSERSITE=1` (conda env config vars), and
  run_all.sh exports it too. `pip check` is clean only with it set.
- qiskit-aer-gpu 0.18.2 replaces qiskit-aer (same import name); the GPU wheel
  brings CUDA 12 + cuQuantum as pip packages. requirements-gpu.txt says how.

### Found by measuring, before changing code (true physics as the model)

| 4 ships x 12 missions, 36-slot horizon | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| exact_dp | 0.4 s, 55.30 t | 0.4 s, 44.79 t | 0.4 s, 68.96 t |
| ALNS 3,000 iters | +3.3% | +10.7% | +28.6% |
| hybrid (v1) | no feasible plan | +12.7% | no feasible plan |

- The v1 hybrid's QUBO rated overloaded, time-infeasible assignments (5-7
  missions on one ship) BELOW the true optimum (seed 0: -9934.9 vs -9915.8);
  0 of its 25 proposals were feasible, before or after repair. A formulation
  problem, not a sampler problem.
- exact_dp was far cheaper than the 2026-10-07 estimate: 0.4 s at 4x12, 22 s and
  1.6 GB peak at 5x16. But every 4x14 and 5x16 instance was infeasible: the
  generator packed all mission windows into a fixed 36-slot horizon.

### Decisions, with reasons

**Fleet horizon (item 4).** `fleet_config.horizon_slots(K, M) = max(36, 14 +
ceil(M/K) * 11)`: one 11-slot wave of start windows per mission per ship. The
default 3x6 (and the test's 2x4) is 2 waves = 36 slots, so those instances are
bit-identical to before; 4x12 gets 47, 5x16 gets 58. Some instances are still
infeasible (4x12 seed 3, 5x16 seed 1) and are reported as such.

**Fleet GA (item 1) -- genetic.py.** Chromosome = ship index per mission;
uniform initial population (so generation 0 IS the random ablation); binary
tournament, uniform crossover, 1/M reassignment mutation (doubled after a
generation that proposed nothing new), elitism 2, Lamarckian write-back of the
repaired assignment. Population 40: at a 1,000-scoring budget pop 40 beat pop 20
on 2 of 3 instances and tied the third (4x12: +4.2/+3.5/+1.8% vs +6.7/+3.5/+10.0%).

**A fair currency for proposal methods.** rand+exact, GA and hybrid v2 all score
through `hybrid.AssignmentScorer`: same repair, same exact best-order-per-ship,
budget = number of DISTINCT exact scorings, best-so-far recorded after each one
(so they are also compared at 25 and 200, not just the final 1,000). ALNS keeps
its own 3,000-iteration budget; wall times are reported for everything.

**Hybrid v2 (item 2) -- solve_hybrid_cuts.** v1 is kept unchanged as
`hybrid_v1`. v2 = the same QUBO + lazy cuts + a stronger repair:
- cuts: for each proposed ship bundle that is infeasible, a minimal infeasible
  subset by deletion filter (valid because feasibility is downward-closed --
  waiting at stations is free), forbidden with a product penalty quadratised by
  Rosenberg auxiliary variables (penalty 2P, so the auxiliary never cheats;
  checked exhaustively in the test). Resample, 8 rounds.
- repair: strip each infeasible bundle to what fits (in start-window order),
  reinsert the stripped missions where they add least fuel. The v1 repair moved
  one mission at a time and gave up when the remainder was still infeasible.
- "hybrid+alns": v2's best plan as ALNS's starting point (alns(start=...)).
- *Tried and rejected: a per-ship capacity constraint* (slack bits, cap = largest
  feasible bundle by Apriori enumeration, which is exact since feasibility is
  downward-closed). Every cap came out 5 while optima use up to 5 missions, so it
  constrained almost nothing, and its dense penalty made v2 worse in 5 of 6 runs
  (4x12, budgets 200/1000: e.g. seed 0 +9.6% -> +16.4%).

**Fleet table cache.** Keyed by scenario only until now -- a retrained model
would have been served tables priced by the old one. Key now includes an md5 of
the pickled model (stable across processes, checked) and the cache moved to
fleet/cache/ so it survives across run folders.

**Gate model (items 7, 8).** solvers/gate.py: Aer SamplerV2 instead of Qiskit's
NumPy StatevectorSampler; GPU from 18 qubits (measured crossover, table in the
module docstring: 22 qubits QAOA 20 iterations = 121 s on 1 CPU thread, 12 s on
32, 2.6 s on the GPU); CVaR(alpha = 0.25) as extra QAOA-CVaR / VQE-CVaR rows
everywhere (quantum_gate, audit_seeds, benchmark --gate, fleet quantum_qaoa
--cvar). The gate benchmark stops at 21 qubits (26 costs ~3 min per QAOA run and
~7 per VQE run; at 5 seeds x 3 instances x 4 variants that is hours) and
quantum_gate runs 26 qubits with 2 trials.

**Bug found: MinimumEigenOptimizer threw away most samples.** With
qiskit-algorithms 0.4 the eigenstate is a dict of PROBABILITIES; qiskit-
optimization treats dict values as AMPLITUDES, squares them and drops < 1e-6.
A bitstring measured once in 1,024 shots is (1/1024)^2 < 1e-6, so only
bitstrings measured at least twice were ever candidates, and at ~21 qubits
nothing survived and solve() crashed (IndexError). Measured on the 3x4 audit
QUBO (12 qubits, QAOA reps 2 x 100 iterations, 15 seeds, same distributions
decoded both ways): the filter kept ~25% of samples; optimal 1/15 -> 3/15,
feasible 9/15 -> 13/15 with every sample decoded. `gate.sample_qp` now decodes
every measured bitstring. **Every QAOA/VQE number from before 2026-10-28 was
produced with this filter** (the README's 2/15, 4/15 audit included).

**Reporting (15, 16, 17).** core/stats.py (Wilson, bootstrap, TTS99, paired
Wilcoxon + exact McNemar), core/results.py (JSON per script), tools/manifest.py,
tools/make_report.py (RESULTS.md per run; `--update-readme` refreshes
`<!-- results:NAME -->` blocks). Each run_all.sh run goes to runs/<date>_<time>/
with log, manifest, timings.tsv (step, seconds, exit code), plots, JSON, a copy
of data/ and models/, RESULTS.md; a failed step is logged and the run continues.
BENCHMARK*.md now lands in the run folder, not the project root.

**Layout (18, 19).** physics_model.py -> core/; run.py at the root dispatches
every script (the old ways still work); tools/ and dashboard/ added; the 21
plots in outputs/ and fleet/outputs/ deleted after re-checking each is
byte-identical in outputs_archive/original_2026-10-06/.

**Dashboard (21).** dashboard/app.py, Streamlit 1.66.1: voyage speed, weather
route, fleet plan, run browser -- thin layers over the existing functions. The
smoke test renders it with streamlit.testing.AppTest and clicks "Optimise speed".

### First full run: runs/2026-10-28_0231 (what it found)

All tables are in that folder's RESULTS.md and in the README blocks. What a
reader should take from them:

- **Phase 1 holds.** Physics+Boosting: test R² 0.994, MAPE 4.9%, decision
  regret 0.00% mean / 0.05% max; the best black box (GB) scores R² 0.989 but
  regret 3.5% / 9.6%. Fitted exponent 3.002. 40 h deadline: 15.0 kn, -26.3%
  fuel; 90 h: 6.67 kn, -66.1%.
- **Penalty weight matters more than the solver.** In the raw route benchmark
  at P = 1× the penalty, only 67% of neal/Tabu/GA runs were feasible, and on
  storm seed 4 brute force over all 2^24 states shows the QUBO's ground state
  is itself infeasible: the solvers were right, the QUBO was wrong. From 2× up,
  100% feasible.
- **Presolve helps the annealers and hurts GSA.** At 22 presolved variables GSA
  is 13% optimal (100% feasible) against 50% on the raw 20-variable instance;
  QGSO 97%, Tabu and GA 100%, neal 80% (mean gap +0.3%).
- **Joint speed+route has a method gap, not a solver gap.** 5×7: QUBO 48.98 t
  vs exact 41.85 t (+17%), worse than the naive 44.15 t. Running the same
  price-on-time loop with an EXACT inner solver gives 53.24 t (+27%), so the
  Lagrangian price itself cannot reach the optimum on this instance (duality
  gap); neal is 1-12% off the priced optimum per solve on top of that.
- **Gate model (decoding fixed, CVaR added).** Audit, 3×4 route, 12 qubits,
  15 seeds: QAOA 2/15 optimal (12/15 feasible), QAOA-CVaR 6/15 (15/15), VQE
  7/15, VQE-CVaR 15/15. Gate benchmark (presolved, 3 instances × 5 seeds),
  % optimal at 6/10/14/16/21 qubits: QAOA 93/60/27/0/0 (feasible 53% at 16,
  20% at 21); QAOA-CVaR 100/73/13/0/0; VQE 47/13/13/13/13; VQE-CVaR
  100/93/53/27/33; neal 100/100/100/93/87. At 26 qubits (2 trials, GPU) QAOA
  found no feasible route; VQE-CVaR was +1.9%. CVaR is the single most useful
  change for both; QAOA still cannot hold one-hot constraints past ~14 qubits.
- **Fleet 3×6** (same instances as 2026-10-06, new fuel model): hybrid v1 and
  v2 10/10 optimal, ALNS 9/10 (misses seed 5 again, +9.0%). At 25 exact
  scorings the QUBO proposals are optimal on 10/10 vs 2/10 for random (exact
  McNemar p = 0.008); by 200, random 9/10 and GA 10/10; at 1,000 everything,
  since there are only 729 assignments.
  The McNemar p at 25 scorings was computed by hand (all 8 discordant
  instances favour the hybrid: 2 × 0.5^8); the per-checkpoint tests were added
  to benchmark_fleet.py after this run started, so they appear from the next
  run on.
- **Fleet scale study, 3×6 to 5×16, 10 instances each (65 min, one core).**
  One instance per size from 4×10 has no feasible plan (4×10 and 4×12 seed 3,
  5×14 and 5×16 seed 1). Exact DP time: ≤ 0.1 s to 4×10, 0.5 s at 4×12,
  3.5 s at 5×14, 25 s at 5×16.

  | mean gap (optimal) | 4×8 | 4×10 | 4×12 | 5×14 | 5×16 |
  |---|---|---|---|---|---|
  | alns_fixed | +5.0% (1) | +4.5% (2) | +4.1% (4) | +3.8% (3) | +5.3% (0) |
  | alns | +1.8% (4) | +4.8% (3) | +7.0% (2) | +5.6% (2) | +6.7% (0) |
  | rand+exact | +2.0% (6) | +9.9% (1) | +14.3% (0) | +17.3% (0) | +24.8% (0) |
  | ga | +0.8% (7) | +1.3% (4) | +2.1% (3) | +6.3% (0) | +7.7% (0) |
  | hybrid_v1 | +7.7% (1) | +11.6% (0) | +25.1% (0) | +23.3% (0) | +34.6% (0) |
  | hybrid (v2) | +0.4% (7) | +5.4% (0) | +11.1% (0) | +17.0% (0) | +21.8% (0) |
  | hybrid+alns | +0.1% (8) | +2.4% (4) | +4.8% (1) | +4.5% (1) | +3.6% (1) |

  Paired Wilcoxon: v2 beats v1 at every size from 4×8 (p ≤ 0.04). The GA
  beats v2 from 4×10 (p = 0.027, 0.004, 0.020, 0.004); ALNS beats v2 at 5×14
  and 5×16 (0.020, 0.004). GA vs ALNS never significant (p ≥ 0.11);
  hybrid+alns vs ALNS never significant (p ≥ 0.125). v2 vs random from 4×10:
  p = 0.055, 0.13, 0.82, 0.13.
- **Why v2 stalls (measured after the run, 3 instances each at 4×8, 4×10,
  4×12, same settings):** it is not short of proposals. It spends all 1,000
  scorings and adds 75-400 cuts (QUBO grows by 58-388 auxiliary variables).
  From 200 to 1,000 scorings it gains 0.4-2.3 points at 4×12 and above; the
  GA gains 15-18. The cuts only forbid infeasible bundles; feasible bundles
  that the pairwise energy underprices stay attractive. The GA selects on true
  cost. (At 3×6 v2 scores only 222-526 distinct assignments: it does run out
  there, but the optimum is always among them.)
- **ALNS adaptive weights do not pay:** the fixed portfolio has the lower mean
  gap at every size from 4×10. Not tested pairwise.
- **Fleet QAOA (Aer):** 2×4 (8 qubits) exact with mean and CVaR, trivially (16
  assignments, all scored). 3×6 (18 qubits, GPU): +1.4%, same plan as neal v1.
  4×6 (24 qubits, GPU, 4.9 s): +4.2% vs neal v1 +0.25%. 25 assignments scored
  each, one instance each.
- **Whole run:** 34 steps, 2.0 h, all exit 0. Slowest: fleet scale 65 min, the
  26-qubit gate comparison 11 min, gate benchmark 9.5 min, raw route benchmark
  8 min.

### Changed after the run (derived from its saved JSON, no rerun)

- `tools/make_report.py`: the scale section now also tabulates, per size, the
  anytime results and the paired tests (read from the per-size benchmark
  files); alns_fixed added to the scale table; the fleet QAOA table names the
  simulator briefly and shows how many assignments were scored out of K^M.
- `fleet/benchmark_fleet.py`: the scale plot now shows alns_fixed, and all
  fleet plots use the validated eight-slot categorical palette (greedy and
  rand+exact were two near-identical greys). This run's `fleet_scale.png` was
  redrawn from `fleet_scale.json`; its other fleet plots keep the old colours.
- `experiments/benchmark.py`: no more "-0.0%" in BENCHMARK*.md (this run's
  files still show it).
- READMEs: all result blocks generated from this run; prose rewritten against
  it. fleet/README.md now leads with the scale-study bottom line.

### What is NOT verified

- Whether any of the fleet differences that are not significant at 10
  instances (hybrid+alns vs ALNS, fixed vs adaptive ALNS, GA vs ALNS) would
  become significant with more instances.
- Anything past 16 missions: there is no exact reference there.
- Fleet QAOA beyond one instance per size; any real-hardware run after the
  2026-10-06 one (8 qubits).
- The dashboard in a browser: it is exercised by AppTest in the smoke test,
  not clicked through by a person.
- `run_all.sh` on the laptop (it should not be run there).

### Proposed and not done (2026-10-28 list, not chosen)

Exact per-ship ordering by subset DP inside ALNS (3), parallel fleet benchmark
(5; the scale study took 65 min on one core), half-hour fleet slots (6),
XY-mixer QAOA (9; the gate benchmark says this is where QAOA needs help), IBM
rerun (10), rough-sea oversampling / prediction intervals / regret-based model
selection / misspecified-physics test (11-14), concepts.pdf refresh (20).

### Next steps

1. Fleet: feed the oracle's true bundle costs back into the assignment QUBO
   (reprice, not just forbid), or seed the GA with the QUBO's first proposals.
   Measure against v2 and the GA with the same scorer and budgets.
2. Parallelise benchmark_fleet.py over instances (idea 5), then rerun the
   scale study with 30 instances per size to settle the non-significant
   comparisons.
3. Joint speed+route: keep the cheapest on-time sample across price
   iterations, or try slack bits; the Lagrangian price has a duality gap on 5×7.
4. QAOA: XY mixer from a valid state (idea 9).
