"""
test_smoke.py  --  one light check of everything that can break silently.

Small on purpose: single-threaded, a few hundred rows, grids of a dozen cells,
6-8 qubits, and it writes only to a temp folder (never to data/, models/ or
outputs/). It is the ONLY thing in this repo meant to be run on a laptop; the
real experiments belong on a workstation.

    python tests/test_smoke.py        # plain asserts, no test framework needed
    python tests/test_smoke.py -v     # also show what each script printed

Where the true physics is used in place of the trained model, that is to keep
the check light -- no training is needed to verify a QUBO or a DP.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"              # before numpy / sklearn / qiskit load
os.environ["FUEL_JOBS"] = "1"

import contextlib
import importlib.util
import io
import itertools
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "fleet")]
warnings.filterwarnings("ignore")

import joblib
import numpy as np

import config
import demo
import generate_data
import optimizer
import run as runner
import train_model
from core import results, stats
from core.physics_model import PhysicsResidualRegressor
from experiments import (audit_seeds, benchmark, joint_speed_route, quantum_gate,  # noqa: F401
                         qubo_route, qubo_speed, speed_compare)
from solvers import backends, ga, gate, gsa, qgso
from tools import make_report, manifest

import fleet_config as fc
import benchmark_fleet
import genetic
import hybrid as hyb
import matheuristic as mh
import oracle
import quantum_qaoa as qq
import qubo_layer as ql
import run_fleet
import scenario as scen_mod

VERBOSE = "-v" in sys.argv
TRUTH = generate_data.TruthModel()

# Send every artefact to a temp folder: the project's real files stay untouched.
TMP = Path(tempfile.mkdtemp(prefix="fuel_smoke_"))
config.DATA_DIR, config.MODELS_DIR = TMP / "data", TMP / "models"
config.DATA_CSV = config.DATA_DIR / "ship_fuel_data.csv"
config.MODEL_PATH = config.MODELS_DIR / "ship_fuel_model.joblib"
config.META_PATH = config.MODELS_DIR / "model_meta.joblib"
config.OUTPUTS_DIR = TMP / "outputs"
fc.OUTPUTS_DIR, fc.CACHE_DIR = TMP / "outputs" / "fleet", TMP / "cache"


def quiet(fn, *args, **kwargs):
    """Call fn, hiding what it prints unless -v was given."""
    with contextlib.redirect_stdout(sys.stdout if VERBOSE else io.StringIO()):
        return fn(*args, **kwargs)


def run_cli(module, *argv):
    """Call a script's main() as if it had been given these arguments."""
    saved, sys.argv = sys.argv, [module.__name__, *argv]
    try:
        quiet(module.main)
    finally:
        sys.argv = saved


def all_bitstrings(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)


def brute_force(qubo):
    """Exact minimum of a small QUBO -> (sample dict, energy)."""
    names, Q = gsa._qubo_matrix(qubo)
    X = all_bitstrings(len(names))
    e = np.einsum("ni,ij,nj->n", X, Q, X)
    i = int(np.argmin(e))
    return {nm: int(X[i, k]) for k, nm in enumerate(names)}, float(e[i])


def storm_costs(rows, cols):
    """Route-cell fuel costs from the TRUE physics (one centred storm)."""
    waves = qubo_route.build_weather(rows, cols)
    base = config.base_scenario_for("Offshore_Patrol")
    return qubo_route.cell_costs(TRUTH, base, waves, 600.0, 16.0)


# --------------------------------------------------------------------------- #
# Phase 1: data -> all models -> save -> optimise -> plot                       #
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end():
    quiet(demo.main, 400)                           # 400 rows: seconds, not minutes
    assert config.MODEL_PATH.exists() and config.META_PATH.exists()
    assert (config.OUTPUTS_DIR / "voyage_fuel_vs_speed.png").exists()
    pipe = quiet(optimizer.load_model)              # same sklearn -> loads, no retrain
    rates = optimizer.predict_rates(pipe, config.DEFAULT_SCENARIO, [8.0, 12.0, 18.0])
    assert rates.shape == (3,) and np.all(np.diff(rates) > 0), rates


def test_load_model_retrains_when_unusable():
    model, meta = joblib.load(config.MODEL_PATH), joblib.load(config.META_PATH)
    calls = []

    def fake_training():                            # stands in for a real retrain
        calls.append(1)
        joblib.dump(meta, config.META_PATH)
        joblib.dump(model, config.MODEL_PATH)

    real, train_model.main = train_model.main, fake_training
    try:
        joblib.dump({**meta, "sklearn_version": "0.0"}, config.META_PATH)
        quiet(optimizer.load_model)                 # saved by "another" sklearn
        assert len(calls) == 1
        config.MODEL_PATH.write_bytes(b"not a pickle")
        quiet(optimizer.load_model)                 # unreadable file
        assert len(calls) == 2
        quiet(optimizer.load_model)                 # healthy again: no retrain
        assert len(calls) == 2
    finally:
        train_model.main = real


# --------------------------------------------------------------------------- #
# The model that fixes the sawtooth, and the metric that exposes it             #
# --------------------------------------------------------------------------- #
def test_truth_and_regret_numbers():
    """The figures quoted in README/journal for the default 600 nm, 90 h voyage."""
    scn = config.DEFAULT_SCENARIO
    xs = np.linspace(4.0, 25.0, 2101)
    economic = xs[np.argmin(optimizer.voyage_fuels(TRUTH, scn, xs, 600.0))]
    assert abs(economic - 6.29) < 0.02, economic
    assert abs(optimizer.voyage_fuel(TRUTH, scn, 600.0 / 90.0, 600.0) - 18.43) < 0.01
    assert abs(optimizer.voyage_fuel(TRUTH, scn, 8.4, 600.0) - 20.08) < 0.01
    assert np.allclose(optimizer.regret_panel(TRUTH, TRUTH), 0.0)
    res = optimizer.optimize_speed(TRUTH, scn, 600.0, 90.0, truth=TRUTH)
    assert abs(res["opt_speed"] - 600.0 / 90.0) < 0.05 and res["regret_pct"] < 0.01


def test_physics_model_recovers_the_cubic_law():
    train = generate_data.generate(3000, seed=1)
    test = generate_data.generate(1000, seed=2)
    X, y = train[config.FEATURE_COLUMNS], train[config.TARGET]
    white = PhysicsResidualRegressor().fit(X, y)    # white box only: instant
    coef = white.coefficients()
    assert abs(coef["speed_exponent"] - 3.0) < 0.1, coef
    assert abs(coef["hotel"] - generate_data.HOTEL_COEF) < 0.005, coef
    truth = TRUTH.predict(test[config.FEATURE_COLUMNS])
    err = np.abs(white.predict(test[config.FEATURE_COLUMNS]) / truth - 1.0) * 100
    assert err.mean() < 3.0, err.mean()             # vs the noise-free physics
    white_regret = optimizer.regret_panel(white, TRUTH)
    assert white_regret.max() < 2.0, white_regret

    boosted = train_model.build_pipeline(train_model.HistGradientBoostingRegressor(
        random_state=0)).fit(X, y)                  # the sawtooth model, same data
    tree_regret = optimizer.regret_panel(boosted, TRUTH)
    print(f"      [3,000 rows] fitted exponent {coef['speed_exponent']:.3f}, hotel "
          f"{coef['hotel']:.4f} | error vs true physics {err.mean():.2f}% | decision "
          f"regret mean/max: physics {white_regret.mean():.2f}/{white_regret.max():.2f}%"
          f", boosting {tree_regret.mean():.2f}/{tree_regret.max():.2f}%")


# --------------------------------------------------------------------------- #
# Solvers                                                                       #
# --------------------------------------------------------------------------- #
def test_ga_and_backends():
    q = {("x0", "x0"): -1.0, ("x1", "x1"): -1.0, ("x2", "x2"): 1.0, ("x0", "x1"): 2.0}
    for solve in (ga.bga_qubo, gsa.bgsa_qubo, qgso.qgso_qubo):
        assert abs(solve(q, n_agents=30, iters=60, seed=1)[1] + 1.0) < 1e-9, solve
    x, f, hist = ga.ga_minimize(lambda X: ((X - 3.0) ** 2).sum(axis=1),
                                [(0.0, 10.0)] * 3, n_agents=30, iters=80,
                                seed=0, vectorized=True)
    assert f < 0.05 and len(hist["best"]) == 80, (x, f)

    cost = storm_costs(3, 4)
    qubo, _ = qubo_route.build_route_qubo(cost, 1, 1, presolve=True)
    _, dp_fuel = qubo_route.exact_best_dp(cost, 1, 1)
    for name, reads in (("neal", 20), ("ga", 3), ("gsa", 3), ("tabu", 1)):
        sampler, _, _ = backends.make_sampler(name)
        t = time.perf_counter()
        ss = backends.sample_qubo(sampler, name, qubo, num_reads=reads, seed=1)
        took = time.perf_counter() - t
        path = qubo_route.decode_route(dict(ss.first.sample), 3, 4, 1, 1)
        assert path is not None, name
        assert abs(qubo_route.path_fuel(path, cost) - dp_fuel) < 1e-9, name
        if name == "tabu":                          # stops on a 20 ms clock, not a count
            assert took < 1.0, took


# --------------------------------------------------------------------------- #
# QUBO formulations checked against exhaustive enumeration                      #
# --------------------------------------------------------------------------- #
def test_route_qubo_and_presolve_are_exact():
    cost = storm_costs(3, 4)
    dp_path, dp_fuel = qubo_route.exact_best_dp(cost, 1, 1)
    for presolve, n_vars in ((False, 12), (True, 6)):
        qubo, info = qubo_route.build_route_qubo(cost, 1, 1, presolve=presolve)
        assert info["n_vars"] == n_vars, info
        sample, energy = brute_force(qubo)
        path = qubo_route.decode_route(sample, 3, 4, 1, 1)
        assert path is not None
        assert abs(qubo_route.path_fuel(path, cost) - dp_fuel) < 1e-9
        assert abs(energy + info["offset"] - dp_fuel) < 1e-6, (energy, info["offset"])
    assert qubo_route.build_route_qubo(storm_costs(5, 7), 2, 2, presolve=True)[1]["n_vars"] == 21
    # a path leaving from the wrong port must be rejected, not priced
    wrong = {f"x_{c}_{r}": 1 for c, r in enumerate([0, 0, 1, 1])}
    assert qubo_route.decode_route(wrong, 3, 4, 1, 1) is None
    assert qubo_route.decode_route(
        {f"x_{c}_{r}": 1 for c, r in enumerate(dp_path)}, 3, 4, 1, 1) == dp_path


def test_joint_speed_route():
    jr = joint_speed_route
    rows, cols, speeds = 2, 3, np.array([10.0, 18.0])
    waves = qubo_route.build_weather(rows, cols)
    hours = 600.0 / cols / speeds
    fuel = (qubo_route.cell_rates(TRUTH, config.base_scenario_for("Offshore_Patrol"),
                                  waves, speeds) * hours / 24.0)
    start = end = 1
    plans = [p for p in itertools.product(
        itertools.product(range(rows), range(len(speeds))), repeat=cols)
        if p[0][0] == start and p[-1][0] == end]    # every valid plan (rows adjacent)

    def best(deadline, lam=0.0):
        ok = [p for p in plans if jr.plan_totals(p, fuel, hours)[1] <= deadline + 1e-9]
        return min(ok, key=lambda p: jr.plan_totals(p, fuel, hours)[0]
                   + lam * jr.plan_totals(p, fuel, hours)[1])

    deadline = 600.0 / 13.0                         # binds: cannot crawl all the way
    exact = jr.exact_joint(fuel, hours, start, end, deadline)
    want = jr.plan_totals(best(deadline), fuel, hours)
    assert np.allclose(jr.plan_totals(exact, fuel, hours), want), (exact, want)
    assert jr.exact_joint(fuel, hours, start, end, 1.0) is None     # impossible

    # the QUBO at lam = 0 is the unconstrained fuel optimum
    sample, _ = brute_force(jr.build_joint_qubo(fuel, hours, start, end, 0.0))
    got = jr.decode_joint(sample, fuel.shape, start, end)
    assert np.isclose(jr.plan_totals(got, fuel, hours)[0],
                      jr.plan_totals(best(np.inf), fuel, hours)[0])

    # price-of-time loop, first with an exact inner solver, then with neal
    plan, _ = jr.solve_lagrangian(fuel, hours, deadline,
                                  lambda lam: list(best(np.inf, lam)))
    tonnes, time_h = jr.plan_totals(plan, fuel, hours)
    assert time_h <= deadline + 1e-9 and tonnes >= want[0] - 1e-9
    print(f"      [joint 2x3x2] exact {want[0]:.2f} t; price-of-time loop "
          f"{tonnes:.2f} t ({(tonnes / want[0] - 1) * 100:+.1f}%)")
    real, optimizer.load_model = optimizer.load_model, lambda: TRUTH
    try:
        run_cli(jr, "--rows", "2", "--cols", "3", "--levels", "2", "--num-reads", "40",
                "--sweeps", "200", "--deadline", f"{deadline:.3f}")
    finally:
        optimizer.load_model = real
    assert list(config.OUTPUTS_DIR.glob("joint_2x3x2*.png"))


# --------------------------------------------------------------------------- #
# The scripts themselves, tiny settings, true physics as the model             #
# --------------------------------------------------------------------------- #
def test_scripts_run():
    real, optimizer.load_model = optimizer.load_model, lambda: TRUTH
    try:
        run_cli(qubo_speed, "--levels", "12", "--num-reads", "20")
        run_cli(qubo_route, "--rows", "3", "--cols", "4", "--num-reads", "30", "--presolve")
        run_cli(speed_compare)
        run_cli(speed_compare, "--legs", "3", "--storms", "1", "--deadline", "45")
        gate = ("--trials", "1", "--reps", "1", "--maxiter", "15",
                "--gsa-agents", "20", "--gsa-iters", "30")
        run_cli(quantum_gate, "--problem", "speed", "--levels", "6", *gate)
        run_cli(quantum_gate, "--problem", "route", "--rows", "3", "--cols", "4",
                "--presolve", *gate)                # 6 qubits: QAOA + VQE included
    finally:
        optimizer.load_model = real
    made = {p.name.split("_")[0] for p in config.OUTPUTS_DIR.glob("*.png")}
    assert {"speed", "route", "compare"} <= made, made
    assert len(list(config.OUTPUTS_DIR.glob("speed_compare_*.png"))) == 2


def test_benchmark_plumbing():
    real, optimizer.load_model = optimizer.load_model, lambda: TRUTH
    saved = benchmark.CLASSICAL[:], {k: dict(v) for k, v in benchmark.BUDGET.items()}
    try:
        benchmark.CLASSICAL[:] = ["neal", "ga"]     # tabu's wall-clock budget is too slow here
        benchmark.BUDGET["ga"] = dict(restarts=2, n_agents=30, iters=40)
        benchmark.BUDGET["qaoa"] = dict(reps=1, maxiter=10)
        problem = benchmark.make_problem(3, 4, 1, seed=3, presolve=True)
        assert problem[2] == 6
        # two worker processes: the same path --jobs N takes on the workstation
        for solver in ("ga", "qaoa"):
            runs = benchmark.run_cell(solver, problem, [0, 1], jobs=2)
            assert len(runs) == 2 and all(dt > 0 for _, dt in runs)
        size_res, order = quiet(benchmark.study_size, [(3, 4), (4, 4)], 2, 2, 0, 1, 1, True)
        assert [n for n, _ in order] == [6, 6]      # two grids, same size: no collision
        assert size_res["3x4"]["neal"]["n"] == 4 and size_res["3x4"]["neal"]["opt_pct"] == 100
        pen_res = quiet(benchmark.study_penalty, 3, 4, 1, [1, 10], 2, 1, 1, True)
        benchmark.plot_size(size_res, order, TMP / "bench_size.png")
        benchmark.plot_penalty(pen_res, 3, 4, TMP / "bench_penalty.png")
        benchmark.write_markdown(size_res, order, pen_res, "3x4", 2, 2, 2, True,
                                 TMP / "BENCHMARK.md")
        assert "full-energy evaluations" in (TMP / "BENCHMARK.md").read_text(encoding="utf-8")
    finally:
        optimizer.load_model = real
        benchmark.CLASSICAL[:] = saved[0]
        benchmark.BUDGET.update(saved[1])


# --------------------------------------------------------------------------- #
# Fleet: the subset DP must agree with brute-force enumeration                  #
# --------------------------------------------------------------------------- #
def test_fleet_exact_dp_matches_enumeration():
    real, optimizer.load_model = optimizer.load_model, lambda: TRUTH
    try:
        scn = scen_mod.build_scenario(0, n_ships=2, n_missions=4)
        # one ship class -> one set of oracle tables (keeps this check light)
        scn.ships[1].ship_class, scn.ships[1].speeds = scn.ships[0].ship_class, scn.ships[0].speeds
        tabs = oracle.get_tables(scn, verbose=False)
        assert len(list(fc.CACHE_DIR.glob("tables_seed0_*_m*.npz"))) == 1
        tabs = oracle.get_tables(scn, verbose=False)        # second call: from cache
        assert len(list(fc.CACHE_DIR.glob("tables_seed0_*.npz"))) == 1
        ev = mh.Evaluator(scn, tabs)

        plan_enum, fuel_enum, _ = mh.exhaustive(ev)
        plan_dp, fuel_dp = mh.exact_dp(ev)
        assert np.isfinite(fuel_enum), "instance infeasible -- pick another seed"
        assert abs(fuel_dp - fuel_enum) < 1e-6, (fuel_dp, fuel_enum)
        assert abs(ev.total(plan_dp) - fuel_dp) < 1e-6, (plan_dp, plan_enum)

        # every solver returns a plan the exact evaluator accepts, never below optimum
        _, f_alns = mh.alns(ev, n_iters=300, seed=1)
        _, f_warm = mh.alns(ev, n_iters=50, seed=1, start=plan_dp)  # warm start keeps it
        assert abs(f_warm - fuel_dp) < 1e-6
        _, f_hyb, _ = hyb.solve_hybrid(ev, num_reads=50, seed=1)
        q = ql.solve_fleet_qubo(scn, tabs, ev, num_reads=50)
        _, f_qaoa, info = qq.solve_assignment_qaoa(ev, reps=1, maxiter=10, verbose=False)
        assert info["n_qubits"] == 8
        # the assignment proposers share one scorer type; 2^4 = 16 assignments exist
        p_ga, f_ga, _ = genetic.solve_ga(ev, budget=16, pop=6, seed=1)
        p_v2, f_v2, info2 = hyb.solve_hybrid_cuts(ev, num_reads=50, budget=16, seed=1)
        p_rd, f_rd, sc = hyb.solve_random(ev, budget=16, seed=1)
        assert sc.n_scored <= 16 and all(a >= b for a, b in zip(sc.trace, sc.trace[1:]))
        for plan, f in ((p_ga, f_ga), (p_v2, f_v2), (p_rd, f_rd)):
            assert f >= fuel_enum - 1e-6 and abs(ev.total(plan) - f) < 1e-6
        assert min(f_ga, f_v2, f_rd) - fuel_enum < 1e-6         # 16 of 16: someone hits it
        for f in (f_alns, f_hyb, f_qaoa, q["true_fuel"]):
            assert f >= fuel_enum - 1e-6
        # repair turns an everything-on-one-ship proposal into a feasible plan or None
        fixed = hyb.repair(ev, {scn.ships[0].id: [0, 1, 2, 3], scn.ships[1].id: []})
        assert fixed is None or all(hyb.bundle_fuel(ev, s, fixed[s.id]) < float("inf")
                                    for s in scn.ships)
        # a 3-way product penalty: only x=y=z=1 pays it, the auxiliary bit never cheats
        qd, aux = {}, {}
        hyb._add_product_penalty(qd, ["a", "b", "c"], 5.0, aux)
        best = {}
        for bits in itertools.product((0, 1), repeat=4):
            v = dict(zip(["a", "b", "c", "aux_0"], bits))
            e = sum(c * v[i] * v[j] for (i, j), c in qd.items())
            k = bits[:3]
            best[k] = min(best.get(k, np.inf), e)
        assert best[(1, 1, 1)] == 5.0 and all(e == 0 for k, e in best.items() if k != (1, 1, 1))
        run_fleet.plot_plan(scn, ev, plan_dp, "smoke", fc.OUTPUTS_DIR / "fleet_plan.png")
        print(f"      [fleet 2x4] exact {fuel_enum:.2f} t | alns {f_alns:.2f} | hybrid "
              f"v1 {f_hyb:.2f} v2 {f_v2:.2f} | GA {f_ga:.2f} | random {f_rd:.2f} | "
              f"QAOA(aer) {f_qaoa:.2f} | qubo+repair {q['true_fuel']:.2f}")
    finally:
        optimizer.load_model = real


def test_fleet_horizon_and_benchmark():
    # the default instance is unchanged by the horizon rule; bigger ones get room
    assert fc.horizon_slots(3, 6) == fc.horizon_slots(2, 4) == 36
    assert fc.horizon_slots(4, 12) == 47 and fc.horizon_slots(5, 16) == 58
    assert scen_mod.build_scenario(1, n_ships=4, n_missions=12).waves.shape[0] == 47
    real, optimizer.load_model = optimizer.load_model, lambda: TRUTH
    try:                                            # the benchmark end to end, tiny
        run_cli(benchmark_fleet, "--seeds", "1", "--sizes", "2x3,2x4", "--iters", "50",
                "--num-reads", "20", "--budget", "12", "--pop", "4", "--checkpoints", "5,12")
    finally:
        optimizer.load_model = real
    d = json.loads((fc.OUTPUTS_DIR / "results" / "fleet_benchmark_2x4.json").read_text())
    assert d["n_valid"] in (0, 1) and (fc.OUTPUTS_DIR / "results" / "fleet_scale.json").exists()
    if d["n_valid"]:
        assert set(d["summary"]) == set(benchmark_fleet.SOLVERS)
        assert all(r["gap"] is None or r["gap"] >= -1e-6
                   for k, r in d["instances"][0].items() if isinstance(r, dict))


def test_stats_and_results():
    lo, hi = stats.wilson(8, 10)
    assert 0.48 < lo < 0.5 and 0.94 < hi < 0.96, (lo, hi)
    assert stats.wilson(10, 10)[1] == 1.0
    assert stats.tts99(1.0, 2.0) == 2.0 and stats.tts99(0.0, 2.0) == float("inf")
    assert abs(stats.tts99(0.5, 1.0) - np.log(0.01) / np.log(0.5)) < 1e-9
    assert stats.bootstrap_ci([3.0]) == (3.0, 3.0)
    lo, hi = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert 1.0 <= lo < 2.5 < hi <= 4.0
    t = stats.paired([0.0] * 10, [0.0] * 8 + [1.0, None])   # the README's 10/10 vs 8/10
    assert t["only_a_opt"] == 2 and t["only_b_opt"] == 0 and t["mcnemar_p"] == 0.5
    path = results.save("smoke_roundtrip", {"a": np.float64(1.5), "b": float("inf"),
                                            "c": np.arange(3), "d": (1, 2)})
    assert json.loads(path.read_text()) == {"a": 1.5, "b": None, "c": [0, 1, 2], "d": [1, 2]}
    # every measured bitstring is a candidate, including those seen only once --
    # MinimumEigenOptimizer squared the probabilities and dropped all of those
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    qp = gate.qubo_to_qp({(f"x{i}", f"x{i}"): 1.0 for i in range(10)})
    qaoa = QAOA(sampler=gate.make_sampler(10, seed=1)[0], optimizer=COBYLA(maxiter=2),
                transpiler=gate.pass_manager())
    samples, _ = gate.sample_qp(qaoa, qp)
    assert any(p == 1 / gate.SHOTS for _, p, _ in samples)
    assert abs(sum(p for _, p, _ in samples) - 1.0) < 1e-9
    assert [s[2] for s in samples] == sorted(s[2] for s in samples)
    with gate_device("CPU"):
        assert gate.device_for(30) == "CPU"
    with gate_device("auto"):
        assert gate.device_for(4) == "CPU"                  # small states stay on the CPU


@contextlib.contextmanager
def gate_device(value):
    saved = os.environ.get("FUEL_QDEVICE")
    os.environ["FUEL_QDEVICE"] = value
    try:
        yield
    finally:
        os.environ.pop("FUEL_QDEVICE") if saved is None else os.environ.update(FUEL_QDEVICE=saved)


def test_runner_and_dashboard():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        saved, sys.argv = sys.argv, ["run.py"]
        try:
            runner.main()                               # the listing
        finally:
            sys.argv = saved
    assert all(name in out.getvalue() for name in ("demo", "run_fleet", "dashboard"))
    for kind, target, _ in runner.TARGETS.values():     # every target exists
        base = ROOT / "fleet" if kind == "fleet" else ROOT
        assert (kind == "module" and importlib.util.find_spec(target)) or \
               (base / target).exists(), target

    from streamlit.testing.v1 import AppTest            # the app renders all four tabs
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    at.button[0].click().run()                          # "Optimise speed" with defaults
    assert not at.exception, at.exception
    assert any("Optimal speed" in m.label for m in at.metric)


def test_report_from_this_run():
    """Every JSON the checks above wrote, turned into RESULTS.md + README blocks."""
    saved, sys.argv = sys.argv, ["manifest", str(config.OUTPUTS_DIR)]
    try:
        quiet(manifest.main)
    finally:
        sys.argv = saved
    rep = make_report.Report(config.OUTPUTS_DIR, config.OUTPUTS_DIR)
    assert "cores" in make_report.render(rep, "run_info")
    assert {"phase1", "fleet_benchmark_2x4"} <= set(rep.R), sorted(rep.R)
    for name in make_report.SECTIONS:
        make_report.render(rep, name)                   # every section builds or is None
    assert "Physics + Boosting" in make_report.render(rep, "phase1_models")
    readme = TMP / "README.md"
    readme.write_text("intro\n<!-- results:phase1_models -->\nold\n"
                      "<!-- /results:phase1_models -->\nend\n", encoding="utf-8")
    assert make_report.update_readme(readme, config.OUTPUTS_DIR) == ["phase1_models"]
    text = readme.read_text(encoding="utf-8")
    assert "\nold\n" not in text and "Test R²" in text and text.endswith("end\n")


if __name__ == "__main__":
    start = time.perf_counter()
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        t = time.perf_counter()
        fn()
        print(f"ok  {name:<46}{time.perf_counter() - t:6.1f}s")
    print(f"\nall {len(tests)} checks passed in {time.perf_counter() - start:.0f}s "
          f"(artefacts in {TMP})")
