"""
quantum_gate.py  --  Phase 2c: the SAME QUBO on GATE-MODEL quantum algorithms,
                     WITH a head-to-head comparison (paths/speeds + metrics + plot).

Phase 2a/2b solved our QUBOs with neal (classical annealing) -- the form a
D-Wave *annealer* uses. This file proves the other half of the story: the very
same QUBO also runs on **gate-model** quantum computers (IBM-style) via the two
standard variational algorithms:

  * QAOA  -- Quantum Approximate Optimization Algorithm (purpose-built for QUBO)
  * VQE   -- Variational Quantum Eigensolver (general ground-state finder)

Both convert the QUBO -> an Ising Hamiltonian (s = 2x - 1) and find its lowest-
energy state with a parameterised quantum circuit + a classical optimiser in the
loop. We run on Aer's local statevector simulator (FREE, offline; on the GPU from
22 qubits when the GPU build is installed -- see solvers/gate.py). The same code
runs on real IBM hardware by swapping the sampler for a Runtime backend.

WHAT THIS SCRIPT SHOWS
----------------------
  * the speed / path each solver chose (drawn on one comparison plot)
  * metrics:  fuel, optimality gap vs exact, feasibility, SUCCESS RATE over
    repeated runs (QAOA/VQE are stochastic), and wall-clock time
  * QAOA-CVaR / VQE-CVaR: the same circuits trained on the best 25% of shots
    instead of the mean (solvers/gate.py) -- does that raise the success rate?

  ** Gate-model on a simulator uses ONE QUBIT PER BINARY VARIABLE, cost ~ 2^qubits.
     ~20 qubits on CPU, ~28 on the 48 GB GPU: few speed levels or a small grid. **

Run (from the project root):
    python -m experiments.quantum_gate --problem speed --levels 8 --trials 5
    python -m experiments.quantum_gate --problem route --rows 3 --cols 4 --storms 1 --seed 2 --trials 5
    python -m experiments.quantum_gate --problem route --rows 4 --cols 6 --presolve   # 24 cells -> 14 qubits
    python -m experiments.quantum_gate --problem route --rows 5 --cols 7 --presolve   # 35 cells -> 21 qubits
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

import config
import optimizer
from core import results
from experiments import qubo_route, qubo_speed
from solvers import backends, ga, gate, gsa

qubo_to_qp = gate.qubo_to_qp            # kept here: benchmark/audit import it from this module


# --------------------------------------------------------------------------- #
# Solvers (each returns a {varname: 0/1} dict so decode() is uniform)          #
# --------------------------------------------------------------------------- #
def _pm_sampler(n_qubits, seed=None):
    # Seeding the sampler (shot RNG) + algorithm_globals (COBYLA start point)
    # makes each gate-model run fully reproducible, independent of call order.
    return gate.pass_manager(), gate.make_sampler(n_qubits, seed)[0]


def run_neal(qubo, seed=42):
    sampler, _, _ = backends.make_sampler("neal")
    ss = backends.sample_qubo(sampler, "neal", qubo, num_reads=200, seed=seed)
    return {k: int(v) for k, v in dict(ss.first.sample).items()}


def run_gsa(qubo, seed=0, n_agents=100, iters=400, restarts=20):
    """Binary Gravitational Search Algorithm on the QUBO (classical metaheuristic).

    Uses `restarts` independent swarms (keeping the best) so GSA gets a budget
    comparable to neal's num_reads -- a FAIR fight. With one swarm only, GSA
    looks far worse than it is on constrained QUBOs."""
    samp = gsa.GSASampler(n_agents=n_agents, iters=iters)
    ss = samp.sample_qubo(qubo, num_reads=restarts, seed=seed)
    return {k: int(v) for k, v in dict(ss.first.sample).items()}


def run_ga(qubo, seed=0, n_agents=100, iters=400, restarts=20):
    """Binary Genetic Algorithm on the QUBO -- same restart budget as run_gsa,
    so the two population methods are compared like for like."""
    samp = ga.GASampler(n_agents=n_agents, iters=iters)
    ss = samp.sample_qubo(qubo, num_reads=restarts, seed=seed)
    return {k: int(v) for k, v in dict(ss.first.sample).items()}


def run_tabu(qubo, seed=0):
    """Tabu search on the QUBO (classical metaheuristic)."""
    sampler, _, _ = backends.make_sampler("tabu")
    ss = backends.sample_qubo(sampler, "tabu", qubo, num_reads=200, seed=seed)
    return {k: int(v) for k, v in dict(ss.first.sample).items()}


def _path_to_vardict(path):
    """Turn an exact-DP route path into a one-hot var_dict (for big-grid exact ref)."""
    return {f"x_{c}_{r}": 1 for c, r in enumerate(path)}


def run_qaoa(qp, reps, maxiter, seed=42, cvar=None):
    """cvar=None: minimise the mean energy (standard); cvar=alpha: CVaR_alpha."""
    algorithm_globals.random_seed = seed
    pm, sampler = _pm_sampler(qp.get_num_vars(), seed)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter),
                reps=reps, transpiler=pm, aggregation=cvar)
    samples, _ = gate.sample_qp(qaoa, qp)        # the best of every measured bitstring
    return gate.to_vardict(qp, samples[0][0])


def run_vqe(qp, n_qubits, reps, maxiter, seed=42, cvar=None):
    algorithm_globals.random_seed = seed
    pm, sampler = _pm_sampler(n_qubits, seed)
    ansatz = pm.run(TwoLocal(n_qubits, "ry", "cz", reps=reps, entanglement="linear"))
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter),
                      aggregation=cvar)
    samples, _ = gate.sample_qp(vqe, qp)
    return gate.to_vardict(qp, samples[0][0])


# --------------------------------------------------------------------------- #
# Evaluate a solver over `trials` runs -> metrics record                       #
# --------------------------------------------------------------------------- #
def evaluate(label, run_once, decode, exact_fuel, trials):
    """Run a solver `trials` times; collect feasibility, quality, success, time."""
    feasible = 0
    successes = 0          # matched the exact optimum
    fuels = []
    best_fuel = float("inf")
    best_sol = None
    best_str = "-"
    t0 = time.perf_counter()
    for i in range(trials):
        sol, fuel, s = decode(run_once(i))
        if sol is not None and fuel == fuel:          # feasible & not NaN
            feasible += 1
            fuels.append(fuel)
            if exact_fuel == exact_fuel and abs(fuel - exact_fuel) < 1e-6:
                successes += 1
            if fuel < best_fuel:
                best_fuel, best_sol, best_str = fuel, sol, s
    avg_time = (time.perf_counter() - t0) / trials

    gap = (float("nan") if best_fuel == float("inf") or exact_fuel != exact_fuel
           else (best_fuel - exact_fuel) / exact_fuel * 100)
    return {
        "label": label,
        "trials": trials,
        "feasible": feasible,
        "successes": successes,
        "best_fuel": (best_fuel if best_fuel != float("inf") else float("nan")),
        "mean_fuel": (float(np.mean(fuels)) if fuels else float("nan")),
        "gap": gap,
        "avg_time": avg_time,
        "best_sol": best_sol,
        "best_str": best_str,
    }


# --------------------------------------------------------------------------- #
# Problem builders                                                             #
# --------------------------------------------------------------------------- #
def build_speed_problem(args):
    scenario = dict(config.DEFAULT_SCENARIO)
    pipe = optimizer.load_model()
    qubo, speeds, fuels, feasible, info = qubo_speed.build_speed_qubo(
        pipe, scenario, args.distance, args.deadline, args.levels)

    def decode(var_dict):
        sel = [i for i in range(len(speeds)) if round(var_dict.get(f"x_{i}", 0)) == 1]
        if len(sel) != 1:
            return None, float("nan"), f"invalid one-hot ({len(sel)} set)"
        i = sel[0]
        return float(speeds[i]), float(fuels[i]), f"{speeds[i]:.2f} kn"

    ctx = {"speeds": speeds, "fuels": fuels, "feasible": feasible,
           "min_speed": info["deadline_min_speed"]}
    meta = {"kind": "speed", "n_qubits": args.levels,
            "title": f"SPEED - {args.levels} levels, {args.distance:.0f} nm, "
                     f"{args.deadline:.0f} h deadline", "unit": "fuel (t)"}
    return qubo, decode, meta, ctx


def build_route_problem(args):
    base = config.base_scenario_for(args.ship_type)
    pipe = optimizer.load_model()
    storms = (qubo_route.random_storms(args.rows, args.cols, args.storms, args.seed)
              if args.storms > 0 else None)
    waves = qubo_route.build_weather(args.rows, args.cols, storms)
    cost = qubo_route.cell_costs(pipe, base, waves, args.distance, args.cruise)
    start_row = end_row = args.rows // 2
    qubo, info = qubo_route.build_route_qubo(
        cost, start_row, end_row,
        penalty_mult=getattr(args, "penalty_mult", 10.0),
        presolve=getattr(args, "presolve", False))

    def decode(var_dict):
        sample = {k: int(round(v)) for k, v in var_dict.items()}
        path = qubo_route.decode_route(sample, args.rows, args.cols,
                                       start_row, end_row)
        if path is None:
            return None, float("nan"), "invalid path"
        return path, qubo_route.path_fuel(path, cost), str(path)

    naive = [int(round(start_row + (end_row - start_row) * c / (args.cols - 1)))
             for c in range(args.cols)]
    # Exact optimum via DP -- scales to ANY grid (NumPy eigensolver would die at 2^n).
    dp_path, dp_fuel = qubo_route.exact_best_dp(cost, start_row, end_row)
    ctx = {"cost": cost, "waves": waves, "start_row": start_row,
           "end_row": end_row, "naive": naive,
           "naive_fuel": qubo_route.path_fuel(naive, cost),
           "dp_path": dp_path, "dp_fuel": dp_fuel}
    meta = {"kind": "route", "n_qubits": info["n_vars"],   # fewer with presolve
            "title": f"ROUTE - {args.rows}x{args.cols} grid, "
                     f"{args.storms or 1} storm(s)", "unit": "fuel (t)"}
    return qubo, decode, meta, ctx


# --------------------------------------------------------------------------- #
# Comparison plots                                                            #
# --------------------------------------------------------------------------- #
STYLE = {                       # consistent colours/markers per solver
    "Exact":  ("#16A34A", "*", 22),
    "neal":   ("#1f77b4", "o", 12),
    "Tabu":   ("#0891B2", "P", 11),
    "GSA":    ("#7C3AED", "v", 11),
    "GA":     ("#65A30D", "^", 11),
    "QAOA-CVaR": ("#B45309", "d", 11),   # before "QAOA": labels match by prefix
    "VQE-CVaR":  ("#9F1239", "X", 10),
    "QAOA":   ("#E8772E", "D", 11),
    "VQE":    ("#D62728", "s", 10),
}


def _style_for(label):
    for key, v in STYLE.items():
        if label.startswith(key):
            return v
    return ("#666666", "x", 10)


def plot_speed_comparison(ctx, records, meta, out):
    speeds, fuels = ctx["speeds"], ctx["fuels"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(speeds, fuels, "-", color="#bbbbbb", lw=1.5, zorder=1,
            label="Candidate speeds (QUBO levels)")
    ax.scatter(speeds, fuels, color="#dddddd", s=30, zorder=1)
    ax.axvspan(speeds[0], ctx["min_speed"], color="red", alpha=0.08,
               label="Too slow (misses deadline)")

    for r in records:
        if r["best_sol"] is None:
            continue
        color, marker, size = _style_for(r["label"])
        ax.plot(r["best_sol"], r["best_fuel"], marker=marker, color=color,
                ms=size, mec="black", mew=0.8, ls="none", zorder=5,
                label=f"{r['label']}: {r['best_str']}, {r['best_fuel']:.1f} t")

    ax.set_xlabel("Speed (knots)")
    ax.set_ylabel("Total voyage fuel (tonnes)")
    ax.set_title(f"Solver comparison — {meta['title']}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_route_comparison(ctx, records, meta, out):
    waves = ctx["waves"]
    n_rows, n_cols = waves.shape
    cols = list(range(n_cols))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * n_cols), max(4.5, 1.0 * n_rows)))
    im = ax.imshow(waves, origin="lower", aspect="auto", cmap="YlOrRd",
                   extent=[-0.5, n_cols - 0.5, -0.5, n_rows - 0.5], vmin=1, vmax=8)
    fig.colorbar(im, ax=ax, label="Wave height (m)")

    # naive reference
    ax.plot(cols, ctx["naive"], "--", color="black", lw=1.5, alpha=0.7,
            label=f"Naive straight ({ctx['naive_fuel']:.1f} t)")

    # each solver's path, with a tiny vertical offset so overlaps stay visible
    offset = -0.12
    for r in records:
        if r["best_sol"] is None:
            continue
        color, marker, _ = _style_for(r["label"])
        gap_txt = ("" if r["label"].startswith("Exact")
                   else f", +{r['gap']:.0f}%" if r["gap"] == r["gap"] and r["gap"] > 0.05
                   else ", optimal")
        ax.plot(cols, [p + offset for p in r["best_sol"]], marker=marker,
                color=color, lw=2.2, ms=8, mec="black", mew=0.5, alpha=0.95,
                label=f"{r['label']} ({r['best_fuel']:.1f} t{gap_txt})")
        offset += 0.08

    ax.scatter([0, n_cols - 1], [ctx["start_row"], ctx["end_row"]], s=240,
               marker="*", color="lime", edgecolor="black", zorder=6,
               label="Start / End port")
    ax.set_xlabel("Stage (west → east)")
    ax.set_ylabel("Track (south → north)")
    ax.set_title(f"Solver comparison — {meta['title']}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare annealing (neal) vs gate-model (QAOA/VQE) on one QUBO.")
    p.add_argument("--problem", choices=["speed", "route"], default="speed")
    p.add_argument("--trials", type=int, default=5,
                   help="repeat runs to measure success rate (QAOA/VQE are stochastic)")
    # speed
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--distance", type=float, default=config.DEFAULT_VOYAGE_NM)
    p.add_argument("--deadline", type=float, default=config.DEFAULT_DEADLINE_HOURS)
    # route
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--storms", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cruise", type=float, default=16.0)
    p.add_argument("--ship-type", default=config.DEFAULT_SCENARIO["ship_type"],
                   choices=list(config.SHIP_SPECS.keys()))
    # effort
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--qseed", type=int, default=42,
                   help="seed for QAOA/VQE reproducibility (deterministic runs)")
    # solver selection + scaling
    p.add_argument("--presolve", action="store_true",
                   help="route only: fix ports + drop unreachable cells "
                        "(fewer qubits, same optimum)")
    p.add_argument("--solvers", default="exact,neal,gsa,ga,qaoa,qaoa_cvar,vqe,vqe_cvar",
                   help="comma list from: exact,neal,tabu,gsa,ga,qaoa,qaoa_cvar,"
                        "vqe,vqe_cvar")
    p.add_argument("--cvar", type=float, default=gate.CVAR_ALPHA,
                   help="alpha for the *_cvar solvers (fraction of best shots)")
    p.add_argument("--gsa-agents", type=int, default=0,
                   help="GSA/GA population size (0 = auto-scale with problem size)")
    p.add_argument("--gsa-iters", type=int, default=0,
                   help="GSA/GA iterations (0 = auto-scale with problem size)")
    p.add_argument("--max-qubits", type=int, default=None,
                   help="skip QAOA/VQE above this many qubits (2^n simulator "
                        "limit; default 26 with a GPU, 20 without)")
    args = p.parse_args()
    if args.max_qubits is None:
        args.max_qubits = 26 if gate.gpu_available() else 20

    # Make the gate-model runs reproducible so a divergence is stable for slides.
    algorithm_globals.random_seed = args.qseed

    if args.problem == "speed":
        qubo, decode, meta, ctx = build_speed_problem(args)
    else:
        qubo, decode, meta, ctx = build_route_problem(args)

    n = meta["n_qubits"]
    n_vars = len({v for k in qubo for v in k})
    print(f"Solver comparison  --  {meta['title']}")
    print(f"  QUBO: {n_vars} variables => {n} qubits on the simulator "
          f"({2**n:,} amplitudes, Aer on {gate.device_for(n)})   |   "
          f"{args.trials} trial(s) each")
    print()

    qp = qubo_to_qp(qubo)
    requested = [s.strip().lower() for s in args.solvers.split(",")]
    gsa_agents = args.gsa_agents or max(80, 4 * n)      # auto-scale GSA effort
    gsa_iters = args.gsa_iters or max(200, 12 * n)
    if {"gsa", "ga"} & set(requested) and (args.gsa_agents or args.gsa_iters) == 0:
        print(f"  GSA/GA effort: {gsa_agents} agents x {gsa_iters} iters (auto-scaled)\n")

    # Exact reference: DP for routes (scales to any grid). For speed the optimum
    # of a one-hot QUBO is simply its cheapest feasible level -- a scan, not an
    # eigensolver whose memory doubles with every extra speed level.
    if args.problem == "route":
        exact_rec = evaluate("Exact (DP)",
                             lambda i: _path_to_vardict(ctx["dp_path"]),
                             decode, float("nan"), trials=1)
    else:
        best_level = int(np.argmin(np.where(ctx["feasible"], ctx["fuels"], np.inf)))
        exact_rec = evaluate("Exact (scan)", lambda i: {f"x_{best_level}": 1},
                             decode, float("nan"), trials=1)
    exact_fuel = exact_rec["best_fuel"]
    exact_rec["gap"] = 0.0
    exact_rec["successes"] = exact_rec["trials"] = 1

    # Each trial gets its own seed so multi-trial stats vary and trial 0 is
    # exactly reproducible. Only the requested solvers run; QAOA/VQE auto-skip
    # above the simulator's qubit ceiling.
    records = [exact_rec]
    if "neal" in requested:
        records.append(evaluate("neal (anneal)", lambda i: run_neal(qubo, seed=42 + i),
                                decode, exact_fuel, args.trials))
    if "tabu" in requested:
        records.append(evaluate("Tabu (search)", lambda i: run_tabu(qubo, seed=i),
                                decode, exact_fuel, args.trials))
    if "gsa" in requested:
        records.append(evaluate(
            "GSA (gravity)",
            lambda i: run_gsa(qubo, seed=args.qseed + i,
                              n_agents=gsa_agents, iters=gsa_iters),
            decode, exact_fuel, args.trials))
    if "ga" in requested:
        records.append(evaluate(
            "GA (genetic)",
            lambda i: run_ga(qubo, seed=args.qseed + i,
                             n_agents=gsa_agents, iters=gsa_iters),
            decode, exact_fuel, args.trials))
    gate_runs = {   # solver -> (label, one trial); *_cvar = same circuit, CVaR objective
        "qaoa": ("QAOA", lambda i, a=None: run_qaoa(
            qp, args.reps, args.maxiter, seed=args.qseed + i, cvar=a)),
        "vqe": ("VQE", lambda i, a=None: run_vqe(
            qp, n, max(1, args.reps - 1), args.maxiter + 100, seed=args.qseed + i, cvar=a)),
    }
    for name in ("qaoa", "qaoa_cvar", "vqe", "vqe_cvar"):
        if name not in requested:
            continue
        label, run = gate_runs[name.split("_")[0]]
        alpha = args.cvar if name.endswith("_cvar") else None
        if alpha is not None:
            label += f"-CVaR({alpha:g})"
        if n > args.max_qubits:
            print(f"  (skipping {label}: {n} qubits > {args.max_qubits}-qubit simulator limit)")
            continue
        records.append(evaluate(label, lambda i, run=run, alpha=alpha: run(i, alpha),
                                decode, exact_fuel, args.trials))

    # --- Metrics table ----------------------------------------------------- #
    print("  COMPARISON METRICS")
    print(f"  {'Method':<16}{'best fuel':>10}{'gap %':>8}{'feasible':>10}"
          f"{'success':>9}{'time/run':>10}   best solution")
    print("  " + "-" * 92)
    for r in records:
        fuel_s = f"{r['best_fuel']:.2f}" if r["best_fuel"] == r["best_fuel"] else "n/a"
        gap_s = "-" if r["label"].startswith("Exact") else (
            f"{r['gap']:+.1f}" if r["gap"] == r["gap"] else "n/a")
        feas_s = f"{r['feasible']}/{r['trials']}"
        succ_s = f"{r['successes']}/{r['trials']}"
        print(f"  {r['label']:<16}{fuel_s:>10}{gap_s:>8}{feas_s:>10}"
              f"{succ_s:>9}{r['avg_time']:>9.2f}s   {r['best_str']}")
    print()
    print("  Legend: gap % = best fuel vs exact optimum | feasible = valid "
          "solutions / trials |\n          success = runs that hit the exact "
          "optimum / trials")

    # --- Comparison plot --------------------------------------------------- #
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_q{args.qseed}" + ("_single" if args.trials == 1 else f"_t{args.trials}")
    if args.problem == "route" and args.presolve:
        tag += "_presolve"
    if args.trials == 1:
        meta["title"] += f"  (single reproducible run, qseed={args.qseed})"
    if meta["kind"] == "speed":
        out = config.OUTPUTS_DIR / f"compare_speed_L{args.levels}{tag}.png"
        plot_speed_comparison(ctx, records, meta, out)
    else:
        out = config.OUTPUTS_DIR / (f"compare_route_{args.rows}x{args.cols}"
                                    f"_s{args.storms}_seed{args.seed}{tag}.png")
        plot_route_comparison(ctx, records, meta, out)
    print(f"\n  Saved comparison plot -> {out}")
    results.save(out.stem, {
        "problem": args.problem, "title": meta["title"], "n_qubits": n,
        "device": gate.device_for(n), "trials": args.trials, "plot": out.name,
        "records": [{k: r[k] for k in ("label", "best_fuel", "mean_fuel", "gap",
                                        "feasible", "successes", "trials", "avg_time",
                                        "best_str")} for r in records]})
    print("\n  Takeaway: same QUBO, three solvers. neal (annealing) is fast and "
          "reliable on\n  constrained QUBOs; QAOA/VQE (gate-model) match on small "
          "problems but need more\n  circuit depth as constraints grow -- the "
          "honest state of the art.")


if __name__ == "__main__":
    main()
