"""
benchmark.py  --  the HONEST solver benchmark (replaces every ad-hoc comparison).

This exists because earlier comparisons in this repo were unfair (unequal budgets)
and cherry-picked (single hand-selected seeds). This harness fixes that by design:

  * FIXED, STATED budget per solver (printed below and in BENCHMARK.md), with
    what that budget means in energy evaluations. The budgets are identical
    across every cell but they are NOT equal compute across solver families --
    which is why they are spelled out instead of called "equal".
  * SEVERAL weather instances per grid size (--instances) x MANY solver seeds
    (--seeds); EVERY run is counted -- no best-of, no selected seed. (Until
    2026-10 each size was ONE instance, so "reliability vs size" was really
    "reliability on these five particular storms".)
  * Reports mean +/- std of the optimality gap, plus % optimal and % feasible --
    and how sure each number is: a 95% Wilson interval on % optimal, a 95%
    bootstrap interval on the mean gap, and TTS99 (seconds to hit the optimum
    once with 99% confidence by repeating the solve; core/stats.py).
  * Ground truth = exact DP (the routing QUBO is a shortest path, solvable exactly).

Studies:
  A. Reliability vs problem size   -> bench_size.png
  B. Sensitivity to penalty weight -> bench_penalty.png
  --gate: Study A for the gate-model solvers only (QAOA, VQE and their CVaR
     variants, neal as the yardstick) on PRESOLVED grids of 6-21 qubits --
     the sizes the Aer simulator (GPU from 18 qubits) makes reachable at this
     many seeds; 26 qubits is in reach for single runs (quantum_gate.py).
Tables go to BENCHMARK*.md and results/benchmark*.json in the outputs folder.
With --presolve the same studies run on the reduced QUBO (ports fixed,
unreachable cells dropped) and every output file gets a `_presolve` suffix.

Run (from the project root):
    python -m experiments.benchmark                 # full run
    python -m experiments.benchmark --quick         # small smoke test
    python -m experiments.benchmark --fast --with-qgso --jobs 16
    python -m experiments.benchmark --fast --with-qgso --presolve --jobs 16
    python -m experiments.benchmark --gate --jobs 16
"""

from __future__ import annotations

import argparse
import os
import time
import types
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import config
from core import results, stats
from experiments import quantum_gate as qg
from solvers import gate
from solvers import qgso as qgso_mod

# ---- FIXED BUDGETS (stated up front; identical across every cell) ---------- #
# gsa, ga and qgso share ONE budget on purpose -- so the only thing measured is
# the update rule (gravity / breeding / quantum rotation gate), not compute.
BUDGET = {
    "neal": dict(num_reads=200, num_sweeps=1000),
    "tabu": dict(num_reads=200),
    "gsa":  dict(restarts=10, n_agents=100, iters=300),
    "ga":   dict(restarts=10, n_agents=100, iters=300),
    "qgso": dict(restarts=10, n_agents=100, iters=300),
    "qaoa": dict(reps=3, maxiter=200),
    "vqe":  dict(reps=2, maxiter=300),
    "qaoa_cvar": dict(reps=3, maxiter=200, alpha=gate.CVAR_ALPHA),
    "vqe_cvar":  dict(reps=2, maxiter=300, alpha=gate.CVAR_ALPHA),
}
CLASSICAL = ["neal", "tabu", "gsa", "ga"]
GATE = ["qaoa", "vqe"]
GPU_JOBS = 4            # parallel solves that each hold a GPU statevector


def budget_text(solver, n):
    """What a solver's budget means in work done, for an n-variable QUBO."""
    b = BUDGET[solver]
    if solver == "neal":
        return f"{b['num_reads'] * b['num_sweeps'] * n:,} single-bit-flip evaluations"
    if solver == "tabu":
        return f"{b['num_reads']} reads x 20 ms time limit each (wall-clock, not evaluations)"
    if solver in ("gsa", "ga", "qgso"):
        return f"{b['restarts'] * b['n_agents'] * b['iters']:,} full-energy evaluations"
    return (f"up to {b['maxiter']} circuit evaluations x {gate.SHOTS} shots "
            f"(statevector, {gate.device_for(n)})")


def run_qgso(qubo, seed, n_agents, iters, restarts):
    """QGSO solve -> {varname: 0/1}; same restart/keep-best scheme as run_gsa."""
    samp = qgso_mod.QGSOSampler(n_agents=n_agents, iters=iters)
    ss = samp.sample_qubo(qubo, num_reads=restarts, seed=seed)
    return {k: int(v) for k, v in dict(ss.first.sample).items()}


def make_problem(rows, cols, storms, seed, penalty_mult=10.0, presolve=False):
    """One routing instance -> (qubo, qp, n_vars, decode, exact_fuel)."""
    a = types.SimpleNamespace(problem="route", rows=rows, cols=cols, storms=storms,
                              seed=seed, cruise=16.0, distance=config.DEFAULT_VOYAGE_NM,
                              ship_type="Offshore_Patrol", penalty_mult=penalty_mult,
                              presolve=presolve)
    qubo, decode, meta, ctx = qg.build_route_problem(a)
    return qubo, qg.qubo_to_qp(qubo), meta["n_qubits"], decode, ctx["dp_fuel"]


def solve_once(solver, b, qubo, qp, n, decode, s):
    """One solve of `solver` (budget `b`) at seed `s`. Returns (path, fuel, seconds).
    The budget is an argument, not a global lookup, so that worker processes use
    the budget the command line asked for."""
    t = time.perf_counter()
    if solver == "neal":
        vd = qg.run_neal(qubo, seed=s)
    elif solver == "tabu":
        vd = qg.run_tabu(qubo, seed=s)
    elif solver == "gsa":
        vd = qg.run_gsa(qubo, seed=s, n_agents=b["n_agents"],
                        iters=b["iters"], restarts=b["restarts"])
    elif solver == "ga":
        vd = qg.run_ga(qubo, seed=s, n_agents=b["n_agents"],
                       iters=b["iters"], restarts=b["restarts"])
    elif solver == "qgso":
        vd = run_qgso(qubo, seed=s, n_agents=b["n_agents"],
                      iters=b["iters"], restarts=b["restarts"])
    elif solver in ("qaoa", "qaoa_cvar"):
        vd = qg.run_qaoa(qp, b["reps"], b["maxiter"], seed=s, cvar=b.get("alpha"))
    elif solver in ("vqe", "vqe_cvar"):
        vd = qg.run_vqe(qp, n, b["reps"], b["maxiter"], seed=s, cvar=b.get("alpha"))
    dt = time.perf_counter() - t
    path, fuel, _ = decode(vd)
    return path, fuel, dt


def run_cell(solver, problem, seeds, jobs=1):
    """Every seed of one solver on one instance -> [(gap % or None, seconds)].
    With jobs > 1 the seeds run in parallel processes (per-solve times are then
    measured on a shared machine: compare them only within one run)."""
    qubo, qp, n, decode, dp_fuel = problem
    qp = qp if solver in GATE else None          # only gate solvers need it
    if solver in GATE and gate.device_for(n) == "GPU":
        jobs = min(jobs if jobs > 0 else GPU_JOBS, GPU_JOBS)   # one GPU, shared
    runs = Parallel(n_jobs=jobs)(
        delayed(solve_once)(solver, BUDGET[solver], qubo, qp, n, decode, s)
        for s in seeds)
    return [((fuel - dp_fuel) / dp_fuel * 100 if path is not None else None, dt)
            for path, fuel, dt in runs]


def summarise(runs):
    """[(gap % or None, seconds)] pooled over instances x seeds -> one table row."""
    gaps = [g for g, _ in runs if g is not None]
    n = len(runs)
    hits = sum(abs(g) < 1e-6 for g in gaps)
    mean_time = float(np.mean([dt for _, dt in runs]))
    return {
        "feas_pct": 100 * len(gaps) / n,
        "opt_pct": 100 * hits / n,
        "opt_ci": [100 * x for x in stats.wilson(hits, n)],
        "mean_gap": (float(np.mean(gaps)) if gaps else float("nan")),
        "std_gap": (float(np.std(gaps)) if gaps else float("nan")),
        "gap_ci": list(stats.bootstrap_ci(gaps)),
        "mean_time": mean_time,
        "tts99": stats.tts99(hits / n, mean_time),
        "n_feasible": len(gaps), "n": n,
    }


# --------------------------------------------------------------------------- #
# Study A: reliability vs problem size                                         #
# --------------------------------------------------------------------------- #
def study_size(sizes, n_seeds, n_instances, max_qubits, storms, jobs, presolve):
    seeds = list(range(n_seeds))
    results = {}          # grid label -> {solver: summary}
    order = []            # (n_vars, grid label), in size order
    print(f"\n=== STUDY A: reliability vs size ({n_instances} instance(s) x "
          f"{n_seeds} seeds, fixed budget) ===")
    for rows, cols in sizes:
        label, runs = f"{rows}x{cols}", {}
        for inst in range(n_instances):          # storm seeds 3, 4, 5, ...
            problem = make_problem(rows, cols, storms, seed=3 + inst,
                                   presolve=presolve)
            n = problem[2]
            for solver in CLASSICAL + (GATE if n <= max_qubits else []):
                runs.setdefault(solver, []).extend(
                    run_cell(solver, problem, seeds, jobs))
        order.append((n, label))
        results[label] = {s: summarise(r) for s, r in runs.items()}
        for solver, r in results[label].items():
            print(f"  {label} n={n:<3} {solver:<9} "
                  f"opt {r['opt_pct']:>3.0f}%  feas {r['feas_pct']:>3.0f}%  "
                  f"gap {r['mean_gap']:+.1f}+-{r['std_gap']:.1f}%  {r['mean_time']:.2f}s"
                  f"  TTS99 {r['tts99']:.2f}s")
    return results, order


# --------------------------------------------------------------------------- #
# Study B: sensitivity to penalty weight                                       #
# --------------------------------------------------------------------------- #
def study_penalty(rows, cols, storms, mults, n_seeds, n_instances, jobs, presolve):
    seeds = list(range(n_seeds))
    results = {}          # penalty_mult -> {solver: summary}
    print(f"\n=== STUDY B: penalty sweep on {rows}x{cols} ({n_instances} "
          f"instance(s) x {n_seeds} seeds) ===")
    for pm in mults:
        runs = {}
        for inst in range(n_instances):
            problem = make_problem(rows, cols, storms, seed=3 + inst,
                                   penalty_mult=pm, presolve=presolve)
            for solver in CLASSICAL:
                runs.setdefault(solver, []).extend(
                    run_cell(solver, problem, seeds, jobs))
        results[pm] = {s: summarise(r) for s, r in runs.items()}
        for solver, r in results[pm].items():
            print(f"  P={pm:>4}x  {solver:<5} opt {r['opt_pct']:>3.0f}%  "
                  f"feas {r['feas_pct']:>3.0f}%  gap {r['mean_gap']:+.1f}+-{r['std_gap']:.1f}%")
    return results


# --------------------------------------------------------------------------- #
# Plots + markdown                                                             #
# --------------------------------------------------------------------------- #
COLORS = {"neal": "#1f77b4", "tabu": "#0891B2", "gsa": "#7C3AED",
          "ga": "#65A30D", "qgso": "#DB2777", "qaoa": "#E8772E", "vqe": "#D62728",
          "qaoa_cvar": "#B45309", "vqe_cvar": "#9F1239"}


def plot_gsa_vs_qgso(size_res, size_order, pen_res, out):
    """Dedicated head-to-head: does the quantum-inspired twist beat plain GSA?"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    for solver in ("gsa", "qgso"):
        cells = [(n, size_res[label][solver]) for n, label in size_order
                 if solver in size_res[label]]
        xs = [n for n, _ in cells]
        gap = [r["mean_gap"] for _, r in cells]
        err = [r["std_gap"] for _, r in cells]
        feas = [r["feas_pct"] for _, r in cells]
        ax1.errorbar(xs, gap, yerr=err, marker="o", capsize=3, lw=2,
                     color=COLORS[solver], label=solver.upper())
        ax2.plot(xs, feas, marker="o", lw=2, color=COLORS[solver],
                 label=solver.upper())
    ax1.axhline(0, color="green", ls="--", lw=1)
    ax1.set_xlabel("QUBO size (vars)"); ax1.set_ylabel("Optimality gap % (mean±std)")
    ax1.set_title("Quality vs size"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("QUBO size (vars)"); ax2.set_ylabel("Feasibility %")
    ax2.set_title("Feasibility vs size"); ax2.legend(); ax2.grid(True, alpha=0.3)
    mults = sorted(pen_res.keys())
    for solver in ("gsa", "qgso"):
        ys = [pen_res[pm][solver]["mean_gap"] for pm in mults]
        es = [pen_res[pm][solver]["std_gap"] for pm in mults]
        ax3.errorbar(mults, ys, yerr=es, marker="o", capsize=3, lw=2,
                     color=COLORS[solver], label=solver.upper())
    ax3.axhline(0, color="green", ls="--", lw=1)
    ax3.set_xlabel("Penalty ×max cell cost"); ax3.set_ylabel("Optimality gap %")
    ax3.set_title("Penalty sensitivity (4x6 grid)"); ax3.legend(); ax3.grid(True, alpha=0.3)
    fig.suptitle("GSA vs QGSO (quantum-inspired) — equal budget, every seed counted",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_size(results, order, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for solver in CLASSICAL + GATE:
        xs, ys, es, fs = [], [], [], []
        for n, label in order:
            if solver in results[label]:
                r = results[label][solver]
                xs.append(n); ys.append(r["mean_gap"]); es.append(r["std_gap"])
                fs.append(r["feas_pct"])
        if xs:
            ax1.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2,
                         color=COLORS[solver], label=solver)
            ax2.plot(xs, fs, marker="o", lw=2, color=COLORS[solver], label=solver)
    ax1.axhline(0, color="green", ls="--", lw=1, label="optimum")
    ax1.set_xlabel("QUBO size (binary variables)")
    ax1.set_ylabel("Optimality gap % (mean +/- std over seeds)")
    ax1.set_title("Solution quality vs size")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("QUBO size (binary variables)")
    ax2.set_ylabel("Feasibility % (valid runs / seeds)")
    ax2.set_title("Constraint satisfaction vs size")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    fig.suptitle("Honest benchmark: solver reliability vs problem size "
                 "(fixed budget, all seeds counted)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_penalty(results, rows, cols, out):
    mults = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for solver in CLASSICAL:
        ys = [results[pm][solver]["mean_gap"] for pm in mults]
        es = [results[pm][solver]["std_gap"] for pm in mults]
        ax.errorbar(mults, ys, yerr=es, marker="o", capsize=3, lw=2,
                    color=COLORS[solver], label=solver)
    ax.axhline(0, color="green", ls="--", lw=1, label="optimum")
    ax.set_xlabel("Penalty weight multiplier  (P = mult x max cell cost)")
    ax.set_ylabel("Optimality gap % (mean +/- std)")
    ax.set_title(f"Sensitivity to penalty weight ({rows}x{cols} route QUBO)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _g(x):
    """Signed gap to 0.1, without "-0.0" from float noise."""
    return f"{round(x, 1) + 0.0:+.1f}"


def _fmt_time(sec):
    return "never" if not np.isfinite(sec) else f"{sec:.2f} s"


def write_markdown(size_res, size_order, pen_res, pen_grid, n_seeds, n_seeds_b,
                   n_instances, presolve, path):
    L = []
    L.append("# Honest Solver Benchmark\n")
    L.append("Generated by `experiments/benchmark.py`. Every solver gets a "
             "**fixed, stated budget**; **every run is counted** (no best-of, no "
             "cherry-picking); results are **mean ± std** pooled over "
             f"**{n_instances} weather instance(s) × solver seeds**. Ground truth "
             "= exact DP. Brackets are 95% intervals (Wilson for % optimal, "
             "bootstrap for the mean gap); TTS99 = time to hit the optimum once "
             "with 99% confidence by repeating the solve.\n")
    L.append("QUBO encoding: " + ("**presolved** (ports fixed, unreachable cells "
             "dropped)." if presolve else "**raw** (every cell is a variable).") + "\n")
    L.append("## Budgets (identical across all cells)\n")
    L.append("Fixed, but not equal compute across solver families — the Budget "
             "column of Study A says what each one buys at that size.\n")
    for k in CLASSICAL + GATE:
        L.append(f"- **{k}**: {BUDGET[k]}")
    L.append("")
    L.append(f"## Study A — reliability vs size ({n_instances} instance(s) × "
             f"{n_seeds} seeds)\n")
    L.append("| Grid | Vars | Solver | % optimal [95% CI] | % feasible | Mean gap ± std [95% CI] "
             "| Time/solve | TTS99 | Budget |")
    L.append("|------|-----:|--------|:---------:|:----------:|:--------------:|-----------:"
             "|------:|--------|")
    for n, label in size_order:
        for solver in CLASSICAL + GATE:
            if solver in size_res[label]:
                r = size_res[label][solver]
                lo, hi = r["opt_ci"]
                glo, ghi = r["gap_ci"]
                gci = f" [{_g(glo)}, {_g(ghi)}]" if np.isfinite(glo) else ""
                L.append(f"| {label} | {n} | {solver} | {r['opt_pct']:.0f}% [{lo:.0f}–{hi:.0f}] | "
                         f"{r['feas_pct']:.0f}% | {_g(r['mean_gap'])} ± {r['std_gap']:.1f}%{gci} | "
                         f"{r['mean_time']:.2f} s | {_fmt_time(r['tts99'])} | "
                         f"{budget_text(solver, n)} |")
    L.append("")
    if pen_res:
        L.append(f"## Study B — penalty-weight sensitivity on {pen_grid} "
                 f"({n_instances} instance(s) × {n_seeds_b} seeds)\n")
        L.append("| Penalty ×max | Solver | % optimal [95% CI] | % feasible | Mean gap ± std |")
        L.append("|-------------:|--------|:---------:|:----------:|:--------------:|")
        for pm in sorted(pen_res.keys()):
            for solver in CLASSICAL:
                r = pen_res[pm][solver]
                lo, hi = r["opt_ci"]
                L.append(f"| {pm}× | {solver} | {r['opt_pct']:.0f}% [{lo:.0f}–{hi:.0f}] | "
                         f"{r['feas_pct']:.0f}% | {_g(r['mean_gap'])} ± {r['std_gap']:.1f}% |")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description="Honest, fair, multi-seed solver benchmark.")
    p.add_argument("--seeds", type=int, default=15, help="seeds for Study A")
    p.add_argument("--seeds-b", type=int, default=10, help="seeds for Study B")
    p.add_argument("--max-qubits", type=int, default=None,
                   help="run QAOA/VQE only at or below this size (default 16; "
                        "with --gate 21 on a GPU, 16 without)")
    p.add_argument("--storms", type=int, default=2)
    p.add_argument("--gsa-restarts", type=int, default=0,
                   help="override GSA/QGSO restart budget (0 = use default 10)")
    p.add_argument("--with-qgso", action="store_true",
                   help="also benchmark QGSO (quantum-inspired GSA) at equal budget")
    p.add_argument("--instances", type=int, default=3,
                   help="weather instances per grid size (1 = the single-instance "
                        "study this harness ran before 2026-10)")
    p.add_argument("--presolve", action="store_true",
                   help="benchmark the presolved QUBO (ports fixed, unreachable "
                        "cells dropped); outputs get a _presolve suffix")
    p.add_argument("--jobs", type=int, default=config.N_JOBS,
                   help="parallel solves (default: FUEL_JOBS, laptop-safe; "
                        "-1 = every core)")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test config")
    p.add_argument("--fast", action="store_true",
                   help="fewer sizes/seeds, lighter swarms, skip slow "
                        "gate-model (see --gate and audit_seeds.py)")
    p.add_argument("--gate", action="store_true",
                   help="gate-model study only: QAOA/VQE (+CVaR) vs neal on "
                        "presolved grids up to --max-qubits (default 21 with a "
                        "GPU, 16 without); outputs get a _gate suffix")
    args = p.parse_args()
    if args.jobs != 1:                 # parallel solves: one Aer thread each
        os.environ["FUEL_AER_THREADS"] = "1"

    swarms = ("gsa", "ga", "qgso")
    if args.gsa_restarts:
        for s in swarms:
            BUDGET[s]["restarts"] = args.gsa_restarts
    if args.with_qgso and "qgso" not in CLASSICAL:
        CLASSICAL.append("qgso")                      # benchmark it everywhere

    if args.gate:
        CLASSICAL[:] = ["neal"]                       # the yardstick
        GATE[:] = ["qaoa", "qaoa_cvar", "vqe", "vqe_cvar"]
        args.presolve = True                          # the only way to reach 20+ qubits
        if args.max_qubits is None:
            args.max_qubits = 21 if gate.gpu_available() else 16
        sizes = [(3, 4), (4, 5), (4, 6), (5, 6), (5, 7)]   # 6, 10, 14, 16, 21 qubits
        if args.max_qubits >= 26:
            sizes.append((5, 8))                          # 26 qubits: hours at 5 seeds
        args.seeds = min(args.seeds, 5)
        mults = []
    elif args.quick:
        sizes = [(3, 4), (4, 4), (4, 6)]
        mults = [1, 5, 20]
        args.seeds, args.seeds_b, args.instances = 4, 3, 1
    elif args.fast:
        sizes = [(3, 4), (4, 5), (4, 6), (5, 6), (6, 7)]  # 12,20,24,30,42 raw vars
        mults = [1, 2, 3, 5, 10, 20]
        args.seeds, args.seeds_b = 10, 10
        args.max_qubits = 0                           # skip gate-model (see --gate)
        for s in swarms:
            BUDGET[s]["restarts"] = args.gsa_restarts or 6
    else:
        sizes = [(3, 4), (4, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
        mults = [1, 2, 3, 5, 10, 20]

    if args.max_qubits is None:
        args.max_qubits = 16
    print("Budgets (fixed across every cell):")
    for k in CLASSICAL + GATE:
        print(f"  {k:<9} {BUDGET[k]}")

    t0 = time.perf_counter()
    size_res, size_order = study_size(sizes, args.seeds, args.instances,
                                      args.max_qubits, args.storms, args.jobs,
                                      args.presolve)
    pen_grid = "4x6"
    pen_res = (study_penalty(4, 6, args.storms, mults, args.seeds_b,
                             args.instances, args.jobs, args.presolve) if mults else {})

    tag = "_gate" if args.gate else "_presolve" if args.presolve else ""
    out = config.OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    plot_size(size_res, size_order, out / f"bench_size{tag}.png")
    if pen_res:
        plot_penalty(pen_res, 4, 6, out / f"bench_penalty{tag}.png")
    if "qgso" in CLASSICAL:
        plot_gsa_vs_qgso(size_res, size_order, pen_res,
                         out / f"bench_gsa_vs_qgso{tag}.png")
    md = out / f"BENCHMARK{tag}.md"
    write_markdown(size_res, size_order, pen_res, pen_grid, args.seeds,
                   args.seeds_b, args.instances, args.presolve, md)
    results.save(f"benchmark{tag}", {
        "presolve": args.presolve, "gate": args.gate, "instances": args.instances,
        "seeds": args.seeds, "seeds_b": args.seeds_b,
        "solvers": [s for s in CLASSICAL + GATE if any(s in size_res[g] for _, g in size_order)],
        "budgets": {k: BUDGET[k] for k in CLASSICAL + GATE},
        "size": [{"grid": label, "n_vars": n, **{s: size_res[label][s]
                                                 for s in size_res[label]}}
                 for n, label in size_order],
        "penalty": {str(pm): pen_res[pm] for pm in sorted(pen_res)},
        "minutes": (time.perf_counter() - t0) / 60})

    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min.")
    print(f"  -> {out}/bench_size{tag}.png, {md.name}, results/benchmark{tag}.json")


if __name__ == "__main__":
    main()
