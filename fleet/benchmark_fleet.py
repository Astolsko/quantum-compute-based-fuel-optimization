"""
benchmark_fleet.py -- honest multi-instance benchmark for the fleet layer.

Methodology (inherited from the Phase-3 lessons, non-negotiable):
  * EVERY instance counted -- no cherry-picked seeds; infeasible instances
    and solver failures are reported, not hidden.
  * Stated budgets: the annealing-style searches get the SAME iteration count;
    the methods that PROPOSE ASSIGNMENTS (random, GA, the QUBO hybrid) get the
    same number of DISTINCT EXACT SCORINGS (hybrid.AssignmentScorer) and their
    best-so-far is recorded after every scoring, so they are also compared at
    every smaller budget (--checkpoints).
  * Ground truth: exact subset DP (matheuristic.exact_dp), exact to 16 missions.
  * Reported: per-instance gaps; mean +/- std with a 95% bootstrap interval;
    optimal hits with a 95% Wilson interval; failures; time; and paired tests
    on the same instances (Wilcoxon on gaps, exact McNemar on "hit the optimum")
    -- because 10/10 vs 8/10 on ten instances is not, by itself, a result.

Solvers
  greedy / sa_naive / alns_fixed / alns     matheuristic.py (classical search)
  rand+exact     random assignments, exact scoring       (ablation)
  ga             genetic algorithm over assignments      (genetic.py)
  hybrid_v1      assignment QUBO (neal) + exact scoring  (the original, unchanged)
  hybrid         v2: + lazy cuts + stronger repair       (hybrid.solve_hybrid_cuts)
  hybrid+alns    v2's best plan as ALNS's starting point (does the QUBO warm-start help?)
  qubo+repair    whole-plan position QUBO                (qubo_layer.py)

Run (from this folder, or `python run.py benchmark_fleet ...` from the root):
    python benchmark_fleet.py --seeds 10                          # 3 ships, 6 missions
    python benchmark_fleet.py --seeds 10 --ships 4 --missions 12
    python benchmark_fleet.py --seeds 10 --sizes 3x6,4x8,4x10,4x12,5x14,5x16   # scale study
"""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fleet_config as fc
import genetic
import hybrid as hyb
import matheuristic as mh
import oracle
import qubo_layer as ql
import scenario as scen_mod

sys.path.insert(0, str(fc.PARENT_DIR))
from core import results, stats                    # read-only

SOLVERS = ["greedy", "sa_naive", "alns_fixed", "alns", "rand+exact", "ga",
           "hybrid_v1", "hybrid", "hybrid+alns", "qubo+repair"]
ANYTIME = ["rand+exact", "ga", "hybrid"]            # scored by AssignmentScorer
PAIRS = [("hybrid", "alns"), ("hybrid", "ga"), ("ga", "alns"),
         ("ga", "rand+exact"), ("hybrid", "rand+exact"), ("hybrid", "hybrid_v1"),
         ("hybrid+alns", "alns")]
# Eight solvers that share a line chart take the eight validated categorical
# slots, in slot order (SCALE_ORDER); the two bar-only solvers stay neutral grey.
SCALE_ORDER = ["alns", "alns_fixed", "rand+exact", "greedy", "hybrid_v1", "ga",
               "hybrid", "hybrid+alns"]
COLORS = dict(zip(SCALE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                "#008300", "#6250d6", "#e34948"]))
COLORS.update({"sa_naive": "#8a8984", "qubo+repair": "#8a8984"})


def run_instance(scn, args):
    """Every solver on one instance -> (exact fuel, {solver: (fuel, seconds)},
    {anytime solver: best-so-far trace}). exact fuel INF = infeasible."""
    tabs = oracle.get_tables(scn, verbose=False)
    ev = mh.Evaluator(scn, tabs)
    t = time.perf_counter()
    _, fe = mh.exact_dp(ev)
    exact_s = time.perf_counter() - t
    if not np.isfinite(fe):
        return fe, exact_s, {}, {}

    seed = scn.seed
    out, traces = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        out[name] = (r, time.perf_counter() - t0)
        return r

    timed("greedy", lambda: (lambda g: ev.total(g) if g else float("inf"))(mh.greedy(ev)))
    timed("sa_naive", lambda: mh.sa_naive(ev, n_iters=args.iters, seed=seed + 1)[1])
    timed("alns_fixed", lambda: mh.alns(ev, n_iters=args.iters, seed=seed + 1,
                                        adaptive=False)[1])
    timed("alns", lambda: mh.alns(ev, n_iters=args.iters, seed=seed + 1)[1])

    sc = hyb.AssignmentScorer(ev)
    timed("rand+exact", lambda: hyb.solve_random(ev, budget=args.budget,
                                                 seed=seed + 1, scorer=sc)[1])
    traces["rand+exact"] = sc.trace
    sc = hyb.AssignmentScorer(ev)
    timed("ga", lambda: genetic.solve_ga(ev, budget=args.budget, pop=args.pop,
                                         seed=seed + 1, scorer=sc)[1])
    traces["ga"] = sc.trace
    timed("hybrid_v1", lambda: hyb.solve_hybrid(ev, num_reads=args.num_reads, seed=42)[1])
    sc = hyb.AssignmentScorer(ev)
    plan_v2 = timed("hybrid", lambda: hyb.solve_hybrid_cuts(
        ev, num_reads=args.num_reads, budget=args.budget, seed=42, scorer=sc))
    traces["hybrid"] = sc.trace
    out["hybrid"] = (plan_v2[1], out["hybrid"][1])
    t0 = time.perf_counter()                      # polish = v2's time + ALNS's
    f_pol = mh.alns(ev, n_iters=args.iters, seed=seed + 1, start=plan_v2[0])[1]
    out["hybrid+alns"] = (f_pol, out["hybrid"][1] + time.perf_counter() - t0)
    timed("qubo+repair", lambda: ql.solve_fleet_qubo(
        scn, tabs, ev, num_reads=args.num_reads, seed=42)["true_fuel"])
    return fe, exact_s, out, traces


def gap(f, fe):
    return (f - fe) / fe * 100 if np.isfinite(f) else None


def run_size(K, M, args):
    """All instances of one size -> summary dict (also printed, plotted, saved)."""
    print(f"\n=== {K} ships x {M} missions: {args.seeds} instances, horizon "
          f"{fc.horizon_slots(K, M)} slots | iters={args.iters}, qubo reads="
          f"{args.num_reads}, exact scorings={args.budget} ===")
    inst, gaps = [], {s: [] for s in SOLVERS}
    times = {s: [] for s in SOLVERS}
    curves = {s: [] for s in ANYTIME}
    for seed in range(args.seeds):
        scn = scen_mod.build_scenario(seed, n_ships=K, n_missions=M)
        fe, exact_s, out, traces = run_instance(scn, args)
        if not np.isfinite(fe):
            print(f"  seed {seed}: instance INFEASIBLE by the exact check "
                  f"-- excluded (reported, not hidden)")
            inst.append({"seed": seed, "feasible": False})
            continue
        row = {"seed": seed, "feasible": True, "exact": fe, "exact_seconds": exact_s}
        line = f"  seed {seed}: exact {fe:6.2f}t ({exact_s:.1f}s) |"
        for s in SOLVERS:
            f, dt = out[s]
            g = gap(f, fe)
            gaps[s].append(g)
            times[s].append(dt)
            row[s] = {"fuel": f, "gap": g, "seconds": dt}
            line += f" {s} " + ("FAIL" if g is None else f"{g:+.1f}%")
        for s in ANYTIME:
            tr = np.array(traces[s], dtype=float)
            curves[s].append(np.where(np.isfinite(tr), (tr - fe) / fe * 100, np.nan))
        print(line, flush=True)
        inst.append(row)

    n_valid = sum(r["feasible"] for r in inst)
    if n_valid == 0:
        print("  No feasible instances.")
        return {"ships": K, "missions": M, "n_valid": 0, "instances": inst}

    summary = {}
    for s in SOLVERS:
        ok = [g for g in gaps[s] if g is not None]
        hits = sum(g < 1e-6 for g in ok)
        summary[s] = {
            "mean_gap": float(np.mean(ok)) if ok else None,
            "std_gap": float(np.std(ok)) if ok else None,
            "gap_ci": list(stats.bootstrap_ci(ok)),
            "optimal": hits, "opt_ci": list(stats.wilson(hits, n_valid)),
            "failed": len(gaps[s]) - len(ok), "mean_seconds": float(np.mean(times[s]))}
    anytime, at_gaps = {}, {}
    for s in ANYTIME:
        anytime[s], at_gaps[s] = {}, {}
        for b in args.checkpoints:
            at = [c[min(b, len(c)) - 1] if len(c) else np.nan for c in curves[s]]
            at_gaps[s][b] = [float(g) if np.isfinite(g) else None for g in at]
            ok = [g for g in at if np.isfinite(g)]
            anytime[s][b] = {"mean_gap": float(np.mean(ok)) if ok else None,
                             "optimal": int(sum(g < 1e-6 for g in ok)),
                             "failed": len(at) - len(ok), "per_instance": at_gaps[s][b]}
    tests = {f"{a} vs {b}": stats.paired(gaps[a], gaps[b]) for a, b in PAIRS}
    # the proposal methods also at every equal budget, not only the final one
    for b in args.checkpoints:
        for x, y in (("hybrid", "rand+exact"), ("hybrid", "ga"), ("ga", "rand+exact")):
            tests[f"{x} vs {y} @{b}"] = stats.paired(at_gaps[x][b], at_gaps[y][b])

    print(f"\n  {'solver':<13}{'mean gap':>10}{'95% CI':>17}{'optimal':>10}"
          f"{'95% CI':>11}{'failed':>8}{'time':>9}")
    print("  " + "-" * 78)
    for s in SOLVERS:
        r = summary[s]
        mg = "     -" if r["mean_gap"] is None else f"{r['mean_gap']:+9.1f}%"
        ci = ("" if r["mean_gap"] is None else
              f"[{r['gap_ci'][0]:+.1f}, {r['gap_ci'][1]:+.1f}]")
        oc = f"{r['opt_ci'][0]:.0%}-{r['opt_ci'][1]:.0%}"
        print(f"  {s:<13}{mg:>10}{ci:>17}{r['optimal']:>6}/{n_valid:<3}{oc:>11}"
              f"{r['failed']:>8}{r['mean_seconds']:>8.2f}s")
    print(f"\n  Anytime (equal number of distinct exact scorings):")
    for s in ANYTIME:
        print(f"    {s:<11}" + "  ".join(
            f"@{b}: " + ("-" if v["mean_gap"] is None else f"{v['mean_gap']:+.1f}%")
            + f" ({v['optimal']}/{n_valid} opt)" for b, v in anytime[s].items()))
    print(f"\n  Paired tests on the same {n_valid} instances (p < 0.05 = a real difference;"
          f" @N = at N exact scorings):")
    for name, t in tests.items():
        md = "-" if t["mean_diff"] is None else f"{t['mean_diff']:+.2f} pts"
        print(f"    {name:<30} mean gap diff {md:>11}  Wilcoxon p={t['wilcoxon_p']:.3f}"
              f"  | optimal only-A {t['only_a_opt']} / only-B {t['only_b_opt']}, "
              f"McNemar p={t['mcnemar_p']:.3f}")

    tag = size_tag(K, M)
    plot_bars(summary, n_valid, K, M, fc.OUTPUTS_DIR / f"fleet_benchmark{tag}.png")
    plot_anytime(curves, n_valid, K, M, args.budget, fc.OUTPUTS_DIR / f"fleet_anytime{tag}.png")
    res = {"ships": K, "missions": M, "horizon": fc.horizon_slots(K, M),
           "seeds": args.seeds, "n_valid": n_valid, "iters": args.iters,
           "num_reads": args.num_reads, "budget": args.budget, "pop": args.pop,
           "solvers": SOLVERS, "summary": summary, "anytime": anytime,
           "paired_tests": tests, "instances": inst,
           "plots": [f"fleet_benchmark{tag}.png", f"fleet_anytime{tag}.png"]}
    results.save(f"fleet_benchmark{tag}", res, fc.OUTPUTS_DIR)
    return res


def size_tag(K, M):
    return ("" if (K, M) == (fc.DEFAULT_N_SHIPS, fc.DEFAULT_N_MISSIONS)
            else f"_{K}x{M}")


def plot_bars(summary, n_valid, K, M, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(SOLVERS))
    for i, s in enumerate(SOLVERS):
        r = summary[s]
        if r["mean_gap"] is None:
            ax.text(i, 0.5, "all\nfailed", ha="center", fontsize=8)
            continue
        lo, hi = r["gap_ci"]
        ax.bar(i, r["mean_gap"], color=COLORS[s], edgecolor="black", lw=0.8)
        ax.errorbar(i, r["mean_gap"], yerr=[[r["mean_gap"] - lo], [hi - r["mean_gap"]]],
                    color="black", capsize=4)
        note = f"{r['optimal']}/{n_valid} opt" + (f"\n{r['failed']} failed" if r["failed"] else "")
        ax.text(i, hi + 0.5, note, ha="center", fontsize=8)
    ax.axhline(0, color="#16A34A", ls="--", lw=1.5, label="exact optimum")
    ax.set_xticks(xs)
    ax.set_xticklabels(SOLVERS, rotation=20)
    ax.set_ylabel("Optimality gap % (mean, 95% bootstrap CI)")
    ax.set_title(f"Fleet solvers on {n_valid} instances ({K} ships x {M} missions) "
                 f"-- stated budgets, every run counted")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fc.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_anytime(curves, n_valid, K, M, budget, out):
    """Mean gap vs number of distinct exact scorings (failed = not yet feasible)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ANYTIME:
        mat = np.full((len(curves[s]), budget), np.nan)
        for i, c in enumerate(curves[s]):
            if len(c):
                mat[i, :len(c)] = c
                mat[i, len(c):] = c[-1]
        with np.errstate(all="ignore"):
            mean = np.nanmean(mat, axis=0)
        ax.plot(np.arange(1, budget + 1), mean, lw=2, color=COLORS[s], label=s)
    ax.axhline(0, color="#16A34A", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Distinct assignments scored exactly (log)")
    ax.set_ylabel("Best gap so far, mean over instances (%)")
    ax.set_title(f"Proposal quality at equal scoring budget ({K}x{M}, {n_valid} instances)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_scale(all_res, out):
    done = [r for r in all_res if r["n_valid"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for s in SCALE_ORDER:
        xs = [r["missions"] for r in done]
        ax1.plot(xs, [r["summary"][s]["mean_gap"] if r["summary"][s]["mean_gap"] is not None
                      else np.nan for r in done], "o-", color=COLORS[s], lw=2, mec="white",
                 mew=1, label=s)
        ax2.plot(xs, [100 * r["summary"][s]["optimal"] / r["n_valid"] for r in done],
                 "o-", color=COLORS[s], lw=2, mec="white", mew=1, label=s)
    labels = [f"{r['ships']}x{r['missions']}" for r in done]
    for ax in (ax1, ax2):
        ax.set_xticks([r["missions"] for r in done])
        ax.set_xticklabels(labels)
        ax.set_xlabel("Ships x missions")
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("Mean optimality gap %")
    ax1.set_title("Solution quality vs size (exact ground truth throughout)")
    ax2.set_ylabel("% of instances solved to optimality")
    ax2.set_title("Hit rate vs size")
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=len(SCALE_ORDER),
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--iters", type=int, default=3000, help="ALNS / SA iterations")
    p.add_argument("--num-reads", type=int, default=1000, help="neal reads per QUBO solve")
    p.add_argument("--budget", type=int, default=1000,
                   help="distinct exact scorings for rand+exact / ga / hybrid")
    p.add_argument("--pop", type=int, default=40, help="GA population")
    p.add_argument("--checkpoints", default="25,200,1000",
                   help="scoring budgets at which the anytime methods are compared")
    p.add_argument("--ships", type=int, default=fc.DEFAULT_N_SHIPS)
    p.add_argument("--missions", type=int, default=fc.DEFAULT_N_MISSIONS,
                   help="up to 16 (limit of the exact ground truth)")
    p.add_argument("--sizes", default=None,
                   help="scale study, e.g. 3x6,4x8,4x12,5x16 (overrides --ships/--missions)")
    args = p.parse_args()
    args.checkpoints = [b for b in map(int, args.checkpoints.split(",")) if b <= args.budget]

    sizes = ([tuple(map(int, s.split("x"))) for s in args.sizes.split(",")]
             if args.sizes else [(args.ships, args.missions)])
    t0 = time.perf_counter()
    all_res = [run_size(K, M, args) for K, M in sizes]
    if len(sizes) > 1:
        plot_scale(all_res, fc.OUTPUTS_DIR / "fleet_scale.png")
        results.save("fleet_scale", {
            "sizes": [{"ships": r["ships"], "missions": r["missions"],
                       "n_valid": r["n_valid"], "seeds": args.seeds,
                       "summary": r.get("summary")} for r in all_res],
            "plot": "fleet_scale.png"}, fc.OUTPUTS_DIR)
        print(f"\n  Saved scale plot -> {fc.OUTPUTS_DIR / 'fleet_scale.png'}")
    print(f"\n  Done in {(time.perf_counter() - t0) / 60:.1f} min. Plots + "
          f"results/*.json -> {fc.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
