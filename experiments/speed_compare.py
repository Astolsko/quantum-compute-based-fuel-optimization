"""
speed_compare.py  --  SA vs GA vs GSA (vs QUBO vs a reference) on the SPEED decision.

Three continuous optimizer families on the same slow-steaming problem, each given
the SAME budget of objective evaluations:

    SA   simulated annealing    (optimizer.py    -- one solution, cooling)
    GA   genetic algorithm      (solvers/ga.py   -- a population that breeds)
    GSA  gravitational search   (solvers/gsa.py  -- a swarm under gravity)

--legs 1 (default): the single cruise-speed problem. It is one-dimensional, so
    every method lands on the same answer: this validates the solvers but cannot
    rank them. The one-hot speed QUBO (neal) and the brute-force grid are shown
    alongside -- the "SA vs GA vs QUBO" comparison.

--legs N: a multi-segment speed PROFILE. The track crosses N legs with different
    sea states and the ship picks a speed for each leg under ONE total-time
    deadline. N coupled decisions: this is where the optimizers actually differ.
    Reference: price time at `lam` tonnes per hour and the legs become
    independent 1-D scans; bisecting lam until the deadline is just met gives a
    feasible profile AND a lower bound, so how far it can be from the true
    optimum is known. (The QUBO form of this problem is
    experiments/joint_speed_route.py --rows 1.)

Run (from the project root):
    python -m experiments.speed_compare
    python -m experiments.speed_compare --distance 600 --deadline 90
    python -m experiments.speed_compare --legs 8 --storms 2 --deadline 50
"""

from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import optimizer
from core import results
from experiments import qubo_route, qubo_speed
from solvers import ga, gsa

POP = 20                    # population size for GA and GSA
EVALS_PER_LEG = 2000        # objective evaluations per leg, identical for all three
PENALTY = 1000.0            # tonnes per hour past the deadline: never worth paying
COLORS = {"SA": "#ff7f0e", "GA": "#65A30D", "GSA": "#7C3AED",
          "QUBO": "#1f77b4", "Reference": "#16A34A"}


def leg_fuels(pipe, scenario, waves, wind, X, leg_nm):
    """Fuel (t) burned on each leg for every speed profile in X
    [(profiles, legs)], in ONE model call."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rates = optimizer.predict_rates(
        pipe, scenario, X.ravel(), wave_height_m=np.tile(waves, len(X)),
        wind_speed_kn=np.tile(wind, len(X)))
    return rates.reshape(X.shape) * leg_nm / (X * 24.0)


def lagrangian_reference(pipe, scenario, waves, wind, leg_nm, deadline, lo, hi,
                         n=400):
    """Reference profile for --legs N, or None if even full speed is too slow.

    With time priced at lam (tonnes per hour) the legs are independent, so each
    is a 1-D scan over a speed grid; bisect lam until the deadline is just met.
    Returns (speeds, fuel, hours, bound): `fuel` is achieved by a feasible
    profile and `bound` is a lower bound on EVERY profile on the grid, so the
    grid optimum is bracketed between them."""
    legs = len(waves)
    grid = np.linspace(lo, hi, n)
    F = leg_fuels(pipe, scenario, waves, wind,
                  np.repeat(grid[:, None], legs, axis=1), leg_nm)    # [n, legs]
    H = leg_nm / grid                                                # hours per leg

    def pick(lam):                       # best grid speed for each leg, given lam
        idx = np.argmin(F + lam * H[:, None], axis=0)
        return idx, float(H[idx].sum())

    if legs * H[-1] > deadline:
        return None
    lam_lo, lam_hi = 0.0, 0.0
    if pick(0.0)[1] > deadline:          # deadline binds: find the price of time
        lam_hi = 1.0
        while pick(lam_hi)[1] > deadline:
            lam_hi *= 2.0
        for _ in range(60):
            mid = (lam_lo + lam_hi) / 2.0
            if pick(mid)[1] > deadline:
                lam_lo = mid
            else:
                lam_hi = mid
    idx, hours = pick(lam_hi)
    fuel = float(F[idx, np.arange(legs)].sum())
    bound = max(float((F + lam * H[:, None]).min(axis=0).sum() - lam * deadline)
                for lam in (lam_lo, lam_hi))
    return grid[idx], fuel, hours, bound


def main() -> None:
    p = argparse.ArgumentParser(
        description="SA vs GA vs GSA on cruise speed (or a speed per leg).")
    p.add_argument("--distance", type=float, default=config.DEFAULT_VOYAGE_NM)
    p.add_argument("--deadline", type=float, default=config.DEFAULT_DEADLINE_HOURS)
    p.add_argument("--legs", type=int, default=1,
                   help="1 = one cruise speed; N = a speed per leg, one deadline")
    p.add_argument("--storms", type=int, default=1,
                   help="storms along the track (--legs N only)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scenario = dict(config.DEFAULT_SCENARIO)
    spec = config.SHIP_SPECS[scenario["ship_type"]]
    legs, leg_nm, deadline = args.legs, args.distance / args.legs, args.deadline
    min_spd, hi = args.distance / deadline, spec["max_speed"]
    if min_spd > hi:
        print(f"INFEASIBLE: needs {min_spd:.1f} kn but max is {hi:.1f} kn.")
        return
    pipe = optimizer.load_model()

    if legs == 1:                        # one leg: the deadline is just a speed floor
        waves = np.array([scenario["wave_height_m"]])
        wind = np.array([scenario["wind_speed_kn"]])
        lo = max(config.PHYSICAL_MIN_SPEED, min_spd)
    else:                                # sea state varies along the track
        storms = qubo_route.random_storms(1, legs, args.storms, args.seed)
        waves = qubo_route.build_weather(1, legs, storms)[0]
        wind = 5.0 + 6.0 * waves         # training correlation
        lo = config.PHYSICAL_MIN_SPEED
    bounds = [(lo, hi)] * legs

    def fuel_hours(X):
        X = np.atleast_2d(X)
        return (leg_fuels(pipe, scenario, waves, wind, X, leg_nm).sum(axis=1),
                (leg_nm / X).sum(axis=1))

    def objective(X):                    # vectorised: one value per profile
        fuel, hours = fuel_hours(X)
        return fuel + PENALTY * np.maximum(0.0, hours - deadline)

    # --- the three optimizers, same evaluation budget ---------------------- #
    evals = EVALS_PER_LEG * legs
    iters = evals // POP
    runs = {}                            # name -> (profile, seconds, evals so far, best so far)
    t = time.perf_counter()
    x, _, hist = optimizer.simulated_annealing(
        lambda v: float(objective(v)[0]), [(lo + hi) / 2.0] * legs, bounds,
        n_iter=evals, seed=args.seed)
    runs["SA"] = (x, time.perf_counter() - t, np.arange(len(hist)), hist)
    t = time.perf_counter()
    x, _, hist = ga.ga_minimize(objective, bounds, n_agents=POP, iters=iters,
                                seed=args.seed, vectorized=True)
    runs["GA"] = (x, time.perf_counter() - t,
                  POP * np.arange(1, iters + 1), hist["best"])
    t = time.perf_counter()
    x, _, hist = gsa.gsa_minimize(objective, bounds, n_agents=POP, iters=iters,
                                  seed=args.seed, vectorized=True)
    runs["GSA"] = (x, time.perf_counter() - t,
                   POP * np.arange(1, iters + 1), hist["best"])

    # --- reference (+ the QUBO on the single-speed problem) ---------------- #
    rows = [(name, r[0], r[1]) for name, r in runs.items()]
    bracket = ""
    if legs == 1:
        qubo, speeds, fuels, _, _ = qubo_speed.build_speed_qubo(
            pipe, scenario, args.distance, deadline)
        t = time.perf_counter()
        q = qubo_speed.solve_speed_qubo(qubo, speeds, fuels)
        rows.append(("QUBO", np.array([q["speed"]]), time.perf_counter() - t))
        ref_x = np.array([optimizer.grid_search(
            lambda s: objective(s[:, None]), (lo, hi))[0]])
        ref_label = "Grid (reference)"
    else:
        ref = lagrangian_reference(pipe, scenario, waves, wind, leg_nm,
                                   deadline, lo, hi)
        ref_x, ref_label = ref[0], "Lagrangian (reference)"
        bracket = (f"  Reference bracket: no profile on its speed grid can beat "
                   f"{ref[3]:.2f} t; the reference achieves {ref[1]:.2f} t.")
    ref_fuel = float(fuel_hours(ref_x)[0][0])
    rows.append((ref_label, ref_x, 0.0))
    base_x = np.full(legs, spec["service_speed"])
    rows.append((f"Baseline ({spec['service_speed']:.0f} kn)", base_x, 0.0))

    names = {"SA": "SA  (annealing)", "GA": "GA  (genetic)",
             "GSA": "GSA (gravitational)", "QUBO": "QUBO (neal, 40 levels)"}
    print("Speed optimization: SA vs GA vs GSA at equal budget")
    print(f"  Voyage {args.distance:.0f} nm in {legs} leg(s), deadline "
          f"{deadline:.0f} h  |  {evals:,} objective evaluations per optimizer")
    if legs > 1:
        print("  Sea state per leg (m): " + " ".join(f"{w:.1f}" for w in waves))
    print()
    print(f"  {'Method':<24}{'fuel (t)':>9}{'hours':>8}{'vs ref':>9}{'time':>8}"
          f"   speed per leg (kn)")
    table = []
    for name, prof, secs in rows:
        fuel, hours = (float(v[0]) for v in fuel_hours(prof))
        table.append({"method": names.get(name, name), "fuel": fuel, "hours": hours,
                      "vs_ref_pct": (fuel / ref_fuel - 1) * 100, "seconds": secs,
                      "speeds": list(prof)})
        late = "" if hours <= deadline + 0.01 else "  LATE"
        print(f"  {names.get(name, name):<24}{fuel:>9.2f}{hours:>8.1f}"
              f"{(fuel / ref_fuel - 1) * 100:>+8.1f}%{secs:>7.1f}s   "
              + " ".join(f"{v:.1f}" for v in prof) + late)
    if bracket:
        print("\n" + bracket)

    # --- plot: convergence + what each method chose ------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, (_, _, xs, ys) in runs.items():
        ax1.plot(xs, ys, color=COLORS[name], lw=2, label=name)
    ax1.axhline(ref_fuel, color=COLORS["Reference"], ls="--", lw=1.5,
                label=f"{ref_label} ({ref_fuel:.1f} t)")
    ax1.set_ylim(ref_fuel * 0.98, ref_fuel * 1.5)
    ax1.set_xlabel("Objective evaluations")
    ax1.set_ylabel("Best voyage fuel so far (t, incl. lateness penalty)")
    ax1.set_title("Convergence at equal budget")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    if legs == 1:                        # the U-curve with each method's pick
        grid = np.linspace(config.PHYSICAL_MIN_SPEED, hi, 300)
        ax2.plot(grid, fuel_hours(grid[:, None])[0], "-", color="#bbbbbb",
                 lw=1.5, label="Voyage fuel curve")
        ax2.axvspan(config.PHYSICAL_MIN_SPEED, min_spd, color="red", alpha=0.08,
                    label="Too slow (misses deadline)")
        for (name, prof, _), marker in zip(rows[:-1], "sv^Do"):
            key = name if name in COLORS else "Reference"
            ax2.plot(prof[0], fuel_hours(prof)[0][0], marker, color=COLORS[key],
                     ms=11, mec="black", label=f"{key}: {prof[0]:.1f} kn")
        ax2.set_xlabel("Speed (knots)")
        ax2.set_ylabel("Total voyage fuel (t)")
        ax2.set_title("Where each optimizer lands on the fuel curve")
    else:                                # speed chosen on each leg vs the sea state
        leg_ids = np.arange(1, legs + 1)
        ax2.bar(leg_ids, waves, color="#FCA5A5", alpha=0.5, label="Wave height (m)")
        ax2.set_ylabel("Wave height (m)")
        ax3 = ax2.twinx()
        for name, prof, _ in rows[:-1]:
            key = name if name in COLORS else "Reference"
            ax3.plot(leg_ids, prof, "o-", color=COLORS[key], lw=2, label=key)
        ax3.set_ylabel("Chosen speed (kn)")
        ax3.legend(loc="upper right", fontsize=9)
        ax2.set_xlabel("Leg")
        ax2.set_title("Speed profile: slow down where the sea is rough")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"SA vs GA vs GSA on speed  ({args.distance:.0f} nm, {legs} "
                 f"leg(s), {deadline:.0f} h deadline)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / (f"speed_compare_{args.distance:.0f}nm_"
                                f"{deadline:.0f}h_L{legs}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Saved plot -> {out}")
    results.save(out.stem, {
        "distance_nm": args.distance, "legs": legs, "deadline_h": deadline,
        "evals": evals, "waves": list(waves), "reference_fuel": ref_fuel,
        "lower_bound": (ref[3] if legs > 1 else None), "rows": table,
        "plot": out.name})


if __name__ == "__main__":
    main()
