"""
joint_speed_route.py  --  Phase 2+: speed AND route in ONE QUBO
                          (with one row: the multi-segment speed profile).

qubo_route.py picks WHERE to sail at a fixed cruise speed; qubo_speed.py picks
ONE speed on a fixed track. A navigator chooses both together -- slow down
through the edge of a storm, or go around it and make the time up on calm water?
Here both decisions sit in one QUBO.

ENCODING
--------
  x[c, r, v] = 1  <=>  "at stage c the ship is in row r, sailing at speed level v"

  H(lam) =  sum (fuel[r,c,v] + lam * hours[v]) x[c,r,v]          (fuel + priced time)
          + 10% surcharge on diagonal moves (a longer leg: more fuel AND more time)
          + P * sum_c (sum_{r,v} x[c,r,v] - 1)^2                  (one cell+speed per stage)
          + P * (wrong start row / wrong end row)
          + P * sum x[c,r,v] x[c+1,r',v']  for |r - r'| >= 2      (no row teleporting)

THE DEADLINE: A PRICE, NOT A PENALTY
------------------------------------
"Total hours <= deadline" is an INEQUALITY. The textbook QUBO treatment is slack
bits: add ~8 binary variables that count the unused hours and square the whole
sum -- which couples every variable to every other one and adds a second penalty
scale to tune. Instead, time gets a PRICE lam (tonnes of fuel per hour): a plain
linear term, no extra qubits, the same sparsity as the route QUBO. A classical
loop bisects lam until the deadline is just met; every step is one QUBO solve,
so the quantum-portable part is unchanged.

What that costs: a price can only reach plans on the convex hull of the
fuel-vs-time trade-off, so the answer may sit slightly above the true optimum.
That gap is MEASURED here, not assumed away -- an exact label-setting DP gives
the true constrained optimum to compare against.

SPECIAL CASES
-------------
  --rows 1     no routing left: a multi-segment SPEED PROFILE (a speed per leg
               under one total-time deadline).
  --levels 1   no speed choice left: the route QUBO of qubo_route.py.

Run (from the project root):
    python -m experiments.joint_speed_route
    python -m experiments.joint_speed_route --rows 1 --cols 8 --storms 2 --levels 4
    python -m experiments.joint_speed_route --rows 5 --cols 7 --storms 2 --deadline 44
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyqubo import Binary

import config
import optimizer
from core import results
from experiments import qubo_route
from solvers import backends

DIAG = 0.10                 # extra distance of a row-changing move (as in qubo_route)


# --------------------------------------------------------------------------- #
# QUBO                                                                         #
# --------------------------------------------------------------------------- #
def build_joint_qubo(fuel, hours, start_row, end_row, lam, penalty_mult=10.0):
    """QUBO of (fuel + lam * time) over x[c, r, v].

    fuel  [rows, cols, levels]  tonnes to cross each cell at each speed level
    hours [levels]              hours per leg at each speed level
    """
    n_rows, n_cols, n_lev = fuel.shape
    w = fuel + lam * hours                       # priced cost of every (cell, speed)
    P = penalty_mult * float(w.max())
    states = [(r, v) for r in range(n_rows) for v in range(n_lev)]
    x = {(c, r, v): Binary(f"x_{c}_{r}_{v}")
         for c in range(n_cols) for r, v in states}

    H = 0
    for c in range(n_cols):
        for r, v in states:
            H += float(w[r, c, v]) * x[(c, r, v)]
        # exactly one (row, speed) per stage
        H += P * (sum(x[(c, r, v)] for r, v in states) - 1) ** 2

    for c in range(n_cols - 1):
        for r, v in states:
            for r2, v2 in states:
                if abs(r - r2) == 1:             # diagonal: longer leg
                    H += (DIAG * float(w[r2, c + 1, v2])
                          * x[(c, r, v)] * x[(c + 1, r2, v2)])
                elif abs(r - r2) >= 2:           # no teleporting
                    H += P * x[(c, r, v)] * x[(c + 1, r2, v2)]

    for r, v in states:                          # fixed ports
        if r != start_row:
            H += P * x[(0, r, v)]
        if r != end_row:
            H += P * x[(n_cols - 1, r, v)]

    return H.compile().to_qubo()[0]


def decode_joint(sample, shape, start_row, end_row):
    """Sample -> [(row, level) for each stage], or None if a constraint is broken."""
    n_rows, n_cols, n_lev = shape
    plan = []
    for c in range(n_cols):
        on = [(r, v) for r in range(n_rows) for v in range(n_lev)
              if sample.get(f"x_{c}_{r}_{v}", 0) == 1]
        if len(on) != 1:
            return None
        plan.append(on[0])
    if plan[0][0] != start_row or plan[-1][0] != end_row:
        return None
    if any(abs(plan[c][0] - plan[c + 1][0]) >= 2 for c in range(n_cols - 1)):
        return None
    return plan


def plan_totals(plan, fuel, hours):
    """(tonnes, hours) of a plan; a diagonal move costs 10% more of both."""
    tonnes = time = 0.0
    for c, (r, v) in enumerate(plan):
        k = 1.0 + DIAG * (c > 0 and plan[c - 1][0] != r)
        tonnes += k * float(fuel[r, c, v])
        time += k * float(hours[v])
    return tonnes, time


# --------------------------------------------------------------------------- #
# Exact reference: label-setting DP over (stage, row)                          #
# --------------------------------------------------------------------------- #
def exact_joint(fuel, hours, start_row, end_row, deadline):
    """The true minimum-fuel plan that meets the deadline (None if impossible).

    Each cell keeps only its non-dominated (hours, fuel) labels: a partial plan
    that is both slower and thirstier than another one reaching the same cell
    can never be part of the optimum."""
    n_rows, n_cols, n_lev = fuel.shape

    def prune(labels):
        keep, least_fuel = [], float("inf")
        for h, f, trail in sorted(labels, key=lambda lab: (lab[0], lab[1])):
            if h <= deadline + 1e-9 and f < least_fuel - 1e-12:
                keep.append((h, f, trail))
                least_fuel = f
        return keep

    # a trail is a linked list (previous trail, (row, level)) -- O(1) to extend
    front = {start_row: prune([(float(hours[v]), float(fuel[start_row, 0, v]),
                                (None, (start_row, v))) for v in range(n_lev)])}
    for c in range(1, n_cols):
        nxt = {}
        for r in range(n_rows):
            cand = []
            for rp in (r - 1, r, r + 1):
                k = 1.0 + DIAG * (rp != r)
                for h, f, trail in front.get(rp, []):
                    for v in range(n_lev):
                        cand.append((h + k * float(hours[v]),
                                     f + k * float(fuel[r, c, v]),
                                     (trail, (r, v))))
            nxt[r] = prune(cand)
        front = nxt

    final = front.get(end_row, [])
    if not final:
        return None
    trail, plan = min(final, key=lambda lab: lab[1])[2], []
    while trail is not None:
        trail, step = trail
        plan.append(step)
    return plan[::-1]


# --------------------------------------------------------------------------- #
# The outer loop: bisect the price of time                                     #
# --------------------------------------------------------------------------- #
def solve_lagrangian(fuel, hours, deadline, solve, iters=10):
    """Find the cheapest plan that meets the deadline by pricing time.

    `solve(lam)` returns a valid plan minimising fuel + lam * hours (or None).
    Returns (best on-time plan or None, [(lam, tonnes, hours), ...])."""
    trace = []

    def on_time(lam):
        plan = solve(lam)
        tonnes, time = (plan_totals(plan, fuel, hours) if plan
                        else (float("nan"), float("inf")))
        trace.append((lam, tonnes, time))
        return plan if time <= deadline + 1e-9 else None

    best = on_time(0.0)
    if best is not None:                         # the deadline does not bind
        return best, trace

    # A price so high that saving any time beats saving any fuel. `unit` is the
    # smallest difference in hours one choice can make.
    unit = min([*np.diff(np.unique(hours)), DIAG * float(hours.min())])
    lo, hi = 0.0, 2.0 * fuel.shape[1] * float(fuel.max()) / unit
    best = on_time(hi)
    if best is None:                             # unreachable even at full haste
        return None, trace
    for _ in range(iters):
        lam = (lo + hi) / 2.0
        plan = on_time(lam)
        if plan is None:
            lo = lam
        else:
            hi = lam
            if plan_totals(plan, fuel, hours)[0] < plan_totals(best, fuel, hours)[0]:
                best = plan
    return best, trace


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def plot_plan(waves, plan, naive, speeds, title, out):
    n_rows, n_cols = waves.shape
    stages = np.arange(n_cols)
    fig, ax = plt.subplots(figsize=(min(14, n_cols + 4), min(8, 0.8 * n_rows + 3)))
    ax.imshow(waves, origin="lower", aspect="auto", cmap="YlOrRd", vmin=1, vmax=8,
              extent=[-0.5, n_cols - 0.5, -0.5, n_rows - 0.5])
    if naive is not None:
        ax.plot(stages, [r for r, _ in naive], "--", color="black", lw=1.5,
                label="Naive: straight line, one constant speed")
    rows = [r for r, _ in plan]
    ax.plot(stages, rows, "-", color="#1f77b4", lw=2, zorder=2,
            label="Joint speed + route plan")
    dots = ax.scatter(stages, rows, c=[speeds[v] for _, v in plan], cmap="viridis",
                      vmin=speeds.min(), vmax=speeds.max(), s=170,
                      edgecolor="black", zorder=3)
    for c, (r, v) in enumerate(plan):
        ax.annotate(f"{speeds[v]:.0f}", (c, r), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8)
    fig.colorbar(dots, ax=ax, label="Speed on this leg (kn)")
    ax.set_title(title + "\n(background: wave height, pale = calm, dark red = storm)")
    ax.set_xlabel("Stage (west -> east)")
    ax.set_ylabel("Track (south -> north)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Joint speed + route in one QUBO.")
    p.add_argument("--rows", type=int, default=3, help="1 = speed profile only")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--levels", type=int, default=3, help="speed levels per leg")
    p.add_argument("--vmin", type=float, default=10.0, help="slowest level (kn)")
    p.add_argument("--vmax", type=float, default=None,
                   help="fastest level (kn); default = the class's service speed")
    p.add_argument("--deadline", type=float, default=None,
                   help="hours; default = the straight line at the mean speed level")
    p.add_argument("--distance", type=float, default=config.DEFAULT_VOYAGE_NM)
    p.add_argument("--storms", type=int, default=0,
                   help="number of random storms (0 = one centred storm)")
    p.add_argument("--seed", type=int, default=0, help="random storm seed")
    p.add_argument("--ship-type", default=config.DEFAULT_SCENARIO["ship_type"],
                   choices=list(config.SHIP_SPECS.keys()))
    p.add_argument("--backend", default="neal",
                   choices=["neal", "dwave", "tabu", "gsa", "ga"])
    p.add_argument("--num-reads", type=int, default=500)
    p.add_argument("--sweeps", type=int, default=1000)
    args = p.parse_args()

    n_rows, n_cols = args.rows, args.cols
    start_row = end_row = n_rows // 2
    vmax = args.vmax or config.SHIP_SPECS[args.ship_type]["service_speed"]
    speeds = (np.array([vmax]) if args.levels == 1
              else np.linspace(args.vmin, vmax, args.levels))
    deadline = args.deadline or args.distance / float(speeds.mean())

    base = config.base_scenario_for(args.ship_type)
    pipe = optimizer.load_model()
    storms = (qubo_route.random_storms(n_rows, n_cols, args.storms, args.seed)
              if args.storms > 0 else None)
    waves = qubo_route.build_weather(n_rows, n_cols, storms)
    hours = args.distance / n_cols / speeds                         # per leg
    fuel = qubo_route.cell_rates(pipe, base, waves, speeds) * hours / 24.0

    sampler, backend_label, _ = backends.make_sampler(args.backend)

    def solve(lam):
        qubo = build_joint_qubo(fuel, hours, start_row, end_row, lam)
        ss = backends.sample_qubo(sampler, args.backend, qubo,
                                  num_reads=args.num_reads, sweeps=args.sweeps)
        for sample in ss.samples():              # lowest energy first
            plan = decode_joint(sample, fuel.shape, start_row, end_row)
            if plan is not None:
                return plan
        return None

    plan, trace = solve_lagrangian(fuel, hours, deadline, solve)
    exact = exact_joint(fuel, hours, start_row, end_row, deadline)
    # Naive: the straight line at the slowest single speed that is still on time.
    naive = next(([(start_row, v)] * n_cols for v in range(len(speeds))
                  if n_cols * hours[v] <= deadline + 1e-9), None)

    print("Joint speed + route: QUBO with a price on time vs exact vs naive")
    print(f"  Ship: {args.ship_type}  |  {args.distance:.0f} nm  |  deadline "
          f"{deadline:.1f} h  |  speed levels (kn): "
          + " ".join(f"{s:.0f}" for s in speeds))
    print(f"  Grid: {n_rows} rows x {n_cols} cols x {len(speeds)} speeds = "
          f"{fuel.size} binary vars  |  sea state {waves.min():.1f}-{waves.max():.1f} m")
    print(f"  Solver: {backend_label}  ({len(trace)} QUBO solves; price of time "
          f"settled at {trace[-1][0]:.2f} t/h)\n")

    def show(name, pl):
        if pl is None:
            print(f"  {name:<28}{'none on time':>18}")
            return float("nan")
        tonnes, time = plan_totals(pl, fuel, hours)
        print(f"  {name:<28}{tonnes:>9.2f}{time:>9.1f}   rows "
              f"{[r for r, _ in pl]}  kn {[int(round(speeds[v])) for _, v in pl]}")
        return tonnes

    print(f"  {'Method':<28}{'fuel (t)':>9}{'hours':>9}")
    f_naive = show("Naive (straight, constant)", naive)
    f_qubo = show(f"QUBO + price ({args.backend})", plan)
    f_exact = show("Exact optimum (DP)", exact)
    print()
    if plan is None:
        print("  The QUBO loop found no on-time plan (deadline unreachable, or "
              "reads too low).")
        return
    if naive is not None:
        print(f"  Fuel saved vs naive : {(1 - f_qubo / f_naive) * 100:.1f}%")
    print(f"  QUBO vs exact       : {(f_qubo / f_exact - 1) * 100:+.2f}%  "
          f"(price-of-time gap + sampler error)")

    out = config.OUTPUTS_DIR / (f"joint_{n_rows}x{n_cols}x{len(speeds)}"
                                f"_s{args.storms}_seed{args.seed}.png")
    plot_plan(waves, plan, naive, speeds,
              f"Speed + route in one QUBO: {f_qubo:.1f} t in "
              f"{plan_totals(plan, fuel, hours)[1]:.1f} h (deadline {deadline:.1f} h)",
              out)
    print(f"\n  Saved plot -> {out}")
    results.save(out.stem, {
        "grid": f"{n_rows}x{n_cols}", "levels": list(speeds), "deadline_h": deadline,
        "storms": args.storms, "seed": args.seed, "n_vars": int(fuel.size),
        "qubo_solves": len(trace), "price_of_time": trace[-1][0],
        "naive_fuel": f_naive, "qubo_fuel": f_qubo, "exact_fuel": f_exact,
        "qubo_hours": plan_totals(plan, fuel, hours)[1], "plot": out.name})


if __name__ == "__main__":
    main()
