"""
qubo_route.py  --  Phase 2b: weather routing as a QUBO (the real headline).

THIS is where QUBO earns its keep. A ship must cross an ocean region from a start
port to a destination port. The region has weather (one or more storms with high
waves). The ship should pick the MINIMUM-FUEL PATH -- which may mean detouring
around storms even though it is longer.

Choosing among the combinatorially many paths is exactly the kind of problem
quantum annealers target. We formulate it as a QUBO and solve with `neal`; the
SAME QUBO could be sent to real D-Wave quantum hardware unchanged.

GRID + ENCODING (a monotone west->east path)
--------------------------------------------
* The region is a grid: n_cols columns (west->east stages) x n_rows rows
  (north-south positions). The ship occupies exactly ONE row per column and
  advances one column at a time (so the path always makes progress to the east).
* Binary variable per cell:  x[c,r] = 1  <=>  "the ship is in row r at column c".
* Each cell has a fuel cost computed by our PHASE 1 ML MODEL, using that cell's
  wave height + wind.

ENERGY  H  = sum_{c,r} cost[c,r] x[c,r]                         (fuel)
           + surcharge for diagonal (row-changing) moves        (extra distance)
           + P * sum_c (sum_r x[c,r] - 1)^2                      (one row per column)
           + P * (force start row at col 0, end row at last col)
           + P * sum  x[c,r] x[c+1,r']  for |r-r'| >= 2          (no row teleporting)

We validate the QUBO answer against an EXACT dynamic-programming solver (the
monotone grid path is a shortest-path problem), which scales to large grids.

PRESOLVE (--presolve)
---------------------
Two of those constraint families only restate things we already know: the ports
are fixed, and a cell more rows away from a port than it is columns away can
never be on a legal path. With --presolve those variables are substituted as
constants instead of being given a qubit and penalty terms: 3x4 goes from 12 to
6 variables, 5x7 from 35 to 21, 11x25 from 275 to 213. Same optimum, smaller
QUBO, fewer penalty cliffs.

Run (from the project root):
    python -m experiments.qubo_route
    python -m experiments.qubo_route --presolve
    python -m experiments.qubo_route --rows 7 --cols 15 --storms 3 --seed 1 --cruise 14
    python -m experiments.qubo_route --rows 9 --cols 19 --storms 4 --ship-type Fast_Patrol
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyqubo import Binary

import config
import generate_data
import optimizer
from core import results
from solvers import backends


# --------------------------------------------------------------------------- #
# Weather field + per-cell fuel cost (from the Phase 1 ML model)               #
# --------------------------------------------------------------------------- #
def build_weather(n_rows, n_cols, storms=None):
    """Wave-height field (metres). `storms` is a list of tuples
    (row_centre, col_centre, amplitude, row_spread, col_spread).
    If None, a single storm sits in the centre (back-compatible default)."""
    if storms is None:
        storms = [((n_rows - 1) / 2.0, (n_cols - 1) / 2.0, 6.0, 1.0, 1.3)]
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_cols)[None, :]
    waves = np.full((n_rows, n_cols), 1.0)            # calm-sea baseline
    for rc, cc, amp, sr, sc in storms:
        waves = waves + amp * np.exp(
            -(((rows - rc) ** 2) / (2 * sr ** 2)
              + ((cols - cc) ** 2) / (2 * sc ** 2))
        )
    return np.clip(waves, 0, 8)


def random_storms(n_rows, n_cols, n, seed=0):
    """Scatter `n` random storms across the interior of the grid."""
    rng = np.random.default_rng(seed)
    storms = []
    for _ in range(n):
        rc = rng.uniform(0, n_rows - 1)
        cc = rng.uniform(n_cols * 0.15, n_cols * 0.85)   # keep off the ports
        amp = rng.uniform(4.0, 7.0)
        sr = rng.uniform(0.8, 1.6)
        sc = rng.uniform(0.8, 2.0)
        storms.append((rc, cc, amp, sr, sc))
    return storms


def cell_rates(pipe, base_scenario, waves, speeds):
    """Fuel RATE (t/day) in every cell at every speed, from ONE model call.
    Returns an array [rows, cols, len(speeds)]."""
    speeds = np.atleast_1d(np.asarray(speeds, dtype=float))
    w = np.repeat(waves.ravel(), len(speeds))
    rates = optimizer.predict_rates(
        pipe, base_scenario, np.tile(speeds, waves.size), wave_height_m=w,
        wind_speed_kn=5.0 + 6.0 * w)    # wind correlates with sea state (as in training data)
    return rates.reshape(*waves.shape, len(speeds))


def cell_costs(pipe, base_scenario, waves, total_nm, cruise_kn):
    """Fuel (tonnes) to traverse each cell's leg, predicted by the ML model."""
    leg_days = total_nm / waves.shape[1] / (cruise_kn * 24.0)
    return cell_rates(pipe, base_scenario, waves, cruise_kn)[:, :, 0] * leg_days


# --------------------------------------------------------------------------- #
# QUBO construction + solve                                                    #
# --------------------------------------------------------------------------- #
def build_route_qubo(cost, start_row, end_row, diag_surcharge=0.10,
                     penalty_mult=10.0, presolve=False):
    """Build the routing QUBO. Returns (qubo_dict, info).

    `penalty_mult` scales the constraint penalty P = penalty_mult * max(cost).
    It only needs to exceed the largest single-cell fuel cost so that breaking a
    constraint is never worth it -- but the *size* of P reshapes the energy
    landscape, so it is swept in the benchmark (large P makes constraint cliffs
    that hurt gradient/population solvers; small P risks violations).

    `presolve` substitutes what is already known as constants -- the two port
    cells are 1, every other cell of the port columns and every cell too many
    rows from a port to be reachable is 0. The Hamiltonian below is unchanged;
    it is simply written over fewer variables."""
    n_rows, n_cols = cost.shape

    def var(c, r):
        if presolve:
            if c == 0:
                return int(r == start_row)
            if c == n_cols - 1:
                return int(r == end_row)
            if abs(r - start_row) > c or abs(r - end_row) > n_cols - 1 - c:
                return 0
        return Binary(f"x_{c}_{r}")

    x = {(c, r): var(c, r) for c in range(n_cols) for r in range(n_rows)}
    P = penalty_mult * float(cost.max())

    H = 0
    # Objective: fuel cost of every occupied cell.
    for c in range(n_cols):
        for r in range(n_rows):
            H += float(cost[r, c]) * x[(c, r)]

    # Diagonal moves cover extra distance -> small surcharge (discourage zigzag).
    for c in range(n_cols - 1):
        for r in range(n_rows):
            for r2 in (r - 1, r + 1):
                if 0 <= r2 < n_rows:
                    H += diag_surcharge * float(cost[r2, c + 1]) * x[(c, r)] * x[(c + 1, r2)]

    # Constraint 1: exactly one row per column.
    for c in range(n_cols):
        H += P * (sum(x[(c, r)] for r in range(n_rows)) - 1) ** 2

    # Constraint 2: fixed start row (col 0) and end row (last col).
    for r in range(n_rows):
        if r != start_row:
            H += P * x[(0, r)]
        if r != end_row:
            H += P * x[(n_cols - 1, r)]

    # Constraint 3: no teleporting -- consecutive rows differ by at most 1.
    for c in range(n_cols - 1):
        for r in range(n_rows):
            for r2 in range(n_rows):
                if abs(r - r2) >= 2:
                    H += P * x[(c, r)] * x[(c + 1, r2)]

    model = H.compile()
    qubo, offset = model.to_qubo()
    # energy(sample) + offset = fuel of the path, for any valid path
    info = {"n_vars": len({v for pair in qubo for v in pair}),
            "n_qubo_terms": len(qubo), "penalty": P, "offset": offset}
    return qubo, info


def decode_route(sample, n_rows, n_cols, start_row, end_row):
    """Turn a QUBO sample into a path (one row per column), or None if it breaks
    any constraint: one row per column, the fixed ports, no row teleporting.

    Port variables that presolve removed are absent from the sample and read as
    the constants they were replaced with."""
    ports = {f"x_0_{start_row}", f"x_{n_cols - 1}_{end_row}"}
    path = []
    for c in range(n_cols):
        on = [r for r in range(n_rows)
              if sample.get(f"x_{c}_{r}", int(f"x_{c}_{r}" in ports)) == 1]
        if len(on) != 1:
            return None
        path.append(on[0])
    if path[0] != start_row or path[-1] != end_row:
        return None
    if any(abs(path[c] - path[c + 1]) >= 2 for c in range(n_cols - 1)):
        return None
    return path


def path_fuel(path, cost, diag_surcharge=0.10):
    """Total fuel for a path = sum of cell costs + diagonal surcharges."""
    total = sum(cost[path[c], c] for c in range(len(path)))
    for c in range(len(path) - 1):
        if path[c] != path[c + 1]:
            total += diag_surcharge * cost[path[c + 1], c + 1]
    return float(total)


# --------------------------------------------------------------------------- #
# Exact validation via dynamic programming (shortest path on the grid)         #
# --------------------------------------------------------------------------- #
def exact_best_dp(cost, start_row, end_row, diag_surcharge=0.10):
    """Exact minimum-fuel monotone path via DP. O(rows*cols) -- scales to any grid."""
    n_rows, n_cols = cost.shape
    INF = float("inf")
    dp = [[INF] * n_rows for _ in range(n_cols)]
    back = [[-1] * n_rows for _ in range(n_cols)]
    dp[0][start_row] = float(cost[start_row, 0])

    for c in range(1, n_cols):
        for r in range(n_rows):
            best, bp = INF, -1
            for rp in (r - 1, r, r + 1):
                if 0 <= rp < n_rows and dp[c - 1][rp] < INF:
                    trans = diag_surcharge * float(cost[r, c]) if rp != r else 0.0
                    val = dp[c - 1][rp] + float(cost[r, c]) + trans
                    if val < best:
                        best, bp = val, rp
            dp[c][r], back[c][r] = best, bp

    total = dp[n_cols - 1][end_row]
    if total == INF:
        return None, INF
    path, r = [end_row], end_row
    for c in range(n_cols - 1, 0, -1):
        r = back[c][r]
        path.append(r)
    path.reverse()
    return path, float(total)


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def plot_routes(waves, naive_path, qubo_path, start_row, path_label, out):
    n_rows, n_cols = waves.shape
    plt.figure(figsize=(min(14, 1.0 * n_cols + 2), min(8, 0.8 * n_rows + 2)))
    im = plt.imshow(waves, origin="lower", aspect="auto", cmap="YlOrRd",
                    extent=[-0.5, n_cols - 0.5, -0.5, n_rows - 0.5], vmin=1, vmax=8)
    plt.colorbar(im, label="Wave height (m)")

    cols = list(range(n_cols))
    plt.plot(cols, naive_path, "o--", color="black", lw=2,
             label="Naive straight route")
    plt.plot(cols, qubo_path, "o-", color="#1f77b4", lw=2.5, label=path_label)
    plt.scatter([0, n_cols - 1], [start_row, qubo_path[-1]], s=220,
                marker="*", color="lime", edgecolor="black", zorder=5,
                label="Start / End port")

    plt.title("Weather routing: QUBO finds the minimum-fuel path around the storms")
    plt.xlabel("Stage (west -> east)")
    plt.ylabel("Track (south -> north)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Weather routing via QUBO + neal.")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--cruise", type=float, default=16.0, help="cruise speed (kn)")
    p.add_argument("--distance", type=float, default=config.DEFAULT_VOYAGE_NM)
    p.add_argument("--storms", type=int, default=0,
                   help="number of random storms (0 = one centred storm)")
    p.add_argument("--seed", type=int, default=0, help="random storm seed")
    p.add_argument("--ship-type", default=config.DEFAULT_SCENARIO["ship_type"],
                   choices=list(config.SHIP_SPECS.keys()))
    p.add_argument("--backend", default="neal",
                   choices=["neal", "dwave", "tabu", "gsa", "ga"],
                   help="who solves the QUBO: neal (classical, default), "
                        "dwave (REAL quantum annealer), tabu (classical), "
                        "gsa (gravitational search), ga (genetic algorithm)")
    p.add_argument("--presolve", action="store_true",
                   help="fix the ports and drop unreachable cells before "
                        "building the QUBO (fewer variables, same optimum)")
    p.add_argument("--num-reads", type=int, default=None,
                   help="number of samples/anneals (default scales with grid)")
    p.add_argument("--sweeps", type=int, default=1000,
                   help="neal annealing sweeps per read (ignored by dwave/tabu)")
    p.add_argument("--start-row", type=int, default=None,
                   help="start port latitude band (default: middle)")
    p.add_argument("--end-row", type=int, default=None,
                   help="end port latitude band (default: middle)")
    args = p.parse_args()

    n_rows, n_cols = args.rows, args.cols
    start_row = args.start_row if args.start_row is not None else n_rows // 2
    end_row = args.end_row if args.end_row is not None else n_rows // 2
    base = config.base_scenario_for(args.ship_type)
    pipe = optimizer.load_model()

    if args.storms > 0:
        storms = random_storms(n_rows, n_cols, args.storms, args.seed)
    else:
        storms = None
    waves = build_weather(n_rows, n_cols, storms)
    cost = cell_costs(pipe, base, waves, args.distance, args.cruise)

    # --- QUBO (quantum-inspired) ------------------------------------------ #
    # Same QUBO regardless of who solves it -- that is the whole point.
    qubo, info = build_route_qubo(cost, start_row, end_row,
                                  presolve=args.presolve)
    sampler, backend_label, is_quantum = backends.make_sampler(args.backend)
    if args.num_reads:
        num_reads = args.num_reads
    elif args.backend == "dwave":
        num_reads = 1000                      # conserve the free QPU minute
    elif args.backend in ("gsa", "ga"):
        num_reads = 12                        # restarts (each is a full population)
    else:
        num_reads = max(500, 150 * n_cols)
    sampleset = backends.sample_qubo(sampler, args.backend, qubo,
                                     num_reads=num_reads, sweeps=args.sweeps)
    qubo_path = decode_route(sampleset.first.sample, n_rows, n_cols,
                             start_row, end_row)
    run_note = backends.describe_run(sampleset, args.backend)

    # --- Validation (exact DP) + naive baseline --------------------------- #
    dp_path, dp_fuel = exact_best_dp(cost, start_row, end_row)
    # Naive baseline: the straight rhumb line from start to end (ignores weather).
    naive_path = [int(round(start_row + (end_row - start_row) * c / (n_cols - 1)))
                  for c in range(n_cols)]
    naive_fuel = path_fuel(naive_path, cost)

    print("Weather routing: QUBO vs exact DP vs naive straight line")
    print(f"  Ship: {args.ship_type}  |  cruise {args.cruise:.0f} kn  |  "
          f"{args.distance:.0f} nm  |  storms: "
          f"{args.storms if args.storms else '1 (centred)'} (seed {args.seed})")
    print(f"  Grid: {n_rows} rows x {n_cols} cols  ({info['n_vars']} binary vars"
          f"{' after presolve' if args.presolve else ''}, "
          f"{info['n_qubo_terms']} QUBO terms, {num_reads} reads)")
    print(f"  Solver: {backend_label}")
    if run_note:
        print(run_note)
    print(f"  Sea state: waves {waves.min():.1f}-{waves.max():.1f} m\n")

    solver_tag = f"QUBO ({args.backend})"
    if qubo_path is None:
        print("  QUBO returned an INVALID path (penalty/reads too low).")
        qubo_fuel, qubo_path, label = float("nan"), naive_path, f"{solver_tag} (invalid)"
    else:
        qubo_fuel, label = path_fuel(qubo_path, cost), f"QUBO route ({args.backend})"

    print(f"  {'Method':<24}{'fuel (t)':>10}   path (rows by stage)")
    print(f"  {'Naive straight':<24}{naive_fuel:>10.2f}   {naive_path}")
    print(f"  {solver_tag:<24}{qubo_fuel:>10.2f}   {qubo_path}")
    print(f"  {'Exact optimum (DP)':<24}{dp_fuel:>10.2f}   {dp_path}")
    print()
    saving = (naive_fuel - qubo_fuel) / naive_fuel * 100
    opt_gap = (qubo_fuel - dp_fuel) / dp_fuel * 100 if dp_fuel else float("nan")
    print(f"  Fuel saved by routing : {saving:.1f}%  "
          f"({naive_fuel - qubo_fuel:.2f} t per leg)")
    if abs(qubo_fuel - dp_fuel) < 1e-6:
        verdict = "YES (found the exact optimum)"
    elif qubo_path == dp_path:
        verdict = "YES (identical path)"
    else:
        verdict = f"within {opt_gap:.2f}% of exact (neal is heuristic)"
    print(f"  QUBO optimal? {verdict}")

    # Decision check against the TRUE physics (possible because the data is
    # synthetic): is the route the model likes also the route that really
    # burns least?
    regret = None
    if qubo_fuel == qubo_fuel:                          # a valid route was found
        true_cost = cell_costs(generate_data.TruthModel(), base, waves,
                               args.distance, args.cruise)
        _, true_best = exact_best_dp(true_cost, start_row, end_row)
        true_fuel = path_fuel(qubo_path, true_cost)
        regret = (true_fuel / true_best - 1) * 100
        print(f"  True physics : this route really burns {true_fuel:.2f} t; the "
              f"truly best route burns {true_best:.2f} t  (decision regret "
              f"{regret:+.1f}%)")

    out = config.OUTPUTS_DIR / (f"route_qubo_{n_rows}x{n_cols}_s{args.storms}"
                                f"_seed{args.seed}_r{start_row}-{end_row}"
                                f"{'_presolve' if args.presolve else ''}.png")
    plot_routes(waves, naive_path, qubo_path, start_row, label, out)
    print(f"\n  Saved plot -> {out}")
    results.save(out.stem, {
        "grid": f"{n_rows}x{n_cols}", "storms": args.storms, "seed": args.seed,
        "presolve": args.presolve, "backend": args.backend, "n_vars": info["n_vars"],
        "num_reads": num_reads, "naive_fuel": naive_fuel, "qubo_fuel": qubo_fuel,
        "dp_fuel": dp_fuel, "saving_pct": saving, "gap_pct": opt_gap,
        "valid": qubo_fuel == qubo_fuel, "regret_pct": regret, "plot": out.name})


if __name__ == "__main__":
    main()
