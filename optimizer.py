"""
optimizer.py  --  Quantum-inspired voyage optimizer (report section 7).

Once we can PREDICT a ship's fuel-burn RATE (tonnes/day), we can OPTIMIZE a
voyage. The maritime twist vs road vehicles:

  * Fuel RATE grows with the cube of speed, so "minimise fuel" alone just says
    "go as slow as possible" -- trivial and operationally useless.
  * The REAL problem is SLOW STEAMING: minimise the *total fuel for the leg*
    while still ARRIVING ON TIME. Total fuel = rate(speed) x voyage_time, and
    voyage_time = distance / speed, so:

        total_fuel(speed) = rate(speed) * distance / (24 * speed)

    Because the ship also burns a speed-independent "hotel load" per day, this
    total-fuel curve is U-SHAPED -- there is an "economic speed". The deadline
    adds a hard floor on speed (you cannot go slower than distance/deadline).

We minimise total voyage fuel with **Simulated Annealing** (the classical cousin
of quantum annealing -- hence "quantum-inspired"), and validate against a
brute-force grid search. We also report CO2 emissions (IMO factor) since the
Coast Guard problem statement calls out environmental compliance.
(The QUBO formulations and route optimization live in experiments/.)

Because the training data is synthetic, the true physics is known. Pass
`truth=generate_data.TruthModel()` and the result also reports DECISION REGRET:
how much more fuel the recommended speed really burns than the true optimum.
Matching the grid only proves the optimizer agrees with the MODEL; regret is the
check against reality.

Run:
    python optimizer.py
    python optimizer.py --distance 600 --deadline 80 --waves 4 --wind 30
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import sklearn

import config


# --------------------------------------------------------------------------- #
# Model loading + objective functions                                          #
# --------------------------------------------------------------------------- #
def load_model():
    """Load the trained pipeline; (re)train if it is missing or was saved by a
    different scikit-learn (its pickles are not portable across versions)."""
    try:
        if joblib.load(config.META_PATH).get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("saved by a different scikit-learn")
        return joblib.load(config.MODEL_PATH)
    except Exception as e:  # noqa: BLE001 -- any unusable model means "retrain"
        print(f"No usable trained model ({type(e).__name__}: {e}) -- training one now...")
        import train_model
        train_model.main()
        return joblib.load(config.MODEL_PATH)


def predict_rates(pipe, scenario: dict, speeds, **columns) -> np.ndarray:
    """Predicted fuel-burn rates (tonnes/day) at an array of speeds, in ONE model
    call. `columns` override scenario values row by row (e.g. wave_height_m=arr)."""
    speeds = np.asarray(speeds, dtype=float).ravel()
    X = pd.DataFrame({**scenario, **columns, config.DECISION_VAR: speeds})
    return np.asarray(pipe.predict(X[config.FEATURE_COLUMNS]), dtype=float)


def predict_rate(pipe, scenario: dict, speed: float) -> float:
    """Predicted fuel-burn rate (tonnes/day) for `scenario` at `speed` knots."""
    return float(predict_rates(pipe, scenario, [speed])[0])


def voyage_fuels(pipe, scenario: dict, speeds, distance_nm: float) -> np.ndarray:
    """Total fuel (tonnes) to cover `distance_nm` at each speed (vectorised)."""
    speeds = np.asarray(speeds, dtype=float).ravel()
    rate_tpd = predict_rates(pipe, scenario, speeds)     # tonnes / day
    return rate_tpd * distance_nm / (speeds * 24.0)      # x voyage time in days


def voyage_fuel(pipe, scenario: dict, speed: float, distance_nm: float) -> float:
    """Total fuel (tonnes) to cover `distance_nm` at `speed` knots."""
    return float(voyage_fuels(pipe, scenario, [speed], distance_nm)[0])


# --------------------------------------------------------------------------- #
# Simulated Annealing  (generic over a vector of decision variables)          #
# --------------------------------------------------------------------------- #
def simulated_annealing(objective, x0, bounds, *, n_iter=2000, t0=5.0,
                        cooling=0.995, seed=0):
    """Minimise `objective(x)` over a box `bounds` via simulated annealing.

    Returns (best_x, best_energy, best_energy_history).
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    step = 0.10 * (hi - lo)          # move size per dimension

    x = np.array(x0, dtype=float)
    energy = objective(x)
    best_x, best_energy = x.copy(), energy
    history = [best_energy]

    temp = t0
    for _ in range(n_iter):
        candidate = np.clip(x + rng.normal(0.0, step), lo, hi)
        cand_energy = objective(candidate)
        delta = cand_energy - energy
        # Accept if better, or probabilistically if worse (the SA twist).
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-9)):
            x, energy = candidate, cand_energy
        if energy < best_energy:
            best_x, best_energy = x.copy(), energy
        history.append(best_energy)
        temp *= cooling

    return best_x, best_energy, history


def grid_search(objective, bounds, n=1000):
    """Brute-force minimum of a 1-D objective -- the reference SA must match.
    `objective` is vectorised (array of x -> array of values): one model call."""
    xs = np.linspace(bounds[0], bounds[1], n)
    fs = np.asarray(objective(xs), dtype=float)
    i = int(np.argmin(fs))
    return xs[i], fs[i], xs, fs


# --------------------------------------------------------------------------- #
# Decision regret: score the model's choice against the TRUE physics           #
# --------------------------------------------------------------------------- #
def regret_panel(pipe, truth, n=400) -> np.ndarray:
    """Decision regret (%) of `pipe` on a fixed panel of voyages: every ship
    class x calm/rough sea x no deadline / tight deadline.

    Regret = extra TRUE fuel burned by sailing at the speed the model believes
    is best, versus the true optimum. It is what R2 cannot show: a model can be
    accurate on average and still put the minimum in the wrong place. Needs the
    true physics, so it exists only for synthetic data."""
    out = []
    for cls, spec in config.SHIP_SPECS.items():
        for waves in (1.5, 4.0):
            scn = {**config.base_scenario_for(cls), "wave_height_m": waves,
                   "wind_speed_kn": 5.0 + 6.0 * waves}     # training correlation
            for lower in (config.PHYSICAL_MIN_SPEED, 0.8 * spec["service_speed"]):
                xs = np.linspace(lower, spec["max_speed"], n)
                true = voyage_fuels(truth, scn, xs, 1.0)   # distance cancels out
                chosen = int(np.argmin(voyage_fuels(pipe, scn, xs, 1.0)))
                out.append((true[chosen] / true.min() - 1.0) * 100.0)
    return np.array(out)


# --------------------------------------------------------------------------- #
# High-level: optimise cruising speed for a voyage leg                         #
# --------------------------------------------------------------------------- #
def optimize_speed(pipe, scenario: dict, distance_nm: float,
                   deadline_hours: float, baseline_speed: float | None = None,
                   seed: int = 0, truth=None) -> dict:
    """Find the fuel-minimising speed that still meets the arrival deadline.
    With `truth` (the true physics) the result also carries decision regret."""
    spec = config.SHIP_SPECS[scenario["ship_type"]]
    max_speed = spec["max_speed"]
    if baseline_speed is None:                       # "business as usual" speed
        baseline_speed = spec["service_speed"]

    # Deadline imposes a minimum speed: distance / deadline.
    deadline_min_speed = distance_nm / deadline_hours
    lower = max(config.PHYSICAL_MIN_SPEED, deadline_min_speed)
    upper = max_speed
    emission_factor = config.EMISSION_FACTORS[scenario["fuel_type"]]

    # Infeasible: even at full speed the ship cannot arrive in time.
    if lower > upper:
        return {
            "feasible": False,
            "deadline_min_speed": deadline_min_speed,
            "max_speed": max_speed,
            "distance_nm": distance_nm,
            "deadline_hours": deadline_hours,
        }

    obj = lambda s: voyage_fuels(pipe, scenario, s, distance_nm)          # noqa: E731

    sa_x, sa_fuel, _ = simulated_annealing(
        lambda v: float(obj(v)[0]), x0=[(lower + upper) / 2],
        bounds=[(lower, upper)], seed=seed
    )
    grid_speed, grid_fuel, xs, _ = grid_search(obj, (lower, upper))
    baseline_fuel = float(obj([baseline_speed])[0])

    def hours(speed):
        return distance_nm / speed

    res = {
        "feasible": True,
        "distance_nm": distance_nm,
        "deadline_hours": deadline_hours,
        "deadline_min_speed": deadline_min_speed,
        "max_speed": max_speed,
        "fuel_type": scenario["fuel_type"],
        "baseline_speed": baseline_speed,
        "baseline_fuel": baseline_fuel,
        "baseline_hours": hours(baseline_speed),
        "baseline_co2": baseline_fuel * emission_factor,
        "opt_speed": float(sa_x[0]),
        "opt_fuel": float(sa_fuel),
        "opt_hours": hours(float(sa_x[0])),
        "opt_co2": float(sa_fuel) * emission_factor,
        "grid_speed": float(grid_speed),
        "grid_fuel": float(grid_fuel),
        "fuel_saving_pct": (baseline_fuel - sa_fuel) / baseline_fuel * 100,
        "co2_saved_t": (baseline_fuel - sa_fuel) * emission_factor,
        "sa_vs_grid_speed_gap": abs(float(sa_x[0]) - float(grid_speed)),
    }
    if truth is not None:               # synthetic data: re-price with real physics
        true = voyage_fuels(truth, scenario, xs, distance_nm)
        true_at_opt = voyage_fuel(truth, scenario, float(sa_x[0]), distance_nm)
        res.update(
            true_opt_speed=float(xs[np.argmin(true)]),
            true_opt_fuel=float(true.min()),
            true_fuel_at_opt=true_at_opt,
            regret_pct=max(0.0, (true_at_opt / float(true.min()) - 1.0) * 100.0),
        )
    return res


def print_report(scenario: dict, res: dict) -> None:
    print("Scenario (fixed inputs):")
    for k, v in scenario.items():
        print(f"  {k:<20} = {v}")

    if not res["feasible"]:
        print("\n  INFEASIBLE: deadline cannot be met even at full speed.")
        print(f"  Need {res['deadline_min_speed']:.1f} kn but max is "
              f"{res['max_speed']:.1f} kn.")
        return

    print(f"\n  Voyage: {res['distance_nm']:.0f} nm, "
          f"deadline {res['deadline_hours']:.0f} h "
          f"(=> min speed {res['deadline_min_speed']:.1f} kn)")
    print()
    print(f"  {'':<14}{'speed':>8}{'time':>9}{'fuel':>9}{'CO2':>9}")
    print(f"  {'':<14}{'(kn)':>8}{'(h)':>9}{'(t)':>9}{'(t)':>9}")
    print(f"  {'Baseline':<14}{res['baseline_speed']:>8.1f}"
          f"{res['baseline_hours']:>9.1f}{res['baseline_fuel']:>9.1f}"
          f"{res['baseline_co2']:>9.1f}")
    print(f"  {'Optimized':<14}{res['opt_speed']:>8.1f}"
          f"{res['opt_hours']:>9.1f}{res['opt_fuel']:>9.1f}"
          f"{res['opt_co2']:>9.1f}")
    print()
    print(f"  Fuel saving : {res['fuel_saving_pct']:.1f}%  "
          f"({res['baseline_fuel'] - res['opt_fuel']:.1f} t fuel, "
          f"{res['co2_saved_t']:.1f} t CO2 per leg)")
    print(f"  Validation  : SA vs grid speed gap "
          f"{res['sa_vs_grid_speed_gap']:.2f} kn (small = the model's optimum found)")
    if "regret_pct" in res:
        print(f"  True physics: that speed really burns {res['true_fuel_at_opt']:.1f} t; "
              f"the true optimum is {res['true_opt_speed']:.1f} kn at "
              f"{res['true_opt_fuel']:.1f} t  (decision regret "
              f"{res['regret_pct']:+.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Optimise ship speed for a voyage.")
    p.add_argument("--distance", type=float, default=config.DEFAULT_VOYAGE_NM,
                   help="voyage distance (nautical miles)")
    p.add_argument("--deadline", type=float, default=config.DEFAULT_DEADLINE_HOURS,
                   help="arrival deadline (hours)")
    p.add_argument("--waves", type=float,
                   default=config.DEFAULT_SCENARIO["wave_height_m"])
    p.add_argument("--wind", type=float,
                   default=config.DEFAULT_SCENARIO["wind_speed_kn"])
    p.add_argument("--current", type=float,
                   default=config.DEFAULT_SCENARIO["current_speed_kn"])
    p.add_argument("--baseline-speed", type=float, default=None)
    args = p.parse_args()

    scenario = dict(config.DEFAULT_SCENARIO)
    scenario["wave_height_m"] = args.waves
    scenario["wind_speed_kn"] = args.wind
    scenario["current_speed_kn"] = args.current

    pipe = load_model()
    res = optimize_speed(pipe, scenario, args.distance, args.deadline,
                         baseline_speed=args.baseline_speed)
    print_report(scenario, res)


if __name__ == "__main__":
    main()
