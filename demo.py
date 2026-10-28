"""
demo.py  --  End-to-end Phase 1 demonstration (maritime / Indian Coast Guard).

Runs the whole pipeline in one command and prints a readable narrative:

    1. Generate synthetic ship-voyage data    (generate_data.py)
    2. Train + compare fuel-rate models        (train_model.py)
    3. Optimise speed for two voyage scenarios (optimizer.py)
    4. Save a total-fuel-vs-speed plot         (outputs/voyage_fuel_vs_speed.png)

The data is synthetic, so the true physics is known: the demo also reports each
model's DECISION REGRET and draws the true fuel curve next to the model's.

Run:
    python demo.py
    FUEL_JOBS=-1 python demo.py     # workstation: train on every core
    python demo.py --n 2000         # smaller dataset
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # save figures to file, no GUI window needed
import matplotlib.pyplot as plt

import config
import generate_data
import optimizer
import train_model
from core import results as run_results


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def plot_voyage_curve(pipe, scenario, distance_nm, res, path, truth=None):
    """Plot total voyage fuel vs speed, with the deadline cutoff + chosen speed."""
    spec = config.SHIP_SPECS[scenario["ship_type"]]
    _, _, xs, fs = optimizer.grid_search(
        lambda s: optimizer.voyage_fuels(pipe, scenario, s, distance_nm),
        (config.PHYSICAL_MIN_SPEED, spec["max_speed"]),
    )

    plt.figure(figsize=(8, 5))
    plt.plot(xs, fs, color="#1f77b4", label="Total voyage fuel (model)")
    if truth is not None:
        plt.plot(xs, optimizer.voyage_fuels(truth, scenario, xs, distance_nm),
                 "--", color="black", lw=1, label="True physics (hidden from the model)")

    # Shade speeds too slow to meet the deadline (infeasible region).
    dmin = res["deadline_min_speed"]
    plt.axvspan(config.PHYSICAL_MIN_SPEED, dmin, color="grey", alpha=0.15)
    plt.axvline(dmin, color="grey", linestyle=":",
                label=f"Deadline limit ({dmin:.1f} kn)")

    plt.scatter([res["baseline_speed"]], [res["baseline_fuel"]], color="#d62728",
                zorder=5, label=f"Baseline ({res['baseline_speed']:.0f} kn)")
    plt.scatter([res["opt_speed"]], [res["opt_fuel"]], color="#2ca02c",
                zorder=5, label=f"Optimized ({res['opt_speed']:.0f} kn)")

    plt.title("Total voyage fuel vs speed (slow steaming under a deadline)")
    plt.xlabel("Speed over ground (knots)")
    plt.ylabel(f"Total fuel for {distance_nm:.0f} nm leg (tonnes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def main(n: int = 10_000) -> None:
    truth = generate_data.TruthModel()      # known only because data is synthetic

    # --- 1. Data --------------------------------------------------------- #
    banner("STEP 1/4  Generating synthetic ship-voyage data")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_data.generate(n=n, seed=42)
    df.to_csv(config.DATA_CSV, index=False)
    print(f"Generated {len(df):,} voyage legs -> {config.DATA_CSV}")
    print(df.head().to_string(index=False))

    # --- 2. Models ------------------------------------------------------- #
    banner("STEP 2/4  Training & comparing fuel-rate models")
    results, best_name, best_pipe = train_model.train_all(df, truth=truth)
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    if train_model.physics_note(best_pipe):
        print(train_model.physics_note(best_pipe))
    train_model.save(best_name, best_pipe, df)
    print(f"Saved -> {config.MODEL_PATH}")

    # --- 3. Optimisation ------------------------------------------------- #
    banner("STEP 3/4  Quantum-inspired optimisation (simulated annealing)")
    scenario = dict(config.DEFAULT_SCENARIO)

    voyages = {
        "Tight schedule (40 h)": config.DEFAULT_DEADLINE_HOURS,
        "Relaxed schedule (90 h) -> slow steaming": 90.0,
    }

    first_res, voyage_res = None, {}
    for title, deadline in voyages.items():
        print(f"\n--- {title} ---")
        res = optimizer.optimize_speed(
            best_pipe, scenario, config.DEFAULT_VOYAGE_NM, deadline, truth=truth
        )
        optimizer.print_report(scenario, res)
        voyage_res[title] = res
        if first_res is None:
            first_res = res

    # --- 4. Plot --------------------------------------------------------- #
    banner("STEP 4/4  Saving total-fuel-vs-speed plot")
    out = config.OUTPUTS_DIR / "voyage_fuel_vs_speed.png"
    plot_voyage_curve(best_pipe, scenario, config.DEFAULT_VOYAGE_NM,
                      first_res, out, truth=truth)
    print(f"Saved plot -> {out}")
    coef = getattr(best_pipe, "coefficients", None)
    run_results.save("phase1", {
        "rows": n, "models": results.to_dict(orient="records"),
        "best_model": best_name, "physics": coef() if coef else None,
        "scenario": scenario, "voyages": voyage_res, "plot": out.name})

    banner("DONE -- Phase 1 maritime pipeline ran end to end.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="End-to-end Phase 1 demo.")
    p.add_argument("--n", type=int, default=10_000,
                   help="number of synthetic voyage legs to generate")
    main(p.parse_args().n)
