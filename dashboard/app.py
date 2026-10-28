"""
dashboard/app.py  --  Phase 3: the interactive front end (Streamlit).

    python run.py dashboard          # then open the URL it prints
    python run.py dashboard --server.port 8600 --server.headless true

Four tabs, each a thin layer over code the scripts already use -- nothing is
re-implemented here, so what the dashboard shows is what the benchmarks measure:

  Voyage speed    optimizer.optimize_speed      ship + voyage + sea state -> speed
  Weather route   experiments/qubo_route.py     route QUBO vs exact DP vs straight line
  Fleet plan      fleet/                        which ship takes which mission
  Results         runs/<run>/                   RESULTS.md and plots of a full run

The model is the trained one in models/ (trained on first use if missing).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "fleet")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
import generate_data
import optimizer
from experiments import qubo_route
from solvers import backends

st.set_page_config(page_title="Fuel Optimization", page_icon="⚓", layout="wide")
TRUTH = generate_data.TruthModel()


@st.cache_resource(show_spinner="Loading the fuel model...")
def model():
    return optimizer.load_model()


def model_name():
    try:
        import joblib
        return joblib.load(config.META_PATH)["best_model"]
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# Tab 1: voyage speed                                                          #
# --------------------------------------------------------------------------- #
def tab_speed():
    st.subheader("Slow steaming: the fuel-minimising speed that still arrives on time")
    with st.form("speed"):
        c1, c2, c3 = st.columns(3)
        ship = c1.selectbox("Ship class", list(config.SHIP_SPECS), index=2)
        distance = c1.number_input("Distance (nm)", 50.0, 3000.0, config.DEFAULT_VOYAGE_NM, 50.0)
        deadline = c1.number_input("Deadline (h)", 2.0, 400.0, config.DEFAULT_DEADLINE_HOURS, 1.0)
        waves = c2.slider("Wave height (m)", 0.0, 8.0, 1.5, 0.1)
        wind = c2.slider("Wind (kn)", 0.0, 50.0, 5.0 + 6.0 * 1.5, 0.5)
        current = c2.slider("Current (kn, + = following)", -3.0, 3.0, 0.0, 0.1)
        load = c3.radio("Load", ["Laden", "Ballast"])
        fuel_type = c3.selectbox("Fuel", list(config.SHIP_SPECS[ship]["fuel"]))
        go = st.form_submit_button("Optimise speed", type="primary")
    if not go:
        st.info("Set the voyage and press **Optimise speed**.")
        return
    scn = {**config.base_scenario_for(ship), "wave_height_m": waves, "wind_speed_kn": wind,
           "current_speed_kn": current, "load_condition": load, "fuel_type": fuel_type}
    res = optimizer.optimize_speed(model(), scn, distance, deadline, truth=TRUTH)
    if not res["feasible"]:
        st.error(f"Infeasible: needs {res['deadline_min_speed']:.1f} kn, the "
                 f"{ship.replace('_', ' ')} tops out at {res['max_speed']:.0f} kn.")
        return
    m = st.columns(4)
    m[0].metric("Optimal speed", f"{res['opt_speed']:.1f} kn",
                f"{res['opt_speed'] - res['baseline_speed']:+.1f} kn vs service speed")
    m[1].metric("Voyage fuel", f"{res['opt_fuel']:.1f} t",
                f"{-res['fuel_saving_pct']:+.1f}%", delta_color="inverse")
    m[2].metric("CO₂ saved", f"{res['co2_saved_t']:.1f} t")
    m[3].metric("Decision regret", f"{res['regret_pct']:.2f}%",
                help="Extra TRUE fuel at the recommended speed vs the true optimum "
                     "(known only because the data is synthetic)")
    xs = np.linspace(config.PHYSICAL_MIN_SPEED, res["max_speed"], 300)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, optimizer.voyage_fuels(model(), scn, xs, distance), label="Model")
    ax.plot(xs, optimizer.voyage_fuels(TRUTH, scn, xs, distance), "--", color="black",
            lw=1, label="True physics")
    ax.axvspan(xs[0], res["deadline_min_speed"], color="grey", alpha=0.15,
               label="Misses the deadline")
    ax.scatter([res["baseline_speed"]], [res["baseline_fuel"]], color="#d62728", zorder=5,
               label=f"Service speed ({res['baseline_speed']:.0f} kn)")
    ax.scatter([res["opt_speed"]], [res["opt_fuel"]], color="#2ca02c", zorder=5,
               label=f"Optimised ({res['opt_speed']:.1f} kn)")
    ax.set_xlabel("Speed over ground (kn)")
    ax.set_ylabel("Voyage fuel (t)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Model: {model_name()}. SA and the 1,000-point grid agree to "
               f"{res['sa_vs_grid_speed_gap']:.2f} kn.")


# --------------------------------------------------------------------------- #
# Tab 2: weather routing                                                       #
# --------------------------------------------------------------------------- #
def tab_route():
    st.subheader("Weather routing as a QUBO -- checked against the exact optimum")
    with st.form("route"):
        c1, c2, c3 = st.columns(3)
        rows = c1.slider("Rows (north-south)", 3, 11, 5)
        cols = c1.slider("Columns (west-east stages)", 4, 25, 7)
        storms = c2.slider("Random storms (0 = one centred)", 0, 6, 0)
        seed = c2.number_input("Storm seed", 0, 999, 0)
        cruise = c2.slider("Cruise speed (kn)", 8.0, 24.0, 16.0, 0.5)
        ship = c3.selectbox("Ship class ", list(config.SHIP_SPECS), index=2)
        backend = c3.selectbox("QUBO solver", ["neal", "tabu", "ga", "gsa"])
        presolve = c3.checkbox("Presolve (fix ports, drop unreachable cells)")
        go = st.form_submit_button("Route", type="primary")
    if not go:
        st.info("Pick a grid and press **Route**.")
        return
    start = end = rows // 2
    waves = qubo_route.build_weather(
        rows, cols, qubo_route.random_storms(rows, cols, storms, int(seed)) if storms else None)
    cost = qubo_route.cell_costs(model(), config.base_scenario_for(ship), waves,
                                 config.DEFAULT_VOYAGE_NM, cruise)
    qubo, info = qubo_route.build_route_qubo(cost, start, end, presolve=presolve)
    sampler, label, _ = backends.make_sampler(backend)
    reads = 12 if backend in ("ga", "gsa") else max(500, 150 * cols)
    t = time.perf_counter()
    ss = backends.sample_qubo(sampler, backend, qubo, num_reads=reads)
    secs = time.perf_counter() - t
    path = qubo_route.decode_route(ss.first.sample, rows, cols, start, end)
    dp_path, dp_fuel = qubo_route.exact_best_dp(cost, start, end)
    naive = [start] * cols
    naive_fuel = qubo_route.path_fuel(naive, cost)
    m = st.columns(4)
    m[0].metric("QUBO size", f"{info['n_vars']} vars", f"{info['n_qubo_terms']} terms",
                delta_color="off")
    if path is None:
        m[1].metric("QUBO route", "invalid")
        st.warning(f"{label} returned a sample that breaks a constraint -- the "
                   "penalty landscape won. Try neal or tabu, or tick presolve.")
    else:
        f = qubo_route.path_fuel(path, cost)
        m[1].metric("QUBO route", f"{f:.2f} t", f"{(f / dp_fuel - 1) * 100:+.2f}% vs exact",
                    delta_color="inverse")
        m[2].metric("Saved vs straight line", f"{(1 - f / naive_fuel) * 100:.1f}%")
    m[3].metric("Solve time", f"{secs:.2f} s", label, delta_color="off")
    fig, ax = plt.subplots(figsize=(min(12, cols + 3), min(6, 0.7 * rows + 2)))
    im = ax.imshow(waves, origin="lower", aspect="auto", cmap="YlOrRd", vmin=1, vmax=8,
                   extent=[-0.5, cols - 0.5, -0.5, rows - 0.5])
    fig.colorbar(im, ax=ax, label="Wave height (m)")
    ax.plot(range(cols), naive, "o--", color="black", label=f"Straight ({naive_fuel:.1f} t)")
    ax.plot(range(cols), [r - 0.08 for r in dp_path], "s-", color="#16A34A",
            label=f"Exact DP ({dp_fuel:.1f} t)")
    if path is not None:
        ax.plot(range(cols), [r + 0.08 for r in path], "o-", color="#1f77b4", lw=2,
                label=f"QUBO ({backend})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Stage (west -> east)")
    ax.set_ylabel("Track")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tab 3: fleet plan                                                            #
# --------------------------------------------------------------------------- #
def tab_fleet():
    import fleet_config as fc
    import genetic
    import hybrid as hyb
    import matheuristic as mh
    import oracle
    import run_fleet
    import scenario as scen_mod

    st.subheader("Fleet planning: which ship takes which mission, in what order")
    with st.form("fleet"):
        c1, c2 = st.columns(2)
        ships = c1.slider("Ships", 2, 5, 3)
        missions = c1.slider("Missions", 3, 16, 6)
        seed = c1.number_input("Instance seed", 0, 999, 0)
        solvers = c2.multiselect("Solvers", ["ALNS", "GA", "Hybrid (QUBO + cuts)", "Exact DP"],
                                 default=["ALNS", "GA", "Exact DP"])
        budget = c2.select_slider("GA / hybrid budget (exact scorings)", [100, 200, 500, 1000],
                                  value=200)
        go = st.form_submit_button("Plan", type="primary")
    if not go:
        st.info(f"Oracle tables for a new instance take ~15-60 s to build (then cached). "
                f"Exact DP is offered up to 16 missions.")
        return
    scn = scen_mod.build_scenario(int(seed), n_ships=ships, n_missions=missions)
    with st.spinner("Building the exact voyage-oracle tables..."):
        ev = mh.Evaluator(scn, oracle.get_tables(scn, verbose=False))
    rows, best = [], (float("inf"), None, "")
    runs = {"ALNS": lambda: mh.alns(ev, 3000, seed=1),
            "GA": lambda: genetic.solve_ga(ev, budget=budget, seed=1)[:2],
            "Hybrid (QUBO + cuts)": lambda: hyb.solve_hybrid_cuts(ev, budget=budget, seed=42)[:2],
            "Exact DP": lambda: mh.exact_dp(ev)}
    for name in solvers:
        if name == "Exact DP" and missions > 16:
            continue
        with st.spinner(f"{name}..."):
            t = time.perf_counter()
            plan, fuel = runs[name]()
            rows.append({"solver": name, "fuel (t)": fuel, "seconds": time.perf_counter() - t})
        if plan is not None and fuel < best[0]:
            best = (fuel, plan, name)
    df = pd.DataFrame(rows)
    exact = df.loc[df.solver == "Exact DP", "fuel (t)"]
    if len(exact) and np.isfinite(exact.iloc[0]):
        df["gap"] = (df["fuel (t)"] / exact.iloc[0] - 1) * 100
    st.dataframe(df, hide_index=True)
    if best[1] is None:
        st.error("No feasible plan found -- the instance may be infeasible; try another seed.")
        return
    out = Path(tempfile.mkdtemp()) / "plan.png"
    run_fleet.plot_plan(scn, ev, best[1], f"Best plan ({best[2]}): {best[0]:.2f} t", out)
    st.image(str(out))
    st.text(mh.describe_plan(ev, best[1]))
    st.caption(f"Horizon {fc.horizon_slots(ships, missions)} slots of {fc.SLOT_HOURS:g} h, "
               f"{fc.N_ROWS}x{fc.N_COLS} cells of {fc.CELL_NM:g} nm.")


# --------------------------------------------------------------------------- #
# Tab 4: results of the full runs                                              #
# --------------------------------------------------------------------------- #
def tab_results():
    st.subheader("Full runs (run_all.sh)")
    runs = sorted((ROOT / "runs").glob("*/"), reverse=True)
    if not runs:
        st.info("No runs yet: `FUEL_JOBS=28 bash run_all.sh` creates runs/<date>_<time>/.")
        return
    run = st.selectbox("Run", runs, format_func=lambda p: p.name)
    man = run / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        st.caption(f"{m['started']} on {m['host']} -- {m['cpu']}, GPU: {m['gpu']}")
    res = run / "RESULTS.md"
    if res.exists():
        with st.expander("RESULTS.md", expanded=False):
            st.markdown(res.read_text(encoding="utf-8"))
    plots = sorted(run.rglob("*.png"))
    pick = st.multiselect("Plots", plots, default=plots[:2],
                          format_func=lambda p: str(p.relative_to(run)))
    for p in pick:
        st.image(str(p), caption=str(p.relative_to(run)))


st.title("⚓ Maritime fuel optimization")
st.caption("Indian Coast Guard PS-77 prototype -- synthetic data, physics-based "
           "fuel model, classical + quantum-inspired optimizers.")
t1, t2, t3, t4 = st.tabs(["Voyage speed", "Weather route", "Fleet plan", "Results"])
with t1:
    tab_speed()
with t2:
    tab_route()
with t3:
    tab_fleet()
with t4:
    tab_results()
