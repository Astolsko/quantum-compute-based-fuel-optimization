"""
Central configuration for the Maritime Fuel Efficiency project.

Domain: SHIPS (Indian Coast Guard -- Problem Statement 77,
"Fuel consumption optimization using AI-based tools").

Keeping the feature schema, ship specifications, file paths, and voyage defaults
in one place means the data generator, the model trainer, and the optimizer all
agree on the same columns and ship classes -- no silent mismatches.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
# Plots + results/*.json. A full run (run_all.sh) sets FUEL_RUN_DIR to its own
# runs/<timestamp>/ folder, so a later run never overwrites an earlier one; a
# script run on its own writes to outputs/.
RUN_DIR = os.environ.get("FUEL_RUN_DIR")
OUTPUTS_DIR = Path(RUN_DIR) if RUN_DIR else ROOT_DIR / "outputs"

DATA_CSV = DATA_DIR / "ship_fuel_data.csv"
MODEL_PATH = MODELS_DIR / "ship_fuel_model.joblib"
META_PATH = MODELS_DIR / "model_meta.joblib"

# --------------------------------------------------------------------------- #
# Compute budget                                                               #
# --------------------------------------------------------------------------- #
# Cores used for model training and (via --jobs) the benchmarks. The default is
# laptop-safe on purpose: an all-core run is what a thin laptop cannot cool.
# On a workstation:   FUEL_JOBS=-1 python demo.py     (-1 = every core)
N_JOBS = int(os.environ.get("FUEL_JOBS", "2"))

# --------------------------------------------------------------------------- #
# Feature schema  (static = ship design, dynamic = voyage conditions)         #
# --------------------------------------------------------------------------- #
# Static numeric: properties of the vessel.
NUMERIC_STATIC = ["displacement_tonnes", "engine_power_kw", "hull_coefficient"]

# Dynamic numeric: conditions during the voyage leg.
NUMERIC_DYNAMIC = ["speed_knots", "wave_height_m", "wind_speed_kn", "current_speed_kn"]

# Categorical features.
CATEGORICAL = ["ship_type", "fuel_type", "load_condition"]

# What we predict: main + auxiliary fuel burn RATE, in tonnes per day.
TARGET = "fuel_rate_tpd"

NUMERIC_FEATURES = NUMERIC_STATIC + NUMERIC_DYNAMIC
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL

# The decision variable the optimizer controls.
DECISION_VAR = "speed_knots"

# Physics prior for the tree models: fuel rate never falls when these rise (+1)
# and never rises when they do (-1). Applied as monotonic constraints.
MONOTONE = {"speed_knots": 1, "wave_height_m": 1, "wind_speed_kn": 1,
            "displacement_tonnes": 1, "hull_coefficient": 1,
            "current_speed_kn": -1}

# --------------------------------------------------------------------------- #
# Ship classes (Indian Coast Guard style patrol fleet)                        #
# --------------------------------------------------------------------------- #
# For each class: sampling priors for the data generator (disp/power/hull/fuel),
# plus operational speeds (knots) used by the optimizer for search bounds and
# the "business as usual" baseline.
#   disp  = (mean, sd) displacement in tonnes
#   power = (mean, sd) installed engine power in kW
#   hull  = (mean, sd) hull/block coefficient (finer hull = lower)
#   fuel  = probability of each fuel type
SHIP_SPECS = {
    "Interceptor": dict(
        p=0.35, disp=(60, 15), power=(3000, 600), hull=(0.42, 0.04),
        fuel={"MGO": 1.0},
        max_speed=35.0, service_speed=28.0,
    ),
    "Fast_Patrol": dict(
        p=0.40, disp=(550, 120), power=(6000, 1200), hull=(0.50, 0.04),
        fuel={"MGO": 0.9, "VLSFO": 0.1},
        max_speed=28.0, service_speed=22.0,
    ),
    "Offshore_Patrol": dict(
        p=0.25, disp=(2000, 400), power=(9000, 1800), hull=(0.62, 0.04),
        fuel={"MGO": 0.6, "VLSFO": 0.3, "HFO": 0.1},
        max_speed=25.0, service_speed=18.0,
    ),
}

# CO2 emitted per tonne of fuel burned (IMO factors, t CO2 / t fuel).
EMISSION_FACTORS = {"MGO": 3.206, "VLSFO": 3.151, "HFO": 3.114}

# --------------------------------------------------------------------------- #
# Voyage / optimization defaults                                              #
# --------------------------------------------------------------------------- #
PHYSICAL_MIN_SPEED = 4.0          # knots; below this a ship loses steerage
DEFAULT_VOYAGE_NM = 600.0         # nautical miles for the demo leg
DEFAULT_DEADLINE_HOURS = 40.0     # must complete the leg within this time

# A representative scenario the optimizer/plot demo uses by default.
# (Everything except `speed_knots`, which the optimizer chooses.)
DEFAULT_SCENARIO = {
    "ship_type": "Offshore_Patrol",
    "fuel_type": "MGO",
    "load_condition": "Laden",
    "displacement_tonnes": 2000.0,
    "engine_power_kw": 9000.0,
    "hull_coefficient": 0.62,
    "wave_height_m": 1.5,
    "wind_speed_kn": 12.0,
    "current_speed_kn": 0.0,
}


def base_scenario_for(ship_type: str) -> dict:
    """A base ship scenario for a class, using that class's typical specs."""
    spec = SHIP_SPECS[ship_type]
    scn = dict(DEFAULT_SCENARIO)
    scn["ship_type"] = ship_type
    scn["displacement_tonnes"] = float(spec["disp"][0])
    scn["engine_power_kw"] = float(spec["power"][0])
    scn["hull_coefficient"] = float(spec["hull"][0])
    scn["fuel_type"] = max(spec["fuel"], key=spec["fuel"].get)   # most likely
    return scn
