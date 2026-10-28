"""
fleet_config.py -- constants for the FLEET-LEVEL problem (Phase 4).

This package reformulates the research problem one level up from a single
voyage: a Coast Guard district has K vessels and M missions (patrol legs with
time windows) in a region with MOVING weather. The decision is
    which ship takes which mission, in what order, departing when,
and every voyage inside that plan is solved EXACTLY by a time-expanded
dynamic program (the "oracle") that uses the Phase-1 ML fuel model.

Nothing in the parent directory is modified -- we import it read-only.

DISCRETISATION
--------------
* Region: N_ROWS x N_COLS grid, each cell CELL_NM nautical miles square.
* Time: at least HORIZON_SLOTS slots of SLOT_HOURS hours (longer when ships
  carry more missions -- see horizon_slots). Weather is a 3-D field
  waves[t, r, c] -- storms DRIFT during the horizon (this is what makes the
  voyage DP time-expanded: the best route depends on WHEN you sail).
* Speeds: each ship class gets N_SPEEDS discrete cruise speeds between
  MIN_FLEET_SPEED and its service speed (all inside the ML model's training
  range, so the model interpolates rather than extrapolates).

MODELLING ASSUMPTIONS (v1 -- documented, revisit later)
------------------------------------------------------
* Waiting is free only when anchored at a port / mission endpoint ("stations");
  ships do not loiter mid-ocean.
* A leg's sea state is taken at the DESTINATION cell at DEPARTURE time
  (legs are 1-3 h; storms drift slowly relative to that).
* Fuel burn uses true leg duration dist/speed; only ARRIVAL TIMES are
  quantised to slots (ceil-free rounding, min 1 slot).
* Ships need not return home after their last mission.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
FLEET_DIR = Path(__file__).resolve().parent
PARENT_DIR = FLEET_DIR.parent                       # existing Phase 1-2 project
# Plots + results/*.json: the run folder when run_all.sh set one, else fleet/outputs.
OUTPUTS_DIR = (Path(os.environ["FUEL_RUN_DIR"]) / "fleet" if os.environ.get("FUEL_RUN_DIR")
               else FLEET_DIR / "outputs")
# Oracle tables (large, regenerable) are cached here across runs, keyed by the
# scenario AND the fuel model that priced them.
CACHE_DIR = FLEET_DIR / "cache"

# --------------------------------------------------------------------------- #
# Region / time discretisation                                                 #
# --------------------------------------------------------------------------- #
N_ROWS = 10                     # north-south cells
N_COLS = 20                     # west-east cells
CELL_NM = 15.0                  # nm per cell edge  (region ~ 300 x 150 nm)
DIAG_NM = CELL_NM * 2 ** 0.5

HORIZON_SLOTS = 36              # planning horizon (slots), the minimum
SLOT_HOURS = 1.0                # hours per slot
WAVE_SLOTS = 11                 # slots per "wave" of mission start windows

# --------------------------------------------------------------------------- #
# Ship speed discretisation                                                    #
# --------------------------------------------------------------------------- #
N_SPEEDS = 5
MIN_FLEET_SPEED = 8.0           # kn -- keeps every class inside training range

# 8 movement directions: (dr, dc, distance_nm)
MOVES = [(-1, 0, CELL_NM), (1, 0, CELL_NM), (0, -1, CELL_NM), (0, 1, CELL_NM),
         (-1, -1, DIAG_NM), (-1, 1, DIAG_NM), (1, -1, DIAG_NM), (1, 1, DIAG_NM)]

# --------------------------------------------------------------------------- #
# Default fleet instance size                                                  #
# --------------------------------------------------------------------------- #
DEFAULT_N_SHIPS = 3
DEFAULT_N_MISSIONS = 6


def horizon_slots(n_ships: int, n_missions: int) -> int:
    """Planning horizon for an instance. Mission windows are staggered in waves,
    one mission per ship per wave; each wave gets WAVE_SLOTS slots, plus 14 of
    lead-in and wind-down. Fixed at 36 slots, every instance past ~2 missions
    per ship was infeasible (measured 2026-10-28: all of 4x14 and 5x16); the
    default 3 ships x 6 missions is 2 waves = 36 slots, exactly as before."""
    return max(HORIZON_SLOTS, 14 + math.ceil(n_missions / n_ships) * WAVE_SLOTS)
