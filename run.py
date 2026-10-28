"""
run.py  --  run any part of the project from the project root.

    python run.py                       # list everything that can be run
    python run.py qubo_route --rows 5 --cols 7
    python run.py run_fleet --seed 0
    python run.py dashboard             # the Streamlit app (Phase 3)

The scripts themselves still run the old ways too (`python demo.py`,
`python -m experiments.qubo_route`, `cd fleet; python run_fleet.py`); this only
saves remembering which of the three a file needs.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name -> (how to run it, target, one line for the listing)
TARGETS = {
    # main pipeline (root)
    "demo":              ("file", "demo.py", "Phase 1 end to end: data -> 7 models -> speed optimisation -> plot"),
    "generate_data":     ("file", "generate_data.py", "synthetic voyage data -> data/"),
    "train_model":       ("file", "train_model.py", "train + compare the fuel models -> models/"),
    "optimizer":         ("file", "optimizer.py", "slow-steaming speed for one voyage"),
    # optional studies
    "qubo_speed":        ("module", "experiments.qubo_speed", "Phase 2a: speed as a QUBO"),
    "qubo_route":        ("module", "experiments.qubo_route", "Phase 2b: weather routing as a QUBO"),
    "quantum_gate":      ("module", "experiments.quantum_gate", "Phase 2c: the same QUBO on QAOA / VQE (+CVaR)"),
    "speed_compare":     ("module", "experiments.speed_compare", "SA vs GA vs GSA on speed (--legs N: speed profile)"),
    "joint_speed_route": ("module", "experiments.joint_speed_route", "speed and route in one QUBO"),
    "benchmark":         ("module", "experiments.benchmark", "the honest route-QUBO solver benchmark"),
    "audit_seeds":       ("module", "experiments.audit_seeds", "15-seed audit of every solver on one QUBO"),
    # fleet (Phase 4-5)
    "run_fleet":         ("fleet", "run_fleet.py", "fleet demo: one instance, every solver, plan plot"),
    "benchmark_fleet":   ("fleet", "benchmark_fleet.py", "fleet benchmark / scale study (--sizes)"),
    "quantum_qaoa":      ("fleet", "quantum_qaoa.py", "fleet assignment QUBO on QAOA (Aer or IBM)"),
    "fleet_scenario":    ("fleet", "scenario.py", "print one synthetic fleet instance"),
    # tools
    "report":            ("module", "tools.make_report", "results tables from a run folder (+ --update-readme)"),
    "test":              ("file", "tests/test_smoke.py", "the light smoke test (~1 min)"),
    "dashboard":         ("streamlit", "dashboard/app.py", "interactive dashboard (Streamlit)"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "list"):
        print(__doc__.strip().split("\n\n")[1] + "\n")
        for name, (_, target, what) in TARGETS.items():
            print(f"  {name:<18} {what}   [{target}]")
        return
    name, args = sys.argv[1], sys.argv[2:]
    if name not in TARGETS:
        sys.exit(f"unknown target {name!r} -- `python run.py` lists them")
    kind, target, _ = TARGETS[name]
    sys.path.insert(0, str(ROOT))
    if kind == "streamlit":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run",
                                  str(ROOT / target), *args]))
    sys.argv = [target, *args]
    if kind == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    elif kind == "fleet":
        sys.path.insert(0, str(ROOT / "fleet"))
        runpy.run_path(str(ROOT / "fleet" / target), run_name="__main__")
    else:
        sys.path.insert(0, str((ROOT / target).parent))
        runpy.run_path(str(ROOT / target), run_name="__main__")


if __name__ == "__main__":
    main()
