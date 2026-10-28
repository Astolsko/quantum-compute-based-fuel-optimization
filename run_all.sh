#!/usr/bin/env bash
# The full run: every result the READMEs report, lightest first.
# Meant for a workstation -- on a laptop run only tests/test_smoke.py.
#
#     conda activate quantum-fuel
#     FUEL_JOBS=28 bash run_all.sh            # cores to use (-1 = every core)
#
# Each run gets its own folder, runs/<date>_<time>/, so a later run never
# overwrites an earlier one:
#     run.log         everything printed
#     manifest.json   package versions, machine, settings (tools/manifest.py)
#     timings.tsv     step, seconds, exit code
#     *.png, BENCHMARK*.md, results/*.json, fleet/   what the steps produced
#     data/, models/  the dataset and trained model this run used
#     RESULTS.md      every results table (tools/make_report.py)
# A failed step is logged and the run goes on; the exit code says if any failed.
# Then refresh the README tables:  python run.py report --update-readme
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1                 # keep ~/.local packages out of the env
export FUEL_JOBS="${FUEL_JOBS:-2}"
JOBS="$FUEL_JOBS"
export FUEL_RUN_DIR="$PWD/runs/$(date +%Y-%m-%d_%H%M)"
mkdir -p "$FUEL_RUN_DIR"
exec > >(tee -a "$FUEL_RUN_DIR/run.log") 2>&1

FAILED=()
step() {                                  # step NAME COMMAND...
    local name=$1; shift
    echo; echo "=== [$name] $* ==="
    local t0=$SECONDS
    "$@"
    local rc=$?
    printf "%s\t%d\t%d\n" "$name" $((SECONDS - t0)) "$rc" >> "$FUEL_RUN_DIR/timings.tsv"
    [ "$rc" -eq 0 ] || { echo "!!! [$name] FAILED (exit $rc)"; FAILED+=("$name"); }
}

python -m tools.manifest "$FUEL_RUN_DIR"
step tests python tests/test_smoke.py

# --- Phase 1: data -> 7 models (+ decision regret) -> speed optimisation ----
step demo python demo.py

# --- Phase 2a/2b: QUBOs for speed and weather routing ----------------------
step qubo_speed_40h  python run.py qubo_speed
step qubo_speed_90h  python run.py qubo_speed --deadline 90
step route_5x7       python run.py qubo_route
step route_5x7_pre   python run.py qubo_route --presolve
step route_7x15      python run.py qubo_route --rows 7  --cols 15 --storms 3 --seed 1
step route_9x19      python run.py qubo_route --rows 9  --cols 19 --storms 4 --seed 3
step route_11x25     python run.py qubo_route --rows 11 --cols 25 --storms 6 --seed 5
step route_11x25_pre python run.py qubo_route --rows 11 --cols 25 --storms 6 --seed 5 --presolve

# --- Phase 2 extras: SA vs GA vs GSA, speed profile, joint speed + route ----
step speed_cmp_40h   python run.py speed_compare
step speed_cmp_90h   python run.py speed_compare --deadline 90
step speed_profile   python run.py speed_compare --legs 8 --storms 2 --deadline 50
step joint_3x5       python run.py joint_speed_route
step joint_profile   python run.py joint_speed_route --rows 1 --cols 8 --storms 2 --levels 4
step joint_5x7       python run.py joint_speed_route --rows 5 --cols 7 --storms 2

# --- Phase 2c: gate-model QAOA / VQE (+CVaR) on the same QUBOs ----------------
step gate_speed      python run.py quantum_gate --problem speed --levels 8
step gate_route_3x4  python run.py quantum_gate --problem route --rows 3 --cols 4 --storms 1 --seed 2
step gate_route_4x6  python run.py quantum_gate --problem route --rows 4 --cols 6 --storms 1 --seed 2 --presolve
step gate_route_5x7  python run.py quantum_gate --problem route --rows 5 --cols 7 --storms 1 --seed 2 --presolve --trials 3
step gate_route_5x8  python run.py quantum_gate --problem route --rows 5 --cols 8 --storms 1 --seed 2 --presolve --trials 2
step audit_seeds     python run.py audit_seeds

# --- The honest benchmark: raw, presolved, gate-model (the long part) -------
step bench_raw       python run.py benchmark --fast --with-qgso --jobs "$JOBS"
step bench_presolve  python run.py benchmark --fast --with-qgso --presolve --jobs "$JOBS"
step bench_gate      python run.py benchmark --gate --jobs "$JOBS"

# --- Fleet (Phase 4-5): demo plans, benchmark + scale study, QAOA ------------
for seed in 0 1 2; do step "fleet_plan_$seed" python run.py run_fleet --seed "$seed"; done
step fleet_scale     python run.py benchmark_fleet --seeds 10 --sizes 3x6,4x8,4x10,4x12,5x14,5x16
step fleet_qaoa_2x4  python run.py quantum_qaoa --backend aer --ships 2 --missions 4
step fleet_qaoa_2x4c python run.py quantum_qaoa --backend aer --ships 2 --missions 4 --cvar 0.25
step fleet_qaoa_3x6  python run.py quantum_qaoa --backend aer --ships 3 --missions 6
step fleet_qaoa_4x6  python run.py quantum_qaoa --backend aer --ships 4 --missions 6

cp -r data models "$FUEL_RUN_DIR/"        # the inputs this run's numbers came from
step report python run.py report --run "$FUEL_RUN_DIR"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All steps succeeded -> $FUEL_RUN_DIR"
else
    echo "FAILED steps: ${FAILED[*]} -> $FUEL_RUN_DIR/run.log"
fi
echo "Refresh the README tables:  python run.py report --run $FUEL_RUN_DIR --update-readme"
[ ${#FAILED[@]} -eq 0 ]
