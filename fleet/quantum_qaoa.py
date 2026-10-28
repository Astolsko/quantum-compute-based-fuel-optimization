"""
quantum_qaoa.py -- run the fleet ASSIGNMENT QUBO on a REAL quantum optimizer (QAOA).

This is the step that turns "quantum-ready" into "quantum-running". The outer
assignment problem is already a QUBO (hybrid.build_assignment_qubo); here we solve
it with QAOA -- a genuine quantum optimization algorithm -- instead of the classical
neal stand-in. The candidate assignments QAOA returns are then scored EXACTLY by the
same classical oracle, exactly like the neal hybrid. Only the outer solver changes.

BACKENDS (swap with one argument)
---------------------------------
  "aer" : local Aer statevector SIMULATOR of a quantum computer. Works now, no
          account; on the GPU from 22 qubits when the GPU build is installed
          (solvers/gate.py), so 3 ships x 6 missions (18 qubits) and 4 x 6
          (24 qubits) are in reach. NOT real hardware.
  "ibm" : a REAL IBM quantum processor via Qiskit Runtime. Needs a free IBM
          Quantum account + API token (see below). Same QAOA, real qubits.

GET A FREE IBM QUANTUM TOKEN (for backend="ibm")
------------------------------------------------
  1. Sign up (free): https://quantum.ibm.com/
  2. Copy your API token from the dashboard.
  3. Save it once:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                          token="YOUR_TOKEN", overwrite=True)
     (or pass token=... straight into solve_assignment_qaoa).

HONEST EXPECTATION
------------------
On these small instances QAOA on real (noisy) hardware will most likely do WORSE
than neal -- near-term quantum is not a performance win here. The point is a
truthful demonstration: the SAME QUBO runs on a real quantum optimizer, judged by
the same exact evaluator. We report whatever number comes back.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

import fleet_config as fc
import hybrid as hyb
import matheuristic as mh

sys.path.insert(0, str(fc.PARENT_DIR))
from solvers import gate                             # read-only (Aer CPU/GPU, CVaR)

qubo_to_qp = gate.qubo_to_qp


# --------------------------------------------------------------------------- #
# Build a QAOA instance on the requested backend                               #
# --------------------------------------------------------------------------- #
def make_qaoa(backend="aer", reps=1, maxiter=80, token=None, seed=42,
              n_qubits=0, cvar=None):
    """Return (qaoa, backend_label). cvar=alpha trains on the best alpha of shots."""
    if backend == "aer":
        sampler, device = gate.make_sampler(n_qubits, seed)
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps,
                    transpiler=gate.pass_manager(), aggregation=cvar)
        return qaoa, f"Aer statevector SIMULATOR on {device} (not real hardware)"

    if backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        # token=None -> use the saved account
        service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                   if token else QiskitRuntimeService())
        qpu = service.least_busy(operational=True, simulator=False)
        pm = generate_preset_pass_manager(optimization_level=1, backend=qpu)
        qaoa = QAOA(sampler=SamplerV2(mode=qpu), optimizer=COBYLA(maxiter=maxiter),
                    reps=reps, transpiler=pm, aggregation=cvar)
        return qaoa, f"IBM QPU [{qpu.name}] (REAL quantum hardware)"

    raise ValueError(f"backend must be 'aer' or 'ibm', got {backend!r}")


# --------------------------------------------------------------------------- #
# Solve the fleet assignment with QAOA, score with the exact oracle            #
# --------------------------------------------------------------------------- #
def solve_assignment_qaoa(ev: mh.Evaluator, backend="aer", reps=1, maxiter=80,
                          topk=25, token=None, seed=42, verbose=True, cvar=None):
    """QAOA (quantum) proposes assignments; the exact oracle scores them.
    Returns (best_plan, best_fuel, info)."""
    qubo, solo, P = hyb.build_assignment_qubo(ev)
    qp = qubo_to_qp(qubo)
    n_qubits = len(qp.variables)
    qaoa, label = make_qaoa(backend, reps, maxiter, token, seed, n_qubits, cvar)

    if verbose:
        print(f"  QAOA on {label}")
        print(f"  assignment QUBO: {n_qubits} qubits, reps={reps}, "
              f"maxiter={maxiter}")

    samples, _ = gate.sample_qp(qaoa, qp)

    # QAOA returns a DISTRIBUTION of candidate bitstrings -- exactly what the
    # hybrid wants. Score the most probable ones exactly, keep the real best.
    samples.sort(key=lambda s: (-s[1], s[2]))          # most probable, then lowest energy
    best_plan, best_f, scored, seen = None, float("inf"), 0, set()
    for x, _, _ in samples:
        vd = gate.to_vardict(qp, x)
        assign = hyb.decode_assignment(vd, ev)
        key = tuple(tuple(sorted(assign[s.id])) for s in ev.scn.ships)
        if key in seen:
            continue
        seen.add(key)
        f, plan = hyb.exact_score(ev, assign)
        if f < best_f:
            best_f, best_plan = f, plan
        scored += 1
        if scored >= topk:
            break

    info = {"backend": label, "n_qubits": n_qubits, "assignments_scored": scored,
            "qaoa_best_energy": min(s[2] for s in samples)}
    return best_plan, best_f, info


def solve_on_hardware(ev: mh.Evaluator, backend_name=None, reps=1, maxiter=60,
                      shots=4096, topk=25, seed=42, verbose=True):
    """Run the assignment QUBO on a REAL IBM QPU, quota-economically.

    Strategy (standard for near-term hardware): optimise the QAOA parameters on
    the local simulator (free, many evaluations), then execute the ONE optimised
    circuit on the real QPU (a single job, minimal QPU time). The hardware
    measurement bitstrings are decoded into assignments and scored EXACTLY by the
    oracle -- identical to the neal/aer hybrids, only the sampler is real silicon.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_optimization.converters import QuadraticProgramToQubo

    qubo, solo, P = hyb.build_assignment_qubo(ev)
    qp = qubo_to_qp(qubo)
    names = [v.name for v in qp.variables]
    n = len(names)

    # 1. optimise QAOA parameters on the simulator
    op, _ = QuadraticProgramToQubo().convert(qp).to_ising()
    qaoa, _ = make_qaoa("aer", reps, maxiter, seed=seed, n_qubits=n)
    res = qaoa.compute_minimum_eigenvalue(op)

    # 2. bind the optimised parameters into the circuit -> measure -> transpile
    circ = res.optimal_circuit.assign_parameters(res.optimal_point)
    circ.measure_all()
    svc = QiskitRuntimeService()
    backend = (svc.backend(backend_name) if backend_name
               else svc.least_busy(operational=True, simulator=False))
    pm_hw = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm_hw.run(circ)

    # 3. ONE job on real hardware
    if verbose:
        print(f"  submitting 1 job to REAL QPU [{backend.name}] "
              f"({n} qubits, {shots} shots)...")
    job = SamplerV2(mode=backend).run([isa], shots=shots)
    if verbose:
        print(f"  job id: {job.job_id()}  -- waiting for the quantum computer...")
    result = job.result()
    counts = result[0].data.meas.get_counts()

    # 4. decode hardware bitstrings -> assignments -> exact oracle score
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    best_plan, best_f, scored, seen = None, float("inf"), 0, set()
    for bits, _cnt in ranked:
        vd = {names[i]: int(bits[-(i + 1)]) for i in range(n)}
        assign = hyb.decode_assignment(vd, ev)
        key = tuple(tuple(sorted(assign[s.id])) for s in ev.scn.ships)
        if key in seen:
            continue
        seen.add(key)
        f, plan = hyb.exact_score(ev, assign)
        if f < best_f:
            best_f, best_plan = f, plan
        scored += 1
        if scored >= topk:
            break

    usage = getattr(job, "usage_estimation", None)
    info = {"backend": backend.name, "n_qubits": n, "job_id": job.job_id(),
            "distinct_bitstrings": len(counts), "assignments_scored": scored,
            "usage": usage}
    return best_plan, best_f, info


if __name__ == "__main__":
    import argparse
    import time

    import oracle
    import scenario as scen_mod
    from core import results

    p = argparse.ArgumentParser(description="Run the fleet assignment on QAOA.")
    p.add_argument("--backend", choices=["aer", "ibm"], default="aer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ships", type=int, default=2, help="smaller = fewer qubits")
    p.add_argument("--missions", type=int, default=4)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--cvar", type=float, default=None,
                   help=f"train on the best alpha of shots (e.g. {gate.CVAR_ALPHA})")
    p.add_argument("--token", default=None, help="IBM Quantum API token")
    args = p.parse_args()

    scn = scen_mod.build_scenario(args.seed, n_ships=args.ships,
                                  n_missions=args.missions)
    tabs = oracle.get_tables(scn, verbose=False)
    ev = mh.Evaluator(scn, tabs)

    t = time.perf_counter()
    plan, f, info = solve_assignment_qaoa(
        ev, backend=args.backend, reps=args.reps, maxiter=args.maxiter,
        token=args.token, seed=args.seed + 1, cvar=args.cvar)
    secs = time.perf_counter() - t

    # references: the exact optimum (subset DP) and the neal hybrid
    pe, fe = mh.exact_dp(ev)
    _, nf, _ = hyb.solve_hybrid(ev, num_reads=200, seed=1)
    gap = (f - fe) / fe * 100
    print()
    print(f"  QAOA hybrid ({args.backend}{', CVaR' if args.cvar else ''}): {f:.2f} t   "
          f"({info['assignments_scored']} assignments scored, {secs:.1f}s)")
    print(f"  neal hybrid              : {nf:.2f} t")
    print(f"  exact optimum            : {fe:.2f} t   QAOA gap {gap:+.2f}%")
    if plan is not None:
        print(mh.describe_plan(ev, plan))
    tag = f"_{args.ships}x{args.missions}_s{args.seed}{'_cvar' if args.cvar else ''}"
    results.save(f"fleet_qaoa{tag}", {
        "backend": info["backend"], "ships": args.ships, "missions": args.missions,
        "seed": args.seed, "n_qubits": info["n_qubits"], "reps": args.reps,
        "maxiter": args.maxiter, "cvar": args.cvar, "qaoa_fuel": f,
        "neal_hybrid_fuel": nf, "exact_fuel": fe, "gap_pct": gap, "seconds": secs,
        "assignments_scored": info["assignments_scored"]}, fc.OUTPUTS_DIR)
