"""HONEST AUDIT: run every solver across MANY seeds on the same 3x4 route QUBO
(the one the qseed-11 diagram used) and report the full distribution -- so the
cherry-picking is visible. No best-of, no selected seed: every run counted.
QAOA/VQE run twice: minimising the mean energy, and CVaR (best 25% of shots).

Run (from the project root):
    python -m experiments.audit_seeds
"""
import types
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from core import results, stats
from experiments import quantum_gate as qg
from solvers import gate

N = 15


def main() -> None:
    a = types.SimpleNamespace(problem="route", rows=3, cols=4, storms=1, seed=2,
                              cruise=16.0, distance=600.0, ship_type="Offshore_Patrol")
    qubo, decode, meta, ctx = qg.build_route_problem(a)
    qp = qg.qubo_to_qp(qubo)
    dp = ctx["dp_fuel"]
    print(f"3x4 route, {N} seeds each. Exact optimum = {dp:.2f} t (path {ctx['dp_path']})\n")
    print(f"{'solver':<12}{'hit opt':>7}{'95% CI':>12}{'feasible':>10}{'mean gap':>10}{'worst gap':>11}")
    print("-" * 64)
    rows = []

    def run_all(name, fn, seeds):
        hits = feas = 0; gaps = []
        for s in seeds:
            path, fuel, _ = decode(fn(s))
            if path is not None:
                feas += 1
                g = (fuel - dp) / dp * 100
                gaps.append(g)
                if abs(g) < 1e-6: hits += 1
        lo, hi = stats.wilson(hits, N)
        mg = f"{np.mean(gaps):+.1f}%" if gaps else "n/a"
        wg = f"{max(gaps):+.1f}%" if gaps else "n/a"
        print(f"{name:<12}{f'{hits}/{N}':>7}{f'{lo:.0%}-{hi:.0%}':>12}{f'{feas}/{N}':>10}"
              f"{mg:>10}{wg:>11}")
        rows.append({"solver": name, "hits": hits, "feasible": feas, "n": N,
                     "hit_ci": [lo, hi], "mean_gap": (np.mean(gaps) if gaps else None),
                     "worst_gap": (max(gaps) if gaps else None)})

    alpha = gate.CVAR_ALPHA
    run_all("neal", lambda s: qg.run_neal(qubo, seed=s), range(N))
    run_all("GSA", lambda s: qg.run_gsa(qubo, seed=s, restarts=20), range(N))
    run_all("GA", lambda s: qg.run_ga(qubo, seed=s, restarts=20), range(N))
    run_all("QAOA", lambda s: qg.run_qaoa(qp, 2, 100, seed=s), range(N))
    run_all("QAOA-CVaR", lambda s: qg.run_qaoa(qp, 2, 100, seed=s, cvar=alpha), range(N))
    run_all("VQE", lambda s: qg.run_vqe(qp, meta["n_qubits"], 1, 200, seed=s), range(N))
    run_all("VQE-CVaR", lambda s: qg.run_vqe(qp, meta["n_qubits"], 1, 200, seed=s,
                                             cvar=alpha), range(N))
    print("\nqseed 11 (the diagram I published) vs this full spread tells the story.")
    results.save("audit_seeds", {"exact_fuel": dp, "seeds": N, "cvar_alpha": alpha,
                                 "rows": rows})


if __name__ == "__main__":      # importing this module must not start the audit
    main()
