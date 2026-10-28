"""
gate.py  --  gate-model quantum plumbing shared by experiments/quantum_gate.py,
             experiments/benchmark.py and fleet/quantum_qaoa.py.

THE SIMULATOR
-------------
Until 2026-10 the QAOA/VQE runs used Qiskit's reference StatevectorSampler: pure
NumPy, one core, ~16-18 qubits in practice. They now use Aer's SamplerV2 -- the
same statevector maths in multithreaded C++, and on CUDA when the GPU build is
installed (requirements-gpu.txt). Measured on the workstation (RTX A6000, 32
Xeon cores), QAOA reps=3, 20 COBYLA iterations, presolved route QUBOs:

    qubits   CPU 1 thread   CPU 32 threads   GPU
      14         1.8 s          1.8 s        1.2 s
      16         2.7 s          1.6 s        1.4 s
      22       121   s         12.1 s        2.6 s
      26           -              -         19   s

A sequential run wants the GPU from ~16 qubits; a parallel benchmark is better
off with many 1-thread CPU workers until the state grows past that. Hence
`auto` = GPU from GPU_MIN_QUBITS upward.

    FUEL_QDEVICE=auto|CPU|GPU      where statevectors are simulated (default auto)
    FUEL_AER_THREADS=N             CPU threads per simulation (0 = all; the
                                   benchmark sets 1 when it runs solves in parallel)

CVaR
----
Plain QAOA/VQE minimise the EXPECTED energy of the measured distribution, so a
circuit that puts weight on many mediocre bitstrings scores well. CVaR_alpha
(Barkoutsos et al. 2020, Quantum 4, 256) minimises the mean of only the best
alpha fraction of shots -- the optimiser is rewarded for making the GOOD tail
likely, which is what a sampler that we decode by "best bitstring" needs. It is
one argument (`aggregation`) in qiskit-algorithms; CVAR_ALPHA is the default.

DECODING (bug fixed 2026-10-28)
-------------------------------
Until then the runs went through qiskit-optimization's MinimumEigenOptimizer.
With qiskit-algorithms 0.4 the eigenstate it receives is a dict of measured
PROBABILITIES, but it treats dict values as AMPLITUDES: it squares them and
drops everything under 1e-6. A bitstring measured once in 1,024 shots has
(1/1024)^2 ~ 1e-6.0 < 1e-6, so only bitstrings measured at least TWICE were
ever candidates -- and at ~20 qubits, where nearly every shot is unique, nothing
survived and solve() crashed. sample_qp() decodes every measured bitstring.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

GPU_MIN_QUBITS = 18         # from here the GPU wins (see the table above)
CVAR_ALPHA = 0.25           # fraction of best shots CVaR averages over
SHOTS = 1024                # as Qiskit's StatevectorSampler default, used before 2026-10


def qubo_to_qp(qubo, name="fuel_qubo"):
    """Convert a pyqubo QUBO dict {(vi,vj): coef} into a Qiskit QuadraticProgram.
    Diagonal (vi==vj) -> linear (x^2 = x for binaries); off-diagonal -> quadratic."""
    varnames = sorted({v for k in qubo for v in k})
    qp = QuadraticProgram(name)
    for v in varnames:
        qp.binary_var(v)
    linear, quadratic = {}, {}
    for (i, j), c in qubo.items():
        if i == j:
            linear[i] = linear.get(i, 0.0) + c
        else:
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + c
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    return "GPU" in AerSimulator().available_devices()


def device_for(n_qubits: int) -> str:
    """'CPU' or 'GPU' for an n-qubit statevector, per FUEL_QDEVICE."""
    want = os.environ.get("FUEL_QDEVICE", "auto").upper()
    if want == "CPU" or not gpu_available():
        return "CPU"
    if want == "GPU":
        return "GPU"
    return "GPU" if n_qubits >= GPU_MIN_QUBITS else "CPU"


@lru_cache(maxsize=1)
def pass_manager():
    """Transpiler targeting the Aer simulator (shared: building it is not free)."""
    return generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())


def make_sampler(n_qubits: int, seed=None):
    """Aer SamplerV2 on the right device. Returns (sampler, device)."""
    device = device_for(n_qubits)
    backend_options = {"method": "statevector", "device": device,
                       "max_parallel_threads": int(os.environ.get("FUEL_AER_THREADS", "0"))}
    sampler = AerSampler(default_shots=SHOTS, seed=seed,
                         options={"backend_options": backend_options})
    return sampler, device


def sample_qp(solver, qp):
    """Run a sampling eigensolver (QAOA / SamplingVQE) on a binary, unconstrained
    QuadraticProgram. Returns (samples, eigen_result); samples is every measured
    bitstring as (x, probability, objective), lowest objective first."""
    op, _ = QuadraticProgramToQubo().convert(qp).to_ising()
    res = solver.compute_minimum_eigenvalue(op)
    state = res.eigenstate
    if hasattr(state, "binary_probabilities"):           # a QuasiDistribution
        state = state.binary_probabilities(num_bits=qp.get_num_vars())
    samples = []
    for bits, p in state.items():
        x = np.fromiter(bits[::-1], dtype=int)            # qubit 0 = variable 0
        samples.append((x, float(p), float(qp.objective.evaluate(x))))
    samples.sort(key=lambda s: s[2])
    return samples, res


def to_vardict(qp, x):
    return {v.name: int(x[i]) for i, v in enumerate(qp.variables)}
