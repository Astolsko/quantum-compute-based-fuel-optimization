"""
ga.py  --  Genetic Algorithm (a fourth optimizer family, added for comparison).

Annealing (SA / neal) improves ONE solution; the gravity swarms (GSA / QGSO)
move a population along attraction forces. A GA is the classic EVOLUTIONARY
alternative (Holland 1975; Goldberg 1989): a population of candidate solutions
BREEDS.

  * SELECTION  binary tournament -- of two random candidates the fitter becomes
               a parent, so good solutions reproduce more often
  * CROSSOVER  two parents are mixed into one child
  * MUTATION   small random changes keep diversity alive
  * ELITISM    the best few survive unchanged, so the best-so-far is never lost

TWO FLAVOURS (the same split and call signatures as gsa.py, so GA drops into
every comparison at the SAME budget of n_agents x iters evaluations)
------------
* Continuous GA -> BLX-alpha blend crossover + Gaussian mutation that shrinks
  over the run. For the speed and speed-profile problems.
* Binary GA     -> uniform crossover + bit-flip mutation. Solves a QUBO, via
  the GASampler adapter (`--backend ga`).

Deliberately the textbook algorithm: no problem-specific repair and no
one-hot-aware operators, so the benchmark measures the GA family itself rather
than how much problem knowledge was baked in.
"""

from __future__ import annotations

import numpy as np

from solvers import gsa                 # reuse _qubo_matrix


def _parents(X, fitness, rng):
    """Binary tournament selection -> two parent matrices, one row per child."""
    n = len(X)
    a, b = rng.integers(n, size=(2, 2 * n))
    winners = X[np.where(fitness[a] <= fitness[b], a, b)]
    return winners[:n], winners[n:]


# --------------------------------------------------------------------------- #
# Continuous GA  (for the speed / speed-profile problems)                       #
# --------------------------------------------------------------------------- #
def ga_minimize(objective, bounds, *, n_agents=30, iters=200, cx_prob=0.9,
                blend=0.5, mut_sigma=0.1, elite=2, seed=0, vectorized=False):
    """Minimise objective(x) over a box `bounds` (list of (lo,hi)) with a GA.
    If `vectorized`, objective takes the whole (N,dim) array and returns (N,)
    fitness in one call. Returns (best_x, best_fitness, history) -- the same
    contract as gsa.gsa_minimize."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)

    X = lo + rng.random((n_agents, dim)) * (hi - lo)
    best_x, best_f = None, np.inf
    history = {"best": [], "mean": []}                    # best-so-far + population mean

    for t in range(iters):
        fitness = (np.asarray(objective(X)) if vectorized
                   else np.array([objective(x) for x in X]))
        order = np.argsort(fitness)                       # best first
        if fitness[order[0]] < best_f:
            best_f, best_x = float(fitness[order[0]]), X[order[0]].copy()
        history["best"].append(best_f)
        history["mean"].append(float(np.mean(fitness)))

        p1, p2 = _parents(X, fitness, rng)
        # BLX-alpha: the child is drawn from the parents' interval, widened by
        # `blend` on both sides so the population can also spread outward.
        u = rng.random((n_agents, dim)) * (1.0 + 2.0 * blend) - blend
        crossed = rng.random((n_agents, 1)) < cx_prob
        child = np.where(crossed, p1 + u * (p2 - p1), p1)
        # Gaussian mutation that shrinks over the run (explore -> exploit).
        child += (rng.normal(0.0, mut_sigma * (1.0 - t / iters), (n_agents, dim))
                  * (hi - lo))
        child = np.clip(child, lo, hi)
        child[:elite] = X[order[:elite]]                  # elitism
        X = child

    return best_x, best_f, history


# --------------------------------------------------------------------------- #
# Binary GA  (for QUBOs)                                                        #
# --------------------------------------------------------------------------- #
def bga_qubo(qubo, *, n_agents=60, iters=150, cx_prob=0.9, mut_rate=None,
             elite=2, seed=0):
    """Minimise a QUBO with a binary GA. Returns (best_sample_dict, energy, history)."""
    names, Q = gsa._qubo_matrix(qubo)
    n = len(names)
    rng = np.random.default_rng(seed)
    if mut_rate is None:
        mut_rate = 1.0 / n                                # ~one flipped bit per child

    X = rng.integers(0, 2, size=(n_agents, n)).astype(float)
    best_x, best_e = None, np.inf
    history = []

    for _ in range(iters):
        fitness = np.einsum("ni,ij,nj->n", X, Q, X)       # batch energy x^T Q x
        order = np.argsort(fitness)
        if fitness[order[0]] < best_e:
            best_e, best_x = float(fitness[order[0]]), X[order[0]].copy()
        history.append(best_e)

        p1, p2 = _parents(X, fitness, rng)
        # Uniform crossover: each bit comes from either parent with equal odds.
        from_p2 = ((rng.random((n_agents, n)) < 0.5)
                   & (rng.random((n_agents, 1)) < cx_prob))
        child = np.where(from_p2, p2, p1)
        flip = rng.random((n_agents, n)) < mut_rate       # bit-flip mutation
        child = np.where(flip, 1.0 - child, child)
        child[:elite] = X[order[:elite]]                  # elitism
        X = child

    sample = {names[k]: int(best_x[k]) for k in range(n)}
    return sample, best_e, history


# --------------------------------------------------------------------------- #
# Sampler adapter: makes the binary GA look like a dwave/neal sampler           #
# --------------------------------------------------------------------------- #
class GASampler:
    """dimod-style sampler so the binary GA drops into the QUBO comparison
    exactly like gsa.GSASampler (independent restarts, lowest energy first)."""

    def __init__(self, n_agents=60, iters=150):
        self.n_agents = n_agents
        self.iters = iters

    def sample_qubo(self, qubo, num_reads=8, seed=0, **_):
        import dimod
        names, _ = gsa._qubo_matrix(qubo)
        samples, energies = [], []
        for r in range(max(1, num_reads)):
            s, e, _ = bga_qubo(qubo, n_agents=self.n_agents,
                               iters=self.iters, seed=seed + r)
            samples.append([s[nm] for nm in names])
            energies.append(e)
        return dimod.SampleSet.from_samples(
            (np.array(samples), names), vartype="BINARY", energy=energies)
