"""
genetic.py -- a GENETIC ALGORITHM for the fleet assignment (Phase 5 comparison).

The quantum-classical hybrid proposes "which ship serves which missions" from a
QUBO; the random ablation proposes them uniformly. A GA is the classical,
evolutionary way to make the same proposals -- so the three are compared on
one footing: identical repair and exact scoring (hybrid.AssignmentScorer, best
order per ship from the oracle) and an equal number of DISTINCT exact scorings.

  chromosome   one gene per mission: the index of the ship that serves it
  initial      uniformly random -- generation 0 IS the random ablation, so
               anything better than solve_random is the GA's own doing
  selection    binary tournament
  crossover    uniform (each mission's ship from either parent)
  mutation     reassign each mission to a random ship with probability 1/M;
               doubled for a generation that produced nothing new (a converged
               population only re-proposes plans that were already scored)
  elitism      the best 2 survive unchanged
  Lamarckian   the REPAIRED assignment is written back into the chromosome, so
               the population learns feasible structure instead of re-making
               the same infeasible bundle

It is the GA family on its natural representation for an assignment problem
(Chu & Beasley 1997 used the same encoding for generalised assignment) -- the
binary QUBO GA in solvers/ga.py is the other half of the GA story.
"""

from __future__ import annotations

import numpy as np

import hybrid as hyb
import matheuristic as mh

INF = float("inf")


def solve_ga(ev: mh.Evaluator, budget=200, pop=20, seed=0, elite=2, scorer=None):
    """GA over assignments, stopped after `budget` distinct exact scorings.
    Returns (best_plan, best_fuel, info)."""
    rng = np.random.default_rng(seed)
    scorer = scorer or hyb.AssignmentScorer(ev)
    ship_ids = [s.id for s in ev.scn.ships]
    K, M = len(ship_ids), len(ev.scn.missions)
    base_mut = 1.0 / M

    def fitness(chrom):
        fuel, plan = scorer.score({sid: [j for j in range(M) if chrom[j] == k]
                                   for k, sid in enumerate(ship_ids)})
        if plan is not None:                       # Lamarckian write-back
            for k, sid in enumerate(ship_ids):
                chrom[plan[sid]] = k
        return fuel

    def tournament(fit):
        a, b = rng.integers(len(fit), size=2)
        return a if fit[a] <= fit[b] else b

    X = rng.integers(0, K, size=(pop, M))
    fit = np.array([fitness(x) for x in X])
    mut, gens = base_mut, 0
    while scorer.n_scored < budget and gens < 50 * max(1, budget // pop):
        gens += 1
        before = scorer.n_scored
        order = np.argsort(fit)
        kids, kid_fit = [X[i].copy() for i in order[:elite]], list(fit[order[:elite]])
        while len(kids) < pop and scorer.n_scored < budget:
            p1, p2 = X[tournament(fit)], X[tournament(fit)]
            child = np.where(rng.random(M) < 0.5, p1, p2)
            flip = rng.random(M) < mut
            child[flip] = rng.integers(0, K, size=int(flip.sum()))
            kids.append(child)
            kid_fit.append(fitness(child))
        X, fit = np.array(kids), np.array(kid_fit)
        mut = base_mut if scorer.n_scored > before else min(0.5, 2 * mut)
    info = {"generations": gens, "assignments_scored": scorer.n_scored, "pop": pop}
    return scorer.best_plan, scorer.best_fuel, info
