"""
hybrid.py -- the QUANTUM-CLASSICAL HYBRID optimizer (Phase 5).

Same two-layer architecture as the ALNS matheuristic, but the OUTER combinatorial
decision is made by a QUBO (quantum-portable) instead of a classical search:

    OUTER  (this file)  : a QUBO decides WHICH SHIP GETS WHICH MISSIONS.
                          Solved with neal now; the identical QUBO runs on a real
                          quantum annealer / QAOA by swapping the backend.
    INNER  (oracle.py + matheuristic.Evaluator, unchanged) : for any assignment,
                          the EXACT cheapest ordering + route + speed + timing.

Why a QUBO can only do the ASSIGNMENT (not the whole plan): a QUBO objective is
pairwise (quadratic), but a ship's true bundle cost is STATEFUL -- it depends on
the order, weather, and timing of the whole chain. So the QUBO uses solo + exact
PAIRWISE costs as an approximation; the quantum sampler returns many candidate
assignments; and the classical oracle then scores each EXACTLY (best ordering per
ship) and keeps the real winner. Quantum proposes, classical verifies.

This is the honest way to employ a quantum technique here: it plugs in at the one
layer that is naturally a QUBO, and its result is judged by the same exact
evaluator the ALNS version uses -- a fair, apples-to-apples comparison.

V1 AND V2
---------
solve_hybrid (v1, unchanged) was optimal on 10/10 instances at 3 ships x 6
missions, and found NO feasible plan on 2 of 3 instances at 4 x 12 (measured
2026-10-28). Diagnosis: the pairwise QUBO cannot see that 3+ missions are
jointly time-infeasible on one ship, so it rates overloaded bundles (5-7
missions on a ship) BELOW the true optimum, and the repair could only move one
mission at a time. More reads cannot fix a mis-specified energy.

solve_hybrid_cuts (v2) keeps the same QUBO and adds two things:
  * LAZY CUTS. Every proposal whose ship bundle is infeasible yields a minimal
    infeasible subset (deletion filter: feasibility is downward-closed because
    waiting at stations is free); that subset is forbidden with a higher-order
    penalty, reduced to quadratic with auxiliary variables (Rosenberg 1975), and
    the QUBO is resampled. Still a QUBO every round -- quantum-portable.
  * A REPAIR that strips each overloaded ship to what fits, then reinserts the
    stripped missions where they add least fuel.

Every assignment-proposing method -- v2, the fleet GA (genetic.py) and the
random ablation -- is scored by the same AssignmentScorer, which counts DISTINCT
exact scorings: they are compared at an equal number of exact evaluations.
"""

from __future__ import annotations

import itertools

import neal
import numpy as np

import matheuristic as mh
import scenario as scen_mod

INF = float("inf")


# --------------------------------------------------------------------------- #
# Exact per-ship bundle cost (best ordering) -- the inner oracle's evaluation  #
# --------------------------------------------------------------------------- #
def best_bundle(ev: mh.Evaluator, ship: scen_mod.Ship, missions):
    """(fuel, best_ordering) for a ship serving an UNORDERED set of missions,
    optimally ordered. Small bundles are enumerated exactly; large ones fall
    back to greedy insertion (rare -- a ship rarely holds >7 missions)."""
    missions = tuple(missions)
    if not missions:
        return 0.0, ()
    key = (ship.id, frozenset(missions))
    hit = ev.bundle_cache.get(key)
    if hit is None:
        hit = ev.bundle_cache[key] = _best_bundle(ev, ship, missions)
    return hit


def _best_bundle(ev, ship, missions):
    if len(missions) <= 7:
        best, arg = INF, None
        for perm in itertools.permutations(missions):
            f, _ = ev.ship_cost(ship, perm)
            if f < best:
                best, arg = f, perm
        return best, (arg or ())
    # greedy-insertion fallback for big bundles
    seq = []
    for j in missions:
        best, pos = INF, 0
        for p in range(len(seq) + 1):
            f, _ = ev.ship_cost(ship, tuple(seq[:p] + [j] + seq[p:]))
            if f < best:
                best, pos = f, p
        seq.insert(pos, j)
    f, _ = ev.ship_cost(ship, tuple(seq))
    return f, tuple(seq)


# --------------------------------------------------------------------------- #
# Build the assignment QUBO                                                    #
# --------------------------------------------------------------------------- #
def build_assignment_qubo(ev: mh.Evaluator, penalty=None):
    """Binary x[s,j] = ship s serves mission j.

    objective   = sum_sj  solo[s,j] x[s,j]                       (solo costs)
                + sum_s sum_{j<k} extra[s,j,k] x[s,j] x[s,k]      (exact pair extra)
    constraint  = P * sum_j ( sum_s x[s,j] - 1 )^2               (each mission once)
    """
    scn = ev.scn
    K, M = len(scn.ships), len(scn.missions)

    solo = np.full((K, M), INF)
    for si, s in enumerate(scn.ships):
        for j in range(M):
            solo[si, j] = ev.ship_cost(s, (j,))[0]

    finite = solo[np.isfinite(solo)]
    if penalty is None:
        penalty = 5.0 * (float(finite.max()) * M if finite.size else 100.0)

    qubo: dict[tuple[str, str], float] = {}

    def add(a, b, c):
        key = (a, b) if a <= b else (b, a)
        qubo[key] = qubo.get(key, 0.0) + c

    # solo (linear) -- infeasible ship/mission gets a big positive cost (avoided)
    for si in range(K):
        for j in range(M):
            v = f"x_{si}_{j}"
            add(v, v, float(solo[si, j]) if np.isfinite(solo[si, j])
                else 3.0 * penalty)

    # exact pairwise bundle interaction on the same ship. An INFEASIBLE pair
    # (the two missions cannot both be served by this ship) gets a big penalty
    # so the QUBO avoids co-assigning them. (Higher-order overloads -- a bundle
    # of 3+ that is jointly infeasible while every pair is fine -- are invisible
    # to a pairwise QUBO; those are handled by classical repair after sampling.)
    for si, s in enumerate(scn.ships):
        for j in range(M):
            for k in range(j + 1, M):
                if np.isfinite(solo[si, j]) and np.isfinite(solo[si, k]):
                    fb, _ = best_bundle(ev, s, (j, k))
                    extra = (fb - solo[si, j] - solo[si, k]
                             if np.isfinite(fb) else penalty)
                    add(f"x_{si}_{j}", f"x_{si}_{k}", float(extra))

    # one-hot per mission: P*(sum_s x - 1)^2 -> linear -P each, quadratic +2P pairs
    for j in range(M):
        vs = [f"x_{si}_{j}" for si in range(K)]
        for v in vs:
            add(v, v, -penalty)
        for a in range(K):
            for b in range(a + 1, K):
                add(vs[a], vs[b], 2.0 * penalty)

    return qubo, solo, penalty


# --------------------------------------------------------------------------- #
# Decode + exact score                                                         #
# --------------------------------------------------------------------------- #
def decode_assignment(sample, ev: mh.Evaluator):
    """Turn a QUBO sample into a valid COMPLETE assignment {ship_id: [missions]}.
    Missions left unassigned or double-assigned are repaired to the cheapest
    feasible solo ship (classical repair of the quantum proposal)."""
    scn = ev.scn
    K, M = len(scn.ships), len(scn.missions)
    assign = {s.id: [] for s in scn.ships}
    for j in range(M):
        chosen = [si for si in range(K) if sample.get(f"x_{si}_{j}", 0) == 1]
        if len(chosen) == 1:
            si = chosen[0]
        else:
            cands = sorted((ev.ship_cost(scn.ships[si], (j,))[0], si)
                           for si in range(K))
            si = cands[0][1]
        assign[scn.ships[si].id].append(j)
    return assign


def _feasible(ev, assign):
    return all(best_bundle(ev, s, assign[s.id])[0] < INF for s in ev.scn.ships)


def exact_score(ev: mh.Evaluator, assign):
    """Exact total fuel + ordered plan. If a ship's bundle is time-INFEASIBLE
    (the QUBO overloaded it), REPAIR minimally: repeatedly move one mission off
    an infeasible ship to the cheapest feasible other ship, until feasible.
    This is the classical half of 'quantum proposes, classical repairs+scores'."""
    assign = {s.id: list(assign[s.id]) for s in ev.scn.ships}
    ships = ev.scn.ships
    for _ in range(4 * len(ev.scn.missions) + 1):
        bad = next((s for s in ships
                    if best_bundle(ev, s, assign[s.id])[0] == INF), None)
        if bad is None:
            break
        # cheapest feasible relocation of one mission off the overloaded ship
        best, mv = INF, None
        for j in list(assign[bad.id]):
            rest = [x for x in assign[bad.id] if x != j]
            fr, _ = best_bundle(ev, bad, rest)
            for s2 in ships:
                if s2.id == bad.id:
                    continue
                f2, _ = best_bundle(ev, s2, assign[s2.id] + [j])
                if f2 < INF and fr < INF and f2 + fr < best:
                    best, mv = f2 + fr, (j, s2.id, rest)
        if mv is None:
            return INF, None
        j, s2id, rest = mv
        assign[bad.id] = rest
        assign[s2id].append(j)
    else:
        return INF, None

    plan, tot = {}, 0.0
    for s in ships:
        f, order = best_bundle(ev, s, assign[s.id])
        if f == INF:
            return INF, None
        plan[s.id] = list(order)
        tot += f
    return tot, plan


# --------------------------------------------------------------------------- #
# The hybrid solver                                                            #
# --------------------------------------------------------------------------- #
def solve_hybrid(ev: mh.Evaluator, num_reads=200, topk=25, seed=0,
                 sampler=None):
    """QUBO (quantum-portable) proposes assignments; oracle scores them exactly.

    `sampler` defaults to neal; pass any dimod-style sampler (e.g. a real
    D-Wave/QAOA sampler) to run the SAME QUBO on quantum hardware.
    Returns (best_plan, best_fuel, info).
    """
    qubo, solo, P = build_assignment_qubo(ev)
    if sampler is None:
        sampler = neal.SimulatedAnnealingSampler()
    ss = sampler.sample_qubo(qubo, num_reads=num_reads, seed=seed)

    best_plan, best_f, scored, seen = None, INF, 0, set()
    for rec in ss.data(["sample", "energy"], sorted_by="energy"):
        assign = decode_assignment(rec.sample, ev)
        key = tuple(tuple(sorted(assign[s.id])) for s in ev.scn.ships)
        if key in seen:
            continue
        seen.add(key)
        f, plan = exact_score(ev, assign)
        if f < best_f:
            best_f, best_plan = f, plan
        scored += 1
        if scored >= topk:
            break
    info = {"n_qubo_vars": len(ev.scn.ships) * len(ev.scn.missions),
            "penalty": P, "assignments_scored": scored}
    return best_plan, best_f, info


# --------------------------------------------------------------------------- #
# v2: a fair scorer, a stronger repair, and the QUBO with lazy cuts            #
# --------------------------------------------------------------------------- #
def bundle_fuel(ev, ship, missions) -> float:
    return best_bundle(ev, ship, missions)[0]


def repair(ev: mh.Evaluator, assign):
    """Make an assignment time-feasible, changing as little as possible.

    1. STRIP: rebuild every infeasible ship bundle by adding its missions in
       start-window order and keeping only those that still fit.
    2. REINSERT each stripped mission (earliest window first) on the ship where
       it adds least fuel, exact bundle costs throughout.
    Returns a feasible {ship_id: [missions]} or None if a mission fits nowhere."""
    ships, missions = ev.scn.ships, ev.scn.missions
    assign = {s.id: list(assign[s.id]) for s in ships}
    by_window = lambda j: (missions[j].es, j)                       # noqa: E731
    pool = []
    for s in ships:
        if bundle_fuel(ev, s, assign[s.id]) < INF:
            continue
        keep = []
        for j in sorted(assign[s.id], key=by_window):
            (keep if bundle_fuel(ev, s, keep + [j]) < INF else pool).append(j)
        assign[s.id] = keep
    for j in sorted(pool, key=by_window):
        best, arg = INF, None
        for s in ships:
            f = bundle_fuel(ev, s, assign[s.id] + [j])
            if f < INF and f - bundle_fuel(ev, s, assign[s.id]) < best:
                best, arg = f - bundle_fuel(ev, s, assign[s.id]), s.id
        if arg is None:
            return None
        assign[arg].append(j)
    return assign


class AssignmentScorer:
    """Exact scoring of proposed assignments, with a budget meter.

    score() repairs a proposal and prices it with the exact best order per ship.
    Only DISTINCT proposals are charged (a repeat is free -- and useless), and
    `trace` keeps the best fuel after every charged scoring, so methods can be
    compared at any equal budget, not just the final one."""

    def __init__(self, ev: mh.Evaluator):
        self.ev = ev
        self.cache: dict[tuple, tuple] = {}
        self.trace: list[float] = []
        self.best_fuel, self.best_plan = INF, None

    @property
    def n_scored(self) -> int:
        return len(self.trace)

    def score(self, assign):
        """(fuel, ordered plan or None) for a proposal {ship_id: [missions]}."""
        key = tuple(tuple(sorted(assign[s.id])) for s in self.ev.scn.ships)
        if key in self.cache:
            return self.cache[key]
        fixed = repair(self.ev, assign)
        if fixed is None:
            out = (INF, None)
        else:
            plan, tot = {}, 0.0
            for s in self.ev.scn.ships:
                f, order = best_bundle(self.ev, s, fixed[s.id])
                plan[s.id], tot = list(order), tot + f
            out = (tot, plan)
        self.cache[key] = out
        if out[0] < self.best_fuel:
            self.best_fuel, self.best_plan = out
        self.trace.append(self.best_fuel)
        return out


def solve_random(ev: mh.Evaluator, budget=200, seed=0, scorer=None):
    """ABLATION: identical exact scoring as the hybrid and the GA, but the
    assignments are drawn UNIFORMLY AT RANDOM. If this matches them, the
    proposal mechanism adds nothing at this scale. Returns (plan, fuel, scorer)."""
    rng = np.random.default_rng(seed)
    scorer = scorer or AssignmentScorer(ev)
    scn = ev.scn
    K, M = len(scn.ships), len(scn.missions)
    tries = 0
    while scorer.n_scored < budget and tries < budget * 30:
        tries += 1
        assign = {s.id: [] for s in scn.ships}
        for j in range(M):
            assign[scn.ships[int(rng.integers(K))].id].append(j)
        scorer.score(assign)
    return scorer.best_plan, scorer.best_fuel, scorer


def minimal_infeasible(ev, ship, bundle):
    """A minimal subset of `bundle` the ship cannot serve (deletion filter).
    Valid because feasibility is downward-closed: dropping a mission never
    makes a ship later, since waiting at stations is free."""
    core = list(bundle)
    for j in list(core):
        rest = [x for x in core if x != j]
        if bundle_fuel(ev, ship, rest) == INF:
            core = rest
    return core


def _add_product_penalty(qubo, names, weight, aux):
    """Add weight * x_1 * x_2 * ... * x_k to a QUBO, k >= 2, using auxiliary
    variables y = x_a AND x_b (Rosenberg penalty 2*weight*(x_a x_b - 2 x_a y
    - 2 x_b y + 3 y): zero iff y = x_a x_b, so cheating never pays). `aux`
    memoises the AND of each prefix so cuts that share missions share qubits."""
    def add(a, b, c):
        key = (a, b) if a <= b else (b, a)
        qubo[key] = qubo.get(key, 0.0) + c

    names = sorted(names)
    acc = names[0]
    for nxt in names[1:-1]:
        pair = (acc, nxt)
        if pair not in aux:
            y = aux[pair] = f"aux_{len(aux)}"
            r = 2.0 * weight
            add(acc, nxt, r)
            add(acc, y, -2.0 * r)
            add(nxt, y, -2.0 * r)
            add(y, y, 3.0 * r)
        acc = aux[pair]
    add(acc, names[-1], weight)


def solve_hybrid_cuts(ev: mh.Evaluator, num_reads=1000, budget=200, rounds=8,
                      seed=0, sampler=None, scorer=None):
    """v2 hybrid: QUBO proposals + lazy cuts + repair, scored exactly.

    Each round samples the QUBO, scores up to budget/rounds new distinct
    proposals, and adds a cut for every minimal infeasible bundle it saw.
    Returns (best_plan, best_fuel, info) like solve_hybrid."""
    qubo, solo, P = build_assignment_qubo(ev)
    qubo = dict(qubo)
    if sampler is None:
        sampler = neal.SimulatedAnnealingSampler()
    scorer = scorer or AssignmentScorer(ev)
    scn = ev.scn
    per_round = -(-budget // rounds)
    cuts, aux = set(), {}
    used = 0
    for rnd in range(rounds):
        if scorer.n_scored >= budget:
            break
        used += 1
        ss = sampler.sample_qubo(qubo, num_reads=num_reads, seed=seed + rnd)
        new_cuts, scored_here = 0, 0
        for rec in ss.data(["sample"], sorted_by="energy"):
            if scored_here >= per_round or scorer.n_scored >= budget:
                break
            assign = decode_assignment(rec.sample, ev)
            before = scorer.n_scored
            scorer.score(assign)
            scored_here += scorer.n_scored - before
            for si, s in enumerate(scn.ships):
                if bundle_fuel(ev, s, assign[s.id]) < INF:
                    continue
                core = tuple(sorted(minimal_infeasible(ev, s, assign[s.id])))
                if len(core) >= 2 and (si, core) not in cuts:
                    cuts.add((si, core))
                    _add_product_penalty(qubo, [f"x_{si}_{j}" for j in core], P, aux)
                    new_cuts += 1
        if scored_here == 0 and new_cuts == 0:
            break                          # nothing new to learn or try
    info = {"n_qubo_vars": len(scn.ships) * len(scn.missions) + len(aux),
            "penalty": P, "assignments_scored": scorer.n_scored,
            "rounds": used, "cuts": len(cuts), "aux_vars": len(aux)}
    return scorer.best_plan, scorer.best_fuel, info


if __name__ == "__main__":
    import oracle
    scn = scen_mod.build_scenario(0)
    tabs = oracle.get_tables(scn, verbose=False)
    ev = mh.Evaluator(scn, tabs)
    plan, f, info = solve_hybrid(ev, num_reads=200, seed=1)
    pe, fe, _ = mh.exhaustive(ev)
    print(f"hybrid (QUBO+oracle): {f:.2f} t   "
          f"({info['n_qubo_vars']} QUBO vars, "
          f"{info['assignments_scored']} assignments scored exactly)")
    print(f"exhaustive optimum  : {fe:.2f} t   gap {(f-fe)/fe*100:+.2f}%")
    if plan is not None:
        print(mh.describe_plan(ev, plan))
    else:
        print("  (no feasible assignment found)")
