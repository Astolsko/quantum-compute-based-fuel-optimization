"""
matheuristic.py -- the NEW OPTIMIZER (Phase 4, layer 2) + honest baselines.

WHAT KIND OF ANIMAL THIS IS
---------------------------
A "matheuristic": a metaheuristic OUTER search over the combinatorial fleet
decisions (which ship serves which missions, in what order), where EVERY
candidate is evaluated EXACTLY by the inner voyage oracle's tables. The outer
search therefore never wastes effort on badly-sailed voyages -- routing and
speed are always optimal inside a candidate plan; only the assignment and
sequencing are searched heuristically.

Design rules carried over from the Phase-2/3 benchmark findings:
  * FEASIBILITY-PRESERVING: solutions are (assignment, sequence) structures,
    never raw bits; time-window violations are rejected, so there is NO
    penalty weight to tune (the knob that wrecked BGSA/QAOA).
  * ADAPTIVE (ALNS-style, Ropke & Pisinger 2006): a portfolio of move
    operators whose selection probabilities adapt to their recent success.
  * HONEST EVALUATION: same iteration budget for every search baseline;
    the exhaustive enumerator provides ground truth on small instances.

Solvers provided:
  greedy            cheapest-insertion dispatcher (the "human planner" bar)
  sa_naive          single-operator simulated annealing, same budget
  alns_fixed        operator portfolio, UNIFORM selection (ablation)
  alns              operator portfolio + adaptive weights  <-- the optimizer
  exhaustive        exact enumeration (M <= 6; kept as the cross-check)
  exact_dp          exact subset DP (M <= 16) -- ground truth for scale studies
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

import fleet_config as fc
import scenario as scen_mod

INF = float("inf")


# --------------------------------------------------------------------------- #
# Exact plan evaluation on top of the oracle tables                            #
# --------------------------------------------------------------------------- #
class Evaluator:
    """Evaluates fleet plans exactly using serve tables; memoises per-ship
    sequence costs (ALNS moves touch 1-2 ships, the rest hit the cache)."""

    def __init__(self, scn: scen_mod.Scenario, tables):
        self.scn = scn
        self.tables = tables
        self._cache: dict[tuple, tuple] = {}
        self.bundle_cache: dict[tuple, tuple] = {}  # hybrid.best_bundle: (ship, set) -> best order
        self.n_evals = 0

    def ship_cost(self, ship: scen_mod.Ship, seq: tuple[int, ...]):
        """(fuel, finish_slot) for one ship serving `seq` in order.

        Pareto DP over finish slots: F[tf] = min fuel to have served the
        prefix and be free BY slot tf. Chaining is a min-plus product with
        serve[o, j, t0, tf]; because waiting at stations is free, an earlier
        finish is never worse, so the done-by (cum-min) representation is
        exact. This is where the fuel-vs-time trade-off inside a sequence is
        resolved optimally -- the search layer never has to think about time.
        """
        key = (ship.id, seq)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sfuel = self.tables[ship.ship_class]["serve_fuel"]
        T = sfuel.shape[2]
        loc = self.scn.home_loc(ship.id)
        # released at slot 0: F[tf] = serve[loc, j0, 0, tf]; then chain.
        F = None
        for j in seq:
            if F is None:
                F = sfuel[loc, j, 0, :].astype(float)          # [tf]
            else:
                # F2[tf] = min over t0 of F[t0] + serve[loc, j, t0, tf]
                F = (F[:, None] + sfuel[loc, j, :, :]).min(axis=0)
            if not np.isfinite(F).any():
                self._cache[key] = (INF, -1)
                return INF, -1
            loc = self.scn.end_loc(j)
        if F is None:                                          # idle ship
            self._cache[key] = (0.0, 0)
            return 0.0, 0
        fuel = float(F.min())
        finish = int(np.argmin(F))          # earliest slot achieving min fuel
        self._cache[key] = (fuel, finish)
        return fuel, finish

    def ship_schedule(self, ship: scen_mod.Ship, seq):
        """Recover the TRUE consistent timeline (mission, release, finish) that
        achieves the optimum total -- by backtracking the Pareto min-plus DP.

        Needed because per-prefix minima do NOT chain: a prefix's cheapest
        finish may be later than the finish the full sequence requires (an
        early mission must sometimes finish sooner, burning slightly more, so a
        later mission still makes its window). Use THIS for schedules/Gantts,
        never independent prefix evaluations.
        """
        sf = self.tables[ship.ship_class]["serve_fuel"]
        T = sf.shape[2]
        loc = self.scn.home_loc(ship.id)
        Fs, choices = [], []
        F = None
        for j in seq:
            if F is None:
                Fnew = sf[loc, j, 0, :].astype(float)
                choice = np.zeros(T, dtype=int)
            else:
                M = F[:, None] + sf[loc, j, :, :]
                choice = np.argmin(M, axis=0)
                Fnew = M[choice, np.arange(T)]
            if not np.isfinite(Fnew).any():
                return None
            Fs.append(Fnew)
            choices.append(choice)
            F = Fnew
            loc = self.scn.end_loc(j)
        if F is None:
            return []
        tf = int(np.argmin(Fs[-1]))
        chain = []
        for L in range(len(seq) - 1, -1, -1):
            t0 = int(choices[L][tf])
            chain.append((seq[L], t0, tf))            # (mission, release, finish)
            tf = t0
        return chain[::-1]

    def total(self, plan: dict[int, list[int]]) -> float:
        """Total fleet fuel; INF if any ship's sequence is infeasible or a
        mission is missing/duplicated."""
        self.n_evals += 1
        served = [j for seq in plan.values() for j in seq]
        if len(served) != len(self.scn.missions) or \
           len(set(served)) != len(self.scn.missions):
            return INF
        tot = 0.0
        for ship in self.scn.ships:
            f, _ = self.ship_cost(ship, tuple(plan[ship.id]))
            if f == INF:
                return INF
            tot += f
        return tot


# --------------------------------------------------------------------------- #
# Greedy cheapest-insertion (baseline: what a careful human planner does)      #
# --------------------------------------------------------------------------- #
def greedy(ev: Evaluator, order=None, rng=None, noise=0.0):
    """Cheapest-insertion. Deterministic by default; `order`/`noise` allow
    randomized restarts (insertion order shuffled, occasionally take the
    2nd-best slot) -- greedy is myopic and can dead-end on tight instances
    that ARE feasible."""
    scn = ev.scn
    plan = {s.id: [] for s in scn.ships}
    if order is None:
        order = sorted(range(len(scn.missions)),
                       key=lambda j: scn.missions[j].es)
    for j in order:
        options = []
        for s in scn.ships:
            base, _ = ev.ship_cost(s, tuple(plan[s.id]))
            for pos in range(len(plan[s.id]) + 1):
                cand = plan[s.id][:pos] + [j] + plan[s.id][pos:]
                f, _ = ev.ship_cost(s, tuple(cand))
                if f == INF:
                    continue
                delta = f - (0.0 if base == INF else base)
                options.append((delta, s.id, pos))
        if not options:
            return None                        # dead end for this order
        options.sort(key=lambda x: x[0])
        pick = 0
        if noise > 0 and rng is not None and len(options) > 1 \
                and rng.random() < noise:
            pick = 1
        _, sid, pos = options[pick]
        plan[sid] = plan[sid][:pos] + [j] + plan[sid][pos:]
    return plan


def construct(ev: Evaluator, seed=0, tries=200):
    """Robust construction: deterministic greedy, then randomized restarts.
    Returns a feasible plan or None (instance likely infeasible)."""
    plan = greedy(ev)
    if plan is not None:
        return plan
    rng = np.random.default_rng(seed)
    n = len(ev.scn.missions)
    for _ in range(tries):
        order = list(rng.permutation(n))
        plan = greedy(ev, order=order, rng=rng, noise=0.3)
        if plan is not None:
            return plan
    return None


# --------------------------------------------------------------------------- #
# Move operators (all feasibility-preserving: reject if INF)                   #
# --------------------------------------------------------------------------- #
def _copy(plan):
    return {k: list(v) for k, v in plan.items()}


def op_relocate_random(plan, ev, rng):
    p = _copy(plan)
    j = int(rng.integers(len(ev.scn.missions)))
    for sid, seq in p.items():
        if j in seq:
            seq.remove(j)
            break
    sid2 = int(rng.integers(len(ev.scn.ships)))
    pos = int(rng.integers(len(p[sid2]) + 1))
    p[sid2].insert(pos, j)
    return p


def op_relocate_worst(plan, ev, rng):
    """Remove the mission with the largest marginal fuel, reinsert greedily."""
    worst_j, worst_delta = None, -INF
    for s in ev.scn.ships:
        seq = plan[s.id]
        base, _ = ev.ship_cost(s, tuple(seq))
        if base == INF:
            continue
        for j in seq:
            rem = tuple(x for x in seq if x != j)
            f, _ = ev.ship_cost(s, rem)
            delta = base - (f if f != INF else base)
            if delta > worst_delta:
                worst_delta, worst_j = delta, j
    if worst_j is None:
        return op_relocate_random(plan, ev, rng)
    p = _copy(plan)
    for sid, seq in p.items():
        if worst_j in seq:
            seq.remove(worst_j)
            break
    best, arg = INF, None
    for s in ev.scn.ships:
        for pos in range(len(p[s.id]) + 1):
            cand = p[s.id][:pos] + [worst_j] + p[s.id][pos:]
            f, _ = ev.ship_cost(s, tuple(cand))
            if f == INF:
                continue
            base, _ = ev.ship_cost(s, tuple(p[s.id]))
            delta = f - (0.0 if base == INF else base)
            if delta < best:
                best, arg = delta, (s.id, pos)
    if arg is None:
        return op_relocate_random(plan, ev, rng)
    sid, pos = arg
    p[sid].insert(pos, worst_j)
    return p


def op_swap(plan, ev, rng):
    p = _copy(plan)
    owners = {j: sid for sid, seq in p.items() for j in seq}
    if len(owners) < 2:
        return p
    j1, j2 = rng.choice(len(ev.scn.missions), size=2, replace=False)
    s1, s2 = owners[int(j1)], owners[int(j2)]
    i1, i2 = p[s1].index(int(j1)), p[s2].index(int(j2))
    p[s1][i1], p[s2][i2] = int(j2), int(j1)
    return p


def op_reorder(plan, ev, rng):
    p = _copy(plan)
    ships_multi = [sid for sid, seq in p.items() if len(seq) >= 2]
    if not ships_multi:
        return op_relocate_random(plan, ev, rng)
    sid = int(rng.choice(ships_multi))
    seq = p[sid]
    i = int(rng.integers(len(seq)))
    j = int(rng.integers(len(seq)))
    m = seq.pop(i)
    seq.insert(j, m)
    return p


def op_destroy_repair(plan, ev, rng):
    """Remove 2-3 random missions, reinsert with NOISY greedy (sometimes the
    2nd-best slot) -- the noise lets repair discover coordinated multi-mission
    exchanges that pure greedy reinsertion can never reconstruct."""
    p = _copy(plan)
    n = len(ev.scn.missions)
    size = int(rng.integers(2, min(3, n) + 1))
    removed = list(rng.choice(n, size=size, replace=False))
    for sid in p:
        p[sid] = [j for j in p[sid] if j not in removed]
    rng.shuffle(removed)
    for j in removed:
        options = []                      # (delta, ship_id, pos)
        for s in ev.scn.ships:
            base, _ = ev.ship_cost(s, tuple(p[s.id]))
            for pos in range(len(p[s.id]) + 1):
                cand = p[s.id][:pos] + [int(j)] + p[s.id][pos:]
                f, _ = ev.ship_cost(s, tuple(cand))
                if f == INF:
                    continue
                delta = f - (0.0 if base == INF else base)
                options.append((delta, s.id, pos))
        if not options:
            return None
        options.sort(key=lambda x: x[0])
        # 70% best slot, 30% second-best (if any): regret-style diversification
        pick = 0 if (len(options) == 1 or rng.random() < 0.7) else 1
        _, sid, pos = options[pick]
        p[sid].insert(pos, int(j))
    return p


OPERATORS = [op_relocate_random, op_relocate_worst, op_swap,
             op_reorder, op_destroy_repair]
OP_NAMES = ["reloc_rand", "reloc_worst", "swap", "reorder", "destroy_repair"]


# --------------------------------------------------------------------------- #
# Search engines (identical budget + acceptance for a fair comparison)         #
# --------------------------------------------------------------------------- #
def _anneal(ev, start_plan, n_iters, rng, operators, adaptive):
    """Shared SA-acceptance engine; `operators` and `adaptive` differ.
    Re-heats (temperature restart from the incumbent BEST) after prolonged
    stagnation -- cheap insurance against freezing in a local basin."""
    cur = _copy(start_plan)
    cur_f = ev.total(cur)
    best, best_f = _copy(cur), cur_f
    w = np.ones(len(operators))
    t0 = max(1e-6, 0.05 * cur_f)
    alpha = (1e-3) ** (1.0 / max(1, n_iters))       # T decays to 0.1% of t0
    temp = t0
    stagnant, reheat_after = 0, max(500, n_iters // 5)

    for _ in range(n_iters):
        if adaptive:
            probs = w / w.sum()
            oi = int(rng.choice(len(operators), p=probs))
        else:
            oi = int(rng.integers(len(operators)))
        cand = operators[oi](cur, ev, rng)
        reward = 0.0
        stagnant += 1
        if cand is not None:
            f = ev.total(cand)
            if f < INF:
                d = f - cur_f
                if d < 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
                    cur, cur_f = cand, f
                    reward = 1.0
                    if f < best_f - 1e-9:
                        best, best_f = _copy(cand), f
                        reward = 3.0
                        stagnant = 0
        if adaptive:
            w[oi] = 0.9 * w[oi] + 0.1 * reward * 10.0
            w[oi] = max(w[oi], 0.05)                # never fully kill an op
        temp *= alpha
        if stagnant >= reheat_after:                # restart from best, warmer
            cur, cur_f = _copy(best), best_f
            temp = t0 * 0.5
            stagnant = 0

    return best, best_f


def sa_naive(ev, n_iters, seed=0):
    """Plain SA: single relocate operator, no adaptivity. Same budget."""
    rng = np.random.default_rng(seed)
    start = construct(ev, seed=seed)
    if start is None:
        return None, INF
    return _anneal(ev, start, n_iters, rng,
                   [op_relocate_random], adaptive=False)


def alns(ev, n_iters, seed=0, adaptive=True, start=None):
    """The new optimizer: operator portfolio (+ adaptive weights). `start` is a
    feasible plan to search from (warm start); default = robust construction."""
    rng = np.random.default_rng(seed)
    if start is None or ev.total(start) == INF:
        start = construct(ev, seed=seed)
    if start is None:
        return None, INF
    return _anneal(ev, start, n_iters, rng, OPERATORS, adaptive=adaptive)


# --------------------------------------------------------------------------- #
# Exhaustive ground truth (small instances)                                    #
# --------------------------------------------------------------------------- #
def exhaustive(ev, max_missions=6):
    scn = ev.scn
    M, K = len(scn.missions), len(scn.ships)
    if M > max_missions:
        return None, INF, 0
    best, best_f, n_seen = None, INF, 0
    for assign in itertools.product(range(K), repeat=M):
        buckets = {s.id: [j for j in range(M) if assign[j] == s.id]
                   for s in scn.ships}
        for perms in itertools.product(
                *[itertools.permutations(buckets[s.id]) for s in scn.ships]):
            plan = {s.id: list(perms[i]) for i, s in enumerate(scn.ships)}
            n_seen += 1
            f = ev.total(plan)
            if f < best_f:
                best, best_f = plan, f
    return best, best_f, n_seen


# --------------------------------------------------------------------------- #
# Exact ground truth by dynamic programming (mid-size instances)               #
# --------------------------------------------------------------------------- #
def exact_dp(ev, max_missions=16):
    """Exact optimum by subset DP -- the same answer as `exhaustive`, without
    listing every plan. Returns (plan, fuel); (None, INF) if the instance is
    infeasible or has more than `max_missions` missions.

    Two layers, both exact:
      1. Per ship, Held-Karp over mission subsets. State = (set served, last
         mission) and, instead of a single number, the whole "fuel to be free
         BY slot tf" vector -- the same min-plus chaining Evaluator.ship_cost
         does for one sequence, minimised over all orderings at once. That is
         valid because min-plus chaining is monotone: the elementwise best
         prefix can only give the elementwise best extension.
      2. Across ships, a set-partition DP: every mission goes to exactly one
         ship; best[subset] = cheapest way for the ships so far to cover it.

    Work is 2^M * M^2 per ship plus 3^M per ship, where enumeration (K^M
    assignments x orderings) is hopeless. Memory is one [2^M, M, T] table per
    ship. Measured on the workstation (2026-10-28): 0.4 s at 4 ships x 12
    missions (T = 36); 22 s and 1.6 GB peak at 5 x 16."""
    scn = ev.scn
    M, T = len(scn.missions), scn.waves.shape[0]
    if M > max_missions:
        return None, INF
    full = (1 << M) - 1
    ends = [scn.end_loc(j) for j in range(M)]

    tables, costs = [], []               # per ship: F[mask, last, tf], cost[mask]
    for ship in scn.ships:
        sf = ev.tables[ship.ship_class]["serve_fuel"].astype(float)
        F = np.full((full + 1, M, T), INF)
        for j in range(M):                                   # first mission from home
            F[1 << j, j] = sf[scn.home_loc(ship.id), j, 0]
        for mask in range(1, full + 1):
            free = [k for k in range(M) if not mask >> k & 1]
            if not free:
                continue
            for j in range(M):
                if mask >> j & 1 and np.isfinite(F[mask, j]).any():
                    # serve k next, released from j's end at any slot t0:
                    # ext[k, tf] = min over t0 of F[mask, j, t0] + sf[end_j, k, t0, tf]
                    ext = (F[mask, j][None, :, None] + sf[ends[j]]).min(axis=1)
                    for k in free:
                        cell = F[mask | 1 << k, k]
                        np.minimum(cell, ext[k], out=cell)
        cost = F.min(axis=(1, 2))                            # best order of each subset
        cost[0] = 0.0                                        # an idle ship burns nothing
        tables.append(F)
        costs.append(cost)

    # Set partition: best[mask] = cheapest cover of `mask` by the ships so far.
    # (plain lists: this loop is 3^M scalar steps, where numpy indexing is slow)
    best = costs[0].tolist()
    took = [list(range(full + 1))]       # took[k][mask] = the part ship k serves
    for cost in costs[1:]:
        cost = cost.tolist()
        new, pick = [INF] * (full + 1), [0] * (full + 1)
        for mask in range(full + 1):
            sub = mask
            while True:                  # every subset `sub` of `mask`
                c = best[mask ^ sub] + cost[sub]
                if c < new[mask]:
                    new[mask], pick[mask] = c, sub
                if sub == 0:
                    break
                sub = (sub - 1) & mask
        best = new
        took.append(pick)
    if best[full] == INF:
        return None, INF

    # Walk back: which missions each ship got, then the order that achieved it.
    plan, mask = {}, full
    for k in range(len(scn.ships) - 1, -1, -1):
        sub = took[k][mask]
        mask ^= sub
        F, order = tables[k], []
        sf = ev.tables[scn.ships[k].ship_class]["serve_fuel"]
        if sub:
            j = int(np.argmin(F[sub].min(axis=1)))           # last mission served
            tf = int(np.argmin(F[sub, j]))
            order.append(j)
            while sub != 1 << j:
                sub ^= 1 << j
                prev = [(F[sub, p] + sf[ends[p], j, :, tf]).min()
                        if sub >> p & 1 else INF for p in range(M)]
                p = int(np.argmin(prev))
                tf = int(np.argmin(F[sub, p] + sf[ends[p], j, :, tf]))
                j = p
                order.append(j)
        plan[scn.ships[k].id] = order[::-1]
    return plan, float(best[full])


def describe_plan(ev, plan):
    lines = []
    for s in ev.scn.ships:
        f, fin = ev.ship_cost(s, tuple(plan[s.id]))
        seq = " -> ".join(f"M{j}" for j in plan[s.id]) or "(idle)"
        fs = f"{f:8.2f} t" if f != INF else "     INF"
        lines.append(f"    {s.name:<8} {seq:<28} {fs}"
                     + (f"  done @ slot {fin}" if plan[s.id] else ""))
    return "\n".join(lines)
