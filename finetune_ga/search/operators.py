"""ga_operators.py — NSGA-II genetic operators: dominance, selection, crossover, mutation.

These functions are pure algorithmic logic with no I/O or training calls, making
them easy to unit-test in isolation.
"""
from __future__ import annotations

import math
import random
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from finetune_ga.core.genome import Genome, assign_genome_id, sample_genome
from finetune_ga.infra.math_utils import pareto_dominates_values
from finetune_ga.search.common import build_budget_level_map, budget_level


_INVALID_OBJECTIVE_VECTOR = (1e12, 1e12, 1e12)


def _objective_vector(g: Genome) -> Tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in getattr(g, 'objectives', ()))
    except (TypeError, ValueError):
        return _INVALID_OBJECTIVE_VECTOR
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        return _INVALID_OBJECTIVE_VECTOR
    return vals  # type: ignore[return-value]


def has_valid_objectives(g: Genome) -> bool:
    """Return True only for finite, correctly shaped objective vectors.

    Invalid, missing, NaN/Inf, or sentinel objective vectors are treated as
    unevaluated/failed candidates by resume and baseline selection paths.
    """
    vals = _objective_vector(g)
    return vals != _INVALID_OBJECTIVE_VECTOR and all(v < 1e8 for v in vals)



# ---------------------------------------------------------------------------
# Dominance helpers
# ---------------------------------------------------------------------------

def dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return pareto_dominates_values(a, b)



def dominates_budget_aware(budget_levels: Dict[str, int], ga: Genome, gb: Genome) -> bool:
    """Return True only when ga and gb are at the same budget level and ga dominates gb.

    Comparing genomes evaluated at different budget levels is meaningless because
    objectives are not comparable across halted and fully-evaluated candidates.
    """
    if budget_level(budget_levels, ga.last_budget) != budget_level(budget_levels, gb.last_budget):
        return False
    a_valid = has_valid_objectives(ga)
    b_valid = has_valid_objectives(gb)
    if a_valid and not b_valid:
        return True
    if not a_valid:
        return False
    return dominates(_objective_vector(ga), _objective_vector(gb))


# ---------------------------------------------------------------------------
# NSGA-II core: non-dominated sort and crowding distance
#
# fast_non_dominated_sort is O(n² × B) where n = population size and
# B = number of budget levels.  Acceptable for pop_size ≤ 50.
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(cfg: Dict[str, Any], pop: List[Genome]) -> List[List[int]]:
    budget_levels = build_budget_level_map(cfg)

    n = len(pop)
    dominated_by_count = [0] * n
    dominates_set = [set() for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates_budget_aware(budget_levels, pop[p], pop[q]):
                dominates_set[p].add(q)
            elif dominates_budget_aware(budget_levels, pop[q], pop[p]):
                dominated_by_count[p] += 1
        if dominated_by_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(pop: List[Genome], front: List[int]) -> Dict[int, float]:
    if not front:
        return {}
    dist = {i: 0.0 for i in front}
    for obj_i in range(3):
        fs = sorted(front, key=lambda idx: _objective_vector(pop[idx])[obj_i])
        dist[fs[0]] = float("inf")
        dist[fs[-1]] = float("inf")
        vmin = _objective_vector(pop[fs[0]])[obj_i]
        vmax = _objective_vector(pop[fs[-1]])[obj_i]
        if vmax - vmin < 1e-12:
            continue
        for k in range(1, len(fs) - 1):
            dist[fs[k]] += (_objective_vector(pop[fs[k + 1]])[obj_i] - _objective_vector(pop[fs[k - 1]])[obj_i]) / (vmax - vmin)
    return dist


def assign_rank_and_crowd(cfg: Dict[str, Any], pop: List[Genome]) -> None:
    fronts = fast_non_dominated_sort(cfg, pop)
    for r, front in enumerate(fronts):
        cd = crowding_distance(pop, front)
        for idx in front:
            pop[idx].rank = r
            pop[idx].crowd = cd.get(idx, 0.0)


def _selection_key(budget_levels: Dict[str, int], g: Genome):
    """Sort key for NSGA-II tournament and survivor selection.

    Priority (ascending tuple = better candidate):
      1. Budget level (descending): higher budget → evaluated on more data.
      2. NSGA-II rank (ascending): closer to Pareto front is better.
      3. Crowding distance (descending): prefer higher diversity.
    """
    return (
        -budget_level(budget_levels, getattr(g, 'last_budget', 'B0')),
        getattr(g, 'rank', 10 ** 9),
        -float(getattr(g, 'crowd', 0.0)),
    )


def nsga2_select(cfg: Dict[str, Any], pop: List[Genome], k: int) -> List[Genome]:
    budget_levels = build_budget_level_map(cfg)
    return sorted(pop, key=lambda g: _selection_key(budget_levels, g))[:k]

def tournament_select(budget_levels: Dict[str, int], pop: List[Genome], rng: random.Random) -> Genome:
    cand = sorted(rng.sample(pop, k=min(2, len(pop))), key=lambda g: _selection_key(budget_levels, g))
    return cand[0]

# ---------------------------------------------------------------------------
# Crossover and mutation
# ---------------------------------------------------------------------------

def crossover(cfg: Dict[str, Any], a: Genome, b: Genome, rng: random.Random) -> Genome:
    def pick(x, y):
        return x if rng.random() < 0.5 else y

    model_names = cfg["model_names"]
    child = Genome(
        base_lr1=pick(a.base_lr1, b.base_lr1),
        base_lr2=pick(a.base_lr2, b.base_lr2),
        n_last_layers=pick(a.n_last_layers, b.n_last_layers),
        dense_units=pick(a.dense_units, b.dense_units),
        dropout=pick(a.dropout, b.dropout),
        l2_weight=pick(a.l2_weight, b.l2_weight),
        batch_size=pick(a.batch_size, b.batch_size),
        aug_rotation=pick(a.aug_rotation, b.aug_rotation),
        aug_zoom=pick(a.aug_zoom, b.aug_zoom),
        aug_contrast=pick(a.aug_contrast, b.aug_contrast),
        lr1_mul={m.lower(): pick(a.lr1_mul[m.lower()], b.lr1_mul[m.lower()]) for m in model_names},
        lr2_mul={m.lower(): pick(a.lr2_mul[m.lower()], b.lr2_mul[m.lower()]) for m in model_names},
    )
    assign_genome_id(child, model_names)
    return child


def mutate_choice_excluding_current(rng, values, current):
    candidates = [float(v) for v in values if float(v) != float(current)]
    if not candidates:
        return float(current)
    return float(rng.choice(candidates))


def mutate(cfg: Dict[str, Any], g: Genome, rng: random.Random) -> Genome:
    """Return a mutated copy of *g* without modifying the original genome."""
    out = deepcopy(g)
    ss = cfg["search_space"]
    lo_mul, hi_mul = ss["lr_mul_bounds"]
    mr = float(cfg["mutation_rate"])

    def maybe(p):
        return rng.random() < p

    if maybe(mr):
        out.base_lr1 = mutate_choice_excluding_current(rng, ss["base_lr1"], out.base_lr1)
    if maybe(mr):
        out.base_lr2 = mutate_choice_excluding_current(rng, ss["base_lr2"], out.base_lr2)
    if maybe(mr):
        out.n_last_layers = int(rng.choice(ss["n_last_layers"]))
    if maybe(mr):
        out.dense_units = int(rng.choice(ss["dense_units"]))
    if maybe(mr):
        _d_lo = float(ss.get('dropout', [0.1, 0.7])[0])
        _d_hi = float(ss.get('dropout', [0.1, 0.7])[1])
        out.dropout = float(max(_d_lo, min(_d_hi, out.dropout + rng.uniform(-0.12, 0.12))))
    if maybe(mr):
        out.l2_weight = float(rng.choice(ss["l2_weight"]))
    if maybe(mr):
        out.batch_size = int(rng.choice(ss["batch_size"]))
    if maybe(mr):
        _lo, _hi = float(ss["aug_rotation"][0]), float(ss["aug_rotation"][1])
        out.aug_rotation = float(max(_lo, min(_hi, out.aug_rotation + rng.uniform(-0.08, 0.08))))
    if maybe(mr):
        _lo, _hi = float(ss["aug_zoom"][0]), float(ss["aug_zoom"][1])
        out.aug_zoom = float(max(_lo, min(_hi, out.aug_zoom + rng.uniform(-0.08, 0.08))))
    if maybe(mr):
        _lo, _hi = float(ss["aug_contrast"][0]), float(ss["aug_contrast"][1])
        out.aug_contrast = float(max(_lo, min(_hi, out.aug_contrast + rng.uniform(-0.08, 0.08))))
    for m in cfg["model_names"]:
        if maybe(mr * 0.6):
            out.lr1_mul[m.lower()] = float(max(lo_mul, min(hi_mul, out.lr1_mul[m.lower()] * math.exp(rng.uniform(-0.12, 0.12)))))
        if maybe(mr * 0.6):
            out.lr2_mul[m.lower()] = float(max(lo_mul, min(hi_mul, out.lr2_mul[m.lower()] * math.exp(rng.uniform(-0.12, 0.12)))))

    assign_genome_id(out, cfg["model_names"])
    return out


def make_offspring(
    cfg: Dict[str, Any],
    parents: List[Genome],
    n: int,
    rng: random.Random,
    allow_crossover: bool = True,
    allow_mutation: bool = True,
    deduplicate: bool = True,
) -> List[Genome]:
    budget_levels = build_budget_level_map(cfg)

    kids: List[Genome] = []
    seen = {p.genome_id for p in parents} if deduplicate else set()
    max_tries = max(50, n * 30)
    tries = 0

    while len(kids) < n and tries < max_tries:
        tries += 1
        p1 = tournament_select(budget_levels, parents, rng) if len(parents) > 1 else parents[0]
        p2 = tournament_select(budget_levels, parents, rng) if len(parents) > 1 else parents[0]
        c = crossover(cfg, p1, p2, rng) if (allow_crossover and rng.random() < float(cfg["crossover_rate"])) else deepcopy(p1)
        if allow_mutation:
            c = mutate(cfg, c, rng)
        if deduplicate and c.genome_id in seen:
            continue
        seen.add(c.genome_id)
        kids.append(c)

    fill_tries = 0
    max_fill_tries = max(100, n * 100)
    while len(kids) < n and fill_tries < max_fill_tries:
        fill_tries += 1
        c = sample_genome(cfg, rng)
        if deduplicate and c.genome_id in seen:
            continue
        seen.add(c.genome_id)
        kids.append(c)

    if len(kids) < n:
        warnings.warn(
            f"Unable to generate {n} unique offspring after {max_tries + max_fill_tries} attempts; "
            "allowing duplicates for the remainder because the search space appears saturated.",
            RuntimeWarning,
        )
        while len(kids) < n:
            p1 = tournament_select(budget_levels, parents, rng) if len(parents) > 1 else parents[0]
            p2 = tournament_select(budget_levels, parents, rng) if len(parents) > 1 else parents[0]
            c = crossover(cfg, p1, p2, rng) if (allow_crossover and rng.random() < float(cfg["crossover_rate"])) else deepcopy(p1)
            if allow_mutation:
                c = mutate(cfg, c, rng)
            kids.append(c)

    return kids


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------
def pareto_front(pop: List[Genome], cfg: Optional[Dict[str, Any]] = None) -> List[Genome]:
    """Return the non-dominated front of *pop*.

    When *cfg* is provided, uses budget-aware dominance (same semantics as
    NSGA-II selection).  When *cfg* is None, uses raw objective dominance
    (appropriate for fully-evaluated single-budget baselines).
    """
    budget_levels = build_budget_level_map(cfg) if cfg is not None else None

    front: List[Genome] = []
    for i, a in enumerate(pop):
        dominated = False
        for j, b in enumerate(pop):
            if i == j:
                continue
            b_dominates_a = (
                dominates_budget_aware(budget_levels, b, a) if budget_levels is not None
                else pareto_dominates_values(_objective_vector(b), _objective_vector(a))
            )
            if b_dominates_a:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda g: (_objective_vector(g)[0], _objective_vector(g)[1], _objective_vector(g)[2]))