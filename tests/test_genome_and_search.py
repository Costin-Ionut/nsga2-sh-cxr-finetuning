import copy
import random
import pytest

from finetune_ga.core.genome import (
    Genome,
    assign_genome_id,
    genome_from_dict,
    genome_to_dict,
    hyper_for_backbone,
    sample_genome,
)
from finetune_ga.search.common import build_budget_level_map
from finetune_ga.search.operators import (
    dominates,
    dominates_budget_aware,
    fast_non_dominated_sort,
    crowding_distance,
    nsga2_select,
    mutate,
    make_offspring,
    pareto_front,
)


@pytest.fixture
def cfg():
    return {
        "model_names": ["mobilenet", "resnet"],
        "mutation_rate": 1.0,
        "crossover_rate": 1.0,
        "budgets": [{"name": "B0"}, {"name": "B1"}, {"name": "B2"}],
        "search_space": {
            "base_lr1": [1e-5, 1e-3],
            "base_lr2": [1e-6, 1e-4],
            "n_last_layers": [0, 5, 10],
            "dense_units": [64, 128],
            "dropout": [0.2, 0.6],
            "l2_weight": [0.0, 1e-5],
            "batch_size": [8, 16],
            "aug_rotation": [0.0, 0.2],
            "aug_zoom": [0.0, 0.2],
            "aug_contrast": [0.0, 0.2],
            "lr_mul_bounds": [0.8, 1.4],
        },
    }


@pytest.fixture
def genome_a():
    g = Genome(
        base_lr1=1e-4,
        base_lr2=1e-5,
        n_last_layers=10,
        dense_units=128,
        dropout=0.3,
        l2_weight=1e-5,
        batch_size=16,
        aug_rotation=0.1,
        aug_zoom=0.05,
        aug_contrast=0.02,
        lr1_mul={"mobilenet": 1.0, "resnet": 1.1},
        lr2_mul={"mobilenet": 0.9, "resnet": 1.2},
        last_budget="B1",
        objectives=(0.1, 10.0, 2.0),
    )
    assign_genome_id(g, ["mobilenet", "resnet"])
    return g


def test_genome_roundtrip_preserves_public_fields(genome_a):
    restored = genome_from_dict(genome_to_dict(genome_a))
    assert restored == genome_a


def test_assign_genome_id_is_deterministic_and_sensitive(genome_a):
    g1 = copy.deepcopy(genome_a)
    g2 = copy.deepcopy(genome_a)
    assert assign_genome_id(g1, ["mobilenet", "resnet"]) == assign_genome_id(g2, ["mobilenet", "resnet"])

    g2.dropout = 0.31
    assert assign_genome_id(g1, ["mobilenet", "resnet"]) != assign_genome_id(g2, ["mobilenet", "resnet"])


def test_sample_genome_respects_search_space_bounds(cfg):
    g = sample_genome(cfg, random.Random(123))
    ss = cfg["search_space"]
    assert ss["base_lr1"][0] <= g.base_lr1 <= ss["base_lr1"][1]
    assert ss["base_lr2"][0] <= g.base_lr2 <= ss["base_lr2"][1]
    assert g.n_last_layers in ss["n_last_layers"]
    assert g.dense_units in ss["dense_units"]
    assert ss["dropout"][0] <= g.dropout <= ss["dropout"][1]
    assert g.l2_weight in ss["l2_weight"]
    assert g.batch_size in ss["batch_size"]
    assert ss["aug_rotation"][0] <= g.aug_rotation <= ss["aug_rotation"][1]
    assert ss["aug_zoom"][0] <= g.aug_zoom <= ss["aug_zoom"][1]
    assert ss["aug_contrast"][0] <= g.aug_contrast <= ss["aug_contrast"][1]
    assert set(g.lr1_mul) == set(cfg["model_names"])
    assert set(g.lr2_mul) == set(cfg["model_names"])
    assert len(g.genome_id) == 12


def test_hyper_for_backbone_clamps_learning_rates_and_unfreeze_cap(genome_a):
    genome_a.base_lr1 = 1.0
    genome_a.base_lr2 = 1.0
    genome_a.n_last_layers = 999
    out = hyper_for_backbone(genome_a, "mobilenet", {"mobilenet": 7})
    assert out == {"lr1": 2e-3, "lr2": 1e-4, "n_last_layers": 7}


def test_dominates_and_budget_aware_semantics(cfg, genome_a):
    budget_levels = build_budget_level_map(cfg)

    worse = copy.deepcopy(genome_a)
    worse.objectives = (0.2, 12.0, 2.1)
    assert dominates(genome_a.objectives, worse.objectives) is True
    assert dominates_budget_aware(budget_levels, genome_a, worse) is True
    worse.last_budget = "B2"
    assert dominates_budget_aware(budget_levels, genome_a, worse) is False


def test_fast_non_dominated_sort_separates_fronts_within_same_budget(cfg, genome_a):
    g1 = copy.deepcopy(genome_a)
    g2 = copy.deepcopy(genome_a)
    g3 = copy.deepcopy(genome_a)
    g1.objectives = (0.10, 8.0, 2.0)
    g2.objectives = (0.20, 9.0, 2.2)
    g3.objectives = (0.05, 20.0, 2.5)
    fronts = fast_non_dominated_sort(cfg, [g1, g2, g3])
    assert fronts[0] == [0, 2]
    assert fronts[1] == [1]


def test_crowding_distance_marks_boundary_points_as_infinite(genome_a):
    g1 = copy.deepcopy(genome_a)
    g2 = copy.deepcopy(genome_a)
    g3 = copy.deepcopy(genome_a)
    g1.objectives = (0.1, 5.0, 2.0)
    g2.objectives = (0.2, 6.0, 3.0)
    g3.objectives = (0.3, 7.0, 4.0)
    cd = crowding_distance([g1, g2, g3], [0, 1, 2])
    assert cd[0] == float("inf")
    assert cd[2] == float("inf")
    assert cd[1] > 0.0


def test_nsga2_select_prioritizes_budget_then_rank_then_crowd(cfg, genome_a):
    low_budget = copy.deepcopy(genome_a)
    low_budget.last_budget = "B0"
    low_budget.rank = 0
    low_budget.crowd = 999

    best = copy.deepcopy(genome_a)
    best.last_budget = "B2"
    best.rank = 0
    best.crowd = 2.0

    weaker = copy.deepcopy(genome_a)
    weaker.last_budget = "B2"
    weaker.rank = 1
    weaker.crowd = 100.0

    selected = nsga2_select(cfg, [weaker, low_budget, best], 2)
    assert selected == [best, weaker]


def test_mutate_keeps_all_numeric_fields_inside_bounds(cfg, genome_a):
    mutated = mutate(cfg, copy.deepcopy(genome_a), random.Random(7))
    ss = cfg["search_space"]
    assert ss["base_lr1"][0] <= mutated.base_lr1 <= ss["base_lr1"][1]
    assert ss["base_lr2"][0] <= mutated.base_lr2 <= ss["base_lr2"][1]
    assert mutated.n_last_layers in ss["n_last_layers"]
    assert mutated.dense_units in ss["dense_units"]
    assert ss["dropout"][0] <= mutated.dropout <= ss["dropout"][1]
    assert mutated.l2_weight in ss["l2_weight"]
    assert mutated.batch_size in ss["batch_size"]
    assert ss["aug_rotation"][0] <= mutated.aug_rotation <= ss["aug_rotation"][1]
    assert ss["aug_zoom"][0] <= mutated.aug_zoom <= ss["aug_zoom"][1]
    assert ss["aug_contrast"][0] <= mutated.aug_contrast <= ss["aug_contrast"][1]
    assert all(0.8 <= v <= 1.4 for v in mutated.lr1_mul.values())
    assert all(0.8 <= v <= 1.4 for v in mutated.lr2_mul.values())
    assert len(mutated.genome_id) == 12


def test_make_offspring_returns_requested_number_of_unique_children(cfg, genome_a):
    p1 = copy.deepcopy(genome_a)
    p2 = copy.deepcopy(genome_a)
    p2.dropout = 0.45
    assign_genome_id(p2, cfg["model_names"])
    kids = make_offspring(cfg, [p1, p2], 5, random.Random(11), deduplicate=True)
    assert len(kids) == 5
    assert len({k.genome_id for k in kids}) == 5


def test_pareto_front_without_budget_awareness_filters_dominated_points(genome_a):
    g1 = copy.deepcopy(genome_a)
    g2 = copy.deepcopy(genome_a)
    g3 = copy.deepcopy(genome_a)
    g1.objectives = (0.10, 5.0, 1.0)
    g2.objectives = (0.20, 6.0, 2.0)
    g3.objectives = (0.05, 7.0, 3.0)
    front = pareto_front([g1, g2, g3], cfg=None)
    assert front == [g3, g1]


def test_nsga2_select_supports_multi_objective_sh_promotion_within_same_budget(cfg, genome_a):
    high_auc = copy.deepcopy(genome_a)
    high_auc.last_budget = "B1"
    high_auc.objectives = (0.05, 50.0, 9.0)

    fast_small = copy.deepcopy(genome_a)
    fast_small.last_budget = "B1"
    fast_small.objectives = (0.15, 5.0, 1.0)

    dominated = copy.deepcopy(genome_a)
    dominated.last_budget = "B1"
    dominated.objectives = (0.25, 6.0, 2.0)

    pop = [high_auc, fast_small, dominated]
    from finetune_ga.search.operators import assign_rank_and_crowd
    assign_rank_and_crowd(cfg, pop)
    selected = nsga2_select(cfg, pop, 2)
    assert high_auc in selected
    assert fast_small in selected
    assert dominated not in selected


def test_make_offspring_warns_and_allows_duplicates_when_unique_search_space_is_exhausted(cfg, monkeypatch):
    rng = random.Random(123)
    parent = sample_genome(cfg, rng)
    parents = [parent]

    def same_genome(_cfg, _rng):
        g = copy.deepcopy(parent)
        assign_genome_id(g, _cfg["model_names"])
        return g

    monkeypatch.setattr("finetune_ga.search.operators.sample_genome", same_genome)

    with pytest.warns(RuntimeWarning, match=r"Unable to generate 2 unique offspring"):
        kids = make_offspring(
            cfg,
            parents,
            n=2,
            rng=rng,
            allow_crossover=False,
            allow_mutation=False,
            deduplicate=True,
        )

    assert len(kids) == 2
    assert kids[0].genome_id == kids[1].genome_id == parent.genome_id


def test_pareto_front_handles_invalid_objectives_without_crashing(genome_a):
    valid = copy.deepcopy(genome_a)
    invalid_nan = copy.deepcopy(genome_a)
    invalid_none = copy.deepcopy(genome_a)

    valid.objectives = (0.10, 5.0, 1.0)
    invalid_nan.objectives = (float("nan"), 1.0, 1.0)
    invalid_none.objectives = None

    front = pareto_front([invalid_nan, valid, invalid_none], cfg=None)

    assert front == [valid]


def test_pareto_front_budget_aware_handles_invalid_objectives_without_crashing(cfg, genome_a):
    valid = copy.deepcopy(genome_a)
    invalid = copy.deepcopy(genome_a)

    valid.last_budget = "B2"
    valid.objectives = (0.10, 5.0, 1.0)
    invalid.last_budget = "B2"
    invalid.objectives = (float("inf"), 1.0, 1.0)

    front = pareto_front([invalid, valid], cfg=cfg)

    assert front == [valid]
