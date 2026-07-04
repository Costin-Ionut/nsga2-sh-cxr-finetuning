import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_test import pick_topk_from_summary, select_single_backbone_run
from finetune_ga.analysis.reports import (
    aggregate_frame,
    best_row,
    build_pretty_test_tables,
    cliffs_delta,
    flatten_summary_rows,
    generate_test_reports,
)
from finetune_ga.core.metrics import (
    compute_confidence_metrics,
    enrich_summary_with_run_metrics,
    hypervolume_2d,
    hypervolume_3d,
    objectives_from_summary,
    replication_metrics,
    summarize_budget_runs,
    wilson_score_interval,
)
from finetune_ga.infra.experiment_records import build_summary_row
from finetune_ga.infra.io_utils import append_jsonl, read_jsonl


def test_replication_metrics_handles_empty_input_defensively():
    out = replication_metrics([])
    assert out == {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "median": 0.0,
        "q1": 0.0,
        "q3": 0.0,
    }


def test_wilson_score_interval_respects_probability_bounds():
    lo, hi = wilson_score_interval(7, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_hypervolume_2d_matches_simple_exact_area():
    hv = hypervolume_2d([(1.0, 4.0), (3.0, 2.0)], ref=(5.0, 5.0))
    assert hv == pytest.approx(8.0)


def test_hypervolume_3d_matches_single_box_volume():
    hv = hypervolume_3d([(1.0, 2.0, 3.0)], ref=(5.0, 6.0, 7.0))
    assert hv == pytest.approx(64.0)


def test_compute_confidence_metrics_returns_expected_perfect_case_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    out = compute_confidence_metrics(y_true, y_prob)
    assert out["brier_score"] < 0.001
    assert out["log_loss"] < 0.03
    assert out["ece_10bins"] < 0.03
    assert out["mean_positive_confidence"] > 0.98
    assert out["mean_negative_confidence"] > 0.98


def test_enrich_summary_with_run_metrics_adds_summary_stats_and_hypervolumes():
    runs = [
        {"best_val_auc": 0.91, "time_s": 100.0, "trainable_params_m": 1.2, "target_size": 224, "rep": 1, "backbone": "m"},
        {"best_val_auc": 0.89, "time_s": 110.0, "trainable_params_m": 1.4, "target_size": 224, "rep": 1, "backbone": "r"},
    ]
    out = enrich_summary_with_run_metrics({}, runs)
    assert out["best_val_auc_mean"] == pytest.approx(0.90)
    assert out["time_s_sum"] == pytest.approx(210.0)
    assert out["hypervolume_2d"] > 0.0
    assert out["hypervolume_3d"] > 0.0


def test_summarize_budget_runs_computes_robustness_and_complexity():
    cfg = {"complexity_weights": {"params_m": 0.55, "time_h": 0.35, "size_ratio": 0.10}}
    runs = [
        {"rep": 1, "backbone": "m", "best_val_auc": 0.95, "best_val_pr_auc": 0.96, "best_val_accuracy": 0.90, "loss_at_best_auc": 0.10, "time_s": 100.0, "trainable_params_m": 1.0, "target_size": 224},
        {"rep": 1, "backbone": "r", "best_val_auc": 0.92, "best_val_pr_auc": 0.93, "best_val_accuracy": 0.88, "loss_at_best_auc": 0.20, "time_s": 120.0, "trainable_params_m": 2.0, "target_size": 224},
        {"rep": 2, "backbone": "m", "best_val_auc": 0.94, "best_val_pr_auc": 0.95, "best_val_accuracy": 0.89, "loss_at_best_auc": 0.15, "time_s": 90.0, "trainable_params_m": 1.5, "target_size": 224},
        {"rep": 2, "backbone": "r", "best_val_auc": 0.91, "best_val_pr_auc": 0.92, "best_val_accuracy": 0.87, "loss_at_best_auc": 0.25, "time_s": 130.0, "trainable_params_m": 2.5, "target_size": 224},
    ]
    out = summarize_budget_runs(cfg, runs, "B2")
    assert out["budget"] == "B2"
    assert out["robust_min_auc"] == pytest.approx(0.91)
    assert out["worst_backbone"] == "r"
    assert out["total_time_s"] == pytest.approx(440.0)
    assert out["mean_trainable_params_m"] == pytest.approx(1.75)
    assert out["complexity_score"] > 0.0


def test_objectives_from_summary_maps_to_minimization_tuple():
    s = {"mean_val_auc": 0.93, "total_time_s": 123.0, "mean_trainable_params_m": 4.5}
    assert objectives_from_summary(s) == pytest.approx((0.07, 123.0, 4.5))


def test_flatten_summary_rows_unpacks_objective_dict_and_preserves_seed():
    rows = [{"genome_id": "g1", "objectives": {"mean_val_auc_loss_obj": 0.1, "selection_time_s_obj": 10.0, "trainable_params_m_obj": 2.0}}]
    out = flatten_summary_rows(rows, seed_val=42)
    assert out[0]["seed"] == 42
    assert out[0]["mean_val_auc_loss_obj"] == 0.1
    assert out[0]["selection_time_s_obj"] == 10.0
    assert out[0]["trainable_params_m_obj"] == 2.0


def test_aggregate_frame_computes_group_statistics():
    df = pd.DataFrame({"tag": ["a", "a", "b"], "score": [1.0, 3.0, 2.0]})
    out = aggregate_frame(df, ["tag"], ["score"])
    a = out[out["tag"] == "a"].iloc[0]
    assert a["score_mean"] == pytest.approx(2.0)
    assert a["score_median"] == pytest.approx(2.0)


def test_best_row_prefers_higher_auc_but_lower_loss_like_metrics():
    df = pd.DataFrame([
        {"name": "x", "test_auc": 0.91, "test_log_loss": 0.30},
        {"name": "y", "test_auc": 0.93, "test_log_loss": 0.40},
        {"name": "z", "test_auc": 0.93, "test_log_loss": 0.20},
    ])
    out = best_row(df, ["test_auc", "test_log_loss"])
    assert out.iloc[0]["name"] == "z"


def test_cliffs_delta_sign_and_magnitude_are_reasonable():
    d = cliffs_delta([5, 6, 7], [1, 2, 3])
    assert d > 0.9
    assert cliffs_delta([], [1, 2]) == 0.0


def test_pick_topk_from_summary_prefers_requested_budget_and_true_three_objective_ranking(tmp_path):
    summary_path = tmp_path / "genome_summary.jsonl"
    rows = [
        {"genome_id": "g1", "budget": "B1", "mean_val_auc": 0.70, "robust_min_auc": 0.70, "total_time_s": 80.0, "mean_trainable_params_m": 2.0},
        {"genome_id": "g1", "budget": "B2", "mean_val_auc": 0.90, "robust_min_auc": 0.82, "total_time_s": 120.0, "mean_trainable_params_m": 5.0},
        {"genome_id": "g2", "budget": "B2", "mean_val_auc": 0.91, "robust_min_auc": 0.80, "total_time_s": 200.0, "mean_trainable_params_m": 7.0},
        {"genome_id": "g3", "budget": "B2", "mean_val_auc": 0.89, "robust_min_auc": 0.79, "total_time_s": 90.0, "mean_trainable_params_m": 3.0},
    ]
    for row in rows:
        append_jsonl(str(summary_path), row)
    cfg = {
        "budgets": [{"name": "B0"}, {"name": "B1"}, {"name": "B2"}],
        "test_selection_policy": {
            "summary_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"],
        },
    }
    top, budget = pick_topk_from_summary(str(summary_path), cfg, preferred_budget_name="B2", k=2)
    assert budget == "B2"
    assert [r["genome_id"] for r in top] == ["g1", "g3"]
    assert all(r["summary_selection_objectives"] == ["auc_loss", "time_s", "trainable_params_m"] for r in top)
    assert all(r["summary_pareto_front_rank"] == 1 for r in top)

def test_pick_topk_from_summary_uses_next_pareto_front_when_k_exceeds_front_size(tmp_path):
    summary_path = tmp_path / "genome_summary.jsonl"
    rows = [
        {"genome_id": "g1", "budget": "B2", "mean_val_auc": 0.90, "total_time_s": 100.0, "mean_trainable_params_m": 5.0},
        {"genome_id": "g2", "budget": "B2", "mean_val_auc": 0.88, "total_time_s": 80.0, "mean_trainable_params_m": 3.0},
        {"genome_id": "g3", "budget": "B2", "mean_val_auc": 0.86, "total_time_s": 130.0, "mean_trainable_params_m": 6.0},
    ]
    for row in rows:
        append_jsonl(str(summary_path), row)
    cfg = {
        "budgets": [{"name": "B2"}],
        "test_selection_policy": {
            "summary_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"],
        },
    }
    top, budget = pick_topk_from_summary(str(summary_path), cfg, preferred_budget_name="B2", k=3)
    assert budget == "B2"
    assert [r["summary_pareto_front_rank"] for r in top] == [1, 1, 2]
    assert {r["genome_id"] for r in top[:2]} == {"g1", "g2"}
    assert top[2]["genome_id"] == "g3"


def test_read_jsonl_skips_malformed_lines_but_keeps_valid_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")
    rows = read_jsonl(str(path))
    assert rows == [{"a": 1}, {"b": 2}]


def test_select_single_backbone_run_uses_true_three_objective_pareto_selection():
    runs = [
        {"genome_id": "g1", "budget": "B2", "backbone": "mobilenet", "best_val_auc": 0.91, "best_val_pr_auc": 0.92, "best_val_accuracy": 0.88, "trainable_params_m": 3.0, "time_s": 100.0},
        {"genome_id": "g1", "budget": "B2", "backbone": "resnet", "best_val_auc": 0.93, "best_val_pr_auc": 0.91, "best_val_accuracy": 0.87, "trainable_params_m": 8.0, "time_s": 150.0},
        {"genome_id": "g1", "budget": "B2", "backbone": "efficientnet", "best_val_auc": 0.92, "best_val_pr_auc": 0.95, "best_val_accuracy": 0.89, "trainable_params_m": 4.0, "time_s": 105.0},
    ]
    cfg = {"test_selection_policy": {"backbone_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"]}}
    bb, row = select_single_backbone_run(runs, genome_id="g1", selected_budget="B2", cfg=cfg)
    assert bb == "efficientnet"
    assert row["ideal_distance"] >= 0.0
    assert row["backbone_selection_objectives"] == ["auc_loss", "time_s", "trainable_params_m"]
    assert row["backbone_selection_runtime_metrics"] == ["best_val_auc", "time_s", "trainable_params_m"]


def test_select_single_backbone_run_skips_infeasible_rows():
    runs = [
        {"genome_id": "g1", "budget": "B2", "backbone": "bad", "best_val_auc": 1.0, "trainable_params_m": 0.1, "time_s": 0.1, "status": "infeasible"},
        {"genome_id": "g1", "budget": "B2", "backbone": "good", "best_val_auc": 0.92, "trainable_params_m": 4.0, "time_s": 105.0, "status": "ok"},
    ]
    cfg = {"test_selection_policy": {"backbone_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"]}}
    bb, row = select_single_backbone_run(runs, genome_id="g1", selected_budget="B2", cfg=cfg)
    assert bb == "good"
    assert row["backbone"] == "good"


def test_collect_probs_and_loss_supports_binary_sigmoid_outputs(monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.evaluate_test import collect_probs_and_loss

    class DummyTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def numpy(self):
            return self._arr

        def __getitem__(self, item):
            return DummyTensor(self._arr[item])

    def _to_numpy(value):
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    class FakeResourceExhaustedError(Exception):
        pass

    class FakeOpError(Exception):
        pass

    fake_tf = SimpleNamespace(
        convert_to_tensor=lambda x: DummyTensor(x),
        cast=lambda x, dtype: DummyTensor(_to_numpy(x).astype(np.float32)),
        float32=np.float32,
        shape=lambda x: DummyTensor(np.asarray(_to_numpy(x).shape, dtype=np.int32)),
        reduce_sum=lambda x: DummyTensor(np.asarray(_to_numpy(x).sum(), dtype=np.float32)),
        get_logger=lambda: SimpleNamespace(setLevel=lambda level: None),
        keras=SimpleNamespace(
            losses=SimpleNamespace(
                sparse_categorical_crossentropy=lambda y_true, y_pred, from_logits=False: DummyTensor(
                    -np.log(np.clip(_to_numpy(y_pred)[np.arange(len(_to_numpy(y_true))), _to_numpy(y_true).astype(int)], 1e-7, 1.0))
                )
            ),
            mixed_precision=SimpleNamespace(global_policy=lambda: "float32"),
        ),
        errors=SimpleNamespace(ResourceExhaustedError=FakeResourceExhaustedError, OpError=FakeOpError),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    class DummyModel:
        def __call__(self, xb, training=False):
            del xb, training
            return np.asarray([[0.2], [0.8]], dtype=float)

    dataset = [(np.zeros((2, 4, 4, 3), dtype=float), DummyTensor([0, 0]))]

    y_true, y_prob, mean_loss = collect_probs_and_loss(DummyModel(), dataset, steps=1)
    assert y_true.tolist() == [0, 0]
    assert y_prob.tolist() == pytest.approx([0.2, 0.8])
    assert mean_loss == pytest.approx(float((-np.log(0.2) - np.log(0.8)) / 2.0))


def test_existing_done_key_includes_selection_rank_and_budget():
    cfg = {
        "test_from_tags": ["nsga2"],
        "test_topk_per_tag": 2,
        "test_e1": 11,
        "test_e2": 5,
        "final_retrain_mode": "train_plus_val_fixed_epochs",
        "final_retrain_uses_validation": False,
        "test_selection_policy": {},
    }
    from experiments.evaluate_test import build_test_protocol_id
    protocol_id = build_test_protocol_id(cfg)
    row_rank1 = {
        "seed": 1, "source_tag": "nsga2", "genome_id": "g1", "backbone": "bb",
        "selection_budget": "B2", "selection_rank": 1, "test_protocol_id": protocol_id,
    }
    row_rank2 = dict(row_rank1, selection_rank=2)
    key1 = (int(row_rank1["seed"]), row_rank1["source_tag"], row_rank1["genome_id"], row_rank1["backbone"], row_rank1["selection_budget"], int(row_rank1["selection_rank"]), row_rank1["test_protocol_id"])
    key2 = (int(row_rank2["seed"]), row_rank2["source_tag"], row_rank2["genome_id"], row_rank2["backbone"], row_rank2["selection_budget"], int(row_rank2["selection_rank"]), row_rank2["test_protocol_id"])
    assert key1 != key2


def test_filter_test_rows_requires_exact_protocol_and_valid_rank():
    from finetune_ga.analysis.reports import filter_test_rows_for_current_protocol
    from experiments.evaluate_test import build_test_protocol_id

    cfg = {
        "test_from_tags": ["ga"],
        "test_topk_per_tag": 2,
        "test_vote_threshold": 0.5,
        "test_tta": False,
        "test_mc_dropout": False,
        "test_eval_all_checkpoints": False,
        "test_ensemble_strategy": "mean",
        "test_require_all_models": False,
    }
    protocol_id = build_test_protocol_id(cfg)
    rows = [
        {"source_tag": "ga", "test_protocol_id": protocol_id, "selection_rank": 1},
        {"source_tag": "ga", "test_protocol_id": None, "selection_rank": 1},
        {"source_tag": "ga", "test_protocol_id": "other", "selection_rank": 1},
        {"source_tag": "ga", "test_protocol_id": protocol_id, "selection_rank": "x"},
        {"source_tag": "other", "test_protocol_id": protocol_id, "selection_rank": 1},
    ]
    filtered = filter_test_rows_for_current_protocol(rows, cfg)
    assert filtered == [{"source_tag": "ga", "test_protocol_id": protocol_id, "selection_rank": 1}]


def test_pick_topk_from_summary_skips_incomplete_or_invalid_rows(tmp_path):
    summary_path = tmp_path / "genome_summary.jsonl"
    rows = [
        {"genome_id": "g1", "budget": "B2", "mean_val_auc": 0.81, "total_time_s": 120.0, "mean_trainable_params_m": 4.0},
        {"genome_id": "g2", "budget": "B2", "mean_val_auc": 0.90, "total_time_s": "bad", "mean_trainable_params_m": 3.0},
        {"genome_id": "g3", "budget": "B2", "mean_val_auc": 0.80, "total_time_s": 100.0},
        {"budget": "B2", "mean_val_auc": 0.99, "total_time_s": 50.0, "mean_trainable_params_m": 1.0},
    ]
    for row in rows:
        append_jsonl(str(summary_path), row)
    cfg = {
        "budgets": [{"name": "B2"}],
        "test_selection_policy": {
            "summary_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"],
        },
    }
    top, budget = pick_topk_from_summary(str(summary_path), cfg, preferred_budget_name="B2", k=3)
    assert budget == "B2"
    assert [r["genome_id"] for r in top] == ["g1"]


def test_existing_done_resume_skips_corrupt_rows():
    from experiments.evaluate_test import _safe_int

    seed = 7
    protocol_id = "proto123"
    rows = [
        {"test_protocol_id": protocol_id, "seed": "bad", "source_tag": "ga", "genome_id": "g1", "backbone": "bb", "selection_budget": "B2", "selection_rank": "bad"},
        {"test_protocol_id": protocol_id, "seed": 9, "source_tag": "ga", "genome_id": "g2", "backbone": "bb", "selection_budget": "B2", "selection_rank": 2},
        {"test_protocol_id": protocol_id, "seed": 1, "source_tag": None, "genome_id": "g3", "backbone": "bb", "selection_budget": "B2", "selection_rank": 1},
    ]
    existing_done = set()
    for r in rows:
        if r.get("test_protocol_id") != protocol_id:
            continue
        source_tag = r.get("source_tag")
        genome_id = r.get("genome_id")
        backbone = r.get("backbone")
        selection_budget = r.get("selection_budget")
        if not source_tag or not genome_id or not backbone or not selection_budget:
            continue
        existing_done.add((
            _safe_int(r.get("seed", seed), seed),
            source_tag,
            genome_id,
            backbone,
            selection_budget,
            _safe_int(r.get("selection_rank", 1), 1),
            r.get("test_protocol_id"),
        ))
    assert existing_done == {
        (7, "ga", "g1", "bb", "B2", 1, protocol_id),
        (9, "ga", "g2", "bb", "B2", 2, protocol_id),
    }


def test_select_best_run_tolerates_invalid_numeric_metrics():
    from finetune_ga.infra.experiment_records import select_best_run

    rows = [
        {"backbone": "bad", "best_val_auc": "bad", "best_val_pr_auc": "bad", "best_val_accuracy": "bad", "loss_at_best_auc": "bad"},
        {"backbone": "good", "best_val_auc": 0.91, "best_val_pr_auc": 0.90, "best_val_accuracy": 0.89, "loss_at_best_auc": 0.3, "time_s": 100.0, "trainable_params_m": 2.0},
    ]
    best = select_best_run(rows)
    assert best["backbone"] == "good"


def test_select_best_run_uses_three_objective_ideal_point_rule():
    from finetune_ga.infra.experiment_records import select_best_run

    rows = [
        {"backbone": "high_auc_but_heavy", "best_val_auc": 0.95, "time_s": 200.0, "trainable_params_m": 10.0},
        {"backbone": "balanced", "best_val_auc": 0.93, "time_s": 110.0, "trainable_params_m": 3.0},
        {"backbone": "tiny_but_weaker", "best_val_auc": 0.89, "time_s": 90.0, "trainable_params_m": 1.5},
    ]
    best = select_best_run(rows)
    assert best["backbone"] == "balanced"
    assert best["ideal_distance"] >= 0.0


def test_summarize_budget_runs_skips_corrupt_rows_instead_of_crashing():
    cfg = {"complexity_weights": {"params_m": 0.55, "time_h": 0.35, "size_ratio": 0.10}}
    runs = [
        {"rep": 1, "backbone": "m", "best_val_auc": 0.95, "best_val_pr_auc": 0.96, "best_val_accuracy": 0.90, "loss_at_best_auc": 0.10, "time_s": 100.0, "trainable_params_m": 1.0, "target_size": 224},
        {"rep": 1, "backbone": "r", "best_val_auc": "bad", "time_s": 120.0},
        {"rep": None, "backbone": "x", "best_val_auc": 0.10},
    ]
    out = summarize_budget_runs(cfg, runs, "B2")
    assert out["n_runs"] == 1
    assert out["n_ok_runs"] == 1
    assert out["n_failed_runs"] == 2
    assert out["success_rate"] == pytest.approx(1 / 3)
    assert out["robust_min_auc_raw_ok_only"] == pytest.approx(0.95)
    assert out["robust_min_auc"] == pytest.approx(0.95 / 3)


def test_filter_test_rows_uses_default_candidate_tags_when_config_omits_tags():
    from finetune_ga.analysis.reports import filter_test_rows_for_current_protocol
    from experiments.evaluate_test import build_test_protocol_id

    cfg = {"test_topk_per_tag": 1, "test_selection_policy": {}}
    protocol_id = build_test_protocol_id(cfg)
    rows = [
        {"source_tag": "nsga2", "test_protocol_id": protocol_id, "selection_rank": 1},
        {"source_tag": "weird_tag", "test_protocol_id": protocol_id, "selection_rank": 1},
    ]
    filtered = filter_test_rows_for_current_protocol(rows, cfg)
    assert filtered == [{"source_tag": "nsga2", "test_protocol_id": protocol_id, "selection_rank": 1}]


def test_build_pretty_test_tables_reports_ci95_as_interval_bounds():
    grouped = pd.DataFrame([
        {
            "source_tag": "nsga2",
            "test_accuracy_ci95_low_mean": 0.81,
            "test_accuracy_ci95_high_mean": 0.91,
            "test_accuracy_ci95_low_min": 0.79,
            "test_accuracy_ci95_high_max": 0.93,
            "test_accuracy_ci95_low_median": 0.82,
            "test_accuracy_ci95_high_median": 0.90,
            "test_precision_ci95_low_mean": 0.71,
            "test_precision_ci95_high_mean": 0.88,
            "test_precision_ci95_low_min": 0.70,
            "test_precision_ci95_high_max": 0.89,
            "test_precision_ci95_low_median": 0.72,
            "test_precision_ci95_high_median": 0.87,
            "test_recall_ci95_low_mean": 0.76,
            "test_recall_ci95_high_mean": 0.92,
            "test_recall_ci95_low_min": 0.75,
            "test_recall_ci95_high_max": 0.93,
            "test_recall_ci95_low_median": 0.77,
            "test_recall_ci95_high_median": 0.91,
            "test_specificity_ci95_low_mean": 0.80,
            "test_specificity_ci95_high_mean": 0.94,
            "test_specificity_ci95_low_min": 0.79,
            "test_specificity_ci95_high_max": 0.95,
            "test_specificity_ci95_low_median": 0.81,
            "test_specificity_ci95_high_median": 0.93,
        }
    ])
    tables = build_pretty_test_tables(grouped)
    uncertainty = tables["table_uncertainty.md"]
    row = uncertainty.iloc[0]
    assert row["Accuracy CI95 interval summary"] == "[0.8100, 0.9100] | span [0.7900, 0.9300] | med [0.8200, 0.9000]"
    assert row["Precision CI95 interval summary"].startswith("[0.7100, 0.8800]")
    assert row["Recall CI95 interval summary"].startswith("[0.7600, 0.9200]")
    assert row["Specificity CI95 interval summary"].startswith("[0.8000, 0.9400]")


def test_generate_test_reports_writes_true_pareto_tradeoff_outputs(tmp_path):
    seed_root = str(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.DataFrame([
        {"source_tag": "fast", "seed": 1, "backbone": "m", "test_auc": 0.88, "test_pr_auc": 0.87, "test_f1": 0.80, "test_mcc": 0.70, "test_accuracy": 0.83,
         "elapsed_total_test_pipeline_s": 90.0, "elapsed_test_train_s": 60.0, "elapsed_test_eval_s": 30.0, "trainable_params_m": 2.0, "complexity_score": 0.2},
        {"source_tag": "accurate", "seed": 1, "backbone": "r", "test_auc": 0.93, "test_pr_auc": 0.91, "test_f1": 0.84, "test_mcc": 0.76, "test_accuracy": 0.86,
         "elapsed_total_test_pipeline_s": 150.0, "elapsed_test_train_s": 100.0, "elapsed_test_eval_s": 50.0, "trainable_params_m": 5.0, "complexity_score": 0.5},
        {"source_tag": "dominated", "seed": 1, "backbone": "e", "test_auc": 0.87, "test_pr_auc": 0.86, "test_f1": 0.79, "test_mcc": 0.68, "test_accuracy": 0.82,
         "elapsed_total_test_pipeline_s": 180.0, "elapsed_test_train_s": 120.0, "elapsed_test_eval_s": 60.0, "trainable_params_m": 6.0, "complexity_score": 0.6},
    ])
    generate_test_reports(seed_root, str(analysis_dir), test_df)
    ranked = pd.read_csv(analysis_dir / "test_diagnostic_tradeoff_pareto_ranked.csv")
    front = pd.read_csv(analysis_dir / "test_diagnostic_tradeoff_pareto_front.csv")
    assert set(front["source_tag"]) == {"fast", "accurate"}
    assert set(ranked[ranked["pareto_front_rank"] == 1]["source_tag"]) == {"fast", "accurate"}
    assert ranked.loc[ranked["source_tag"] == "dominated", "pareto_front_rank"].iloc[0] > 1
    assert (analysis_dir / "test_diagnostic_tradeoff_pareto_front.png").exists()
    assert not (analysis_dir / "pareto_front.png").exists()


def test_build_summary_row_requires_robust_min_auc():
    with pytest.raises(KeyError):
        build_summary_row(
            gen_idx=0,
            tag='t',
            genome_id='g',
            budget_name='B1',
            summary={'total_time_s': 10.0, 'mean_trainable_params_m': 1.0},
            objectives=(0.1, 10.0, 1.0),
            best_run={'selection_objectives': {}, 'selection_normalized_objectives': {}},
        )


def test_artifact_contract_lists_canonical_reports():
    from finetune_ga.analysis.artifact_contract import get_artifact_specs

    names = {spec.name for spec in get_artifact_specs()}
    assert 'selection_validation_pareto_ranked_candidates.csv' in names
    assert 'selection_top_validation_candidates.csv' in names
    assert 'test_diagnostic_tradeoff_pareto_ranked.csv' in names
    assert 'validation_comparison_by_method.csv' in names


def test_final_package_audit_detects_cache_artifacts(tmp_path):
    from finetune_ga.analysis.final_package_audit import run_final_package_audit

    (tmp_path / '__pycache__').mkdir()
    (tmp_path / 'bad.pyc').write_bytes(b'x')
    ok, report = run_final_package_audit(tmp_path)
    assert ok is False
    assert 'pycache_dirs_present=' in report
    assert 'pyc_files_present=' in report




def test_final_package_audit_reports_runtime_contract_selfcheck(tmp_path):
    from finetune_ga.analysis.final_package_audit import run_final_package_audit

    for name in ['README.md', 'requirements.txt', 'pyproject.toml', 'config.json', 'run_paper_reproduction.sh']:
        (tmp_path / name).write_text('x', encoding='utf-8')
    (tmp_path / 'FINAL_PACKAGE_AUDIT.txt').write_text('x', encoding='utf-8')
    (tmp_path / 'MIGRATION_REPORT.md').write_text('x', encoding='utf-8')

    ok, report = run_final_package_audit(tmp_path)
    assert ok is True
    assert 'runtime_contract_selfcheck=PASS' in report

def test_metric_contract_strict_validation_flags_unregistered_metrics():
    from finetune_ga.analysis.metric_contract import validate_metric_contract

    issues = validate_metric_contract(['mean_val_auc', 'mystery_metric'], allowed_roles={'selection_primary', 'selection_cost'}, strict=True)
    assert issues == ['unregistered_metric=mystery_metric']


def test_aggregate_frame_adds_count_and_ci95_columns():
    df = pd.DataFrame({"tag": ["a", "a", "a"], "score": [1.0, 2.0, 3.0]})
    out = aggregate_frame(df, ["tag"], ["score"])
    row = out.iloc[0]
    assert row["count_runs"] == 3
    assert row["score_count"] == 3
    assert row["score_ci95_low"] < row["score_mean"] < row["score_ci95_high"]


def test_generate_test_reports_deduplicates_rows_before_export(tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    seed_root = str(tmp_path)
    rows = [
        {"genome_id": "g1", "search_seed": 1, "final_train_seed": 11, "final_eval_seed": 101, "source_tag": "m1", "selection_rank": 1, "backbone": "b", "test_auc": 0.9, "test_pr_auc": 0.8, "test_f1": 0.7, "test_mcc": 0.6, "test_accuracy": 0.85},
        {"genome_id": "g1", "search_seed": 1, "final_train_seed": 11, "final_eval_seed": 101, "source_tag": "m1", "selection_rank": 1, "backbone": "b", "test_auc": 0.9, "test_pr_auc": 0.8, "test_f1": 0.7, "test_mcc": 0.6, "test_accuracy": 0.85},
    ]
    df = pd.DataFrame(rows)
    generate_test_reports(seed_root, str(analysis_dir), df)
    exported = pd.read_csv(analysis_dir / 'test_results_shared.csv')
    assert len(exported) == 1


def test_best_row_handles_nan_without_breaking_sort_order():
    df = pd.DataFrame([
        {"name": "bad", "test_auc": np.nan, "test_log_loss": 0.1},
        {"name": "good", "test_auc": 0.9, "test_log_loss": 0.2},
    ])
    out = best_row(df, ["test_auc", "test_log_loss"])
    assert out.iloc[0]["name"] == "good"


def test_build_pretty_test_tables_includes_compiled_loss_in_calibration_table():
    grouped = pd.DataFrame([{
        "source_tag": "method_a",
        "test_brier_score_mean": 0.11,
        "test_loss_mean": 0.22,
        "test_log_loss_mean": 0.23,
        "test_ece_10bins_mean": 0.05,
    }])
    tables = build_pretty_test_tables(grouped)
    calibration = tables["table_calibration.md"]
    assert "Compiled loss" in calibration.columns


def test_metric_contract_registers_test_loss_and_calibration_metrics():
    from finetune_ga.analysis.metric_contract import validate_metric_contract

    issues = validate_metric_contract([
        'test_loss', 'test_log_loss', 'test_brier_score', 'test_ece_10bins', 'test_loss_mean', 'test_log_loss_std'
    ], allowed_roles={'test_primary', 'test_cost_diagnostic', 'test_calibration', 'test_diagnostic', 'test_uncertainty'}, strict=True)
    assert issues == []


def test_artifact_contract_global_specs_require_explicit_loss_columns():
    from finetune_ga.analysis.artifact_contract import get_artifact_specs

    specs = {spec.name: spec for spec in get_artifact_specs(category='global')}
    test_cmp = specs['test_comparison_by_method.csv']
    ranking = specs['diagnostic_method_ranking.csv']
    assert 'test_loss_mean' in test_cmp.required_columns
    assert 'test_log_loss_mean' in test_cmp.required_columns
    assert 'test_brier_score_mean' in test_cmp.required_columns
    assert 'test_loss_mean' in ranking.required_columns
    assert 'test_log_loss_mean' in ranking.required_columns


def test_complexity_score_is_minmax_normalized_and_bounded():
    from finetune_ga.core.metrics import compute_complexity_components, compute_complexity_score

    cfg = {
        "complexity_weights": {
            "params_m": 0.55,
            "time_h": 0.35,
            "size_ratio": 0.10,
            "params_m_min": 1.0,
            "params_m_max": 100.0,
            "time_h_min": 0.0,
            "time_h_max": 24.0,
            "size_ratio_min": 0.5,
            "size_ratio_max": 1.5,
            "clip_normalized": True,
        }
    }

    components = compute_complexity_components(
        trainable_params_m=40.0,
        total_time_s=3600.0,
        target_size=224,
        cfg=cfg,
    )
    score = compute_complexity_score(
        trainable_params_m=40.0,
        total_time_s=3600.0,
        target_size=224,
        cfg=cfg,
    )

    assert 0.0 <= score <= 1.0
    assert 0.0 <= components["params_norm"] <= 1.0
    assert 0.0 <= components["time_norm"] <= 1.0
    assert 0.0 <= components["size_norm"] <= 1.0
    assert components["params_norm"] == pytest.approx((40.0 - 1.0) / 99.0)
    assert components["time_norm"] == pytest.approx(1.0 / 24.0)
    assert components["size_norm"] == pytest.approx(0.5)


def test_select_single_backbone_run_uses_configured_runtime_metrics():
    runs = [
        {"genome_id": "g1", "budget": "B2", "backbone": "auc_favored", "best_val_auc": 0.99, "alternate_val_auc": 0.70, "trainable_params_m": 2.0, "time_s": 10.0},
        {"genome_id": "g1", "budget": "B2", "backbone": "configured_metric_favored", "best_val_auc": 0.70, "alternate_val_auc": 0.99, "trainable_params_m": 2.0, "time_s": 10.0},
    ]
    cfg = {
        "test_selection_policy": {
            "backbone_selection_objectives": ["auc_loss", "time_s", "trainable_params_m"],
            "backbone_selection_runtime_metrics": ["alternate_val_auc", "time_s", "trainable_params_m"],
        }
    }

    bb, row = select_single_backbone_run(runs, genome_id="g1", selected_budget="B2", cfg=cfg)

    assert bb == "configured_metric_favored"
    assert row["backbone_selection_runtime_metrics"] == [
        "alternate_val_auc",
        "time_s",
        "trainable_params_m",
    ]
