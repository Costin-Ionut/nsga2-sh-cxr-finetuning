import pytest

from finetune_ga.core.genome import Genome, assign_genome_id
from finetune_ga.core.training import (
    best_epoch_by_auc,
    choose_better_stage,
    is_resource_exhaustion_error,
    make_infeasible_result,
    trainable_params_m,
)


class FakeVar:
    def __init__(self, shape):
        self.shape = shape


class FakeModel:
    def __init__(self, shapes):
        self.trainable_variables = [FakeVar(s) for s in shapes]


class FakeHistory:
    def __init__(self, history):
        self.history = history


def test_best_epoch_by_auc_returns_auc_loss_and_index_of_best_epoch():
    h = FakeHistory({"val_auc": [0.70, 0.85, 0.80], "val_loss": [0.9, 0.4, 0.5]})
    assert best_epoch_by_auc(h) == (0.85, 0.4, 1)


def test_best_epoch_by_auc_handles_missing_history_safely():
    assert best_epoch_by_auc(FakeHistory({})) == (0.0, 1e9, -1)


def test_choose_better_stage_uses_auc_then_loss_tie_breaker():
    assert choose_better_stage((0.80, 0.5, 1), (0.90, 0.9, 2)) == (0.90, 0.9, 2, 2)
    assert choose_better_stage((0.90, 0.6, 1), (0.90, 0.5, 2)) == (0.90, 0.5, 2, 2)


def test_trainable_params_m_counts_parameters_in_millions():
    model = FakeModel([(10, 10), (5,), None])
    assert trainable_params_m(model) == pytest.approx(0.000105)


def test_is_resource_exhaustion_error_detects_common_message_patterns():
    exc = RuntimeError("CUDA_ERROR_OUT_OF_MEMORY while allocating tensor")
    assert is_resource_exhaustion_error(exc) is True
    assert is_resource_exhaustion_error(RuntimeError("some other failure")) is False


def test_make_infeasible_result_marks_candidate_without_rescue():
    cfg = {"model_names": ["mobilenet"], "search_seeds": [42], "final_seeds": [42], "selection_source_seed": 42}
    budget = {"name": "B1"}
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
        lr1_mul={"mobilenet": 1.0},
        lr2_mul={"mobilenet": 1.0},
    )
    assign_genome_id(g, ["mobilenet"])
    err = RuntimeError("out of memory")
    out = make_infeasible_result(cfg, g, "mobilenet", budget, rep=0, reason="resource_exhausted", error=err)
    assert out["status"] == "infeasible"
    assert out["failure_reason"] == "resource_exhausted"
    assert out["best_val_auc"] == 0.0
    assert out["trainable_params_m"] == pytest.approx(1e9)
    assert out["error_type"] == "RuntimeError"


def test_safe_int_resume_checkpoint_values_do_not_crash(tmp_path):
    from finetune_ga.infra.io_utils import save_experiment_state
    from finetune_ga.baselines.full_budget import load_completed_index_state
    import random
    from finetune_ga.baselines.single_objective_ga import load_checkpoint
    from finetune_ga.search.nsga2 import load_nsga2_checkpoint

    root = str(tmp_path)
    save_experiment_state(root, "full_budget", {"completed_index": "bad"})
    assert load_completed_index_state(root, "full_budget", resume=True) == 0

    save_experiment_state(root, "single_objective_ga", {"gen_completed": "bad", "population": []})
    assert load_checkpoint(root, "single_objective_ga", random.Random(0))[0] == 0

    save_experiment_state(root, "nsga2", {"gen_completed": "bad", "population": []})
    assert load_nsga2_checkpoint(root, "nsga2", random.Random(0))[0] == 0


def test_nsga2_checkpoint_skips_invalid_population_entries(tmp_path):
    from finetune_ga.infra.io_utils import save_experiment_state
    from finetune_ga.search.nsga2 import load_nsga2_checkpoint
    import random

    root = str(tmp_path)
    save_experiment_state(root, "nsga2", {
        "gen_completed": "bad",
        "population": [{"oops": 1}],
    })

    gen_completed, pop = load_nsga2_checkpoint(root, "nsga2", random.Random(0))
    assert gen_completed == 0
    assert pop == []


def test_single_objective_checkpoint_skips_invalid_population_entries(tmp_path):
    import json
    import os
    import random

    from finetune_ga.baselines.single_objective_ga import load_checkpoint

    root = str(tmp_path / "runs")
    state_dir = os.path.join(root, "single_objective_ga")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "checkpoint_state.json"), "w", encoding="utf-8") as f:
        json.dump({"gen_completed": "bad", "population": [{"oops": 1}, 7, None]}, f)

    gen_completed, pop = load_checkpoint(root, "single_objective_ga", random.Random(0))
    assert gen_completed == 0
    assert pop == []


def test_random_full_budget_resume_imports_safe_int_and_handles_bad_completed_index(tmp_path):
    import json
    import os
    from unittest.mock import patch

    from finetune_ga.baselines import random_full_budget

    root_dir = str(tmp_path / "seed_42")
    exp_dir = os.path.join(root_dir, "random_full_budget")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "checkpoint_state.json"), "w", encoding="utf-8") as f:
        json.dump({"completed_index": "bad", "population": []}, f)

    cfg = {"pop_size": 1, "generations": 1}
    ctx = {
        "active_seed": 42,
        "root_dir": root_dir,
        "paths": {"exp_dir": exp_dir},
    }

    with patch.dict(os.environ, {"RESUME": "1"}, clear=False), \
         patch("finetune_ga.baselines.random_full_budget.initialize_baseline_run", return_value=ctx), \
         patch("finetune_ga.baselines.random_full_budget.sample_genome", return_value=type("G", (), {"genome_id": "g1", "objectives": (0,0,0), "last_budget": "B2"})()), \
         patch("finetune_ga.baselines.random_full_budget.genome_to_dict", return_value={"genome_id": "g1"}), \
         patch("finetune_ga.baselines.random_full_budget.assign_rank_and_crowd"), \
         patch("finetune_ga.baselines.random_full_budget.nsga2_select", return_value=[]), \
         patch("finetune_ga.baselines.random_full_budget.evaluate_genome_at_final_budget"), \
         patch("finetune_ga.baselines.random_full_budget.save_completed_index_state"), \
         patch("finetune_ga.baselines.random_full_budget.load_done_keys_from_path", return_value=set()):
        random_full_budget.run_for_seed(cfg, 42)


def test_single_attempt_clears_augmenter_cache_after_attempt(monkeypatch):
    from finetune_ga.core.genome import Genome, assign_genome_id
    from finetune_ga.core import training

    cfg = {
        "model_names": ["mobilenet"],
        "unfreeze_cap": {"mobilenet": 16},
        "num_classes": 2,
        "test_e1": 1,
        "test_e2": 1,
        "early_stopping_patience_stage1": 1,
        "early_stopping_patience_stage2": 1,
    }
    budget = {"name": "B1", "steps_factor": 1.0, "es_patience_s1": 1, "es_patience_s2": 1}
    g = Genome(
        base_lr1=1e-4,
        base_lr2=1e-5,
        n_last_layers=4,
        dense_units=64,
        dropout=0.2,
        l2_weight=1e-5,
        batch_size=8,
        aug_rotation=0.1,
        aug_zoom=0.1,
        aug_contrast=0.1,
        lr1_mul={"mobilenet": 1.0},
        lr2_mul={"mobilenet": 1.0},
    )
    assign_genome_id(g, ["mobilenet"])

    events = []

    monkeypatch.setattr(training, "seed_everything", lambda seed: events.append(("seed", int(seed))))
    monkeypatch.setattr(training, "_require_tensorflow", lambda: None)
    monkeypatch.setattr(training, "get_dataset_counts", lambda: {"train": 8, "val": 4})
    monkeypatch.setattr(training, "steps_safe", lambda *args, **kwargs: 1)
    monkeypatch.setattr(training, "full_steps", lambda *args, **kwargs: 1)
    monkeypatch.setattr(training, "print_run_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "ensure_head_trainable_backbone_frozen", lambda model: object())

    class FakeStrategy:
        num_replicas_in_sync = 1
        def scope(self):
            class _Scope:
                def __enter__(self_inner):
                    return None
                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return _Scope()

    class FakeMetric:
        def __init__(self, *args, **kwargs):
            pass

    class FakePolicy:
        def __str__(self):
            return "mixed_float16"

    class FakeKerasMetrics:
        class SparseCategoricalAccuracy:
            def __init__(self, *args, **kwargs):
                pass

    class FakeMixedPrecision:
        @staticmethod
        def global_policy():
            return FakePolicy()

    class FakeKeras:
        metrics = FakeKerasMetrics
        mixed_precision = FakeMixedPrecision

    class FakeTF:
        keras = FakeKeras

    monkeypatch.setattr(training, "STRATEGY", FakeStrategy())
    monkeypatch.setattr(training, "SoftmaxBinaryAUC", FakeMetric)
    monkeypatch.setattr(training, "tf", FakeTF())

    class StopHere(Exception):
        pass

    def fake_clear_augmenter_cache():
        events.append(("clear_aug", None))

    def fake_apply_preprocessing(*args, **kwargs):
        events.append(("apply", kwargs.get("augment")))
        raise StopHere()

    monkeypatch.setattr("finetune_ga.data.preprocessing.clear_augmenter_cache", fake_clear_augmenter_cache)
    monkeypatch.setattr("finetune_ga.data.preprocessing.apply_preprocessing", fake_apply_preprocessing)
    monkeypatch.setattr("finetune_ga.data.loader.load_train_val_data", lambda **kwargs: (object(), object()))
    monkeypatch.setattr("finetune_ga.models.backbone.build_model", lambda *args, **kwargs: (object(), 224))

    with pytest.raises(StopHere):
        training._train_one_backbone_single_attempt(
            cfg, g, "mobilenet", budget, rep=0, active_seed=42
        )

    assert events[0][0] == "seed"
    assert events[1] == ("apply", True)
    assert events[2][0] == "clear_aug"


def test_evaluate_test_clears_augmenter_cache_before_final_retrain(monkeypatch, tmp_path):
    import experiments.evaluate_test as evaluate_test

    events = []

    monkeypatch.setattr(evaluate_test, "build_test_protocol_id", lambda cfg: "proto")
    monkeypatch.setattr(evaluate_test, "get_seeded_root_dir", lambda out_dir, seed, seeds: str(tmp_path / f"seed_{seed}"))
    monkeypatch.setattr(evaluate_test, "seed_root_for", lambda out_dir, seed, seeds: str(tmp_path / "seed_42"))
    monkeypatch.setattr(evaluate_test, "pick_topk_from_summary", lambda *args, **kwargs: ([{"genome_id": "g1"}], "B2"))
    monkeypatch.setattr(evaluate_test, "read_jsonl", lambda path: [])
    monkeypatch.setattr(evaluate_test, "select_single_backbone_run", lambda *args, **kwargs: ("mobilenet", {
        "lr1": 1e-4, "lr2": 1e-5, "n_last_layers": 4, "batch_size": 8,
        "dense_units": 64, "dropout": 0.2, "l2_weight": 1e-5,
        "aug_rotation": 0.1, "aug_zoom": 0.1, "aug_contrast": 0.1,
    }))
    monkeypatch.setattr(evaluate_test, "write_selection_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_test, "write_test_row_everywhere", lambda *args, **kwargs: None)

    class FakeStrategy:
        def scope(self):
            class _Scope:
                def __enter__(self_inner):
                    return None
                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return _Scope()

    class FakeModel:
        def build(self, shape):
            return None

    class FakeKerasBackend:
        @staticmethod
        def clear_session():
            return None

    class FakeOptimizers:
        class Adam:
            def __init__(self, *args, **kwargs):
                pass

    class FakeKerasMetrics:
        class SparseCategoricalAccuracy:
            def __init__(self, *args, **kwargs):
                pass

    class FakeKeras:
        backend = FakeKerasBackend
        optimizers = FakeOptimizers
        metrics = FakeKerasMetrics

    class FakeErrors:
        class OpError(Exception):
            pass

    class FakeTF:
        keras = FakeKeras
        errors = FakeErrors

    def fake_apply_preprocessing(ds, **kwargs):
        events.append(("apply", kwargs.get("augment")))
        raise RuntimeError("stop-after-apply")

    rt = {
        "STRATEGY": FakeStrategy(),
        "seed_everything": lambda seed: None,
        "append_jsonl": lambda *args, **kwargs: None,
        "SoftmaxBinaryAUC": lambda *args, **kwargs: object(),
        "compute_binary_classification_metrics": lambda *args, **kwargs: {},
        "compute_complexity_score": lambda *args, **kwargs: 0.0,
        "ensure_head_trainable_backbone_frozen": lambda model: object(),
        "set_finetune_last_n_layers": lambda *args, **kwargs: None,
        "trainable_params_m": lambda model: 0.0,
        "full_steps": lambda count, bs: 1,
        "build_model": lambda *args, **kwargs: (FakeModel(), 224),
        "load_final_retrain_data": lambda **kwargs: (object(), object()),
        "apply_preprocessing": fake_apply_preprocessing,
        "clear_augmenter_cache": lambda: events.append("clear_aug"),
        "fallback_attempts_for_backbone": lambda bb, bs, base_size, n_last: [{"batch_size": bs, "target_size": base_size, "n_last_layers": n_last}],
        "is_resource_exhaustion_error": lambda exc: False,
        "clear_tf_memory": lambda: None,
        "make_run_artifact_paths": lambda *args, **kwargs: {},
        "persist_final_artifacts": lambda *args, **kwargs: {},
    }
    monkeypatch.setattr(evaluate_test, "_require_training_runtime", lambda: rt)
    monkeypatch.setattr(evaluate_test, "tf", FakeTF())
    monkeypatch.setattr(evaluate_test, "get_dataset_counts", lambda refresh=True: {"train": 8, "val": 4})

    root = tmp_path / "seed_42" / "nsga2"
    root.mkdir(parents=True)
    (root / "genome_summary.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "runs.jsonl").write_text("{}\n", encoding="utf-8")

    cfg = {
        "out_dir": str(tmp_path),
        "search_seeds": [42],
        "final_seeds": [42],
        "selection_source_seed": 42,
        "num_classes": 2,
        "test_candidate_tags": ["nsga2"],
        "test_selection_policy": {},
        "budgets": [{"name": "B2"}],
        "test_e1": 1,
        "test_e2": 1,
    }

    with pytest.raises(RuntimeError, match="stop-after-apply"):
        evaluate_test.evaluate_for_seed(cfg, active_seed=42, existing_done=set(), selection_seed=42)

    assert events[0] == "clear_aug"
    assert events[1] == ("apply", True)


def test_nsga2_passes_active_seed_to_train_one_backbone(monkeypatch, tmp_path):
    from finetune_ga.search import nsga2 as mod
    from finetune_ga.core.genome import Genome

    g = Genome(
        base_lr1=1e-4, base_lr2=1e-5, n_last_layers=4,
        dense_units=128, dropout=0.2, l2_weight=1e-4, batch_size=8, aug_rotation=0.1, aug_zoom=0.1, aug_contrast=0.1,
        lr1_mul={'mobilenet': 1.0}, lr2_mul={'mobilenet': 1.0}, genome_id='g1'
    )
    seen = {}

    monkeypatch.setattr(mod, 'prepare_output_paths', lambda root_dir, tag: {'exp_dir': str(tmp_path)})
    monkeypatch.setattr(mod, 'write_run_row_everywhere', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'write_genome_summary_everywhere', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'summarize_budget_runs', lambda cfg, rows, budget_name: {'mean_val_auc': 0.8, 'total_time_s': 1.0, 'mean_trainable_params_m': 1.0})
    monkeypatch.setattr(mod, 'objectives_from_summary', lambda summ: [0.2, 1.0, 1.0])
    monkeypatch.setattr(mod, 'select_best_run', lambda rows: rows[0])
    monkeypatch.setattr(mod, 'build_summary_row', lambda **kwargs: kwargs)
    monkeypatch.setattr(mod, 'assign_rank_and_crowd', lambda cfg, pop: None)
    monkeypatch.setattr(mod, 'nsga2_select', lambda cfg, pop, k: pop[:k])

    def fake_train(cfg, genome, bb, budget, rep, **kwargs):
        seen['active_seed'] = kwargs.get('active_seed')
        return {'genome_id': genome.genome_id, 'budget': budget['name'], 'backbone': bb, 'rep': rep, 'best_val_auc': 0.8, 'time_s': 1.0, 'trainable_params_m': 1.0}

    monkeypatch.setattr(mod, 'train_one_backbone', fake_train)

    def fake_run_search_tasks_ephemeral(cfg, tasks, on_result=None):
        from finetune_ga.core.genome import genome_from_dict

        rows = []
        for task in tasks:
            genome = genome_from_dict(task["genome"])
            row = mod.train_one_backbone(
                task["cfg"], genome, task["backbone"], task["budget"], int(task["rep"]),
                root_dir=task["root_dir"], tag=task["tag"], gen_idx=int(task["gen_idx"]),
                active_seed=int(task["active_seed"]), mode=task.get("mode", "search"),
            )
            rows.append(row)
            if on_result is not None:
                on_result(row)
        return rows

    monkeypatch.setattr(mod, '_run_search_tasks_ephemeral', fake_run_search_tasks_ephemeral)

    cfg = {'budgets': [{'name': 'B0', 'reps': 1}], 'promote_frac': 0.5, 'model_names': ['mobilenet']}
    mod.evaluate_population_with_sh(cfg, 1, [g], str(tmp_path), 'nsga2', False, active_seed=123)
    assert seen['active_seed'] == 123


def test_ensure_head_trainable_backbone_frozen_prefers_layer_zero():
    import types
    from finetune_ga.core import training as mod

    original_tf = mod.tf
    FakeModelBase = type('FakeModelBase', (), {})
    mod.tf = types.SimpleNamespace(keras=types.SimpleNamespace(Model=FakeModelBase, layers=types.SimpleNamespace(BatchNormalization=type('BN', (), {}))))

    class Backbone(FakeModelBase):
        def __init__(self):
            self.trainable = True
            self.layers = []

    class Pool:
        def __init__(self):
            self.trainable = True

    class Seq:
        def __init__(self):
            self.layers = [Backbone(), Pool()]

    try:
        model = Seq()
        backbone = mod.ensure_head_trainable_backbone_frozen(model)
        assert backbone is model.layers[0]
        assert model.layers[0].trainable is False
        assert model.layers[1].trainable is True
    finally:
        mod.tf = original_tf


def test_fallback_attempts_uses_final_img_size_in_final_mode(monkeypatch):
    """Regression: train_one_backbone must seed fallback attempts from FINAL_IMG_SIZE
    when mode='final', not always from SEARCH_IMG_SIZE (dual-imgsize bug)."""
    from finetune_ga.core import training as mod
    from finetune_ga.infra.config import SEARCH_IMG_SIZE, FINAL_IMG_SIZE

    captured = {}

    def fake_search_attempt(cfg, g, bb, budget, rep, *, override_target_size, **kwargs):
        captured['search_target_size'] = override_target_size
        return {
            'genome_id': g.genome_id, 'budget': budget['name'], 'backbone': bb,
            'rep': rep, 'best_val_auc': 0.8, 'time_s': 1.0,
            'trainable_params_m': 1.0, 'status': 'ok',
        }

    def fake_final_attempt(cfg, g, bb, budget, rep, *, override_target_size, **kwargs):
        captured['final_target_size'] = override_target_size
        return {
            'genome_id': g.genome_id, 'budget': budget['name'], 'backbone': bb,
            'rep': rep, 'best_val_auc': 0.8, 'time_s': 1.0,
            'trainable_params_m': 1.0, 'status': 'ok',
        }

    monkeypatch.setattr(mod, '_train_one_backbone_single_attempt', fake_search_attempt)
    monkeypatch.setattr(mod, '_train_one_backbone_final_single_attempt', fake_final_attempt)
    monkeypatch.setattr(mod, '_require_tensorflow', lambda: None)

    import random
    from finetune_ga.core.genome import sample_genome

    cfg = {
        'model_names': ['mobilenet'],
        'search_space': {
            'base_lr1': [1e-3], 'base_lr2': [1e-5], 'n_last_layers': [4],
            'dense_units': [256], 'dropout': [0.3, 0.5], 'l2_weight': [0.0],
            'batch_size': [32], 'aug_rotation': [0.0, 0.1], 'aug_zoom': [0.0, 0.1],
            'aug_contrast': [0.0, 0.1], 'lr_mul_bounds': [0.5, 2.0],
        },
        'unfreeze_cap': {'mobilenet': 10},
        'search_runtime': {}, 'final_runtime': {},
    }
    g = sample_genome(cfg, random.Random(1))
    budget = {'name': 'B0', 'e1': 1, 'e2': 0, 'steps_factor': 1.0, 'reps': 1}

    mod.train_one_backbone(cfg, g, 'mobilenet', budget, 0, mode='search')
    assert captured['search_target_size'] == SEARCH_IMG_SIZE, (
        f"search mode should use SEARCH_IMG_SIZE={SEARCH_IMG_SIZE}, got {captured['search_target_size']}"
    )

    mod.train_one_backbone(cfg, g, 'mobilenet', budget, 0, mode='final')
    assert captured['final_target_size'] == FINAL_IMG_SIZE, (
        f"final mode should use FINAL_IMG_SIZE={FINAL_IMG_SIZE}, got {captured['final_target_size']}"
    )


def test_dataset_config_snapshot_dataset_version_tolerates_none(monkeypatch):
    """Regression: dataset_config_snapshot must not raise AttributeError when
    _dataset_version() returns None and KAGGLE_DATASET_VERSION is unset."""
    import os
    from finetune_ga.infra import config as cfg_mod

    monkeypatch.delenv('KAGGLE_DATASET_VERSION', raising=False)
    monkeypatch.setattr(cfg_mod, '_dataset_version', lambda: None)

    snap = cfg_mod.dataset_config_snapshot(refresh=False)
    # Accessing dataset_version on the lazy snapshot must not crash and must be
    # None or a string — never raises AttributeError.
    ver = snap.get('dataset_version')
    assert ver is None or isinstance(ver, str), (
        f"dataset_version should be None or str, got {type(ver)}"
    )
