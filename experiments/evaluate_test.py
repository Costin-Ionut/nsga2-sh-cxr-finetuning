import os
import json
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import math

from finetune_ga.infra import runtime

tf = runtime.tf

from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.io_utils import atomic_write_json, prepare_output_paths, read_jsonl, safe_int
from finetune_ga.infra.run_utils import (
    get_seeded_root_dir,
    get_final_seeds,
    get_search_seeds,
    get_selection_source_seed,
    seed_root_for,
)
from finetune_ga.infra.config import IMG_SIZE, FINAL_IMG_SIZE, get_dataset_counts
from finetune_ga.infra.test_protocol import build_test_protocol_id, resolve_test_candidate_tags
from finetune_ga.selection.multiobjective import (
    FINAL_SELECTION_PROTOCOL_NAME,
    FINAL_SELECTION_OBJECTIVES,
    FINAL_SELECTION_RUNTIME_METRICS,
    prepare_selection_row,
    candidate_has_finite_objectives,
    select_by_ideal_point,
    non_dominated_sort_rows,
)


class InvalidFinalMetricsError(RuntimeError):
    """Raised only when final-test numeric outputs are invalid."""
    pass


def _safe_int(value: Any, default: int) -> int:
    return safe_int(value, default)


def _budget_rank_map(cfg: Dict[str, Any]) -> Dict[str, int]:
    return {str(b['name']): idx for idx, b in enumerate(cfg['budgets'])}


def pick_topk_from_summary(summary_path: str, cfg: Dict[str, Any], preferred_budget_name: str = "B2", k: int = 1) -> Tuple[List[Dict[str, Any]], str | None]:
    rows = read_jsonl(summary_path)
    if not rows:
        return [], None

    budget_rank = _budget_rank_map(cfg)
    available_budgets = [str(r.get('budget'))
                         for r in rows if r.get('budget') is not None]
    if not available_budgets:
        return [], None

    if preferred_budget_name in available_budgets:
        selected_budget = preferred_budget_name
    else:
        selected_budget = max(
            available_budgets, key=lambda name: budget_rank.get(name, -1))

    policy = cfg.get('test_selection_policy', {}) if isinstance(cfg, dict) else {}
    summary_objectives = tuple(policy.get('summary_selection_objectives', FINAL_SELECTION_OBJECTIVES))

    def _summary_metric(row: Dict[str, Any], metric: str, default: float = 0.0) -> float:
        raw_value = row.get(metric)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    per_genome: Dict[str, Dict[str, Any]] = {}
    for s in rows:
        if str(s.get('budget')) != selected_budget:
            continue
        gid = s.get('genome_id')
        if not gid:
            continue
        candidate = dict(s)
        try:
            mean_val_auc = _summary_metric(candidate, 'mean_val_auc', 0.0)
            total_time_s = _summary_metric(candidate, 'mean_time_s_per_run', _summary_metric(candidate, 'total_time_s', float('inf')))
            mean_trainable_params_m = _summary_metric(candidate, 'mean_trainable_params_m', float('inf'))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (mean_val_auc, total_time_s, mean_trainable_params_m)):
            continue
        candidate['mean_val_auc'] = mean_val_auc
        candidate['mean_time_s_per_run'] = max(0.0, total_time_s)
        candidate['mean_trainable_params_m'] = max(0.0, mean_trainable_params_m)
        candidate = prepare_selection_row(
            candidate,
            auc_key='mean_val_auc',
            time_key='mean_time_s_per_run',
            params_key='mean_trainable_params_m',
            runtime_metrics=['mean_val_auc', 'mean_time_s_per_run', 'mean_trainable_params_m'],
        )
        candidate['summary_selection_objectives'] = list(summary_objectives)
        candidate['summary_selection_runtime_metrics'] = ['mean_val_auc', 'mean_time_s_per_run', 'mean_trainable_params_m']

        existing = per_genome.get(str(gid))
        if existing is None:
            per_genome[str(gid)] = candidate
            continue
        chosen, _ = select_by_ideal_point([existing, candidate], summary_objectives, tie_break_auc_key='mean_val_auc', tie_break_name_key='genome_id')
        per_genome[str(gid)] = chosen

    candidates = list(per_genome.values())
    if not candidates:
        return [], selected_budget

    fronts = non_dominated_sort_rows(candidates, summary_objectives)
    ranked: List[Dict[str, Any]] = []
    for front_index, front in enumerate(fronts, start=1):
        _, scored_front = select_by_ideal_point(front, summary_objectives, tie_break_auc_key='mean_val_auc', tie_break_name_key='genome_id')
        for row in scored_front:
            row_copy = dict(row)
            row_copy['summary_pareto_front_rank'] = int(front_index)
            ranked.append(row_copy)

    return ranked[:k], selected_budget


def _json_safe_number(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def collect_probs_and_loss(model, dataset, steps: int | None = None):
    import numpy as np
    tf_module = runtime.get_tf_module_or_none()
    if tf_module is None:
        raise ModuleNotFoundError("TensorFlow is required to collect probabilities")

    y_true, y_prob = [], []
    total_loss = 0.0
    total_examples = 0

    iterator = iter(dataset)
    step_limit = None if steps is None else max(0, int(steps))
    step_idx = 0

    while step_limit is None or step_idx < step_limit:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            break

        pred = model(xb, training=False)
        pred = tf_module.convert_to_tensor(pred)
        prob = np.asarray(pred.numpy())

        # Guard final-test collection against invalid numeric model outputs.
        # The check reuses the arrays already materialized for metrics, avoiding
        # extra TensorFlow synchronizations in the normal path.
        if not np.all(np.isfinite(prob)):
            raise InvalidFinalMetricsError("Model produced NaN/Inf predictions during final evaluation")

        loss_vec = tf_module.keras.losses.sparse_categorical_crossentropy(
            y_true=yb,
            y_pred=pred,
            from_logits=False,
        )
        loss_values = np.asarray(tf_module.cast(loss_vec, tf_module.float32).numpy(), dtype=float)
        if not np.all(np.isfinite(loss_values)):
            raise InvalidFinalMetricsError("Final evaluation loss contains NaN/Inf values")

        batch_size = int(loss_values.shape[0])
        total_loss += float(np.sum(loss_values))
        total_examples += batch_size

        if prob.ndim == 2 and prob.shape[1] >= 2:
            positive_prob = prob[:, 1]
        elif prob.ndim == 2 and prob.shape[1] == 1:
            positive_prob = prob[:, 0]
        elif prob.ndim == 1:
            positive_prob = prob
        else:
            raise ValueError(f"Unsupported probability output shape: {prob.shape}")

        positive_prob = np.asarray(positive_prob, dtype=float)
        if not np.all(np.isfinite(positive_prob)):
            raise InvalidFinalMetricsError("Final evaluation probabilities contain NaN/Inf values")

        y_true.append(yb.numpy())
        y_prob.append(positive_prob)
        step_idx += 1

    if not y_true:
        raise InvalidFinalMetricsError("Final evaluation dataset is empty or produced zero batches")

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    mean_loss = float(total_loss / max(total_examples, 1))

    return y_true, y_prob, mean_loss


def write_test_row_everywhere(seed_root: str, tag: str, row: Dict[str, Any]):
    from finetune_ga.infra.io_utils import append_jsonl, atomic_write_json
    shared_dir = prepare_output_paths(
        seed_root, tag, row["backbone"])["shared_dir"]
    tag_paths = prepare_output_paths(seed_root, tag, row["backbone"])
    append_jsonl(os.path.join(shared_dir, "test_results_shared.jsonl"), row)
    append_jsonl(os.path.join(tag_paths["exp_dir"], "test_results.jsonl"), row)
    append_jsonl(os.path.join(
        tag_paths["per_model_dir"], "test_results.jsonl"), row)


def write_selection_manifest(seed_root: str, tag: str, selected_rows: List[Dict[str, Any]], cfg: Dict[str, Any], selected_budget: str | None, selected_backbones: Dict[str, Dict[str, Any]] | None = None, selection_source_seed: int | None = None, active_seed: int | None = None):
    exp_dir = prepare_output_paths(seed_root, tag)["exp_dir"]
    out_path = os.path.join(exp_dir, "selected_for_final_retrain_test.json")
    manifest = _build_selection_manifest(tag, selected_budget, cfg, selected_rows, selected_backbones, selection_source_seed=selection_source_seed, active_seed=active_seed)
    atomic_write_json(out_path, manifest, validate_finite=True)


def _build_selection_manifest(tag: str, selected_budget: str | None, cfg: Dict[str, Any], selected_rows: List[Dict[str, Any]], selected_backbones: Dict[str, Dict[str, Any]] | None = None, selection_source_seed: int | None = None, active_seed: int | None = None) -> Dict[str, Any]:
    policy = cfg.get('test_selection_policy', {}) if isinstance(cfg, dict) else {}
    return {
        'tag': tag,
        'protocol': {
            'test_protocol_id': build_test_protocol_id(cfg),
            'selected_budget': selected_budget,
            'selection_protocol': {
                'name': FINAL_SELECTION_PROTOCOL_NAME,
            },
            'summary_selection_objectives': list(policy.get('summary_selection_objectives', FINAL_SELECTION_OBJECTIVES)),
            'summary_selection_runtime_metrics': list(policy.get('summary_selection_runtime_metrics', ['mean_val_auc', 'mean_time_s_per_run', 'mean_trainable_params_m'])),
            'backbone_selection_objectives': list(policy.get('backbone_selection_objectives', FINAL_SELECTION_OBJECTIVES)),
            'backbone_selection_runtime_metrics': list(policy.get('backbone_selection_runtime_metrics', FINAL_SELECTION_RUNTIME_METRICS)),
            'final_retrain_mode': cfg.get('final_retrain_mode', 'train_plus_val_fixed_epochs'),
            'final_retrain_uses_validation': bool(cfg.get('final_retrain_uses_validation', False)),
        },
        'selection': {
            'summary_topk_strategy': 'pareto_fronts_then_ideal_distance',
            'runtime_selection_strategy': 'pareto_front_then_ideal_distance',
            'objective_direction': 'minimize',
            'selection_source_seed': int(selection_source_seed if selection_source_seed is not None else get_selection_source_seed(cfg)),
            'final_evaluation_seed': int(active_seed),
        },
        'candidates': {
            'selected_rows': selected_rows,
            'selected_backbones': selected_backbones or {},
        },
        'artifacts': {
            'manifest_file': 'selected_for_final_retrain_test.json',
        },
    }


def _get_backbone_selection_policy(cfg: Dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    policy = cfg.get('test_selection_policy', {}) if isinstance(cfg, dict) else {}

    objectives = tuple(policy.get('backbone_selection_objectives', FINAL_SELECTION_OBJECTIVES))
    runtime_metrics = tuple(policy.get('backbone_selection_runtime_metrics', FINAL_SELECTION_RUNTIME_METRICS))

    if len(objectives) != 3:
        raise ValueError(
            'test_selection_policy.backbone_selection_objectives must contain exactly 3 objectives'
        )
    if len(runtime_metrics) != 3:
        raise ValueError(
            'test_selection_policy.backbone_selection_runtime_metrics must contain exactly 3 values: '
            '[auc_metric, time_metric, params_metric]'
        )

    return objectives, runtime_metrics


def _prepare_runtime_selection_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if str(row.get('status', 'ok')).strip().lower() != 'ok':
        raise ValueError('runtime selection requires status=ok')

    objectives, runtime_metrics = _get_backbone_selection_policy(cfg)
    auc_key, time_key, params_key = runtime_metrics

    row_copy = prepare_selection_row(
        row,
        auc_key=auc_key,
        time_key=time_key,
        params_key=params_key,
        runtime_metrics=runtime_metrics,
    )
    if not candidate_has_finite_objectives(row_copy.get('selection_objectives', {}), objectives):
        raise ValueError('invalid runtime selection objectives')

    row_copy['backbone_selection_objectives'] = list(objectives)
    row_copy['backbone_selection_runtime_metrics'] = list(runtime_metrics)
    return row_copy


def select_single_backbone_run(all_runs: List[Dict[str, Any]], genome_id: str, selected_budget: str, cfg: Dict[str, Any]) -> Tuple[str | None, Dict[str, Any] | None]:
    objectives, runtime_metrics = _get_backbone_selection_policy(cfg)
    auc_key, time_key, params_key = runtime_metrics

    candidates: List[Dict[str, Any]] = []
    for r in all_runs:
        if r.get('genome_id') != genome_id or r.get('budget') != selected_budget:
            continue
        try:
            candidates.append(_prepare_runtime_selection_row(r, cfg))
        except (TypeError, ValueError):
            continue

    if not candidates:
        return None, None

    best, front = select_by_ideal_point(
        candidates,
        objectives,
        tie_break_auc_key=auc_key,
        tie_break_name_key='backbone',
        stable_id_key='genome_id',
    )
    if not best:
        return None, None

    selected = dict(best)
    selected['pareto_candidate_count'] = len(front)
    selected['pareto_candidates'] = [
        {
            'backbone': row.get('backbone'),
            'rep': row.get('rep'),
            auc_key: row.get(auc_key),
            time_key: row.get(time_key),
            params_key: row.get(params_key),
            'selection_objectives': dict(row.get('selection_objectives', {})),
            'selection_normalized_objectives': dict(row.get('selection_normalized_objectives', {})),
            'ideal_distance': row.get('ideal_distance'),
        }
        for row in front
    ]
    return str(selected.get('backbone')), selected

def _require_training_runtime():
    if runtime.get_tf_module_or_none() is None:
        raise ModuleNotFoundError('TensorFlow is required to run')

    from finetune_ga.infra.runtime import STRATEGY, SoftmaxBinaryAUC
    from finetune_ga.infra.run_utils import seed_everything
    from finetune_ga.infra.io_utils import append_jsonl, atomic_write_json
    from finetune_ga.core.metrics import (
        compute_binary_classification_metrics,
        compute_complexity_score,
    )
    from finetune_ga.core.final_artifacts import make_run_artifact_paths, persist_final_artifacts
    from finetune_ga.core.training import (
        ensure_head_trainable_backbone_frozen,
        set_finetune_last_n_layers,
        trainable_params_m,
        fallback_attempts_for_backbone,
        is_resource_exhaustion_error,
        clear_tf_memory,
    )
    from finetune_ga.infra.math_utils import full_steps
    from finetune_ga.models.backbone import build_model
    from finetune_ga.data.loader import load_final_retrain_data
    from finetune_ga.data.preprocessing import apply_preprocessing, clear_augmenter_cache
    return {
        'STRATEGY': STRATEGY,
        'seed_everything': seed_everything,
        'append_jsonl': append_jsonl,
        'SoftmaxBinaryAUC': SoftmaxBinaryAUC,
        'compute_binary_classification_metrics': compute_binary_classification_metrics,
        'compute_complexity_score': compute_complexity_score,
        'ensure_head_trainable_backbone_frozen': ensure_head_trainable_backbone_frozen,
        'set_finetune_last_n_layers': set_finetune_last_n_layers,
        'trainable_params_m': trainable_params_m,
        'fallback_attempts_for_backbone': fallback_attempts_for_backbone,
        'is_resource_exhaustion_error': is_resource_exhaustion_error,
        'clear_tf_memory': clear_tf_memory,
        'full_steps': full_steps,
        'build_model': build_model,
        'load_final_retrain_data': load_final_retrain_data,
        'apply_preprocessing': apply_preprocessing,
        'clear_augmenter_cache': clear_augmenter_cache,
        'make_run_artifact_paths': make_run_artifact_paths,
        'persist_final_artifacts': persist_final_artifacts,
    }




def _final_failure_test_row(*, protocol_id: str, selection_rank: int, active_seed: int, selection_seed: int, tag: str,
                            selected_budget: str, gid: str, bb: str, r: Dict[str, Any], status: str,
                            final_status: str, message: str, effective: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "test_protocol_id": protocol_id,
        "selection_rank": int(selection_rank),
        "seed": int(active_seed),
        "active_seed": int(active_seed),
        "search_seed": int(selection_seed),
        "selection_source_seed": int(selection_seed),
        "final_train_seed": int(active_seed),
        "final_eval_seed": int(active_seed),
        "source_tag": tag,
        "tag": tag,
        "mode": "final_retrain_test",
        "status": status,
        "final_status": final_status,
        "final_error": message,
        "selection_budget": selected_budget,
        "budget": selected_budget,
        "evaluation_protocol": "final_retrain_train_plus_val_fixed_epochs_then_test",
        "backbone_selection_protocol": FINAL_SELECTION_PROTOCOL_NAME,
        "final_retrain_dataset": "train_plus_val",
        "final_retrain_used_validation": False,
        "final_retrain_selection_used_validation_only": True,
        "genome_id": gid,
        "backbone": bb,
        "rep": r.get("rep"),
        "target_size": effective.get("img_size"),
        "test_loss": None,
        "test_auc": None,
        "test_pr_auc": None,
        "test_accuracy": None,
        "test_f1": None,
        "test_mcc": None,
        "trainable_params_m": None,
        "complexity_score": None,
        "elapsed_test_train_s": 0.0,
        "elapsed_test_eval_s": 0.0,
        "elapsed_total_test_pipeline_s": 0.0,
        "stage1_time_s": 0.0,
        "stage2_time_s": 0.0,
        "epochs_stage1_ran": 0,
        "epochs_stage2_ran": 0,
        "final_effective_batch_size": effective.get("batch_size"),
        "final_effective_img_size": effective.get("img_size"),
        "final_effective_fine_tune_layers": effective.get("fine_tune_layers"),
        "final_fallback_used": bool(effective.get("fallback_used", False)),
        "lr1": r.get("lr1"), "lr2": r.get("lr2"), "n_last_layers": r.get("n_last_layers"),
        "batch_size": r.get("batch_size"), "dense_units": r.get("dense_units"),
        "dropout": r.get("dropout"), "l2_weight": r.get("l2_weight"),
    }


def _run_final_retrain_test_with_fallbacks(cfg: Dict[str, Any], *, active_seed: int, selection_seed: int,
                                           protocol_id: str, tag: str, selected_budget: str,
                                           selection_rank: int, gid: str, bb: str, r: Dict[str, Any],
                                           seed_root: str) -> Dict[str, Any]:
    rt = _require_training_runtime()
    STRATEGY = rt['STRATEGY']
    SoftmaxBinaryAUC = rt['SoftmaxBinaryAUC']
    compute_binary_classification_metrics = rt['compute_binary_classification_metrics']
    compute_complexity_score = rt['compute_complexity_score']
    ensure_head_trainable_backbone_frozen = rt['ensure_head_trainable_backbone_frozen']
    set_finetune_last_n_layers = rt['set_finetune_last_n_layers']
    trainable_params_m = rt['trainable_params_m']
    full_steps = rt['full_steps']
    build_model = rt['build_model']
    load_final_retrain_data = rt['load_final_retrain_data']
    apply_preprocessing = rt['apply_preprocessing']
    clear_augmenter_cache = rt['clear_augmenter_cache']
    fallback_attempts_for_backbone = rt['fallback_attempts_for_backbone']
    is_resource_exhaustion_error = rt['is_resource_exhaustion_error']
    clear_tf_memory = rt['clear_tf_memory']
    make_run_artifact_paths = rt['make_run_artifact_paths']
    persist_final_artifacts = rt['persist_final_artifacts']

    lr1, lr2 = float(r["lr1"]), float(r["lr2"])
    n_last = int(r["n_last_layers"])
    bs = int(r["batch_size"])
    dense = int(r["dense_units"])
    drop = float(r["dropout"])
    l2w = float(r["l2_weight"])
    base_size = 299 if str(bb).lower() == "inception" else FINAL_IMG_SIZE
    attempts = fallback_attempts_for_backbone(bb, bs, base_size, n_last)
    if not attempts:
        attempts = [{"batch_size": bs, "target_size": base_size, "n_last_layers": n_last}]
    last_error = None
    last_effective = {"batch_size": bs, "img_size": base_size, "fine_tune_layers": n_last, "fallback_used": False}

    for attempt_index, attempt in enumerate(attempts):
        eff_bs = int(attempt.get("batch_size", bs))
        requested_size = int(attempt.get("target_size", base_size))
        eff_layers = int(attempt.get("n_last_layers", n_last))
        last_effective = {"batch_size": eff_bs, "img_size": requested_size, "fine_tune_layers": eff_layers, "fallback_used": attempt_index > 0}
        try:
            try:
                tf.keras.backend.clear_session()
            except (RuntimeError, ValueError, AttributeError):
                pass
            import gc
            gc.collect()
            if attempt_index > 0:
                print(f"[FINAL-OOM-FALLBACK] retry={attempt_index}/{len(attempts)-1} backbone={bb} img={requested_size} bs={eff_bs} unfreeze_last={eff_layers}")
            with STRATEGY.scope():
                model, target_size = build_model(bb, int(cfg["num_classes"]), requested_size, dense, drop, l2w)
            model.build((None, target_size, target_size, 3))
            last_effective["img_size"] = int(target_size)
            clear_augmenter_cache()
            trainval_ds, test_ds = load_final_retrain_data(
                target_size=target_size, batch_size=eff_bs, repeat_train=True,
                seed=int(active_seed), model_name=bb, include_test=True)
            trainval_ds = apply_preprocessing(
                trainval_ds, model_name=bb, augment=True,
                aug_rotation=float(r["aug_rotation"]),
                aug_zoom=float(r["aug_zoom"]), aug_contrast=float(r["aug_contrast"])
            )
            test_ds = apply_preprocessing(test_ds, model_name=bb, augment=False)
            counts = get_dataset_counts(refresh=True)
            trainval_count = int(counts['train']) + int(counts['val'])
            sp = full_steps(trainval_count, eff_bs)
            def build_final_metrics():
                # Keras metrics keep internal state; build fresh instances for each compile.
                return [
                    SoftmaxBinaryAUC(name="auc", curve="ROC"),
                    SoftmaxBinaryAUC(name="pr_auc", curve="PR"),
                    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                ]

            t0 = time.perf_counter()
            with STRATEGY.scope():
                backbone_model = ensure_head_trainable_backbone_frozen(model)
                model.compile(optimizer=tf.keras.optimizers.Adam(lr1, clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=build_final_metrics())
            stage1_start = time.perf_counter()
            h1 = model.fit(trainval_ds, epochs=int(cfg.get("test_e1", 11)), steps_per_epoch=sp, callbacks=[], verbose=2)
            stage1_time = time.perf_counter() - stage1_start

            with STRATEGY.scope():
                set_finetune_last_n_layers(backbone_model, eff_layers)
                model.compile(optimizer=tf.keras.optimizers.Adam(lr2, clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=build_final_metrics())
            stage2_start = time.perf_counter()
            h2 = model.fit(trainval_ds, epochs=int(cfg.get("test_e2", 5)), steps_per_epoch=sp, callbacks=[], verbose=2)
            stage2_time = time.perf_counter() - stage2_start
            train_elapsed = time.perf_counter() - t0
            eval_start = time.perf_counter()
            y_true, y_prob, test_loss = collect_probs_and_loss(model, test_ds)
            eval_time = time.perf_counter() - eval_start
            extra = compute_binary_classification_metrics(y_true, y_prob)
            trainable_params_m_value = trainable_params_m(model)
            artifact_paths = make_run_artifact_paths(
                seed_root, f"{tag}_final_test", gid, bb, selected_budget, int(r.get("rep", 0) or 0)
            )
            artifact_payload = persist_final_artifacts(
                model, h1, h2, artifact_paths, bb, selected_budget, int(r.get("rep", 0) or 0)
            )
            return {
                "test_protocol_id": protocol_id,
                "selection_rank": int(selection_rank),
                "seed": int(active_seed),
                "active_seed": int(active_seed),
                "search_seed": int(selection_seed),
                "selection_source_seed": int(selection_seed),
                "final_train_seed": int(active_seed),
                "final_eval_seed": int(active_seed),
                "source_tag": tag,
                "tag": tag,
                "mode": "final_retrain_test",
                "status": "ok",
                "final_status": "ok",
                "selection_budget": selected_budget,
                "budget": selected_budget,
                "evaluation_protocol": "final_retrain_train_plus_val_fixed_epochs_then_test",
                "backbone_selection_protocol": FINAL_SELECTION_PROTOCOL_NAME,
                "final_retrain_dataset": "train_plus_val",
                "final_retrain_used_validation": False,
                "final_retrain_selection_used_validation_only": True,
                "genome_id": gid,
                "backbone": bb,
                "rep": r.get("rep"),
                "target_size": int(target_size),
                "test_loss": _json_safe_number(float(test_loss)),
                **{f"test_{k}": _json_safe_number(v) for k, v in extra.items()},
                "trainable_params_m": _json_safe_number(float(trainable_params_m_value)),
                "complexity_score": _json_safe_number(compute_complexity_score(trainable_params_m_value, train_elapsed + eval_time, int(target_size), cfg)),
                "elapsed_test_train_s": _json_safe_number(float(train_elapsed)),
                "elapsed_test_eval_s": _json_safe_number(float(eval_time)),
                "elapsed_total_test_pipeline_s": _json_safe_number(float(train_elapsed + eval_time)),
                "stage1_time_s": _json_safe_number(float(stage1_time)),
                "stage2_time_s": _json_safe_number(float(stage2_time)),
                "epochs_stage1_ran": len(h1.history.get("loss", [])),
                "epochs_stage2_ran": len(h2.history.get("loss", [])),
                "final_retrain_schedule": {
                    "stage1_epochs": int(cfg.get("test_e1", 11)),
                    "stage2_epochs": int(cfg.get("test_e2", 5)),
                    "early_stopping": False,
                    "reduce_lr_on_plateau": False,
                },
                "final_effective_batch_size": int(eff_bs),
                "final_effective_img_size": int(target_size),
                "final_effective_fine_tune_layers": int(eff_layers),
                "final_fallback_used": bool(attempt_index > 0),
                "lr1": lr1, "lr2": lr2, "n_last_layers": n_last,
                "batch_size": bs, "dense_units": dense, "dropout": drop, "l2_weight": l2w,
                **artifact_payload,
            }
        except InvalidFinalMetricsError as exc:
            # Invalid numeric final predictions/metrics are not recoverable by trying a
            # smaller batch/image, but they should still be exported as a
            # controlled failed row rather than aborting the full final sweep.
            return _final_failure_test_row(
                protocol_id=protocol_id, selection_rank=selection_rank, active_seed=active_seed,
                selection_seed=selection_seed, tag=tag, selected_budget=selected_budget, gid=gid, bb=bb, r=r,
                status="failed_invalid_final_metrics", final_status="invalid_final_metrics",
                message=f"{type(exc).__name__}: {exc}", effective=last_effective,
            )
        except (MemoryError, RuntimeError, OSError, tf.errors.OpError) as exc:
            if not is_resource_exhaustion_error(exc):
                raise
            last_error = exc
            clear_tf_memory()
            clear_augmenter_cache()
            print(f"[FINAL-OOM-FALLBACK] OOM detected for {bb} attempt={attempt_index}: {type(exc).__name__}: {exc}")
    return _final_failure_test_row(
        protocol_id=protocol_id, selection_rank=selection_rank, active_seed=active_seed,
        selection_seed=selection_seed, tag=tag, selected_budget=selected_budget, gid=gid, bb=bb, r=r,
        status="failed_resource_exhausted", final_status="resource_exhausted_after_fallbacks",
        message=f"{type(last_error).__name__}: {last_error}" if last_error is not None else "resource_exhausted_after_fallbacks",
        effective=last_effective,
    )

def evaluate_for_seed(cfg: Dict[str, Any], active_seed: int, existing_done: set, selection_seed: int):
    rt = _require_training_runtime()
    STRATEGY = rt['STRATEGY']
    seed_everything = rt['seed_everything']
    SoftmaxBinaryAUC = rt['SoftmaxBinaryAUC']
    compute_binary_classification_metrics = rt['compute_binary_classification_metrics']
    compute_complexity_score = rt['compute_complexity_score']
    ensure_head_trainable_backbone_frozen = rt['ensure_head_trainable_backbone_frozen']
    set_finetune_last_n_layers = rt['set_finetune_last_n_layers']
    trainable_params_m = rt['trainable_params_m']
    full_steps = rt['full_steps']
    build_model = rt['build_model']
    load_final_retrain_data = rt['load_final_retrain_data']
    apply_preprocessing = rt['apply_preprocessing']
    clear_augmenter_cache = rt['clear_augmenter_cache']

    seed_root = get_seeded_root_dir(cfg["out_dir"], int(active_seed), get_final_seeds(cfg))
    seed_everything(int(active_seed))
    selection_root = seed_root_for(cfg["out_dir"], selection_seed, get_search_seeds(cfg))

    candidate_tags = resolve_test_candidate_tags(cfg)
    top_per_tag = int(cfg.get("test_topk_per_tag", 1))
    protocol_id = build_test_protocol_id(cfg)

    for tag in candidate_tags:
        summary_path = os.path.join(selection_root, tag, "genome_summary.jsonl")
        runs_path = os.path.join(selection_root, tag, "runs.jsonl")
        if not os.path.exists(summary_path) or not os.path.exists(runs_path):
            continue
        preferred_budget = cfg['budgets'][-1]['name']
        top_entries, selected_budget = pick_topk_from_summary(
            summary_path, cfg=cfg, preferred_budget_name=preferred_budget, k=top_per_tag)
        if not selected_budget:
            continue
        print(f"[{tag}][seed={active_seed}] selected {len(top_entries)} candidate(s) from budget {selected_budget} under test protocol {protocol_id}")
        all_runs = read_jsonl(runs_path)
        selected_backbones: Dict[str, Dict[str, Any]] = {}
        for selection_rank, entry in enumerate(top_entries, start=1):
            gid = entry["genome_id"]
            selected_bb, selected_run = select_single_backbone_run(
                all_runs, genome_id=gid, selected_budget=selected_budget, cfg=cfg
            )
            if selected_bb is None or selected_run is None:
                print(
                    f"[skip missing validation winner] seed={active_seed} tag={tag} genome={gid}")
                continue
            selected_backbones[gid] = {
                "backbone": selected_bb,
                "selection_metrics": {
                    "objective_direction": "minimize",
                    "objectives": cfg.get("test_selection_policy", {}).get("backbone_selection_objectives", ["auc_loss", "time_s", "trainable_params_m"]),
                    "runtime_metrics": cfg.get("test_selection_policy", {}).get("backbone_selection_runtime_metrics", ["best_val_auc", "time_s", "trainable_params_m"]),
                    "ideal_distance": selected_run.get("ideal_distance"),
                },
                "selected_run": dict(selected_run),
            }
        write_selection_manifest(
            seed_root,
            tag,
            top_entries,
            cfg=cfg,
            selected_budget=selected_budget,
            selected_backbones=selected_backbones,
            selection_source_seed=selection_seed,
            active_seed=int(active_seed),
        )

        for selection_rank, entry in enumerate(top_entries, start=1):
            gid = entry["genome_id"]
            selected = selected_backbones.get(gid)
            if not selected:
                continue
            bb = str(selected["backbone"])
            r = dict(selected["selected_run"])
            done_key = (active_seed, tag, gid, bb, selected_budget,
                        int(selection_rank), protocol_id)
            if done_key in existing_done:
                print(
                    f"[skip existing test] seed={active_seed} tag={tag} genome={gid} backbone={bb}")
                continue
            print(
                f"\n=== TEST TRAIN: seed={active_seed} tag={tag} genome={gid} backbone={bb} ===")
            lr1, lr2 = float(r["lr1"]), float(r["lr2"])
            n_last = int(r["n_last_layers"])
            bs = int(r["batch_size"])
            dense = int(r["dense_units"])
            drop = float(r["dropout"])
            l2w = float(r["l2_weight"])

            row = _run_final_retrain_test_with_fallbacks(
                cfg,
                active_seed=int(active_seed),
                selection_seed=int(selection_seed),
                protocol_id=protocol_id,
                tag=tag,
                selected_budget=selected_budget,
                selection_rank=int(selection_rank),
                gid=gid,
                bb=bb,
                r=r,
                seed_root=seed_root,
            )
            write_test_row_everywhere(seed_root, tag, row)
            existing_done.add(done_key)


def main():
    cfg = load_config()
    existing_done = set()
    protocol_id = build_test_protocol_id(cfg)
    final_seeds = get_final_seeds(cfg)
    selection_seed = get_selection_source_seed(cfg)
    for seed in final_seeds:
        seed_root = seed_root_for(cfg["out_dir"], int(seed), final_seeds)
        shared_path = os.path.join(
            seed_root, "shared", "test_results_shared.jsonl")
        for r in read_jsonl(shared_path):
            if r.get("test_protocol_id") != protocol_id:
                continue
            source_tag = r.get("source_tag")
            genome_id = r.get("genome_id")
            backbone = r.get("backbone")
            selection_budget = r.get("selection_budget")
            if not source_tag or not genome_id or not backbone or not selection_budget:
                continue
            existing_done.add((
                _safe_int(r.get("final_eval_seed", r.get("seed", seed)), int(seed)),
                source_tag,
                genome_id,
                backbone,
                selection_budget,
                _safe_int(r.get("selection_rank", 1), 1),
                r.get("test_protocol_id"),
            ))

    for seed in final_seeds:
        evaluate_for_seed(cfg, int(seed), existing_done, selection_seed=selection_seed)

    print("\nDONE. Per-seed test results saved under <out_dir>/seed_x/shared/test_results_shared.jsonl and experiment folders.")


if __name__ == "__main__":
    main()
