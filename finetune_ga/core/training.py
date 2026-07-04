"""Training pipeline: v2 packaging/artifacts with fast data path and OOM fallback."""
from __future__ import annotations

import gc
import math
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from finetune_ga.infra import runtime
from finetune_ga.infra.math_utils import steps_safe, full_steps
from finetune_ga.infra.run_utils import seed_everything, stable_crc32
from finetune_ga.core.genome import Genome, genome_to_dict, genome_fingerprint, hyper_for_backbone
from finetune_ga.infra.config import IMG_SIZE, SEARCH_IMG_SIZE, FINAL_IMG_SIZE, get_dataset_counts
from finetune_ga.core.final_artifacts import (
    make_run_artifact_paths,
    build_final_stage1_callbacks,
    build_final_stage2_callbacks,
    persist_final_artifacts,
)

# Backward-compatible module aliases for tests and targeted monkeypatching.
STRATEGY = runtime.STRATEGY
tf = runtime.tf
_require_tensorflow = runtime._require_tensorflow
SoftmaxBinaryAUC = runtime.SoftmaxBinaryAUC


def _fit_verbose(mode: str) -> int:
    raw = os.environ.get('TRAIN_FIT_VERBOSE', '').strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    mode_normalized = str(mode or '').strip().lower()
    if mode_normalized == 'final':
        return 1
    if mode_normalized == 'search':
        return 0
    return 0


def _training_log_mode(mode: str) -> str:
    raw = os.environ.get('TRAIN_LOG_MODE', '').strip().lower()
    if raw:
        if raw in {'off', 'quiet', 'none'}:
            return 'off'
        if raw in {'minimal', 'concise'}:
            return 'minimal'
        if raw == 'epoch':
            return 'epoch'
        return 'minimal'
    mode_normalized = str(mode or '').strip().lower()
    if mode_normalized == 'final':
        return 'epoch'
    if mode_normalized == 'search':
        return 'minimal'
    return 'minimal'


def _safe_metric(value: Any, default: float) -> float:
    """Return a finite float for metrics that may contain NaN/inf from Keras history."""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return 'n/a'
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_f):
        return 'n/a'
    return f"{value_f:.4f}"


def _make_epoch_logger(mode: str, stage_name: str, total_epochs: int, backbone: str, genome_id: str, budget_name: str, rep: int):
    if runtime.get_tf_module_or_none() is None or _training_log_mode(mode) == 'off':
        return None

    class EpochMetricLogger(runtime.get_tf_module_or_none().keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"[FIT] genome={genome_id} | bb={backbone} | budget={budget_name} | rep={rep} | "
                f"stage={stage_name} | epoch={epoch + 1}/{total_epochs} | "
                f"loss={_format_metric_value(logs.get('loss'))} | "
                f"auc={_format_metric_value(logs.get('auc'))} | "
                f"val_loss={_format_metric_value(logs.get('val_loss'))} | "
                f"val_auc={_format_metric_value(logs.get('val_auc'))}"
            )

    return EpochMetricLogger()


def best_epoch_by_auc(history, *, uses_validation: bool = True) -> Tuple[float, float, int]:
    auc_key = 'val_auc' if uses_validation else 'auc'
    loss_key = 'val_loss' if uses_validation else 'loss'
    auc = history.history.get(auc_key, [])
    loss = history.history.get(loss_key, [])
    if not auc:
        return 0.0, 1e9, -1

    finite_auc: List[Tuple[float, int]] = []
    for idx, raw_auc in enumerate(auc):
        auc_val = _safe_metric(raw_auc, float('nan'))
        if math.isfinite(auc_val):
            finite_auc.append((auc_val, idx))

    if not finite_auc:
        return 0.0, 1e9, -1

    best_auc, idx = max(finite_auc, key=lambda item: item[0])
    best_loss = _safe_metric(loss[idx], 1e9) if idx < len(loss) else 1e9
    return float(best_auc), float(best_loss), int(idx)


def choose_better_stage(s1: Tuple[float, float, int], s2: Tuple[float, float, int]) -> Tuple[float, float, int, int]:
    if s2[0] > s1[0]:
        return s2[0], s2[1], s2[2], 2
    if s2[0] < s1[0]:
        return s1[0], s1[1], s1[2], 1
    if s2[1] < s1[1]:
        return s2[0], s2[1], s2[2], 2
    return s1[0], s1[1], s1[2], 1


def _save_stage_snapshot(
    model,
    path: Optional[str],
    *,
    backbone: str,
    budget_name: str,
    rep: int,
    stage: int,
) -> bool:
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model.save_weights(path)
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f'Could not save stage{stage} snapshot weights for {backbone}/{budget_name}/rep{rep}: '
            f'{type(exc).__name__}: {exc}'
        )
        return False


def _restore_stage_snapshot(
    model,
    path: Optional[str],
    *,
    backbone: str,
    budget_name: str,
    rep: int,
    stage: int,
) -> bool:
    if not path:
        return False
    try:
        model.load_weights(path)
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f'Could not restore stage{stage} snapshot weights for {backbone}/{budget_name}/rep{rep}: '
            f'{type(exc).__name__}: {exc}. Exported model may not match best_from_stage.'
        )
        return False


def _delete_stage_snapshot(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        warnings.warn(f'Could not delete temporary stage snapshot {path}: {type(exc).__name__}: {exc}')


def trainable_params_m(model) -> float:
    n = 0
    for v in model.trainable_variables:
        shape = v.shape
        if shape is None:
            continue
        n += int(np.prod(shape))
    return float(n) / 1e6


def ensure_head_trainable_backbone_frozen(model):
    backbone = None
    tf_module = getattr(tf, 'keras', None)
    if tf_module is None:
        _require_tensorflow()
        tf_module = tf.keras
    if len(model.layers) > 0 and isinstance(model.layers[0], tf_module.Model):
        backbone = model.layers[0]
    else:
        for layer in model.layers:
            if isinstance(layer, tf_module.Model):
                backbone = layer
                break

    if backbone is None:
        raise RuntimeError('Backbone not found in Sequential.')
    backbone.trainable = False
    return backbone


def set_finetune_last_n_layers(backbone, n_last_layers: int) -> None:
    total_layers = len(backbone.layers)
    n_last_layers = int(max(0, min(int(n_last_layers), total_layers)))

    if n_last_layers <= 0:
        backbone.trainable = False
        return

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-n_last_layers:]:
        tf_module = getattr(tf, 'keras', None)
        if tf_module is None:
            _require_tensorflow()
            tf_module = tf.keras
        if not isinstance(layer, tf_module.layers.BatchNormalization):
            layer.trainable = True


def print_run_header(
    g,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    lr1: float,
    lr2: float,
    n_last_layers: int,
    sp: int,
    vp: int,
    target_size: int,
    effective_batch_size: Optional[int] = None
) -> None:
    shown_bs = int(effective_batch_size if effective_batch_size is not None else g.batch_size)
    print(
        f"\n[RUN] genome={g.genome_id} | bb={backbone} | budget={budget['name']} | rep={rep}\n"
        f"      img={target_size} | bs={shown_bs} | sp={sp} | vp={vp}\n"
        f"      lr1={lr1:.2e} | lr2={lr2:.2e} | unfreeze_last={n_last_layers} | dense={g.dense_units} | drop={g.dropout:.2f} | l2={g.l2_weight:.1e}\n"
        f"      aug: rot={g.aug_rotation:.2f} zoom={g.aug_zoom:.2f} con={g.aug_contrast:.2f}\n"
        f"      replicas={getattr(STRATEGY, 'num_replicas_in_sync', 1)} | policy={runtime.get_tf_module_or_none().keras.mixed_precision.global_policy()}"
    )


def clear_tf_memory() -> None:
    if runtime.get_tf_module_or_none() is None:
        return
    try:
        # Keras preprocessing layers cached in finetune_ga.data.preprocessing
        # belong to the current Keras graph/session. Drop them before and after
        # clear_session() so a later run never reuses stale Random* layers.
        try:
            from finetune_ga.data.preprocessing import clear_augmenter_cache
            clear_augmenter_cache()
        except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
            warnings.warn(f"Could not clear augmenter cache before TF session reset: {type(exc).__name__}: {exc}")

        tf.keras.backend.clear_session()

        try:
            from finetune_ga.data.preprocessing import clear_augmenter_cache
            clear_augmenter_cache()
        except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
            warnings.warn(f"Could not clear augmenter cache after TF session reset: {type(exc).__name__}: {exc}")

        # clear_session() can reset Keras global dtype policy to float32.
        # Re-apply the configured precision policy so subsequent models keep
        # the same mixed-precision behavior as the first run in this process.
        runtime.apply_precision_policy()
    except (RuntimeError, ValueError, AttributeError) as exc:
        warnings.warn(f"Could not clear TF session cleanly: {type(exc).__name__}: {exc}")
    gc.collect()


def is_resource_exhaustion_error(exc: BaseException) -> bool:
    tf_module = runtime.get_tf_module_or_none()
    if tf_module is not None and isinstance(exc, tf_module.errors.ResourceExhaustedError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    needles = [
        'resourceexhausted',
        'resource exhausted',
        'oom',
        'out of memory',
        'failed to allocate memory',
        'cuda_error_out_of_memory',
        'cudnn_status_alloc_failed',
        'allocator gpu',
        'blas xgemm launch failed',
    ]
    return any(n in msg for n in needles) or type(exc).__name__ == 'ResourceExhaustedError'


def fallback_attempts_for_backbone(backbone: str, batch_size: int, target_size: int, n_last_layers: int) -> List[Dict[str, Any]]:
    backbone = str(backbone).lower()
    min_size = {'inception': 75}.get(backbone, 96)

    preferred_sizes = [target_size]
    if target_size >= 299:
        preferred_sizes += [256, 224, 192, 160, 128]
    elif target_size >= 224:
        preferred_sizes += [192, 160, 128]
    elif target_size >= 160:
        preferred_sizes += [160, 128, 96]
    else:
        preferred_sizes += [128, 96]

    size_candidates: List[int] = []
    for s in preferred_sizes:
        s = int(max(s, min_size))
        if s not in size_candidates:
            size_candidates.append(s)

    batch_candidates: List[int] = []
    b = int(max(1, batch_size))
    while b >= 1:
        if b not in batch_candidates:
            batch_candidates.append(b)
        if b == 1:
            break
        b = max(1, b // 2)

    attempts: List[Dict[str, Any]] = []
    seen: set = set()
    for size in size_candidates:
        for b in batch_candidates:
            for n_last in [int(n_last_layers), 0]:
                key = (size, b, n_last)
                if key in seen:
                    continue
                seen.add(key)
                attempts.append({
                    'target_size': int(size),
                    'batch_size': int(b),
                    'n_last_layers': int(max(0, n_last)),
                })
    return attempts


def make_infeasible_result(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    reason: str,
    *,
    error: Optional[BaseException] = None,
    gen_idx: int = 0,
    batch_size: Optional[int] = None,
    target_size: Optional[int] = None,
    active_seed: int = 0,
    mode: Optional[str] = None,
    attempt_index: int = 0,
    used_oom_fallback: bool = False,
    auc_thresholds: Optional[int] = None,
    compute_pr_auc: bool = False,
    min_steps_fraction: float = 1.0,
    model_export_ok: bool = False,
    history_plot_ok: bool = False,
    n_last_layers: Optional[int] = None,
) -> Dict[str, Any]:
    fp = genome_fingerprint(g, cfg['model_names'])
    seed = (
        stable_crc32(fp)
        ^ stable_crc32(backbone)
        ^ stable_crc32(budget['name'])
        ^ (rep * 1337)
        ^ int(active_seed)
    ) & 0xFFFFFFFF
    target_size = int(target_size if target_size is not None else (SEARCH_IMG_SIZE if str(backbone).lower() != 'inception' else 299))
    batch_size = int(batch_size if batch_size is not None else g.batch_size)

    return {
        'gen': int(gen_idx),
        'genome_id': g.genome_id,
        'genome': genome_to_dict(g),
        'backbone': backbone,
        'budget': budget['name'],
        'budget_meta': budget,
        'rep': rep,
        'seed': int(seed),
        'active_seed': int(active_seed),
        'target_size': int(target_size),
        'lr1': 0.0,
        'lr2': 0.0,
        'n_last_layers': int(g.n_last_layers if n_last_layers is None else n_last_layers),
        'dense_units': int(g.dense_units),
        'dropout': _safe_metric(g.dropout, 0.0),
        'l2_weight': _safe_metric(g.l2_weight, 0.0),
        'batch_size': int(batch_size),
        'aug_rotation': _safe_metric(g.aug_rotation, 0.0),
        'aug_zoom': _safe_metric(g.aug_zoom, 0.0),
        'aug_contrast': _safe_metric(g.aug_contrast, 0.0),
        'best_val_auc': 0.0,
        'best_val_pr_auc': None,
        'best_val_pr_auc_computed': False,
        'best_val_accuracy': 0.0,
        'best_train_auc': 0.0,
        'loss_at_best_auc': 1e9,
        'best_auc_epoch': -1,
        'best_from_stage': 0,
        'time_s': 1e9,
        'stage1_time_s': 0.0,
        'stage2_time_s': 0.0,
        'trainable_params_m': 1e9,
        'steps_per_epoch': 0,
        'val_steps': 0,
        'epochs_stage1_ran': 0,
        'epochs_stage2_ran': 0,
        'best_epoch_stage1': -1,
        'best_epoch_stage2': -1,
        'num_replicas_in_sync': int(getattr(STRATEGY, 'num_replicas_in_sync', 1)),
        'mixed_precision_policy': (str(tf.keras.mixed_precision.global_policy()) if getattr(tf, 'keras', None) is not None else 'unavailable'),
        'checkpoint_dir': None,
        'best_weights_path': None,
        'best_model_path': None,
        'history_plot_path': None,
        'is_best_checkpoint_saved': False,
        'status': 'infeasible',
        'failure_reason': str(reason),
        'error_type': type(error).__name__ if error is not None else None,
        'error_message': str(error) if error is not None else None,
        'attempt_index': int(attempt_index),
        'used_oom_fallback': bool(used_oom_fallback),
        'mode': mode,
        'final_training_protocol': ('fixed_schedule_no_selection' if mode == 'final' else None),
        'auc_thresholds': (int(auc_thresholds) if auc_thresholds is not None else None),
        'compute_pr_auc': bool(compute_pr_auc),
        'min_steps_fraction': _safe_metric(min_steps_fraction, 0.0),
        'model_export_ok': bool(model_export_ok),
        'history_plot_ok': bool(history_plot_ok),
    }



def _build_search_stage1_callbacks(*, mode: str, budget: Dict[str, Any], backbone: str, genome_id: str, budget_name: str, rep: int):
    callbacks = [
        runtime.get_tf_module_or_none().keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=int(budget.get('es_patience_s1', 2)),
            restore_best_weights=True,
        )
    ]
    epoch_logger = _make_epoch_logger(mode, 'stage1', int(budget['e1']), backbone, genome_id, budget_name, rep)
    if epoch_logger is not None:
        callbacks.append(epoch_logger)
    return callbacks


def _build_search_stage2_callbacks(*, mode: str, budget: Dict[str, Any], backbone: str, genome_id: str, budget_name: str, rep: int):
    callbacks = [
        runtime.get_tf_module_or_none().keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=int(budget.get('es_patience_s2', 2)),
            restore_best_weights=True,
        ),
        runtime.get_tf_module_or_none().keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=1,
            min_lr=1e-8,
        ),
    ]
    epoch_logger = _make_epoch_logger(mode, 'stage2', int(budget['e2']), backbone, genome_id, budget_name, rep)
    if epoch_logger is not None:
        callbacks.append(epoch_logger)
    return callbacks




def _run_training_with_fallbacks(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    gen_idx: int,
    active_seed: int,
    mode: str,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    _require_tensorflow()
    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {'search', 'final'}:
        raise ValueError(f"Unsupported training mode '{mode}'. Expected 'search' or 'final'.")

    hyp = hyper_for_backbone(g, backbone, cfg['unfreeze_cap'])
    base_size = (FINAL_IMG_SIZE if mode_normalized == 'final' else SEARCH_IMG_SIZE) if backbone.lower() != 'inception' else 299
    attempts = fallback_attempts_for_backbone(backbone, g.batch_size, base_size, hyp['n_last_layers'])
    tf_module = runtime.get_tf_module_or_none()
    tf_exceptions = (tf_module.errors.OpError,) if tf_module is not None else ()
    last_error: Optional[BaseException] = None
    last_attempt = attempts[0] if attempts else {'batch_size': g.batch_size, 'target_size': base_size, 'n_last_layers': hyp['n_last_layers']}

    for attempt_index, attempt in enumerate(attempts):
        last_attempt = attempt
        try:
            if attempt_index > 0:
                print(
                    f"[OOM-FALLBACK] backbone={backbone} | retry={attempt_index}/{len(attempts)-1} | "
                    f"img={attempt['target_size']} | bs={attempt['batch_size']} | unfreeze_last={attempt['n_last_layers']}"
                )
            result = _train_one_backbone_single_attempt(
                cfg,
                g,
                backbone,
                budget,
                rep,
                override_batch_size=attempt['batch_size'],
                override_target_size=attempt['target_size'],
                override_n_last_layers=attempt['n_last_layers'],
                attempt_index=attempt_index,
                artifacts=artifacts,
                gen_idx=gen_idx,
                active_seed=int(active_seed),
                mode=mode_normalized,
            )
            result['attempt_index'] = int(attempt_index)
            result['used_oom_fallback'] = bool(attempt_index > 0)
            return result
        except (MemoryError, RuntimeError, OSError, *tf_exceptions) as exc:
            if not is_resource_exhaustion_error(exc):
                raise
            last_error = exc
            clear_tf_memory()
            print(f"[OOM-FALLBACK] OOM detected for {backbone} on attempt {attempt_index}: {type(exc).__name__}: {exc}")

    search_runtime = cfg.get('search_runtime', {}) or {}
    return make_infeasible_result(
        cfg,
        g,
        backbone,
        budget,
        rep,
        reason='resource_exhausted_after_fallbacks',
        error=last_error,
        gen_idx=gen_idx,
        batch_size=int(last_attempt.get('batch_size', g.batch_size)),
        target_size=int(last_attempt.get('target_size', base_size)),
        active_seed=int(active_seed),
        mode=mode_normalized,
        attempt_index=max(0, len(attempts) - 1),
        used_oom_fallback=bool(len(attempts) > 1),
        auc_thresholds=int(search_runtime.get('auc_thresholds', 200)),
        compute_pr_auc=bool(search_runtime.get('compute_pr_auc', False)),
        min_steps_fraction=float(search_runtime.get('min_steps_fraction', 0.5)),
        model_export_ok=False,
        history_plot_ok=False,
        n_last_layers=int(last_attempt.get('n_last_layers', hyp['n_last_layers'])),
    )


def _run_final_training_with_fallbacks(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    gen_idx: int,
    active_seed: int,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    _require_tensorflow()
    hyp = hyper_for_backbone(g, backbone, cfg['unfreeze_cap'])
    base_size = FINAL_IMG_SIZE if backbone.lower() != 'inception' else 299
    attempts = fallback_attempts_for_backbone(backbone, g.batch_size, base_size, hyp['n_last_layers'])
    tf_module = runtime.get_tf_module_or_none()
    tf_exceptions = (tf_module.errors.OpError,) if tf_module is not None else ()
    last_error: Optional[BaseException] = None
    last_attempt = attempts[0] if attempts else {'batch_size': g.batch_size, 'target_size': base_size, 'n_last_layers': hyp['n_last_layers']}

    for attempt_index, attempt in enumerate(attempts):
        last_attempt = attempt
        try:
            if attempt_index > 0:
                print(
                    f"[OOM-FALLBACK] backbone={backbone} | retry={attempt_index}/{len(attempts)-1} | "
                    f"img={attempt['target_size']} | bs={attempt['batch_size']} | unfreeze_last={attempt['n_last_layers']}"
                )
            result = _train_one_backbone_final_single_attempt(
                cfg,
                g,
                backbone,
                budget,
                rep,
                override_batch_size=attempt['batch_size'],
                override_target_size=attempt['target_size'],
                override_n_last_layers=attempt['n_last_layers'],
                attempt_index=attempt_index,
                artifacts=artifacts,
                gen_idx=gen_idx,
                active_seed=int(active_seed),
            )
            result['attempt_index'] = int(attempt_index)
            result['used_oom_fallback'] = bool(attempt_index > 0)
            return result
        except (MemoryError, RuntimeError, OSError, *tf_exceptions) as exc:
            if not is_resource_exhaustion_error(exc):
                raise
            last_error = exc
            clear_tf_memory()
            print(f"[OOM-FALLBACK] OOM detected for {backbone} on attempt {attempt_index}: {type(exc).__name__}: {exc}")

    return make_infeasible_result(
        cfg, g, backbone, budget, rep,
        reason='resource_exhausted_after_fallbacks',
        error=last_error,
        gen_idx=gen_idx,
        batch_size=int(last_attempt.get('batch_size', g.batch_size)),
        target_size=int(last_attempt.get('target_size', base_size)),
        active_seed=int(active_seed),
        mode='final',
        attempt_index=max(0, len(attempts) - 1),
        used_oom_fallback=bool(len(attempts) > 1),
        auc_thresholds=int(cfg.get('final_runtime', {}).get('auc_thresholds', 1000)),
        compute_pr_auc=bool(cfg.get('final_runtime', {}).get('compute_pr_auc', True)),
        min_steps_fraction=float(cfg.get('final_runtime', {}).get('min_steps_fraction', 1.0)),
        model_export_ok=False,
        history_plot_ok=False,
        n_last_layers=int(last_attempt.get('n_last_layers', hyp['n_last_layers'])),
    )


def train_one_backbone_search(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    root_dir: Optional[str] = None,
    tag: str = 'default',
    gen_idx: int = 0,
    active_seed: int = 0,
) -> Dict[str, Any]:
    return _run_training_with_fallbacks(
        cfg,
        g,
        backbone,
        budget,
        rep,
        gen_idx=gen_idx,
        active_seed=int(active_seed),
        mode='search',
        artifacts=None,
    )


def train_one_backbone_final(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    root_dir: Optional[str] = None,
    tag: str = 'default',
    gen_idx: int = 0,
    active_seed: int = 0,
) -> Dict[str, Any]:
    artifacts = None
    if root_dir is not None:
        artifacts = make_run_artifact_paths(root_dir, tag, g.genome_id, backbone, budget['name'], rep)
    return _run_final_training_with_fallbacks(
        cfg,
        g,
        backbone,
        budget,
        rep,
        gen_idx=gen_idx,
        active_seed=int(active_seed),
        artifacts=artifacts,
    )


def train_one_backbone(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    root_dir: Optional[str] = None,
    tag: str = 'default',
    gen_idx: int = 0,
    active_seed: int = 0,
    mode: str = 'search',
) -> Dict[str, Any]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized == 'search':
        return train_one_backbone_search(
            cfg, g, backbone, budget, rep,
            root_dir=root_dir, tag=tag, gen_idx=gen_idx, active_seed=int(active_seed),
        )
    if mode_normalized == 'final':
        return train_one_backbone_final(
            cfg, g, backbone, budget, rep,
            root_dir=root_dir, tag=tag, gen_idx=gen_idx, active_seed=int(active_seed),
        )
    raise ValueError(f"Unsupported training mode '{mode}'. Expected 'search' or 'final'.")



def _train_one_backbone_single_attempt(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    override_batch_size: Optional[int] = None,
    override_target_size: Optional[int] = None,
    override_n_last_layers: Optional[int] = None,
    attempt_index: int = 0,
    artifacts: Optional[Dict[str, str]] = None,
    gen_idx: int = 0,
    active_seed: int = 0,
    mode: str = 'search',
) -> Dict[str, Any]:
    return _train_one_backbone_attempt_impl(
        cfg, g, backbone, budget, rep,
        override_batch_size=override_batch_size,
        override_target_size=override_target_size,
        override_n_last_layers=override_n_last_layers,
        attempt_index=attempt_index,
        artifacts=artifacts,
        gen_idx=gen_idx,
        active_seed=active_seed,
        mode=mode,
    )


def _train_one_backbone_final_single_attempt(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    override_batch_size: Optional[int] = None,
    override_target_size: Optional[int] = None,
    override_n_last_layers: Optional[int] = None,
    attempt_index: int = 0,
    artifacts: Optional[Dict[str, str]] = None,
    gen_idx: int = 0,
    active_seed: int = 0,
) -> Dict[str, Any]:
    return _train_one_backbone_attempt_impl(
        cfg, g, backbone, budget, rep,
        override_batch_size=override_batch_size,
        override_target_size=override_target_size,
        override_n_last_layers=override_n_last_layers,
        attempt_index=attempt_index,
        artifacts=artifacts,
        gen_idx=gen_idx,
        active_seed=active_seed,
        mode='final',
    )




def _train_one_backbone_attempt_impl(
    cfg: Dict[str, Any],
    g: Genome,
    backbone: str,
    budget: Dict[str, Any],
    rep: int,
    *,
    override_batch_size: Optional[int] = None,
    override_target_size: Optional[int] = None,
    override_n_last_layers: Optional[int] = None,
    attempt_index: int = 0,
    artifacts: Optional[Dict[str, str]] = None,
    gen_idx: int = 0,
    active_seed: int = 0,
    mode: str = 'search',
) -> Dict[str, Any]:
    _require_tensorflow()
    from finetune_ga.models.backbone import build_model
    from finetune_ga.data.loader import load_train_val_data, load_final_retrain_data
    from finetune_ga.data.preprocessing import apply_preprocessing, clear_augmenter_cache

    start = time.perf_counter()

    fp = genome_fingerprint(g, cfg['model_names'])
    seed = (
        stable_crc32(fp)
        ^ stable_crc32(backbone)
        ^ stable_crc32(budget['name'])
        ^ (rep * 1337)
        ^ int(active_seed)
    ) & 0xFFFFFFFF
    seed_everything(int(seed))

    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {'search', 'final'}:
        raise ValueError(f"Unsupported training mode '{mode}'. Expected 'search' or 'final'.")

    runtime_cfg_key = 'search_runtime' if mode_normalized == 'search' else 'final_runtime'
    runtime_cfg = cfg.get(runtime_cfg_key, {}) or {}
    uses_validation = (mode_normalized == 'search') or bool(cfg.get('final_retrain_uses_validation', False))
    min_steps_fraction = float(runtime_cfg.get('min_steps_fraction', 0.5 if mode_normalized == 'search' else 1.0))
    min_steps_fraction = min(1.0, max(0.0, min_steps_fraction))
    auc_thresholds = int(runtime_cfg.get('auc_thresholds', 200 if mode_normalized == 'search' else 1000))
    compute_pr_auc = bool(runtime_cfg.get('compute_pr_auc', mode_normalized == 'final'))

    try:

        hyp = hyper_for_backbone(g, backbone, cfg['unfreeze_cap'])
        lr1, lr2 = hyp['lr1'], hyp['lr2']
        n_last_layers = int(hyp['n_last_layers'] if override_n_last_layers is None else override_n_last_layers)
        default_target_size = FINAL_IMG_SIZE if mode_normalized == 'final' else SEARCH_IMG_SIZE
        requested_target_size = int((default_target_size if backbone.lower() != 'inception' else 299) if override_target_size is None else override_target_size)
        requested_batch_size = int(g.batch_size if override_batch_size is None else override_batch_size)
        es_patience_s1 = int(budget.get('es_patience_s1', cfg.get('early_stopping_patience_stage1', 2)))
        es_patience_s2 = int(budget.get('es_patience_s2', cfg.get('early_stopping_patience_stage2', 2)))
        budget = dict(budget)
        budget['es_patience_s1'] = es_patience_s1
        budget['es_patience_s2'] = es_patience_s2

        with STRATEGY.scope():
            model, target_size = build_model(
                model_name=backbone,
                num_classes=int(cfg['num_classes']),
                img_size=requested_target_size,
                dense_units=g.dense_units,
                dropout=g.dropout,
                l2_weight=g.l2_weight,
            )
            def build_metrics() -> List[Any]:
                stage_metrics: List[Any] = [
                    SoftmaxBinaryAUC(name='auc', curve='ROC', num_thresholds=auc_thresholds),
                    runtime.get_tf_module_or_none().keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                ]
                if compute_pr_auc:
                    stage_metrics.insert(1, SoftmaxBinaryAUC(name='pr_auc', curve='PR', num_thresholds=auc_thresholds))
                return stage_metrics

            backbone_model = ensure_head_trainable_backbone_frozen(model)

        if uses_validation:
            train_ds, val_ds = load_train_val_data(
                target_size=target_size,
                batch_size=requested_batch_size,
                repeat_train=True,
                shuffle_train=True,
                seed=int(seed),
                model_name=backbone,
            )
        else:
            train_ds, _ = load_final_retrain_data(
                target_size=target_size,
                batch_size=requested_batch_size,
                repeat_train=True,
                shuffle_train=True,
                seed=int(seed),
                model_name=backbone,
            )
            val_ds = None

        counts = get_dataset_counts()
        train_count = int(counts['train']) if uses_validation else int(counts['train']) + int(counts['val'])

        raw_sp = steps_safe(train_count, requested_batch_size, float(budget['steps_factor']))
        full_train_sp = full_steps(train_count, requested_batch_size)
        min_train_sp = max(1, int(min_steps_fraction * full_train_sp)) if full_train_sp > 0 else 1
        sp = min(full_train_sp, max(raw_sp, min_train_sp))

        vp = full_steps(counts['val'], requested_batch_size) if uses_validation else 0

        train_ds = apply_preprocessing(
            train_ds,
            model_name=backbone,
            augment=True,
            aug_rotation=g.aug_rotation,
            aug_zoom=g.aug_zoom,
            aug_contrast=g.aug_contrast,
            cache=False,
        )
        if val_ds is not None:
            val_ds = apply_preprocessing(
                val_ds,
                model_name=backbone,
                augment=False,
                cache=False,
            )

        print_run_header(
            g,
            backbone,
            budget,
            rep,
            lr1,
            lr2,
            n_last_layers,
            sp,
            vp,
            target_size,
            effective_batch_size=requested_batch_size,
        )
        if attempt_index > 0:
            print(
                f"      fallback attempt={attempt_index} | "
                f"effective_bs={requested_batch_size} | effective_img={target_size}"
            )

        tparams1 = trainable_params_m(model)
        stage1_snapshot_path = artifacts.get('stage1_snapshot_path') if artifacts is not None else None
        stage1_snapshot_saved = False

        stage1_search_callbacks = _build_search_stage1_callbacks(
            mode=mode_normalized,
            budget=budget,
            backbone=backbone,
            genome_id=g.genome_id,
            budget_name=budget['name'],
            rep=rep,
        ) if uses_validation else ([logger] if (logger := _make_epoch_logger(mode_normalized, 'stage1', int(budget['e1']), backbone, g.genome_id, budget['name'], rep)) is not None else [])

        if artifacts is None:
            cb1 = stage1_search_callbacks
        else:
            cb1 = build_final_stage1_callbacks(
                search_callbacks=stage1_search_callbacks,
            )

        stage1_start = time.perf_counter()
        with STRATEGY.scope():
            model.compile(
                optimizer=runtime.get_tf_module_or_none().keras.optimizers.Adam(lr1, clipnorm=1.0),
                loss='sparse_categorical_crossentropy',
                metrics=build_metrics(),
            )

        fit_kwargs = {
            'x': train_ds,
            'epochs': int(budget['e1']),
            'steps_per_epoch': sp,
            'callbacks': cb1,
            'verbose': _fit_verbose(mode_normalized),
        }
        if uses_validation and val_ds is not None:
            fit_kwargs['validation_data'] = val_ds
            fit_kwargs['validation_steps'] = vp
        h1 = model.fit(**fit_kwargs)
        stage1_time_s = time.perf_counter() - stage1_start
        s1 = best_epoch_by_auc(h1, uses_validation=uses_validation) if uses_validation else (None, None, -1)

        # Save Stage1 best weights only if Stage2 will run. This prevents
        # exporting Stage2 weights when Stage1 is the selected best stage,
        # without adding a redundant Stage2 snapshot.
        if (
            uses_validation
            and artifacts is not None
            and int(budget.get('e2', 0)) > 0
        ):
            stage1_snapshot_saved = _save_stage_snapshot(
                model,
                stage1_snapshot_path,
                backbone=backbone,
                budget_name=budget['name'],
                rep=rep,
                stage=1,
            )

        s2 = (None, None, -1) if not uses_validation else (0.0, 1e9, -1)
        stage2_time_s = 0.0
        tparams2 = tparams1
        h2 = None

        if int(budget['e2']) > 0:
            set_finetune_last_n_layers(backbone_model, n_last_layers)
            tparams2 = trainable_params_m(model)

            stage2_search_callbacks = _build_search_stage2_callbacks(
                mode=mode_normalized,
                budget=budget,
                backbone=backbone,
                genome_id=g.genome_id,
                budget_name=budget['name'],
                rep=rep,
            ) if uses_validation else ([logger] if (logger := _make_epoch_logger(mode_normalized, 'stage2', int(budget['e2']), backbone, g.genome_id, budget['name'], rep)) is not None else [])

            if artifacts is None:
                cb2 = stage2_search_callbacks
            else:
                cb2 = build_final_stage2_callbacks(
                    search_callbacks=stage2_search_callbacks,
                )

            with STRATEGY.scope():
                model.compile(
                    optimizer=runtime.get_tf_module_or_none().keras.optimizers.Adam(lr2, clipnorm=1.0),
                    loss='sparse_categorical_crossentropy',
                    metrics=build_metrics(),
                )

            stage2_start = time.perf_counter()
            fit_kwargs = {
                'x': train_ds,
                'epochs': int(budget['e2']),
                'steps_per_epoch': sp,
                'callbacks': cb2,
                'verbose': _fit_verbose(mode_normalized),
            }
            if uses_validation and val_ds is not None:
                fit_kwargs['validation_data'] = val_ds
                fit_kwargs['validation_steps'] = vp
            h2 = model.fit(**fit_kwargs)
            stage2_time_s = time.perf_counter() - stage2_start
            s2 = best_epoch_by_auc(h2, uses_validation=uses_validation) if uses_validation else (None, None, -1)

        if uses_validation:
            best_auc, best_loss, best_epoch, best_stage = choose_better_stage(s1, s2)
            winning_hist = h1 if best_stage == 1 else h2
            if winning_hist is not None and best_epoch >= 0:
                pr_key = 'val_pr_auc'
                acc_key = 'val_accuracy'
                auc_key = 'auc'
                _pr = winning_hist.history.get(pr_key, [])
                _acc = winning_hist.history.get(acc_key, [])
                _train_auc = winning_hist.history.get(auc_key, [])
                metric_pr_best = _safe_metric(_pr[best_epoch], 0.0) if best_epoch < len(_pr) else 0.0
                metric_acc_best = _safe_metric(_acc[best_epoch], 0.0) if best_epoch < len(_acc) else 0.0
                train_auc_best = _safe_metric(_train_auc[best_epoch], 0.0) if best_epoch < len(_train_auc) else 0.0
            else:
                metric_pr_best = 0.0
                metric_acc_best = 0.0
                train_auc_best = 0.0
        else:
            best_auc, best_loss, best_epoch, best_stage = (None, None, None, None)
            metric_pr_best = None
            metric_acc_best = None
            all_train_auc: List[float] = []
            all_train_pr_auc: List[float] = []
            all_train_accuracy: List[float] = []

            def _extend_finite_metric_values(target: List[float], values: Any) -> None:
                for value in values or []:
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value_f):
                        target.append(value_f)

            for hist in (h1, h2):
                if hist is None:
                    continue
                _extend_finite_metric_values(all_train_auc, hist.history.get('auc', []))
                _extend_finite_metric_values(all_train_pr_auc, hist.history.get('pr_auc', []))
                _extend_finite_metric_values(all_train_accuracy, hist.history.get('accuracy', []))
            train_auc_best = max(all_train_auc) if all_train_auc else None
            train_pr_auc_best = max(all_train_pr_auc) if all_train_pr_auc else None
            train_accuracy_best = max(all_train_accuracy) if all_train_accuracy else None

        # If Stage1 won after Stage2 mutated the model, restore Stage1 weights
        # before final artifact export so exported weights match best_from_stage.
        if (
            uses_validation
            and artifacts is not None
            and best_stage == 1
            and int(budget.get('e2', 0)) > 0
        ):
            if stage1_snapshot_saved:
                try:
                    _restore_stage_snapshot(
                        model,
                        stage1_snapshot_path,
                        backbone=backbone,
                        budget_name=budget['name'],
                        rep=rep,
                        stage=1,
                    )
                finally:
                    _delete_stage_snapshot(stage1_snapshot_path)
            else:
                warnings.warn(
                    f'Stage 1 was best but snapshot missing for {backbone}/{budget["name"]}/rep{rep}. '
                    'Exported model may not match best_from_stage.'
                )

        # Stage1 snapshots are temporary handoff files used only to restore the
        # Stage1 winner after Stage2 has run. If Stage1 won, the finally block
        # above has already deleted the snapshot; otherwise clean it up here.
        elif stage1_snapshot_saved:
            _delete_stage_snapshot(stage1_snapshot_path)

        elapsed = time.perf_counter() - start
        tparams = float(max(tparams1, tparams2))

        checkpoint_payload = {
            'checkpoint_dir': None,
            'best_weights_path': None,
            'best_model_path': None,
            'history_plot_path': None,
            'is_best_checkpoint_saved': False,
            'model_export_ok': False,
            'history_plot_ok': False,
        }
        if artifacts is not None:
            checkpoint_payload = persist_final_artifacts(
                model,
                h1,
                h2,
                artifacts,
                backbone,
                budget['name'],
                rep,
            )

        return {
            'gen': int(gen_idx),
            'genome_id': g.genome_id,
            'genome': genome_to_dict(g),
            'backbone': backbone,
            'budget': budget['name'],
            'budget_meta': budget,
            'rep': rep,
            'seed': int(seed),
            'active_seed': int(active_seed),
            'target_size': int(target_size),
            'lr1': _safe_metric(lr1, 0.0),
            'lr2': _safe_metric(lr2, 0.0),
            'n_last_layers': int(n_last_layers),
            'dense_units': int(g.dense_units),
            'dropout': _safe_metric(g.dropout, 0.0),
            'l2_weight': _safe_metric(g.l2_weight, 0.0),
            'batch_size': int(requested_batch_size),
            'aug_rotation': _safe_metric(g.aug_rotation, 0.0),
            'aug_zoom': _safe_metric(g.aug_zoom, 0.0),
            'aug_contrast': _safe_metric(g.aug_contrast, 0.0),
            'best_val_auc': (_safe_metric(best_auc, 0.0) if uses_validation else None),
            'best_val_pr_auc': (_safe_metric(metric_pr_best, 0.0) if (compute_pr_auc and uses_validation and metric_pr_best is not None) else None),
            'best_val_pr_auc_computed': bool(compute_pr_auc and uses_validation),
            'best_val_accuracy': (_safe_metric(metric_acc_best, 0.0) if uses_validation and metric_acc_best is not None else None),
            'best_train_pr_auc': (_safe_metric(train_pr_auc_best, 0.0) if (not uses_validation and train_pr_auc_best is not None) else None),
            'best_train_accuracy': (_safe_metric(train_accuracy_best, 0.0) if (not uses_validation and train_accuracy_best is not None) else None),
            'best_train_auc': (_safe_metric(train_auc_best, 0.0) if train_auc_best is not None else None),
            'loss_at_best_auc': (_safe_metric(best_loss, 1e9) if uses_validation and best_loss is not None else None),
            'best_auc_epoch': (int(best_epoch) if uses_validation and best_epoch is not None else None),
            'best_from_stage': (int(best_stage) if uses_validation and best_stage is not None else None),
            'time_s': _safe_metric(elapsed, 0.0),
            'stage1_time_s': _safe_metric(stage1_time_s, 0.0),
            'stage2_time_s': _safe_metric(stage2_time_s, 0.0),
            'trainable_params_m': _safe_metric(tparams, 0.0),
            'steps_per_epoch': int(sp),
            'val_steps': int(vp),
            'final_retrain_uses_validation': bool(uses_validation if mode_normalized == 'final' else True),
            'final_training_protocol': ('fixed_schedule_no_selection' if mode_normalized == 'final' and not uses_validation else None),
            'epochs_stage1_ran': len(h1.history.get('loss', [])),
            'epochs_stage2_ran': len(h2.history.get('loss', [])) if h2 is not None else 0,
            'best_epoch_stage1': (int(s1[2]) if uses_validation else None),
            'best_epoch_stage2': (int(s2[2]) if uses_validation else None),
            'num_replicas_in_sync': int(getattr(STRATEGY, 'num_replicas_in_sync', 1)),
            'mixed_precision_policy': str(runtime.get_tf_module_or_none().keras.mixed_precision.global_policy()),
            'mode': mode_normalized,
            'auc_thresholds': int(auc_thresholds),
            'compute_pr_auc': bool(compute_pr_auc),
            'min_steps_fraction': _safe_metric(min_steps_fraction, 0.0),
            'attempt_index': int(attempt_index),
            'used_oom_fallback': bool(attempt_index > 0),
            **checkpoint_payload,
            'status': 'ok',
        }
    finally:
        clear_augmenter_cache()
