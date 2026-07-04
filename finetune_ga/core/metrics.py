"""metrics.py — Evaluation metrics: statistics, hypervolume, classification, budget summaries.

This module is TensorFlow-optional: most functions run without it.
Only compute_binary_classification_metrics() requires TF (for AUC computation).
"""
from __future__ import annotations

import math
import re
import warnings
from statistics import NormalDist
from typing import Any, Dict, List, Tuple

import numpy as np

from finetune_ga.infra.config import IMG_SIZE, SEARCH_IMG_SIZE, FINAL_IMG_SIZE


def _finite_float(value: Any, default: float) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def replication_metrics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q1': 0.0,
            'q3': 0.0,
        }
    arr = np.array(values, dtype=float)
    q1, q3 = (
        np.percentile(arr, [25, 75]) if len(arr) > 1
        else (float(arr[0]), float(arr[0]))
    )
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        'mean': float(np.mean(arr)),
        'std': std,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q1': float(q1),
        'q3': float(q3),
    }


def _default_target_size_for_row(row: Dict[str, Any]) -> float:
    backbone = str(row.get('backbone', '') or '').strip().lower()
    if backbone == 'inception':
        return 299.0
    mode = str(row.get('mode', 'search') or 'search').strip().lower()
    return float(FINAL_IMG_SIZE if mode == 'final' else SEARCH_IMG_SIZE)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def _minmax_normalize(
    value: float,
    min_value: float,
    max_value: float,
    *,
    clip: bool = True,
) -> float:
    """Return a finite min-max normalized value for publication tables."""
    try:
        value_f = float(value)
        min_f = float(min_value)
        max_f = float(max_value)
    except (TypeError, ValueError):
        return 1.0

    if not all(math.isfinite(v) for v in (value_f, min_f, max_f)):
        return 1.0
    if max_f <= min_f:
        return 0.0

    normalized = (value_f - min_f) / (max_f - min_f)
    if clip:
        normalized = max(0.0, min(1.0, normalized))
    return float(normalized)


def compute_complexity_components(
    trainable_params_m: float,
    total_time_s: float,
    target_size: int,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Return normalized components used by compute_complexity_score().

    Components are costs, so lower is better. Ranges are configurable through
    cfg['complexity_weights'] and default to the ranges documented in config.json.
    """
    cw = cfg.get('complexity_weights', {}) if isinstance(cfg, dict) else {}

    params_min = float(cw.get('params_m_min', 1.0))
    params_max = float(cw.get('params_m_max', 100.0))
    time_min = float(cw.get('time_h_min', 0.0))
    time_max = float(cw.get('time_h_max', 24.0))
    size_min = float(cw.get('size_ratio_min', 0.5))
    size_max = float(cw.get('size_ratio_max', 1.5))
    clip = bool(cw.get('clip_normalized', True))

    time_h = float(total_time_s) / 3600.0
    size_ratio = float(target_size) / 224.0

    return {
        'params_norm': _minmax_normalize(
            float(trainable_params_m), params_min, params_max, clip=clip,
        ),
        'time_norm': _minmax_normalize(time_h, time_min, time_max, clip=clip),
        'size_norm': _minmax_normalize(size_ratio, size_min, size_max, clip=clip),
        'time_h': float(time_h),
        'size_ratio': float(size_ratio),
    }


def compute_complexity_score(
    trainable_params_m: float,
    total_time_s: float,
    target_size: int,
    cfg: Dict[str, Any] | None = None,
) -> float:
    """Weighted normalized diagnostic complexity score.

    The score is a weighted sum of min-max normalized cost components:
    trainable parameters, wall-clock time in hours, and input-size ratio. The
    output is finite and, with clipping enabled, lies in [0, 1].
    """
    cw = cfg.get('complexity_weights', {}) if isinstance(cfg, dict) else {}
    w_params = float(cw.get('params_m', 0.55))
    w_time = float(cw.get('time_h', 0.35))
    w_size = float(cw.get('size_ratio', 0.10))

    weight_sum = w_params + w_time + w_size
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError('complexity_weights must have a positive finite weight sum')

    components = compute_complexity_components(
        trainable_params_m,
        total_time_s,
        target_size,
        cfg,
    )

    return float(
        (
            w_params * components['params_norm']
            + w_time * components['time_norm']
            + w_size * components['size_norm']
        )
        / weight_sum
    )

def wilson_score_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) / n) +
                           (z * z) / (4.0 * n * n)) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


# ---------------------------------------------------------------------------
# Hypervolume (exact sweep-line algorithms)
# ---------------------------------------------------------------------------

def hypervolume_2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Exact 2D hypervolume via a sweep-line algorithm over the nondominated frontier."""
    pts = [
        (float(x), float(y))
        for x, y in points
        if float(x) <= ref[0] and float(y) <= ref[1]
    ]
    if not pts:
        return 0.0

    pts = sorted(pts, key=lambda t: (t[0], t[1]))

    frontier: List[Tuple[float, float]] = []
    best_y = float('inf')
    for x, y in pts:
        if y < best_y:
            frontier.append((x, y))
            best_y = y

    hv, prev_y = 0.0, ref[1]
    for x, y in frontier:
        hv += max(0.0, ref[0] - x) * max(0.0, prev_y - y)
        prev_y = y
    return float(hv)


def hypervolume_3d(
    points: List[Tuple[float, float, float]], ref: Tuple[float, float, float]
) -> float:
    """Exact 3D hypervolume via a 2D sweep-line over x-slabs.

    O(n² log n) in the number of points — acceptable for the small population
    sizes used in this project.
    """
    pts = [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in points
        if all(float(p[i]) <= ref[i] for i in range(3))
    ]
    if not pts:
        return 0.0
    xs = sorted(set(p[0] for p in pts))
    hv = 0.0
    for i, x in enumerate(xs):
        x_next = xs[i + 1] if i + 1 < len(xs) else ref[0]
        if x_next <= x:
            continue
        slice_pts = [(p[1], p[2]) for p in pts if p[0] <= x]
        hv += hypervolume_2d(slice_pts, ref=(ref[1], ref[2])) * (x_next - x)
    return float(hv)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_confidence_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.size == 0:
        return {'brier_score': 0.0, 'log_loss': 0.0, 'ece_10bins': 0.0,
                'mean_positive_confidence': 0.0, 'mean_negative_confidence': 0.0}
    eps = 1e-7
    probs = np.clip(y_prob, eps, 1.0 - eps)
    brier = float(np.mean((probs - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(probs) +
                     (1 - y_true) * np.log(1 - probs)))
    pred = (probs >= 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1.0 - probs)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if not np.any(mask):
            continue
        acc = np.mean((pred[mask] == y_true[mask]).astype(float))
        avg_conf = float(np.mean(conf[mask]))
        ece += float(np.mean(mask.astype(float))) * abs(acc - avg_conf)
    return {
        'brier_score': brier,
        'log_loss': log_loss,
        'ece_10bins': float(ece),
        'mean_positive_confidence': float(np.mean(probs[y_true == 1])) if np.any(y_true == 1) else 0.0,
        'mean_negative_confidence': float(np.mean(1.0 - probs[y_true == 0])) if np.any(y_true == 0) else 0.0,
    }


def compute_binary_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Full classification report. Requires TensorFlow (for AUC computation)."""
    from finetune_ga.infra.runtime import tf, _require_tensorflow
    _require_tensorflow()
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    if y_prob.size and not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains NaN/Inf values")
    auc_valid = bool({0, 1}.issubset(set(np.unique(y_true).tolist()))) if y_true.size else False
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(len(y_true))

    acc = safe_div(tp + tn, n)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    bal_acc = 0.5 * (recall + specificity)
    npv = safe_div(tn, tn + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    prevalence = safe_div(tp + fn, n)
    mcc_denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_denominator <= 0.0:
        mcc = 0.0
    else:
        mcc = float(((tp * tn) - (fp * fn)) / math.sqrt(mcc_denominator))

    if auc_valid:
        auc_metric = tf.keras.metrics.AUC(curve='ROC', num_thresholds=1000)
        auc_metric.update_state(y_true, y_prob)
        pr_auc_metric = tf.keras.metrics.AUC(curve='PR', num_thresholds=1000)
        pr_auc_metric.update_state(y_true, y_prob)
        auc_value = float(auc_metric.result().numpy())
        pr_auc_value = float(pr_auc_metric.result().numpy())
    else:
        auc_value = float('nan')
        pr_auc_value = float('nan')

    acc_lo, acc_hi = wilson_score_interval(tp + tn, n)
    rec_lo, rec_hi = wilson_score_interval(
        tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    spec_lo, spec_hi = wilson_score_interval(
        tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0)
    prec_lo, prec_hi = wilson_score_interval(
        tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)

    out = {
        'auc': auc_value,
        'pr_auc': pr_auc_value,
        'auc_valid': bool(auc_valid),
        'pr_auc_valid': bool(auc_valid),
        'accuracy': float(acc), 'precision': float(precision), 'recall': float(recall),
        'specificity': float(specificity), 'f1': float(f1),
        'balanced_accuracy': float(bal_acc), 'npv': float(npv),
        'fpr': float(fpr), 'fnr': float(fnr), 'mcc': float(mcc),
        'prevalence': float(prevalence), 'support': int(n),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy_ci95_low': float(acc_lo), 'accuracy_ci95_high': float(acc_hi),
        'recall_ci95_low': float(rec_lo), 'recall_ci95_high': float(rec_hi),
        'specificity_ci95_low': float(spec_lo), 'specificity_ci95_high': float(spec_hi),
        'precision_ci95_low': float(prec_lo), 'precision_ci95_high': float(prec_hi),
    }
    out.update(compute_confidence_metrics(y_true, y_prob))
    return out


# ---------------------------------------------------------------------------
# Run enrichment helpers (internal)
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _numeric_metric_keys(rows: List[Dict[str, Any]]) -> List[str]:
    blocked = {
        'genome', 'budget_meta', 'objectives', 'tag', 'backbone', 'budget', 'genome_id',
        'mixed_precision_policy', 'checkpoint_dir', 'best_model_path', 'best_weights_path',
        'history_plot_path', 'pr_curve_plot_path', 'roc_curve_plot_path',
        'is_best_checkpoint_saved',
    }
    keys: set = set()
    for r in rows:
        for k, v in r.items():
            if k in blocked:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, str):
                keys.add(k)
    return sorted(keys)


def enrich_summary_with_run_metrics(out: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return out
    for key in _numeric_metric_keys(runs):
        vals = [
            float(r[key])
            for r in runs
            if isinstance(r.get(key), (int, float, np.integer, np.floating)) and not isinstance(r.get(key), bool)
        ]
        if not vals:
            continue
        for stat_name, stat_val in replication_metrics(vals).items():
            out[f'{key}_{stat_name}'] = float(stat_val)
        if key.endswith('_time_s') or key in {'time_s'}:
            out[f'{key}_sum'] = float(np.sum(vals))

    pts2, pts3 = [], []
    for r in runs:
        auc = r.get('best_val_auc')
        tm = r.get('time_s')
        prm = r.get('trainable_params_m')
        if isinstance(auc, (int, float, np.integer, np.floating)) and isinstance(tm, (int, float, np.integer, np.floating)):
            pts2.append((1.0 - float(auc), float(tm)))
        if (isinstance(auc, (int, float, np.integer, np.floating))
                and isinstance(tm, (int, float, np.integer, np.floating))
                and isinstance(prm, (int, float, np.integer, np.floating))):
            pts3.append((1.0 - float(auc), float(tm), float(prm)))

    if pts2:
        out['hypervolume_2d'] = float(hypervolume_2d(
            pts2, ref=(1.0, max(max(p[1] for p in pts2) * 1.05, 1.0))))
    if pts3:
        out['hypervolume_3d'] = float(hypervolume_3d(pts3, ref=(
            1.0,
            max(max(p[1] for p in pts3) * 1.05, 1.0),
            max(max(p[2] for p in pts3) * 1.05, 1.0),
        )))
    return out


# ---------------------------------------------------------------------------
# Budget summarisation
# ---------------------------------------------------------------------------

def _coerce_summary_run(row: Dict[str, Any]) -> Dict[str, Any] | None:
    status = row.get('status')
    if status is not None and str(status).strip().lower() != 'ok':
        return None
    try:
        rep = int(row.get('rep'))
        backbone = str(row.get('backbone'))
        # Search/selection summaries must be based on validation AUC only.
        # Do not fall back to train AUC: that would silently change the
        # optimisation target and make the selection protocol harder to defend.
        raw_auc = row.get('best_val_auc')
        if raw_auc is None:
            return None
        auc = float(raw_auc)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(auc):
        return None
    if not backbone or backbone == 'None':
        return None
    out = dict(row)
    out['rep'] = rep
    out['backbone'] = backbone
    out['best_val_auc'] = auc
    for key, default in [
        ('best_val_accuracy', 0.0),
        ('loss_at_best_auc', 1e9),
        ('time_s', 0.0),
        ('trainable_params_m', 0.0),
        ('target_size', None),
    ]:
        raw_value = row.get(key, default)
        if key == 'target_size' and raw_value is None:
            out[key] = _default_target_size_for_row(row)
            continue
        out[key] = _finite_float(
            raw_value,
            _default_target_size_for_row(row) if key == 'target_size' else float(default),
        )
    pr_auc_raw = row.get('best_val_pr_auc', None)
    out['best_val_pr_auc'] = None if pr_auc_raw is None else _finite_float(pr_auc_raw, 0.0)
    return out


def summarize_budget_runs(
    cfg: Dict[str, Any], runs: List[Dict[str, Any]], budget_name: str
) -> Dict[str, Any]:
    """Summarise training runs for a given budget level.

    Primary objective is ``mean_val_auc``. Search therefore favours the
    highest average validation AUC across all evaluated backbone/replication
    runs, while wall-clock time and mean trainable parameters remain the
    other two Pareto objectives.
    """
    total_input_runs = int(len(runs))
    valid_runs = [r for r in (_coerce_summary_run(row)
                              for row in runs) if r is not None]
    n_ok_runs = int(len(valid_runs))
    n_failed_runs = max(0, total_input_runs - n_ok_runs)
    success_rate = float(n_ok_runs / total_input_runs) if total_input_runs else 0.0
    failure_rate = float(1.0 - success_rate) if total_input_runs else 0.0
    status_counts: Dict[str, int] = {}
    failure_reason_counts: Dict[str, int] = {}
    for raw in runs:
        status = str(raw.get('status', 'ok')).strip().lower() or 'unknown'
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != 'ok':
            reason = str(raw.get('failure_reason') or raw.get('final_error') or 'unknown').strip() or 'unknown'
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    reps = sorted(set(r['rep'] for r in valid_runs))
    min_auc_per_rep: List[float] = []
    worst_backbone_per_rep: List[str] = []
    for rep in reps:
        rr = sorted([r for r in valid_runs if r['rep'] == rep],
                    key=lambda x: x['best_val_auc'])
        if not rr:
            continue
        min_auc_per_rep.append(float(rr[0]['best_val_auc']))
        worst_backbone_per_rep.append(rr[0]['backbone'])

    robust_min_auc_raw = float(min(min_auc_per_rep)) if min_auc_per_rep else 0.0
    robust_min_auc = float(robust_min_auc_raw * success_rate)
    worst_backbone = (
        worst_backbone_per_rep[int(np.argmin(min_auc_per_rep))]
        if min_auc_per_rep else None
    )

    aucs = [float(r['best_val_auc'])
            for r in valid_runs] if valid_runs else [0.0]
    pr_aucs = [float(r['best_val_pr_auc']) for r in valid_runs if r.get('best_val_pr_auc') is not None]
    accs = [float(r.get('best_val_accuracy', 0.0))
            for r in valid_runs] if valid_runs else [0.0]
    losses = [float(r.get('loss_at_best_auc', 1e9))
              for r in valid_runs] if valid_runs else [1e9]
    total_time = float(sum(float(r.get('time_s', 0.0)) for r in valid_runs))
    mean_time_per_run = float(total_time / len(valid_runs)) if valid_runs else 1e9
    mean_tparams = float(np.mean([float(
        r.get('trainable_params_m', 0.0)) for r in valid_runs])) if valid_runs else 1e9
    mean_target_size = float(np.mean([float(r.get(
        'target_size', _default_target_size_for_row(r))) for r in valid_runs])) if valid_runs else float(IMG_SIZE)

    mean_val_auc_raw = float(np.mean(aucs)) if aucs else 0.0
    # Failed/infeasible runs are not silently dropped from selection.
    # The validation AUC objectives are penalized by the observed success rate,
    # while raw ok-only values remain available for audit tables.
    mean_val_auc = float(mean_val_auc_raw * success_rate)

    out: Dict[str, Any] = {
        'budget': budget_name,
        'mean_val_auc': mean_val_auc,
        'mean_val_auc_raw_ok_only': mean_val_auc_raw,
        'robust_min_auc': robust_min_auc,
        'robust_min_auc_raw_ok_only': robust_min_auc_raw,
        'n_input_runs': total_input_runs,
        'n_ok_runs': n_ok_runs,
        'n_failed_runs': n_failed_runs,
        'failure_rate': failure_rate,
        'success_rate': success_rate,
        'status_counts': status_counts,
        'failure_reason_counts': failure_reason_counts,
        'worst_backbone': worst_backbone,
        'total_time_s': total_time,
        'mean_time_s_per_run': mean_time_per_run,
        'mean_trainable_params_m': mean_tparams,
        'mean_target_size': mean_target_size,
        'n_runs': int(len(valid_runs)),
    }
    for prefix, vals in [('auc', aucs), ('pr_auc', pr_aucs), ('accuracy', accs), ('loss', losses)]:
        for k, v in replication_metrics(vals).items():
            out[f'{prefix}_{k}'] = float(v)

    # Complexity score: weighted combination of trainable params, wall-clock time,
    # and input resolution.  Weights configurable via cfg['complexity_weights'].
    out['complexity_score'] = compute_complexity_score(
        mean_tparams,
        mean_time_per_run,
        int(mean_target_size),
        cfg,
    )
    out = enrich_summary_with_run_metrics(out, valid_runs)
    return out


def objectives_from_summary(s: Dict[str, Any]) -> Tuple[float, float, float]:
    if 'mean_val_auc' not in s:
        raise KeyError('mean_val_auc is required in summary rows for objective construction')
    mean_auc = _finite_float(s.get('mean_val_auc'), 0.0)
    mean_time = _finite_float(s.get('mean_time_s_per_run', s.get('total_time_s', 0.0)), 0.0)
    mean_params = _finite_float(s.get('mean_trainable_params_m'), 0.0)
    return (
        float(max(0.0, 1.0 - mean_auc)),
        float(max(0.0, mean_time)),
        float(max(0.0, mean_params)),
    )
