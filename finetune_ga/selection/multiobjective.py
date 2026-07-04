from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, List, Tuple

from finetune_ga.infra.math_utils import pareto_dominates_values
from finetune_ga.infra.io_utils import safe_float

FINAL_SELECTION_METHOD = "pareto_ideal_point_3d"
FINAL_SELECTION_NORMALIZATION = "minmax"
FINAL_SELECTION_PROTOCOL_NAME = "validation_only_pareto_ideal_point_3d"
FINAL_SELECTION_OBJECTIVES = ("auc_loss", "time_s", "trainable_params_m")
FINAL_SELECTION_RUNTIME_METRICS = ("best_val_auc", "time_s", "trainable_params_m")


def build_selection_objectives(row: Mapping[str, Any], *, auc_key: str, time_key: str, params_key: str) -> Dict[str, float]:
    auc_val = safe_float(row.get(auc_key), float('nan'))
    time_val = safe_float(row.get(time_key), float('inf'))
    params_val = safe_float(row.get(params_key), float('inf'))
    return {
        'auc_loss': float('nan') if not math.isfinite(auc_val) else max(0.0, 1.0 - auc_val),
        'time_s': max(0.0, time_val) if math.isfinite(time_val) else float('inf'),
        'trainable_params_m': max(0.0, params_val) if math.isfinite(params_val) else float('inf'),
    }


def candidate_has_finite_objectives(objectives: Mapping[str, Any], objective_keys: Sequence[str] = FINAL_SELECTION_OBJECTIVES) -> bool:
    try:
        return all(math.isfinite(float(objectives[k])) for k in objective_keys)
    except (KeyError, TypeError, ValueError):
        return False


def prepare_selection_row(
    row: Mapping[str, Any], *, auc_key: str, time_key: str, params_key: str,
    method: str = FINAL_SELECTION_METHOD, runtime_metrics: Sequence[str] | None = None,
) -> Dict[str, Any]:
    row_copy = dict(row)
    objectives = build_selection_objectives(row_copy, auc_key=auc_key, time_key=time_key, params_key=params_key)
    row_copy['selection_objectives'] = objectives
    row_copy['selection_method'] = method
    row_copy['selection_runtime_metrics'] = list(runtime_metrics or [auc_key, time_key, params_key])
    row_copy['selection_auc_key'] = auc_key
    row_copy['selection_time_key'] = time_key
    row_copy['selection_params_key'] = params_key
    return row_copy


def dominates(a: Mapping[str, float], b: Mapping[str, float], objective_keys: Sequence[str]) -> bool:
    a_vals = [float(a[k]) for k in objective_keys]
    b_vals = [float(b[k]) for k in objective_keys]
    return pareto_dominates_values(a_vals, b_vals)


def pareto_front_rows(rows: Sequence[Mapping[str, Any]], objective_keys: Sequence[str] = FINAL_SELECTION_OBJECTIVES) -> List[Dict[str, Any]]:
    prepared: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    for row in rows:
        row_copy = dict(row)
        objectives = dict(row_copy.get('selection_objectives', {}))
        if not objectives:
            objectives = {k: safe_float(row_copy.get(k), float('inf')) for k in objective_keys}
        row_copy['selection_objectives'] = objectives
        if not candidate_has_finite_objectives(objectives, objective_keys):
            continue
        prepared.append((row_copy, objectives))
    front: List[Dict[str, Any]] = []
    for idx, (row, objectives) in enumerate(prepared):
        dominated = False
        for jdx, (_, other_objectives) in enumerate(prepared):
            if idx == jdx:
                continue
            if dominates(other_objectives, objectives, objective_keys):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def non_dominated_sort_rows(rows: Sequence[Mapping[str, Any]], objective_keys: Sequence[str] = FINAL_SELECTION_OBJECTIVES) -> List[List[Dict[str, Any]]]:
    remaining = [dict(r) for r in rows]
    fronts: List[List[Dict[str, Any]]] = []
    while remaining:
        front = pareto_front_rows(remaining, objective_keys)
        if not front:
            break
        fronts.append(front)
        signatures = {
            (
                str(r.get('genome_id', '')),
                str(r.get('budget', '')),
                str(r.get('backbone', '')),
                str(r.get('rep', '')),
                tuple(sorted(dict(r.get('selection_objectives', {})).items())),
            )
            for r in front
        }
        remaining = [
            r for r in remaining
            if (
                str(r.get('genome_id', '')),
                str(r.get('budget', '')),
                str(r.get('backbone', '')),
                str(r.get('rep', '')),
                tuple(sorted(dict(r.get('selection_objectives', {})).items())),
            ) not in signatures
        ]
    return fronts


def _normalized_value(value: float, values: Sequence[float]) -> float:
    lo, hi = min(values), max(values)
    if not math.isfinite(value) or not math.isfinite(lo) or not math.isfinite(hi):
        return 1.0
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _selection_runtime_keys(row: Mapping[str, Any], tie_break_auc_key: str) -> tuple[str, str, str]:
    runtime_keys = list(row.get('selection_runtime_metrics', []))
    auc_key = str(row.get('selection_auc_key', runtime_keys[0] if len(runtime_keys) > 0 else tie_break_auc_key))
    time_key = str(row.get('selection_time_key', runtime_keys[1] if len(runtime_keys) > 1 else 'time_s'))
    params_key = str(row.get('selection_params_key', runtime_keys[2] if len(runtime_keys) > 2 else 'trainable_params_m'))
    return auc_key, time_key, params_key


def _finalize_scored_front(
    scored: Sequence[Mapping[str, Any]], *, tie_break_auc_key: str, tie_break_name_key: str, stable_id_key: str,
) -> List[Dict[str, Any]]:
    if not scored:
        return []
    exemplar = dict(scored[0])
    _, time_key, params_key = _selection_runtime_keys(exemplar, tie_break_auc_key)
    scored_rows = [dict(r) for r in scored]
    scored_rows.sort(key=lambda r: (
        float(r.get('ideal_distance', float('inf'))),
        -safe_float(r.get(tie_break_auc_key), 0.0),
        safe_float(r.get(params_key), float('inf')),
        safe_float(r.get(time_key), float('inf')),
        str(r.get(tie_break_name_key, '')),
        str(r.get(stable_id_key, '')),
        str(r.get('rep', '')),
    ))
    return scored_rows


def select_by_ideal_point(
    rows: Sequence[Mapping[str, Any]], objective_keys: Sequence[str] = FINAL_SELECTION_OBJECTIVES, *,
    tie_break_auc_key: str = 'best_val_auc', tie_break_name_key: str = 'backbone', stable_id_key: str = 'genome_id',
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not rows:
        return {}, []
    front = pareto_front_rows(rows, objective_keys)
    if not front:
        return {}, []
    objective_values = {k: [float(dict(r).get('selection_objectives', {}).get(k, float('inf'))) for r in front] for k in objective_keys}
    scored: List[Dict[str, Any]] = []
    for row in front:
        row_copy = dict(row)
        objectives = dict(row_copy.get('selection_objectives', {}))
        normalized = {k: _normalized_value(float(objectives.get(k, float('inf'))), objective_values[k]) for k in objective_keys}
        row_copy['selection_objectives'] = objectives
        row_copy['selection_normalized_objectives'] = normalized
        row_copy['ideal_distance'] = float(math.sqrt(sum(normalized[k] ** 2 for k in objective_keys)))
        scored.append(row_copy)
    ordered = _finalize_scored_front(scored, tie_break_auc_key=tie_break_auc_key, tie_break_name_key=tie_break_name_key, stable_id_key=stable_id_key)
    return (ordered[0] if ordered else {}), ordered


def rank_candidates_multiobjective(
    rows: Sequence[Mapping[str, Any]], *, auc_key: str, time_key: str, params_key: str,
    objective_keys: Sequence[str] = FINAL_SELECTION_OBJECTIVES, report_role_candidate: str | None = None,
    report_role_ranked: str | None = None, tie_break_name_key: str = 'genome_id', stable_id_key: str = 'genome_id',
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = prepare_selection_row(
            dict(row), auc_key=auc_key, time_key=time_key, params_key=params_key,
            runtime_metrics=[auc_key, time_key, params_key],
        )
        if report_role_candidate:
            row_dict['report_role'] = report_role_candidate
        prepared.append(row_dict)
    fronts = non_dominated_sort_rows(prepared, objective_keys)
    ranked_rows: List[Dict[str, Any]] = []
    for front_rank, front in enumerate(fronts, start=1):
        _, scored_front = select_by_ideal_point(
            front, objective_keys, tie_break_auc_key=auc_key, tie_break_name_key=tie_break_name_key, stable_id_key=stable_id_key,
        )
        for ideal_rank, row in enumerate(scored_front, start=1):
            row_copy = dict(row)
            objectives = dict(row_copy.get('selection_objectives', {}))
            normalized = dict(row_copy.get('selection_normalized_objectives', {}))
            row_copy['pareto_front_rank'] = front_rank
            row_copy['pareto_within_front_rank'] = ideal_rank
            row_copy['selection_auc'] = safe_float(row_copy.get(auc_key), float('nan'))
            row_copy['selection_auc_loss'] = objectives.get('auc_loss')
            row_copy['selection_time_s'] = safe_float(row_copy.get(time_key), float('inf'))
            row_copy['selection_params_m'] = safe_float(row_copy.get(params_key), float('inf'))
            row_copy['selection_norm_auc_loss'] = normalized.get('auc_loss')
            row_copy['selection_norm_time_s'] = normalized.get('time_s')
            row_copy['selection_norm_params_m'] = normalized.get('trainable_params_m')
            row_copy['selection_stable_id_key'] = stable_id_key
            if report_role_ranked:
                row_copy['report_role'] = report_role_ranked
            ranked_rows.append(row_copy)
    front_rows = [dict(r) for r in ranked_rows if int(r.get('pareto_front_rank', 0)) == 1]
    return ranked_rows, front_rows
