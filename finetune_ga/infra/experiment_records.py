from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from finetune_ga.infra.io_utils import read_jsonl, safe_float, upsert_jsonl
from finetune_ga.selection.multiobjective import (
    FINAL_SELECTION_METHOD,
    FINAL_SELECTION_OBJECTIVES,
    FINAL_SELECTION_RUNTIME_METRICS,
    build_selection_objectives,
    prepare_selection_row,
    pareto_front_rows,
    non_dominated_sort_rows,
    select_by_ideal_point,
)


OBJECTIVE_AUC_KEYS = ('mean_val_auc_loss_obj', 'robust_min_auc_loss_obj')
OBJECTIVE_TIME_KEYS = ('selection_time_s_obj',)
OBJECTIVE_PARAMS_KEYS = ('trainable_params_m_obj',)

RUN_KEY_FIELDS = ('genome_id', 'budget', 'backbone', 'rep')
SUMMARY_KEY_FIELDS = ('tag', 'genome_id', 'budget')


def _dedupe_rows_by_key(rows: Iterable[Mapping[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    seen: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    order: List[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in seen:
            order.append(key)
        seen[key] = dict(row)
    return [seen[key] for key in order]


def dedupe_run_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_rows_by_key(rows, RUN_KEY_FIELDS)


def dedupe_summary_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_rows_by_key(rows, SUMMARY_KEY_FIELDS)


def upsert_summary_row(path: str, row: Dict[str, Any]) -> None:
    upsert_jsonl(path, row, SUMMARY_KEY_FIELDS)


def select_best_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = prepare_selection_row(row, auc_key='best_val_auc', time_key='time_s', params_key='trainable_params_m', runtime_metrics=FINAL_SELECTION_RUNTIME_METRICS)
        prepared.append(row_copy)
    best, _ = select_by_ideal_point(prepared, FINAL_SELECTION_OBJECTIVES)
    return best


def load_done_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, str, int]]:
    done = set()
    for row in rows:
        try:
            done.add((str(row['genome_id']), str(row['budget']), str(row['backbone']), int(row['rep'])))
        except (KeyError, TypeError, ValueError):
            continue
    return done


def load_done_keys_from_path(runs_path: str) -> set[tuple[str, str, str, int]]:
    return load_done_keys(read_jsonl(runs_path))


def extract_objectives(obj: Mapping[str, Any] | None) -> Dict[str, Any]:
    obj = dict(obj or {})
    time_val = next((obj[k] for k in OBJECTIVE_TIME_KEYS if k in obj), None)
    params_val = next((obj[k] for k in OBJECTIVE_PARAMS_KEYS if k in obj), None)
    return {
        'mean_val_auc_loss_obj': obj.get('mean_val_auc_loss_obj'),
        'robust_min_auc_loss_obj': obj.get('robust_min_auc_loss_obj'),
        'selection_time_s_obj': time_val,
        'trainable_params_m_obj': params_val,
    }


def build_summary_row(*, gen_idx: int, tag: str, genome_id: str, budget_name: str, summary: Dict[str, Any], objectives: tuple[float, float, float], best_run: Dict[str, Any], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if 'robust_min_auc' not in summary:
        raise KeyError('build_summary_row requires robust_min_auc in summary to compute robust_min_auc_loss_obj without fallback')
    row = {
        'gen': int(gen_idx),
        'tag': tag,
        'genome_id': genome_id,
        'budget': budget_name,
        'mode': 'summary',
        'status': 'ok',
        'seed': best_run.get('active_seed', best_run.get('seed')),
        'active_seed': best_run.get('active_seed', best_run.get('seed')),
        'backbone': best_run.get('backbone', 'NA'),
        'rep': best_run.get('rep'),
        **summary,
        'best_run_backbone': best_run.get('backbone'),
        'best_run_rep': best_run.get('rep'),
        'best_run_seed': best_run.get('seed'),
        'best_run_val_auc': best_run.get('best_val_auc'),
        'best_run_val_pr_auc': best_run.get('best_val_pr_auc'),
        'best_run_val_accuracy': best_run.get('best_val_accuracy'),
        'best_model_path': best_run.get('best_model_path'),
        'best_weights_path': best_run.get('best_weights_path'),
        'best_history_plot_path': best_run.get('history_plot_path'),
        'best_run_selection_method': FINAL_SELECTION_METHOD,
        'best_run_selection_runtime_metrics': list(FINAL_SELECTION_RUNTIME_METRICS),
        'best_run_selection_details': {
            'method': FINAL_SELECTION_METHOD,
            'objective_direction': 'minimize',
            'objectives': list(FINAL_SELECTION_OBJECTIVES),
            'runtime_metrics': list(FINAL_SELECTION_RUNTIME_METRICS),
            'selection_objectives': dict(best_run.get('selection_objectives', {})),
            'selection_normalized_objectives': dict(best_run.get('selection_normalized_objectives', {})),
            'ideal_distance': best_run.get('ideal_distance'),
        },
        'objectives': {
            'mean_val_auc_loss_obj': float(objectives[0]),
            'robust_min_auc_loss_obj': float(1.0 - float(summary['robust_min_auc'])),
            'selection_time_s_obj': float(objectives[1]),
            'trainable_params_m_obj': float(objectives[2]),
        },
    }
    row.setdefault('selection_time_s_per_run', row.get('mean_time_s_per_run', row.get('total_time_s')))
    row.setdefault('diagnostic_total_search_time_s', row.get('total_time_s'))
    if extra:
        row.update(extra)
    return row
