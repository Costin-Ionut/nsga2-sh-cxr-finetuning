"""analyze_results.py — Aggregate, compare and visualise experiment results.

Plotting helpers live in plot_utils.py.
Statistical helpers (hypervolume, replication_metrics) live in metrics.py.
"""
from __future__ import annotations

import glob
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.experiment_records import dedupe_run_rows, dedupe_summary_rows, extract_objectives
from finetune_ga.analysis.plots import ensure_dir, save_plot, save_boxplot, write_markdown_table
from finetune_ga.infra.io_utils import read_jsonl, prepare_output_paths, safe_float_or_none, atomic_write_json
from finetune_ga.infra.test_protocol import build_test_protocol_id, resolve_test_candidate_tags
from finetune_ga.selection.multiobjective import prepare_selection_row, non_dominated_sort_rows, select_by_ideal_point, rank_candidates_multiobjective

from finetune_ga.core.metrics import hypervolume_2d, hypervolume_3d
from finetune_ga.analysis.metric_contract import metric_contract_dataframe, validate_metric_contract, resolve_metric_name
from finetune_ga.analysis.artifact_contract import artifact_contract_dataframe, get_artifact_specs, REQUIRED_SELECTION_COLUMNS
from finetune_ga.analysis.final_package_audit import run_final_package_audit
from finetune_ga.infra.repro_manifest import build_run_manifest

try:
    from scipy.stats import wilcoxon, mannwhitneyu
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False




def filter_test_rows_for_current_protocol(rows: List[Dict], cfg: Dict) -> List[Dict]:
    allowed_tags = set(resolve_test_candidate_tags(cfg))
    topk = int(cfg.get("test_topk_per_tag", 1))
    protocol_id = build_test_protocol_id(cfg)
    filtered = []
    for row in rows:
        tag = row.get("source_tag")
        if allowed_tags and tag not in allowed_tags:
            continue
        if row.get("test_protocol_id") != protocol_id:
            continue
        try:
            selection_rank = int(row.get("selection_rank", 1))
        except (TypeError, ValueError):
            continue
        if selection_rank > topk:
            continue
        filtered.append(row)
    return filtered

SUMMARY_METRICS = [
    "mean_val_auc", "robust_min_auc", "auc_mean", "auc_std", "pr_auc_mean", "accuracy_mean",
    "loss_mean", "total_time_s", "mean_trainable_params_m", "complexity_score",
    "hypervolume_2d", "hypervolume_3d",
    "failure_rate", "success_rate", "n_input_runs", "n_ok_runs", "n_failed_runs",
]
TEST_METRICS = [
    "test_auc", "test_pr_auc", "test_accuracy", "test_precision", "test_recall",
    "test_specificity", "test_f1", "test_balanced_accuracy", "test_mcc",
    "test_npv", "test_fpr", "test_fnr", "test_brier_score", "test_loss", "test_log_loss",
    "test_ece_10bins", "test_mean_positive_confidence", "test_mean_negative_confidence",
    "test_prevalence", "test_support", "test_tp", "test_tn", "test_fp", "test_fn",
    "test_accuracy_ci95_low", "test_accuracy_ci95_high",
    "test_recall_ci95_low", "test_recall_ci95_high",
    "test_specificity_ci95_low", "test_specificity_ci95_high",
    "test_precision_ci95_low", "test_precision_ci95_high",
    "elapsed_test_train_s", "elapsed_test_eval_s", "elapsed_total_test_pipeline_s",
    "trainable_params_m", "complexity_score",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collect_seed_roots(root_dir, seeds):
    if len(seeds) > 1:
        return [os.path.join(root_dir, f"seed_{int(s)}") for s in seeds
                if os.path.isdir(os.path.join(root_dir, f"seed_{int(s)}"))]
    return [root_dir]

def infer_seed_value(rows: List[Dict], default: int = 0) -> int:
    for row in rows or []:
        if not isinstance(row, dict):
            continue

        for key in ("seed", "active_seed", "best_run_seed", "search_seed", "final_seed"):
            value = row.get(key)
            if value is None or value == "":
                continue

            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue

    return int(default)

def flatten_summary_rows(rows: List[Dict], seed_val: int) -> List[Dict]:
    out = []
    for r in dedupe_summary_rows(rows):
        rr = dict(r)
        rr.setdefault("seed", seed_val)
        obj = rr.pop('objectives', {}) or {}
        if isinstance(obj, dict):
            rr.update(extract_objectives(obj))
        out.append(rr)
    return out


def _safe_count_key(df: pd.DataFrame, preferred: list[str] | tuple[str, ...]) -> str | None:
    for col in preferred:
        if col in df.columns:
            return col
    return None


def sanitize_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _sort_fill_value(column: str) -> float:
    lowered = column.lower()
    minimize_tokens = ("time", "loss", "fpr", "fnr", "brier", "log_loss", "ece")
    if any(token in lowered for token in minimize_tokens):
        return 1e9
    return -1.0


def _prepare_sorted_frame(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    out = sanitize_metric_frame(df)
    for col in sort_cols:
        if col in out.columns:
            out[col] = out[col].fillna(_sort_fill_value(col))
    return out




def ensure_col(df: pd.DataFrame, name: str, default=None) -> pd.DataFrame:
    if df is not None and name not in df.columns:
        df[name] = default
    return df

def add_time_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if 'selection_time_s_per_run' not in out.columns:
        if 'mean_time_s_per_run' in out.columns:
            out['selection_time_s_per_run'] = out['mean_time_s_per_run']
        elif 'total_time_s' in out.columns:
            out['selection_time_s_per_run'] = out['total_time_s']
    if 'diagnostic_total_search_time_s' not in out.columns and 'total_time_s' in out.columns:
        out['diagnostic_total_search_time_s'] = out['total_time_s']
    return out

def aggregate_frame(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    use_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not use_metrics:
        return pd.DataFrame()
    df = sanitize_metric_frame(df)
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.size().reset_index(name='count_runs')
    for metric in use_metrics:
        series = grouped[metric]
        stats = series.agg(['mean', 'std', 'min', 'max', 'median', 'count']).reset_index()
        stats.rename(columns={'count': f'{metric}_count'}, inplace=True)
        stats[f'{metric}_std'] = stats['std'] if 'std' in stats.columns else np.nan
        se = stats[f'{metric}_std'] / np.sqrt(stats[f'{metric}_count'].replace(0, np.nan))
        stats[f'{metric}_ci95_low'] = stats['mean'] - 1.96 * se
        stats[f'{metric}_ci95_high'] = stats['mean'] + 1.96 * se
        stats = stats.drop(columns=['std'])
        stats.rename(columns={
            'mean': f'{metric}_mean',
            'min': f'{metric}_min',
            'max': f'{metric}_max',
            'median': f'{metric}_median',
        }, inplace=True)
        out = out.merge(stats, on=group_cols, how='left')
    return out


def best_row(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in sort_cols if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame()
    ascending = [
        True if any(k in c for k in ("time", "loss", "fpr", "fnr", "brier", "log_loss", "ece")) else False
        for c in cols
    ]
    sortable = _prepare_sorted_frame(df, cols)
    return sortable.sort_values(cols, ascending=ascending).head(1)


def cliffs_delta(x, y):
    """Non-parametric effect size (Cliff's delta) for two unpaired samples.

    Returns a value in [-1, 1].  Thresholds (Romano et al., 2006):
    |d| < 0.147 = negligible, < 0.33 = small, < 0.474 = medium, else large.
    """
    x, y = list(x), list(y)
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    greater = sum(1 for xi in x for yj in y if xi > yj)
    less    = sum(1 for xi in x for yj in y if xi < yj)
    return (greater - less) / n




def _safe_float(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return safe_float_or_none(value)


def _format_number(value, decimals: int = 4) -> str:
    val = _safe_float(value)
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _metric_cell(row: pd.Series, metric: str, decimals: int = 4, ci_metric: str | None = None) -> str:
    mean_val = _safe_float(row.get(f"{metric}_mean"))
    if mean_val is None:
        return "—"
    parts = [f"{mean_val:.{decimals}f}"]
    std_val = _safe_float(row.get(f"{metric}_std"))
    if std_val is not None:
        parts.append(f"± {std_val:.{decimals}f}")
    median_val = _safe_float(row.get(f"{metric}_median"))
    if median_val is not None:
        parts.append(f"med {_format_number(median_val, decimals)}")
    min_val = _safe_float(row.get(f"{metric}_min"))
    max_val = _safe_float(row.get(f"{metric}_max"))
    if min_val is not None and max_val is not None:
        parts.append(f"range [{min_val:.{decimals}f}, {max_val:.{decimals}f}]")
    if ci_metric:
        lo = _safe_float(row.get(f"{ci_metric}_mean"))
        hi = _safe_float(row.get(f"{ci_metric.replace('_low', '_high')}_mean"))
        if lo is not None and hi is not None:
            parts.append(f"CI95 [{lo:.{decimals}f}, {hi:.{decimals}f}]")
    return " | ".join(parts)


def _interval_cell(row: pd.Series, low_metric: str, high_metric: str | None = None, decimals: int = 4) -> str:
    high_metric = high_metric or low_metric.replace('_low', '_high')
    low_mean = _safe_float(row.get(f"{low_metric}_mean"))
    high_mean = _safe_float(row.get(f"{high_metric}_mean"))
    if low_mean is None or high_mean is None:
        return "—"
    parts = [f"[{low_mean:.{decimals}f}, {high_mean:.{decimals}f}]"]
    low_min = _safe_float(row.get(f"{low_metric}_min"))
    high_max = _safe_float(row.get(f"{high_metric}_max"))
    if low_min is not None and high_max is not None:
        parts.append(f"span [{low_min:.{decimals}f}, {high_max:.{decimals}f}]")
    low_median = _safe_float(row.get(f"{low_metric}_median"))
    high_median = _safe_float(row.get(f"{high_metric}_median"))
    if low_median is not None and high_median is not None:
        parts.append(f"med [{low_median:.{decimals}f}, {high_median:.{decimals}f}]")
    return " | ".join(parts)


def build_pretty_test_tables(grouped_by_method: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if grouped_by_method.empty:
        return {}
    method_col = "source_tag" if "source_tag" in grouped_by_method.columns else grouped_by_method.columns[0]
    base = grouped_by_method[[method_col]].copy()
    base.rename(columns={method_col: "Method"}, inplace=True)

    main = base.copy()
    main["AUC"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_auc"), axis=1)
    main["PR-AUC"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_pr_auc"), axis=1)
    main["Accuracy"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_accuracy", ci_metric="test_accuracy_ci95_low"), axis=1)
    main["Precision"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_precision", ci_metric="test_precision_ci95_low"), axis=1)
    main["Recall"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_recall", ci_metric="test_recall_ci95_low"), axis=1)
    main["Specificity"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_specificity", ci_metric="test_specificity_ci95_low"), axis=1)
    main["F1"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_f1"), axis=1)
    main["Balanced Acc."] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_balanced_accuracy"), axis=1)
    main["MCC"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_mcc"), axis=1)

    efficiency = base.copy()
    efficiency["Train time (s)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "elapsed_test_train_s"), axis=1)
    efficiency["Eval time (s)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "elapsed_test_eval_s"), axis=1)
    efficiency["Total pipeline time (s)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "elapsed_total_test_pipeline_s"), axis=1)
    efficiency["Trainable params (M)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "trainable_params_m"), axis=1)
    efficiency["Complexity"] = grouped_by_method.apply(lambda r: _metric_cell(r, "complexity_score"), axis=1)

    calibration = base.copy()
    calibration["Brier score"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_brier_score"), axis=1)
    calibration["Compiled loss"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_loss"), axis=1)
    calibration["Log loss"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_log_loss"), axis=1)
    calibration["ECE (10 bins)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_ece_10bins"), axis=1)
    calibration["Mean positive conf."] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_mean_positive_confidence"), axis=1)
    calibration["Mean negative conf."] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_mean_negative_confidence"), axis=1)
    calibration["NPV"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_npv"), axis=1)
    calibration["FPR"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_fpr"), axis=1)
    calibration["FNR"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_fnr"), axis=1)

    uncertainty = base.copy()
    uncertainty["CI95 semantics"] = "Summary of per-run Wilson interval bounds; diagnostic, not CI of aggregated method mean"
    uncertainty["Accuracy CI95 interval summary"] = grouped_by_method.apply(lambda r: _interval_cell(r, "test_accuracy_ci95_low"), axis=1)
    uncertainty["Precision CI95 interval summary"] = grouped_by_method.apply(lambda r: _interval_cell(r, "test_precision_ci95_low"), axis=1)
    uncertainty["Recall CI95 interval summary"] = grouped_by_method.apply(lambda r: _interval_cell(r, "test_recall_ci95_low"), axis=1)
    uncertainty["Specificity CI95 interval summary"] = grouped_by_method.apply(lambda r: _interval_cell(r, "test_specificity_ci95_low"), axis=1)

    confusion = base.copy()
    confusion["TP"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_tp"), axis=1)
    confusion["TN"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_tn"), axis=1)
    confusion["FP"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_fp"), axis=1)
    confusion["FN"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_fn"), axis=1)
    confusion["Prevalence"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_prevalence"), axis=1)
    confusion["Support"] = grouped_by_method.apply(lambda r: _metric_cell(r, "test_support"), axis=1)

    return {
        "table_main_performance.md": main,
        "table_efficiency.md": efficiency,
        "table_calibration.md": calibration,
        "table_uncertainty.md": uncertainty,
        "table_confusion.md": confusion,
    }


def build_pretty_summary_tables(grouped_by_method: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if grouped_by_method.empty:
        return {}
    method_col = "tag" if "tag" in grouped_by_method.columns else grouped_by_method.columns[0]
    base = grouped_by_method[[method_col]].copy()
    base.rename(columns={method_col: "Method"}, inplace=True)

    search = base.copy()
    search["Mean val AUC"] = grouped_by_method.apply(lambda r: _metric_cell(r, "mean_val_auc"), axis=1)
    search["Robust min AUC (diagnostic)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "robust_min_auc"), axis=1)
    search["AUC mean (diagnostic)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "auc_mean"), axis=1)
    search["PR-AUC mean"] = grouped_by_method.apply(lambda r: _metric_cell(r, "pr_auc_mean"), axis=1)
    search["Accuracy mean"] = grouped_by_method.apply(lambda r: _metric_cell(r, "accuracy_mean"), axis=1)
    search["Search time (s)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "total_time_s"), axis=1)
    search["Trainable params (M)"] = grouped_by_method.apply(lambda r: _metric_cell(r, "mean_trainable_params_m"), axis=1)
    search["Complexity"] = grouped_by_method.apply(lambda r: _metric_cell(r, "complexity_score"), axis=1)
    search["Hypervolume 2D"] = grouped_by_method.apply(lambda r: _metric_cell(r, "hypervolume_2d"), axis=1)
    search["Hypervolume 3D"] = grouped_by_method.apply(lambda r: _metric_cell(r, "hypervolume_3d"), axis=1)
    return {"table_search_summary.md": search}


def build_pretty_backbone_table(grouped_by_backbone: pd.DataFrame) -> pd.DataFrame:
    if grouped_by_backbone.empty:
        return pd.DataFrame()
    base = grouped_by_backbone[["backbone"]].copy() if "backbone" in grouped_by_backbone.columns else grouped_by_backbone.iloc[:, :1].copy()
    base.columns = ["Backbone"]
    base["AUC"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "test_auc"), axis=1)
    base["Accuracy"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "test_accuracy"), axis=1)
    base["F1"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "test_f1"), axis=1)
    base["MCC"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "test_mcc"), axis=1)
    base["Total pipeline time (s)"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "elapsed_total_test_pipeline_s"), axis=1)
    base["Trainable params (M)"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "trainable_params_m"), axis=1)
    base["Complexity"] = grouped_by_backbone.apply(lambda r: _metric_cell(r, "complexity_score"), axis=1)
    return base


def _existing_subset(columns: list[str] | tuple[str, ...], df: pd.DataFrame) -> list[str]:
    return [c for c in columns if c in df.columns]


def dedupe_report_rows(df: pd.DataFrame, subset: list[str] | tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    use_subset = _existing_subset(subset, df)
    if not use_subset:
        return df
    return df.drop_duplicates(subset=use_subset, keep='last').reset_index(drop=True)


def _float_csv_kwargs() -> dict:
    return {'index': False, 'float_format': '%.6f'}


def save_report_csv(df: pd.DataFrame, path: str | Path) -> None:
    add_time_aliases(sanitize_metric_frame(df)).to_csv(path, **_float_csv_kwargs())


def _load_strict_jsonl(path: str) -> list[dict]:
    return read_jsonl(path, strict=True)


def _report_manifest_payload(scope: str, *, analysis_dir: str, min_runs: int | None = None) -> dict:
    try:
        run_manifest = build_run_manifest()
        manifest_status = 'ok'
    except (OSError, ValueError):
        run_manifest = {}
        manifest_status = 'failed'
    payload = {
        'scope': scope,
        'analysis_dir': str(analysis_dir),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'min_runs_for_ranking': min_runs,
        'git_commit_hash': run_manifest.get('git_commit_hash'),
        'config_sha256': run_manifest.get('config_sha256'),
        'search_seeds': run_manifest.get('search_seeds'),
        'final_seeds': run_manifest.get('final_seeds'),
        'protocol_id': run_manifest.get('protocol_id'),
        'manifest_status': manifest_status,
    }
    return payload


def write_reporting_manifest(target_dir: str | Path, scope: str, *, min_runs: int | None = None) -> None:
    target = Path(target_dir) / 'report_manifest.json'
    atomic_write_json(str(target), _report_manifest_payload(scope, analysis_dir=str(target_dir), min_runs=min_runs), validate_finite=True)


def save_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, path: str, label_col: str | None = None) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plot_df = df[[c for c in [x_col, y_col, label_col] if c and c in df.columns]].dropna()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col])
    if label_col and label_col in plot_df.columns:
        for _, row in plot_df.iterrows():
            plt.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=8, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(path)


# ---------------------------------------------------------------------------
# Statistical comparisons
# ---------------------------------------------------------------------------

def _resolved_eval_seed_series(df: pd.DataFrame) -> pd.Series:
    if 'final_eval_seed' in df.columns:
        return df['final_eval_seed']
    if 'seed' in df.columns:
        return df['seed']
    return pd.Series([None] * len(df), index=df.index, dtype='object')


def _resolved_search_seed_series(df: pd.DataFrame) -> pd.Series:
    if 'search_seed' in df.columns:
        return df['search_seed']
    if 'seed' in df.columns:
        return df['seed']
    return pd.Series([None] * len(df), index=df.index, dtype='object')


def _resolved_final_train_seed_series(df: pd.DataFrame) -> pd.Series:
    if 'final_train_seed' in df.columns:
        return df['final_train_seed']
    if 'seed' in df.columns:
        return df['seed']
    return pd.Series([None] * len(df), index=df.index, dtype='object')


def _attach_pairing_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out['resolved_eval_seed'] = _resolved_eval_seed_series(out)
    out['resolved_search_seed'] = _resolved_search_seed_series(out)
    out['resolved_final_train_seed'] = _resolved_final_train_seed_series(out)
    return out


def _pairing_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ['source_tag', 'resolved_search_seed', 'resolved_final_train_seed', 'resolved_eval_seed']
    return [c for c in preferred if c in df.columns]


def run_statistical_comparisons(test_df, reference_tag: str, metric: str, analysis_dir: str) -> None:
    """Pairwise Wilcoxon signed-rank tests against a reference method."""
    if not _SCIPY_AVAILABLE:
        print("[WARN] scipy not installed — skipping statistical tests. pip install scipy")
        return
    if test_df.empty or metric not in test_df.columns:
        return

    methods = sorted(test_df["source_tag"].dropna().unique())
    if reference_tag not in methods:
        reference_tag = methods[0] if methods else None
    if reference_tag is None:
        return

    test_df = _attach_pairing_keys(test_df)
    pair_cols = _pairing_group_columns(test_df)
    ref_vals_by_seed = {}
    for group_key, g in test_df.groupby(pair_cols, dropna=False):
        method = group_key[0] if isinstance(group_key, tuple) else group_key
        pair_key = tuple(group_key[1:]) if isinstance(group_key, tuple) else tuple()
        if method == reference_tag:
            vals = g[metric].dropna()
            if len(vals) > 0:
                ref_vals_by_seed[pair_key] = float(vals.mean())

    rows = []
    n_comparisons = len([m for m in methods if m != reference_tag])
    bonferroni_alpha = 0.05 / max(1, n_comparisons)

    for method in methods:
        if method == reference_tag:
            continue
        cmp_vals_by_seed = {}
        for group_key, g in test_df.groupby(pair_cols, dropna=False):
            m = group_key[0] if isinstance(group_key, tuple) else group_key
            pair_key = tuple(group_key[1:]) if isinstance(group_key, tuple) else tuple()
            if m == method:
                vals = g[metric].dropna()
                if len(vals) > 0:
                    cmp_vals_by_seed[pair_key] = float(vals.mean())
        common_seeds = sorted(set(ref_vals_by_seed) & set(cmp_vals_by_seed))
        ref_flat = [ref_vals_by_seed[s] for s in common_seeds]
        cmp_flat = [cmp_vals_by_seed[s] for s in common_seeds]
        if not ref_flat or not cmp_flat:
            continue
        delta = cliffs_delta(ref_flat, cmp_flat)
        try:
            if len(ref_flat) == len(cmp_flat) and len(ref_flat) >= 2:
                stat, p = wilcoxon(ref_flat, cmp_flat, alternative="two-sided")
                test_name = "wilcoxon_signed_rank"
            elif len(ref_flat) >= 3 and len(cmp_flat) >= 3:
                stat, p = mannwhitneyu(ref_flat, cmp_flat, alternative="two-sided")
                test_name = "mann_whitney_u"
            else:
                p, stat, test_name = float("nan"), float("nan"), "insufficient_data"
        except ValueError:
            p, stat, test_name = float("nan"), float("nan"), "insufficient_data"
        effect_label = (
            "negligible" if abs(delta) < 0.147 else
            "small"      if abs(delta) < 0.330 else
            "medium"     if abs(delta) < 0.474 else "large"
        )
        rows.append({
            "reference": reference_tag, "comparison": method, "metric": metric,
            "test": test_name, "n_ref": len(ref_flat), "n_cmp": len(cmp_flat),
            "ref_median": float(np.median(ref_flat)), "cmp_median": float(np.median(cmp_flat)),
            "cliffs_delta": round(delta, 4), "effect_size": effect_label,
            "p_value": round(float(p), 6), "bonferroni_alpha": round(bonferroni_alpha, 6),
            "significant_bonferroni": bool(float(p) < bonferroni_alpha),
        })
    if rows:
        out_path = os.path.join(analysis_dir, f"statistical_tests_{metric}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Statistical tests ({metric}) -> {out_path}")



def build_multiobjective_summary_reports(
    summary_df: pd.DataFrame,
    *,
    report_role_candidate: str = 'multiobjective_validation_candidate',
    report_role_ranked: str = 'multiobjective_validation_ranked',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_df = add_time_aliases(summary_df)
    ranked_rows, front_rows = rank_candidates_multiobjective(
        summary_df.to_dict(orient='records'),
        auc_key='mean_val_auc',
        time_key='selection_time_s_per_run',
        params_key='mean_trainable_params_m',
        report_role_candidate=report_role_candidate,
        report_role_ranked=report_role_ranked,
        tie_break_name_key='genome_id',
        stable_id_key='genome_id',
    )
    ranked_df = add_time_aliases(pd.DataFrame(ranked_rows))
    if not ranked_df.empty:
        ranked_df['selection_metric_scope'] = 'validation_selection_contract'
        ranked_df['selection_auc_key'] = 'mean_val_auc'
        ranked_df['selection_time_key'] = 'selection_time_s_per_run'
        ranked_df['selection_params_key'] = 'mean_trainable_params_m'
        ranked_df['selection_metric_roles'] = 'selection_primary,selection_cost,selection_cost'
        ranked_df['selection_summary_note'] = 'Rows are Pareto-ranked on validation metrics only; auxiliary metrics are descriptive only.'
        ranked_df['auc_loss'] = ranked_df['selection_auc_loss']
        ranked_df['mean_val_auc_loss_obj'] = ranked_df['selection_auc_loss']
        ranked_df['selection_time_s_obj'] = ranked_df['selection_time_s']
        ranked_df['trainable_params_m_obj'] = ranked_df['selection_params_m']
    front_df = add_time_aliases(pd.DataFrame(front_rows))
    if not front_df.empty:
        front_df['selection_metric_scope'] = 'validation_selection_contract'
        front_df['selection_summary_note'] = 'Front 1 only. Validation Pareto candidates used for downstream selection.'
        front_df['auc_loss'] = front_df['selection_auc_loss']
        front_df['mean_val_auc_loss_obj'] = front_df['selection_auc_loss']
        front_df['selection_time_s_obj'] = front_df['selection_time_s']
        front_df['trainable_params_m_obj'] = front_df['selection_params_m']
    return ranked_df, front_df


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def _missing_columns(df: pd.DataFrame, required: list[str] | tuple[str, ...]) -> list[str]:
    return [c for c in required if c not in df.columns]


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _strict_metric_issues_for_report(df: pd.DataFrame, filename: str) -> list[str]:
    issues: list[str] = []
    metadata_columns = {
    'genome_id',
    'budget',
    'budget_name',
    'tag',
    'seed',
    'active_seed',
    'backbone',
    'rank',
    'source_tag',
    'analysis_scope',
    'time_metric_scope',
    'selection_metric_scope',
    'selection_summary_note',
    'report_role',
    'pareto_front_rank',
    'pareto_within_front_rank',
    'ideal_distance',
    'auc_loss',
    'selection_time_s_obj',
    'trainable_params_m_obj',
    'mode',
    'status',
    'ranking_eligible',
    'min_runs_required',
    'raw_row_count',
    'count_runs',
    'source_file',
    'selection_auc_key',
    'selection_time_key',
    'selection_params_key',
    'selection_metric_roles',
    'selection_objectives',
    'selection_normalized_objectives',
    'selection_stable_id_key',
    'selection_method',
    'selection_runtime_metrics',
}
    if filename.startswith('selection_'):
        allowed_roles = {
            'selection_primary',
            'metadata',
            'selection_cost',
            'diagnostic_robustness',
            'diagnostic_performance',
            'diagnostic_cost',
            'test_cost_diagnostic',
        }
    elif filename.startswith('diagnostic_'):
        allowed_roles = {
            'selection_primary',
            'metadata',
            'selection_cost',
            'diagnostic_robustness',
            'diagnostic_performance',
            'diagnostic_cost',
            'test_primary',
            'test_calibration',
            'test_diagnostic',
            'test_uncertainty',
            'test_cost_diagnostic',
            
        }
    elif filename.startswith('test_diagnostic_'):
        allowed_roles = {
            'test_primary',
            'metadata',
            'test_calibration',
            'test_diagnostic',
            'test_uncertainty',
            'test_cost_diagnostic',
        }
    elif filename in {'validation_comparison_by_method.csv'}:
        allowed_roles = {
            'selection_primary',
            'metadata',
            'selection_cost',
            'diagnostic_performance',
            'diagnostic_robustness',
            'diagnostic_cost',
            'test_cost_diagnostic',
        }
    elif filename in {'test_comparison_by_method.csv'}:
        allowed_roles = {
            'test_primary',
            'metadata',
            'test_calibration',
            'test_diagnostic',
            'test_uncertainty',
            'test_cost_diagnostic',
        }
    elif filename in {'diagnostic_method_ranking.csv'}:
        allowed_roles = {
            'test_primary',
            'metadata',
            'test_calibration',
            'test_diagnostic',
            'test_uncertainty',
            'test_cost_diagnostic',
            'diagnostic_robustness',
            'diagnostic_cost',
            'diagnostic_performance',
        }
    else:
        return issues

    candidate_metrics: list[str] = []
    for col in df.columns:
        if col in metadata_columns:
            continue
        candidate_metrics.append(col)
    issues.extend(validate_metric_contract(candidate_metrics, allowed_roles=allowed_roles, strict=True))
    return sorted(set(issues))


def _validate_report_roles(df: pd.DataFrame, filename: str) -> list[str]:
    issues: list[str] = []
    if 'report_role' not in df.columns:
        return issues
    roles = set(df['report_role'].dropna().astype(str))
    if filename.startswith('selection_'):
        bad_roles = sorted(r for r in roles if not r.startswith('multiobjective_validation_'))
        if bad_roles:
            issues.append(f'invalid_selection_report_role={filename}:' + ','.join(bad_roles))
    elif filename.startswith('diagnostic_'):
        bad_roles = sorted(r for r in roles if not r.startswith('descriptive_') and not r.startswith('diagnostic_'))
        if bad_roles:
            issues.append(f'invalid_diagnostic_report_role={filename}:' + ','.join(bad_roles))
    elif filename.startswith('test_diagnostic_'):
        bad_roles = sorted(r for r in roles if not r.startswith('test_diagnostic_'))
        if bad_roles:
            issues.append(f'invalid_test_diagnostic_report_role={filename}:' + ','.join(bad_roles))
    elif filename == 'validation_comparison_by_method.csv':
        bad_roles = sorted(r for r in roles if r != 'diagnostic_validation_method_comparison')
        if bad_roles:
            issues.append(f'invalid_global_validation_report_role={filename}:' + ','.join(bad_roles))
    elif filename == 'test_comparison_by_method.csv':
        bad_roles = sorted(r for r in roles if r != 'diagnostic_test_method_comparison')
        if bad_roles:
            issues.append(f'invalid_global_test_report_role={filename}:' + ','.join(bad_roles))
    elif filename == 'diagnostic_method_ranking.csv':
        bad_roles = sorted(r for r in roles if r != 'diagnostic_test_metric_ranking')
        if bad_roles:
            issues.append(f'invalid_global_ranking_report_role={filename}:' + ','.join(bad_roles))
    return issues


def _write_artifact_contract_registry(target_dir: Path, visible_artifacts: list[str]) -> None:
    artifact_df = artifact_contract_dataframe(visible_artifacts)
    save_report_csv(artifact_df, target_dir / 'artifact_contract_registry.csv')
    write_markdown_table(artifact_df, target_dir / 'artifact_contract_registry.md', 'Artifact Contract Registry')


def write_seed_reporting_audit(analysis_dir: str, *, summary_df: pd.DataFrame | None = None,
                               pareto_ranked_df: pd.DataFrame | None = None, test_pareto_df: pd.DataFrame | None = None) -> None:
    issues: list[str] = []
    checks: list[str] = []
    analysis_path = Path(analysis_dir)

    visible_artifacts = [p.name for p in analysis_path.glob('*.csv')]
    _write_artifact_contract_registry(analysis_path, visible_artifacts)

    contract_df = metric_contract_dataframe((summary_df.columns if summary_df is not None else []))
    save_report_csv(contract_df, analysis_path / 'metric_contract_registry.csv')
    write_markdown_table(contract_df, analysis_path / 'metric_contract_registry.md', 'Metric Contract Registry')

    if summary_df is not None and not summary_df.empty:
        checks.append(f'summary_rows={len(summary_df)}')
    if pareto_ranked_df is not None and not pareto_ranked_df.empty:
        missing = _missing_columns(pareto_ranked_df, REQUIRED_SELECTION_COLUMNS)
        if missing:
            issues.append('selection_report_missing_columns=' + ','.join(missing))
        order_view = pareto_ranked_df[['pareto_front_rank', 'pareto_within_front_rank']].copy()
        sorted_view = order_view.sort_values(['pareto_front_rank', 'pareto_within_front_rank']).reset_index(drop=True)
        if not order_view.reset_index(drop=True).equals(sorted_view):
            issues.append('selection_report_not_sorted_by_pareto_ranks')
        checks.append(f'selection_rows={len(pareto_ranked_df)}')
    if test_pareto_df is not None and not test_pareto_df.empty:
        expected_scope = {'test_only_diagnostic_not_used_for_selection'}
        expected_time_scope = {'test_pipeline_runtime_seconds_not_comparable_to_validation_selection_time'}
        values = set(test_pareto_df.get('analysis_scope', pd.Series(dtype=str)).dropna().astype(str))
        if values != expected_scope:
            issues.append('test_tradeoff_invalid_analysis_scope=' + ','.join(sorted(values)))
        time_scope_values = set(test_pareto_df.get('time_metric_scope', pd.Series(dtype=str)).dropna().astype(str))
        if time_scope_values != expected_time_scope:
            issues.append('test_tradeoff_invalid_time_metric_scope=' + ','.join(sorted(time_scope_values)))
        checks.append(f'test_tradeoff_rows={len(test_pareto_df)}')

    for spec in get_artifact_specs():
        if spec.category == 'global':
            continue
        search_paths = [analysis_path / spec.name, analysis_path.parent / 'best' / spec.name]
        existing = next((p for p in search_paths if p.exists()), None)
        if existing is None:
            issues.append(f'missing_canonical_report={spec.name}')
            continue
        df = _load_csv_if_exists(existing)
        if df is None or df.empty:
            issues.append(f'empty_canonical_report={spec.name}')
            continue
        missing = _missing_columns(df, spec.required_columns)
        if missing:
            issues.append(f'canonical_report_missing_columns={spec.name}:' + ','.join(missing))
            continue
        issues.extend(_validate_report_roles(df, spec.name))
        issues.extend(_strict_metric_issues_for_report(df, spec.name))
        checks.append(f'validated_report={spec.name}')

    lines = ['AUDIT STATUS: PASS' if not issues else 'AUDIT STATUS: FAIL', '']
    if checks:
        lines.append('Checks:')
        lines.extend(f'- {c}' for c in checks)
        lines.append('')
    if issues:
        lines.append('Issues:')
        lines.extend(f'- {i}' for i in issues)
    else:
        lines.append('Issues: none')
    (analysis_path / 'AUDIT_SEED_REPORTING.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if issues:
        print('[WARN] Seed reporting audit found issues.')


def write_global_reporting_audit(global_dir: str) -> None:
    issues: list[str] = []
    checks: list[str] = []
    global_path = Path(global_dir)

    visible_artifacts = [p.name for p in global_path.glob('*.csv')]
    _write_artifact_contract_registry(global_path, visible_artifacts)

    for spec in get_artifact_specs(category='global'):
        existing = global_path / spec.name
        if not existing.exists():
            issues.append(f'missing_global_report={spec.name}')
            continue
        df = _load_csv_if_exists(existing)
        if df is None or df.empty:
            issues.append(f'empty_global_report={spec.name}')
            continue
        missing = _missing_columns(df, spec.required_columns)
        if missing:
            issues.append(f'global_report_missing_columns={spec.name}:' + ','.join(missing))
            continue
        issues.extend(_validate_report_roles(df, spec.name))
        issues.extend(_strict_metric_issues_for_report(df, spec.name))
        checks.append(f'validated_global_report={spec.name}')

    lines = ['AUDIT STATUS: PASS' if not issues else 'AUDIT STATUS: FAIL', '']
    if checks:
        lines.append('Checks:')
        lines.extend(f'- {c}' for c in checks)
        lines.append('')
    if issues:
        lines.append('Issues:')
        lines.extend(f'- {i}' for i in issues)
    else:
        lines.append('Issues: none')
    (global_path / 'AUDIT_GLOBAL_REPORTING.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if issues:
        print('[WARN] Global reporting audit found issues.')

# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


def build_failure_audit(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate failure/infeasible reporting so failed runs are visible."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "tag", "seed", "genome_id", "budget", "n_input_runs", "n_ok_runs",
        "n_failed_runs", "failure_rate", "success_rate", "status_counts",
        "failure_reason_counts",
    ] if c in summary_df.columns]
    if not cols:
        return pd.DataFrame()
    out = summary_df[cols].copy()
    for col in ["n_input_runs", "n_ok_runs", "n_failed_runs"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["failure_rate", "success_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [c for c in ["tag", "seed", "budget", "failure_rate"] if c in out.columns]
    ascending = [True, True, True, False][:len(sort_cols)]
    return out.sort_values(sort_cols, ascending=ascending) if sort_cols else out
def generate_summary_reports(seed_root: str, analysis_dir: str, summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_df = dedupe_report_rows(summary_df, ["tag", "seed", "genome_id", "budget"])
    save_report_csv(summary_df, os.path.join(analysis_dir, "summary_all.csv"))
    failure_audit_df = build_failure_audit(summary_df)
    if not failure_audit_df.empty:
        save_report_csv(failure_audit_df, os.path.join(analysis_dir, "failure_audit_by_candidate.csv"))

    per_tag_seed = []
    for (tag, seed), g in summary_df.groupby(["tag", "seed"]):
        pts3 = list(zip(g["mean_val_auc_loss_obj"], g["selection_time_s_obj"], g["trainable_params_m_obj"])) if {"mean_val_auc_loss_obj", "selection_time_s_obj", "trainable_params_m_obj"}.issubset(g.columns) else []
        pts2 = list(zip(g["mean_val_auc_loss_obj"], g["selection_time_s_obj"])) if {"mean_val_auc_loss_obj", "selection_time_s_obj"}.issubset(g.columns) else []
        time_ref = max(g["selection_time_s_obj"].max() * 1.05, 1.0) if "selection_time_s_obj" in g.columns else 1.0
        params_ref = max(g["trainable_params_m_obj"].max() * 1.05, 1.0) if "trainable_params_m_obj" in g.columns else 1.0
        per_tag_seed.append({
            "tag": tag, "seed": int(seed), "n_candidates": int(len(g)),
            "best_robust_min_auc": float(g["robust_min_auc"].max()),
            "mean_robust_min_auc": float(g["robust_min_auc"].mean()),
            "std_robust_min_auc": float(g["robust_min_auc"].std(ddof=0)),
            "mean_auc_mean": float(g["auc_mean"].mean()),
            "mean_auc_std": float(g["auc_std"].mean()),
            "mean_complexity_score": float(g["complexity_score"].mean()),
            "mean_total_time_s": float(g["total_time_s"].mean()),
            "hypervolume_2d": float(hypervolume_2d(pts2, ref=(1.0, time_ref))) if pts2 else 0.0,
            "hypervolume_3d": float(hypervolume_3d(pts3, ref=(1.0, time_ref, params_ref))) if pts3 else 0.0,
        })
    per_tag_seed_df = pd.DataFrame(per_tag_seed).sort_values(["tag", "seed"])
    save_report_csv(per_tag_seed_df, os.path.join(analysis_dir, "aggregate_metrics_per_seed.csv"))
    save_report_csv(aggregate_frame(per_tag_seed_df, ["tag"], [c for c in per_tag_seed_df.columns if c not in {"tag", "seed", "n_candidates"}]), os.path.join(analysis_dir, "aggregate_metrics_across_seeds.csv"))

    best_dir = prepare_output_paths(seed_root, "analysis")["best_dir"]
    summary_df.sort_values(["mean_val_auc", "auc_mean", "robust_min_auc", "pr_auc_mean", "accuracy_mean"], ascending=False).head(50).assign(report_role='descriptive_auc_ranking').pipe(lambda _df: save_report_csv(_df, os.path.join(best_dir, "diagnostic_validation_auc_ranked_candidates.csv")))
    summary_df.sort_values(["mean_val_auc", "auc_mean", "robust_min_auc"], ascending=False).groupby("tag", as_index=False).head(10).assign(report_role='descriptive_auc_ranking_per_tag').pipe(lambda _df: save_report_csv(_df, os.path.join(analysis_dir, "diagnostic_validation_auc_ranked_per_tag.csv")))

    pareto_ranked_df, pareto_front_df = build_multiobjective_summary_reports(summary_df)
    if not pareto_ranked_df.empty:
        save_report_csv(pareto_ranked_df, os.path.join(best_dir, 'selection_validation_pareto_ranked_candidates.csv'))
    if not pareto_front_df.empty:
        save_report_csv(pareto_front_df, os.path.join(analysis_dir, 'selection_validation_pareto_front.csv'))

    grouped_summary = aggregate_frame(summary_df, ["tag"], SUMMARY_METRICS)
    save_report_csv(grouped_summary, os.path.join(analysis_dir, "summary_grouped_by_tag.csv"))
    for filename, pretty_df in build_pretty_summary_tables(grouped_summary).items():
        write_markdown_table(pretty_df, os.path.join(analysis_dir, filename), filename.replace("_", " ").replace(".md", "").title())

    plt.figure(figsize=(8, 5))
    for tag, g in summary_df.groupby("tag"):
        plt.scatter(g["total_time_s"] / 60.0, g["mean_val_auc"], label=tag, alpha=0.65)
    plt.xlabel("Total search time [min]")
    plt.ylabel("Mean validation AUC")
    plt.legend()
    save_plot(os.path.join(analysis_dir, "selection_mean_val_auc_vs_time.png"))

    plt.figure(figsize=(8, 5))
    for tag, g in summary_df.groupby("tag"):
        plt.scatter(g["mean_trainable_params_m"], g["mean_val_auc"], label=tag, alpha=0.65)
    plt.xlabel("Trainable params [M]")
    plt.ylabel("Mean validation AUC")
    plt.legend()
    save_plot(os.path.join(analysis_dir, "selection_mean_val_auc_vs_complexity.png"))

    diag_box_data, diag_box_labels = [], []
    for tag, g in summary_df.groupby("tag"):
        vals = g["robust_min_auc"].dropna().values
        if len(vals) > 0:
            diag_box_data.append(vals)
            diag_box_labels.append(tag)
    save_boxplot(diag_box_data, diag_box_labels, "Robust min AUC (diagnostic)", os.path.join(analysis_dir, "diagnostic_robust_min_auc_boxplot.png"))
    if "mean_val_auc" in summary_df.columns:
        mv_box_data, mv_box_labels = [], []
        for tag, g in summary_df.groupby("tag"):
            vals = g["mean_val_auc"].dropna().values
            if len(vals) > 0:
                mv_box_data.append(vals)
                mv_box_labels.append(tag)
        save_boxplot(mv_box_data, mv_box_labels, "Mean validation AUC", os.path.join(analysis_dir, "selection_mean_val_auc_boxplot.png"))
    return pareto_ranked_df, pareto_front_df


def generate_test_reports(seed_root: str, analysis_dir: str, test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()
    test_df = dedupe_report_rows(test_df, ["genome_id", "search_seed", "final_train_seed", "final_eval_seed", "source_tag", "selection_rank", "backbone"])
    test_df = sanitize_metric_frame(test_df)
    save_report_csv(test_df, os.path.join(analysis_dir, "test_results_shared.csv"))
    grouped_metrics = aggregate_frame(test_df, ["source_tag", "backbone"], TEST_METRICS)
    grouped_by_tag = aggregate_frame(test_df, ["source_tag"], TEST_METRICS)
    grouped_by_backbone = aggregate_frame(test_df, ["backbone"], TEST_METRICS)
    save_report_csv(grouped_metrics, os.path.join(analysis_dir, "test_grouped_metrics.csv"))
    save_report_csv(grouped_by_tag, os.path.join(analysis_dir, "test_grouped_by_tag.csv"))

    _ref_tag = "nsga2" if "nsga2" in test_df.get("source_tag", pd.Series()).unique() else (
        test_df.groupby("source_tag")["test_auc"].mean().idxmax()
        if "test_auc" in test_df.columns and not test_df.empty else None
    )
    if _ref_tag:
        for _metric in ["test_auc", "test_f1", "test_mcc"]:
            run_statistical_comparisons(test_df, _ref_tag, _metric, analysis_dir)

    save_report_csv(grouped_by_backbone, os.path.join(analysis_dir, "test_grouped_by_backbone.csv"))
    for filename, pretty_df in build_pretty_test_tables(grouped_by_tag).items():
        write_markdown_table(pretty_df, os.path.join(analysis_dir, filename), filename.replace("_", " ").replace(".md", "").title())
    pretty_backbone = build_pretty_backbone_table(grouped_by_backbone)
    write_markdown_table(pretty_backbone, os.path.join(analysis_dir, "table_backbone_comparison.md"), "Table Backbone Comparison")
    best_dir = Path(prepare_output_paths(seed_root, "analysis")["best_dir"])
    best_dir.mkdir(parents=True, exist_ok=True)
    save_report_csv(_prepare_sorted_frame(test_df, ["test_auc", "test_pr_auc", "test_f1", "test_mcc"]).sort_values(["test_auc", "test_pr_auc", "test_f1", "test_mcc"], ascending=False).head(50), best_dir / "diagnostic_best_test_results.csv")
    save_report_csv(_prepare_sorted_frame(test_df, ["test_auc", "test_pr_auc", "test_f1", "test_mcc"]).sort_values(["test_auc", "test_pr_auc", "test_f1", "test_mcc"], ascending=False).groupby("source_tag", as_index=False).head(10), os.path.join(analysis_dir, "diagnostic_best_test_per_tag.csv"))
    overall_best = best_row(test_df, ["test_auc", "test_pr_auc", "test_f1", "test_mcc", "test_accuracy"])
    if not overall_best.empty:
        save_report_csv(overall_best, best_dir / "diagnostic_best_overall_single_run.csv")

    plt.figure(figsize=(12, 5))
    plot_df = sanitize_metric_frame(test_df.groupby(["source_tag", "backbone"], as_index=False)["test_auc"].mean()).sort_values("test_auc", ascending=False)
    plt.bar(np.arange(len(plot_df)), plot_df["test_auc"].values)
    plt.xticks(np.arange(len(plot_df)), [f"{a}:{b}" for a, b in zip(plot_df["source_tag"], plot_df["backbone"])], rotation=90)
    plt.ylabel("Mean Test AUC across seeds")
    save_plot(os.path.join(analysis_dir, "test_auc_barplot.png"))

    box_data, box_labels = [], []
    for method, g in test_df.groupby("source_tag"):
        vals = g["test_auc"].dropna().values
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(method)
    save_boxplot(box_data, box_labels, "Test AUC", os.path.join(analysis_dir, "test_auc_boxplot.png"))

    for metric, ylabel, filename in [("test_accuracy", "Test Accuracy", "test_accuracy_boxplot.png"), ("test_recall", "Test Recall", "test_recall_boxplot.png"), ("test_f1", "Test F1", "test_f1_boxplot.png"), ("test_mcc", "Test MCC", "test_mcc_boxplot.png")]:
        metric_box_data, metric_box_labels = [], []
        if metric not in test_df.columns:
            continue
        for method, g in test_df.groupby("source_tag"):
            vals = g[metric].dropna().values
            if len(vals) > 0:
                metric_box_data.append(vals)
                metric_box_labels.append(method)
        save_boxplot(metric_box_data, metric_box_labels, ylabel, os.path.join(analysis_dir, filename))

    plot_df = grouped_by_tag.copy()
    if not plot_df.empty:
        save_scatter_plot(plot_df, "elapsed_total_test_pipeline_s_mean", "test_auc_mean", "Total pipeline time (s)", "Mean test AUC", os.path.join(analysis_dir, "diagnostic_test_auc_vs_time.png"), label_col="source_tag")
        save_scatter_plot(plot_df, "trainable_params_m_mean", "test_auc_mean", "Trainable params (M)", "Mean test AUC", os.path.join(analysis_dir, "diagnostic_test_auc_vs_params.png"), label_col="source_tag")
        ranked_plot_rows, front_plot_rows = rank_candidates_multiobjective(
            plot_df.to_dict(orient='records'),
            auc_key='test_auc_mean',
            time_key='elapsed_total_test_pipeline_s_mean',
            params_key='trainable_params_m_mean',
            report_role_candidate='test_diagnostic_tradeoff_candidate',
            report_role_ranked='test_diagnostic_tradeoff_ranked',
            tie_break_name_key='source_tag',
            stable_id_key='source_tag',
        )
        pareto_ranked_df = pd.DataFrame(ranked_plot_rows)
        pareto_front_rows_df = pd.DataFrame(front_plot_rows)
        if not pareto_ranked_df.empty:
            pareto_ranked_df['analysis_scope'] = 'test_only_diagnostic_not_used_for_selection'
            pareto_ranked_df['time_metric_scope'] = 'test_pipeline_runtime_seconds_not_comparable_to_validation_selection_time'
            pareto_ranked_df['selection_metric_scope'] = 'test_diagnostic_contract'
            save_report_csv(pareto_ranked_df, os.path.join(analysis_dir, 'test_diagnostic_tradeoff_pareto_ranked.csv'))
            if not pareto_front_rows_df.empty:
                pareto_front_rows_df['analysis_scope'] = 'test_only_diagnostic_not_used_for_selection'
                pareto_front_rows_df['time_metric_scope'] = 'test_pipeline_runtime_seconds_not_comparable_to_validation_selection_time'
                pareto_front_rows_df['selection_metric_scope'] = 'test_diagnostic_contract'
                save_report_csv(pareto_front_rows_df, os.path.join(analysis_dir, 'test_diagnostic_tradeoff_pareto_front.csv'))
            plt.figure(figsize=(8, 5))
            plt.scatter(pareto_ranked_df['elapsed_total_test_pipeline_s_mean'], pareto_ranked_df['test_auc_mean'], alpha=0.55)
            for _, row in pareto_ranked_df.iterrows():
                plt.annotate(str(row.get('source_tag', '')), (row['elapsed_total_test_pipeline_s_mean'], row['test_auc_mean']), fontsize=8, alpha=0.8)
            if not pareto_front_rows_df.empty:
                ordered_front = pareto_front_rows_df.sort_values(['elapsed_total_test_pipeline_s_mean', 'trainable_params_m_mean', 'test_auc_mean'], ascending=[True, True, False])
                plt.plot(ordered_front['elapsed_total_test_pipeline_s_mean'], ordered_front['test_auc_mean'], linewidth=1.5)
                plt.scatter(ordered_front['elapsed_total_test_pipeline_s_mean'], ordered_front['test_auc_mean'])
            plt.xlabel('Total test pipeline time (s)')
            plt.ylabel('Mean test AUC')
            plt.title('Test Pareto trade-off (diagnostic only, not used for selection)')
            save_plot(os.path.join(analysis_dir, 'test_diagnostic_tradeoff_pareto_front.png'))
        write_reporting_manifest(analysis_dir, 'seed_analysis')
        return pareto_ranked_df
    write_reporting_manifest(analysis_dir, 'seed_analysis')
    return pd.DataFrame()


def write_experiment_level_reports(seed_root: str):
    for tag_dir in sorted(Path(seed_root).iterdir()):
        if not tag_dir.is_dir() or tag_dir.name in {"shared", "best", "analysis"}:
            continue
        summary_path = tag_dir / "genome_summary.jsonl"
        runs_path = tag_dir / "runs.jsonl"
        test_path = tag_dir / "test_results.jsonl"
        if summary_path.exists():
            summary_rows = _load_strict_jsonl(str(summary_path))
            sdf = pd.DataFrame(flatten_summary_rows(summary_rows, infer_seed_value(summary_rows)))
            if not sdf.empty:
                save_report_csv(sdf, tag_dir / "validation_summary.csv")
                save_report_csv(aggregate_frame(sdf, ["budget"], SUMMARY_METRICS), tag_dir / "validation_summary_by_budget.csv")
                top, top_front = build_multiobjective_summary_reports(
                    sdf,
                    report_role_candidate='multiobjective_validation_candidate',
                    report_role_ranked='multiobjective_validation_ranked',
                )
                if not top.empty:
                    save_report_csv(top, tag_dir / "selection_top_validation_candidates.csv")
                    top_md_cols = [
                        "genome_id", "budget", "pareto_front_rank", "pareto_within_front_rank", "ideal_distance",
                        "mean_val_auc", "total_time_s", "mean_trainable_params_m",
                        "mean_val_auc_loss_obj", "auc_loss", "selection_time_s_obj", "trainable_params_m_obj",
                        "selection_norm_auc_loss", "selection_norm_time_s", "selection_norm_params_m",
                        "robust_min_auc", "auc_mean", "pr_auc_mean", "accuracy_mean",
                    ]
                    write_markdown_table(
                        top[[c for c in top_md_cols if c in top.columns]],
                        tag_dir / "selection_top_validation_candidates.md",
                        f"Top validation candidates for {tag_dir.name} (Pareto-ranked)",
                    )
                if not top_front.empty:
                    save_report_csv(top_front, tag_dir / "selection_top_validation_pareto_front.csv")
        if runs_path.exists():
            rdf = pd.DataFrame(dedupe_run_rows(_load_strict_jsonl(str(runs_path))))
            if not rdf.empty:
                save_report_csv(rdf, tag_dir / "all_training_runs.csv")
                save_report_csv(aggregate_frame(rdf, ["backbone", "budget"], ["best_val_auc", "best_val_pr_auc", "best_val_accuracy", "time_s", "trainable_params_m"]), tag_dir / "training_runs_by_backbone_budget.csv")
        if test_path.exists():
            tdf = pd.DataFrame(dedupe_report_rows(pd.DataFrame(_load_strict_jsonl(str(test_path))), ["genome_id", "search_seed", "final_train_seed", "final_eval_seed", "source_tag", "selection_rank", "backbone"]))
            if not tdf.empty:
                save_report_csv(tdf, tag_dir / "test_results.csv")
                save_report_csv(best_row(tdf, ["test_auc", "test_pr_auc", "test_f1", "test_mcc"]), tag_dir / "best_test_candidate.csv")



def write_global_reports(root_dir: str, search_seeds: List[int], search_seed_roots: List[str], final_seeds: List[int], final_seed_roots: List[str]) -> str:
    all_test_rows, all_summary_rows = [], []
    for seed_root in search_seed_roots:
        analysis_dir = os.path.join(seed_root, "analysis")
        summ_csv = os.path.join(analysis_dir, "summary_all.csv")
        test_csv = os.path.join(analysis_dir, "test_results_shared.csv")
        if os.path.exists(summ_csv):
            all_summary_rows.append(pd.read_csv(summ_csv))
    global_dir = os.path.join(root_dir, "analysis_global")
    ensure_dir(global_dir)
    if all_summary_rows:
        gsum = dedupe_report_rows(pd.concat(all_summary_rows, ignore_index=True), ["tag", "seed", "genome_id", "budget"])
        save_report_csv(gsum, os.path.join(global_dir, "summary_all_seeds.csv"))
        validation_cmp = aggregate_frame(gsum, ["tag"], SUMMARY_METRICS)
        validation_cmp['report_role'] = 'diagnostic_validation_method_comparison'
        save_report_csv(validation_cmp, os.path.join(global_dir, "validation_comparison_by_method.csv"))
        contract_df = metric_contract_dataframe(gsum.columns)
        save_report_csv(contract_df, os.path.join(global_dir, 'metric_contract_registry.csv'))
        write_markdown_table(contract_df, os.path.join(global_dir, 'metric_contract_registry.md'), 'Metric Contract Registry')
    for seed_root in final_seed_roots:
        analysis_dir = os.path.join(seed_root, "analysis")
        test_csv = os.path.join(analysis_dir, "test_results_shared.csv")
        if os.path.exists(test_csv):
            all_test_rows.append(pd.read_csv(test_csv))
    if all_test_rows:
        gtest = _attach_pairing_keys(dedupe_report_rows(pd.concat(all_test_rows, ignore_index=True), ["genome_id", "search_seed", "final_train_seed", "final_eval_seed", "source_tag", "selection_rank", "backbone"]))
        save_report_csv(gtest, os.path.join(global_dir, "test_all_seeds.csv"))
        test_cmp = aggregate_frame(gtest, ["source_tag"], TEST_METRICS)
        test_cmp['report_role'] = 'diagnostic_test_method_comparison'
        save_report_csv(test_cmp, os.path.join(global_dir, "test_comparison_by_method.csv"))
        ranking_metrics = [m for m in [
            "test_auc",
            "test_pr_auc",
            "test_f1",
            "test_mcc",
            "test_loss",
            "test_log_loss",
            "test_brier_score",
            "test_ece_10bins",
            "elapsed_total_test_pipeline_s",
            "trainable_params_m",
            "complexity_score",
        ] if m in gtest.columns]
        ranking = aggregate_frame(gtest, ["source_tag"], ranking_metrics)
        pair_cols = [c for c in ["resolved_search_seed", "resolved_final_train_seed", "resolved_eval_seed"] if c in gtest.columns]
        pair_counts = gtest.groupby("source_tag", dropna=False).size().reset_index(name="raw_row_count")
        if pair_cols:
            unique_pairs = gtest.groupby("source_tag", dropna=False)[pair_cols].apply(lambda frame: frame.drop_duplicates().shape[0]).reset_index(name="count_runs")
            ranking = ranking.drop(columns=["count_runs"], errors="ignore").merge(unique_pairs, on="source_tag", how="left")
        ranking = ranking.merge(pair_counts, on="source_tag", how="left")
        for col, default in {
            "count_runs": 0,
            "ranking_eligible": False,

            "test_auc_mean": np.nan,
            "test_pr_auc_mean": np.nan,
            "test_f1_mean": np.nan,
            "test_mcc_mean": np.nan,
            "test_loss_mean": np.nan,
            "test_log_loss_mean": np.nan,
            "test_brier_score_mean": np.nan,
            "test_ece_10bins_mean": np.nan,
            "elapsed_total_test_pipeline_s_mean": np.nan,

            "mean_val_auc": np.nan,
            "robust_min_auc": np.nan,
            "mean_time_s_per_run": np.nan,
            "total_time_s": np.nan,

            "status": "NA",
            "mode": "NA",
            "tag": "NA",
            "seed": np.nan,
            "active_seed": np.nan,
        }.items():
            ensure_col(ranking, col, default)
        ranking = add_time_aliases(ranking)
        unique_run_count = int(ranking["count_runs"].max()) if "count_runs" in ranking.columns and not ranking.empty else 0
        min_runs = 2 if unique_run_count > 1 else 1
        ranking["min_runs_required"] = min_runs
        ranking["ranking_eligible"] = ranking.get("count_runs", pd.Series([0] * len(ranking))).fillna(0).astype(int) >= min_runs
        ranking = _prepare_sorted_frame(ranking, [c for c in ["test_auc_mean", "test_pr_auc_mean", "test_f1_mean", "test_mcc_mean"] if c in ranking.columns])
        ranking = pd.concat([
            ranking[ranking["ranking_eligible"]].sort_values([c for c in ["test_auc_mean", "test_pr_auc_mean", "test_f1_mean", "test_mcc_mean"] if c in ranking.columns], ascending=False),
            ranking[~ranking["ranking_eligible"]],
        ], ignore_index=True)
        ranking['report_role'] = 'diagnostic_test_metric_ranking'
        save_report_csv(ranking, os.path.join(global_dir, "diagnostic_method_ranking.csv"))
        write_reporting_manifest(global_dir, 'global_analysis', min_runs=min_runs)
        write_markdown_table(ranking, os.path.join(global_dir, "diagnostic_method_ranking.md"), "Diagnostic method ranking across all seeds")
    return global_dir


def main():
    cfg = load_config()
    root_dir = cfg["out_dir"]
    search_seeds = cfg["search_seeds"]
    final_seeds = cfg["final_seeds"]
    search_seed_roots = collect_seed_roots(root_dir, search_seeds)
    final_seed_roots = collect_seed_roots(root_dir, final_seeds)
    report_roots = []
    seen = set()
    for _root in list(search_seed_roots) + list(final_seed_roots):
        if _root not in seen:
            report_roots.append(_root)
            seen.add(_root)
    for seed_root in report_roots:
        analysis_dir = os.path.join(seed_root, "analysis")
        ensure_dir(analysis_dir)
        seed_name = os.path.basename(seed_root)
        try:
            seed_val = int(seed_name.split("seed_")[-1]) if seed_name.startswith("seed_") else int((search_seeds or final_seeds)[0])
        except ValueError:
            seed_val = int((search_seeds or final_seeds)[0])
        summary_rows = []
        for path in glob.glob(os.path.join(seed_root, "*", "genome_summary.jsonl")):
            summary_rows.extend(flatten_summary_rows(_load_strict_jsonl(path), seed_val))
        summary_df = pd.DataFrame(summary_rows)
        test_rows = filter_test_rows_for_current_protocol(_load_strict_jsonl(os.path.join(seed_root, "shared", "test_results_shared.jsonl")), cfg)
        test_df = pd.DataFrame(test_rows)
        summary_pareto_df, _ = generate_summary_reports(seed_root, analysis_dir, summary_df)
        test_pareto_df = generate_test_reports(seed_root, analysis_dir, test_df)
        write_seed_reporting_audit(analysis_dir, summary_df=summary_df, pareto_ranked_df=summary_pareto_df, test_pareto_df=test_pareto_df)
        write_experiment_level_reports(seed_root)
        print(f"Analysis exported to {analysis_dir}")
    global_dir = write_global_reports(root_dir, search_seeds, search_seed_roots, final_seeds, final_seed_roots)
    write_global_reporting_audit(global_dir)


if __name__ == "__main__":
    main()
