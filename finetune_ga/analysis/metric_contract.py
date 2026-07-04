from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    name: str
    role: str
    split: str
    used_for_selection: bool
    display_group: str
    description: str


METRIC_SPECS: List[MetricSpec] = [
    MetricSpec('mean_val_auc', 'selection_primary', 'validation', True, 'selection', 'Primary validation utility used for Pareto selection.'),
    MetricSpec('mean_time_s_per_run', 'selection_cost', 'validation', True, 'selection', 'Mean validation/search runtime in seconds per evaluated run used for Pareto selection.'),
    MetricSpec('mean_trainable_params_m', 'selection_cost', 'validation', True, 'selection', 'Model trainable-parameter cost used for Pareto selection.'),
    MetricSpec('selection_auc', 'selection_primary', 'validation', True, 'selection', 'Canonical validation selection utility copied into canonical reports.'),
    MetricSpec('selection_auc_loss', 'selection_cost', 'validation', True, 'selection', 'Canonical validation loss-form objective used in ideal-point selection.'),
    MetricSpec('selection_time_s', 'selection_cost', 'validation', True, 'selection', 'Canonical validation runtime cost copied into canonical reports.'),
    MetricSpec('selection_params_m', 'selection_cost', 'validation', True, 'selection', 'Canonical validation parameter cost copied into canonical reports.'),
    MetricSpec('selection_norm_auc_loss', 'selection_cost', 'validation', True, 'selection', 'Normalized validation AUC-loss objective used for ideal-point ranking.'),
    MetricSpec('selection_norm_time_s', 'selection_cost', 'validation', True, 'selection', 'Normalized validation runtime objective used for ideal-point ranking.'),
    MetricSpec('selection_norm_params_m', 'selection_cost', 'validation', True, 'selection', 'Normalized validation parameter objective used for ideal-point ranking.'),
    MetricSpec('mean_val_auc_loss_obj', 'selection_cost', 'validation', True, 'selection', 'Loss-form objective equal to 1 - mean_val_auc for Pareto minimization.'),
    MetricSpec('robust_min_auc_loss_obj', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Loss-form diagnostic equal to 1 - robust_min_auc.'),
    MetricSpec('selection_time_s_obj', 'selection_cost', 'validation', True, 'selection', 'Runtime objective in seconds used by Pareto minimization.'),
    MetricSpec('trainable_params_m_obj', 'selection_cost', 'validation', True, 'selection', 'Trainable-parameter objective in millions used by Pareto minimization.'),
    MetricSpec('robust_min_auc', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Conservative robustness diagnostic across runs/backbones.'),
    MetricSpec('auc_mean', 'diagnostic_performance', 'validation', False, 'diagnostic', 'Descriptive mean validation AUC, not a selection objective.'),
    MetricSpec('auc_std', 'diagnostic_performance', 'validation', False, 'diagnostic', 'Descriptive dispersion of validation AUC, not a selection objective.'),
    MetricSpec('pr_auc_mean', 'diagnostic_performance', 'validation', False, 'diagnostic', 'Descriptive mean validation PR-AUC.'),
    MetricSpec('accuracy_mean', 'diagnostic_performance', 'validation', False, 'diagnostic', 'Descriptive mean validation accuracy.'),
    MetricSpec('loss_mean', 'diagnostic_performance', 'validation', False, 'diagnostic', 'Descriptive mean validation loss.'),
    MetricSpec('total_time_s', 'diagnostic_cost', 'validation', False, 'diagnostic', 'Descriptive total validation/search runtime across all runs for a genome; reported for completeness, not used for selection.'),
    MetricSpec('failure_rate', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Fraction of failed/infeasible runs for a method or candidate.'),
    MetricSpec('success_rate', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Fraction of successful runs for a method or candidate.'),
    MetricSpec('n_input_runs', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Number of input runs considered before filtering.'),
    MetricSpec('n_ok_runs', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Number of successful runs after filtering.'),
    MetricSpec('n_failed_runs', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Number of failed or infeasible runs.'),
    MetricSpec('count_runs', 'diagnostic_robustness', 'mixed', False, 'diagnostic', 'Number of distinct runs used in aggregate reports.'),
    MetricSpec('active_seed', 'metadata', 'mixed', False, 'metadata', 'Seed value active for this report row.'),
    MetricSpec('diagnostic_total_search_time_s', 'diagnostic_cost', 'validation', False, 'diagnostic', 'Total search runtime used in diagnostic summaries.'),
    MetricSpec('min_runs_required', 'metadata', 'mixed', False, 'metadata', 'Minimum number of runs required for a ranking or report row.'),
    MetricSpec('mode', 'metadata', 'mixed', False, 'metadata', 'Execution or evaluation mode associated with the report row.'),
    MetricSpec('ranking_eligible', 'metadata', 'mixed', False, 'metadata', 'Whether the row is eligible for ranking.'),
    MetricSpec('raw_row_count', 'metadata', 'mixed', False, 'metadata', 'Number of raw rows contributing to an aggregate report row.'),
    MetricSpec('selection_time_s_per_run', 'selection_cost', 'validation', True, 'selection', 'Selection runtime normalized per run.'),
    MetricSpec('status', 'metadata', 'mixed', False, 'metadata', 'Status label associated with the report row.'),
    MetricSpec('complexity_score', 'test_cost_diagnostic', 'mixed', False, 'diagnostic', 'Composite diagnostic cost score derived from parameter footprint and runtime.'),
    MetricSpec('hypervolume_2d', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Validation hypervolume diagnostic over two objectives.'),
    MetricSpec('hypervolume_3d', 'diagnostic_robustness', 'validation', False, 'diagnostic', 'Validation hypervolume diagnostic over three objectives.'),

    MetricSpec('test_auc', 'test_primary', 'test', False, 'test', 'Primary held-out test AUC.'),
    MetricSpec('test_pr_auc', 'test_primary', 'test', False, 'test', 'Held-out test PR-AUC.'),
    MetricSpec('test_accuracy', 'test_primary', 'test', False, 'test', 'Held-out test accuracy.'),
    MetricSpec('test_precision', 'test_primary', 'test', False, 'test', 'Held-out test precision.'),
    MetricSpec('test_recall', 'test_primary', 'test', False, 'test', 'Held-out test recall.'),
    MetricSpec('test_specificity', 'test_primary', 'test', False, 'test', 'Held-out test specificity.'),
    MetricSpec('test_f1', 'test_primary', 'test', False, 'test', 'Held-out test F1.'),
    MetricSpec('test_balanced_accuracy', 'test_primary', 'test', False, 'test', 'Held-out balanced accuracy.'),
    MetricSpec('test_mcc', 'test_primary', 'test', False, 'test', 'Held-out MCC.'),
    MetricSpec('test_npv', 'test_primary', 'test', False, 'test', 'Held-out negative predictive value.'),
    MetricSpec('test_fpr', 'test_primary', 'test', False, 'test', 'Held-out false positive rate.'),
    MetricSpec('test_fnr', 'test_primary', 'test', False, 'test', 'Held-out false negative rate.'),
    MetricSpec('test_brier_score', 'test_calibration', 'test', False, 'test_calibration', 'Held-out Brier score on positive-class probabilities.'),
    MetricSpec('test_loss', 'test_calibration', 'test', False, 'test_calibration', 'Held-out compiled sparse categorical cross-entropy computed in the single-pass evaluation loop.'),
    MetricSpec('test_log_loss', 'test_calibration', 'test', False, 'test_calibration', 'Held-out binary log-loss on positive-class probabilities.'),
    MetricSpec('test_ece_10bins', 'test_calibration', 'test', False, 'test_calibration', 'Held-out expected calibration error using 10 bins.'),
    MetricSpec('test_mean_positive_confidence', 'test_calibration', 'test', False, 'test_calibration', 'Mean predicted confidence for positive-class examples.'),
    MetricSpec('test_mean_negative_confidence', 'test_calibration', 'test', False, 'test_calibration', 'Mean predicted confidence for negative-class examples.'),
    MetricSpec('test_prevalence', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Observed positive-class prevalence in the held-out test set.'),
    MetricSpec('test_support', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Number of evaluated held-out test examples.'),
    MetricSpec('test_tp', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Held-out true positives at the reporting threshold.'),
    MetricSpec('test_tn', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Held-out true negatives at the reporting threshold.'),
    MetricSpec('test_fp', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Held-out false positives at the reporting threshold.'),
    MetricSpec('test_fn', 'test_diagnostic', 'test', False, 'test_diagnostic', 'Held-out false negatives at the reporting threshold.'),
    MetricSpec('test_accuracy_ci95_low', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Lower Wilson bound for per-run held-out accuracy.'),
    MetricSpec('test_accuracy_ci95_high', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Upper Wilson bound for per-run held-out accuracy.'),
    MetricSpec('test_recall_ci95_low', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Lower Wilson bound for per-run held-out recall.'),
    MetricSpec('test_recall_ci95_high', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Upper Wilson bound for per-run held-out recall.'),
    MetricSpec('test_specificity_ci95_low', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Lower Wilson bound for per-run held-out specificity.'),
    MetricSpec('test_specificity_ci95_high', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Upper Wilson bound for per-run held-out specificity.'),
    MetricSpec('test_precision_ci95_low', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Lower Wilson bound for per-run held-out precision.'),
    MetricSpec('test_precision_ci95_high', 'test_uncertainty', 'test', False, 'test_uncertainty', 'Upper Wilson bound for per-run held-out precision.'),
    MetricSpec('elapsed_test_train_s', 'test_cost_diagnostic', 'test', False, 'test_diagnostic', 'Test-stage final training runtime before held-out evaluation.'),
    MetricSpec('elapsed_test_eval_s', 'test_cost_diagnostic', 'test', False, 'test_diagnostic', 'Single-pass held-out evaluation runtime.'),
    MetricSpec('elapsed_total_test_pipeline_s', 'test_cost_diagnostic', 'test', False, 'test_diagnostic', 'Test pipeline runtime; diagnostic only, never used for selection.'),
    MetricSpec('trainable_params_m', 'test_cost_diagnostic', 'test', False, 'test_diagnostic', 'Test-time parameter footprint; diagnostic only.'),

    MetricSpec('test_auc_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test AUC across methods.'),
    MetricSpec('test_auc_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test AUC across methods.'),
    MetricSpec('test_pr_auc_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test PR-AUC across methods.'),
    MetricSpec('test_pr_auc_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test PR-AUC across methods.'),
    MetricSpec('test_accuracy_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test accuracy across methods.'),
    MetricSpec('test_accuracy_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test accuracy across methods.'),
    MetricSpec('test_precision_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test precision across methods.'),
    MetricSpec('test_precision_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test precision across methods.'),
    MetricSpec('test_recall_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test recall across methods.'),
    MetricSpec('test_recall_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test recall across methods.'),
    MetricSpec('test_specificity_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test specificity across methods.'),
    MetricSpec('test_specificity_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test specificity across methods.'),
    MetricSpec('test_f1_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test F1 across methods.'),
    MetricSpec('test_f1_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test F1 across methods.'),
    MetricSpec('test_balanced_accuracy_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out balanced accuracy across methods.'),
    MetricSpec('test_balanced_accuracy_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out balanced accuracy across methods.'),
    MetricSpec('test_mcc_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out test MCC across methods.'),
    MetricSpec('test_mcc_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out test MCC across methods.'),
    MetricSpec('test_npv_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out negative predictive value across methods.'),
    MetricSpec('test_npv_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out negative predictive value across methods.'),
    MetricSpec('test_fpr_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out false positive rate across methods.'),
    MetricSpec('test_fpr_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out false positive rate across methods.'),
    MetricSpec('test_fnr_mean', 'test_primary', 'test', False, 'global', 'Global mean held-out false negative rate across methods.'),
    MetricSpec('test_fnr_std', 'test_primary', 'test', False, 'global', 'Global dispersion of held-out false negative rate across methods.'),
    MetricSpec('test_brier_score_mean', 'test_calibration', 'test', False, 'global', 'Global mean held-out Brier score across methods.'),
    MetricSpec('test_brier_score_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of held-out Brier score across methods.'),
    MetricSpec('test_loss_mean', 'test_calibration', 'test', False, 'global', 'Global mean compiled sparse categorical cross-entropy across methods.'),
    MetricSpec('test_loss_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of compiled sparse categorical cross-entropy across methods.'),
    MetricSpec('test_log_loss_mean', 'test_calibration', 'test', False, 'global', 'Global mean binary log-loss on positive-class probabilities across methods.'),
    MetricSpec('test_log_loss_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of binary log-loss across methods.'),
    MetricSpec('test_ece_10bins_mean', 'test_calibration', 'test', False, 'global', 'Global mean held-out expected calibration error across methods.'),
    MetricSpec('test_ece_10bins_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of held-out expected calibration error across methods.'),
    MetricSpec('test_mean_positive_confidence_mean', 'test_calibration', 'test', False, 'global', 'Global mean positive-class confidence across methods.'),
    MetricSpec('test_mean_positive_confidence_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of positive-class confidence across methods.'),
    MetricSpec('test_mean_negative_confidence_mean', 'test_calibration', 'test', False, 'global', 'Global mean negative-class confidence across methods.'),
    MetricSpec('test_mean_negative_confidence_std', 'test_calibration', 'test', False, 'global', 'Global dispersion of negative-class confidence across methods.'),
    MetricSpec('test_prevalence_mean', 'test_diagnostic', 'test', False, 'global', 'Global mean test prevalence across methods.'),
    MetricSpec('test_prevalence_std', 'test_diagnostic', 'test', False, 'global', 'Global dispersion of test prevalence across methods.'),
    MetricSpec('test_support_mean', 'test_diagnostic', 'test', False, 'global', 'Global mean number of evaluated test examples across methods.'),
    MetricSpec('test_support_std', 'test_diagnostic', 'test', False, 'global', 'Global dispersion of evaluated test examples across methods.'),
    MetricSpec('elapsed_test_train_s_mean', 'test_cost_diagnostic', 'test', False, 'global', 'Global mean final training runtime across methods.'),
    MetricSpec('elapsed_test_train_s_std', 'test_cost_diagnostic', 'test', False, 'global', 'Global dispersion of final training runtime across methods.'),
    MetricSpec('elapsed_test_eval_s_mean', 'test_cost_diagnostic', 'test', False, 'global', 'Global mean single-pass held-out evaluation runtime across methods.'),
    MetricSpec('elapsed_test_eval_s_std', 'test_cost_diagnostic', 'test', False, 'global', 'Global dispersion of single-pass held-out evaluation runtime across methods.'),
    MetricSpec('elapsed_total_test_pipeline_s_mean', 'test_cost_diagnostic', 'test', False, 'global', 'Aggregated test pipeline runtime in canonical diagnostic reports.'),
    MetricSpec('elapsed_total_test_pipeline_s_std', 'test_cost_diagnostic', 'test', False, 'global', 'Global dispersion of diagnostic test pipeline runtime across methods.'),
    MetricSpec('trainable_params_m_mean', 'test_cost_diagnostic', 'test', False, 'global', 'Aggregated parameter footprint in canonical diagnostic reports.'),
    MetricSpec('trainable_params_m_std', 'test_cost_diagnostic', 'test', False, 'global', 'Global dispersion of diagnostic parameter footprint across methods.'),
    MetricSpec('complexity_score_mean', 'test_cost_diagnostic', 'mixed', False, 'global', 'Global mean composite diagnostic complexity score across methods.'),
    MetricSpec('complexity_score_std', 'test_cost_diagnostic', 'mixed', False, 'global', 'Global dispersion of composite diagnostic complexity score across methods.'),

    MetricSpec('mean_val_auc_mean', 'selection_primary', 'validation', True, 'global', 'Global mean of validation utility across methods.'),
    MetricSpec('mean_time_s_per_run_mean', 'selection_cost', 'validation', True, 'global', 'Global mean validation/search runtime in seconds per evaluated run across methods.'),
    MetricSpec('mean_trainable_params_m_mean', 'selection_cost', 'validation', True, 'global', 'Global mean validation parameter cost across methods.'),
    MetricSpec('mean_val_auc_std', 'selection_primary', 'validation', True, 'global', 'Global dispersion of validation utility across methods.'),
    MetricSpec('mean_time_s_per_run_std', 'selection_cost', 'validation', True, 'global', 'Global dispersion of validation/search runtime in seconds per evaluated run across methods.'),
    MetricSpec('mean_trainable_params_m_std', 'selection_cost', 'validation', True, 'global', 'Global dispersion of validation parameter cost across methods.'),
    MetricSpec('total_time_s_mean', 'diagnostic_cost', 'validation', False, 'global', 'Global mean descriptive total validation/search runtime across methods.'),
    MetricSpec('total_time_s_std', 'diagnostic_cost', 'validation', False, 'global', 'Global dispersion of descriptive total validation/search runtime across methods.'),
]

AGGREGATE_SUFFIXES = (
    '_ci95_high', '_ci95_low',
    '_mean', '_std', '_min', '_max', '_median',
    '_q1', '_q3', '_sum', '_count',
)

def metric_contract_dataframe(extra_metrics: Iterable[str] | None = None) -> pd.DataFrame:
    rows = [asdict(spec) for spec in METRIC_SPECS]
    known = {spec.name for spec in METRIC_SPECS}

    metrics = [] if extra_metrics is None else list(extra_metrics)

    for metric in sorted(set(metrics)):
        if metric in known:
            continue
        rows.append({
            'name': metric,
            'role': 'unclassified_auxiliary',
            'split': 'mixed',
            'used_for_selection': False,
            'display_group': 'auxiliary',
            'description': 'Auxiliary metric not explicitly registered in the paper-ready contract.',
        })
    return pd.DataFrame(rows)


def metric_spec_map() -> dict[str, MetricSpec]:
    return {spec.name: spec for spec in METRIC_SPECS}


def resolve_metric_name(metric: str) -> str | None:
    specs = metric_spec_map()

    if metric in specs:
        return metric


    for suffix in AGGREGATE_SUFFIXES:
        if metric.endswith(suffix):
            base = metric[: -len(suffix)]

            if base in specs:
                return base

            resolved_base = resolve_metric_name(base)
            if resolved_base is not None:
                return resolved_base

    return None


def validate_metric_contract(metrics: Iterable[str], *, allowed_roles: Iterable[str] | None = None, strict: bool = False) -> list[str]:
    specs = metric_spec_map()
    allowed = set(allowed_roles or [])
    issues: list[str] = []
    for metric in metrics:
        resolved = resolve_metric_name(metric)
        spec = specs.get(resolved) if resolved is not None else None
        if spec is None:
            if strict:
                issues.append(f'unregistered_metric={metric}')
            continue
        if allowed and spec.role not in allowed:
            issues.append(f'role_not_allowed={metric}:{spec.role}')
    return issues
