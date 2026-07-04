from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    category: str
    required: bool
    required_columns: tuple[str, ...]
    description: str


REQUIRED_SELECTION_COLUMNS = (
    'pareto_front_rank', 'pareto_within_front_rank', 'ideal_distance',
    'selection_auc', 'selection_auc_loss', 'selection_time_s', 'selection_params_m',
    'mean_val_auc_loss_obj', 'selection_time_s_obj', 'trainable_params_m_obj',
    'selection_norm_auc_loss', 'selection_norm_time_s', 'selection_norm_params_m',
)

ARTIFACT_SPECS: List[ArtifactSpec] = [
    ArtifactSpec(
        'selection_validation_pareto_ranked_candidates.csv', 'selection', True,
        REQUIRED_SELECTION_COLUMNS + ('selection_metric_scope', 'selection_summary_note', 'report_role', 'mean_val_auc', 'total_time_s', 'mean_trainable_params_m', 'robust_min_auc', 'auc_mean', 'pr_auc_mean', 'accuracy_mean'),
        'Canonical ranked validation candidates under the Pareto selection contract.',
    ),
    ArtifactSpec(
        'selection_validation_pareto_front.csv', 'selection', True,
        REQUIRED_SELECTION_COLUMNS + ('selection_metric_scope', 'selection_summary_note', 'report_role', 'mean_val_auc', 'total_time_s', 'mean_trainable_params_m', 'robust_min_auc', 'auc_mean', 'pr_auc_mean', 'accuracy_mean'),
        'Canonical Pareto-front validation candidates under the Pareto selection contract.',
    ),
    ArtifactSpec(
        'selection_top_validation_candidates.csv', 'selection', True,
        REQUIRED_SELECTION_COLUMNS + ('selection_metric_scope', 'selection_summary_note', 'report_role', 'mean_val_auc', 'total_time_s', 'mean_trainable_params_m', 'robust_min_auc', 'auc_mean', 'pr_auc_mean', 'accuracy_mean', 'tag'),
        'Per-method canonical top validation candidates ranked by Pareto then ideal distance.',
    ),
    ArtifactSpec(
        'selection_top_validation_pareto_front.csv', 'selection', True,
        REQUIRED_SELECTION_COLUMNS + ('selection_metric_scope', 'selection_summary_note', 'report_role', 'mean_val_auc', 'total_time_s', 'mean_trainable_params_m', 'robust_min_auc', 'auc_mean', 'pr_auc_mean', 'accuracy_mean', 'tag'),
        'Per-method Pareto front for validation candidates.',
    ),
    ArtifactSpec(
        'diagnostic_validation_auc_ranked_candidates.csv', 'diagnostic', True,
        ('report_role', 'genome_id', 'budget', 'tag', 'mean_val_auc', 'auc_mean', 'robust_min_auc', 'pr_auc_mean', 'accuracy_mean', 'total_time_s', 'mean_trainable_params_m'),
        'Diagnostic AUC-ranked validation candidates; not used for selection.',
    ),
    ArtifactSpec(
        'diagnostic_validation_auc_ranked_per_tag.csv', 'diagnostic', True,
        ('report_role', 'genome_id', 'budget', 'tag', 'mean_val_auc', 'auc_mean', 'robust_min_auc', 'pr_auc_mean', 'accuracy_mean', 'total_time_s', 'mean_trainable_params_m'),
        'Diagnostic AUC-ranked per-tag validation candidates; not used for selection.',
    ),
    ArtifactSpec(
        'test_diagnostic_tradeoff_pareto_ranked.csv', 'test_diagnostic', True,
        ('pareto_front_rank', 'pareto_within_front_rank', 'ideal_distance', 'analysis_scope', 'time_metric_scope', 'selection_metric_scope', 'report_role', 'test_auc_mean', 'elapsed_total_test_pipeline_s_mean', 'trainable_params_m_mean', 'source_tag'),
        'Diagnostic Pareto-ranked test trade-off report; explicitly not used for selection.',
    ),
    ArtifactSpec(
        'test_diagnostic_tradeoff_pareto_front.csv', 'test_diagnostic', True,
        ('pareto_front_rank', 'pareto_within_front_rank', 'ideal_distance', 'analysis_scope', 'time_metric_scope', 'selection_metric_scope', 'report_role', 'test_auc_mean', 'elapsed_total_test_pipeline_s_mean', 'trainable_params_m_mean', 'source_tag'),
        'Diagnostic Pareto front on test trade-off; explicitly not used for selection.',
    ),
    ArtifactSpec(
        'validation_comparison_by_method.csv', 'global', True,
        ('tag', 'report_role', 'mean_val_auc_mean', 'mean_val_auc_std', 'total_time_s_mean', 'total_time_s_std', 'mean_trainable_params_m_mean', 'mean_trainable_params_m_std'),
        'Global aggregate comparison across validation methods.',
    ),
    ArtifactSpec(
        'test_comparison_by_method.csv', 'global', True,
        ('source_tag', 'report_role', 'test_auc_mean', 'test_auc_std', 'test_pr_auc_mean', 'test_pr_auc_std', 'test_f1_mean', 'test_f1_std', 'test_mcc_mean', 'test_mcc_std', 'test_loss_mean', 'test_loss_std', 'test_log_loss_mean', 'test_log_loss_std', 'test_brier_score_mean', 'test_brier_score_std', 'test_ece_10bins_mean', 'test_ece_10bins_std', 'elapsed_total_test_pipeline_s_mean', 'elapsed_total_test_pipeline_s_std', 'trainable_params_m_mean', 'trainable_params_m_std'),
        'Global aggregate comparison across test methods.',
    ),
    ArtifactSpec(
        'diagnostic_method_ranking.csv', 'global', True,
        ('source_tag', 'report_role', 'test_auc_mean', 'test_pr_auc_mean', 'test_f1_mean', 'test_mcc_mean', 'test_loss_mean', 'test_log_loss_mean', 'test_brier_score_mean', 'test_ece_10bins_mean', 'elapsed_total_test_pipeline_s_mean'),
        'Global diagnostic ranking across methods.',
    ),
]


def artifact_contract_dataframe(extra_artifacts: Iterable[str] | None = None) -> pd.DataFrame:
    rows = [asdict(spec) | {'required_columns': ','.join(spec.required_columns)} for spec in ARTIFACT_SPECS]
    known = {spec.name for spec in ARTIFACT_SPECS}
    for artifact in sorted(set(extra_artifacts or [])):
        if artifact in known:
            continue
        rows.append({
            'name': artifact,
            'category': 'unclassified',
            'required': False,
            'required_columns': '',
            'description': 'Artifact not explicitly registered in the canonical artifact contract.',
        })
    return pd.DataFrame(rows)


def get_artifact_specs(*, category: str | None = None) -> List[ArtifactSpec]:
    specs = ARTIFACT_SPECS
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    return list(specs)
