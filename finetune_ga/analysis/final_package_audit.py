from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from finetune_ga.analysis.artifact_contract import get_artifact_specs
from finetune_ga.analysis.metric_contract import validate_metric_contract, resolve_metric_name


EXPECTED_TOP_LEVEL_AUDITS = {
    'FINAL_PACKAGE_AUDIT.txt',
    'MIGRATION_REPORT.md',
}
EXPECTED_SOURCE_FILES = {
    'README.md',
    'requirements.txt',
    'pyproject.toml',
    'config.json',
    'run_paper_reproduction.sh',
}


def _iter_matching(root: Path, predicate) -> list[str]:
    matches: list[str] = []
    for path in root.rglob('*'):
        if predicate(path):
            matches.append(str(path.relative_to(root)))
    return sorted(matches)


def check_no_pycache(root: Path) -> list[str]:
    return _iter_matching(root, lambda p: p.is_dir() and p.name == '__pycache__')


def check_no_pyc(root: Path) -> list[str]:
    return _iter_matching(root, lambda p: p.is_file() and p.suffix == '.pyc')


def check_no_pytest_cache(root: Path) -> list[str]:
    return _iter_matching(root, lambda p: '.pytest_cache' in p.parts)


def check_expected_top_level_audits(root: Path) -> list[str]:
    extras = []
    for p in root.iterdir():
        if p.is_file() and ('AUDIT' in p.name or 'MIGRATION' in p.name):
            if p.name not in EXPECTED_TOP_LEVEL_AUDITS:
                extras.append(p.name)
    return sorted(extras)


def check_required_artifacts(root: Path) -> list[str]:
    missing: list[str] = []
    for spec in get_artifact_specs():
        if not spec.required:
            continue
        found = list(root.rglob(spec.name))
        if not found:
            missing.append(spec.name)
    return sorted(missing)


def check_unexpected_runtime_artifacts(root: Path) -> list[str]:
    found: list[str] = []
    required_names = {spec.name for spec in get_artifact_specs() if spec.required}
    for path in root.rglob('*.csv'):
        if path.name in required_names:
            found.append(str(path.relative_to(root)))
    return sorted(found)


def check_expected_source_files(root: Path) -> list[str]:
    missing: list[str] = []
    for name in sorted(EXPECTED_SOURCE_FILES):
        if not (root / name).exists():
            missing.append(name)
    return missing


def _sample_value(column: str):
    if column in {'report_role'}:
        return 'sample'
    if column in {'analysis_scope'}:
        return 'test_only_diagnostic_not_used_for_selection'
    if column in {'time_metric_scope'}:
        return 'test_pipeline_runtime_seconds_not_comparable_to_validation_selection_time'
    if column in {'selection_metric_scope'}:
        return 'validation_selection_contract'
    if column in {'selection_summary_note'}:
        return 'sample'
    if column in {'tag', 'source_tag', 'genome_id', 'budget', 'budget_name', 'backbone'}:
        return 'sample'
    if column in {'seed', 'pareto_front_rank', 'pareto_within_front_rank'}:
        return 1
    return 1.0


def _role_for_artifact(name: str) -> str:
    if name.startswith('selection_'):
        return 'multiobjective_validation_ranked'
    if name.startswith('diagnostic_'):
        return 'descriptive_diagnostic'
    if name.startswith('test_diagnostic_'):
        return 'test_diagnostic_tradeoff'
    if name == 'validation_comparison_by_method.csv':
        return 'diagnostic_validation_method_comparison'
    if name in {'test_comparison_by_method.csv', 'diagnostic_method_ranking.csv'}:
        return 'diagnostic_test_method_comparison' if name == 'test_comparison_by_method.csv' else 'diagnostic_test_metric_ranking'
    return 'sample'


def _runtime_contract_selfcheck() -> list[str]:
    issues: list[str] = []
    with TemporaryDirectory() as td:
        root = Path(td)
        for spec in get_artifact_specs():
            row = {col: _sample_value(col) for col in spec.required_columns}
            row['report_role'] = _role_for_artifact(spec.name)
            # Ensure tag/source_tag coexist where needed by grouping artifacts
            if 'tag' in spec.required_columns:
                row['tag'] = 'sample'
            if 'source_tag' in spec.required_columns:
                row['source_tag'] = 'sample'
            df = pd.DataFrame([row])
            path = root / spec.name
            df.to_csv(path, index=False)
            loaded = pd.read_csv(path)
            missing = [c for c in spec.required_columns if c not in loaded.columns]
            if missing:
                issues.append(f'runtime_contract_missing_columns={spec.name}:' + ','.join(missing))
                continue
            metadata = {'genome_id','budget','budget_name','tag','seed','backbone','rank','source_tag','analysis_scope','time_metric_scope','selection_metric_scope','selection_summary_note','report_role','pareto_front_rank','pareto_within_front_rank','ideal_distance','auc_loss','selection_time_s_obj','trainable_params_m_obj'}
            metric_like = [c for c in loaded.columns if c not in metadata]
            metric_issues = validate_metric_contract(metric_like, strict=True)
            if metric_issues:
                issues.extend(f'runtime_contract_metric_issue={spec.name}:{mi}' for mi in metric_issues)
        return issues


def run_final_package_audit(root_dir: str | Path, *, require_runtime_artifacts: bool = False) -> tuple[bool, str]:
    root = Path(root_dir)
    checks: list[str] = []
    issues: list[str] = []

    pycache = check_no_pycache(root)
    if pycache:
        issues.append('pycache_dirs_present=' + ', '.join(pycache))
    else:
        checks.append('no_pycache_dirs=PASS')

    pyc = check_no_pyc(root)
    if pyc:
        issues.append('pyc_files_present=' + ', '.join(pyc))
    else:
        checks.append('no_pyc_files=PASS')

    pytest_cache = check_no_pytest_cache(root)
    if pytest_cache:
        issues.append('pytest_cache_present=' + ', '.join(pytest_cache))
    else:
        checks.append('no_pytest_cache=PASS')

    extra_audits = check_expected_top_level_audits(root)
    if extra_audits:
        issues.append('unexpected_top_level_audit_files=' + ', '.join(extra_audits))
    else:
        checks.append('top_level_audit_files=PASS')

    missing_source_files = check_expected_source_files(root)
    if missing_source_files:
        issues.append('missing_expected_source_files=' + ', '.join(missing_source_files))
    else:
        checks.append('expected_source_files=PASS')

    required_specs = [spec.name for spec in get_artifact_specs() if spec.required]
    if not required_specs:
        issues.append('artifact_contract_has_no_required_specs')
    else:
        checks.append(f'artifact_contract_declares_required_artifacts={len(required_specs)}')

    runtime_missing = check_required_artifacts(root)
    runtime_present = check_unexpected_runtime_artifacts(root)
    if require_runtime_artifacts:
        if runtime_missing:
            issues.append('missing_required_artifacts=' + ', '.join(runtime_missing))
        else:
            checks.append('required_runtime_artifacts_present=PASS')
    else:
        if runtime_present:
            issues.append('unexpected_runtime_artifacts_in_source_package=' + ', '.join(runtime_present))
        else:
            checks.append('source_package_contains_no_runtime_artifacts=PASS')
        runtime_contract_issues = _runtime_contract_selfcheck()
        if runtime_contract_issues:
            issues.extend(runtime_contract_issues)
        else:
            checks.append('runtime_contract_selfcheck=PASS')

    lines = ['FINAL PACKAGE AUDIT: PASS' if not issues else 'FINAL PACKAGE AUDIT: FAIL', '']
    lines.append('Checks:')
    lines.extend(f'- {c}' for c in checks)
    lines.append('')
    if issues:
        lines.append('Issues:')
        lines.extend(f'- {i}' for i in issues)
    else:
        lines.append('Issues: none')
    return (not issues, '\n'.join(lines) + '\n')
