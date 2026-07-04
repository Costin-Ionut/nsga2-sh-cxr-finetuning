import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

# config.json lives at the project root (two levels above this file)
DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / 'config.json'


def get_config_path(path: str | None = None) -> str:
    candidate = path or os.environ.get('CONFIG_PATH')
    if candidate is None:
        return str(DEFAULT_CONFIG_PATH)
    candidate = str(candidate).strip()
    if not candidate:
        return str(DEFAULT_CONFIG_PATH)
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return str(candidate_path)
    if candidate_path.exists():
        return str(candidate_path.resolve())
    repo_relative = DEFAULT_CONFIG_PATH.with_name(candidate)
    return str(repo_relative)


def _validate_budget(budget: Dict[str, Any], index: int) -> None:
    required = {'name', 'e1', 'e2', 'steps_factor'}
    missing = sorted(required - set(budget))
    if missing:
        raise ValueError(f"Budget #{index} is missing keys: {', '.join(missing)}")
    if float(budget['steps_factor']) <= 0:
        raise ValueError(f"Budget {budget['name']} must have steps_factor > 0")

    has_legacy_reps = 'reps' in budget
    has_mode_reps = ('search_reps' in budget) or ('final_reps' in budget)
    if not has_legacy_reps and not has_mode_reps:
        raise ValueError(
            f"Budget {budget['name']} must define reps or at least one of search_reps/final_reps"
        )

    for reps_key in ('reps', 'search_reps', 'final_reps'):
        if reps_key in budget and int(budget[reps_key]) <= 0:
            raise ValueError(f"Budget {budget['name']} must have {reps_key} > 0")

    if 'promote_frac' in budget and not (0.0 < float(budget['promote_frac']) <= 1.0):
        raise ValueError(f"Budget {budget['name']} must have promote_frac in (0, 1].")
    if int(budget['e1']) < 0 or int(budget['e2']) < 0:
        raise ValueError(f"Budget {budget['name']} must have non-negative epoch counts")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_top = ['num_classes', 'model_names', 'budgets', 'out_dir', 'search_space']
    missing_top = [k for k in required_top if k not in cfg]
    if missing_top:
        raise ValueError(f"Configuration is missing keys: {', '.join(missing_top)}")

    if not isinstance(cfg['model_names'], list) or not cfg['model_names']:
        raise ValueError('model_names must be a non-empty list')
    if len(set(cfg['model_names'])) != len(cfg['model_names']):
        raise ValueError('model_names must be unique')
    if not isinstance(cfg['budgets'], list) or not cfg['budgets']:
        raise ValueError('budgets must be a non-empty list')

    seen_budget_names = set()
    for idx, budget in enumerate(cfg['budgets'], start=1):
        _validate_budget(budget, idx)
        name = str(budget['name'])
        if name in seen_budget_names:
            raise ValueError(f'Duplicate budget name: {name}')
        seen_budget_names.add(name)

    unexpected_legacy = [k for k in ('seed', 'seeds') if k in cfg]
    if unexpected_legacy:
        raise ValueError('Legacy seed keys are not allowed; use only search_seeds, final_seeds, and selection_source_seed')

    if 'search_seeds' not in cfg or not isinstance(cfg['search_seeds'], list) or not cfg['search_seeds']:
        raise ValueError('search_seeds must be a non-empty list')
    if 'final_seeds' not in cfg or not isinstance(cfg['final_seeds'], list) or not cfg['final_seeds']:
        raise ValueError('final_seeds must be a non-empty list')
    if 'selection_source_seed' not in cfg:
        raise ValueError('selection_source_seed is required')

    cfg['search_seeds'] = [int(s) for s in cfg['search_seeds']]
    cfg['final_seeds'] = [int(s) for s in cfg['final_seeds']]
    cfg['selection_source_seed'] = int(cfg['selection_source_seed'])
    if cfg['selection_source_seed'] not in cfg['search_seeds']:
        raise ValueError('selection_source_seed must be included in search_seeds')
    cfg['out_dir'] = str(cfg['out_dir'])

    if 'promote_frac' in cfg:
        cfg['promote_frac'] = float(cfg['promote_frac'])
        if not (0.0 < cfg['promote_frac'] <= 1.0):
            raise ValueError('promote_frac must be in (0, 1]')

    test_topk = int(cfg.get('test_topk_per_tag', 1))
    if test_topk < 1:
        raise ValueError('test_topk_per_tag must be >= 1')
    cfg['test_topk_per_tag'] = test_topk

    test_tags = []
    for tag in cfg.get('test_from_tags', []):
        tag_name = str(tag).strip()
        if not tag_name:
            continue
        if tag_name.startswith('ablation_'):
            warnings.warn(f"Tag {tag_name!r} excluded from final evaluation (prefix ablation_).")
            continue
        if tag_name not in test_tags:
            test_tags.append(tag_name)
    cfg['test_from_tags'] = test_tags
    cfg.setdefault('final_retrain_mode', 'train_plus_val_fixed_epochs')
    cfg.setdefault('final_retrain_uses_validation', False)
    cfg.setdefault('test_selection_policy', {
        'summary_budget': 'highest_available_budget',
        'selection_protocol': {
            'name': 'validation_only_pareto_ideal_point_3d',
            'method': 'pareto_ideal_point_3d',
            'normalization': 'minmax',
        },
        'summary_selection_objectives': ['auc_loss', 'time_s', 'trainable_params_m'],
        'summary_selection_runtime_metrics': ['mean_val_auc', 'mean_time_s_per_run', 'mean_trainable_params_m'],
        'per_tag_candidates_sent_to_test': test_topk,
        'backbone_selection_objectives': ['auc_loss', 'time_s', 'trainable_params_m'],
        'backbone_selection_runtime_metrics': ['best_val_auc', 'time_s', 'trainable_params_m'],
    })
    return cfg


def load_config(path: str | None = None) -> Dict[str, Any]:
    config_path = Path(get_config_path(path))
    with config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['_config_path'] = str(config_path)
    return validate_config(cfg)
