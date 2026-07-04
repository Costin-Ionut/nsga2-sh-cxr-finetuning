import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


# ---------------------------------------------------------------------------
# Dataset / config mocks — împiedică operații de filesystem la import
# ---------------------------------------------------------------------------

_MOCK_DATASET_STATE = {
    'dataset_root': '/mock/dataset',
    'train_folder': '/mock/dataset/train',
    'val_folder': '/mock/dataset/val',
    'test_folder': '/mock/dataset/test',
    'num_train_images': 1000,
    'num_val_images': 200,
    'num_test_images': 200,
    'is_resolved': True,
}


@pytest.fixture(autouse=True)
def mock_dataset_config(monkeypatch, request):
    """
    Patchuiește starea globală a dataset-ului din config.py înainte de orice test.
    Fără asta, simpla importare a finetune_ga.infra.config declanșează
    numărarea imaginilor pe disk, care eșuează în CI sau fără dataset.
    """
    import finetune_ga.infra.config as cfg_module

    if getattr(request.module, '__name__', '').endswith('test_config_and_data'):
        return

    monkeypatch.setattr(cfg_module, '_RUNTIME_DATASET_STATE', dict(_MOCK_DATASET_STATE))
    monkeypatch.setattr(cfg_module, 'DATASET_ROOT', _MOCK_DATASET_STATE['dataset_root'])
    monkeypatch.setattr(cfg_module, 'TRAIN_FOLDER', _MOCK_DATASET_STATE['train_folder'])
    monkeypatch.setattr(cfg_module, 'VAL_FOLDER', _MOCK_DATASET_STATE['val_folder'])
    monkeypatch.setattr(cfg_module, 'TEST_FOLDER', _MOCK_DATASET_STATE['test_folder'])
    monkeypatch.setattr(cfg_module, 'NUM_TRAIN_IMAGES', _MOCK_DATASET_STATE['num_train_images'])
    monkeypatch.setattr(cfg_module, 'NUM_VAL_IMAGES', _MOCK_DATASET_STATE['num_val_images'])
    monkeypatch.setattr(cfg_module, 'NUM_TEST_IMAGES', _MOCK_DATASET_STATE['num_test_images'])

    monkeypatch.setattr(cfg_module, 'get_dataset_counts',
                        lambda refresh=False: {
                            'train': _MOCK_DATASET_STATE['num_train_images'],
                            'val':   _MOCK_DATASET_STATE['num_val_images'],
                            'test':  _MOCK_DATASET_STATE['num_test_images'],
                        })
    monkeypatch.setattr(cfg_module, 'get_dataset_paths',
                        lambda refresh=False: {
                            'dataset_root': _MOCK_DATASET_STATE['dataset_root'],
                            'train_folder': _MOCK_DATASET_STATE['train_folder'],
                            'val_folder':   _MOCK_DATASET_STATE['val_folder'],
                            'test_folder':  _MOCK_DATASET_STATE['test_folder'],
                        })
    monkeypatch.setattr(cfg_module, 'validate_dataset_layout',
                        lambda strict=True, refresh=False: True)


# ---------------------------------------------------------------------------
# Fixture opțional: cfg minim pentru teste care au nevoie de un dict de config
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_cfg():
    """Config minimal valid, folosit de testele care apelează sample_genome,
    make_offspring, summarize_budget_runs etc. fără să pornească training."""
    return {
        'search_seeds': [42],
        'final_seeds': [42],
        'selection_source_seed': 42,
        'num_classes': 2,
        'model_names': ['mobilenet', 'efficientnet'],
        'pop_size': 4,
        'generations': 2,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'promote_frac': 0.5,
        'out_dir': '/tmp/test_runs',
        'unfreeze_cap': {'mobilenet': 90, 'efficientnet': 140},
        'budgets': [
            {'name': 'B0', 'e1': 1, 'e2': 0, 'steps_factor': 0.25, 'reps': 1},
            {'name': 'B1', 'e1': 2, 'e2': 1, 'steps_factor': 0.5,  'reps': 1},
        ],
        'search_space': {
            'base_lr1':     [3e-5, 1e-4, 2e-4],
            'base_lr2':     [2e-6, 5e-6],
            'n_last_layers':[0, 5, 10],
            'dense_units':  [128, 256],
            'dropout':      [0.2, 0.5],
            'l2_weight':    [0.0, 1e-5],
            'batch_size':   [16, 32],
            'aug_rotation': [0.0, 0.1],
            'aug_zoom':     [0.0, 0.1],
            'aug_contrast': [0.0, 0.1],
            'lr_mul_bounds':[0.8, 1.4],
        },
        'complexity_weights': {'params_m': 0.55, 'time_h': 0.35, 'size_ratio': 0.10},
        'test_selection_policy': {
            'selection_protocol': {'name': 'validation_only_pareto_ideal_point_3d', 'method': 'pareto_ideal_point_3d', 'normalization': 'minmax'},
            'summary_selection_objectives': ['auc_loss', 'time_s', 'trainable_params_m'],
            'summary_selection_runtime_metrics': ['mean_val_auc', 'total_time_s', 'mean_trainable_params_m'],
            'backbone_selection_objectives': ['auc_loss', 'time_s', 'trainable_params_m'],
        },
    }