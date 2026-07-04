from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from finetune_ga.data.loader import _manifest_file_list
from finetune_ga.infra.config import dataset_config_snapshot
from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.run_utils import git_commit_hash
from finetune_ga.infra.test_protocol import build_test_protocol_id


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_sha256(config_path: str) -> str:
    path = Path(config_path)
    return _sha256_bytes(path.read_bytes())


def _split_manifest(split_folder: str) -> Dict[str, Any]:
    folder = Path(split_folder)
    if not folder.is_dir():
        return {
            'path': str(folder),
            'exists': False,
            'count': 0,
            'classes': {},
            'file_list_sha256': None,
        }

    rel_paths = _manifest_file_list(folder)
    class_counts: Dict[str, int] = {}
    for rel in rel_paths:
        parts = rel.split('/')
        class_name = parts[0] if parts else 'unknown'
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    payload = '\n'.join(rel_paths).encode('utf-8')
    return {
        'path': str(folder),
        'exists': True,
        'count': len(rel_paths),
        'classes': dict(sorted(class_counts.items())),
        'file_list_sha256': _sha256_bytes(payload),
    }


def build_run_manifest(config_path: str | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    snap = dataset_config_snapshot(refresh=True)
    protocol_id = build_test_protocol_id(cfg)
    cfg_path = str(Path(cfg['_config_path']).resolve())
    manifest = {
        'git_commit_hash': git_commit_hash(),
        'config_path': cfg_path,
        'config_sha256': config_sha256(cfg_path),
        'protocol_id': protocol_id,
        'dataset_manifest': {
            'dataset_root': snap['dataset_root'],
            'dataset_root_source': snap['dataset_root_source'],
            'runtime_environment': snap['runtime_environment'],
            'dataset_handle': snap['dataset_handle'],
            'dataset_version': snap['dataset_version'],
            'auto_split_dataset': bool(snap['auto_split_dataset']),
            'deterministic_dataset_required': bool(snap['deterministic_dataset_required']),
            'splits': {
                'train': _split_manifest(snap['train_folder']),
                'val': _split_manifest(snap['val_folder']),
                'test': _split_manifest(snap['test_folder']),
            },
        },
        'search_seeds': [int(s) for s in cfg['search_seeds']],
        'final_seeds': [int(s) for s in cfg['final_seeds']],
        'selection_source_seed': int(cfg['selection_source_seed']),
        'out_dir': str(cfg['out_dir']),
        'test_from_tags': list(cfg.get('test_from_tags', [])),
        'test_topk_per_tag': int(cfg.get('test_topk_per_tag', 1)),
    }
    return manifest


def write_run_manifest(config_path: str | None = None, output_dir: str | None = None) -> str:
    manifest = build_run_manifest(config_path)
    out_dir = Path(output_dir or manifest['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'run_manifest.json'
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return str(out_path)
