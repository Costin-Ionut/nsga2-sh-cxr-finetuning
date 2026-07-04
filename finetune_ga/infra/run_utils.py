"""run_utils.py — Experiment bookkeeping: seeding, environment capture, seed-dir helpers."""
from __future__ import annotations

import os
import platform
import subprocess
import zlib
from typing import Any, Dict

import numpy as np

from finetune_ga.infra import runtime
from finetune_ga.infra.runtime import tf, STRATEGY, _require_tensorflow
from finetune_ga.infra.utils import env_flag
from finetune_ga.infra.config import ENABLE_XLA


def _strict_tf_determinism_required() -> bool:
    return env_flag('STRICT_TF_DETERMINISM', '1')


def seed_everything(seed: int, *, include_tensorflow: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow for reproducibility.

    The training/search logic stays unchanged; this only reduces run-to-run
    variance by enabling deterministic TensorFlow execution when supported.
    """
    import random

    os.environ['PYTHONHASHSEED'] = str(seed)

    # Seed Python and NumPy first so they are always seeded regardless of
    # whether TensorFlow is available.
    random.seed(seed)
    np.random.seed(seed)

    if not include_tensorflow or runtime.get_tf_module_or_none() is None:
        return

    try:
        tf.keras.utils.set_random_seed(int(seed))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        if _strict_tf_determinism_required():
            raise RuntimeError('TensorFlow could not apply full reproducibility seeding.') from exc
        tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError, ValueError) as exc:
        if _strict_tf_determinism_required():
            raise RuntimeError('TensorFlow deterministic ops could not be enabled.') from exc


def stable_crc32(s: str) -> int:
    return int(zlib.crc32(s.encode('utf-8')) & 0xFFFFFFFF)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return 'unknown'


def get_env_info(*, include_tensorflow: bool = True) -> Dict[str, Any]:
    tf_module = runtime.get_tf_module_or_none() if include_tensorflow else None
    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'tf_version': getattr(tf_module, '__version__', None) if tf_module is not None else None,
        'np_version': np.__version__,
        'git_commit': git_commit_hash(),
        'xla_enabled': bool(ENABLE_XLA),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'multi_gpu_mode': os.environ.get('MULTI_GPU_MODE'),
        'tf_runtime_imported_in_this_process': bool(tf_module is not None),
    }
    if tf_module is None:
        info.update({'devices': [], 'num_replicas_in_sync': 1,
                     'mixed_precision_policy': None, 'num_gpus': 0})
        return info
    info.update({
        'devices': [d.name for d in tf.config.list_logical_devices()],
        'num_replicas_in_sync': int(getattr(STRATEGY, 'num_replicas_in_sync', 1)),
        'mixed_precision_policy': str(tf.keras.mixed_precision.global_policy()),
    })
    try:
        gpus = tf.config.list_physical_devices('GPU')
        info['num_gpus'] = len(gpus)
        if gpus:
            info['gpu_names'] = [g.name for g in gpus]
    except (RuntimeError, ValueError):
        pass
    return info


# ---------------------------------------------------------------------------
# Seed / root-dir helpers
# ---------------------------------------------------------------------------


def _require_cfg_key(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def get_search_seeds(cfg: Dict[str, Any]) -> list[int]:
    return [int(s) for s in _require_cfg_key(cfg, 'search_seeds')]


def get_final_seeds(cfg: Dict[str, Any]) -> list[int]:
    return [int(s) for s in _require_cfg_key(cfg, 'final_seeds')]


def get_selection_source_seed(cfg: Dict[str, Any]) -> int:
    return int(_require_cfg_key(cfg, 'selection_source_seed'))


def get_budget_reps(cfg: Dict[str, Any], budget: Dict[str, Any], *, mode: str) -> int:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"search", "final"}:
        raise ValueError(f"Unsupported mode {mode!r}. Expected 'search' or 'final'.")

    mode_key = f"{mode_normalized}_reps"
    if mode_key in budget:
        reps = int(budget[mode_key])
    else:
        reps = int(budget.get("reps", 1))

    if reps <= 0:
        raise ValueError(f"Budget {budget.get('name', '<unknown>')} must have {mode_key} > 0")
    return reps


def get_budget_promote_frac(cfg: Dict[str, Any], budget: Dict[str, Any]) -> float:
    raw = budget.get("promote_frac", cfg.get("promote_frac", 1.0))
    value = float(raw)
    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"Budget {budget.get('name', '<unknown>')} must have promote_frac in (0, 1]."
        )
    return value


def seed_root_for(base_out_dir: str, seed: int, seed_list: list[int]) -> str:
    return os.path.join(str(base_out_dir), f'seed_{int(seed)}') if len(seed_list) > 1 else str(base_out_dir)


def get_seeded_root_dir(base_out_dir: str, active_seed: int, seed_list: list[int]) -> str:
    return seed_root_for(str(base_out_dir), int(active_seed), [int(s) for s in seed_list])
