from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List

from finetune_ga.infra.io_utils import prepare_output_paths, clear_runs_cache, save_run_config_snapshot
from finetune_ga.infra.run_utils import (
    seed_everything,
    get_env_info,
    get_seeded_root_dir,
    get_search_seeds,
)
from finetune_ga.infra.experiment_config import load_config
from finetune_ga.search.task_executor import pin_single_gpu_if_single_worker, is_parallel_search_enabled


def ensure_env_file(run_dir: str, *, include_tensorflow: bool = True) -> None:
    env_path = os.path.join(run_dir, "env.json")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(get_env_info(include_tensorflow=include_tensorflow), f, indent=2, allow_nan=False)


def initialize_baseline_run(cfg: Dict[str, Any], tag: str, active_seed: int, *, ensure_dataset_counts: bool = False) -> Dict[str, Any]:
    parallel_search = is_parallel_search_enabled(cfg)
    if not parallel_search:
        pin_single_gpu_if_single_worker(cfg)
    root_dir = get_seeded_root_dir(cfg['out_dir'], int(active_seed), get_search_seeds(cfg))
    seed_everything(int(active_seed), include_tensorflow=not parallel_search)
    paths = prepare_output_paths(root_dir, tag)
    ensure_env_file(paths["exp_dir"], include_tensorflow=not parallel_search)
    save_run_config_snapshot(paths["exp_dir"], cfg, tag=tag, active_seed=int(active_seed))
    if ensure_dataset_counts:
        from finetune_ga.infra.config import get_dataset_counts
        get_dataset_counts(refresh=True)
    return {
        "tag": tag,
        "active_seed": active_seed,
        "root_dir": root_dir,
        "paths": paths,
    }


def run_seeded_main(run_for_seed: Callable[[Dict[str, Any], int], None]) -> None:
    cfg = load_config()
    seeds: Iterable[int] = get_search_seeds(cfg)
    for seed in seeds:
        clear_runs_cache()
        run_for_seed(cfg, int(seed))
