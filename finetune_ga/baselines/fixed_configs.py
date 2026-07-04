from __future__ import annotations

import json
import os
from typing import List

from finetune_ga.core.genome import Genome, assign_genome_id
from finetune_ga.infra.io_utils import load_experiment_state, safe_int
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.baselines.full_budget import (
    evaluate_genome_at_final_budget,
    save_completed_index_state,
)
from finetune_ga.infra.experiment_records import load_done_keys_from_path
from finetune_ga.search.task_executor import SearchTaskExecutor


def make_fixed_genomes(cfg) -> List[Genome]:
    model_names = cfg["model_names"]

    def full(mult=1.0):
        return {m: float(mult) for m in model_names}

    genomes = [
        Genome(8e-5, 6e-6, 5, 128, 0.25, 1e-5, 32, 0.02, 0.02, 0.02, full(1.0), full(1.0)),
        Genome(1.5e-4, 1.0e-5, 10, 256, 0.35, 1e-6, 32, 0.06, 0.06, 0.06, full(1.0), full(1.0)),
        Genome(2.2e-4, 1.6e-5, 20, 512, 0.45, 1e-5, 16, 0.10, 0.10, 0.10, full(1.0), full(1.0)),
        Genome(1.1e-4, 8e-6, 40, 256, 0.40, 1e-4, 16, 0.14, 0.14, 0.14, full(1.0), full(1.0)),
        Genome(3.5e-4, 2.0e-5, 60, 512, 0.55, 0.0, 16, 0.18, 0.18, 0.18, full(1.0), full(1.0)),
    ]
    for genome in genomes:
        assign_genome_id(genome, model_names)
    return genomes


def _write_manifest(exp_dir: str, genomes: List[Genome]) -> None:
    with open(os.path.join(exp_dir, "fixed_configs_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([
            {
                "index": idx,
                "genome_id": g.genome_id,
                "base_lr1": g.base_lr1,
                "base_lr2": g.base_lr2,
                "n_last_layers": g.n_last_layers,
                "dense_units": g.dense_units,
                "dropout": g.dropout,
                "l2_weight": g.l2_weight,
                "batch_size": g.batch_size,
                "aug_rotation": g.aug_rotation,
                "aug_zoom": g.aug_zoom,
                "aug_contrast": g.aug_contrast,
                "lr1_mul": g.lr1_mul,
                "lr2_mul": g.lr2_mul,
            }
            for idx, g in enumerate(genomes, start=1)
        ], f, indent=2, allow_nan=False)


def run_for_seed(cfg, active_seed: int):
    tag = "fixed_configs"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    root_dir = ctx["root_dir"]
    exp_dir = ctx["paths"]["exp_dir"]
    runs_path = os.path.join(exp_dir, "runs.jsonl")
    summary_path = os.path.join(exp_dir, "genome_summary.jsonl")
    genomes = make_fixed_genomes(cfg)
    resume = bool(os.environ.get("RESUME", "1") == "1")
    state = load_experiment_state(root_dir, tag) if resume else {}
    completed_index = safe_int(state.get("completed_index", 0), 0)
    done_keys = load_done_keys_from_path(runs_path)

    with SearchTaskExecutor(cfg) as task_executor:
        for idx, genome in enumerate(genomes, start=1):
            if idx <= completed_index:
                continue
            print(f"\n===== FIXED CONFIG {idx}/{len(genomes)} | seed={ctx['active_seed']} =====")
            evaluate_genome_at_final_budget(
                cfg,
                genome,
                tag=tag,
                root_dir=root_dir,
                runs_path=runs_path,
                summary_path=summary_path,
                done_keys=done_keys,
                extra_summary={"fixed_config_index": idx},
                active_seed=int(active_seed),
                task_executor=task_executor,
            )
            save_completed_index_state(root_dir, tag, idx)

    _write_manifest(exp_dir, genomes)
    print(f"DONE. Fixed-config baseline in {exp_dir}")

def main():
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
