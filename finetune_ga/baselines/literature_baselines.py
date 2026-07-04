from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.baselines.full_budget import (
    evaluate_genome_at_final_budget,
    save_completed_index_state,
)
from finetune_ga.infra.experiment_records import load_done_keys_from_path
from finetune_ga.search.task_executor import SearchTaskExecutor
from finetune_ga.core.genome import Genome, assign_genome_id
from finetune_ga.infra.io_utils import load_experiment_state, safe_int


STANDARD_BACKBONE_MULT = 1.0


def _full(model_names: List[str], mult: float = STANDARD_BACKBONE_MULT) -> Dict[str, float]:
    return {m: float(mult) for m in model_names}


def make_feature_extractor_genomes(cfg) -> List[Genome]:
    model_names = cfg["model_names"]
    genome = Genome(
        base_lr1=1e-3,
        base_lr2=1e-5,
        n_last_layers=0,
        dense_units=256,
        dropout=0.30,
        l2_weight=1e-5,
        batch_size=32,
        aug_rotation=0.05,
        aug_zoom=0.05,
        aug_contrast=0.05,
        lr1_mul=_full(model_names),
        lr2_mul=_full(model_names),
    )
    assign_genome_id(genome, model_names)
    return [genome]


def make_standard_finetune_genomes(cfg) -> List[Genome]:
    model_names = cfg["model_names"]
    genome = Genome(
        base_lr1=1e-4,
        base_lr2=1e-5,
        n_last_layers=10,
        dense_units=256,
        dropout=0.30,
        l2_weight=1e-5,
        batch_size=32,
        aug_rotation=0.08,
        aug_zoom=0.08,
        aug_contrast=0.08,
        lr1_mul=_full(model_names),
        lr2_mul=_full(model_names),
    )
    assign_genome_id(genome, model_names)
    return [genome]


def make_manual_tuned_genomes(cfg) -> List[Genome]:
    model_names = cfg["model_names"]
    grid: List[Tuple[float, int, float, int]] = [
        (1e-3, 5, 0.30, 32),
        (1e-3, 10, 0.30, 32),
        (1e-4, 10, 0.30, 32),
        (1e-4, 20, 0.30, 32),
        (1e-4, 20, 0.40, 16),
        (5e-5, 10, 0.40, 16),
        (5e-5, 20, 0.40, 16),
        (5e-5, 40, 0.50, 16),
    ]
    genomes: List[Genome] = []
    for lr1, n_last_layers, dropout, batch_size in grid:
        genome = Genome(
            base_lr1=lr1,
            base_lr2=min(1e-5, lr1 / 10.0),
            n_last_layers=n_last_layers,
            dense_units=256,
            dropout=dropout,
            l2_weight=1e-5,
            batch_size=batch_size,
                aug_rotation=0.08,
            aug_zoom=0.08,
            aug_contrast=0.08,
            lr1_mul=_full(model_names),
            lr2_mul=_full(model_names),
        )
        assign_genome_id(genome, model_names)
        genomes.append(genome)
    return genomes


BASELINE_SPECS = {
    "baseline_feature_extractor": {
        "description": "Transfer-learning baseline with frozen backbone and trained classifier head only.",
        "genome_factory": make_feature_extractor_genomes,
    },
    "baseline_standard_finetune": {
        "description": "Single standard fine-tuning recipe without search, using partial unfreezing.",
        "genome_factory": make_standard_finetune_genomes,
    },
    "baseline_manual_tuned": {
        "description": "Small manual tuning grid over common fine-tuning settings, without evolutionary or Bayesian search.",
        "genome_factory": make_manual_tuned_genomes,
    },
}


def _write_manifest(exp_dir: str, tag: str, description: str, genomes: List[Genome]) -> None:
    payload = {
        "tag": tag,
        "description": description,
        "selection_rule": "Choose the best validation-performing configuration for this tag, then select a single backbone on validation before final test retrain.",
        "num_genomes": len(genomes),
        "genomes": [
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
        ],
    }
    with open(os.path.join(exp_dir, f"{tag}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def run_baseline_for_seed(cfg, active_seed: int, *, tag: str) -> None:
    spec = BASELINE_SPECS[tag]
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    root_dir = ctx["root_dir"]
    exp_dir = ctx["paths"]["exp_dir"]
    runs_path = os.path.join(exp_dir, "runs.jsonl")
    summary_path = os.path.join(exp_dir, "genome_summary.jsonl")
    genomes = spec["genome_factory"](cfg)
    resume = bool(os.environ.get("RESUME", "1") == "1")
    state = load_experiment_state(root_dir, tag) if resume else {}
    completed_index = safe_int(state.get("completed_index", 0), 0)
    done_keys = load_done_keys_from_path(runs_path)

    with SearchTaskExecutor(cfg) as task_executor:
        for idx, genome in enumerate(genomes, start=1):
            if idx <= completed_index:
                continue
            print(f"\n===== {tag.upper()} {idx}/{len(genomes)} | seed={ctx['active_seed']} =====")
            evaluate_genome_at_final_budget(
                cfg,
                genome,
                tag=tag,
                root_dir=root_dir,
                runs_path=runs_path,
                summary_path=summary_path,
                done_keys=done_keys,
                extra_summary={"baseline_index": idx, "baseline_description": spec["description"]},
                extra_run_fields={"baseline_tag": tag},
                active_seed=int(active_seed),
                task_executor=task_executor,
            )
            save_completed_index_state(root_dir, tag, idx)

    _write_manifest(exp_dir, tag, spec["description"], genomes)
    print(f"DONE. {tag} baseline in {exp_dir}")

def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run curated literature-style baselines.")
    parser.add_argument(
        "--tag",
        required=True,
        choices=sorted(BASELINE_SPECS.keys()),
        help="Baseline tag to run.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_seeded_main(lambda cfg, active_seed: run_baseline_for_seed(cfg, active_seed, tag=args.tag))


if __name__ == "__main__":
    main()
