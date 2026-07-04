"""ablation_runner.py — Single entry point for all ablation experiments.

Replaces the four separate ablation_no_*.py files.

Usage:
    python ablation_runner.py --ablation no_crossover
    python ablation_runner.py --ablation no_mutation
    python ablation_runner.py --ablation no_elitism
    python ablation_runner.py --ablation no_dedup
"""
from __future__ import annotations

import argparse

from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.run_utils import get_search_seeds
from finetune_ga.infra.io_utils import clear_runs_cache
from finetune_ga.search.nsga2 import run_nsga2_experiment

ABLATION_CONFIGS = {
    "no_crossover": dict(tag="ablation_no_crossover", allow_crossover=False),
    "no_mutation":  dict(tag="ablation_no_mutation",  allow_mutation=False),
    "no_elitism":   dict(tag="ablation_no_elitism",   use_elitism=False),
    "no_dedup":     dict(tag="ablation_no_dedup",     deduplicate=False),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ablation experiment.")
    parser.add_argument(
        "--ablation", required=True, choices=list(ABLATION_CONFIGS),
        help="Which component to ablate.",
    )
    args = parser.parse_args()
    kwargs = ABLATION_CONFIGS[args.ablation]

    cfg = load_config()
    seeds = get_search_seeds(cfg)
    for seed in seeds:
        print(f"\n########## ABLATION={args.ablation} seed={seed} ##########")
        clear_runs_cache()
        run_nsga2_experiment(cfg, active_seed=int(seed), **kwargs)


if __name__ == "__main__":
    main()
