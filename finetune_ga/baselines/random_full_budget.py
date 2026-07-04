from __future__ import annotations

import ast
import json
import os
import random
from typing import List

from finetune_ga.core.genome import Genome, sample_genome, genome_to_dict, genome_from_dict
from finetune_ga.infra.io_utils import load_experiment_state, safe_int
from finetune_ga.search.operators import assign_rank_and_crowd, nsga2_select
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.baselines.full_budget import (
    evaluate_genome_at_final_budget,
    save_completed_index_state,
)
from finetune_ga.infra.experiment_records import load_done_keys_from_path
from finetune_ga.search.task_executor import SearchTaskExecutor


def _load_population_from_state(state: dict, tag: str) -> List[Genome]:
    pop: List[Genome] = []
    raw_population = state.get("population", [])

    if not isinstance(raw_population, list):
        if raw_population not in (None, []):
            print(
                f"[WARN] Invalid checkpoint population container for {tag}: "
                f"{type(raw_population).__name__}; expected list. Ignoring checkpoint population."
            )
        return pop

    for i, item in enumerate(raw_population):
        try:
            if not isinstance(item, dict):
                raise TypeError(f"population[{i}] is {type(item).__name__}, expected dict")
            pop.append(genome_from_dict(item))
        except Exception as exc:
            print(
                f"[WARN] Skipping corrupted checkpoint genome at index {i} "
                f"for {tag}: {type(exc).__name__}: {exc}"
            )

    return pop


def run_for_seed(cfg, active_seed: int):
    tag = "random_full_budget"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    active_seed = ctx["active_seed"]
    root_dir = ctx["root_dir"]
    exp_dir = ctx["paths"]["exp_dir"]
    resume = bool(os.environ.get("RESUME", "1") == "1")
    rng = random.Random(int(active_seed) + 2026)
    runs_path = os.path.join(exp_dir, "runs.jsonl")
    summary_path = os.path.join(exp_dir, "genome_summary.jsonl")

    state = load_experiment_state(root_dir, tag) if resume else {}

    if state.get("rng_state"):
        try:
            rng.setstate(ast.literal_eval(state["rng_state"]))
        except (ValueError, SyntaxError, TypeError) as exc:
            print(f"[WARN] Could not restore RNG state for {tag}: {type(exc).__name__}: {exc}")

    pop = _load_population_from_state(state, tag)

    completed_index = safe_int(state.get("completed_index", 0), 0)
    completed_index = max(0, min(completed_index, len(pop)))

    if not pop:
        pop = [sample_genome(cfg, rng) for _ in range(int(cfg["pop_size"]) * int(cfg["generations"]))]
        completed_index = 0

    done_keys = load_done_keys_from_path(runs_path)

    with SearchTaskExecutor(cfg) as task_executor:
        for idx, genome in enumerate(pop, start=1):
            if idx <= completed_index:
                continue

            print(f"\n===== RANDOM FULL BUDGET {idx}/{len(pop)} | seed={active_seed} =====")

            evaluate_genome_at_final_budget(
                cfg,
                genome,
                tag=tag,
                root_dir=root_dir,
                runs_path=runs_path,
                summary_path=summary_path,
                done_keys=done_keys,
                active_seed=int(active_seed),
                task_executor=task_executor,
            )

            save_completed_index_state(
                root_dir,
                tag,
                idx,
                population=[genome_to_dict(x) for x in pop],
                rng_state=repr(rng.getstate()),
                active_seed=int(active_seed),
            )

    assign_rank_and_crowd(cfg, pop)
    best = nsga2_select(cfg, pop, k=min(10, len(pop)))

    with open(os.path.join(exp_dir, "best_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "genome_id": g.genome_id,
                    "objective_names": ["mean_val_auc_loss_obj", "selection_time_s_obj", "trainable_params_m_obj"],
                    "objectives": list(g.objectives),
                    "mean_val_auc_loss_obj": float(g.objectives[0]),
                    "mean_val_auc": float(1.0 - g.objectives[0]),
                    "selection_time_s_obj": float(g.objectives[1]),
                    "trainable_params_m_obj": float(g.objectives[2]),
                    "last_budget": g.last_budget,
                }
                for g in best
            ],
            f,
            indent=2,
            allow_nan=False,
        )

    print(f"DONE. Random full-budget baseline in {exp_dir}")

def main():
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
