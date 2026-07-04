from __future__ import annotations

import os
import math
import random
from typing import List, Tuple

from finetune_ga.core.genome import Genome, sample_genome
from finetune_ga.search.nsga2 import evaluate_population_with_sh
from finetune_ga.search.task_executor import SearchTaskExecutor
from finetune_ga.search.common import load_population_checkpoint, save_population_checkpoint, write_pareto_snapshot
from finetune_ga.infra.io_utils import atomic_write_json
from finetune_ga.search.operators import make_offspring, assign_rank_and_crowd, pareto_front
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main


def fitness_key(g: Genome) -> Tuple[float, float, float]:
    try:
        auc_loss, total_time_s, trainable_params_m = (float(v) for v in g.objectives)
    except (TypeError, ValueError):
        return (-1e9, -1e12, -1e12)
    if not math.isfinite(auc_loss):
        auc_loss = 1e9
    if not math.isfinite(total_time_s):
        total_time_s = 1e12
    if not math.isfinite(trainable_params_m):
        trainable_params_m = 1e12
    mean_val_auc = 1.0 - auc_loss
    return (mean_val_auc, -max(0.0, total_time_s), -max(0.0, trainable_params_m))


def select_top(pop: List[Genome], k: int) -> List[Genome]:
    return sorted(pop, key=fitness_key, reverse=True)[:k]


def save_checkpoint(root_dir: str, tag: str, gen_completed: int, pop: List[Genome], rng: random.Random) -> None:
    save_population_checkpoint(root_dir, tag, gen_completed, pop, rng)


def load_checkpoint(root_dir: str, tag: str, rng: random.Random) -> Tuple[int, List[Genome]]:
    return load_population_checkpoint(root_dir, tag, rng)


def run_for_seed(cfg, active_seed: int) -> None:
    tag = "single_objective_ga"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=False)
    root_dir = ctx["root_dir"]
    exp_dir = ctx["paths"]["exp_dir"]
    resume = bool(os.environ.get("RESUME", "1") == "1")
    rng = random.Random(int(ctx["active_seed"]) + 4040)
    start_gen, pop = load_checkpoint(root_dir, tag, rng) if resume else (0, [])
    if not pop:
        pop = [sample_genome(cfg, rng) for _ in range(int(cfg["pop_size"]))]
        start_gen = 0

    with SearchTaskExecutor(cfg) as task_executor:
        for gen in range(start_gen + 1, int(cfg["generations"]) + 1):
            print(f"\n================= {tag.upper()} GENERATION {gen}/{cfg['generations']} =================")
            parents = select_top(pop, k=max(2, len(pop) // 2)) if gen > 1 else pop[:]
            elite = select_top(pop, k=max(1, int(round(len(pop) * 0.15)))) if gen > 1 else []
            if gen == 1:
                combined = pop[:]
            else:
                children = make_offspring(
                    cfg,
                    parents if parents else pop,
                    n=max(1, len(pop) - len(elite)),
                    rng=rng,
                    allow_crossover=True,
                    allow_mutation=True,
                    deduplicate=True,
                )
                combined = (elite + children) if elite else (pop + children)
            evaluate_population_with_sh(
                cfg, gen, combined, root_dir,
                tag=tag, resume=resume, active_seed=int(active_seed),
                task_executor=task_executor,
            )
            pop = select_top(combined, k=int(cfg["pop_size"]))

            atomic_write_json(
                os.path.join(exp_dir, f"best_gen_{gen}.json"),
                [
                    {
                        "genome_id": g.genome_id,
                        "fitness": fitness_key(g)[0],
                        "objective_names": ["mean_val_auc_loss_obj", "selection_time_s_obj", "trainable_params_m_obj"],
                        "objectives": list(g.objectives),
                        "mean_val_auc_loss_obj": float(g.objectives[0]),
                        "mean_val_auc": float(1.0 - g.objectives[0]),
                        "selection_time_s_obj": float(g.objectives[1]),
                        "trainable_params_m_obj": float(g.objectives[2]),
                        "last_budget": g.last_budget,
                    }
                    for g in pop[:min(10, len(pop))]
                ],
                validate_finite=True,
            )
            save_checkpoint(root_dir, tag, gen_completed=gen, pop=pop, rng=rng)
            print("--- Single-objective best (top few) ---")
            for i, g in enumerate(pop[:6], start=1):
                mean_auc = 1.0 - g.objectives[0]
                print(f"{i:02d}) mean_val_auc={mean_auc:.4f} time={g.objectives[1]/60:.1f}min trainable={g.objectives[2]:.2f}M last={g.last_budget}")

    assign_rank_and_crowd(cfg, pop)
    pf = pareto_front(pop, cfg=cfg)
    atomic_write_json(
        os.path.join(exp_dir, "pareto_final.json"),
        [
            {"genome_id": g.genome_id, "objective_names": ["mean_val_auc_loss_obj", "selection_time_s_obj", "trainable_params_m_obj"], "objectives": list(g.objectives), "mean_val_auc_loss_obj": float(g.objectives[0]), "mean_val_auc": float(1.0 - g.objectives[0]), "selection_time_s_obj": float(g.objectives[1]), "trainable_params_m_obj": float(g.objectives[2]), "last_budget": g.last_budget}
            for g in pf
        ],
        validate_finite=True,
    )
    print(f"DONE. Single-objective GA baseline in {exp_dir}")

def main() -> None:
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
