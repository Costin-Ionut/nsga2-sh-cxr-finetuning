"""nsga2_sh_main.py — NSGA-II + Successive Halving main experiment loop.

Genetic operators (dominance, selection, crossover, mutation) live in
operators.py. Training logic lives in training.py. This file owns only the
NSGA-II/SH-specific policy, while shared orchestration lives in search/common.py.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import random
from typing import Any, Callable, Dict, List, Tuple

from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.experiment_records import build_summary_row, dedupe_run_rows, load_done_keys_from_path, select_best_run
from finetune_ga.infra.run_utils import (
    get_budget_promote_frac,
    get_budget_reps,
    seed_everything,
    get_env_info,
    get_seeded_root_dir,
    get_search_seeds,
)
from finetune_ga.infra.io_utils import (
    read_jsonl,
    prepare_output_paths,
    clear_runs_cache,
    save_run_config_snapshot,
)
from finetune_ga.core.genome import Genome, sample_genome, assign_genome_id, genome_to_dict
from finetune_ga.core.metrics import summarize_budget_runs, objectives_from_summary
from finetune_ga.search.common import (
    load_population_checkpoint,
    log_top_population,
    run_generational_search,
    save_population_checkpoint,
    write_genome_summary_everywhere,
    write_pareto_snapshot,
    write_run_row_everywhere,
)
from finetune_ga.search.operators import assign_rank_and_crowd, nsga2_select, make_offspring, pareto_front, has_valid_objectives
from finetune_ga.search.task_executor import (
    SearchTaskExecutor,
    pin_single_gpu_if_single_worker,
    is_parallel_search_enabled,
    _resolve_search_worker_count,
    _resolve_visible_gpu_ids,
    _run_search_task_local,
)


def train_one_backbone_search(*args, **kwargs):
    from finetune_ga.core.training import train_one_backbone_search as _train_one_backbone_search
    return _train_one_backbone_search(*args, **kwargs)


def train_one_backbone(*args, **kwargs):
    from finetune_ga.core.training import train_one_backbone as _train_one_backbone
    return _train_one_backbone(*args, **kwargs)


def _build_search_tasks(
    cfg: Dict[str, Any],
    survivors: List[Genome],
    budget: Dict[str, Any],
    *,
    root_dir: str,
    tag: str,
    gen_idx: int,
    active_seed: int,
    done_keys: set[tuple[str, str, str, int]],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    budget_name = str(budget["name"])
    search_reps = get_budget_reps(cfg, budget, mode='search')

    for genome in survivors:
        genome_payload = genome_to_dict(genome)
        for rep in range(search_reps):
            for backbone in cfg["model_names"]:
                key = (genome.genome_id, budget_name, backbone, rep)
                if key in done_keys:
                    continue
                tasks.append({
                    "cfg": cfg,
                    "genome": genome_payload,
                    "backbone": backbone,
                    "budget": budget,
                    "rep": int(rep),
                    "root_dir": root_dir,
                    "tag": tag,
                    "gen_idx": int(gen_idx),
                    "active_seed": int(active_seed),
                    "mode": "search",
                })
    return tasks


def evaluate_population_with_sh(
    cfg: Dict[str, Any], gen_idx: int, pop: List[Genome],
    root_dir: str, tag: str, resume: bool, active_seed: int,
    task_executor: SearchTaskExecutor | None = None,
    on_budget_complete: Callable[[str, List[Genome]], None] | None = None,
) -> None:
    exp_dir = prepare_output_paths(root_dir, tag)["exp_dir"]
    runs_path = os.path.join(exp_dir, "runs.jsonl")
    done_keys = load_done_keys_from_path(runs_path) if (resume and os.path.exists(runs_path)) else set()

    survivors = pop[:]
    budgets = cfg["budgets"]

    current_genome_ids = {g.genome_id for g in pop}
    runs_by_genome_budget: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    if resume and os.path.exists(runs_path):
        for row in dedupe_run_rows(read_jsonl(runs_path)):
            genome_id = row.get("genome_id")
            budget_name = row.get("budget")
            if genome_id is None or budget_name is None:
                continue
            if genome_id not in current_genome_ids:
                continue
            key = (genome_id, budget_name)
            runs_by_genome_budget.setdefault(key, []).append(row)

    for bi, budget in enumerate(budgets):
        budget_name = budget["name"]
        tasks = _build_search_tasks(
            cfg,
            survivors,
            budget,
            root_dir=root_dir,
            tag=tag,
            gen_idx=gen_idx,
            active_seed=int(active_seed),
            done_keys=done_keys,
        )
        print(f"[SEARCH] generation={gen_idx} | budget={budget_name} | survivors={len(survivors)} | scheduled_tasks={len(tasks)}")

        def record_completed_row(row: Dict[str, Any]) -> None:
            key = (str(row["genome_id"]), str(row["budget"]), str(row["backbone"]), int(row["rep"]))
            key_gb = (str(row["genome_id"]), str(row["budget"]))
            if key in done_keys:
                return
            write_run_row_everywhere(root_dir, tag, row)
            done_keys.add(key)
            runs_by_genome_budget.setdefault(key_gb, []).append(row)

        if task_executor is not None:
            task_executor.run_tasks(tasks, on_result=record_completed_row)
        else:
            _run_search_tasks_ephemeral(cfg, tasks, on_result=record_completed_row)

        for g in survivors:
            key_gb = (g.genome_id, budget_name)
            collected = runs_by_genome_budget.get(key_gb, [])
            if not collected:
                continue

            summ = summarize_budget_runs(cfg, collected, budget_name)
            obj = objectives_from_summary(summ)
            g.objectives = obj
            g.last_budget = budget_name

            best_run = select_best_run(collected)
            summary_row = build_summary_row(
                gen_idx=gen_idx,
                tag=tag,
                genome_id=g.genome_id,
                budget_name=budget_name,
                summary=summ,
                objectives=obj,
                best_run=best_run,
            )
            write_genome_summary_everywhere(root_dir, tag, summary_row)

        evaluated_at_budget = [
            g for g in pop
            if getattr(g, "objectives", None) is not None and str(getattr(g, "last_budget", "")) == str(budget_name)
        ]
        if evaluated_at_budget:
            pf_budget = pareto_front(evaluated_at_budget, cfg=cfg)
            budget_slug = str(budget_name).replace(os.sep, "_").replace("/", "_")
            write_pareto_snapshot(
                exp_dir,
                f"pareto_gen_{gen_idx}_budget_{budget_slug}.json",
                pf_budget,
            )
            print(
                f"[SAVE] generation={gen_idx} | budget={budget_name} | "
                f"saved budget Pareto snapshot ({len(pf_budget)} genomes)"
            )
        if on_budget_complete is not None:
            on_budget_complete(str(budget_name), pop)

        if bi < len(budgets) - 1:
            evaluated_survivors = [g for g in survivors if getattr(g, "objectives", None) is not None]
            if not evaluated_survivors:
                continue

            assign_rank_and_crowd(cfg, evaluated_survivors)
            promote_frac = get_budget_promote_frac(cfg, budget)
            survivors = nsga2_select(
                cfg,
                evaluated_survivors,
                k=max(1, int(math.ceil(len(evaluated_survivors) * promote_frac))),
            )


def _run_search_tasks_ephemeral(cfg: Dict[str, Any], tasks: List[Dict[str, Any]], on_result=None) -> List[Dict[str, Any]]:
    with SearchTaskExecutor(cfg) as executor:
        return executor.run_tasks(tasks, on_result=on_result)


def save_nsga2_checkpoint(root_dir: str, tag: str, gen_completed: int, pop: List[Genome], rng: random.Random) -> None:
    save_population_checkpoint(root_dir, tag, gen_completed, pop, rng)


def load_nsga2_checkpoint(root_dir: str, tag: str, rng: random.Random) -> Tuple[int, List[Genome]]:
    return load_population_checkpoint(root_dir, tag, rng)


def run_nsga2_experiment(
    cfg: Dict[str, Any], active_seed: int, tag: str = "nsga2",
    allow_crossover: bool = True, allow_mutation: bool = True,
    use_elitism: bool = True, deduplicate: bool = True, no_conditioning: bool = False,
) -> None:
    parallel_search = is_parallel_search_enabled(cfg)
    if not parallel_search:
        pin_single_gpu_if_single_worker(cfg)
    seed_everything(int(active_seed), include_tensorflow=not parallel_search)
    root_dir = get_seeded_root_dir(cfg["out_dir"], int(active_seed), get_search_seeds(cfg))
    paths = prepare_output_paths(root_dir, tag)
    save_run_config_snapshot(paths["exp_dir"], cfg, tag=tag, active_seed=int(active_seed))
    env_path = os.path.join(paths["exp_dir"], "env.json")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(get_env_info(include_tensorflow=not parallel_search), f, indent=2, allow_nan=False)

    resume = bool(os.environ.get("RESUME", "1") == "1")
    rng = random.Random(int(active_seed))
    from finetune_ga.infra.config import get_dataset_counts
    get_dataset_counts(refresh=True)

    def maybe_force_no_conditioning(g: Genome) -> Genome:
        if not no_conditioning:
            return g
        for m in cfg["model_names"]:
            g.lr1_mul[m.lower()] = 1.0
            g.lr2_mul[m.lower()] = 1.0
        assign_genome_id(g, cfg["model_names"])
        return g

    start_gen, pop = load_nsga2_checkpoint(root_dir, tag, rng) if resume else (0, [])
    if not pop:
        pop = [maybe_force_no_conditioning(sample_genome(cfg, rng)) for _ in range(int(cfg["pop_size"]))]
        start_gen = 0

    with SearchTaskExecutor(cfg) as task_executor:
        def per_generation(gen: int, population: List[Genome]) -> List[Genome]:
            is_initial_generation = (start_gen == 0 and gen == 1)

            if not is_initial_generation:
                assign_rank_and_crowd(cfg, population)
                parents = nsga2_select(cfg, population, k=max(2, len(population) // 2))
            else:
                parents = []

            elite = nsga2_select(cfg, population, k=max(1, int(round(len(population) * 0.15)))) if (not is_initial_generation and use_elitism) else []

            if is_initial_generation:
                print(f"[GEN 1] Evaluating initial random population ({len(population)} genomes, NSGA-II initialisation phase)")
                combined = population[:]
            else:
                if start_gen > 0 and gen == start_gen + 1:
                    print(f"[RESUME] Restarting from checkpoint after completed generation {start_gen}; rank/crowding are recomputed from stored population.")
                children = make_offspring(
                    cfg, parents if parents else population, n=max(1, len(population) - len(elite)),
                    rng=rng, allow_crossover=allow_crossover, allow_mutation=allow_mutation, deduplicate=deduplicate,
                )
                conditioned_elite = [maybe_force_no_conditioning(deepcopy(g)) for g in elite] if elite else []
                conditioned_children = [maybe_force_no_conditioning(c) for c in children]
                combined = conditioned_elite + conditioned_children
            def save_budget_state(budget_name: str, budget_population: List[Genome]) -> None:
                # Separate from checkpoint_state.json so resume still treats the
                # generation as incomplete until all budgets finish.
                save_population_checkpoint(
                    root_dir,
                    f"{tag}_latest_budget",
                    gen_completed=gen,
                    pop=budget_population,
                    rng=rng,
                )

            evaluate_population_with_sh(
                cfg,
                gen,
                combined,
                root_dir,
                tag=tag,
                resume=resume,
                active_seed=int(active_seed),
                task_executor=task_executor,
                on_budget_complete=save_budget_state,
            )
            evaluated = [g for g in combined if has_valid_objectives(g)] or combined
            if len(evaluated) < len(combined):
                print(f"[WARN] gen={gen}: {len(combined)-len(evaluated)} genome(s) have sentinel objectives.")
            assign_rank_and_crowd(cfg, evaluated)
            next_pop = nsga2_select(cfg, evaluated, k=min(int(cfg["pop_size"]), len(evaluated)))
            pf = pareto_front(next_pop, cfg=cfg)
            write_pareto_snapshot(paths["exp_dir"], f"pareto_gen_{gen}.json", pf)
            log_top_population(pf, title="--- Pareto (top few) ---")
            return next_pop

        def save_checkpoint(gen: int, population: List[Genome]) -> None:
            save_nsga2_checkpoint(root_dir, tag, gen_completed=gen, pop=population, rng=rng)

        def final_population(population: List[Genome]) -> List[Genome]:
            return pareto_front(population, cfg=cfg)

        run_generational_search(
            cfg=cfg,
            tag=tag,
            root_dir=root_dir,
            exp_dir=paths["exp_dir"],
            start_gen=start_gen,
            initial_population=pop,
            per_generation=per_generation,
            save_checkpoint=save_checkpoint,
            final_population=final_population,
            completion_message=f"\nDONE. Outputs in: {root_dir}",
        )


def main() -> None:
    cfg = load_config()
    seeds = get_search_seeds(cfg)
    for seed in seeds:
        print(f"\n########## RUN seed={seed} ##########")
        clear_runs_cache()
        run_nsga2_experiment(cfg, active_seed=int(seed), tag="nsga2")


if __name__ == "__main__":
    main()
