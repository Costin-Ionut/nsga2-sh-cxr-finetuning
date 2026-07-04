from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Any, Dict, List

from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.core.genome import Genome, sample_genome, genome_to_dict
from finetune_ga.core.metrics import objectives_from_summary, summarize_budget_runs
from finetune_ga.infra.experiment_records import (
    build_summary_row,
    dedupe_run_rows,
    load_done_keys_from_path,
    select_best_run,
)
from finetune_ga.infra.io_utils import prepare_output_paths, read_jsonl
from finetune_ga.search.common import (
    load_population_checkpoint,
    log_top_population,
    run_generational_search,
    save_population_checkpoint,
    write_genome_summary_everywhere,
    write_pareto_snapshot,
    write_run_row_everywhere,
)
from finetune_ga.infra.run_utils import get_budget_reps
from finetune_ga.search.task_executor import SearchTaskExecutor
from finetune_ga.search.operators import (
    assign_rank_and_crowd,
    make_offspring,
    nsga2_select,
    pareto_front,
    has_valid_objectives,
)


def _sort_run_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("genome_id", "")),
            str(r.get("budget", "")),
            str(r.get("backbone", "")),
            int(r.get("rep", 0)),
        ),
    )


def _evaluate_genome_full_budget_everywhere(
    cfg: Dict[str, Any],
    genome: Genome,
    *,
    tag: str,
    root_dir: str,
    done_keys: set,
    gen_idx: int = 0,
    active_seed: int = 0,
    task_executor: SearchTaskExecutor | None = None,
) -> Dict[str, Any]:
    budget = cfg["budgets"][-1]
    budget_name = str(budget["name"])

    exp_runs_path = os.path.join(prepare_output_paths(root_dir, tag)["exp_dir"], "runs.jsonl")

    search_reps = get_budget_reps(cfg, budget, mode='search')
    genome_payload = genome_to_dict(genome)
    tasks: List[Dict[str, Any]] = []
    for rep in range(search_reps):
        for bb in cfg["model_names"]:
            key = (genome.genome_id, budget_name, bb, rep)
            if key in done_keys:
                continue
            tasks.append({
                "cfg": cfg,
                "genome": genome_payload,
                "backbone": bb,
                "budget": budget,
                "rep": int(rep),
                "root_dir": root_dir,
                "tag": tag,
                "gen_idx": int(gen_idx),
                "active_seed": int(active_seed),
                "mode": "search",
            })

    def record_completed_row(row: Dict[str, Any]) -> None:
        row.update({"gen": int(gen_idx), "tag": tag})
        key = (str(row["genome_id"]), str(row["budget"]), str(row["backbone"]), int(row["rep"]))
        if key in done_keys:
            return
        write_run_row_everywhere(root_dir, tag, row)
        done_keys.add(key)

    if task_executor is None:
        with SearchTaskExecutor(cfg) as executor:
            executor.run_tasks(tasks, on_result=record_completed_row)
    else:
        task_executor.run_tasks(tasks, on_result=record_completed_row)

    all_rows = [
        r for r in dedupe_run_rows(read_jsonl(exp_runs_path))
        if r.get("genome_id") == genome.genome_id and r.get("budget") == budget_name
    ]

    summary = summarize_budget_runs(cfg, all_rows, budget_name)
    objectives = objectives_from_summary(summary)
    genome.objectives = objectives
    genome.last_budget = budget_name
    best_run = select_best_run(all_rows)

    summary_row = build_summary_row(
        gen_idx=gen_idx,
        tag=tag,
        genome_id=genome.genome_id,
        budget_name=budget_name,
        summary=summary,
        objectives=objectives,
        best_run=best_run,
    )
    write_genome_summary_everywhere(root_dir, tag, summary_row)

    return {
        "budget": budget,
        "all_rows": all_rows,
        "summary": summary,
        "objectives": objectives,
        "best_run": best_run,
    }

def _is_fully_evaluated(cfg: Dict[str, Any], genome: Genome, done_keys: set) -> bool:
    budget = cfg["budgets"][-1]
    budget_name = budget["name"]
    budget_reps = get_budget_reps(cfg, budget, mode='search')

    return all(
        (genome.genome_id, budget_name, backbone, rep) in done_keys
        for backbone in cfg["model_names"]
        for rep in range(budget_reps)
    )


def run_for_seed(cfg, active_seed: int) -> None:
    tag = "ga_no_sh"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    active_seed = ctx["active_seed"]
    root_dir = ctx["root_dir"]
    exp_dir = ctx["paths"]["exp_dir"]

    resume = bool(os.environ.get("RESUME", "1") == "1")
    rng = random.Random(int(active_seed) + 5050)

    start_gen, pop = load_population_checkpoint(root_dir, tag, rng) if resume else (0, [])

    if not pop:
        pop = [sample_genome(cfg, rng) for _ in range(int(cfg["pop_size"]))]
        start_gen = 0

    task_executor_holder: Dict[str, SearchTaskExecutor | None] = {"executor": None}

    def per_generation(gen: int, population: List[Genome]) -> List[Genome]:
        if gen > 1:
            assign_rank_and_crowd(cfg, population)
            parents = nsga2_select(cfg, population, k=max(2, len(population) // 2))
            elite = nsga2_select(cfg, population, k=max(1, int(round(len(population) * 0.15))))
            children = make_offspring(
                cfg,
                parents if parents else population,
                n=max(1, len(population) - len(elite)),
                rng=rng,
                allow_crossover=True,
                allow_mutation=True,
                deduplicate=True,
            )
            combined = [deepcopy(g) for g in elite] + children
        else:
            combined = population[:]

        print(f"[GEN {gen}] Evaluating {len(combined)} genomes at final budget (no SH)")

        done_keys = load_done_keys_from_path(os.path.join(exp_dir, "runs.jsonl"))

        for idx, genome in enumerate(combined, start=1):
            print(f"[GEN {gen}] FULL BUDGET {idx}/{len(combined)} | genome={genome.genome_id}")

            if _is_fully_evaluated(cfg, genome, done_keys):
                if has_valid_objectives(genome):
                    print(f"[GEN {gen}] SKIP already fully evaluated | genome={genome.genome_id}")
                    continue
                print(f"[GEN {gen}] REUSE already fully evaluated summary | genome={genome.genome_id}")
                _evaluate_genome_full_budget_everywhere(
                    cfg,
                    genome,
                    tag=tag,
                    root_dir=root_dir,
                    done_keys=done_keys,
                    gen_idx=gen,
                    active_seed=int(active_seed),
                    task_executor=task_executor_holder["executor"],
                )
                continue

            _evaluate_genome_full_budget_everywhere(
                cfg,
                genome,
                tag=tag,
                root_dir=root_dir,
                done_keys=done_keys,
                gen_idx=gen,
                active_seed=int(active_seed),
                task_executor=task_executor_holder["executor"],
            )

        evaluated = [g for g in combined if has_valid_objectives(g)] or combined
        if len(evaluated) < len(combined):
            print(f"[WARN] gen={gen}: {len(combined) - len(evaluated)} genome(s) have sentinel objectives.")

        assign_rank_and_crowd(cfg, evaluated)
        next_pop = nsga2_select(cfg, evaluated, k=min(int(cfg["pop_size"]), len(evaluated)))
        pf = pareto_front(next_pop, cfg=cfg)
        write_pareto_snapshot(exp_dir, f"pareto_gen_{gen}.json", pf)
        log_top_population(pf, title="--- Pareto (top few) ---")
        return next_pop

    def save_checkpoint(gen: int, population: List[Genome]) -> None:
        save_population_checkpoint(root_dir, tag, gen_completed=gen, pop=population, rng=rng)

    def final_population(population: List[Genome]) -> List[Genome]:
        return pareto_front(population, cfg=cfg)

    with SearchTaskExecutor(cfg) as task_executor:
        task_executor_holder["executor"] = task_executor
        run_generational_search(
            cfg=cfg,
            tag=tag,
            root_dir=root_dir,
            exp_dir=exp_dir,
            start_gen=start_gen,
            initial_population=pop,
            per_generation=per_generation,
            save_checkpoint=save_checkpoint,
            final_population=final_population,
            completion_message=f"DONE. GA no-SH baseline in {exp_dir}",
        )


def main() -> None:
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
