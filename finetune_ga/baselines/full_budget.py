from __future__ import annotations

import os
from typing import Any, Dict, Optional

from finetune_ga.infra.experiment_records import build_summary_row, dedupe_run_rows, load_done_keys_from_path, select_best_run, upsert_summary_row
from finetune_ga.infra.io_utils import append_jsonl, read_jsonl, save_experiment_state, load_experiment_state, safe_int
from finetune_ga.core.genome import genome_to_dict
from finetune_ga.core.metrics import summarize_budget_runs, objectives_from_summary
from finetune_ga.infra.run_utils import get_budget_reps
from finetune_ga.search.task_executor import SearchTaskExecutor


def _sort_run_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("genome_id", "")),
            str(r.get("budget", "")),
            str(r.get("backbone", "")),
            int(r.get("rep", 0)),
        ),
    )


def _run_tasks_with_optional_executor(
    cfg: Dict[str, Any],
    tasks: list[Dict[str, Any]],
    task_executor: SearchTaskExecutor | None,
    on_result=None,
) -> list[Dict[str, Any]]:
    if task_executor is not None:
        return task_executor.run_tasks(tasks, on_result=on_result)
    with SearchTaskExecutor(cfg) as executor:
        return executor.run_tasks(tasks, on_result=on_result)


def evaluate_genome_at_final_budget(
    cfg: Dict[str, Any],
    genome,
    *,
    tag: str,
    root_dir: str,
    runs_path: str,
    summary_path: str,
    done_keys: set,
    gen_idx: int = 0,
    extra_summary: Optional[Dict[str, Any]] = None,
    extra_run_fields: Optional[Dict[str, Any]] = None,
    active_seed: int = 0,
    task_executor: SearchTaskExecutor | None = None,
) -> Dict[str, Any]:
    budget = cfg["budgets"][-1]
    budget_name = str(budget["name"])
    search_reps = get_budget_reps(cfg, budget, mode='search')
    genome_payload = genome_to_dict(genome)

    tasks: list[Dict[str, Any]] = []
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
        if extra_run_fields:
            row.update(extra_run_fields)
        key = (str(row["genome_id"]), str(row["budget"]), str(row["backbone"]), int(row["rep"]))
        if key in done_keys:
            return
        append_jsonl(runs_path, row)
        done_keys.add(key)

    _run_tasks_with_optional_executor(cfg, tasks, task_executor, on_result=record_completed_row)

    all_rows = [
        r for r in dedupe_run_rows(read_jsonl(runs_path))
        if r.get("genome_id") == genome.genome_id and r.get("budget") == budget_name
    ]
    summary = summarize_budget_runs(cfg, all_rows, budget_name)
    objectives = objectives_from_summary(summary)
    genome.objectives = objectives
    genome.last_budget = budget_name
    best_run = select_best_run(all_rows)
    upsert_summary_row(summary_path, build_summary_row(
        gen_idx=gen_idx,
        tag=tag,
        genome_id=genome.genome_id,
        budget_name=budget_name,
        summary=summary,
        objectives=objectives,
        best_run=best_run,
        extra=extra_summary,
    ))
    return {
        "budget": budget,
        "all_rows": all_rows,
        "summary": summary,
        "objectives": objectives,
        "best_run": best_run,
    }


def load_completed_index_state(root_dir: str, tag: str, *, resume: bool) -> int:
    state = load_experiment_state(root_dir, tag) if resume else {}
    return safe_int(state.get("completed_index", 0), 0)


def save_completed_index_state(root_dir: str, tag: str, completed_index: int, **extra: Any) -> None:
    payload = {"completed_index": int(completed_index)}
    payload.update(extra)
    save_experiment_state(root_dir, tag, payload)
