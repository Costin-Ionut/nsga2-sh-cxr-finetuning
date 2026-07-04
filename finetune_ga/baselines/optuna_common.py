from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

try:
    import optuna
except (ImportError, ModuleNotFoundError) as exc:
    print(f"[WARN] Optuna import failed: {type(exc).__name__}: {exc}")
    optuna = None

from finetune_ga.infra.experiment_records import build_summary_row, dedupe_run_rows, load_done_keys, select_best_run, upsert_summary_row
from finetune_ga.infra.io_utils import append_jsonl, read_jsonl
from finetune_ga.core.genome import genome_to_dict
from finetune_ga.core.metrics import summarize_budget_runs, objectives_from_summary
from finetune_ga.core.training import train_one_backbone_search
from finetune_ga.infra.run_utils import get_budget_reps
from finetune_ga.search.task_executor import SearchTaskExecutor


def build_optuna_study(tag: str, active_seed: int, run_dir: str, *, sampler_seed: int, pruner) -> Any:
    if optuna is None:
        return None
    storage_url = f"sqlite:///{os.path.join(run_dir, 'optuna_study.db')}"
    study_name = f"{tag}_seed_{active_seed}"
    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


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


def _run_trial_tasks_serial(
    cfg: Dict[str, Any],
    tasks: list[Dict[str, Any]],
    *,
    trial,
    tag: str,
    trial_extra_fields: Optional[Callable[[Any], Dict[str, Any]]],
    runs_path: str,
    done_keys: set,
    prune_callback: Optional[Callable[[Any, list], None]],
) -> list[Dict[str, Any]]:
    collected: list[Dict[str, Any]] = []
    for task in tasks:
        from finetune_ga.core.genome import genome_from_dict
        genome = genome_from_dict(task["genome"])
        row = train_one_backbone_search(
            cfg, genome, task["backbone"], task["budget"], int(task["rep"]),
            root_dir=task["root_dir"], tag=tag, gen_idx=int(task["gen_idx"]),
            active_seed=int(task["active_seed"]),
        )
        row.update({"gen": int(task["gen_idx"]), "tag": tag, "trial_number": int(trial.number), "worker_gpu_id": -1})
        extra = trial_extra_fields(trial) if trial_extra_fields else None
        if extra:
            row.update(extra)
        key = (str(row["genome_id"]), str(row["budget"]), str(row["backbone"]), int(row["rep"]))
        if key not in done_keys:
            append_jsonl(runs_path, row)
            collected.append(row)
            done_keys.add(key)
        if prune_callback:
            prune_callback(trial, collected)
    return collected


def optimize_single_budget(
    cfg: Dict[str, Any],
    *,
    study,
    tag: str,
    root_dir: str,
    run_dir: str,
    suggest_genome: Callable[[Dict[str, Any], Any], Any],
    prune_callback: Optional[Callable[[Any, list], None]] = None,
    trial_extra_fields: Optional[Callable[[Any], Dict[str, Any]]] = None,
    active_seed: int,
    task_executor: SearchTaskExecutor | None = None,
) -> None:
    if optuna is None:
        print("Optuna is not installed. Run: pip install optuna")
        return

    runs_path = os.path.join(run_dir, "runs.jsonl")
    summary_path = os.path.join(run_dir, "genome_summary.jsonl")
    budget = cfg["budgets"][-1]
    budget_name = str(budget["name"])
    done_keys = load_done_keys(read_jsonl(runs_path))
    n_trials_total = int(cfg["pop_size"]) * int(cfg["generations"])
    n_trials_remaining = max(0, n_trials_total - len(study.trials))
    if n_trials_remaining <= 0:
        print(f"Optuna study already complete for seed {int(active_seed)}.")
        return

    allow_parallel_pruning = bool(
        str(os.environ.get("OPTUNA_PARALLEL_PRUNING", cfg.get("optuna_parallel_pruning", "0"))).lower()
        in {"1", "true", "yes", "on"}
    )

    def objective(trial):
        genome = suggest_genome(cfg, trial)
        search_reps = get_budget_reps(cfg, budget, mode="search")
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
                    "gen_idx": 0,
                    "active_seed": int(active_seed),
                    "mode": "search",
                })

        collected: list[Dict[str, Any]] = []
        # Keep pruned BOHB/Hyperband serial by default so pruning granularity and
        # published methodology stay unchanged. Set OPTUNA_PARALLEL_PRUNING=1 only
        # if batch-level pruning is desired and documented.
        if prune_callback and not allow_parallel_pruning:
            collected = _run_trial_tasks_serial(
                cfg,
                tasks,
                trial=trial,
                tag=tag,
                trial_extra_fields=trial_extra_fields,
                runs_path=runs_path,
                done_keys=done_keys,
                prune_callback=prune_callback,
            )
        else:
            def record_completed_row(row: Dict[str, Any]) -> None:
                row.update({"gen": 0, "tag": tag, "trial_number": int(trial.number)})
                extra = trial_extra_fields(trial) if trial_extra_fields else None
                if extra:
                    row.update(extra)
                key = (str(row["genome_id"]), str(row["budget"]), str(row["backbone"]), int(row["rep"]))
                if key in done_keys:
                    return
                append_jsonl(runs_path, row)
                collected.append(row)
                done_keys.add(key)

            if task_executor is None:
                with SearchTaskExecutor(cfg) as executor:
                    executor.run_tasks(tasks, on_result=record_completed_row)
            else:
                task_executor.run_tasks(tasks, on_result=record_completed_row)
            if prune_callback and collected:
                prune_callback(trial, collected)

        all_rows = [r for r in dedupe_run_rows(read_jsonl(runs_path)) if r.get("genome_id") == genome.genome_id and r.get("budget") == budget_name]
        summary = summarize_budget_runs(cfg, all_rows, budget_name)
        objectives = objectives_from_summary(summary)
        best_run = select_best_run(all_rows)
        extra = {"trial_number": int(trial.number)}
        if trial_extra_fields:
            maybe = trial_extra_fields(trial)
            if maybe:
                extra.update(maybe)
        upsert_summary_row(summary_path, build_summary_row(
            gen_idx=0,
            tag=tag,
            genome_id=genome.genome_id,
            budget_name=budget_name,
            summary=summary,
            objectives=objectives,
            best_run=best_run,
            extra=extra,
        ))
        return float(summary.get("mean_val_auc", summary.get("auc_mean", summary["robust_min_auc"])))

    study.optimize(objective, n_trials=n_trials_remaining)
    if study.best_trial is not None:
        print("Best trial mean_val_auc:", study.best_value)
    print(f"Results appended to {runs_path} and {summary_path}")
