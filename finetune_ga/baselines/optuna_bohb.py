from __future__ import annotations

from typing import Any, Dict
import os

from finetune_ga.core.genome import Genome, assign_genome_id
from finetune_ga.infra.experiment_config import load_config
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.baselines.optuna_common import optuna, build_optuna_study, optimize_single_budget
from finetune_ga.search.task_executor import SearchTaskExecutor, force_pin_current_process_to_first_gpu


def suggest_genome(cfg: Dict[str, Any], trial) -> Genome:
    ss = cfg["search_space"]
    model_names = cfg["model_names"]
    lo_mul, hi_mul = ss["lr_mul_bounds"]
    genome = Genome(
        base_lr1=float(trial.suggest_categorical("base_lr1", ss["base_lr1"])),
        base_lr2=float(trial.suggest_categorical("base_lr2", ss["base_lr2"])),
        n_last_layers=int(trial.suggest_categorical("n_last_layers", ss["n_last_layers"])),
        dense_units=int(trial.suggest_categorical("dense_units", ss["dense_units"])),
        dropout=float(trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1])),
        l2_weight=float(trial.suggest_categorical("l2_weight", ss["l2_weight"])),
        batch_size=int(trial.suggest_categorical("batch_size", ss["batch_size"])),
        aug_rotation=float(trial.suggest_float("aug_rotation", ss["aug_rotation"][0], ss["aug_rotation"][1])),
        aug_zoom=float(trial.suggest_float("aug_zoom", ss["aug_zoom"][0], ss["aug_zoom"][1])),
        aug_contrast=float(trial.suggest_float("aug_contrast", ss["aug_contrast"][0], ss["aug_contrast"][1])),
        lr1_mul={b: float(trial.suggest_float(f"lr1_mul_{b}", lo_mul, hi_mul, log=True)) for b in model_names},
        lr2_mul={b: float(trial.suggest_float(f"lr2_mul_{b}", lo_mul, hi_mul, log=True)) for b in model_names},
    )
    assign_genome_id(genome, model_names)
    return genome


def _prune_callback(trial, collected) -> None:
    values = []
    for r in collected:
        raw = r.get("best_val_auc")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue

    if not values:
        return

    seen_max = max(values)
    trial.report(seen_max, step=len(collected))
    if trial.should_prune():
        raise optuna.TrialPruned()


def run_for_seed(cfg, active_seed: int) -> None:
    tag = "optuna_bohb_style"
    allow_parallel_pruning = str(
        os.environ.get("OPTUNA_PARALLEL_PRUNING", cfg.get("optuna_parallel_pruning", "0"))
    ).lower() in {"1", "true", "yes", "on"}
    if not allow_parallel_pruning:
        # BOHB/Hyperband pruning is intentionally serial by default. Pin before
        # initialize_baseline_run() seeds TensorFlow, even when a global
        # SEARCH_PARALLEL_WORKERS=2 is set for other baselines.
        force_pin_current_process_to_first_gpu()

    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    if optuna is None:
        print("Optuna is not installed. Run: pip install optuna")
        return
    budget = cfg["budgets"][-1]
    study = build_optuna_study(
        tag,
        int(ctx["active_seed"]),
        ctx["paths"]["exp_dir"],
        sampler_seed=int(ctx["active_seed"]),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, int(budget["e1"])),
            reduction_factor=3,
        ),
    )
    if allow_parallel_pruning:
        # Optional batch-level pruning mode: faster, but changes pruning granularity.
        with SearchTaskExecutor(cfg) as task_executor:
            optimize_single_budget(
                cfg,
                study=study,
                tag=tag,
                root_dir=ctx["root_dir"],
                run_dir=ctx["paths"]["exp_dir"],
                suggest_genome=suggest_genome,
                prune_callback=_prune_callback,
                active_seed=int(active_seed),
                task_executor=task_executor,
            )
    else:
        # Default: preserve BOHB/Hyperband pruning semantics and avoid creating an
        # unused worker pool while optuna_common executes pruned trials serially.
        optimize_single_budget(
            cfg,
            study=study,
            tag=tag,
            root_dir=ctx["root_dir"],
            run_dir=ctx["paths"]["exp_dir"],
            suggest_genome=suggest_genome,
            prune_callback=_prune_callback,
            active_seed=int(active_seed),
            task_executor=None,
        )


def main() -> None:
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
