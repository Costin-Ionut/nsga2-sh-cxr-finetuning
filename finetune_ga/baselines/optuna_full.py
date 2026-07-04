from __future__ import annotations

from finetune_ga.infra.experiment_config import load_config
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main
from finetune_ga.baselines.optuna_bohb import suggest_genome
from finetune_ga.baselines.optuna_common import optuna, build_optuna_study, optimize_single_budget
from finetune_ga.search.task_executor import SearchTaskExecutor


def run_for_seed(cfg, active_seed: int) -> None:
    tag = "optuna_full_budget"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    if optuna is None:
        print("Optuna is not installed. Run: pip install optuna")
        return
    study = build_optuna_study(
        tag,
        int(ctx["active_seed"]),
        ctx["paths"]["exp_dir"],
        sampler_seed=int(ctx["active_seed"]) + 1,
        pruner=optuna.pruners.NopPruner(),
    )
    with SearchTaskExecutor(cfg) as task_executor:
        optimize_single_budget(
            cfg,
            study=study,
            tag=tag,
            root_dir=ctx["root_dir"],
            run_dir=ctx["paths"]["exp_dir"],
            suggest_genome=suggest_genome,
            active_seed=int(active_seed),
            task_executor=task_executor,
        )


def main() -> None:
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
