from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from finetune_ga.infra.repro_manifest import write_run_manifest

SCRIPT_STEPS = [
    ["-m", "finetune_ga.search.nsga2"],

    # Main paper baselines
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_feature_extractor"],
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_standard_finetune"],
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_manual_tuned"],
    ["-m", "finetune_ga.baselines.random_sh"],

    # Appendix
    # ["-m", "finetune_ga.baselines.no_conditioning"],

    # Appendix
    # ["-m", "finetune_ga.baselines.random_full_budget"],

    # Appendix
    # ["-m", "finetune_ga.baselines.ga_no_sh"],

    
    # ["-m", "finetune_ga.baselines.optuna_bohb"],
    # Appendix
    # ["-m", "finetune_ga.baselines.optuna_full"],

    # Main paper
    # ["-m", "finetune_ga.baselines.single_objective_ga"],

    # Main
    # ["-m", "finetune_ga.baselines.fixed_configs"],

    # Main paper ablations
    ["-m", "experiments.run_ablations", "--ablation", "no_crossover"],
    ["-m", "experiments.run_ablations", "--ablation", "no_mutation"],
    ["-m", "experiments.run_ablations", "--ablation", "no_elitism"],

    # Appendix
    # ["-m", "experiments.run_ablations", "--ablation", "no_dedup"],

    ["-m", "experiments.evaluate_test"],
    ["-m", "experiments.analyze"],
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("RESUME", "1")
    root = Path(__file__).resolve().parents[1]
    manifest_path = write_run_manifest(env.get("CONFIG_PATH"))
    print(f"[RUN_PAPER_SUITE] Wrote run manifest: {manifest_path}")
    for step_parts in SCRIPT_STEPS:
        cmd = [sys.executable, *step_parts]
        label = " ".join(step_parts)
        print(f"\n{'=' * 20} RUNNING {label} {'=' * 20}")
        code = subprocess.call(cmd, env=env, cwd=str(root))
        if code != 0:
            raise SystemExit(f"Step failed: {label} (exit code {code})")


if __name__ == "__main__":
    main()