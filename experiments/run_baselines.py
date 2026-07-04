from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASELINE_STEPS = [
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_feature_extractor"],
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_standard_finetune"],
    ["-m", "finetune_ga.baselines.literature_baselines", "--tag", "baseline_manual_tuned"],
    ["-m", "finetune_ga.baselines.random_sh"],
    ["-m", "finetune_ga.baselines.no_conditioning"],
    ["-m", "finetune_ga.baselines.random_full_budget"],
    ["-m", "finetune_ga.baselines.ga_no_sh"],
    ["-m", "finetune_ga.baselines.optuna_bohb"],
    ["-m", "finetune_ga.baselines.optuna_full"],
    ["-m", "finetune_ga.baselines.single_objective_ga"],
    ["-m", "finetune_ga.baselines.fixed_configs"],
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("RESUME", "1")
    root = Path(__file__).resolve().parents[1]
    for step_parts in BASELINE_STEPS:
        cmd = [sys.executable, *step_parts]
        label = " ".join(step_parts)
        print(f"\n{'=' * 20} RUNNING {label} {'=' * 20}")
        code = subprocess.call(cmd, env=env, cwd=str(root))
        if code != 0:
            raise SystemExit(f"Step failed: {label} (exit code {code})")


if __name__ == "__main__":
    main()
