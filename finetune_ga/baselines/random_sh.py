from __future__ import annotations

import ast
import os
import random
from typing import List

from finetune_ga.core.genome import Genome, sample_genome
from finetune_ga.infra.io_utils import save_experiment_state, load_experiment_state, safe_int
from finetune_ga.search.nsga2 import evaluate_population_with_sh
from finetune_ga.search.task_executor import SearchTaskExecutor
from finetune_ga.baselines.common import initialize_baseline_run, run_seeded_main


def run_for_seed(cfg, active_seed: int):
    tag = "random_sh"
    ctx = initialize_baseline_run(cfg, tag, active_seed, ensure_dataset_counts=True)
    active_seed = ctx["active_seed"]
    root_dir = ctx["root_dir"]
    resume = bool(os.environ.get("RESUME", "1") == "1")
    rng = random.Random(int(active_seed) + 123)
    state = load_experiment_state(root_dir, tag) if resume else {}
    if state.get("rng_state"):
        try:
            rng.setstate(ast.literal_eval(state["rng_state"]))
        except (ValueError, SyntaxError, TypeError) as exc:
            print(f"[WARN] Could not restore RNG state for {tag}: {type(exc).__name__}: {exc}")
    start_gen = safe_int(state.get("gen_completed", 0), 0)
    pop_size = int(cfg["pop_size"])
    gens = int(cfg["generations"])

    with SearchTaskExecutor(cfg) as task_executor:
        for gen in range(start_gen + 1, gens + 1):
            print(f"\n================= RANDOM+SH GENERATION {gen}/{gens} | seed={active_seed} =================")
            candidates: List[Genome] = [sample_genome(cfg, rng) for _ in range(pop_size)]
            evaluate_population_with_sh(
                cfg, gen, candidates, root_dir,
                tag=tag, resume=resume, active_seed=int(active_seed),
                task_executor=task_executor,
            )
            save_experiment_state(root_dir, tag, {"gen_completed": gen, "rng_state": repr(rng.getstate())})

    print(f"\nDONE. Random+SH results in: {root_dir}/{tag}")


def main():
    run_seeded_main(run_for_seed)


if __name__ == "__main__":
    main()
