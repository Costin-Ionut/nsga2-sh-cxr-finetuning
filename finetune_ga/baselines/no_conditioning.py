from finetune_ga.infra.experiment_config import load_config
from finetune_ga.infra.run_utils import get_search_seeds
from finetune_ga.infra.io_utils import clear_runs_cache
from finetune_ga.search.nsga2 import run_nsga2_experiment


def main():
    cfg = load_config()
    seeds = get_search_seeds(cfg)
    for seed in seeds:
        clear_runs_cache()
        run_nsga2_experiment(cfg, active_seed=int(seed), tag="nsga2_no_conditioning", no_conditioning=True)


if __name__ == "__main__":
    main()
