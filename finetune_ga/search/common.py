from __future__ import annotations

import ast
import json
import os
import random
from typing import Any, Callable, Dict, List, Tuple

from finetune_ga.core.genome import Genome, genome_from_dict, genome_to_dict
from finetune_ga.infra.experiment_records import upsert_summary_row
from finetune_ga.infra.io_utils import (
    append_jsonl,
    atomic_write_json,
    load_experiment_state,
    prepare_output_paths,
    safe_int,
    save_experiment_state,
)


def build_budget_level_map(cfg: Dict[str, Any]) -> Dict[str, int]:
    return {str(b["name"]): idx for idx, b in enumerate(cfg["budgets"])}


def budget_level(budget_levels: Dict[str, int], budget_name: str | None) -> int:
    if budget_name is None:
        return -1
    return int(budget_levels.get(str(budget_name), -1))


def _normalize_minimal_result_fields(row: Dict[str, Any], tag: str) -> Dict[str, Any]:
    out = dict(row)
    out.setdefault('tag', tag)
    out.setdefault('mode', 'search')
    out.setdefault('status', 'ok')
    out.setdefault('budget', 'NA')
    out.setdefault('genome_id', 'NA')
    out.setdefault('backbone', 'NA')
    out.setdefault('rep', None)
    out.setdefault('active_seed', out.get('seed'))
    out.setdefault('seed', out.get('active_seed'))
    return out


def write_run_row_everywhere(root_dir: str, tag: str, row: Dict[str, Any]) -> None:
    row = _normalize_minimal_result_fields(row, tag)
    paths = prepare_output_paths(root_dir, tag, row["backbone"])
    append_jsonl(os.path.join(paths["exp_dir"], "runs.jsonl"), row)
    append_jsonl(os.path.join(paths["per_model_dir"], "runs.jsonl"), row)
    append_jsonl(os.path.join(paths["shared_dir"], "runs_all.jsonl"), row)


def write_genome_summary_everywhere(root_dir: str, tag: str, summary_row: Dict[str, Any]) -> None:
    summary_row = _normalize_minimal_result_fields(summary_row, tag)
    summary_row['mode'] = summary_row.get('mode') or 'summary'
    paths = prepare_output_paths(root_dir, tag, "all_models")
    upsert_summary_row(os.path.join(paths["exp_dir"], "genome_summary.jsonl"), summary_row)
    upsert_summary_row(os.path.join(paths["per_model_dir"], "genome_summary.jsonl"), summary_row)
    upsert_summary_row(os.path.join(paths["shared_dir"], "genome_summary_all.jsonl"), summary_row)


def save_population_checkpoint(root_dir: str, tag: str, gen_completed: int, pop: List[Genome], rng: random.Random) -> None:
    save_experiment_state(
        root_dir,
        tag,
        {
            "gen_completed": int(gen_completed),
            "population": [genome_to_dict(g) for g in pop],
            "rng_state": repr(rng.getstate()),
        },
    )


def load_population_checkpoint(root_dir: str, tag: str, rng: random.Random) -> Tuple[int, List[Genome]]:
    state = load_experiment_state(root_dir, tag)
    if not state:
        return 0, []

    if state.get("rng_state"):
        try:
            rng.setstate(ast.literal_eval(state["rng_state"]))
        except (ValueError, SyntaxError, TypeError) as exc:
            print(f"[WARN] Could not restore RNG state for {tag}: {type(exc).__name__}: {exc}")

    raw_population = state.get("population", [])
    pop: List[Genome] = []

    if isinstance(raw_population, list):
        for idx, item in enumerate(raw_population):
            try:
                if not isinstance(item, dict):
                    raise TypeError(f"population[{idx}] is {type(item).__name__}, expected dict")
                pop.append(genome_from_dict(item))
            except (KeyError, TypeError, ValueError) as exc:
                print(
                    f"[WARN] Skipping invalid checkpoint genome for {tag} "
                    f"at index {idx}: {type(exc).__name__}: {exc}"
                )
    else:
        if raw_population not in (None, []):
            print(
                f"[WARN] Ignoring invalid checkpoint population for {tag}: "
                f"expected list, got {type(raw_population).__name__}"
            )

    gen_completed = max(0, safe_int(state.get("gen_completed", 0), 0))
    return gen_completed, pop


def write_pareto_snapshot(exp_dir: str, filename: str, pf: List[Genome]) -> None:
    objective_names = ['mean_val_auc_loss_obj', 'selection_time_s_obj', 'trainable_params_m_obj']
    atomic_write_json(
        os.path.join(exp_dir, filename),
        [
            {
                'genome_id': g.genome_id,
                'objective_names': objective_names,
                'objectives': list(g.objectives),
                'mean_val_auc_loss_obj': float(g.objectives[0]),
                'mean_val_auc': float(1.0 - g.objectives[0]),
                'selection_time_s_obj': float(g.objectives[1]),
                'trainable_params_m_obj': float(g.objectives[2]),
                'last_budget': g.last_budget,
            }
            for g in pf
        ],
        validate_finite=True,
    )


def log_top_population(population: List[Genome], *, title: str, limit: int = 6, value_label: str = "mean_val_auc") -> None:
    print(title)
    for i, genome in enumerate(population[:limit], start=1):
        print(
            f"{i:02d}) {value_label}={1.0 - genome.objectives[0]:.4f} "
            f"time={genome.objectives[1] / 60:.1f}min "
            f"trainable={genome.objectives[2]:.2f}M "
            f"last={genome.last_budget}"
        )


def run_generational_search(
    *,
    cfg: Dict[str, Any],
    tag: str,
    root_dir: str,
    exp_dir: str,
    start_gen: int,
    initial_population: List[Genome],
    per_generation: Callable[[int, List[Genome]], List[Genome]],
    save_checkpoint: Callable[[int, List[Genome]], None],
    final_population: Callable[[List[Genome]], List[Genome]],
    completion_message: str,
) -> List[Genome]:
    pop = list(initial_population)
    for gen in range(start_gen + 1, int(cfg["generations"]) + 1):
        print(f"\n================= {tag.upper()} GENERATION {gen}/{cfg['generations']} =================")
        pop = per_generation(gen, pop)
        save_checkpoint(gen, pop)
    final_pop = final_population(pop)
    write_pareto_snapshot(exp_dir, "pareto_final.json", final_pop)
    print(completion_message)
    return final_pop
