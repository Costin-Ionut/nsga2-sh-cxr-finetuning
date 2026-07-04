from __future__ import annotations

import json
from pathlib import Path

from finetune_ga.baselines import literature_baselines as lb


CFG = {
    "search_seeds": [5],
    "final_seeds": [5],
    "selection_source_seed": 5,
    "model_names": ["mobilenet", "resnet50"],
    "budgets": [{"name": "B2", "reps": 1}],
}


def test_literature_baseline_genome_factories_create_expected_candidates():
    feature = lb.make_feature_extractor_genomes(CFG)
    standard = lb.make_standard_finetune_genomes(CFG)
    manual = lb.make_manual_tuned_genomes(CFG)

    assert len(feature) == 1
    assert feature[0].n_last_layers == 0
    assert len(standard) == 1
    assert standard[0].n_last_layers == 10
    assert len(manual) == 8
    assert all(genome.genome_id for genome in feature + standard + manual)


def test_run_baseline_for_seed_writes_runs_summary_and_manifest(tmp_path, monkeypatch):
    tag = "baseline_manual_tuned"
    exp_dir = tmp_path / tag
    exp_dir.mkdir(parents=True)

    monkeypatch.setattr(
        lb,
        "initialize_baseline_run",
        lambda cfg, tag, active_seed, ensure_dataset_counts=False: {
            "tag": tag,
            "active_seed": active_seed,
            "root_dir": str(tmp_path),
            "paths": {"exp_dir": str(exp_dir)},
        },
    )
    monkeypatch.setattr(lb, "load_experiment_state", lambda root_dir, tag: {})
    monkeypatch.setattr(lb, "save_completed_index_state", lambda root_dir, tag, idx: None)

    def fake_evaluate_genome_at_final_budget(
        cfg,
        genome,
        *,
        tag,
        root_dir,
        runs_path,
        summary_path,
        done_keys,
        gen_idx=0,
        extra_summary=None,
        extra_run_fields=None,
        active_seed=0,
        task_executor=None,
        **kwargs,
    ):
        run_row = {
            "status": "ok",
            "genome_id": genome.genome_id,
            "budget": cfg["budgets"][-1]["name"],
            "backbone": cfg["model_names"][0],
            "rep": 0,
            "seed": cfg["search_seeds"][0],
            "best_val_auc": 0.88,
            "best_val_pr_auc": 0.85,
            "best_val_accuracy": 0.82,
            "loss_at_best_auc": 0.12,
            "time_s": 42.0,
            "trainable_params_m": 1.7,
            "target_size": 224,
        }
        if extra_run_fields:
            run_row.update(extra_run_fields)
        with open(runs_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(run_row) + "\n")
        summary_row = {
            "genome_id": genome.genome_id,
            "budget": cfg["budgets"][-1]["name"],
            "baseline_description": extra_summary["baseline_description"],
            "baseline_index": extra_summary["baseline_index"],
            "objectives": {"robust_min_auc_loss_obj": 0.12, "selection_time_s_obj": 42.0, "trainable_params_m_obj": 1.7},
        }
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_row) + "\n")
        done_keys.add((genome.genome_id, cfg["budgets"][-1]["name"], cfg["model_names"][0], 0))
        return {"summary": summary_row}

    monkeypatch.setattr(lb, "evaluate_genome_at_final_budget", fake_evaluate_genome_at_final_budget)

    lb.run_baseline_for_seed(CFG, 5, tag=tag)

    runs_path = exp_dir / "runs.jsonl"
    summary_path = exp_dir / "genome_summary.jsonl"
    manifest_path = exp_dir / f"{tag}_manifest.json"

    runs_rows = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    summary_rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text())

    assert len(runs_rows) == 8
    assert all(row["baseline_tag"] == tag for row in runs_rows)
    assert len(summary_rows) == 8
    assert {row["baseline_index"] for row in summary_rows} == set(range(1, 9))
    assert manifest["tag"] == tag
    assert manifest["num_genomes"] == 8
    assert len(manifest["genomes"]) == 8
