from __future__ import annotations

import json
from pathlib import Path

from finetune_ga.core.genome import Genome, assign_genome_id
from finetune_ga.search import nsga2


def _make_genome(model_names: list[str], *, lr: float, n_last_layers: int) -> Genome:
    genome = Genome(
        base_lr1=lr,
        base_lr2=lr / 10.0,
        n_last_layers=n_last_layers,
        dense_units=128,
        dropout=0.25,
        l2_weight=1e-5,
        batch_size=16,
        aug_rotation=0.05,
        aug_zoom=0.05,
        aug_contrast=0.05,
        lr1_mul={name: 1.0 for name in model_names},
        lr2_mul={name: 1.0 for name in model_names},
    )
    assign_genome_id(genome, model_names)
    return genome


def test_run_nsga2_experiment_smoke_writes_runs_summaries_and_pareto(tmp_path, monkeypatch):
    cfg = {
        "out_dir": str(tmp_path),
        "search_seeds": [7],
        "final_seeds": [7],
        "selection_source_seed": 7,
        "model_names": ["backbone_a", "backbone_b"],
        "budgets": [{"name": "B1", "reps": 1}],
        "pop_size": 2,
        "generations": 1,
        "promote_frac": 0.5,
        "complexity_weights": {"params_m": 0.55, "time_h": 0.35, "size_ratio": 0.10},
    }

    genomes = [
        _make_genome(cfg["model_names"], lr=1e-4, n_last_layers=5),
        _make_genome(cfg["model_names"], lr=2e-4, n_last_layers=10),
    ]
    genome_iter = iter(genomes)

    monkeypatch.setattr(nsga2, "seed_everything", lambda seed, **kwargs: None)
    monkeypatch.setattr(nsga2, "get_seeded_root_dir", lambda base_out_dir, active_seed, seed_list: str(tmp_path / "seed_7"))
    monkeypatch.setattr(nsga2, "get_env_info", lambda **kwargs: {"python": "test"})
    monkeypatch.setattr(nsga2, "sample_genome", lambda cfg, rng: next(genome_iter))
    monkeypatch.setattr("finetune_ga.infra.config.get_dataset_counts", lambda refresh=True: {"ok": True})

    def fake_train_one_backbone(cfg, genome, backbone, budget, rep, *, root_dir=None, tag=None, gen_idx=None, active_seed=0, mode='search'):
        auc_base = 0.80 if genome.n_last_layers == 5 else 0.86
        auc = auc_base + (0.01 if backbone == "backbone_b" else 0.0)
        return {
            "status": "ok",
            "genome_id": genome.genome_id,
            "budget": budget["name"],
            "backbone": backbone,
            "rep": rep,
            "seed": active_seed,
            "best_val_auc": auc,
            "best_val_pr_auc": auc - 0.02,
            "best_val_accuracy": auc - 0.03,
            "loss_at_best_auc": 1.0 - auc,
            "time_s": 30.0 + float(genome.n_last_layers),
            "trainable_params_m": 1.5 + float(genome.n_last_layers) / 10.0,
            "target_size": 224,
            "best_model_path": str(Path(root_dir or tmp_path) / "model.keras"),
            "best_weights_path": str(Path(root_dir or tmp_path) / "weights.weights.h5"),
            "history_plot_path": str(Path(root_dir or tmp_path) / "history.png"),
        }

    monkeypatch.setattr(nsga2, "train_one_backbone", fake_train_one_backbone)

    class FakeSearchTaskExecutor:
        def __init__(self, cfg):
            self.cfg = cfg

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run_tasks(self, tasks, on_result=None):
            from finetune_ga.core.genome import genome_from_dict

            rows = []
            for task in tasks:
                genome = genome_from_dict(task["genome"])
                row = fake_train_one_backbone(
                    task["cfg"], genome, task["backbone"], task["budget"], int(task["rep"]),
                    root_dir=task["root_dir"], tag=task["tag"], gen_idx=int(task["gen_idx"]),
                    active_seed=int(task["active_seed"]), mode=task.get("mode", "search"),
                )
                row.update({"gen": int(task["gen_idx"]), "tag": task["tag"], "worker_gpu_id": -1})
                rows.append(row)
                if on_result is not None:
                    on_result(row)
            return rows

    monkeypatch.setattr(nsga2, "SearchTaskExecutor", FakeSearchTaskExecutor)

    nsga2.run_nsga2_experiment(cfg, active_seed=7, tag="nsga2_smoke")

    exp_dir = tmp_path / "seed_7" / "nsga2_smoke"
    runs_path = exp_dir / "runs.jsonl"
    summary_path = exp_dir / "genome_summary.jsonl"
    pareto_path = exp_dir / "pareto_final.json"

    assert runs_path.exists()
    assert summary_path.exists()
    assert pareto_path.exists()

    run_rows = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    summary_rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    pareto_rows = json.loads(pareto_path.read_text())

    assert len(run_rows) == 4
    assert {row["status"] for row in run_rows} == {"ok"}
    assert len(summary_rows) == 2
    assert {row["budget"] for row in summary_rows} == {"B1"}
    assert all("objectives" in row for row in summary_rows)
    assert pareto_rows, "pareto front should not be empty"
