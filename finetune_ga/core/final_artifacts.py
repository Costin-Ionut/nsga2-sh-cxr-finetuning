from __future__ import annotations

import os
import warnings
from typing import Any, Dict


def make_run_artifact_paths(root_dir: str, tag: str, genome_id: str, backbone: str, budget_name: str, rep: int) -> Dict[str, str]:
    run_dir = os.path.join(root_dir, tag, 'runs', f'{genome_id}__{backbone}__{budget_name}__rep{rep}')
    os.makedirs(run_dir, exist_ok=True)
    return {
        'run_dir': run_dir,
        'weights_path': os.path.join(run_dir, 'final.weights.h5'),
        'model_path': os.path.join(run_dir, 'final_model.keras'),
        # Temporary snapshot used only to restore correct best-stage weights
        # before exporting final model (avoids incorrect Stage2 overwrite).
        # Avoids a redundant Stage2 snapshot and keeps I/O minimal.
        'stage1_snapshot_path': os.path.join(run_dir, 'stage1_snapshot.weights.h5'),
    }


def build_final_stage1_callbacks(*, search_callbacks):
    return list(search_callbacks)


def build_final_stage2_callbacks(*, search_callbacks):
    return list(search_callbacks)


def plot_training_history(h1, h2, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError, OSError):
        return

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, title in zip(axes, ('auc', 'loss'), ('AUC', 'Loss')):
            for hist, label in ((h1, 'Stage 1'), (h2, 'Stage 2')):
                if hist is None:
                    continue
                vals = hist.history.get(f'val_{metric}', []) or hist.history.get(metric, [])
                if vals:
                    ax.plot(vals, label=label)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
        plt.tight_layout()
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=80)
        plt.close(fig)
    except (OSError, RuntimeError, ValueError) as exc:
        warnings.warn(f'Could not save history plot to {out_path}: {type(exc).__name__}: {exc}')


def persist_final_artifacts(model, h1, h2, artifacts: Dict[str, str], backbone: str, budget_name: str, rep: int) -> Dict[str, Any]:
    os.makedirs(artifacts['run_dir'], exist_ok=True)

    final_weights_path = artifacts['weights_path']
    final_model_path = artifacts['model_path']
    history_plot_path = os.path.join(artifacts['run_dir'], 'training_history.png')

    checkpoint_saved = False
    try:
        model.save_weights(final_weights_path)
        checkpoint_saved = True
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f'Could not save final weights for {backbone}/{budget_name}/rep{rep}: '
            f'{type(exc).__name__}: {exc}'
        )

    model_export_ok = False
    try:
        model.save(final_model_path)
        model_export_ok = True
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f'Could not save final model for {backbone}/{budget_name}/rep{rep}: '
            f'{type(exc).__name__}: {exc}'
        )

    history_plot_ok = False
    if h1 is not None or h2 is not None:
        try:
            plot_training_history(h1, h2, history_plot_path)
            history_plot_ok = os.path.exists(history_plot_path)
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f'Could not save history plot for {backbone}/{budget_name}/rep{rep}: '
                f'{type(exc).__name__}: {exc}'
            )

    return {
        'checkpoint_dir': artifacts['run_dir'],
        'best_weights_path': final_weights_path if checkpoint_saved else None,
        'best_model_path': final_model_path if model_export_ok else None,
        'history_plot_path': history_plot_path if history_plot_ok else None,
        'is_best_checkpoint_saved': bool(checkpoint_saved),
        'model_export_ok': bool(model_export_ok),
        'history_plot_ok': bool(history_plot_ok),
    }
