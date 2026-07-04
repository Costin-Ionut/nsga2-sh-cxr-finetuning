"""plot_utils.py — Reusable plotting helpers for experiment analysis."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_boxplot(series_list, labels, ylabel, path, rotation=45) -> None:
    if not series_list:
        return
    plt.figure(figsize=(10, 5))
    plt.boxplot(series_list, tick_labels=labels, orientation='vertical')
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    save_plot(path)


def write_markdown_table(df: pd.DataFrame, path: str, title: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("No data available.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")
