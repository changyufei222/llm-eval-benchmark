from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_metric_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return

    plot_frame = numeric.copy()
    if "model_label" in frame.columns and "method" in frame.columns:
        labels = frame["model_label"].astype(str) + "\n" + frame["method"].astype(str)
    elif "method" in frame.columns:
        labels = frame["method"].astype(str)
    elif "model_label" in frame.columns:
        labels = frame["model_label"].astype(str)
    else:
        labels = frame.index.astype(str)

    ax = plot_frame.plot(kind="bar", figsize=(9, 4.5))
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
