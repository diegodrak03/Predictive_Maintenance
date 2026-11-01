import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from pm import config

# Categorical slots 1 and 2 of the default palette, plus ink for reference lines
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#444444"


def save_figure(fig, name):
    """
    Save a figure to results/figures/<name>.png (150 dpi, tight) and close it.

    Returns:
        Path: The saved file.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def style_axes(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def proba_histogram(y, proba, threshold=config.SELECTED_THRESHOLD, title=None):
    """
    Histogram of predicted probabilities by class, log y axis, with the threshold marked.
    """
    y = np.asarray(y)
    bins = np.linspace(0, 1, 51)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba[y == 0], bins=bins, color=BLUE, alpha=0.7, label="No failure", edgecolor="white", linewidth=0.5)
    ax.hist(proba[y == 1], bins=bins, color=ORANGE, alpha=0.7, label="Failure", edgecolor="white", linewidth=0.5)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1, label=f"Threshold {threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted failure probability [-]")
    ax.set_ylabel("Number of processes (log scale)")
    ax.set_title(title or "Out-of-fold predicted probabilities by class")
    ax.legend(frameon=False)
    style_axes(ax)
    return fig


def pr_curve(y, proba, threshold=config.SELECTED_THRESHOLD, min_precision=config.MIN_PRECISION, title=None):
    """
    Precision-recall curve with the minimum precision line and the selected threshold marked.
    """
    precision, recall, thresholds = precision_recall_curve(y, proba)

    # Operating point: the first curve point whose threshold reaches the selected one
    i = np.searchsorted(thresholds, threshold)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=BLUE, linewidth=2, label="Precision-recall curve")
    ax.axhline(min_precision, color=INK, linestyle="--", linewidth=1, label=f"Precision = {min_precision}")
    ax.plot(recall[i], precision[i], "o", color=ORANGE, markersize=9, markeredgecolor="white",
            markeredgewidth=2, label=f"Threshold {threshold}")
    ax.annotate(f"P = {precision[i]:.3f}, R = {recall[i]:.3f}", (recall[i], precision[i]),
                xytext=(-10, -25), textcoords="offset points", ha="right", color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall [-]")
    ax.set_ylabel("Precision [-]")
    ax.set_title(title or "Precision-recall curve (out-of-fold)")
    ax.legend(frameon=False, loc="lower left")
    style_axes(ax)
    return fig

