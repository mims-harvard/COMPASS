import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.stats import binomtest
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest


def plot_calibration(
    y_true,
    y_prob,
    *,
    ax=None,
    label=None,
    color=None,
    n_bins=10,
    strategy="quantile",      # "quantile" or "uniform"
    show_ci=True,
    show_counts=False,
    smooth=False,             # LOESS-like local smoothing for visualization
    smooth_window=0.05,
):
    """
    Plot a robust calibration (reliability) curve.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary ground truth labels (0/1).
    y_prob : array-like, shape (n_samples,)
        Predicted probabilities in [0, 1].
    ax : matplotlib.axes.Axes
        Axis to plot on.
    label : str
        Label for legend.
    color : str
        Line color.
    n_bins : int
        Number of bins (used as upper bound; empty bins are dropped).
    strategy : str
        'quantile' (recommended) or 'uniform'.
    show_ci : bool
        Whether to show binomial confidence intervals.
    show_counts : bool
        Whether to annotate sample size per bin.
    smooth : bool
        Whether to overlay a smoothed calibration curve (visual aid only).
    smooth_window : float
        Half window size for smoothing in probability space.
    """

    # ---------- setup ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    # ---------- define bins ----------
    if strategy == "quantile":
        bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    # handle discrete predictors (e.g. PD-L1)
    bins = np.unique(bins)
    if len(bins) <= 2:
        # fallback: unique probability values
        bins = np.unique(y_prob)
        bins = np.r_[bins, bins[-1] + 1e-6]

    bin_ids = np.digitize(y_prob, bins, right=True)

    mean_pred, frac_pos, lower, upper, counts = [], [], [], [], []

    # ---------- compute per-bin statistics ----------
    for b in range(1, len(bins)):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue

        p_mean = y_prob[idx].mean()
        y_rate = y_true[idx].mean()

        mean_pred.append(p_mean)
        frac_pos.append(y_rate)
        counts.append(idx.sum())

        if show_ci:
            k = int(y_true[idx].sum())
            n = idx.sum()
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95)
            lower.append(ci.low)
            upper.append(ci.high)

    mean_pred = np.asarray(mean_pred)
    frac_pos = np.asarray(frac_pos)

    # ---------- plot reliability curve ----------
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        color=color,
        label=label,
        zorder=3,
    )

    if show_ci and len(lower) == len(mean_pred):
        ax.fill_between(
            mean_pred,
            lower,
            upper,
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=2,
        )

    # ---------- optional smoothing (visual only) ----------
    if smooth:
        grid = np.linspace(0.01, 0.99, 100)
        smoothed = []

        for g in grid:
            m = np.abs(y_prob - g) <= smooth_window
            if m.sum() < 5:
                smoothed.append(np.nan)
            else:
                smoothed.append(y_true[m].mean())

        ax.plot(
            grid,
            smoothed,
            color=color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )

    # ---------- annotations ----------
    if show_counts:
        for x, y, n in zip(mean_pred, frac_pos, counts):
            ax.text(x, y, f"n={n}", fontsize=8, ha="center", va="bottom")

    # ---------- formatting ----------
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted response probability")
    ax.set_ylabel("Observed response rate")

    if label:
        ax.legend(frameon=False)

    return ax
