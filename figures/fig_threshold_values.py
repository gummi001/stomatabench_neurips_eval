"""Figures S6-S11: set-level F1, count MAE, FP, FN, signed count bias and density MAE at the four other confidence
settings (shared, model-specific MS, fixed 0.50 and 0.75) in the layout of Figure 2, with every value printed in its cell.

F1 on viridis from 0 to 1; count MAE, FP, FN and density MAE on the warm map with a log scale; bias on the orange-purple
diverging map with a symmetric log scale. Dataset level, scene-primary images, IoU 0.50 where matched.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.patches import Rectangle

from data import BIOLOGY, OPERATING, THRESHOLD_PROTOCOLS, dataset_rows, read
from fig_species_heatmaps import BIAS, em, number
from fig_species_r2_heatmap import ink_for
from style import DATASETS, FAMILY_BAND, INK, MODELS, MUTED, SEQUENTIAL, STRONG, TEXT_WIDTH, TINT, WARM, label, model_rows, \
    save

# (key, source, column, title, digits)
METRICS = [("f1", "op", "f1", "F1 (IoU 0.50)", 3), ("count_mae", "bio", "count_mae", "Count MAE (stomata per image)", 2),
           ("fp", "op", "fp", "False positives (IoU 0.50)", 0), ("fn", "op", "fn", "False negatives (IoU 0.50)", 0),
           ("bias", "bio", "count_signed_bias", "Signed count bias (stomata per image)", 2),
           ("density_mae", "bio", "density_mae", "Density MAE (stomata mm$^{-2}$, calibrated images)", 2)]
LABEL_W, PANEL_GAP = 0.86, 0.07  # inches
TOP, BAND, HEADS, ROW, BOTTOM = 0.33, 0.17, 0.13, 0.104, 0.04  # inches: title and key, set band, column labels, row


def scale(key, values):
    v = np.concatenate([x.to_numpy().ravel() for x in values])
    if key == "f1":
        return plt.get_cmap(SEQUENTIAL), Normalize(0, 1), [0, 0.5, 1], ["0", "0.5", "1"], "higher is better"
    if key == "bias":
        top = float(np.abs(v).max())
        return (BIAS, SymLogNorm(1, vmin=-top, vmax=top, base=10), [-10, 0, 10], ["−10", "0", "10"],
                "orange overcounts, purple undercounts (symmetric log)")
    low = max(float(v.min()), 1.0)
    ticks = [t for t in (1, 10, 100, 1_000, 10_000, 100_000) if low <= t <= v.max()]
    if len(ticks) < 3:  # a short range: 2-5-10 steps
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000) if low <= t <= v.max()]
    labels = [f"{t:,}" for t in ticks]
    return WARM, LogNorm(low, float(v.max())), ticks, labels, "lower is better (log scale)"


def figure(key, title, digits, values, settings, stem, out):
    """values: {dataset: DataFrame (MODELS x settings)}."""
    cmap, norm, ticks, tick_labels, direction = scale(key, values.values())
    rows, headers = model_rows()
    n_rows = len(rows) + len(headers)
    n_cols = len(settings)
    cell = (TEXT_WIDTH - LABEL_W - 0.02 - (len(DATASETS) - 1) * PANEL_GAP) / (len(DATASETS) * n_cols)
    height = TOP + BAND + HEADS + n_rows * ROW + BOTTOM
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(0.04, 0.12, title, ha="left", va="top", fontsize=7.5, fontweight="bold", color=INK)
    cax = fig.add_axes([3.55 / TEXT_WIDTH, 1 - 0.20 / height, 1.85 / TEXT_WIDTH, 0.055 / height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.ax.minorticks_off()
    cb.set_ticks(ticks, labels=tick_labels)
    cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.4, pad=1, which="both")
    cax.set_title(direction, fontsize=6, color=MUTED, pad=2)

    y_row = lambda y: TOP + BAND + HEADS + y * ROW
    for family, y in headers:
        ax.add_patch(Rectangle((0.005, y_row(y)), TEXT_WIDTH - 0.01, ROW, facecolor=FAMILY_BAND, edgecolor="none"))
        ax.text(0.03, y_row(y) + ROW / 2, family, ha="left", va="center", fontsize=5.8, fontweight="bold",
                fontstyle="italic", color=INK)
    for m, y in rows.items():
        ax.text(LABEL_W - 0.04, y_row(y) + ROW / 2, label(m), ha="right", va="center", fontsize=6, color=INK)
    for p, d in enumerate(DATASETS):
        x0 = LABEL_W + p * (n_cols * cell + PANEL_GAP)
        width = n_cols * cell
        ax.add_patch(Rectangle((x0, TOP), width, BAND, facecolor=TINT[d], edgecolor="none"))
        ax.plot([x0, x0 + width], [TOP + BAND] * 2, color=STRONG[d], lw=1.0, solid_capstyle="butt")
        ax.text(x0 + width / 2, TOP + BAND / 2, d, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)
        for i, s in enumerate(settings):
            ax.text(x0 + (i + 0.5) * cell, TOP + BAND + HEADS / 2 + 0.005, s, ha="center", va="center", fontsize=5.2,
                    color=MUTED)
        v = values[d]
        for m, y in rows.items():
            for i in range(n_cols):
                value = float(v.loc[m].iloc[i])
                face = cmap(norm(max(value, norm.vmin) if isinstance(norm, LogNorm) else value))
                ax.add_patch(Rectangle((x0 + i * cell + 0.004, y_row(y) + 0.006), cell - 0.008, ROW - 0.012, facecolor=face,
                                       edgecolor="none"))
                text = number(value, digits)
                size = min(5.2, (cell * 72 - 1.4) / em(text))
                ax.text(x0 + (i + 0.5) * cell, y_row(y) + ROW / 2, text, ha="center", va="center", fontsize=size,
                        color=ink_for(face))
    save(fig, stem, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    frames = {"op": read(a.root, OPERATING), "bio": read(a.root, BIOLOGY)}
    (shared,) = set(np.round(dataset_rows(frames["op"], "Shared")["threshold"], 3))
    names = [f"{t:.2f}" if p == "Fixed" else f"{shared:.3f}" if p == "Shared" else "MS" for p, t in THRESHOLD_PROTOCOLS]
    for key, src, col, title, digits in METRICS:
        values = {d: pd.DataFrame({i: dataset_rows(frames[src], p, t).xs(d, level="dataset").loc[MODELS, col]
                                   for i, (p, t) in enumerate(THRESHOLD_PROTOCOLS)}) for d in DATASETS}
        figure(key, title, digits, values, names, f"fig_thr_{key}", a.out)


if __name__ == "__main__":
    main()
