"""Figure 2: F1 and count MAE of every model at the five confidence settings, one panel per set.

Columns of every block: fixed 0.25, the shared threshold, the model-specific threshold (MS), fixed 0.50 and 0.75.
One colour scale per metric for all five sets: F1 on viridis from 0 to 1, count MAE on a warm log scale.
Dataset level, scene-primary images, F1 at IoU 0.50.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

from data import BIOLOGY, OPERATING, THRESHOLD_PROTOCOLS, dataset_rows, read
from style import DATASETS, INK, MODELS, MUTED, SEQUENTIAL, STRONG, TEXT_WIDTH, TINT, WARM, family_bands, label, model_rows, \
    save

SETTINGS = [("Fixed", 0.25)] + THRESHOLD_PROTOCOLS  # 0.25, shared, model-specific, 0.50, 0.75
METRICS = [("F1", "op", "f1"), ("Count MAE", "bio", "count_mae")]  # (name, source, column)
KEY_Y, KEY_H, KEY_W, TITLE_PAD = 0.085, 0.03, 0.18, 2.5  # key row (figure fraction) and title gap (points)
SIZE = 8  # labels; keys one step smaller


def key_title(fig, x, text):
    fig.text(x, KEY_Y + KEY_H + TITLE_PAD / 72 / fig.get_figheight(), text, ha="center", va="baseline", fontsize=SIZE - 0.5,
             color=INK)


def colour_key(fig, x, title, note, cmap, norm, ticks):
    cax = fig.add_axes([x, KEY_Y, KEY_W, KEY_H])
    bar = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.minorticks_off()
    bar.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=SIZE - 0.5, length=2, width=0.4, pad=1)
    cax.set_xlabel(note, fontsize=SIZE - 1, color=MUTED, labelpad=1.5)
    key_title(fig, x + KEY_W / 2, title)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    frames = {"op": read(a.root, OPERATING), "bio": read(a.root, BIOLOGY)}
    values = {(d, col): pd.DataFrame({i: dataset_rows(frames[src], p, t).xs(d, level="dataset").loc[MODELS, col]
                                      for i, (p, t) in enumerate(SETTINGS)})
              for d in DATASETS for _, src, col in METRICS}
    shared, = set(np.round(dataset_rows(frames["op"], "Shared")["threshold"], 6))
    labels = [f"{t:.2f}" if p == "Fixed" else f"{shared:.3f}" if p == "Shared" else "MS" for p, t in SETTINGS]
    mae = np.concatenate([values[(d, "count_mae")].to_numpy().ravel() for d in DATASETS])
    scales = {"f1": (plt.get_cmap(SEQUENTIAL), Normalize(0, 1)), "count_mae": (WARM, LogNorm(mae.min(), mae.max()))}
    rows, headers = model_rows()
    n_rows = len(rows) + len(headers)

    fig, axes = plt.subplots(1, len(DATASETS), sharey=True, figsize=(TEXT_WIDTH, 3.35),
                             gridspec_kw=dict(left=0.155, right=0.985, bottom=0.2, top=0.86, wspace=0.08))
    for ax, d, letter in zip(axes, DATASETS, "abcde"):
        ax.set_facecolor("none")
        for k, (name, _, col) in enumerate(METRICS):
            cmap, norm = scales[col]
            v = values[(d, col)]
            x0 = k * (len(SETTINGS) + 0.6)
            for m, y in rows.items():
                for i in range(len(SETTINGS)):
                    ax.add_patch(Rectangle((x0 + i, y - 0.5), 1, 1, facecolor=cmap(norm(v.loc[m, i])), edgecolor="white", lw=0.3))
            ax.text(x0 + len(SETTINGS) / 2, -0.7, name, ha="center", va="bottom", fontsize=SIZE - 0.5, color=INK)
        width = 2 * len(SETTINGS) + 0.6
        ax.set_xlim(0, width)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.axis("off")
        top = -2.0
        ax.add_patch(Rectangle((0, top - 1.8), width, 1.8, facecolor=TINT[d], edgecolor="none", clip_on=False))
        ax.plot([0, width], [top, top], color=STRONG[d], lw=1.4, clip_on=False, solid_capstyle="butt")
        ax.text(width / 2, top - 0.9, d, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(0.25, top - 0.9, letter, ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
    for m, y in rows.items():
        axes[0].text(-0.3, y, label(m), ha="right", va="center", fontsize=SIZE, color=INK)
    family_bands(fig, axes[0], headers, fontsize=SIZE)

    colour_key(fig, 0.02, "F1: higher is better", "matched at IoU 0.50", *scales["f1"], [0, 0.25, 0.5, 0.75, 1])
    colour_key(fig, 0.26, "Count MAE: lower is better", "stomata per image, log scale", *scales["count_mae"],
               [t for t in (5, 10, 20, 50) if mae.min() <= t <= mae.max()])
    x0, cell = 0.5, 0.235 / TEXT_WIDTH
    for i, text in enumerate(labels):
        fig.add_artist(Rectangle((x0 + i * cell, KEY_Y), cell, KEY_H, transform=fig.transFigure, facecolor="#E9E9E9",
                                 edgecolor="white", lw=0.8))
        fig.text(x0 + (i + 0.5) * cell, KEY_Y + KEY_H / 2, text, ha="center", va="center", fontsize=SIZE - 1, color=INK)
    key_title(fig, x0 + 2.5 * cell, "Columns of every block")
    notes = ["Both selected on OD-Valid:", f"{shared:.3f} = shared threshold", "MS = model-specific threshold"]
    for text, y in zip(notes, (1.6, 0.5, -0.6)):
        fig.text(x0 + 5 * cell + 0.012, KEY_Y + KEY_H * y, text, ha="left", va="center", fontsize=SIZE - 1, color=INK)
    save(fig, "fig_threshold_sensitivity", a.out)


if __name__ == "__main__":
    main()
