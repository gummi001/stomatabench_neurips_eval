"""Figures S12-S13: count MAE and density MAE with 95% paired-image bootstrap intervals at the fixed 0.25 threshold,
one panel per set, one row per model.

Reads the dataset-level rows of the bootstrap CSV and checks every point estimate against the biological metrics.
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from data import BIOLOGY, BOOTSTRAP, dataset_rows, read
from style import DATASETS, GRID, INK, MODELS, MUTED, STRONG, model_figure, save, set_header

# (figure key, endpoint, metric, biological column, x-axis label)
CI_METRICS = [("count_mae", "count", "MAE", "count_mae", "Count MAE (stomata per image)"),
              ("density_mae", "density", "MAE", "density_mae", "Density MAE (stomata mm$^{-2}$)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    b = read(a.root, BOOTSTRAP)
    ds = b[b["group_level"].eq("dataset")].set_index(["dataset", "model", "endpoint", "metric"])
    bio = dataset_rows(read(a.root, BIOLOGY), "Fixed", 0.25)
    for key, endpoint, metric, column, axis in CI_METRICS:
        fig, axes, rows = model_figure(3.35, top=0.785, bottom=0.11)
        for ax, d, letter in zip(axes, DATASETS, "abcde"):
            r = ds.xs((d, endpoint, metric), level=("dataset", "endpoint", "metric")).loc[MODELS]
            assert np.allclose(r["point_estimate"], bio.xs(d, level="dataset").loc[MODELS, column])
            lo, est, hi = r["ci_lower_2p5"], r["point_estimate"], r["ci_upper_97p5"]
            for m, y in rows.items():
                ax.plot([lo[m], hi[m]], [y, y], color=STRONG[d], lw=1.3, solid_capstyle="butt", zorder=3)
                ax.scatter(est[m], y, marker="o", s=18, zorder=4, clip_on=False, facecolor=STRONG[d], edgecolor=INK, lw=0.6)
            ax.set_xlim(0, float(hi.max()) * 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            set_header(ax, d, f"{int(r['n_images'].max()):,} images", letter)
        fig.supxlabel(axis, y=0.002, fontsize=8)
        handles = [Line2D([], [], marker="o", color=MUTED, lw=1.3, markerfacecolor=MUTED, markeredgecolor=INK,
                          markeredgewidth=0.6, markersize=4.4)]
        fig.legend(handles, ["Point estimate with its 95% paired-image bootstrap interval"], loc="upper center", ncol=1,
                   bbox_to_anchor=(0.56, 1.0), fontsize=7.5, handlelength=1.4, columnspacing=1.6)
        save(fig, f"fig_ci_{key}", a.out)


if __name__ == "__main__":
    main()
