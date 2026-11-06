"""Figure 3: how many of the 15 detectors have within-species density R2 above 0 at the fixed 0.25 threshold, one stacked
bar per calibrated species (brown = detectors below 0, teal = above), two columns grouped by set.

Species enter as in the macro summaries: R2 defined for all 15 detectors and at least 3 calibrated images. The species
counts are checked against tables/pooled_and_macro_species_density.csv.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from data import POOLED, read, species_order, species_values, verify_species_protocol
from style import DATASETS, DIVERGING, INK, MODELS, MUTED, OOD, STRONG, TEXT_WIDTH, TINT, save

BROWN, TEAL = plt.get_cmap(DIVERGING)(0.12), plt.get_cmap(DIVERGING)(0.85)  # the density R2 colours of the heatmaps
M = len(MODELS)
COLUMNS = [["OD-Valid", "T-MAWP", "T-MSWP"], ["T-DCWP", "T-GSH"]]


def counts(root):
    """One row per species with a defined R2 for all detectors: set, name, level, n, descriptive flag, below, above."""
    verify_species_protocol(root)
    sp = species_values(root)
    tables = read(root, POOLED).drop_duplicates("dataset").set_index("dataset")
    rows = []
    for d in DATASETS:
        sub = sp[sp["dataset"].eq(d)]
        info = species_order(sub)
        r2 = sub.pivot(index="scientific_name", columns="model", values="density_r2").loc[info["scientific_name"], MODELS]
        keep = np.isfinite(r2).all(axis=1).to_numpy() & (info["density_n"] >= 3).to_numpy()
        assert keep.sum() == tables.loc[d, "density_species_with_defined_r2"], d
        for r, (name, v) in zip(info[keep].itertuples(), r2[keep].iterrows()):
            rows.append(dict(dataset=d, species=name, level=r.ood_level, n=int(r.density_n),
                             descriptive=isinstance(r.small_n_flag, str), below=int((v < 0).sum()), above=int((v > 0).sum())))
    t = pd.DataFrame(rows)
    assert (t["below"] + t["above"]).eq(M).all()
    return t


def legend(fig, extra=(), fontsize=7.5):
    handles = [Patch(facecolor=BROWN), Patch(facecolor=TEAL)] + [h for h, _ in extra]
    labels = ["$R^2$ below 0: worse than the species-mean reference", "$R^2$ above 0"] + [l for _, l in extra]
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=fontsize,
               handlelength=1.2, columnspacing=1.4)


def page(t, out):
    """Two columns of species bars, each set under its own header in the table colour."""
    units = [sum(t["dataset"].eq(d).sum() + 2.0 for d in col) - 0.4 for col in COLUMNS]  # header 1.6, gap 0.4 per set
    height, top, bottom = 7.9, 0.55, 0.05  # inches
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    col_w, gap = (TEXT_WIDTH - 0.10) / 2, 0.10
    size = dict(header=9, species_count=8, level=7.5, name=8, n=7.5, count=8, legend=8.5)
    renderer = fig.canvas.get_renderer()

    def width(text, fontsize, style="normal"):
        label = fig.text(0, 0, text, fontsize=fontsize, fontstyle=style)
        w = label.get_window_extent(renderer).width / fig.dpi
        label.remove()
        return w

    level_w = 0.22
    name_w = max(width(r.species, size["name"], "italic") for r in t.itertuples()) + 0.05
    n_w = max(width(str(n), size["n"]) for n in t["n"]) + 0.05
    count_w = width(str(M), size["count"]) + 0.05
    bar_w = col_w - level_w - name_w - n_w - 0.04 - count_w
    assert bar_w >= 0.4, bar_w
    for c, sets in enumerate(COLUMNS):
        x0 = c * (col_w + gap)
        ax = fig.add_axes([0, bottom / height, 1, (height - top - bottom) / height])
        ax.set_xlim(0, TEXT_WIDTH)
        ax.set_ylim(max(units), 0)
        ax.axis("off")
        y = 0.0
        for d in sets:
            g = t[t["dataset"].eq(d)]
            ax.add_patch(Rectangle((x0, y + 0.1), col_w, 1.2, facecolor=TINT[d], edgecolor="none"))
            ax.plot([x0, x0 + col_w], [y + 1.3, y + 1.3], color=STRONG[d], lw=1.4, solid_capstyle="butt")
            ax.text(x0 + 0.05, y + 0.7, d, ha="left", va="center", fontsize=size["header"], fontweight="bold", color=INK)
            ax.text(x0 + col_w - 0.05, y + 0.7, f"{len(g)} species", ha="right", va="center", fontsize=size["species_count"],
                    color=MUTED)
            y += 1.6
            for r in g.itertuples():
                ax.add_patch(Rectangle((x0, y + 0.08), level_w - 0.03, 0.84, facecolor=OOD[r.level], edgecolor="none"))
                ax.text(x0 + (level_w - 0.03) / 2, y + 0.5, r.level, ha="center", va="center", fontsize=size["level"],
                        color=INK)
                ax.text(x0 + level_w + 0.03, y + 0.5, r.species, ha="left", va="center",
                        fontsize=size["name"], fontstyle="italic", color=INK)
                ax.text(x0 + level_w + name_w + n_w - 0.03, y + 0.5, str(r.n), ha="right", va="center", fontsize=size["n"],
                        color=MUTED)
                bx = x0 + level_w + name_w + n_w
                ax.add_patch(Rectangle((bx, y + 0.15), bar_w * r.below / M, 0.7, facecolor=BROWN, lw=0))
                ax.add_patch(Rectangle((bx + bar_w * r.below / M, y + 0.15), bar_w * r.above / M, 0.7, facecolor=TEAL,
                                       lw=0))
                ax.text(bx + bar_w + count_w, y + 0.5, str(r.below), ha="right", va="center", fontsize=size["count"], color=INK)
                y += 1
            y += 0.4
    legend(fig, [(Patch(facecolor="none", edgecolor="none"), f"Number: detectors with $R^2$ below 0 (of {M})")],
           fontsize=size["legend"])
    save(fig, "fig_species_r2_counts_page", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    page(counts(a.root), a.out)


if __name__ == "__main__":
    main()
