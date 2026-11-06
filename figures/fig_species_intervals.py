"""Figures S14-S18: per-species 95% paired-image bootstrap intervals at the fixed 0.25 threshold as forest plots, for
every T-MAWP species and the calibrated Gibbs wheat images, four species per page.

Each species is one block with one panel per measure (count MAE, density MAE, density R2) and one row per detector.
Density R2 below -1 is drawn at -1 with a left-pointing marker. Gibbs wheat has density only, because the species'
count rows also include the uncalibrated StomataTracker images. Reads the species-level rows of the bootstrap CSV.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from data import BOOTSTRAP, SPECIES, read
from style import FAMILY_BAND, GRID, INK, MODELS, MUTED, OOD, STRONG, TEXT_WIDTH, TINT, label, model_rows, save

PER_PAGE = 4
CLIP = -1.0
# (endpoint, metric, panel title, digits)
MEASURES = [("count", "MAE", "Count MAE (stomata per image)", 2), ("density", "MAE", "Density MAE (stomata mm$^{-2}$)", 1),
            ("density", "predictive_R2", "Density $R^2$", 2)]
LABEL_W, GAP, VALUE_W = 0.98, 0.17, 0.30  # inches: model names, space between panels, printed estimates
PANEL_W = (TEXT_WIDTH - LABEL_W - 0.02 - 2 * GAP) / 3 - VALUE_W
HEAD, TITLE, ROW, TICKS, SPACE = 0.17, 0.15, 0.068, 0.15, 0.13  # inches


def targets(root):
    """The species of the interval tables, in their order: T-MAWP by OOD level and name, then Gibbs wheat."""
    sp = read(root, SPECIES)
    mawp = sp[sp["dataset"].eq("T-MAWP")].drop_duplicates("scientific_name").sort_values(["ood_level", "scientific_name"])
    wheat = sp[sp["dataset"].eq("T-GSH") & sp["scientific_name"].eq("Triticum aestivum")].drop_duplicates("scientific_name")
    return [("T-MAWP", r.scientific_name, r.ood_level) for r in mawp.itertuples()] + [
        ("T-GSH", r.scientific_name, r.ood_level) for r in wheat.itertuples()]


def text_for(v, digits):
    if v < -10:
        return "<−10"
    s = f"{v:.{digits}f}"
    return (s[1:] if s.startswith("-") and float(s) == 0 else s).replace("-", "−")


def block(fig, top, height, d, name, level, part):
    """One species at top (inches from the figure top)."""
    rows, headers = model_rows()
    n_rows = len(rows) + len(headers)
    y_in = lambda y: top + HEAD + TITLE + (y + 0.5) * ROW  # row centre, inches from the top
    to_fig = lambda x, y: (x / TEXT_WIDTH, 1 - y / height)
    fig.add_artist(Rectangle(to_fig(0.005, top + HEAD), (TEXT_WIDTH - 0.01) / TEXT_WIDTH, HEAD / height,
                             transform=fig.transFigure, facecolor=TINT[d], edgecolor="none"))
    fig.add_artist(plt.Line2D([0.005 / TEXT_WIDTH, 1 - 0.005 / TEXT_WIDTH], [1 - (top + HEAD) / height] * 2,
                              transform=fig.transFigure, color=STRONG[d], lw=1.0))
    fig.add_artist(Rectangle(to_fig(0.03, top + HEAD - 0.025), 0.2 / TEXT_WIDTH, (HEAD - 0.05) / height,
                             transform=fig.transFigure, facecolor=OOD[level], edgecolor="none"))
    fig.text(*to_fig(0.13, top + HEAD / 2), level, ha="center", va="center", fontsize=6, color=INK)
    n = int(part["n_images"].max())
    n_density = int(part.xs(("density", "MAE"), level=("endpoint", "metric"))["n_images"].iloc[0])
    scope = (f"{n_density:,} calibrated Gibbs images; density only" if name == "Triticum aestivum"
             else f"{n:,} images")
    fig.text(*to_fig(0.29, top + HEAD / 2), name, ha="left", va="center", fontsize=7, fontweight="bold", fontstyle="italic",
             color=INK)
    fig.text(*to_fig(TEXT_WIDTH - 0.04, top + HEAD / 2), f"{d}  ·  {scope}", ha="right", va="center", fontsize=6.5,
             color=MUTED)
    for family, y in headers:
        fig.add_artist(Rectangle(to_fig(0.005, y_in(y) + ROW / 2), (TEXT_WIDTH - 0.01) / TEXT_WIDTH, ROW / height,
                                 transform=fig.transFigure, facecolor=FAMILY_BAND, edgecolor="none", zorder=-1))
        fig.text(*to_fig(0.03, y_in(y)), family, ha="left", va="center", fontsize=5.4, fontweight="bold", fontstyle="italic",
                 color=INK)
    for m, y in rows.items():
        fig.text(*to_fig(LABEL_W - 0.04, y_in(y)), label(m), ha="right", va="center", fontsize=5.6, color=INK)
    for i, (endpoint, metric, title, digits) in enumerate(MEASURES):
        x0 = LABEL_W + i * (PANEL_W + VALUE_W + GAP)
        ax = fig.add_axes([x0 / TEXT_WIDTH, 1 - (top + HEAD + TITLE + n_rows * ROW) / height, PANEL_W / TEXT_WIDTH,
                           n_rows * ROW / height])
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_facecolor("none")
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(axis="x", labelsize=5.6, length=1.5, width=0.5, pad=1)
        fig.text(*to_fig(x0 + (PANEL_W + VALUE_W) / 2, top + HEAD + TITLE / 2 + 0.01), title, ha="center", va="center",
                 fontsize=6.2, color=INK)
        if name == "Triticum aestivum" and endpoint == "count":
            ax.axis("off")
            fig.text(*to_fig(x0 + PANEL_W / 2, top + HEAD + TITLE + n_rows * ROW / 2), "not reported:\ncount rows include\n"
                     "uncalibrated images", ha="center", va="center", fontsize=5.6, color=MUTED)
            continue
        r = part.xs((endpoint, metric), level=("endpoint", "metric")).loc[MODELS]
        lo, est, hi = r["ci_lower_2p5"], r["point_estimate"], r["ci_upper_97p5"]
        if metric == "predictive_R2":
            ax.set_xlim(CLIP - 0.12, 1.05)
            ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
            ax.set_xticks([-1, 0, 1], ["≤−1", "0", "1"])
        else:
            ax.set_xlim(0, float(hi.max()) * 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        for m, y in rows.items():
            ax.plot([max(lo[m], CLIP), max(hi[m], CLIP)], [y, y], color=STRONG[d], lw=1.0, solid_capstyle="butt", zorder=3)
            ax.scatter(max(est[m], CLIP), y, marker="<" if est[m] < CLIP else "o", s=7, zorder=4, clip_on=False,
                       facecolor=STRONG[d], edgecolor=INK, lw=0.4)
            fig.text(*to_fig(x0 + PANEL_W + VALUE_W - 0.04, y_in(y)), text_for(est[m], digits), ha="right", va="center",
                     fontsize=5.4, color=INK)


def page(blocks, b, stem, out):
    one = HEAD + TITLE + (len(MODELS) + 5) * ROW + TICKS + SPACE
    height = len(blocks) * one
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    for i, (d, name, level) in enumerate(blocks):
        block(fig, i * one, height, d, name, level, b.loc[(d, name)])
    save(fig, stem, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    ap.add_argument("pages", nargs="*", type=int, help="page numbers to draw (default: all)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    b = read(a.root, BOOTSTRAP)
    b = b[b["group_level"].eq("species")].set_index(["dataset", "scientific_name", "model", "endpoint", "metric"]).sort_index()
    species = targets(a.root)
    pages = [species[i:i + PER_PAGE] for i in range(0, len(species), PER_PAGE)]
    for k, blocks in enumerate(pages, 1):
        if not a.pages or k in a.pages:
            page(blocks, b, f"fig_ci_species_{k}", a.out)


if __name__ == "__main__":
    main()
