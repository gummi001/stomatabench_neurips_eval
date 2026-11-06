"""Figures S21-S42 and S51-S58: species x detector heatmaps with every value printed in its cell.

S21-S42 (fixed 0.25): one pair of pages per metric, (a) OD-Valid, T-MAWP and T-DCWP, (b) T-MSWP and T-GSH, for COCO
AP50:95, precision, recall, F1, TP, FP, FN, count MAE, count R2, density MAE and signed count bias.
S51-S58 (shared, model-specific, 0.50 and 0.75): count MAE, pages split as the density R2 pages,
(a) OD-Valid, T-MAWP, T-MSWP and T-GSH, (b) T-DCWP.
Layout of fig_species_r2_heatmap.py. Count MAE has one colour scale for all settings, FP and FN share one.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.patches import Rectangle

from data import BIOLOGY, read, species_order, species_values
from fig_species_r2_heatmap import BAND, LEFT, N_RIGHT, PAGES, RHO_RIGHT, ROW, SET_GAP, SETTINGS, TOP, columns, ink_for, text_for
from style import DIVERGING, INK, MODELS, MUTED, NA, OOD, SEQUENTIAL, STRONG, TEXT_WIDTH, TINT, WARM, label, save

S3_PAGES = [("a", ["OD-Valid", "T-MAWP", "T-DCWP"]), ("b", ["T-MSWP", "T-GSH"])]
LOW = 1.0  # the light end of the log scales
BIAS = plt.get_cmap("PuOr_r")  # purple below 0 (undercount), orange above (overcount)
# Context columns: (header, column, format) twice.
CONTEXT = {"count": [("$N$", "n_images", "{:,.0f}"), ("$\\bar{c}_{\\mathrm{GT}}$", "count_gt_mean", "{:.1f}")],
           "boxes": [("$N$", "n_images", "{:,.0f}"), ("GT", "gt_boxes", "{:,.0f}")],
           "density": [("$N$", "density_n", "{:,.0f}"), ("$\\bar{\\rho}_{\\mathrm{GT}}$", "density_gt_mean", "{:.1f}")]}
# Fixed 0.25 metrics: (key, column, heading, context, digits, scale). The key names the figures (fig_species_<key>_<a|b>).
S3_METRICS = [
    ("ap", "AP50_95", "COCO AP$_{50:95}$ (%), all released scores", "count", 1, "ap"),
    ("precision", "precision", "Precision at $\\tau$ = 0.25", "count", 2, "unit"),
    ("recall", "recall", "Recall at $\\tau$ = 0.25", "count", 2, "unit"),
    ("f1", "f1", "F1 at $\\tau$ = 0.25", "count", 2, "unit"),
    ("tp", "tp", "True positives at $\\tau$ = 0.25", "boxes", 0, "tp"),
    ("fp", "fp", "False positives at $\\tau$ = 0.25", "boxes", 0, "fpfn"),
    ("fn", "fn", "False negatives at $\\tau$ = 0.25", "boxes", 0, "fpfn"),
    ("count_mae", "count_mae", "Count MAE at $\\tau$ = 0.25", "count", 1, "count_mae"),
    ("count_r2", "count_r2", "Count $R^2$ at $\\tau$ = 0.25", "count", 2, "r2"),
    ("density_mae", "density_mae", "Density MAE at $\\tau$ = 0.25", "density", 1, "density_mae"),
    ("count_bias", "count_signed_bias", "Signed count bias at $\\tau$ = 0.25", "count", 1, "bias"),
]


def number(v, digits):
    s = f"{v:.{digits}f}"
    return (s[1:] if s.startswith("-") and float(s) == 0 else s).replace("-", "−")


def cell_text(v, digits, scale):
    if np.isnan(v):
        return "–"
    return text_for(v) if scale == "r2" else number(v, digits)


def em(s):
    """Width of a printed value in em of Times (digits half an em)."""
    return sum(0.25 if c in ".," else 0.564 if c in "−<" else 0.5 for c in s)


def log_ticks(vmax):
    ticks = [10 ** k for k in range(int(np.floor(np.log10(vmax))) + 1)]
    labels = ["≤1"] + [f"{t:g}" if t < 1000 else f"$10^{{{int(np.log10(t))}}}$" for t in ticks[1:]]
    return ticks, labels


def scales(sp, s4_values):
    """Colour map, norm, colour-bar ticks and labels, extend and colour-bar title per scale."""
    mae_max = max([sp["count_mae"].max()] + [v["count_mae"].max() for v in s4_values])
    fpfn_max = max(sp["fp"].max(), sp["fn"].max())
    bias_max = sp["count_signed_bias"].abs().max()
    seq = plt.get_cmap(SEQUENTIAL)
    out = {"ap": (seq, Normalize(0, 100), [0, 50, 100], ["0", "50", "100"], "neither", "lighter is higher"),
           "unit": (seq, Normalize(0, 1), [0, 0.5, 1], ["0", "0.5", "1"], "neither", "lighter is higher"),
           "r2": (plt.get_cmap(DIVERGING), Normalize(-1, 1), [-1, 0, 1], ["≤−1", "0", "1"], "min",
                  "teal above the species mean (0), brown below"),
           "bias": (BIAS, SymLogNorm(1, vmin=-bias_max, vmax=bias_max, base=10), [-100, -10, 0, 10, 100],
                    ["−100", "", "0", "", "100"], "neither", "orange overcounts, purple undercounts (symmetric log)")}
    for name, cmap, vmax, title in [("tp", seq, sp["tp"].max(), "log scale: lighter is more"),
                                    ("fpfn", WARM, fpfn_max, "log scale: lighter is fewer"),
                                    ("count_mae", WARM, mae_max, "log scale: lighter is lower error"),
                                    ("density_mae", WARM, sp["density_mae"].max(), "log scale: lighter is lower error")]:
        out[name] = (cmap, LogNorm(LOW, vmax), *log_ticks(vmax), "min", title)
    return out


def colour(v, scale, cmap, norm):
    if scale == "r2":
        v = max(v, -1.0)
    elif isinstance(norm, LogNorm):
        v = max(v, LOW)
    return cmap(norm(v))


def page(sets, frame, column, digits, scale, spec, context, heading, stem, bar_top, out, density=False):
    """One heatmap page: sets in order, rows from frame (one row per dataset, model and species)."""
    cmap, norm, ticks, tick_labels, extend, bar_title = spec
    xs, w = columns()
    blocks = []
    for d in sets:
        sub = frame[frame["dataset"].eq(d)]
        if density:
            sub = sub[sub["density_n"].gt(0)]
        blocks.append((d, species_order(sub), sub))
    texts = [cell_text(v, digits, scale) for _, _, sub in blocks for v in sub[column]]
    size = min(4.9, (w * 72 - 1.6) / max(em(t) for t in texts))
    bottom = 0.10 if bar_top else 0.42
    n_rows = sum(len(info) for _, info, _ in blocks)
    height = TOP + len(blocks) * (BAND + SET_GAP) + n_rows * ROW + bottom
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(0.04, 0.12, heading, ha="left", va="top", fontsize=7, fontweight="bold", color=INK)
    for x, m in zip(xs, MODELS):
        ax.text(x + w / 2, TOP - 0.04, label(m), rotation=90, ha="center", va="bottom", fontsize=5.8, color=INK)
    for right, (head, _, _) in zip([N_RIGHT, RHO_RIGHT], CONTEXT[context]):
        ax.text(right, TOP - 0.04, head, ha="right", va="bottom", fontsize=6.2, color=MUTED)

    y = TOP
    for d, info, sub in blocks:
        y += SET_GAP
        ax.add_patch(Rectangle((0.005, y), TEXT_WIDTH - 0.01, BAND, facecolor=TINT[d], edgecolor="none"))
        ax.plot([0.005, TEXT_WIDTH - 0.005], [y + BAND, y + BAND], color=STRONG[d], lw=1.0, solid_capstyle="butt")
        ax.text(0.04, y + BAND / 2, f"{d}  ·  {len(info)} {'calibrated ' if density else ''}species", ha="left",
                va="center", fontsize=7, fontweight="bold", color=INK)
        y += BAND
        grid = sub.pivot(index="scientific_name", columns="model", values=column).loc[info["scientific_name"], MODELS]
        for r, row in zip(info.itertuples(), grid.to_numpy()):
            ax.add_patch(Rectangle((0.02, y + 0.008), 0.22, ROW - 0.016, facecolor=OOD[r.ood_level], edgecolor="none"))
            ax.text(0.13, y + ROW / 2, r.ood_level, ha="center", va="center", fontsize=5.6, color=INK)
            ax.text(0.29, y + ROW / 2, r.scientific_name, ha="left", va="center", fontsize=6.2, fontstyle="italic", color=INK)
            for right, (_, col, fmt) in zip([N_RIGHT, RHO_RIGHT], CONTEXT[context]):
                ax.text(right, y + ROW / 2, fmt.format(getattr(r, col)), ha="right", va="center", fontsize=5.8, color=MUTED)
            for x, v in zip(xs, row):
                if np.isnan(v):
                    ax.add_patch(Rectangle((x + 0.004, y + 0.006), w - 0.008, ROW - 0.012, facecolor="white", edgecolor=NA,
                                           hatch="//////", lw=0))
                    ax.text(x + w / 2, y + ROW / 2, "–", ha="center", va="center", fontsize=5, color=MUTED)
                    continue
                face = colour(v, scale, cmap, norm)
                ax.add_patch(Rectangle((x + 0.004, y + 0.006), w - 0.008, ROW - 0.012, facecolor=face, edgecolor="none"))
                ax.text(x + w / 2, y + ROW / 2, cell_text(v, digits, scale), ha="center", va="center", fontsize=size,
                        color=ink_for(face))
            y += ROW

    if bar_top:  # under the heading, left of the model names
        cax = fig.add_axes([0.06 / TEXT_WIDTH, 1 - 0.43 / height, 1.3 / TEXT_WIDTH, 0.06 / height])
    else:  # under the grid
        cax = fig.add_axes([xs[3] / TEXT_WIDTH, 0.13 / height, (xs[-4] + w - xs[3]) / TEXT_WIDTH, 0.065 / height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=extend)
    cb.outline.set_linewidth(0.4)
    cb.ax.minorticks_off()
    cb.set_ticks(ticks, labels=tick_labels)
    cb.ax.tick_params(labelsize=6.2, length=1.5, width=0.4, pad=1, which="both")
    cax.set_title(bar_title, fontsize=6.2, color=MUTED, pad=2, loc="left" if bar_top else "center")
    save(fig, stem, out)
    plt.close(fig)


def species_frame(root):
    """Species values at fixed 0.25 with the mean annotated stomata per image, and the biological file."""
    sp = species_values(root)
    assert set(sp["image_scope"]) == {"scene_primary"}
    bio = read(root, BIOLOGY)
    keep = (bio["group_level"].eq("species") & bio["image_scope"].eq("scene_primary") & bio["confidence_protocol"].eq("Fixed")
            & np.isclose(bio["threshold"], 0.25))
    m = sp.merge(bio.loc[keep, ["dataset", "model", "scientific_name", "count_gt_mean", "count_mae", "count_n"]],
                 on=["dataset", "model", "scientific_name"], suffixes=("", "_bio"), validate="one_to_one")
    assert len(m) == len(sp) and (m["count_n"] == m["n_images"]).all()
    assert np.allclose(m["count_mae"], m["count_mae_bio"])
    return m, bio


def at_setting(sp, bio, protocol, threshold):
    """Species count MAE at another confidence setting, with the context columns of the 0.25 values, and its heading."""
    keep = bio["group_level"].eq("species") & bio["image_scope"].eq("scene_primary") & bio["confidence_protocol"].eq(protocol)
    if threshold is not None:
        keep &= np.isclose(bio["threshold"], threshold)
    r = bio.loc[keep, ["dataset", "model", "scientific_name", "threshold", "count_mae", "count_n", "count_gt_mean"]]
    info = sp[["dataset", "model", "scientific_name", "ood_level", "n_images", "count_gt_mean"]]
    m = info.merge(r, on=["dataset", "model", "scientific_name"], suffixes=("", "_at"), validate="one_to_one")
    assert len(m) == len(sp) == len(r), protocol
    assert (m["count_n"] == m["n_images"]).all() and np.allclose(m["count_gt_mean"], m["count_gt_mean_at"])
    assert m["count_mae"].notna().all()
    if protocol == "Shared":
        (t,) = set(r["threshold"].round(3))
        name = f"shared $\\tau$ = {t:.3f}"
    elif protocol == "Model-specific":
        name = "model-specific $\\tau$"
    else:
        name = f"$\\tau$ = {threshold:.2f}"
    return m, name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sp, bio = species_frame(a.root)
    s4 = [(key, *at_setting(sp, bio, protocol, threshold)) for key, protocol, threshold in SETTINGS if key != "fixed025"]
    spec = scales(sp, [values for _, values, _ in s4])
    for key, column, heading, context, digits, scale in S3_METRICS:
        for letter, sets in S3_PAGES:
            page(sets, sp, column, digits, scale, spec[scale], context, heading, f"fig_species_{key}_{letter}", True, a.out,
                 density=context == "density")
    for key, values, setting in s4:
        for letter, sets in PAGES:
            page(sets, values, "count_mae", 1, "count_mae", spec["count_mae"], "count", f"Confidence: {setting}",
                 f"fig_species_count_mae_{key}_{letter}", False, a.out)


if __name__ == "__main__":
    main()
