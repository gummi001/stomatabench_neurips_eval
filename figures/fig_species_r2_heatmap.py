"""Figures S19-S20 and S43-S50: within-species density R2 of every detector and calibrated species, printed in each
cell, on two pages per confidence setting: (a) OD-Valid, T-MAWP, T-MSWP and T-GSH, (b) T-DCWP.

Rows are species in the order of the SI species tables, grouped by set, with the number of calibrated images N and the
mean annotated density; columns are the 15 detectors in table order. Colour: the diverging map clipped at -1; values
below -10 print as <-10; a species whose annotated density does not vary has no R2 (printed as -).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Rectangle

from data import BIOLOGY, read, species_order, species_values, verify_species_protocol
from style import DIVERGING, FAMILIES, INK, MODELS, MUTED, NA, OOD, STRONG, TEXT_WIDTH, TINT, label, save

# Confidence settings: (file key, protocol in the biological file, fixed threshold or None).
SETTINGS = [("fixed025", "Fixed", 0.25), ("shared", "Shared", None), ("model_specific", "Model-specific", None),
            ("fixed050", "Fixed", 0.50), ("fixed075", "Fixed", 0.75)]
PAGES = [("a", ["OD-Valid", "T-MAWP", "T-MSWP", "T-GSH"]), ("b", ["T-DCWP"])]
ROW, BAND, SET_GAP = 0.104, 0.19, 0.07  # inches: species row, set header band, space before a set
TOP, BOTTOM, LEFT = 0.66, 0.42, 2.14  # inches: model labels above, colour bar below, species labels at the left
N_RIGHT, RHO_RIGHT = LEFT - 0.42, LEFT - 0.05  # right edges of the N and mean-density columns
FLOOR = -10  # values below this print as <-10
NORM, CMAP = Normalize(-1, 1), plt.get_cmap(DIVERGING)


def columns():
    """x of each model column (inches) and the column width, with half a column between families."""
    units = len(MODELS) + 0.5 * (len(FAMILIES) - 1)
    width = (TEXT_WIDTH - LEFT - 0.03) / units
    xs, x = [], LEFT
    for _, n in FAMILIES:
        xs += [x + i * width for i in range(n)]
        x += (n + 0.5) * width
    return np.array(xs), width


def text_for(v):
    if np.isnan(v):
        return "–"
    if v < FLOOR:
        return f"<{FLOOR}".replace("-", "−")
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "−")


def ink_for(face):
    r, g, b = to_rgb(face)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def at_setting(sp, bio, protocol, threshold):
    """The species values with density R2 at another confidence setting, and the setting's label."""
    keep = bio["group_level"].eq("species") & bio["image_scope"].eq("scene_primary") & bio["confidence_protocol"].eq(protocol)
    if threshold is not None:
        keep &= np.isclose(bio["threshold"], threshold)
    r = bio.loc[keep, ["dataset", "model", "scientific_name", "threshold", "density_r2", "density_n", "density_gt_mean"]]
    assert len(r) == len(sp) and not r.duplicated(["dataset", "model", "scientific_name"]).any(), protocol
    m = sp.drop(columns="density_r2").merge(r, on=["dataset", "model", "scientific_name"], suffixes=("", "_at"),
                                             validate="one_to_one")
    assert np.allclose(m["density_n"], m["density_n_at"]) and np.allclose(m["density_gt_mean"], m["density_gt_mean_at"],
                                                                          equal_nan=True)
    if protocol == "Shared":
        (t,) = set(r["threshold"].round(3))
        name = f"shared $\\tau$ = {t:.3f}"
    elif protocol == "Model-specific":
        name = "model-specific $\\tau$"
    else:
        name = f"$\\tau$ = {threshold:.2f}"
    return m, name


def page(sets, sp, stem, setting, out):
    xs, w = columns()
    blocks = []
    for d in sets:
        sub = sp[sp["dataset"].eq(d)]
        calibrated = sub[sub["density_n"].gt(0)]
        blocks.append((d, species_order(calibrated), calibrated))
    n_rows = sum(len(info) for _, info, _ in blocks)
    height = TOP + len(blocks) * (BAND + SET_GAP) + n_rows * ROW + BOTTOM
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(0.04, 0.12, f"Confidence: {setting}", ha="left", va="top", fontsize=7, fontweight="bold", color=INK)
    for x, m in zip(xs, MODELS):
        ax.text(x + w / 2, TOP - 0.04, label(m), rotation=90, ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.text(N_RIGHT, TOP - 0.04, "$N$", ha="right", va="bottom", fontsize=6.2, color=MUTED)
    ax.text(RHO_RIGHT, TOP - 0.04, "$\\bar{\\rho}_{\\mathrm{GT}}$", ha="right", va="bottom", fontsize=6.2, color=MUTED)

    y = TOP
    for d, info, sub in blocks:
        y += SET_GAP
        ax.add_patch(Rectangle((0.005, y), TEXT_WIDTH - 0.01, BAND, facecolor=TINT[d], edgecolor="none"))
        ax.plot([0.005, TEXT_WIDTH - 0.005], [y + BAND, y + BAND], color=STRONG[d], lw=1.0, solid_capstyle="butt")
        ax.text(0.04, y + BAND / 2, f"{d}  ·  {len(info)} calibrated species", ha="left", va="center", fontsize=7,
                fontweight="bold", color=INK)
        y += BAND
        values = sub.pivot(index="scientific_name", columns="model", values="density_r2").loc[info["scientific_name"], MODELS]
        for r, row in zip(info.itertuples(), values.to_numpy()):
            ax.add_patch(Rectangle((0.02, y + 0.008), 0.22, ROW - 0.016, facecolor=OOD[r.ood_level], edgecolor="none"))
            ax.text(0.13, y + ROW / 2, r.ood_level, ha="center", va="center", fontsize=5.6, color=INK)
            ax.text(0.29, y + ROW / 2, r.scientific_name, ha="left", va="center", fontsize=6.2, fontstyle="italic", color=INK)
            ax.text(N_RIGHT, y + ROW / 2, f"{int(r.density_n):,}", ha="right", va="center", fontsize=5.8, color=MUTED)
            ax.text(RHO_RIGHT, y + ROW / 2, f"{r.density_gt_mean:.1f}", ha="right", va="center", fontsize=5.8, color=MUTED)
            for x, v in zip(xs, row):
                if np.isnan(v):
                    ax.add_patch(Rectangle((x + 0.004, y + 0.006), w - 0.008, ROW - 0.012, facecolor="white", edgecolor=NA,
                                           hatch="//////", lw=0))
                    ax.text(x + w / 2, y + ROW / 2, "–", ha="center", va="center", fontsize=5, color=MUTED)
                    continue
                face = CMAP(NORM(max(v, -1.0)))
                ax.add_patch(Rectangle((x + 0.004, y + 0.006), w - 0.008, ROW - 0.012, facecolor=face, edgecolor="none"))
                ax.text(x + w / 2, y + ROW / 2, text_for(v), ha="center", va="center", fontsize=4.9, color=ink_for(face))
            y += ROW

    cax = fig.add_axes([(xs[3]) / TEXT_WIDTH, 0.13 / height, (xs[-4] + w - xs[3]) / TEXT_WIDTH, 0.065 / height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", extend="min")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.2, length=1.5, width=0.4, pad=1)
    cb.set_ticks([-1, 0, 1], labels=["≤−1", "0", "1"])
    cax.set_title("Within-species density $R^2$: brown below the species-mean reference (0), teal above", fontsize=6.2,
                  color=MUTED, pad=2)
    save(fig, stem, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    verify_species_protocol(a.root)
    sp = species_values(a.root)
    assert set(sp["image_scope"]) == {"scene_primary"}
    bio = read(a.root, BIOLOGY)
    for key, protocol, threshold in SETTINGS:
        values, setting = at_setting(sp, bio, protocol, threshold)
        if key == "fixed025":
            assert np.allclose(values["density_r2"], sp["density_r2"], equal_nan=True)
        for letter, sets in PAGES:
            stem = f"fig_species_r2_heatmap_{letter}" if key == "fixed025" else f"fig_species_r2_heatmap_{key}_{letter}"
            page(sets, values, stem, setting, a.out)


if __name__ == "__main__":
    main()
