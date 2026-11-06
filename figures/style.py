"""Shared style of the result figures: colours, fonts, model rows and save().

Colours are the table colours of the paper's tex and the strong dataset colours of Figure 1.
Matplotlib settings follow figures4papers (Chen Liu, github.com/ChenLiu-1996/figures4papers, CC BY-NC 4.0).
Figures are drawn at the NeurIPS text width, so the font sizes here are the printed sizes.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import common

MODELS, DATASETS = list(common.MODELS), list(common.DATASETS)
FAMILIES = [("Two-stage convolutional detectors", 4), ("One-stage convolutional detectors", 3),
            ("YOLO-family detectors", 4), ("Transformer detectors", 2), ("Foundation-model-derived detectors", 2)]
TEXT_WIDTH = 5.5  # inches, NeurIPS \textwidth

# The twelve \definecolor colours of the tex: set headers (traingray, tmwpblue, tdcwpyellow, tmswporange, tgshred;
# OD-Valid takes the OD-Train grey), family rows (archgray) and OOD levels (oodL0 to oodL5).
TINT = {"OD-Valid": "#F2F2F2", "T-MAWP": "#EDF5FF", "T-DCWP": "#FFF9DB", "T-MSWP": "#FFF0E0", "T-GSH": "#FDECEC"}
FAMILY_BAND = "#F4F4F4"
OOD = {"L0": "#EAF2FF", "L1": "#F1F8E9", "L2": "#FFF8D6", "L3": "#FFE8CC", "L4": "#FDE0DD", "L5": "#E8DAEF"}
# Figure 1 tile borders.
STRONG = {"OD-Valid": "#7F7F7F", "T-MAWP": "#4A90D9", "T-DCWP": "#C9A227", "T-MSWP": "#E08A3C", "T-GSH": "#C0392B"}
INK, MUTED, GRID = "#222222", "#5E5E5E", "#E4E4E4"
# Value colours: viridis for magnitudes, BrBG for density R2 (teal above 0, brown below).
SEQUENTIAL, DIVERGING = "viridis", "BrBG"
# The second metric beside a viridis one: light yellow to dark brown, starting above white.
WARM = ListedColormap(plt.get_cmap("YlOrBr")(np.linspace(0.12, 1, 256)), name="warm")
NA = "#BDBDBD"  # hatch colour of cells with no value

# Times (NimbusRomNo9L in the NeurIPS style); Times New Roman has the same metrics.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # TrueType: NeurIPS rejects Type 3 fonts
    "svg.fonttype": "none",
})


def model_rows():
    """({model: y}, [(family, y)]): a header row for each family, then its models, top to bottom in table order."""
    y, rows, headers = 0, {}, []
    models = iter(MODELS)
    for family, n in FAMILIES:
        headers.append((family, y))
        y += 1
        for _ in range(n):
            rows[next(models)] = y
            y += 1
    return rows, headers


def label(model):
    """Display name as in the tables (1x -> 1×), in matplotlib mathtext."""
    return re.sub(r"-(\d)x$", r"-\1$\\times$", model)


def family_bands(fig, ax, headers, fontsize=7.5):
    """Family rows: a grey band across the figure with the family name at the left (y in ax's data)."""
    band = blended_transform_factory(fig.transFigure, ax.transData)
    for family, y in headers:
        fig.add_artist(Rectangle((0.005, y - 0.5), 0.985, 1, transform=band, facecolor=FAMILY_BAND, edgecolor="none", zorder=-1))
        fig.text(0.01, y, family, transform=band, ha="left", va="center", fontsize=fontsize, fontweight="bold",
                 fontstyle="italic", color=INK)


def set_header(ax, d, subtitle=None, letter=None):
    """Panel header in the set's table colour, with its Figure 1 colour underneath, and the panel letter at its left."""
    h = 0.115 if subtitle else 0.075
    ax.add_patch(Rectangle((0, 1.012), 1, h, transform=ax.transAxes, facecolor=TINT[d], edgecolor="none", clip_on=False))
    ax.plot([0, 1], [1.012, 1.012], transform=ax.transAxes, color=STRONG[d], lw=1.4, clip_on=False, solid_capstyle="butt")
    ax.text(0.5, 1.085 if subtitle else 1.05, d, transform=ax.transAxes, ha="center", va="center", fontweight="bold",
            fontsize=8, color=INK)
    if subtitle:
        ax.text(0.5, 1.038, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=7, color=MUTED)
    if letter:
        ax.text(0.03, 1.05 + (0.02 if subtitle else 0), letter, transform=ax.transAxes, ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK)


def model_figure(height, top=0.80, bottom=0.10, wspace=0.17):
    """One panel per set, side by side, sharing model rows grouped by family. Returns fig, axes, rows."""
    rows, headers = model_rows()
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True, figsize=(TEXT_WIDTH, height),
                             gridspec_kw=dict(left=0.135, right=0.985, bottom=bottom, top=top, wspace=wspace))
    for ax in axes:
        ax.set_facecolor("none")
        for _, y in headers:
            ax.axhspan(y - 0.5, y + 0.5, color=FAMILY_BAND, lw=0, zorder=2.5)
        ax.set_ylim(len(rows) + len(headers) - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(rows.values()), [label(m) for m in rows])
    family_bands(fig, axes[0], headers)
    return fig, axes, rows


def save(fig, stem, folder):
    """Vector PDF for the paper and a 300 dpi PNG preview in folder."""
    folder = Path(folder)
    for ext, dpi in [("pdf", None), ("png", 300)]:
        fig.savefig(folder / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    w, h = plt.imread(folder / f"{stem}.png").shape[1::-1]
    print(f"wrote {folder / stem}.pdf and .png ({w / 300:.2f} x {h / 300:.2f} in; text width is {TEXT_WIDTH} in)")
