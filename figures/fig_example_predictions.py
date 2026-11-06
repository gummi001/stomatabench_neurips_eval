"""Figures S59-S61: one image per evaluation set with its annotations and every detector's predictions at the fixed
0.25 threshold (true positives, false positives, missed stomata), five detectors per figure.

The image of a set is the scene-primary image whose annotated count is closest to the set's median (T-GSH: calibrated
Gibbs wheat). Matching is the evaluation's one-to-one matching at IoU 0.50; the counts under a tile are for the whole
image. Needs coco.json files, images and predictions (python scripts/download.py eval, then images).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from data import PER_IMAGE, read
from style import DATASETS, INK, MODELS, MUTED, OOD, STRONG, TINT, label, save

THRESHOLD, IOU = 0.25, 0.50
CROP = 0.5  # side of the central square, as a fraction of the shorter image side
PER_FIGURE = 5
TILE, LEFT, SIZE = 0.56, 0.66, 0.85  # tile and row-label width (inches) and text scale: two figures fit on a page
# Okabe-Ito colours: true positive, false positive, missed stoma; annotations in the first column.
TP_C, FP_C, FN_C, GT_C = "#009E73", "#D55E00", "#56B4E9", "#F0E442"


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    w = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    h = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = w * h
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


def match(preds, gts):
    """(true positives, false positives, missed stomata) as box lists, with the evaluation's one-to-one matching."""
    free, tp, fp = list(range(len(gts))), [], []
    for p in sorted(preds, key=lambda b: -b["score"]):
        best, best_iou = None, 0.0
        for g in free:
            v = iou(p["bbox"], gts[g])
            if v > best_iou:
                best, best_iou = g, v
        if best is not None and best_iou >= IOU:
            tp.append(p["bbox"])
            free.remove(best)
        else:
            fp.append(p["bbox"])
    return tp, fp, [gts[g] for g in free]


def choose(per_image):
    """The image of each set: annotated count closest to the set's median (T-GSH: calibrated Gibbs wheat)."""
    rows = []
    first = per_image[per_image["model"].eq(MODELS[0]) & per_image["is_scene_primary"]]
    for d in DATASETS:
        pool = first[first["dataset"].eq(d)]
        if d == "T-GSH":
            pool = pool[pool["source_subset"].eq("wheat_gibbs") & pool["has_density"]]
        median = pool["gt_count"].median()
        pick = pool.assign(gap=(pool["gt_count"] - median).abs()).sort_values(["gap", "coco_image_id"]).iloc[0]
        rows.append(dict(dataset=d, coco_image_id=int(pick["coco_image_id"]), basename=pick["basename"],
                         scientific_name=pick["scientific_name"], ood_level=pick["ood_level"], gt_count=int(pick["gt_count"]),
                         pool=len(pool), pool_median=median))
    return pd.DataFrame(rows)


def prepare(choice, data):
    """Per chosen image: the crop (a square centred on the annotated stomata, inside the image) and the annotated boxes."""
    rows = []
    for ch in choice.itertuples():
        folder = data / "datasets" / ch.dataset
        coco = json.loads((folder / "coco.json").read_text(encoding="utf-8"))
        img = next(i for i in coco["images"] if i["id"] == ch.coco_image_id)
        assert img["file_name"].endswith(ch.basename), (img["file_name"], ch.basename)
        gts = [a["bbox"] for a in coco["annotations"] if a["image_id"] == ch.coco_image_id]
        assert len(gts) == ch.gt_count, ch.dataset
        path = next(q for q in (folder / "images" / ch.basename, folder / "images_2" / ch.basename) if q.exists())
        image = Image.open(path).convert("RGB")
        w, h = image.size
        side = CROP * min(w, h)
        cx = np.median([bx + bw / 2 for bx, _, bw, _ in gts])
        cy = np.median([by + bh / 2 for _, by, _, bh in gts])
        x0, y0 = min(max(cx - side / 2, 0), w - side), min(max(cy - side / 2, 0), h - side)
        crop = image.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((360, 360))
        rows.append(dict(ch=ch, gts=gts, crop=crop, x0=x0, y0=y0, side=side))
    return rows


def predictions(per_image, model, ch, data):
    """The model's boxes with score >= 0.25 on the chosen image, checked against the per-image count."""
    preds = json.loads((data / "predictions" / model / f"{ch.dataset}.json").read_text(encoding="utf-8"))
    kept = [b for b in preds if b["image_id"] == ch.coco_image_id and b["score"] >= THRESHOLD]
    release = per_image[per_image["dataset"].eq(ch.dataset) & per_image["model"].eq(model)
                        & per_image["coco_image_id"].eq(ch.coco_image_id)]
    assert len(release) == 1 and int(release["fixed_0p25"].iloc[0]) == len(kept), (ch.dataset, model)
    return kept


def draw(per_image, rows, models, stem, data, out):
    """One row per set and one column per model after the annotations."""
    n_rows, n_cols = len(rows), 1 + len(models)
    gap, top, text_h = 0.04, 0.30 * SIZE, 0.13 * SIZE
    bottom = 0.02
    fig_w = LEFT + n_cols * TILE + gap * (n_cols - 1)
    height = top + n_rows * (TILE + text_h + gap) + bottom
    fig = plt.figure(figsize=(fig_w, height))
    for c, name in enumerate(["Annotation"] + [label(m) for m in models]):
        fig.text((LEFT + c * (TILE + gap) + TILE / 2) / fig_w, 1 - 0.17 * SIZE / height, name, ha="center", va="center",
                 fontsize=6.8 * SIZE, fontweight="bold", color=INK)
    checked = []
    for r, row in enumerate(rows):
        ch, gts, x0, y0, side = row["ch"], row["gts"], row["x0"], row["y0"], row["side"]
        y_top = 1 - (top + r * (TILE + text_h + gap)) / height
        fig.add_artist(Rectangle((0.002, y_top - TILE / height), (LEFT - 0.06) / fig_w, TILE / height,
                                 transform=fig.transFigure, facecolor=TINT[ch.dataset], edgecolor="none"))
        fig.add_artist(Line2D([0.002, 0.002], [y_top - TILE / height, y_top], transform=fig.transFigure,
                              color=STRONG[ch.dataset], lw=2))
        fig.text(0.035 / fig_w, y_top - 0.12 * SIZE / height, ch.dataset, ha="left", va="top", fontsize=7.2 * SIZE,
                 fontweight="bold", color=INK)
        fig.add_artist(Rectangle((0.035 / fig_w, y_top - 0.40 * SIZE / height), 0.2 * SIZE / fig_w, 0.13 * SIZE / height,
                                 transform=fig.transFigure, facecolor=OOD[ch.ood_level], edgecolor="none"))
        fig.text((0.035 + 0.1 * SIZE) / fig_w, y_top - 0.335 * SIZE / height, ch.ood_level, ha="center", va="center",
                 fontsize=6 * SIZE, color=INK)
        fig.text(0.035 / fig_w, y_top - 0.48 * SIZE / height, ch.scientific_name.replace(" ", "\n", 1), ha="left", va="top",
                 fontsize=6 * SIZE, fontstyle="italic", color=INK, linespacing=1.0)
        for c in range(n_cols):
            ax = fig.add_axes([(LEFT + c * (TILE + gap)) / fig_w, y_top - TILE / height, TILE / fig_w, TILE / height])
            ax.imshow(row["crop"], extent=(x0, x0 + side, y0 + side, y0))
            ax.set_xlim(x0, x0 + side)
            ax.set_ylim(y0 + side, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color(MUTED)
            if c == 0:
                layers, note = [(gts, GT_C, "-")], f"{len(gts)} stomata"
            else:
                m = models[c - 1]
                tp, fp, fn = match(predictions(per_image, m, ch, data), gts)
                checked.append(dict(dataset=ch.dataset, model=m, tp=len(tp), fp=len(fp), fn=len(fn)))
                layers = [(fn, FN_C, (0, (2, 1.2))), (fp, FP_C, "-"), (tp, TP_C, "-")]
                note = f"TP {len(tp)}  FP {len(fp)}  FN {len(fn)}"
            for boxes, colour, style in layers:
                for bx, by, bw, bh in boxes:
                    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor=colour, lw=0.7 * SIZE, linestyle=style))
            fig.text((LEFT + c * (TILE + gap) + TILE / 2) / fig_w, y_top - (TILE + 0.075 * SIZE) / height, note, ha="center",
                     va="center", fontsize=5.6 * SIZE, color=INK)
    save(fig, stem, out)
    plt.close(fig)
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--data", type=Path, default=Path("data/StomataBench"),
                    help="dataset folder with datasets/<set>/coco.json, images and predictions/")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    per_image = read(a.root, PER_IMAGE)
    choice = choose(per_image)
    rows = prepare(choice, a.data)
    checked = []
    for i in range(0, len(MODELS), PER_FIGURE):
        checked += draw(per_image, rows, MODELS[i:i + PER_FIGURE], f"fig_qualitative_{i // PER_FIGURE + 1}", a.data, a.out)
    choice.to_csv(a.out / "example_predictions_choice.csv", index=False)
    print(choice.to_string(index=False))
    print(pd.DataFrame(checked).to_string(index=False))


if __name__ == "__main__":
    main()
