"""TP, FP, FN, precision, recall and F1 under six confidence protocols, at IoU 0.30 and 0.50.

Boxes are taken in their released order (descending score within each image) and
matched greedily one-to-one to unmatched ground truth. Every unmatched box is a
false positive of one kind: duplicate (overlaps an already matched stoma at the
matching IoU), localization (best IoU at least 0.10) or background.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DATASETS, MODELS, iter_json_array, load_master, load_thresholds,
                    protocol_columns, read_part, sha256, write_json)

IOU_THRESHOLDS = (0.30, 0.50)
LOCALIZATION_FLOOR = 0.10
PART_COLUMNS = ["dataset", "model", "coco_image_id", "scene_id", "is_scene_primary",
                "scientific_name", "source_subset", "calibration_group", "gt_count"]
GROUPS = {
    "dataset": [],
    "species": ["scientific_name"],
    "species_acquisition": ["scientific_name", "source_subset", "calibration_group"],
}


def iou_key(iou):
    return f"iou{int(round(iou * 100)):03d}"


def pairwise_iou(pred, gt):
    """IoU matrix between xywh boxes."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p, g = pred.astype(np.float64, copy=True), gt.astype(np.float64, copy=True)
    p[:, 2:] = p[:, :2] + np.maximum(0.0, p[:, 2:])
    g[:, 2:] = g[:, :2] + np.maximum(0.0, g[:, 2:])
    left = np.maximum(p[:, None, 0], g[None, :, 0])
    top = np.maximum(p[:, None, 1], g[None, :, 1])
    right = np.minimum(p[:, None, 2], g[None, :, 2])
    bottom = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_p = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 1])
    area_g = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def cumulative_matches(iou, threshold):
    """Cumulative TP and the three FP kinds after each box, in score order."""
    n_pred, n_gt = iou.shape
    tp, dup, loc, bg = (np.zeros(n_pred + 1, dtype=np.int32) for _ in range(4))
    matched = np.zeros(n_gt, dtype=bool)
    for i in range(n_pred):
        tp[i + 1], dup[i + 1], loc[i + 1], bg[i + 1] = tp[i], dup[i], loc[i], bg[i]
        if n_gt == 0:
            bg[i + 1] += 1
            continue
        free = np.flatnonzero(~matched)
        if len(free):
            j = int(np.argmax(iou[i, free]))
            if float(iou[i, free[j]]) >= threshold:
                matched[int(free[j])] = True
                tp[i + 1] += 1
                continue
        best = float(np.max(iou[i]))
        if best >= threshold:
            dup[i + 1] += 1
        elif best >= LOCALIZATION_FLOOR:
            loc[i + 1] += 1
        else:
            bg[i + 1] += 1
    return tp, dup, loc, bg


def load_gt(path):
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {int(im["id"]): Path(str(im["file_name"])).name for im in coco["images"]}
    boxes = {i: [] for i in names}
    for ann in coco["annotations"]:
        if int(ann.get("category_id", 1)) == 1:
            boxes[int(ann["image_id"])].append([float(v) for v in ann["bbox"]])
    return {i: np.asarray(b, dtype=np.float64).reshape((-1, 4)) for i, b in boxes.items()}, names


def kept(scores, threshold):
    return len(scores) if not math.isfinite(threshold) else int(np.count_nonzero(scores >= threshold))


def build_part(pred_path, part_path, meta, gt_boxes, model, model_threshold):
    """Write per-image kept boxes, TP and FP kinds for every protocol and IoU."""
    protocols = protocol_columns(model_threshold)
    empty = {}
    for key, _, _ in protocols:
        empty[f"selected__{key}"] = 0
        for match_iou in IOU_THRESHOLDS:
            for kind in ("tp", "duplicate_fp", "localization_fp", "background_fp"):
                empty[f"{kind}__{iou_key(match_iou)}__{key}"] = 0
    ids = set(meta["coco_image_id"].astype(int))
    results = {i: {} for i in ids}

    def score_image(image_id, scores, boxes):
        scores = np.asarray(scores, dtype=np.float64)
        iou = pairwise_iou(np.asarray(boxes, dtype=np.float64).reshape((-1, 4)), gt_boxes[image_id])
        selected = {key: kept(scores, thr) for key, _, thr in protocols}
        out = {f"selected__{key}": n for key, n in selected.items()}
        for match_iou in IOU_THRESHOLDS:
            tp, dup, loc, bg = cumulative_matches(iou, match_iou)
            k = iou_key(match_iou)
            for key, n in selected.items():
                out[f"tp__{k}__{key}"] = int(tp[n])
                out[f"duplicate_fp__{k}__{key}"] = int(dup[n])
                out[f"localization_fp__{k}__{key}"] = int(loc[n])
                out[f"background_fp__{k}__{key}"] = int(bg[n])
        results[image_id] = out

    current, scores, boxes = None, [], []
    prev_id, prev_score, records = -1, math.inf, 0
    for rec in iter_json_array(pred_path):
        if int(rec.get("category_id", 1)) != 1:
            continue
        records += 1
        image_id, score = int(rec["image_id"]), float(rec["score"])
        if image_id not in ids:
            raise ValueError(f"{pred_path}: unknown image id {image_id}")
        if image_id < prev_id or (image_id == prev_id and score > prev_score + 1e-12):
            raise ValueError(f"{pred_path}: boxes must be sorted by image id, then by descending score")
        if current is not None and image_id != current:
            score_image(current, scores, boxes)
            scores, boxes = [], []
        current = image_id
        scores.append(score)
        boxes.append([float(v) for v in rec["bbox"]])
        prev_id, prev_score = image_id, score
    if current is not None:
        score_image(current, scores, boxes)
    for i in ids:
        if not results[i]:
            results[i] = dict(empty)

    with open(part_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PART_COLUMNS + list(empty))
        writer.writeheader()
        for row in meta.itertuples(index=False):
            image_id = int(row.coco_image_id)
            out = {"dataset": row.dataset, "model": model, "coco_image_id": image_id,
                   "scene_id": row.scene_id, "is_scene_primary": bool(row.is_scene_primary),
                   "scientific_name": row.scientific_name, "source_subset": row.source_subset,
                   "calibration_group": row.calibration_group, "gt_count": int(row.gt_stomata_count)}
            out.update(results[image_id])
            writer.writerow(out)
    return {"dataset": str(meta["dataset"].iloc[0]), "model": model, "images": len(meta),
            "prediction_records": records, "prediction_sha256": sha256(pred_path),
            "model_specific_threshold": model_threshold}


def aggregate(parts, thresholds, out):
    rows = []
    for part in parts:
        df = read_part(part)
        model, dataset = str(df["model"].iloc[0]), str(df["dataset"].iloc[0])
        protocols = protocol_columns(thresholds[model])
        for scope, scoped in (("all", df), ("scene_primary", df[df["is_scene_primary"]])):
            for level, cols in GROUPS.items():
                groups = [((), scoped)] if not cols else scoped.groupby(cols[0] if len(cols) == 1 else cols,
                                                                         sort=True, dropna=False)
                for gkey, g in groups:
                    gkey = gkey if isinstance(gkey, tuple) else (gkey,)
                    labels = dict(zip(cols, gkey))
                    gt = int(g["gt_count"].sum())
                    for key, protocol, thr in protocols:
                        selected = int(g[f"selected__{key}"].sum())
                        for match_iou in IOU_THRESHOLDS:
                            k = iou_key(match_iou)
                            tp = int(g[f"tp__{k}__{key}"].sum())
                            dup = int(g[f"duplicate_fp__{k}__{key}"].sum())
                            loc = int(g[f"localization_fp__{k}__{key}"].sum())
                            bg = int(g[f"background_fp__{k}__{key}"].sum())
                            fp, fn = selected - tp, gt - tp
                            if fp != dup + loc + bg:
                                raise ValueError(f"FP kinds do not add up: {dataset}/{model}/{scope}/{level}/{key}/{match_iou}")
                            nan = float("nan")
                            rows.append({
                                "dataset": dataset, "model": model, "image_scope": scope, "group_level": level,
                                "scientific_name": labels.get("scientific_name", ""),
                                "source_subset": labels.get("source_subset", ""),
                                "calibration_group": labels.get("calibration_group", ""),
                                "confidence_protocol": protocol, "threshold": thr, "matching_iou": match_iou,
                                "n_images": len(g), "gt_boxes": gt, "selected_predictions": selected,
                                "tp": tp, "fn": fn, "fp": fp,
                                "duplicate_fp": dup, "localization_fp": loc, "background_fp": bg,
                                "precision": tp / (tp + fp) if tp + fp else nan,
                                "recall": tp / (tp + fn) if tp + fn else nan,
                                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan,
                                "small_n_flag": "descriptive_only_n_lt_5" if len(g) < 5 else "",
                            })
    pd.DataFrame(rows).to_csv(out / "operating_point_metrics_long.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True, help="selected_thresholds.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    (a.out / "per_model_dataset_parts").mkdir(parents=True, exist_ok=True)
    master = load_master(a.master)
    thresholds = load_thresholds(a.thresholds)

    parts, audit = [], []
    for dataset in DATASETS:
        meta = master[master["dataset"].eq(dataset)]
        gt_boxes, names = load_gt(a.data / "datasets" / dataset / "coco.json")
        if set(meta["coco_image_id"].astype(int)) != set(gt_boxes):
            raise ValueError(f"{dataset}: master and COCO image ids differ")
        by_id = meta.set_index("coco_image_id")
        for image_id, name in names.items():
            if str(by_id.loc[image_id, "basename"]) != name:
                raise ValueError(f"{dataset}: basename mismatch at image {image_id}")
            if int(by_id.loc[image_id, "gt_stomata_count"]) != len(gt_boxes[image_id]):
                raise ValueError(f"{dataset}: GT count mismatch at image {image_id}")
        for model in MODELS:
            part = a.out / "per_model_dataset_parts" / f"{dataset}__{model}.csv"
            audit.append(build_part(pred_dir / model / f"{dataset}.json", part, meta,
                                    gt_boxes, model, thresholds[model]))
            parts.append(part)
            print(f"{dataset:8s} {model:17s} {audit[-1]['prediction_records']:,} boxes", flush=True)
    aggregate(parts, thresholds, a.out)
    pd.DataFrame(audit).to_csv(a.out / "prediction_audit.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "master_sha256": sha256(a.master),
        "thresholds_sha256": sha256(a.thresholds),
        "matching": "greedy one-to-one in descending score order, no cap on boxes",
        "iou_thresholds": list(IOU_THRESHOLDS),
        "localization_floor": LOCALIZATION_FLOOR,
    })


if __name__ == "__main__":
    main()
