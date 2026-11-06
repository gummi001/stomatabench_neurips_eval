"""COCO AP and paired image bootstrap of AP for the example species and models of Table S21.

The bootstrap reweights the per-image COCOeval matches; at weight 1 it reproduces COCOeval.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import load_master

AP_REPLICATES = 1000
TARGETS = {
    "tmawp_malus_domestica": ("T-MAWP", "Malus domestica", None, 20260811),
    "tmswp_ulmus_americana": ("T-MSWP", "Ulmus americana Planch", None, 20260812),
    "tgsh_wheat_gibbs": ("T-GSH", "Triticum aestivum", "wheat_gibbs", 20260813),
}
TARGET_MODELS = {
    "tmawp_malus_domestica": ["YOLOv12", "YOLOv11", "FRCNN-R50-3x", "RT-DETRv2", "RT-DETRv4"],
    "tmswp_ulmus_americana": ["SAM3-OD", "RT-DETRv2", "GDINO-SB", "YOLOv11", "YOLOv8"],
    "tgsh_wheat_gibbs": ["GDINO-SB", "SAM3-OD", "FRCNN-X101", "Retina-R101-3x", "YOLOv26", "RT-DETRv2"],
}
METRICS = ["AP50_95", "AP50", "AP75"]


def ap_from_weights(records, weights):
    """COCO AP50:95, AP50 and AP75 when image i is counted weights[b, i] times."""
    n_boot = weights.shape[0]
    per_iou = np.full((n_boot, 10), np.nan)
    recall_thresholds = np.linspace(0, 1, 101)
    gt_by_image = np.asarray([np.count_nonzero(~np.asarray(r["gtIgnore"], bool)) for r in records])
    positive_gt = weights @ gt_by_image
    for t in range(10):
        scores, image_index, matched, ignored = [], [], [], []
        for i, r in enumerate(records):
            s = np.asarray(r["dtScores"], float)
            scores.append(s)
            image_index.append(np.full(len(s), i, dtype=np.int32))
            matched.append(np.asarray(r["dtMatches"])[t] > 0)
            ignored.append(np.asarray(r["dtIgnore"])[t].astype(bool))
        scores = np.concatenate(scores) if scores else np.empty(0)
        image_index = np.concatenate(image_index) if image_index else np.empty(0, dtype=np.int32)
        matched = np.concatenate(matched) if matched else np.empty(0, dtype=bool)
        ignored = np.concatenate(ignored) if ignored else np.empty(0, dtype=bool)
        order = np.argsort(-scores, kind="mergesort")
        image_index, matched, ignored = image_index[order], matched[order], ignored[order]
        for start in range(0, n_boot, 25):
            stop = min(start + 25, n_boot)
            w = weights[start:stop, image_index]
            tp = np.cumsum(w * (matched & ~ignored), axis=1)
            fp = np.cumsum(w * (~matched & ~ignored), axis=1)
            recall = tp / np.maximum(positive_gt[start:stop, None], 1)
            precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            for j in range(stop - start):
                pos = np.searchsorted(recall[j], recall_thresholds, side="left")
                ok = pos < precision.shape[1]
                sampled = np.zeros(101)
                sampled[ok] = precision[j, pos[ok]]
                per_iou[start + j, t] = sampled.mean()
    return np.column_stack([per_iou.mean(axis=1), per_iou[:, 0], per_iou[:, 5]]) * 100


def run_one(data, pred_dir, master, target, model, replicates):
    dataset, species, source, seed = TARGETS[target]
    mask = master["dataset"].eq(dataset) & master["scientific_name"].eq(species) & master["is_scene_primary"]
    if source is not None:
        mask &= master["source_subset"].eq(source)
    rows = master.loc[mask]
    ids = sorted(rows["coco_image_id"].astype(int).tolist())
    if not ids:
        raise ValueError(f"empty target {target}")
    wanted = set(ids)
    preds = json.loads((pred_dir / model / f"{dataset}.json").read_text(encoding="utf-8"))
    preds = [p for p in preds if int(p["image_id"]) in wanted]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(data / "datasets" / dataset / "coco.json"))
    gt.dataset.setdefault("info", {})
    gt.dataset.setdefault("licenses", [])
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(gt, gt.loadRes(preds), "bbox")
        evaluator.params.imgIds = ids
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    official = np.asarray(evaluator.stats[[0, 1, 2]]) * 100
    n = len(evaluator._paramsEval.imgIds)
    records = [evaluator.evalImgs[i] for i in range(n)]
    if any(r is None for r in records) or len(evaluator._paramsEval.areaRng) != 4:
        raise ValueError("unexpected COCOeval image/area layout")
    diff = float(np.max(np.abs(ap_from_weights(records, np.ones((1, n), dtype=np.int16))[0] - official)))
    if diff > 1e-8:
        raise ValueError(f"reweighted AP does not reproduce COCOeval: max difference {diff}")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    weights = np.zeros((replicates, n), dtype=np.int16)
    for b, d in enumerate(draws):
        weights[b] = np.bincount(d, minlength=n)
    boot = ap_from_weights(records, weights)
    base = {"target_id": target, "dataset": dataset, "scientific_name": species,
            "source_subset": source, "model": model, "n_images": n}
    point = {**dict(zip(METRICS, official)), **base, "gt_boxes": int(rows["gt_stomata_count"].sum()),
             "released_predictions": len(preds), "max_custom_official_abs_difference": diff}
    long = [{**base, "metric": m, "point_estimate": official[k],
             "bootstrap_mean": float(boot[:, k].mean()),
             "ci_lower_2p5": float(np.quantile(boot[:, k], .025)),
             "ci_upper_97p5": float(np.quantile(boot[:, k], .975)),
             "valid_replicates": replicates, "seed": seed} for k, m in enumerate(METRICS)]
    return point, long


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=AP_REPLICATES)
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    a.out.mkdir(parents=True, exist_ok=True)
    master = load_master(a.master)
    points, longs = [], []
    for target, models in TARGET_MODELS.items():
        for model in models:
            point, long = run_one(a.data, pred_dir, master, target, model, a.replicates)
            points.append(point)
            longs += long
            print(f"{target:22s} {model:15s} AP50:95={point['AP50_95']:.3f}", flush=True)
    pd.DataFrame(points).to_csv(a.out / "target_ap_point_metrics.csv", index=False)
    pd.DataFrame(longs).to_csv(a.out / "target_ap_bootstrap_metrics.csv", index=False)


if __name__ == "__main__":
    main()
