"""COCO bbox metrics per set and per species, on all images and on scene-primary images.

Every released box is scored (no confidence filter), with maxDets = [1, 10, 100].
COCOeval matches each image once; the set, scene-primary and species views are
then accumulated from the same per-image results.
"""
import argparse
import contextlib
import copy
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import DATASETS, MODELS, TEST_SETS, load_master, write_json

STAT_NAMES = [
    "AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def accumulate_subset(evaluator, eval_imgs, params, image_ids):
    """COCO stats for a subset of images, sliced from one evaluate() pass.

    pycocotools' accumulate(p) does not locate p.imgIds inside _paramsEval.imgIds,
    so the K x A x I evalImgs list is sliced by hand.
    """
    position = {image_id: i for i, image_id in enumerate(params.imgIds)}
    picked = [position[x] for x in sorted(image_ids)]
    n_cat = len(params.catIds if params.useCats else [-1])
    n_area, n_img = len(params.areaRng), len(params.imgIds)
    subset = []
    for k in range(n_cat):
        for r in range(n_area):
            offset = k * n_area * n_img + r * n_img
            subset.extend(eval_imgs[offset + i] for i in picked)
    p = copy.deepcopy(params)
    p.imgIds = sorted(image_ids)
    evaluator.evalImgs = subset
    evaluator._paramsEval = copy.deepcopy(p)
    evaluator.params = copy.deepcopy(p)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        evaluator.summarize()
    return {name: float(evaluator.stats[i] * 100) for i, name in enumerate(STAT_NAMES)}


def write_tables(rows, out):
    df = pd.DataFrame(rows)
    df.to_csv(out / "coco_metrics_long.csv", index=False)
    for subset in sorted(df.subset.unique()):
        sf = df[df.subset == subset]
        for metric in ["AP50_95", "AP50", "AP75", "AR100"]:
            table = sf.pivot(index="model", columns="dataset", values=metric).reindex(index=MODELS, columns=DATASETS)
            table["Heldout_mean"] = table[list(TEST_SETS)].mean(axis=1)
            table.to_csv(out / f"coco_{metric}_{subset}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    a.out.mkdir(parents=True, exist_ok=True)
    master = load_master(a.master)

    set_rows, species_rows = [], []
    for dataset in a.datasets:
        meta = master[master["dataset"].eq(dataset)]
        ids = {
            "all": sorted(meta["coco_image_id"].tolist()),
            "dedup": sorted(meta.loc[meta["is_scene_primary"], "coco_image_id"].tolist()),
        }
        species = {}
        for name, g in meta.groupby("scientific_name", sort=True):
            species[str(name)] = {
                "all": sorted(g["coco_image_id"].tolist()),
                "scene_primary": sorted(g.loc[g["is_scene_primary"], "coco_image_id"].tolist()),
                "family": str(g["family"].iloc[0]),
                "order": str(g["order"].iloc[0]),
                "ood_level": str(g["ood_level"].iloc[0]),
            }
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(a.data / "datasets" / dataset / "coco.json"))
        gt.dataset.setdefault("info", {})
        gt.dataset.setdefault("licenses", [])
        gt_per_image = Counter(int(x["image_id"]) for x in gt.dataset["annotations"] if int(x.get("category_id", 1)) == 1)

        for model in a.models:
            path = pred_dir / model / f"{dataset}.json"
            preds = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(preds, list):
                raise TypeError(f"{path}: expected a top-level list")
            unknown = {int(p["image_id"]) for p in preds} - set(ids["all"])
            if unknown:
                raise ValueError(f"{model}/{dataset}: {len(unknown)} prediction image ids absent from GT")
            scores = [float(p["score"]) for p in preds]
            if not all(math.isfinite(x) for x in scores):
                raise ValueError(f"{model}/{dataset}: non-finite scores")
            pred_per_image = Counter(int(p["image_id"]) for p in preds)

            with contextlib.redirect_stdout(io.StringIO()):
                dt = gt.loadRes(preds)
                evaluator = COCOeval(gt, dt, "bbox")
                evaluator.params.maxDets = [1, 10, 100]
                evaluator.evaluate()
            eval_imgs, params = evaluator.evalImgs, copy.deepcopy(evaluator._paramsEval)

            for subset in ("all", "dedup"):
                sub = ids[subset]
                set_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "subset": subset,
                    "num_images": len(sub),
                    "num_gt": sum(gt_per_image[x] for x in sub),
                    "num_predictions": sum(pred_per_image[x] for x in sub),
                    "num_images_with_predictions": sum(1 for x in sub if pred_per_image[x] > 0),
                    "released_score_min": min(scores) if scores else np.nan,
                    "released_score_max": max(scores) if scores else np.nan,
                    **accumulate_subset(evaluator, eval_imgs, params, sub),
                })
            for name, spec in species.items():
                for scope in ("all", "scene_primary"):
                    sub = spec[scope]
                    species_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "image_scope": scope,
                        "scientific_name": name,
                        "family": spec["family"],
                        "order": spec["order"],
                        "ood_level": spec["ood_level"],
                        "n_images": len(sub),
                        "gt_boxes": sum(gt_per_image[x] for x in sub),
                        "confidence_protocol": "Available support",
                        "threshold": np.nan,
                        "max_dets": 100,
                        **accumulate_subset(evaluator, eval_imgs, params, sub),
                        "small_n_flag": "descriptive_only_n_lt_5" if len(sub) < 5 else "",
                    })
            print(f"{dataset:8s} {model:17s} AP50:95 all={set_rows[-2]['AP50_95']:.3f} "
                  f"scene-primary={set_rows[-1]['AP50_95']:.3f}", flush=True)
            del preds, dt, evaluator, eval_imgs

    write_tables(set_rows, a.out)
    pd.DataFrame(species_rows).to_csv(a.out / "species_coco_metrics_long.csv", index=False)
    write_json(a.out / "run_metadata.json", {"max_dets": [1, 10, 100], "confidence_filter": None,
                                             "set_rows": len(set_rows), "species_rows": len(species_rows)})


if __name__ == "__main__":
    main()
