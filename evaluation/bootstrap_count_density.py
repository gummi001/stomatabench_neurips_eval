"""Paired image bootstrap of count and density MAE and R2 at the fixed 0.25 threshold.

Scene-primary images are resampled with replacement, the same draw for all 15
models. Intervals are the 2.5 and 97.5 percentiles over replicates with a defined value.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASETS, MODELS, REPLICATES, SEED, predictive_r2, read_part, stable_seed, write_json

COUNT_COLUMN = "fixed_0p25"
THRESHOLD = 0.25


def bootstrap_group(truth, predictions, replicates, seed, batch_size=50):
    """Bootstrap MAE and R2, arrays of shape replicates x models."""
    n_images, n_models = predictions.shape
    mae = np.empty((replicates, n_models))
    r2 = np.full((replicates, n_models), np.nan)
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < replicates:
        batch = min(batch_size, replicates - offset)
        idx = rng.integers(0, n_images, size=(batch, n_images), endpoint=False)
        t = truth[idx]
        sst = np.square(t - t.mean(axis=1, keepdims=True)).sum(axis=1)
        valid = sst > 0
        for m in range(n_models):
            error = predictions[idx, m] - t
            mae[offset:offset + batch, m] = np.mean(np.abs(error), axis=1)
            sse = np.square(error).sum(axis=1)
            r2[offset:offset + batch, m][valid] = 1.0 - sse[valid] / sst[valid]
        offset += batch
    return mae, r2


def interval(values):
    v = values[np.isfinite(values)]
    if not len(v):
        return {"bootstrap_mean": float("nan"), "ci_lower_2p5": float("nan"),
                "ci_upper_97p5": float("nan"), "valid_replicates": 0}
    return {"bootstrap_mean": float(np.mean(v)), "ci_lower_2p5": float(np.quantile(v, 0.025)),
            "ci_upper_97p5": float(np.quantile(v, 0.975)), "valid_replicates": int(len(v))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count-density", type=Path, required=True, help="output folder of evaluate_count_density.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        meta, preds = None, []
        for model in MODELS:
            df = read_part(a.count_density / "per_model_dataset_parts" / f"{dataset}__{model}.csv")
            df = df[df["is_scene_primary"]].sort_values("coco_image_id").reset_index(drop=True)
            if meta is None:
                meta = df[["coco_image_id", "scientific_name", "has_density",
                           "image_area_mm2", "gt_count", "gt_density_per_mm2"]].copy()
            elif not np.array_equal(df["coco_image_id"].to_numpy(), meta["coco_image_id"].to_numpy()):
                raise ValueError(f"{dataset}: image order differs for {model}")
            preds.append(df[COUNT_COLUMN].to_numpy(float))
        counts = np.column_stack(preds)

        groups = [("dataset", "", np.ones(len(meta), dtype=bool))]
        groups += [("species", str(s), meta["scientific_name"].eq(s).to_numpy())
                   for s in sorted(meta["scientific_name"].unique())]
        for level, species, mask in groups:
            endpoints = [("count", mask, meta.loc[mask, "gt_count"].to_numpy(float), counts[mask])]
            dmask = mask & meta["has_density"].to_numpy()
            if np.any(dmask):
                area = meta.loc[dmask, "image_area_mm2"].to_numpy(float)
                endpoints.append(("density", dmask, meta.loc[dmask, "gt_density_per_mm2"].to_numpy(float),
                                  counts[dmask] / area[:, None]))
            for endpoint, _, truth, pred in endpoints:
                boot_mae, boot_r2 = bootstrap_group(truth, pred, a.replicates,
                                                    stable_seed(a.seed, dataset, level, species, endpoint))
                for m, model in enumerate(MODELS):
                    base = {"dataset": dataset, "group_level": level, "scientific_name": species, "model": model,
                            "image_scope": "scene_primary", "confidence_protocol": "Fixed", "threshold": THRESHOLD,
                            "endpoint": endpoint, "n_images": len(truth),
                            "small_n_flag": "descriptive_only_n_lt_5" if len(truth) < 5 else "",
                            "replicates_requested": a.replicates}
                    rows.append({**base, "metric": "MAE",
                                 "point_estimate": float(np.mean(np.abs(pred[:, m] - truth))),
                                 **interval(boot_mae[:, m])})
                    rows.append({**base, "metric": "predictive_R2",
                                 "point_estimate": predictive_r2(truth, pred[:, m]),
                                 **interval(boot_r2[:, m])})
        print(f"{dataset} done", flush=True)

    pd.DataFrame(rows).to_csv(a.out / "bootstrap_biological_fixed025_scene_primary.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "replicates": a.replicates, "seed": a.seed,
        "resampling_unit": "scene-primary image, paired across models",
        "confidence_protocol": "Fixed", "threshold": THRESHOLD,
        "interval": "percentile 2.5% to 97.5% over finite replicates",
    })


if __name__ == "__main__":
    main()
