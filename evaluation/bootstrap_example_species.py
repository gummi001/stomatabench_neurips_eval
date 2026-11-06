"""Point values and paired image bootstrap for the three example species of Table S21.

Scene-primary, calibrated images of each species; fixed 0.25 threshold; matching at IoU 0.50.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODELS, REPLICATES, SEED, read_part, write_json

TARGETS = [
    ("tmawp_malus_domestica", "T-MAWP", "Malus domestica", None),
    ("tmswp_ulmus_americana", "T-MSWP", "Ulmus americana Planch", None),
    ("tgsh_wheat_gibbs", "T-GSH", "Triticum aestivum", "wheat_gibbs"),
]


def r2(y, p):
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((p - y) ** 2) / den) if den > 0 else np.nan


def corr(y, p):
    return float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan


def ccc(y, p):
    """Lin's concordance correlation coefficient."""
    ym, pm = y.mean(), p.mean()
    den = np.mean((y - ym) ** 2) + np.mean((p - pm) ** 2) + (ym - pm) ** 2
    return float(2 * np.mean((y - ym) * (p - pm)) / den) if den > 0 else np.nan


def calculate(gt, pred, area, tp, fp, fn):
    gd, pdens = gt / area, pred / area
    stp, sfp, sfn = tp.sum(), fp.sum(), fn.sum()
    pr = corr(gd, pdens)
    return {
        "TP": float(stp), "FP": float(sfp), "FN": float(sfn),
        "precision": stp / (stp + sfp) if stp + sfp else np.nan,
        "recall": stp / (stp + sfn) if stp + sfn else np.nan,
        "F1": 2 * stp / (2 * stp + sfp + sfn) if 2 * stp + sfp + sfn else np.nan,
        "count_MAE": float(np.mean(np.abs(pred - gt))),
        "count_predictive_R2": r2(gt, pred),
        "density_MAE": float(np.mean(np.abs(pdens - gd))),
        "density_predictive_R2": r2(gd, pdens),
        "density_pearson_r": pr,
        "density_pearson_r2": pr * pr if np.isfinite(pr) else np.nan,
        "density_CCC": ccc(gd, pdens),
        "density_signed_bias": float(np.mean(pdens - gd)),
    }


def subset(df, species, source, density):
    keep = df["is_scene_primary"] & df["scientific_name"].eq(species)
    if source is not None:
        keep &= df["source_subset"].eq(source)
    if density:
        keep &= df["has_density"]
    return df.loc[keep].copy().sort_values("coco_image_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count-density", type=Path, required=True, help="output folder of evaluate_count_density.py")
    ap.add_argument("--detection", type=Path, required=True, help="output folder of evaluate_detection.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cd_parts, det_parts = a.count_density / "per_model_dataset_parts", a.detection / "per_model_dataset_parts"

    rng = np.random.default_rng(a.seed)
    points, boots, targets = [], [], []
    for target_id, dataset, species, source in TARGETS:
        ref = subset(read_part(cd_parts / f"{dataset}__{MODELS[0]}.csv"), species, source, True)
        if ref.empty:
            raise ValueError(f"no rows for {target_id}")
        ids = ref["coco_image_id"].astype(int).to_numpy()
        n = len(ids)
        draws = rng.integers(0, n, size=(a.replicates, n))
        base = {"target_id": target_id, "dataset": dataset, "scientific_name": species, "source_subset": source}
        targets.append({**base, "n_images": n, "gt_boxes": int(ref["gt_count"].sum()),
                        "ood_level": ref["ood_level"].iloc[0],
                        "mean_gt_density": float(ref["gt_density_per_mm2"].mean())})
        for model in MODELS:
            cd = subset(read_part(cd_parts / f"{dataset}__{model}.csv"), species, source, True)
            det = subset(read_part(det_parts / f"{dataset}__{model}.csv"), species, source, False)
            if not np.array_equal(cd["coco_image_id"].astype(int), ids) or not np.array_equal(det["coco_image_id"].astype(int), ids):
                raise ValueError(f"image mismatch: {target_id}/{model}")
            gt = cd["gt_count"].to_numpy(float)
            pred = cd["fixed_0p25"].to_numpy(float)
            area = cd["image_area_mm2"].to_numpy(float)
            tp = det["tp__iou050__fixed_0p25"].to_numpy(float)
            fp = sum(det[c].to_numpy(float) for c in ["duplicate_fp__iou050__fixed_0p25",
                                                      "localization_fp__iou050__fixed_0p25",
                                                      "background_fp__iou050__fixed_0p25"])
            fn = gt - tp
            if not np.array_equal(pred, tp + fp):
                raise ValueError(f"kept boxes != TP + FP: {target_id}/{model}")
            point = calculate(gt, pred, area, tp, fp, fn)
            points.append({**base, "model": model, "n_images": n, **point})
            dist = {name: np.empty(a.replicates) for name in point}
            for b, idx in enumerate(draws):
                for name, value in calculate(gt[idx], pred[idx], area[idx], tp[idx], fp[idx], fn[idx]).items():
                    dist[name][b] = value
            for name, values in dist.items():
                v = values[np.isfinite(values)]
                boots.append({**base, "model": model, "n_images": n, "metric": name,
                              "point_estimate": point[name],
                              "bootstrap_mean": v.mean() if len(v) else np.nan,
                              "ci_lower_2p5": np.quantile(v, .025) if len(v) else np.nan,
                              "ci_upper_97p5": np.quantile(v, .975) if len(v) else np.nan,
                              "valid_replicates": len(v)})
        print(f"{target_id} done", flush=True)
    pd.DataFrame(targets).to_csv(a.out / "example_species_targets.csv", index=False)
    pd.DataFrame(points).to_csv(a.out / "example_species_point_metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(a.out / "example_species_bootstrap_metrics.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "replicates": a.replicates, "seed": a.seed, "image_scope": "scene_primary",
        "confidence_protocol": "fixed 0.25", "matching_iou": 0.50,
        "bootstrap_unit": "image within each target, paired across models",
    })


if __name__ == "__main__":
    main()
