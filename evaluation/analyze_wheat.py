"""Density errors and detection counts on the calibrated Gibbs wheat images of T-GSH.

Scene-primary images with a physical scale, under each of the six confidence protocols.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODELS, read_part, write_json

FAMILIES = {
    **{m: "two-stage CNN" for m in MODELS[:4]},
    **{m: "one-stage CNN" for m in MODELS[4:7]},
    **{m: "YOLO" for m in MODELS[7:11]},
    **{m: "real-time transformer" for m in MODELS[11:13]},
    "GDINO-SB": "foundation-derived", "SAM3-OD": "foundation-derived",
}
PROTOCOLS = {
    "Available support": "available_support", "Shared 0.254": "shared_0p254",
    "Model-specific": "model_specific", "Fixed 0.25": "fixed_0p25",
    "Fixed 0.50": "fixed_0p50", "Fixed 0.75": "fixed_0p75",
}


def r2(y, p):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count-density", type=Path, required=True, help="output folder of evaluate_count_density.py")
    ap.add_argument("--detection", type=Path, required=True, help="output folder of evaluate_detection.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        df = read_part(a.count_density / "per_model_dataset_parts" / f"T-GSH__{model}.csv")
        wheat = df[df["is_scene_primary"] & df["scientific_name"].eq("Triticum aestivum")
                   & df["source_subset"].eq("wheat_gibbs") & df["has_density"]]
        gt_count = wheat["gt_count"].to_numpy(float)
        area = wheat["image_area_mm2"].to_numpy(float)
        gt = gt_count / area
        for protocol, column in PROTOCOLS.items():
            pred_count = wheat[column].to_numpy(float)
            pred = pred_count / area
            error = pred - gt
            mse, bias = float(np.mean(error ** 2)), float(np.mean(error))
            r = float(np.corrcoef(gt, pred)[0, 1]) if np.std(pred) > 0 else np.nan
            rows.append({
                "model": model, "architecture_group": FAMILIES[model], "protocol": protocol,
                "n_images": len(wheat), "gt_count_total": int(gt_count.sum()),
                "pred_count_total": int(pred_count.sum()),
                "pred_count_difference_percent": 100 * (pred_count.sum() / gt_count.sum() - 1),
                "zero_prediction_images": int(np.count_nonzero(pred_count == 0)),
                "zero_prediction_fraction": float(np.mean(pred_count == 0)),
                "underprediction_fraction": float(np.mean(pred < gt)),
                "gt_density_mean": float(gt.mean()), "gt_density_sd": float(gt.std()),
                "gt_density_cv": float(gt.std() / gt.mean()),
                "pred_density_mean": float(pred.mean()), "pred_density_sd": float(pred.std()),
                "pred_density_cv": float(pred.std() / pred.mean()) if pred.mean() else np.nan,
                "pred_density_max": float(pred.max()),
                "density_mae": float(np.mean(np.abs(error))), "density_signed_bias": bias,
                "density_mse": mse, "density_r2": r2(gt, pred),
                "density_pearson_r": r,
                "affine_in_sample_r2_ceiling": r ** 2 if np.isfinite(r) else np.nan,
                "bias_fraction_of_mse": bias * bias / mse,
                "scatter_fraction_of_mse": 1 - bias * bias / mse,
                "failure_component": "bias-dominated" if bias * bias / mse > .5 else "scatter-dominated",
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(a.out / "wheat_model_protocol_metrics.csv", index=False)

    op = pd.read_csv(a.detection / "operating_point_metrics_long.csv", low_memory=False)
    op = op[op["dataset"].eq("T-GSH") & op["image_scope"].eq("scene_primary")
            & op["group_level"].eq("species_acquisition") & op["scientific_name"].eq("Triticum aestivum")
            & op["source_subset"].eq("wheat_gibbs")].copy()
    op.insert(2, "architecture_group", op["model"].map(FAMILIES))
    op.to_csv(a.out / "wheat_operating_point_metrics.csv", index=False)

    ms = metrics[metrics["protocol"].eq("Model-specific")]
    write_json(a.out / "run_metadata.json", {
        "n_scene_primary_images": int(ms["n_images"].iloc[0]),
        "gt_boxes": int(ms["gt_count_total"].iloc[0]),
        "negative_predictive_r2_model_protocol_cells": int((metrics["density_r2"] < 0).sum()),
        "model_protocol_cells": len(metrics),
        "model_specific_max_affine_in_sample_r2_percent": 100 * float(ms["affine_in_sample_r2_ceiling"].max()),
    })


if __name__ == "__main__":
    main()
