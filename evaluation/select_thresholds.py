"""Select each model's confidence threshold, and one shared threshold, on OD-Valid.

Objective: count MAE averaged over images within each OD-Valid species, then over
the five species. Grid 0.001 to 0.950; the lowest threshold wins a tie. A model's
threshold is chosen within its released score range (rounded outward to the grid).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODELS, THRESHOLD_GRID, iter_json_array, load_master, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    a.out.mkdir(parents=True, exist_ok=True)

    master = load_master(a.master)
    valid = master[master["dataset"].eq("OD-Valid")].sort_values("coco_image_id").reset_index(drop=True)
    if len(valid) != 138 or valid["scientific_name"].nunique() != 5:
        raise ValueError("OD-Valid must contain 138 images and five species")
    image_ids = valid["coco_image_id"].astype(int).tolist()
    position = {image_id: i for i, image_id in enumerate(image_ids)}
    truth = valid["gt_stomata_count"].to_numpy(int)
    species_idx = [np.flatnonzero(valid["scientific_name"].eq(s).to_numpy())
                   for s in sorted(valid["scientific_name"].unique())]
    grid = THRESHOLD_GRID
    objective_matrix = np.empty((len(grid), len(MODELS)))
    curve_rows, selected_rows = [], []

    for m, model in enumerate(MODELS):
        scores_by_image = [[] for _ in image_ids]
        lo, hi = float("inf"), -float("inf")
        for rec in iter_json_array(pred_dir / model / "OD-Valid.json"):
            if int(rec.get("category_id", 1)) != 1:
                continue
            image_id = int(rec["image_id"])
            if image_id not in position:
                raise ValueError(f"{model}: unknown OD-Valid image id {image_id}")
            score = float(rec["score"])
            scores_by_image[position[image_id]].append(score)
            lo, hi = min(lo, score), max(hi, score)
        scores = [np.asarray(v, dtype=np.float64) for v in scores_by_image]
        floor, ceiling = math.floor(lo * 1000.0) / 1000.0, math.ceil(hi * 1000.0) / 1000.0

        objectives = np.empty(len(grid))
        for t, thr in enumerate(grid):
            predicted = np.asarray([np.count_nonzero(s >= thr) for s in scores], dtype=int)
            error = np.abs(predicted - truth)
            objectives[t] = float(np.mean([float(np.mean(error[i])) for i in species_idx]))
            curve_rows.append({
                "model": model,
                "threshold": thr,
                "species_balanced_count_mae": objectives[t],
                "within_rounded_grid_score_support": bool(floor <= thr <= ceiling),
                "observed_score_min": lo,
                "observed_score_max": hi,
            })
        objective_matrix[:, m] = objectives

        supported = np.flatnonzero((grid >= floor) & (grid <= ceiling))
        if not len(supported):
            raise ValueError(f"{model}: no grid threshold within score support")
        best = float(np.min(objectives[supported]))
        best_i = int(supported[np.flatnonzero(np.isclose(objectives[supported], best, rtol=0, atol=1e-12))[0]])
        fixed_i = int(np.flatnonzero(np.isclose(grid, 0.25))[0])
        selected_rows.append({
            "model": model,
            "selected_threshold_species_balanced_mae": float(grid[best_i]),
            "best_species_balanced_count_mae": best,
            "tau0p25_species_balanced_count_mae": float(objectives[fixed_i]),
            "observed_score_min": lo,
            "observed_score_max": hi,
            "grid_support_floor": floor,
            "grid_support_ceiling": ceiling,
            "candidate_thresholds_within_support": int(len(supported)),
        })
        print(f"{model:17s} threshold={grid[best_i]:.3f} objective={best:.6f}", flush=True)

    shared = objective_matrix.mean(axis=1)
    shared_best = float(np.min(shared))
    shared_i = int(np.flatnonzero(np.isclose(shared, shared_best, rtol=0, atol=1e-12))[0])
    pd.DataFrame({"threshold": grid, "equal_model_equal_species_count_mae": shared}).to_csv(
        a.out / "shared_threshold_curve.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(a.out / "model_threshold_curves.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(a.out / "selected_thresholds.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "master_sha256": sha256(a.master),
        "validation_images": len(valid),
        "validation_species": int(valid["scientific_name"].nunique()),
        "shared_threshold": float(grid[shared_i]),
        "shared_objective": shared_best,
    })
    print(f"shared threshold {grid[shared_i]:.3f} objective={shared_best:.6f}")


if __name__ == "__main__":
    main()
