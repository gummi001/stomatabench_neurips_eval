"""Bootstrap and leave-one-species-out uncertainty of the OD-Valid threshold selection.

Bootstrap: images are resampled within each of the five OD-Valid species, which keep
equal weight, and the threshold is selected again. Leave-one-species-out: select on
four species and score the fifth.
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (MODELS, REPLICATES, SEED, THRESHOLD_GRID, iter_json_array, load_master,
                    load_thresholds, write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True, help="selected_thresholds.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    a.out.mkdir(parents=True, exist_ok=True)

    master = load_master(a.master)
    valid = master[master["dataset"].eq("OD-Valid")].sort_values("coco_image_id").reset_index(drop=True)
    selected = load_thresholds(a.thresholds)
    grid = THRESHOLD_GRID
    truth = valid["gt_stomata_count"].to_numpy(int)
    position = {int(i): k for k, i in enumerate(valid["coco_image_id"].astype(int))}
    species_names = sorted(valid["scientific_name"].unique())
    species_idx = {s: np.flatnonzero(valid["scientific_name"].eq(s).to_numpy()) for s in species_names}
    fixed_i = int(np.flatnonzero(np.isclose(grid, 0.25))[0])
    rng = np.random.default_rng(a.seed)
    summary, frequency, loso, curves = [], [], [], []

    for model in MODELS:
        scores_by_image = [[] for _ in range(len(valid))]
        lo, hi = math.inf, -math.inf
        for rec in iter_json_array(pred_dir / model / "OD-Valid.json"):
            if int(rec.get("category_id", 1)) != 1:
                continue
            score = float(rec["score"])
            scores_by_image[position[int(rec["image_id"])]].append(score)
            lo, hi = min(lo, score), max(hi, score)
        errors = np.empty((len(valid), len(grid)))
        for k, s in enumerate(scores_by_image):
            s = np.asarray(s, dtype=np.float64)
            errors[k] = np.abs(np.count_nonzero(s[:, None] >= grid[None, :], axis=0) - truth[k])
        species_curves = np.vstack([errors[i].mean(axis=0) for i in species_idx.values()])
        objective = species_curves.mean(axis=0)
        supported = np.flatnonzero((grid >= math.floor(lo * 1000) / 1000) & (grid <= math.ceil(hi * 1000) / 1000))
        nominal_i = int(supported[np.argmin(objective[supported])])
        nominal = float(grid[nominal_i])
        if not np.isclose(nominal, selected[model], rtol=0, atol=1e-12):
            raise ValueError(f"{model}: threshold does not reproduce select_thresholds.py")

        boot = np.zeros((a.replicates, len(grid)))
        for idx in species_idx.values():
            n = len(idx)
            weights = rng.multinomial(n, np.full(n, 1.0 / n), size=a.replicates)
            boot += (weights @ errors[idx]) / n / len(species_names)
        boot_thr = grid[supported[np.argmin(boot[:, supported], axis=1)]]
        lower, upper = float(np.quantile(boot_thr, 0.025)), float(np.quantile(boot_thr, 0.975))
        counts = Counter(float(v) for v in boot_thr)
        mode_thr, mode_n = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0]
        in_interval = objective[np.flatnonzero((grid >= lower) & (grid <= upper))]
        best = float(objective[nominal_i])
        plateau = supported[objective[supported] <= best * 1.01 + 1e-12]
        summary.append({
            "model": model,
            "nominal_threshold": nominal,
            "nominal_species_balanced_count_mae": best,
            "bootstrap_median_threshold": float(np.median(boot_thr)),
            "bootstrap_ci_lower_2p5": lower,
            "bootstrap_ci_upper_97p5": upper,
            "bootstrap_mode_threshold": mode_thr,
            "bootstrap_mode_fraction": mode_n / a.replicates,
            "bootstrap_unique_thresholds": len(counts),
            "max_original_objective_within_bootstrap_interval": float(np.max(in_interval)),
            "max_objective_penalty_within_interval": float(np.max(in_interval) - best),
            "one_percent_plateau_lower": float(grid[plateau[0]]),
            "one_percent_plateau_upper": float(grid[plateau[-1]]),
            "one_percent_plateau_grid_points": len(plateau),
            "observed_score_min": lo,
            "observed_score_max": hi,
        })
        for thr, n in sorted(counts.items()):
            frequency.append({"model": model, "threshold": thr, "bootstrap_selection_count": n,
                              "bootstrap_selection_fraction": n / a.replicates})
        supported_set = set(supported)
        for t, thr in enumerate(grid):
            curves.append({"model": model, "threshold": thr, "species_balanced_count_mae": objective[t],
                           "within_rounded_score_support": bool(t in supported_set)})
        for h, heldout in enumerate(species_names):
            other = species_curves[[k for k in range(len(species_names)) if k != h]].mean(axis=0)
            pick = int(supported[np.argmin(other[supported])])
            loso.append({
                "model": model,
                "heldout_species": heldout,
                "selected_threshold_on_other_four_species": float(grid[pick]),
                "other_four_species_objective": float(other[pick]),
                "heldout_species_mae_at_loso_threshold": float(species_curves[h, pick]),
                "heldout_species_mae_at_full_selection": float(species_curves[h, nominal_i]),
                "heldout_species_mae_at_fixed025": float(species_curves[h, fixed_i]),
            })
        print(f"{model} done", flush=True)

    pd.DataFrame(summary).to_csv(a.out / "threshold_bootstrap_summary.csv", index=False)
    pd.DataFrame(frequency).to_csv(a.out / "threshold_bootstrap_frequencies.csv", index=False)
    pd.DataFrame(loso).to_csv(a.out / "threshold_leave_one_species_out.csv", index=False)
    pd.DataFrame(curves).to_csv(a.out / "threshold_objective_curves.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "replicates": a.replicates, "seed": a.seed,
        "bootstrap": "images resampled within each OD-Valid species; species keep equal weight",
        "interval": "percentile interval of the selected thresholds",
        "tie_rule": "lower grid threshold",
    })


if __name__ == "__main__":
    main()
