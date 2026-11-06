"""Per species: does the best model change between fixed thresholds, and does the AP leader lead counts?

Scene-primary images. Winners are the models with the best value (ties kept).
AP is COCO AP50:95 over all boxes; count and density metrics are at the fixed 0.25 threshold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import write_json

FIXED = (0.25, 0.50, 0.75)


def winner_set(group, column, higher):
    values = group[column].to_numpy(float)
    if not np.isfinite(values).any():
        return set()
    target = np.nanmax(values) if higher else np.nanmin(values)
    return set(group.loc[np.isclose(group[column], target, rtol=0, atol=1e-12), "model"].astype(str))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count-density", type=Path, required=True, help="output folder of evaluate_count_density.py")
    ap.add_argument("--coco", type=Path, required=True, help="output folder of evaluate_coco.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(a.count_density / "biological_metrics_long.csv", low_memory=False)
    species = metrics[metrics["group_level"].eq("species")]
    species_ap = pd.read_csv(a.coco / "species_coco_metrics_long.csv", low_memory=False)

    fixed = species[species["image_scope"].eq("scene_primary") & species["confidence_protocol"].eq("Fixed")
                    & species["threshold"].isin(list(FIXED))].copy()
    rows = []
    for (dataset, name), g in fixed.groupby(["dataset", "scientific_name"], sort=True):
        row = {"dataset": dataset, "scientific_name": name, "n_images": int(g["count_n"].iloc[0])}
        for metric, higher in (("density_r2", True), ("density_mae", False), ("count_r2", True), ("count_mae", False)):
            sets = {}
            for thr in FIXED:
                sets[thr] = winner_set(g[np.isclose(g["threshold"], thr)], metric, higher)
                row[f"{metric}_winner_{thr:.2f}"] = ";".join(sorted(sets[thr]))
            row[f"{metric}_winner_changed_025_to_075"] = sets[0.25] != sets[0.75]
            row[f"{metric}_winner_any_change"] = len({tuple(sorted(v)) for v in sets.values()}) > 1
        rows.append(row)
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(a.out / "fixed_threshold_species_winner_sensitivity.csv", index=False)
    sensitivity.groupby("dataset", as_index=False).agg(
        species=("scientific_name", "nunique"),
        density_r2_changed_025_to_075=("density_r2_winner_changed_025_to_075", "sum"),
        density_r2_any_change=("density_r2_winner_any_change", "sum"),
        count_mae_changed_025_to_075=("count_mae_winner_changed_025_to_075", "sum"),
        count_mae_any_change=("count_mae_winner_any_change", "sum"),
    ).to_csv(a.out / "fixed_threshold_winner_sensitivity_summary.csv", index=False)

    merged = species_ap[species_ap["image_scope"].eq("scene_primary")].merge(
        fixed[np.isclose(fixed["threshold"], 0.25)][["dataset", "scientific_name", "model", "count_n", "density_n",
                                                     "count_mae", "count_r2", "density_mae", "density_r2"]],
        on=["dataset", "scientific_name", "model"], validate="one_to_one")
    rows = []
    for (dataset, name), g in merged.groupby(["dataset", "scientific_name"], sort=True):
        if len(g) != 15:
            raise ValueError(f"{dataset}/{name}: expected 15 models, found {len(g)}")
        ap_winners = winner_set(g, "AP50_95", True)
        row = {"dataset": dataset, "scientific_name": name, "n_images": int(g["count_n"].iloc[0]),
               "ap_winners": ";".join(sorted(ap_winners))}
        for metric, higher in (("count_mae", False), ("count_r2", True), ("density_mae", False), ("density_r2", True)):
            winners = winner_set(g, metric, higher)
            rho = spearmanr(g["AP50_95"].to_numpy(float), g[metric].to_numpy(float), nan_policy="omit")
            row[f"{metric}_winners"] = ";".join(sorted(winners))
            row[f"ap_{metric}_winner_overlap"] = bool(ap_winners & winners)
            row[f"ap_{metric}_spearman"] = float(rho.statistic)
        rows.append(row)
    relationship = pd.DataFrame(rows)
    relationship.to_csv(a.out / "species_ap_biology_relationship.csv", index=False)

    rows = []
    for dataset, g in relationship.groupby("dataset", sort=True):
        row = {"dataset": dataset, "species": len(g)}
        for metric in ("count_mae", "count_r2", "density_mae", "density_r2"):
            rho = g[f"ap_{metric}_spearman"].dropna().to_numpy(float)
            row[f"{metric}_winner_overlap_count"] = int(g[f"ap_{metric}_winner_overlap"].sum())
            row[f"{metric}_spearman_median"] = float(np.median(rho)) if len(rho) else float("nan")
            row[f"{metric}_spearman_q25"] = float(np.quantile(rho, 0.25)) if len(rho) else float("nan")
            row[f"{metric}_spearman_q75"] = float(np.quantile(rho, 0.75)) if len(rho) else float("nan")
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out / "species_ap_biology_relationship_summary.csv", index=False)
    write_json(a.out / "run_metadata.json", {"image_scope": "scene_primary", "count_threshold": 0.25,
                                             "ap_confidence_filter": None, "species_rows": len(relationship)})


if __name__ == "__main__":
    main()
