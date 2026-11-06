"""Collect the result tables of the paper into outputs/tables/ as CSV files.

Reads the outputs of the evaluation scripts (see scripts/run_eval.sh for the folder names).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASETS, MODELS, as_bool

ROLE = {
    "OD-Train": "training",
    "OD-Valid": "development and threshold selection",
    "T-MAWP": "held-out test",
    "T-DCWP": "held-out test",
    "T-MSWP": "held-out test",
    "T-GSH": "held-out test",
}
SHIFT = {
    "OD-Train": "training distribution",
    "OD-Valid": "familiar species, new plants/accessions",
    "T-MAWP": "taxonomic",
    "T-DCWP": "taxonomic and microscope",
    "T-MSWP": "microscope and annotation",
    "T-GSH": "taxonomic, source, and annotation",
}
# Rows of Table S21, in display order.
EXAMPLES = {
    ("T-MAWP", "Malus domestica"): ["YOLOv12", "YOLOv11", "FRCNN-R50-3x", "RT-DETRv2", "RT-DETRv4"],
    ("T-MSWP", "Ulmus americana Planch"): ["SAM3-OD", "GDINO-SB", "YOLOv11", "YOLOv8", "RT-DETRv2"],
    ("T-GSH", "Triticum aestivum"): ["GDINO-SB", "SAM3-OD", "FRCNN-X101", "Retina-R101-3x", "YOLOv26", "RT-DETRv2"],
}


def fixed025(df, level, iou=None):
    keep = (df["image_scope"].eq("scene_primary") & df["group_level"].eq(level)
            & df["confidence_protocol"].eq("Fixed") & np.isclose(df["threshold"], .25))
    if iou is not None:
        keep &= np.isclose(df["matching_iou"], iou)
    return df.loc[keep].copy()


def dataset_inventory(master):
    rows = []
    for dataset in ["OD-Train", *DATASETS]:
        f = master.loc[master["dataset"].eq(dataset)]
        p = f.loc[f["is_scene_primary"]]
        rows.append({
            "dataset": dataset, "role": ROLE[dataset], "shift": SHIFT[dataset],
            "images_all": len(f), "images_scene_primary": len(p), "redundant_views": len(f) - len(p),
            "annotations_all": int(f["gt_stomata_count"].sum()),
            "annotations_scene_primary": int(p["gt_stomata_count"].sum()),
            "taxon_labels": int(f["scientific_name"].nunique()),
            "density_images_all": int(f["has_density"].sum()),
            "density_images_scene_primary": int(p["has_density"].sum()),
            "ood_levels": ", ".join(sorted(f["ood_level"].dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def tmawp_composition(master):
    f = master.loc[master["dataset"].eq("T-MAWP")]
    return f.groupby(["ood_level", "scientific_name", "family", "order"], sort=False).agg(
        accessions=("accession_id", lambda x: ", ".join(sorted({str(v) for v in x.dropna() if str(v).strip()}))),
        images=("global_image_id", "size"),
        stomata=("gt_stomata_count", "sum"),
        mean_density=("gt_density_per_mm2", "mean"),
    ).reset_index()


def example_species(root):
    points = pd.read_csv(root / "example_species" / "example_species_point_metrics.csv")
    boot = pd.read_csv(root / "example_species" / "example_species_bootstrap_metrics.csv")
    ap_points = pd.read_csv(root / "example_ap" / "target_ap_point_metrics.csv")
    ap_boot = pd.read_csv(root / "example_ap" / "target_ap_bootstrap_metrics.csv")

    def pick(df, dataset, species, model, metric=None):
        keep = df["dataset"].eq(dataset) & df["scientific_name"].eq(species) & df["model"].eq(model)
        if metric is not None:
            keep &= df["metric"].eq(metric)
        return df.loc[keep].iloc[0]

    rows = []
    for (dataset, species), models in EXAMPLES.items():
        for model in models:
            point = pick(points, dataset, species, model)
            ap_hit = pick(ap_boot, dataset, species, model, "AP50_95")
            r2_hit = pick(boot, dataset, species, model, "density_predictive_R2")
            mae_hit = pick(boot, dataset, species, model, "density_MAE")
            rows.append({
                "dataset": dataset, "species": species, "model": model, "n_images": int(point["n_images"]),
                "AP50_95": pick(ap_points, dataset, species, model)["AP50_95"],
                "AP50_95_lower": ap_hit["ci_lower_2p5"], "AP50_95_upper": ap_hit["ci_upper_97p5"],
                "F1": point["F1"],
                "density_r2": point["density_predictive_R2"],
                "density_r2_lower": r2_hit["ci_lower_2p5"], "density_r2_upper": r2_hit["ci_upper_97p5"],
                "density_mae": point["density_MAE"],
                "density_mae_lower": mae_hit["ci_lower_2p5"], "density_mae_upper": mae_hit["ci_upper_97p5"],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="outputs folder of scripts/run_eval.sh")
    ap.add_argument("--master", type=Path, required=True)
    a = ap.parse_args()
    out = a.root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(a.master, low_memory=False)
    master["is_scene_primary"] = as_bool(master["is_scene_primary"], "is_scene_primary")
    master["has_density"] = as_bool(master["has_density"], "has_density")
    dataset_inventory(master).to_csv(out / "dataset_inventory.csv", index=False)
    tmawp_composition(master).to_csv(out / "tmawp_species_composition.csv", index=False)

    coco = pd.read_csv(a.root / "coco" / "coco_metrics_long.csv")
    coco["image_scope"] = coco["subset"].replace({"dedup": "scene_primary"})
    coco.loc[coco["image_scope"].eq("scene_primary")].to_csv(out / "official_coco_scene_primary.csv", index=False)

    op = pd.read_csv(a.root / "detection" / "operating_point_metrics_long.csv", low_memory=False)
    bio = pd.read_csv(a.root / "count_density" / "biological_metrics_long.csv", low_memory=False)
    macro = pd.read_csv(a.root / "count_density" / "macro_species_metrics.csv")
    fixed025(op, "dataset", .5).merge(
        fixed025(bio, "dataset")[["dataset", "model", "count_mae", "count_signed_bias",
                                   "density_n", "density_mae", "density_signed_bias"]],
        on=["dataset", "model"], validate="one_to_one",
    ).to_csv(out / "operating_point_fixed025_iou050.csv", index=False)

    macro025 = macro.loc[macro["image_scope"].eq("scene_primary") & macro["confidence_protocol"].eq("Fixed")
                         & np.isclose(macro["threshold"], .25)]
    pooled = fixed025(bio, "dataset").merge(macro025, on=["dataset", "model", "image_scope", "confidence_protocol",
                                                          "threshold"], validate="one_to_one")
    pooled[["dataset", "model", "density_n", "density_r2", "density_mae", "density_signed_bias",
            "density_species_with_defined_r2", "macro_species_density_r2_mean", "macro_species_density_r2_median",
            "macro_species_density_r2_q25", "macro_species_density_r2_q75",
            "macro_species_density_r2_positive_fraction", "macro_species_density_mae_mean"]].to_csv(
        out / "pooled_and_macro_species_density.csv", index=False)

    selected = pd.read_csv(a.root / "thresholds" / "selected_thresholds.csv")
    uncertainty = pd.read_csv(a.root / "threshold_uncertainty" / "threshold_bootstrap_summary.csv")
    selected.merge(uncertainty, on="model", how="left", suffixes=("", "_bootstrap")).to_csv(
        out / "validation_selected_thresholds.csv", index=False)

    example_species(a.root).to_csv(out / "example_species_compact.csv", index=False)

    species_ap = pd.read_csv(a.root / "coco" / "species_coco_metrics_long.csv")
    species = species_ap.loc[species_ap["image_scope"].eq("scene_primary")].merge(
        fixed025(op, "species", .5)[["dataset", "model", "scientific_name", "tp", "fp", "fn", "precision", "recall", "f1"]],
        on=["dataset", "model", "scientific_name"], validate="one_to_one",
    ).merge(
        fixed025(bio, "species")[["dataset", "model", "scientific_name", "count_n", "count_mae", "count_r2", "density_n",
                                  "density_mae", "density_r2", "density_signed_bias", "density_gt_sd"]],
        on=["dataset", "model", "scientific_name"], validate="one_to_one",
    )
    if species.duplicated(["dataset", "model", "scientific_name"]).any():
        raise ValueError("species table has duplicate rows")
    species.to_csv(out / "species_metrics_fixed025_scene_primary.csv", index=False)

    parts = sorted((a.root / "count_density" / "per_model_dataset_parts").glob("*.csv"))
    per_image = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
    expected = len(MODELS) * int(master["dataset"].isin(DATASETS).sum())
    if len(parts) != len(MODELS) * len(DATASETS) or len(per_image) != expected:
        raise ValueError(f"per-image table has {len(per_image)} rows from {len(parts)} files, expected {expected}")
    if per_image.duplicated(["dataset", "model", "coco_image_id"]).any():
        raise ValueError("per-image table has duplicate dataset/model/image rows")
    per_image.to_csv(out / "per_image_measurements.csv.gz", index=False, compression="gzip")
    print(f"wrote {len(list(out.iterdir()))} tables to {out}")


if __name__ == "__main__":
    main()
