"""Loaders of the result CSVs the figures read, from a folder laid out like results/ (the outputs of scripts/run_eval.sh)."""
from pathlib import Path

import numpy as np
import pandas as pd

from style import DATASETS, MODELS

HERE = Path(__file__).resolve().parent
OPERATING = "detection/operating_point_metrics_long.csv"
BIOLOGY = "count_density/biological_metrics_long.csv"
BOOTSTRAP = "bootstrap/bootstrap_biological_fixed025_scene_primary.csv"
SPECIES = "tables/species_metrics_fixed025_scene_primary.csv"
POOLED = "tables/pooled_and_macro_species_density.csv"
PER_IMAGE = "tables/per_image_measurements.csv.gz"
# Confidence protocols other than fixed 0.25: (protocol, fixed threshold or None).
THRESHOLD_PROTOCOLS = [("Shared", None), ("Model-specific", None), ("Fixed", 0.50), ("Fixed", 0.75)]


def read(root, name):
    return pd.read_csv(Path(root) / name, low_memory=False)


def dataset_rows(frame, protocol, threshold=None):
    """One dataset-level, scene-primary row per model and dataset for a confidence protocol (IoU 0.50 where matched)."""
    keep = frame["group_level"].eq("dataset") & frame["image_scope"].eq("scene_primary") & frame["confidence_protocol"].eq(protocol)
    if "matching_iou" in frame:
        keep &= np.isclose(frame["matching_iou"], 0.5)
    if threshold is not None:
        keep &= np.isclose(frame["threshold"], threshold)
    rows = frame[keep]
    assert len(rows) == len(MODELS) * len(DATASETS) and not rows.duplicated(["dataset", "model"]).any()
    return rows.set_index(["model", "dataset"])


def verify_species_protocol(root):
    """Check that TP, FP and FN of the species CSV are the fixed 0.25, IoU 0.50 rows of the operating-point file."""
    sp = read(root, SPECIES)
    op = read(root, OPERATING)
    op = op.loc[op["group_level"].eq("species") & op["image_scope"].eq("scene_primary") & np.isclose(op["matching_iou"], 0.5)
                & op["confidence_protocol"].eq("Fixed") & np.isclose(op["threshold"], 0.25)]
    merged = sp.merge(op, on=["dataset", "model", "scientific_name"], suffixes=("", "_op"), validate="one_to_one")
    if len(merged) != len(sp):
        raise ValueError("species rows missing from the operating-point file")
    for col in ["tp", "fp", "fn"]:
        if not np.array_equal(merged[col].to_numpy(), merged[f"{col}_op"].to_numpy()):
            raise ValueError(f"species {col} is not the fixed 0.25, IoU 0.50 value")


def species_values(root):
    """Per-species values at fixed 0.25 (scene-primary): the species CSV plus mean GT density from the biological file."""
    bio = read(root, BIOLOGY)
    bio = bio[bio["group_level"].eq("species") & bio["image_scope"].eq("scene_primary") & bio["confidence_protocol"].eq("Fixed")
              & np.isclose(bio["threshold"], 0.25)]
    sp = read(root, SPECIES)
    m = sp.merge(bio[["dataset", "model", "scientific_name", "density_n", "density_gt_mean", "density_r2", "density_mae",
                      "count_signed_bias", "count_gt_sd"]],
                 on=["dataset", "model", "scientific_name"], suffixes=("", "_bio"), validate="one_to_one")
    assert len(m) == len(sp)
    for c in ("density_n", "density_r2", "density_mae"):
        assert np.allclose(m[c].astype(float), m[f"{c}_bio"].astype(float), equal_nan=True)
    return m


def species_order(sub):
    """One row per species of a set, in the order of the SI species tables (species_order.csv)."""
    order = pd.read_csv(HERE / "species_order.csv")
    position = {key: i for i, key in enumerate(zip(order["dataset"], order["scientific_name"]))}
    info = sub.drop_duplicates("scientific_name").copy()
    info["position"] = [position[key] for key in zip(info["dataset"], info["scientific_name"])]
    return info.sort_values("position")
