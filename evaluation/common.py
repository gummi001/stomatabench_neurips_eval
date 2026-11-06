"""Constants and helpers shared by the StomataBench evaluation scripts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = (
    "FRCNN-R50-1x", "FRCNN-R50-3x", "FRCNN-R101-1x", "FRCNN-X101",
    "Retina-R50-1x", "Retina-R50-3x", "Retina-R101-3x", "YOLOv8",
    "YOLOv11", "YOLOv12", "YOLOv26", "RT-DETRv2", "RT-DETRv4",
    "GDINO-SB", "SAM3-OD",
)
DATASETS = ("OD-Valid", "T-MAWP", "T-DCWP", "T-MSWP", "T-GSH")
TEST_SETS = ("T-MAWP", "T-DCWP", "T-MSWP", "T-GSH")

# Confidence protocols of the paper. The shared threshold is the one
# select_thresholds.py picks on OD-Valid; evaluate_counts.py checks that.
SHARED_THRESHOLD = 0.254
FIXED_THRESHOLDS = (0.25, 0.50, 0.75)
THRESHOLD_GRID = np.round(np.arange(0.001, 0.951, 0.001), 3)

SEED = 20260802
REPLICATES = 2000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed, *parts):
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def iter_json_array(path, chunk_chars=4 * 1024 * 1024):
    """Stream the objects of a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer, position, started = "", 0, False
    with open(path, "r", encoding="utf-8") as f:
        eof = False
        while True:
            chunk = f.read(chunk_chars)
            if chunk:
                buffer += chunk
            else:
                eof = True
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    position += 1
                    started = True
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield value
                position = end
            if position:
                buffer, position = buffer[position:], 0
            if eof:
                break
    raise ValueError(f"Unexpected end of JSON array in {path}")


def as_bool(series, name):
    s = series.astype(str).str.strip().str.lower()
    bad = ~s.isin(["true", "false"])
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} values are not true/false")
    return s.eq("true")


def load_master(path):
    """Read the master CSV, keep the five evaluation sets and check the density calibration."""
    df = pd.read_csv(path, low_memory=False)
    required = {
        "dataset", "coco_image_id", "basename", "gt_stomata_count",
        "scientific_name", "source_subset", "calibration_group",
        "is_scene_primary", "has_density", "image_area_mm2",
        "gt_density_per_mm2", "scene_id", "ood_level", "family", "order",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master missing columns: {sorted(missing)}")
    df = df[df["dataset"].isin(DATASETS)].copy()
    if df.duplicated(["dataset", "coco_image_id"]).any():
        raise ValueError("master has duplicate dataset/coco_image_id rows")
    df["coco_image_id"] = pd.to_numeric(df["coco_image_id"], errors="raise").astype(int)
    df["gt_stomata_count"] = pd.to_numeric(df["gt_stomata_count"], errors="raise").astype(int)
    df["is_scene_primary"] = as_bool(df["is_scene_primary"], "is_scene_primary")
    df["has_density"] = as_bool(df["has_density"], "has_density")
    df["image_area_mm2"] = pd.to_numeric(df["image_area_mm2"], errors="coerce")
    df["gt_density_per_mm2"] = pd.to_numeric(df["gt_density_per_mm2"], errors="coerce")
    density = df[df["has_density"]]
    if density[["image_area_mm2", "gt_density_per_mm2"]].isna().any().any():
        raise ValueError("has_density rows contain missing area or GT density")
    error = np.max(np.abs(density["gt_stomata_count"] / density["image_area_mm2"] - density["gt_density_per_mm2"]))
    if error > 1e-8:
        raise ValueError(f"density calibration mismatch, max error={error}")
    return df.sort_values(["dataset", "coco_image_id"]).reset_index(drop=True)


def load_thresholds(path):
    """Per-model thresholds from select_thresholds.py (selected_thresholds.csv)."""
    df = pd.read_csv(path)
    values = dict(zip(df["model"], df["selected_threshold_species_balanced_mae"].astype(float)))
    if set(values) != set(MODELS):
        raise ValueError("threshold table does not contain exactly the 15 paper models")
    return values


def protocol_columns(model_threshold):
    """(column, protocol label, threshold) for the six confidence protocols."""
    return [
        ("available_support", "Available support", float("nan")),
        ("shared_0p254", "Shared", SHARED_THRESHOLD),
        ("model_specific", "Model-specific", float(model_threshold)),
        ("fixed_0p25", "Fixed", FIXED_THRESHOLDS[0]),
        ("fixed_0p50", "Fixed", FIXED_THRESHOLDS[1]),
        ("fixed_0p75", "Fixed", FIXED_THRESHOLDS[2]),
    ]


def predictive_r2(truth, prediction):
    """1 - SSE/SST about the observed mean; NaN when the truth does not vary."""
    if len(truth) < 2:
        return float("nan")
    sst = float(np.square(truth - truth.mean()).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - np.square(truth - prediction).sum() / sst)


def read_part(path):
    df = pd.read_csv(path, low_memory=False)
    df["is_scene_primary"] = as_bool(df["is_scene_primary"], "is_scene_primary")
    if "has_density" in df.columns:
        df["has_density"] = as_bool(df["has_density"], "has_density")
    return df


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
