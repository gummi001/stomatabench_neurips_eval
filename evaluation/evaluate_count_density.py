"""Stomatal count and density error per image, species and set, under six confidence protocols.

Count: number of boxes kept at a confidence threshold, on every annotated image.
Density: that count divided by image_area_mm2, only where has_density is true.
No IoU matching is involved. Results are given for all images and for scene-primary images.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DATASETS, MODELS, SHARED_THRESHOLD, iter_json_array, load_master,
                    load_thresholds, predictive_r2, protocol_columns, read_part, sha256, write_json)

PART_COLUMNS = [
    "dataset", "model", "coco_image_id", "basename", "scene_id",
    "is_scene_primary", "scientific_name", "source_subset",
    "calibration_group", "ood_level", "family", "order", "has_density",
    "image_area_mm2", "gt_count", "gt_density_per_mm2",
]
GROUPS = {
    "dataset": [],
    "species": ["scientific_name"],
    "species_acquisition": ["scientific_name", "source_subset", "calibration_group"],
}


def build_part(pred_path, part_path, meta, model, model_threshold):
    """Write one row per image with the number of boxes kept under each protocol."""
    protocols = protocol_columns(model_threshold)
    counts = {int(i): np.zeros(len(protocols), dtype=np.int64) for i in meta["coco_image_id"]}
    records, lo, hi = 0, math.inf, -math.inf
    prev_id, prev_score = -1, math.inf
    for rec in iter_json_array(pred_path):
        if int(rec.get("category_id", 1)) != 1:
            continue
        records += 1
        image_id, score = int(rec["image_id"]), float(rec["score"])
        lo, hi = min(lo, score), max(hi, score)
        if image_id < prev_id or (image_id == prev_id and score > prev_score + 1e-12):
            raise ValueError(f"{pred_path}: boxes must be sorted by image id, then by descending score")
        prev_id, prev_score = image_id, score
        if image_id not in counts:
            raise ValueError(f"{pred_path}: unknown image id {image_id}")
        c = counts[image_id]
        c[0] += 1
        for k, (_, _, thr) in enumerate(protocols[1:], start=1):
            if score >= thr:
                c[k] += 1

    part_path.parent.mkdir(parents=True, exist_ok=True)
    with open(part_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PART_COLUMNS + [p[0] for p in protocols])
        writer.writeheader()
        for row in meta.itertuples(index=False):
            image_id = int(row.coco_image_id)
            out = {
                "dataset": row.dataset, "model": model, "coco_image_id": image_id,
                "basename": row.basename, "scene_id": row.scene_id,
                "is_scene_primary": bool(row.is_scene_primary), "scientific_name": row.scientific_name,
                "source_subset": row.source_subset, "calibration_group": row.calibration_group,
                "ood_level": row.ood_level, "family": row.family, "order": row.order,
                "has_density": bool(row.has_density), "image_area_mm2": row.image_area_mm2,
                "gt_count": int(row.gt_stomata_count), "gt_density_per_mm2": row.gt_density_per_mm2,
            }
            out.update({protocols[k][0]: int(counts[image_id][k]) for k in range(len(protocols))})
            writer.writerow(out)

    per_image = np.vstack(list(counts.values()))
    return {
        "dataset": str(meta["dataset"].iloc[0]),
        "model": model,
        "prediction_sha256": sha256(pred_path),
        "images": len(meta),
        "images_without_available_predictions": int(np.count_nonzero(per_image[:, 0] == 0)),
        "records": records,
        "score_min": float(lo) if math.isfinite(lo) else None,
        "score_max": float(hi) if math.isfinite(hi) else None,
        "model_specific_threshold": model_threshold,
    }


def pearson_r(truth, prediction):
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def endpoint_metrics(truth, prediction, prefix):
    error = prediction - truth
    n = len(truth)
    nan = float("nan")
    return {
        f"{prefix}_n": n,
        f"{prefix}_gt_total": float(truth.sum()),
        f"{prefix}_pred_total": float(prediction.sum()),
        f"{prefix}_mae": float(np.mean(np.abs(error))) if n else nan,
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(error)))) if n else nan,
        f"{prefix}_signed_bias": float(np.mean(error)) if n else nan,
        f"{prefix}_r2": predictive_r2(truth, prediction),
        f"{prefix}_pearson_r": pearson_r(truth, prediction),
        f"{prefix}_gt_mean": float(np.mean(truth)) if n else nan,
        f"{prefix}_gt_sd": float(np.std(truth, ddof=1)) if n > 1 else nan,
        f"{prefix}_pred_mean": float(np.mean(prediction)) if n else nan,
        f"{prefix}_pred_sd": float(np.std(prediction, ddof=1)) if n > 1 else nan,
    }


def aggregate(parts, thresholds, out):
    """Pooled (set), species and species-by-acquisition metrics, then macro species summaries."""
    rows = []
    for part in parts:
        df = read_part(part)
        model, dataset = str(df["model"].iloc[0]), str(df["dataset"].iloc[0])
        for scope, scoped in (("all", df), ("scene_primary", df[df["is_scene_primary"]])):
            for key, label, nominal in protocol_columns(0.0):
                thr = thresholds[model] if key == "model_specific" else nominal
                for level, cols in GROUPS.items():
                    groups = [((), scoped)] if not cols else scoped.groupby(cols[0] if len(cols) == 1 else cols,
                                                                             sort=True, dropna=False)
                    for gkey, g in groups:
                        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
                        row = {"dataset": dataset, "model": model, "image_scope": scope, "group_level": level,
                               "confidence_protocol": label, "threshold": thr,
                               "scientific_name": "", "source_subset": "", "calibration_group": ""}
                        row.update(dict(zip(cols, gkey)))
                        row.update(endpoint_metrics(g["gt_count"].to_numpy(float), g[key].to_numpy(float), "count"))
                        d = g[g["has_density"]]
                        if len(d):
                            truth = d["gt_density_per_mm2"].to_numpy(float)
                            pred = d[key].to_numpy(float) / d["image_area_mm2"].to_numpy(float)
                            row.update(endpoint_metrics(truth, pred, "density"))
                        else:
                            row.update(endpoint_metrics(np.array([]), np.array([]), "density"))
                        row["small_n_count_flag"] = "descriptive_only_n_lt_5" if len(g) < 5 else ""
                        row["small_n_density_flag"] = "descriptive_only_n_lt_5" if 0 < len(d) < 5 else ""
                        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "biological_metrics_long.csv", index=False)

    # Macro species summaries: every species with data counts in macro MAE;
    # R2 needs at least three images in the species.
    keys = ["dataset", "model", "image_scope", "confidence_protocol", "threshold"]
    macro_rows = []
    species = metrics[metrics["group_level"].eq("species")]
    for values, g in species.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, values))
        row["species_count"] = int(g["scientific_name"].nunique())
        for ep in ("count", "density"):
            eligible = g[g[f"{ep}_n"] > 0]
            r2 = eligible.loc[eligible[f"{ep}_n"] >= 3, f"{ep}_r2"].dropna().to_numpy(float)
            mae = eligible[f"{ep}_mae"].dropna().to_numpy(float)
            some_r2, some_mae = len(r2) > 0, len(mae) > 0
            nan = float("nan")
            row[f"{ep}_species_with_data"] = len(eligible)
            row[f"{ep}_species_with_defined_r2"] = len(r2)
            row[f"macro_species_{ep}_r2_mean"] = float(np.mean(r2)) if some_r2 else nan
            row[f"macro_species_{ep}_r2_median"] = float(np.median(r2)) if some_r2 else nan
            row[f"macro_species_{ep}_r2_q25"] = float(np.quantile(r2, 0.25)) if some_r2 else nan
            row[f"macro_species_{ep}_r2_q75"] = float(np.quantile(r2, 0.75)) if some_r2 else nan
            row[f"macro_species_{ep}_r2_positive_fraction"] = float(np.mean(r2 > 0)) if some_r2 else nan
            row[f"macro_species_{ep}_mae_mean"] = float(np.mean(mae)) if some_mae else nan
            row[f"macro_species_{ep}_mae_median"] = float(np.median(mae)) if some_mae else nan
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(out / "macro_species_metrics.csv", index=False)

    pooled = metrics[metrics["group_level"].eq("dataset")]
    pooled.merge(macro, on=keys, validate="one_to_one", suffixes=("_pooled", "_macro")).to_csv(
        out / "pooled_vs_macro_species.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder with datasets/ and metadata/")
    ap.add_argument("--predictions", type=Path, default=None, help="prediction folder (default: <data>/predictions)")
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True, help="selected_thresholds.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pred_dir = a.predictions or a.data / "predictions"
    a.out.mkdir(parents=True, exist_ok=True)

    selection = json.loads((a.thresholds.parent / "run_metadata.json").read_text(encoding="utf-8"))
    if not np.isclose(selection["shared_threshold"], SHARED_THRESHOLD, rtol=0, atol=1e-12):
        raise ValueError(f"shared threshold on OD-Valid is {selection['shared_threshold']}, not {SHARED_THRESHOLD}")
    master = load_master(a.master)
    thresholds = load_thresholds(a.thresholds)

    parts, audit = [], []
    for dataset in DATASETS:
        meta = master[master["dataset"].eq(dataset)]
        for model in MODELS:
            part = a.out / "per_model_dataset_parts" / f"{dataset}__{model}.csv"
            audit.append(build_part(pred_dir / model / f"{dataset}.json", part, meta, model,
                                    thresholds[model]))
            parts.append(part)
            print(f"{dataset:8s} {model:17s} {audit[-1]['records']:,} boxes", flush=True)
    aggregate(parts, thresholds, a.out)
    pd.DataFrame(audit).to_csv(a.out / "prediction_audit.csv", index=False)
    write_json(a.out / "run_metadata.json", {
        "master_sha256": sha256(a.master),
        "thresholds_sha256": sha256(a.thresholds),
        "count_scope": "all annotated images, no IoU matching",
        "density_scope": "has_density true; kept boxes divided by image_area_mm2",
    })


if __name__ == "__main__":
    main()
