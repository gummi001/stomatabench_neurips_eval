import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_dir, file_name):
    """Find an image by base name in images/ or, for T-MSWP, images_2/."""
    image_dir = Path(image_dir)
    for folder in (image_dir, image_dir.parent / "images_2"):
        p = folder / Path(file_name).name
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found: {file_name} under {image_dir}")


def set_detectron2_thresholds(cfg, score_threshold):
    """
    Set score thresholds robustly for common Detectron2 architectures.
    ROI_HEADS: Faster R-CNN / Fast R-CNN / Mask R-CNN style.
    RETINANET: RetinaNet style.
    """
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold

    if hasattr(cfg.MODEL, "PROPOSAL_GENERATOR"):
        # Not always present/needed, but safe to leave untouched.
        pass


def build_predictor(config_path, weights_path, score_threshold, device):
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device

    set_detectron2_thresholds(cfg, score_threshold)

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config_path = Path(args.config)
    weights_path = Path(args.weights)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    images = coco["images"]

    print("=" * 80)
    print(f"Model:      {args.model_id}")
    print(f"Config:     {config_path}")
    print(f"Weights:    {weights_path}")
    print(f"Dataset:    {args.coco_json}")
    print(f"Image dir:  {args.image_dir}")
    print(f"# images:   {len(images):,}")
    print(f"Threshold:  {args.score_threshold}")
    print(f"Category:   {args.category_id}")
    print(f"Out:        {out_json}")
    print("=" * 80)

    predictor, cfg = build_predictor(
        config_path=config_path,
        weights_path=weights_path,
        score_threshold=args.score_threshold,
        device=args.device,
    )

    predictions = []
    start = time.time()

    for im in tqdm(images, desc=f"Inferencing {args.model_id}"):
        image_id = im["id"]
        file_name = im["file_name"]

        image_path = resolve_image_path(args.image_dir, file_name)
        img = cv2.imread(str(image_path))

        if img is None:
            raise RuntimeError(f"cv2 could not read image: {image_path}")

        height, width = img.shape[:2]

        with torch.no_grad():
            outputs = predictor(img)

        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_boxes") or not instances.has("scores"):
            continue

        boxes_xyxy = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        for box, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box.tolist()

            # Clip to original image dimensions.
            x1 = max(0.0, min(float(x1), float(width)))
            y1 = max(0.0, min(float(y1), float(height)))
            x2 = max(0.0, min(float(x2), float(width)))
            y2 = max(0.0, min(float(y2), float(height)))

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(args.category_id),
                    "bbox": [
                        round(float(x1), 4),
                        round(float(y1), 4),
                        round(float(bw), 4),
                        round(float(bh), 4),
                    ],
                    "score": round(float(score), 6),
                }
            )

    elapsed = time.time() - start

    with open(out_json, "w") as f:
        json.dump(predictions, f)

    meta = {
        "model_id": args.model_id,
        "framework": "detectron2",
        "config": str(config_path),
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "coco_json": str(args.coco_json),
        "image_dir": str(args.image_dir),
        "out_json": str(out_json),
        "num_images": len(images),
        "num_predictions": len(predictions),
        "score_threshold": args.score_threshold,
        "category_id": args.category_id,
        "device": args.device,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": elapsed,
    }

    meta_path = out_json.with_name(out_json.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved predictions: {out_json}")
    print(f"Saved metadata:    {meta_path}")
    print(f"Num predictions:   {len(predictions):,}")
    print(f"Elapsed seconds:   {elapsed:.1f}")


if __name__ == "__main__":
    main()
