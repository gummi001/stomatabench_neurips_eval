import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    images = coco["images"]

    print(f"Model: {args.model_id}")
    print(f"Weights: {args.weights}")
    print(f"Dataset images: {len(images)}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Output: {out_json}")

    model = YOLO(args.weights)

    predictions = []
    start = time.time()

    for im in tqdm(images, desc=f"Inferencing {args.model_id}"):
        image_id = im["id"]
        file_name = im["file_name"]
        image_path = resolve_image_path(args.image_dir, file_name)

        results = model.predict(
            source=str(image_path),
            conf=args.score_threshold,
            iou=args.iou,
            max_det=args.max_det,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        if not results:
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy()
        scores = r.boxes.conf.detach().cpu().numpy()

        # original image size after ultralytics loading
        h, w = r.orig_shape

        for box, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box.tolist()

            # clip to image boundary
            x1 = max(0.0, min(float(x1), float(w)))
            y1 = max(0.0, min(float(y1), float(h)))
            x2 = max(0.0, min(float(x2), float(w)))
            y2 = max(0.0, min(float(y2), float(h)))

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
        "framework": "ultralytics",
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "coco_json": str(args.coco_json),
        "image_dir": str(args.image_dir),
        "out_json": str(out_json),
        "num_images": len(images),
        "num_predictions": len(predictions),
        "score_threshold": args.score_threshold,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "category_id": args.category_id,
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
    print(f"Num predictions:   {len(predictions)}")
    print(f"Elapsed seconds:   {elapsed:.1f}")


if __name__ == "__main__":
    main()
