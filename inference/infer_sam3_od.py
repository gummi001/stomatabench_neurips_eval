import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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


def load_coco_items(coco_json, image_dir):
    coco = COCO(coco_json)
    imgs = coco.loadImgs(coco.getImgIds())

    items = []
    size_map = {}

    for im in imgs:
        image_id = int(im["id"])
        file_name = im["file_name"]
        height = int(im["height"])
        width = int(im["width"])
        path = resolve_image_path(image_dir, file_name)

        items.append((image_id, str(path), height, width))
        size_map[image_id] = (height, width)

    return items, size_map


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_xywh(x, y, w, h, image_width, image_height):
    x = max(0.0, min(float(x), float(image_width)))
    y = max(0.0, min(float(y), float(image_height)))
    w = max(0.0, min(float(w), float(image_width) - x))
    h = max(0.0, min(float(h), float(image_height) - y))
    return x, y, w, h


def save_coco_predictions(raw_preds, size_map, out_json, category_id=1, max_dets_per_image=1000):
    per_image = defaultdict(list)

    for pred in raw_preds:
        image_id = int(pred["image_id"])

        if image_id not in size_map:
            continue

        height, width = size_map[image_id]
        x, y, w, h = pred["bbox"]
        x, y, w, h = clip_xywh(x, y, w, h, width, height)

        if w <= 0 or h <= 0:
            continue

        per_image[image_id].append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [
                    round(float(x), 4),
                    round(float(y), 4),
                    round(float(w), 4),
                    round(float(h), 4),
                ],
                "score": round(float(pred["score"]), 6),
            }
        )

    final = []

    for image_id, dets in per_image.items():
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        final.extend(dets[:max_dets_per_image])

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(final, f)

    return final


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "module"]:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_sam3_model_and_processor(base_weights, finetuned_checkpoint, bpe_path, device):
    print("=" * 80)
    print("Loading SAM3-OD")
    print(f"Base weights:       {base_weights}")
    print(f"Fine-tuned ckpt:    {finetuned_checkpoint}")
    print(f"BPE path:           {bpe_path}")
    print(f"Device:             {device}")
    print("=" * 80)

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        checkpoint_path=str(base_weights),
    )

    ckpt = torch.load(str(finetuned_checkpoint), map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded fine-tuned checkpoint keys: {len(state_dict)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if unexpected:
        print("First unexpected keys:", unexpected[:10])

    model.to(device)
    model.eval()

    processor = Sam3Processor(model)

    print("SAM3-OD ready.")
    return model, processor


@torch.no_grad()
def infer_sam3(processor, items, text_prompt, score_threshold):
    raw_preds = []

    for image_id, path, height, width in tqdm(items, desc="SAM3-OD"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"WARNING: failed to load {path}: {e}")
            continue

        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=text_prompt)

        boxes = out["boxes"]
        scores = out["scores"]

        if isinstance(boxes, torch.Tensor):
            if boxes.ndim == 3:
                boxes = boxes[0]
            boxes = boxes.detach().cpu().numpy()

        if isinstance(scores, torch.Tensor):
            if scores.ndim == 2:
                scores = scores[0]
            scores = scores.detach().cpu().numpy()

        for box_xyxy, score in zip(boxes, scores):
            score = float(score)

            if score < score_threshold:
                continue

            raw_preds.append(
                {
                    "image_id": int(image_id),
                    "bbox": xyxy_to_xywh(box_xyxy),
                    "score": score,
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return raw_preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", default="SAM3-OD")
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-json", required=True)

    parser.add_argument("--text-prompt", default="stomata")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    start = time.time()

    print("=" * 80)
    print(f"Model ID:        {args.model_id}")
    print(f"Dataset:         {args.coco_json}")
    print(f"Image dir:       {args.image_dir}")
    print(f"Text prompt:     {args.text_prompt}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Category ID:     {args.category_id}")
    print(f"Max det/image:   {args.max_det}")
    print(f"Device:          {args.device}")
    print("=" * 80)

    items, size_map = load_coco_items(args.coco_json, args.image_dir)
    print(f"Loaded {len(items)} images")

    model, processor = load_sam3_model_and_processor(
        base_weights=args.base_weights,
        finetuned_checkpoint=args.finetuned_checkpoint,
        bpe_path=args.bpe_path,
        device=args.device,
    )

    raw_preds = infer_sam3(
        processor=processor,
        items=items,
        text_prompt=args.text_prompt,
        score_threshold=args.score_threshold,
    )

    final_preds = save_coco_predictions(
        raw_preds=raw_preds,
        size_map=size_map,
        out_json=args.out_json,
        category_id=args.category_id,
        max_dets_per_image=args.max_det,
    )

    elapsed = time.time() - start

    meta = {
        "model_id": args.model_id,
        "framework": "sam3_custom",
        "base_weights": str(args.base_weights),
        "base_weights_sha256": sha256_file(args.base_weights),
        "finetuned_checkpoint": str(args.finetuned_checkpoint),
        "finetuned_checkpoint_sha256": sha256_file(args.finetuned_checkpoint),
        "bpe_path": str(args.bpe_path),
        "coco_json": str(args.coco_json),
        "image_dir": str(args.image_dir),
        "out_json": str(args.out_json),
        "text_prompt": args.text_prompt,
        "num_images": len(items),
        "num_raw_predictions": len(raw_preds),
        "num_saved_predictions": len(final_preds),
        "score_threshold": args.score_threshold,
        "max_det": args.max_det,
        "category_id": args.category_id,
        "device": args.device,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": elapsed,
    }

    meta_path = Path(args.out_json).with_name(Path(args.out_json).stem + "_meta.json")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved predictions: {args.out_json}")
    print(f"Saved metadata:    {meta_path}")
    print(f"Raw predictions:   {len(raw_preds):,}")
    print(f"Saved predictions: {len(final_preds):,}")
    print(f"Elapsed seconds:   {elapsed:.1f}")


if __name__ == "__main__":
    main()
