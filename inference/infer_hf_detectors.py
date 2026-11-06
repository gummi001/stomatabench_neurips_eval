import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)

def pad_pil_to_multiple(image, multiple=32, fill=0):
    """
    Pad PIL RGB image on right/bottom so H and W are divisible by `multiple`.
    Returns padded_image, original_size, padded_size.
    """
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if new_w == w and new_h == h:
        return image, (h, w), (h, w)

    padded = Image.new("RGB", (new_w, new_h), color=(fill, fill, fill))
    padded.paste(image, (0, 0))
    return padded, (h, w), (new_h, new_w)
    
def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_model_dir(model_dir):
    model_dir = Path(model_dir)
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    for p in candidates:
        if p.exists():
            return sha256_file(p)
    return None


def set_seed(seed: int):
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


def load_coco_items(coco_json: str, image_dir: str):
    coco = COCO(coco_json)
    imgs = coco.loadImgs(coco.getImgIds())

    items = []
    size_map = {}

    for im in imgs:
        image_id = int(im["id"])
        file_name = im["file_name"]
        h = int(im["height"])
        w = int(im["width"])
        path = resolve_image_path(image_dir, file_name)

        items.append((image_id, str(path), h, w))
        size_map[image_id] = (h, w)

    return coco, items, size_map


def clip_xywh(x, y, w, h, W, H):
    x = max(0.0, min(float(x), float(W)))
    y = max(0.0, min(float(y), float(H)))
    w = max(0.0, min(float(w), float(W) - x))
    h = max(0.0, min(float(h), float(H) - y))
    return x, y, w, h


def save_coco_predictions(raw_preds, size_map, out_json, category_id=1, max_dets_per_image=1000):
    per_image = defaultdict(list)

    for pred in raw_preds:
        image_id = int(pred["image_id"])
        if image_id not in size_map:
            continue

        H, W = size_map[image_id]
        x, y, w, h = pred["bbox"]
        x, y, w, h = clip_xywh(x, y, w, h, W, H)

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

def infer_rtdetr_auto(model_dir, items, device, score_threshold):
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        do_resize=False,
        do_normalize=True,
        use_fast=False,
    )
    model = AutoModelForObjectDetection.from_pretrained(model_dir)
    model.to(device).eval()

    raw_preds = []

    with torch.no_grad():
        for image_id, path, H, W in tqdm(items, desc="RT-DETRv1"):
            image = Image.open(path).convert("RGB")

            # RT-DETRv1 with do_resize=False can fail on arbitrary image sizes.
            # Pad to multiple of 32 to avoid FPN feature-map mismatch.
            image_pad, (orig_H, orig_W), (pad_H, pad_W) = pad_pil_to_multiple(
                image, multiple=32, fill=0
            )

            encoded = processor(images=image_pad, return_tensors="pt").to(device)
            outputs = model(**encoded)

            # Postprocess to padded image size, then clipping happens later
            # in save_coco_predictions() using the original COCO H,W.
            dets = processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=[(pad_H, pad_W)],
            )[0]

            for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                raw_preds.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.detach().cpu()),
                    }
                )

            del encoded, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return raw_preds


def infer_rtdetrv2(model_dir, items, device, score_threshold):
    processor = RTDetrImageProcessor.from_pretrained(
        model_dir,
        do_resize=False,
        do_normalize=True,
        use_fast=False,
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(model_dir)
    model.to(device).eval()

    raw_preds = []

    with torch.no_grad():
        for image_id, path, H, W in tqdm(items, desc="RT-DETRv2"):
            image = Image.open(path).convert("RGB")

            # RT-DETRv2 can fail on arbitrary image sizes because FPN feature maps
            # must align. Padding preserves original scale while making H,W divisible by 32.
            image_pad, (orig_H, orig_W), (pad_H, pad_W) = pad_pil_to_multiple(
                image, multiple=32, fill=0
            )

            inputs = processor(images=image_pad, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            target_sizes = torch.tensor([(pad_H, pad_W)], device=outputs.logits.device)

            dets = processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )[0]

            for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                raw_preds.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.detach().cpu()),
                    }
                )

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return raw_preds


def infer_gdino(model_dir, processor_id, items, device, score_threshold, text_prompt, text_threshold):
    # Try local processor first. If local GDINO folder cannot load tokenizer,
    # use IDEA-Research/grounding-dino-base as processor fallback.
    try:
        processor = AutoProcessor.from_pretrained(model_dir)
        print(f"Loaded GDINO processor from local model dir: {model_dir}")
    except Exception as e:
        print(f"Local GDINO processor failed: {e}")
        print(f"Falling back to processor: {processor_id}")
        processor = AutoProcessor.from_pretrained(processor_id)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir)
    model.to(device).eval()

    raw_preds = []

    with torch.no_grad():
        for image_id, path, H, W in tqdm(items, desc="GDINO-SB"):
            image = Image.open(path).convert("RGB")

            inputs = processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            dets = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]

            boxes = dets["boxes"].detach().cpu().numpy()
            scores = dets["scores"].detach().cpu().numpy()

            for box, score in zip(boxes, scores):
                s = float(score)
                if s < score_threshold:
                    continue

                x1, y1, x2, y2 = box.tolist()
                raw_preds.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": s,
                    }
                )

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return raw_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", required=True, choices=["rtdetrv1", "rtdetrv2", "gdino"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # GDINO-only args
    parser.add_argument("--processor-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--text-prompt", default="a stomata .")
    parser.add_argument("--text-threshold", type=float, default=0.25)

    args = parser.parse_args()

    set_seed(args.seed)

    start = time.time()

    print("=" * 80)
    print(f"Model ID:        {args.model_id}")
    print(f"Model kind:      {args.model_kind}")
    print(f"Model dir:       {args.model_dir}")
    print(f"Dataset:         {args.coco_json}")
    print(f"Image dir:       {args.image_dir}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Category ID:     {args.category_id}")
    print(f"Max det/image:   {args.max_det}")
    print(f"Device:          {args.device}")
    if args.model_kind == "gdino":
        print(f"Text prompt:     {args.text_prompt}")
        print(f"Text threshold:  {args.text_threshold}")
    print("=" * 80)

    coco, items, size_map = load_coco_items(args.coco_json, args.image_dir)
    print(f"Loaded {len(items)} images")

    if args.model_kind == "rtdetrv1":
        raw_preds = infer_rtdetr_auto(
            model_dir=args.model_dir,
            items=items,
            device=args.device,
            score_threshold=args.score_threshold,
        )
    elif args.model_kind == "rtdetrv2":
        raw_preds = infer_rtdetrv2(
            model_dir=args.model_dir,
            items=items,
            device=args.device,
            score_threshold=args.score_threshold,
        )
    elif args.model_kind == "gdino":
        raw_preds = infer_gdino(
            model_dir=args.model_dir,
            processor_id=args.processor_id,
            items=items,
            device=args.device,
            score_threshold=args.score_threshold,
            text_prompt=args.text_prompt,
            text_threshold=args.text_threshold,
        )
    else:
        raise ValueError(args.model_kind)

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
        "model_kind": args.model_kind,
        "framework": "transformers",
        "model_dir": str(args.model_dir),
        "checkpoint_sha256": sha256_model_dir(args.model_dir),
        "coco_json": str(args.coco_json),
        "image_dir": str(args.image_dir),
        "out_json": str(args.out_json),
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

    if args.model_kind == "gdino":
        meta["processor_id"] = args.processor_id
        meta["text_prompt"] = args.text_prompt
        meta["text_threshold"] = args.text_threshold

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
