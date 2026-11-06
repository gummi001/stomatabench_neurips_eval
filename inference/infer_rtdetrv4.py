import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


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


def preprocess_pil_for_rtdetrv4(image, size=(640, 640)):
    """
    Matches the working RT-DETRv4 notebook:
    RGB -> resize to 640x640 -> float tensor in [0, 1].
    """
    image = image.convert("RGB").resize(size, resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


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


def extract_state_dict(obj):
    if isinstance(obj, dict):
        tensor_count = sum(torch.is_tensor(v) for v in obj.values())
        if tensor_count > 10:
            return obj

        for key in ["module", "model", "state_dict", "ema"]:
            if key in obj:
                sd = extract_state_dict(obj[key])
                if sd is not None:
                    return sd

    return None


def normalize_state_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "_orig_mod.", "model.", "ema."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def load_rtdetrv4(repo_dir, config_path, checkpoint_path, device):
    """
    Loader matched to the working RT-DETRv4 notebook.
    Critical detail: disable HGNetv2 pretrained BEFORE cfg.model is built.
    """

    repo_dir = Path(repo_dir).resolve()
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    from engine.core import YAMLConfig

    cfg = YAMLConfig(
        str(config_path),
        test_only=True,
        seed=0,
        use_amp=False,
    )

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print("Task:", cfg.yaml_cfg.get("task"))
    print("num_classes:", cfg.yaml_cfg.get("num_classes"))
    print("remap_mscoco_category:", cfg.yaml_cfg.get("remap_mscoco_category"))

    model = cfg.model
    postprocessor = cfg.postprocessor

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    candidates = []

    if isinstance(checkpoint, dict):
        for name in ["ema", "model", "state_dict"]:
            if name in checkpoint:
                sd = extract_state_dict(checkpoint[name])
                if sd is not None:
                    candidates.append((name, normalize_state_keys(sd)))

        root_sd = extract_state_dict(checkpoint)
        if root_sd is not None:
            candidates.append(("checkpoint_root", normalize_state_keys(root_sd)))
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if not candidates:
        raise RuntimeError("Could not find a usable state_dict in the checkpoint.")

    model_keys = set(model.state_dict().keys())
    scored = []

    for name, sd in candidates:
        overlap = len(model_keys & set(sd.keys()))
        print(f"Candidate {name}: overlap {overlap}/{len(model_keys)}")
        scored.append((overlap, name, sd))

    best_overlap, best_name, best_sd = max(scored, key=lambda x: x[0])
    print("Using checkpoint candidate:", best_name)

    missing, unexpected = model.load_state_dict(best_sd, strict=False)

    print(f"Loaded: {best_name}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])

    if best_overlap < 0.5 * len(model_keys):
        raise RuntimeError(
            f"Checkpoint overlap is too low: {best_overlap}/{len(model_keys)}. "
            "Check config/checkpoint pair."
        )

    model.to(device).eval()
    print("RT-DETRv4 model ready.")

    return model, postprocessor


def infer_rtdetrv4(
    repo_dir,
    config_path,
    checkpoint_path,
    items,
    device,
    score_threshold,
    image_size,
    max_det,
):
    model, postprocessor = load_rtdetrv4(
        repo_dir=repo_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    raw_preds = []
    eval_size = (image_size, image_size)

    with torch.no_grad():
        for image_id, path, height, width in tqdm(items, desc="RT-DETRv4"):
            image = Image.open(path).convert("RGB")

            im_data = preprocess_pil_for_rtdetrv4(image, size=eval_size)
            im_data = im_data.unsqueeze(0).to(device)

            orig_size = torch.tensor([[width, height]], dtype=torch.float32).to(device)

            outputs = model(im_data)
            results = postprocessor(outputs, orig_size)[0]

            boxes = results["boxes"].detach().cpu()
            scores = results["scores"].detach().cpu()

            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            if len(scores) > 0:
                order = torch.argsort(scores, descending=True)[:max_det]
                boxes = boxes[order]
                scores = scores[order]

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.tolist()

                raw_preds.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

            del im_data, outputs, results

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return raw_preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", default="RT-DETRv4")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--out-json", required=True)

    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    start = time.time()

    print("=" * 80)
    print(f"Model ID:        {args.model_id}")
    print(f"Repo dir:        {args.repo_dir}")
    print(f"Config:          {args.config}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Dataset:         {args.coco_json}")
    print(f"Image dir:       {args.image_dir}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Category ID:     {args.category_id}")
    print(f"Max det/image:   {args.max_det}")
    print(f"Image size:      {args.image_size}")
    print(f"Device:          {args.device}")
    print("=" * 80)

    items, size_map = load_coco_items(args.coco_json, args.image_dir)
    print(f"Loaded {len(items)} images")

    raw_preds = infer_rtdetrv4(
        repo_dir=args.repo_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        items=items,
        device=args.device,
        score_threshold=args.score_threshold,
        image_size=args.image_size,
        max_det=args.max_det,
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
        "framework": "rtdetrv4_custom",
        "repo_dir": str(args.repo_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "coco_json": str(args.coco_json),
        "image_dir": str(args.image_dir),
        "out_json": str(args.out_json),
        "num_images": len(items),
        "num_raw_predictions": len(raw_preds),
        "num_saved_predictions": len(final_preds),
        "score_threshold": args.score_threshold,
        "max_det": args.max_det,
        "category_id": args.category_id,
        "image_size": args.image_size,
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