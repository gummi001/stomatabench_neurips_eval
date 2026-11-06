#!/bin/bash
# YOLOv8, YOLOv11, YOLOv12 and YOLOv26 (Ultralytics) on OD-Valid and the four test sets.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
WEIGHTS=${WEIGHTS:-$ROOT/data/weights}
OUT=${OUT:-$ROOT/outputs/predictions}
GPU=${1:-0}

SETS=("OD-Valid" "T-MAWP" "T-DCWP" "T-MSWP" "T-GSH")
MODELS=("YOLOv8" "YOLOv11" "YOLOv12" "YOLOv26")

for MODEL in "${MODELS[@]}"; do
  for SET in "${SETS[@]}"; do
    OUT_JSON="$OUT/$MODEL/$SET.json"
    if [ -s "$OUT_JSON" ]; then echo "exists, skipping: $OUT_JSON"; continue; fi
    mkdir -p "$OUT/$MODEL"
    echo "== $MODEL on $SET"
    CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/inference/infer_yolo.py" \
      --model-id "$MODEL" \
      --weights "$WEIGHTS/$MODEL/StomataBench-OD_$MODEL.pt" \
      --image-dir "$DATA/datasets/$SET/images" \
      --coco-json "$DATA/datasets/$SET/coco.json" \
      --out-json "$OUT_JSON" \
      --score-threshold 0.001 \
      --iou 0.7 \
      --imgsz 1280 \
      --max-det 1000 \
      --device 0
  done
done
