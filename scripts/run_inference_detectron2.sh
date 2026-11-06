#!/bin/bash
# Faster R-CNN and RetinaNet (Detectron2) on OD-Valid and the four test sets.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
WEIGHTS=${WEIGHTS:-$ROOT/data/weights}
OUT=${OUT:-$ROOT/outputs/predictions}
GPU=${1:-0}

SETS=("OD-Valid" "T-MAWP" "T-DCWP" "T-MSWP" "T-GSH")

run_model () {
  MODEL="$1"; CONFIG="$2"; CKPT="$3"; SET="$4"
  OUT_JSON="$OUT/$MODEL/$SET.json"
  if [ -s "$OUT_JSON" ]; then echo "exists, skipping: $OUT_JSON"; return; fi
  mkdir -p "$OUT/$MODEL"
  echo "== $MODEL on $SET"
  CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/inference/infer_detectron2.py" \
    --model-id "$MODEL" \
    --config "$WEIGHTS/$MODEL/$CONFIG" \
    --weights "$WEIGHTS/$MODEL/$CKPT" \
    --image-dir "$DATA/datasets/$SET/images" \
    --coco-json "$DATA/datasets/$SET/coco.json" \
    --out-json "$OUT_JSON" \
    --score-threshold 0.001 \
    --category-id 1 \
    --device cuda
}

for SET in "${SETS[@]}"; do
  run_model FRCNN-R50-1x   config3.yaml StomataBench-OD_FASTER-R-CNN-R50-FPN-1x.pth      "$SET"
  run_model FRCNN-R50-3x   config3.yaml StomataBench-OD_FASTER-R-CNN-R50-FPN-3x.pth      "$SET"
  run_model FRCNN-R101-1x  config3.yaml StomataBench-OD_FASTER-R-CNN-R101-FPN-1x.pth     "$SET"
  run_model FRCNN-X101     config.yaml  StomataBench-OD_FRCNN-X101.pth                   "$SET"
  run_model Retina-R50-1x  config5.yaml StomataBench-OD_Retina-R50-1x.pth                "$SET"
  run_model Retina-R50-3x  config5.yaml StomataBench-OD_RETINA-NET-RESNET50-FPN-3x.pth   "$SET"
  run_model Retina-R101-3x config5.yaml StomataBench-OD_RETINA-NET-RESNET101-FPN-3x.pth  "$SET"
done
