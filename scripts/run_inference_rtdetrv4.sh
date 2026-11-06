#!/bin/bash
# RT-DETRv4 (official code) on OD-Valid and the four test sets.
# RTDETRV4_REPO is a clone of https://github.com/RT-DETRs/RT-DETRv4 at commit df44122, with
# configs/rtv4_hgnetv2_l_stomata.yml and configs/stomata_detection.yml of this repository
# copied to its configs/stomata/.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
WEIGHTS=${WEIGHTS:-$ROOT/data/weights}
OUT=${OUT:-$ROOT/outputs/predictions}
GPU=${1:-0}

RTDETRV4_REPO=${RTDETRV4_REPO:?set RTDETRV4_REPO to a clone of the official RT-DETRv4 code}
RTDETRV4_CONFIG=${RTDETRV4_CONFIG:-$RTDETRV4_REPO/configs/stomata/rtv4_hgnetv2_l_stomata.yml}
RTDETRV4_CKPT=${RTDETRV4_CKPT:-$WEIGHTS/RT-DETRv4/StomataBench-OD_RT-DETRv4.pth}

SETS=("OD-Valid" "T-MAWP" "T-DCWP" "T-MSWP" "T-GSH")

for SET in "${SETS[@]}"; do
  OUT_JSON="$OUT/RT-DETRv4/$SET.json"
  if [ -s "$OUT_JSON" ]; then echo "exists, skipping: $OUT_JSON"; continue; fi
  mkdir -p "$OUT/RT-DETRv4"
  echo "== RT-DETRv4 on $SET"
  CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/inference/infer_rtdetrv4.py" \
    --model-id RT-DETRv4 \
    --repo-dir "$RTDETRV4_REPO" \
    --config "$RTDETRV4_CONFIG" \
    --checkpoint "$RTDETRV4_CKPT" \
    --image-dir "$DATA/datasets/$SET/images" \
    --coco-json "$DATA/datasets/$SET/coco.json" \
    --out-json "$OUT_JSON" \
    --score-threshold 0.001 \
    --category-id 1 \
    --max-det 1000 \
    --image-size 640 \
    --device cuda:0
done
