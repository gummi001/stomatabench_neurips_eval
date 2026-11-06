#!/bin/bash
# RT-DETRv2 and GDINO-SB (HuggingFace Transformers) on OD-Valid and the four test sets.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
WEIGHTS=${WEIGHTS:-$ROOT/data/weights}
OUT=${OUT:-$ROOT/outputs/predictions}
GPU=${1:-0}

SETS=("OD-Valid" "T-MAWP" "T-DCWP" "T-MSWP" "T-GSH")

run_model () {
  MODEL="$1"; KIND="$2"; SET="$3"; shift 3
  OUT_JSON="$OUT/$MODEL/$SET.json"
  if [ -s "$OUT_JSON" ]; then echo "exists, skipping: $OUT_JSON"; return; fi
  mkdir -p "$OUT/$MODEL"
  echo "== $MODEL on $SET"
  CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/inference/infer_hf_detectors.py" \
    --model-id "$MODEL" \
    --model-kind "$KIND" \
    --model-dir "$WEIGHTS/$MODEL" \
    --image-dir "$DATA/datasets/$SET/images" \
    --coco-json "$DATA/datasets/$SET/coco.json" \
    --out-json "$OUT_JSON" \
    --score-threshold 0.001 \
    --category-id 1 \
    --max-det 1000 \
    --device cuda:0 \
    "$@"
}

for SET in "${SETS[@]}"; do
  run_model RT-DETRv2 rtdetrv2 "$SET"
  run_model GDINO-SB gdino "$SET" \
    --processor-id IDEA-Research/grounding-dino-base \
    --text-prompt "a stomata ." \
    --text-threshold 0.25
done
