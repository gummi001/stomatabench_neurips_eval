#!/bin/bash
# SAM3-OD (official SAM 3 code) on OD-Valid and the four test sets.
# SAM3_BASE_WEIGHTS is sam3.pt from https://huggingface.co/facebook/sam3 and
# SAM3_BPE_PATH is assets/bpe_simple_vocab_16e6.txt.gz of https://github.com/facebookresearch/sam3.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
WEIGHTS=${WEIGHTS:-$ROOT/data/weights}
OUT=${OUT:-$ROOT/outputs/predictions}
GPU=${1:-0}

SAM3_BASE_WEIGHTS=${SAM3_BASE_WEIGHTS:?set SAM3_BASE_WEIGHTS to the official sam3.pt}
SAM3_BPE_PATH=${SAM3_BPE_PATH:?set SAM3_BPE_PATH to bpe_simple_vocab_16e6.txt.gz}
SAM3_FINETUNED_CKPT=${SAM3_FINETUNED_CKPT:-$WEIGHTS/SAM3-OD/StomataBench-OD_SAM3.pt}

SETS=("OD-Valid" "T-MAWP" "T-DCWP" "T-MSWP" "T-GSH")

for SET in "${SETS[@]}"; do
  OUT_JSON="$OUT/SAM3-OD/$SET.json"
  if [ -s "$OUT_JSON" ]; then echo "exists, skipping: $OUT_JSON"; continue; fi
  mkdir -p "$OUT/SAM3-OD"
  echo "== SAM3-OD on $SET"
  CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/inference/infer_sam3_od.py" \
    --model-id SAM3-OD \
    --base-weights "$SAM3_BASE_WEIGHTS" \
    --finetuned-checkpoint "$SAM3_FINETUNED_CKPT" \
    --bpe-path "$SAM3_BPE_PATH" \
    --image-dir "$DATA/datasets/$SET/images" \
    --coco-json "$DATA/datasets/$SET/coco.json" \
    --out-json "$OUT_JSON" \
    --text-prompt "stomata" \
    --score-threshold 0.001 \
    --category-id 1 \
    --max-det 1000 \
    --device cuda:0
done
