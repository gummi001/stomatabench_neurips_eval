#!/bin/bash
# Every result figure of the paper, from the result CSVs of scripts/run_eval.sh (RESULTS=results for the released ones).
# CPU only. The last step also needs images, coco.json files and predictions (python scripts/download.py eval, images).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
RESULTS=${RESULTS:-$ROOT/outputs}
OUT=${OUT:-$ROOT/outputs/figures}
PY=${PYTHON:-python}
F="$ROOT/figures"

step () { echo; echo "== $1"; }

step "1/8 Figure 2: F1 and count MAE at five confidence settings"
$PY "$F/fig_threshold_sensitivity.py" --root "$RESULTS" --out "$OUT"

step "2/8 Figure 3: detectors below the species-mean reference"
$PY "$F/fig_species_r2_counts_all.py" --root "$RESULTS" --out "$OUT"

step "3/8 Figures S6-S11: set-level values at the other confidence settings"
$PY "$F/fig_threshold_values.py" --root "$RESULTS" --out "$OUT"

step "4/8 Figures S12-S13: bootstrap intervals per set"
$PY "$F/fig_bootstrap_intervals.py" --root "$RESULTS" --out "$OUT"

step "5/8 Figures S14-S18: bootstrap intervals per species"
$PY "$F/fig_species_intervals.py" --root "$RESULTS" --out "$OUT"

step "6/8 Figures S19-S20, S43-S50: within-species density R2"
$PY "$F/fig_species_r2_heatmap.py" --root "$RESULTS" --out "$OUT"

step "7/8 Figures S21-S42, S51-S58: species heatmaps"
$PY "$F/fig_species_heatmaps.py" --root "$RESULTS" --out "$OUT"

step "8/8 Figures S59-S61: example predictions"
if [ -d "$DATA/datasets/OD-Valid/images" ]; then
    $PY "$F/fig_example_predictions.py" --root "$RESULTS" --data "$DATA" --out "$OUT"
else
    echo "skipped: no images in $DATA (python scripts/download.py images)"
fi

echo; echo "done: $OUT"
