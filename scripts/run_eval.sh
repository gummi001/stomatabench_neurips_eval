#!/bin/bash
# Every evaluation step of the paper, from the released predictions to the result tables.
# CPU only. Needs coco.json files, the master CSV and predictions (python scripts/download.py eval).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/data/StomataBench}
MASTER=${MASTER:-$DATA/metadata/StomataBench_master_v10.csv}
PRED=${PRED:-$DATA/predictions}
OUT=${OUT:-$ROOT/outputs}
PY=${PYTHON:-python}
E="$ROOT/evaluation"
THR="$OUT/thresholds/selected_thresholds.csv"

step () { echo; echo "== $1"; }

step "1/11 confidence thresholds on OD-Valid"
$PY "$E/select_thresholds.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --out "$OUT/thresholds"

step "2/11 COCO AP per set and per species"
$PY "$E/evaluate_coco.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --out "$OUT/coco"

step "3/11 count and density"
$PY "$E/evaluate_count_density.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --thresholds "$THR" --out "$OUT/count_density"

step "4/11 TP, FP, FN"
$PY "$E/evaluate_detection.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --thresholds "$THR" --out "$OUT/detection"

step "5/11 bootstrap intervals for count and density"
$PY "$E/bootstrap_count_density.py" --count-density "$OUT/count_density" --out "$OUT/bootstrap"

step "6/11 threshold uncertainty"
$PY "$E/threshold_uncertainty.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --thresholds "$THR" --out "$OUT/threshold_uncertainty"

step "7/11 example species (Table S21)"
$PY "$E/bootstrap_example_species.py" --count-density "$OUT/count_density" --detection "$OUT/detection" --out "$OUT/example_species"

step "8/11 example species AP (Table S21)"
$PY "$E/bootstrap_example_ap.py" --data "$DATA" --predictions "$PRED" --master "$MASTER" --out "$OUT/example_ap"

step "9/11 threshold and AP agreement per species"
$PY "$E/analyze_threshold_ap.py" --count-density "$OUT/count_density" --coco "$OUT/coco" --out "$OUT/threshold_ap"

step "10/11 Gibbs wheat"
$PY "$E/analyze_wheat.py" --count-density "$OUT/count_density" --detection "$OUT/detection" --out "$OUT/wheat"

step "11/11 result tables"
$PY "$E/build_result_tables.py" --root "$OUT" --master "$MASTER"

echo; echo "done: $OUT"
