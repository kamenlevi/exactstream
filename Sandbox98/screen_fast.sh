#!/bin/bash
> (a quote from the project owner, kept private)
# process on the same slices (sb/train_multi.py), shallow rollout (depth 12), read out at depth 15 (bands 0-14, kept/16),
# each lever compared paired against the in-batch control on band 7-14. Winners are confirmed at depth 30 (screen.sh with
# BASE=s30o OPT=s30o LR=1e-5 EPOCHS=2, the long format).
#   ./screen_fast.sh BATCH "LEVER1,LEVER2" "LINE" [STEPS] [DEPTH] [NMBPP] [NHE]
#   STEPS  default: the calibrated budget (results/screens/CALIBRATION.md); a lever needs a budget at which the control
#          itself measurably moves, otherwise NO EFFECT means untested
#   Memory: ~1.5 GB + ~0.75 GB per arm (+ ~1.5 GB for the step): by day keep arms (control included) <= 3 (~5 GB).
set -euo pipefail
cd "$(dirname "$0")"
BATCH=$1; LEVERS=$2; LINE=$3; STEPS=${4:-${FAST_STEPS:-$(python3 -c "import json; print(json.load(open('results/screens/calibration_v2.json'))['budget'] or '')" 2>/dev/null)}}; DEPTH=${5:-12}; NM=${6:-50}; NH=${7:-30}
[ -n "$STEPS" ] || { echo "no STEPS given and no calibrated FAST_STEPS default yet (results/screens/CALIBRATION.md)"; exit 2; }
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
TAG=fast_${BATCH}
ARMS="ctl:-"; RUNS="s30,${TAG}_ctl"
IFS=, read -ra LV <<< "$LEVERS"
for L in "${LV[@]}"; do ARMS="$ARMS,$L:$L"; RUNS="$RUNS,${TAG}_$L@$L"; done
N=$(( ${#LV[@]} + 1 ))
[ $N -le 3 ] || echo "warning: $N arms ~$(( 15 + 8 * N ))/10 GB peak: over the by-day ~5 GB line; run at night or split the batch"
mkdir -p results/screens logs
T0=$(date +%s)
$PY sb/train_multi.py --tag $TAG --arms $ARMS --depth $DEPTH --steps $STEPS > logs/$TAG.log 2>&1
$PY sb/evalx.py --runs $RUNS --sets mbpp,he --nm $NM --nh $NH --depth $((DEPTH + 3)) --mode chain --out results/screens/$TAG.json > logs/eval_$TAG.log 2>&1
{
  echo "== FAST screen $BATCH (levers $LEVERS; $STEPS steps at rollout depth $DEPTH, read-out depth $((DEPTH + 3)); $NM MBPP + $NH HE windows; control in-batch) $(date '+%a %d %b %H:%M')"
  echo "line: $LINE"
  echo "-- control vs s30 (must hold: MBPP 7-14 not down by more than 1.5 pt)"
  $PY sb/compare.py results/screens/$TAG.json "s30|mbpp|chain" results/screens/$TAG.json "${TAG}_ctl|mbpp|chain" --band 7-14
  for L in "${LV[@]}"; do
    for ST in mbpp he; do
      echo "-- $L, $ST"
      $PY sb/compare.py results/screens/$TAG.json "${TAG}_ctl|$ST|chain" results/screens/$TAG.json "${TAG}_$L|$ST|chain" --band 7-14 --json results/screens/${TAG}_${L}_$ST.json
    done
  done
  echo "wall $(( $(date +%s) - T0 )) s ($N arms; $(grep -o '[0-9.]*s/step' logs/$TAG.log | tail -1) per step for all arms)"
} | tee -a results/SCREENS.log
