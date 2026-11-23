#!/bin/bash
# One-command screen (<= 15 min on a shared GPU): a lever vs the cached control, paired on the same windows.
#   ./screen.sh NAME LEVER "LINE" [STEPS] [NMBPP] [NHE]
#   NAME   short id (ckpt/scr_NAME, results/screens/NAME.json)
#   LEVER  levers/LEVER.py (see README), or "-" for the control itself
#   LINE   the pre-registered "result -> action" line, written BEFORE the run (it is printed with the result)
#   STEPS  training steps (96 anchors each) from the base drafter s30 on the screen pool (default 300; 0 = eval-only lever)
#   lr: constant 2e-6 after a 20-step warm-up (LR=...; results/screens/SETUP.md: at 1e-5 the control drifted 8 pt below s30)
#   NMBPP / NHE   held-out windows (default 50 MBPP + 30 HumanEval; the full sets are 100 / 60)
# Control = s30 + the same STEPS on the same pool slices in the same order, no lever (trained once per STEPS, cached).
set -euo pipefail
cd "$(dirname "$0")"
NAME=$1; LEVER=$2; LINE=$3; STEPS=${4:-300}; NM=${5:-50}; NH=${6:-30}
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
LR=${LR:-2e-6}; SEED=${SEED:-1115}; BASE=${BASE:-s30}; OPT=${OPT:-}; EPOCHS=${EPOCHS:-1}
TX="--epochs $EPOCHS"; [ -n "$OPT" ] && TX="$TX --resume-opt $OPT"     # long screens: continue the base's Adam
mkdir -p results/screens logs
CTL=ctl_${BASE}_${STEPS}_lr${LR}${OPT:+_opt}
[ "$STEPS" = 0 ] && CTL=$BASE
T0=$(date +%s)
if [ "$STEPS" != 0 ] && [ ! -f ckpt/$CTL.safetensors ]; then
  $PY sb/train.py --run $CTL --start $BASE --files pool --steps $STEPS --const --lr $LR --seed $SEED $TX > logs/$CTL.log 2>&1
fi
CE=results/screens/_eval_${CTL}_${NM}_${NH}.json
[ -f $CE ] || $PY sb/evalx.py --runs $CTL --sets mbpp,he --n 0 --mode chain --out $CE --nm $NM --nh $NH > logs/eval_$CTL.log 2>&1
if [ "$LEVER" = "-" ]; then echo "control $CTL ready: $CE"; exit 0; fi
RUN=scr_${NAME}_${STEPS}_lr${LR}${OPT:+_opt}
if [ "$STEPS" != 0 ]; then
  $PY sb/train.py --run $RUN --start $BASE --files pool --steps $STEPS --const --lr $LR --seed $SEED $TX --lever $LEVER > logs/$RUN.log 2>&1
  EVR=$RUN
else
  EVR=$BASE
fi
VE=results/screens/_eval_${RUN}_${NM}_${NH}.json
$PY sb/evalx.py --runs $EVR --sets mbpp,he --n 0 --mode chain --lever $LEVER --out $VE --nm $NM --nh $NH > logs/eval_$RUN.log 2>&1
{
  echo "== screen $NAME (lever $LEVER, $STEPS steps, $NM MBPP + $NH HE windows; control $CTL) $(date '+%a %d %b %H:%M')"
  echo "line: $LINE"
  for ST in mbpp he; do
    echo "-- $ST"
    $PY sb/compare.py $CE "$CTL|$ST|chain" $VE "$EVR|$ST|chain" --band 23-30 --json results/screens/${NAME}_$ST.json
  done
  echo "wall $(( $(date +%s) - T0 )) s"
} | tee -a results/SCREENS.log
