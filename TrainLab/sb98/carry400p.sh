#!/bin/bash
# WEEK98 screen 2 (MTP Head): THE CARRY on Sandbox98. Pre-registered in REPORT 'WEEK98 S2 CARRY400' before it ran.
# ARMS (env) picks the batch: 2a = "n1:-,ca:carrya,cac:carryac" (default), 2b = "ar:allrows" (compared against 2a's n1;
# same slices, order and schedule). All from the native head, fresh Adam, base files seed 1114, rollout depth 30, a
# 400-step cosine (20-step warm-up to 1e-5, to 1e-6 at step 400): DEPTHN's dn_n1 recipe (MBPP-50 23-30 48.37).
cd "$HOME/Sandbox98" || exit 1
OUT=$HOME/TrainLab/data/sb98; mkdir -p "$OUT"
L=$HOME/TrainLab/sb98/levers
TAG=${TAG:-mhc2a}
ARMS=${ARMS:-"n1:-,ca:$L/carrya,cac:$L/carryac"}
CTL=${CTL:-${TAG}_n1}                       # the control run name (2b: mhc2a_n1)
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=${MEMGB:-6}
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
cmpj() { $PY sb/compare.py "$OUT/$1.json" "$2|$4|chain" "$OUT/$1.json" "$3|$4|chain" --band 23-30 --json "$OUT/${1}_${4}_$2_vs_$3.json"; }
num() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['23-30']; print('%.4f %.4f' % (d['diff'], d['p_sign']))" "$1"; }
log "screen 2 CARRY400 $TAG: arms $ARMS"
/Users/kamen/TrainLab/sb98/run_paused.sh ${TAG}_train "$OUT/${TAG}_train.log" 90 $PY sb/train_multi.py --tag $TAG --arms "$ARMS" --start native --resume-opt "" \
  --files base --seed 1114 --depth 30 --steps 400 || { log "training failed"; exit 1; }
RUNS="dn_n1"; LEVS=""
for spec in ${ARMS//,/ }; do nm=${spec%%:*}; lv=${spec#*:}
  if [ "$lv" = "-" ]; then RUNS="$RUNS,${TAG}_$nm"; else RUNS="$RUNS,${TAG}_$nm@$lv"; LEVS="$LEVS $nm"; fi; done
[ "$CTL" != "${TAG}_n1" ] && RUNS="$RUNS,$CTL"
log "read-out: $RUNS"
/Users/kamen/TrainLab/sb98/run_paused.sh ${TAG}_eval "$OUT/${TAG}_eval.log" 30 $PY sb/evalx.py --runs "$RUNS" --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain \
  --out "$OUT/${TAG}.json" || exit 1
{ for A in $LEVS; do for ST in mbpp he; do echo "-- $A vs $CTL, $ST"; cmpj $TAG $CTL ${TAG}_$A $ST; done; done
  echo "-- control check: $CTL vs dn_n1 (DEPTHN's same recipe), mbpp"; cmpj $TAG dn_n1 $CTL mbpp
} > "$OUT/${TAG}_compare.txt" 2>&1
R=""
for A in $LEVS; do read D P < <(num "$OUT/${TAG}_mbpp_${CTL}_vs_${TAG}_$A.json")
  V=$(awk -v d="$D" -v p="$P" 'BEGIN{ if (d >= 0.02 && p < 0.05) print "PROMISING"; else if (d <= -0.02 && p < 0.05) print "HURTS"; else print "NO EFFECT"}')
  R="$R$A $V ($D, p $P); "; done
read DC PC < <(num "$OUT/${TAG}_mbpp_dn_n1_vs_$CTL.json")
echo "CARRY400 $TAG, MBPP-50 23-30 vs $CTL: $R control $CTL - dn_n1 = $DC (p $PC); details $OUT/${TAG}_compare.txt" > "$OUT/${TAG}_RESULT.txt"
log "$(cat "$OUT/${TAG}_RESULT.txt")"; log "CARRY400 $TAG done"
