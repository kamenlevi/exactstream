#!/bin/bash
# WEEK98 screen 1 (MTP Head): depth done properly on Sandbox98. Pre-registered in REPORT 'WEEK98 S1 DEPTHC' before it ran.
# One process, three arms, all from the native head with a fresh Adam, on s30's slices (base files, seed 1114), s30's
# schedule (20-step warm-up to 1e-5, cosine to 1e-6 over 1,889 steps = one pass), rollout depth 30:
#   n1  = the native single layer (the control; at the end it should reproduce s30)
#   c2  = + a second layer copied from the native one (non-zero), in the main Adam (levers/deepc)
#   c2h = the same copy on its own Adam at 3e-5 (levers/deepch; deep2's lr, constant after the warm-up)
# Stop line at step 380 (20 %): BOTH copies <= -2.0 pt vs n1 at MBPP-50 23-30 with sign p < 0.05 -> stop (KILL).
# Sandbox98's scripts run unmodified; its guard_lib.sh is sourced (share lock, PAUSE, quiet-ticket yield, swap growth).
cd "$HOME/Sandbox98" || exit 1
OUT=$HOME/TrainLab/data/sb98; mkdir -p "$OUT"
LC=$HOME/TrainLab/sb98/levers/deepc; LH=$HOME/TrainLab/sb98/levers/deepch
TAG=mhd1
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=9          # training ~5.7 GB + the stop-line read-out beside it ~3.4 GB
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
ev() {   # OUTNAME RUNS...: read out at depth 31 on MBPP-50 + HE-30 (chain)
  local o=$1; shift
  $PY sb/evalx.py --runs "$(IFS=,; echo "$*")" --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out "$OUT/$o.json"
}
cmpj() {  # JSON A B SET
  $PY sb/compare.py "$OUT/$1.json" "$2|$4|chain" "$OUT/$1.json" "$3|$4|chain" --band 23-30 --json "$OUT/${1}_${4}_$2_vs_$3.json"
}
num() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['23-30']; print('%.4f %.4f' % (d['diff'], d['p_sign']))" "$1"; }
bad() { awk -v d="$1" -v p="$2" 'BEGIN{exit !(d <= -0.02 && p < 0.05)}'; }
log "screen 1 DEPTHC: train $TAG (n1, c2, c2h) 1,889 steps, copies at 380"
guarded ${TAG}_train "$OUT/${TAG}_train.log" 90 $PY sb/train_multi.py --tag $TAG --arms "n1:-,c2:$LC,c2h:$LH" --start native \
  --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 --save-at 380 &
GP=$!
# ---- the stop line at step 380
until { [ -f ckpt/${TAG}_n1_s380.safetensors ] && [ -f ckpt/${TAG}_c2_s380.safetensors ] && [ -f ckpt/${TAG}_c2h_s380.safetensors ]; } \
  || ! kill -0 $GP 2>/dev/null; do sleep 20; done
if [ -f ckpt/${TAG}_c2h_s380.safetensors ]; then
  log "stop-line read-out at step 380"
  ev ${TAG}_s380 ${TAG}_n1_s380 "${TAG}_c2_s380@$LC" "${TAG}_c2h_s380@$LH" > "$OUT/${TAG}_s380_eval.log" 2>&1
  { for A in c2 c2h; do for ST in mbpp he; do echo "-- $A vs n1, $ST (step 380)"; cmpj ${TAG}_s380 ${TAG}_n1_s380 ${TAG}_${A}_s380 $ST; done; done
  } > "$OUT/${TAG}_s380_compare.txt" 2>&1
  read D1 P1 < <(num "$OUT/${TAG}_s380_mbpp_${TAG}_n1_s380_vs_${TAG}_c2_s380.json")
  read D2 P2 < <(num "$OUT/${TAG}_s380_mbpp_${TAG}_n1_s380_vs_${TAG}_c2h_s380.json")
  log "step 380, MBPP-50 23-30 vs n1: c2 $D1 (p $P1), c2h $D2 (p $P2)"
  if bad "$D1" "$P1" && bad "$D2" "$P2"; then
    log "STOP LINE: both copies are worse than n1 at step 380 -> KILL, stopping the training"
    pkill -TERM -f "sb/train_multi.py --tag $TAG"; wait $GP
    echo "DEPTHC KILL at the stop line (step 380): MBPP-50 23-30 vs n1: c2 $D1 (p $P1), c2h $D2 (p $P2)" > "$OUT/${TAG}_RESULT.txt"
    log "DEPTHC done"; exit 0
  fi
  log "stop line passed: training continues to 1,889"
fi
wait $GP; RC=$?
for A in n1 c2 c2h; do [ -f ckpt/${TAG}_$A.safetensors ] || { log "training did not finish (rc $RC)"; echo "DEPTHC: training did not finish (rc $RC)" > "$OUT/${TAG}_RESULT.txt"; exit 1; }; done
# ---- the final read-out
log "final read-out (s30, n1, c2, c2h)"
guarded ${TAG}_eval "$OUT/${TAG}_eval.log" 30 $PY sb/evalx.py --runs "s30,${TAG}_n1,${TAG}_c2@$LC,${TAG}_c2h@$LH" --sets mbpp,he \
  --nm 50 --nh 30 --depth 31 --mode chain --out "$OUT/${TAG}_final.json" || exit 1
{ for A in c2 c2h; do for ST in mbpp he; do echo "-- $A vs n1, $ST (1,889 steps)"; cmpj ${TAG}_final ${TAG}_n1 ${TAG}_$A $ST; done; done
  echo "-- control check: n1 vs s30, mbpp"; cmpj ${TAG}_final s30 ${TAG}_n1 mbpp
  for A in c2 c2h; do $PY -I -c "import mlx.core as mx; w = mx.load('ckpt/${TAG}_$A.safetensors'); print('$A: extra layer |o_proj| max %.3g, |down_proj| max %.3g; native layer |o_proj| max %.3g' % (float(mx.abs(w['xl.0.o_proj.weight']).max()), float(mx.abs(w['xl.0.down_proj.weight']).max()), float(mx.abs(w['o_proj.weight']).max())))"; done
} > "$OUT/${TAG}_final_compare.txt" 2>&1
R=""
for A in c2 c2h; do
  read D P < <(num "$OUT/${TAG}_final_mbpp_${TAG}_n1_vs_${TAG}_$A.json")
  V=$(awk -v d="$D" -v p="$P" 'BEGIN{ if (d >= 0.02 && p < 0.05) print "DEPTH HELPS"; else if (d <= -0.02 && p < 0.05) print "HURTS"; else print "NO EFFECT"}')
  R="$R$A $V ($D, p $P); "
done
read DC PC < <(num "$OUT/${TAG}_final_mbpp_s30_vs_${TAG}_n1.json")
echo "DEPTHC at 1,889 steps, MBPP-50 23-30 vs n1: $R control n1 - s30 = $DC (p $PC); details $OUT/${TAG}_final_compare.txt" > "$OUT/${TAG}_RESULT.txt"
log "$(cat "$OUT/${TAG}_RESULT.txt")"; log "DEPTHC done"
