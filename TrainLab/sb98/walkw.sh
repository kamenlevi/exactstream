#!/bin/bash
# WEEK98 S5 WALKW (MTP Head): walk-weighted and close-call-weighted TTT on Sandbox98. Pre-registered in REPORT
# 'WEEK98 S5 WALKW' (+ amendment 1) before it runs.
# 1. s30's walk on the base training windows -> check-start anchors (walk_starts.py; + the depth-0 diagnostic);
# 2. two arms in one process, dn_n1's recipe (from native, fresh Adam, base files seed 1114, rollout depth 30, 400-step
#    cosine): ww (levers/walkw: check-start anchors weigh 3x in every depth's KL) and mw (levers/marginw: rows whose
#    target margin is < 1 nat weigh 3x). The control is mhc2a_n1 (= dn_n1 exactly: same start, rows, seed, order);
# 3. read-out (evalx, depth 31, MBPP-50 + HE-30) + compare on kept/32; walk_diag.py for depth 0 at each head's own
#    check starts and walk-strict 0-14. Everything paused-safe (run_paused.sh). Scratch tag mhx (overwrites).
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; L=$HOME/TrainLab/sb98/levers; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=5
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
$S/run_paused.sh walkw_starts $O/walkw_starts.log 40 $PY $S/walk_starts.py || { log "walk table failed"; exit 1; }
log "$(tail -2 $O/walkw_starts.log)"
$S/run_paused.sh mhx_train $O/mhx_train.log 60 $PY sb/train_multi.py --tag mhx --arms "ww:$L/walkw,mw:$L/marginw" --start native \
  --resume-opt "" --files base --seed 1114 --depth 30 --steps 400 || { log "training failed"; exit 1; }
grep -q "DONE mhx" $O/mhx_train.log || { log "training did not finish"; exit 1; }
$S/run_paused.sh mhx_eval $O/mhx_eval.log 20 $PY sb/evalx.py --runs "mhc2a_n1,mhx_ww@$L/walkw,mhx_mw@$L/marginw" --sets mbpp,he \
  --nm 50 --nh 30 --depth 31 --mode chain --out $O/walkw.json || exit 1
for A in ww mw; do for ST in mbpp he; do
  echo "-- $A vs n1, $ST"
  $PY sb/compare.py $O/walkw.json "mhc2a_n1|$ST|chain" $O/walkw.json "mhx_$A|$ST|chain" --band 23-30 --json $O/walkw_${A}_$ST.json
done; done > $O/walkw_compare.txt 2>&1
$S/run_paused.sh walkw_diag $O/walkw_diag.log 20 $PY $S/walk_diag.py $O/walkw_diag.json mhc2a_n1 mhx_ww mhx_mw
R=""
for A in ww mw; do
  read D P < <(python3 -I -c "import json; d=json.load(open('$O/walkw_${A}_mbpp.json'))['kept']; print('%.4f %.4f' % (d['diff'], d['p_sign']))")
  V=$(awk -v d="$D" -v p="$P" 'BEGIN{ if (d >= 0.10 && p < 0.05) print "PROMISING"; else if (d <= -0.10 && p < 0.05) print "HURTS"; else print "NO EFFECT"}')
  R="$R$A $V ($D, p $P); "
done
echo "WALKW, MBPP-50 kept/32 vs n1: $R bands + HE in walkw_compare.txt; check-start depth 0 + walk-strict in walkw_diag.log" > $O/walkw_RESULT.txt
peer say --to "AI/LLM project recommendations" "MTP Head (auto): $(head -c 240 $O/walkw_RESULT.txt)" >/dev/null 2>&1
log "$(cat $O/walkw_RESULT.txt)"
