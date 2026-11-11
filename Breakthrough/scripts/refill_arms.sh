#!/bin/sh
# R-41F REFILL in ONE quiet lock (runner_refill.sh's step); stops at the first failure:
#   1. hashes: scripts/cand_check.py CC_MODE=hash on candidate b9d5b491ad (BT_DIR = scratchpad btr) at its defaults
#      (engine v3o, M-01, B12_PFR_MIN 512, B12_PFR_REFILL 1): p2k / p8k / write16 = R-48's; all three take PFR + refill
#   2. arms, one process each (scripts/v3o_creep.py on agent/write14, agent/edit10, agent/text23 with 10 s idle gaps):
#      A = today's live behaviour (B12_PFR_MIN=4096 B12_PFR_REFILL=0: short rows without PFR, text23 with PFR, no
#      refill), B = the candidate's defaults, A2 = A again if >= 240 s are left; then scripts/refill_sum.py (the verdict)
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/refill; Q=$HOME/.local/share/quiet
BTR=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btr
KEYS=agent/write14,agent/edit10,agent/text23
T0=$(date +%s); LIM=${RF_LIM:-840}
date '+%H:%M:%S start'
mkdir -p $BTR && cp -p scripts/_bt_b12_refill.py $BTR/bt_b12.py
[ "$(shasum -a 256 $BTR/bt_b12.py | cut -c1-10)" = b9d5b491ad ] || { echo "candidate is not b9d5b491ad"; exit 2; }
while [ -e $Q/PAUSE ]; do sleep 10; done
env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u B12_M01 -u BIGMOE_M01 -u B12_FRS_NOCYCLE \
  -u B12_PFR_MIN -u B12_PFR_REFILL BT_DIR=$BTR CC_MODE=hash CC_OUT=$OUT $PY scripts/cand_check.py rf_hash > "$OUT/rf_hash.txt" 2>&1; rc=$?
grep -E "RESULT|VERDICT" "$OUT/rf_hash.txt" | cut -c1-330
[ $rc -eq 0 ] && grep -q "HASH VERDICT PASS" "$OUT/rf_hash.txt" || { echo "hash check FAILED (rc $rc)"; tail -5 "$OUT/rf_hash.txt"; exit 3; }
[ "$(grep -E ' RESULT ' "$OUT/rf_hash.txt" | grep -c 'PFR yes, refill \[')" = 3 ] || { echo "not every prompt took PFR + refill: the screen does not cover the new path"; exit 3; }
date '+%H:%M:%S hashes PASS'
for arm in A B A2; do
  if [ $arm = A2 ] && [ $(( $(date +%s) - T0 )) -gt $((LIM - 240)) ]; then echo "A2 skipped: under 240 s left"; break; fi
  while [ -e $Q/PAUSE ]; do sleep 10; done
  case $arm in A|A2) M=4096; R=0;; B) M=512; R=1;; esac
  B12_PFR_MIN=$M B12_PFR_REFILL=$R F12_BT=$BTR/bt_b12.py F12_ENGINE=v3o F12_SET=agent F12_KEYS=$KEYS F12_GAP=10 F12_OUT=$OUT \
    $PY scripts/v3o_creep.py rf_$arm > "$OUT/rf_$arm.txt" 2>&1 || { echo "$arm failed:"; tail -5 "$OUT/rf_$arm.txt"; exit 1; }
  date "+%H:%M:%S $arm done"
done
$PY scripts/refill_sum.py "$OUT"
