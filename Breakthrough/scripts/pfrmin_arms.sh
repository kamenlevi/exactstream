#!/bin/sh
# R-41E PFRMIN in ONE quiet lock (runner_pfrmin.sh's step); stops at the first failure:
#   1. hashes: scripts/cand_check.py CC_MODE=hash on candidate 9c3057ce62 (BT_DIR = scratchpad btp) at its defaults
#      (engine v3o, M-01, B12_PFR_MIN 512): p2k / p8k / write16 must equal R-48's hashes; p2k must now take PFR
#   2. the short-prompt spike, each arm its own process (scripts/v3o_creep.py, agent/write14 + agent/edit10, 10 s idle
#      gaps): arm a4096 = B12_PFR_MIN=4096 (the old rule), arm b512 = the candidate's default
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/pfrmin; Q=$HOME/.local/share/quiet
BTP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btp
date '+%H:%M:%S start'
[ "$(shasum -a 256 $BTP/bt_b12.py | cut -c1-10)" = 9c3057ce62 ] || { echo "candidate is not 9c3057ce62"; exit 2; }
while [ -e $Q/PAUSE ]; do sleep 10; done
env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u B12_M01 -u BIGMOE_M01 -u B12_FRS_NOCYCLE \
  -u B12_PFR_MIN BT_DIR=$BTP CC_MODE=hash CC_OUT=$OUT $PY scripts/cand_check.py pfrmin_hash > "$OUT/pfrmin_hash.txt" 2>&1; rc=$?
grep -E "RESULT|VERDICT" "$OUT/pfrmin_hash.txt" | cut -c1-300
[ $rc -eq 0 ] && grep -q "HASH VERDICT PASS" "$OUT/pfrmin_hash.txt" || { echo "hash check FAILED (rc $rc)"; tail -5 "$OUT/pfrmin_hash.txt"; exit 3; }
grep -E "p2k RESULT" "$OUT/pfrmin_hash.txt" | grep -q "PFR yes" || { echo "p2k did not take PFR: the screen does not cover the new path"; exit 3; }
date '+%H:%M:%S hashes PASS'
for arm in a4096 b512; do
  while [ -e $Q/PAUSE ]; do sleep 10; done
  case $arm in a4096) M=4096;; b512) M=512;; esac
  B12_PFR_MIN=$M F12_BT=$BTP/bt_b12.py F12_ENGINE=v3o F12_SET=agent F12_KEYS=agent/write14,agent/edit10 F12_GAP=10 \
    F12_OUT=$OUT $PY scripts/v3o_creep.py spike_$arm > "$OUT/spike_$arm.txt" 2>&1 || { echo "$arm failed:"; tail -5 "$OUT/spike_$arm.txt"; exit 1; }
  date "+%H:%M:%S $arm done"
  grep -E "RESULT row" "$OUT/spike_$arm.txt" | grep -oE "agent/[a-z0-9]+ RESULT row [0-9]+|suite tokens equal [A-Za-z]+|prefill [0-9.]+ s, PFR [a-z]+, prefill max [0-9.]+ \(rise [-+0-9.]+\)|max [0-9.]+ \(rise [-+0-9.]+\)|lifetime peak [0-9.]+|[0-9.]+ s decode" | paste -s -d' ' - | sed 's/agent\//\n  agent\//g'
done
