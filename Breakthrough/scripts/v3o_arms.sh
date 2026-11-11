#!/bin/sh
# R-45 v3o in ONE quiet lock (runner_v3o.sh's step); stops at the first failure:
#   1. build engine v3o's ext (v3o/ext/build; nothing has loaded it) if missing or older than its source
#   2. hashes: scripts/cand_check.py CC_MODE=hash on the v3o candidate fd405861fa (BT_DIR = scratchpad bto) at its
#      defaults (engine v3o, M-01 on, records3m): p2k / p8k / write16 must equal R-48's hashes (= v3n's) and write16
#      b12llfx's tokens
#   3. creep, each arm its own process (scripts/v3o_creep.py): v3o, then v3n (live 23b42972ef) if >= 300 s are left;
#      6 agent requests (write14, edit10, write13, twice), a 10 s idle gap + gc after each
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/v3o; Q=$HOME/.local/share/quiet
BTO=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/bto
SO=v3o/ext/build/bigmoe_ext.cpython-312-darwin.so
KEYS=agent/write14,agent/edit10,agent/write13,agent/write14,agent/edit10,agent/write13
T0=$(date +%s); LIM=${V3O_LIM:-840}
date '+%H:%M:%S start'
if [ ! -f $SO ] || [ v3o/ext/bigmoe_ext.cpp -nt $SO ]; then
  cmake -S v3o/ext -B v3o/ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" \
    -DNB_CMAKE_DIR="$HOME/BigMoE/.nb215/nanobind/cmake" > v3o/ext/build.cmake.log 2>&1 \
    || { echo "cmake configure failed (v3o/ext/build.cmake.log)"; exit 2; }
  cmake --build v3o/ext/build -j 4 > v3o/ext/build.log 2>&1 || { echo "build failed:"; grep -E 'error' v3o/ext/build.log | head; exit 2; }
  [ -f $SO ] || { echo "build made no .so"; exit 2; }
  date '+%H:%M:%S built v3o ext'
fi
[ "$(shasum -a 256 $BTO/bt_b12.py | cut -c1-10)" = fd405861fa ] || { echo "candidate is not fd405861fa"; exit 2; }
while [ -e $Q/PAUSE ]; do sleep 10; done
env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u B12_M01 -u BIGMOE_M01 -u B12_FRS_NOCYCLE \
  BT_DIR=$BTO CC_MODE=hash CC_OUT=$OUT $PY scripts/cand_check.py v3o_hash > "$OUT/v3o_hash.txt" 2>&1; rc=$?
grep -E "RESULT|VERDICT|m01_on" "$OUT/v3o_hash.txt" | cut -c1-300
[ $rc -eq 0 ] && grep -q "HASH VERDICT PASS" "$OUT/v3o_hash.txt" || { echo "hash check FAILED (rc $rc)"; tail -5 "$OUT/v3o_hash.txt"; exit 3; }
date '+%H:%M:%S hashes PASS'
for arm in v3o v3n; do
  if [ $arm = v3n ] && [ $(( $(date +%s) - T0 )) -gt $((LIM - 300)) ]; then echo "v3n arm skipped: under 300 s left"; break; fi
  while [ -e $Q/PAUSE ]; do sleep 10; done
  case $arm in v3o) BT=$BTO/bt_b12.py;; v3n) BT=$HOME/Breakthrough/scripts/bt_b12.py;; esac
  F12_BT=$BT F12_ENGINE=$arm F12_SET=agent F12_KEYS=$KEYS F12_GAP=10 F12_OUT=$OUT \
    $PY scripts/v3o_creep.py ${arm}_creep > "$OUT/${arm}_creep.txt" 2>&1 || { echo "$arm creep failed:"; tail -5 "$OUT/${arm}_creep.txt"; exit 1; }
  date "+%H:%M:%S $arm creep done"
  grep -E "SUMMARY" "$OUT/${arm}_creep.txt" | cut -c1-600
done
