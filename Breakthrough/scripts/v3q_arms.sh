#!/bin/sh
# R-45V v3q (Breakthrough Sun 1 Nov 04:5x, from Fit12's R-45U; lines in SCOREBOARD, written before the run): engine v3q = v3p +
# one shared bounce pool (BigMoE 9f6c4f5's design; exact). On a SHARED lock (FAST INTEGRATION: an exactness + memory
# screen), waiting for no HOLD / PAUSE, >= 20 GB available and a granted share (refused while a quiet run holds the Mac):
#   1. build v3q/ext/build
#   2. hashes: cand_check CC_MODE=hash on candidate 9250688a7a with B12_MODE=exact (fold-exact: R-48's hashes)
#   3. creep, fast mode (the live defaults), one process per arm on agent write14 / edit10 / write13 x 2 (10 s gaps,
#      tokens vs b12llff3): arm v3p (live cf8bd7840a), arm v3q (the candidate)
# Swap guard: an arm that grows swap by > 2 GB is stopped. Log: data/v3q/arms.txt
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/v3q; Q=$HOME/.local/share/quiet
BTQ=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btq
KEYS=agent/write14,agent/edit10,agent/write13,agent/write14,agent/edit10,agent/write13
mkdir -p $BTQ $OUT && cp -p scripts/_bt_b12_v3q.py $BTQ/bt_b12.py
[ "$(shasum -a 256 $BTQ/bt_b12.py | cut -c1-10)" = 9250688a7a ] || { echo "candidate is not 9250688a7a"; exit 2; }
avail() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub("\\.", "", $NF); s += $NF} END {print int(s * 16384 / 1e9)}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
until [ ! -e $Q/HOLD ] && [ ! -e $Q/PAUSE ] && [ "$(avail)" -ge 20 ] && \
      peer lock share 25m --cpu 400 --gpu 50 "Breakthrough R-45V v3q bounce pool: build + hashes + 2 creep arms (~15 min, ~11.5 GB)" > /dev/null 2>&1; do sleep 60; done
date "+%H:%M:%S share granted (available $(avail) GB, swap $(swap_mb) MB)"
SO=v3q/ext/build/bigmoe_ext.cpython-312-darwin.so
cmake -S v3q/ext -B v3q/ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" \
  -DNB_CMAKE_DIR="$HOME/BigMoE/.nb215/nanobind/cmake" > v3q/ext/build.cmake.log 2>&1 \
  && cmake --build v3q/ext/build -j 4 > v3q/ext/build.log 2>&1 && [ -f $SO ] || { echo "v3q build failed"; grep -E error v3q/ext/build.log | head; peer unlock; exit 2; }
date "+%H:%M:%S built v3q"
run() {   # run NAME CMD...: the command in the background, stopped on +2 GB swap
  n=$1; shift
  SW0=$(swap_mb); ( "$@" > "$OUT/$n.txt" 2>&1 ) & P=$!
  while kill -0 $P 2>/dev/null; do sleep 10
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then echo "swap grew > 2 GB: stopping $1"; kill $(tree $P) 2>/dev/null; sleep 5; kill -9 $(tree $P) 2>/dev/null; fi
  done; wait $P
}
env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u B12_M01 -u BIGMOE_M01 -u B12_FRS_NOCYCLE -u B12_PFR_MIN \
  B12_MODE=exact BT_DIR=$BTQ CC_MODE=hash CC_OUT=$OUT $PY scripts/cand_check.py v3q_hash > "$OUT/v3q_hash.txt" 2>&1; rc=$?
grep -E "RESULT|VERDICT" "$OUT/v3q_hash.txt" | cut -c1-260
[ $rc -eq 0 ] && grep -q "HASH VERDICT PASS" "$OUT/v3q_hash.txt" || { echo "hash check FAILED (rc $rc)"; tail -5 "$OUT/v3q_hash.txt"; peer unlock; exit 3; }
date "+%H:%M:%S hashes PASS"
for arm in v3p v3q; do
  case $arm in v3p) BT=$HOME/Breakthrough/scripts/bt_b12.py;; v3q) BT=$BTQ/bt_b12.py;; esac
  ( unset B12_ENGINE B12_MODE B12_PFR_MIN
    export F12_ARM=def F12_BT=$BT F12_SET=agent F12_KEYS=$KEYS F12_REF=b12llff3 F12_GAP=10 F12_OUT=$OUT
    run creep_$arm $PY scripts/v3o_creep.py creep_$arm )
  date "+%H:%M:%S $arm done (rc $?)"
  grep -E "SUMMARY" "$OUT/creep_$arm.txt" | cut -c1-500
  grep -E "RESULT row" "$OUT/creep_$arm.txt" | grep -oE "row [0-9]+|suite tokens equal [A-Za-z]+|after gc: footprint [0-9.]+|bounce pool \[[0-9, ]+\]" | paste -sd' ' -
done
peer unlock
