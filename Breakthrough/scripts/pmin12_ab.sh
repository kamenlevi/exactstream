#!/bin/sh
# PMIN12 A/B on a SHARED GPU (FAST INTEGRATION: a proof-of-concept screen; timings "shared GPU, rough"; lines in SCOREBOARD
# 'PMIN12 A/B LINES'): scripts/pmin12_ab.py on a pinned copy of the live bt_b12 (31e4333196). Waits while HOLD / PAUSE exist,
# outside 08:01-23:50 (small runs by day), and until vm_stat shows >= 21 GB available (peak ~12.3 + 8); takes `peer lock
# share` (refused while a quiet run holds the Mac: retry); stops the run if swap grows by > 2 GB (the whole process tree).
# PM_* env (mode, keys, arms, tag) passes through to pmin12_ab.py. Log: data/pmin12/run<PM_TAG>.txt (by the caller)
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/pmin12; Q=$HOME/.local/share/quiet
BTP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btpmin
mkdir -p $BTP $OUT && cp -p scripts/bt_b12.py $BTP/bt_b12.py
[ "$(shasum -a 256 $BTP/bt_b12.py | cut -c1-10)" = 31e4333196 ] || { echo "live bt_b12 is not 31e4333196"; exit 2; }
avail() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub("\\.", "", $NF); s += $NF} END {print int(s * 16384 / 1e9)}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
in_hours() { h=$(date +%H%M); [ "$h" -ge 0801 ] && [ "$h" -le 2350 ]; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
until in_hours && [ ! -e $Q/HOLD ] && [ ! -e $Q/PAUSE ] && [ "$(avail)" -ge 21 ]; do sleep 30; done
until peer lock share 18m --cpu 400 --gpu 50 "Breakthrough PMIN12 A/B: MTP p_min arms on the 12 GB line (shared GPU, ~11 min, ~12.3 GB)" > /dev/null 2>&1; do sleep 60; done
date "+%H:%M:%S start (available $(avail) GB, swap $(swap_mb) MB)"
SW0=$(swap_mb)
BT_DIR=$BTP $PY scripts/pmin12_ab.py > "$OUT/ab${PM_TAG}.txt" 2>&1 & P=$!
while kill -0 $P 2>/dev/null; do
  sleep 10
  if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then echo "swap grew > 2 GB: stopping"; kill $(tree $P) 2>/dev/null; sleep 5; kill -9 $(tree $P) 2>/dev/null; fi
done
wait $P; rc=$?
peer unlock > /dev/null 2>&1
date "+%H:%M:%S done (rc $rc)"
grep -E "SUMMARY|Traceback|Error" "$OUT/ab${PM_TAG}.txt" | cut -c1-300
