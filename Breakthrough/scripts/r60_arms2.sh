#!/bin/sh
# [r60_arms2: Kamen 19:5x one-hour window, deadline R60_DEADLINE (default 20:45 today), an arm starts only if R60_ARM_S (600 s) remain, killed at the deadline; R60_MATH 4] R-60 MTPDROP arms on a SHARED GPU (FAST INTEGRATION: a proof-of-concept screen; timings "shared GPU, rough"): each arm
# its own process (scripts/r60_screen.py on the candidate scripts/_bt_b12_r60.py = live cf8bd7840a + B12_MTP_HEAD), then
# scripts/r60_sum.py. Waits while HOLD / PAUSE exist, outside 08:01-23:50 (small runs by day), and until vm_stat shows
# >= 20 GB available (peak ~11.5 + 8); takes `peer lock share` per arm (refused while a quiet run holds the Mac: retry);
# stops an arm whose run grows swap by > 2 GB (the whole process tree). Log: data/r60/arms.txt
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/r60; Q=$HOME/.local/share/quiet
BTR=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btr60
mkdir -p $BTR $OUT && cp -p scripts/_bt_b12_r60.py $BTR/bt_b12.py
[ "$(shasum -a 256 $BTR/bt_b12.py | cut -c1-10)" = 68db9798c6 ] || { echo "candidate is not 68db9798c6"; exit 2; }
avail() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub("\\.", "", $NF); s += $NF} END {print int(s * 16384 / 1e9)}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
in_hours() { h=$(date +%H%M); [ "$h" -ge 0801 ] && [ "$h" -le 2350 ]; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
DL=$(date -j -f "%H:%M" "${R60_DEADLINE:-20:45}" +%s); export R60_MATH=${R60_MATH:-4}
for arm in ${R60_ARMS:-stock d22}; do
  [ $(( DL - $(date +%s) )) -ge ${R60_ARM_S:-600} ] || { echo "$arm skipped: under ${R60_ARM_S:-600} s to the deadline"; break; }
  until in_hours && [ ! -e $Q/HOLD ] && [ ! -e $Q/PAUSE ] && [ "$(avail)" -ge 20 ]; do sleep 30; [ $(( DL - $(date +%s) )) -ge ${R60_ARM_S:-600} ] || { echo "$arm: no room before the deadline"; exit 4; }; done
  until peer lock share 15m --cpu 400 --gpu 50 "Breakthrough R-60 MTPDROP arm $arm (shared GPU, ~6 min, ~11.5 GB)" > /dev/null 2>&1; do sleep 60; done
  date "+%H:%M:%S start $arm (available $(avail) GB, swap $(swap_mb) MB)"
  SW0=$(swap_mb)
  BT_DIR=$BTR R60_ARM=$arm $PY scripts/r60_screen.py > "$OUT/$arm.txt" 2>&1 & P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 10
    if [ $(date +%s) -ge $DL ]; then echo "deadline: stopping $arm"; kill $(tree $P) 2>/dev/null; sleep 5; kill -9 $(tree $P) 2>/dev/null; fi
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then echo "swap grew > 2 GB: stopping $arm"; kill $(tree $P) 2>/dev/null; sleep 5; kill -9 $(tree $P) 2>/dev/null; fi
  done
  wait $P; rc=$?
  peer unlock > /dev/null 2>&1
  date "+%H:%M:%S done $arm (rc $rc)"
  grep -E "RESULT|Traceback|Error" "$OUT/$arm.txt" | cut -c1-230
  [ $rc -eq 0 ] || { echo "$arm failed"; tail -5 "$OUT/$arm.txt"; exit 1; }
done
$PY scripts/r60_sum.py
