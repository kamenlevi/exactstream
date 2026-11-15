#!/bin/bash
# run_c55_engine.sh (Breakthrough, CHECK55 step 0 on the full engine; the orchestrator booked ~04:15 Mon, after EvalSuite's
# close-call base in the 45 GB block). Waits for BOTH ~/EvalSuite/results/CARD_lx45cc1_DONE and CARD_lx45cc1_rev_DONE
# written after this runner started, or EvalSuite's chain-end marker ~/EvalSuite/results/CHAIN_1116_END (written on any
# end of its Sunday chain, after its processes exit and unlock; EvalSuite 11:4x), or ~/Breakthrough/data/m1/check55/
# engine/GO (a manual trigger), until the deadline.
# Then: the head test (catches a half-edited peer file), a QUIET ticket, the NO-SWAP gate (>= 39 GB = peak ~31 + 8; up to
# 10 min under the ticket), the run (r57_c55.py 160 256 with c55_engine_env.sh), swap guard +2 GB, a 20-min cap, unlock.
# Usage: run_c55_engine.sh [deadline "YYYY-mm-dd HH:MM"]   (default 2026-11-16 15:00; the orchestrator 14:5x: heat stretches
# the night chain ~1.9x, apps stay closed to 15:00 on school days). The wall cap is 60 min (fanwatch freezes count on it).
cd ~/Breakthrough || exit 9
H=~/Breakthrough/scripts/m1/check55
D=~/Breakthrough/data/m1/check55/engine
mkdir -p "$D"
L=$D/run_c55_engine.log
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:-2026-11-16 15:00}:00" +%s) || { say "bad deadline"; exit 2; }
T0=$(date +%s)
E=~/EvalSuite/results
newer() { [ -f "$1" ] && [ "$(stat -f %m "$1")" -ge "$T0" ]; }
say "armed (pid $$): waits for CARD_lx45cc1_DONE + CARD_lx45cc1_rev_DONE newer than now, or CHAIN_1116_END, or $D/GO; deadline $(date -r "$DL" '+%a %H:%M')"
while true; do
  if newer "$E/CARD_lx45cc1_DONE" && newer "$E/CARD_lx45cc1_rev_DONE"; then say "close-call base DONE (both orders)"; break; fi
  if newer "$E/CHAIN_1116_END"; then say "EvalSuite's chain ended (CHAIN_1116_END: its processes exited, unlocked)"; break; fi
  if [ -f "$D/GO" ]; then say "manual GO file"; break; fi
  if [ "$(date +%s)" -ge "$DL" ]; then say "SKIP: deadline reached without the trigger"; exit 4; fi
  sleep 60
done
source "$H/c55_engine_env.sh"
if ! ~/BigMoE/.venv/bin/python -I -B "$H/c55_headtest.py" >> "$L" 2>&1; then say "SKIP: the head test failed (log above)"; exit 5; fi
say "head test PASS; queueing a quiet ticket"
export PEER_QUEUE_TIMEOUT=21600
peer lock quiet --queue 40m "Breakthrough: CHECK55 full-engine profile (exact-45 loop C 160, MEM ~31 GB, ~5 min)" >> "$L" 2>&1 \
  || { say "SKIP: quiet lock not granted"; exit 3; }
sleep 20
a=$(ns)
say "quiet granted; NO-SWAP $a GB; power: $(pmset -g | grep -iE 'powermode|lowpowermode' | tr -s ' ' | paste -sd ';' -)"
end=$(( $(date +%s) + 600 ))
while [ "$a" -lt 39 ]; do
  [ "$(date +%s)" -ge $end ] && { say "SKIP: NO-SWAP $a < 39 GB for 10 min under the ticket"; peer unlock >/dev/null 2>&1; exit 6; }
  sleep 20; a=$(ns)
done
sw0=$(swmb)
~/BigMoE/.venv/bin/python -I -B "$H/r57_c55.py" 160 256 > >(grep --line-buffered -vE "^\[(tokenizer|mtplx|qwen4_exp|moe-sorted)" >> "$L") 2>&1 &
py=$!
cap=$(( $(date +%s) + 3600 ))
while kill -0 $py 2>/dev/null; do
  sleep 10
  s=$(swmb)
  if [ "$s" -gt $((sw0 + 2048)) ]; then say "swap grew > 2 GB ($sw0 -> $s MB): stop"; kill -TERM $py; sleep 5; kill -9 $py 2>/dev/null; fi
  if [ "$(date +%s)" -gt $cap ]; then say "60-min cap: stop"; kill -TERM $py; sleep 5; kill -9 $py 2>/dev/null; fi
done
wait $py; rc=$?
peer unlock >/dev/null 2>&1
say "C55-ENGINE-DONE rc $rc; swap $(swmb) MB (start $sw0)"
