#!/bin/bash
# release/smoke_runner.sh <work dir> <start epoch> <last start epoch> [tag]
# The package's GPU smoke / final run in NewIdeas' slot (Leap, Sun 2026-11-08 19:5x): waits for <start epoch>, queues a
# quiet ticket (the first boundary of EvalSuite's chain), then gates: no PAUSE / HOLD, NO-SWAP >= 51 GB; if the gate
# fails it unlocks and tries again 15 min later until <last start epoch>. Under the lock: clean_clone_test.sh run on the
# prepared clone in <work dir>, with a swap guard (+2 GB since the start stops it) and a 25 min wall-clock cap; then
# peer unlock. Writes <work dir>/runner.log; last line RUNNER DONE / NOT STARTED / STOPPED.
W=${1:?work dir}; START=${2:?start epoch}; LAST=${3:?last start epoch}; TAG=${4:-smoke}
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
LOG="$W/runner.log"; Q="$HOME/.local/share/quiet"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }   # descendants by PID, deepest first
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%.2f", $6 / 1024}'; }
[ -x "$W/clone/.venv/bin/python" ] || { say "no prepared clone in $W: NOT STARTED"; exit 2; }
say "armed ($TAG): start $(date -r "$START" '+%a %H:%M'), last start $(date -r "$LAST" '+%a %H:%M')"
while [ "$(date +%s)" -lt "$START" ]; do sleep 30; done
while :; do
  now=$(date +%s)
  [ "$now" -le "$LAST" ] || { say "past the last start: NOT STARTED"; exit 3; }
  if [ -e "$Q/HOLD" ] || { [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; }; then
    say "HOLD / PAUSE set: wait 5 min"; sleep 300; continue
  fi
  say "queue a quiet ticket"
  PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 20m "Leap: engine package $TAG run (clean clone, 6 configs x 2 IFEval x CAP 256 + 1 text prompt, one process at a time; MEM 47 GB; ~5 min)" >> "$LOG" 2>&1
  rc=$?; [ $rc = 0 ] || { say "peer lock rc $rc: retry in 5 min"; sleep 300; continue; }
  sleep 20                                                   # gated waiters start together: look again after 20 s
  a=$(noswap)
  if [ "$a" -lt 51 ] || [ -e "$Q/HOLD" ] || { [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; }; then
    say "gate failed under the lock (NO-SWAP $a GB, need 51; or HOLD / PAUSE): unlock, retry in 15 min"
    peer unlock >> "$LOG" 2>&1; sleep 900; continue
  fi
  S0=$(swap_gb); say "granted: NO-SWAP $a GB, swap $S0 GB: run"
  bash "$W/clone/tools/clean_clone_test.sh" "$W/clone" "$W" run >> "$LOG" 2>&1 &
  jp=$!; t0=$(date +%s); why=""
  while kill -0 $jp 2>/dev/null; do
    sleep 5
    if awk -v s="$(swap_gb)" -v s0="$S0" 'BEGIN { exit !(s - s0 > 2) }'; then why="SWAP GUARD (+2 GB)"; fi
    [ $(( $(date +%s) - t0 )) -gt 1500 ] && why="25 min cap"
    if [ -n "$why" ]; then
      say "$why: stop"; tree=$(desc $jp); kill -TERM $tree $jp 2>/dev/null; sleep 5; kill -9 $tree $jp 2>/dev/null; break
    fi
  done
  wait $jp 2>/dev/null; rc=$?
  peer unlock >> "$LOG" 2>&1
  say "$(tail -1 "$W/report.txt" 2>/dev/null) (rc $rc, $(( $(date +%s) - t0 )) s, swap $(swap_gb) GB)"
  [ -n "$why" ] && { say "RUNNER STOPPED ($why)"; exit 4; }
  say "RUNNER DONE"; exit 0
done
