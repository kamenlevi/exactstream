#!/bin/sh
# Breakthrough: the post-R-41D bt_b12 candidate check (NewIdeas Thu 18:3x, item 1; lines in SCOREBOARD, written before the
# run). Candidate = $1/bt_b12.py (= ce6d4c63d2, the R-45 fix, + the M-01 default: engine v3n, BIGMOE_M01=1), at ITS defaults:
#   cand_hash  p2k, p8k, write16 (cand_check.py CC_MODE=hash): every hash = R-48's (= the fold-exact regression's) and
#              write16's tokens = the suite's b12llfx row
#   cand_rows  R-41C's 16 math rows in ONE process (CC_MODE=rows): tokens = b12llfx on all 16, lifetime peak <= 11.85 GB.
#              ~15 min of decode, so the process spans quiet locks: before a row that would not end inside the lock it
#              writes data/cand/cand_rows.wait and idles (no GPU work); this runner then unlocks, queues the next lock and
#              writes cand_rows.go with the new deadline. No new lock after 23:15 (the silent night): the run is stopped.
#   then, if scripts/r49_capture_hook.sh exists and >= 5 min of the lock are left: that hook as a step (R-49's 12 GB
#   routing traces; a separate file so this script never changes while it runs)
# Log: data/cand/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/cand; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=$1
[ -f "$SP/bt_b12.py" ] || { echo "usage: runner_cand.sh <dir with the candidate bt_b12.py>"; exit 2; }
CLEAN="env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u B12_M01 -u BIGMOE_M01 -u B12_FRS_NOCYCLE"
log "runner_cand start (pid $$), candidate $SP/bt_b12.py"
caffeinate -i -w $$ &
until_ts() { $PY -I -S -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["until"]))' "$LOCKF"; }
lock() {   # lock MIN WHY: the held lock (HELD=1, first call only) or wait until ok, queue one, check ok again
  if [ "${HELD:-0}" = 1 ] && keepalive; then HELD=0; log "on the lock the session already holds"; return 0; fi
  until ok; do sleep 30; done
  log "queued ($1 min): $2"
  PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue "$1"m "Breakthrough $2" || { log "no lock"; return 1; }
  ok || { log "lock came with PAUSE / ACT / ramp / out of hours, released"; peer unlock; return 1; }
}
lock 15 "post-R-41D bt_b12 candidate: hashes + R-41C's 16 math rows in one process (~18 min over 2 locks)" || exit 1
U=$(until_ts)
step cand_hash 300 $CLEAN BT_DIR=$SP CC_MODE=hash $PY scripts/cand_check.py cand_hash || { peer unlock; log "runner_cand stopped"; exit 1; }
name=cand_rows
rm -f "$OUT/cand_rows.wait" "$OUT/cand_rows.go" "$OUT/cand_rows.jsonl"
log "start cand_rows (deadline in $((U - 45 - $(date +%s))) s)"
( $CLEAN BT_DIR=$SP CC_MODE=rows CC_NROWS=16 CC_DEADLINE=$((U - 45)) $PY scripts/cand_check.py cand_rows \
  > "$OUT/cand_rows.txt" 2>&1 ) & P=$!
SW0=$(swap_mb); i=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5
  if [ -e "$OUT/cand_rows.wait" ]; then
    log "cand_rows idles before row $(cut -d' ' -f1 "$OUT/cand_rows.wait") (needs ~$(cut -d' ' -f2 "$OUT/cand_rows.wait") s)"
    peer unlock
    if [ "$(date +%H%M)" -gt 2315 ]; then stop "no time for the next lock before the silent night"; rc=125; break; fi
    if ! lock 10 "post-R-41D candidate: the rest of R-41C's 16 math rows (the same process)"; then
      stop "no next lock"; rc=125; break
    fi
    U=$(until_ts); rm -f "$OUT/cand_rows.wait"
    echo $((U - 45)) > "$OUT/cand_rows.go.tmp"; mv "$OUT/cand_rows.go.tmp" "$OUT/cand_rows.go"
    log "cand_rows go (deadline in $((U - 45 - $(date +%s))) s)"; SW0=$(swap_mb); i=0
    continue
  fi
  if paused; then continue; fi
  i=$((i + 5))
  if [ $((i % 120)) -eq 0 ]; then keepalive || { stop "lost the lock"; rc=125; break; }; fi
  SW=$(swap_mb); if [ $((SW - SW0)) -gt 2048 ]; then stop "swap grew ${SW0} -> ${SW} MB"; rc=126; break; fi
  if act; then stop "health monitor at ACT"; rc=127; break; fi
  if [ "$(date +%s)" -gt $((U + 30)) ]; then stop "past the lock's end"; rc=124; break; fi
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
log "done cand_rows (rc $rc): $(grep -E 'ROWS VERDICT|Traceback|Error' "$OUT/cand_rows.txt" | tail -2 | cut -c1-300 | tr '\n' ' ')"
LEFT=$((U - $(date +%s)))
if [ $rc -eq 0 ] && [ -x scripts/r49_capture_hook.sh ] && [ $LEFT -ge 300 ] && keepalive; then
  step r49_capture $((LEFT - 30)) scripts/r49_capture_hook.sh
fi
peer unlock
log "cand_hash: $(grep -h 'HASH VERDICT' "$OUT/cand_hash.txt" | cut -c1-200) | cand_rows: $(grep -h 'ROWS VERDICT' "$OUT/cand_rows.txt" | cut -c1-260)"
log "runner_cand done"
