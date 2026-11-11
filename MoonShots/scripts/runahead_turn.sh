#!/bin/bash
# M-16 RUNAHEAD step 2: one 45 GB-route quiet turn for scripts/runahead_live.py (the live dual pass at C 224). BigMoE's turn45.sh
# pattern (queue the quiet ticket first, check memory after the grant) + lpom_runner.sh's guards, logs in MoonShots only:
#   wait for NOTBEFORE, no PAUSE / closed lid / ACT verdict / soft_end, NO-SWAP >= PEAK + MARGIN before queueing;
#   PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue; after the grant settle 30 s, need NO-SWAP >= PEAK + MARGIN (else release,
#   wait 2 min, queue again; give up after MAXFAILS); run with keepalive, a time cap, PAUSE / lost lock / swap +2 GB stops the
#   whole process group; release the ticket.
# Run as a harness background task (a nohup'd queue dies): bash scripts/runahead_turn.sh [C=224]
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
CL=${1:-224}                                            # one or more C values, e.g. "224 256": run in order under one ticket
C=${CL%% *}
peak() { awk -v c=$1 'BEGIN { printf "%d", 33.2 + (c - 192) * 0.14 + 1.5 }'; }   # BigMoE's c45 peak + the harness
PEAK=$(for c in $CL; do peak $c; echo; done | sort -n | head -1)   # the grant gate keeps the ticket if ANY C fits;
MARGIN=${MARGIN:-8}                                    # the per-C check below skips the ones that don't (orchestrator 23:1x: MARGIN=4)
D="$HOME/MoonShots"; PY="$HOME/BigMoE/.venv/bin/python"; Q="$HOME/.local/share/quiet"
OUT="$D/data/runahead"; mkdir -p "$OUT"; LOG="$OUT/turn.log"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
WHY="MoonShots M-16 RUNAHEAD step 2 (45 GB route, C $C, ~10 min)"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
blocked() { [ -e "$Q/PAUSE" ] || ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' \
            || grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null || pgrep -f "quiet/soft_end.sh" >/dev/null; }
busy45() { peer ls 2>/dev/null | grep -E "benchmarking: quiet lock|waiting for a quiet" | grep -v "MoonShots" | grep -ciE "45 GB|S45|C45|C 224|C 256|C 160|D-23|REF2c|45-route"; }
lockok() { /usr/bin/python3 -I -S -c '
import json, sys, time
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$LOCKF"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
say "start: C $C, PEAK $PEAK GB + $MARGIN, NOTBEFORE ${NOTBEFORE:-none}"
fails=0
while :; do
  # NewIdeas 12:4x: queue at NOTBEFORE (23:01), FIFO behind BigMoE C45 and MTP Head D-23c; no wait for others' tickets and
  # no memory wait before queueing (the FIFO and the NO-SWAP check after the grant keep the 45 GB processes apart)
  blocked && { sleep 60; continue; }
  [ -n "$NOTBEFORE" ] && [ "$(date +%H%M)" -lt "$(echo "$NOTBEFORE" | tr -d :)" ] && [ "$(date +%H%M)" -ge 1200 ] && { sleep 20; continue; }
  if [ $fails -eq 0 ] && lockok; then                 # the session took the ticket before starting this (the peers hook
    say "quiet lock already held by this session (NO-SWAP $(avail_gb) GB)"   # blocks starting a program in others' locks)
  else
    say "queue the quiet ticket (NO-SWAP $(avail_gb) GB)"
    PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 30m "$WHY" >> "$LOG" 2>&1 || { say "queue failed: retry in 2 min"; sleep 120; continue; }
  fi
  sleep 30
  a=$(avail_gb)
  if [ "$a" -lt $((PEAK + MARGIN)) ]; then
    fails=$((fails + 1)); peer unlock >> "$LOG" 2>&1
    [ $fails -ge ${MAXFAILS:-2} ] && { say "granted $fails time(s) with only $a GB NO-SWAP: giving up"; exit 8; }
    say "granted but only $a GB NO-SWAP (< $((PEAK + MARGIN))): released, wait 2 min"; sleep 120; continue
  fi
  break
done
say "granted, NO-SWAP $a GB: run C list $CL"
sh "$D/scripts/keepalive_while.sh" $$ & GL=$!
set -m
T0=$(date +%s); rc=0
for C in $CL; do
  need=$(( $(peak $C) + MARGIN ))
  [ -f "$Q/PAUSE" ] && { say "PAUSE: stop before C $C"; break; }
  lockok || { say "lost the lock before C $C"; break; }
  a=$(avail_gb)
  [ "$a" -lt "$need" ] && { say "C $C skipped: NO-SWAP $a GB < $need"; continue; }
  [ $(( $(date +%s) - T0 )) -gt 1200 ] && { say "C $C skipped: the ticket's 30 min are nearly used"; break; }
  S0=$(swapmb); say "run C $C (NO-SWAP $a GB, swap $S0 MB)"
  env R57_PRESET=45 BIGMOE_EXT_BUILD=build "$PY" "$D/scripts/runahead_live.py" $C 256 > "$OUT/ra_${C}.log" 2>&1 &
  P=$!
  i=0; why=""
  while kill -0 $P 2>/dev/null; do
    sleep 5; i=$((i + 5))
    if [ $i -ge ${CAPS:-1200} ]; then why="time cap"; break; fi
    if [ -f "$Q/PAUSE" ]; then why="PAUSE"; break; fi
    if [ $((i % 30)) -eq 0 ] && ! lockok; then why="lost the lock"; break; fi
    if [ $(( $(swapmb) - S0 )) -gt 2048 ]; then why="swap +2 GB"; break; fi
  done
  if [ -n "$why" ] && kill -0 $P 2>/dev/null; then
    kill -- -$P 2>/dev/null; sleep 3; kill -9 -- -$P 2>/dev/null
    say "run C $C stopped: $why"
  fi
  wait $P 2>/dev/null; rc=$?
  say "done C $C rc=$rc: $(grep -E '^Lines|footprint|Traceback|Error' "$OUT/ra_${C}.log" | cut -c1-200 | tr '\n' ' ')"
  [ -n "$why" ] && break
  sleep 20
done
kill $GL 2>/dev/null
peer unlock >> "$LOG" 2>&1
say "turn over, ticket released"
exit $rc
