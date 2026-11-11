#!/bin/bash
# waiter.sh - Strata head-to-head, unattended after CERT (Sat 7 Nov 23:5x; the orchestrator's; Kamen OK'd Strata + MTP).
# Waits until (a) both IQ3_XXS shards are complete and sha256-verified, (b) EvalSuite's CERT has ended, (c) >= 45 GB of disk is
# free and (d) the Mac is calm (no PAUSE file, no soft_end.sh, lid open or an external display: EvalSuite runner 55's rule); then
> (a quote from the project owner, kept private)
# 20 s later `peer wait --for bench` (gated waiters can start together), then driver.sh, report.py, `peer unlock`.
# If the driver stops for PAUSE / lid / soft_end (4) or for lock time (6), it unlocks, waits for the conditions again, re-queues
# and resumes (finished arm+pass runs are kept), at most H2H_ATTEMPTS times; no new start after H2H_START_BY.
#
# START IT as the orchestrator's own background task (attached: a detached runner's quiet lock lapses; `peer lock` must run
# under the session that holds it), from the orchestrator's usual working folder:
#     bash /Users/kamen/StrataEval/h2h/waiter.sh
# Look at the conditions once (no lock, no hashing, nothing started):  bash /Users/kamen/StrataEval/h2h/waiter.sh --check
# While it runs, the orchestrator's own session must not take or queue another peer lock (one lock per session; the waiter's
# `peer unlock` would also drop it).  Stop it: kill <waiter pid> (TERM: it stops the driver and the engine, then unlocks).
# Log: ~/StrataEval/h2h/logs/waiter.log (its last line says how it ended); results: ~/StrataEval/h2h/results/REPORT.md.
set -u
H=/Users/kamen/StrataEval/h2h
M=/Users/kamen/StrataEval/models/IQ3_XXS
EV=${H2H_EV:-/Users/kamen/EvalSuite}                       # (H2H_EV, H2H_RUNNER_PAT, H2H_LOG, H2H_SETTLE_S: test hooks only)
RUNNER_PAT=${H2H_RUNNER_PAT:-'day_runner55[.]sh$'}
PY=/Users/kamen/StrataEval/src/.venv/bin/python
mkdir -p "$H/logs" "$H/state"
LOG=${H2H_LOG:-$H/logs/waiter.log}
SETTLE=${H2H_SETTLE_S:-20}
RESULTS=${H2H_OUT:-$H/results}
DL_LOG=${H2H_DL_LOG:-/private/tmp/claude-501/-Users-kamen-OjoX/3c9dd883-f531-48f1-879c-0636665ab84b/tasks/bxpa8ksw1.output}
CERT_START=$(date -j -f "%Y-%m-%d %H:%M" "${H2H_CERT_START:-2026-11-08 01:30}" +%s)
START_BY=$(date -j -f "%Y-%m-%d %H:%M" "${H2H_START_BY:-2026-11-08 14:00}" +%s)
LOCK_MIN=${H2H_LOCK_MIN:-120}
> (a quote from the project owner, kept private)
MAX_ATTEMPTS=${H2H_ATTEMPTS:-3}
DISK_MIN=${H2H_DISK_MIN:-45}
MIN_AVAIL=${H2H_MIN_AVAIL_GB:-0}                           # 11-08: the driver's avail_gb must show this before the lock is queued (0 = no gate)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
say() { if [ $CHECK = 1 ]; then echo "$*"; else echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; fi; }

quiet_held() { peer ls </dev/null 2>/dev/null | grep -q "benchmarking: quiet lock until"; }
runner_pids() { ps -Ao pid=,command= | awk -v pat="$RUNNER_PAT" '$2 ~ /(^|\/)(ba)?sh$/ && $3 ~ pat {print $1}' | tr '\n' ' '; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
calm_state() {
  [ -e "$HOME/.local/share/quiet/PAUSE" ] && { echo "PAUSE file present"; return 1; }
  ps -Ao command | grep -qE '^/bin/zsh [^ ]*soft_end\.sh' && { echo "soft_end.sh running"; return 1; }
  lid_shut && { echo "lid closed, no external display"; return 1; }
  echo "calm"; return 0
}
disk_state() {
  local f; f=$(df -g /Users/kamen/StrataEval | awk 'NR==2 {print $4}')
  echo "$f GB free (need $DISK_MIN)"; [ "$f" -ge "$DISK_MIN" ]
}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
mem_state() {
  local a; a=$(avail_gb); echo "$a GB available (need $MIN_AVAIL)"; awk -v a="$a" -v m="$MIN_AVAIL" 'BEGIN {exit !(a >= m)}'
}
RUNNER_SEEN=0      # set in the main shell (cert_state runs in a $(...) subshell and only reads it)
cert_state() {
  local r; r=$(runner_pids)
  if [ -n "${r// /}" ]; then echo "CERT runner 55 present (pid $r): waiting"; return 1; fi
  if [ -f "$EV/results/CERT_DONE" ] && [ "$(stat -f %m "$EV/results/CERT_DONE")" -ge "$CERT_START" ]; then
    echo "ended: results/CERT_DONE says '$(head -c 100 "$EV/results/CERT_DONE" | tr '\n' ' ')'"; return 0
  fi
  if [ "$RUNNER_SEEN" = 1 ]; then echo "ended: runner 55 is gone (no CERT_DONE newer than the 01:30 slot)"; return 0; fi
  echo "not seen: no runner 55 process and no CERT_DONE since the 01:30 slot: waiting"; return 1
}
model_state() {   # both shards verified: the download log's OK lines, or this waiter's own size + sha256 check
  local name size oid n=0 ok=0 own
  while read -r name size oid; do
    n=$((n + 1))
    if grep -F "$name OK (" "$DL_LOG" 2>/dev/null | grep -q "sha256 matches"; then ok=$((ok + 1)); continue; fi
    own=$H/state/$name.sha256ok
    if [ -f "$own" ] && [ "$(cat "$own")" = "$oid" ]; then ok=$((ok + 1)); continue; fi
    [ "$(stat -f %z "$M/$name" 2>/dev/null || echo 0)" = "$size" ] || continue
    [ $CHECK = 1 ] && continue
    ps -Ao command | grep -F -- "-o $M/$name" | grep -q '^curl ' && continue          # still being written
    quiet_held && continue                                                            # never hash 76 GB during a measurement
    say "model: $name has its full size but no OK line in $DL_LOG: hashing it here (background priority)"
    local sum; sum=$(taskpolicy -b shasum -a 256 "$M/$name" | cut -d' ' -f1)
    if [ "$sum" = "$oid" ]; then echo "$oid" > "$own"; ok=$((ok + 1)); say "model: $name sha256 OK (own check)"
    else say "MODEL SHA MISMATCH: $name $sum != $oid"; echo "MISMATCH"; return 2; fi
  done < "$M/files.txt"
  echo "$ok / $n shards verified"; [ "$n" -gt 0 ] && [ "$ok" = "$n" ]
}

RUNNER0=$(runner_pids)
[ -n "${RUNNER0// /}" ] && RUNNER_SEEN=1
if [ $CHECK = 1 ]; then
  echo "(a) model: $(model_state)"; echo "(b) CERT:  $(cert_state)"; echo "(c) disk:  $(disk_state)"; echo "(d) calm:  $(calm_state)"; echo "(e) mem:   $(mem_state)"
  echo "deadline for a new start: $(date -r "$START_BY" '+%a %H:%M'); lock: $LOCK_MIN min, \"$WHY\""
  exit 0
fi

LOCKED=0; QUEUED=0; DPID=""; LPID=""
cleanup() {
  if [ -n "$DPID" ] && kill -0 "$DPID" 2>/dev/null; then kill -TERM -- -"$DPID" 2>/dev/null || kill -TERM "$DPID"; wait "$DPID" 2>/dev/null; fi
  if [ -n "$LPID" ] && kill -0 "$LPID" 2>/dev/null; then kill -INT "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; fi
  if [ "$LOCKED" = 1 ] || [ "$QUEUED" = 1 ]; then peer unlock >> "$LOG" 2>&1; say "peer unlock (cleanup)"; LOCKED=0; QUEUED=0; fi
}
trap 'say "WAITER END: stopped by a signal"; cleanup; exit 130' INT TERM
set -m
say "waiter start (pid $$): runner 55 at start: ${RUNNER0:-none}; download log $DL_LOG; CERT slot $(date -r "$CERT_START" '+%a %H:%M'); no new start after $(date -r "$START_BY" '+%a %H:%M')"
attempt=0; last=""; lastlog=0
while :; do
  now=$(date +%s)
  if [ "$now" -gt "$START_BY" ]; then say "WAITER END: GAVE UP: not started (or not resumed) before $(date -r "$START_BY" '+%a %H:%M')"; exit 0; fi
  ma=$(model_state); mrc=$?
  [ $mrc = 2 ] && { say "WAITER END: STOPPED: model sha256 mismatch"; exit 2; }
  rp=$(runner_pids); [ -n "${rp// /}" ] && RUNNER_SEEN=1
  ce=$(cert_state); crc=$?
  di=$(disk_state); drc=$?
  ca=$(calm_state); carc=$?
  me=$(mem_state); merc=$?
  key="$ma|$crc|$drc|$ca|$merc"
  if [ "$key" != "$last" ] || [ $((now - lastlog)) -ge 600 ]; then say "waiting: model [$ma] | CERT [$ce] | disk [$di] | [$ca] | mem [$me]"; last=$key; lastlog=$now; fi
  if [ $mrc = 0 ] && [ $crc = 0 ] && [ $drc = 0 ] && [ $carc = 0 ] && [ $merc = 0 ]; then
    attempt=$((attempt + 1))
    say "conditions met (attempt $attempt): queueing: peer lock quiet --queue ${LOCK_MIN}m \"$WHY\""
    QUEUED=1
    peer lock quiet --queue "${LOCK_MIN}m" "$WHY" >> "$LOG" 2>&1 &
    LPID=$!
    wait "$LPID"; lrc=$?; LPID=""; QUEUED=0
    if [ $lrc != 0 ]; then
      say "the queue ended without the lock (rc $lrc): re-queueing after the conditions are checked again"
      attempt=$((attempt - 1)); sleep 30; continue
    fi
    LOCKED=1
    export H2H_LOCK_UNTIL=$(( $(date +%s) + LOCK_MIN * 60 )) H2H_LOCK_WHY="$WHY"
    say "quiet lock held until $(date -r "$H2H_LOCK_UNTIL" '+%H:%M'); ${SETTLE} s, then peer wait --for bench"
    sleep "$SETTLE"
    peer wait --for bench --timeout 15m >> "$LOG" 2>&1 || say "peer wait --for bench did not clear in 15 min: going on (the run records it)"
    bash "$H/driver.sh" >> "$LOG" 2>&1 &
    DPID=$!
    wait "$DPID"; rc=$?; DPID=""
    say "driver rc $rc"
    "$PY" -I "$H/report.py" --results "$RESULTS" > "$H/logs/report.out" 2>&1; say "report rc $? ($RESULTS/REPORT.md)"
    peer unlock >> "$LOG" 2>&1; LOCKED=0; say "peer unlock"
    case $rc in
      0) say "WAITER END: DONE ($RESULTS/REPORT.md)"; exit 0 ;;
      4|6) if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then say "driver stopped ($rc): waiting to resume"; sleep 60; continue; fi
           say "WAITER END: INCOMPLETE after $attempt attempts (driver rc $rc)"; exit 0 ;;
      *) say "WAITER END: STOPPED (driver rc $rc; see $H/results/driver.log)"; exit 0 ;;
    esac
  fi
  sleep 60
done
