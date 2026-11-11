#!/bin/bash
# A 45 GB-route memory turn (the orchestrator's FAST INTEGRATION addendum, Fri 30 Oct 23:2x): 45 GB runs queue a quiet
# ticket FIRST and check memory after the grant (a queued quiet ticket stops new shared runs; the grant waits for running
# ones, so memory drains). One 45 GB ticket at a time across sessions, 15 min apart (EvalSuite's REF2b chunks get the gap):
#   1. wait while another session holds or waits for a 45 GB-route quiet ticket, and until 15 min after the last one;
#   2. wait for no PAUSE, an open lid, no ACT verdict, no soft_end ramp (not for memory);
#   3. `peer lock quiet --queue`; after the grant settle 30 s and need vm_stat available >= PEAK + 8 GB, else release,
#      wait 2 min and queue again;
#   4. run CMD under the ticket (it gets HAVE_LOCK=1 KEEP_LOCK=1), then release the ticket.
# WAITMEM=GB (BigMoE 11-01): also wait until NO-SWAP >= GB BEFORE queueing, so a day-time memory shortfall costs no grant
# (each failed grant drains everyone's shared runs); NOTBEFORE=HH:MM: queue no earlier than that time today (night runs).
# DAY9 (11-02 13:4x): NOTBEFORE / NOTAFTER are fixed to clock times at the start (NOTAFTER = its next occurrence, NOTBEFORE =
# its last occurrence before that), so an evening launch for 01:00-05:30 waits instead of reading 22:00 as past 05:30;
# AFTER='<sh condition>': queue only once it holds (another session's step first, e.g. the orchestrator's gap order).
# usage (as a harness background task, not nohup: a nohup'd queue dies): turn45.sh PEAK_GB MINUTES "why" CMD...
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
PEAK=$1; MIN=$2; WHY=$3; shift 3
MARGIN=${MARGIN:-8}                                     # GB over the peak after the grant (the orchestrator 23:4x: 4 under a quiet lock)
Q="$HOME/.local/share/quiet"
LOG="$HOME/BigMoE/data/turn45.log"
say() { echo "$(date +%H:%M:%S) [$WHY] $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }   # NO-SWAP (the orchestrator 23:5x, RULES): free + speculative + purgeable + file-backed
busy45() { peer ls 2>/dev/null | grep -E "benchmarking: quiet lock|waiting for a quiet" | grep -v "BigMoE" | grep -ciE "45 GB|S45|STACK|C 224|C 256|C 288|C 192|C 160|R-59|45-route"; }
why_blocked() { [ -e "$Q/PAUSE" ] && { echo "PAUSE"; return; }   # the first gate that holds, named (11-08: CHS4 hung 2 h silent)
  ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { echo "HEALTH ACT"; return; }
  local p; p=$(pgrep -f "quiet/soft_en[d][.]sh" | head -1); [ -n "$p" ] && echo "soft_end pid $p: $(ps -o command= -p $p | cut -c1-100)"; }
# 11-08 09:2x ROOT CAUSE of CHS4's 2-h hang: the plain pattern "quiet/soft_end.sh" also matched ANOTHER turn45's pgrep process
# (its argv holds the pattern). Two waiting turn45 loops that line up on the same second then block each other every minute,
# forever (FU + PFB at 09:13-09:18: 'soft_end pid N' with an empty command, adjacent pids). The bracket form cannot match a pgrep.
blocked() { [ -n "$(why_blocked)" ]; }
lastg=; held=0                                          # hold GATE: log the gate when it changes, then every ~30 min
hold() { if [ "$1" = "$lastg" ]; then held=$((held + 1)); else held=0; lastg=$1; fi
         [ $((held % 30)) -eq 0 ] && say "waiting: $1 ($held min)"; sleep 60; }
at_next() { local t; t=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date -r $2 +%Y-%m-%d) $1:00" +%s); [ $t -le $2 ] && t=$((t + 86400)); echo $t; }
T0=$(date +%s); NA_E=; NB_E=                           # at_next HH:MM EPOCH: that clock time's first occurrence after EPOCH
[ -n "$NOTAFTER" ] && NA_E=$(at_next $NOTAFTER $T0)
if [ -n "$NOTBEFORE" ]; then
  if [ -n "$NA_E" ]; then NB_E=$(at_next $NOTBEFORE $((NA_E - 86400)))
  else NB_E=$(at_next $NOTBEFORE $T0); [ $((NB_E - T0)) -gt 43200 ] && NB_E=$T0; fi   # alone: passed today = go now
fi
say "start: PEAK $PEAK GB, ticket $MIN min, window ${NB_E:+from $(date -r $NB_E '+%a %H:%M') }${NA_E:+to $(date -r $NA_E '+%a %H:%M')}${AFTER:+, after: $AFTER}, cmd: $*"
last=$(( $(date +%s) - 900 + ${GAP0:-900} ))           # a 15-min gap from now too (Leap's / Fit12's turns end unseen);
                                                        # GAP0=0 when the last 45 GB ticket is known to be over 15 min ago
while :; do
  n=0; [ -z "$NOWAIT45" ] && n=$(busy45); now=$(date +%s)
  # NIGHT (the orchestrator 01:4x): NOWAIT45=1 skips the wait for others' 45 GB tickets (the quiet queue itself is FIFO,
  # one ticket at a time), GAP=0 drops the 15-min gap
  if [ -z "$NOWAIT45" ] && [ "${n:-0}" -gt 0 ]; then last=$now; hold "others' 45 GB tickets"; continue; fi
  [ $((now - last)) -lt ${GAP:-900} ] && { hold "gap ${GAP:-900} s"; continue; }
  g=$(why_blocked); [ -n "$g" ] && { hold "$g"; continue; }
  [ -n "$NB_E" ] && [ "$now" -lt "$NB_E" ] && { hold "NOTBEFORE $NOTBEFORE"; continue; }
  # NOTAFTER=HH:MM (DAY9 11-02): no queueing from then on (a day turn that never found the memory: exit 3, the night takes it)
  [ -n "$NA_E" ] && [ "$now" -ge "$NA_E" ] && { say "past NOTAFTER $NOTAFTER: not queued"; exit 3; }
  [ -n "$AFTER" ] && ! /bin/sh -c "$AFTER" >/dev/null 2>&1 && { hold "AFTER"; continue; }
  [ -n "$WAITMEM" ] && [ "$(avail_gb)" -lt "$WAITMEM" ] && { hold "WAITMEM $WAITMEM GB"; continue; }
  say "queue the quiet ticket (available $(avail_gb) GB)"
  # QTIMEOUT=s (Mon 07:2x, NewIdeas: "queue at 09:40 with a 60 min queue"): how long the ticket may wait; default 6 h
  PEER_QUEUE_TIMEOUT=${QTIMEOUT:-21600} peer lock quiet --queue ${MIN}m "$WHY" >> "$LOG" 2>&1 || { say "queue failed: retry in 2 min"; sleep 120; continue; }
  sleep 30
  a=$(avail_gb)
  if [ "$a" -lt $((PEAK + MARGIN)) ]; then
    fails=$((${fails:-0} + 1))
    peer unlock >> "$LOG" 2>&1
    # each failed grant drains everyone's shared runs for nothing (the orchestrator 23:3x): give up after two
    [ $fails -ge ${MAXFAILS:-2} ] && { say "granted $fails time(s) with only $a GB no-swap (< $((PEAK + MARGIN))): giving up this turn"; exit 8; }
    say "granted but only $a GB no-swap (< $((PEAK + MARGIN))): released, wait 2 min, queue again"
    sleep 120; last=0; continue
  fi
  say "granted, $a GB available: run"
  HAVE_LOCK=1 KEEP_LOCK=1 "$@"; rc=$?
  peer unlock >> "$LOG" 2>&1
  say "done rc=$rc, ticket released"
  exit $rc
done
