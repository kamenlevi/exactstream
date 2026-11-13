#!/bin/bash
# night_onebuild.sh (Leap 11-13 02:4x): the PF-B one-build test at 45 GB (onebuild_45_runs.sh v2) under ONE quiet ticket, the
# night45_1113.sh way (the orchestrator 02:3x: "Re-arming onebuild for the next night the same way is fine (one quiet ticket,
# gates after the grant). Size its gate first"). The 1.2.0 stage 3 is done (PASS Fri 02:29), so this runs onebuild only.
#  1. From FROM: if WAITFOR is set (an extended regex for the start of a `peer ls` line, e.g. "^- Run a larger-than-RAM MoE"),
#     wait until that session's quiet ticket or lock shows and queue behind it; by NOTIX queue anyway. WAITFOR empty: queue at
#     FROM.
#  2. One quiet ticket, attached (PEER_QUEUE_TIMEOUT up to LAST). The reason says "45 GB" and names no other session (other
#     sessions' waiters drop lines that carry their own name); keep this session's status line free of it too.
#  3. After the grant settle 30 s; gates: no engine, PAUSE (waits up to 3 min) / HOLD, lid open, no HEALTH ACT, Kamen idle
#     >= 600 s, NO-SWAP >= 52 (the lowest per-run gate: a 10K run's peak 47.5 GB + 4, the quiet-lock margin). Then
#     onebuild_45_runs.sh with its own per-run gates (10K runs 52; 30K R0 53, R3 / R2 54, R1 56).
#  4. Release the moment the gate fails or the runs end. Swap +2 GB stops all (onebuild's own guard).
# v2 (Fri 13 Nov 15:5x, the orchestrator's start rule for Fri -> Sat: one quiet ticket at Sat 00:30, or earlier on its "go";
# nothing may END after Sat 10:00): GO file (touched when the orchestrator says go) counts as FROM reached; a 4th argument,
# CUTOFF, goes to onebuild_45_runs.sh v3 (it skips a run that could end after it); the ticket asks 90 min.
# Usage: [WAITFOR=regex] [BLOCKS="10k 30k"] night_onebuild.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<NOTIX ...>" "<LAST ...>" "<CUTOFF ...>"
OB=/Users/kamen/Leap/scripts/prefill15/onebuild/onebuild_45_runs.sh
OBW=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/onebuild45; LOG=$OBW/night_onebuild.log
Q=$HOME/.local/share/quiet
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); NOTIX=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$3" +%s)
CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${4:?CUTOFF}" +%s); GOF=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/GO_night
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
tix() { [ -n "$WAITFOR" ] || { echo 0; return; }; peer ls 2>/dev/null | grep -E "$WAITFOR" | grep -cE "· (benchmarking|waiting for a quiet window)"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
gatewhy() {
  [ -n "$(procs)" ] && { echo "an engine runs ($(procs))"; return; }
  [ -e $Q/HOLD ] && { echo "HOLD"; return; }
  [ -e $Q/PAUSE ] && { echo "PAUSE"; return; }
  lid_closed && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { echo "HEALTH ACT"; return; }
> (a quote from the project owner, kept private)
  [ "$(noswap)" -lt $1 ] && { echo "NO-SWAP $(noswap) < $1"; return; }
}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
LOCKED=0; CP=""
cleanup() { if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then for c in $(pgrep -P $CP 2>/dev/null); do kill -TERM $c 2>/dev/null; done; kill -TERM $CP; fi
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
[ -f $OB ] || { say "STOP: no $OB"; exit 2; }
for n in 10k 30k; do [ -f $OBW/prompt_$n.txt ] || { say "STOP: no $OBW/prompt_$n.txt"; exit 2; }; done
say "armed (pid $$): PF-B one-build at 45 GB under one quiet ticket, blocks [${BLOCKS:-10k 30k}]; from $1 (or $GOF)${WAITFOR:+ behind /$WAITFOR/}, queue anyway at $2, last $3, runs end by $4"
NOTE=""
while :; do
  now=$(date +%s)
  [ $now -ge $LAST ] && { say "SKIP: past $3 before queueing"; exit 0; }
  if [ $now -ge $FROM ] || [ -e $GOF ]; then
    [ -z "$WAITFOR" ] && break
    [ "$(tix)" -gt 0 ] && { say "the awaited ticket or lock is in peer ls: queue behind it"; break; }
    [ $now -ge $NOTIX ] && { say "no awaited ticket by $2: queue anyway"; break; }
    w="waiting for the awaited ticket"
  else
    w="before $1"
  fi
  [ "$w" != "$NOTE" ] && say "wait: $w"; NOTE=$w
  sleep 30
done
TO=$(( LAST - $(date +%s) )); [ $TO -lt 120 ] && { say "SKIP: under 2 min to $3"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 90m "Leap: 45 GB PF-B one-build test (bitwise vs PF-D, peaks), MEM 52-56 GB, ~45 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant"; exit 0
fi
say "granted (NO-SWAP $(noswap) GB); settle 30 s"
sleep 30
[ $(date +%s) -ge $LAST ] && { say "granted after $3: release"; exit 0; }
w=0; while [ -e $Q/PAUSE ] && [ $w -lt 180 ]; do sleep 15; w=$((w + 15)); done
why=$(gatewhy 52)
[ -n "$why" ] && { say "gate failed after the grant: $why -> release"; exit 0; }
say "onebuild (NO-SWAP $(noswap) GB): onebuild_45_runs.sh, per-run gates"
CUTOFF=$CUT BLOCKS="${BLOCKS:-10k 30k}" bash $OB $OBW > $OBW/onebuild_runs.out 2>&1 &
CP=$!; wait $CP; rc=$?; CP=""
say "  onebuild rc $rc"
grep -E "^onebuild R|SKIP" $OBW/onebuild.log 2>/dev/null | tail -10 | sed 's/^/  /' | tee -a $LOG
say "DONE"
exit 0
