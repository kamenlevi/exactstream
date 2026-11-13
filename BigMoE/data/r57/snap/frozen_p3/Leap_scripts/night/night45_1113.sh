#!/bin/bash
# night45_1113.sh (Leap 11-13 01:5x). The orchestrator 01:4x: "Run your 45 GB stage and onebuild right after BigMoE's A/B, ahead
# of TTT30B, if they fit. If they don't fit, they go to the next night." Its steps, as set:
#  1. From FROM, wait until the 45 GB A/B's quiet ticket or lock shows on its session's `peer ls` line ("- Run a larger-than-
#     RAM MoE ..."; a "· benchmarking" or "· waiting for a quiet window" segment). Never queue before it: an early grant at
#     ~50 GB fails, and `peer unlock` also drops a queue place. Its turn45 queues no later than 04:00, so if no ticket has
#     shown by NOTIX (04:00) the A/B did not run and this queues anyway.
#  2. Queue ONE quiet ticket behind it, attached (this process waits for the grant; a harness background task, not nohup),
#     with PEER_QUEUE_TIMEOUT up to LAST. The reason says "45 GB" (other 45 GB turns see it) and never the other session's
#     name: TTT30B's others_quiet() and turn45's busy45() drop lines that carry it. This session's status line avoids it too.
#  3. After the grant settle 30 s. Gates: no engine, no PAUSE (waits up to 3 min) / HOLD, lid open, no HEALTH ACT, Kamen idle
#     >= 600 s, NO-SWAP >= 52 -> stage 3 of the 1.2.0 clean-clone check on the published v1.2.0 (pkg_120b 21fb59b, clone
#     v120b): exact-45 (pin), lossy-45, chat-45, as rel120_check.sh's stage 3. Then, 20 s later, the same gates with NO-SWAP
#     >= 56 -> onebuild_45_runs.sh (the pre-registered 8 runs, section 3 of data/prefill15_1112.md). All under the one lock.
#  4. Release the moment a gate fails or the work ends (the next job never waits for nothing). Swap +2 GB stops all.
# Usage: night45_1113.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<NOTIX ...>" "<LAST ...>"
V=/Users/kamen/Leap/release/cleanclone/v120b; CL=$V/clone; PKG=/Users/kamen/Leap/release/out/pkg_120b; CCT=$CL/tools/clean_clone_test.sh
WH=/Users/kamen/Leap/release/cleanclone/wheels; OB=/Users/kamen/Leap/scripts/prefill15/onebuild/onebuild_45_runs.sh
OBW=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/onebuild45; LOG=$V/night45.log
Q=$HOME/.local/share/quiet
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); NOTIX=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$3" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
abtix() { peer ls 2>/dev/null | grep "^- Run a larger-than-RAM MoE" | grep -cE "· (benchmarking|waiting for a quiet window)"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
gatewhy() {   # $1 = the NO-SWAP floor; prints the first gate that fails, or nothing
  [ -n "$(procs)" ] && { echo "an engine runs ($(procs))"; return; }
  [ -e $Q/HOLD ] && { echo "HOLD"; return; }
  [ -e $Q/PAUSE ] && { echo "PAUSE"; return; }
  lid_closed && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { echo "HEALTH ACT"; return; }
> (a quote from the project owner, kept private)
  [ "$(noswap)" -lt $1 ] && { echo "NO-SWAP $(noswap) < $1"; return; }
}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
LOCKED=0; CP=""; GUARD=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then for c in $(pgrep -P $CP 2>/dev/null); do kill -TERM $c 2>/dev/null; done; kill -TERM $CP; fi
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            touch $V/night45.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
[ "$(git -C $CL rev-parse --short HEAD)" = "$(git -C $PKG rev-parse --short HEAD)" ] || { say "STOP: the clone is not at the package's HEAD"; exit 2; }
[ -x $OB ] || [ -f $OB ] || { say "STOP: no $OB"; exit 2; }
for n in 10k 30k; do [ -f $OBW/prompt_$n.txt ] || { say "STOP: no $OBW/prompt_$n.txt"; exit 2; }; done
say "armed (pid $$): 45 GB clean-clone stage of v1.2.0 ($(git -C $CL rev-parse --short HEAD)) + onebuild under one quiet ticket queued behind the 45 GB A/B; from $1, queue anyway at $2, last $3"
# 1. wait for the A/B's ticket or lock
NOTE=""
while :; do
  now=$(date +%s)
  [ $now -ge $LAST ] && { say "SKIP: past $3 before queueing (next night)"; exit 0; }
  if [ $now -ge $FROM ]; then
    [ "$(abtix)" -gt 0 ] && { say "the 45 GB A/B's ticket or lock is in peer ls: queue behind it"; break; }
    [ $now -ge $NOTIX ] && { say "no A/B ticket or lock by $2: queue anyway"; break; }
    w="waiting for the A/B's ticket"
  else
    w="before $1"
  fi
  [ "$w" != "$NOTE" ] && say "wait: $w"; NOTE=$w
  sleep 30
done
# 2. one quiet ticket, attached
TO=$(( LAST - $(date +%s) )); [ $TO -lt 120 ] && { say "SKIP: under 2 min to $3"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 75m "Leap: 45 GB clean-clone check of 1.2.0 + PF-B one-build (8 runs), MEM 54 GB, ~45 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant: next night"; exit 0
fi
say "granted (NO-SWAP $(noswap) GB); settle 30 s"
sleep 30
[ $(date +%s) -ge $LAST ] && { say "granted after $3: release (next night)"; exit 0; }
# 3. gates, then stage 3
w=0; while [ -e $Q/PAUSE ] && [ $w -lt 180 ]; do sleep 15; w=$((w + 15)); done
why=$(gatewhy 52)
[ -n "$why" ] && { say "gate failed after the grant: $why -> release (next night)"; exit 0; }
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
say "stage 3 (NO-SWAP $(noswap) GB): clean_clone_test.sh run, CONFIGS=exact-45 lossy-45 chat-45"
CONFIGS="exact-45 lossy-45 chat-45" TEXT_CONFIG=chat-45 CHAT_SERVE=0 WHEELS=$WH bash "$CCT" "$PKG" "$V" run > $V/stage3.out 2>&1 &
CP=$!; wait $CP; rc=$?; CP=""
say "  stage 3 rc $rc: $(tail -1 $V/report.txt)"
# then onebuild
sleep 20
w=0; while [ -e $Q/PAUSE ] && [ $w -lt 180 ]; do sleep 15; w=$((w + 15)); done
why=$(gatewhy 56)
[ -n "$why" ] && { say "onebuild not started: $why -> release (next night)"; exit 0; }
say "onebuild (NO-SWAP $(noswap) GB): onebuild_45_runs.sh, 8 runs"
bash $OB $OBW > $OBW/onebuild_runs.out 2>&1 &
CP=$!; wait $CP; rc=$?; CP=""
say "  onebuild rc $rc"
grep "^onebuild R" $OBW/onebuild.log 2>/dev/null | tail -4 | sed 's/^/  /' | tee -a $LOG
say "DONE"
exit 0
