#!/bin/bash
# run_c25.sh (Leap, Mon 16 Nov 2026 12:4x; CHECK25): one run of c25_loop.py (BigMoE's lossy-45 + 1c-tiers loop in the G32T
# test mode, c25_engine's arms on 32-row checks) with c25_env.py, in a slot NewIdeas gives (Tue by day ~09:30-10:00; a smoke
# today when memory frees). Waits, holding nothing, until START; never starts if it cannot end by HARD (latest start =
# HARD - EST); then: HOLD / PAUSE / lid / HEALTH ACT checks, the head test (the real loop up to its model load), a quiet ticket
# (queued until the latest start), settle 20 s, the NO-SWAP gate on the lower of two reads 30 s apart (>= GATE, default 46 =
# t1cx45t1c1's key-test peak 42.4 + 4), a swap guard (+2 GB stops it), the run killed at HARD. On ANY end: the ticket
# released, then the marker ~/Leap/data/week98/c25/C25_<TAG>_DONE = "c25 end <time>: rc N (why)".
# Env knobs (passed through to c25_env.py): C25_TAG, C25_ARMS, C25_ROUNDS, C25_WARM, C25_KEYS; C25_GATE, C25_EST_MIN;
# C25_AFTER=<marker path>: also wait for that file, newer than arming (another job's end marker); C25_AFTER_PID=<pid>:
# the fallback when that job's runner is gone without its marker (start 60 s later); C25_C: the pool size (default 256);
# C25_WAIT_NOSWAP=<GB>: hold nothing until NO-SWAP >= GB on two reads before the ticket (C25_POLL / C25_GAP seconds;
# C25_WAIT_PIDFILE: also hold while that file's process group is alive; C25_WAIT_ENDMARK: if such a step ran, also wait
# for that marker to be written after the arm; C25_WAIT_NOBIG=<GB>: also hold while another process is over that RSS); C25_CAP_MIN: kill the run that
# many minutes after the grant; C25_NOHEAD: skip the head test; C25_POST=<shell command>: run it after the loop, inside the
# same quiet window (before the hard end / cap), output to <TAG>_post.txt.
# DRY=1: fake lock, 1 s waits, gate 1, START now, C25_CMD (default: sleep 3) in place of the loop.
# Usage: run_c25.sh "<START YYYY-MM-DD HH:MM:SS>" "<HARD END ...>"
E=/Users/kamen/Leap/data/week98/c25; H=/Users/kamen/Leap/scripts/week98/c25; D=/Users/kamen/Leap/data/week98/screens
Q=$HOME/.local/share/quiet; PY=/Users/kamen/BigMoE/.venv/bin/python
DRY=${DRY:-0}; GATE=${C25_GATE:-46}; EST=${C25_EST_MIN:-40}; ZZ=30; ST=20; TAG=${C25_TAG:-c25}
[ $DRY = 1 ] && { GATE=1; ZZ=1; ST=1; TAG=${TAG}_dry; }
export C25_TAG=$TAG
mkdir -p $E
MARK=$E/C25_${TAG}_DONE; LOG=$E/run_${TAG}.log
START=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?START}" +%s) || exit 2
HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?HARD END}" +%s) || exit 2
[ $DRY = 1 ] && START=$(date +%s)
LATEST=$(( HARD - EST * 60 ))
T0=$(date +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [run_c25 $TAG] $*" | tee -a $LOG >> $D/screens.log; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
pmode() { pmset -g | awk '/^ *powermode/ {print $2; exit}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
RC=9; WHY="started"; LOCKED=0; CP=""; GUARD=""
fin() {
  [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
  [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
  if [ $LOCKED = 1 ]; then [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; say "ticket released"; fi
  echo "c25 end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY)" > $MARK
  say "wrote $MARK (rc $RC, $WHY); NO-SWAP $(noswap) GB, swap $(swap) MB"
}
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
stop_checks() { [ -e $Q/HOLD ] && { WHY="HOLD"; RC=3; exit 3; }
                [ -e $Q/PAUSE ] && { WHY="PAUSE"; RC=3; exit 3; }
                lid_closed && { WHY="lid closed"; RC=3; exit 3; }
                grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { WHY="HEALTH ACT"; RC=3; exit 3; }; }
away_ok() {   # the orchestrator 21:0x (Kamen: lower-RAM tests while he is at the Mac): away >= 900 s around the reads
  local s; s=$(bash ${LEAP_AWAY_SH:-/Users/kamen/.local/share/quiet/away_secs.sh} 2>/dev/null)
> (a quote from the project owner, kept private)
gate_ok() { local a b; away_ok || return 1; a=$(noswap); sleep $ZZ; b=$(noswap); say "  gate $GATE: read 1 $a GB; read 2 (+$ZZ s) $b GB"
            [ $a -ge $GATE ] && [ $b -ge $GATE ] && away_ok; }
rm -f $MARK
say "armed (pid $$, DRY $DRY): start $(date -r $START '+%m-%d %H:%M'), latest start $(date -r $LATEST '+%m-%d %H:%M'), hard end $(date -r $HARD '+%m-%d %H:%M'); gate $GATE GB; arms ${C25_ARMS:-all}; rounds ${C25_ROUNDS:-12}"
while [ $(date +%s) -lt $START ]; do sleep $(( DRY == 1 ? 1 : 20 )); done
if [ -n "${C25_AFTER:-}" ]; then   # wait for another job's end marker, newer than arming (or the latest start)
  say "waiting for $C25_AFTER (newer than arming)"
  while ! { [ -f "$C25_AFTER" ] && [ "$(stat -f %m "$C25_AFTER")" -ge $T0 ]; }; do
    [ $(date +%s) -ge $LATEST ] && { WHY="no marker $C25_AFTER before the latest start"; RC=3; exit 3; }
    if [ -n "${C25_AFTER_PID:-}" ] && ! kill -0 $C25_AFTER_PID 2>/dev/null; then   # its runner died without a marker
      sleep $(( DRY == 1 ? 1 : 60 )); [ -f "$C25_AFTER" ] && [ "$(stat -f %m "$C25_AFTER")" -ge $T0 ] && break
      say "pid $C25_AFTER_PID is gone and no marker: the fallback (start)"; break
    fi
    sleep $(( DRY == 1 ? 1 : 20 ))
  done
  [ -f "$C25_AFTER" ] && say "marker: $(head -c 160 "$C25_AFTER")"
fi
[ $(date +%s) -ge $LATEST ] && { WHY="no start before the latest start"; RC=3; exit 3; }
cd /Users/kamen/Leap || { WHY="no ~/Leap"; RC=2; exit 2; }
step_live() {   # another job's running step: C25_WAIT_PIDFILE holds its process group id (Sandbox98's guarded steps
  [ -n "${C25_WAIT_PIDFILE:-}" ] && [ -f "$C25_WAIT_PIDFILE" ] || return 1   # yield = kill their step on any quiet ticket)
  local g; g=$(cat "$C25_WAIT_PIDFILE" 2>/dev/null); [ -n "$g" ] || return 1
  pgrep -g "$g" >/dev/null 2>&1
}
big_other() {   # another process over C25_WAIT_NOBIG GB resident (NewIdeas' slot gate: no other > 16 GB job running)
  [ -n "${C25_WAIT_NOBIG:-}" ] || return 1
  [ "$(ps -axo rss=,pid= | awk -v L=$(( C25_WAIT_NOBIG * 1024 * 1024 )) -v me=$$ '$1 > L && $2 != me' | wc -l | tr -d ' ')" -gt 0 ]
}
end_pending() {   # that job ran a step during this wait and its end marker (C25_WAIT_ENDMARK, written on any end) is not new yet
  [ "$SAW" = 1 ] && [ -n "${C25_WAIT_ENDMARK:-}" ] || return 1
  [ -f "$C25_WAIT_ENDMARK" ] && [ "$(stat -f %m "$C25_WAIT_ENDMARK")" -ge $T0W ] && return 1
  return 0
}
if [ -n "${C25_WAIT_NOSWAP:-}" ]; then   # hold nothing until the memory is free (e.g. another job's process has exited)
  POLL=$(( DRY == 1 ? 1 : ${C25_POLL:-20} )); GAP=$(( DRY == 1 ? 1 : ${C25_GAP:-$ZZ} )); SAW=0; T0W=$(date +%s)
  say "waiting for NO-SWAP >= $C25_WAIT_NOSWAP GB on two reads $GAP s apart, polling every $POLL s (holding nothing)${C25_WAIT_PIDFILE:+; no live step in $C25_WAIT_PIDFILE}${C25_WAIT_ENDMARK:+ (if one runs: then its end marker $C25_WAIT_ENDMARK)}"
  while :; do
    [ $(date +%s) -ge $LATEST ] && { WHY="the memory / the other job never freed before the latest start (NO-SWAP $(noswap) GB)"; RC=3; exit 3; }
    if step_live; then [ $SAW = 0 ] && say "another job's step is live ($C25_WAIT_PIDFILE): wait for it${C25_WAIT_ENDMARK:+ and its end marker}"; SAW=1; sleep $POLL; continue; fi
    end_pending && { sleep $POLL; continue; }
    if big_other; then [ "${SAWB:-0}" = 0 ] && say "a process over $C25_WAIT_NOBIG GB is running: wait"; SAWB=1; sleep $POLL; continue; fi
    a=$(noswap)
    if [ $a -ge $C25_WAIT_NOSWAP ]; then sleep $GAP; b=$(noswap); [ $b -ge $C25_WAIT_NOSWAP ] && ! step_live && ! big_other && { say "memory free: $a / $b GB"; break; }; fi
    sleep $POLL
  done
fi
stop_checks
if [ -n "${C25_NOHEAD:-}" ] || { [ $DRY = 1 ] && [ -n "${C25_SKIP_HEAD:-}" ]; }; then say "head test skipped (C25_NOHEAD: it passed for this code ${C25_NOHEAD:-})"; else
  ( eval "$(python3 $H/c25_env.py)" && $PY -I -B $H/c25_headtest.py ) >> $LOG 2>&1 || { WHY="the head test failed (log)"; RC=5; exit 5; }
  say "head test PASS"
fi
to=$(( LATEST - $(date +%s) )); [ $to -lt 60 ] && { WHY="no time left to queue"; RC=3; exit 3; }
say "queue the quiet ticket (timeout ${to} s; NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then echo "(dry) fake ticket" >> $LOG; else
  PEER_QUEUE_TIMEOUT=$to peer lock quiet --queue 60m "Leap CHECK25: lossy 32-row check engine A/Bs (c25), MEM 43 GB" >> $LOG 2>&1 \
    || { WHY="the ticket left the queue without a grant"; RC=4; exit 4; }
fi
LOCKED=1; say "granted (power mode $(pmode)); settle $ST s"; sleep $ST
if [ -n "${C25_CAP_MIN:-}" ]; then   # the quiet window's cap, counted from the grant
  H2=$(( $(date +%s) + C25_CAP_MIN * 60 )); [ $H2 -lt $HARD ] && HARD=$H2
  say "the run is capped at $(date -r $HARD '+%H:%M:%S') (${C25_CAP_MIN} min from the grant)"
fi
stop_checks
> (a quote from the project owner, kept private)
[ $(date +%s) -ge $LATEST ] && { WHY="granted after the latest start"; RC=3; exit 3; }
s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) [run_c25] SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
say "run (power mode $(pmode), swap $(swap) MB, NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then ( bash -c "${C25_CMD:-sleep 3}" ) >> $LOG 2>&1 & else
  ( eval "$(python3 $H/c25_env.py)" && exec $PY -I -B $H/c25_loop.py ${C25_C:-256} 2048 ) > $E/${TAG}_run.txt 2>&1 &
fi
CP=$!
while kill -0 $CP 2>/dev/null; do
  if [ $(date +%s) -ge $HARD ]; then say "HARD END: stopping the run"; killtree $CP; sleep 5; WHY="cut at the hard end"; RC=6; exit 6; fi
  stop_checks_soft() { [ -e $Q/HOLD ] || [ -e $Q/PAUSE ] || lid_closed; }
  if stop_checks_soft; then say "HOLD / PAUSE / lid during the run: stopping"; killtree $CP; sleep 5; WHY="stopped (HOLD / PAUSE / lid)"; RC=3; exit 3; fi
  sleep $(( DRY == 1 ? 1 : 10 ))
done
wait $CP; RC=$?
case $RC in 0) WHY="ran";; *) WHY="the run failed (${TAG}_run.txt)";; esac
if [ $RC = 0 ] && [ -n "${C25_POST:-}" ] && [ $(date +%s) -lt $HARD ]; then   # more measurements in the same quiet window
  say "post: $C25_POST"
  ( bash -c "$C25_POST" ) >> $E/${TAG}_post.txt 2>&1 &
  CP=$!
  while kill -0 $CP 2>/dev/null; do
    if [ $(date +%s) -ge $HARD ]; then say "HARD END during the post: stopping it"; killtree $CP; sleep 5; WHY="ran; the post was cut at the hard end"; exit 0; fi
    sleep $(( DRY == 1 ? 1 : 5 ))
  done
  wait $CP; PRC=$?
  WHY="ran; post rc $PRC (${TAG}_post.txt)"
fi
exit $RC
