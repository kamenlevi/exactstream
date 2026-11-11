#!/bin/zsh
# WHOLE-CHIP-2 runner (Leap, Tue 2026-11-03 17:5x; TABLE_350 'WHOLE-CHIP-2'): waits for 00:44 (the orchestrator
# 18:0x: Fit12's T3R + P2-PREFETCH quiet slot is 00:30-00:43:30 and drops a grant after 00:35:30, so no ticket before
# 00:44; BigMoE's mission 20 queues from 00:45) and the night window (until 14:00), for no PAUSE / HOLD, and for a quiet
# lock in the FIFO; then H (native/wc_handoff, cap 150 s) and S (scripts/wc_screen.py sustained, cap 420 s): a HARD
# ~10 min in all; then unlocks. Stops on swap growth > 2 GB.
# WC2_NOW=1 (the orchestrator 18:5x, Kamen's plan change: small breakthrough tests run now, with Kamen on the Mac): no
# clock wait; a 10 s background-load baseline before H, and scripts/wc2_load.py samples the background load (CPU and GPU
# per app, outside this runner's process tree) all through H and S (data/wc2_load.jsonl, marks in data/wc2_marks.txt).
# Run ATTACHED (run_in_background) so the lock stays with this session.
cd ~/Leap
Q=~/.local/share/quiet
LOG=data/wc2_night.log
MARKS=data/wc2_marks.txt
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
mark() { echo "$1 $(date +%s)" >> $MARKS; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); n+=$NF} END {printf "%d", n*ps/1e9}'; }
run_capped() {                       # run_capped SECONDS cmd ...: the command and its children are killed at the cap
  local cap=$1; shift                # wall clock ($SECONDS): the heat watcher's SIGSTOP freezes this runner too, so a
  "$@" &                             # count of loop seconds let S run 801 s against a 420 s cap (11-03 19:1x)
  local pid=$! t0=$SECONDS
  while kill -0 $pid 2>/dev/null; do
    if [ $((SECONDS - t0)) -ge $cap ]; then
      say "cap ${cap} s hit: stopping pid $pid and its children"
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; sleep 3
      pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null
      break
    fi
    sleep 1
  done
  wait $pid 2>/dev/null
  return $?
}
if [ -z "$WC2_NOW" ]; then
  say "runner up (pid $$); waiting for 00:44"
  while true; do
    hm=$((10#$(date +%H%M)))
    if [ $hm -ge 44 ] && [ $hm -lt 1400 ]; then break; fi
    sleep 30
  done
else
> (a quote from the project owner, kept private)
fi
while [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; do sleep 60; done
say "queueing a quiet lock"
PEER_QUEUE_TIMEOUT=43200 peer lock quiet --queue 11m "Leap WHOLE-CHIP-2: Neural Engine handoff + sustained re-run, hard cap 10 min, MEM 4 GB" >> $LOG 2>&1
say "lock: $(peer ls 2>/dev/null | grep -c 'quiet run') quiet runs listed"
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD set after the grant: stopping"; peer unlock >> $LOG 2>&1; exit 0; fi
av=$(avail_gb)
if [ $av -lt 8 ]; then say "NO-SWAP gate: ${av} GB available < 8 (peak 4 + 4): stopping"; peer unlock >> $LOG 2>&1; exit 1; fi
s0=$(swap_mb)
say "available ${av} GB, swap ${s0} MB"
LPID=
if [ -n "$WC2_NOW" ]; then
  : > $MARKS
  nice -n 10 /usr/bin/python3 scripts/wc2_load.py sample $$ data/wc2_load.jsonl 2 >> $LOG 2>&1 &
  LPID=$!
  mark base0; sleep 10; mark base1
fi
W=data/wc
say "H: wc_handoff (K 4 / 8 / 12 / 16, 20 checks, 0.70 ms layers, 200 GB/s share)"
mark H0
run_capped 150 native/wc_handoff $W/ane_e4_a4_s1.mlpackage,$W/ane_e8_a4_s1.mlpackage,$W/ane_e12_a4_s1.mlpackage,$W/ane_e16_a4_s1.mlpackage 20 0.70 62 0.71 200 > data/wc2_handoff.txt 2>&1
say "H exit $?; swap +$(( $(swap_mb) - s0 )) MB"
mark H1
if [ $(( $(swap_mb) - s0 )) -gt 2048 ]; then
  say "swap guard: stopping"; [ -n "$LPID" ] && kill $LPID 2>/dev/null; peer unlock >> $LOG 2>&1; exit 1
fi
say "S: sustained re-run"
mark S0
run_capped 420 env WC_COND="${WC2_COND:-quiet lock, night; WHOLE-CHIP-2 S}" WC_PH=12 WC_WARM=45 WC_ALONE=0 \
  WC_UNITS=a4x4,q4,a4x4+q4,a4x4,q4,a4x4+q4,a4x4,q4,a4x4+q4 \
  ~/BigMoE/.venv/bin/python scripts/wc_screen.py > data/wc2_sustained.log 2>&1
say "S exit $?; swap +$(( $(swap_mb) - s0 )) MB"
mark S1
[ -n "$LPID" ] && kill $LPID 2>/dev/null
peer unlock >> $LOG 2>&1
say "done, unlocked"
[ -n "$LPID" ] && /usr/bin/python3 scripts/wc2_load.py summary data/wc2_load.jsonl $MARKS > data/wc2_load_summary.txt 2>&1
exit 0
