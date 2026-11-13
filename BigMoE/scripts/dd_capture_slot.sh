#!/bin/bash
# DEEPDRAFT capture in BigMoE's slot (Fri 13 Nov; the orchestrator 10:4x, after TTT30B PAYS: "the capture gets your slot
# when the gate's chain ends, open-ended until Kamen says stop ... stop at a chunk boundary when another session's quiet
# ticket or lock shows, then resume ... don't hold a quiet lock for hours").
# Runs scripts/dd_capture.py (DLOG_SRC=magicoder, the n1112 continuation: it skips prompts it already has) under a SHARE
# lock, never a quiet one:
#  - start: after the pid AFTER_PID ends (EvalSuite's 45 GB P2 gate chain), then whenever the gates clear;
#  - gates: no other session's quiet lock held, lapsed or queued (lock segments on the "^- " lines of peer ls only: a
#    held lock with nothing run under it for 8+ min shows as "quiet lock lapsed" and still counts), lid open,
#    no HEALTH ACT, NO-SWAP >= the capture's peak + 8 GB;
#  - yield: when a gate closes, touch the stop file: the capture ends at its next prompt boundary (median 14 s, p90 19 s per
#    prompt on night 1). After 120 s a SIGTERM, after 60 s more a SIGKILL (a heat freeze can hold it stopped); its npz
#    writes are atomic and a torn index line is skipped on resume. Then release its share of the lock, wait, resume;
#  - C: DD_C if set, else dd_capture_night.sh's table on NO-SWAP - RESERVE_GB (memory kept for a TTT arm beside it); peak
#    ~ 23.7 GB at C 100 + 0.164 GB per C;
#  - swap guard: +2 GB over the try's start stops the try (then wait and resume);
> (a quote from the project owner, kept private)
#    DLOG-ONPATH-FAIL (rc 3), or 4 failed tries in a row.
# The session's one peer lock is shared with dd_ttt_slot.sh through scripts/dd_slot_lock.sh (MEM = the sum of both jobs).
# usage (a harness background task, not nohup): AFTER_PID=5975 [DD_C=100 | RESERVE_GB=20] bash scripts/dd_capture_slot.sh
#        DRY=1: print the gates and the plan, run nothing
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
export BIGMOE_EXT_BUILD=${BIGMOE_EXT_BUILD:-build_qom}    # as dd_capture_night.sh (amendment 2 of the capture PREREG)
NIGHT=${DD_NIGHT:-n1112}
D=$HOME/FableGod/deepdraft/capture; mkdir -p "$D"
LOG=$D/slot_$NIGHT.log; CLOG=$D/capture_$NIGHT.log
STOPF=$D/.yield_$NIGHT
source scripts/dd_slot_lock.sh                          # STOP, SLOT_STATE, slot_lock: one session lock for both launchers
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -n "$DRY" ] && echo "$*"; }
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
RESERVE=${RESERVE_GB:-0}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
others_quiet() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }   # lapsed = still held (Sat 14 Nov 01:0x)
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB"; }
pick_c() { local a=$1; if [ "$a" -ge 52 ]; then echo 224; elif [ "$a" -ge 47 ]; then echo 192; elif [ "$a" -ge 42 ]; then echo 160;
  elif [ "$a" -ge 37 ]; then echo 128; elif [ "$a" -ge 32 ]; then echo 100; else echo 0; fi; }
peak_gb() { echo $(( (237 + ($1 - 100) * 164 / 100 + 9) / 10 )); }     # ceil(23.7 + 0.164 (C - 100))
want_c() { if [ -n "$DD_C" ]; then [ "$(avail_gb)" -ge $(( $(peak_gb $DD_C) + 8 )) ] && echo "$DD_C" || echo 0;
  else pick_c $(( $(avail_gb) - RESERVE )); fi; }

say "slot start (pid $$; night $NIGHT; AFTER_PID ${AFTER_PID:-none}; $( [ -n "$DD_C" ] && echo "C $DD_C fixed" || echo "reserve $RESERVE GB" ); stop file $STOP)"
if [ -n "$DRY" ]; then say "DRY: gates now: $(gate_state); C now $(want_c)"; exit 0; fi
if [ -n "$AFTER_PID" ]; then
  while kill -0 "$AFTER_PID" 2>/dev/null; do [ -e "$STOP" ] && { say "STOP_SLOT before the start: done"; exit 0; }; sleep 60; done
  say "pid $AFTER_PID ended: $(gate_state)"
fi
rm -f "$STOPF"; fails=0; tries=0; swaps=0
while :; do
  [ -e "$STOP" ] && { say "STOP_SLOT: done"; break; }
  n=0
  until gates_clear && [ "$(want_c)" -gt 0 ]; do
    [ -e "$STOP" ] && break
    [ $((n % 30)) -eq 0 ] && say "waits: $(gate_state); C $(want_c)"
    n=$((n + 1)); sleep 60
  done
  [ -e "$STOP" ] && { say "STOP_SLOT: done"; break; }
  C=$(want_c); MEM=$(peak_gb $C); S0=$(swap_mb)
  echo "$$ $MEM capture C $C" > "$SLOT_STATE/capture.mem"; slot_lock
  tries=$((tries + 1))
  say "try $tries: C $C (peak ~$MEM GB), swap0 $S0 MB: $(gate_state)"
  L0=$(wc -l < "$CLOG" 2>/dev/null || echo 0)
  DLOG_SRC=magicoder DLOG_NIGHT=$NIGHT DLOG_TARGET=50000000 DLOG_CAP_GB=${DD_CAP_GB:-80} DLOG_MIN_FREE_GB=115 \
    DLOG_BUDGET=43200 DLOG_HARD=43500 DLOG_STOPFILE="$STOPF" "$PY" scripts/dd_capture.py $C >> "$CLOG" 2>&1 &
  P=$!; why=""
  while kill -0 $P 2>/dev/null; do
    sleep 20
    if [ -e "$STOP" ]; then why=STOP_SLOT
    elif ! gates_clear; then why="yield ($(gate_state))"
    elif [ "$(swap_mb)" -gt $((S0 + 2048)) ]; then why="swap $(swap_mb) MB > $S0 + 2048"
    fi
    if [ -n "$why" ]; then
      say "$why: stop file -> capture pid $P"
      touch "$STOPF"; t=0
      while kill -0 $P 2>/dev/null && [ $t -lt 120 ]; do sleep 5; t=$((t + 5)); done
      kill -0 $P 2>/dev/null && { say "no prompt boundary within 120 s: SIGTERM"; kill -TERM $P; t=0
        while kill -0 $P 2>/dev/null && [ $t -lt 60 ]; do sleep 5; t=$((t + 5)); done
        kill -0 $P 2>/dev/null && { say "still alive 60 s after SIGTERM (frozen?): SIGKILL"; kill -KILL $P; }; }
      break
    fi
  done
  wait $P; rc=$?
  rm -f "$SLOT_STATE/capture.mem" "$STOPF"; slot_lock
  NEWL=$(tail -n +$((L0 + 1)) "$CLOG" 2>/dev/null)
  last=$(echo "$NEWL" | grep -E "stopping|done|DLOG-ONPATH|session:" | tail -2 | tr '\n' ' ' | cut -c1-240)
  say "try $tries ended rc $rc${why:+ ($why)}: $last"
  [ "$why" = STOP_SLOT ] && { say "STOP_SLOT: done"; break; }
  case "$why" in swap*) swaps=$((swaps + 1)); [ $swaps -ge 3 ] && { say "3 swap-guard stops: slot ends"; break; };; esac
  [ $rc -eq 3 ] && { say "DLOG-ONPATH-FAIL: not retried, slot ends"; break; }
  if echo "$NEWL" | grep -qE "cap [0-9.]+ GB reached|free disk .* < |every language has"; then say "the capture's own stop: slot ends"; break; fi
  if [ $rc -eq 0 ] && [ -z "$why" ] && echo "$NEWL" | grep -q "session: 0 prompts"; then say "no prompts left: slot ends"; break; fi
  if [ $rc -ne 0 ] && [ -z "$why" ]; then
    fails=$((fails + 1)); [ $fails -ge 4 ] && { say "4 failed tries in a row: slot ends"; break; }
  else fails=0; fi
  sleep 60
done
say "slot end: $(ls "$D/$NIGHT" 2>/dev/null | grep -c '\.npz$') files, $(du -sh "$D/$NIGHT" 2>/dev/null | cut -f1)"
