#!/bin/bash
# run_paused.sh NAME LOG MINUTES CMD... (MTP Head, Sun 11-15 01:4x): run CMD under a share lock; while another session is
# benchmarking or waiting for a quiet window, SIGSTOP it and release the lock, then re-lock and SIGCONT when clear.
# Never kills and never restarts (Sandbox98's guarded() kills and restarts from step 0, up to 3 times: tonight's
# frequent quiet windows ended S4's first run at step 9). Exits with CMD's exit code.
source "$HOME/Sandbox98/scripts/guard_lib.sh"      # quiet_others, noswap, swap_mb (read-only use)
name=$1 log=$2 mins=$3; shift 3
lk() {   # logs the lock's reply and exit code (06:06: the old silent call may have failed unseen); 3 tries, 30 s apart
  local i out rc
  for i in 1 2 3; do
    out=$(peer lock share ${mins}m --cpu 200 --gpu 25 "MTP Head: $name, MEM ${MEMGB:-6} GB" 2>&1); rc=$?
    echo "$(date +%H:%M:%S) lock try $i rc $rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    [ $rc -eq 0 ] && return 0
    sleep 30
  done
  echo "$(date +%H:%M:%S) LOCK FAILED 3x for $name: running without a share lock"; return 1
}
clear() {   # WAITBIG=N (Mon 16 Nov 14:2x): also wait while another session holds a lock with MEM >= N GB. 13:26-14:15,
            # 6 runs started in the gaps between EvalSuite's back-to-back 39 GB processes and the guard killed each one.
  [ "$(quiet_others)" -eq 0 ] && [ "$(noswap)" -ge ${NOSWAP_MIN:-12} ] && [ ! -f "$HOME/.local/share/quiet/PAUSE" ] \
    && { [ -z "$WAITBIG" ] || [ "$(GUARD_BIG_GB=$WAITBIG big_others)" -eq 0 ]; }
}
until clear; do sleep 20; done
[ -n "$WAITBIG" ] && { sleep 60; until clear; do sleep 20; done; }   # still clear a minute later
lk
echo "$(date +%H:%M:%S) start $name (swap $(swap_mb) MB, NO-SWAP $(noswap) GB)"
"$@" > "$log" 2>&1 &
P=$!; paused=0; SW0=$(swap_mb)
while kill -0 $P 2>/dev/null; do
  sleep 10
  if [ "$(noswap)" -lt 6 ] || [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then   # 06:06 incident: kill, do not pause
    echo "$(date +%H:%M:%S) MEMORY GUARD: NO-SWAP $(noswap) GB, swap +$(( $(swap_mb) - SW0 )) MB -> killing $name"
    kill -CONT $P 2>/dev/null; kill -TERM $P; sleep 10; kill -KILL $P 2>/dev/null; wait $P; peer unlock >/dev/null 2>&1; exit 3
  fi
  if [ "$(quiet_others)" -gt 0 ] || [ -f "$HOME/.local/share/quiet/PAUSE" ]; then
    if [ $paused = 0 ]; then kill -STOP $P; paused=1; peer unlock >/dev/null 2>&1; echo "$(date +%H:%M:%S) paused $name (quiet window)"; fi
  elif [ $paused = 1 ]; then
    lk; kill -CONT $P; paused=0; echo "$(date +%H:%M:%S) resumed $name"
  fi
done
wait $P; rc=$?
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) done $name (exit $rc)"
exit $rc
