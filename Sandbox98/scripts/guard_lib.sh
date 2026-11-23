# Polite guards for Sandbox98 runners (sourced): quiet tickets, PAUSE, NO-SWAP, swap growth; yield + rerun.
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
QD=${GUARD_QDIR:-$HOME/.local/share/quiet}     # PAUSE (and HOLD) live here; GUARD_QDIR: tests only
# Other sessions' rows are the "- " lines (my own row is "You:"; never skip myself with grep -v <my name>: that hides
# every session whose status names me). Lock state sits in " · " segments (memory peer-ls-lock-segments); MEM is read
# inside the lock segments only, so a status text that mentions "MEM n GB" doesn't count.
others() { peer ls 2>/dev/null | grep "^- "; }
locks_of() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
quiet_others() { others | locks_of | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
mems() { others | locks_of | grep -oE "MEM ~?[0-9]+(\.[0-9]+)? GB" | grep -oE "[0-9]+(\.[0-9]+)?"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
big_others() { mems | awk -v b=${GUARD_BIG_GB:-1000} '$1 >= b' | grep -c . ; }   # a big slot (e.g. 45 GB) held by another session
mem_others() { mems | awk -v b=${GUARD_MEM_OTHERS_GT:-4.99} '$1 > b' | grep -c . ; }   # by day: one job above ~4-5 GB at a time
gpu_others() {   # others' shared runs at >= GUARD_GPU_OTHERS_MAX % GPU (a long reason can lose its MEM tag: memory peer-ls-truncates-mem)
  others | locks_of | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' |
    awk -v g=${GUARD_GPU_OTHERS_MAX:-101} '$1 >= g' | grep -c . ; }
SWAP0=$(swap_mb)
ready() {
  [ -f $QD/PAUSE ] && return 1
  if [ -n "${GUARD_HELD_FN:-}" ] && $GUARD_HELD_FN; then return 1; fi   # e.g. a shut lid with no external display
  [ "$(quiet_others)" -eq 0 ] || return 1
  [ "$(big_others)" -eq 0 ] || return 1
  [ "$(noswap)" -ge ${GUARD_NOSWAP:-12} ] || return 1
  if [ -n "${NO_MEM_OTHERS:-}" ]; then [ "$(mem_others)" -eq 0 ] && [ "$(gpu_others)" -eq 0 ] || return 1; fi
  [ $(( $(swap_mb) - SWAP0 )) -lt 1500 ] || { echo "$(date +%H:%M:%S) swap grew > 1.5 GB since the start: stopping"; GEXIT="swap grew > 1.5 GB since the start"; exit 0; }
  return 0
}
lock_step() {   # MINUTES NAME: (re)take this step's share lock for MINUTES; 0 when peer took it
  local out; out=$(peer lock share ${1}m --cpu 200 --gpu ${GUARD_GPU:-25} "Sandbox98 MEM ${MEMGB:-3.6} GB ${GUARD_MEMNOTE:-measured peak}: step $2" 2>&1 | head -1)
  echo "$(date +%H:%M:%S) lock $2 for $1 min: $out"
  case "$out" in "shared run reserved"*) return 0 ;; *) return 1 ;; esac
}
eta_s() {   # LOG: "i seconds-left" from the step's last "step i/N ... X s/step" line (X = the run's own mean), counting
  # GUARD_TAIL_STEPS more step-equivalents for the work after training (read-outs); nothing before step 40 (warm-up)
  tail -c 20000 "$1" 2>/dev/null | grep -oE "step [0-9]+/[0-9]+ .* [0-9.]+s/step" | tail -1 |
    sed -E 's#step ([0-9]+)/([0-9]+) .* ([0-9.]+)s/step#\1 \2 \3#' |
    awk -v t=${GUARD_TAIL_STEPS:-0} '$1 >= 40 {printf "%d %d", $1, ($2 - $1 + t) * $3}'
}
stop_now() {   # 0 = stop for good (GUARD_STOP_FN, e.g. Kamen's HOLD or a hard end; it sets GSTOP to the reason)
  [ -n "${GUARD_STOP_FN:-}" ] && $GUARD_STOP_FN
}
stop_group() {   # SP WAIT: TERM the run's process group, wait up to WAIT s until ALL of it has exited (a SIGTERM state
  # save must finish; a wrapper like /usr/bin/time dies at once), then KILL what is left
  local sp=$1 w=${2:-5} t=0
  kill -TERM -- -$sp 2>/dev/null
  while pgrep -g $sp >/dev/null 2>&1 && [ $t -lt $w ]; do sleep 1; t=$((t + 1)); done
  pgrep -g $sp >/dev/null 2>&1 && kill -KILL -- -$sp 2>/dev/null
  return 0
}
guarded() {   # NAME LOG MINUTES CMD...   run CMD in its own session under a share lock; yield to quiet tickets
  # and come back (up to GUARD_MAX_YIELDS times, default 40). Hooks (L1CF, 11-15): GUARD_HELD_FN (0 = held, e.g. a shut
  # lid: wait / yield like PAUSE), GUARD_STOP_FN (0 = stop for good, e.g. HOLD or a hard end: TERM the step, which saves
  # its state, unlock, return 8; it sets GSTOP), GUARD_PIDFILE (the step's process group, for the runner's own traps).
  # A refused share lock no longer starts the step: it waits and re-checks. With GUARD_RUN set, a step that finished (exit 0) leaves
  # logs/.done/$GUARD_RUN/NAME, and a restarted runner skips it.
  # Memory guard while it runs (orchestrator 06:3x): GUARD_SWAP_KILL_MB (swap growth since this start) and
  # GUARD_NOSWAP_KILL (NO-SWAP GB, two reads in a row) kill the step (SIGTERM first: it saves its state) and stop the runner.
  local name=$1 log=$2 mins=$3; shift 3
  local tries=0 mark=""
  if [ -n "${GUARD_RUN:-}" ]; then
    mark=logs/.done/$GUARD_RUN/$name; mkdir -p logs/.done/$GUARD_RUN
    [ -f "$mark" ] && { echo "$(date +%H:%M:%S) $name: done before, skip"; return 0; }
  fi
  while true; do
    until ready; do
      stop_now && { echo "$(date +%H:%M:%S) $name: not started ($GSTOP)"; return 8; }
      sleep ${GUARD_POLL:-20}
    done
    sleep ${GUARD_SETTLE:-60}; ready || continue
    stop_now && { echo "$(date +%H:%M:%S) $name: not started ($GSTOP)"; return 8; }
    echo "$(date +%H:%M:%S) start $name (swap $(swap_mb) MB, NO-SWAP $(noswap) GB)"
    lock_step $mins $name || { echo "$(date +%H:%M:%S) $name: share lock refused, waiting"; sleep 60; continue; }
    local lock_end=$(( $(date +%s) + mins * 60 )) last_i=-1 sized=0
    $PY -I -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" > $log 2>&1 &
    local SP=$! yielded=0 sw0=$(swap_mb) low=0
    [ -n "${GUARD_PIDFILE:-}" ] && echo $SP > "$GUARD_PIDFILE"
    while kill -0 $SP 2>/dev/null; do
      sleep ${GUARD_TICK:-20}
      # Hold the share lock to the step's real end (orchestrator 09:4x: a 60 min lock lapsed while LSQ ran ~2x longer
      # under a 5-way GPU share). At each new progress line from step 40 on: needed end = now + 1.15 x (steps left +
      # GUARD_TAIL_STEPS) x the run's own s/step + 5 min. The first such line always re-sizes the lock (to that + 10 min);
      # later ones only when the lock would end sooner, or is more than 30 min too long. Fallback: renew for 20 min when
      # the lock is 5 min from lapsing (read-outs that run long).
      local now=$(date +%s) e=$(eta_s $log)
      if [ -n "$e" ] && [ "${e% *}" != "$last_i" ]; then
        last_i=${e% *}
        local want=$(( now + ${e#* } * 115 / 100 + 300 ))
        if [ $sized = 0 ] || [ $want -gt $lock_end ] || [ $want -lt $((lock_end - 1800)) ]; then
          lock_step $(( (want + 600 - now + 59) / 60 )) $name && { lock_end=$((want + 600)); sized=1; }
        fi
      fi
      if [ $((lock_end - now)) -lt 300 ]; then
        if lock_step 20 $name; then lock_end=$((now + 1200)); else lock_end=$((now + 360)); fi   # refused: retry in ~1 min
      fi
      if stop_now; then
        echo "$(date +%H:%M:%S) $name: stop ($GSTOP): TERM, it saves its state"; stop_group $SP ${GUARD_KILL_WAIT:-5}; wait $SP
        peer unlock 2>&1 | tail -1; [ -n "${GUARD_PIDFILE:-}" ] && rm -f "$GUARD_PIDFILE"; return 8
      fi
      if [ "$(quiet_others)" -gt 0 ] || [ -f $QD/PAUSE ] || [ "$(big_others)" -gt 0 ] || { [ -n "${GUARD_HELD_FN:-}" ] && $GUARD_HELD_FN; }; then
        echo "$(date +%H:%M:%S) quiet ticket, PAUSE, a big slot or a hold appeared: yield (kill $name)"; stop_group $SP ${GUARD_KILL_WAIT:-5}
        yielded=1; break
      fi
      local sw=$(swap_mb)
      if [ -n "${GUARD_SWAP_KILL_MB:-}" ] && [ $((sw - sw0)) -gt $GUARD_SWAP_KILL_MB ]; then
        echo "$(date +%H:%M:%S) MEMORY GUARD: swap $sw MB = +$((sw - sw0)) MB since $name started (line +$GUARD_SWAP_KILL_MB MB): kill and stop"
        GEXIT="memory guard: swap +$((sw - sw0)) MB"; stop_group $SP ${GUARD_KILL_WAIT:-5}; wait $SP; peer unlock 2>&1 | tail -1; exit 3
      fi
      if [ -n "${GUARD_NOSWAP_KILL:-}" ]; then
        local ns=$(noswap)
        if [ "$ns" -lt $GUARD_NOSWAP_KILL ]; then low=$((low + 1)); else low=0; fi
        if [ $low -ge 2 ]; then
          echo "$(date +%H:%M:%S) MEMORY GUARD: NO-SWAP $ns GB < $GUARD_NOSWAP_KILL GB twice in a row: kill and stop"
          GEXIT="memory guard: NO-SWAP $ns GB twice"; stop_group $SP ${GUARD_KILL_WAIT:-5}; wait $SP; peer unlock 2>&1 | tail -1; exit 3
        fi
      fi
    done
    wait $SP; local rc=$?
    peer unlock 2>&1 | tail -1
    [ -n "${GUARD_PIDFILE:-}" ] && rm -f "$GUARD_PIDFILE"
    if [ $yielded = 0 ]; then
      echo "$(date +%H:%M:%S) done $name (exit $rc, swap $(swap_mb) MB)"
      [ $rc = 0 ] && [ -n "$mark" ] && touch "$mark"
      return $rc
    fi
    tries=$((tries + 1))
    [ $tries -ge ${GUARD_MAX_YIELDS:-40} ] && { echo "$(date +%H:%M:%S) $name: yielded $tries times, giving up"; GEXIT="yielded $tries times, gave up"; exit 1; }
    echo "$(date +%H:%M:%S) $name: yield $tries, will resume when the Mac is free"
  done
}
