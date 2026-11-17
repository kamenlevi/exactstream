#!/bin/bash
# BREAK98 MID CAPTURE runner (Tue 17 Nov 2026; the orchestrator 07:4x, NewIdeas 07:41 / 07:44): gates -> share lock ->
# scripts/dd_mid.py (layers 2/4/8/16/24 of the 125B's pre-mixer widened stream on the bank's heplus24 + mbppbank answer
# rows) -> unlock -> a marker on EVERY end.
# Gates, polled every 20 s until START_BY ("YYYY-mm-dd HH:MM", required): no HOLD; no PAUSE and the lid open; no other
#   session's lock with MEM >= 16 GB (held, shared or queued) and no quiet lock held; away_secs >= AWAYMIN (default 900);
#   NO-SWAP (free + purgeable + file-backed) >= GATE GB on two reads 30 s apart (default 28: the bank's run was MEM ~20 GB).
# BIGMOE_EXT_BUILD defaults to build_qom, as the bank's own launch (11-14): records3 is native+p11+pad, which needs an
#   ext with set_direct_split (dd_bank's copied default build_next lacks it).
# Mid-run: dd_mid.py stops at a file boundary when away_secs < AWAYMIN or the stop file (data/deepdraft/ceiling/MID.STOP)
#   appears; this runner writes the stop file at STOP_AT (optional, "YYYY-mm-dd HH:MM": e.g. before another slot) and
#   TERMs the capture 4 min later if it still runs, or at once when swap grows > 2 GB. A finished tag stays; a re-run
#   skips it. The unfinished tag's .tmp file is removed.
# Marker data/deepdraft/ceiling/MID_CAPTURE_DONE: "MID CAPTURE END <mm-dd HH:MM:SS>: PASS|FAIL|STOPPED|NOT RUN|ERROR (...)".
# Log data/deepdraft/ceiling/mid_capture.log (+ the capture's own output in mid_capture_py.log).
# Sims: SIMD (a folder for the log / marker / stop file), FAKE_AWAYF (a file with the away seconds), FAKE_PEERLS (a
#   peer ls text), PYCMD (replaces the capture command), NOLOCK=1 (no peer lock).
set -u
cd ~/BigMoE || exit 9
Q=$HOME/.local/share/quiet; W=${SIMD:-data/deepdraft/ceiling}
L=$W/mid_capture.log; PL=$W/mid_capture_py.log; MK=$W/MID_CAPTURE_DONE; STOPF=$W/MID.STOP
GATE=${GATE:-28}; AWAYMIN=${AWAYMIN:-900}; LOCKMEM=${LOCKMEM:-24}; LOCKMIN=${LOCKMIN:-50}; NOLOCK=${NOLOCK:-0}
PYCMD=${PYCMD:-.venv/bin/python -B scripts/dd_mid.py 100 heplus24,mbppbank}
export BIGMOE_EXT_BUILD=${BIGMOE_EXT_BUILD:-build_qom}
case "$AWAYMIN$GATE$LOCKMEM$LOCKMIN" in *[!0-9.]*) echo "AWAYMIN / GATE / LOCKMEM / LOCKMIN must be numbers" >&2; exit 2;; esac
SB=$(date -j -f "%Y-%m-%d %H:%M:%S" "${START_BY:?START_BY (YYYY-mm-dd HH:MM) is required}:00" +%s) || exit 2
SA=0; [ -n "${STOP_AT:-}" ] && { SA=$(date -j -f "%Y-%m-%d %H:%M:%S" "$STOP_AT:00" +%s) || exit 2; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
others_locks() {  # other sessions' lock segments (rows start "- "; mine is "You:"), as g1_follow.sh
  { if [ -n "${FAKE_PEERLS:-}" ]; then cat "$FAKE_PEERLS"; else peer ls 2>/dev/null; fi; } | grep "^- " \
    | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"
}
big_lock() { others_locks | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'; }
quiet_held() { others_locks | grep -q "^· benchmarking"; }
away_ok() {
  local a; if [ -n "${FAKE_AWAYF:-}" ]; then a=$(cat "$FAKE_AWAYF" 2>/dev/null); else a=$(bash $Q/away_secs.sh 2>/dev/null); fi
  AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge "$AWAYMIN" ]
}
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
gate_ok() { local r1 r2; r1=$(noswap); sleep 30; r2=$(noswap); GR="$r1 / $r2"
  awk -v a="$r1" -v b="$r2" -v g="$GATE" 'BEGIN {exit !(a >= g && b >= g)}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }

STATUS="ERROR (the runner ended early)"; LOCKED=0; PY=""; MADE_STOP=0
fin() {
  trap - EXIT INT TERM
  [ -n "$PY" ] && kill -0 "$PY" 2>/dev/null && { kill -TERM "$PY" 2>/dev/null; wait "$PY" 2>/dev/null; }
  rm -f "${W:?}"/mid_heplus24*.npz.tmp "${W:?}"/mid_mbppbank*.npz.tmp
  [ "$LOCKED" = 1 ] && { peer unlock > /dev/null 2>&1; say "share lock released"; }
  [ "$MADE_STOP" = 1 ] && rm -f "$STOPF"
  echo "MID CAPTURE END $(date '+%m-%d %H:%M:%S'): $STATUS" > $MK
  say "END: $STATUS (marker $MK)"
}
trap fin EXIT
trap 'STATUS="STOPPED (runner TERM / INT)"; exit 0' INT TERM

rm -f "$STOPF"
say "armed: start by $START_BY${STOP_AT:+, stop at $STOP_AT}; gate NO-SWAP >= $GATE, away >= $AWAYMIN s, lock MEM $LOCKMEM GB for $LOCKMIN min; BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD; cmd $PYCMD"
LASTW=""
while :; do
  w=""
  if [ "$(date +%s)" -ge "$SB" ]; then STATUS="NOT RUN (gates not open by $START_BY: ${LASTW:-?})"; exit 0
  elif [ -e $Q/HOLD ]; then w="HOLD"
  elif [ -e $Q/PAUSE ] || lid_closed; then w="PAUSE or the lid closed"
  elif big_lock; then w="another session's MEM >= 16 GB lock"
  elif quiet_held; then w="another session's quiet lock"
> (a quote from the project owner, kept private)
  elif gate_ok && away_ok; then break
  else w="NO-SWAP $GR GB < $GATE"
  fi
  [ "$LASTW" = "$w" ] || { say "waiting: $w"; LASTW="$w"; }
  sleep 20
done
S0=$(swap_mb)
say "start: gates open (no HOLD / PAUSE / big or quiet lock; lid open), NO-SWAP $GR GB >= $GATE; away_secs $AW >= $AWAYMIN; swap $S0 MB"
if [ "$NOLOCK" != 1 ]; then
  LO=$(peer lock share ${LOCKMIN}m --cpu 200 --gpu 100 "MEM $LOCKMEM GB: BigMoE BREAK98 mid capture (dd_mid.py, ~30 min)" 2>&1); LR=$?
  say "share lock rc $LR: $(echo "$LO" | head -n 1 | cut -c1-160)"
  [ $LR = 0 ] && LOCKED=1
fi
AWAYMIN=$AWAYMIN MID_STOP=$STOPF $PYCMD > $PL 2>&1 &
PY=$!
say "capture started: pid $PY (output $PL)"
TERMED=""
while kill -0 $PY 2>/dev/null; do
  sleep 10
  now=$(date +%s)
  if [ "$SA" -gt 0 ] && [ "$now" -ge "$SA" ] && [ ! -e "$STOPF" ]; then touch "$STOPF"; MADE_STOP=1; say "STOP_AT $STOP_AT: stop file written (the capture stops at the next file boundary)"; fi
  if [ "$SA" -gt 0 ] && [ "$now" -ge $((SA + 240)) ] && [ -z "$TERMED" ]; then TERMED="STOP_AT + 4 min"; kill -TERM $PY 2>/dev/null; fi
  S1=$(swap_mb); if awk -v a="$S0" -v b="$S1" 'BEGIN {exit !(b - a > 2048)}'; then [ -z "$TERMED" ] && { TERMED="swap guard: swap $S0 -> $S1 MB"; say "$TERMED"; kill -TERM $PY 2>/dev/null; }; fi
done
wait $PY; RC=$?; PY=""
TAIL=$(grep -E 'WROTE|STOPPED|CHECK FAILED|Traceback|Error' $PL | tail -n 3 | cut -c1-200 | tr '\n' ' ')
DONE=$(ls $W 2>/dev/null | grep -E '^mid_(heplus24|mbppbank)\.npz$' | tr '\n' ' ')
case "$RC" in
  0) STATUS="PASS (rc 0; wrote ${DONE:-?}; all bitwise checks pass)";;
  3) STATUS="FAIL (rc 3; a bitwise check failed: $TAIL)";;
  4) STATUS="STOPPED (rc 4; done: ${DONE:-none}; $TAIL)";;
  *) if [ -n "$TERMED" ]; then STATUS="STOPPED (rc $RC, $TERMED; done: ${DONE:-none})"; else STATUS="ERROR (rc $RC; $TAIL)"; fi;;
esac
exit 0
