#!/bin/bash
# run_comqom.sh (Breakthrough, Mon 16 Nov 2026 13:5x): the optional comment-QOM map arms, amendment (f) of
# docs/CHECK_BYTES_900.md (NewIdeas 13:4x: slot (c), Tue ~07:00-07:15, read-out by ~07:30; information only).
# Starts at >= Tue 07:00 once no map process runs (pgrep 'run_ma[p].py', read twice 30 s apart): tonight's G1 / G2
# screen, or its Tue-morning fallback, goes first. Latest start Tue 08:20, hard stop 08:40 (TERM, KILL 20 s later).
# HOLD / lid / HEALTH ACT skip; a PAUSE is waited out; NO-SWAP >= 24 GB; a share lock (MEM 16 GB); swap guard +2 GB;
# the run (comqom_run_map.py, cwd ~/Breakthrough), then the read-out (comqom_read.py); an end marker on every end.
# Simulation overrides: CQ_D, CQ_Q, CQ_START / CQ_LATEST / CQ_HARD (epoch), CQ_PAT, CQ_NS_MIN, CQ_NOLOCK=1, CQ_JOB.
cd ~/Breakthrough || exit 9
D=${CQ_D:-$HOME/Breakthrough/data/m1/comqom_map}
Q=${CQ_Q:-$HOME/.local/share/quiet}
PAT=${CQ_PAT:-run_ma[p][.]py}
NSMIN=${CQ_NS_MIN:-24}
mkdir -p "$D"
L=$D/run.log
END=$D/COMQOM_END
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
START=${CQ_START:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 07:00:00" +%s)}
LATEST=${CQ_LATEST:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 08:20:00" +%s)}
HARD=${CQ_HARD:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 08:40:00" +%s)}
LOCKED=0
fin() { echo "COMQOM END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
quit() { [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; }; say "$1"; fin "$1"; exit 7; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
stop_checks() {
  [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD present"
  lid_closed && quit "SKIP ($1): lid closed"
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && quit "SKIP ($1): HEALTH ACT"
  if [ -e "$Q/PAUSE" ]; then
    say "PAUSE present ($1): waiting it out until $(date -r "$LATEST" '+%a %H:%M') at the latest"
    while [ -e "$Q/PAUSE" ] && [ "$(date +%s)" -lt "$LATEST" ]; do
      sleep 10
      [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD (during a PAUSE)"
      lid_closed && quit "SKIP ($1): lid closed (during a PAUSE)"
    done
    [ -e "$Q/PAUSE" ] && quit "SKIP ($1): PAUSE still present at the latest start"
    say "PAUSE cleared ($1)"
  fi
}
mapfree() { ! pgrep -f "$PAT" >/dev/null; }
say "armed (pid $$): starts at >= $(date -r "$START" '+%a %H:%M') once no '$PAT' process runs (read twice); latest start $(date -r "$LATEST" '+%a %H:%M'), hard stop $(date -r "$HARD" '+%a %H:%M')"
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$START" ] && mapfree; then
    sleep 30
    if mapfree; then say "trigger: the start time passed and no map process runs (read twice)"; break; fi
  fi
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: not started by the latest start (map process running: $(mapfree && echo n || echo y))"
  sleep 60
done
stop_checks "after the trigger"
a=$(ns)
say "NO-SWAP $a GB (need $NSMIN)"
while [ "$a" -lt "$NSMIN" ]; do
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: NO-SWAP $a < $NSMIN GB at the latest start"
  stop_checks "NO-SWAP wait"
  sleep 20; a=$(ns)
done
if [ -z "${CQ_NOLOCK:-}" ]; then
  peer lock share 20m --cpu 300 --gpu 60 "Breakthrough: comment-QOM map arms (3 arms x 12 code windows, ~8 min), MEM 16 GB" >> "$L" 2>&1 \
    || quit "SKIP: the share lock was refused"
  LOCKED=1
fi
sw0=$(swmb)
say "running (NO-SWAP $(ns) GB, swap $sw0 MB)"
if [ -n "${CQ_JOB:-}" ]; then JOB=($CQ_JOB); else JOB=("$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/comqom_run_map.py); fi
"${JOB[@]}" > >(grep --line-buffered -vE "^\[(tokenizer|mtplx|qwen4_exp|moe-sorted)" >> "$L") 2>&1 &
py=$!
stopped=""
while kill -0 $py 2>/dev/null; do
  sleep 10
  s=$(swmb)
  [ "$s" -gt $((sw0 + 2048)) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
  [ "$(date +%s)" -ge "$HARD" ] && stopped="hard stop $(date -r "$HARD" '+%H:%M')"
  { [ -e "$Q/HOLD" ] || lid_closed; } && stopped="HOLD or the lid closed"
  if [ -n "$stopped" ]; then say "stop: $stopped"; kill -TERM $py; sleep 20; kill -9 $py 2>/dev/null; break; fi
done
wait $py; rc=$?
[ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; }
say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0)"
rd="skipped (simulation)"
if [ -z "${CQ_JOB:-}" ]; then
  if "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/comqom_read.py >> "$L" 2>&1; then rd="written (data/m1/comqom_map.out)"; else rd="FAILED (log)"; fi
fi
fin "run rc $rc${stopped:+ ($stopped)}; read-out $rd"
