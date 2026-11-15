#!/bin/bash
# run_l2_quiet.sh (Breakthrough, CHECK55 lever 2's pre-registered timing verdict, docs/CHECK55_LEVER2.md sections 4 + 7).
# The orchestrator 21:4x: chain it right after the engine profile tonight; quiet ticket, MEM < 1 GB, hard end 07:00 (Kamen
# may use the Mac after that), no quiet ticket before 23:30; if the profile runs late, skip (a 15-min quiet window Monday).
# Waits for the profile runner (pid $1) to exit, then:
# - not before 23:30 Sun; latest start 06:45 Mon (the runs fit a 12-min cap before 07:00);
# - HOLD / lid closed / HEALTH ACT skip; a PAUSE is waited out to the latest start, holding no lock;
# - a quiet ticket; bench_moe_dense.py l2_bench_quiet 7 16 chain,mine (THE VERDICT: dense_pre's default + B + moe_tail vs
#   K-73's chain, layers 0 / 24 / 47, 16 and 32 rows); then piece_moe_dense.py l2_pieces_quiet (information only);
# - a 12-min cap on both; unlock.
# Overrides for a dry simulation only: L2_D (log dir), L2_Q (the quiet dir), L2_START / L2_LATEST (epochs), L2_STEM.
cd ~/Breakthrough || exit 9
H=~/Breakthrough/scripts/m1/check55
D=${L2_D:-~/Breakthrough/data/m1/check55/lever2}
mkdir -p "$D"
L=$D/run_l2_quiet.log
Q=${L2_Q:-~/.local/share/quiet}
STEM=${L2_STEM:-quiet}
PP=${1:?usage: run_l2_quiet.sh PROFILE_RUNNER_PID}
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
START=${L2_START:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-15 23:30:00" +%s)}
LATEST=${L2_LATEST:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 06:45:00" +%s)}
LOCKED=0
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
quit() { [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; }; say "$1"; exit 7; }
stop_checks() {
  [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD present"
  lid_closed && quit "SKIP ($1): lid closed"
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && quit "SKIP ($1): HEALTH ACT"
  if [ -e "$Q/PAUSE" ]; then
    say "PAUSE present ($1): waiting it out until $(date -r $LATEST '+%a %H:%M') at the latest"
    while [ -e "$Q/PAUSE" ] && [ "$(date +%s)" -lt $LATEST ]; do
      sleep 10
      [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD (during a PAUSE)"
      lid_closed && quit "SKIP ($1): lid closed (during a PAUSE)"
    done
    [ -e "$Q/PAUSE" ] && quit "SKIP ($1): PAUSE still present at the latest start"
    say "PAUSE cleared ($1)"
  fi
}
say "armed (pid $$): waits for the profile runner pid $PP to exit; window $(date -r $START '+%a %H:%M') .. latest start $(date -r $LATEST '+%a %H:%M')"
while kill -0 "$PP" 2>/dev/null; do sleep 60; done
say "profile runner $PP exited; its last line: $(grep -v '^\[' ~/Breakthrough/data/m1/check55/engine/run_c55_engine.log | tail -1 | cut -c1-160)"
while [ "$(date +%s)" -lt $START ]; do sleep 60; done
[ "$(date +%s)" -gt $LATEST ] && quit "SKIP: past the latest start $(date -r $LATEST '+%a %H:%M') (the profile ran late): take a 15-min quiet window Monday"
while true; do
  stop_checks "before the ticket"
  export PEER_QUEUE_TIMEOUT=$(( LATEST - $(date +%s) > 60 ? LATEST - $(date +%s) : 60 ))
  peer lock quiet --queue 15m "Breakthrough: CHECK55 lever 2 quiet bench (MoE dense kernels, the pre-registered verdict), MEM 1 GB, ~5 min" >> "$L" 2>&1 \
    || quit "SKIP: quiet lock not granted by the latest start"
  LOCKED=1
  sleep 20
  if [ -e "$Q/HOLD" ] || lid_closed || grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; then stop_checks "under the ticket"; fi
  if [ -e "$Q/PAUSE" ]; then say "PAUSE under the ticket: released, waiting it out, then queueing again"; peer unlock >/dev/null 2>&1; LOCKED=0; continue; fi
  [ "$(date +%s)" -gt $LATEST ] && quit "SKIP: the ticket came after the latest start"
  break
done
say "quiet granted; NO-SWAP $(ns) GB; power: $(pmset -g | grep -iE 'powermode|lowpowermode' | tr -s ' ' | paste -sd ';' -)"
cap=$(( $(date +%s) + 720 ))
for job in "bench_moe_dense.py l2_bench_$STEM 7 16 chain,mine" "piece_moe_dense.py l2_pieces_$STEM 9 96 16:4:2:4,4:4:2:4"; do
  set -- $job
  ~/BigMoE/.venv/bin/python -I -B "$H/$1" "${@:2}" > >(grep --line-buffered -vE "^\[" >> "$L") 2>&1 &
  py=$!
  while kill -0 $py 2>/dev/null; do
    sleep 5
    if [ "$(date +%s)" -gt $cap ]; then say "12-min cap: stop $1"; kill -TERM $py; sleep 5; kill -9 $py 2>/dev/null; fi
    if [ -e "$Q/HOLD" ] || lid_closed; then say "HOLD or the lid closed during $1: stop"; kill -TERM $py; sleep 5; kill -9 $py 2>/dev/null; fi
  done
  wait $py; say "$1 rc $?"
  [ "$(date +%s)" -gt $cap ] && break
done
peer unlock >/dev/null 2>&1; LOCKED=0
say "L2-QUIET-DONE"
