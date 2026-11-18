#!/bin/bash
# run_rec1a_c25_on_marker.sh (Breakthrough, Wed 18 Nov 2026 11:3x): NewIdeas 11:2x moves the module 1a engine run
# (rec1a_c25, pre-registered in docs/MODULE1A.md 'Tue 06:5x') from Thu 07:30 to tonight, right after BigMoE's night
# chain (CERTIFY98 capture + count check, ~02:40). Leap's slot #4 starts on this run's END marker.
# Waits, holding nothing, for BigMoE's end marker MARK (absent at arming; any content counts), then runs the FROZEN
# runner data/m1/rec1a_engine/run_rec1a_c25_1119.frozen.sh UNCHANGED with START = now and HARD = now + DUR minutes
# (no later than CAP). That runner keeps all its gates and guards: no c25 / rec1a loop, Kamen away on two reads,
# HOLD / PAUSE / lid / HEALTH ACT, the flag check, NO-SWAP >= 43 on two reads, the share lock 'MEM 35 GB', swap +2 GB,
# the yield when Kamen is back, the hard stop. It writes the END marker data/m1/rec1a_engine/REC1A_END
# ("REC1A END <MM-DD HH:MM:SS>: <why>") on any end: pass, fail, skip, yield, hard stop.
# This script writes the same END marker itself when no run starts: MARK not there by BY, HOLD while waiting, under
# 40 minutes left before CAP, or the runner exiting without writing it. Killed by a signal: no marker.
# Usage: run_rec1a_c25_on_marker.sh <MARK> "<BY YYYY-MM-DD HH:MM:SS>" "<CAP YYYY-MM-DD HH:MM:SS>"
# Env: RJW_DUR_MIN (90), RJ_TAG (rec1a_1119); simulation only: RJW_RUNNER, RJW_END, RJW_LOG, RJW_Q, RJW_POLL (60).
cd "$HOME/Breakthrough" || exit 2
D=data/m1/rec1a_engine
RUNNER=${RJW_RUNNER:-$D/run_rec1a_c25_1119.frozen.sh}
END=${RJW_END:-$D/REC1A_END}
TAG=${RJ_TAG:-rec1a_1119}
L=${RJW_LOG:-$D/${TAG}_wrapper.log}
Q=${RJW_Q:-$HOME/.local/share/quiet}
DUR=${RJW_DUR_MIN:-90}; POLL=${RJW_POLL:-60}
MARK=${1:?MARK}
BY=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?BY}" +%s) || exit 2
CAP=$(date -j -f "%Y-%m-%d %H:%M:%S" "${3:?CAP}" +%s) || exit 2
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
fin() { echo "REC1A END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; exit 7; }
[ -e "$MARK" ] && { say "REFUSED: $MARK already exists at arming"; exit 3; }
[ -e "$END" ] && { say "REFUSED: $END already exists at arming (Leap gates on it)"; exit 3; }
say "armed (pid $$, tag $TAG): waits for $MARK until $(date -r "$BY" '+%a %H:%M'), then runs $RUNNER with HARD = start + $DUR min, at most $(date -r "$CAP" '+%a %H:%M'); END marker $END"
while [ ! -e "$MARK" ]; do
  [ "$(date +%s)" -ge "$BY" ] && fin "SKIP (wrapper): $MARK not there by $(date -r "$BY" '+%a %H:%M')"
  [ -e "$Q/HOLD" ] && fin "SKIP (wrapper): HOLD present while waiting for $MARK"
  sleep "$POLL"
done
now=$(date +%s)
say "trigger: $MARK appeared ($(head -c 160 "$MARK" | tr -c '[:print:]' ' '))"
hard=$(( now + DUR * 60 )); [ "$hard" -gt "$CAP" ] && hard=$CAP
[ $(( hard - now )) -lt 2400 ] && fin "SKIP (wrapper): only $(( (hard - now) / 60 )) min left before $(date -r "$CAP" '+%a %H:%M')"
S=$(date -r "$now" '+%Y-%m-%d %H:%M:%S'); H=$(date -r "$hard" '+%Y-%m-%d %H:%M:%S')
say "running the frozen runner: START $S, HARD $H"
RJ_TAG=$TAG bash "$RUNNER" "$S" "$H"; rc=$?
if [ -e "$END" ]; then say "the runner ended rc $rc: $(cat "$END")"; exit 0; fi
fin "the runner exited rc $rc without its END marker"
