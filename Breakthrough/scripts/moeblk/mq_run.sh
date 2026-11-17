#!/bin/bash
# mq_run.sh (Breakthrough, Tue 17 Nov 2026 11:4x; MOE BLOCK proof): one QUIET run, launched on the orchestrator's word:
# (1) the pre-registered block bench (docs/MOEBLOCK.md f6f806e) on frozen copies data/moeblk/frozen_q2:
#     today / knob (tgl16eu, knob_v2) / f1 / f2, layers 20 + 44, 32 + 16 rows, 12 blocks x 6 calls x 5 rounds;
# (2) tgl16's dependency ablations (32 rows; abl4 = no MMA, abl32 = MMA on another simdgroup's tile, abl36 = both).
# Gates: no BREAK98_HOLD / HOLD / PAUSE / lid closed / HEALTH ACT; away_secs >= 900 on two reads 30 s apart; NO-SWAP >= 14
# (peak ~6 GB + 8). Then `peer lock quiet --queue`; each step capped at 6 min; unlock; marker data/moeblk/MQ_END on any
# end. Never starts after MQ_LATEST (default today 17:30).   Env: DRY=1 (fake lock, RJ_JOB), XQ_DRYDIR, RJ_AWAY_CMD.
cd "$HOME/Breakthrough" || exit 2
Q=$HOME/.local/share/quiet; PY=$HOME/BigMoE/.venv/bin/python
DRY=${DRY:-0}; ZZ=30; GATE=${MQ_GATE:-14}; [ $DRY = 1 ] && { ZZ=1; GATE=0; }
H98=$HOME/NewIdeas/research/week98/BREAK98_HOLD
D=data/moeblk; [ $DRY = 1 ] && D=${XQ_DRYDIR:?DRY needs XQ_DRYDIR}; L=$D/mq_runner.log; END=$D/MQ_END
LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${MQ_LATEST:-2026-11-17 17:30:00}" +%s)
F=data/moeblk/frozen_q2
LOCKED=0
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
fin() { echo "MQ END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
unlock() { [ $LOCKED = 1 ] && { [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; LOCKED=0; say "lock released"; }; }
quit() { unlock; fin "$1"; exit 7; }
trap 'unlock' EXIT
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
away() { local a; a=$( { [ -n "$RJ_AWAY_CMD" ] && $RJ_AWAY_CMD || bash "$Q/away_secs.sh"; } 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 900;; *) echo "$a";; esac; }
away2() { [ "$(away)" -ge 900 ] || return 1; sleep $ZZ; [ "$(away)" -ge 900 ]; }
blocked() {
  [ -e "$H98" ] && { echo "BREAK98_HOLD"; return; }
  [ -e "$Q/HOLD" ] && { echo "HOLD"; return; }
  [ -e "$Q/PAUSE" ] && { echo "PAUSE"; return; }
  ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { echo "HEALTH ACT"; return; }
  [ "$(ns)" -lt "$GATE" ] && { echo "NO-SWAP $(ns) < $GATE GB"; return; }
  echo ""
}
blocked_static() {   # the holds only: memory and the away rule are checked at the grant (the orchestrator 11:52: queue now)
  [ -e "$H98" ] && { echo "BREAK98_HOLD"; return; }
  [ -e "$Q/HOLD" ] && { echo "HOLD"; return; }
  [ -e "$Q/PAUSE" ] && { echo "PAUSE"; return; }
  ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { echo "HEALTH ACT"; return; }
  echo ""
}
say "armed (pid $$): holds, then a queued quiet ticket (memory + away at the grant); latest start $(date -r "$LATEST" '+%a %H:%M')"
while :; do
  b=$(blocked_static)
  [ -z "$b" ] && break
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: blocked at the latest start ($b)"
  say "waiting: $b"; sleep $(( DRY == 1 ? 1 : 30 ))
done
if [ $DRY = 1 ]; then say "DRY: fake quiet lock"; LOCKED=1; else
  say "queueing the quiet ticket"
  peer lock quiet --queue 20m "MEM 6 GB: Breakthrough MoE-block bench (today / knob / F1 / F2, 2 layers x 2 sizes) + expert ablations (~6 min)" >> "$L" 2>&1 \
    || quit "SKIP: the quiet lock was refused or timed out"
  LOCKED=1
fi
[ "$(date +%s)" -ge "$((LATEST + 3600))" ] && quit "SKIP at the grant: granted after the latest start + 1 h"
b=$(blocked); [ -n "$b" ] && quit "SKIP at the grant: $b"
> (a quote from the project owner, kept private)
say "granted; running (NO-SWAP $(ns) GB)"
run() {
  if [ $DRY = 1 ]; then ${RJ_JOB:-sleep 2} >> "$D/mq_run.txt" 2>&1 & else "$PY" -I -B "$@" >> "$D/mq_run.txt" 2>&1 & fi
  local p=$! t=0
  while kill -0 $p 2>/dev/null; do sleep 1; t=$((t + 1)); [ $t -ge 360 ] && { kill -TERM $p; sleep 5; kill -9 $p 2>/dev/null; say "capped: $*"; }; done
  wait $p; echo $?
}
export MOEBLK_OUTD="$HOME/Breakthrough/data/moeblk" XRW_OUTD="$HOME/Breakthrough/data/xrw"
r1=$(run $F/moeblk/bench.py --layers 20,44 --nb 12 --reps 6 --rounds 5 --rows 32,16 --arms today,knob,f1,f2 --tag q1)
r2=$(run $F/xrw/xrw_bench.py --rounds 7 --nb 24 --sizes 32 --cands tgl16eu,tgl16,tgl16abl4,tgl16abl32,tgl16abl36 --tag q2_abl)
unlock
fin "done: block bench rc $r1; ablations rc $r2"
