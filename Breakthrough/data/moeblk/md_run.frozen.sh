#!/bin/bash
# md_run.sh (Breakthrough, Tue 17 Nov 2026 12:2x; MOE BLOCK DIAG, docs/MOEBLOCK.md "DIAG" + both amendments, c614b4c+):
# one QUIET run on frozen copies data/moeblk/frozen_q3 (md5s in frozen_q3.md5):
# (A) bench.py reps 6 x rounds 5: today / knob / f1 / f2 / f2c, layers 20 + 44, 32 + 16 rows, 12 blocks (the 11:55 grid);
# (B) bench.py reps 1 x rounds 9: the same + the components c_exp / c_route / c_shared / c_comb (serialized calls).
# Gates: no BREAK98_HOLD / HOLD / PAUSE / lid closed / HEALTH ACT; away_secs >= 900 on two reads 30 s apart; NO-SWAP >= 13
# (peak ~5 GB + 8). Then `peer lock quiet --queue`; each step capped at 4 min; unlock; marker data/moeblk/MD_END on any
# end. Queues nothing before MD_START; never starts after MD_LATEST (default today 17:30).   Env: DRY=1 (fake lock, RJ_JOB), XQ_DRYDIR, RJ_AWAY_CMD.
cd "$HOME/Breakthrough" || exit 2
Q=$HOME/.local/share/quiet; PY=$HOME/BigMoE/.venv/bin/python
DRY=${DRY:-0}; ZZ=30; GATE=${MD_GATE:-13}; [ $DRY = 1 ] && { ZZ=1; GATE=0; }
H98=$HOME/NewIdeas/research/week98/BREAK98_HOLD
D=data/moeblk; [ $DRY = 1 ] && D=${XQ_DRYDIR:?DRY needs XQ_DRYDIR}; L=$D/md_runner.log; END=$D/MD_END
LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${MD_LATEST:-2026-11-17 17:30:00}" +%s)
F=data/moeblk/frozen_q3
LOCKED=0
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
fin() { echo "MD END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
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
START=$(date -j -f "%Y-%m-%d %H:%M:%S" "${MD_START:-2026-11-17 00:00:00}" +%s)
say "armed (pid $$): start $(date -r "$START" '+%a %H:%M'); holds, then a queued quiet ticket (memory + away at the grant); latest start $(date -r "$LATEST" '+%a %H:%M')"
while [ "$(date +%s)" -lt "$START" ]; do sleep $(( DRY == 1 ? 1 : 30 )); done   # no ticket before the start: a queued ticket stalls others' gates
while :; do                     # Kamen at the Mac: no start (the AWAY rule, 11-16 21:0x); wait, never skip, until LATEST
  while :; do
> (a quote from the project owner, kept private)
    [ -z "$b" ] && break
    [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: blocked at the latest start ($b)"
    say "waiting: $b"; sleep $(( DRY == 1 ? 1 : 30 ))
  done
  if [ $DRY = 1 ]; then say "DRY: fake quiet lock"; LOCKED=1; else
    say "queueing the quiet ticket"
    peer lock quiet --queue 20m "MEM 5 GB: Breakthrough MoE-block GPU split (2 short bench passes, ~2 min)" >> "$L" 2>&1 \
      || quit "SKIP: the quiet lock was refused or timed out"
    LOCKED=1
  fi
  [ "$(date +%s)" -ge "$((LATEST + 3600))" ] && quit "SKIP at the grant: granted after the latest start + 1 h"
> (a quote from the project owner, kept private)
  [ -z "$b" ] && break
  unlock; say "at the grant: $b; back to waiting"
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: blocked at the grant after the latest start ($b)"
  sleep $(( DRY == 1 ? 1 : 120 ))
done
say "granted; running (NO-SWAP $(ns) GB)"
run() {
  if [ $DRY = 1 ]; then ${RJ_JOB:-sleep 2} >> "$D/md_run.txt" 2>&1 & else "$PY" -I -B "$@" >> "$D/md_run.txt" 2>&1 & fi
  local p=$! t=0
  while kill -0 $p 2>/dev/null; do sleep 1; t=$((t + 1)); [ $t -ge 240 ] && { kill -TERM $p; sleep 5; kill -9 $p 2>/dev/null; say "capped: $*"; }; done
  wait $p; echo $?
}
export MOEBLK_OUTD="$HOME/Breakthrough/data/moeblk" XRW_OUTD="$HOME/Breakthrough/data/xrw"
r1=$(run $F/moeblk/bench.py --layers 20,44 --nb 12 --reps 6 --rounds 5 --rows 32,16 --arms today,knob,f1,f2,f2c --tag q3_r6)
r2=$(run $F/moeblk/bench.py --layers 20,44 --nb 12 --reps 1 --rounds 9 --rows 32,16 --arms today,knob,f1,f2,f2c,c_exp,c_route,c_shared,c_comb --tag q3_r1)
unlock
fin "done: run A rc $r1; run B rc $r2"
