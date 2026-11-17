#!/bin/bash
# xq_after_tue1.sh (Breakthrough, Tue 17 Nov 2026 09:5x; EXPERT REWRITE): the quiet run of screens X0 + X1
# (docs/EXPERT_REWRITE.md, pre-registered 0aee02c + ec211aa) + the first candidates' interleaved kernel bench
# (xrw_bench.py: orig = coh1cq vs tgl16, tgl16e, tgl16eu, w4a8, w4a8u; layer 20, 24 real blocks) + tgl16's timing ablations (32 rows); frozen copies in data/xrw/frozen_q3; re-armed 10:3x) on a quiet ticket queued the moment Leap's tue1 ends
# (NewIdeas 09:5x + Leap 09:48: its end marker ~/Leap/data/week98/c25/C25_tue1_DONE, written on any end, by ~12:00).
# Gates: the marker newer than arming; no BREAK98_HOLD, HOLD, PAUSE, lid closed or HEALTH ACT; Kamen away (away_secs
# >= 900 on two reads 30 s apart); NO-SWAP >= 12 GB (peak ~4 GB + 8). Then `peer lock quiet --queue`, both screens
# (each capped at 5 min), unlock. Never starts after LATEST. Marker data/xrw/XQ_END = "XQ END <time>: <why>" on any end.
# Usage: xq_after_tue1.sh   Env: DRY=1 (fake marker XQ_MARK, fake lock, 1 s waits, RJ_JOB in place of the screens)
cd "$HOME/Breakthrough" || exit 2
Q=$HOME/.local/share/quiet; PY=$HOME/BigMoE/.venv/bin/python
DRY=${DRY:-0}; ZZ=30; [ $DRY = 1 ] && ZZ=1
M=${XQ_MARK:-$HOME/Leap/data/week98/c25/C25_tue1_DONE}
H98=$HOME/NewIdeas/research/week98/BREAK98_HOLD
D=data/xrw; [ $DRY = 1 ] && D=${XQ_DRYDIR:?DRY needs XQ_DRYDIR}; L=$D/xq_runner.log; END=$D/XQ_END
T0=$(date +%s); LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${XQ_LATEST:-2026-11-17 14:30:00}" +%s)
LOCKED=0
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
fin() { echo "XQ END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
unlock() { [ $LOCKED = 1 ] && { [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; LOCKED=0; say "lock released"; }; }
quit() { unlock; fin "$1"; exit 7; }
trap 'unlock' EXIT
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
away() { local a; a=$( { [ -n "$RJ_AWAY_CMD" ] && $RJ_AWAY_CMD || bash "$Q/away_secs.sh"; } 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 900;; *) echo "$a";; esac; }
away2() { [ "$(away)" -ge 900 ] || return 1; sleep $ZZ; [ "$(away)" -ge 900 ]; }
blocked() {   # why the screens cannot start now ('' = clear)
  [ -e "$H98" ] && { echo "BREAK98_HOLD"; return; }
  [ -e "$Q/HOLD" ] && { echo "HOLD"; return; }
  [ -e "$Q/PAUSE" ] && { echo "PAUSE"; return; }
  ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { echo "HEALTH ACT"; return; }
  [ "$(ns)" -lt 12 ] && { echo "NO-SWAP $(ns) < 12 GB"; return; }
  echo ""
}
say "armed (pid $$): waits for $M newer than arming, then the gates and a queued quiet ticket; latest start $(date -r "$LATEST" '+%a %H:%M')"
while :; do
  if [ -e "$M" ] && [ "$(stat -f %m "$M")" -ge "$T0" ]; then break; fi
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: no tue1 end marker by the latest start"
  sleep $(( DRY == 1 ? 1 : 20 ))
done
say "tue1 marker: $(head -c 200 "$M")"
while :; do
  b=$(blocked)
> (a quote from the project owner, kept private)
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: blocked at the latest start ($b)"
  say "waiting: $b"; sleep $(( DRY == 1 ? 1 : 30 ))
done
if [ $DRY = 1 ]; then say "DRY: fake quiet lock"; LOCKED=1; else
  say "queueing the quiet ticket"
  peer lock quiet --queue 20m "MEM 4 GB: Breakthrough expert rewrite screens X0 + X1 + candidate and ablation kernel benches (~6 min)" >> "$L" 2>&1 \
    || quit "SKIP: the quiet lock was refused or timed out"
  LOCKED=1
fi
b=$(blocked); [ -n "$b" ] && quit "SKIP at the grant: $b"
> (a quote from the project owner, kept private)
say "granted; running (NO-SWAP $(ns) GB)"
run() {   # one screen, capped at 5 min
  if [ $DRY = 1 ]; then ${RJ_JOB:-sleep 2} >> "$D/xq_run.txt" 2>&1 & else "$PY" -I -B "$@" >> "$D/xq_run.txt" 2>&1 & fi
  local p=$! t=0
  while kill -0 $p 2>/dev/null; do sleep 1; t=$((t + 1)); [ $t -ge 300 ] && { kill -TERM $p; sleep 5; kill -9 $p 2>/dev/null; say "capped: $*"; }; done
  wait $p; echo $?
}
F=data/xrw/frozen_q3                                   # frozen copies (md5s in data/xrw/frozen_q3.md5), outputs to data/xrw
export XRW_OUTD="$HOME/Breakthrough/data/xrw"
r0=$(run $F/x0_overlap.py --rounds 5 --reps 10 --tag quiet)
r1=$(run $F/x1_rates.py --rounds 5 --reps 10 --tag quiet)
r2=$(run $F/xrw_bench.py --rounds 9 --nb 24 --cands tgl16,tgl16e,tgl16eu,w4a8,w4a8u --tag quiet)
r3=$(run $F/xrw_bench.py --rounds 7 --nb 24 --sizes 32 --cands tgl16,tgl16abl1,tgl16abl2,tgl16abl4,tgl16abl8,tgl16abl16 --tag quiet_abl)
unlock
v=$(grep -h -o -- "-> [A-Z]*$" "$D/x0_overlap_quiet.txt" 2>/dev/null | tail -1)
fin "done: X0 rc $r0 ${v:-no verdict line}; X1 rc $r1; bench rc $r2; ablations rc $r3"
