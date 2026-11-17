#!/bin/bash
# V3 KEY TEST (BigMoE, Tue 17 Nov 2026; the orchestrator 14:4x / 14:5x): Leap's ref2_switches_v3 (b4331e4: REF2_BOF and
# REF2_HC_B, default OFF; its in-loop ON-vs-OFF bit test never ran) end to end on the REAL r57 loop. Snapshot g1v3 (= g1b
# with Leap's scripts + third_party at b4331e4; build verified file by file), exact-45 (data/stack/x45rpl1_env.txt line 2,
# R57_PRESET=45 R57_REPS=1 R57_WARM=1), R57_LOGHASH=1, the same 4 code keys twice: OFF, then ON (REF2_BOF=1 REF2_HC_B=1).
# Verdict: scripts/v3kt_compare.py (PASS only if every key's tokens AND every check's verify logits are bitwise identical,
# ON engaged and OFF clean). Each run goes through scripts/kt_launch.py (the screen in-process + a V3KT counter line at exit).
# Gates as mid_run.sh, polled every 20 s until START_BY ("YYYY-mm-dd HH:MM"): no HOLD / PAUSE, lid open, no other session's
# lock with MEM >= 16 GB or quiet lock, away_secs >= 900, NO-SWAP >= GATE (36) on two reads 30 s apart; re-checked before
# the ON run. Mid-run: away < 900 on two reads 30 s apart, HOLD or the lid closed stop the run (YIELDED); swap +2 GB stops it.
# A share lock "MEM 34 GB" while it runs. EVERY end writes data/r57/V3KT_RESULT: "V3KT <verdict> <mm-dd HH:MM:SS>: <why>".
# V3KT_SIM=1: SIMD = the output dir (log, result, records); PYCMD = a command run per arm with $1 = off|on; FAKE_AWAYF /
# FAKE_PEERLS / FAKE_NOSWAP files; no lock.
cd ~/BigMoE || exit 9
Q=$HOME/.local/share/quiet; SIM=${V3KT_SIM:-0}; D=data/r57
SCREEN=data/r57/snap/g1v3/r57_screen_g1v3.py
ENVA=$(sed -n 2p data/stack/x45rpl1_env.txt | tr ',' ' ')
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,mbpp/2,mbpp/13
COMMON="R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_LOGHASH=1 R57_KEYS=$KEYS"
GATE=${GATE:-36}; AWAYMIN=900; OFF=v3kt1117off; ON=v3kt1117on
[ "$SIM" = 1 ] && D=${SIMD:?SIMD is required with V3KT_SIM=1}
L=$D/v3kt_1117.log; RES=$D/V3KT_RESULT
SB=$(date -j -f "%Y-%m-%d %H:%M:%S" "${START_BY:?START_BY (YYYY-mm-dd HH:MM) is required}:00" +%s) || exit 2
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
others_locks() { { if [ -n "${FAKE_PEERLS:-}" ]; then cat "$FAKE_PEERLS"; else peer ls 2>/dev/null; fi; } | grep "^- " \
    | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
big_lock() { others_locks | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'; }
quiet_held() { others_locks | grep -q "^· benchmarking"; }
away() { if [ -n "${FAKE_AWAYF:-}" ]; then cat "$FAKE_AWAYF" 2>/dev/null; else bash $Q/away_secs.sh 2>/dev/null; fi; }
away_ok() { local a; a=$(away); AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge "$AWAYMIN" ]; }
noswap() { if [ -n "${FAKE_NOSWAP:-}" ]; then cat "$FAKE_NOSWAP"; return; fi
           vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
gate_ok() { local r1 r2; r1=$(noswap); sleep $GS; r2=$(noswap); GR="$r1 / $r2"
            awk -v a=$r1 -v b=$r2 -v g=$GATE 'BEGIN {exit !(a >= g && b >= g)}'; }
GS=30; [ "$SIM" = 1 ] && GS=1
VERDICT="ERROR"; WHY="the runner ended early"; LOCKED=0; CP=""; WD=""
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
fin() { [ -n "$WD" ] && kill $WD 2>/dev/null
        if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then killtree $CP; for i in $(seq 1 120); do kill -0 $CP 2>/dev/null || break; sleep 1; done; fi
        [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "lock released"; }
        echo "V3KT $VERDICT $(date '+%m-%d %H:%M:%S'): $WHY" > $RES; say "END: V3KT $VERDICT: $WHY"; }
trap fin EXIT; trap 'WHY="$WHY; stopped by a signal"; exit 1' INT TERM
rm -f $RES; say "armed: start by $(date -r $SB '+%H:%M'); gates NO-SWAP >= $GATE, away >= $AWAYMIN, no HOLD / PAUSE / lid / big or quiet lock; snapshot g1v3 (Leap b4331e4); keys $KEYS$( [ "$SIM" = 1 ] && echo "; SIMULATION")"
wait_gates() {   # $1 = what for; returns when every gate is open, or ends the runner NOT RUN at START_BY (before the OFF run) / YIELDED (before ON)
  local w
  while :; do
    if [ -e $Q/HOLD ]; then w="HOLD"; elif [ -e $Q/PAUSE ]; then w="PAUSE"; elif lid_closed; then w="lid closed"
    elif big_lock; then w="another session's lock MEM >= 16 GB"; elif quiet_held; then w="another session's quiet lock"
> (a quote from the project owner, kept private)
    else say "gates open for $1 (NO-SWAP $GR, away $AW)"; return 0; fi
    [ "$w" != "${LASTW:-}" ] && say "waiting ($1): $w"; LASTW=$w
    [ "$(date +%s)" -ge "$SB" ] && { [ "$1" = OFF ] && VERDICT="NOT RUN" || VERDICT="YIELDED"; WHY="gates not open by $(date -r $SB '+%H:%M') before $1: $w"; exit 0; }
    sleep 20
  done; }
run_arm() {   # $1 = off|on
  local tag=$OFF sw=""; [ $1 = on ] && { tag=$ON; sw="REF2_BOF=1 REF2_HC_B=1"; }
  rm -f $D/${tag}_runs.json $D/${tag}_tokens.json
  if [ "$SIM" = 1 ]; then ( exec bash -c "$PYCMD" _ $1 ) > $D/$tag.log 2>&1 &
  else ( exec env $ENVA $COMMON R57_TAG=$tag $sw .venv/bin/python scripts/kt_launch.py $SCREEN 160 256 ) > $D/$tag.log 2>&1 &
  fi
  CP=$!; local s0; s0=$(swapmb); local lowa=0
  while kill -0 $CP 2>/dev/null; do
    sleep 15
    [ $(( $(swapmb) - s0 )) -gt 2048 ] && { VERDICT="ERROR"; WHY="swap guard +2 GB during $1 ($(swapmb) MB)"; exit 1; }
    if [ -e $Q/HOLD ] || lid_closed; then VERDICT="YIELDED"; WHY="HOLD or lid closed during $1"; exit 0; fi
> (a quote from the project owner, kept private)
  done
  wait $CP; RC=$?; CP=""; say "$1 run rc $RC ($(grep -c . $D/$tag.log) log lines; $(grep '^V3KT ' $D/$tag.log | tail -1 | cut -c1-200))"
  [ $RC = 0 ] || { VERDICT="ERROR"; WHY="the $1 run ended rc $RC (log $D/$tag.log)"; exit 1; }; }
wait_gates OFF
if [ "$SIM" != 1 ]; then peer lock share 30m --cpu 200 --gpu 100 "MEM 34 GB: BigMoE v3 key test (REF2_BOF / REF2_HC_B OFF vs ON, bitwise, ~12 min)" >> $L 2>&1 || { WHY="no share lock"; exit 1; }; fi
LOCKED=1; say "OFF run start"; run_arm off
wait_gates ON; say "ON run start"; run_arm on
OUT=$(python3 scripts/v3kt_compare.py $D $OFF $ON $KEYS 2>&1); CRC=$?
case $CRC in 0) VERDICT="PASS";; 1) VERDICT="FAIL";; *) VERDICT="ERROR";; esac; WHY="${OUT#*: }"; exit 0
