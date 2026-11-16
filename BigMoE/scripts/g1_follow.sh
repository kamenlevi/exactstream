#!/bin/bash
# The G1 key test's arming wrapper (BigMoE, Mon 16 Nov; NewIdeas 12:3x-12:5x, EvalSuite PREREG_lossygate amendment 5a).
# It starts scripts/g1_keytest.py --snap g1a when ALL hold:
#   - EvalSuite's 1c-tiers chain has ended: $ES/$ESMK with mtime >= ESMIN AND pid ESPID gone. ESMK defaults to
#     T1C_MON3_END (Breakthrough 12:5x: T1C_MON2_END marks the chain's PAUSE for Sandbox98's SIB test, not its end; the
#     resumed mon3 chain writes T1C_MON3_END). ESPID = the mon3 chain's pid (EvalSuite sends it; required). ESMIN defaults to
#     the arming time;
#   - Leap's model-level int8 slice test passed: $W/G1A_GO exists (BigMoE creates it on Leap's PASS message);
#   - no other session's lock of MEM >= 16 GB held or queued; no t30c drafter job; no process matching AFTER (optional);
#   - no HOLD / PAUSE, the lid open;
#   - NO-SWAP >= GATE on two reads 30 s apart (default 40: the orchestrator's peak - 2.6, as for t1c1).
# The at-Mac yield is lifted (the orchestrator 11:5x, relaying Kamen), so there is no away gate; the memory gate stays.
# The key test: --gate GATE --away-min 0 --start-by DEADLINE --hard DEADLINE + HARDMIN (default 8 min: clear of M4 at 23:05).
# EVERY end writes $W/G1A_KEYTEST_DONE: "G1A KEYTEST END <mm-dd HH:MM:SS>: <PASS|FAIL|STOPPED|NOT RUN> (<why>)" (Breakthrough's
# 4-arm map screen gates on it), unless $W/G1A.NOMARK exists.
# usage: ESPID=<mon3 chain pid> DEADLINE="2026-11-16 22:55" bash scripts/g1_follow.sh
# Simulation only: SIMD (log / marker dir), STUB (a python file run in place of the key test), FAKE_ES (EvalSuite results dir),
# ESPID, ESMIN ("YYYY-mm-dd HH:MM:SS"), KT_EXTRA (extra key-test arguments, e.g. "--stub X --nolock --out D"), POLL (s).
set -u
cd ~/BigMoE || exit 1
DEADLINE=${DEADLINE:?YYYY-mm-dd HH:MM}; POLL=${POLL:-60}; STUB=${STUB:-}; GATE=${GATE:-40}; AFTER=${AFTER:-}; HARDMIN=${HARDMIN:-8}
W=${SIMD:-data/r57}; L=$W/g1_follow.log; MK=$W/G1A_KEYTEST_DONE; NOMARK=$W/G1A.NOMARK; GO=$W/G1A_GO; Q=$HOME/.local/share/quiet
ES=${FAKE_ES:-$HOME/EvalSuite/results}; ESPID=${ESPID:?the EvalSuite chain pid}; ESMK=${ESMK:-T1C_MON3_END}; KT_EXTRA=${KT_EXTRA:-}
ESMIN=${ESMIN:-$(date '+%Y-%m-%d %H:%M:%S')}
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DEADLINE:00" +%s) || exit 2
ESMT=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ESMIN" +%s) || exit 2
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
newer() { [ -e "$1" ] && [ "$(stat -f %m "$1")" -ge "$2" ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
big_lock() {      # another session's lock (held or queued) with MEM >= 16 GB in its reason
  peer ls 2>/dev/null | grep -iE "quiet|shared|queued|lock" | grep -v "BigMoE" | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'
}
t30c_up() { ps -axo args | grep -qE "^(/bin/)?bash [^ ]*sb98/t30c_run\.sh|^[^ ]*python[^ ]* (-B )?scripts/dd_ttt\.py (train|eval) .*t30c"; }
after_up() { [ -n "$AFTER" ] && pgrep -f "$AFTER" > /dev/null; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
gate_ok() { local r1 r2; r1=$(noswap); sleep 30; r2=$(noswap); GR="$r1 / $r2"   # g1_keytest.py's noswap_gb (GB = 1e9)
            awk -v a=$r1 -v b=$r2 -v g=$GATE 'BEGIN {exit !(a >= g && b >= g)}'; }
STATUS="NOT RUN (armed $(date '+%m-%d %H:%M'))"; CP=""; LASTW=""; LASTG=0
cleanup() { if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then kill -TERM $CP 2>/dev/null
              for i in $(seq 1 90); do kill -0 $CP 2>/dev/null || break; sleep 1; done; fi
            if [ -e $NOMARK ]; then say "END without the marker (NOMARK): $STATUS"
            else echo "G1A KEYTEST END $(date '+%m-%d %H:%M:%S'): $STATUS" > $MK; say "END: $STATUS (marker $MK)"; fi; }
trap cleanup EXIT; trap 'STATUS="STOPPED (a signal while ${PHASE:-waiting})"; exit 1' INT TERM
say "armed: deadline $(date -r $DL '+%m-%d %H:%M'); gate $GATE; after '${AFTER:-none}'; EvalSuite end marker $ES/$ESMK (mtime >= $ESMIN, pid $ESPID gone); go file $GO; marker $MK"
PHASE=waiting
while :; do
  now=$(date +%s)
  [ $now -ge $DL ] && { STATUS="NOT RUN (the deadline $(date -r $DL '+%m-%d %H:%M') passed while waiting: ${LASTW:-?})"; exit 0; }
  [ -e $Q/HOLD ] && { STATUS="NOT RUN (HOLD)"; exit 0; }
  if ! { newer $ES/$ESMK $ESMT && ! kill -0 $ESPID 2>/dev/null; }; then w="EvalSuite's chain ($ESMK, pid $ESPID)"
  elif [ ! -e $GO ]; then w="Leap's slice test (no $GO)"
  elif big_lock; then w="another session's MEM >= 16 GB lock"
  elif t30c_up; then w="t30c"
  elif after_up; then w="a process matching '$AFTER'"
  elif [ -e $Q/PAUSE ] || lid_closed; then w="PAUSE or the lid closed"
  elif gate_ok; then break
  else w="NO-SWAP $GR GB < $GATE"
       [ $(( $(date +%s) - LASTG )) -ge 300 ] && { say "waiting: $w"; LASTG=$(date +%s); }
  fi
  [ "$LASTW" = "$w" ] || { case "$w" in NO-SWAP*) ;; *) say "waiting: $w";; esac; LASTW="$w"; }
  sleep $POLL
done
[ $(date +%s) -ge $DL ] && { STATUS="NOT RUN (the deadline $(date -r $DL '+%m-%d %H:%M') passed during the gate reads)"; exit 0; }
say "start: EvalSuite's chain ended, Leap's slice test passed, NO-SWAP $GR GB >= $GATE; the key test's own gate next"
PHASE="the key test"
HARDT=$(date -r $(( DL + 60 * HARDMIN )) '+%Y-%m-%d %H:%M'); SBY=$(date -r $DL '+%Y-%m-%d %H:%M')
if [ -n "$STUB" ]; then ( exec .venv/bin/python $STUB ) >> $L 2>&1 & else
  ( exec .venv/bin/python scripts/g1_keytest.py --snap g1a --gate $GATE --away-min 0 --start-by "$SBY" --hard "$HARDT" $KT_EXTRA ) >> $L 2>&1 &
fi
CP=$!; wait $CP; rc=$?; CP=""
R=${SIMD:-data/r57}/g1_keytest_g1a.json
v=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('envs',{}); print(d.get('status','?'), '|', '; '.join(f\"{k}: {v.get('verdict')} peak {v.get('peak_gb')} GB margin {v.get('at_peak_margin_gb')} GB swap {v.get('swap_gb')} swapouts {v.get('swapouts')}\" + (f\" fails {v.get('fails')}\" if v.get('fails') else '') for k, v in e.items()))" $R 2>/dev/null | cut -c1-500)
case "$v" in PASS*) STATUS="PASS (rc $rc; $v)";; STOPPED*) STATUS="STOPPED (rc $rc; $v)";; "") STATUS="FAIL (rc $rc; no result file)";; *) STATUS="FAIL (rc $rc; $v)";; esac
