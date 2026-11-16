#!/bin/bash
# The 1c-tiers key test's arming wrapper (BigMoE, Mon 16 Nov; the orchestrator 09:1x). Two modes:
#   MODE=follower  Tuesday: after EvalSuite's night1117 chain (results/CHAIN_1117_END newer than arming) AND its tier-0a
#                  follower (results/T0A_1117_END newer than arming, or its pid gone); no start after DEADLINE (Tue 13:00).
#   MODE=today     after BigMoE's 127-node dry run (data/deepdraft/ceiling/N127_DRY_1117_DONE, mtime >= 09:05) and
#                  MoonShots' M3 (no p3_dd_tree.py process), with no other session's lock of MEM >= 16 GB held or queued.
# Both: Kamen away (away_secs >= 900), no HOLD / PAUSE, the lid open, no process matching AFTER (a pgrep -f pattern, e.g.
# another session's job; default none), and NO-SWAP >= GATE on two reads 30 s apart (default 42, the at-peak rule; the
# orchestrator 11:0x: 41 = peak - 2 once MTP Head's S12 has ended); then scripts/t1c_keytest.py --snap t1c1 --gate GATE
# (its own two reads) --away-min 900 (its share lock "MEM 45 GB", swap guard +2 GB, HOLD / lid stops).
# EVERY end writes data/r57/T1C1_KEYTEST_DONE: "T1C1 KEYTEST END <mm-dd HH:MM:SS>: <PASS|FAIL|STOPPED|NOT RUN> (<why>)"
# (EvalSuite's 1c-tiers gate arms only on PASS), unless data/r57/T1C1.NOMARK.
# usage: MODE=follower DEADLINE="2026-11-17 13:00" bash scripts/t1c_follow.sh   |   MODE=today DEADLINE="2026-11-16 21:00" ...
# Simulation only: SIMD (log / marker dir), STUB (a python file run in place of the key test), FAKE_ES (EvalSuite results dir),
# POLL (seconds, default 60).
set -u
cd ~/BigMoE || exit 1
MODE=${MODE:?follower or today}; DEADLINE=${DEADLINE:?YYYY-mm-dd HH:MM}; POLL=${POLL:-60}; STUB=${STUB:-}
GATE=${GATE:-42}; AFTER=${AFTER:-}; LASTG=0
W=${SIMD:-data/r57}; L=$W/t1c_follow.log; MK=$W/T1C1_KEYTEST_DONE; NOMARK=$W/T1C1.NOMARK; Q=$HOME/.local/share/quiet
ES=${FAKE_ES:-$HOME/EvalSuite/results}; T0APID=${T0APID:-45029}; DRYMK=${DRYMK:-data/deepdraft/ceiling/N127_DRY_1117_DONE}
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DEADLINE:00" +%s) || exit 2
ARMED=$(date +%s); DRYT=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 09:05:00" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [$MODE] $*" >> $L; }
newer() { [ -e "$1" ] && [ "$(stat -f %m "$1")" -ge "$2" ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
away() { bash $Q/away_secs.sh 2>/dev/null | tail -1; }
big_lock() {      # another session's lock (held or queued) with MEM >= 16 GB in its reason
  peer ls 2>/dev/null | grep -iE "quiet|shared|queued|lock" | grep -v "BigMoE" | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'
}
t30c_up() { ps -axo args | grep -qE "^(/bin/)?bash [^ ]*sb98/t30c_run\.sh|^[^ ]*python[^ ]* (-B )?scripts/dd_ttt\.py (train|eval) .*t30c"; }   # the drafter line first
after_up() { [ -n "$AFTER" ] && pgrep -f "$AFTER" > /dev/null; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
gate_ok() { local r1 r2; r1=$(noswap); sleep 30; r2=$(noswap); GR="$r1 / $r2"   # t1c_keytest.py's noswap_gb (GB = 1e9)
            awk -v a=$r1 -v b=$r2 -v g=$GATE 'BEGIN {exit !(a >= g && b >= g)}'; }
STATUS="NOT RUN (armed $(date -r $ARMED '+%m-%d %H:%M'))"; CP=""
cleanup() { if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then kill -TERM $CP 2>/dev/null
              for i in $(seq 1 90); do kill -0 $CP 2>/dev/null || break; sleep 1; done; fi
            if [ -e $NOMARK ]; then say "END without the marker (NOMARK): $STATUS"
            else echo "T1C1 KEYTEST END $(date '+%m-%d %H:%M:%S'): $STATUS" > $MK; say "END: $STATUS (marker $MK)"; fi; }
trap cleanup EXIT; trap 'STATUS="STOPPED (a signal while ${PHASE:-waiting})"; exit 1' INT TERM
say "armed: deadline $(date -r $DL '+%m-%d %H:%M'); gate $GATE; after '${AFTER:-none}'; marker $MK"
PHASE=waiting
while :; do
  now=$(date +%s)
  [ $now -ge $DL ] && { STATUS="NOT RUN (the deadline $(date -r $DL '+%m-%d %H:%M') passed while waiting)"; exit 0; }
  [ -e $Q/HOLD ] && { STATUS="NOT RUN (HOLD)"; exit 0; }
  if [ "$MODE" = follower ]; then
    pre=0; newer $ES/CHAIN_1117_END $ARMED && { newer $ES/T0A_1117_END $ARMED || ! kill -0 $T0APID 2>/dev/null; } && pre=1
  else
    pre=0; newer $DRYMK $DRYT && ! pgrep -f "p3_dd_tre[e].py" > /dev/null && ! big_lock && ! t30c_up && pre=1
  fi
  if [ $pre = 1 ] && after_up; then
    [ "${LASTW:-}" = "after" ] || { say "predecessors done; waiting for no process matching '$AFTER'"; LASTW=after; }
    pre=0
  fi
  if [ $pre = 1 ]; then
    a=$(away)
    if [ ! -e $Q/PAUSE ] && ! lid_closed && [ "${a:-0}" -ge 900 ]; then
      if gate_ok; then break; fi
      [ $(( $(date +%s) - LASTG )) -ge 300 ] && { say "NO-SWAP $GR GB < $GATE (away ${a:-?})"; LASTG=$(date +%s); }
      LASTW=gate
    else
      [ "${LASTW:-}" = "gates" ] || { say "predecessors done; waiting for away >= 900 / no PAUSE / the lid open (away ${a:-?})"; LASTW=gates; }
    fi
  fi
  sleep $POLL
done
say "start: predecessors done, away $(away) s, NO-SWAP $GR GB >= $GATE; the key test's own gate next"
PHASE="the key test"
HARDT=$(date -r $(( DL + 1800 )) '+%Y-%m-%d %H:%M'); SBY=$(date -r $DL '+%Y-%m-%d %H:%M')
if [ -n "$STUB" ]; then ( exec .venv/bin/python $STUB ) >> $L 2>&1 & else
  ( exec .venv/bin/python scripts/t1c_keytest.py --snap t1c1 --gate $GATE --away-min 900 --start-by "$SBY" --hard "$HARDT" ) >> $L 2>&1 &
fi
CP=$!; wait $CP; rc=$?; CP=""
R=${SIMD:-data/r57}/t1c_keytest_t1c1.json
v=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('envs',{}); print(d.get('status','?'), '|', '; '.join(f\"{k}: {v.get('verdict')} peak {v.get('peak_gb')} GB swap {v.get('swap_gb')}\" + (f\" fails {v.get('fails')}\" if v.get('fails') else '') for k, v in e.items()))" $R 2>/dev/null | cut -c1-400)
case "$v" in PASS*) STATUS="PASS (rc $rc; $v)";; STOPPED*) STATUS="STOPPED (rc $rc; $v)";; "") STATUS="FAIL (rc $rc; no result file)";; *) STATUS="FAIL (rc $rc; $v)";; esac
