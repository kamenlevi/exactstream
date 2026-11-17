#!/bin/bash
# The G1 key test's arming wrapper (BigMoE, Mon 16 Nov; NewIdeas 12:3x-12:5x, EvalSuite PREREG_lossygate amendment 5a).
# It starts scripts/g1_keytest.py --snap $SNAP --variants $VARIANTS when ALL hold (SNAP default g1a, VARIANTS default g1x;
# Mon 17:0x: g1b with amendment 5d's eight variants g1x,g1rx,g1cx,g1crx,g1cdx,g1cdrx,g1lx,g1lrx after the base):
#   - EvalSuite's 1c-tiers chain has ended: $ES/$ESMK with mtime >= ESMIN AND its process gone: pid ESPID, or no process
#     whose command line matches the pgrep -f pattern ESPAT (one of the two is required). EvalSuite 13:1x: the gate is
#     T1C_RESF2_END (mtime >= 13:15) + tools/t1c_resume_follower2.sh gone (it writes the marker only at its very end, after
#     V1 -> pause 2 -> B2 -> V2, or on any stop); T1C_MON2_END / T1C_MON3_END can be written at the chain's PAUSES.
#     ESMK defaults to T1C_RESF2_END; ESMIN defaults to the arming time;
#   - Leap's model-level int8 slice test passed: $W/G1A_GO exists (BigMoE creates it on Leap's PASS message);
#   - no other session's lock of MEM >= 16 GB held or queued; no t30c drafter job; no process matching AFTER (optional);
#   - no HOLD / PAUSE, the lid open;
#   - NO-SWAP >= GATE on two reads 30 s apart (default 40: the orchestrator's peak - 2.6, as for t1c1).
# The at-Mac yield is lifted (the orchestrator 11:5x, relaying Kamen), so the away gate is off by default; the memory gate stays.
# The key test: --gate GATE --away-min 0 --start-by DEADLINE --hard DEADLINE + HARDMIN (default 8 min: clear of M4 at 23:05).
# EVERY end writes $W/G1A_KEYTEST_DONE: "G1A KEYTEST END <mm-dd HH:MM:SS>: <PASS|FAIL|STOPPED|NOT RUN> (<why>)" (Breakthrough's
# 4-arm map screen gates on it), unless $W/G1A.NOMARK exists.
# usage: ESPAT='t1c_resume_follower[2]\.sh' ESMIN="2026-11-16 13:15:00" DEADLINE="2026-11-16 22:55" bash scripts/g1_follow.sh
# Simulation only: SIMD (log / marker dir), STUB (a python file run in place of the key test), FAKE_ES (EvalSuite results dir),
# ESPID, ESMIN ("YYYY-mm-dd HH:MM:SS"), KT_EXTRA (extra key-test arguments, e.g. "--stub X --nolock --out D"), POLL (s).
# Mon 17:3x (NewIdeas' order 17:2x: this test BEFORE EvalSuite's B2, right after Breakthrough's map screen): ESGATE=0
# drops the EvalSuite chain gate; PREVMK / PREVMIN = another job's end marker (mtime >= PREVMIN) to wait for; KT_ABSENT
# = a pgrep -f pattern the key test checks before each env (it stops if one matches: EvalSuite's runner 65 waits in a
# memory loop that does not re-check the shared-lock yield, so it could launch between two envs).
# Mon 20:5x (the orchestrator 20:4x, relaying Kamen: lower-RAM tests while he uses the Mac; the key test waits for the night
# window, right after M4): AWAYMIN = the seconds since Kamen last touched the Mac (away_secs.sh, HIDIdleTime) needed to start
# (0 = off; 900 = the orchestrator's 15 min). Checked in the loop and again after the two NO-SWAP reads; the key test itself
# keeps --away-min 0 (its one-shot check would end STOPPED and write the marker that releases the jobs chained behind it).
# FAKE_AWAYF (simulation only): a file holding the away seconds.
# Tue 11-17 01:0x (the COMQOM probe re-run): MKNAME / NOMARKNAME / FOLLOW_LOG rename the end marker, the NOMARK file and
# the log (defaults G1A_KEYTEST_DONE / G1A.NOMARK / g1_follow.log); G1KT_SFX (exported) suffixes the key test's
# records and result; G1KT_XKEYS (exported) replaces the COMQOM envs' extra keys; MKLABEL replaces the marker's first
# words (default "G1A KEYTEST"; the probe: "G1P PROBE").
# Mon 21:0x (the orchestrator: if M4 does not run tonight, the key test runs on away + NO-SWAP alone, start by 01:30):
# PREVALT = another end file of the previous job, also mtime >= PREVMIN: M4's data/r57/m4_1116_AWAY_SKIP (M4 NOT RUN
# because Kamen stayed on the Mac to its latest start; it ends with NOMARK, so it writes no M4 marker).
set -u
cd ~/BigMoE || exit 1
DEADLINE=${DEADLINE:?YYYY-mm-dd HH:MM}; POLL=${POLL:-60}; STUB=${STUB:-}; GATE=${GATE:-40}; AFTER=${AFTER:-}; HARDMIN=${HARDMIN:-8}
SNAP=${SNAP:-g1a}; VARIANTS=${VARIANTS:-g1x}
W=${SIMD:-data/r57}; L=$W/${FOLLOW_LOG:-g1_follow.log}; MK=$W/${MKNAME:-G1A_KEYTEST_DONE}; NOMARK=$W/${NOMARKNAME:-G1A.NOMARK}; GO=$W/G1A_GO; Q=$HOME/.local/share/quiet
ES=${FAKE_ES:-$HOME/EvalSuite/results}; ESPID=${ESPID:-}; ESPAT=${ESPAT:-}; ESMK=${ESMK:-T1C_RESF2_END}; KT_EXTRA=${KT_EXTRA:-}
ESGATE=${ESGATE:-1}; PREVMK=${PREVMK:-}; PREVMIN=${PREVMIN:-}; KT_ABSENT=${KT_ABSENT:-}; AWAYMIN=${AWAYMIN:-0}; PREVALT=${PREVALT:-}
case "$AWAYMIN" in ""|*[!0-9]*) echo "AWAYMIN must be a whole number of seconds" >&2; exit 2;; esac
[ "$ESGATE" = 0 ] || [ -n "$ESPID$ESPAT" ] || { echo "ESPID (a pid) or ESPAT (a pgrep -f pattern) is required" >&2; exit 2; }
ESMIN=${ESMIN:-$(date '+%Y-%m-%d %H:%M:%S')}
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DEADLINE:00" +%s) || exit 2
ESMT=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ESMIN" +%s) || exit 2
PVT=0; if [ -n "$PREVMK" ]; then PVT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${PREVMIN:?PREVMIN is required with PREVMK}" +%s) || exit 2; fi
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
newer() { [ -e "$1" ] && [ "$(stat -f %m "$1")" -ge "$2" ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
others_locks() {  # other sessions' lock segments (memory peer-ls-lock-segments: their rows start "- ", mine is "You:";
                  # never grep -v a name: statuses that mention BigMoE would hide their locks). FAKE_PEERLS: a sim file
  { if [ -n "${FAKE_PEERLS:-}" ]; then cat "$FAKE_PEERLS"; else peer ls 2>/dev/null; fi; } | grep "^- " \
    | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"
}
big_lock() {      # another session's lock (held, shared or queued) with MEM >= 16 GB in its reason
  others_locks | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'
}
quiet_held() {    # another session's quiet lock held (a benchmark): a share run beside it would spoil its timing
  others_locks | grep -q "^· benchmarking"
}
t30c_up() { ps -axo args | grep -qE "^(/bin/)?bash [^ ]*sb98/t30c_run\.sh|^[^ ]*python[^ ]* (-B )?scripts/dd_ttt\.py (train|eval) .*t30c"; }
after_up() { [ -n "$AFTER" ] && pgrep -f "$AFTER" > /dev/null; }
away_ok() {      # Kamen away >= AWAYMIN s (no reading = at the Mac); AW = the last reading
  local a; if [ -n "${FAKE_AWAYF:-}" ]; then a=$(cat "$FAKE_AWAYF" 2>/dev/null); else a=$(bash $Q/away_secs.sh 2>/dev/null); fi
  AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge "$AWAYMIN" ]
}
es_gone() { if [ -n "$ESPAT" ]; then ! pgrep -f "$ESPAT" > /dev/null; else ! kill -0 $ESPID 2>/dev/null; fi; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
gate_ok() { local r1 r2; r1=$(noswap); sleep 30; r2=$(noswap); GR="$r1 / $r2"   # g1_keytest.py's noswap_gb (GB = 1e9)
            awk -v a=$r1 -v b=$r2 -v g=$GATE 'BEGIN {exit !(a >= g && b >= g)}'; }
STATUS="NOT RUN (armed $(date '+%m-%d %H:%M'))"; CP=""; LASTW=""; LASTG=0
cleanup() { if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then kill -TERM $CP 2>/dev/null
              for i in $(seq 1 90); do kill -0 $CP 2>/dev/null || break; sleep 1; done; fi
            if [ -e $NOMARK ]; then say "END without the marker (NOMARK): $STATUS"
            else echo "${MKLABEL:-G1A KEYTEST} END $(date '+%m-%d %H:%M:%S'): $STATUS" > $MK; say "END: $STATUS (marker $MK)"; fi; }
trap cleanup EXIT; trap 'STATUS="STOPPED (a signal while ${PHASE:-waiting})"; exit 1' INT TERM
say "armed: snap $SNAP, variants $VARIANTS${G1KT_XKEYS:+; extra keys $G1KT_XKEYS}${G1KT_SFX:+; record suffix $G1KT_SFX}; deadline $(date -r $DL '+%m-%d %H:%M'); gate $GATE; after '${AFTER:-none}'; EvalSuite end marker $ES/$ESMK (mtime >= $ESMIN, ${ESPAT:+no process matching '$ESPAT'}${ESPID:+pid $ESPID gone}); go file $GO; marker $MK; EvalSuite gate $ESGATE; previous job ${PREVMK:-none}${PREVALT:+ or $PREVALT}${PREVMK:+ (mtime >= $PREVMIN)}; key-test absent pattern '${KT_ABSENT:-none}'; away gate ${AWAYMIN} s"
PHASE=waiting
while :; do
  now=$(date +%s)
  [ $now -ge $DL ] && { STATUS="NOT RUN (the deadline $(date -r $DL '+%m-%d %H:%M') passed while waiting: ${LASTW:-?})"; exit 0; }
  [ -e $Q/HOLD ] && { STATUS="NOT RUN (HOLD)"; exit 0; }
  if [ "$ESGATE" != 0 ] && ! { newer $ES/$ESMK $ESMT && es_gone; }; then w="EvalSuite's chain ($ESMK${ESPAT:+, '$ESPAT'}${ESPID:+, pid $ESPID})"
  elif [ -n "$PREVMK" ] && ! newer "$PREVMK" $PVT && ! { [ -n "$PREVALT" ] && newer "$PREVALT" $PVT; }; then
       w="the previous job's end marker ($PREVMK${PREVALT:+ or $PREVALT}, mtime >= $PREVMIN)"
  elif [ ! -e $GO ]; then w="Leap's slice test (no $GO)"
  elif big_lock; then w="another session's MEM >= 16 GB lock"
  elif quiet_held; then w="another session's quiet lock (a benchmark)"
  elif t30c_up; then w="t30c"
  elif after_up; then w="a process matching '$AFTER'"
  elif [ -e $Q/PAUSE ] || lid_closed; then w="PAUSE or the lid closed"
> (a quote from the project owner, kept private)
  elif gate_ok; then
> (a quote from the project owner, kept private)
  else w="NO-SWAP $GR GB < $GATE"
       [ $(( $(date +%s) - LASTG )) -ge 300 ] && { say "waiting: $w"; LASTG=$(date +%s); }
  fi
  [ "$LASTW" = "$w" ] || { case "$w" in NO-SWAP*) ;; *) say "waiting: $w";; esac; LASTW="$w"; }
  sleep $POLL
done
[ $(date +%s) -ge $DL ] && { STATUS="NOT RUN (the deadline $(date -r $DL '+%m-%d %H:%M') passed during the gate reads)"; exit 0; }
say "start: the gates are open (EvalSuite gate $ESGATE${PREVMK:+, the previous job ended}, G1A_GO, no big lock / t30c / AFTER match), NO-SWAP $GR GB >= $GATE$( [ "$AWAYMIN" -gt 0 ] && echo "; away_secs $AW >= $AWAYMIN"); the key test's own gate next"
PHASE="the key test"
HARDT=$(date -r $(( DL + 60 * HARDMIN )) '+%Y-%m-%d %H:%M'); SBY=$(date -r $DL '+%Y-%m-%d %H:%M')
if [ -n "$STUB" ]; then ( exec .venv/bin/python $STUB ) >> $L 2>&1 & else
  ( export G1KT_ABSENT="$KT_ABSENT"; exec .venv/bin/python scripts/g1_keytest.py --snap $SNAP --variants $VARIANTS --gate $GATE --away-min 0 --start-by "$SBY" --hard "$HARDT" $KT_EXTRA ) >> $L 2>&1 &
fi
CP=$!; wait $CP; rc=$?; CP=""
R=${SIMD:-data/r57}/g1_keytest_${SNAP}${G1KT_SFX:-}.json
v=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('envs',{}); print(d.get('status','?'), '|', '; '.join(f\"{k}: {v.get('verdict')} peak {v.get('peak_gb')} GB margin {v.get('at_peak_margin_gb')} GB swap {v.get('swap_gb')} swapouts {v.get('swapouts')}\" + (f\" fails {v.get('fails')}\" if v.get('fails') else '') for k, v in e.items()))" $R 2>/dev/null | cut -c1-1600)
case "$v" in PASS*) STATUS="PASS (rc $rc; $v)";; STOPPED*) STATUS="STOPPED (rc $rc; $v)";; "") STATUS="FAIL (rc $rc; no result file)";; *) STATUS="FAIL (rc $rc; $v)";; esac
