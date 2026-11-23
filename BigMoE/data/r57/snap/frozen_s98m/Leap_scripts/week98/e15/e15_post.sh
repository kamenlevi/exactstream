#!/bin/bash
# e15_post.sh (Leap, Tue 17 Nov 2026 06:3x): E15 (NewIdeas' A8 kill test, adaptk_k8.txt; its 06:32 condition: INSIDE slot #1's
# process) as the last step of tue1_post.sh, in tue1's quiet window, after the c25 loop has exited. One run of e15_loop.py
# (E14's loop; e15_engine time mode: A16, A8, A16b, A8b, C32one on G32T's 3 windows; ~2-3 min, peak ~31 GB like E14).
# Gates (any fail -> skip, marker written): HOLD / PAUSE / lid / HEALTH ACT; away >= 900 (Kamen's rule) before and after the
# memory reads; >= 6 min left before tue1's cap (its log's "capped at"); no other process over 16 GB; NO-SWAP >= 39 on two
# reads 20 s apart; the head test. The run: a swap guard (+2 GB -> TERM), a cap of 8 min (TERM the tree). tue1's runner
# still holds the ticket and kills this tree at its hard end; the away guard TERMs the runner when Kamen is back.
# Then (Tue 06:4x) the SW pass: the ref2 v3 switch bit tests (BigMoE's (a) / (b) + on / off; E15_MODE=sw, REF2_BOF=1
# REF2_HC_B=1) on v3-PATCHED COPIES (data/week98/e15/v3tree; the live files untouched), with its own gates: >= 4 min left
# before the cap, away >= 900, NO-SWAP >= 39 twice, the live files still at the patch's base md5s, the copies at theirs,
# its own head test (the loop head with E15_V3TREE) + v3_import_check.py (the block past EM.load); cap 6 min; output e15_sw_run.txt. Skipped if the time pass was skipped.
# On ANY end: ~/Leap/data/week98/e15/E15_DONE = "E15 end <time>: rc N (why); sw: <status>". Log: data/week98/e15/run_e15.log.
# DRY=1: AWAY_CMD / NOSWAP_CMD / CAPLINE stubs, 1 s waits, E15_CMD (default: sleep 2) in place of the loop; marker E15_DRY_DONE.
E=/Users/kamen/Leap/data/week98/e15; H=/Users/kamen/Leap/scripts/week98/e15; Q=$HOME/.local/share/quiet
PY=/Users/kamen/BigMoE/.venv/bin/python; T1LOG=/Users/kamen/Leap/data/week98/c25/run_tue1.log
DRY=${DRY:-0}; MARK=$E/E15_DONE; LOG=$E/run_e15.log; ZZ=20; GATE=39; CAPS=480
[ $DRY = 1 ] && { MARK=$E/E15_DRY_DONE; LOG=$E/run_e15_dry.log; ZZ=1; }
AWAY_CMD=${AWAY_CMD:-$Q/away_secs.sh}
say() { echo "$(date '+%m-%d %H:%M:%S') [e15$([ $DRY = 1 ] && echo _dry)] $*" >> $LOG; echo "[e15] $*"; }
noswap() { if [ -n "${NOSWAP_CMD:-}" ]; then $NOSWAP_CMD; else vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; fi; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
away() { local a; a=$($AWAY_CMD 2>/dev/null); echo "${a%%.*}"; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
RC=9; WHY="started"; CP=""; GUARD=""; SWS="not reached"
V3=/Users/kamen/Leap/data/week98/e15/v3tree/scripts
fin() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
        [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
        echo "E15 end $(date '+%m-%d %H:%M:%S'): rc $RC ($WHY); sw: $SWS" > $MARK
        say "wrote $MARK (rc $RC, $WHY); NO-SWAP $(noswap) GB, swap $(swap) MB"; }
trap fin EXIT
trap 'WHY="stopped by a signal"; RC=1; exit 1' INT TERM
skip() { WHY="skipped: $1"; RC=3; exit 3; }
cd /Users/kamen/Leap || skip "no ~/Leap"
say "start (pid $$, DRY $DRY)"
[ -e $Q/HOLD ] && skip "HOLD"; [ -e $Q/PAUSE ] && skip "PAUSE"; lid_closed && skip "lid closed"
grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && skip "HEALTH ACT"
> (a quote from the project owner, kept private)
cl=${CAPLINE:-$(grep "the run is capped at" $T1LOG 2>/dev/null | tail -1 | grep -oE "capped at [0-9:]+" | awk '{print $3}')}
[ -n "$cl" ] || skip "no 'capped at' line in tue1's log"
cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $cl" +%s) || skip "bad cap time $cl"
left=$(( cap - $(date +%s) )); [ $left -ge 360 ] || skip "only $left s left before tue1's cap ($cl)"
big=$(ps -axo rss=,pid=,comm= | awk '$1 > 16*1024*1024 {print $2" "$3" "int($1/1048576)" GB"}' | head -3)
[ -z "$big" ] || [ $DRY = 1 ] || skip "another process over 16 GB: $big"
r1=$(noswap); sleep $ZZ; r2=$(noswap); say "gate $GATE: NO-SWAP reads $r1 / $r2 GB; $left s left before the cap ($cl); away $a s"
[ "$r1" -ge $GATE ] && [ "$r2" -ge $GATE ] || skip "NO-SWAP $r1 / $r2 below $GATE GB"
a=$(away); [ -n "$a" ] && [ "$a" -ge 900 ] || skip "away $a s < 900 after the memory reads"
if [ $DRY = 1 ] && [ -n "${E15_SKIP_HEAD:-}" ]; then say "head test skipped (dry)"; else
  ( source $H/e15_env.sh && $PY -I -B $H/e15_headtest.py ) >> $LOG 2>&1 || { WHY="the head test failed (log)"; RC=5; exit 5; }
  say "head test PASS"
fi
s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) [e15] SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
say "run (swap $s0 MB, NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then ( bash -c "${E15_CMD:-sleep 2}" ) >> $LOG 2>&1 & else
  ( source $H/e15_env.sh && exec $PY -I -B $H/e15_loop.py 160 256 ) > $E/e15_run.txt 2>&1 &
fi
CP=$!; t0=$(date +%s)
while kill -0 $CP 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -ge $CAPS ] && { say "CAP ${CAPS} s: stopping the run"; killtree $CP; sleep 5; WHY="cut at the ${CAPS} s cap"; RC=6; exit 6; }
  sleep $(( DRY == 1 ? 1 : 5 ))
done
wait $CP; RC=$?; CP=""
case $RC in 0) WHY="ran";; *) WHY="the run failed (e15_run.txt)";; esac
say "time pass: rc $RC ($WHY)"
sw_pass() {   # its own gates; never changes RC; sets SWS
  local left r1 r2 a lm vm t0 rc
  left=$(( cap - $(date +%s) )); [ $left -ge 240 ] || { SWS="skipped: only $left s left before the cap"; return; }
  a=$(away); [ -n "$a" ] && [ "$a" -ge 900 ] || { SWS="skipped: away $a s < 900"; return; }
  lm=$(cd /Users/kamen/Leap && md5 -q scripts/ref2.py scripts/week98/bofi.py scripts/week98/bof.py | cut -c1-8 | tr '\n' ' ')
  [ "$lm" = "${LIVE_MD5:-33021885 7d37a80d e5e29b6f }" ] || { SWS="skipped: the live files changed ($lm)"; return; }
  vm=$(md5 -q $V3/ref2.py $V3/week98/bofi.py $V3/week98/bof.py 2>/dev/null | cut -c1-8 | tr '\n' ' ')
  [ "$vm" = "${V3_MD5:-589b840c beebe5af edb1ed1e }" ] || { SWS="skipped: the v3 copies are not the patch's ($vm)"; return; }
  r1=$(noswap); sleep $ZZ; r2=$(noswap); say "sw gate: NO-SWAP $r1 / $r2 GB; $left s left; away $a s"
  [ "$r1" -ge $GATE ] && [ "$r2" -ge $GATE ] || { SWS="skipped: NO-SWAP $r1 / $r2 below $GATE GB"; return; }
  if [ $DRY = 1 ] && [ -n "${E15_SKIP_HEAD:-}" ]; then say "sw head test skipped (dry)"; else
    ( source $H/e15_env.sh && export E15_MODE=sw REF2_BOF=1 REF2_HC_B=1 E14_ROUNDS=1 R57_TAG=e15sw E15_V3TREE=$V3 && \
      $PY -I -B $H/e15_headtest.py ) >> $LOG 2>&1 || { SWS="the sw head test failed (log)"; return; }
    say "sw head test PASS"
    ( source $H/e15_env.sh && export E15_MODE=sw REF2_BOF=1 REF2_HC_B=1 E15_V3TREE=$V3 && \
      $PY -I -B $H/v3_import_check.py ) >> $LOG 2>&1 || { SWS="the v3 import check failed (log)"; return; }
    say "v3 import check PASS"
  fi
  a=$(away); [ -n "$a" ] && [ "$a" -ge 900 ] || { SWS="skipped: away $a s < 900 after the memory reads"; return; }
  say "sw run (v3 copies in $V3)"; SWS="started, then stopped with the step"
  if [ $DRY = 1 ]; then ( bash -c "${E15_SWCMD:-sleep 2}" ) >> $LOG 2>&1 & else
    ( source $H/e15_env.sh && export E15_MODE=sw REF2_BOF=1 REF2_HC_B=1 E14_ROUNDS=1 R57_TAG=e15sw E15_V3TREE=$V3 && \
      exec $PY -I -B $H/e15_loop.py 160 256 ) > $E/e15_sw_run.txt 2>&1 &
  fi
  CP=$!; t0=$(date +%s)
  while kill -0 $CP 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -ge 360 ] && { say "sw CAP 360 s: stopping it"; killtree $CP; sleep 5; CP=""; SWS="cut at the 360 s cap"; return; }
    sleep $(( DRY == 1 ? 1 : 5 ))
  done
  wait $CP; rc=$?; CP=""
  SWS="rc $rc ($([ $rc = 0 ] && echo "ran; verdict in e15_sw_run.txt" || echo "failed (e15_sw_run.txt)"): $(grep -o 'E15 SW VERDICT: [A-Z]*' $E/e15_sw_run.txt 2>/dev/null | tail -1))"
}
sw_pass
say "sw pass: $SWS"
exit $RC
