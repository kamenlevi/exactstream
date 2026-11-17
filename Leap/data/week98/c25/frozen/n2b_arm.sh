#!/bin/bash
# n2b_arm.sh (Leap, Wed 18 Nov 2026 00:2x; re-arm of n2_arm.sh, the orchestrator 00:2x: M4 ended NOT RUN 23:45, BigMoE re-arms
# it to wait for NO-SWAP >= 36 until 04:00; M4 FIRST: slot #2 waits for M4's end marker newer than this arm, or 03:00): SLOT #2 (Tue night 11-17/18; NewIdeas: after BigMoE's night (M4), ahead of G1,
# behind any BREAK98 test): the check-module A/B on the c25 engine (75748ff). 16-row arms for Saturday's TREE-16 path:
# F16 (base), F16g (G1's lambda 1.8e-2), F16gd (+ GDN16), F16gk / F16gu (+ Breakthrough's tgl16e / tgl16eu), F16gx (G1 +
# GDN16 + tgl16eu = the flags BigMoE would get). 32-row: F32s (required), F32c6i, F32c6g (+ gdn_fast), F32c6p (+ HC32P),
# F32c6x (the bitwise stack), F32c6k / F32c6u (+ tgl16e / tgl16eu), F32c6xu (the whole stack). 3 windows x 12 rounds,
# profile + pass K (first window), no post. Gates in order: ~/BigMoE/data/r57/NIGHT1117_BIGMOE_END newer than this arm
# (any end, NOT RUN included); no BREAK98_HOLD; Kamen away >= 900 s; then the runner's NO-SWAP >= 36 on two reads, no
# process over 16 GB, the quiet ticket. Latest start 05:50, hard end 06:30 (slot #3 from 07:15). The away guard starts
# with the runner (same pid after exec). End marker ~/Leap/data/week98/c25/C25_n2_DONE on ANY end (also no start).
A0=$(date +%s); MK=/Users/kamen/BigMoE/data/r57/NIGHT1117_BIGMOE_END; BH=/Users/kamen/NewIdeas/research/week98/BREAK98_HOLD
E=/Users/kamen/Leap/data/week98/c25; LOG=$E/run_n2.log; MARK=$E/C25_n2_DONE; AWAY=/Users/kamen/.local/share/quiet/away_secs.sh
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-18 05:50:00" +%s); T300=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-18 03:00:00" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [n2_arm] $*" >> $LOG; }
end3() { say "$1: end (no start)"; echo "c25 end $(date '+%m-%d %H:%M:%S'): rc 3 ($1)" > $MARK; exit 3; }
trap 'say "stopped by a signal before the run"; echo "c25 end $(date "+%m-%d %H:%M:%S"): rc 1 (stopped by a signal before the run)" > $MARK; exit 1' INT TERM
rm -f $MARK
say "armed (pid $$; n2b): waits for $MK newer than $(date -r $A0 '+%m-%d %H:%M'), then no BREAK98_HOLD, then away >= 900; latest start 05:50"
while :; do
  if [ -e "$MK" ] && [ $(stat -f %m "$MK") -ge $A0 ]; then say "BigMoE marker: $(head -c 200 "$MK" | tr '\n' ' ')"; break; fi
  if [ $(date +%s) -ge $T300 ]; then say "03:00 and no new M4 end marker: slot #2 goes now (M4 first only to 03:00)"; break; fi
  [ $(date +%s) -ge $LAST ] && end3 "no BigMoE night marker before 05:50"
  sleep 30
done
while [ -e "$BH" ]; do [ $(date +%s) -ge $LAST ] && end3 "BREAK98_HOLD held to 05:50"; sleep 30; done
while :; do a=$($AWAY 2>/dev/null); a=${a%%.*}; [ -n "$a" ] && [ "$a" -ge 900 ] 2>/dev/null && break
> (a quote from the project owner, kept private)
trap - INT TERM
say "gates open (away $($AWAY) s): the runner + its away guard"
FROM_EPOCH=$(date +%s) /Users/kamen/Leap/data/week98/c25/guard/tue1_away_guard.sh $$ < /dev/null > /dev/null 2>&1 &
export C25_TAG=n2 C25_C=160 C25_GATE=36 C25_ROUNDS=12 C25_WARM=2 C25_EST_MIN=30 C25_CAP_MIN=40
export C25_PROF=1 C25_PASSK=1 C25_KROUNDS=3 C25_KARMS=F16,F16g,F16gu,F16gx,F32c6i,F32c6xu
export C25_ARMS=F16,F16g,F16gd,F16gk,F16gu,F16gx,F32s,F32c6i,F32c6g,F32c6p,F32c6x,F32c6k,F32c6u,F32c6xu
export C25_WAIT_NOSWAP=36 C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
unset C25_AFTER C25_POST
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_n2.sh "$(date '+%Y-%m-%d %H:%M:%S')" "2026-11-18 06:30:00"
