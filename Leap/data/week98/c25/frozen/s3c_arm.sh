#!/bin/bash
# s3c_arm.sh (Leap, Wed 18 Nov 2026 04:2x; the orchestrator 04:2x: run slot #3 NOW (GPU idle since 02:13, Kamen asleep):
# no Q8 / 07:15 wait; memory gate 28 -> 26 after 20 min + a swap guard (+2 GB -> TERM), as slot #2's n2e). Was s3b_arm.sh
# (15:3x; re-arm of s3_arm.sh 14:49 with F16ne, the orchestrator 15:3x): SLOT #3 (Wed 18 Nov morning; NewIdeas: after MoonShots' cuts): the repeat of the
# deciding arms + g1c's other sets. 16-row: F16 (g1c HE+ code rows: 1c tiers lambda 0.016, s 0.98), F16g (== F16, drift
# check), F16gu (+ tgl16eu: THE deciding number for the orchestrator's Wed-evening call), F16gx (+ GDN16 + tgl16eu = the
# HE+ flags), F16ne (F16n + GDN16 + tgl16e EXACT; F16nx - F16ne = tgl16eu on the base path), F16n (tiers OFF = lossy-45's base path: g1c on MBPP / HumanEval-X / IFEval), F16nx (+ GDN16 + tgl16eu). 32-row:
# F32s (required), F32c6i, F32c6x (the bitwise stack), F32c6u, F32c6xu. 3 windows x 12 rounds, profile + pass K. Gates in
# order: MoonShots' ~/MoonShots/week98/p3/Q8_DONE newer than this arm, or 07:15 with no p3_q8 process; slot #2's
# C25_n2_DONE (any end); no BREAK98_HOLD; Kamen away >= 900 s; then the runner's NO-SWAP >= 36, no process over 16 GB, the
# quiet ticket. Latest start 11:00, hard end 11:30. Away guard with the runner. Marker C25_s3_DONE on ANY end.
A0=$(date +%s); QM=/Users/kamen/MoonShots/week98/p3/Q8_DONE; BH=/Users/kamen/NewIdeas/research/week98/BREAK98_HOLD
E=/Users/kamen/Leap/data/week98/c25; LOG=$E/run_s3.log; MARK=$E/C25_s3_DONE; N2=$E/C25_n2_DONE
AWAY=/Users/kamen/.local/share/quiet/away_secs.sh
T715=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-18 07:15:00" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-18 11:00:00" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [s3_arm] $*" >> $LOG; }
end3() { say "$1: end (no start)"; echo "c25 end $(date '+%m-%d %H:%M:%S'): rc 3 ($1)" > $MARK; exit 3; }
trap 'say "stopped by a signal before the run"; echo "c25 end $(date "+%m-%d %H:%M:%S"): rc 1 (stopped by a signal before the run)" > $MARK; exit 1' INT TERM
rm -f $MARK
say "armed (pid $$; s3c, run now): waits for Q8_DONE newer than $(date -r $A0 '+%m-%d %H:%M') or 07:15 + no p3_q8; then C25_n2_DONE, no BREAK98_HOLD, away >= 900; latest start 11:00"
while :; do
  say "the orchestrator 04:2x: slot #3 runs now (no Q8 / 07:15 wait)"; break
  if [ $(date +%s) -ge $T715 ] && ! pgrep -f "p3_q[8]" > /dev/null; then say "07:15 and no p3_q8 process"; break; fi
  [ $(date +%s) -ge $LAST ] && end3 "MoonShots' Q8 gate never opened before 11:00"
  sleep 30
done
while [ ! -e "$N2" ]; do [ $(date +%s) -ge $LAST ] && end3 "no C25_n2_DONE before 11:00"; sleep 30; done
say "slot #2 marker: $(head -c 160 "$N2")"
while [ -e "$BH" ]; do [ $(date +%s) -ge $LAST ] && end3 "BREAK98_HOLD held to 11:00"; sleep 30; done
while :; do a=$($AWAY 2>/dev/null); a=${a%%.*}; [ -n "$a" ] && [ "$a" -ge 900 ] 2>/dev/null && break
> (a quote from the project owner, kept private)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
T0G=$(date +%s); G=""
while :; do a=$(noswap); sleep 20; b=$(noswap); lo=$(( a < b ? a : b ))
  if [ $lo -ge 28 ]; then G=28; break; fi
  if [ $(( $(date +%s) - T0G )) -ge 1200 ] && [ $lo -ge 26 ]; then G=26; break; fi
  [ $(date +%s) -ge $LAST ] && end3 "NO-SWAP below the gate (28, 26 after 20 min) to 05:50 (last reads $a / $b GB)"
  sleep 10; done
say "memory gate $G GB passed (reads $a / $b GB)"
trap - INT TERM
say "gates open (away $($AWAY) s): the runner + its away guard"
FROM_EPOCH=$(date +%s) /Users/kamen/Leap/data/week98/c25/guard/tue1_away_guard.sh $$ < /dev/null > /dev/null 2>&1 &
S0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); S0=${S0%%.*}; RP=$$
say "swap guard: TERM the runner if swap > $S0 + 2048 MB"
( while kill -0 $RP 2>/dev/null; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); u=${u%%.*}
    if [ -n "$u" ] && [ "$u" -gt $((S0 + 2048)) ]; then say "SWAP GUARD: swap $u MB > $S0 + 2048: TERM runner $RP"; kill -TERM $RP; exit 0; fi
    sleep 10; done ) < /dev/null > /dev/null 2>&1 &
export C25_TAG=s3 C25_C=160 C25_GATE=$G C25_ROUNDS=12 C25_WARM=2 C25_EST_MIN=30 C25_CAP_MIN=40
export C25_PROF=1 C25_PASSK=1 C25_KROUNDS=3 C25_KARMS=F16,F16n,F16ne,F16nx,F32c6i,F32c6xu
export C25_ARMS=F16,F16g,F16gu,F16gx,F16n,F16ne,F16nx,F32s,F32c6i,F32c6x,F32c6u,F32c6xu
export C25_WAIT_NOSWAP=$G C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
unset C25_AFTER C25_POST
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_s3.sh "$(date '+%Y-%m-%d %H:%M:%S')" "2026-11-18 11:30:00"
