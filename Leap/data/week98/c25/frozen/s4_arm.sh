#!/bin/bash
# s4_arm.sh (Leap, Wed 18 Nov 2026 07:5x): SLOT #4 (Thu 19 Nov 08:00; NewIdeas 07:4x / 07:5x): Breakthrough's knob_v3 arms.
# (a) tgl16ef = tgl16e + the front fold, EXACT: F16nf vs F16ne (no-tier, bitwise expected) and F32c6xf vs F32c6xe (the 32-row
# stack, bitwise expected); xfront.COUNTS fold > 0, plain_* == 0. (b) w4a8u vs tgl16eu for tier 1 #2a's go / no-go: F16nw vs
# F16nx (no-tier: >= ~1 ms, deterministic) and F16gw vs F16gx (tier twin: expert GB F16gw <= F16gx). knob_v3's kernel files ==
# knob_v2's (md5) and its dispatch differs only by the tgl16ef branch, so w4a8u here == EvalSuite's pinned v2 bit for bit.
# Gates: 08:00 (after Breakthrough's rec1a_c25 07:30-07:50), no BREAK98_HOLD, away >= 900, NO-SWAP 28 -> 26 after 20 min, then
# the runner (no process over 16 GB, the quiet ticket); swap guard +2 GB; latest start 10:30, hard end 11:00. Marker C25_s4_DONE.
A0=$(date +%s); QM=/Users/kamen/MoonShots/week98/p3/Q8_DONE; BH=/Users/kamen/NewIdeas/research/week98/BREAK98_HOLD
E=/Users/kamen/Leap/data/week98/c25; LOG=$E/run_s4.log; MARK=$E/C25_s4_DONE; N2=$E/C25_n2_DONE
AWAY=/Users/kamen/.local/share/quiet/away_secs.sh
T800=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-19 08:00:00" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-19 10:30:00" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') [s4_arm] $*" >> $LOG; }
end3() { say "$1: end (no start)"; echo "c25 end $(date '+%m-%d %H:%M:%S'): rc 3 ($1)" > $MARK; exit 3; }
trap 'say "stopped by a signal before the run"; echo "c25 end $(date "+%m-%d %H:%M:%S"): rc 1 (stopped by a signal before the run)" > $MARK; exit 1' INT TERM
rm -f $MARK
say "armed (pid $$; s4): waits for Thu 08:00, then no BREAK98_HOLD, away >= 900, NO-SWAP 28 -> 26; latest start 10:30"
while :; do
  [ $(date +%s) -ge $T800 ] && { say "08:00"; break; }
  [ $(date +%s) -ge $LAST ] && end3 "no 08:00 before 10:30 (impossible)"
  sleep 30
done
while false && [ ! -e "$N2" ]; do [ $(date +%s) -ge $LAST ] && end3 "no C25_n2_DONE before 10:30"; sleep 30; done
while [ -e "$BH" ]; do [ $(date +%s) -ge $LAST ] && end3 "BREAK98_HOLD held to 10:30"; sleep 30; done
while :; do a=$($AWAY 2>/dev/null); a=${a%%.*}; [ -n "$a" ] && [ "$a" -ge 900 ] 2>/dev/null && break
> (a quote from the project owner, kept private)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
T0G=$(date +%s); G=""
while :; do a=$(noswap); sleep 20; b=$(noswap); lo=$(( a < b ? a : b ))
  if [ $lo -ge 28 ]; then G=28; break; fi
  if [ $(( $(date +%s) - T0G )) -ge 1200 ] && [ $lo -ge 26 ]; then G=26; break; fi
  [ $(date +%s) -ge $LAST ] && end3 "NO-SWAP below the gate (28, 26 after 20 min) to 10:30 (last reads $a / $b GB)"
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
export C25_TAG=s4 C25_C=160 C25_GATE=$G C25_ROUNDS=12 C25_WARM=2 C25_EST_MIN=30 C25_CAP_MIN=40
export C25_PROF=1 C25_PASSK=1 C25_KROUNDS=3 C25_KARMS=F16ne,F16nf,F16nx,F16nw
export C25_ARMS=F16n,F16ne,F16nf,F16nx,F16nw,F16g,F16gx,F16gw,F32s,F32c6x,F32c6xe,F32c6xf
export C25_XKNOB=knob_v3
export C25_WAIT_NOSWAP=$G C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
unset C25_AFTER C25_POST
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_s4.sh "$(date '+%Y-%m-%d %H:%M:%S')" "2026-11-19 11:00:00"
