#!/bin/bash
# check25_q4.sh (Leap, Mon 16 Nov 2026 13:1x; CHECK25): the expert one-call at 32 rows, sync-free posts (exp32_bench v2)
# (scripts/week98/qmm32n_bench.py both) in a <= 5-min QUIET lock queued now (NewIdeas 12:2x; the orchestrator 12:2x:
# "If it's granted while EvalSuite's gate (pid 60135) is between processes and the GPU is idle, measure. If no grant by
# 13:30, measure after EvalSuite's gate ends (~16:00) ... Don't time beside a running 45 GB process").
# Rules: no start after DEADLINE (13:30); after the grant wait 20 s, then skip if any process holds > 10 GB RSS (a 45 GB
# runner is running); swap +2 GB -> stop; HOLD / PAUSE -> stop; the bench is capped at 300 s; always unlock at the end.
# Marker: data/week98/check25/Q4_DONE (one line "q4 end <time>: rc N (why)") on ANY end.
cd ~/Leap || exit 9
D=data/week98/check25
LOG=$D/q4.log
MK=$D/Q4_DONE
mkdir -p $D
rm -f $MK
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 13:17:00" +%s)
say() { echo "$(date +%T) $*" >> $LOG; }
fin() { [ -n "$WP" ] && kill -TERM $WP 2>/dev/null; [ -n "$BP" ] && kill -TERM $BP 2>/dev/null; peer unlock >/dev/null 2>&1; say "end rc $1 ($2)"; echo "q4 end $(date +%T): rc $1 ($2)" > $MK; exit $1; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }
big_rss() { ps -axo rss=,pid=,comm= | awk '$1 > 10000000 {print $2" "$3" "int($1/1048576)"GB"}' | head -3; }
Q=$HOME/.local/share/quiet
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
held() { [ -e $Q/HOLD ] || [ -e $Q/PAUSE ] || lid_closed; }

WP=""; BP=""
trap 'fin 1 "stopped by a signal (ticket dropped)"' INT TERM
say "q4 armed pid $$: queue a quiet ticket (10 min), no start after 2026-11-16 13:17:00"
peer lock quiet --queue 10m "Leap CHECK25: expert one-call timing at 32 rows (<= 3 min, MEM 3 GB)" >> $LOG 2>&1 &
WP=$!
while kill -0 $WP 2>/dev/null; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then kill -TERM $WP 2>/dev/null; wait $WP 2>/dev/null; fin 3 "no grant within 10 min: ticket dropped"; fi
  sleep 5
done
wait $WP; RC=$?
say "lock call returned rc $RC"
[ "$RC" = "0" ] || fin 4 "lock call failed"
[ "$(date +%s)" -gt "$DEADLINE" ] && fin 3 "granted after the 10-min cap: not run"
sleep 20
B=$(big_rss)
[ -n "$B" ] && fin 5 "a > 10 GB process is running at the grant (+20 s): $B"
held && fin 6 "HOLD / PAUSE / lid closed"
S0=$(swap_mb)
say "grant clean: swap ${S0} MB; bench starts"
~/BigMoE/.venv/bin/python scripts/week98/exp32_bench.py >> $LOG 2>&1 &
BP=$!
T0=$(date +%s)
while kill -0 $BP 2>/dev/null; do
  S=$(swap_mb)
  if [ "$(echo "$S - $S0 > 2048" | bc)" = "1" ]; then kill -TERM $BP; wait $BP 2>/dev/null; fin 7 "swap +2 GB (${S0} -> ${S} MB): stopped"; fi
  if [ $(( $(date +%s) - T0 )) -gt 330 ]; then kill -TERM $BP; wait $BP 2>/dev/null; fin 8 "bench over 330 s: stopped"; fi
  held && { kill -TERM $BP; wait $BP 2>/dev/null; fin 6 "HOLD / PAUSE / lid during the bench"; }
  sleep 2
done
wait $BP; BRC=$?
fin $BRC "bench done in $(( $(date +%s) - T0 )) s; swap ${S0} -> $(swap_mb) MB"
