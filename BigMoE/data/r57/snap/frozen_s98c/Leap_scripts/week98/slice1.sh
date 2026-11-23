#!/bin/bash
# slice1.sh (Leap, Mon 16 Nov 2026 13:1x; CHECK25): f32one_test.py (the FAST stack's 32-row verify + int8d on a 4-layer
# slice, ~8-9 GB) in EvalSuite's pause for Sandbox98's SIB (NewIdeas 13:1x: EvalSuite stops after B1 ~13:1x until sib_END
# or 15:30). Waits, holding nothing, for NO-SWAP >= GATE (default 13 = the slice's ~9 GB peak + 4) on two reads 30 s
# apart; no start after LATEST (15:10); a share lock; swap guard +2 GB; HOLD / PAUSE / lid stop it; the run capped at
# 25 min; unlock on ANY end (TERM trap); marker data/week98/check25/SLICE1_DONE = "slice1 end <time>: rc N (why)".
cd ~/Leap || exit 9
D=data/week98/check25; LOG=$D/slice1.log; MK=$D/SLICE1_DONE; Q=$HOME/.local/share/quiet
GATE=${SLICE_GATE:-13}; LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 15:10:00" +%s)
rm -f $MK
say() { echo "$(date +%T) $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
held() { [ -e $Q/HOLD ] || [ -e $Q/PAUSE ] || lid_closed; }
LOCKED=0; BP=""
fin() { [ -n "$BP" ] && kill -TERM $BP 2>/dev/null; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1
        say "end rc $1 ($2)"; echo "slice1 end $(date +%T): rc $1 ($2)" > $MK; exit $1; }
trap 'fin 1 "stopped by a signal"' INT TERM
say "slice1 armed pid $$: gate NO-SWAP >= $GATE GB on two reads, no start after 15:10"
while :; do
  [ "$(date +%s)" -gt "$LATEST" ] && fin 3 "no room before 15:10"
  held && { sleep 30; continue; }
  a=$(noswap)
  if [ "$a" -ge "$GATE" ]; then sleep 30; b=$(noswap); say "gate reads $a / $b GB"; [ "$b" -ge "$GATE" ] && break; fi
  sleep 30
done
peer lock share 25m --cpu 300 --gpu 40 "Leap CHECK25 slice test f32one_test (4 layers, MEM 9 GB, bits / logits, no timing)" >> $LOG 2>&1 \
  || fin 4 "share lock failed"
LOCKED=1
S0=$(swap_mb); say "run (NO-SWAP $(noswap) GB, swap $S0 MB)"
~/BigMoE/.venv/bin/python scripts/week98/f32one_test.py >> $LOG 2>&1 &
BP=$!; T0=$(date +%s)
while kill -0 $BP 2>/dev/null; do
  [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 7 "swap +2 GB"; }
  [ $(( $(date +%s) - T0 )) -gt 1500 ] && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 8 "over 25 min"; }
  held && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 6 "HOLD / PAUSE / lid"; }
  sleep 5
done
wait $BP; RC=$?; BP=""
fin $RC "f32one_test done in $(( $(date +%s) - T0 )) s; swap $S0 -> $(swap_mb) MB"
