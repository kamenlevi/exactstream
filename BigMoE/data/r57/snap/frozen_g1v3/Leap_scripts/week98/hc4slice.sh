#!/bin/bash
# hc4slice.sh (Leap, Mon 16 Nov 2026 13:5x; V3 for EvalSuite's G2): hc4_slice_test.py (Breakthrough's hc4 switch 210d95d +
# Leap's hc4i on a 4-layer slice, ~9 GB, ~1 min, bits / numbers, no timing) in EvalSuite's pause 2 (beside Sandbox98's
# 5 GB SIB step; never in V1's segment gaps): waits, holding nothing, for a NEW '[resf3] pause 2 ... began' line in
# EvalSuite's t1c_chain.log (HC4_CHAINLOG; HC4_NOPAUSE skips it), then NO-SWAP >= GATE (13) on two reads 30 s apart; no
# start after 17:30; a 10-min share lock (MEM 10 GB: SIB yields only to quiet tickets and >= 30 GB locks); swap guard
# +2 GB; HOLD / PAUSE / lid stop it; capped at 10 min; unlock on ANY end (TERM trap); HC4_DRY=1 ends after the gates.
# Marker data/week98/check25/HC4SLICE_DONE = "hc4slice end <time>: rc N (why)".
cd ~/Leap || exit 9
D=data/week98/check25; LOG=$D/hc4slice.log; MK=$D/HC4SLICE_DONE; Q=$HOME/.local/share/quiet
GATE=${SLICE_GATE:-13}; LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 17:30:00" +%s)
rm -f $MK
say() { echo "$(date +%T) $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
held() { [ -e $Q/HOLD ] || [ -e $Q/PAUSE ] || lid_closed; }
LOCKED=0; BP=""
fin() { [ -n "$BP" ] && kill -TERM $BP 2>/dev/null; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1
        say "end rc $1 ($2)"; echo "hc4slice end $(date +%T): rc $1 ($2)" > $MK; exit $1; }
trap 'fin 1 "stopped by a signal"' INT TERM
CL=${HC4_CHAINLOG:-$HOME/EvalSuite/results/t1c_chain.log}   # EvalSuite 13:22: pause 2 logs '[resf3] pause 2 ... began' here
p2() { local n; n=$(grep -c -E "\[resf3\] pause 2.*began" "$CL" 2>/dev/null); echo "${n:-0}"; }
P20=$(p2); [ -n "${HC4_NOPAUSE:-}" ] && P20=-1
say "hc4slice armed pid $$: waits for a NEW '[resf3] pause 2 ... began' line (now $P20), then NO-SWAP >= $GATE GB on two reads; no start after 17:30"
while [ "$(p2)" -le "$P20" ]; do
  [ "$(date +%s)" -gt "$LATEST" ] && fin 3 "no pause 2 before 17:30"
  sleep 20
done
say "pause 2 began: $(grep -E "\[resf3\] pause 2.*began" "$CL" | tail -1 | cut -c1-160)"
while :; do
  [ "$(date +%s)" -gt "$LATEST" ] && fin 3 "no room before 17:30"
  held && { sleep 30; continue; }
  a=$(noswap)
  if [ "$a" -ge "$GATE" ]; then sleep 30; b=$(noswap); say "gate reads $a / $b GB"; [ "$b" -ge "$GATE" ] && break; fi
  sleep 30
done
[ -n "${HC4_DRY:-}" ] && fin 0 "dry: the gates passed"
peer lock share 10m --cpu 300 --gpu 40 "Leap V3 slice test hc4_slice_test (4 layers, MEM 10 GB, bits / numbers, no timing)" >> $LOG 2>&1 \
  || fin 4 "share lock failed"
LOCKED=1
S0=$(swap_mb); say "run (NO-SWAP $(noswap) GB, swap $S0 MB)"
~/BigMoE/.venv/bin/python scripts/week98/hc4_slice_test.py >> $LOG 2>&1 &
BP=$!; T0=$(date +%s)
while kill -0 $BP 2>/dev/null; do
  [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 7 "swap +2 GB"; }
  [ $(( $(date +%s) - T0 )) -gt 600 ] && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 8 "over 10 min"; }
  held && { kill -TERM $BP; wait $BP 2>/dev/null; BP=""; fin 6 "HOLD / PAUSE / lid"; }
  sleep 5
done
wait $BP; RC=$?; BP=""
fin $RC "hc4_slice_test done in $(( $(date +%s) - T0 )) s; swap $S0 -> $(swap_mb) MB"
