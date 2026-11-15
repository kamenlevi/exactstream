#!/bin/bash
# Leap, Fri 2026-11-06 23:4x. Replaces gap_night.sh (keyed on Sat 00:30). Kamen opened the window at 23:30; NewIdeas' QUEUE
# (7e74d6c): BigMoE's 200 STACK -> Leap's C 224 GPU-idle run -> Fit12 -> EvalSuite runner 49. Re-keyed on BigMoE's run:
#   queue Leap's quiet ticket once BigMoE holds its quiet lock (FIFO: Leap is next at its release), or, if BigMoE's turn45
#   waiter is gone (its night over or dropped), once NO-SWAP >= 42.3 GB.
# At the grant: C from NO-SWAP by today's measured C 224 stack peaks (38.6-39.3 GB footprint; the 7.45 GB R57_VTIER pool is
# purgeable, not in the footprint) + 3: 224 >= 42.3, 208 >= 40.2, 192 >= 38.1, 160 >= 33.9 (2.1 GB per 16 slots), else
# release and stop. Then:
#   1. bstackF: the full 16-row loop's GPU busy vs wall (+ FULL16 warm and DENSE-ONLY forwards), ioreg sampler on
#   2. togs: the SB cadence A/B in one process (ns:STEPB_SBN = 4 | 2, 10 passes)
# then unlock and write the read-outs to ~/Leap/data/stepb/gap_night.txt. Gives up at Saturday 12:00.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_night.txt"
TL="$HOME/BigMoE/data/turn45.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
N0=$(wc -l < "$TL" 2>/dev/null || echo 0)
say "gap_night2 waiter start (pid $$; NO-SWAP $(avail_gb) GB; BigMoE's turn45.log at line $N0)"
why=""
while :; do
  now=$(date +%Y%m%d%H%M)
  [ "$now" -ge 202611071200 ] && { say "Sat 12:00 reached: giving up"; exit 0; }
  if peer ls 2>/dev/null | grep -F 'Run a larger-than-RAM MoE' | grep -q 'benchmarking: quiet lock until'; then why="BigMoE holds its quiet lock"; break; fi
  if tail -n +$((N0 + 1)) "$TL" 2>/dev/null | grep -F 'BigMoE: 200 STACK' | grep -q 'available: run'; then why="BigMoE's 200 STACK run started"; break; fi
  if ! pgrep -f "scripts/turn45.s[h]" >/dev/null && ge "$(avail_gb)" 42.3; then why="no BigMoE waiter and NO-SWAP $(avail_gb) GB"; break; fi
  sleep 30
done
say "$why: queueing Leap's ticket"
PEER_QUEUE_TIMEOUT=10800 peer lock quiet --queue 20m "Leap: C 224 GPU-busy vs wall of the 16-row check + SB cadence A/B (NewIdeas' QUEUE: after BigMoE), ~15 min, MEM ~39 GB" >> "$OUT" 2>&1 \
  || { say "no quiet lock: nothing runs"; exit 1; }
say "lock granted"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
sleep 20
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9 (C 160's peak + 3): released, nothing runs"; peer unlock >> "$OUT" 2>&1; exit 2; }
say "NO-SWAP $A GB -> C $C"
S="$HOME/Leap/data/stepb/gpu_util_night.txt"; rm -f "$S"
./gpu_util_sampler.sh "$S" & SP=$!
HAVE_LOCK=1 STEPB_C=$C STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="8" STEPB_KREP=0 STEPB_TAGX=night STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
kill $SP 2>/dev/null
HAVE_LOCK=1 STEPB_C=$C STEPB_ARMS="togs" STEPB_PASSES=10 STEPB_AT="" STEPB_KREP=0 STEPB_TOGGLE="ns:STEPB_SBN=4|2" STEPB_TAGX=sbn STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_night_flat2.json 2>/dev/null | head -1)
[ -n "$J" ] && "$HOME/BigMoE/.venv/bin/python" -B gap_read.py "$J" "$S" >> "$OUT" 2>&1
T=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_togs_*_sbn_screen.txt 2>/dev/null | head -1)
[ -n "$T" ] && grep -E "TOGGLE" "$T" >> "$OUT"
say "gap_night2 end"
