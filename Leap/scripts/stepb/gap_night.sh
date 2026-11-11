#!/bin/bash
# Leap, Fri 2026-11-06 21:3x (the orchestrator 21:1x slot; NewIdeas 21:2x queue rule): wait until Saturday >= 00:30 AND BigMoE
# (~/BigMoE) holds its quiet lock, then queue Leap's quiet ticket (first in line for BigMoE's release ~01:35), then run:
#   1. bstackF: the full 16-row check's GPU-busy vs wall at C 224 (+ the dense-only forwards), ioreg sampler on
#   2. togs: the SB cadence A/B in one process (ns:STEPB_SBN = 4 | 2, 10 passes)
# then unlock and write the read-outs to ~/Leap/data/stepb/gap_night.txt. Gives up at Saturday 06:00.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_night.txt"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
say "gap_night waiter start (pid $$)"
while :; do
  now=$(date +%Y%m%d%H%M)
  [ "$now" -ge 202611070600 ] && { say "06:00 reached without BigMoE's lock: giving up"; exit 0; }
  if [ "$now" -ge 202611070030 ] && peer ls 2>/dev/null | grep -F '~/BigMoE' | grep -q 'benchmarking: quiet lock until'; then
    break
  fi
  sleep 30
done
say "BigMoE holds its quiet lock: queueing Leap's ticket"
PEER_QUEUE_TIMEOUT=10800 peer lock quiet --queue 15m "Leap: C 224 GPU-busy vs wall of the 16-row check + SB cadence A/B (orchestrator's slot after BigMoE), ~6 min, MEM ~40 GB" >> "$OUT" 2>&1 \
  || { say "no quiet lock: nothing runs"; exit 1; }
say "lock granted"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
S="$HOME/Leap/data/stepb/gpu_util_night.txt"; rm -f "$S"
./gpu_util_sampler.sh "$S" & SP=$!
HAVE_LOCK=1 STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="8" STEPB_KREP=0 STEPB_TAGX=night STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
kill $SP 2>/dev/null
HAVE_LOCK=1 STEPB_ARMS="togs" STEPB_PASSES=10 STEPB_AT="" STEPB_KREP=0 STEPB_TOGGLE="ns:STEPB_SBN=4|2" STEPB_TAGX=sbn STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_night_flat2.json 2>/dev/null | head -1)
[ -n "$J" ] && "$HOME/BigMoE/.venv/bin/python" -B gap_read.py "$J" "$S" >> "$OUT" 2>&1
T=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_togs_*_sbn_screen.txt 2>/dev/null | head -1)
[ -n "$T" ] && grep -E "TOGGLE" "$T" >> "$OUT"
say "gap_night end"
