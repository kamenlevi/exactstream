#!/bin/bash
# M-18 3-bit sure-row arm, queued behind MTP Head's E2 (orchestrator Mon 11:2x: E2 = DAY9 #8 goes first when memory frees).
# Run ATTACHED (Bash run_in_background): every 60 s, start only when (a) >= 26 GB is free (map peak ~17 GB + 8), (b) no
# other session holds or waits for a quiet window, and (c) E2 is not waiting for memory: its chain has ended, or one of its
# steps (capture / training) is running and has its memory. Then a share lock and run_day9.sh with the M-18 names, attached
# to this process. Gives up after T3_WAIT_H hours (default 10).
cd ~/FableGod/decision_map || exit 1
LOG=out/runner.log
NAMES="gate gate_top2t4_cf099 gate_top2t3_cf099 gate_top2t3_cf0999 hc8w_mix"
free_gb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
e2_alive() { pgrep -f "turns/e2_chain" >/dev/null; }
e2_step() { pgrep -f "turns/e2_cap.sh|turns/e2_train_x" >/dev/null; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|waiting for a quiet window"; }
END=$(( $(date +%s) + ${T3_WAIT_H:-10} * 3600 ))
echo "[dm-t3w] watching from $(date '+%a %H:%M') (free $(free_gb) GB; E2 chain alive: $(e2_alive && echo yes || echo no))" >> $LOG
while [ $(date +%s) -lt $END ]; do
  f=$(free_gb)
  if [ "$f" -ge 26 ] && ! quiet_others && { ! e2_alive || e2_step; }; then
    echo "[dm-t3w] go $(date '+%a %H:%M'): free $f GB; E2 chain alive $(e2_alive && echo yes || echo no), step running $(e2_step && echo yes || echo no)" >> $LOG
    peer lock share 60m --cpu 200 --gpu 50 "Breakthrough M-18 3-bit sure-row arm: decision map flips (~17 GB, no timing)" >> $LOG 2>&1
    DAY9_NAMES="$NAMES" bash run_day9.sh
    exit 0
  fi
  sleep 60
done
echo "[dm-t3w] gave up $(date '+%a %H:%M') (window ended)" >> $LOG
