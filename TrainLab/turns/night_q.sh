#!/bin/sh
# MTP Head night queue (Sat 10-31, orchestrator 01:4x NIGHT MODE: big and long runs tonight, one quiet ticket at a time,
# FIFO, no gaps; stop queueing new tickets after 06:45). After D-20's queuer (turn47q) is done:
#   A. shipped-line A/B redo, adapt mode (turn53, A-B-A, the largest C that fits)      B. the same with a fixed 7-draft chain
#   C. chat captures (turn51: the rest of data/d22c + the eval streams)                  D. train b4 + its read-out (turn56)
#   E. b4's real-runtime verdict (turn57, 32 prompts)
# Each step re-queues until its output exists (or its ticket budget is spent). Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night_q: $*" >> $LOG; }
late() { [ "$(date +%H%M)" -ge 0645 ] && [ "$(date +%H%M)" -lt 1200 ]; }
ticket() {   # $1 = why, $2 = turn script, rest = env
  WHY=$1; JOB=$2; shift 2
  late && { say "past 06:45: no new ticket ($WHY)"; return 2; }
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head: $WHY" >> $LOG 2>&1 || { say "queue failed ($WHY)"; return 1; }
  env "$@" ./run_locked.sh 15 "$JOB"
}
while pgrep -f "turns/turn47q.sh" > /dev/null; do sleep 30; done
say "start (D-20 queuer done)"
for MODE in adapt k7; do
  for t in 1 2 3; do
    [ -s data/d22_ship_b3_${MODE}_aba.json ] && break
    ticket "D-22 shipped-line A/B $MODE (A-B-A, b3)" turns/turn53.sh D22_SHIP_MODE=$MODE || [ $? -ne 2 ] || exit 0
  done
done
for t in 1 2 3 4 5; do
  [ "$(ls data/d22c/*.npz 2>/dev/null | wc -l)" -ge 147 ] && [ "$(ls data/d16w/mgsm_*.npz 2>/dev/null | wc -l)" -ge 30 ] && break
  ticket "D-22 chat captures (training windows + eval streams)" turns/turn51.sh X=1 || [ $? -ne 2 ] || exit 0
done
for t in 1 2 3 4 5; do
  [ -s data/d22_eval_b4.txt ] && break
  ticket "D-22 training run b4 (code/math + all chat windows) + read-out" turns/turn56.sh X=1 || [ $? -ne 2 ] || exit 0
done
for t in 1 2 3 4; do
  [ -s data/d22r_b4_vs_stock.txt ] && [ "$(ls data/d22r_b4/*.npz 2>/dev/null | wc -l)" -ge 32 ] && break
  ticket "D-22 real-runtime verdict of b4 (32 prompts)" turns/turn57.sh X=1 || [ $? -ne 2 ] || exit 0
done
say "done"
