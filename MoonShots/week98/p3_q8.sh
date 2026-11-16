#!/bin/sh
# WEEK98 Q8 + SCREENS (MoonShots; NewIdeas Mon 16 Nov 21:1x and 22:3x): on the 125B, M3's harness (p3_dd_tree.py, gate
# PASSed Mon 08:4x), in NewIdeas' priority order Q8 > H32 > H16, then Q8H32 (stacked) if time is left:
#   Q8    Q4 with every 4 -> 8 (t30b's trained Linears + the head's HC mixes at 8 bits g64, router 8; full head, top-10)
#   H32   the screen: FR-Spec top-32K + the session extras (screen_masks, as H8); fp32 weights
#   H16   the same at top-16K
#   Q8H32 Q8's weights with the 32K screen
# Each cut on both sets (heplus24 first, then mbppbank), then scores of none / Q4 / H8 / Q8 / H32 / H16 / Q8H32 vs the full
# head (BigMoE's tree_t30b dumps) to p3/q8_scores.{txt,json}. Slot: Wed 18 Nov 06:30 (or earlier in a free >= 12 GB
# window, NewIdeas' rule); done before the band-screen fallback (latest start 07:15): no dump starts after Q8_CUTOFF
# (HHMM, default 0710). Share lock MEM 12 GB taken by the session before launch; this script drops it at the end. One
# dump process at a time under the memory guard (14 GB cap, swap +1 GB kill), vm_stat gate first.
cd "$HOME/MoonShots/week98" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
mkdir -p p3
LOG=p3/q8_run.log
avail_gb() { vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("[.]","",f); gsub("[.]","",i); gsub("[.]","",s); printf "%d\n", (f+i+s)*16384/1e9}'; }
memgate() {
  n=0
  while [ "$(avail_gb)" -lt 20 ]; do
    [ $n -eq 0 ] && echo "[q8] waiting for memory: $(avail_gb) GB available < 20 at $(date '+%H:%M:%S')" >> $LOG
    n=$((n + 1)); [ $n -gt 60 ] && { echo "[q8] STOP: memory gate not met in 10 min" >> $LOG; return 4; }
    sleep 10
  done
}
late() { [ "$(date '+%H%M')" -gt "${Q8_CUTOFF:-0710}" ] && [ "$(date '+%H%M')" -lt 1200 ]; }
run() {   # $1 set, $2 cut
  memgate || return 4
  if late; then echo "[q8] cutoff ${Q8_CUTOFF:-0710}: $2 $1 not started ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  echo "[q8] $2 $1 start $(date '+%H:%M:%S'), available $(avail_gb) GB" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=p3/dump_$2_$1.log sh jcc_guard.sh "$PY" -I -B p3_dd_tree.py --set "$1" --cut "$2"
  rc=$?; echo "[q8] $2 $1 rc $rc at $(date '+%H:%M:%S')" >> $LOG; return $rc
}
echo "[q8] start $(date '+%a %d %b %H:%M:%S'), away $(sh $HOME/.local/share/quiet/away_secs.sh) s" >> $LOG
stop=0
for c in Q8 H32 H16 Q8H32; do
  for s in heplus24 mbppbank; do
    run $s $c; rc=$?
    [ $rc -eq 8 ] && { stop=1; break; }
    [ $rc -ne 0 ] && echo "[q8] $c $s FAILED rc $rc (continuing with the next cut)" >> $LOG && break
  done
  [ $stop -eq 1 ] && break
done
peer unlock
echo "[q8] dumps done, lock released $(date '+%H:%M:%S')" >> $LOG
"$PY" -I -B p3_score.py --cuts none,Q4,H8,Q8,H32,H16,Q8H32 --tag q8 >> $LOG 2>&1
echo "[q8] scored $(date '+%H:%M:%S') rc $?" >> $LOG
touch p3/Q8_DONE
