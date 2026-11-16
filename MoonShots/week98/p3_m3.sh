#!/bin/sh
# WEEK98 PILLAR 3 M3 (MoonShots, Mon 16 Nov 2026; NewIdeas 08:4x GO, beside t30c; share lock MEM 12 GB): a one-window
# smoke of the most hooked cut (C8), the bitwise gate (--cut none on both sets vs BigMoE's dumps), then the cuts
# C1 C3 C8 Q4 H8 on both sets, then the scores. One dump process at a time, each under the memory guard (14 GB cap,
# swap +1 GB kill) and a vm_stat gate before it starts (available >= peak 12 + 8 GB shared).
cd "$HOME/MoonShots/week98" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
mkdir -p p3
LOG=p3/m3_run.log
avail_gb() { vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("[.]","",f); gsub("[.]","",i); gsub("[.]","",s); printf "%d\n", (f+i+s)*16384/1e9}'; }
memgate() {
  n=0
  while [ "$(avail_gb)" -lt 20 ]; do
    [ $n -eq 0 ] && echo "[m3] waiting for memory: $(avail_gb) GB available < 20 at $(date '+%H:%M:%S')" >> $LOG
    n=$((n + 1)); [ $n -gt 180 ] && { echo "[m3] STOP: memory gate not met in 30 min" >> $LOG; exit 4; }
    sleep 10
  done
}
run() {   # $1 set, $2 cut
  memgate
  echo "[m3] $2 $1 start $(date '+%H:%M:%S'), available $(avail_gb) GB" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=p3/dump_$2_$1.log sh jcc_guard.sh "$PY" -I -B p3_dd_tree.py --set "$1" --cut "$2"
  rc=$?; echo "[m3] $2 $1 rc $rc at $(date '+%H:%M:%S')" >> $LOG; return $rc
}
echo "[m3] start $(date '+%H:%M:%S')" >> $LOG
memgate
MAXGB=14 SWAPGB=1 LOG=p3/smoke_C8.log sh jcc_guard.sh "$PY" -I -B p3_dd_tree.py --set heplus24 --cut C8 --part 0/24
rc=$?; echo "[m3] smoke C8 heplus24 part 0/24 rc $rc at $(date '+%H:%M:%S')" >> $LOG
[ $rc -ne 0 ] && { echo "[m3] STOP: smoke failed" >> $LOG; exit 5; }
rm -f p3/tree_t30b_heplus24_C8.part0of24.npz
for s in heplus24 mbppbank; do
  run $s none || { echo "[m3] STOP: gate dump failed" >> $LOG; exit 6; }
  "$PY" -I -B p3_dd_tree.py --set $s --cut none --gate >> $LOG 2>&1
done
if grep -q "\[p3 gate\].*FAIL" $LOG; then echo "[m3] STOP: gate FAIL" >> $LOG; exit 7; fi
for c in C1 C3 C8 Q4 H8; do
  for s in heplus24 mbppbank; do run $s $c; done
done
echo "[m3] dumps done $(date '+%H:%M:%S')" >> $LOG
"$PY" -I -B p3_score.py >> $LOG 2>&1
echo "[m3] scored $(date '+%H:%M:%S') rc $?" >> $LOG
