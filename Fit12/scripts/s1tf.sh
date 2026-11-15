#!/bin/sh
# Fit12 s1tf.sh (Sun 15 Nov 01:3x): runs s1tf.py (WEEK98 S1 DF-TF; its LINE is in its header) under ONE share lock (MEM 12 GB).
# Gate: no quiet ticket for 5 min, no other MEM > 5 GB lock, NO-SWAP (free + purgeable + file-backed: the orchestrator's corrected
# formula) >= 20 GB, no PAUSE / HOLD. Stops: footprint peak > 13.0 GB (exact bytes), swap +1 GB, 15 min.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1tf
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
fp() { footprint -f bytes -p "$1" 2>/dev/null | awk '/phys_footprint_peak:/ {printf "%.3f", $2 / 1e9; exit}'; }
cd $F || exit 9
mkdir -p $O; rm -f $O/DONE
TE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-15 05:30" +%s)   # start by Sun 05:30 (the result before 06:00)
TG=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-15 04:30" +%s)   # G32 fallback: after 04:30 the plain gate applies
# (Sun 02:4x) BigMoE posted 'G32 loaded' (the orchestrator's 01:56 condition); G32's MEM lock now holds this gate via nbig
while :; do
  [ "$(date +%s)" -ge "$TE" ] && { log "s1tf: no slot by 05:30: NOT STARTED"; echo "rc 5" > $O/DONE; exit 5; }
  if [ "$(nquiet)" -eq 0 ] && [ "$(nbig)" -eq 0 ] && [ ! -e "$PAUSEF" ] && [ ! -e "$HOLDF" ] && [ "$(nosw)" -ge 20 ]; then
    peer wait --timeout 5m >/dev/null 2>&1 && break
  fi
  sleep 30
done
peer lock share 20m --cpu 200 --gpu 100 "MEM 12 GB Fit12 WEEK98 S1 DF-TF (DFlash2 teacher-forced at block 32 on the bank), ~10 min" >/dev/null 2>&1
SW0=$(swapmb); T0=$(date +%s); rc=0
log "s1tf: start (NO-SWAP $(nosw) GB, swap $SW0 MB)"; echo "$(date +%H:%M:%S) SCR s1tf start" >> $LOG
# (Sun 02:4x, the orchestrator 02:35) a HEAD test first: the first 2 anchors through the exact command; the full screen
# runs only if its self-check (j = 1 through the reused cache == the native pass) holds
S1TF_HEAD=2 $PY -B $F/scripts/s1tf.py $O > $O/s1tf_head.txt 2>&1; hr=$?
log "s1tf: head test rc $hr: $(grep -o "HEAD 2: self-check[^;]*" $O/s1tf_head.txt | head -1)"
if [ $hr -ne 0 ]; then peer unlock >/dev/null 2>&1; echo "rc 8 head test failed (rc $hr) $(date '+%H:%M:%S')" > $O/DONE
  echo "$(date +%H:%M:%S) SCR s1tf head test FAILED (rc $hr)" >> $LOG; tail -5 $O/s1tf_head.txt; exit 8; fi
$PY -B $F/scripts/s1tf.py $O > $O/s1tf_run.txt 2>&1 &
P=$!; pk=""
while kill -0 $P 2>/dev/null; do
  sleep 5
  v=$(fp $P); [ -n "$v" ] && pk=$v
  if [ -n "$pk" ] && awk -v p="$pk" 'BEGIN {exit !(p > 13.0)}'; then rc=126; log "s1tf: footprint $pk GB > 13.0: stopping"
  elif [ $(( $(swapmb) - SW0 )) -gt 1024 ]; then rc=127; log "s1tf: swap +1 GB: stopping"
  elif [ $(( $(date +%s) - T0 )) -ge 900 ]; then rc=124; log "s1tf: 15 min: stopping"; fi
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
log "s1tf: end rc $rc ($(( $(date +%s) - T0 )) s), footprint peak ${pk:-n/a} GB (exact), swap $(swapmb) MB"
echo "$(date +%H:%M:%S) SCR s1tf end (rc $rc)" >> $LOG; echo "rc $rc peak ${pk:-n/a} GB $(date '+%H:%M:%S')" > $O/DONE
tail -6 $O/s1tf_run.txt
exit $rc
