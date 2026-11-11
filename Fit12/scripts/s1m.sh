#!/bin/sh
# Fit12 S1m (Tue 10 Nov 11:4x): the memory pre-check NewIdeas asked for before a lower-memory S1q arm (11:1x / 11:2x): S1q's
# config (arm q0r, BIGMOE_MAXK=92, C_m 51 / C_l 32, CAP = item 1,024) on mbpp/228 ALONE, the key that took the S1q gate over 12.0
# GB (12.027, exact). One process, F12_MEMLOG=1 (MLX active / peak per check and phase: t3_onecall.MEMLOG) plus an external
# sampler every 2 s (footprint -f bytes: phys_footprint and its peak) -> $O/s1m_fp.txt. A DIAGNOSTIC: no 12.0 stop; a safety
# stop at 13.0 GB, swap +1 GB, 6 min, or the clock at 14:50. GATE as s1q.sh: no quiet ticket for 5 min, no MEM > 5 GB lock,
# an idle GPU, NO-SWAP >= 20, no PAUSE / HOLD; starts by 14:40 or not at all.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1m
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD
DL=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-10 14:40" +%s); CLK=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-10 14:50" +%s)
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
fp() { footprint -f bytes -p "$1" 2>/dev/null | awk '/phys_footprint:/ {a = $2} /phys_footprint_peak:/ {printf "%.3f %.3f", a / 1e9, $2 / 1e9; exit}'; }
cd $F || exit 9
mkdir -p $O; rm -f $O/DONE
while :; do
  [ "$(date +%s)" -ge "$DL" ] && { log "S1m: no clear slot by 14:40: NOT STARTED"; echo "rc 5 not started" > $O/DONE; exit 5; }
  if [ "$(nquiet)" -eq 0 ] && [ "$(nbig)" -eq 0 ] && [ ! -e "$PAUSEF" ] && [ ! -e "$HOLDF" ] && [ "$(nosw)" -ge 20 ]; then
    peer wait --timeout 5m >/dev/null 2>&1 || { sleep 30; continue; }
    peer lock share 15m --cpu 200 --gpu 100 "MEM 12 GB Fit12 S1m: S1q memory pre-check on mbpp/228, ~2 min" >/dev/null 2>&1
    sleep 3; g=$(gpu); n_=0
    while [ "${g:-0}" -gt 10 ] && [ $n_ -lt 24 ]; do sleep 5; g=$(gpu); n_=$((n_ + 1)); done
    [ "${g:-0}" -le 10 ] && [ "$(nquiet)" -eq 0 ] && [ "$(nbig)" -eq 0 ] && break
    peer unlock >/dev/null 2>&1
  fi
  sleep 30
done
SW0=$(swapmb); T0=$(date +%s); rc=0
log "S1m: start (NO-SWAP $(nosw) GB, swap $SW0 MB, GPU ${g}%)"; echo "$(date +%H:%M:%S) SCR s1m start" >> $LOG
env BIGMOE_MAXK=92 F12_T3ARMS=q0r F12_CAP=item F12_CMCL=51,32 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 \
  F12_MEMLOG=1 F12_OUT=$O F12_IDS_OUT=$O/s1mh_ids.json F12_KEYS=mbpp/228 $PY -B $F/scripts/fit_onecall.py s1mh > $O/s1mh.txt 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  sleep 2
  v=$(fp $P); [ -n "$v" ] && echo "$(date +%H:%M:%S) $v" >> $O/s1m_fp.txt
  pk=$(echo "$v" | awk '{print $2}')
  if [ -n "$pk" ] && awk -v p="$pk" 'BEGIN {exit !(p > 13.0)}'; then rc=126; log "S1m: safety stop at $pk GB"
  elif [ $(( $(swapmb) - SW0 )) -gt 1024 ]; then rc=127; log "S1m: swap +1 GB: stopping"
  elif [ $(( $(date +%s) - T0 )) -ge 360 ] || [ "$(date +%s)" -ge "$CLK" ]; then rc=124; log "S1m: time stop"; fi
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
log "S1m: end rc $rc ($(( $(date +%s) - T0 )) s); footprint peak $(tail -1 $O/s1m_fp.txt 2>/dev/null | awk '{print $3}') GB exact; harness $(grep -o 'done; lifetime [0-9.]* GB' $O/s1mh.txt | tail -1)"
echo "$(date +%H:%M:%S) SCR s1m end (rc $rc)" >> $LOG; echo "rc $rc $(date '+%H:%M:%S')" > $O/DONE
exit $rc
