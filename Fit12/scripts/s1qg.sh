#!/bin/sh
# Fit12 s1qg.sh (Tue 10 Nov 10:4x; EvalSuite 10:4x, route (i): Fit12 runs it, EvalSuite pre-registers (a new PREREG) and scores).
# The 12 GB CODE GATE on S1q (QOM0 at C_m 51 / C_l 32 with the GPU residency probe; PLAN_12GB_90_fit12.md 'S1q RESULT').
# - harness: the FROZEN /Users/kamen/Fit12/snap/s1q1110/fit_onecall.py (6fd6d2951273; = the S1q run's 914e1ea452de but for its
#   sys.path line) + t3_onecall.py 24c21d9902a7 (S1q's) + t2_twopass + fit_pr1, all read-only; env exactly
#   /Users/kamen/Fit12/data/gate/code12_s1q_env.txt's line: BIGMOE_MAXK=92 F12_T3ARMS=q0r F12_CAP=item F12_CMCL=51,32 F12_W1=4
#   F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1; F12_IDS_OUT per chunk ({key: {"ex": [[ids]]}} + _meta.json);
# - KEYS: fwd = /Users/kamen/Fit12/data/gate/s1qg_fwd_keys.txt (the 60 dec_humaneval keys, then the 100 dec_mbpp keys, in the
#   items files' order); rev = that list reversed. BOTH ORDERS (EvalSuite: no history-independence assumed for QOM0 + the
#   probe): fwd first, then rev; chunks of CH keys (default 20), one process each (qgfNN / qgrNN); keys already in an order's
#   chunk file are skipped (resume); a stopped key is in no file and runs again. CAP = each item's max_tokens (768 / 1,024).
# - s1qg_merge.py writes s1qg_{fwd,rev}_{tokens,runs,meta}.json.
# HOLD: while $O/WAIT exists (released by hand on EvalSuite's GO, after its PREREG).
# START of a chunk (polled every 30 s): no PAUSE / HOLD; NO-SWAP >= 20 GB; no quiet ticket waiting and no other MEM > 5 GB
#   lock (yield at chunk ends); before CSTART (Tue 14:40: every chunk must END by 14:50; the orchestrator: apps reopen at 15:00).
# LOCK per chunk: peer lock share, reason starting "MEM 12 GB"; released after every chunk.
# STOP (every 10 s while a chunk runs; ends the run, done keys kept): the process's phys_footprint_peak in EXACT bytes
#   (footprint -f bytes) > 12.0 GB; swap > the run's start + 0.5 GB; the clock at HSTOP (Tue 14:50).
F=/Users/kamen/Fit12; LOG=$F/logs/summary.txt
PY=/Users/kamen/BigMoE/.venv/bin/python; H=$F/snap/s1q1110/fit_onecall.py
O=$F/data/scr/s1qg; KF=$F/data/gate/s1qg_fwd_keys.txt; CH=${QG_CH:-20}
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD
CSTART=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-10 14:40}" +%s) || exit 9
HSTOP=$(date -j -f "%Y-%m-%d %H:%M" "${2:-2026-11-10 14:50}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
fpeak() { footprint -f bytes -p "$1" 2>/dev/null | awk '/phys_footprint_peak:/ {printf "%.3f", $2 / 1e9; exit}'; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
[ -e $O/GO ] || touch $O/WAIT
FWD=$(paste -sd, $KF); REV=$(tail -r $KF | paste -sd, -)
done_keys() { $PY -I -c 'import glob, json, sys
s = set()
for f in glob.glob(sys.argv[1] + "/" + sys.argv[2] + "*_ids.json"):
    if not f.endswith("_meta.json"):
        s |= set(json.load(open(f)))
print(",".join(sorted(s)))' $O $1; }
left() { d=",$(done_keys $1),"; echo "$2" | tr ',' '\n' | while read -r k; do case "$d" in *",$k,"*) ;; *) echo "$k";; esac; done | head -$CH | paste -sd, -; }
rc=0; first=1; SW0=$(swapmb)
run_chunk() {   # $1 = tag, $2 = keys; sets rc
  T=$1; K_=$2
  while :; do
    [ "$(date +%s)" -ge "$CSTART" ] && { log "past the chunk start-by $(date -r $CSTART '+%H:%M') before $T: the run stops (no verdict until it resumes)"; rc=5; return; }
    if [ ! -e $O/WAIT ] && [ ! -e "$PAUSEF" ] && [ ! -e "$HOLDF" ] && [ "$(nosw)" -ge 20 ] && [ "$(nquiet)" -eq 0 ] && [ "$(nbig)" -eq 0 ]; then break; fi
    sleep 30
  done
  peer lock share 30m --cpu 200 --gpu 100 "MEM 12 GB Fit12 S1q code gate (EvalSuite): $T" >/dev/null 2>&1
  SWc=$(swapmb); [ $first -eq 1 ] && SW0=$SWc
  log "$T: $(echo "$K_" | tr ',' '\n' | wc -l | tr -d ' ') keys from $(echo "$K_" | cut -d, -f1); NO-SWAP $(nosw) GB, swap $SWc MB (run start $SW0)"
  echo "$(date +%H:%M:%S) SCR s1qg $T start" >> $LOG
  env BIGMOE_MAXK=92 F12_T3ARMS=q0r F12_CAP=item F12_CMCL=51,32 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 \
    F12_OUT=$O F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$K_ $PY -B $H $T > $O/$T.txt 2>&1 &
  P=$!; first=0; why=""
  while kill -0 $P 2>/dev/null; do
    sleep 10
    pk=$(fpeak $P)
    if [ -n "$pk" ] && awk -v p="$pk" 'BEGIN {exit !(p > 12.0)}'; then why="footprint peak $pk GB (exact bytes) > 12.0"; rc=126
    elif [ $(( $(swapmb) - SW0 )) -gt 512 ]; then why="swap $(swapmb) MB > start $SW0 + 512"; rc=127
    elif [ "$(date +%s)" -ge "$HSTOP" ]; then why="the clock reached $(date -r $HSTOP '+%H:%M')"; rc=124; fi
    [ -n "$why" ] && { log "$T: STOP: $why"; pkill -TERM -P $P 2>/dev/null; kill -TERM $P; sleep 5; pkill -KILL -P $P 2>/dev/null; kill -KILL $P 2>/dev/null; break; }
    [ -n "$pk" ] && echo "$pk" > $O/.peak_$T
  done
  wait $P; w=$?
  peer unlock >/dev/null 2>&1
  log "$T end rc $w (footprint peak $(cat $O/.peak_$T 2>/dev/null || echo n/a) GB exact; harness: $(grep -o 'done; lifetime [0-9.]* GB' $O/$T.txt | tail -1))"
  echo "$(date +%H:%M:%S) SCR s1qg $T end (rc $w)" >> $LOG
  [ $rc -eq 0 ] && [ $w -ne 0 ] && { rc=$w; log "$T failed: stopping the run"; }
}
log "s1qg armed: fwd + rev x 160 keys, chunks of $CH; held while $O/WAIT exists; chunk start-by $(date -r $CSTART '+%a %H:%M'), hard stop $(date -r $HSTOP '+%H:%M')"
for ORD in f r; do
  [ $ORD = f ] && KS=$FWD || KS=$REV
  n=$(ls $O 2>/dev/null | grep -cE "^qg${ORD}[0-9]+_ids\.json$")
  while [ $rc -eq 0 ]; do
    K_=$(left qg$ORD "$KS"); [ -z "$K_" ] && { log "order $ORD: all 160 keys done"; break; }
    n=$((n + 1)); run_chunk "$(printf "qg${ORD}%02d" $n)" "$K_"
  done
  [ $rc -ne 0 ] && break
done
$PY -I -B $F/scripts/s1qg_merge.py $O $KF > $O/s1qg_merge.txt 2>&1; cat $O/s1qg_merge.txt
echo "rc $rc ended $(date '+%a %H:%M:%S')" > $O/DONE
echo "$(date +%H:%M:%S) SCR s1qg run end (rc $rc)" >> $LOG
exit $rc
