#!/bin/sh
> (a quote from the project owner, kept private)
# more tan 15 min. and dont close spotify". The same screen as s1u.sh (pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
# '12 GB -> 90: WHY <= 4 ROWS'; lines unchanged) with s1u.sh's 23:00 / idle / quiet-queue / GPU-idle gates dropped on that word.
# It runs BESIDE Kamen's apps (Chrome, Spotify): timings may be noisy. Nothing of his is closed.
# Kept: HOLD blocks the start; NO-SWAP >= 16 GB at the start (peak ~11.4); swap guard +1 GB; one share lock (MEM 12 GB).
# HARD STOP at 15:00 wall clock from the start (heat pauses included): TERM, then KILL the process and its children, release
# the lock, read whatever prompts finished. Processes: s1uh (arm u01b, the one-call check), then s1ue (arm ex = A', the
# same-lock reference), 16 HumanEval at CAP 256 (chw14ha's keys). The s1u.done marker is NOT touched here.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1u
Q=/Users/kamen/.local/share/quiet; HOLDF=$Q/HOLD
K=humaneval/HumanEval
HK=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE $O/WAIT
[ -e "$HOLDF" ] && { log "HOLD exists: not started"; echo "rc 5 not started (HOLD) $(date '+%a %H:%M:%S')" > $O/DONE; exit 5; }
A0=$(nosw); [ "$A0" -ge 16 ] || { log "NO-SWAP $A0 GB < 16: not started"; echo "rc 5 not started (NO-SWAP $A0) $(date '+%a %H:%M:%S')" > $O/DONE; exit 5; }
> (a quote from the project owner, kept private)
SW0=$(swapmb); T0=$(date +%s); HS=$((T0 + 900)); rc=0
log "S1u NOW: started (NO-SWAP $A0 GB, swap $SW0 MB, GPU $(gpu)%; other locks: $(others | locks | tr '\n' ' ')); hard stop $(date -r $HS '+%H:%M:%S')"
echo "$(date +%H:%M:%S) SCR s1u NOW start (Kamen's go; beside his apps): u01b + A' ref, 16 HumanEval, share lock, hard stop $(date -r $HS '+%H:%M:%S')" >> $LOG
> (a quote from the project owner, kept private)
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py $F/scripts/s1u_now.sh; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_onecall.py; } > $O/MANIFEST.txt 2>&1
for J in "s1uh|u01b" "s1ue|ex"; do
  T=$(echo "$J" | cut -d'|' -f1); A=$(echo "$J" | cut -d'|' -f2)
  [ $rc -ne 0 ] && break
  [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1u: past the 15-min stop before $T: not started"; break; }
  log "S1u: $T (arm $A) start"
  env F12_OUT=$O F12_CAP=256 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=$A \
    F12_CMCL=47,32 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$HK $PY -B $F/scripts/fit_onecall.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% swap $(swapmb) MB | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "S1u: swap +1 GB: stopping"; }
    [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1u: 15-min hard stop: stopping"; }
    [ $rc -ne 0 ] && { pkill -TERM -P $P 2>/dev/null; kill -TERM $P; sleep 5; pkill -KILL -P $P 2>/dev/null; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "S1u: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR s1u $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR s1u NOW end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
> (a quote from the project owner, kept private)
grep -E 'Error|Traceback|AssertionError' $O/s1uh.txt $O/s1ue.txt 2>/dev/null | head -3
for f in $F/data/scr/chw14/chw14ha*; do ln -sf $f $O/; done
$PY -I -B $F/scripts/s1u_read.py $O s1uh:u01b s1ue:ex > $O/s1u_read.txt 2>&1
$PY -I -B $F/scripts/paper_rows.py $O/s1u_rows "$O/s1uh.txt|s1uh|u01b|S1u code" "$O/s1ue.txt|s1ue|ex|A' code (same lock)" > /dev/null 2>&1
$PY -I -B $F/scripts/chw13_read.py $O chw14ha s1ue /nonexistent.json > $O/s1u_rep.txt 2>&1
cut -c1-300 $O/s1u_read.txt $O/s1u_rows.txt $O/s1u_rep.txt
exit $rc
