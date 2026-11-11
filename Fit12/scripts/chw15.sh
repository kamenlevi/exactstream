#!/bin/sh
# Fit12 T2 FIXED COST PROFILE (Sun 8 Nov 11:4x; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 FIXED COST
# PROFILE'): one process under one SHARE lock (MEM 12 GB): chw15p = P2' (W1 cf:0.9:2 + PR1 + tail) on chat12's 12 IFEval, CAP
# 128, with F12_GSTAT=1 (per exact call: stamps + the loader's clocks). GATE: 6 CLEAR minutes (1 once $O/GO exists; held while
# $O/WAIT exists): no other quiet lock / ticket, no other lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid,
# >= 20 GB free; then no quiet ticket for 5 min, the share lock, an idle GPU (else the lock goes back). Starts by DL (arg 1,
# default "2026-11-08 18:00") or not at all. Stops on swap +1 GB or 20 min. Reports: chw13_read.py (chw14c2 linked in as the
# reference) -> chw15_eq.txt; chw15_read.py -> chw15_prof.txt; c1u16_read.py.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/chw15
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; PRE=$F/PLAN_12GB_90_fit12.md
IK=ifeval/1220,ifeval/3280,ifeval/1561,ifeval/2853,ifeval/374,ifeval/3407,ifeval/2239,ifeval/2590,ifeval/2870,ifeval/2439,ifeval/1219,ifeval/1262
K=humaneval/HumanEval
HK=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 18:00}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "chw15 armed: 6 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    if [ -e "$PRE" ]; then
      q=$(nquiet); b=$(nbig); a=$(freegb)
      if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
      need=6; [ -e $O/GO ] && need=1
      [ $clear -ge $((need * 2)) ] && break
    fi
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "chw15: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR chw15 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "chw15: clear for $((clear / 2)) min ($a GB free): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "chw15: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 15m --cpu 200 --gpu 100 "Fit12 T2 fixed-cost profile at 12 GB (stamps only, 12 IFEval, stage-2 rule), MEM 12 GB, ~5 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "chw15: locked, but blocked / busy ($(freegb) GB free, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR chw15 start: T2 fixed-cost profile (P2' + PR1 + tail + F12_GSTAT on 12 IFEval CAP 128), share lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "chw15 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for J in "chw15p|cf:0.9:2|128|$IK"; do
  T=$(echo "$J" | cut -d'|' -f1); W=$(echo "$J" | cut -d'|' -f2); C=$(echo "$J" | cut -d'|' -f3); KS=$(echo "$J" | cut -d'|' -f4)
  [ $rc -ne 0 ] && break
  log "chw15: $T (W1 $W, CAP $C) start"
  env F12_OUT=$O F12_CAP=$C F12_W1=$W F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_GSTAT=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
    F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$KS $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "chw15: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1200 ] && { rc=124; log "chw15: 20 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "chw15: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR chw15 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR chw15 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/chw15p.txt 2>/dev/null | head -3
for f in $F/data/scr/chw14/chw14c2*; do ln -sf $f $O/; done      # stage 2 P2' = the same env (run-to-run check)
$PY -I -B $F/scripts/chw13_read.py $O chw14c2 chw15p > $O/chw15_eq.txt 2>&1
for T in chw15p; do $PY -I -B $F/scripts/c1u16_read.py $O/$T.txt $T ex /nonexistent none ex > $O/${T}_report.txt 2>&1; done
$PY -I -B $F/scripts/chw15_read.py $O chw15p > $O/chw15_prof.txt 2>&1; cut -c1-230 $O/chw15_eq.txt $O/chw15_prof.txt
exit $rc
