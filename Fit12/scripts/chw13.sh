#!/bin/sh
# Fit12 T2 CHAT WIDTH stage 1b (Sun 8 Nov 10:0x; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 CHAT WIDTH
# stage 1b'). Three processes back to back under one SHARE lock (MEM 12 GB), AFTER pf13 (its DONE), arm ex + PR1 (F12_PR1:
# keep-1 commits replay the PLE layer on 2 rows) + the tail rule (F12_W1TAIL: no 1-row last chunk), chat12's 12 IFEval, CAP 128:
#  A' chw13a: W1 4 (the reference)   P' chw13p: W1 cf:0.9:2:1.01:3 (the policy)   Q' chw13q: W1 rand:2,3,4 (the probe)
# P' = A' and Q' = A' in tokens AND kept-row logits hash 12 / 12 = exact-preserving w.r.t. A'. Speed P' vs A' counts only
# with an idle GPU (others' locks logged each minute in $O/locks.txt).
# GATE: /Users/kamen/Fit12/data/scr/pf13/DONE exists, then 6 CLEAR minutes (1 once $O/GO exists; held while $O/WAIT exists):
# no other quiet lock / ticket, no other lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid, >= 20 GB free; then
# no quiet ticket for 5 min, a 25-min SHARE lock, an idle GPU (else the lock goes back). Starts by DL (arg 1, default
# "2026-11-08 21:30") or not at all. Stops on swap +1 GB or 25 min. Report: chw13_read.py -> $O/chw13_report.txt.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/chw13
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; PRE=$F/data/scr/pf13/DONE
IK=ifeval/1220,ifeval/3280,ifeval/1561,ifeval/2853,ifeval/374,ifeval/3407,ifeval/2239,ifeval/2590,ifeval/2870,ifeval/2439,ifeval/1219,ifeval/1262
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 21:30}" +%s) || exit 9
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
log "chw13 armed: after $PRE, 6 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
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
    log "chw13: no clear slot by the deadline (pf13 DONE $([ -e $PRE ] && echo yes || echo no)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR chw13 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "chw13: clear for $((clear / 2)) min ($a GB free): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "chw13: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 25m --cpu 200 --gpu 100 "Fit12 T2 chat width stage 1b at 12 GB (PR1 + widths 2-4, 12 IFEval CAP 128, 3 arms), MEM 12 GB, ~8 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "chw13: locked, but blocked / busy ($(freegb) GB free, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR chw13 start: T2 stage 1b (ex + PR1 + tail, 12 IFEval, CAP 128; W1 4 / cf policy / rand 2-4), share lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "chw13 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for J in "chw13a|4" "chw13p|cf:0.9:2:1.01:3" "chw13q|rand:2,3,4"; do
  T=$(echo "$J" | cut -d'|' -f1); W=$(echo "$J" | cut -d'|' -f2)
  [ $rc -ne 0 ] && break
  log "chw13: $T (W1 $W) start"
  env F12_OUT=$O F12_CAP=128 F12_W1=$W F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
    F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$IK $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "chw13: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1500 ] && { rc=124; log "chw13: 25 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "chw13: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR chw13 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR chw13 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/chw13a.txt $O/chw13p.txt $O/chw13q.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/chw13_read.py $O chw13a chw13p chw13q > $O/chw13_report.txt 2>&1
cut -c1-230 $O/chw13_report.txt
exit $rc
