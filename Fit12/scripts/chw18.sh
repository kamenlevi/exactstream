#!/bin/sh
# Fit12 chw18 (Sun 8 Nov; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 SMOKE + HISTORY TEST'): ONE share
# lock (MEM 12 GB), three processes, in this order. It replaces chw17.sh (smoke) and chw16.sh (profile), stopped while they
# waited in their gates:
#  chw18s: the FROZEN harness /Users/kamen/Fit12/snap/chw1108/fit_t3.py, P2' (cf:0.9:2 + PR1 + tail) on ifeval/1220,
#          ifeval/3280 (tokens + lsha must = chw14c2, the same positions) and ifbench/10 (it must load and decode), CAP 128
#  chw18r: the same harness and env on ifbench/10 FIRST, then chat12's 12 IFEval in REVERSED order, CAP 128 (history test:
#          every prompt sees other pools / caches than in chw14c2 / chw18s; tokens + lsha must be equal)
#  chw16p: the live scripts with F12_GSTAT=1 on the 12 IFEval in order (chw16's pre-registered profile; same tag and folder)
# GATE: 6 CLEAR minutes (1 once $O/GO exists; held while $O/WAIT exists): no other quiet lock / ticket, no other lock with
# MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid, >= 20 GB free; then no quiet ticket for 5 min, the share lock, an
# idle GPU (else the lock goes back). Starts by DL (arg 1, default "2026-11-08 22:15") or not at all. Stops on swap +1 GB or
# 25 min. Reports: $O/chw18_eq.txt (vs chw14c2), $O/chw18_hist.txt (chw18r vs chw18s), chw16's chw16_eq.txt / chw16_prof.txt.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/chw18; O16=$F/data/scr/chw16
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; SN=$F/snap/chw1108
IK=ifeval/1220,ifeval/3280,ifeval/1561,ifeval/2853,ifeval/374,ifeval/3407,ifeval/2239,ifeval/2590,ifeval/2870,ifeval/2439,ifeval/1219,ifeval/1262
RK=ifbench/10,ifeval/1262,ifeval/1219,ifeval/2439,ifeval/2870,ifeval/2590,ifeval/2239,ifeval/3407,ifeval/374,ifeval/2853,ifeval/1561,ifeval/3280,ifeval/1220
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 22:15}" +%s) || exit 9
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
mkdir -p $O $O16
rm -f $O/DONE $O16/DONE
log "chw18 armed: 6 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(freegb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
    need=6; [ -e $O/GO ] && need=1
    [ $clear -ge $((need * 2)) ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "chw18: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR chw18 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    echo "rc 5 not started (inside chw18) $(date '+%a %H:%M:%S')" > $O16/DONE
    exit 5
  fi
  log "chw18: clear for $((clear / 2)) min ($a GB free): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "chw18: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 20m --cpu 200 --gpu 100 "Fit12 12 GB smoke + history test for EvalSuite's IFBench arm, then a 12-prompt profile (3 processes, CAP 128), MEM 12 GB, ~10 min" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "chw18: locked, but blocked / busy ($(freegb) GB free, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR chw18 start: snapshot smoke (chw18s) + history (chw18r) + chw16p profile, P2' + PR1 + tail, CAP 128, share lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "chw18 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $SN/*.py $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $SN/fit_t3.py; } > $O/MANIFEST.txt 2>&1
cp $O/MANIFEST.txt $O16/MANIFEST.txt
for J in "chw18s|$O|$SN/fit_t3.py|0|ifeval/1220,ifeval/3280,ifbench/10" "chw18r|$O|$SN/fit_t3.py|0|$RK" "chw16p|$O16|$F/scripts/fit_t3.py|1|$IK"; do
  T=$(echo "$J" | cut -d'|' -f1); D=$(echo "$J" | cut -d'|' -f2); S=$(echo "$J" | cut -d'|' -f3); G=$(echo "$J" | cut -d'|' -f4); KS=$(echo "$J" | cut -d'|' -f5)
  [ $rc -ne 0 ] && break
  log "chw18: $T ($S, GSTAT $G) start"
  GS=""; [ "$G" = 1 ] && GS="F12_GSTAT=1"        # the snapshot runs exactly chw17's env (no GSTAT variable at all)
  env F12_OUT=$D F12_CAP=128 F12_W1=cf:0.9:2 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 $GS F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
    F12_IDS_OUT=$D/${T}_ids.json F12_KEYS=$KS $PY -B $S $T > $D/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "chw18: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1500 ] && { rc=124; log "chw18: 25 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "chw18: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR chw18 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR chw18 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
echo "rc $rc (inside chw18) started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O16/DONE
grep -E 'Error|Traceback|AssertionError' $O/chw18s.txt $O/chw18r.txt $O16/chw16p.txt 2>/dev/null | head -3
for f in $F/data/scr/chw14/chw14c2*; do ln -sf $f $O/; ln -sf $f $O16/; done      # stage 2 P2' = the same env
$PY -I -B $F/scripts/chw13_read.py $O chw14c2 chw18s chw18r > $O/chw18_eq.txt 2>&1; cp $O/chw13_report.json $O/chw18_eq.json
$PY -I -B $F/scripts/chw13_read.py $O chw18s chw18r > $O/chw18_hist.txt 2>&1; cp $O/chw13_report.json $O/chw18_hist.json
grep -E '^\[chw18[sr]\] ifbench' $O/chw18s.txt $O/chw18r.txt | cut -c1-230 >> $O/chw18_hist.txt
$PY -I -B $F/scripts/chw13_read.py $O16 chw14c2 chw16p > $O16/chw16_eq.txt 2>&1
$PY -I -B $F/scripts/c1u16_read.py $O16/chw16p.txt chw16p ex /nonexistent none ex > $O16/chw16p_report.txt 2>&1
$PY -I -B $F/scripts/chw15_read.py $O16 chw16p > $O16/chw16_prof.txt 2>&1
cut -c1-230 $O/chw18_eq.txt $O/chw18_hist.txt $O16/chw16_eq.txt $O16/chw16_prof.txt
exit $rc
