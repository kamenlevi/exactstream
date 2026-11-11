#!/bin/sh
# Fit12 S1p = the S1u PROFILE run (Mon 9 Nov 23:1x; derived from s1u.sh 45bdb608be31). NewIdeas' number check 22:55: S1u's
# 108.2 ms 'outside reads' is a fit split [P]. S1p measures it: ONE share lock (MEM 12 GB), ONE process on chw14ha's 16
# HumanEval at CAP 256, PR1 + tail, arm u01b (the one-call check, as S1u) with F12_GSTAT=1. t3_onecall.TS1 then logs, per wide
# call, the stamps, the loader's proc / io clocks and the disk bytes (<tag>_<key>_u01b_p1ts.json).
# The pre-registered line is unchanged: outside reads <= 90 ms. It is now read on the MEASURED loader-idle time (s1u_prof.py).
# GATE: held while $O/WAIT exists (NewIdeas' GO is relayed as $O/GO); NO-SWAP (free + speculative + purgeable + file-backed)
# >= 20 GB (peak ~11.4 + 8); no quiet ticket for 5 min, no MEM > 5 GB lock, an idle GPU; no PAUSE / HOLD / ACT / ramp / lid.
# Starts by DL (arg 1, default Tue 18:00) or not at all. Stops at 10 min from its start or on swap +1 GB.
# S1P_IDLE=1 (tonight, NewIdeas 23:05) adds: Kamen idle (HIDIdleTime >= 600 s) and no Python of his >= 2 GB in `peer top`.
# Reports: $O/s1p_prof.txt (s1u_prof.py), $O/s1p_read.txt (s1u_read.py), $O/s1p_rows.txt (paper_rows.py).
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1p
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
K=humaneval/HumanEval
HK=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
ep() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
DL=$(ep "${1:-2026-11-10 18:00}") || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/ {print int($NF / 1000000000); exit}'; }
kchat() { peer top 2>/dev/null | awk '/^Your user.s apps and tools/ {b = 1; next} b && /^[^ ]/ {b = 0}
  b && /[Pp]ython/ {for (i = 2; i <= NF; i++) if ($i == "GB" && $(i - 1) + 0 >= 2) f = 1} END {exit !f}'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
idle_ok() { i_=$(idle_s); [ "${i_:-0}" -ge 600 ]; }                 # unreadable -> not idle
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed \
  || { [ "${S1P_IDLE:-0}" = 1 ] && { ! idle_ok || kchat; }; }; }   # (S1p) S1P_IDLE=1: tonight's rule (Kamen idle, no chat of his)
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
[ -e $O/GO ] || touch $O/WAIT
log "S1p armed: held while $O/WAIT exists; start by $(date -r $DL '+%a %H:%M'); a 10-min hard stop from its start"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(nosw)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
    [ $clear -ge 2 ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "S1p: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR s1p NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "S1p: clear for 1 min (NO-SWAP $a GB, idle $(idle_s) s): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "S1p: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 20m --cpu 200 --gpu 100 "Fit12 12 GB S1p: the one-call check profiled, 16 HumanEval, MEM 12 GB, ~4 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(nosw)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "S1p: locked, but blocked / busy (NO-SWAP $(nosw) GB, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); HS=$((T0 + 600)); rc=0
echo "$(date +%H:%M:%S) SCR s1p start: one-call check u01b + A' ref, 16 HumanEval, share lock (NO-SWAP $(nosw) GB, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "s1u manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_onecall.py $F/scripts/t3_onecall.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_onecall.py; } > $O/MANIFEST.txt 2>&1
for J in "s1ph|u01b"; do
  T=$(echo "$J" | cut -d'|' -f1); A=$(echo "$J" | cut -d'|' -f2)
  [ $rc -ne 0 ] && break
  [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1p: past the hard stop before $T: not started"; break; }
  log "S1p: $T (arm $A) start"
  env F12_OUT=$O F12_CAP=256 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=$A \
    F12_CMCL=47,32 F12_GSTAT=1 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$HK $PY -B $F/scripts/fit_onecall.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "S1p: swap +1 GB: stopping"; }
    [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1p: hard stop $(date -r $HS '+%H:%M'): stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "S1p: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR s1p $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR s1p end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/s1uh.txt $O/s1ue.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/s1u_prof.py $O s1ph u01b > $O/s1p_prof.txt 2>&1
$PY -I -B $F/scripts/s1u_read.py $O s1ph:u01b > $O/s1p_read.txt 2>&1
$PY -I -B $F/scripts/paper_rows.py $O/s1p_rows "$O/s1ph.txt|s1ph|u01b|S1u code, profiled" > /dev/null 2>&1
cut -c1-300 $O/s1p_prof.txt $O/s1p_read.txt $O/s1p_rows.txt
exit $rc
