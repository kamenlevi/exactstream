#!/bin/sh
# Fit12 S1u (Mon 9 Nov 21:5x; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md '12 GB -> 90: WHY <= 4 ROWS', before
# any run). ONE share lock (MEM 12 GB), two processes on 16 HumanEval at CAP 256 (chw14ha's keys), PR1 + tail:
#  s1uh: arm u01b = the one-call check (fit_onecall.py + t3_onecall.py: one wide 16-row call per check, U = every row's ranks
#        0-1 + the bonus row's ranks 2-9; committed from that call alone). LOSSY: a speed screen.
#  s1ue: arm ex = A' (W1 4), the same-lock reference (its tokens + lsha must = chw14ha's 16 / 16).
# GATE (NewIdeas' board, QUEUE.md 'MON 21:07 TONIGHT, UPDATED'): held while $O/WAIT exists (NewIdeas' GO is relayed as $O/GO);
# not before Mon 23:00 (the Mac is Kamen's until then); Kamen idle (HIDIdleTime >= 600 s, no Python of his >= 2 GB in
# `peer top`); NO-SWAP (free + speculative + purgeable + file-backed) >= 20 GB (peak ~11.4 + 8); no quiet ticket for 5 min,
# no MEM > 5 GB lock, an idle GPU. Starts by DL (arg 1, default Tue 00:58) or not at all. Stops at Tue 01:15 (everything
# ends by then) or on swap +1 GB. Kamen's idleness gates the start only (NewIdeas' rule), not a running screen.
# Reports: $O/s1u_read.txt (s1u_read.py), $O/s1u_rows.txt (paper_rows.py), $O/s1u_rep.txt (s1ue vs chw14ha).
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1u
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
K=humaneval/HumanEval
HK=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
ep() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
NB=$(ep "2026-11-09 23:00") || exit 9; DL=$(ep "${1:-2026-11-10 00:58}") || exit 9; HS=$(ep "2026-11-10 01:15") || exit 9
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
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || [ "$(date +%s)" -lt "$NB" ] || act || ramp || lid_closed \
  || ! idle_ok || kchat; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
[ -e $O/GO ] || touch $O/WAIT
log "S1u armed: held while $O/WAIT exists; not before $(date -r $NB '+%a %H:%M'); start by $(date -r $DL '+%a %H:%M'), hard stop $(date -r $HS '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(nosw)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
    [ $clear -ge 2 ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "S1u: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR s1u NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "S1u: clear for 1 min (NO-SWAP $a GB, idle $(idle_s) s): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "S1u: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 20m --cpu 200 --gpu 100 "Fit12 12 GB S1u one-call check screen: 16 HumanEval x 2 arms, MEM 12 GB, ~12 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(nosw)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "S1u: locked, but blocked / busy (NO-SWAP $(nosw) GB, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR s1u start: one-call check u01b + A' ref, 16 HumanEval, share lock (NO-SWAP $(nosw) GB, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "s1u manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_onecall.py $F/scripts/t3_onecall.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_onecall.py; } > $O/MANIFEST.txt 2>&1
for J in "s1uh|u01b" "s1ue|ex"; do
  T=$(echo "$J" | cut -d'|' -f1); A=$(echo "$J" | cut -d'|' -f2)
  [ $rc -ne 0 ] && break
  [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1u: past the hard stop before $T: not started"; break; }
  log "S1u: $T (arm $A) start"
  env F12_OUT=$O F12_CAP=256 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=$A \
    F12_CMCL=47,32 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$HK $PY -B $F/scripts/fit_onecall.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "S1u: swap +1 GB: stopping"; }
    [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1u: hard stop $(date -r $HS '+%H:%M'): stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "S1u: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR s1u $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR s1u end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/s1uh.txt $O/s1ue.txt 2>/dev/null | head -3
for f in $F/data/scr/chw14/chw14ha*; do ln -sf $f $O/; done      # A' code from Sun (the reference's reference)
$PY -I -B $F/scripts/s1u_read.py $O s1uh:u01b s1ue:ex > $O/s1u_read.txt 2>&1
$PY -I -B $F/scripts/paper_rows.py $O/s1u_rows "$O/s1uh.txt|s1uh|u01b|S1u code" "$O/s1ue.txt|s1ue|ex|A' code (same lock)" > /dev/null 2>&1
$PY -I -B $F/scripts/chw13_read.py $O chw14ha s1ue /nonexistent.json > $O/s1u_rep.txt 2>&1
cut -c1-260 $O/s1u_read.txt $O/s1u_rows.txt $O/s1u_rep.txt
exit $rc
