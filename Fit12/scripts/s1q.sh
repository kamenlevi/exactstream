#!/bin/sh
# Fit12 S1q = QOM0 at C_m 51 / C_l 32 with the residency probe (Tue 10 Nov 10:3x; derived from s1p.sh bdd45f4ca3b1; the
# route after S1u closed: PLAN_12GB_90_fit12.md '12 GB -> 90: QOM0 needs C_m 51'). Pre-registered (before any run):
# ONE share lock (MEM 12 GB), ONE process on chw14ha's 16 HumanEval at CAP 256, PR1 + tail, arm q0r (t3_onecall.qom0_moe: one
# wide 16-row call per check; U + the drafted rows' resident ranks 2-9 in the same post; bonus row exact; committed from that
# call), BIGMOE_MAXK=92, F12_CMCL=51,32, F12_GSTAT=1 (the loader's clocks per wide call, as S1p).
# LINES: VALID if the probe's misreads are 0 (every resident it adds is resident for post_ids too) and the lifetime peak is
#   <= 12.0 GB; then the build may go to the gate (the map passes QOM0 at 51 slots: 1.16 % code, Breakthrough). Speed is
#   reported (whole decode, the measured split), not lined: QOM0 reads U's misses like S1u and computes more experts.
# GATE: held while $O/WAIT exists; NO-SWAP >= 20 GB (peak ~11.75 + 8); no quiet ticket for 5 min, no MEM > 5 GB lock, an idle
# GPU; no PAUSE / HOLD / ACT / ramp / lid. Starts by DL (arg 1, default Tue 18:00) or not at all. Stops at 10 min from its
# start or on swap +1 GB. Reports: $O/s1q_prof.txt, $O/s1q_read.txt, $O/s1q_rows.txt, $O/s1q_q0.txt (the QOM0 counters).
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/s1q
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
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
[ -e $O/GO ] || touch $O/WAIT
log "S1q armed: held while $O/WAIT exists; start by $(date -r $DL '+%a %H:%M'); a 10-min hard stop from its start"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(nosw)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
    [ $clear -ge 2 ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "S1q: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR s1q NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "S1q: clear for 1 min (NO-SWAP $a GB): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "S1q: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 20m --cpu 200 --gpu 100 "MEM 12 GB Fit12 S1q: QOM0 one-call check at C_m 51, 16 HumanEval, ~4 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(nosw)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "S1q: locked, but blocked / busy (NO-SWAP $(nosw) GB, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); HS=$((T0 + 600)); rc=0
echo "$(date +%H:%M:%S) SCR s1q start: QOM0 one-call check q0r at C_m 51, 16 HumanEval, share lock (NO-SWAP $(nosw) GB, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "s1q manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_onecall.py $F/scripts/t3_onecall.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py $F/scripts/s1q.sh; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_onecall.py; } > $O/MANIFEST.txt 2>&1
for J in "s1qh|q0r"; do
  T=$(echo "$J" | cut -d'|' -f1); A=$(echo "$J" | cut -d'|' -f2)
  [ $rc -ne 0 ] && break
  log "S1q: $T (arm $A) start"
  env BIGMOE_MAXK=92 F12_OUT=$O F12_CAP=256 F12_W1=4 F12_WL=4 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=$A \
    F12_CMCL=51,32 F12_GSTAT=1 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$HK $PY -B $F/scripts/fit_onecall.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "S1q: swap +1 GB: stopping"; }
    [ "$(date +%s)" -ge "$HS" ] && { rc=124; log "S1q: hard stop $(date -r $HS '+%H:%M'): stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "S1q: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR s1q $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR s1q end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/s1qh.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/s1u_prof.py $O s1qh q0r > $O/s1q_prof.txt 2>&1
$PY -I -B $F/scripts/s1u_read.py $O s1qh:q0r > $O/s1q_read.txt 2>&1
$PY -I -B $F/scripts/paper_rows.py $O/s1q_rows "$O/s1qh.txt|s1qh|q0r|QOM0 code at C_m 51, profiled" > /dev/null 2>&1
grep -o 'QOM0: [^|]*' $O/s1qh.txt > $O/s1q_q0.txt 2>/dev/null
cut -c1-300 $O/s1q_prof.txt $O/s1q_read.txt $O/s1q_rows.txt; head -20 $O/s1q_q0.txt
exit $rc
