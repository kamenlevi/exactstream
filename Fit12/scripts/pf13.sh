#!/bin/sh
# Fit12 PREFILL BUG CHECK (Sun 8 Nov 08:4x; orchestrator 08:2x + Leap a521e91: MLX 0.32.2's sorted NAX gather_qmm leaves rows
# unwritten when one call has > 32,767 sorted rows and rows % 64 != 0; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
# 'PREFILL BUG CHECK'). Two processes under one QUIET lock, AFTER the CHAT WIDTH test (chw12's DONE), arm ex, CAP 32, BigMoE's
# PFX prompts (agent/edit10 1,565, agent/write13 4,730, agent/write11 7,932 tokens, ascending):
#  1) pf13f: the shipped grid (bt_b12 fast mode: PFR 'balanced' = one MoE chunk per prompt up to 16K) + the gather_qmm audit
#     + Leap's pad (F12_GQPAD: an exposed sorted call is padded to a multiple of 64): the re-measure (TTFT, tok/s, SSD, peak).
#  2) pf13k: F12_PFRGRID=keep F12_SMOE=3072: every MoE call <= 30,720 rows (no call can reach the bug): the bitwise reference.
# GATE: /Users/kamen/Fit12/data/scr/chw12/DONE exists, then 6 CLEAR minutes (1 once $O/GO exists; held while $O/WAIT exists):
# no other quiet lock / ticket, no other lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid, >= 16 GB free. A
# 15-min quiet lock (queue <= 60 min, else back to the gate), an idle GPU. Starts by DL (arg 1, default "2026-11-08 21:30")
# or not at all. Stops on swap +1 GB or 20 min. Report: pf13_read.py -> $O/pf13_report.txt. Out: /Users/kamen/Fit12/data/scr/pf13.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/pf13
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; PRE=$F/data/scr/chw12/DONE
PK=agent/edit10,agent/write13,agent/write11
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
log "pf13 armed: after $PRE, 6 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    if [ -e "$PRE" ]; then
      q=$(nquiet); b=$(nbig); a=$(freegb)
      if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 16 ]; then clear=$((clear + 1)); else clear=0; fi
      need=6; [ -e $O/GO ] && need=1
      [ $clear -ge $((need * 2)) ] && break
    fi
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "pf13: no clear slot by the deadline (chw12 DONE $([ -e $PRE ] && echo yes || echo no)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR pf13 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "pf13: chw12 done, clear for $((clear / 2)) min ($a GB free): asking for the quiet lock"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 30m "Fit12 12 GB prefill bug check (MLX sorted-gather audit + Leap's pad, 1.6K / 4.7K / 7.9K, 2 arms), MEM 12 GB, ~8 min (orchestrator 08:2x)" >/dev/null 2>&1; then
    log "pf13: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1                       # the previous holder's exit: two samples <= 10 % in a row, <= 2 min
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "pf13: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR pf13 start: prefill bug check (ex, PFX prompts, CAP 32; shipped grid + audit + pad, then keep grid s_moe 3072), quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "pf13 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py /Users/kamen/Breakthrough/v3p/bigmoe_v3f/tier.py; \
  sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for J in "pf13f|4096||1" "pf13k|3072|keep|0"; do
  T=$(echo "$J" | cut -d'|' -f1); S=$(echo "$J" | cut -d'|' -f2); G=$(echo "$J" | cut -d'|' -f3); PD=$(echo "$J" | cut -d'|' -f4)
  [ $rc -ne 0 ] && break
  log "pf13: $T (s_moe $S, grid ${G:-shipped}, pad $PD) start"
  env F12_OUT=$O F12_CAP=32 F12_SMOE=$S F12_PFRGRID=$G F12_GQAUDIT=1 F12_GQPAD=$PD F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
    F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$PK $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "pf13: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1200 ] && { rc=124; log "pf13: 20 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "pf13: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR pf13 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR pf13 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback' $O/pf13f.txt $O/pf13k.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/pf13_read.py $O pf13f pf13k > $O/pf13_report.txt 2>&1
cut -c1-220 $O/pf13_report.txt
exit $rc
