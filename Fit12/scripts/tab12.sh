#!/bin/sh
# Fit12 (Sat 7 Nov 17:2x; NewIdeas 17:2x, QUEUE c8dfaac; the orchestrator 17:1x): the two 12 GB cells of Kamen's 45 / 24 / 12 GB
# x exact / lossy table (due Sun ~08:00), on BigMoE's S5 prompt set: 12 HumanEval (68, 16, 91, 6, 79, 19, 161, 17, 56, 141, 101,
# 63) + MBPP12 (16, 100, 146, 161, 183, 208, 224, 246, 268, 296, 305, 319), CAP 256 as S5. Each arm ALONE in its own process,
# back to back under one quiet lock, on 12GB-CLEAN's footing (C_m 47 / C_l 32, F12_RERUN=1, F12_PHLOG=1):
#   slot 3, 12E = arm ex, the verified-greedy loop (NOT bit-exact: 2-4-row verify calls, keep-1 replays; PR1 not installed);
#   slot 4, 12F = arm t95ck1u, the theta 0.95 FAST arm (P1-K1 KU 4, cf 0.9; t80ck1u / t90ck1u failed EvalSuite's code gate,
#   the map passes theta 0.95: 1.10 % code flips, 37.9 tok/s [P]).
# Each run also writes its full ids in EvalSuite's tools/r57rows.py form (F12_IDS_OUT: $O/tab12{e,f}_ids.json + _meta.json).
# Report per run: /Users/kamen/Fit12/scripts/c1u16_read.py (whole decode, full period, end to end; per set; lifetime peak);
# 12E's tokens vs ex16 (the 12 shared HumanEval keys), 12F's vs 12E.
# GATE: from now, 10 CLEAR minutes in a row (1 once $O/GO exists, which NewIdeas may touch at S5''s end; held while $O/WAIT
# exists): no other quiet lock / ticket, no other lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid, >= 16 GB free.
# Then a 30-min quiet lock (queue <= 60 min, else back to the gate) and an idle GPU. Starts by DL (arg 1, default
# "2026-11-08 00:50": it ends before CERT's 01:30) or not at all. Stops on swap +1 GB or 35 min. Writes $O/DONE at the end.
# Out: /Users/kamen/Fit12/data/scr/tab12. Run ATTACHED (Bash run_in_background). Usage: sh tab12.sh [DL]
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/tab12
H=humaneval/HumanEval; Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
KEYS=$H/68,$H/16,$H/91,$H/6,$H/79,$H/19,$H/161,$H/17,$H/56,$H/141,$H/101,$H/63,mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 00:50}" +%s) || exit 9
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
log "tab12 armed: 10 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(freegb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 16 ]; then clear=$((clear + 1)); else clear=0; fi
    need=10; [ -e $O/GO ] && need=1
    [ $clear -ge $((need * 2)) ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "tab12: no clear slot by the deadline (quiet $q, big $b, $a GB free, PAUSE $([ -e $PAUSEF ] && echo on || echo off)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR tab12 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "tab12: clear for $((clear / 2)) min ($a GB free): asking for the quiet lock"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 30m "Fit12 table 12 GB cells: 12E verified-greedy + 12F theta 0.95, S5 prompts (12 HE + MBPP12), MEM 12 GB, ~20 min (NewIdeas c8dfaac)" >/dev/null 2>&1; then
    log "tab12: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1                       # the previous holder's exit: two samples <= 10 % in a row, <= 2 min
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "tab12: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR tab12 start: fit_t3.py 12E (ex) then 12F (t95ck1u), each alone, S5 prompts (12 HE + MBPP12), cap 256, quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "tab12 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for TA in tab12e:ex tab12f:t95ck1u; do
  T=${TA%%:*}; A=${TA##*:}
  [ $rc -ne 0 ] && break
  log "tab12: $T (arm $A) start"
  env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=$A F12_CMCL=47,32 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$KEYS \
    $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "tab12: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 2100 ] && { rc=124; log "tab12: 35 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "tab12: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR tab12 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR tab12 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback' $O/tab12e.txt $O/tab12f.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/c1u16_read.py $O/tab12e.txt tab12e ex $F/data/scr/ex16 ex16 ex > $O/tab12e_report.txt 2>&1
$PY -I -B $F/scripts/c1u16_read.py $O/tab12f.txt tab12f t95ck1u $O tab12e ex > $O/tab12f_report.txt 2>&1
for T in tab12e tab12f; do echo "== $T"; grep -E "^(WHOLE|FULL|END TO END|  humaneval|  mbpp|Per check|Tokens|Lifetime|PAUSE|Harness)" $O/${T}_report.txt | cut -c1-250; done
exit $rc
