#!/bin/sh
# Fit12 (Thu 5 Nov 20:1x; 12 GB plan lever 1, /Users/kamen/Fit12/PLAN_12GB_90_fit12.md): the certificate calibration run.
# /Users/kamen/Fit12/scripts/fit_t3.py arm g1k1u (t80ck1u's pass 1 on EVERY check, then the exact check, so the outputs =
# ex) with F12_CAL=1 on T3R2's 16 HumanEval prompts (256 tokens each, C_m 47 / C_l 32 as T3R2): one dict per check with
# pass 1's per-row p1 / top-2 margin / argmax, the draft's cf and the exact tokens, for an offline replay of any certificate.
# The pinned bt_b12 (8db7, engine v3p) reads records3m (BIGMOE_M01=1), so Thu's records3 conversion does not touch it.
# One process, GPU, lifetime ~11.4 GB. Share lock (no timing is read). Starts only with no quiet ticket (peer wait), no
# PAUSE and >= 20 GB free (NO-SWAP: peak + 8 GB under a shared lock); stops on swap +1 GB or 30 min; starts by 22:00 or not at all. Run ATTACHED.
# Out: /Users/kamen/Fit12/data/scr/t3cal (c51.txt, c51_*_{g1k1u,g1k2}{,_cal,_ph,_toks}.json; g1k2 = the top-2 pass 1 (Breakthrough 19:4x)). Usage: sh t3cal.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/t3cal
K=humaneval/HumanEval; PAUSEF=/Users/kamen/.local/share/quiet/PAUSE
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
mkdir -p $O
peer wait --timeout 60m >/dev/null 2>&1 || { echo "t3cal: a quiet ticket for 60 min: not started"; exit 3; }
while { [ -e $PAUSEF ] || [ "$(freegb)" -lt 20 ]; } && [ "$(date +%H%M)" -lt 2155 ]; do sleep 30; done   # start by 22:00 (the orchestrator 20:2x), else Fri
[ -e $PAUSEF ] && { echo "t3cal: PAUSE at 21:55: not started (Fri by day, after EvalSuite runner 45)"; exit 4; }
[ "$(freegb)" -lt 20 ] && { echo "t3cal: only $(freegb) GB free at 21:55: not started (Fri by day, after EvalSuite runner 45)"; exit 5; }
peer lock share 35m --cpu 200 --gpu 100 "Fit12 12 GB lever-1 certificate calibration (g1k1u + g1k2, F12_CAL), 16 HumanEval prompts, MEM 12 GB, ~15 min" >/dev/null 2>&1
SW0=$(swapmb)
echo "$(date +%H:%M:%S) SCR t3cal start: fit_t3.py c51 arms g1k1u + g1k2 F12_CAL=1, 16 HumanEval prompts ($(freegb) GB free, swap $SW0 MB)" >> $LOG
env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_CAL=1 F12_K1U_KU=4 F12_T3ARMS=g1k1u,g1k2 F12_CMCL=47,32 \
  F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 \
  $PY -B $F/scripts/fit_t3.py c51 > $O/c51.txt 2>&1 &
P=$!; j=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5; j=$((j + 5))
  [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; echo "t3cal: swap +1 GB: stopping"; }
  [ $j -ge 1800 ] && { rc=124; echo "t3cal: 30 min: stopping"; }
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR t3cal end (rc $rc, $j s, swap $(swapmb) MB): $(ls $O/*_cal.json 2>/dev/null | wc -l | tr -d ' ') cal files" >> $LOG
grep -E 'g1k(1u|2) theta|Error|Traceback' $O/c51.txt | cut -c1-220
exit $rc
