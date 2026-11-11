#!/bin/sh
# Fit12 (Fri 6 Nov 09:5x; 12 GB plan lever 3, the orchestrator 09:3x): Leap's Friday kernel switches priced on the 12 GB T3
# path. None is read there as-is (no ref2 / d7b_nax / r57), so /Users/kamen/Fit12/scripts/f12_kports.py ports them (pinned
# kernels /Users/kamen/Fit12/pinned/leap_1106): exd = ex + the drafter on qmm16c with the per-shape ks (D7BNAX_32=2's
# <= 16-row rule; exact: tokens must = ex); t80ck2a / t80ck2n / t80ck2f = t80ck2 + pass 1's attention as one SDPA (att_single)
# / + pass 1's 4-bit dense on qmm16c + per-shape ks (DN-KS form) / + both + the drafter (FAST; pass 1 only).
# /Users/kamen/Fit12/scripts/fit_t3.py on T3R2's 16 HumanEval prompts (256 tokens, C_m 47 / C_l 32), F12_ALT=1 (odd prompts
# run the non-ex arms in reverse: every non-ex arm has the same mean position). One process, GPU, lifetime ~11.4 GB.
# Share lock (proof of concept: ROUGH timings, the orchestrator 09:3x). Starts only with no quiet ticket, no PAUSE and
# >= 20 GB free (NO-SWAP: peak + 8 GB under a shared lock), by 14:30; stops on swap +1 GB, 35 min or 15:00. Run ATTACHED.
# Out: /Users/kamen/Fit12/data/scr/kp1 (kp1.txt, kp1_*_{arm}{,_toks,_ph}.json). Usage: sh kp1.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/kp1
K=humaneval/HumanEval; PAUSEF=/Users/kamen/.local/share/quiet/PAUSE
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
cd $F || exit 9
mkdir -p $O
peer wait --timeout 60m >/dev/null 2>&1 || { echo "kp1: a quiet ticket for 60 min: not started"; exit 3; }
while { [ -e $PAUSEF ] || [ "$(freegb)" -lt 20 ]; } && [ "$(date +%H%M)" -lt 1430 ]; do sleep 30; done
[ -e $PAUSEF ] && { echo "kp1: PAUSE at 14:30: not started"; exit 4; }
[ "$(freegb)" -lt 20 ] && { echo "kp1: only $(freegb) GB free at 14:30: not started"; exit 5; }
peer lock share 35m --cpu 200 --gpu 100 "Fit12 12 GB lever 3: Leap's Fri switches ported (drafter qmm16c, pass-1 att_single / DN-KS), 6 arms x 16 HumanEval, MEM 12 GB, ~20 min, rough" >/dev/null 2>&1
SW0=$(swapmb)
echo "$(date +%H:%M:%S) SCR kp1 start: fit_t3.py kp1 arms ex,t80ck2,exd,t80ck2a,t80ck2n,t80ck2f F12_KPORTS=1, 16 HumanEval prompts ($(freegb) GB free, swap $SW0 MB)" >> $LOG
env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_KPORTS=1 F12_T3ARMS=ex,t80ck2,exd,t80ck2a,t80ck2n,t80ck2f F12_CMCL=47,32 \
  F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 \
  $PY -B $F/scripts/fit_t3.py kp1 > $O/kp1.txt 2>&1 &
P=$!; j=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5; j=$((j + 5))
  [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; echo "kp1: swap +1 GB: stopping"; }
  [ $j -ge 2100 ] && { rc=124; echo "kp1: 35 min: stopping"; }
  [ "$(date +%H%M)" -ge 1500 ] && { rc=125; echo "kp1: 15:00: stopping"; }
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR kp1 end (rc $rc, $j s, swap $(swapmb) MB): $(ls $O/kp1_*_toks.json 2>/dev/null | wc -l | tr -d ' ') arm outputs" >> $LOG
grep -E 'KPORTS|qmm16c N|single SDPA|Error|Traceback' $O/kp1.txt | cut -c1-200
exit $rc
