#!/bin/sh
# Fit12 (Fri 6 Nov 09:5x, re-armed 09:5x as kp2 after kp1 hit its swap guard beside a 39 GB job; 12 GB plan lever 3): Leap's Friday kernel switches priced on the 12 GB T3
# path. None is read there as-is (no ref2 / d7b_nax / r57), so /Users/kamen/Fit12/scripts/f12_kports.py ports them (pinned
# kernels /Users/kamen/Fit12/pinned/leap_1106): exd = ex + the drafter on qmm16c with the per-shape ks (D7BNAX_32=2's
# <= 16-row rule; exact: tokens must = ex); t80ck2a / t80ck2n / t80ck2f = t80ck2 + pass 1's attention as one SDPA (att_single)
# / + pass 1's 4-bit dense on qmm16c + per-shape ks (DN-KS form) / + both + the drafter (FAST; pass 1 only).
# /Users/kamen/Fit12/scripts/fit_t3.py on T3R2's 16 HumanEval prompts (256 tokens, C_m 47 / C_l 32), F12_ALT=1 (odd prompts
# run the non-ex arms in reverse: every non-ex arm has the same mean position). One process, GPU, lifetime ~11.4 GB.
# Share lock (proof of concept: ROUGH timings, the orchestrator 09:3x). SLOT GATE (kp1 09:42: +1.08 GB swap in 50 s beside
# EvalSuite's 39 GB agent check): no quiet lock / ticket, no other lock showing MEM > 5 GB, no PAUSE, >= 24 GB free, for 10
# clear minutes (MTP Head's PORT gate needs 3, so NewIdeas' order goes first); starts by 14:15. Was: no quiet ticket, no PAUSE and
# >= 20 GB free (NO-SWAP: peak + 8 GB under a shared lock), by 14:30; stops on swap +1 GB, 35 min or 15:00. Run ATTACHED.
# Out: /Users/kamen/Fit12/data/scr/kp2 (kp2.txt, kp2_*_{arm}{,_toks,_ph}.json). Usage: sh kp2.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/kp2
K=humaneval/HumanEval; PAUSEF=/Users/kamen/.local/share/quiet/PAUSE
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
cd $F || exit 9
mkdir -p $O
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5' | wc -l | tr -d " "; }
clear=0
while [ "$(date +%H%M)" -lt 1415 ]; do
  q=$(nquiet); b=$(nbig); a=$(freegb)
  if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && [ ! -e $PAUSEF ] && [ "$a" -ge 24 ]; then clear=$((clear + 1)); else clear=0; fi
  [ $clear -ge 20 ] && break
  sleep 30
done
[ $clear -ge 20 ] || { echo "kp2: no 10-min clear slot by 14:15 (quiet $q, big $b, $a GB free): not started"; echo "$(date +%H:%M:%S) SCR kp2 NOT STARTED: no clear slot by 14:15 (quiet $q, big $b, $a GB free)" >> $LOG; exit 5; }
peer wait --timeout 5m >/dev/null 2>&1 || { echo "kp2: a quiet ticket appeared: not started"; exit 3; }
peer lock share 35m --cpu 200 --gpu 100 "Fit12 12 GB lever 3: Leap's Fri switches ported (drafter qmm16c, pass-1 att_single / DN-KS), 6 arms x 16 HumanEval, MEM 12 GB, ~20 min, rough" >/dev/null 2>&1
SW0=$(swapmb)
echo "$(date +%H:%M:%S) SCR kp2 start: fit_t3.py kp2 arms ex,t80ck2,exd,t80ck2a,t80ck2n,t80ck2f F12_KPORTS=1, 16 HumanEval prompts ($(freegb) GB free, swap $SW0 MB)" >> $LOG
env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_ALT=1 F12_KPORTS=1 F12_T3ARMS=ex,t80ck2,exd,t80ck2a,t80ck2n,t80ck2f F12_CMCL=47,32 \
  F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 \
  $PY -B $F/scripts/fit_t3.py kp2 > $O/kp2.txt 2>&1 &
P=$!; j=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5; j=$((j + 5))
  [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; echo "kp2: swap +1 GB: stopping"; }
  [ $j -ge 2100 ] && { rc=124; echo "kp2: 35 min: stopping"; }
  [ "$(date +%H%M)" -ge 1500 ] && { rc=125; echo "kp2: 15:00: stopping"; }
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR kp2 end (rc $rc, $j s, swap $(swapmb) MB): $(ls $O/kp2_*_toks.json 2>/dev/null | wc -l | tr -d ' ') arm outputs" >> $LOG
grep -E 'KPORTS|qmm16c N|single SDPA|Error|Traceback' $O/kp2.txt | cut -c1-200
exit $rc
