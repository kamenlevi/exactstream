#!/bin/bash
# EvalSuite runner 53, Sat 2026-11-07 (attached): CERT's missing exact references (docs/PREREG_cert_2026-11-08.md):
# NAX6-N's AR (ref2tnaxn.r3, the decode stage on the converted store) on dec_humanevalx (104) + dec_mathx (50), right after
# runner 49 (tonight's chain) ends (NewIdeas). Chunks of 45 min; a chunk starts only if start + the longest chunk so far
# ends by Sat 16:00 (NewIdeas 23:35: no 10:00 gate; S4 by 16:00), and none starts once S4's runner 54 has written
# results/S4_READY (it yields to S4). Rows left over resume in a later slot. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day53_runner.log
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-07 16:00" +%s)   # NewIdeas 23:35: no 10:00 gate; S4 by 16:00
export EVALSUITE_DEADLINE=$END
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
unset BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_EXT_BUILD BIGMOE_MAXK EVALSUITE_RELAX_VARIANT
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
RUN=ref2tnaxn.r3
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
mem_busy() { peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
calm() { while [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; do sleep 30; done; }
alive49() { [ -f results/q49.pid ] && kill -0 "$(cat results/q49.pid)" 2>/dev/null; }
echo $$ > results/q53.pid; trap 'rm -f results/q53.pid' EXIT
say "runner 53 start (pid $$); waits for runner 49's end, then $RUN on humanevalx + mathx ($(count results/$RUN/decode.jsonl)/154 rows)"
while alive49; do sleep 60; done
sleep 300      # BigMoE's s4_final.sh writes final_env.txt as runner 49 ends: give S4 (runner 54) the first slot
# yield to an S4 not yet run on this final_env.txt (S4_DONE newer than the file = S4 ran on it)
{ [ -e results/S4_READY ] || { [ -s "$HOME/BigMoE/data/stack/final_env.txt" ] && ! [ results/S4_DONE -nt "$HOME/BigMoE/data/stack/final_env.txt" ]; }; } && { say "final_env.txt / S4_READY is in: runner 53 yields to S4 at $(count results/$RUN/decode.jsonl)/154"; exit 0; }
EST=3000; NP=0
while [ "$(count results/$RUN/decode.jsonl)" -lt 154 ]; do
  calm
  while mem_busy; do sleep 60; calm; done
  [ -e results/S4_READY ] && { say "S4 is ready (results/S4_READY): runner 53 yields at $(count results/$RUN/decode.jsonl)/154; the rest resumes after S4"; exit 0; }
  [ $(( $(date +%s) + EST )) -le "$END" ] || { say "no chunk fits before 16:00 ($((EST / 60)) min): runner stops at $(count results/$RUN/decode.jsonl)/154"; exit 6; }
  fg=$(freegb); C=192; NEED=36; [ "$fg" -ge 40 ] || { C=160; NEED=32; }; [ "$fg" -ge 36 ] || { C=128; NEED=27; }
  b=$(count "results/$RUN/decode.jsonl"); t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode "$RUN" --sets humanevalx,mathx --chunk-minutes 45 --share --C "$C" \
    --need "$NEED" --minutes 60 >> "$LOG" 2>&1; rc=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$EST" ] && EST=$d
  say "chunk rc=$rc at C $C: $(count results/$RUN/decode.jsonl)/154 rows ($((d / 60)) min)"
  if [ "$(count "results/$RUN/decode.jsonl")" -le "$b" ]; then
    NP=$((NP + 1)); [ "$NP" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }; sleep 60
  else NP=0; fi
done
touch results/CERTREF_DONE
say "runner 53 end: $(count results/$RUN/decode.jsonl)/154 rows"
