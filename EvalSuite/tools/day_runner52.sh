#!/bin/bash
# EvalSuite runner 52, Fri 2026-11-06 (attached): the MBPP check's exact reference (docs/PREREG_mbpp_2026-11-06.md
# amendment 2): NAX6-N's AR (ref2tnaxn.r2, the decode stage on the converted store) on dec_mbpp, in NewIdeas' slot
# 16:15-16:55 (Kamen extended the big-test window to 17:00). One decode chunk, sized so it ENDS by 16:55; the cli's
# deadline is 16:55 too. Rows left over resume tonight in runner 49's MBPP phase. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day52_runner.log
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 16:15" +%s)
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 16:55" +%s)
export EVALSUITE_DEADLINE=$END
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
unset BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_EXT_BUILD BIGMOE_MAXK EVALSUITE_RELAX_VARIANT
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
mem_busy() { peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
echo $$ > results/q52.pid; trap 'rm -f results/q52.pid' EXIT
say "runner 52 start (pid $$); waits for 16:15, then one ref2tnaxn.r2 MBPP chunk ending by 16:55 ($(count results/ref2tnaxn.r2/decode.jsonl)/100 rows now)"
while [ "$(date +%s)" -lt "$START" ]; do sleep 30; done
while mem_busy || [ -e "$HOME/.local/share/quiet/PAUSE" ]; do
  [ $(( END - $(date +%s) )) -ge 1080 ] || { say "no 18 min left before 16:55: runner stops"; exit 6; }
  sleep 30
done
fg=$(freegb); C=192; NEED=36
[ "$fg" -ge 40 ] || { C=160; NEED=32; }
[ "$fg" -ge 36 ] || { C=128; NEED=27; }
LEFT=$(( (END - $(date +%s)) / 60 ))
CM=$(( LEFT - 8 ))                         # the load (~2 min) + the last row (<= ~3 min) + margin inside the slot
[ "$CM" -ge 10 ] || { say "only $LEFT min left: runner stops"; exit 6; }
say "chunk: --chunk-minutes $CM at C $C ($fg GB NO-SWAP)"
EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode ref2tnaxn.r2 --sets mbpp --chunk-minutes "$CM" --share --C "$C" \
  --need "$NEED" --minutes "$LEFT" >> "$LOG" 2>&1; rc=$?
say "chunk rc=$rc: $(count results/ref2tnaxn.r2/decode.jsonl)/100 rows"
say "runner 52 end"
