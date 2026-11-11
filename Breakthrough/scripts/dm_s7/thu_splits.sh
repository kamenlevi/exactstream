#!/bin/bash
# Thursday's split arms (the orchestrator 09:1x; Fable C's audit test 1; NewIdeas' _ball masks c22da10; Breakthrough, Thu
# 2026-11-05 09:2x): right after BigMoE's in-place records3 conversion settles (~10:20), before EvalSuite's runners 43 + 44.
#   1. from 10:10 (the start of my slot: from then on every exit hands on with GO), wait until records3's index is uniformly
#      CONVERTED ("native+p11+pad": the in-place rewrite must be done, a row-major store is not enough) and no conversion
#      process runs; last start 11:15, else the splits go to Friday;
#   2. the scores-capture smoke (gslot.sh; checks the routing / baseline top-1 on the converted store);
#   3. only if it passed: the s67 runner on the three code-window arms: qom_0_split0.9_live_ball+code,
#      qom_0_split0.99_live_ball+code, their twin qom_0_live_ball+code (code_read.py read-outs vs the twin and each other);
#   4. on EVERY exit from 10:10 on: stop my runs, unlock, touch ~/EvalSuite/results/q44/GO (runner 44 follows runner 43).
# The bq pair, DENSE-3 and cf 0.4 wait for Friday's "Thursday leftovers". Locks say "MEM <n> GB" (runner 43 and MTP Head's
# PORT check wait while a share lock shows MEM > 5 GB: one at a time). BIGMOE_EXT_BUILD=build_nat for every map process.
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/thu_splits.log; X=$H/data/s67_extra.txt
GO_DIR=${THU_GO_DIR:-~/EvalSuite/results/q44}
say() { echo "[thu] $* $(date '+%a %H:%M:%S')" >> $LOG; }
kids() { local p=$1 c; for c in $(pgrep -P $p); do kids $c; echo $c; done; }
handoff() {   # once: stop my runs (TERM, then KILL if frozen; never SIGCONT), unlock, then GO for EvalSuite's runner 44
  [ -n "$DONE_HANDOFF" ] && return; DONE_HANDOFF=1; trap - EXIT INT TERM HUP
  for c in $(kids $$); do kill -TERM $c 2>/dev/null; done; sleep 5
  for c in $(kids $$); do ps -o stat= -p $c 2>/dev/null | grep -q T && kill -KILL $c 2>/dev/null; done
  peer unlock >> $LOG 2>&1
  [ -z "$ACTIVE" ] && { say "stopped before my slot began ($1): no GO"; return; }
  if [ -d $GO_DIR ]; then touch $GO_DIR/GO && say "handed on: touched $GO_DIR/GO ($1)"
  else say "handed on ($1), but $GO_DIR does not exist: GO NOT written (not mine to create)"; fi
}
trap 'handoff "exit"' EXIT; trap 'handoff "stopped"; exit 1' INT TERM HUP
at=$(date -j -f "%Y-%m-%d %H:%M" "${THU_AT:-2026-11-05 10:10}" +%s); go_by=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 11:15" +%s)
say "armed: slot from 10:10 (GO to $GO_DIR on every exit from then on), after the conversion settles; last start 11:15, hard stop 11:45"
while [ $(date +%s) -lt $at ]; do sleep 20; done
ACTIVE=1
layouts() { $PY -c "import json, os; L = json.load(open(os.path.expanduser('~/BigMoE/models/bare/records3/index.json')))['layers']; print(' '.join(sorted({str(v.get('layout')) for v in L.values()})))" 2>/dev/null; }
last=
while :; do
  [ $(date +%s) -ge $go_by ] && { say "11:15 passed before the conversion settled (layouts '$(layouts)'); the splits go to Friday"; exit 0; }
  l=$(layouts); conv=$(pgrep -f "convert_records[3]" | head -1)
  if [ "$l" = "native+p11+pad" ] && [ -z "$conv" ]; then say "records3 converted (layouts: $l), no conversion running"; break; fi
  w="records3 layouts '$l', conversion pid '${conv:-none}'"; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 30
done
GS_LASTGO="2026-11-05 11:15:00" GS_DEADLINE="2026-11-05 11:45:00" bash $H/scripts/dm_s7/gslot.sh scap_smoke 3 16 \
  env DM_OUT=$H/data/scap_smoke $PY $H/scripts/dm_s7/scap_pass.py smoke & wait $!
rc=$?; say "scores-capture smoke: $([ $rc = 0 ] && echo PASS || echo 'FAILED or not run') (data/gslot/scap_smoke.log)"
[ $rc = 0 ] || { say "the arms do not run on an unchecked store; the splits go to Friday"; exit 0; }
cp $X $H/data/s67_extra_wed.txt 2>/dev/null
printf '# Thursday split arms (the orchestrator 09:1x): <arm> <est min> <start by HH:MM> <primary twin> <info twins...>\nqom_0_split0.9_live_ball+code 4 11:15 qom_0_live_ball+code qom_0_split0.99_live_ball+code\nqom_0_split0.99_live_ball+code 4 11:15 qom_0_live_ball+code qom_0_split0.9_live_ball+code\nqom_0_live_ball+code 4 11:15 qom_0_live_bq\nALONE 0 00:00\n' > $X
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-05 11:15:00" S67_DEADLINE="2026-11-05 11:45:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
say "runner ended (rc $?)"
for a in qom_0_split0.9_live_ball+code qom_0_split0.99_live_ball+code; do
  [ -f ~/FableGod/decision_map/out/$a.json ] && say "code read: $($PY $H/scripts/dm_s7/code_read.py $a qom_0_live_ball+code 2>&1 | grep '^VERDICT')"
done
exit 0
