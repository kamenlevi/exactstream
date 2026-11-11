#!/bin/bash
# Thursday's map leftovers (the orchestrator 07:0x; Breakthrough, Thu 2026-11-05): from 13:10, in the spare after the records3
# conversion and EvalSuite's records3 check, never beside its 34 GB share runs (the guards: another session's shared run
# > 5 GB, quiet windows, heat, lid, NO-SWAP). Every map process runs with BIGMOE_EXT_BUILD=build_nat (BigMoE's layout-aware
# ext; gated.py reads it). Last start 14:45, hard stop 15:00 (Kamen's day; T0 holds the one > 5 GB job).
#   1. wait for 13:10, then until records3's index is uniform (every layer converted, or every layer row-major) and no
#      conversion process runs: never read a half-converted store;
#   2. the scores-capture SMOKE (gslot.sh, 1 window): checks the routing / baseline top-1 against the map's stored data;
#   3. only if it passed: the s67 runner with data/s67_extra.txt = DENSE-3 (twin screen 6), then
#      qom_0.3_live_c147_bonus4g (twins: the cf 0.3 arm, the cf 0.5 chat arm), each only if it can start by 14:45;
#   4. DENSE-3's pre-registered read-out (dense3_check.py: the INVALID rule) if it ran; the runner's own read-out gives
#      map_kl and the code verdict (dm_screen_read vs screen 6).
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/thu_leftovers.log; X=$H/data/s67_extra.txt
say() { echo "[thu] $* $(date '+%a %H:%M:%S')" >> $LOG; }
at=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 13:10" +%s); go_by=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 14:45" +%s)
say "armed: start >= 13:10, last start 14:45, hard stop 15:00; BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD"
while [ $(date +%s) -lt $at ]; do sleep 30; done
layouts() { $PY -c "import json, os; L = json.load(open(os.path.expanduser('~/BigMoE/models/bare/records3/index.json')))['layers']; print(' '.join(sorted({str(v.get('layout')) for v in L.values()})))" 2>/dev/null; }
while :; do
  [ $(date +%s) -ge $go_by ] && { say "14:45 passed before the store settled; everything to Friday"; exit 0; }
  l=$(layouts); conv=$(pgrep -f "convert_records[3]" | head -1)
  if [ -n "$l" ] && [ "$(echo $l | wc -w | tr -d ' ')" = 1 ] && [ -z "$conv" ]; then say "records3 layouts: $l (uniform), no conversion running"; break; fi
  say "waiting: records3 layouts '$l', conversion pid '${conv:-none}'"; sleep 60
done
GS_LASTGO="2026-11-05 14:45:00" GS_DEADLINE="2026-11-05 15:00:00" bash $H/scripts/dm_s7/gslot.sh scap_smoke 3 16 \
  env DM_OUT=$H/data/scap_smoke $PY $H/scripts/dm_s7/scap_pass.py smoke
rc=$?; say "scores-capture smoke: $([ $rc = 0 ] && echo PASS || echo 'FAILED or not run') (data/gslot/scap_smoke.log)"
[ $rc = 0 ] || { say "the arms do not run on an unchecked store; Friday"; exit 1; }
cp $X $H/data/s67_extra_wed.txt 2>/dev/null
# the arm list is read from data/thu_extra.txt at this point (Thu 07:3x: editable until the smoke ends, no re-arm needed);
# without it, the 07:0x list (DENSE-3, then the cf 0.4 arm)
if [ -s $H/data/thu_extra.txt ]; then { cat $H/data/thu_extra.txt; echo "ALONE 0 13:00"; } > $X; say "arm list from data/thu_extra.txt: $(grep -v '^#' $X | awk '{print $1}' | tr '\n' ' ')"
else printf '# Thursday leftovers (the orchestrator 07:0x)\nqom_0_live_ball_c147_t3+dense3 10 14:45 qom_0_live_ball_c147_t3\nqom_0.3_live_c147_bonus4g 5 14:45 qom_0.3_live_c147_bonus3g qom_0.5_live_c147_bonus5g\nALONE 0 13:00\n' > $X; fi
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-05 14:45:00" S67_DEADLINE="2026-11-05 15:00:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh
say "runner ended (rc $?)"
D3=qom_0_live_ball_c147_t3+dense3
if [ -f ~/FableGod/decision_map/out/$D3.json ]; then
  $PY $H/scripts/dm_s7/dense3_check.py >> $LOG 2>&1
  say "DENSE-3 read-out: $(grep '^VERDICT' $LOG | tail -1); code flips: $(grep -F "| $D3 |" $H/data/s67_slot.log | tail -1 | cut -d'|' -f3)"
fi
say "done"
