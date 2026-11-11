#!/bin/bash
# Friday night's map arms (Breakthrough, Thu 2026-11-05 09:xx; the orchestrator 06:5x / 07:0x / 07:2x / 09:0x; Fable C's audit
# AUDIT_1105 sections 2-3 and 7; NewIdeas' masks). From Thu 23:45 (FRI_AT; was Fri 00:30 until Kamen 23:1x, via the
# orchestrator: "start the night now"), behind MTP Head's P-OFF and BigMoE's ticket in the quiet
# queue (the runners wait while another session holds or waits for a quiet window, or runs > 5 GB). Every map process runs
# with BIGMOE_EXT_BUILD=build_nat (records3 is converted on Thursday).
#   1. wait for FRI_AT (Thu 23:45), then until records3's index is uniform and no conversion runs;
#   2. the s67 runner on data/fri_extra.txt (test 1's arms on all 36 windows, the code-only and conditional arms, the tail);
#   3. the all-rows scores capture (gslot.sh, scap_pass.py run), then build_massb.py (Fable C's rulings: ranks 0-1 always,
#      then the budget; the rank profile for a row without scores), check_massb, install_c3.sh massB_0.5, and a second runner
#      for massB_0.5 alone;
#   4. on EVERY exit (the end, an abort, a stop): stop my runs, unlock, then touch ~/EvalSuite/results/q45/GO (the
#      orchestrator 09:0x: EvalSuite's runner 45 starts on it; it needs GO by ~11:00, so nothing of mine runs past 10:30).
# The low-priority tail (topk8, the capture, massB) is cut first by its start-by times; the hard stop is 10:30.
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/fri_c3.log; X=$H/data/s67_extra.txt; DM=~/FableGod/decision_map
GO_DIR=${FRI_GO_DIR:-~/EvalSuite/results/q45}   # (a scratch dir for the handoff test)
say() { echo "[fri] $* $(date '+%a %H:%M:%S')" >> $LOG; }
kids() { local p=$1 c; for c in $(pgrep -P $p); do kids $c; echo $c; done; }
handoff() {   # once: stop my runs (TERM, then KILL if frozen; never SIGCONT), unlock, then GO for EvalSuite's runner 45
  [ -n "$DONE_HANDOFF" ] && return; DONE_HANDOFF=1; trap - EXIT INT TERM HUP
  for c in $(kids $$); do kill -TERM $c 2>/dev/null; done; sleep 5
  for c in $(kids $$); do ps -o stat= -p $c 2>/dev/null | grep -q T && kill -KILL $c 2>/dev/null; done
  peer unlock >> $LOG 2>&1
  [ -z "$ACTIVE" ] && { say "stopped before the night slot began ($1): no GO (not my slot yet)"; return; }
  if [ -d $GO_DIR ]; then touch $GO_DIR/GO && say "handed on: touched $GO_DIR/GO ($1)"
  else say "handed on ($1), but $GO_DIR does not exist: GO NOT written (not mine to create)"; fi
}
trap 'handoff "exit"' EXIT; trap 'handoff "stopped"; exit 1' INT TERM HUP
at=$(date -j -f "%Y-%m-%d %H:%M" "${FRI_AT:-2026-11-05 23:45}" +%s); stop_by=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 10:00" +%s)   # Thu 23:2x: was Fri 00:30
say "armed: start >= $(date -r $at '+%a %H:%M') behind the quiet queue; hard stop 10:30; GO for EvalSuite runner 45 on every exit; BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD"
while [ $(date +%s) -lt $at ]; do sleep 30; done
ACTIVE=1   # from here on, every exit hands the GPU on with GO
layouts() { $PY -c "import json, os; L = json.load(open(os.path.expanduser('~/BigMoE/models/bare/records3/index.json')))['layers']; print(' '.join(sorted({str(v.get('layout')) for v in L.values()})))" 2>/dev/null; }
while :; do
  [ $(date +%s) -ge $stop_by ] && { say "10:00 passed before the store settled; nothing ran"; exit 0; }
  l=$(layouts); conv=$(pgrep -f "convert_records[3]" | head -1)
  if [ -n "$l" ] && [ "$(echo $l | wc -w | tr -d ' ')" = 1 ] && [ -z "$conv" ]; then say "records3 layouts: $l (uniform), no conversion running"; break; fi
  say "waiting: records3 layouts '$l', conversion pid '${conv:-none}'"; sleep 60
done
cp $X $H/data/s67_extra_thu.txt 2>/dev/null
{ cat $H/data/fri_extra.txt; echo "ALONE 0 00:00"; } > $X; say "arm list (data/fri_extra.txt): $(grep -v '^#' $X | grep -v '^WHEN\|^ALONE' | awk '{print $1}' | tr '\n' ' ')"
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-06 10:15:00" S67_DEADLINE="2026-11-06 10:30:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!   # a child + wait: a stop runs the trap at once
say "main runner ended (rc $?)"
# the R6 tail: the all-rows capture, the masks, then the arm (each only if it still fits by 10:30)
GS_LASTGO="2026-11-06 10:05:00" GS_DEADLINE="2026-11-06 10:30:00" bash $H/scripts/dm_s7/gslot.sh scap_run 6 16 \
  env DM_OUT=$H/data/scap $PY $H/scripts/dm_s7/scap_pass.py run & wait $!
rc=$?; say "scores capture: $([ $rc = 0 ] && echo PASS || echo 'FAILED or not run') (data/gslot/scap_run.log)"
[ $rc = 0 ] || exit 0
$PY $H/scripts/dm_s7/build_massb.py $H/data/scap/scores $H/data/c3/massB_0.5 >> $LOG 2>&1 || { say "build_massb failed"; exit 0; }
C3_SRC=$H/data/c3 C3_SNAP=$DM/out/c3 bash $H/scripts/dm_s7/install_c3.sh massB_0.5 >> $LOG 2>&1 || { say "massB_0.5 install failed"; exit 0; }
say "massB_0.5 built and installed"
printf '# Friday R6 (the orchestrator 07:2x; Fable C rulings 2-3)\nmassB_0.5 5 10:20 c3_b2d\nALONE 0 00:00\n' > $X
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-06 10:20:00" S67_DEADLINE="2026-11-06 10:30:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
say "massB runner ended (rc $?)"
exit 0
