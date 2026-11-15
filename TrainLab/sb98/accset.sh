#!/bin/bash
# WEEK98 S7 ACCSET (MTP Head): accept-set training for close-call acceptance on Sandbox98. Pre-registered in REPORT
# 'WEEK98 S7 ACCSET' (af0ffee) before it runs.
# 1. screen: cs1 + cs1k in one process (measured peak 4.1 GB), then cs2; 400 steps from native on dn_n1's recipe;
# 2. read-out (cc_readout.py): n1 (control), the three arms, s30 (reference); verdicts by the pre-registered line;
# 3. if cs1 or cs1k is PROMISING: the confirm (the larger MBPP delta-1 lift) on s30's recipe (1,889 steps, one arm) and
#    its read-out vs s30. Paused-safe (run_paused.sh); scratch tag mhx (reused files); -> data/sb98/accset_RESULT.txt.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; L=$HOME/TrainLab/sb98/levers; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=5
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
log "screen training 1: cs1 + cs1k (400 steps)"
$S/run_paused.sh accset_train1 $O/accset_train1.log 30 $PY sb/train_multi.py --tag mhx \
  --arms "cs1:$L/accset1,cs1k:$L/accset1k" --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 400 \
  || { log "training 1 failed"; exit 1; }
grep -q "DONE mhx" $O/accset_train1.log || { log "training 1 did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/accset_train1.log | tail -1)"
log "screen training 2: cs2 (400 steps)"
$S/run_paused.sh accset_train2 $O/accset_train2.log 20 $PY sb/train_multi.py --tag mhx \
  --arms "cs2:$L/accset2" --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 400 \
  || { log "training 2 failed"; exit 1; }
grep -q "DONE mhx" $O/accset_train2.log || { log "training 2 did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/accset_train2.log | tail -1)"
log "screen read-out"
REF="mbpp=4.0764,he=5.7914" JUDGE="mhx_cs1=1,mhx_cs1k=1,mhx_cs2=2" $S/run_paused.sh accset_read $O/accset_read.log 30 \
  $PY $S/cc_readout.py $O/accset_screen.json mhc2a_n1 mhx_cs1 mhx_cs1k mhx_cs2 s30 || { log "read-out failed"; exit 1; }
grep -q "SELF-CHECK.*FAIL" $O/accset_read.log && { log "read-out self-check FAILED"; exit 1; }
grep '^VERDICT' $O/accset_read.log > $O/accset_RESULT.txt
cat $O/accset_RESULT.txt
BEST=$(python3 -I -c "
import json
v = json.load(open('$O/accset_screen.json'))['verdict']
c = [(v[r]['mbpp_diff'], r) for r in ('mhx_cs1', 'mhx_cs1k') if v[r]['call'] == 'PROMISING']
print(max(c)[1] if c else '')")
if [ -z "$BEST" ]; then
  echo "SCREEN: neither delta-1 arm PROMISING -> no confirm (pre-registered: stop and report)" >> $O/accset_RESULT.txt
  log "no confirm"; exit 0
fi
LV=accset1; [ "$BEST" = mhx_cs1k ] && LV=accset1k
echo "SCREEN: $BEST PROMISING -> confirm with levers/$LV at s30's recipe (1,889 steps) vs s30" >> $O/accset_RESULT.txt
[ -f ckpt/mhx_cf.safetensors ] || mv ckpt/mhx_cs2.safetensors ckpt/mhx_cf.safetensors   # reuse a scratch file
log "confirm training: cf = $LV, 1,889 steps"
$S/run_paused.sh accset_conf $O/accset_conf_train.log 60 $PY sb/train_multi.py --tag mhx \
  --arms "cf:$L/$LV" --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 \
  || { log "confirm training failed"; exit 1; }
grep -q "DONE mhx" $O/accset_conf_train.log || { log "confirm training did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/accset_conf_train.log | tail -1)"
JUDGE="mhx_cf=1" $S/run_paused.sh accset_cread $O/accset_conf_read.log 20 \
  $PY $S/cc_readout.py $O/accset_confirm.json s30 mhx_cf || { log "confirm read-out failed"; exit 1; }
grep '^VERDICT' $O/accset_conf_read.log | sed 's/^VERDICT/CONFIRM/; s/PROMISING/PASS/' >> $O/accset_RESULT.txt
log "$(tail -1 $O/accset_RESULT.txt)"
