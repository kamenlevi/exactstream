#!/bin/bash
# WEEK98 S8b MARGIN (MTP Head): an argmax-margin loss for the close-call rule on Sandbox98. Pre-registered in REPORT
# 'WEEK98 S8b MARGIN' (f88e870) before it runs.
# 1. screen: mg1 (SURV) + mg1k (exact survival) in one process, 400 steps from native on dn_n1's recipe;
# 2. read-out (cc_readout.py): n1, mg1, mg1k, S7's cs1; mg1k is judged vs n1, mg1 vs cs1 (same survival rule);
# 3. if either is PROMISING: the confirm (the larger MBPP lift) at s30's recipe (1,889 steps) vs s30.
# Paused-safe (run_paused.sh); scratch tag mhx (reused files); -> data/sb98/margin_RESULT.txt.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; L=$HOME/TrainLab/sb98/levers; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=5
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_mg1.safetensors ] || mv ckpt/mhx_cs1k.safetensors ckpt/mhx_mg1.safetensors      # reuse S7's scratch files
[ -f ckpt/mhx_mg1k.safetensors ] || mv ckpt/mhx_cf.safetensors ckpt/mhx_mg1k.safetensors
log "screen training: mg1 + mg1k (400 steps)"
$S/run_paused.sh margin_train $O/margin_train.log 30 $PY sb/train_multi.py --tag mhx \
  --arms "mg1:$L/margin1,mg1k:$L/margin1k" --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 400 \
  || { log "training failed"; exit 1; }
grep -q "DONE mhx" $O/margin_train.log || { log "training did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/margin_train.log | tail -1)"
log "screen read-out"
REF="mbpp=4.0764,he=5.7914" PAIRS="mhx_mg1:mhx_cs1" JUDGE="mhx_mg1k=1,mhx_mg1@mhx_cs1=1" \
  $S/run_paused.sh margin_read $O/margin_read.log 30 \
  $PY $S/cc_readout.py $O/margin_screen.json mhc2a_n1 mhx_mg1 mhx_mg1k mhx_cs1 || { log "read-out failed"; exit 1; }
grep -q "SELF-CHECK.*FAIL" $O/margin_read.log && { log "read-out self-check FAILED"; exit 1; }
grep '^VERDICT' $O/margin_read.log > $O/margin_RESULT.txt
cat $O/margin_RESULT.txt
BEST=$(python3 -I -c "
import json
v = json.load(open('$O/margin_screen.json'))['verdict']
c = [(v[r]['mbpp_diff'], r) for r in ('mhx_mg1k', 'mhx_mg1@mhx_cs1') if v[r]['call'] == 'PROMISING']
print(max(c)[1] if c else '')")
if [ -z "$BEST" ]; then
  echo "SCREEN: neither arm PROMISING -> no confirm (pre-registered: stop; the train-for-the-rule family closes on the sandbox)" >> $O/margin_RESULT.txt
  log "no confirm"; exit 0
fi
if [ "$BEST" = mhx_mg1k ]; then LV=margin1k; SPARE=mhx_mg1; else LV=margin1; SPARE=mhx_mg1k; fi
echo "SCREEN: $BEST PROMISING -> confirm with levers/$LV at s30's recipe (1,889 steps) vs s30" >> $O/margin_RESULT.txt
[ -f ckpt/mhx_mf.safetensors ] || mv ckpt/$SPARE.safetensors ckpt/mhx_mf.safetensors
log "confirm training: mf = $LV, 1,889 steps"
$S/run_paused.sh margin_conf $O/margin_conf_train.log 60 $PY sb/train_multi.py --tag mhx \
  --arms "mf:$L/$LV" --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 \
  || { log "confirm training failed"; exit 1; }
grep -q "DONE mhx" $O/margin_conf_train.log || { log "confirm training did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/margin_conf_train.log | tail -1)"
JUDGE="mhx_mf=1" $S/run_paused.sh margin_cread $O/margin_conf_read.log 20 \
  $PY $S/cc_readout.py $O/margin_confirm.json s30 mhx_mf || { log "confirm read-out failed"; exit 1; }
grep '^VERDICT' $O/margin_conf_read.log | sed 's/^VERDICT/CONFIRM/; s/PROMISING/PASS/' >> $O/margin_RESULT.txt
log "$(tail -1 $O/margin_RESULT.txt)"
