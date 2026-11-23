#!/bin/sh
# Leap L5 (DFL2E_PREREG.md, QUEUE L1-04): DFlash2 extended to block 16 by a fine-tune on the 125B's own code / math
# outputs, scored at block 16 on the held-out tf_code / tf_gsm. One quiet lock, ~1.5 h, unattended. Resumable: each
# step skips what is already on disk, and a stopped training resumes. Uses tonight's tf_code / tf_gsm features
# (data/dfl2d_code, dfl2d_math, from the L4 batch).
# Run FROM ~/Leap (the quiet-window thermostat recognises research jobs by their working folder; orchestrator 10-24):
#   cd ~/Leap && nohup ~/Leap/run_locked.sh 100 /bin/sh ~/Leap/scripts/dfl2e_batch.sh > /dev/null 2>&1 &
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
WT=$D/dfl2e_ft.safetensors
cd $HOME/Leap
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== Leap L5 block-16 $(date +%H:%M:%S): memory free ${FREE}%, $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
step() { echo "== $1 $(date +%H:%M:%S)"; }
fail() { echo "== $1 FAILED"; exit 1; }
$PY -m py_compile $S/dfl2e_data.py $S/dfl2d_gen.py $S/dfl2c_extract.py $S/dfl2c_train.py $S/dfl2_eval.py $S/dfl2_exit.py \
  || fail syntax
[ -d $D/dfl2d_code ] && [ -d $D/dfl2d_math ] || fail "tf_code / tf_gsm features (run the L4 batch first)"

step "data: held-out code / math prompts"
[ -s $D/dfl2e_samples.jsonl ] || $PY $S/dfl2e_data.py || fail data
step "original drafter at block 16 (depths 8-15 untrained: the baseline)"
[ -s $D/dfl2_result_l5_orig16_code.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_code --bs 16 --tag _l5_orig16_code || fail eval
[ -s $D/dfl2_result_l5_orig16_math.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_math --bs 16 --tag _l5_orig16_math || fail eval
step "generation: the 125B's own code / math outputs (C=128, <= 50K tokens or 40 min)"
[ -e $D/dfl2e_gen.done ] || { DFL2D_IN=$D/dfl2e_samples.jsonl DFL2D_OUT=$D/dfl2e_windows.jsonl DFL2D_C=128 DFL2D_MAXT=400 \
  $PY $S/dfl2d_gen.py 50000 40 && touch $D/dfl2e_gen.done; } || fail generation
step "features + labels on them"
DFL2C_IN=$D/dfl2e_windows.jsonl DFL2C_OUT=$D/dfl2e $PY $S/dfl2c_extract.py || fail extraction
step "training at block 16"
if grep -q '"status": "done"' $D/dfl2e_ft.state.json 2>/dev/null; then
  echo "(already trained)"
else
  R=""; [ -s $D/dfl2e_ft.state.json ] && R="--resume"
  $PY $S/dfl2c_train.py --data $D/dfl2e --out $WT --bs 16 --micro 1 --epochs 2 --minutes 30 $R; rc=$?
  [ $rc -eq 0 ] || { echo "== training stopped (rc $rc): run this batch again to resume"; exit $rc; }
fi
step "fine-tuned drafter at block 16, and at block 8"
$PY $S/dfl2_eval.py --dir $D/dfl2d_code --weights $WT --bs 16 --tag _l5_ft16_code || fail eval
$PY $S/dfl2_eval.py --dir $D/dfl2d_math --weights $WT --bs 16 --tag _l5_ft16_math || fail eval
$PY $S/dfl2_eval.py --dir $D/dfl2d_code --weights $WT --tag _l5_ft8_code || fail eval
step "QUEUE L1-09: early-exit screen (layers 36 / 40 vs 48, code / GSM8K / agentic)"
[ -s $D/exit_result.txt ] || $PY $S/dfl2_exit.py || fail "exit screen"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
