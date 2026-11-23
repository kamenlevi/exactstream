#!/bin/sh
# Leap, Sat 10-24 (DFL2D_PREREG.md): the DFlash2 tree bound on code / math, then L4 (DFlash2 trained on the 125B's own
# agentic outputs) and its verdict. One quiet lock, <= 2 h, unattended. Resumable: each step skips what is already on
# disk, and a stopped training resumes from its checkpoint, so after a stop just run the same command again.
# Run FROM ~/Leap (the quiet-window thermostat keys on the working folder):
#   cd ~/Leap && nohup ~/Leap/run_locked.sh 120 /bin/sh ~/Leap/scripts/dfl2d_batch.sh > /dev/null 2>&1 &
# 03:35: the first run's generation died at 17 turns, when a quiet-window SIGSTOP tripped BigMoE's spin gates. Rerun
# resumes there, at C = 192 (~28 GB; 04:55: generation and extraction, for more data per minute; pausewait.py now
# makes passes wait at their boundary through a quiet window).
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
WT=$D/dfl2d_ft.safetensors
cd $HOME/Leap
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== Leap tree + L4 $(date +%H:%M:%S): memory free ${FREE}%, $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
$PY -m py_compile $S/dfl2d_data.py $S/dfl2d_gen.py $S/dfl2_tree.py $S/dfl2_eval.py $S/dfl2_extract.py $S/dfl2c_extract.py \
  $S/dfl2c_train.py $S/dfl2_exit.py $S/pausewait.py $S/dfl2_trunc.py && sh -n $HOME/Leap/run_locked.sh || { echo "== syntax FAILED"; exit 1; }
step() { echo "== $1 $(date +%H:%M:%S)"; }
fail() { echo "== $1 FAILED"; exit 1; }

step "data: turns to generate"
[ -s $D/dfl2d_samples.jsonl ] || $PY $S/dfl2d_data.py 110 || fail data
step "features: the 12 held-out agentic turns, code, GSM8K"
DFL2_ONLY=agent DFL2_OUT=$D/dfl2d_eval $PY $S/dfl2_extract.py || fail "eval features"
DFL2_ONLY=tf_code DFL2_OUT=$D/dfl2d_code $PY $S/dfl2_extract.py || fail "code features"
DFL2_ONLY=tf_gsm DFL2_OUT=$D/dfl2d_math $PY $S/dfl2_extract.py || fail "math features"
step "tree bound (original drafter)"
[ -s $D/tree_result.txt ] || $PY $S/dfl2_tree.py $D/dfl2d_code $D/dfl2d_math || fail tree
step "original drafter, fixed counting (must reproduce L3 per turn in the old counting)"
[ -s $D/dfl2_result_l4_orig_agent.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_eval --every --tag _l4_orig_agent || fail eval
[ -s $D/dfl2_result_l4_orig_code.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_code --tag _l4_orig_code || fail eval
[ -s $D/dfl2_result_l4_orig_math.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_math --tag _l4_orig_math || fail eval
step "block 16 on the agentic turns (original; L5's code / math fine-tune if it exists): does block 16 or code training help agentic?"
[ -s $D/dfl2_result_l4_orig16_agent.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_eval --bs 16 --tag _l4_orig16_agent || fail eval
[ -s $D/dfl2e_ft.safetensors ] && { [ -s $D/dfl2_result_l4_l5ft16_agent.txt ] || $PY $S/dfl2_eval.py --dir $D/dfl2d_eval --weights $D/dfl2e_ft.safetensors --bs 16 --tag _l4_l5ft16_agent || fail eval; }
step "side analysis: verify fewer rows when the drafter is unsure (dfl2_trunc.py; GPU, ~1 min each; failures do not stop the batch)"
[ -s $D/trunc_orig8_code.txt ] || $PY $S/dfl2_trunc.py --dir $D/dfl2d_code --bs 8 --tag _orig8_code || echo "== trunc failed"
[ -s $D/trunc_orig16_code.txt ] || $PY $S/dfl2_trunc.py --dir $D/dfl2d_code --bs 16 --tag _orig16_code || echo "== trunc failed"
[ -s $D/dfl2e_ft.safetensors ] && { [ -s $D/trunc_l5ft16_code.txt ] || $PY $S/dfl2_trunc.py --dir $D/dfl2d_code --weights $D/dfl2e_ft.safetensors --bs 16 --tag _l5ft16_code || echo "== trunc failed"; }
step "L1-09 control (~4 min): an exit after the last layer must equal the full model; the curve over the last layers"
[ -s $D/exit_control.txt ] || DFL2X_EXITS=43,45,46,47 DFL2X_N=6 DFL2X_OUT=$D/exit_control.txt $PY $S/dfl2_exit.py || echo "== exit control failed"
step "generation: the 125B's own agentic outputs (<= 36K tokens or 45 min)"
[ -e $D/dfl2d_gen.done ] || { DFL2D_C=192 $PY $S/dfl2d_gen.py 36000 45 && touch $D/dfl2d_gen.done; } || fail generation
step "features + labels on them"
DFL2C_C=192 DFL2C_IN=$D/dfl2d_windows.jsonl DFL2C_OUT=$D/dfl2d $PY $S/dfl2c_extract.py || fail extraction
step "training"
if grep -q '"status": "done"' $D/dfl2d_ft.state.json 2>/dev/null; then
  echo "(already trained)"
else
  R=""; [ -s $D/dfl2d_ft.state.json ] && R="--resume"
  $PY $S/dfl2c_train.py --data $D/dfl2d --out $WT --epochs 2 --minutes 30 $R; rc=$?
  [ $rc -eq 0 ] || { echo "== training stopped (rc $rc): run this batch again to resume"; exit $rc; }
fi
step "fine-tuned drafter"
$PY $S/dfl2_eval.py --dir $D/dfl2d_eval --weights $WT --every --tag _l4_ft_agent || fail eval
$PY $S/dfl2_eval.py --dir $D/dfl2d_code --weights $WT --tag _l4_ft_code || fail eval
step "L1-09 early-exit screen (if the L5 batch ran out of lock time before it)"
[ -s $D/exit_result.txt ] || $PY $S/dfl2_exit.py || fail "exit screen"
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
