#!/bin/sh
# Leap L3 batch 2 (one quiet lock, <= ~85 min; DFL2C_PREREG.md): the fine-tune, then the fine-tuned drafter on the 12
# held-out agentic turns (tau8 + the hybrid bound vs MTP) and on code. The original's numbers are from batch 1.
# Arguments go to dfl2c_train.py (e.g. --epochs 2, or --resume after a stop).
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
WT=$D/dfl2c_ft.safetensors
cd $HOME
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== L3 batch 2 $(date +%H:%M:%S): memory free ${FREE}%, $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free; train $*"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
$PY -m py_compile $S/dfl2c_train.py $S/dfl2_eval.py || { echo "== syntax FAILED"; exit 1; }
$PY $S/dfl2c_train.py "$@"; rc=$?
[ $rc -eq 0 ] || { echo "== training stopped (rc $rc): continue later with --resume"; exit $rc; }
echo "== eval of the fine-tuned drafter $(date +%H:%M:%S)"
$PY $S/dfl2_eval.py --dir $D/dfl2c_eval --weights $WT --tag _l3_ft_agent || { echo "== eval FAILED"; exit 1; }
$PY $S/dfl2_eval.py --dir $D/dfl2c_eval --weights $WT --every --tag _l3_ft_agent_hyb || { echo "== eval FAILED"; exit 1; }
$PY $S/dfl2_eval.py --dir $D/dfl2c_code --weights $WT --tag _l3_ft_code || { echo "== eval FAILED"; exit 1; }
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
