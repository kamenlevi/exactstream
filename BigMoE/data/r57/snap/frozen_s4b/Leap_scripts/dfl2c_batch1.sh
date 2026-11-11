#!/bin/sh
# Leap L3 batch 1 (one quiet lock, ~25 min; DFL2C_PREREG.md): training windows from Kamen's sessions, the 125B's features
# and labels on them (a 3-window head first so the trainer's forward check fails fast), the 12 held-out agentic turns'
# and the code sequences' features (KEEP 2055), the original drafter's eval on them (must give 2.86 / 6.15), the
# trainer's step timing.
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
cd $HOME
FREE=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
echo "== L3 batch 1 $(date +%H:%M:%S): memory free ${FREE}%, $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
[ -n "$FREE" ] && [ "$FREE" -lt 45 ] && { echo "== not enough free memory (need >= 45%), stopping"; exit 3; }
$PY -m py_compile $S/dfl2c_data.py $S/dfl2c_extract.py $S/dfl2c_train.py $S/dfl2_eval.py && sh -n $HOME/Leap/run_locked.sh \
  || { echo "== syntax FAILED"; exit 1; }
echo "== data $(date +%H:%M:%S)"
[ -s $D/dfl2c_windows.jsonl ] || $PY $S/dfl2c_data.py 250000 || { echo "== data FAILED"; exit 1; }
echo "== extract, first 3 windows $(date +%H:%M:%S)"
DFL2C_LIMIT=3 $PY $S/dfl2c_extract.py || { echo "== extract FAILED"; exit 1; }
echo "== trainer forward check $(date +%H:%M:%S)"
$PY $S/dfl2c_train.py --check || { echo "== trainer check FAILED"; exit 1; }
echo "== extract, all windows $(date +%H:%M:%S)"
$PY $S/dfl2c_extract.py || { echo "== extract FAILED"; exit 1; }
echo "== features of the 12 held-out turns and the code sequences $(date +%H:%M:%S)"
DFL2_ONLY=agent DFL2_OUT=$D/dfl2c_eval $PY $S/dfl2_extract.py || { echo "== eval features FAILED"; exit 1; }
DFL2_ONLY=tf_code DFL2_OUT=$D/dfl2c_code $PY $S/dfl2_extract.py || { echo "== code features FAILED"; exit 1; }
echo "== original drafter on them $(date +%H:%M:%S)"
$PY $S/dfl2_eval.py --dir $D/dfl2c_eval --tag _l3_orig_agent || { echo "== eval FAILED"; exit 1; }
$PY $S/dfl2_eval.py --dir $D/dfl2c_code --tag _l3_orig_code || { echo "== eval FAILED"; exit 1; }
echo "== trainer timing $(date +%H:%M:%S)"
$PY $S/dfl2c_train.py --time || { echo "== trainer timing FAILED"; exit 1; }
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
