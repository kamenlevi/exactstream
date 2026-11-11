#!/bin/sh
# E5a (MTP Head, Sun 11-01 21:2x): a quick DFlash-X screen on D-15a's split (16 train / 8 held-out multilingual answers, where
# a full-body tune on the same data did nothing: kept 2.07 -> 2.06). Branch-only training (sticky/e5_train.py, body frozen),
# 2 epochs, L1-20's recipe at lr 2e-5 (Fable's) and 2e-4 (a zero-init branch moves little in 34 steps); walk of the 8 held-out
# answers with the branch (TL_E5); read-out = d15_read.py (kept per check, sure-token depth-1 hit, per language) vs L1-20.
# Shared lock (small model, peak ~13 GB), done before 23:00.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
for arm in lo:2e-5 hi:2e-4; do
  t=${arm%%:*}; lr=${arm#*:}
  W=$HOME/TrainLab/data/e5a_$t.safetensors
  rm -f data/e5a_$t.state.json
  echo "== E5a $t (lr $lr) train $(date +%H:%M:%S)"
  $PY sticky/e5_train.py --data $HOME/TrainLab/data/d15_multi --init $HOME/Leap/data/l120_ft.safetensors --noval --out $W \
    --bs 32 --micro 1 --S 16 --A 16 --epochs 2 --minutes 8 --pmask 0.5 --lr $lr 2>&1 | grep -v "Token indices" | tail -4
  echo "== E5a $t walk $(date +%H:%M:%S)"
  TL_E5=$W TL_WALK_TAG=_e5a$t TL_WALK_SETS="mheld=$HOME/TrainLab/data/d15_held/*.npz" $PY sticky/d02b_walk.py 2>&1 \
    | grep -v "Token indices" | tail -4
  D15_TAG=e5a$t $PY sticky/d15_read.py
done
echo "== E5a done $(date +%H:%M:%S)"
