#!/bin/sh
# S10c all-layer M1a (NewIdeas Tue 20:4x, after b34 BETWEEN): detached runner for a pacing sleep. Waits for BigMoE's b35
# capture (x at every layer; meta.json written and 2 min old), then PAUSE and others' quiet locks, then fits S10b's frozen
# recipe (M1a lr 3e-4 wd 10, 5 epochs) at lead 2 on every layer whose target exists, judge models on prompts 0-3, and
# exports them folded (m1a_46pairs.npz / .json / _check.json, the m1a_8pairs form). CPU, 4 BLAS threads, nice 10.
cd $HOME/TrainLab
D=$HOME/BigMoE/data/b35/s10b
LOG=$HOME/TrainLab/logs/s10b_fit_allL.out
echo "$(date '+%m-%d %H:%M:%S') runner waiting for $D/meta.json" > $LOG
until [ -f $D/meta.json ] && [ $(( $(date +%s) - $(stat -f %m $D/meta.json) )) -gt 120 ]; do sleep 30; done
while [ -f $HOME/.local/share/quiet/PAUSE ]; do sleep 30; done
peer wait --timeout 120m >> $LOG 2>&1
echo "$(date '+%m-%d %H:%M:%S') fit starts" >> $LOG
S10B_LEADS=2 S10B_LABEL=ld S10B_FROZEN=M1a,3e-4,10,5 S10B_EXPORT=$HOME/TrainLab/data S10B_EXPORT_NAME=m1a_all \
  nice -n 10 $HOME/BigMoE/.venv/bin/python sticky/s10b_fit.py $D b35allL >> $LOG 2>&1
echo "rc $? $(date '+%m-%d %H:%M:%S')" >> $LOG
