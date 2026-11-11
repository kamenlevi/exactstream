#!/bin/sh
# DS2 (Leap, Mon 2026-10-26, pre-registered in ~/Leap/TABLE_350.md 'DS2'; DS's pre-registered next step): the compute
# slope at 100 % data. L1-20's recipe on all of its windows (data/l120) at 401 steps (--epochs 4; L1-20 = 201 steps),
# scored like DS: c100 = T(C) / T(L1-20) (scripts/ds_report.py). Weights + checkpoints on a 3 GB RAM disk, as in
# ds_batch.sh (the SSD is at its 100 GB floor), detached on exit; a stopped run restarts from scratch.
# Run under a held quiet lock via run_locked.sh (~57 min), cwd ~/Leap.   usage: ds2_batch.sh
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts; D=$HOME/Leap/data; RD=/Volumes/LeapRAM; WT=$RD/ds_C.safetensors
MODES="32:1step,32:tail_0.9,16:1step"
say() { echo "[ds2] $* $(date +%H:%M:%S); swap $(sysctl -n vm.swapusage | cut -d'(' -f1| sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
unram() { [ -d $RD ] && hdiutil detach $RD > /dev/null 2>&1; }
trap unram EXIT
trap 'exit 143' TERM INT
say "start"
[ -s $D/ds_l120.json ] || { say "need data/ds_l120.json (ds_batch.sh makes it)"; exit 1; }
if [ ! -d $RD ]; then
  DEV=$(hdiutil attach -nomount ram://6291456 | awk '{print $1}')
  diskutil erasevolume HFS+ LeapRAM $DEV > /dev/null || { say "RAM disk FAILED"; hdiutil detach $DEV; exit 1; }
fi
if [ ! -s $D/ds_C.json ]; then
  say "train C (100 %, 4 epochs = 401 steps)"
  $PY $S/dfl2c_train.py --data $D/l120 --out $WT --bs 32 --micro 1 --epochs 4 --minutes 60 --pmask 0.5 \
    > $D/ds_C.train.log 2>&1 || { say "training C stopped"; tail -5 $D/ds_C.train.log; exit 1; }
  cp $RD/ds_C.state.json $D/
  say "score C"
  R19_WEIGHTS=$WT R19_TAG=_ds_C R19_MODES=$MODES R19_JSON=$D/ds_C.json \
    $PY $S/refine19.py > $D/ds_C.score.log 2>&1 || { say "scoring C FAILED"; tail -5 $D/ds_C.score.log; exit 1; }
fi
rm -f $WT
$PY $S/ds_report.py
say "done"
