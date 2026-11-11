#!/bin/sh
# DS (NewIdeas 03:27; Leap, Mon 2026-10-26, pre-registered in ~/Leap/TABLE_350.md): drafter data slope. L1-20's recipe
# (block 32, partial masking 0.5, lr 2e-5, same trainer) on a stratified half of its 99 train windows (data/ds50, symlinks;
# data/ds50_split.json), scored with L1-20's replay on the 24 held-out HumanEval samples, per sample.
#   B = 50 % at 201 steps (4 epochs; = L1-20's steps): s_fix = T(L1-20) / T(B), NewIdeas' lines
#   A = 50 % at 101 steps (2 epochs; = L1-20's epochs): run only when ds_report.py says so (s_fix < 1.12)
# Weights and the trainer's 10-minute checkpoints (1.1 GB, 2.2 GB while one is rewritten) go to a 3 GB RAM disk, not the
# SSD: at 10:26 Mon the first checkpoint took the disk from 100 to 99 GB free, the health monitor went to ACT and
# run_locked.sh stopped arm B at step 85. Each arm trains from scratch (L1-20 ran without a resume); its state / val log
# is copied to data/, its weights go with the RAM disk. Per-sample JSONs stay in data/.
# Run under a held quiet lock via run_locked.sh, cwd ~/Leap.   usage: ds_batch.sh
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts; D=$HOME/Leap/data; RD=/Volumes/LeapRAM
MODES="32:1step,32:tail_0.9,16:1step"
say() { echo "[ds] $* $(date +%H:%M:%S); swap $(sysctl -n vm.swapusage | cut -d'(' -f1| sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
unram() { [ -d $RD ] && hdiutil detach $RD > /dev/null 2>&1; }
trap unram EXIT
trap 'exit 143' TERM INT
say "start"
if [ ! -d $RD ]; then
  DEV=$(hdiutil attach -nomount ram://6291456 | awk '{print $1}')
  diskutil erasevolume HFS+ LeapRAM $DEV > /dev/null || { say "RAM disk FAILED"; hdiutil detach $DEV; exit 1; }
fi
say "RAM disk $RD: $(df -m $RD | tail -1 | awk '{print $4}') MB free"
if [ ! -s $D/ds_l120.json ]; then
  R19_WEIGHTS=$D/l120_ft.safetensors R19_TAG=_ds_l120 R19_MODES=$MODES R19_JSON=$D/ds_l120.json \
    $PY $S/refine19.py > $D/ds_l120.log 2>&1 || { say "L1-20 re-score FAILED"; tail -5 $D/ds_l120.log; exit 1; }
  say "L1-20 re-scored: $(grep 'block 32 1step' $D/ds_l120.log | cut -c1-60)"
fi
arm() {   # $1 = arm, $2 = epochs
  WT=$RD/ds_$1.safetensors
  if [ ! -s $D/ds_$1.json ]; then
    say "train $1 ($2 epochs)"
    $PY $S/dfl2c_train.py --data $D/ds50 --out $WT --bs 32 --micro 1 --epochs $2 --minutes 50 --pmask 0.5 \
      > $D/ds_$1.train.log 2>&1 || { say "training $1 stopped"; tail -5 $D/ds_$1.train.log; exit 1; }
    cp $RD/ds_$1.state.json $D/
    say "score $1"
    R19_WEIGHTS=$WT R19_TAG=_ds_$1 R19_MODES=$MODES R19_JSON=$D/ds_$1.json \
      $PY $S/refine19.py > $D/ds_$1.score.log 2>&1 || { say "scoring $1 FAILED"; tail -5 $D/ds_$1.score.log; exit 1; }
  fi
  rm -f $WT
}
arm B 4
$PY $S/ds_report.py; rc=$?
if [ $rc -eq 10 ]; then
  arm A 2
  $PY $S/ds_report.py
fi
say "done"
