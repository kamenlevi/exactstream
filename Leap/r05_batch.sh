#!/bin/sh
# R-05 (SD1 resized to the 10-min protocol; Leap, pre-registered Mon 22:33 EET in TABLE_350 'R-05'): arms U (L1-20's
# recipe) and W (--wloss, acceptance-weighted depths), each 21 steps (~3 min) from the L1-20 checkpoint, scored by the
# 1-step replay on the 24 held-out HumanEval samples at block 32 (primary) and 16. Weights go to a 3 GB RAM disk (the
# SSD is under its floor); per-sample JSONs to data/. Run under a held quiet lock via run_locked.sh, cwd ~/Leap.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/Leap/scripts; D=$HOME/Leap/data; RD=/Volumes/LeapRAM
say() { echo "[r05] $* $(date +%H:%M:%S); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
unram() { [ -d $RD ] && hdiutil detach $RD > /dev/null 2>&1; }
trap 'kill $CHILD 2>/dev/null; wait; unram; exit 143' TERM INT
if [ ! -d $RD ]; then
  DEV=$(hdiutil attach -nomount ram://6291456 | awk '{print $1}')
  diskutil erasevolume HFS+ LeapRAM $DEV > /dev/null || { say "RAM disk FAILED"; hdiutil detach $DEV; exit 1; }
fi
say "start; RAM disk $(df -m $RD | tail -1 | awk '{print $4}') MB free"
for arm in U W; do
  F=""; [ $arm = W ] && F="--wloss"
  $PY $S/r05_train.py --data $D/l120 --init $D/l120_ft.safetensors --out $RD/r05_$arm.safetensors --bs 32 \
    --pmask 0.5 --micro 1 --epochs 0.2 --minutes 3.2 --noval $F > $D/r05_$arm.train.log 2>&1 & CHILD=$!; wait $CHILD
  [ $? -eq 0 ] || { say "train $arm FAILED"; tail -5 $D/r05_$arm.train.log; unram; exit 1; }
  grep -E "step 1/|DONE|time budget" $D/r05_$arm.train.log
  R19_WEIGHTS=$RD/r05_$arm.safetensors R19_MODES=32:1step,16:1step R19_JSON=$D/r05_$arm.json R19_TAG=_r05_$arm \
    $PY $S/refine19.py > $D/r05_$arm.score.log 2>&1 & CHILD=$!; wait $CHILD
  [ $? -eq 0 ] || { say "score $arm FAILED"; tail -5 $D/r05_$arm.score.log; unram; exit 1; }
  grep "1step" $D/r05_$arm.score.log | head -2
  say "arm $arm done"
done
unram
$PY $S/r05_report.py
say "done"
