#!/bin/sh
# L1-20 batch 2 (Leap, Sat 2026-10-24): fine-tune DFlash2 at block 32 with partial masking (--pmask 0.5) on batch 1's
# features (local: the drafter is < 1B; <= 50 min), then score it with the same replay as L1-19 (1- and 2-step, blocks
# 16 / 32) on the 24 held-out HumanEval samples. Resumable training. ~60 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
WT=$D/l120_ft.safetensors
echo "== L1-20 b2 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1)"
if grep -q '"status": "done"' $D/l120_ft.state.json 2>/dev/null; then
  echo "(already trained)"
else
  R=""; [ -s $D/l120_ft.state.json ] && R="--resume"
  $PY $S/dfl2c_train.py --data $D/l120 --out $WT --bs 32 --micro 1 --epochs 2 --minutes 50 --pmask 0.5 $R; rc=$?
  [ $rc -eq 0 ] || { echo "== training stopped (rc $rc): run this batch again to resume"; exit $rc; }
fi
echo "== scoring $(date +%H:%M:%S)"
R19_WEIGHTS=$WT R19_TAG=_l120 $PY $S/refine19.py || { echo "== scoring FAILED"; exit 1; }
echo "== done $(date +%H:%M:%S)"
