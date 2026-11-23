#!/bin/bash
# DEPTH screen (results/screens/DEPTH_LINES.md, written first). Polite guards; MEM 7 GB, only with no other memory-heavy job.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=15 MEMGB=7 NO_MEM_OTHERS=1
guarded depth_train logs/depth_train.log 35 $PY sb/train_multi.py --tag dp --arms ctl:-,deep2:deep2,tie2:tie2,distill:distill --depth 30 --steps 400 --mem 8 || exit 1
unset NO_MEM_OTHERS; export MEMGB=4.5
guarded depth_eval logs/depth_eval.log 20 $PY sb/evalx.py --runs s30,dp_ctl,dp_deep2@deep2,dp_tie2@tie2,dp_distill@distill --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/dp.json || exit 1
{
  echo "== DEPTH screen (DEPTH_LINES.md) $(date '+%a %d %b %H:%M')"
  echo "-- control vs s30 (must hold)"
  for ST in mbpp he; do $PY sb/compare.py results/screens/dp.json "s30|$ST|chain" results/screens/dp.json "dp_ctl|$ST|chain" --band 23-30; done
  for A in deep2 tie2 distill; do for ST in mbpp he; do
    echo "-- $A vs ctl, $ST"
    $PY sb/compare.py results/screens/dp.json "dp_ctl|$ST|chain" results/screens/dp.json "dp_$A|$ST|chain" --band 23-30 --json results/screens/dp_${A}_$ST.json
  done; done
  $PY -c "import mlx.core as mx; w = mx.load('ckpt/dp_deep2.safetensors'); print('deep2 extra layer |o_proj| max %.3g, |down_proj| max %.3g' % (float(mx.abs(w['xl.0.o_proj.weight']).max()), float(mx.abs(w['xl.0.down_proj.weight']).max())))"
} | tee -a results/SCREENS.log
echo "depth done"
