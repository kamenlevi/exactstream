#!/bin/bash
# Fair depth test (results/screens/DEPTHN_LINES.md, written first). Polite guards.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=5 NO_MEM_OTHERS=1
guarded depthn logs/depthn_train.log 25 $PY sb/train_multi.py --tag dn --arms n1:-,n2:deep2 --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 400 || exit 1
unset NO_MEM_OTHERS; export MEMGB=4.5
guarded depthn_eval logs/depthn_eval.log 15 $PY sb/evalx.py --runs s30,dn_n1,dn_n2@deep2 --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/dn.json || exit 1
{
  echo "== fair depth test (DEPTHN_LINES.md) $(date '+%a %d %b %H:%M')"
  for ST in mbpp he; do echo "-- n2 vs n1, $ST"; $PY sb/compare.py results/screens/dn.json "dn_n1|$ST|chain" results/screens/dn.json "dn_n2|$ST|chain" --band 23-30 --json results/screens/dn_$ST.json; done
  echo "-- n1 (400 steps) vs s30 (1,889 steps), mbpp"; $PY sb/compare.py results/screens/dn.json "s30|mbpp|chain" results/screens/dn.json "dn_n1|mbpp|chain" --band 23-30
  $PY -c "import mlx.core as mx; w = mx.load('ckpt/dn_n2.safetensors'); print('n2 extra layer |o_proj| max %.3g, |down_proj| max %.3g' % (float(mx.abs(w['xl.0.o_proj.weight']).max()), float(mx.abs(w['xl.0.down_proj.weight']).max())))"
} | tee -a results/SCREENS.log
echo "depthn done"
