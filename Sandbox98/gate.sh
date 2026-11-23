#!/bin/bash
# The validity gate (GATE.md): base drafter s30, then the read-outs and the report. ~40 min on a shared GPU.
# usage: ./gate.sh [LR]   (LR from the lr screen; default 5e-5)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
LR=${1:-5e-5}
mkdir -p logs results
[ -f ckpt/s30.safetensors ] || $PY sb/train.py --run s30 --start native --files base --steps ${BASE_STEPS:-1200} --rows 96 --lr $LR > logs/s30.log 2>&1
$PY sb/evalx.py --runs native,s30 --sets mbpp,he --mode chain --out results/gate_chain.json > logs/gate_chain.log 2>&1
$PY sb/evalx.py --runs s30 --sets mbpp,he --mode true,block:8 --out results/gate_true.json > logs/gate_true.log 2>&1
for m in "L2 2" "L12 12" "L22 22" "Lall 2,7,12,17,22"; do
  set -- $m
  $PY sb/fitmap.py --name $1 --taps $2 >> logs/fitmap.log 2>&1
done
$PY sb/evalx.py --runs s30 --sets mbpp --mode map:L2,map:L12,map:L22,map:Lall,noise:0.3,noise:0.5,noise:0.7 \
  --out results/gate_maps.json > logs/gate_maps.log 2>&1
$PY sb/gatereport.py > logs/gatereport.log 2>&1
echo "gate done: results/gate_report.md"
