#!/bin/sh
# verify-pass split (scripts/verify_split.py): pass 1 saves the continuation, pass 2 measures with fresh pools.
cd ~/BigMoE
memory_pressure | tail -1
rm -f data/verify_split_cont.npy
R="env BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 .venv/bin/python scripts/verify_split.py 0.50 18"
$R 2>&1 | grep -E "saved|Error|Traceback"
./scripts/gpu_quiet.sh > /dev/null
$R 2>&1 | grep -E "^S=|Error|Traceback|line [0-9]"
