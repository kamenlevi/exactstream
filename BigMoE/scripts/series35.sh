#!/bin/sh
# 35B timing series under one quiet lock: plain ref, streamed uniform/ends, MTP k=2 ref/streamed.
cd ~/BigMoE
PY=~/BigMoE/.venv/bin/python
run() { ( "$@" 2>&1 | grep -v -i warn | tail -1 ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
run $PY scripts/bench35.py ref 1.0 200
run $PY scripts/bench35.py stream 0.6 200 uniform
run $PY scripts/bench35.py stream 0.6 200 ends
run $PY scripts/bench35_mtp.py stream 0.6 200 2 ends
run $PY scripts/bench35_mtp.py ref 1.0 200 2
