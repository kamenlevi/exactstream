#!/bin/sh
# R-49: run by runner_cand.sh as a step inside its quiet lock (only with >= 5 min of the lock left): the 12 GB routing
# capture (scripts/r49_capture.py, live bt_b12 at its defaults), only the prompts that fit before the lock's end - 40 s.
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
U=$($PY -I -S -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["until"]))' "$HOME/.claude-peers/locks/$CLAUDE_PID.json") || exit 1
exec env -u B12_ENGINE -u BIGMOE_M01 -u BT_DIR R49_DEADLINE=$((U - 40)) $PY scripts/r49_capture.py
