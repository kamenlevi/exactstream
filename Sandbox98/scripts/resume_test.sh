#!/bin/bash
# Resume test for sb/resume.py on a 2-file lever1c smoke: run, SIGTERM after a saved state, rerun -> "resumed at step".
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
rm -f logs/rtest1.log logs/rtest2.log
$PY -I sb/lever1c.py --smoke 2 --minutes 0.3 --evals 0.1 --save-every 2 --tag rtest > logs/rtest1.log 2>&1 &
P=$!
for _ in $(seq 1 240); do grep -q "read at 0.1" logs/rtest1.log && break; kill -0 $P 2>/dev/null || break; sleep 1; done
sleep 4; kill -TERM $P; wait $P; echo "run 1 exit $? ($(grep -c . logs/rtest1.log) lines): $(grep -E 'SIGTERM|Traceback|Error' logs/rtest1.log | head -3)"
ls -la ckpt/lever1c_rtest_smoke_state.safetensors 2>&1 | awk '{print "state file", $5, "bytes"}'
$PY -I sb/lever1c.py --smoke 2 --minutes 0.3 --evals 0.1 --save-every 2 --tag rtest > logs/rtest2.log 2>&1
echo "run 2 exit $?: $(grep -E 'resumed|DONE|Traceback|Error' logs/rtest2.log | head -4)"
ls ckpt/lever1c_rtest_smoke_state* 2>/dev/null && echo "state file NOT removed" || echo "state file removed at the end"
