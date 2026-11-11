#!/bin/sh
# R-31 (b) fix + screen (NewIdeas Tue ~21:0x: profile first, then theta 0.08 first if a cheap fix holds). The 16-row
# profile (turn24, 21:00): > 8-row expert calls take the engine's full-E path, 0.30 s per position vs 0.07 at 4 rows.
# Fix (TL_CHUNK=8): every arm's verify passes run the expert computation in 8-row calls inside the block copy (routing
# still sees the whole 16-row check; rows are independent there). Gate before the screen: 64 positions at 16 rows,
# chunked, <= 0.105 s per position (1.5x the 4-row cost) and RSS < 30 GB, else stop. Screen (REPORT 'R-31 UNIQ-Q',
# amendment 21:0x): v3 at 16 rows (chunked control) + u080, every 8th sequence of ifeval / code / multi.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
$PY tests/test_dynk.py | tail -1 | grep -q "ALL OK" || { echo "== test_dynk FAILED: nothing started"; exit 1; }
echo "== R-31 gate $(date +%H:%M:%S): 16 rows chunked, 64 positions"
TL_PROF=1 TL_MAXPOS=64 TL_CHUNK=8 TL_TAG=prof16c TL_VARIANT=plan4433333333 TL_LIMIT=200 \
  $PY sticky/budget125_tf.py 16 p4433333333 tf_code 99 2>&1 | grep -E "^PROF|Traceback|Error" | tee logs/r31_gate.txt
$PY - <<'PYEOF' || { echo "== R-31 gate FAILED: no screen"; exit 1; }
import re, sys
t = open("/Users/kamen/TrainLab/logs/r31_gate.txt").read()
sp = [float(x) for x in re.findall(r"16 rows [\d.]+s \(([\d.]+) s/pos\)", t)]
rss = [float(x) for x in re.findall(r"RSS ([\d.]+) GB", t)]
tdc = [int(x) for x in re.findall(r"tier_decode_calls \+(\d+)", t)]
ok = len(sp) >= 3 and max(sp[1:]) <= 0.105 and max(rss) < 30 and min(tdc) > 0
print(f"== R-31 gate {'PASS' if ok else 'FAIL'}: s/pos {sp}, RSS max {max(rss) if rss else 0} GB, decode tier calls {tdc}")
sys.exit(0 if ok else 1)
PYEOF
echo "== R-31 screen $(date +%H:%M:%S): v3 at 16 rows (chunked control) + u080, ifeval / code / multi, every 8th"
TL_CHUNK=8 TL_LIMIT=$(( ${TL_LIMIT:-570} - 60 )) TL_TAG=r31 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 16 p4433333333,u080 tf_ifeval,tf_code,tf_multi 8 2>&1 | grep --line-buffered -v "Token indices"
