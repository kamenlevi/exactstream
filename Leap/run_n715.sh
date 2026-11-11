#!/bin/bash
# N7-15 + B6 under one quiet lock (Leap, Tue 2026-11-03 01:4x): self-test -> screen -> ref2 wiring + self_test -> B6.
# Each step: a swap guard (stop at swap +2 GB) and a time cap; ref2 is restored at once if its self_test fails.
cd ~/Leap
PY=~/BigMoE/.venv/bin/python
export N7_COND="quiet lock" B6_COND="quiet lock"
LOG=~/Leap/data/run_n715.log
swap() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6}'; }
sw0=$(swap)
step() {
  local name=$1 cap=$2; shift 2
  echo "== $name start $(date +%H:%M:%S) (swap $(swap) MB)" | tee -a $LOG
  "$@" >> $LOG 2>&1 &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    sleep 5; t=$((t + 5))
    if [ "$(echo "$(swap) - $sw0 > 2048" | bc)" = 1 ]; then
      echo "== $name STOPPED: swap +2 GB" | tee -a $LOG; kill $pid; sleep 2; kill -9 $pid 2>/dev/null; return 2; fi
    if [ $t -ge $cap ]; then
      echo "== $name STOPPED: over $cap s" | tee -a $LOG; kill $pid; sleep 2; kill -9 $pid 2>/dev/null; return 3; fi
  done
  wait $pid; local rc=$?
  echo "== $name end rc $rc $(date +%H:%M:%S)" | tee -a $LOG
  return $rc
}
step "N7-15 self-test" 600 $PY scripts/n7_repack_check.py
if tail -1 data/n7_repack_check.txt | grep -q "self-test: PASS"; then
  step "N7-15 screen" 600 $PY scripts/n7_repack.py
  step "ref2 wiring" 60 $PY scripts/ref2_n715_patch.py
  step "ref2 self_test" 300 $PY -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/Leap/scripts')); sys.path.insert(0, os.path.expanduser('~/BigMoE'))
import mlx.core as mx, ref2
ref2.apply()
a, b = mx.zeros((2, 10), mx.int32), mx.zeros((2, 2), mx.int32)
x, y = ref2._native_repack({}, {}, a, b)
inert = x is a and y is b and not ref2.CFG['nax_repack_p1']
r = ref2.self_test()
ok = (all(bool(v) for v in r.values()) if isinstance(r, dict) else bool(r)) and inert
print('self_test', r, 'inert', inert)
print('REF2 SELF_TEST ALL PASS' if ok else 'REF2 SELF_TEST FAIL')
"
  if ! grep -q "REF2 SELF_TEST ALL PASS" $LOG; then
    cp data/ref2_before_n715.py scripts/ref2.py.restore && mv scripts/ref2.py.restore scripts/ref2.py
    echo "== ref2 RESTORED from data/ref2_before_n715.py (self_test did not pass)" | tee -a $LOG
  fi
else
  echo "== N7-15 self-test did not PASS: screen and wiring skipped" | tee -a $LOG
fi
echo "== all done $(date +%H:%M:%S) (swap $(swap) MB)" | tee -a $LOG
