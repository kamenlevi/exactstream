#!/bin/bash
# G1/G2 map screen: yield watcher for a run beside EvalSuite's waiting B2 (Breakthrough, Mon 16 Nov 2026 17:4x). Armed
# only on NewIdeas' yes + EvalSuite's no-objection, next to the frozen runner's manual GO.
# While the map process (scripts/m1/g1g2_run_map.py) runs, every YIELD_DT s (30): B2's view without my run =
# NO-SWAP (free + purgeable + file-backed) + the map process's current phys_footprint. If that is >= YIELD_NEED GB (43,
# B2's gate) on two reads in a row, TERM the map process: it exits 143, and the frozen runner then unlocks, reads out the
# finished arms and writes G1G2_MAP_END (the rest re-run later with --skip-done).
# Ends when the map process ends, when B2's runner pid is gone, or when no map process starts within YIELD_WAIT s (1200).
# Usage: bash scripts/m1/g1g2_yield.sh <B2 runner pid>     log: data/m1/g1g2_map/yield.log (YIELD_LOG: tests only)
B2PID=${1:?usage: g1g2_yield.sh B2_RUNNER_PID}
NEED=${YIELD_NEED:-43}
DT=${YIELD_DT:-30}
WAIT=${YIELD_WAIT:-1200}
L=${YIELD_LOG:-$HOME/Breakthrough/data/m1/g1g2_map/yield.log}
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
fp() { /usr/bin/python3 -I -B -c '
import ctypes, sys
class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [("u%d" % i, ctypes.c_uint64) for i in range(40)]
ru = RU()
ok = ctypes.CDLL(None).proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(ru)) == 0
print("%.1f" % (ru.u7 / 1e9) if ok else "")' "$1" 2>/dev/null; }
mp() { pgrep -f 'scripts/m1/g1g2_run_ma[p][.]py' | head -1; }

say "armed (pid $$): B2 runner pid $B2PID; yield when NO-SWAP + the map process's footprint >= $NEED GB on two reads ${DT} s apart"
t0=$(date +%s)
p=$(mp)
while [ -z "$p" ]; do
  [ $(( $(date +%s) - t0 )) -ge "$WAIT" ] && { say "no map process ${WAIT} s after arming: exit"; exit 0; }
  sleep 5
  p=$(mp)
done
say "map process pid $p"
hit=0; n=0
while kill -0 "$p" 2>/dev/null; do
  kill -0 "$B2PID" 2>/dev/null || { say "B2 runner $B2PID gone: nothing to yield to; exit"; exit 0; }
  a=$(ns); f=$(fp "$p")
  if [ -n "$f" ]; then
    v=$(awk -v a="$a" -v f="$f" 'BEGIN {printf "%.1f", a + f}')
    if awk -v v="$v" -v n="$NEED" 'BEGIN {exit !(v >= n)}'; then
      hit=$((hit + 1)); say "read: NO-SWAP $a + footprint $f = $v GB >= $NEED ($hit of 2)"
      if [ "$hit" -ge 2 ]; then say "YIELD: TERM map process $p (B2 would fit without my run)"; kill -TERM "$p"; exit 0; fi
    else
      [ "$hit" -gt 0 ] && say "read: NO-SWAP $a + footprint $f = $v GB < $NEED (reset)"
      hit=0
      [ $((n % 10)) = 0 ] && say "status: NO-SWAP $a + footprint $f = $v GB < $NEED"
    fi
  fi
  n=$((n + 1))
  sleep "$DT"
done
say "map process ended: exit"
