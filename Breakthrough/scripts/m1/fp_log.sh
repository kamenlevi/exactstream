#!/bin/bash
# Footprint logger (Breakthrough, Mon 16 Nov 2026 17:4x): waits up to 5 min for a process matching PATTERN, then logs its
# current phys_footprint and lifetime max every DT s until it ends. Logs only; it never stops anything.
# Usage: bash scripts/m1/fp_log.sh PATTERN LOG [DT]
PAT=${1:?pattern}; L=${2:?log}; DT=${3:-15}
fp() { /usr/bin/python3 -I -B -c '
import ctypes, sys
class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [("u%d" % i, ctypes.c_uint64) for i in range(40)]
ru = RU()
ok = ctypes.CDLL(None).proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(ru)) == 0
print("footprint %.2f GB, lifetime max %.2f GB" % (ru.u7 / 1e9, ru.u28 / 1e9) if ok else "")' "$1" 2>/dev/null; }
t0=$(date +%s); p=$(pgrep -f "$PAT" | head -1)
while [ -z "$p" ]; do
  [ $(( $(date +%s) - t0 )) -ge 300 ] && { echo "$(date '+%H:%M:%S') no process for '$PAT' in 5 min" >> "$L"; exit 0; }
  sleep 2; p=$(pgrep -f "$PAT" | head -1)
done
echo "$(date '+%H:%M:%S') pid $p" >> "$L"
while kill -0 "$p" 2>/dev/null; do
  r=$(fp "$p"); [ -n "$r" ] && echo "$(date '+%H:%M:%S') $r" >> "$L"
  sleep "$DT"
done
echo "$(date '+%H:%M:%S') pid $p ended" >> "$L"
