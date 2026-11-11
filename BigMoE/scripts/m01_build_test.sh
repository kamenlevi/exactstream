#!/bin/bash
# R-44 (a) M-01: build the ext into ext/build_m01 (same source as ext/build; BIGMOE_M01 unset = today's behaviour), then
# (1) the records3 loader test on the new build with M-01 off (unchanged paths), (2) the bitwise M-01 loader test on
# records3m, (3) part-load timing records3 (M-01 off) vs records3m. CPU / SSD only; run under a quiet lock.
# usage: HAVE_LOCK=1 bash scripts/m01_build_test.sh > data/m01/build_test.out 2>&1
set -u
cd "$HOME/BigMoE" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
date '+start %H:%M:%S'
cmake -S ext -B ext/build_m01 -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" > ext/build_m01.cmake.log 2>&1 \
  || { echo "cmake configure failed (ext/build_m01.cmake.log)"; exit 1; }
cmake --build ext/build_m01 -j 4 2>&1 | grep -E 'error|Error|Built target' | head -20
[ -f ext/build_m01/bigmoe_ext.cpython-312-darwin.so ] || { echo "build failed"; exit 1; }
date '+built %H:%M:%S'
echo "== (1) records3 paths on build_m01, M-01 off"
BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3.py 23 96
echo "== (2) M-01 loader paths on records3m"
BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3m.py 0,23,47 96
echo "== (3) part-load timing, records3 with M-01 off"
BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3m.py --r3 0,23,47 96
date '+done %H:%M:%S'
