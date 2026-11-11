#!/bin/sh
# R-48 M01-12 in ONE quiet lock (runner_m01.sh's step); stops at the first failure:
#   1. build engine v3n's ext (v3n/ext/build; nobody has it loaded) if missing or older than its source
#   2. CPU / SSD loader test (scripts/test_v3n_m01.py, L0 / L23 / L47 x 96 experts): BIGMOE_M01=1 on records3m, then
#      --r3 (M-01 off on records3); both must PASS bitwise
#   3. page cache: no records3 / records3m file > 1 % resident (evicts with msync(MS_INVALIDATE) if one is)
#   4. A1 S1 S2 A2, each its own process (scripts/m01_ab.py): A = engine v3m on records3, S = v3n + BIGMOE_M01=1 on
#      BigMoE's records3m; then scripts/m01_sum.py (the verdict against the lines in SCOREBOARD 4b90156)
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/m01; B=$HOME/BigMoE/models/bare
SO=v3n/ext/build/bigmoe_ext.cpython-312-darwin.so
date '+%H:%M:%S start'
if [ ! -f $SO ] || [ v3n/ext/bigmoe_ext.cpp -nt $SO ]; then
  cmake -S v3n/ext -B v3n/ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" \
    -DNB_CMAKE_DIR="$HOME/BigMoE/.nb215/nanobind/cmake" > v3n/ext/build.cmake.log 2>&1 \
    || { echo "cmake configure failed (v3n/ext/build.cmake.log)"; exit 2; }
  cmake --build v3n/ext/build -j 4 > v3n/ext/build.log 2>&1 || { echo "build failed:"; grep -E 'error' v3n/ext/build.log | head; exit 2; }
  [ -f $SO ] || { echo "build made no .so"; exit 2; }
  date '+%H:%M:%S built v3n ext'
fi
$PY -c 'import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' scripts/m01_ab.py scripts/m01_sum.py \
  scripts/ubc_check.py scripts/test_v3n_m01.py v3n/bigmoe_v3f/tier.py v3n/bigmoe_v3f/stream.py || exit 2
BIGMOE_M01=1 $PY scripts/test_v3n_m01.py 0,23,47 96 > "$OUT/loader_m01.txt" 2>&1; rc=$?
cat "$OUT/loader_m01.txt"; [ $rc -eq 0 ] || { echo "loader test (M-01) FAILED"; exit 2; }
env -u BIGMOE_M01 $PY scripts/test_v3n_m01.py --r3 0,23,47 96 > "$OUT/loader_r3.txt" 2>&1; rc=$?
cat "$OUT/loader_r3.txt"; [ $rc -eq 0 ] || { echo "loader test (--r3, M-01 off) FAILED"; exit 2; }
date '+%H:%M:%S loader tests PASS'
$PY scripts/ubc_check.py --evict "$B/records3" "$B/records3m" || { echo "page cache above the line"; exit 3; }
for arm in A1 S1 S2 A2; do
  case $arm in A*) E="B12_ENGINE=v3m";; S*) E="B12_ENGINE=v3n BIGMOE_M01=1";; esac
  rm -f "$OUT/$arm.jsonl"
  env -u BIGMOE_M01 -u B12_FOLD -u B12_PFR_GRID -u B12_FOLD_AW -u B12_FOLD_CM -u BT_DIR $E M01_OUT="$OUT" \
    $PY scripts/m01_ab.py $arm > "$OUT/$arm.txt" 2>&1 || { echo "$arm failed:"; tail -5 "$OUT/$arm.txt"; exit 1; }
  date "+%H:%M:%S $arm done"
  grep -E "RESULT|TOTAL" "$OUT/$arm.txt" | cut -c1-300
done
$PY scripts/m01_sum.py "$OUT"
