#!/bin/sh
# Build the patched copy used by test_sched35.py: BigMoE's bigmoe package with this patch's stream.py/gated.py,
# and bigmoe_ext built from this patch's ext/ (MLX 0.32.2, BigMoE's venv and vendored nanobind). Reads ~/BigMoE only.
set -e
P=$(cd "$(dirname "$0")" && pwd)
T=${SCHED_TREE:-/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/bm}
mkdir -p "$T/ext/build"
rsync -a --delete --exclude __pycache__ ~/BigMoE/bigmoe/ "$T/bigmoe/"
cp "$P/py/stream.py" "$P/py/gated.py" "$T/bigmoe/"
cmake -S "$P/ext" -B "$T/cmake" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$HOME/BigMoE/.venv/bin/python \
      -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake > "$T/cmake.log" 2>&1 || { tail -20 "$T/cmake.log"; exit 1; }
nice -n 10 cmake --build "$T/cmake" -j 2 2>&1 | grep -E "error|warning: unused|Error" || true
cp "$T/cmake"/bigmoe_ext*.so "$T/ext/build/"
ls -la "$T/ext/build/"
