#!/bin/bash
# R-44 (b) POM + LP1: build the ext (ext/build_m01, and ext/build when nobody has it loaded), convert records3p (resumable,
# 4 layers in parallel, at most MIN minutes), then the bitwise checks: m03_suite_check (2-plane slot == the Eval suite's
# tier-2 math), test_records3p (POM / LP1 ext paths + the LP1-patched GPU kernels), test_records3m (M-01 paths) and
# test_records3 (M-01 off). CPU / SSD + a tiny GPU test; run under a quiet lock.
# usage: HAVE_LOCK=1 bash scripts/m03_build_test.sh [MIN=10] > data/m03/build_test.out 2>&1
set -u
cd "$HOME/BigMoE" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
MIN=${1:-10}
date '+start %H:%M:%S'
(cd ext/build_m01 && make -j 4 2>&1 | grep -E 'error|Built target bigmoe_ext' | head -10)
[ ext/build_m01/bigmoe_ext.cpython-312-darwin.so -nt ext/bigmoe_ext.cpp ] || { echo "build_m01 FAILED"; exit 1; }
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then echo "ext/build in use: not rebuilt"
else (cd ext/build && make -j 4 2>&1 | grep -E 'error|Built target bigmoe_ext' | head -5); echo "ext/build rebuilt"; fi
date '+built %H:%M:%S'
"$PY" scripts/m03_convert.py --minutes "$MIN" 2>&1 | tail -6
date '+converted %H:%M:%S'
N=$("$PY" -c 'import json; print(len(json.load(open("models/bare/records3p/index.json"))["layers"]))')
echo "records3p layers verified: $N / 48"
LAY=$("$PY" -c '
import json; d = json.load(open("models/bare/records3p/index.json"))["layers"]
print(",".join(str(L) for L in (0, 23, 47) if str(L) in d) or "0")')
echo "== (1) 2-plane slot vs the Eval suite's tier 2 (layers $LAY)"
BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/m03_suite_check.py "$LAY" 8
echo "== (2) POM / LP1 ext paths + GPU LP1 kernels (layers $LAY)"
BIGMOE_M01=1 BIGMOE_POM=4 BIGMOE_POM_LP1=1 BIGMOE_PF_JIT=1 BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3p.py "$LAY"
echo "== (3) M-01 paths on the new build"
BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3m.py 0,23,47 96 | tail -4
echo "== (4) records3 paths, M-01 off"
BIGMOE_EXT_BUILD=build_m01 "$PY" scripts/test_records3.py 23 96 | tail -2
date '+done %H:%M:%S'
