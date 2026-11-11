#!/bin/sh
# Rebuild both extensions; PF1 (parallel adopt) profile + A/B; MLX 0.32.4 identity (ar2) and spec125 k=3 on both MLX.
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
rm -rf ext/build324 && cmake -S ext -B ext/build324 -DPython_EXECUTABLE=$HOME/BigMoE/.venv324/bin/python -DCMAKE_BUILD_TYPE=Release \
  -DNB_CMAKE_DIR=$HOME/BigMoE/.venv324/lib/python3.12/site-packages/nanobind/cmake > /dev/null 2>&1
(cd ext/build324 && make 2>&1 | grep -E "error|\[100" | tail -1; strings bigmoe_ext*.so | grep -oE 'v2[0-9]_system[a-z0-9_]*' | sort -u)
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_SPARE=12 BIGMOE_PREFETCH=12"
F='s/.*decode +([0-9.]+) tok.*prefill\+first +([0-9.]+)s.*peak ([0-9.]+) GB.*sha ([0-9a-f]+).*/decode \1 tok\/s prefill+first \2 s peak \3 GB sha \4/'
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
w; sh -c "env $REF BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1 .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E 'PF1|^ar|Error|Traceback' | sed -E '$F' | sed 's|^|[PF1] |'"
w; sh -c "env $REF .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback' | sed -E '$F' | sed 's|^|[base] |'"
w; sh -c "env $REF .venv324/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback|bigmoe:' | sed -E '$F' | sed 's|^|[mlx 0.32.4] |'"
w; sh -c "env $REF .venv324/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback' | sed -E 's/max_draft=5 lookup=0://; s/commit refused 0, //' | sed 's|^|[mlx 0.32.4] |'"
# demand-miss split (50 %; 60 % needs the apps paused): victim B=12, then spares 12/12
R0="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
for e in "BIGMOE_PF_VICTIM=12 BIGMOE_PREFETCH=12" "BIGMOE_SPARE=12 BIGMOE_PREFETCH=12"; do
  w; sh -c "env $R0 $e .venv/bin/python scripts/miss_split.py 0.50 256 2>&1 | grep -E '^plain|^spec3|Error|Traceback' | sed 's|^|[split $e] |'"
done
# widened gate: MTP k=4 (S=5) with BIGMOE_MAXK=50 vs k=3; tokens must stay c09a8163f9
w; sh -c "env $REF BIGMOE_MAXK=50 .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,4:5:0,4:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:' | sed -E 's/max_draft=5 lookup=0://; s/commit refused 0, //' | sed 's|^|[MAXK=50] |'"
# PF2 layer-major prefill (MoonShots e56f23d) on top of PF1: ref vs lmp, sha must match
for T in 4096 16384; do
  w; sh -c "env $REF BIGMOE_PF_PREFILL=1 .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 $T 2>&1 | grep -E '^prompt|^ref|^lmp|Error|Traceback' | sed 's|^|[PF2 $T] |'"
done
