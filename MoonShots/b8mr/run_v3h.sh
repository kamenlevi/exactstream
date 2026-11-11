#!/bin/sh
# Seq-stamp fix (defer/v3h = Breakthrough's v3g + stamps; defer/v3d2 = my v3d + stamps). Builds both exts, then
# (1) quiet bitwise: B12LL on v3h vs v3g (shas must be equal), v3 exact on v3d2 vs v3d; (3) speed from the same quiet
# runs (interleaved) + the loader's stamp counters; (2) the handoff test under 50 % GPU load on v3h (code + prose x 3):
# torn calls must be 0. Launch holding a quiet lock; unlocks at the end. Log data/b8mr/v3h.txt.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py; D=$M/defer/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/v3h.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "v3h turn start"
for E in v3h v3d2; do
  B=$M/defer/$E/ext
  cmake -S $B -B $B/build -DPython_EXECUTABLE=$PY -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake -DCMAKE_BUILD_TYPE=Release \
    > $B/build_cmake.log 2>&1 && cmake --build $B/build -j 6 > $B/build.log 2>&1
  log "build $E: exit $?; $(ls $B/build/*.so 2>/dev/null | head -1); $(grep -iE ' error' $B/build.log | head -2 | cut -c1-200)"
done
ls $M/defer/v3h/ext/build/*.so > /dev/null 2>&1 || { log "v3h build FAILED"; exit 1; }
H=../MoonShots/defer/v3h
for C in a b; do
  for T in v3g v3h; do
    E=v3g; [ $T = v3h ] && E=$H
    for P in code prose; do
      env B12_ENGINE=$E MR_FRAC=0.10 B12_LATE3=2,10 MR_ARM=wait MR_TAG=_fix_${T}_$C $PY $R gen $P 192 > "$OUT/fix_${P}_${T}_$C.txt" 2>&1
      log "B12LL $T $P $C: $(grep -E '^\[(code|prose)_|^\[stamps|Error|Traceback' "$OUT/fix_${P}_${T}_$C.txt" | tail -2 | tr '\n' ' ' | cut -c1-430)"
    done
  done
done
for E in v3d v3d2; do
  for P in code prose; do
    env DH_ENGINE=$E DH_ARM=v3 DH_TAG=_fix_$E $PY $D gen $P 160 > "$OUT/fix_v3_${P}_$E.txt" 2>&1
    log "v3 exact $E $P: $(grep -E '^\[(code|prose)_|^\[stamps|Error|Traceback' "$OUT/fix_v3_${P}_$E.txt" | tail -2 | tr '\n' ' ' | cut -c1-430)"
  done
done
for C in gpu gpu2 gpu3; do
  load_on
  for P in code prose; do
    env B12_ENGINE=$H MR_FRAC=0.10 B12_LATE3=2,10 MR_CSUM=1 MR_HANDOFF=1 MR_ARM=wait MR_TAG=_fixcs_$C $PY $R gen $P 192 > "$OUT/fixcs_gen_${P}_$C.txt" 2>&1
    log "handoff v3h $P $C: $(grep -E '^\[handoff|^\[stamps|^\[(code|prose)_|Error|Traceback' "$OUT/fixcs_gen_${P}_$C.txt" | tail -3 | tr '\n' ' ' | cut -c1-700)"
  done
  load_off
done
log "v3h turn finished"
