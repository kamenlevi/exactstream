#!/bin/sh
# dh_w4 prototype, GPU turn 1 (launch holding a quiet lock; unlocks at the end; summary in data/defer/summary.txt):
# (0) build the ext copy (full-slot phase B); (s) SSD batch policy microbench; (q) IFEval all 30 sequences, v3 control vs dh_w4 (tf_dense.py); (1) correctness: teacher-forced logits at S=4 and S=8 for v3 / dh10 / force /
# ref / dh, then tf_compare (dh10 == v3 and force == ref bit for bit; a failure is logged, the turn goes on); (2) all-hits overhead: fixed-state verify
# passes at S=4 and S=8 for v3 / dh / force; (3) streamed MTP decode A/B at 24 GB, code / prose / edit: v3, dh, d1
# (one layer late), pipe (dh + pipelined late reads, if built); round 2 (dh, v3) if time allows.
M=$HOME/MoonShots; OUT=$M/data/defer; PY=$HOME/BigMoE/.venv/bin/python; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/summary.txt"; }
fin() { peer unlock > /dev/null 2>&1; log "$1"; exit 0; }
TSTART=$(date +%s)
log "turn 1 start"
if ! ls $M/defer/v3d/ext/build/bigmoe_ext*.so > /dev/null 2>&1; then
  (mkdir -p $M/defer/v3d/ext/build && cd $M/defer/v3d/ext/build && \
   cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PY -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake && make -j8) \
   > "$OUT/build.txt" 2>&1
  ls $M/defer/v3d/ext/build/bigmoe_ext*.so > /dev/null 2>&1 || fin "STOP: ext build failed (build.txt)"
  log "ext built"
fi
if ! ls $M/defer/v3d/ext/build_pipe/bigmoe_ext*.so > /dev/null 2>&1; then     # variant: pipelined late reads (optional)
  (mkdir -p $M/defer/v3d/ext/build_pipe && cd $M/defer/v3d/ext/build_pipe && \
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DDEFER_PIPE=1 -DPython_EXECUTABLE=$PY \
     -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake && make -j8) > "$OUT/build_pipe.txt" 2>&1
  log "pipe ext: $(ls $M/defer/v3d/ext/build_pipe/bigmoe_ext*.so > /dev/null 2>&1 && echo built || echo FAILED)"
fi
# SSD batch policy for the late reads (seq = today's second batch vs issuing them early); ~20 s, cold records3 reads
clang -O2 -o "$OUT/ssd_batch" $M/defer/ssd_batch.c > "$OUT/ssd_batch_build.txt" 2>&1 && \
  "$OUT/ssd_batch" $HOME/BigMoE/models/bare/records3 15 > "$OUT/ssd_batch.txt" 2>&1
log "ssd_batch: $(grep -c . "$OUT/ssd_batch.txt") lines (data/defer/ssd_batch.txt)"
# pipe delay: the smallest d whose phase A stays within 10 % of seq's (mean over the (nw, nl) cells), else seq (-1)
PIPE_US=$($PY - "$OUT/ssd_batch.txt" <<'PYEOF'
import sys, collections
A, B = collections.defaultdict(list), collections.defaultdict(list)
for l in open(sys.argv[1]):
    f = l.split()
    if len(f) == 5 and f[0].isdigit():
        A[f[2]].append(float(f[3])); B[f[2]].append(float(f[4]))
if "seq" not in A:
    print(-1); sys.exit()
mA = {k: sum(v) / len(v) for k, v in A.items()}; mB = {k: sum(v) / len(v) for k, v in B.items()}
ok = sorted((int(k[1:]), k) for k in mA if k != "seq" and mA[k] <= 1.10 * mA["seq"] and mB[k] < mB["seq"])
print(ok[0][0] if ok else -1)
PYEOF
)
log "pipe delay rule (all cells within 10 %): ${PIPE_US} us; used instead: 250 us + 200 us per waited read beyond the first"
PIPE_US=250; PIPE_STEP=200                     # ssd_batch 11:01: nw=1 d250 keeps phase A, saves ~0.19 ms of the late batch
# quality: the denser IFEval re-check, ALL 30 sequences (every 1: the full-set level, no subset sampling) on Breakthrough's real v3f decoder (independent of the build below)
if [ -z "$SKIP_Q" ]; then
for A in ctl dhw4; do
  if [ $A = ctl ]; then E="B12_W=10 B12_HC=0"; else E="B12_W=4 B12_LATE3=4,10 B12_HC=0"; fi
  env $E $PY $M/defer/tf_dense.py ${A}_c128 0.25 tf_ifeval 1 > "$OUT/tfdense_$A.txt" 2>&1
  log "tf_dense $A: $(grep -E '_c128 \[|Error|Traceback' "$OUT/tfdense_$A.txt" | tail -2 | cut -c1-300 | tr '\n' ' ')"
done
$PY $M/defer/tf_dense_cmp.py ctl_c128 dhw4_c128 e1o0 > "$OUT/tfdense_cmp.txt" 2>&1
log "tf_dense_cmp: $(tr '\n' ' ' < "$OUT/tfdense_cmp.txt" | cut -c1-700)"
else log "IFEval re-check skipped (done at 11:19)"; fi
$PY -m py_compile $M/defer/run.py $M/defer/v3d/bigmoe_v3d/tier.py $M/defer/v3d/bigmoe_v3d/stream.py $M/defer/v3d/tierglu.py \
  > "$OUT/pycompile.txt" 2>&1 || fin "STOP: py_compile failed ($(head -c 300 "$OUT/pycompile.txt"))"
for S in 4 8; do
  for A in v3 dh10 force ref dh force_sub; do
    if [ $A = force_sub ]; then E="DH_ARM=force BIGMOE_DH_SUB=1 DH_TAG=_sub"; else E="DH_ARM=$A"; fi
    env $E $PY $M/defer/run.py tf 32 $S > "$OUT/tf_${A}_S$S.txt" 2>&1
    log "tf $A S=$S: $(grep -E '^\[tf_|Error|Traceback' "$OUT/tf_${A}_S$S.txt" | tail -2 | cut -c1-200 | tr '\n' ' ')"
  done
done
$PY $M/defer/tf_compare.py > "$OUT/tf_compare.txt" 2>&1; ok=$?
log "tf_compare (exit $ok): $(tr '\n' ' ' < "$OUT/tf_compare.txt" | cut -c1-900)"
[ $ok -eq 0 ] || log "WARNING: a correctness check failed: speed numbers below are UNVERIFIED (see tf_compare.txt)"
for S in 4 8; do
  for A in v3 dh force dh_sub force_sub; do
    case $A in dh_sub) E="DH_ARM=dh BIGMOE_DH_SUB=1 DH_TAG=_sub";; force_sub) E="DH_ARM=force BIGMOE_DH_SUB=1 DH_TAG=_sub";; *) E="DH_ARM=$A";; esac
    env $E $PY $M/defer/run.py fixed $S 48 > "$OUT/fixed_${A}_S$S.txt" 2>&1
    log "fixed $A S=$S: $(grep -E '^\[fixed_|Error|Traceback' "$OUT/fixed_${A}_S$S.txt" | tail -2 | cut -c1-220 | tr '\n' ' ')"
  done
done
T0=$(date +%s)
PIPE_OK=0; [ "$PIPE_US" -ge 0 ] 2>/dev/null && ls $M/defer/v3d/ext/build_pipe/bigmoe_ext*.so > /dev/null 2>&1 && PIPE_OK=1
for R in 1 2; do
  if [ $R -eq 2 ] && [ $(( $(date +%s) - T0 )) -gt 1500 ]; then log "round 2 skipped (time)"; break; fi
  for P in code prose edit; do
    if [ $R -eq 1 ]; then ARMS="v3 dh d1"; [ $PIPE_OK -eq 1 ] && ARMS="v3 dh d1 pipe"; else ARMS="dh v3"; fi
    for A in $ARMS; do
      if [ $A = d1 ]; then
        env DH_ARM=dh BIGMOE_DH_D=1 $PY $M/defer/run.py gen $P 256 > "$OUT/gen_${P}_${A}_r$R.txt" 2>&1
        cp "$OUT/gen_${P}_dh_d1.json" "$OUT/gen_${P}_${A}_r$R.json" 2>/dev/null
      elif [ $A = pipe ]; then
        env DH_ARM=dh BIGMOE_EXT_BUILD=build_pipe BIGMOE_DEFER_PIPE_US=$PIPE_US BIGMOE_DEFER_PIPE_STEP_US=$PIPE_STEP DH_TAG=_pipe \
          $PY $M/defer/run.py gen $P 256 \
          > "$OUT/gen_${P}_${A}_r$R.txt" 2>&1
        cp "$OUT/gen_${P}_dh_pipe.json" "$OUT/gen_${P}_${A}_r$R.json" 2>/dev/null
      else
        DH_ARM=$A $PY $M/defer/run.py gen $P 256 > "$OUT/gen_${P}_${A}_r$R.txt" 2>&1
        cp "$OUT/gen_${P}_${A}.json" "$OUT/gen_${P}_${A}_r$R.json" 2>/dev/null
      fi
      log "gen $P $A r$R: $(grep -E '^\[(code|prose|edit)_|Error|Traceback' "$OUT/gen_${P}_${A}_r$R.txt" | tail -2 | cut -c1-260 | tr '\n' ' ')"
    done
  done
done
# JIT prefetch interplay: dh with JIT off, and v3 with JIT off (code)
for A in dh v3; do
  env DH_ARM=$A DH_NOJIT=1 $PY $M/defer/run.py gen code 256 > "$OUT/gen_code_${A}_nojit.txt" 2>&1
  log "nojit $A code: $(grep -E '^\[code_|Error|Traceback' "$OUT/gen_code_${A}_nojit.txt" | tail -2 | cut -c1-200 | tr '\n' ' ')"
done
# per-layer timeline (ext trace: kind 5 = phase-A release, kind 1 = phase-B release) for the tail / late-batch analysis
for A in dh d1; do
  if [ $A = d1 ]; then E="DH_ARM=dh BIGMOE_DH_D=1"; else E="DH_ARM=dh"; fi
  env $E DH_TRACE=1 $PY $M/defer/run.py gen code 256 > "$OUT/gen_code_${A}_trace.txt" 2>&1
  log "traced $A code: $(grep -E '^\[code_|Error|Traceback' "$OUT/gen_code_${A}_trace.txt" | tail -2 | cut -c1-200 | tr '\n' ' ')"
done
# speculative (lowest priority, skipped past 50 min): d1's quality gate on v3 tiers with the build that would ship, same
# scorer / sets / sampling as the gating run (4 sets every 8th); the v3d control also cross-checks Breakthrough's control
if [ $(( $(date +%s) - TSTART )) -lt 3000 ]; then
  for A in v3 d1; do
    if [ $A = d1 ]; then E="DH_ARM=dh BIGMOE_DH_D=1"; else E="DH_ARM=v3"; fi
    env $E $PY $M/defer/step_dense.py my_${A}_c128 tf_gsm,tf_code,tf_ifeval,tf_multi 8 > "$OUT/step_${A}.txt" 2>&1
    log "step_dense $A: $(grep -E '_c128 \[|Error|Traceback' "$OUT/step_${A}.txt" | tail -2 | cut -c1-320 | tr '\n' ' ')"
  done
  $PY $M/defer/tf_dense_cmp.py my_v3_c128 my_d1_c128 e8o0 > "$OUT/step_cmp.txt" 2>&1
  log "step cmp (d1 vs v3d control): $(tr '\n' ' ' < "$OUT/step_cmp.txt" | cut -c1-900)"
else
  log "d1 quality gate skipped (time)"
fi
fin "turn 1 finished"
