#!/bin/sh
# Breakthrough Wed 2026-10-21: dense diet for the RAM floor (TASKS.md, NewIdeas): floor at C=16 per step + teacher-
# forced KLD (exact routing, both subsets) of HC mixers 4-bit / 8-bit. Waits for day_bands.sh (same CLAUDE_PID).
B=$HOME/Breakthrough; OUT=$B/data/day; LOG=$B/data/expS/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=10800
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
step() {   # step NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^\[|^bt_|^125B|^[A-Za-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
  return $rc
}
while pgrep -f "scripts/day_(speed[34]|bands).sh" > /dev/null; do sleep 20; done   # earlier runners (same CLAUDE_PID)
peer lock quiet --queue 60m "Breakthrough: frugal speed levers (late rank 1, precise prefetch) + RAM floors + 5% baseline (~50 min)" > /dev/null 2>&1 || { log "no lock (diet)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (diet)"
# 17:50 (dd in place, lines 1-33 unchanged): timing first, then the RAM floors, then the quality screen of the late-rank
# levers (tf_frugal_var lAtB: frugal 'base' vs base_l1t2 = a cold rank 1 half a layer late); a second turn re-queues for
# whatever did not fit + the 5 % screen (base vs base_l2t4 at 0.05) + HC 4-bit KLD + spare timings.
# NewIdeas 17:40: precise look-ahead prefetch of ranks 0-1 is the main latency lever (a half-layer window is ~0.2-0.3 ms
# in the new engine vs 0.30 ms per cold read): first, with MoEPrefetch's corrected d=1 predictor (top-10 agreement 0.62->0.75)
CORR=$HOME/MoEPrefetch/data/corr125_r128.npz
for PR in code prose; do
  step pf2c_$PR 400 env PROMPT=$PR BT_PF=1 BT_SPARE=4 BIGMOE_PREFETCH=2 BIGMOE_PF_CORR=$CORR $PY $S/bt_frugal_spec.py 0.10 speed 3:3:0
  step pf4c_$PR 400 env PROMPT=$PR BT_PF=1 BT_SPARE=6 BIGMOE_PREFETCH=4 BIGMOE_PF_CORR=$CORR $PY $S/bt_frugal_spec.py 0.10 speed 3:3:0
done
step pf2_code 400 env PROMPT=code BT_PF=1 BT_SPARE=4 BIGMOE_PREFETCH=2 $PY $S/bt_frugal_spec.py 0.10 speed 3:3:0
if ls $B/fork125bd/ext/build/bigmoe_ext*.so > /dev/null 2>&1; then
  step df_check10 300 env BT_K=10 $PY $S/bt_defer.py 0.10 check
  step df_checkf 300 env BT_FRUGAL=1 $PY $S/bt_defer.py 0.10 check
  step df_checkf1 300 env BT_FRUGAL=1 BT_LATE1=1 $PY $S/bt_defer.py 0.10 check
  if grep -q "^check" $OUT/df_checkf1.txt; then
    for PR in code prose; do
      step df_f_$PR 300 env PROMPT=$PR BT_FRUGAL=1 $PY $S/bt_defer.py 0.10 speed
      step df_f1_$PR 300 env PROMPT=$PR BT_FRUGAL=1 BT_LATE1=1 $PY $S/bt_defer.py 0.10 speed
    done
  fi
fi
step floor_base 300 env $PY $S/floor_diet.py
step floor_hot0 300 env MTPLX_NGRAM_HOT_MB=0 $PY $S/floor_diet.py
step floor_hot256 300 env MTPLX_NGRAM_HOT_MB=256 $PY $S/floor_diet.py
step floor_hc8 300 env $PY $S/floor_diet.py hc8
step floor_hc4 300 env $PY $S/floor_diet.py hc4
step floor_all 300 env MTPLX_NGRAM_HOT_MB=0 $PY $S/floor_diet.py hc4,lm3
kld() {   # kld VAR FRAC: skip a variant whose file exists (tf_frugal_var tags non-10 % runs _rNN)
  rt=""; [ "$2" = "0.10" ] || rt=_r$(printf %02d $(echo "$2" | awk '{print int($1 * 100 + 0.5)}'))
  [ -f $B/data/tfvar/$1${rt}_e8.npz ] && return 0
  step tfvar8_$1$rt 1200 env BIGMOE_MAXK=80 $PY $S/tf_frugal_var.py $1 $2 tf_ifeval,tf_multi 8
}
for v in base base_l1t2 base_Sgc base_l1t2_Sgc; do kld $v 0.10; done
peer unlock > /dev/null 2>&1; log "lock released (diet)"
peer lock quiet --queue 60m "Breakthrough: 5% late-rank screen + HC 4-bit KLD + spare timings (~45 min)" > /dev/null 2>&1 || { log "no lock (diet2)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (diet2)"
# 19:10 (dd in place): build v3f (CPU), then TrainLab's pre-routed frugal (pre2: the waited top-2 of layers 8-39 picked a
# layer early by the corrected d=1 look-ahead, the true router's weights) vs frugal on both subsets at 10 % and 5 %
# (NewIdeas: before the B12 speed runs), then B12 checks / speed / GSM8K-20, then the 5 % late-rank screens and HC 4-bit.
if ! ls $B/v3f/ext/build/bigmoe_ext*.so > /dev/null 2>&1; then
  log "building v3f"
  /opt/homebrew/bin/cmake -S $B/v3f/ext -B $B/v3f/ext/build -DCMAKE_BUILD_TYPE=Release -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake \
    -DPython_EXECUTABLE=$PY > $B/v3f/ext/build_cmake.log 2>&1 && /opt/homebrew/bin/cmake --build $B/v3f/ext/build -j 6 > $B/v3f/ext/build.log 2>&1
fi
kld base_pre2 0.10; kld base_pre2_Sgc 0.10
kld base 0.05; kld base_pre2 0.05; kld base_Sgc 0.05; kld base_pre2_Sgc 0.05
pre_ok() { $PY -c '
import numpy as np, sys
r = []
for a, b in (("base_pre2", "base"), ("base_pre2_Sgc", "base_Sgc")):
    za, zb = np.load(f"{sys.argv[1]}/{a}_e8.npz"), np.load(f"{sys.argv[1]}/{b}_e8.npz")
    k = lambda z: np.concatenate([z[f] for f in z.files if f.startswith("kld_")]).mean()
    r.append(k(za) / k(zb))
print("pre2 / frugal KLD", [round(v, 3) for v in r]); sys.exit(0 if max(r) <= 1.05 else 1)' $B/data/tfvar >> $LOG 2>&1; }
if ls $B/v3f/ext/build/bigmoe_ext*.so > /dev/null 2>&1; then
  step b12_check_plain 300 env B12_HC=0 B12_LATE1=0 $PY $S/bt_b12.py 0.10 check
  step b12_check 300 env $PY $S/bt_b12.py 0.10 check
  step b12_check_pre 300 env B12_PRE=1 $PY $S/bt_b12.py 0.10 check
  if grep -q "^check" $OUT/b12_check.txt; then
    for PR in code prose; do
      step b12_$PR 300 env PROMPT=$PR $PY $S/bt_b12.py 0.10 speed
      step b12_nolate_$PR 300 env PROMPT=$PR B12_LATE1=0 $PY $S/bt_b12.py 0.10 speed
      grep -q "^check" $OUT/b12_check_pre.txt && step b12_pre_$PR 300 env PROMPT=$PR B12_PRE=1 $PY $S/bt_b12.py 0.10 speed
    done
    step b12_gsm20 900 env $PY $S/bt_b12.py 0.10 gsm20
    if grep -q "^check" $OUT/b12_check_pre.txt && pre_ok; then   # NewIdeas: GSM8K-50 decode before pre-routing replaces late rank 1
      step b12_pre_gsm50 1500 env B12_PRE=1 $PY $S/bt_b12.py 0.10 gsm50
    fi
  fi
else
  log "v3f build FAILED (v3f/ext/build.log)"
fi
kld base_l2t4 0.05; kld w3 0.05   # 5 % margin candidates (NewIdeas' 35B: w3 0.125, cold-in-50 %-late 0.138 < frugal-10 % 0.163)
for v in exact_Sgc exact_hc4 exact_hc4_Sgc; do kld $v 0.10; done
peer unlock > /dev/null 2>&1; log "lock released (diet2)"
peer lock quiet --queue 45m "Breakthrough: B12 checks + speed (fixed HC 4-bit path) + leftover KLD (~40 min)" > /dev/null 2>&1 || { log "no lock (diet3)"; exit 1; }
log "lock taken (diet3)"
# 19:55 (dd in place): turn 2's B12 checks with HC 4-bit failed to build Metal kernels (BigMoE's compiled HC read was built
# for bf16 mixers; hc8.py: never combine): bt_b12 now turns the HC fusions off when B12_HC != 0. Plain B12 checked fine.
# pre2 failed its screen (+26 % hard / +11 % easy KLD vs frugal): no B12_PRE runs.
step b12_check_late 300 env B12_HC=0 $PY $S/bt_b12.py 0.10 check
step b12_check 300 env $PY $S/bt_b12.py 0.10 check
if grep -q "^check" $OUT/b12_check.txt; then
  for PR in code prose; do
    step b12_$PR 300 env PROMPT=$PR $PY $S/bt_b12.py 0.10 speed
    step b12_nolate_$PR 300 env PROMPT=$PR B12_LATE1=0 $PY $S/bt_b12.py 0.10 speed
    step b12_plain_$PR 300 env PROMPT=$PR B12_HC=0 B12_LATE1=0 $PY $S/bt_b12.py 0.10 speed
  done
  step b12_gsm20 900 env $PY $S/bt_b12.py 0.10 gsm20
fi
kld base_pre2 0.10; kld base_pre2_Sgc 0.10; kld base 0.05; kld base_pre2 0.05; kld base_Sgc 0.05; kld base_pre2_Sgc 0.05
kld base_l2t4 0.05; kld w3 0.05
for v in exact_Sgc exact_hc4 exact_hc4_Sgc; do kld $v 0.10; done
peer unlock > /dev/null 2>&1; log "lock released (diet3)"
exit 0
