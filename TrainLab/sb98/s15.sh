#!/bin/bash
# WEEK98 S15 SPARSETAP (MTP Head), pre-registered in REPORT 'WEEK98 S15 SPARSETAP': tapmlp at tap 2 with the tap refreshed
# only every K depths (lever tapmlpk), K = 8 and 4, s30's full recipe; read vs s30. Waits for S14's runner (pid $1); runs
# only if S14's verdict is SURVIVES or PARTIAL.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4 TAPMLP_TAP=2
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
V=$(grep "^VERDICT S14" $O/s14_RESULT.txt 2>/dev/null)
log "S14: ${V:-no verdict}"
case "$V" in *SURVIVES*|*PARTIAL*) ;; *) log "S15 is moot (S14 not SURVIVES / PARTIAL)"; exit 0;; esac
[ -f ckpt/mhx_fk8.safetensors ] || mv ckpt/mhc2a_cac.safetensors ckpt/mhx_fk8.safetensors
[ -f ckpt/mhx_fk4.safetensors ] || mv ckpt/mhc2a_n1.safetensors ckpt/mhx_fk4.safetensors
: > $O/s15_RESULT.txt
for k in 8 4; do
  export TAPMLP_K=$k
  $S/run_paused.sh s15_fk$k $O/s15_fk${k}_train.log 45 $PY sb/train_multi.py --tag mhx --arms "fk$k:$L/tapmlpk" \
    --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 || { log "fk$k training failed"; exit 1; }
  grep -q "DONE mhx" $O/s15_fk${k}_train.log || { log "fk$k did not finish"; exit 1; }
  log "$(grep 'DONE mhx' $O/s15_fk${k}_train.log | tail -1)"
  $S/run_paused.sh s15_eval$k $O/s15_eval$k.log 25 $PY sb/evalx.py --runs "s30,mhx_fk$k@$L/tapmlpk" --sets mbpp,he \
    --nm 50 --nh 30 --depth 31 --mode chain --out $O/s15_k$k.json || { log "evalx failed"; exit 1; }
  $S/run_paused.sh s15_walk$k $O/s15_walk$k.log 25 $PY $S/cc_readout.py $O/s15_walk$k.json s30 "mhx_fk$k@$L/tapmlpk" \
    || { log "cc_readout failed"; exit 1; }
  for ST in mbpp he; do
    $PY sb/compare.py $O/s15_k$k.json "s30|$ST|chain" $O/s15_k$k.json "mhx_fk$k|$ST|chain" --band 23-30 \
      --json $O/s15_fk${k}_$ST.json >> $O/s15_compare.txt 2>&1
  done
  python3 -I -c "
import json
O, k = '$O', '$k'
w = json.load(open(f'{O}/s15_walk{k}.json'))['res']
ft2 = {ST: json.load(open(f'{O}/s14_ft2_{ST}.json'))['23-30']['diff'] for ST in ('mbpp', 'he')}
per = {}
for ST in ('mbpp', 'he'):
    x = json.load(open(f'{O}/s15_fk{k}_{ST}.json')); b, kp = x['23-30'], x['kept']; rel = kp['B'] / kp['A'] - 1
    per[ST] = (b['diff'], rel)
    wa, wb = w[f's30|{ST}']['0']['walk_strict_bands'], w[f'mhx_fk{k}|{ST}']['0']['walk_strict_bands']
    ws = ' '.join(f'{bd} {100 * wa[bd][0]:.1f}->{100 * wb[bd][0]:.1f} (n {wb[bd][1]})' for bd in wa)
    ratio = f'{b[\"diff\"] / ft2[ST]:.2f}x ft2' if ft2[ST] >= 0.005 else 'ft2 < +0.5 pt: not judged'
    print(f\"mhx_fk{k} (tap 2, refresh every {k}) vs s30 {ST}: 23-30 {100 * b['A']:.2f} -> {100 * b['B']:.2f} ({100 * b['diff']:+.2f} pt, CI {100 * b['ci95'][0]:+.2f}..{100 * b['ci95'][1]:+.2f}, p {b['p_sign']:.3g}; {ratio}); kept/32 {kp['A']:.3f} -> {kp['B']:.3f} ({100 * rel:+.1f} %) | walk-strict {ws}\")
judged = {s: ft2[s] >= 0.005 for s in per}
if all(judged.values()) and all(per[s][0] >= 0.5 * ft2[s] and per[s][1] >= 0.02 for s in per):
    v = 'KEEPS'
elif all(judged.values()) and all(per[s][0] < 0.25 * ft2[s] for s in per):
    v = 'LOSES IT'
else:
    v = 'PARTIAL'
print(f'VERDICT S15 K {k}: {v}')
" >> $O/s15_RESULT.txt
done
cat $O/s15_RESULT.txt
