#!/bin/bash
# WEEK98 S14 SHALLOWTAP (MTP Head), pre-registered in REPORT 'WEEK98 S14 SHALLOWTAP': S12's tapmlp arm at tap 2 (layers
# 0-2 of 24), s30's full recipe; read vs s30. Waits for S13's runner (pid $1) first: one job of mine at a time.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4 TAPMLP_TAP=2
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
log "S13 runner gone; S14 starts"
[ -f ckpt/mhx_ft2.safetensors ] || mv ckpt/mhc2a_ca.safetensors ckpt/mhx_ft2.safetensors
$S/run_paused.sh s14_ft2 $O/s14_ft2_train.log 45 $PY sb/train_multi.py --tag mhx --arms "ft2:tapmlp" --start native \
  --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 || { log "ft2 training failed"; exit 1; }
grep -q "DONE mhx" $O/s14_ft2_train.log || { log "ft2 did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/s14_ft2_train.log | tail -1)"
$S/run_paused.sh s14_eval $O/s14_eval.log 25 $PY sb/evalx.py --runs "s30,mhx_ft2@tapmlp" --sets mbpp,he --nm 50 --nh 30 \
  --depth 31 --mode chain --out $O/s14.json || { log "evalx failed"; exit 1; }
$S/run_paused.sh s14_walk $O/s14_walk.log 25 $PY $S/cc_readout.py $O/s14_walk.json s30 "mhx_ft2@tapmlp" \
  || { log "cc_readout failed"; exit 1; }
: > $O/s14_RESULT.txt
for ST in mbpp he; do
  $PY sb/compare.py $O/s14.json "s30|$ST|chain" $O/s14.json "mhx_ft2|$ST|chain" --band 23-30 --json $O/s14_ft2_$ST.json \
    >> $O/s14_compare.txt 2>&1
done
python3 -I -c "
import json
O = '$O'; w = json.load(open(f'{O}/s14_walk.json'))['res']; ref = {'mbpp': 0.0478, 'he': 0.0455}
per = {}
for ST in ('mbpp', 'he'):
    x = json.load(open(f'{O}/s14_ft2_{ST}.json')); b, k = x['23-30'], x['kept']; rel = k['B'] / k['A'] - 1
    per[ST] = (b['diff'], rel)
    wa, wb = w[f's30|{ST}']['0']['walk_strict_bands'], w[f'mhx_ft2|{ST}']['0']['walk_strict_bands']
    ws = ' '.join(f'{bd} {100 * wa[bd][0]:.1f}->{100 * wb[bd][0]:.1f} (n {wb[bd][1]})' for bd in wa)
    print(f\"mhx_ft2 (tap 2) vs s30 {ST}: 23-30 {100 * b['A']:.2f} -> {100 * b['B']:.2f} ({100 * b['diff']:+.2f} pt, CI {100 * b['ci95'][0]:+.2f}..{100 * b['ci95'][1]:+.2f}, p {b['p_sign']:.3g}; {b['diff'] / ref[ST]:.2f}x tap 12's gain); kept/32 {k['A']:.3f} -> {k['B']:.3f} ({100 * rel:+.1f} %) | walk-strict {ws}\")
if all(per[s][0] >= 0.5 * ref[s] and per[s][1] >= 0.02 for s in per):
    v = 'SURVIVES (price it)'
elif all(per[s][0] < 0.25 * ref[s] for s in per):
    v = 'COLLAPSES (tapmlp closes)'
else:
    v = 'PARTIAL'
print('VERDICT S14 tap 2: ' + v)
" >> $O/s14_RESULT.txt
cat $O/s14_RESULT.txt
