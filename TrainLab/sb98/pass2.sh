#!/bin/bash
# WEEK98 S10 PASS2 (MTP Head): the sandbox pre-screen of a second training pass (t30c's analog), pre-registered in REPORT
# 'WEEK98 S10 PASS2' (de5852f) + amendment 1 (017ce44).
# Stage 1:
#   p2f = s30 + one pass of FRESH data (pool2, 2,570 steps, fresh Adam, cosine), judged vs s30 by NewIdeas' line (per
#   set: 23-30 >= +1.0 pt AND kept/32 >= +2 %); --save-at 680 keeps p2f_s680 (mid-cosine, info only);
#   p2s = s30 + the same first 680 pool2 slices with its OWN cosine (info only: the analog of a ~1,500-step t30c).
# Stage 2 (P2B=1, info only): p2b = s30 + a second pass over its own base files (1,889 steps).
# One arm per process (~3.4 GB peak). Paused-safe (run_paused.sh). Disk flat: reused scratch files (see the amendment).
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_p2f.safetensors ] || mv ckpt/mhx_mg1.safetensors ckpt/mhx_p2f.safetensors
[ -f ckpt/mhx_p2s.safetensors ] || mv ckpt/mhx_mg1k.safetensors ckpt/mhx_p2s.safetensors
[ -f ckpt/mhx_p2f_s680.safetensors ] || mv ckpt/mhx_cs1.safetensors ckpt/mhx_p2f_s680.safetensors

readout() {   # $1 = tag, $2.. = runs (the first is the control s30)
  local tag=$1; shift; local runs="$*"
  $S/run_paused.sh ${tag}_eval $O/${tag}_eval.log 30 $PY sb/evalx.py --runs "${runs// /,}" --sets mbpp,he --nm 50 --nh 30 \
    --depth 31 --mode chain --out $O/${tag}.json || { log "evalx failed"; exit 1; }
  $S/run_paused.sh ${tag}_walk $O/${tag}_walk.log 30 $PY $S/cc_readout.py $O/${tag}_walk.json $runs \
    || { log "cc_readout failed"; exit 1; }
  for a in ${runs#s30 }; do for ST in mbpp he; do
    $PY sb/compare.py $O/${tag}.json "s30|$ST|chain" $O/${tag}.json "$a|$ST|chain" --band 23-30 --json $O/${tag}_${a}_$ST.json \
      >> $O/${tag}_compare.txt 2>&1
    python3 -I -c "
import json
c = json.load(open('$O/${tag}_${a}_$ST.json')); w = json.load(open('$O/${tag}_walk.json'))['res']
b, k = c['23-30'], c['kept']
rel = k['B'] / k['A'] - 1
ok = b['diff'] >= 0.010 and rel >= 0.02
wa, wb = w['s30|$ST']['0']['walk_strict_bands'], w['$a|$ST']['0']['walk_strict_bands']
ws = ' '.join(f\"{band} {100 * wa[band][0]:.1f}->{100 * wb[band][0]:.1f} (n {wb[band][1]})\" for band in wa)
tag = '' if '$a' == 'mhx_p2f' else ' [info only]'
print(f\"$a vs s30 $ST{tag}: 23-30 {100 * b['A']:.2f} -> {100 * b['B']:.2f} ({100 * b['diff']:+.2f} pt, CI {100 * b['ci95'][0]:+.2f}..{100 * b['ci95'][1]:+.2f}, p {b['p_sign']:.3g}); kept/32 {k['A']:.3f} -> {k['B']:.3f} ({100 * rel:+.1f} %) -> {'PASS' if ok else 'FAIL'} | walk-strict {ws}\")
" >> $O/pass2_RESULT.txt
  done; done
}

: > $O/pass2_RESULT.txt
log "stage 1: train p2f (s30 + one pool2 pass, 2,570 steps, copy at 680)"
$S/run_paused.sh pass2_p2f $O/pass2_p2f_train.log 90 $PY sb/train_multi.py --tag mhx --arms "p2f:-" --start s30 \
  --resume-opt "" --files pool2 --seed 1116 --depth 30 --steps 2570 --save-at 680 || { log "p2f training failed"; exit 1; }
grep -q "DONE mhx" $O/pass2_p2f_train.log || { log "p2f training did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/pass2_p2f_train.log | tail -1)"
log "stage 1: train p2s (s30 + the same first 680 pool2 slices, own cosine)"
$S/run_paused.sh pass2_p2s $O/pass2_p2s_train.log 30 $PY sb/train_multi.py --tag mhx --arms "p2s:-" --start s30 \
  --resume-opt "" --files pool2 --seed 1116 --depth 30 --steps 680 || { log "p2s training failed"; exit 1; }
grep -q "DONE mhx" $O/pass2_p2s_train.log || { log "p2s training did not finish"; exit 1; }
log "$(grep 'DONE mhx' $O/pass2_p2s_train.log | tail -1)"
log "stage 1 read-outs"
readout pass2 s30 mhx_p2f mhx_p2f_s680 mhx_p2s
python3 -I -c "
r = open('$O/pass2_RESULT.txt').read().splitlines()
f = [l for l in r if l.startswith('mhx_p2f vs s30')]
both = len(f) == 2 and all('-> PASS' in l for l in f)
he = any(l.startswith('mhx_p2f vs s30 he') and '-> PASS' in l for l in f)
print('VERDICT p2f: ' + ('PASS on both sets: t30c gets Tuesday night' if both else ('HE PASS only (MBPP fails)' if he else 'FAIL: t30c does not get Tuesday night')))
" >> $O/pass2_RESULT.txt
cat $O/pass2_RESULT.txt
if [ "${P2B:-0}" = 1 ]; then
  log "stage 2: train p2b (s30 + a second pass of base, 1,889 steps; reuses p2s's file)"
  mv ckpt/mhx_p2s.safetensors ckpt/mhx_p2b.safetensors
  $S/run_paused.sh pass2_p2b $O/pass2_p2b_train.log 45 $PY sb/train_multi.py --tag mhx --arms "p2b:-" --start s30 \
    --resume-opt "" --files base --seed 1115 --depth 30 --steps 1889 || { log "p2b training failed"; exit 1; }
  grep -q "DONE mhx" $O/pass2_p2b_train.log || { log "p2b training did not finish"; exit 1; }
  readout pass2b s30 mhx_p2b
  tail -2 $O/pass2_RESULT.txt
fi
