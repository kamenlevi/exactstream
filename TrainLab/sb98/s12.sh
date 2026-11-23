#!/bin/bash
# WEEK98 S12 FULLSTATE (MTP Head; NewIdeas' ITEM 5), pre-registered in REPORT 'WEEK98 S12 FULLSTATE': carryac and tapmlp
# at s30's full budget, each one arm per process (<= 4 GB) on s30's recipe (native head, fresh Adam, base files, seed
# 1114, depth 30, 1,889 steps, cosine), plus f30 = the same recipe without a lever (the same-code control).
# Paused-safe (run_paused.sh: NO-SWAP >= 12 GB to start a stage, no quiet lock of others, kill below 6 GB).
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
# disk flat: the PASS2 scratch files (read-outs committed in ee806ed + the p2b addendum) become S12's
[ -f ckpt/mhx_fca.safetensors ] || mv ckpt/mhx_p2f_s680.safetensors ckpt/mhx_fca.safetensors
[ -f ckpt/mhx_ftm.safetensors ] || mv ckpt/mhx_p2b.safetensors ckpt/mhx_ftm.safetensors
[ -f ckpt/mhx_f30.safetensors ] || mv ckpt/mhx_p2f.safetensors ckpt/mhx_f30.safetensors
train() {   # $1 = arm, $2 = lever ('-' = none), $3 = lock minutes
  log "train $1 (lever $2): s30's recipe, 1,889 steps from native"
  $S/run_paused.sh s12_$1 $O/s12_$1_train.log $3 $PY sb/train_multi.py --tag mhx --arms "$1:$2" --start native \
    --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 || { log "$1 training failed"; exit 1; }
  grep -q "DONE mhx" $O/s12_$1_train.log || { log "$1 did not finish"; exit 1; }
  log "$(grep 'DONE mhx' $O/s12_$1_train.log | tail -1)"
}
train fca $L/carryac 45
train ftm tapmlp 45
train f30 - 35
$PY -I -c "
import mlx.core as mx
a, b = mx.load('ckpt/s30.safetensors'), mx.load('ckpt/mhx_f30.safetensors')
same = sorted(a) == sorted(b) and all(mx.array_equal(a[k], b[k]).item() for k in a)
diff = max((mx.abs(a[k].astype(mx.float32) - b[k].astype(mx.float32)).max().item() for k in a if k in b), default=-1)
print(f'f30 vs s30 weights: {\"BITWISE EQUAL\" if same else \"DIFFERENT\"} (keys {len(a)} / {len(b)}, max abs diff {diff:.3g})')
" > $O/s12_RESULT.txt 2>&1
SP="s30 mhx_f30 mhx_fca@$L/carryac mhx_ftm@tapmlp"
$S/run_paused.sh s12_eval $O/s12_eval.log 40 $PY sb/evalx.py --runs "${SP// /,}" --sets mbpp,he --nm 50 --nh 30 \
  --depth 31 --mode chain --out $O/s12.json || { log "evalx failed"; exit 1; }
PAIRS="mhx_fca:mhx_f30,mhx_ftm:mhx_f30,mhx_f30:s30" $S/run_paused.sh s12_walk $O/s12_walk.log 40 $PY $S/cc_readout.py \
  $O/s12_walk.json $SP || { log "cc_readout failed"; exit 1; }
for a in mhx_fca mhx_ftm; do for c in mhx_f30 s30; do for ST in mbpp he; do
  $PY sb/compare.py $O/s12.json "$c|$ST|chain" $O/s12.json "$a|$ST|chain" --band 23-30 --json $O/s12_${a}_${c}_$ST.json \
    >> $O/s12_compare.txt 2>&1
done; done
python3 -I -c "
import json
O = '$O'; a = '$a'
w = json.load(open(f'{O}/s12_walk.json'))['res']
calls = {}
for c in ('mhx_f30', 's30'):
    per = {}
    for ST in ('mbpp', 'he'):
        x = json.load(open(f'{O}/s12_{a}_{c}_{ST}.json')); b, k = x['23-30'], x['kept']
        per[ST] = (b['diff'], b['p_sign'], k['B'] / k['A'] - 1)
        wa, wb = w[f'{c}|{ST}']['0']['walk_strict_bands'], w[f'{a}|{ST}']['0']['walk_strict_bands']
        ws = ' '.join(f'{bd} {100 * wa[bd][0]:.1f}->{100 * wb[bd][0]:.1f} (n {wb[bd][1]})' for bd in wa)
        print(f\"{a} vs {c} {ST}: 23-30 {100 * b['A']:.2f} -> {100 * b['B']:.2f} ({100 * b['diff']:+.2f} pt, CI {100 * b['ci95'][0]:+.2f}..{100 * b['ci95'][1]:+.2f}, p {b['p_sign']:.3g}); kept/32 {k['A']:.3f} -> {k['B']:.3f} ({100 * per[ST][2]:+.1f} %) | walk-strict {ws}\")
    if all(d >= 0.020 and r >= 0.02 for d, _, r in per.values()):
        calls[c] = 'PROMISING'
    elif any(d <= -0.020 and p < 0.05 for d, p, _ in per.values()):
        calls[c] = 'HURTS'
    else:
        calls[c] = 'NO EFFECT'
print(f\"VERDICT {a}: {calls['mhx_f30']} vs f30 (the line); {calls['s30']} vs s30 (beside)\")
" >> $O/s12_RESULT.txt
done
cat $O/s12_RESULT.txt
