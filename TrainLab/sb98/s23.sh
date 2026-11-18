#!/bin/bash
# WEEK98 S23 SMOE (MTP Head), pre-registered in REPORT 'WEEK98 S23 SMOE': the head's MLP as an upcycled fine-grained MoE
# (lever smoe), E 16 / K 4 and E 32 / K 8, s30's full recipe from native; read-out sb98/tree8.py vs s30.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_sm16.safetensors ] || mv ckpt/mhx_h2.safetensors ckpt/mhx_sm16.safetensors
[ -f ckpt/mhx_sm32.safetensors ] || mv ckpt/mhx_h3.safetensors ckpt/mhx_sm32.safetensors
: > $O/s23_RESULT.txt
for spec in sm16:16:4 sm32:32:8; do IFS=: read a e k <<< "$spec"
  export SMOE_E=$e SMOE_K=$k
  for i in 1 2 3; do
    $S/run_paused.sh s23_$a $O/s23_${a}_train.log 60 $PY sb/train_multi.py --tag mhx --arms "$a:$L/smoe" --start native \
      --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 && grep -q "DONE mhx" $O/s23_${a}_train.log && break
    log "$a try $i failed"; sleep 120
  done
  log "$(grep 'DONE mhx' $O/s23_${a}_train.log | tail -1)"
  for i in 1 2 3; do
    PAIRS="mhx_$a:s30" $S/run_paused.sh s23_read_$a $O/s23_read_$a.log 30 $PY $S/tree8.py $O/s23_$a.json s30 "mhx_$a@$L/smoe" && break
    sleep 120
  done
  $PY -I -B -c "
import sys, os, glob, numpy as np; sys.path.insert(0, os.path.expanduser('~/Sandbox98'))
import mlx.core as mx
from sb import core as C
from sb.train import load_lever
D = C.load_drafter('mhx_$a', strict=False); L = load_lever('$L/smoe'); L.setup(D, None)
D.load_weights(os.path.join(C.CKPT, 'mhx_$a.safetensors'), strict=False)
f = sorted(g for g in glob.glob(os.path.join(C.ROOT, 'data', 'he', '*.npz')) if not g.endswith('.taps.npz'))[0]
W = C.load_W(np.load(f)); x = mx.array(W[:, :256]).astype(mx.bfloat16) if W.ndim == 3 else mx.array(W[None, :256]).astype(mx.bfloat16)
u = L.usage(D, D.input_layernorm(x)); print('expert usage (share of tokens per expert, HE window 0 trunk rows):', [round(v, 3) for v in u.tolist()])
" >> $O/s23_RESULT.txt 2>&1
  python3 -I -c "
import json
c = json.load(open('$O/s23_$a.json'))['cmp']; g = lambda st, kk: c[f'mhx_$a@s30|{st}|{kk}']['ratio']
ok = all(g(st, 'K32|g1.0') >= 0.97 for st in ('mbpp', 'he'))
print('VERDICT $a (E $e, K $k) vs s30: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(st, \"K16|g1.0\"):.4f} t32 x{g(st, \"K32|g1.0\"):.4f} chain x{g(st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
" >> $O/s23_RESULT.txt
done
cat $O/s23_RESULT.txt
