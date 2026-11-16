#!/bin/bash
# WEEK98 S13 QATSCREEN (MTP Head), pre-registered in REPORT 'WEEK98 S13 QATSCREEN': QAT 4-bit on the sandbox head (lever
# qat4) at Sandbox98's screen recipe (s30 + 300 steps on pool, const lr after a 20-step warm-up, seed 1115), lr 2e-6
# and 1e-5, vs the cached bf16 controls; post-hoc Q4 = s30@qat4. Read-out: sb98/tree8.py (chain-32 / tree-16 / tree-32).
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_q4a.safetensors ] || mv ckpt/mhx_f30.safetensors ckpt/mhx_q4a.safetensors
[ -f ckpt/mhx_q4b.safetensors ] || mv ckpt/mhx_fca.safetensors ckpt/mhx_q4b.safetensors
for spec in q4a:2e-6 q4b:1e-5; do a=${spec%%:*}; lr=${spec#*:}
  log "train $a: QAT 4-bit, s30 + 300 steps, lr $lr"
  $S/run_paused.sh s13_$a $O/s13_${a}_train.log 20 $PY sb/train.py --run mhx_$a --start s30 --files pool --steps 300 \
    --const --lr $lr --seed 1115 --epochs 1 --lever $L/qat4 || { log "$a training failed"; exit 1; }
  log "$(tail -1 $O/s13_${a}_train.log | cut -c1-160)"
done
PAIRS="s30@qat4:s30,mhx_q4a:ctl_s30_300_lr2e-6,mhx_q4b:ctl_s30_300_lr1e-5,mhx_q4a:s30,mhx_q4b:s30" \
  $S/run_paused.sh s13_read $O/s13_read.log 45 $PY $S/tree8.py $O/s13.json s30 "s30@$L/qat4" ctl_s30_300_lr2e-6 \
  "mhx_q4a@$L/qat4" ctl_s30_300_lr1e-5 "mhx_q4b@$L/qat4" || { log "read-out failed"; exit 1; }
python3 -I -c "
import json
c = json.load(open('$O/s13.json'))['cmp']
g = lambda run, ctl, st, k: c[f'{run}@{ctl}|{st}|{k}']['ratio']
ph = {st: g('s30@qat4', 's30', st, 'K32|g1.0') for st in ('mbpp', 'he')}
inf = any(v <= 0.98 for v in ph.values())
print('post-hoc Q4 / s30 at tree-32: ' + ', '.join(f'{st} x{v:.4f}' for st, v in ph.items()) + ('' if inf else ' -> UNINFORMATIVE (no 4-bit loss on the 2B head)'))
for run, ctl in (('mhx_q4a', 'ctl_s30_300_lr2e-6'), ('mhx_q4b', 'ctl_s30_300_lr1e-5')):
    r = {(st, k): g(run, ctl, st, k) for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0', 'chain')}
    ok = all(r[(st, k)] >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
    s = {(st, k): g(run, 's30', st, k) for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0')}
    print(f'VERDICT {run} vs {ctl}: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{r[(st, \"K16|g1.0\")]:.4f} t32 x{r[(st, \"K32|g1.0\")]:.4f} chain x{r[(st, \"chain\")]:.4f}' for st in ('mbpp', 'he'))
          + ' | vs s30: ' + '; '.join(f'{st} t16 x{s[(st, \"K16|g1.0\")]:.4f} t32 x{s[(st, \"K32|g1.0\")]:.4f}' for st in ('mbpp', 'he')))
" > $O/s13_RESULT.txt
grep -E "chain-32|SELF-CHECK" $O/s13_read.log >> $O/s13_RESULT.txt
cat $O/s13_RESULT.txt
