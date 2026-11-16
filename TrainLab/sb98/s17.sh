#!/bin/bash
# WEEK98 S17 BITS (MTP Head), pre-registered in REPORT 'WEEK98 S17 BITS': post-hoc 8 / 6-bit on s30 and 6-bit QAT from native.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
rd() {   # $1 = bits, $2 = out tag, $3.. = specs (PAIRS from the env)
  local b=$1 t=$2; shift 2
  for i in 1 2 3; do QAT_BITS=$b $S/run_paused.sh s17_$t $O/s17_$t.log 30 $PY $S/tree8.py $O/s17_$t.json "$@" && return 0; sleep 120; done
  log "read-out $t failed"; exit 1
}
PAIRS="s30@qat4:s30" rd 8 p8 s30 "s30@$L/qat4"
[ -f ckpt/mhx_qn6.safetensors ] || mv ckpt/mhd1_n1_s380.safetensors ckpt/mhx_qn6.safetensors
for i in 1 2 3; do
  QAT_BITS=6 $S/run_paused.sh s17_qn6 $O/s17_qn6_train.log 45 $PY sb/train_multi.py --tag mhx --arms "qn6:$L/qat4" \
    --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 && grep -q "DONE mhx" $O/s17_qn6_train.log && break
  log "qn6 try $i failed"; sleep 120
done
PAIRS="s30@qat4:s30,mhx_qn6:s30" rd 6 b6 s30 "s30@$L/qat4" "mhx_qn6@$L/qat4"
python3 -I -c "
import json
O = '$O'
for f, runs in (('s17_p8', (('s30@qat4', 'p8 post-hoc 8-bit'),)), ('s17_b6', (('s30@qat4', 'p6 post-hoc 6-bit'), ('mhx_qn6', 'qn6 6-bit QAT from native')))):
    c = json.load(open(f'{O}/{f}.json'))['cmp']
    for r, lab in runs:
        g = lambda st, k: c[f'{r}@s30|{st}|{k}']['ratio']
        ok = all(g(st, k) >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
        print(f'VERDICT {lab} vs s30: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(st, \"K16|g1.0\"):.4f} t32 x{g(st, \"K32|g1.0\"):.4f} chain x{g(st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
" > $O/s17_RESULT.txt
cat $O/s17_RESULT.txt
