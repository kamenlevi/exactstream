#!/bin/bash
# WEEK98 S16 QATCOS (MTP Head), pre-registered in REPORT 'WEEK98 S16 QATCOS': QAT with an anneal. c16 = p2s's recipe (bf16),
# q16 = c16 + qat4, qn = s30's full recipe + qat4; read-out sb98/tree8.py.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_c16.safetensors ] || mv ckpt/mhx_q4a.safetensors ckpt/mhx_c16.safetensors
[ -f ckpt/mhx_q16.safetensors ] || mv ckpt/mhx_q4b.safetensors ckpt/mhx_q16.safetensors
[ -f ckpt/mhx_qn.safetensors ] || mv ckpt/mhd1_n1.safetensors ckpt/mhx_qn.safetensors
tr() {   # $1 = arm, $2 = lever, $3.. = recipe
  local a=$1 lv=$2; shift 2
  for i in 1 2 3; do
    $S/run_paused.sh s16_$a $O/s16_${a}_train.log 40 $PY sb/train_multi.py --tag mhx --arms "$a:$lv" "$@" && \
      grep -q "DONE mhx" $O/s16_${a}_train.log && { log "$(grep 'DONE mhx' $O/s16_${a}_train.log | tail -1)"; return 0; }
    log "$a try $i failed"; sleep 120
  done
  exit 1
}
tr c16 - --start s30 --resume-opt "" --files pool2 --seed 1116 --depth 30 --steps 680
tr q16 $L/qat4 --start s30 --resume-opt "" --files pool2 --seed 1116 --depth 30 --steps 680
tr qn $L/qat4 --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889
for i in 1 2 3; do
  PAIRS="mhx_q16:mhx_c16,mhx_q16:s30,mhx_c16:s30,mhx_qn:s30" $S/run_paused.sh s16_read $O/s16_read.log 45 $PY $S/tree8.py \
    $O/s16.json s30 mhx_c16 "mhx_q16@$L/qat4" "mhx_qn@$L/qat4" && break
  log "read-out try $i failed"; sleep 120
done
python3 -I -c "
import json
c = json.load(open('$O/s16.json'))['cmp']
g = lambda r, b, st, k: c[f'{r}@{b}|{st}|{k}']['ratio']
for run, ctl in (('mhx_q16', 'mhx_c16'), ('mhx_qn', 's30')):
    ok = all(g(run, ctl, st, k) >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
    print(f'VERDICT {run} vs {ctl}: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(run, ctl, st, \"K16|g1.0\"):.4f} t32 x{g(run, ctl, st, \"K32|g1.0\"):.4f} chain x{g(run, ctl, st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
print('c16 vs s30 (the anneal control): ' + '; '.join(f'{st} t16 x{g(\"mhx_c16\", \"s30\", st, \"K16|g1.0\"):.4f} t32 x{g(\"mhx_c16\", \"s30\", st, \"K32|g1.0\"):.4f}' for st in ('mbpp', 'he')))
print('q16 vs s30: ' + '; '.join(f'{st} t16 x{g(\"mhx_q16\", \"s30\", st, \"K16|g1.0\"):.4f} t32 x{g(\"mhx_q16\", \"s30\", st, \"K32|g1.0\"):.4f}' for st in ('mbpp', 'he')))
" > $O/s16_RESULT.txt
grep -E "chain-32" $O/s16_read.log >> $O/s16_RESULT.txt
cat $O/s16_RESULT.txt
