#!/bin/bash
# WEEK98 S20 SMALLHEAD (MTP Head), pre-registered in REPORT 'WEEK98 S20 SMALLHEAD': the head's MLP cut to 1536 / 768 neurons
# (lever mlpw), s30's full recipe from native; read-out sb98/tree8.py vs s30.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -f ckpt/mhx_h2.safetensors ] || mv ckpt/mhd1_c2.safetensors ckpt/mhx_h2.safetensors
[ -f ckpt/mhx_h3.safetensors ] || mv ckpt/mhd1_c2h.safetensors ckpt/mhx_h3.safetensors
: > $O/s20_RESULT.txt
for spec in h2:1536 h3:768; do a=${spec%%:*}; w=${spec#*:}
  export MLPW=$w
  for i in 1 2 3; do
    $S/run_paused.sh s20_$a $O/s20_${a}_train.log 50 $PY sb/train_multi.py --tag mhx --arms "$a:$L/mlpw" --start native \
      --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889 && grep -q "DONE mhx" $O/s20_${a}_train.log && break
    log "$a try $i failed"; sleep 120
  done
  log "$(grep 'DONE mhx' $O/s20_${a}_train.log | tail -1)"
  for i in 1 2 3; do
    PAIRS="mhx_$a:s30" $S/run_paused.sh s20_read_$a $O/s20_read_$a.log 30 $PY $S/tree8.py $O/s20_$a.json s30 "mhx_$a@$L/mlpw" && break
    sleep 120
  done
  python3 -I -c "
import json
c = json.load(open('$O/s20_$a.json'))['cmp']; g = lambda st, k: c[f'mhx_$a@s30|{st}|{k}']['ratio']
ok = all(g(st, k) >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
print('VERDICT $a (MLP $w) vs s30: ' + ('HOLDS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(st, \"K16|g1.0\"):.4f} t32 x{g(st, \"K32|g1.0\"):.4f} chain x{g(st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
" >> $O/s20_RESULT.txt
done
cat $O/s20_RESULT.txt
