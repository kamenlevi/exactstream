#!/bin/bash
# WEEK98 S19 SCREENN (MTP Head), pre-registered in REPORT 'WEEK98 S19 SCREENN': post-hoc screened reads of s30 at N 16K / 32K.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
: > $O/s19_RESULT.txt
for n in 16384 32768; do
  for i in 1 2 3; do
    SCREEN_N=$n PAIRS="s30!s:s30" $S/run_paused.sh s19_$n $O/s19_$n.log 30 $PY $S/tree8.py $O/s19_$n.json s30 "s30!s" && break
    sleep 120
  done
  python3 -I -c "
import json
c = json.load(open('$O/s19_$n.json'))['cmp']; g = lambda st, k: c[f's30!s@s30|{st}|{k}']['ratio']
ok = all(g(st, k) >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
print('VERDICT screen N $n vs s30: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(st, \"K16|g1.0\"):.4f} t32 x{g(st, \"K32|g1.0\"):.4f} chain x{g(st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
" >> $O/s19_RESULT.txt
done
cat $O/s19_RESULT.txt
