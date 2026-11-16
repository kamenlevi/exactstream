#!/bin/bash
# WEEK98 S18 SCREEN (MTP Head), pre-registered in REPORT 'WEEK98 S18 SCREEN': the head trained under the 8K vocab screen
# (lever scr) on c16's annealed recipe (s16) and on s30's recipe from native (sn); read-out sb98/tree8.py ('!s' = screened).
# Waits for the runner pid $1 (S17) first.
cd "$HOME/Sandbox98" || exit 1
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98; L=$S/levers
PY=$HOME/BigMoE/.venv/bin/python
export MEMGB=4
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
log "S18 starts"
[ -f ckpt/mhx_s16.safetensors ] || mv ckpt/mhd1_c2_s380.safetensors ckpt/mhx_s16.safetensors
[ -f ckpt/mhx_sn.safetensors ] || mv ckpt/mhd1_c2h_s380.safetensors ckpt/mhx_sn.safetensors
tr() {   # $1 = arm, $2.. = recipe
  local a=$1; shift
  for i in 1 2 3; do
    $S/run_paused.sh s18_$a $O/s18_${a}_train.log 50 $PY sb/train_multi.py --tag mhx --arms "$a:$L/scr" "$@" && \
      grep -q "DONE mhx" $O/s18_${a}_train.log && { log "$(grep 'DONE mhx' $O/s18_${a}_train.log | tail -1)"; return 0; }
    log "$a try $i failed"; sleep 120
  done
  exit 1
}
tr s16 --start s30 --resume-opt "" --files pool2 --seed 1116 --depth 30 --steps 680
tr sn --start native --resume-opt "" --files base --seed 1114 --depth 30 --steps 1889
for i in 1 2 3; do
  PAIRS="s30!s:s30,mhx_s16!s:mhx_c16,mhx_sn!s:s30,mhx_s16!s:s30" $S/run_paused.sh s18_read $O/s18_read.log 60 $PY $S/tree8.py \
    $O/s18.json s30 "s30!s" mhx_c16 "mhx_s16!s" "mhx_sn!s" && break
  log "read-out try $i failed"; sleep 120
done
python3 -I -c "
import json
c = json.load(open('$O/s18.json'))['cmp']
g = lambda r, b, st, k: c[f'{r}@{b}|{st}|{k}']['ratio']
ph = {st: g('s30!s', 's30', st, 'K32|g1.0') for st in ('mbpp', 'he')}
print('post-hoc screen s30!s / s30 at tree-32: ' + ', '.join(f'{st} x{v:.4f}' for st, v in ph.items()) + ('' if any(v <= 0.99 for v in ph.values()) else ' -> UNINFORMATIVE'))
for run, ctl in (('mhx_s16!s', 'mhx_c16'), ('mhx_sn!s', 's30')):
    ok = all(g(run, ctl, st, k) >= 0.99 for st in ('mbpp', 'he') for k in ('K16|g1.0', 'K32|g1.0'))
    print(f'VERDICT {run} vs {ctl}: ' + ('PASS' if ok else 'FAIL') + ' | ' + '; '.join(f'{st} t16 x{g(run, ctl, st, \"K16|g1.0\"):.4f} t32 x{g(run, ctl, st, \"K32|g1.0\"):.4f} chain x{g(run, ctl, st, \"chain\"):.4f}' for st in ('mbpp', 'he')))
print('s16 vs s30: ' + '; '.join(f'{st} t16 x{g(\"mhx_s16!s\", \"s30\", st, \"K16|g1.0\"):.4f} t32 x{g(\"mhx_s16!s\", \"s30\", st, \"K32|g1.0\"):.4f}' for st in ('mbpp', 'he')))
" > $O/s18_RESULT.txt
grep -E "chain-32" $O/s18_read.log >> $O/s18_RESULT.txt
cat $O/s18_RESULT.txt
