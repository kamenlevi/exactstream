#!/bin/sh
# D-16 retry (MTP Head, Fri 10-30 02:1x; REPORT 'D-16'). turn36's H arm diverged (loss 2.77 -> 8.46 at step 3): hc-lr
# 1e-3 is far too high for a zero-init projection with fan-in 10,240 (one Adam step moves every output by up to
# lr x 10,240). Two arms at fan-in-matched rates (main layers: fan-in 2,560 at 2e-5 -> 5e-6 here) and a third lower:
# Hc 5e-6, Hb 1.5e-6, training only, each skipped when done; then the paired per-step loss screen vs U (CPU).
# Guards: >= 34 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-16 retry $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 34 ] || { echo "== < 34 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d16_train.py; exit; fi
  done ) & G=$!
D=$HOME/TrainLab/data
for A in Hc:5e-6 Hb:1.5e-6; do
  N=${A%%:*}; R=${A#*:}
  grep -q '"status": "done"' $D/d16_$N.state.json 2>/dev/null && continue
  rm -f $D/d16_$N.state.json
  echo "== train $N (hc-lr $R) $(date +%H:%M:%S)"
  $PY sticky/d16_train.py --hc $D/d16w --data ~/Leap/data/l120 --init $HOME/Leap/data/l120_ft.safetensors --noval \
    --out $D/d16_$N.safetensors --bs 32 --micro 1 --epochs 0.2 --minutes 6 --pmask 0.5 --div fkl --hc-lr $R 2>&1 \
    | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
$PY sticky/d16_loss.py d16_H d16_Hc d16_Hb
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
