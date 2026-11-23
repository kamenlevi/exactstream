#!/bin/bash
# After the DEPTH batch (DEPTH_LINES.md additions, written first): timing, ladder placement, and the conditional window-cap
# and deep3 runs. Polite guards.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
until grep -q "depth done" logs/depth_runner.log 2>/dev/null; do
  pgrep -f "scripts/depth.sh" >/dev/null || { grep -q "depth done" logs/depth_runner.log || { echo "depth batch ended without a result"; exit 1; }; }
  sleep 20
done
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=4
guarded timedraft logs/timedraft.log 10 $PY sb/timedraft.py --ctx 400 --layers 1,2,3
{ echo; echo "## Depth results ($(date '+%a %d %b %H:%M'))"; echo; echo "Placement (addition 2):"; echo; $PY sb/ladder.py results/screens/dp.json; echo; echo "Timing (addition 1, shared GPU):"; grep "\[td\]" logs/timedraft.log; } >> results/screens/DEPTH_LINES.md
pass_of() { $PY -c "import json; r = json.load(open('results/screens/dp_$1_mbpp.json'))['23-30']; print('yes' if r['diff'] >= 0.02 and r['p_sign'] < 0.05 else 'no')"; }
d23() { $PY -c "import json; print(json.load(open('results/screens/dp_$1_mbpp.json'))['23-30']['diff'])"; }
for A in deep2 tie2; do
  if [ "$(pass_of $A)" = yes ]; then
    echo "$A passes: window-cap eval (256)"
    guarded wcap_$A logs/wcap_$A.log 15 $PY sb/evalx.py --runs dp_$A@$A --sets mbpp --nm 50 --depth 31 --mode chain --window 256 --out results/screens/dp_${A}_w256.json
    { echo; echo "Window cap 256 on $A (addition 3):"; $PY sb/compare.py results/screens/dp.json "dp_$A|mbpp|chain" results/screens/dp_${A}_w256.json "dp_$A|mbpp|chain" --band 23-30; } >> results/screens/DEPTH_LINES.md
  fi
done
if $PY -c "import sys; sys.exit(0 if $(d23 deep2) >= 0.01 else 1)"; then
  echo "deep2 >= +1 pt: deep3"
  export MEMGB=5 NO_MEM_OTHERS=1
  guarded deep3 logs/deep3.log 25 $PY sb/train_multi.py --tag dp3 --arms deep3:deep3 --depth 30 --steps 400 || exit 1
  unset NO_MEM_OTHERS; export MEMGB=4.5
  guarded deep3_eval logs/deep3_eval.log 15 $PY sb/evalx.py --runs dp3_deep3@deep3 --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/dp3.json
  { echo; echo "deep3 (addition 4) vs dp_ctl and vs deep2:"; for ST in mbpp he; do
      $PY sb/compare.py results/screens/dp.json "dp_ctl|$ST|chain" results/screens/dp3.json "dp3_deep3|$ST|chain" --band 23-30
      $PY sb/compare.py results/screens/dp.json "dp_deep2|$ST|chain" results/screens/dp3.json "dp3_deep3|$ST|chain" --band 23-30; done; } >> results/screens/DEPTH_LINES.md
else
  echo "deep2 below +1 pt: no deep3 (addition 4's condition)" | tee -a results/screens/DEPTH_LINES.md
fi
echo "depth post done"
