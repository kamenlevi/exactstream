#!/bin/bash
# distill re-test (results/screens/DISTILL2_LINE.md, written first): after the fresh-pool capture; polite guards.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
until grep -q "\[cap\] DONE" logs/capture_mc2.log 2>/dev/null; do
  pgrep -f "sb/capture.py --sets mc2" >/dev/null || { grep -q "\[cap\] DONE" logs/capture_mc2.log || { echo "capture ended without DONE"; exit 1; }; }
  sleep 10
done
$PY sb/addmc2.py
source scripts/guard_lib.sh
export GUARD_SETTLE=15 MEMGB=4.5
guarded distill2 logs/distill2.log 20 $PY sb/train_multi.py --tag d2 --arms ctl:-,distill:distill --depth 30 --steps 400 || exit 1
guarded distill2_eval logs/distill2_eval.log 15 $PY sb/evalx.py --runs s30,d2_ctl,d2_distill@distill --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/d2.json || exit 1
{
  echo "== distill re-test (DISTILL2_LINE.md) $(date '+%a %d %b %H:%M')"
  echo "-- control vs s30 (must hold)"
  $PY sb/compare.py results/screens/d2.json "s30|mbpp|chain" results/screens/d2.json "d2_ctl|mbpp|chain" --band 23-30
  $PY sb/compare.py results/screens/d2.json "s30|he|chain" results/screens/d2.json "d2_ctl|he|chain" --band 23-30
  for ST in mbpp he; do
    echo "-- distill vs ctl, $ST"
    $PY sb/compare.py results/screens/d2.json "d2_ctl|$ST|chain" results/screens/d2.json "d2_distill|$ST|chain" --band 23-30 --json results/screens/d2_distill_$ST.json
  done
} | tee -a results/SCREENS.log
echo "distill2 done"
