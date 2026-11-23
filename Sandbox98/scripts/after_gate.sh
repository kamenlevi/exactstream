#!/bin/bash
# Queued 19:2x (before the gate finished): once gate.sh reports, the screen control and the first screen (t30f transfer).
cd "$(dirname "$0")/.."
until grep -q "gate done" logs/gate.log 2>/dev/null; do
  if grep -q "Traceback" logs/gate*.log logs/s30.log 2>/dev/null; then echo "gate failed: no screens"; exit 1; fi
  sleep 10
done
./screen.sh ctl - "control" 300 50 30
./screen.sh feat feat "t30f (feature loss to the true deep state at every depth) was NO EFFECT on the 125B. Here: MBPP 23-30 >= +2.0 pt over the control with sign p < 0.05 -> DISAGREES with the 125B: find why before trusting stand-in levers; within +-2 pt -> AGREES (NO EFFECT); <= -2.0 pt with p < 0.05 -> HURTS here (note it)." 300 50 30
echo "screens done"
