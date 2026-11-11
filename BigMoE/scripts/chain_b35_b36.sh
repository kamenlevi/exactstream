#!/bin/bash
# b35 (S10c all-layer x capture) then b36 (R-27 per-layer pools A/B + RAM24 probe), one detached chain: b36 starts as soon
# as b35 has released its quiet lock (its " trace: " log line), or b35 has exited; b35's CPU check runs on meanwhile.
cd "$HOME/BigMoE" || exit 1
bash scripts/b35_runner.sh & p=$!
while kill -0 $p 2>/dev/null && ! grep -q " trace: " data/b35/b35.log 2>/dev/null; do sleep 15; done
SIZES="$HOME/MoonShots/data/laysz_engine_147_46.json" bash scripts/b36_runner.sh
wait $p
