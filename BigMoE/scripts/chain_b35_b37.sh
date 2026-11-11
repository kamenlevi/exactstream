#!/bin/bash
# After b36 has released its quiet lock (its " traces: " log line, or its chain has ended): b35 again (the all-layer x
# capture; its first try failed at 21:01 on BSD seq's trailing comma), then b37 (R-37 SB-24 A/B) once b35 has released
# its lock. Each runner takes its own 10-min quiet lock.
cd "$HOME/BigMoE" || exit 1
while pgrep -f "chain_b35_b36.sh" > /dev/null && ! grep -q " traces: " data/b36/b36.log 2>/dev/null; do sleep 15; done
mv data/b35 "data/b35_failed_$(date +%H%M)" 2>/dev/null; mkdir -p data/b35
bash scripts/b35_runner.sh & p=$!
while kill -0 $p 2>/dev/null && ! grep -q " trace: " data/b35/b35.log 2>/dev/null; do sleep 15; done
bash scripts/b37_runner.sh
wait $p
