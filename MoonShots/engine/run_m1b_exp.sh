#!/bin/sh
# M1b overhead experiments (all-hits repeat mode; launch holding a quiet lock): per-layer async_eval off (BigMoE and the
# engine with BigMoE's MoE block), and the gate-free hit-path prototype (engine/oracle).
OUT=$HOME/MoonShots/data/day15; PY=$HOME/BigMoE/.venv/bin/python; E=$HOME/MoonShots/engine
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
run() { tag=$1; shift; env "$@" M1B_REPEAT=1 $PY $E/test_m1b_speed.py $MODE 0.25 32 64 > "$OUT/m1b_$tag.txt" 2>&1
        echo "$(date +%H:%M) done m1b $tag: $(grep -E 'verify pass|oracle path|Traceback|[A-Za-z]*Error' "$OUT/m1b_$tag.txt" | cut -c1-200 | tr '\n' ' ')" >> "$OUT/summary.txt"; }
MODE=ref run ref_ae0 BIGMOE_ASYNC_EVERY=0
MODE=eng run eng_bigmoe_ae0 BIGMOE_ASYNC_EVERY=0 M1B_MOE=bigmoe
MODE=eng run eng_oracle M1B_MOE=oracle
peer unlock > /dev/null 2>&1
