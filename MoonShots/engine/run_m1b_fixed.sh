#!/bin/sh
# M1b true all-hits (fixed-state repeat: the same 4 positions re-run from a restored state; launch holding a quiet lock):
# BigMoE's model, the engine with BigMoE's MoE block (gate path), the engine's gate-free hit-path prototype.
OUT=$HOME/MoonShots/data/day15; PY=$HOME/BigMoE/.venv/bin/python; E=$HOME/MoonShots/engine
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
run() { tag=$1; mode=$2; shift 2; env "$@" M1B_REPEAT=1 M1B_FIXED=1 $PY $E/test_m1b_speed.py $mode 0.25 32 64 > "$OUT/m1b_$tag.txt" 2>&1
        echo "$(date +%H:%M) done m1b $tag: $(grep -E 'verify pass|oracle path|Traceback|[A-Za-z]*Error' "$OUT/m1b_$tag.txt" | cut -c1-200 | tr '\n' ' ')" >> "$OUT/summary.txt"; }
run fixed_ref ref
run fixed_eng_bigmoe eng M1B_MOE=bigmoe
run fixed_eng_oracle eng M1B_MOE=oracle
peer unlock > /dev/null 2>&1
