#!/bin/bash
# CHAIN SIZE follow-up: prompt-aware lm_head lists on s30 (eval-only). Smoke, then the full read-out. Guarded.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=listctx MEMGB=5 GUARD_NOSWAP=13
guarded lc_smoke logs/listctx_smoke.log 5 $PY sb/listctx.py --nm 2 --nh 2 || { echo "listctx smoke failed"; exit 1; }
grep -q "DONE" logs/listctx_smoke.log || { echo "listctx smoke: no DONE line"; exit 1; }
guarded lc logs/listctx.log 15 $PY sb/listctx.py || exit 1
echo "listctx done: $(grep 'DONE' logs/listctx.log)"
