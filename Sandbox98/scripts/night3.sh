#!/bin/bash
# Night queue 3 (orchestrator 02:2x): the CARRY400 confirmation (sb/carryconf.py; skipped if logs/SKIP_CARRYCONF exists),
# then S5 (sb/s5inject.py). Starts after the CHAIN SIZE runner exits. Memory rule (orchestrator 01:56 / 02:2x): only after
# BigMoE's G32 test has loaded (its lock gone, or held >= 10 min) and with the run's peak + 8 GB free (GUARD_NOSWAP).
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=night3 GUARD_KILL_WAIT=15
while pgrep -f "night_cs_s[5].sh" > /dev/null; do sleep 15; done
echo "$(date +%H:%M:%S) chain-size runner has exited: $(tail -1 logs/night_cs_s5.log)"
g32_first=""
while true; do                                  # G32 loaded: its lock gone, or held >= 10 min
  if peer ls 2>/dev/null | grep -q "G32 32-row gate test"; then
    [ -z "$g32_first" ] && g32_first=$(date +%s) && echo "$(date +%H:%M:%S) G32's lock seen; waiting until it has loaded"
    [ $(( $(date +%s) - g32_first )) -ge 600 ] && break
  else
    break
  fi
  sleep 30
done
echo "$(date +%H:%M:%S) G32 gate passed (NO-SWAP $(noswap) GB)"
if [ ! -f logs/SKIP_CARRYCONF ]; then
  export MEMGB=5 GUARD_NOSWAP=13
  guarded cc_smoke logs/carryconf_smoke.log 6 $PY sb/carryconf.py --smoke || { echo "carryconf smoke failed"; exit 1; }
  grep -q "DONE verdict" logs/carryconf_smoke.log || { echo "carryconf smoke: no DONE line"; exit 1; }
  if [ ! -f logs/SKIP_CARRYCONF ]; then
    guarded cc logs/carryconf.log 70 $PY sb/carryconf.py || exit 1
    echo "carryconf done: $(grep 'DONE verdict' logs/carryconf.log)"
  fi
fi
export MEMGB=7 GUARD_NOSWAP=15
guarded s5_smoke logs/s5inject_smoke.log 6 $PY sb/s5inject.py --smoke 2 --minutes 0.1 --evals 0.05 || { echo "s5 smoke failed"; exit 1; }
grep -q "DONE verdict" logs/s5inject_smoke.log || { echo "s5 smoke: no DONE line"; exit 1; }
guarded s5 logs/s5inject.log 25 $PY sb/s5inject.py --minutes 8 --evals 2,4 || exit 1
echo "s5 done: $(grep 'DONE verdict' logs/s5inject.log)"
