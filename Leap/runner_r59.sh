#!/bin/sh
# Leap's fifty-ninth detached screen runner (Sun 2026-11-01 07:3x; FAST INTEGRATION: a lead on a SHARED GPU slot, rough):
# R-69L: the 45 GB default's live loop at C 160 (REF2-T on the gate + t 0.3), processes A (default) - B (BIGMOE_TIER_COMBINE=1
# BIGMOE_OOO=1) - A, 4 prompts, CAP 256. NO-SWAP >= 37.7 GB (peak ~29.7 + 8) before each process; swap guard +2 GB.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner59] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap10() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() { [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && [ "$(noswap10)" -ge 377 ]; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
G="BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_TIER=1 R57_TRUNC=0.3 R57_ARMS=gate+t R57_PROMPTS=4 R57_REPS=1 R57_WARM=1"
one() {  # one <tag> <extra env...>
  tag=$1; shift
  w=0; while ! ok_pre; do [ $w -eq 0 ] && log "$tag: waiting (PAUSE / lid / ACT / NO-SWAP $(noswap10) / 10 GB < 37.7)"; w=1; sleep 60; done
  log "$tag: start (NO-SWAP $(noswap10) / 10 GB)"
  SW0=$(swap_mb)
  env $G R57_TAG=$tag "$@" perl -e 'alarm shift; exec @ARGV' 900 $PY scripts/k69_live.py 160 256 >> logs/job.out 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    SW=$(swap_mb)
    if [ $((SW - SW0)) -gt 2048 ]; then log "SWAP GUARD: swap $SW MB (start $SW0): stopping $tag"; T=$(tree $P); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $P; log "$tag: rc $?"
}
{
  log "then: R-69L (C 160, A - B - A processes: default vs TIER_COMBINE + OOO, shared, rough)"
  while ! peer lock share 40m --cpu 200 --gpu 90 "Leap R-69L: 45 GB default live loop at C 160 (~30 GB), default vs TIER_COMBINE+OOO, A-B-A processes (~15 min, rough)" >/dev/null 2>&1; do sleep 120; done
  log "share lock"
  one r69l_A1
  one r69l_B R69_KEEP_COMBINE=1 BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1
  one r69l_A2
  peer unlock >/dev/null 2>&1
  for t in r69l_A1 r69l_B r69l_A2; do echo "== $t"; grep -E "R-69L|\[gate\+t\]|gate\+t:|lifetime" data/r69l/${t}_screen.txt; done
  $PY - <<'PYEOF'
import json, os
D = os.path.expanduser("~/Leap/data/r69l")
T = {t: json.load(open(f"{D}/{t}_tokens.json")) for t in ("r69l_A1", "r69l_B", "r69l_A2") if os.path.exists(f"{D}/{t}_tokens.json")}
keys = sorted(set().union(*[set(v) for v in T.values()])) if T else []
for k in keys:
    runs = {t: v.get(k, {}).get("gate+t", [None])[0] for t, v in T.items()}
    ref = runs.get("r69l_A1")
    print(f"tokens {k}: " + ", ".join(f"{t} == A1: {r == ref}" for t, r in runs.items() if t != "r69l_A1"))
PYEOF
  log "done"
} >> data/runner_r.log 2>&1
