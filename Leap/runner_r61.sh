#!/bin/sh
# Leap's sixty-first detached runner (Sun 2026-11-01; FAST INTEGRATION: one every-token check for a stack of exact-safe
# changes; an adoption verdict, so a QUIET ticket queued from 23:10): K-73L = the 45 GB default (R57_PRESET=45: REF2-T + gate + t 0.3 + OOO + TIER_COMBINE)
# at C 160, processes A (default) - B (+ K73_EXACT=1: R-70 router, R-72 att_rope + idx_prep, R-73 tail + hc_write) - A,
# 4 prompts, CAP 256, every token vs REF2-T's rows (BigMoE s45t_B gate+t). Night only (NOTBEFORE 23:00, Kamen's day rule:
# small tests by day); waits for NO-SWAP >= 37.7 GB (peak ~29.7 + 8) and for no other process over 8 GB RSS (two
# 45 GB-route processes never run at once); swap guard +2 GB; PAUSE / HOLD / lid / ACT respected.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner61] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap10() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
bigproc() { ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me {n++} END {print n + 0}'; }
night() { h=$(date +%H); [ "$h" -ge 23 ] || [ "$h" -lt 7 ]; }
ok_pre() { night && [ ! -e $Q/PAUSE ] && [ ! -e $Q/HOLD ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
           && [ "$(noswap10)" -ge 377 ] && [ "$(bigproc)" -eq 0 ]; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
G="R57_PRESET=45 BIGMOE_EXT_BUILD=build_next R57_PROMPTS=4 R57_REPS=1 R57_WARM=1 R57_REFTOK=$HOME/BigMoE/data/r57/s45t_B_tokens.json:gate+t"
one() {  # one <tag> <extra env...>
  tag=$1; shift
  w=0; while ! ok_pre; do [ $w -eq 0 ] && log "$tag: waiting (night / PAUSE / HOLD / lid / ACT / NO-SWAP $(noswap10) / 10 GB < 37.7 / big procs $(bigproc))"; w=1; sleep 60; done
  sleep 60; while ! ok_pre; do sleep 60; done          # a second look a minute later: not a gap between another runner's processes
  log "$tag: start (NO-SWAP $(noswap10) / 10 GB)"
  SW0=$(swap_mb)
  env $G R57_TAG=$tag "$@" perl -e 'alarm shift; exec @ARGV' 900 $PY scripts/k73_live.py 160 256 >> logs/job.out 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    SW=$(swap_mb)
    if [ $((SW - SW0)) -gt 2048 ]; then log "SWAP GUARD: swap $SW MB (start $SW0): stopping $tag"; T=$(tree $P); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $P; log "$tag: rc $?"
}
{
  log "queued: K-73L (C 160, A - B - A processes: default vs + the exact-safe stack; shared, rough; every token vs REF2-T rows)"
  # a QUIET ticket, queued from 23:10 behind tonight's C45 turn / D-23 / EvalSuite's runner 36: the queue keeps 45 GB-route
  # processes apart (memory), and it is an adoption verdict, so its timing is clean too. Nothing is held while waiting.
  while [ "$(date +%H%M)" -lt 2310 ] && [ "$(date +%H)" -ge 7 ]; do sleep 120; done
  log "queueing the quiet ticket"
  until PEER_QUEUE_TIMEOUT=28800 peer lock quiet --queue 40m "Leap K-73L: exact-safe kernel stack on the 45 GB default (adoption verdict, every-token check vs REF2-T), C 160 (~30 GB), A-B-A ~20 min" >/dev/null 2>&1; do
    log "quiet ticket lapsed or refused: re-queueing"; sleep 60
  done
  log "quiet lock"
  night || { log "granted by day: releasing it, stopping (night only)"; peer unlock >/dev/null 2>&1; exit 0; }
  one k73_A1
  one k73_B K73_EXACT=1
  one k73_A2
  peer unlock >/dev/null 2>&1
  for t in k73_A1 k73_B k73_A2; do echo "== $t"; grep -E "K-73|\[gate\+t\]|gate\+t:|tier flags|lifetime" data/r73l/${t}_screen.txt; done
  log "done"
} >> data/runner_r.log 2>&1
