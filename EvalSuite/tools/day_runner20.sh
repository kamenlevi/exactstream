#!/bin/bash
# EvalSuite runner 20, 2026-10-24 (detached, from a live evalsuite session). Orchestrator ~12:50: core verdict on U2 =
# B12LL + Fit12's PR1 fix (fit_hook.load_b12ll_budget("", pr1=True), Fit12 12:57), normal priority (not [low]), against
# the 4-bit ref and against B12LL (U1). Engine pinned to v3f without trims (cli.SHIP_AR), like U1's rows: bt_b12's
# defaults moved to v3g + trims at a50d103. First a short U1 replay (b12ll.r3: 4 IFEval + 4 HumanEval+ rows) under the
# current bt_b12, to show the engine still gives U1's tokens, so U2-vs-U1 differences are PR1's. Same chunk rules as
# tools/day_runner18.sh; deadline 22:00.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day20_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-24 22:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
mark() { ls -lT "results/$1" "results/$1/tfstep_parts" 2>/dev/null | md5 -q; }   # changes when a chunk writes anything

job() {   # job NAME VARIANT STAGE ARGS...   (normal priority: research and verdict tickets queue first come first served)
  local name=$1 v=$2 stuck=0 rc before
  shift 2
  while [ "$(left)" -gt 900 ]; do
    before=$(mark "$v")
    ./bin/evalsuite gpu "$@" --minutes 45 >> "$LOG" 2>&1; rc=$?
    say "$name: chunk rc=$rc"
    if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
      say "$name: the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"
      exit 3
    fi
    [ "$rc" = 0 ] && { say "$name: done"; return 0; }
    if [ "$(mark "$v")" = "$before" ]; then
      stuck=$((stuck + 1))
      [ "$stuck" -ge 2 ] && { say "$name: two chunks without progress, skipping"; return 1; }
      sleep 300
    else
      stuck=0
    fi
  done
  say "$name: deadline"
  return 1
}
same() { say "same tokens $1 vs $2: $(.venv/bin/python tools/same_tokens.py "$1" "$2" 2>&1 | tr '\n' ';')"; }

say "runner 20 start (pid $$)"
job "decode b12ll.r3 U1 replay (8 rows)" b12ll.r3 decode b12ll.r3 --sets ifeval,humaneval --limit 4 && same b12ll.r3 b12ll
job "decode b12llu2 core" b12llu2 decode b12llu2 --sets math,ifeval,humaneval
./bin/evalsuite report --variant b12llu2 --decode > results/report_b12llu2.txt 2>&1
say "report b12llu2 vs ref: $(tail -1 results/report_b12llu2.txt)"
./bin/evalsuite report --variant b12llu2 --decode --ref b12ll > results/report_b12llu2_vs_b12ll.txt 2>&1
say "report b12llu2 vs b12ll: $(tail -1 results/report_b12llu2_vs_b12ll.txt)"
same b12llu2 b12ll
say "runner 20 end"
