#!/bin/bash
# EvalSuite runner 21, 2026-10-24 (detached, from a live evalsuite session). Orchestrator ~15:30: core verdicts on Leap's
# L1-17 (fewer experts per token on B12LL), normal priority (not [low]), against the 4-bit ref and against B12LL; adopting
> (a quote from the project owner, kept private)
# top-8, renormalized) and load_b12ll_budget("", topp=0.8) = b12llka80 (the smallest rank prefix holding >= 80 % of the
# top-10 router weight); prefill, one-row calls and the MTP head keep top-10. First the hook's no-ops (topk=10 =
# b12llk10, topp=1.0 = b12llka100) on 6 rows each: they must give B12LL's tokens exactly, else the runner stops. A stage
# that yields (exit 75: someone touched results/.yield_<cli pid> to pause it at a row boundary) is queued again at
# normal priority. Same chunk rules as tools/day_runner20.sh; deadline 02:30.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day21_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-25 02:30" +%s)
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
    [ "$rc" = 75 ] && { say "$name: paused at a row boundary, queued again"; continue; }
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

ident() {   # ident VARIANT: a no-op setting of the hook must give B12LL's tokens on all 6 rows, else the runner stops
  local v=$1 out a b
  job "decode $v identity (6 rows)" "$v" decode "$v" --sets ifeval,humaneval --limit 3 \
    || { say "$v: the identity check did not finish: runner stops"; exit 4; }
  out=$(.venv/bin/python tools/same_tokens.py "$v" b12ll 2>&1 | tail -1)
  say "identity $v vs b12ll: $out"
  read -r a b <<< "$(echo "$out" | sed -n 's/^all: \([0-9]*\)\/\([0-9]*\) rows.*/\1 \2/p')"
  [ -n "$a" ] && [ "$a" = "$b" ] && [ "$b" = 6 ] || { say "$v: NOT token-identical to B12LL: runner stops (tell Fit12)"; exit 4; }
}

verdict() {   # verdict VARIANT: core sets; tables vs the 4-bit ref and vs B12LL; token identity vs B12LL
  local v=$1
  job "decode $v core" "$v" decode "$v" --sets math,ifeval,humaneval || return 1
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v vs ref: $(tail -1 "results/report_$v.txt")"
  ./bin/evalsuite report --variant "$v" --decode --ref b12ll > "results/report_${v}_vs_b12ll.txt" 2>&1
  say "report $v vs b12ll: $(tail -1 "results/report_${v}_vs_b12ll.txt")"
  same "$v" b12ll
}

say "runner 21 start (pid $$)"
ident b12llk10
ident b12llka100
verdict b12llk8
verdict b12llka80
say "runner 21 end"
