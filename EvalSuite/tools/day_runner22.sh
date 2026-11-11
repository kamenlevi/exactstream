#!/bin/bash
# EvalSuite runner 22, 2026-10-24 (detached, from a live evalsuite session). Replaces runner 21 at its b12llk8 chunk
# boundary: runner 21's bash is stopped, its last chunk (cli 90280: b12llk8 chunk 2, queued 16:51:46) finishes on its own,
# and this runner waits for that cli to exit. New after runner 21 (Fit12 16:30, the orchestrator's catch): b12llk8 is
# U1-based, while Fit12's stack TK2 is PR1 + top-8 (+ tau 0.7, which moves no tokens under PR1), so b12llk8u2 =
# fit_hook.load_b12ll_budget("", pr1=True, topk=8) is judged against U2 (b12llu2), the 4-bit ref and B12LL, after its
# no-op b12llk10u2 (pr1=True, topk=10 = U2 bitwise) on 6 rows. Order: b12llk8 (rest) -> b12llk10u2 identity ->
# b12llk8u2 -> b12llka80. Normal priority; same chunk rules as tools/day_runner21.sh; deadline 02:30.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day22_runner.log
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
core_done() { [ "$(grep -c '"set": "math"\|"set": "ifeval"\|"set": "humaneval"' "results/$1/decode.jsonl" 2>/dev/null)" -ge 210 ]; }

ident() {   # ident VARIANT BASE: the hook's no-op must give BASE's tokens on all 6 rows, else the runner stops
  local v=$1 base=$2 out a b
  job "decode $v identity (6 rows)" "$v" decode "$v" --sets ifeval,humaneval --limit 3 \
    || { say "$v: the identity check did not finish: runner stops"; exit 4; }
  out=$(.venv/bin/python tools/same_tokens.py "$v" "$base" 2>&1 | tail -1)
  say "identity $v vs $base: $out"
  read -r a b <<< "$(echo "$out" | sed -n 's/^all: \([0-9]*\)\/\([0-9]*\) rows.*/\1 \2/p')"
  [ -n "$a" ] && [ "$a" = "$b" ] && [ "$b" = 6 ] || { say "$v: NOT token-identical to $base: runner stops (tell Fit12)"; exit 4; }
}

verdict() {   # verdict VARIANT BASE...: core sets; tables vs the 4-bit ref and vs each BASE; token identity vs each BASE
  local v=$1 b
  shift
  core_done "$v" || job "decode $v core" "$v" decode "$v" --sets math,ifeval,humaneval || return 1
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v vs ref: $(tail -1 "results/report_$v.txt")"
  for b in "$@"; do
    ./bin/evalsuite report --variant "$v" --decode --ref "$b" > "results/report_${v}_vs_$b.txt" 2>&1
    say "report $v vs $b: $(tail -1 "results/report_${v}_vs_$b.txt")"
    same "$v" "$b"
  done
}

say "runner 22 start (pid $$): waiting for runner 21's last b12llk8 chunk (cli 90280)"
while kill -0 90280 2>/dev/null; do sleep 20; done
say "cli 90280 ended; b12llk8 rows: $(wc -l < results/b12llk8/decode.jsonl | tr -d ' ')"
verdict b12llk8 b12ll
ident b12llk10u2 b12llu2
verdict b12llk8u2 b12llu2 b12ll
verdict b12llka80 b12ll
say "runner 22 end"
