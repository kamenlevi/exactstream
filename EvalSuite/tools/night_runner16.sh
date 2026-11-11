#!/bin/bash
# EvalSuite runner 16, 2026-10-24 (detached, from a live evalsuite session; started inside runner 15's lock at a job
# boundary). Orchestrator ~01:00 (NewIdeas' fallback order): B12LL-b20 fails the suite on truncations alone (15 vs 13;
# IFEval -2.0 passes), so Fit12's UGB-0.5 (b16 only on verify passes where the MTP head is confident; exact B12LL routing
# on the rest; ~/Fit12/scripts/fit_hook.py load_b12ll_budget('b16', ugb=0.5)) is judged next. First a check that the
# gate-off path is exact B12LL (gate 1.01 never opens: 6 math rows must equal B12LL's token for token), with peak
# footprints logged (< 12 GB). Then b12llugb core (+ vs b12ll), b12llb20 new (resume), b8fast.r2, f24b16.r2 rest,
# b12llb16 new last (the first to drop). Same chunk rules as tools/day_runner6.sh; deadline 07:05 (night_plan.md).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night16_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-24 07:05" +%s)
export PEER_QUEUE_TIMEOUT=14400
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
mark() { ls -lT "results/$1" "results/$1/tfstep_parts" 2>/dev/null | md5 -q; }   # changes when a chunk writes anything

job() {   # job NAME VARIANT STAGE ARGS...
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
vs() { ./bin/evalsuite report --variant "$1" --decode --ref "$2" > "results/report_$1_vs_$2.txt" 2>&1; say "report $1 vs $2: $(tail -1 "results/report_$1_vs_$2.txt")"; }
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
same_as_b12ll() {   # same_as_b12ll VARIANT: rows token-identical to B12LL's, and the peak footprint
  python3 -c "
import json, sys
a = {r['key']: r for r in map(json.loads, open('results/b12ll/decode.jsonl'))}
b = {r['key']: r for r in map(json.loads, open('results/$1/decode.jsonl'))}
ks = [k for k in b if k in a]
fp = json.load(open('results/$1/decode.json')).get('footprint_peak_gb')
print(f'{sum(a[k][\"completion_ids\"] == b[k][\"completion_ids\"] for k in ks)}/{len(ks)} rows token-identical to B12LL; peak footprint {fp:.2f} GB')
" 2>&1
}
say "runner 16 start (pid $$)"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "check b12llugbx (gate never opens)" b12llugbx decode b12llugbx --sets math --limit 6
say "gate-off check: $(same_as_b12ll b12llugbx)"
job "decode b12llugb core" b12llugb decode b12llugb --sets $CORE; table b12llugb; vs b12llugb b12ll
say "b12llugb: $(same_as_b12ll b12llugb)"
job "decode b12llb20 new" b12llb20 decode b12llb20 --sets $NEW; table b12llb20; vs b12llb20 b12ll
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
job "decode f24b16.r2 core" f24b16.r2 decode f24b16.r2 --sets $CORE; table f24b16.r2; vs f24b16.r2 f24nat
job "decode b12llb16 new" b12llb16 decode b12llb16 --sets $NEW; table b12llb16; vs b12llb16 b12ll
say "runner 16 end"
