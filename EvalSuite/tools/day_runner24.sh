#!/bin/bash
# EvalSuite runner 24, 2026-10-27 (detached, from a live evalsuite session). Breakthrough Tue 22:3x (NewIdeas' call):
# B12_FOLD on vs off on the standard 12 GB set, docs/PREREG_fold_2026-10-27.md. Arms b12llfoff / b12llfon (B12LL in
# bt_b12's shipped env, v3l fold off / v3m fold on), 239 rows (agent, math, ifeval, humaneval) in 10 chunks with the same
# row ranges in both arms, ABBA, one process + one queued quiet lock per chunk (RULES: <= 15 min, released between).
# No chunk starts while soft_end.sh ramps or PAUSE exists; cli.gpu stops a stage's whole process group on time-up, a
# lost lock or > 2 GB of swap growth. Deadline Wed 07:00. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day24_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-28 07:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
SETS=agent,math,ifeval,humaneval
CHUNKS="17 16 12 13 34 24 22 34 33 34"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
rows() { if [ -f "results/$1/decode.jsonl" ]; then wc -l < "results/$1/decode.jsonl" | tr -d ' '; else echo 0; fi; }
calm() {   # no timed chunk while soft_end.sh ramps or PAUSE exists (RULES; Fit12 Tue 19:30)
  local n=0
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ]; do
    [ $((n % 20)) = 0 ] && say "waiting: soft_end.sh or PAUSE"
    n=$((n + 1)); sleep 30
  done
}
arm() {   # arm VARIANT TARGET: decode VARIANT up to TARGET rows in all; each call = one chunk process + one quiet lock
  local v=$1 target=$2 tries=0 have rc
  while [ "$(left)" -gt 900 ]; do
    have=$(rows "$v")
    [ "$have" -ge "$target" ] && return 0
    calm
    ./bin/evalsuite gpu decode "$v" --sets $SETS --max-rows $((target - have)) --minutes 16 >> "$LOG" 2>&1; rc=$?
    say "$v: chunk to $target rows rc=$rc ($(rows "$v") rows)"
    if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
      say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
    fi
    if tail -4 "$LOG" | grep -q "swap guard"; then say "swap guard: runner stops"; exit 4; fi
    [ "$(rows "$v")" -ge "$target" ] && return 0
    tries=$((tries + 1))
    [ "$tries" -ge 3 ] && { say "$v: three calls without reaching $target rows: runner stops"; exit 5; }
    sleep 60
  done
  say "deadline: runner stops"; exit 6
}

say "runner 24 start (pid $$)"
target=0; i=0
for k in $CHUNKS; do
  target=$((target + k)); i=$((i + 1))
  if [ $((i % 2)) = 1 ]; then arm b12llfoff $target; arm b12llfon $target; else arm b12llfon $target; arm b12llfoff $target; fi
  say "chunk $i done (both arms at $target rows)"
done
for spec in "b12llfoff ref" "b12llfon ref" "b12llfon b12llfoff" "b12llfoff b12ll"; do   # tables (CPU)
  set -- $spec
  out=results/report_$1$([ "$2" = ref ] || echo "_vs_$2").txt
  ./bin/evalsuite report --variant "$1" --decode $([ "$2" = ref ] || echo "--ref $2") > "$out" 2>&1
  say "report $1 vs $2: $(tail -1 "$out")"
done
for pair in "b12llfon b12llfoff" "b12llfoff b12ll" "b12llfon b12ll"; do
  say "same tokens: $(.venv/bin/python tools/same_tokens.py $pair 2>&1 | tail -1)"
done
[ -f tools/fold_compare.py ] && .venv/bin/python tools/fold_compare.py > results/fold_compare.txt 2>&1 \
  && say "fold_compare: results/fold_compare.txt"
say "runner 24 end"
