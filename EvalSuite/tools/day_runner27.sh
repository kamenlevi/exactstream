#!/bin/bash
# EvalSuite runner 27, 2026-10-29 (detached, from a live evalsuite session): R-41C part B, resumed after Kamen's full
# stop (Thu 02:11 to ~15:35; runner 26 was killed while it waited for 07:30). docs/PREREG_r41c_2026-10-28.md,
# amendment 2. Part A is done (runner 26). Here: b12llff's other 194 core rows in chunks (no new row after 9 min,
# stopped at 16), one process + one queued quiet lock each; then b12llfx on the core rows where b12llff differs from
# b12ll (--keys), base_r41c, the reports and results/r41c_B.txt. Same code as runner 26: the pin must equal its
# f22f5bd31102 at the start, and a change stops the runner (exit 7). No chunk while soft_end.sh ramps, PAUSE exists,
# the lid is closed with no external display, or < 30 GB is free. Deadline Fri 2026-10-30 08:00. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day27_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-30 08:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nset() { if [ -f "results/$1/decode.jsonl" ]; then grep -c "\"set\": \"$2\"" "results/$1/decode.jsonl"; else echo 0; fi; }
ncore() { echo $(( $(nset "$1" math) + $(nset "$1" ifeval) + $(nset "$1" humaneval) )); }
nkeys() { grep -c . "$1"; }   # keys in a diffkeys file (it holds one empty line when nothing differs)
missing() {   # missing VARIANT KEYFILE: how many of the file's keys VARIANT has not decoded
  $PY - "$1" "$2" <<'EOF'
import json, os, sys
p = f"results/{sys.argv[1]}/decode.jsonl"
have = {json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()
print(len(set(open(sys.argv[2]).read().split()) - have))
EOF
}
free_gb() { memory_pressure 2>/dev/null | tail -1 | sed 's/.*: *//; s/%//' | awk '{print int($1 * 64 / 100)}'; }
lid_shut() {   # lid closed and no external display (lid_guard.sh's test): research holds
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut \
        || [ "$(free_gb)" -lt 30 ]; do
    waitlog "waiting: soft_end.sh, PAUSE, a closed lid or < 30 GB free ($(free_gb) GB)"
    sleep 30
  done
}
pin() {
  { shasum -a 256 "$HOME/Breakthrough/scripts/bt_b12.py"
    (cd "$HOME/Breakthrough" && find v3l v3m -type f \( -name '*.py' -o -name '*.so' -o -name '*.metal' \) \
       ! -path '*__pycache__*' | sort | xargs shasum)
    git -C "$HOME/SpecDecode" rev-parse HEAD; git -C "$HOME/SpecDecode" status --porcelain --untracked-files=no
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mtplx)-.*dist-info$'; } 2>&1 | shasum | cut -c1-12
}
PIN0=$(pin)
checkpin() {
  local p; p=$(pin)
  [ "$p" = "$PIN0" ] || { say "code changed (bt_b12.py, v3l / v3m engines, SpecDecode or MLX / mtplx): pin $PIN0 -> $p: runner stops"; return 1; }
}
chunk() {   # chunk VARIANT STAGE-ARGS...: one process, one quiet lock; returns 1 if the runner must stop
  local v=$1 rc; shift
  calm
  checkpin || exit 7
  [ "$(left)" -gt 1200 ] || { say "deadline near: $v not started: runner stops"; exit 6; }
  ./bin/evalsuite gpu decode "$v" "$@" >> "$LOG" 2>&1; rc=$?
  say "$v: chunk rc=$rc ($*)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then say "swap guard: runner stops"; exit 3; fi
  return 0
}
NOPROG=0
progress() {   # progress BEFORE AFTER: three calls in a row without a new row stop the runner
  if [ "$2" -le "$1" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
}
keys_step() {   # keys_step VARIANT SETS KEYFILE: chunks until VARIANT has every key in KEYFILE
  local v=$1 sets=$2 kf=$3 b
  while [ "$(missing "$v" "$kf")" -gt 0 ]; do
    b=$(missing "$v" "$kf")
    chunk "$v" --sets "$sets" --keys "$kf" --chunk-minutes 9 --minutes 16
    progress "$(missing "$v" "$kf")" "$b"
  done
}

say "runner 27 start (pid $$); code pin $PIN0"
[ "$PIN0" = f22f5bd31102 ] || { say "the code differs from runner 26's (pin f22f5bd31102): runner stops"; exit 7; }
# part B: the other 194 core rows
while [ "$(ncore b12llff)" -lt 210 ]; do
  b=$(ncore b12llff)
  chunk b12llff --sets math,ifeval,humaneval --chunk-minutes 9 --minutes 16
  progress "$b" "$(ncore b12llff)"
done
# part B's follow-up: b12llfx on the core rows where b12llff differs from b12ll (rows b12llfx has are skipped)
$PY tools/r41c.py diffkeys b12llff b12ll math,ifeval,humaneval > results/r41c_B_diff.txt
say "part B: b12llff differs from b12ll on $(nkeys results/r41c_B_diff.txt) core rows; b12llfx decodes them"
[ "$(nkeys results/r41c_B_diff.txt)" -gt 0 ] && keys_step b12llfx math,ifeval,humaneval results/r41c_B_diff.txt
$PY tools/r41c.py base > results/r41c_base.txt 2>&1 && say "$(head -1 results/r41c_base.txt)"
./bin/evalsuite report --variant b12llff --decode --ref base_r41c > results/report_b12llff_vs_base_r41c.txt 2>&1
say "report b12llff vs base_r41c: $(tail -1 results/report_b12llff_vs_base_r41c.txt)"
./bin/evalsuite report --variant b12llff --decode --ref b12ll > results/report_b12llff_vs_b12ll.txt 2>&1
say "report b12llff vs b12ll: $(tail -1 results/report_b12llff_vs_b12ll.txt)"
$PY tools/r41c.py b > results/r41c_B.txt 2>&1
$PY tools/fold_compare.py b12llfx b12llff > results/fold_compare_r41c.txt 2>&1
touch results/R41C_B_DONE
say "runner 27 end"
