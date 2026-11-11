#!/bin/bash
# EvalSuite runner 26, 2026-10-28 (detached, from a live evalsuite session): R-41C fold confirmation,
# docs/PREREG_r41c_2026-10-28.md. Part A: b12llfx (fold-exact = the new bt_b12 defaults) on runner 24's 45 rows, the
# 29 agent rows in ONE process, then math 16. Part B: b12llff (fold-fast) on all 239 rows: its first 45 alternate with
# A (ABBA: A agent, B agent, B math 16, A math 16), its other 194 core rows from Thu 07:30 in chunks (no new row after
# 9 min, stopped at 16). Follow-ups through --keys: b12llox on A's differing rows (tonight), b12llfx on the core rows
# where b12llff differs from b12ll (after B). One process + one queued quiet lock per chunk, released between chunks.
# No chunk before 22:30 tonight (MTP Head's D-09 first, NewIdeas), with < 30 GB free, while soft_end.sh ramps, PAUSE
# exists, or the lid is closed with no external display. Code pin: bt_b12.py, the v3l / v3m engines, SpecDecode, the
# MLX / mtplx installs; a change stops the runner (exit 7). Deadline Thu 2026-10-29 14:00. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day26_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-29 14:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-28 22:30" +%s)
MORNING=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-29 07:30" +%s)
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nset() { if [ -f "results/$1/decode.jsonl" ]; then grep -c "\"set\": \"$2\"" "results/$1/decode.jsonl"; else echo 0; fi; }
ncore() { echo $(( $(nset "$1" math) + $(nset "$1" ifeval) + $(nset "$1" humaneval) )); }
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
wait_until() { while [ "$(date +%s)" -lt "$1" ]; do waitlog "waiting until $(date -r "$1" '+%a %H:%M') ($2)"; sleep 60; done; }
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
step() {   # step VARIANT SET TARGET MINUTES [STAGE-ARGS...]: chunks until VARIANT has TARGET rows of SET
  local v=$1 s=$2 target=$3 m=$4 b; shift 4
  while [ "$(nset "$v" "$s")" -lt "$target" ]; do
    b=$(nset "$v" "$s")
    chunk "$v" --sets "$s" --minutes "$m" "$@"
    if [ "$s" = agent ] && [ "$b" = 0 ] && [ "$(nset "$v" agent)" -lt 29 ]; then   # bash 3.2: no associative arrays
      eval "REDO_$(echo "$v" | tr -c 'a-zA-Z0-9\n' '_')=1"
      say "$v: agent process cut short at $(nset "$v" agent) rows: its 29-row process is re-run under $v.r1 after the ABBA chunks"
    fi
    progress "$b" "$(nset "$v" "$s")"
  done
}
keys_step() {   # keys_step VARIANT SETS KEYFILE: chunks until VARIANT has every key in KEYFILE
  local v=$1 sets=$2 kf=$3 b
  while [ "$(missing "$v" "$kf")" -gt 0 ]; do
    b=$(missing "$v" "$kf")
    chunk "$v" --sets "$sets" --keys "$kf" --chunk-minutes 9 --minutes 16
    progress "$(missing "$v" "$kf")" "$b"
  done
}

say "runner 26 start (pid $$); code pin $PIN0"
# amendment 21:3x (PREREG): the math pair first, while the GPU is idle before 22:05 (B math 16, then A math 16), each
# back to back; no row starts after 22:02 and cli stops a stage at 22:05, so no lock of ours spans 22:05-22:30
CUT=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-28 22:02" +%s)
for v in b12llff b12llfx; do
  now=$(date +%s)
  [ $(( CUT - now )) -ge 600 ] || break           # under 10 min left: the pair continues after the agent pair
  [ "$(nset "$v" math)" -ge 16 ] && continue
  calm; checkpin || exit 7
  b=$(nset "$v" math)
  ./bin/evalsuite gpu decode "$v" --sets math --max-rows 16 --chunk-minutes $(( (CUT - now) / 60 )) \
    --minutes $(( (CUT + 180 - now) / 60 )) >> "$LOG" 2>&1
  say "$v: math chunk before 22:05 rc=$? ($b -> $(nset "$v" math) rows)"
done
wait_until "$START" "MTP Head's D-09 has the GPU first (NewIdeas)"
# the agent pair back to back (A, B), then whatever the math pair still lacks (B, A): ABBA mirrored
step b12llfx agent 29 30 --max-rows 29
step b12llff agent 29 30 --max-rows 29
step b12llff math 16 25 --max-rows 16
step b12llfx math 16 25 --max-rows 16
for v in b12llfx b12llff; do
  eval "r=\${REDO_$v:-0}"
  if [ "$r" = 1 ]; then step "$v.r1" agent 29 30 --max-rows 29; fi
done
# part A's follow-up: b12llox (old envs on the new code) on the rows where b12llfx differs from b12llfoff
$PY tools/r41c.py diffkeys b12llfx b12llfoff agent,math > results/r41c_A_diff.txt
if [ -s results/r41c_A_diff.txt ]; then
  say "part A: $(wc -l < results/r41c_A_diff.txt | tr -d ' ') rows differ from b12llfoff: b12llox decodes them"
  keys_step b12llox agent,math results/r41c_A_diff.txt
fi
$PY tools/r41c.py a > results/r41c_A.txt 2>&1
$PY tools/fold_compare.py b12llfx b12llff > results/fold_compare_r41c.txt 2>&1
say "part A: $(grep -m1 '^Part A' results/r41c_A.txt)"
touch results/R41C_A_DONE
# part B: the other 194 core rows from the morning
wait_until "$MORNING" "B's core rows from the morning (NewIdeas)"
while [ "$(ncore b12llff)" -lt 210 ]; do
  b=$(ncore b12llff)
  chunk b12llff --sets math,ifeval,humaneval --chunk-minutes 9 --minutes 16
  progress "$b" "$(ncore b12llff)"
done
# part B's follow-up: b12llfx on the core rows where b12llff differs from b12ll (rows b12llfx has are skipped)
$PY tools/r41c.py diffkeys b12llff b12ll math,ifeval,humaneval > results/r41c_B_diff.txt
say "part B: b12llff differs from b12ll on $(wc -l < results/r41c_B_diff.txt | tr -d ' ') core rows; b12llfx decodes them"
[ -s results/r41c_B_diff.txt ] && keys_step b12llfx math,ifeval,humaneval results/r41c_B_diff.txt
$PY tools/r41c.py base > results/r41c_base.txt 2>&1 && say "$(head -1 results/r41c_base.txt)"
./bin/evalsuite report --variant b12llff --decode --ref base_r41c > results/report_b12llff_vs_base_r41c.txt 2>&1
say "report b12llff vs base_r41c: $(tail -1 results/report_b12llff_vs_base_r41c.txt)"
./bin/evalsuite report --variant b12llff --decode --ref b12ll > results/report_b12llff_vs_b12ll.txt 2>&1
say "report b12llff vs b12ll: $(tail -1 results/report_b12llff_vs_b12ll.txt)"
$PY tools/r41c.py b > results/r41c_B.txt 2>&1
$PY tools/fold_compare.py b12llfx b12llff > results/fold_compare_r41c.txt 2>&1
touch results/R41C_B_DONE
say "runner 26 end"
