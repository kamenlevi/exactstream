#!/bin/bash
# EvalSuite runner 28, 2026-10-29 (detached, from a live evalsuite session): R-41D, the fold arms with R-45's fix
# (docs/PREREG_r41d_2026-10-29.md). The arms run in the order given (tools/day_runner28.sh b12llff2 b12llfx2, or one
# arm). First the pre-flight: one scratch row per arm (ARM.r9, the first agent row) -> results/r41d_preflight.txt; a
# missing row or one that differs from its baseline's stops the runner (exit 4) before any verdict row. Then each arm
# decodes runner 24's 45 rows: agent 29 in ONE process (the creep test), then math 16 in one process, each
# with one queued quiet lock. Then tools/r41d.py -> results/r41d.txt and the marker results/R41D_DONE. Code pin:
# bt_b12.py (the swapped one; model.BT_B12_SHA also asserts its sha per arm at load), the v3l / v3m engines,
# SpecDecode, MLX / mtplx; a change stops the runner (exit 7). No process while soft_end.sh ramps, PAUSE exists, the
# lid is closed with no external display, or < 30 GB is free. Deadline Fri 2026-10-30 08:00. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day28_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-30 08:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nset() { if [ -f "results/$1/decode.jsonl" ]; then grep -c "\"set\": \"$2\"" "results/$1/decode.jsonl"; else echo 0; fi; }
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
step() {   # step VARIANT SET TARGET MINUTES [STAGE-ARGS...]: chunks until VARIANT has TARGET rows of SET
  local v=$1 s=$2 target=$3 m=$4 b; shift 4
  while [ "$(nset "$v" "$s")" -lt "$target" ]; do
    b=$(nset "$v" "$s")
    chunk "$v" --sets "$s" --minutes "$m" "$@"
    if [ "$s" = agent ] && [ "$b" = 0 ] && [ "$(nset "$v" agent)" -lt 29 ]; then   # bash 3.2: no associative arrays
      eval "REDO_$(echo "$v" | tr -c 'a-zA-Z0-9\n' '_')=1"
      say "$v: agent process cut short at $(nset "$v" agent) rows: its 29-row process is re-run under $v.r1 at the end"
    fi
    progress "$b" "$(nset "$v" "$s")"
  done
}

say "runner 28 start (pid $$); arms: $*; code pin $PIN0"
[ $# -ge 1 ] || { say "no arm given: runner stops"; exit 2; }
SHA=$(shasum -a 256 "$HOME/Breakthrough/scripts/bt_b12.py" | cut -c1-10)
[ "$SHA" = ce6d4c63d2 ] || { say "bt_b12.py is $SHA, not ce6d4c63d2 (amendment 2): runner stops"; exit 7; }
for v in "$@"; do   # pre-flight: the arm's switches and bt_b12 sha load (model.load asserts them), on one scratch row
  [ "$(nset "$v.r9" agent)" -ge 1 ] || chunk "$v.r9" --sets agent --max-rows 1 --minutes 15
done
$PY - "$@" > results/r41d_preflight.txt 2>&1 <<'EOF'
import json, sys
BASE = {"b12llfx2": "b12llfx", "b12llff2": "b12llff"}
def rows(v):
    try:
        return [json.loads(l) for l in open(f"results/{v}/decode.jsonl")]
    except FileNotFoundError:
        return []
ok = True
for v in sys.argv[1:]:
    got = rows(v + ".r9")
    if not got:
        print(f"{v}.r9: no row"); ok = False; continue
    a = got[0]
    b = {r["key"]: r for r in rows(BASE[v])}.get(a["key"])
    same = b is not None and a["completion_ids"] == b["completion_ids"]
    print(f"{v}.r9 {a['key']}: {len(a['completion_ids'])} tokens, identical to {BASE[v]}: {same}, mem {a.get('mem')}")
    ok = ok and same
print("pre-flight:", "OK" if ok else "STOP")
EOF
say "$(tr '\n' ' ' < results/r41d_preflight.txt)"
grep -q "^pre-flight: OK" results/r41d_preflight.txt || { say "pre-flight did not pass: no verdict row runs (runner stops)"; exit 4; }
for v in "$@"; do
  step "$v" agent 29 30 --max-rows 29
  step "$v" math 16 25 --max-rows 16
done
for v in "$@"; do   # an agent process cut short: its 29-row process once more, in full, for the memory test
  eval "r=\${REDO_$(echo "$v" | tr -c 'a-zA-Z0-9\n' '_'):-0}"
  if [ "$r" = 1 ]; then step "$v.r1" agent 29 30 --max-rows 29; fi
done
$PY tools/r41d.py "$@" > results/r41d.txt 2>&1
for v in "$@"; do say "$(grep -m1 "^$v: " results/r41d.txt)"; done
touch results/R41D_DONE
say "runner 28 end"
