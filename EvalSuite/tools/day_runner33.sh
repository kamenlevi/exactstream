#!/bin/bash
# EvalSuite runner 33, 2026-10-31 (detached, from a live evalsuite session): b12llff3 for decision 5
# (docs/PREREG_b12llff3_2026-10-31.md). Pre-flight: one scratch row (b12llff3.r9 = runner 24's first agent row) vs
# b12llff's -> results/r41g_preflight.txt; a missing or different row stops the runner (exit 4) before any verdict row.
# Then runner 24's 29 agent rows in ONE process under one quiet ticket (the creep test), then tools/r41g.py ->
# results/r41g.txt and the marker results/R41G_DONE. Code pin: bt_b12.py (677c6558b6; model.BT_B12_SHA also asserts it
# at load), Breakthrough's v3o engine, SpecDecode, MLX / mtplx, evalsuite/; a change stops the runner (exit 7). No process
# while soft_end.sh ramps, PAUSE exists or the lid is closed with no external display. Deadline Sat 2026-10-31 12:00.
# Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day33_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-31 12:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
unset EVALSUITE_BIGMOE_CODE EVALSUITE_STAGE_ENV
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nset() { if [ -f "results/$1/decode.jsonl" ]; then grep -c "\"set\": \"$2\"" "results/$1/decode.jsonl"; else echo 0; fi; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {   # queue first; cli.gpu gates memory (NO-SWAP, need 14 + 4) after the grant
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
    waitlog "waiting: soft_end.sh, PAUSE or a closed lid"
    sleep 30
  done
}
pin() {
  { shasum -a 256 "$HOME/Breakthrough/scripts/bt_b12.py"
    (cd "$HOME/Breakthrough" && find v3o -type f \( -name '*.py' -o -name '*.so' -o -name '*.metal' \) ! -path '*__pycache__*' \
       | sort | xargs shasum)
    git -C "$HOME/SpecDecode" rev-parse HEAD; git -C "$HOME/SpecDecode" status --porcelain --untracked-files=no
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
PIN0=$(pin)
chunk() {   # chunk VARIANT STAGE-ARGS...: one process, one quiet ticket
  local v=$1 rc; shift
  calm
  [ "$(pin)" = "$PIN0" ] || { say "code changed (bt_b12 / v3o / SpecDecode / evalsuite / MLX): runner stops"; exit 7; }
  [ "$(left)" -gt 1200 ] || { say "deadline near: $v not started: runner stops"; exit 6; }
  ./bin/evalsuite gpu decode "$v" "$@" >> "$LOG" 2>&1; rc=$?
  say "$v: chunk rc=$rc ($*)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then say "swap guard: runner stops"; exit 3; fi
  return 0
}

say "runner 33 start (pid $$); code pin $PIN0"
SHA=$(shasum -a 256 "$HOME/Breakthrough/scripts/bt_b12.py" | cut -c1-10)
[ "$SHA" = 677c6558b6 ] || { say "bt_b12.py is $SHA, not 677c6558b6: runner stops"; exit 7; }
[ "$(nset b12llff3.r9 agent)" -ge 1 ] || chunk b12llff3.r9 --sets agent --max-rows 1 --minutes 15
$PY - > results/r41g_preflight.txt 2>&1 <<'PYEOF'
import json
def rows(v):
    try:
        return [json.loads(l) for l in open(f"results/{v}/decode.jsonl")]
    except FileNotFoundError:
        return []
got = rows("b12llff3.r9")
if not got:
    print("b12llff3.r9: no row"); print("pre-flight: STOP"); raise SystemExit
a = got[0]
b = {r["key"]: r for r in rows("b12llff")}.get(a["key"])
same = b is not None and a["completion_ids"] == b["completion_ids"]
print(f"b12llff3.r9 {a['key']}: {len(a['completion_ids'])} tokens, identical to b12llff: {same}, mem {a.get('mem')}")
print("pre-flight:", "OK" if same else "STOP")
PYEOF
say "$(tr '\n' ' ' < results/r41g_preflight.txt)"
grep -q "^pre-flight: OK" results/r41g_preflight.txt || { say "pre-flight did not pass: no verdict row runs (runner stops)"; exit 4; }
NOPROG=0
while [ "$(nset b12llff3 agent)" -lt 29 ]; do
  b=$(nset b12llff3 agent)
  chunk b12llff3 --sets agent --max-rows 29 --minutes 30
  if [ "$(nset b12llff3 agent)" -le "$b" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
  if [ "$(nset b12llff3 agent)" -lt 29 ] && [ "$b" = 0 ]; then
    say "b12llff3: the 29-row process was cut short at $(nset b12llff3 agent) rows (the memory test needs one process): see the PREREG"
  fi
done
$PY tools/r41g.py > results/r41g.txt 2>&1
say "$(grep -m1 '^b12llff3: ' results/r41g.txt)"
touch results/R41G_DONE
say "runner 33 end"
