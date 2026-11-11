#!/bin/bash
# EvalSuite runner 30, 2026-10-30 (detached, from a live evalsuite session): the REF2b reference rows + decision 4's
# same-scores test (docs/PREREG_ref2b_2026-10-30.md). Decodes the 68 rows of results/ref2b_keys.txt (65 that reach a
# bucketed AR step + 3 spot checks) with variant ref2b (snap/leap-faff8e5, key buckets on), in chunks that start no
# new row after 12 min, each under one queued quiet lock (cli.gpu: memory gate 34 + 4 GB, swap guard, 17-min stop).
# Then tools/ref2b_assemble.py (the checks, then the copy of the other 142 rows) -> results/ref2b_regen.txt; on a
# failed check the runner stops (exit 5) with nothing copied. Then the reports: ref2b vs ref.r2 (the verdict) and
# ref2b vs ref2 (information), and the marker results/REF2B_DONE. BigMoE runs from a frozen copy
# (snap/bigmoe-6c64148: bigmoe/ + ext/build + ext/build_next, env.BIGMOE_CODE; PREREG amendment 1), as R-40's runner 25
# did. Code pin: the two snapshots, evalsuite/, MLX / mlx_lm / mtplx; a change stops the runner (exit 7). No chunk while
# soft_end.sh ramps, PAUSE exists, the lid is closed with no external display, or < 34 GB is free. Deadline Sat
# 2026-10-31 08:00. Stop it by pid only, and its cli child too (a killed runner leaves a queued chunk behind).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day30_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-31 08:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"   # frozen BigMoE code (env.BIGMOE_CODE)
PY=.venv/bin/python
KEYS_SHA=e405642d19cb                            # results/ref2b_keys.txt as pre-registered (sha256[:12])
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { if [ -f results/ref2b/decode.jsonl ]; then wc -l < results/ref2b/decode.jsonl | tr -d ' '; else echo 0; fi; }
free_gb() {   # NO-SWAP: free + speculative + purgeable + file-backed pages (vm_stat; orchestrator 10-30 23:5x, RULES.md:
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'
}              # 'inactive' holds anonymous pages only swap can free; shared lock: peak + 8 GB)
SWAPTRIPS=0
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut \
        || [ "$(free_gb)" -lt 0 ]; do                   # amendment 6: queue first; cli gates memory after the grant
    waitlog "waiting: soft_end.sh, PAUSE, a closed lid or < 30 GB no-swap ($(free_gb) GB)"
    sleep 30
  done
}
pin() {
  { shasum snap/leap-faff8e5/*.py
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    find evalsuite -name '*.py' ! -name live16b.py ! -path '*__pycache__*' | sort | xargs shasum   # live16b: R-56b's stage
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
PIN0=$(pin)
checkpin() {
  local p; p=$(pin)
  [ "$p" = "$PIN0" ] || { say "code changed (the REF2b / BigMoE snapshots, evalsuite, MLX / mtplx): pin $PIN0 -> $p: runner stops"; return 1; }
}
NOPROG=0
say "runner 30 start (pid $$); code pin $PIN0"
[ "$(shasum -a 256 results/ref2b_keys.txt | cut -c1-12)" = "$KEYS_SHA" ] || { say "results/ref2b_keys.txt is not the pre-registered list: runner stops"; exit 2; }
while [ "$(nrows)" -lt 68 ]; do
  b=$(nrows)
  calm
  checkpin || exit 7
  [ "$(left)" -gt 1500 ] || { say "deadline near: runner stops at $b/68 rows"; exit 6; }
  ./bin/evalsuite gpu decode ref2b --keys results/ref2b_keys.txt --sets math,ifeval,humaneval --chunk-minutes 6 \
    --C 96 --need 22 --minutes 15 >> "$LOG" 2>&1; rc=$?   # amendment 6: a quiet ticket per chunk (FIFO), C 96 (~22 GB)
  say "ref2b: chunk rc=$rc ($(nrows)/68 rows)"
  if [ "$rc" = 75 ]; then say "yielded to a quiet ticket: asking again in 2 min"; sleep 120; continue; fi
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then   # amendment 4: the stage was stopped; wait, then the gates again
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min, then the memory gate again"; sleep 120; continue
  fi
  if [ "$(nrows)" -le "$b" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
done
$PY tools/ref2b_assemble.py > results/ref2b_regen.txt 2>&1; rc=$?
say "assemble rc=$rc: $(tail -1 results/ref2b_regen.txt)"
[ "$rc" = 0 ] || { say "a REF2b check failed: nothing copied, no report: runner stops"; exit 5; }
./bin/evalsuite report --variant ref2b --decode --ref ref.r2 > results/report_ref2b_vs_ref.r2.txt 2>&1
say "report ref2b vs ref.r2 (verdict): $(tail -1 results/report_ref2b_vs_ref.r2.txt)"
./bin/evalsuite report --variant ref2b --decode --ref ref2 > results/report_ref2b_vs_ref2.txt 2>&1
say "report ref2b vs ref2 (information): $(tail -1 results/report_ref2b_vs_ref2.txt)"
touch results/REF2B_DONE
say "runner 30 end"
