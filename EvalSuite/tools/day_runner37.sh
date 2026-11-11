#!/bin/bash
# EvalSuite runner 37, 2026-11-01 (detached, from a live evalsuite session): PMIN12 (docs/PREREG_pmin12_2026-11-01.md).
# Arms b12llff4p (B12_PMIN 0.5) and b12llff4 (the control) on bt_b12 8db71cc537 (v3p, asserted at load), decision 5's 239
# rows: the 210 core rows in ~9-min chunks under shared locks (cli --share --need 16, arms alternating; >= 20 GB free is
# checked here first), then runner 24's 29 agent rows in ONE process per arm (the creep test; ~17 min each).
# Phase A (day): no chunk starts after 22:45 and each chunk's EVALSUITE_DEADLINE is 22:58: no lock of this session is held
# or asked for when runner 36 queues at 23:05 (cli.gpu's `peer unlock` would cancel its ticket). An agent process starts
# in Phase A only with >= 55 min left (shared lock, --minutes 40). Amendment 1 (relaunch Sun 14:51): before it, the agent
# processes waited for Phase B. With all 478 rows done, the runner goes straight to the reports.
# Phase B: once runner 36 has exited, what is left (agent processes under a quiet ticket); deadline Mon 2026-11-02 12:00.
# Then the reports (b12llff4p vs b12llff4 = the verdict; vs b12llff = information), tools/pmin12.py -> results/pmin12.txt
# and results/PMIN12_DONE. Code pin: bt_b12.py, Breakthrough's v3p engine, SpecDecode, evalsuite/, tools/d4.py, pmin12.py
# and r41c.py, MLX / mtplx; a change stops the runner (exit 7). Stop it by pid only, and its cli child too. Before a
# relaunch of runner 36, stop this runner if it is in Phase B.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day37_runner.log
A_END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 22:45" +%s)
A_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 22:58" +%s)
B_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 12:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset EVALSUITE_BIGMOE_CODE EVALSUITE_STAGE_ENV B12_PMIN F12_R53A
PY=.venv/bin/python
P=b12llff4p; C=b12llff4
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
nset() { if [ -f "results/$1/decode.jsonl" ]; then grep -c "\"set\": \"$2\"" "results/$1/decode.jsonl"; else echo 0; fi; }
ncore() { echo $(( $(nset "$1" math) + $(nset "$1" ifeval) + $(nset "$1" humaneval) )); }
nall() { echo $(( $(ncore $P) + $(ncore $C) + $(nset $P agent) + $(nset $C agent) )); }
free_gb() {   # NO-SWAP (cli.free_gb): free + speculative + purgeable + file-backed pages
  vm_stat | awk '/page size of/{ps=$8} /^Pages free:|^Pages speculative:|^Pages purgeable:|^File-backed pages:/{gsub(/\./,"",$NF); s+=$NF} END{printf "%d", s*ps/1e9}'
}
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
    waitlog "waiting: soft_end.sh, PAUSE or a closed lid"
    sleep 30
  done
}
pin() {
  { shasum -a 256 "$HOME/Breakthrough/scripts/bt_b12.py"
    (cd "$HOME/Breakthrough" && find v3p -type f \( -name '*.py' -o -name '*.so' -o -name '*.metal' \) ! -path '*__pycache__*' \
       | sort | xargs shasum)
    git -C "$HOME/SpecDecode" rev-parse HEAD; git -C "$HOME/SpecDecode" status --porcelain --untracked-files=no
    shasum tools/d4.py tools/pmin12.py tools/r41c.py
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
in_a() { [ "$PHASE" = A ] && [ "$(date +%s)" -lt "$A_END" ]; }
next_job() {   # the next chunk: core rows (arms alternating) until both have 210; then the agent processes
  local p c
  p=$(ncore $P); c=$(ncore $C)
  if [ "$p" -lt 210 ] || [ "$c" -lt 210 ]; then
    if [ "$p" -le "$c" ] && [ "$p" -lt 210 ] || [ "$c" -ge 210 ]; then v=$P; else v=$C; fi
    echo "share $v --sets math,ifeval,humaneval --chunk-minutes 9 --minutes 14"; return
  fi
  for v in $P $C; do
    [ "$(nset $v agent)" -lt 29 ] || continue
    if [ "$PHASE" = B ]; then echo "quiet $v --sets agent --max-rows 29 --minutes 40"
    elif [ $(( A_DEADLINE - $(date +%s) )) -ge 3300 ]; then echo "share $v --sets agent --max-rows 29 --minutes 40"; fi
    return
  done
}
SWAPTRIPS=0; NOPROG=0
PIN0=$(pin); PHASE=A
say "runner 37 start (pid $$); code pin $PIN0; Phase A until 22:45 (shared locks while >= 20 GB is free)"
while :; do
  if [ "$PHASE" = A ] && ! in_a; then
    say "Phase A over ($(nall) of 478 rows): waiting for runner 36 to exit"
    while pgrep -f 'tools/day_runner36\.sh' >/dev/null; do sleep 60; done
    PHASE=B; say "Phase B (runner 36 is gone)"
  fi
  job=$(next_job)
  [ -n "$job" ] || { [ "$PHASE" = A ] && [ "$(nall)" -lt 478 ] && { sleep 60; continue; }; break; }
  set -- $job; mode=$1; v=$2; shift 2
  calm
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  if [ "$PHASE" = A ]; then dl=$A_DEADLINE; else dl=$B_DEADLINE; fi
  [ $(( dl - $(date +%s) )) -gt 1200 ] || { [ "$PHASE" = A ] && { sleep 60; continue; }; say "deadline near: runner stops at $(nall)/478 rows"; exit 6; }
  if [ "$(free_gb)" -lt 20 ]; then
    waitlog "waiting: $(free_gb) GB free, the run needs 20 (peak ~12 + 8)"; sleep 120; continue
  fi
  b=$(nall)
  if [ "$mode" = share ]; then
    EVALSUITE_DEADLINE=$dl ./bin/evalsuite gpu decode "$v" "$@" --share --need 16 >> "$LOG" 2>&1; rc=$?
  else
    EVALSUITE_DEADLINE=$dl ./bin/evalsuite gpu decode "$v" "$@" --need 16 >> "$LOG" 2>&1; rc=$?
  fi
  n=$(nall)
  say "$v ($mode, Phase $PHASE): chunk rc=$rc; rows: $P core $(ncore $P) agent $(nset $P agent), $C core $(ncore $C) agent $(nset $C agent)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min"; sleep 120; continue
  fi
  if [ "$rc" != 0 ] && [ "$n" -le "$b" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  if [[ "$*" == *"--sets agent"* ]] && [ "$(nset $v agent)" -lt 29 ]; then
    say "$v: agent process ended at $(nset $v agent)/29 rows: not one process any more (the creep test is weaker; see the RESULT)"
  fi
  if [ "$n" -le "$b" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 4 ] && { say "four chunks without a new row: runner stops"; exit 3; }
    sleep 120
  else
    NOPROG=0
  fi
done
./bin/evalsuite report --variant $P --decode --ref $C > results/report_${P}_vs_${C}.txt 2>&1
./bin/evalsuite report --variant $P --decode --ref b12llff > results/report_${P}_vs_b12llff.txt 2>&1
./bin/evalsuite report --variant $C --decode --ref b12llff > results/report_${C}_vs_b12llff.txt 2>&1
$PY tools/pmin12.py > results/pmin12.txt 2>&1
say "$(grep -m1 '^PMIN12: [A-Z]' results/pmin12.txt)"
touch results/PMIN12_DONE
say "runner 37 end"
