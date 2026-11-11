#!/bin/bash
# EvalSuite runner 25, 2026-10-28 (detached, from a live evalsuite session): R-40 REF2 vs today's reference,
# docs/PREREG_ref2_2026-10-28.md. Arms (args, default "ref2"): ref2 = the reference harness + Leap's ref2.apply() +
# ref2.generate; ref.r2 = today's stock baseline, only if step 0's identity check failed. Core 210 rows (math, ifeval,
# humaneval). Time chunks: a chunk starts no new row after 9 min and cli.gpu stops it at 16 min; one process + one
# queued quiet lock per chunk, released between chunks; the arms alternate chunk by chunk. No chunk while soft_end.sh
# ramps, PAUSE exists, or the lid is closed with no external display. Deadline Thu 2026-10-29 12:00. Stop it by pid only.
# ref2 chunks wait for results/REF2_GO (touched when Leap confirms its streamed check). BigMoE code: a frozen copy,
# snap/bigmoe-d0daa68 (= the identity check's code), so BigMoE's daily changes cannot split the arms; REF2: a frozen
# copy too, snap/leap-86ca915 (model.REF2_CODE). Code pin: both copies; a changed pin stops the runner.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day25_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-29 12:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-d0daa68"   # frozen BigMoE code for both arms (env.BIGMOE_CODE)
SETS=math,ifeval,humaneval
TARGET=210
ARMS="${*:-ref2}"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
rows() { if [ -f "results/$1/decode.jsonl" ]; then wc -l < "results/$1/decode.jsonl" | tr -d ' '; else echo 0; fi; }
lid_shut() {   # lid closed and no external display (lid_guard.sh's test): research holds
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
calm() {
  local n=0
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
    [ $((n % 20)) = 0 ] && say "waiting: soft_end.sh, PAUSE or a closed lid"
    n=$((n + 1)); sleep 30
  done
}
pin_b() { (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum) 2>&1 \
           | shasum | cut -c1-12; }   # the frozen copy must not change
pin_l() { (cd "$HOME/EvalSuite/snap/leap-86ca915" && shasum ref2.py leapk.py qmm16.py qmmg25.py gmr24.py) 2>&1 | shasum \
           | cut -c1-12; }   # the frozen REF2 (model.REF2_CODE) must not change
PIN_B0=$(pin_b); PIN_L0=""
checkpin() {   # checkpin VARIANT: 1 = the code changed since the run began
  local b; b=$(pin_b)
  [ "$b" = "$PIN_B0" ] || { say "BigMoE code changed (bigmoe/, ext/build .so): pin $PIN_B0 -> $b: runner stops"; return 1; }
  [ "$1" = ref2 ] || return 0
  if [ -z "$PIN_L0" ]; then PIN_L0=$(pin_l); say "REF2 files pinned at the first ref2 chunk: $PIN_L0"; return 0; fi
  [ "$(pin_l)" = "$PIN_L0" ] || { say "REF2 files changed: pin $PIN_L0 -> $(pin_l): runner stops"; return 1; }
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
chunk() {   # chunk VARIANT: one process, one quiet lock; returns 1 if the runner must stop
  local v=$1 rc before
  before=$(rows "$v")
  calm
  ./bin/evalsuite gpu decode "$v" --sets $SETS --chunk-minutes 9 --minutes 16 >> "$LOG" 2>&1; rc=$?
  say "$v: chunk rc=$rc ($before -> $(rows "$v") rows)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; return 1
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then say "swap guard: runner stops"; return 1; fi
  local key n
  key=$(echo "$v" | tr -c 'a-zA-Z0-9\n' '_')    # bash 3.2 (macOS): no associative arrays
  if [ "$(rows "$v")" -le "$before" ]; then
    eval "n=\$(( \${NOPROG_$key:-0} + 1 ))"; eval "NOPROG_$key=$n"
    [ "$n" -ge 3 ] && { say "$v: three chunks without a new row: runner stops"; return 1; }
    sleep 60
  else
    eval "NOPROG_$key=0"
  fi
  return 0
}

say "runner 25 start (pid $$), arms: $ARMS; BigMoE pin $PIN_B0"
while [ "$(left)" -gt 900 ]; do
  pending=0; ran=0
  for v in $ARMS; do
    [ "$(rows "$v")" -ge "$TARGET" ] && continue
    pending=1
    if [ "$v" = ref2 ] && [ ! -e results/REF2_GO ]; then waitlog "ref2 waits for Leap's go (results/REF2_GO)"; continue; fi
    checkpin "$v" || exit 7
    chunk "$v" || exit 3
    ran=1
    [ "$(left)" -gt 900 ] || break
  done
  [ "$pending" = 0 ] && break
  [ "$ran" = 0 ] && sleep 60
done
for v in $ARMS; do
  [ "$(rows "$v")" -ge "$TARGET" ] || { say "deadline: $v at $(rows "$v") rows: runner stops"; exit 6; }
done
say "all arms at $TARGET rows"
base=ref; [[ " $ARMS " == *" ref.r2 "* ]] && base=ref.r2
./bin/evalsuite report --variant ref2 --decode > results/report_ref2.txt 2>&1
say "report ref2 vs ref: $(tail -1 results/report_ref2.txt)"
if [ "$base" = ref.r2 ]; then
  ./bin/evalsuite report --variant ref2 --decode --ref ref.r2 > results/report_ref2_vs_ref.r2.txt 2>&1
  say "report ref2 vs ref.r2: $(tail -1 results/report_ref2_vs_ref.r2.txt)"
  ./bin/evalsuite report --variant ref.r2 --decode > results/report_ref.r2.txt 2>&1
  say "report ref.r2 vs ref: $(tail -1 results/report_ref.r2.txt)"
  ./bin/evalsuite report --variant floor --decode --ref ref.r2 > results/report_floor_vs_ref.r2.txt 2>&1
  say "report floor vs ref.r2 (noise yardstick): $(tail -1 results/report_floor_vs_ref.r2.txt)"
fi
say "same tokens: $(.venv/bin/python tools/same_tokens.py ref2 $base 2>&1 | tail -1)"
.venv/bin/python tools/ref2_summary.py $base > results/ref2_summary.txt 2>&1 && say "summary: results/ref2_summary.txt"
say "runner 25 end"
