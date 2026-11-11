#!/bin/bash
# The morning's second part, Wed 4 Nov, after scripts/morning_1104.sh (A done: 3b BETWEEN, 5a u3n == u3c, item 3 ALL
# EQUAL; B: 2 lever 1 ran). Two things went wrong there and are fixed before this starts:
#   - item 0 failed its first test (rc 3, files restored): the patched tier.py's native+p11+pad path needs set_direct_split,
#     which build_nat lacks; scripts/item0_live.sh now tests on build_w1 (the current source). Item 0 is run by hand
#     before this script (CPU), so block C finds the patch live and ext/build rebuilt;
#   - item 1 (W1B 24 GB) crashed at its 12-token warm-up: tier.py's block takes 9-16-row calls since 11-01 (01f652c) and
#     assumes TIER_COMBINE's combined output, but the 24 GB stack's MAXK 80 sends them to the sync prefill path
#     (per-expert output). Fixed in tier.py (such calls take the stock block, as before 11-01; no working path changes).
# In order, one quiet ticket per block (no gaps between my own blocks), every arm ending by ~12:55 (no arm starts after
# 12:30, the orchestrator 07:2x):
#   B1  1 scripts/w1b_24.sh (the sprint's 24 GB baseline row; build_m01 rebuilt 06:22)
#   C   4 scripts/mix_e2e.sh at C 160 -> 5 scripts/default45_wed.sh (both need item 0 live: rc 0 in data/item0.log)
#   D   5b scripts/u3c_wed.sh (its own u3n2 gate on the rebuilt ext/build first)
# Log: data/morning2_1104.log. usage (a harness background task, not nohup): bash scripts/morning2_1104.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
LOG="$HOME/BigMoE/data/morning2_1104.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
before() { [ "$(date +%H%M)" -lt "$1" ]; }
SELF=data/r57/snap/morning2_1104_run.sh
freeze() { cp scripts/r57_screen.py "data/r57/snap/r57_screen_$1.py" && cp scripts/c45_runner.sh "data/r57/snap/c45_runner_$1.sh"; }
export E2E_DEADLINE=1230 U3C_DEADLINE=1230 D45_DEADLINE=1230
live0() { grep -q "ITEM 0 LIVE" data/item0.log 2>/dev/null && [ "$(grep -E 'ITEM 0 LIVE|item 0 start' data/item0.log | tail -1 | grep -c 'ITEM 0 LIVE')" = 1 ]; }
case "$1" in
blockB1)
  export C45_SWAP0=$(swap_gb)
  say "B1: 1 W1B 24 GB (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  bash scripts/w1b_24.sh; say "B1: 1 rc=$? ($(grep -E 'done w24|STOP' data/w1b24/w1b24.log | tail -2 | sed 's/^[0-9-]* [0-9:]* //' | tr '\n' ' '))"
  say "B1: end" ;;
blockC)
  export C45_SWAP0=$(swap_gb)
  freeze e2e
  say "C: 4 mix_e2e at C 160 (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  E2E_C=160 bash scripts/mix_e2e.sh; say "C: 4 rc=$? ($(grep -E '^.*(mix|mixc): ' data/e2e/e2e.log | tail -2 | sed 's/^[0-9-]* [0-9:]* //' | tr '\n' ' ' | cut -c1-200))"
  if before 1200; then
    freeze d45
    bash scripts/default45_wed.sh; say "C: 5 rc=$? ($(grep Verdict data/d45/d45_report.txt 2>/dev/null | cut -c1-120))"
  else
    say "C: 5 not run (past 12:00)"
  fi
  say "C: end" ;;
blockD)
  export C45_SWAP0=$(swap_gb)
  say "D: 5b U3C len on code (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  bash scripts/u3c_wed.sh; say "D: 5b rc=$? ($(grep Verdict data/u3c/u3c_report.txt 2>/dev/null | cut -c1-120))"
  say "D: end" ;;
"")
  cp "$0" "$SELF" 2>/dev/null; [ "$0" = "$SELF" ] || exec /bin/bash "$SELF" main
  ;;
esac
[ "$1" = main ] || exit 0
T45() { GAP0=0 GAP=0 MARGIN=4 NOTAFTER=$1 bash scripts/turn45.sh "$2" "$3" "$4" /bin/bash "$SELF" "$5"; }
say "MORNING2 START (pid $$; NO-SWAP $(noswap) GB, swap $(swap_gb) GB; item 0 live: $(live0 && echo yes || echo NO))"
if before 1145; then
  T45 11:45 26 30 "BigMoE morning B1: W1B 24 GB (<= 24 GB), MEM 26 GB" blockB1; say "block B1 rc=$?"
else say "block B1 not started (past 11:45)"; fi
if live0 && before 1150; then
  T45 11:50 33 70 "BigMoE morning C: mix_e2e + the combined default, MEM 33 GB" blockC; say "block C rc=$?"
else say "block C not started (item 0 not live, or past 11:50)"; fi
if before 1200; then
  T45 12:00 33 35 "BigMoE 5b: U3C len on code vs U3 (FAST), MEM 33-38 GB" blockD; say "block D rc=$?"
else say "block D not started (past 12:00)"; fi
say "MORNING2 DONE (NO-SWAP $(noswap) GB, swap $(swap_gb) GB)"
