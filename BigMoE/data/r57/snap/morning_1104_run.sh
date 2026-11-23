#!/bin/bash
# The morning block, Wed 4 Nov (the orchestrator 06:2x: the GPU is BigMoE's from ~08:30, after FLOOR and Breakthrough's map
# batch, to 12:50; Kamen wants the day's tests done and prepared before 13:45). Started by hand on Breakthrough's "map batch
# done". One quiet ticket per block (turn45 with no 15-min gaps between my own blocks), in order:
#   A  exactness, C 160: 3b scripts/pr2_run.sh (~4 min) -> 5a scripts/u3_smoke.sh at C 160 (~10 min; its fresh u3c re-checks
#      item 3's 10 HumanEval at full length against the night's u3c_160)
#   0  CPU, no ticket: scripts/item0_live.sh (the layout patch live + ext/build rebuilt; Fit12 06:2x: YES). Not started while
#      another process loads a ~/BigMoE ext build: retried every 2 min for 20 min, then block C is skipped
#   B  timing: 1 scripts/w1b_24.sh (~15 min, <= 24 GB) -> 2 scripts/w1_lever1.sh (~50 min; C 224 if NO-SWAP >= 44 at its
#      start, else C 160: the WED_TESTS rule)
#   C  needs 0: 4 scripts/mix_e2e.sh at C 160 (~20 min) -> 5 scripts/default45_wed.sh, the combined default, if it exists
# Nothing starts that cannot end by 12:55: a block starts only before its latest start (A 12:15, B 11:15, C 11:50) and every
# runner stops starting arms at 12:30. The 2 GB swap rule runs per block (C45_SWAP0). Fresh frozen harness copies per block.
# Log: data/morning_1104.log (the watcher reads only this log).
# usage (a harness background task, not nohup: turn45's queue must stay attached): bash scripts/morning_1104.sh
#        (it runs a frozen copy of itself, so this file can change meanwhile)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
LOG="$HOME/BigMoE/data/morning_1104.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
before() { [ "$(date +%H%M)" -lt "$1" ]; }
SELF=data/r57/snap/morning_1104_run.sh
freeze() { cp scripts/r57_screen.py "data/r57/snap/r57_screen_$1.py" && cp scripts/c45_runner.sh "data/r57/snap/c45_runner_$1.sh"; }
# the orchestrator 07:2x: every arm must END by ~12:55 (Fit12's T3R2 goes quiet at 13:05, hard stop 13:40, no slack), so no
# arm starts after 12:30 (the longest arm here is < 25 min; item 5 only starts before 12:00)
export E2E_DEADLINE=1230 U3_DEADLINE=1230 W1_DEADLINE=1230 PR2_DEADLINE=1230 D45_DEADLINE=1230
case "$1" in
blockA)
  export C45_SWAP0=$(swap_gb)
  say "A: 3b pre-route re-run (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  bash scripts/pr2_run.sh; say "A: 3b rc=$? ($(tail -1 data/r57/pr2_160_preroute.txt 2>/dev/null | cut -c1-150))"
  grep -h "Verdict" data/r57/pr2_160_preroute.txt >> "$LOG" 2>/dev/null
  for f in data/r57/u3c_160_*.json data/r57/u3c_160_screen.txt; do [ -e "$f" ] && [ ! -e "$f.night" ] && cp "$f" "$f.night"; done
  freeze u3
  U3_C=160 bash scripts/u3_smoke.sh; say "A: 5a rc=$? ($(grep -E 'u3n vs u3c|U3 rc|stopped' data/u3/u3.log | tail -2 | tr '\n' ' ' | cut -c1-200))"
  .venv/bin/python -I -c "
import json
a = json.load(open('data/r57/u3c_160_tokens.json.night')); b = json.load(open('data/r57/u3c_160_tokens.json'))
r = json.load(open('data/ab/ref_tokens.json'))
ks = [k for k in a if k in r]
bad = [k for k in ks if b.get(k, {}).get('gate+t') != r[k]['gate+t']]
print('item 3 second check: fresh u3c_160 vs data/ab/ref_tokens.json on %d keys, full length: %s' % (len(ks), 'ALL EQUAL' if not bad else 'DIFFER ' + str(bad)))
" >> "$LOG" 2>&1
  say "A: end" ;;
blockB)
  export C45_SWAP0=$(swap_gb)
  say "B: 1 W1B 24 GB (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  bash scripts/w1b_24.sh; say "B: 1 rc=$?"
  a=$(noswap); W1C=160; [ "$a" -ge 44 ] && W1C=224
  freeze w1
  say "B: 2 lever 1 at C $W1C (NO-SWAP $a GB)"
  W1_C=$W1C bash scripts/w1_lever1.sh; say "B: 2 rc=$? ($(grep -c 'rc=0' data/w1/w1.log) ok lines in data/w1/w1.log)"
  say "B: end" ;;
blockC)
  export C45_SWAP0=$(swap_gb)
  freeze e2e
  say "C: 4 mix_e2e at C 160 (swap base $C45_SWAP0 GB, NO-SWAP $(noswap) GB)"
  E2E_C=160 bash scripts/mix_e2e.sh; say "C: 4 rc=$? ($(grep -E '^.*(mix|mixc): ' data/e2e/e2e.log | tail -2 | sed 's/^[0-9-]* [0-9:]* //' | tr '\n' ' ' | cut -c1-200))"
  if [ -f scripts/default45_wed.sh ] && before 1200; then
    freeze d45
    bash scripts/default45_wed.sh; say "C: 5 rc=$?"
  else
    say "C: 5 not run ($([ -f scripts/default45_wed.sh ] && echo 'past 12:00' || echo 'no scripts/default45_wed.sh'))"
  fi
  say "C: end" ;;
"")
  cp "$0" "$SELF" 2>/dev/null; [ "$0" = "$SELF" ] || exec /bin/bash "$SELF" main
  ;;
esac
[ "$1" = main ] || exit 0
T45() { GAP0=0 GAP=0 MARGIN=4 NOTAFTER=$1 bash scripts/turn45.sh "$2" "$3" "$4" /bin/bash "$SELF" "$5"; }
say "MORNING START (pid $$; NO-SWAP $(noswap) GB, swap $(swap_gb) GB)"
if before 1215; then
  T45 12:15 33 40 "BigMoE morning A: 3b pre-route re-run + 5a U3 smoke, C 160, MEM 33 GB" blockA; say "block A rc=$?"
else say "block A not started (past 12:15)"; fi
I0=9
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  bash scripts/item0_live.sh >> "$LOG" 2>&1; I0=$?
  [ $I0 = 2 ] && [ $i -lt 11 ] && grep -q "is loaded by" <(tail -3 data/item0.log) && { say "item 0 waits (an ext build is loaded): retry in 2 min"; sleep 120; continue; }
  break
done
say "ITEM 0 rc=$I0 ($(grep -E 'ITEM 0 LIVE|NOT STARTED|FAILED|RESTORED' data/item0.log | tail -1 | cut -c16-220))"
if before 1115; then
  T45 11:15 33 120 "BigMoE morning B: W1B 24 GB + lever 1 at 45 GB, MEM 44 GB" blockB; say "block B rc=$?"
else say "block B not started (past 11:15)"; fi
if [ $I0 = 0 ] && before 1150; then
  T45 11:50 33 70 "BigMoE morning C: mix_e2e + the combined default, MEM 33 GB" blockC; say "block C rc=$?"
else say "block C not started (item 0 rc $I0 or past 11:50)"; fi
say "MORNING DONE (NO-SWAP $(noswap) GB, swap $(swap_gb) GB)"
