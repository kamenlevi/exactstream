#!/bin/bash
# t1c_resume_follower4.sh: B2 -> V2 of 1c-tiers AFTER BigMoE's g1b key test (PREREG_lossygate amendment 4i; NewIdeas'
# order 17:3x: Breakthrough's map screen, then BigMoE's g1b key test, then B2 -> V2). Runner 65 runs mem_busy only at
# the top of gate(), not inside its memory loop, so a runner waiting for memory would launch in a gap between the key
# test's nine processes. The waiting B2 (chain2 tag mon4) was therefore stopped at 17:33:40 (TERM), and no day_runner6x
# process exists until the key test has ended. This follower:
#   1. waits until a key-test process (scripts/g1_follow.sh or scripts/g1_keytest.py) has been SEEN, is gone on two reads
#      30 s apart, and BigMoE's end marker (data/r57/G1A_KEYTEST_DONE, written on every end; a re-arm writes none) was
#      written at or after the first sighting;
#   2. then the memory gate: NO-SWAP >= 43 GB on two reads 30 s apart within CHAIN_FB_MIN (10) min of that end runs
#      tools/t1c_chain2.sh tag mon5 on runner 65 (gate 43); otherwise amendment 4e's fallback, the same chain on
#      tools/day_runner66.sh with CARD_GATE=42 (NewIdeas 17:3x);
#   3. no start after CHAIN_LASTSTART (20:00): then 4d's Tuesday follower is the route.
# results/T1C_RESF2_END is written on EVERY end of this follower, and only then: it is the 1c-tiers end others wait for.
# Stop it by pid only, with a plain kill (TERM): it stops the chain it owns.
#   CHAIN_MYPID=<this session's Claude pid> tools/t1c_resume_follower4.sh
# Tests only: CHAIN_SCRIPT, CHAIN_KT (the marker), CHAIN_KTPAT (the key-test process pattern), CHAIN_FREEGB (a command
# that prints NO-SWAP GB), CHAIN_POLL (s between reads), CHAIN_PAIR (s between a read pair), CHAIN_LASTSTART.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/t1c_chain.log; END=results/T1C_RESF2_END
MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}; CH=${CHAIN_SCRIPT:-tools/t1c_chain2.sh}
KT=${CHAIN_KT:-$HOME/BigMoE/data/r57/G1A_KEYTEST_DONE}; KTPAT=${CHAIN_KTPAT:-scripts/g1_follo[w]\.sh|scripts/g1_keytes[t]\.py}
FDAY=${CHAIN_DAY:-2026-11-16}; CUTS=${CHAIN_CUTOFF:-2026-11-16 22:30}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
LAST=$(ep "${CHAIN_LASTSTART:-2026-11-16 20:00}"); FB=${CHAIN_FB_MIN:-10}; POLL=${CHAIN_POLL:-30}; PAIR=${CHAIN_PAIR:-30}
say() { echo "$(date '+%m-%d %H:%M:%S') [resf4] $*" >> "$L"; }
freegb() {
  if [ -n "${CHAIN_FREEGB:-}" ]; then $CHAIN_FREEGB
  else .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; fi
}
STATUS="waiting for BigMoE's key test"; CPID=""
finish() {
  local rc=$?
  if [ -n "$CPID" ] && ps -o command= -p "$CPID" 2>/dev/null | grep -q t1c_chain; then   # stopped mid-run
    kill -TERM "$CPID" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$CPID" 2>/dev/null || break; sleep 1; done
  fi
  echo "T1C RESF2 END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
ktup() { pgrep -f -- "$KTPAT" > /dev/null; }
late() { [ "$(date +%s)" -ge "$LAST" ]; }
say "follower4 (pid $$; session $MYPID): B2 -> V2 after BigMoE's key test: a key-test process seen, then gone (two reads ${PAIR} s apart) and $(basename "$KT") written after the sighting; then NO-SWAP >= 43 twice (runner 65) within $FB min, else runner 66 at CARD_GATE=42; no start after $(date -r "$LAST" '+%m-%d %H:%M'); one end marker $END"
SEEN=0; n=0
while :; do
  late && { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M') (the key test not ended): no run; 4d's Tuesday follower is the route"; say "$STATUS"; exit 6; }
  if ktup; then
    [ "$SEEN" = 0 ] && { SEEN=$(date +%s); say "the key test is up: $(pgrep -fl -- "$KTPAT" | head -2 | cut -c1-120 | tr '\n' ' ')"; }
  elif [ "$SEEN" != 0 ] && [ -e "$KT" ] && [ "$(stat -f %m "$KT")" -ge "$SEEN" ]; then
    sleep "$PAIR"; ktup || break                                  # read twice: still gone
  fi
  n=$((n + 1)); [ $((n % 20)) = 0 ] && say "waiting: $([ "$SEEN" = 0 ] && echo 'no key-test process seen yet' || echo 'the key test runs or ended without its marker')"
  sleep "$POLL"
done
KTEND=$(date +%s); say "the key test ended: $(head -1 "$KT" | cut -c1-200)"
STATUS="the memory gate after the key test"; RUN=65
while :; do
  late && { STATUS="no start by $(date -r "$LAST" '+%m-%d %H:%M') (the memory gate): no run; 4d's Tuesday follower is the route"; say "$STATUS"; exit 6; }
  a=$(freegb)
  if [ "${a:-0}" -ge 43 ]; then sleep "$PAIR"; b=$(freegb); [ "${b:-0}" -ge 43 ] && { say "NO-SWAP $a / $b GB >= 43: runner 65 (gate 43)"; break; }; fi
  [ "$(date +%s)" -ge $(( KTEND + FB * 60 )) ] && { RUN=66; say "43 not open $FB min after the key test (NO-SWAP $a GB): amendment 4e's fallback, runner 66 at CARD_GATE=42"; break; }
  sleep "$POLL"
done
STATUS="the chain (tag mon5) on runner $RUN: B2 -> V2"
if [ "$RUN" = 66 ]; then
  CHAIN_RUNNER=tools/day_runner66.sh CARD_GATE=42 CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$CUTS" CHAIN_NOYIELD=1 CHAIN_TAG=mon5 CHAIN_MYPID=$MYPID "$CH" &
else
  CHAIN_PHASE=wed CHAIN_DAY=$FDAY CHAIN_CUTOFF="$CUTS" CHAIN_NOYIELD=1 CHAIN_TAG=mon5 CHAIN_MYPID=$MYPID "$CH" &
fi
CPID=$!; say "the chain (tag mon5, runner $RUN) runs as pid $CPID"; wait "$CPID"; rc=$?; CPID=""
STATUS="done: the chain mon5 (runner $RUN) ended rc $rc ($(head -c 160 results/T1C_MON5_END 2>/dev/null | tr '\n' ' '))"
exit "$rc"
