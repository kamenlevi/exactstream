#!/bin/bash
# night1117_chain.sh (Leap, Mon 16 Nov 2026 21:0x). Tonight's Leap slot, chained after BigMoE's post-M4 key test.
#   BigMoE 20:5x: "please chain try B / E14 on G1A_KEYTEST_DONE (newer than your arming), not on a fixed 00:30".
#   The orchestrator 20:4x: the ~39 GB GPU-busy run stays gated until the night window; lower-RAM tests while Kamen
#   is at the Mac.
# Order: M4 (23:05 -> 00:25), BigMoE's key test (~12-15 min, share lock MEM 45 GB), try B, E14, then EvalSuite's
#   night1117 chain (it gates on E14's marker). try B = the c25 GPU busy + CPU split, TAG gpubB. E14 = E1-E4.
#
#   1. Wait, holding nothing, for the key test's end marker ~/BigMoE/data/r57/G1A_KEYTEST_DONE, written after arming.
#      A STOPPED marker counts only when no BigMoE g1 waiter script runs on two reads G s apart; else it was a stop
#      before a re-arm, and the chain waits for the next marker. Nothing starts before 23:30 (the night window:
#      apps quit 23:00).
#   2. try B runs only if it can start by 01:46. Otherwise it is skipped: slot #1 (Tue 09:30) records the same
#      profile + CPU split.
#      - First a pre-wait (to 01:46) for: Kamen away >= 900 s, no PAUSE, the lid open, no other process over 16 GB,
#        and NO-SWAP >= 39 on two reads G s apart.
#      - Then the frozen runner run_c25_gpubB.sh (md5 1018e07b), START now, hard end min(now + 24 min, 02:10).
#        It is run_c25_gpub.sh (c621bb42, gpubA's) plus the away gate (>= 900 s before and after its post-grant
#        memory reads; the orchestrator 21:0x).
#      - The runner does the rest: its own memory wait, the head test, a quiet ticket, a 20 min cap from the grant,
#        and the marker C25_gpubB_DONE on any end.
#   3. E14 goes from max(now, Tue 00:30 - 30 s).
#      - The same pre-wait, up to 02:10.
#      - Then exec the frozen run_e14.sh (md5 fe3bf178), START max(now, Tue 00:30), hard end 03:00.
#        It is scripts/week98/e14/run_e14.sh (2e0b985b, armed since Mon 00:33) plus the same away gate.
#      - The START matters: EvalSuite counts E14's marker only after a run line dated 11-17.
#      - If the pre-wait times out: the marker E1E4_1117_DONE "rc 3 (not started: ...)", and no run.
#   HOLD at any point ends the chain with markers. PAUSE or a shut lid only holds it (the runners re-check at their
#   starts).
#   On TERM / INT / HUP before E14's runner takes over:
#      - try B's runner is stopped (it writes its own marker);
#      - then the markers not yet written: C25_gpubB_DONE and E1E4_1117_DONE "... rc 1 (stopped by a signal: the chain,
#        before ...)".
# DRY=1 + CH_* overrides: the stub test (scratch markers, short times, stub runners).
set -u
L=/Users/kamen/Leap; DRY=${DRY:-0}
C25R=${CH_C25R:-$L/data/week98/c25/frozen/run_c25_gpubB.sh}
E14R=${CH_E14R:-$L/data/week98/e14/frozen/run_e14.sh}
KT=${CH_KT:-/Users/kamen/BigMoE/data/r57/G1A_KEYTEST_DONE}
WPAT=${CH_WPAT:-g1_follo[w][^ ]*[.]sh}
MB=${CH_MB:-$L/data/week98/c25/C25_gpubB_DONE}
ME=${CH_ME:-$L/data/week98/screens/E1E4_1117_DONE}
LOG=${CH_LOG:-$L/data/week98/c25/run_chain1117.log}
AWAY=${CH_AWAY:-/Users/kamen/.local/share/quiet/away_secs.sh}
Q=${CH_Q:-$HOME/.local/share/quiet}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
EARLY=${CH_EARLY:-$(ep "2026-11-16 23:30:00")}
TB_LAST=${CH_TB_LAST:-$(ep "2026-11-17 01:46:00")}
TB_HMAX=${CH_TB_HMAX:-$(ep "2026-11-17 02:10:00")}
E_START=${CH_E_START:-$(ep "2026-11-17 00:30:00")}
E_LAST=${CH_E_LAST:-$(ep "2026-11-17 02:10:00")}
E_HARD=${CH_E_HARD:-2026-11-17 03:00:00}
GATE=${CH_GATE:-39}; NOBIG=${CH_NOBIG:-16}; P=${CH_POLL:-30}; G=${CH_GAP:-60}; TBMIN=${CH_TBMIN:-24}
T0=$(date +%s); STAGE=wait; CHILD=""; SLP=""; MB_DONE=""; KTWHY=""; WAITWHY=""
fmt() { date -r $1 '+%Y-%m-%d %H:%M:%S'; }
hm() { date -r $1 '+%m-%d %H:%M'; }
say() { echo "$(date '+%m-%d %H:%M:%S') [chain1117$([ $DRY = 1 ] && echo _dry)] $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
big_other() { [ "$(ps -axo rss=,pid= | awk -v L=$(( NOBIG * 1024 * 1024 )) -v me=$$ '$1 > L && $2 != me' | wc -l | tr -d ' ')" -gt 0 ]; }
away() { local a; a=$(bash $AWAY 2>/dev/null); echo ${a:-0}; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
held() { [ -e $Q/HOLD ]; }
nap() { sleep $1 & SLP=$!; wait $SLP; SLP=""; }
mark_b() { [ -n "$MB_DONE" ] && return 0; echo "c25 end $(date '+%m-%d %H:%M:%S'): rc $1 ($2)" > $MB; MB_DONE=1; say "wrote $MB (rc $1, $2)"; }
mark_e() { echo "E1-E4 end $(date '+%m-%d %H:%M:%S'): rc $1 ($2)" > $ME; say "wrote $ME (rc $1, $2)"; }
on_stop() {
  trap - TERM INT HUP
  [ -n "$SLP" ] && kill $SLP 2>/dev/null
  say "stopped by a signal (stage $STAGE)"
  if [ -n "$CHILD" ] && kill -0 $CHILD 2>/dev/null; then kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; MB_DONE=1; fi
  mark_b 1 "stopped by a signal: the chain, before try B"
  mark_e 1 "stopped by a signal: the chain, before E14"
  exit 1; }
end_hold() {
  say "HOLD: end"
  mark_b 3 "NOT RUN: HOLD (the chain, before try B)"
  mark_e 3 "not started: HOLD (the chain, before E14)"
  exit 3; }
trap on_stop TERM INT HUP
prewait() {   # $1 deadline, $2 label. Returns 0 when free (see the header), 1 at the deadline (WAITWHY = the last hold)
  local a b c last=""
  while :; do
    held && end_hold
    [ $(date +%s) -ge $1 ] && { WAITWHY=${WAITWHY:-the deadline}; return 1; }
    c=""
    if [ -e $Q/PAUSE ]; then c=pause; WAITWHY="PAUSE"
    elif [ $DRY = 0 ] && lid_closed; then c=lid; WAITWHY="the lid shut"
> (a quote from the project owner, kept private)
    elif big_other; then c=big; WAITWHY="a process over $NOBIG GB"
    else a=$(noswap)
      if [ $a -ge $GATE ]; then nap $G; b=$(noswap)
        if [ $b -ge $GATE ] && ! big_other && [ $(away) -ge 900 ] && ! held; then
          say "  $2: free (NO-SWAP $a / $b GB, away $(away) s)"; WAITWHY=""; return 0; fi
        c=mem; WAITWHY="NO-SWAP $a / $b GB < $GATE"
      else c=mem; WAITWHY="NO-SWAP $a GB < $GATE"; fi
    fi
    [ "$c" != "$last" ] && say "  $2: hold: $WAITWHY"; last=$c
    nap $P
  done; }

rm -f $MB $ME
say "armed (pid $$, DRY $DRY): the key-test marker $KT newer than arming; nothing before $(hm $EARLY); try B by $(hm $TB_LAST) (hard end <= $(hm $TB_HMAX)); E14 from $(hm $E_START), pre-wait to $(hm $E_LAST), hard end $E_HARD; runners c25 $(md5 -q $C25R | cut -c1-8), e14 $(md5 -q $E14R | cut -c1-8)"
# 1. the key test's end marker
TK=$T0
while :; do
  held && end_hold
  [ $(date +%s) -ge $TB_LAST ] && { KTWHY="no key-test marker by $(hm $TB_LAST)"; say "$KTWHY"; break; }
  if [ -f $KT ] && [ "$(stat -f %m $KT)" -ge $TK ]; then
    l=$(head -c 160 $KT | tr '\n' ' ')
    if echo "$l" | grep -q STOPPED; then
      w1=$(pgrep -f "$WPAT" | head -1); nap $G; w2=$(pgrep -f "$WPAT" | head -1)
      if [ -n "$w1$w2" ]; then
        TK=$(( $(stat -f %m $KT) + 1 )); say "key-test marker STOPPED and a BigMoE g1 waiter runs (pid ${w2:-$w1}; a re-arm): wait for its next marker"; nap $P; continue; fi
    fi
    say "the key test ended: $l"; break
  fi
  nap $P
done
if [ $(date +%s) -lt $EARLY ]; then
  say "wait for the night window ($(hm $EARLY))"
  while [ $(date +%s) -lt $EARLY ]; do held && end_hold; nap $P; done
fi
# 2. try B
if [ -n "$KTWHY" ]; then
  mark_b 3 "NOT RUN: $KTWHY; slot #1 (Tue 09:30) records the same profile + CPU split"
elif ! prewait $TB_LAST "try B pre-wait"; then
  mark_b 3 "NOT RUN: not free by $(hm $TB_LAST) ($WAITWHY); slot #1 (Tue 09:30) records the same profile + CPU split"
else
  now=$(date +%s); HB=$(( now + TBMIN * 60 )); [ $HB -gt $TB_HMAX ] && HB=$TB_HMAX
  STAGE=tryB; say "try B: $C25R \"$(fmt $now)\" \"$(fmt $HB)\" (TAG ${CH_TAG:-gpubB})"
  ( export C25_TAG=${CH_TAG:-gpubB} C25_C=160 C25_GATE=39 C25_ROUNDS=8 C25_WARM=2 C25_EST_MIN=15 C25_CAP_MIN=20
    export C25_PROF=1 C25_CPUSPLIT=1 C25_ARMS=F32c6i,F16,F32s,F32o
    export C25_WAIT_NOSWAP=39 C25_WAIT_NOBIG=16 C25_POLL=20 C25_GAP=60
    cd $L && exec /bin/bash $C25R "$(fmt $now)" "$(fmt $HB)" ) &
  CHILD=$!; wait $CHILD; RCB=$?; CHILD=""; MB_DONE=1
  say "try B ended (rc $RCB): $(head -c 160 $MB 2>/dev/null)"
fi
# 3. E14
STAGE=e14
if [ $(date +%s) -lt $(( E_START - 30 )) ]; then
  say "E14 waits for $(hm $E_START)"
  while [ $(date +%s) -lt $(( E_START - 30 )) ]; do held && end_hold; nap $P; done
fi
if ! prewait $E_LAST "E14 pre-wait"; then
  mark_e 3 "not started: not free by $(hm $E_LAST) ($WAITWHY); the chain"; exit 3
fi
SE=$(date +%s); [ $SE -lt $E_START ] && SE=$E_START
export E14_ARMS=A16,C32,C32one,C32oneB,A16b,A16h,A16c,C32c
say "E14: exec $E14R \"$(fmt $SE)\" \"$E_HARD\" (arms $E14_ARMS)"
trap - TERM INT HUP
cd $L && exec /bin/bash $E14R "$(fmt $SE)" "$E_HARD"
