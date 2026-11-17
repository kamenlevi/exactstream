#!/bin/bash
# t1c_chain2.sh = t1c_chain.sh (d7d9001d6549811d) + three settings for running the gate to completion (PREREG_lossygate
# amendment 4f; Kamen 11-16 11:5x via the orchestrator: the day limits and the at-Mac yield are lifted until he says
# otherwise), and one fix. A new file, because t1c_chain.sh serves armed phases.
#   CHAIN_CUTOFF (YYYY-mm-dd HH:MM): the runner's cutoff instead of DAY 14:30; the last start is 15 min before it.
#   CHAIN_NOYIELD=1: no Kamen-at-the-Mac rule, neither as a start condition nor as a mid-run yield. HOLD, PAUSE and a
#     shut lid still hold; the memory gate, the +2 GB swap guard and every runner stop are unchanged.
#   CHAIN_TAG: the log tag and the end marker results/T1C_<TAG>_END (default: the phase).
#   The fix: a recorded VOID / INCOMPLETE tier-1 verdict stops the chain (rc 9, a decision) instead of counting as done.
# t1c_chain.sh: the 1c-tiers lossy gate (docs/PREREG_lossygate amendments 4 / 4a / 4b), CODE-ONLY, through runner 65,
# base lx45t1c1 and variant t1cx45t1c1 on BigMoE's snapshot t1c1. One script, three windows (NewIdeas 08:4x; 09:2x):
#   CHAIN_PHASE=mon (today, NewIdeas 09:2x: t30c is off, the orchestrator agrees): as tue, without the T0A gate.
#   CHAIN_PHASE=tue: B0 base mini-panel -> V0 variant mini-panel (tier-0 KILL vs B0) -> B1 base forward -> V1 variant
#     forward (vs B1, IFEval identity), as far as the day window allows. It never starts B2 / V2.
#   CHAIN_PHASE=wed (armed on NewIdeas' word, after the drafter line): the steps not done yet, in order: a cut tier-1
#     run resumes (CARD_RESUME=1), then B2 base reverse -> V2 variant reverse (both orders pooled).
# A step already done (its verdict in its DONE file) is skipped. V0's KILL or V1's FAIL ends the gate; a step that ends
# without a verdict, VOID or INCOMPLETE stops the chain. After V1 / V2: tools/lossy_speed.py (information).
# Tier 0 and tier 1 share a TG, so a tier-0 verdict is copied to results/CARD_<tg>_DONE.t0. A tier-0 run cut by a yield
# or a window is moved aside (<dir>.cut<N>) and starts fresh (amendment 4); a tier-1 run resumes.
# Start gates (tue; mon has all but the first): results/T0A_1117_END written after the tier-0a follower's arming (Mon 08:40:20, any end); BigMoE's
# dry-run marker ~/BigMoE/data/deepdraft/ceiling/N127_DRY_1117_DONE with mtime >= Mon 09:05; and the t1c1 real-loop key
# test's marker ~/BigMoE/data/r57/T1C1_KEYTEST_DONE (mtime >= Mon 09:20) saying PASS with every env's peak <= 42.5 GB
# (a FAIL or a higher peak: no run, rc 5; STOPPED / NOT RUN: it waits for a re-test); all read twice, 30 s apart.
# BREAK98 hold (NewIdeas Tue 07:3x; PREREG_lossygate 4j / 5b dated lines Tue 07:4x): while $HOME/NewIdeas/research/week98/
#   BREAK98_HOLD exists, nothing starts; when it appears mid-run, the running step yields at once (the runner and its
#   children TERM, peer unlock) and resumes once it is gone (a cut tier-1 run with CARD_RESUME=1). CHAIN_BHOLD: tests.
# Both phases: nothing starts while HOLD, PAUSE, a shut lid (no external display) or Kamen at the Mac, and no
# start after 14:15 of the phase's day; the cutoff is 14:30 (the capture starts no earlier than Wed 23:30). Runner 65
# gates each process on NO-SWAP 43 GB (CARD_MEM 39) and yields at each process start to another session's run above
# 5 GB. Kamen at the Mac mid-run: two reads 30 s apart and the step yields (runner + children stop, peer unlock).
# EVERY end writes results/T1C_<PHASE>_END, after the runner and its children exit and peer unlock. Stop it by pid only,
# with a plain kill (TERM). kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1, CARD_REF_LATER=1 for references made by earlier steps); no
# wait, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> CHAIN_PHASE=mon|tue|wed [CHAIN_DRY=1] tools/t1c_chain.sh
# Tests only: CHAIN_QDIR, CHAIN_T0A, CHAIN_N127, CHAIN_KT, CHAIN_DAY (YYYY-mm-dd), CHAIN_AWAY, CHAIN_RUNNER.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; PH=${CHAIN_PHASE:?set CHAIN_PHASE to mon, tue or wed}; MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
case "$PH" in mon) DAY=${CHAIN_DAY:-2026-11-16} ;; tue) DAY=${CHAIN_DAY:-2026-11-17} ;; wed) DAY=${CHAIN_DAY:-2026-11-18} ;; *) echo "CHAIN_PHASE is mon, tue or wed"; exit 2 ;; esac
TGL=${CHAIN_TAG:-$PH}; UPH=$(echo "$TGL" | tr a-z A-Z); L=results/t1c_chain.log; END=results/T1C_${UPH}_END; NOY=${CHAIN_NOYIELD:-0}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
CUT="${CHAIN_CUTOFF:-$DAY 14:30}"; GIVEUP=$(( $(ep "$CUT") - 900 ))
T0A=${CHAIN_T0A:-results/T0A_1117_END}; T0AAFTER=$(ep "2026-11-16 08:40")
N127=${CHAIN_N127:-$HOME/BigMoE/data/deepdraft/ceiling/N127_DRY_1117_DONE}; N127AFTER=$(ep "2026-11-16 09:05")
KT=${CHAIN_KT:-$HOME/BigMoE/data/r57/T1C1_KEYTEST_DONE}; KTAFTER=$(ep "2026-11-16 09:20")   # BigMoE t1c_follow.sh c7cf64a8
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
BHOLD=${CHAIN_BHOLD:-$HOME/NewIdeas/research/week98/BREAK98_HOLD}   # NewIdeas Tue 07:3x: while it exists, no process starts (BREAK98 first)
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner65.sh}
BMS=$HOME/BigMoE/data/stack; EX=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2
TREES=data/r57/snap/t1c1=04180c59e76e39b5deffd8e38c5ac277c179a005,data/r57/snap/frozen_t1c1=14bb1462b292d2cb366401cf4d30e8967e79a124,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
say() { echo "$(date '+%m-%d %H:%M:%S') [$TGL] $*" >> "$L"; }
STATUS="armed"; RPID=""; RSTART=0
stoprunner() {  # the runner and its children: TERM, KILL after 60 s
  { [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; } || { RPID=""; return 0; }
  say "stopping runner $RPID and its children"
  pkill -TERM -P "$RPID"; kill -TERM "$RPID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$RPID" 2>/dev/null || break; sleep 10; done
  pkill -KILL -P "$RPID" 2>/dev/null; kill -KILL "$RPID" 2>/dev/null
  wait "$RPID" 2>/dev/null; RPID=""
}
finish() {
  local rc=$?
  stoprunner                                                 # stopped mid-run: the runner and its children first
  [ "$DRY" = 1 ] && { say "DRY end (rc $rc): no marker"; exit "$rc"; }
  peer unlock > /dev/null 2>&1
  echo "T1C $UPH END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
away() { "$AWAY" 2>/dev/null; }
held() {        # 0 = held: Kamen's HOLD, the PAUSE file, the lid shut with no external display (runner 64's lid_shut), or
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }          # Kamen at the Mac (one read; no reading counts as away)
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
  [ -e "$BHOLD" ] && { HELD="BREAK98_HOLD ($(head -1 "$BHOLD" 2>/dev/null | cut -c1-80))"; return 0; }
> (a quote from the project owner, kept private)
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
back() {        # 0 = Kamen is at the Mac on two reads 30 s apart: a running step yields (never with CHAIN_NOYIELD=1)
  [ "$NOY" = 1 ] && return 1
  local a; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null || return 1
  sleep 30; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null
}
gates() {       # 0 = the phase's start gates hold (tue: the tier-0a end marker, BigMoE's dry-run marker, and the t1c1 key
  [ "$PH" = wed ] && { WHY="wed: armed on NewIdeas' word"; return 0; }    # test's PASS; wed: none); 1 = not yet;
  [ "$PH" = mon ] || { [ -e "$T0A" ] && [ "$(stat -f %m "$T0A")" -ge "$T0AAFTER" ]; } || { WHY="no T0A_1117_END yet"; return 1; }   # 2 = never:
  { [ -e "$N127" ] && [ "$(stat -f %m "$N127")" -ge "$N127AFTER" ]; } || { WHY="no N127 dry-run marker yet"; return 1; }  # a key-
  { [ -e "$KT" ] && [ "$(stat -f %m "$KT")" -ge "$KTAFTER" ]; } || { WHY="no t1c1 key-test marker yet"; return 1; }   # test FAIL
  local kl pk; kl=$(head -1 "$KT")                                                                  # or a peak > 42.5 GB
  echo "$kl" | grep -qE '^T1C1 KEYTEST END [0-9-]+ [0-9:]+: FAIL' && { WHY="the t1c1 key test FAILED: $(echo "$kl" | cut -c1-160)"; return 2; }
  echo "$kl" | grep -qE '^T1C1 KEYTEST END [0-9-]+ [0-9:]+: PASS( |$)' || { WHY="the t1c1 key test is not PASS yet: $(echo "$kl" | cut -c1-120)"; return 1; }
  pk=$(echo "$kl" | grep -oE 'peak [0-9]+(\.[0-9]+)? GB' | awk '{print $2}' | sort -n | tail -1)
  if [ -n "$pk" ] && awk -v p="$pk" 'BEGIN { exit !(p > 42.5) }'; then WHY="the t1c1 key test's peak $pk GB is above 42.5 (amendment 4a): re-gate first"; return 2; fi
  WHY="$([ "$PH" = tue ] && echo "T0A ($(head -c 90 "$T0A" | tr '\n' ' ')); ")N127 ($(head -c 70 "$N127" | tr '\n' ' ')); key test PASS (peak ${pk:+$pk GB}${pk:-not given})"; return 0
}
run() {         # label, the run's TG (for its DONE file), then runner 65's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
  if [ "$DRY" != 1 ] && [ -e "$BHOLD" ]; then say "$lab: not started: BREAK98_HOLD ($(head -1 "$BHOLD" 2>/dev/null | cut -c1-80))"; return 75; fi
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/t1c_chain_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do     # a HOLD mid-run: the finish trap stops the runner and its children
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
    if [ "$DRY" != 1 ] && [ -e "$BHOLD" ] && kill -0 "$RPID" 2>/dev/null; then   # the BREAK98 hold: a clean handoff (NewIdeas 07:3x)
      say "BREAK98_HOLD ($(head -1 "$BHOLD" 2>/dev/null | cut -c1-80)): $lab yields"
      stoprunner; peer unlock > /dev/null 2>&1; return 75
    fi
    if [ "$DRY" != 1 ] && back && kill -0 "$RPID" 2>/dev/null; then
> (a quote from the project owner, kept private)
      stoprunner; peer unlock > /dev/null 2>&1; return 75
    fi
    sleep 15
  done
  wait "$RPID"; rc=$?; RPID=""
  local df=results/CARD_${tg}_DONE
  say "$lab: rc $rc; $(if [ -e "$df" ] && [ "$(stat -f %m "$df")" -ge "$RSTART" ]; then cut -c1-220 "$df"; else echo 'no DONE from this run'; fi)"
  return "$rc"
}
waitfree() {    # after a yield: until nothing holds (a HOLD ends the chain); 1 = no start left before the cutoff
  local n=0 h
  while :; do
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD while yielding: $STATUS"; say "HOLD while yielding"; exit 8; }
    h=$(held && echo "$HELD")
    [ -z "$h" ] && { say "nothing holds (away_secs $(away) s): continuing"; return 0; }
    [ $(( $(date +%s) + 900 )) -le "$(ep "$CUT")" ] || { say "still held at $(date +%H:%M) ($h): no start left before $CUT"; return 1; }
    [ $((n % 30)) = 0 ] && say "yielding: $h"
    n=$((n + 1)); sleep 60
  done
}
verdict() {     # TG, tier, run dir: the step's recorded verdict line (none: rc 1). Tier 0: DONE.t0. Tier 1: DONE (tier1,
  local tg=$1 tier=$2 rd=$3 f                      # VOID or INCOMPLETE) with the tier-1 run dir present, since a tier-0
  if [ "$tier" = 0 ]; then f=results/CARD_${tg}_DONE.t0; else f=results/CARD_${tg}_DONE; fi   # VOID shares the file
  [ -s "$f" ] || return 1
  if [ "$tier" = 1 ]; then { [ -d "results/$rd" ] && head -1 "$f" | grep -qE 'tier1|VOID|INCOMPLETE'; } || return 1; fi
  head -1 "$f"
}
step() {        # label, TG, tier, run dir, then runner 65's settings. 0 = a verdict is recorded; 75 = cut by the window;
  local lab=$1 tg=$2 tier=$3 rd=$4 rc k=0; shift 4  # 9 = no verdict, VOID or INCOMPLETE. RAN=1 once a runner was started
  if verdict "$tg" "$tier" "$rd" | grep -qE 'VOID|INCOMPLETE'; then say "$lab: a recorded VOID / INCOMPLETE verdict: $(verdict "$tg" "$tier" "$rd" | cut -c1-140): a decision, not a re-run"; return 9; fi
  if verdict "$tg" "$tier" "$rd" > /dev/null; then say "$lab: already done: $(verdict "$tg" "$tier" "$rd" | cut -c1-160)"; return 0; fi
  RAN=1
  while :; do
    if [ -e "results/$rd" ] && [ "$DRY" != 1 ]; then
      if [ "$tier" = 0 ]; then                       # a cut tier-0 run starts fresh (amendment 4)
        k=$((k + 1)); while [ -e "results/$rd.cut$k" ]; do k=$((k + 1)); done
        mv "results/$rd" "results/$rd.cut$k"; say "$lab: a cut tier-0 run moved aside: results/$rd.cut$k"
        run "$lab, fresh" "$tg" "$@"; rc=$?
      else
        run "$lab, resumed" "$tg" "$@" CARD_RESUME=1; rc=$?
      fi
    else
      run "$lab" "$tg" "$@"; rc=$?
    fi
    [ "$rc" = 75 ] || break
    waitfree || return 75
  done
  [ "$DRY" = 1 ] && return "$rc"
  if [ "$rc" = 6 ]; then                           # exit 6: the cutoff, unless the runner wrote INCOMPLETE (a real stop)
    { [ -e "results/CARD_${tg}_DONE" ] && [ "$(stat -f %m "results/CARD_${tg}_DONE")" -ge "$RSTART" ] \
      && grep -q INCOMPLETE "results/CARD_${tg}_DONE"; } && return 9
    say "$lab: the runner stopped at the window's cutoff (rc 6): it resumes in the next window"; return 75
  fi
  [ "$rc" = 0 ] || return "$rc"
  if [ "$tier" = 0 ]; then
    head -1 "results/CARD_${tg}_DONE" | grep -qE '^LOSSY tier0 \(CODE-ONLY mode\): (GO|KILL)' || return 9
    cp "results/CARD_${tg}_DONE" "results/CARD_${tg}_DONE.t0"
  fi
  verdict "$tg" "$tier" "$rd" | grep -qE 'VOID|INCOMPLETE' && return 9
  return 0
}
speed() {       # information beside the verdict (amendment 3a): kept per check, arm vs base
  [ "$DRY" = 1 ] && return
  local o=results/lossygate/speed_$1_$(date +%Y-%m-%d).txt
  .venv/bin/python tools/lossy_speed.py "$1" "$2" > "$o" 2>&1
  say "speed $1 vs $2 (rc $?): $(tail -1 "$o" | cut -c1-160)"
}
say "chain2 (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): 1c-tiers, phase $PH, tag $TGL (cutoff $CUT$([ "$NOY" = 1 ] && echo '; no at-Mac rule')); runner $(basename "$RUNNER"); NO-SWAP gate per process: the runner's"
if [ "$DRY" != 1 ]; then
> (a quote from the project owner, kept private)
  n=0; lasth=""
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    [ "$(date +%s)" -lt "$GIVEUP" ] || { STATUS="no start by $(date -r "$GIVEUP" '+%m-%d %H:%M') (${h:+held: $h; }$(gates; echo "$WHY")): no run"; say "$STATUS"; exit 6; }
    gates; g=$?
    [ "$g" = 2 ] && { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
    if [ -z "$h" ] && [ "$g" = 0 ]; then
      w1=$WHY; sleep 30
      if ! held && gates; then say "start: $w1; read again 30 s later"; break; fi
      say "the start condition did not hold 30 s later: waiting"
    fi
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$(gates; echo "$WHY")"
    sleep 60
  done
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
fi
P=(CARD_SCOPE=code "CARD_TREES=$TREES" CARD_SCREEN_SHA=8d56cb717843 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$CUT" CARD_MEM=39)
B=(CARD_TAG=lx45t1c1 "CARD_FENV=$BMS/lx45t1c1_env.txt" CARD_FBLOB=c9ebc0cfe9cb6697aefa2cc44c08b30b035fb58e)
V=(CARD_TAG=t1cx45t1c1 "CARD_FENV=$BMS/t1cx45t1c1_env.txt" CARD_FBLOB=d4b1217db0234aff6848e820fe1199f68cb28211 CARD_ENGAGE=lossy_check:e3+edrop+lmsl_rows)
LATER=(); [ "$DRY" = 1 ] && LATER=(CARD_REF_LATER=1)
R=""
ends() { STATUS="${R%;}; $1"; say "$1"; exit 0; }      # the chain's end before its last step: the status says why
STATUS="B0"; RAN=0; step "B0 base tier 0" lx45t1c1 0 card_lx45t1c1_t0code "${P[@]}" "${B[@]}" CARD_TIER=0 "CARD_REF=$EX"; r=$?
R="$R B0 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || ends "stopped: B0 rc $r (75: cut by the window, a fresh start next; 9: no tier-0 verdict, VOID or INCOMPLETE)"
STATUS="V0"; RAN=0; step "V0 variant tier 0 vs B0" t1cx45t1c1 0 card_t1cx45t1c1_t0code "${P[@]}" "${V[@]}" CARD_TIER=0 CARD_REF=card_lx45t1c1_t0code "${LATER[@]}"; r=$?
R="$R V0 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || ends "stopped: V0 rc $r (75: cut by the window, a fresh start next; 9: no verdict)"
if [ "$DRY" != 1 ] && verdict t1cx45t1c1 0 card_t1cx45t1c1_t0code | grep -q ': KILL'; then ends "ended: V0 KILL, the variant ends at tier 0"; fi
STATUS="B1"; RAN=0; step "B1 base forward" lx45t1c1 1 card_lx45t1c1_t1code "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX"; r=$?
R="$R B1 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || ends "stopped: B1 rc $r (75: cut by the window, it resumes in the next; 9: VOID or INCOMPLETE)"
STATUS="V1"; RAN=0; step "V1 variant forward vs B1" t1cx45t1c1 1 card_t1cx45t1c1_t1code "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_REF=card_lx45t1c1_t1code CARD_IFEVAL_IDENT=1 "${LATER[@]}"; r=$?
R="$R V1 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || ends "stopped: V1 rc $r (75: cut by the window, it resumes in the next; 9: VOID or INCOMPLETE)"
[ "$RAN" = 1 ] && speed t1cx45t1c1 lx45t1c1
if [ "$DRY" != 1 ] && ! verdict t1cx45t1c1 1 card_t1cx45t1c1_t1code | grep -q '^LOSSY tier1 (CODE-ONLY mode): PASS'; then ends "ended: V1 did not pass, the variant ends"; fi
[ "$PH" != wed ] && [ "$DRY" != 1 ] && ends "done for the day ($PH): V1 PASS (provisional); B2 -> V2 on Wednesday, after the drafter line"
STATUS="B2"; RAN=0; step "B2 base reverse" lx45t1c1_rev 1 card_lx45t1c1_t1code_rev "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX" CARD_ORDER=rev; r=$?
R="$R B2 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || ends "stopped: B2 rc $r (75: cut by the window, it resumes after the proofs; 9: VOID or INCOMPLETE)"
STATUS="V2"; RAN=0; step "V2 variant reverse vs B2, pooled" t1cx45t1c1_rev 1 card_t1cx45t1c1_t1code_rev "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_ORDER=rev \
  CARD_REF=card_lx45t1c1_t1code_rev CARD_IFEVAL_IDENT=1 CARD_FWD_RUN=card_t1cx45t1c1_t1code CARD_FWD_REF=card_lx45t1c1_t1code "${LATER[@]}"; r=$?
R="$R V2 rc $r;"; [ "$r" = 0 ] && [ "$RAN" = 1 ] && speed t1cx45t1c1_rev lx45t1c1_rev
STATUS="done:${R%;} (verdicts in the CARD_*_DONE files; the both-orders verdict in CARD_t1cx45t1c1_rev_DONE)"
