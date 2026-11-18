#!/bin/bash
# stk1_chain.sh: blob 1's tier 1 #1 (PREREG_lossygate amendment 6, sections 2, 3 and 8; the orchestrator via NewIdeas,
# received Wed 11:3x: TONIGHT), CODE-ONLY, through runner 69 (tools/day_runner69.sh) on BigMoE's blob 1, against the EXACT
# references. Built from tools/stk_chain.sh (ede7aa9, never armed): the same run / held / back / waitfree / step / verdict /
# cardmem code, with the AWAY rule ON; no G1 decision and no base orders (the references are stored).
#   Steps: SMOKE: runner 69's card tier on tools/stk_smoke_keys.txt (3 keys, both orders, CARD_RUNSUF=_smoke), a function
#   check, not a verdict (rc 0, a "(tier 2" DONE line that is not VOID / INCOMPLETE), run WITHOUT engagement terms (its one
#   code key cannot show them; tier 1 does); then the card gate's memory by the registered rule, gate = max(43, the
#   smoke's lifetime peak + 0.5 GB rounded up to a whole GB), CARD_MEM = gate - 4, written to results/STK_CARD_MEM ->
#   T1F blob 1 forward vs EXACT (ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; IFEval by decision 4) -> the xrw check
#   (tools/xrw_check.py, amendment 6's process-wide rules over every kept run) -> T1R blob 1 reverse vs EXACT, both orders
#   pooled -> the xrw check. A T1F FAIL or interim STOP-FAIL, or a VOID xrw check, ends the chain (no T1R).
#   Start gates, all read twice 30 s apart: no EvalSuite runner alive; the markers in CHAIN_AFTER (each written after the
#   arming); the env file's git blob is the pinned one; BigMoE's key test CHAIN_KTMK (~/BigMoE/data/r57/BLOB1_KEYTEST_END)
#   newer than the env file with first line '^BLOB1 KEYTEST PASS' ('FAIL' or 'NOT RUN' ends the chain, no run); and
#   amendment 6 FILED: the committed PREREG (HEAD) holds "## Amendment 6 (" and names the blob sha.
#   One window: no first start after CHAIN_LASTSTART (default Thu 00:30: then tier 1 #1 runs Thursday as set), runner
#   cutoff CHAIN_CUTOFF (default Thu 09:00).
#   Nothing starts while HOLD, PAUSE, NewIdeas' BREAK98_HOLD, a shut lid (no external display) or Kamen at the Mac
#   (away_secs.sh < 900, one read; no reading counts as away). Mid-run, Kamen at the Mac on two reads 30 s apart, or
#   BREAK98_HOLD appearing (amendment 6 section 7, as g1_chain.sh), makes the step yield (the runner and its children TERM,
#   peer unlock); it resumes once nothing holds. Runner 69 gates each process on NO-SWAP
#   CARD_MEM + 4 GB (43), keeps the +2 GB swap guard, and yields at each process start to another session's run > 5 GB.
#   EVERY end writes results/STK1_CHAIN_END (Part A's chain follows it), after the runner and its children exit and peer
#   unlock. Stop it by pid only, with a plain kill (TERM). kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1, CARD_REF_LATER=1); no wait, no gate, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/stk1_chain.sh
# Pins, set at arming from the amendment: CHAIN_B1_ENV / CHAIN_B1_BLOB / CHAIN_B1_TAG; CHAIN_TREES / CHAIN_SCREEN_SHA /
# CHAIN_SO_SHA; CHAIN_XKNOB (<path>/xknob.py:<sha256>); CHAIN_AFTER (marker|pgrep pattern|time; semicolon list).
# Tests only: CHAIN_QDIR, CHAIN_AWAY, CHAIN_RUNNER, CHAIN_KTMK, CHAIN_TICK, CHAIN_REREAD, CHAIN_RUNPAT, CHAIN_PREREG.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
TGL=stk1; L=results/stk1_chain.log; END=${CHAIN_END:-results/STK1_CHAIN_END}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
hm() { date -r "$1" '+%m-%d %H:%M'; }
CUT="${CHAIN_CUTOFF:-2026-11-19 09:00}"; LASTS=$(ep "${CHAIN_LASTSTART:-2026-11-19 00:30}")
ARMED=$(date +%s); TICK=${CHAIN_TICK:-60}; REREAD=${CHAIN_REREAD:-30}
KTMK=${CHAIN_KTMK:-$HOME/BigMoE/data/r57/BLOB1_KEYTEST_END}; PREREG=${CHAIN_PREREG:-docs/PREREG_lossygate_2026-11-15.md}
AFTER=${CHAIN_AFTER-}
RUNPAT=${CHAIN_RUNPAT:-tools/day_runner[67][0-9]\.sh}     # a pgrep -f pattern (runners 60-79: Part A runs runner 70)
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
BHOLD=${CHAIN_BHOLD:-$HOME/NewIdeas/research/week98/BREAK98_HOLD}   # NewIdeas: while it exists, no process starts
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner69.sh}
EX=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; NAXN=ref2tnaxn.r6
CC1=close_call:accepted@run                               # CC(1), an accept rule on the code sets: > 0 over the run's kept code answers
SMOKEKF=tools/stk_smoke_keys.txt
VFENV=${CHAIN_B1_ENV:?pin CHAIN_B1_ENV}; VBLOB=${CHAIN_B1_BLOB:?pin CHAIN_B1_BLOB}; VT=${CHAIN_B1_TAG:?pin CHAIN_B1_TAG}
XKNOB=${CHAIN_XKNOB:?pin CHAIN_XKNOB to <path>/xknob.py:<sha256>}
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
  echo "STK1 CHAIN END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
away() { "$AWAY" 2>/dev/null; }
held() {        # 0 = held: Kamen's HOLD, the PAUSE file, Kamen at the Mac (one read; no reading counts as away), or the
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }          # lid shut with no external display (runner 64's lid_shut)
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
  [ -e "$BHOLD" ] && { HELD="BREAK98_HOLD ($(head -1 "$BHOLD" 2>/dev/null | cut -c1-80))"; return 0; }
> (a quote from the project owner, kept private)
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
back() {        # 0 = Kamen is at the Mac on two reads 30 s apart: a running step yields
  local a; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null || return 1
  sleep "$REREAD"; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null
}
gates() {       # 0 = the start gates hold; 1 = not yet; 5 = never
  pgrep -f "$RUNPAT" > /dev/null && { WHY="an EvalSuite runner is alive"; return 1; }
  local IFS_=$IFS a_ mk_ pt_ tm_ k_; IFS=';'
  for a_ in $AFTER; do
    IFS=$IFS_; mk_=${a_%%|*}; pt_=${a_#*|}; tm_=${pt_#*|}; pt_=${pt_%%|*}
    { [ -e "$mk_" ] && [ "$(stat -f %m "$mk_")" -ge "$ARMED" ]; } && continue
    { [ "$(date +%s)" -ge "$(ep "$tm_")" ] && ! pgrep -f "$pt_" > /dev/null; } && continue
    WHY="no $mk_ after the arming$(pgrep -f "$pt_" > /dev/null && echo " (its job runs or waits)" || echo " (none of its job before $tm_)")"; IFS=$IFS_; return 1
  done; IFS=$IFS_
  [ -s "$VFENV" ] || { WHY="no env file $VFENV"; return 1; }
  [ "$(git -C "$HOME/BigMoE" hash-object "$VFENV" 2>/dev/null)" = "$VBLOB" ] \
    || { WHY="the env file's blob $(git -C "$HOME/BigMoE" hash-object "$VFENV" 2>/dev/null) is not the pinned $VBLOB"; return 5; }
  { [ -e "$KTMK" ] && [ "$(stat -f %m "$KTMK")" -ge "$(stat -f %m "$VFENV")" ]; } || { WHY="no $KTMK newer than $VFENV"; return 1; }
  k_=$(head -1 "$KTMK")
  echo "$k_" | grep -qE '^BLOB1 KEYTEST (FAIL|NOT RUN)' && { WHY="the key test did not pass: $(echo "$k_" | cut -c1-120)"; return 5; }
  echo "$k_" | grep -q '^BLOB1 KEYTEST PASS' || { WHY="$KTMK's first line is not a verdict yet: $(echo "$k_" | cut -c1-80)"; return 1; }
  { git show "HEAD:$PREREG" 2>/dev/null | grep -q '^## Amendment 6 (' && git show "HEAD:$PREREG" | grep -q "$VBLOB"; } \
    || { WHY="amendment 6, naming blob $VBLOB, is not committed in $PREREG"; return 1; }
  WHY="key test: $(echo "$k_" | cut -c1-100); amendment 6 filed ($(git log -1 --format=%h -- "$PREREG"))"
  return 0
}
run() {         # label, the run's TG (for its DONE file), then runner 69's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
  if [ "$DRY" != 1 ] && [ -e "$BHOLD" ]; then say "$lab: not started: BREAK98_HOLD ($(head -1 "$BHOLD" 2>/dev/null | cut -c1-80))"; return 75; fi
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/stk1_chain_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do     # a HOLD mid-run: the finish trap stops the runner and its children
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
    if [ "$DRY" != 1 ] && [ -e "$BHOLD" ] && kill -0 "$RPID" 2>/dev/null; then   # the BREAK98 hold: a clean handoff (as g1_chain.sh)
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
    n=$((n + 1)); sleep "$TICK"
  done
}
verdict() {     # TG, tier, run dir: the step's recorded verdict line (none: rc 1). Tier 1: DONE (tier1, VOID or INCOMPLETE)
  local tg=$1 tier=$2 rd=$3 f=results/CARD_${1}_DONE   # with the run dir present. Tier 2 (the smoke): a "(tier 2" line
  [ -s "$f" ] || return 1
  if [ "$tier" = 1 ]; then { [ -d "results/$rd" ] && head -1 "$f" | grep -qE 'tier1|VOID|INCOMPLETE'; } || return 1; fi
  if [ "$tier" = 2 ]; then { [ -d "results/$rd" ] && head -1 "$f" | grep -qE '\(tier 2|VOID|INCOMPLETE'; } || return 1; fi
  head -1 "$f"
}
step() {        # label, TG, tier, run dir, then runner 69's settings. 0 = a verdict is recorded; 75 = cut by the window;
  local lab=$1 tg=$2 tier=$3 rd=$4 rc; shift 4      # 9 = no verdict, VOID or INCOMPLETE. RAN=1 once a runner was started
  if verdict "$tg" "$tier" "$rd" | grep -qE 'VOID|INCOMPLETE'; then say "$lab: a recorded VOID / INCOMPLETE verdict: $(verdict "$tg" "$tier" "$rd" | cut -c1-140): a decision, not a re-run"; return 9; fi
  if verdict "$tg" "$tier" "$rd" > /dev/null; then say "$lab: already done: $(verdict "$tg" "$tier" "$rd" | cut -c1-160)"; return 0; fi
  RAN=1
  while :; do
    if [ -e "results/$rd" ] && [ "$DRY" != 1 ]; then
      run "$lab, resumed" "$tg" "$@" CARD_RESUME=1; rc=$?   # a cut tier-1 run or the smoke's two runs: they resume
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
    say "$lab: the runner stopped at the cutoff (rc 6): the rest goes to a re-arm"; return 75
  fi
  [ "$rc" = 0 ] || return "$rc"
  verdict "$tg" "$tier" "$rd" | grep -qE 'VOID|INCOMPLETE' && return 9
  return 0
}
xrw() {         # amendment 6's process-wide checks over a tier-1 run; a VOID ends the chain (9)
  [ "$DRY" = 1 ] && { say "DRY: the xrw check would read results/$1"; return 0; }
  local o=results/lossygate/xrw_$1_$(date +%Y-%m-%d).txt
  .venv/bin/python tools/xrw_check.py "$1" --mode tgl16eu --xknob "$XKNOB" --code-sum tree:sib_accepted > "$o" 2>&1; local rc=$?
  say "$(tail -1 "$o" | cut -c1-240)"
  [ "$rc" = 0 ] && return 0
  echo "$(tail -1 "$o")" > "results/CARD_${2}_XRW_VOID"; return 9
}
cardmem() {     # the card gate's memory by the registered rule, from the smoke's processes (results/card/es_cd<VT>_smoke_*)
  [ "$DRY" = 1 ] && { say "DRY: the card gate's memory rule would read results/card/es_cd${VT}_smoke_*"; return 0; }
  python3 - "$VT" > results/STK_CARD_MEM.tmp 2>&1 <<'CM' || { say "the card memory rule failed: $(tail -1 results/STK_CARD_MEM.tmp | cut -c1-160)"; return 9; }
import glob, math, re, sys, time
vt = sys.argv[1]
fs = sorted(glob.glob(f"results/card/es_cd{vt}_smoke_*.log"))
pk = [float(x) for f in fs for x in re.findall(r"lifetime peak (?:after prefill )?[0-9.]+ -> (?:after decode )?([0-9.]+) GB", open(f).read())]
assert pk, f"no lifetime peak in {len(fs)} smoke process logs"
p = max(pk); gate = max(43, math.ceil(p + 0.5))
print(f"STK CARD MEM {time.strftime('%m-%d %H:%M:%S')}: gate {gate} GB NO-SWAP, CARD_MEM {gate - 4} (the rule: max(43, the smoke's "
      f"lifetime peak + 0.5 GB, rounded up); the smoke's lifetime peak {p:.2f} GB over {len(fs)} processes ({vt}_smoke)")
CM
  mv results/STK_CARD_MEM.tmp results/STK_CARD_MEM; say "$(cat results/STK_CARD_MEM)"
}
steps() {       # 0 = tier 1 ended (done, or a decision: a T1F FAIL); 75 = cut by the window; other = a stop
  local r                                          # (9: no verdict, VOID or INCOMPLETE)
  STATUS="SMOKE"; RAN=0; step "SMOKE blob 1 card tier, 3 keys, both orders (a function check; no engagement terms)" "${VT}_smoke" 2 \
    "card_${VT}_smoke_t2code_fwd" "${P[@]}" "${V[@]}" CARD_ENGAGE= CARD_TIER=2 "CARD_REF=$NAXN" "CARD_KEYSFILE=$SMOKEKF" CARD_RUNSUF=_smoke; r=$?
  R="$R SMOKE rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  if [ "$DRY" = 1 ]; then cardmem                  # the card gate's memory: once per smoke (a newer STK_CARD_MEM is kept)
  elif ! { [ -s results/STK_CARD_MEM ] && [ "$(stat -f %m results/STK_CARD_MEM)" -ge "$(stat -f %m "results/CARD_${VT}_smoke_DONE")" ]; }; then
    cardmem || return 9
  fi
  STATUS="T1F"; RAN=0; step "T1F blob 1 forward vs EXACT" "$VT" 1 "card_${VT}_t1code" "${P[@]}" "${V[@]}" CARD_TIER=1 "CARD_REF=$EX" "${LATER[@]}"; r=$?
  R="$R T1F rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  if [ "$DRY" != 1 ] && ! verdict "$VT" 1 "card_${VT}_t1code" | grep -q '^LOSSY tier1 (CODE-ONLY mode): PASS'; then
    ENDS="ended: T1F did not pass ($(verdict "$VT" 1 "card_${VT}_t1code" | cut -c1-120)); no T1R"; return 0
  fi
  xrw "card_${VT}_t1code" "$VT" || { ENDS="ended: T1F is VOID by the xrw check (results/CARD_${VT}_XRW_VOID); no T1R"; return 9; }
  STATUS="T1R"; RAN=0; step "T1R blob 1 reverse vs EXACT, pooled" "${VT}_rev" 1 "card_${VT}_t1code_rev" "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_ORDER=rev \
    "CARD_REF=$EX" "CARD_FWD_RUN=card_${VT}_t1code" "CARD_FWD_REF=$EX" "${LATER[@]}"; r=$?
  R="$R T1R rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  xrw "card_${VT}_t1code_rev" "${VT}_rev" || { ENDS="ended: T1R is VOID by the xrw check (results/CARD_${VT}_rev_XRW_VOID)"; return 9; }
  ENDS="done: blob 1 ($VT); the both-orders tier-1 verdict in CARD_${VT}_rev_DONE; the card gate's memory in STK_CARD_MEM"; return 0
}
startwait() {   # wait until the gates hold and nothing holds, on two reads 30 s apart. 0 = start; 1 = no start before the
  local n=0 lasth="" h g w1                        # latest start; 5 = never
> (a quote from the project owner, kept private)
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    [ "$(date +%s)" -lt "$LASTS" ] || { WHY="${h:+held: $h; }$(gates; echo "$WHY")"; return 1; }
    gates; g=$?
    [ "$g" = 5 ] && return 5
    if [ -z "$h" ] && [ "$g" = 0 ]; then
      w1=$WHY; sleep "$REREAD"
      if ! held && gates; then say "start: $w1; read again ${REREAD} s later"; return 0; fi
      say "the start condition did not hold ${REREAD} s later: waiting"
    fi
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$(gates; echo "$WHY")"
    sleep "$TICK"
  done
}
P=(CARD_SCOPE=code "CARD_TREES=${CHAIN_TREES:?pin CHAIN_TREES}" "CARD_SCREEN_SHA=${CHAIN_SCREEN_SHA:?pin CHAIN_SCREEN_SHA}" \
   "CARD_SO_SHA=${CHAIN_SO_SHA:?pin CHAIN_SO_SHA}" "CARD_CUTOFF=$CUT" CARD_MEM=39)
V=(CARD_TAG="$VT" "CARD_FENV=$VFENV" CARD_FBLOB="$VBLOB" "CARD_ENGAGE=$CC1")
LATER=(); [ "$DRY" = 1 ] && LATER=(CARD_REF_LATER=1)
R=""; ENDS=""
say "stk1_chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): blob 1's tier 1 #1 through $(basename "$RUNNER"), $VT, blob $VBLOB; cutoff $CUT, latest first start $(hm "$LASTS"); after [${AFTER:-none}]; key test $KTMK; xknob $XKNOB; the AWAY rule on"
if [ "$DRY" = 1 ]; then steps; exit $?; fi
STATUS="waiting"
startwait; s=$?
[ "$s" = 5 ] && { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
[ "$s" = 1 ] && { STATUS="no start by $(hm "$LASTS"): $WHY"; say "$STATUS"; exit 6; }
say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
steps; r=$?
if [ "$r" = 0 ]; then STATUS="${R%;}; $ENDS"; say "$ENDS"; exit 0; fi
if [ "$r" = 75 ]; then STATUS="cut by the cutoff:${R%;} (the rest goes to a re-arm)"; say "$STATUS"; exit 75; fi
STATUS="stopped:${R%;}${ENDS:+; $ENDS} (rc $r: 9 = no verdict, VOID or INCOMPLETE)"; say "$STATUS"; exit "$r"
