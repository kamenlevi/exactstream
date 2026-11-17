#!/bin/bash
# stk_chain.sh: the lossy STACK's tier 1 (PREREG_lossygate, the stack's amendment; the orchestrator Tue 06:4x / 06:5x;
# NewIdeas Tue 07:0x), CODE-ONLY, through runner 69 (tools/day_runner69.sh) on BigMoE's snapshot g1b. Built from
# tools/g1_chain.sh (c1852dfa566ea6e5): the same run / held / back / waitfree / step code, with the AWAY rule ON.
#   The env (read once, at the first start): "stk" (G1's pick + CC(1)) when G1's both-orders tier-1 line is the clean
#   PASS ("LOSSY tier1 (CODE-ONLY mode) [2 orders pooled]: PASS" in results/CARD_<pick>45g1b_rev_DONE); "stk0" (lossy-45 +
#   CC(1)) on anything short of it: a V0 KILL, a V1 or V2 FAIL, or no G1 verdict by the chain's latest start. Both envs
#   carry lcb6 in every lever's sets (for the card gate, which a later follower runs).
#   Steps: the base orders G1's chain made (B1 card_lx45g1b_t1code, B2 card_lx45g1b_t1code_rev), each run first if
#   missing -> SMOKE: runner 69's card tier on tools/stk_smoke_keys.txt (3 keys, both orders, CARD_RUNSUF=_smoke), a
#   function check, not a verdict (rc 0, a "(tier 2" DONE line that is not VOID / INCOMPLETE); then the card gate's
#   memory by the registered rule, gate = max(43, the smoke's lifetime peak + 0.5 GB rounded up to a whole GB), CARD_MEM =
#   gate - 4, written to results/STK_CARD_MEM -> T1F the stack forward vs B1 (IFEval identity) -> T1R the stack reverse
#   vs B2, both orders pooled. A FAIL or an interim STOP-FAIL of T1F ends the chain (no card gate). After T1F / T1R:
#   tools/lossy_speed.py (information).
#   Start gates, all read twice 30 s apart: no EvalSuite runner and no G1 chain alive; the markers in CHAIN_AFTER (the
#   jobs ahead in NewIdeas' Wednesday order; each written after the arming); BigMoE's ~/BigMoE/data/r57/STK_KEYTEST_DONE
#   newer than the chosen env file, with a line "<env>: PASS" for that env; and the env's decision (above).
#   One window: by day to CHAIN_CUTOFF (default Wed 21:30), no first start after CHAIN_LASTSTART (default Wed 18:45).
#   Nothing starts while HOLD, PAUSE, a shut lid (no external display) or Kamen at the Mac (away_secs.sh < 900, one
#   read; no reading counts as away). Mid-run, Kamen at the Mac on two reads 30 s apart makes the step yield (the runner
#   and its children TERM, peer unlock); it resumes once nothing holds. Runner 69 gates each process on NO-SWAP
#   CARD_MEM + 4 GB (43), keeps the +2 GB swap guard, and yields at each process start to another session's run > 5 GB.
#   EVERY end writes results/STK_<TAG>_END (tag stkw), after the runner and its children exit and peer unlock. Stop it
#   by pid only, with a plain kill (TERM). kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1, CARD_REF_LATER=1 for references made by earlier steps) on the
# env CHAIN_DRYENV names (stk or stk0); no wait, no gate, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1 CHAIN_DRYENV=stk] tools/stk_chain.sh
# Pins, set at arming from the amendment: CHAIN_STK_ENV / CHAIN_STK_BLOB / CHAIN_STK_TAG / CHAIN_STK_PICK (G1's pick
# that stk carries), CHAIN_STK0_ENV / CHAIN_STK0_BLOB / CHAIN_STK0_TAG; CHAIN_AFTER (marker paths, comma list).
# Tests only: CHAIN_QDIR, CHAIN_AWAY, CHAIN_RUNNER, CHAIN_KTMK, CHAIN_PICK, CHAIN_TICK, CHAIN_REREAD, CHAIN_RUNPAT,
# CHAIN_G1PAT, CHAIN_RES (the results dir the gates read).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
TGL=${CHAIN_TAG:-stkw}; UPH=$(echo "$TGL" | tr a-z A-Z); L=results/stk_chain.log; END=${CHAIN_END:-results/STK_CHAIN_END}   # Leap's slot #3 chains on it
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
hm() { date -r "$1" '+%m-%d %H:%M'; }
CUT="${CHAIN_CUTOFF:-2026-11-18 21:30}"; LAST0=$(ep "${CHAIN_LASTSTART:-2026-11-18 18:45}"); LASTS=$LAST0   # 17:30 if B2 must run, 16:15 if B1 too
ARMED=$(date +%s); TICK=${CHAIN_TICK:-60}; REREAD=${CHAIN_REREAD:-30}; RES=${CHAIN_RES:-results}
KTMK=${CHAIN_KTMK:-$HOME/BigMoE/data/r57/STK_KEYTEST_DONE}; PICKF=${CHAIN_PICK:-$RES/G1_PICK}
# the jobs ahead (NewIdeas 07:0x), each "marker|pgrep pattern|time": the marker written after the arming, or, from that time
# on, no process matching the pattern (the job not armed, or gone): MoonShots' cuts, then Breakthrough's band-screen fallback
AFTER=${CHAIN_AFTER-$HOME/MoonShots/week98/p3/Q8_DONE|p3_q[8]\.sh|2026-11-18 07:10;$HOME/Breakthrough/data/m1/lband_map/LBAND_END|run_lban[d]|2026-11-18 07:15}
RUNPAT=${CHAIN_RUNPAT:-tools/day_runner6[0-9]\.sh}; G1PAT=${CHAIN_G1PAT:-tools/g1_chai[n]\.sh}   # pgrep -f patterns
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner69.sh}
EX=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; NAXN=ref2tnaxn.r6
TREES=data/r57/snap/g1b=fb0ee18288dae0ac5528006557dfaf8487d63c3b,data/r57/snap/frozen_g1b=2584e36a814ae956ac5c2d9c9090c4d787298502,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
ONE_C=lossy_check:e3+edrop+lmsl_rows; RES_=lossy_check:res_opt@run,lossy_check:opt_post@run; COM=lossy_check:com_rows@run,lossy_check:com_opt@run
CC1=close_call:accepted@run                               # CC(1), an accept rule: > 0 over the run's kept code answers
SMOKEKF=tools/stk_smoke_keys.txt
envof() {       # stk | stk0 -> VFENV / VBLOB / VT / VENG (the amendment's pins); 1 = not pinned
  case "$1" in
    stk)  VFENV=${CHAIN_STK_ENV:-}; VBLOB=${CHAIN_STK_BLOB:-}; VT=${CHAIN_STK_TAG:-}
          case "${CHAIN_STK_PICK:-}" in   # G1's pick that stk carries -> its CARD_ENGAGE terms (5d's), + CC(1)
            g1|g1l) VENG=$ONE_C,$CC1 ;; g1r|g1lr) VENG=$ONE_C,$RES_,$CC1 ;; g1c|g1cd) VENG=$ONE_C,$COM,$CC1 ;;
            g1cr|g1cdr) VENG=$ONE_C,$COM,$RES_,$CC1 ;; *) return 1 ;;
          esac ;;
    stk0) VFENV=${CHAIN_STK0_ENV:-}; VBLOB=${CHAIN_STK0_BLOB:-}; VT=${CHAIN_STK0_TAG:-}; VENG=$CC1 ;;
    *) return 1 ;;
  esac
  [ -n "$VFENV" ] && [ -n "$VBLOB" ] && [ -n "$VT" ]
}
g1pick() { head -1 "$PICKF" 2>/dev/null | sed -nE 's/^G1 PICK [0-9]{2}-[0-9]{2} [0-9:]{8}: ([a-z0-9]+)( .*)?$/\1/p'; }
g1dec() {       # G1's decision -> DEC (stk | stk0 | "") and DECWHY, from its verdict files
  local p t; p=$(g1pick); DEC=""; DECWHY="no G1 pick"
  [ -n "$p" ] || return 0
  t=${p}45g1b
  if head -1 "$RES/CARD_${t}_rev_DONE" 2>/dev/null | grep -qE '^LOSSY tier1 \(CODE-ONLY mode\) \[2 orders pooled\]: PASS'; then
    DEC=stk; DECWHY="G1 ($p) PASS in both orders: $(head -1 "$RES/CARD_${t}_rev_DONE" | cut -c1-80)"
    [ "$p" = "${CHAIN_STK_PICK:-}" ] || { DEC=""; DECWHY="G1's pick $p is not the pick stk carries (${CHAIN_STK_PICK:-unset}): no start"; return 5; }
  elif head -1 "$RES/CARD_${t}_rev_DONE" 2>/dev/null | grep -qE '^LOSSY tier1 \(CODE-ONLY mode\) \[2 orders pooled\]: FAIL'; then
    DEC=stk0; DECWHY="G1 ($p) FAIL in both orders"
  elif head -1 "$RES/CARD_${t}_DONE.t0" 2>/dev/null | grep -q ': KILL'; then
    DEC=stk0; DECWHY="G1 ($p) V0 KILL"
  elif head -1 "$RES/CARD_${t}_DONE" 2>/dev/null | grep -qE '^LOSSY tier1 \(CODE-ONLY mode\): FAIL|^LOSSY [^ ]+ tier1: FAIL'; then
    DEC=stk0; DECWHY="G1 ($p) V1 FAIL"
  else DECWHY="no G1 decision yet (pick $p)"; fi
  return 0
}
say() { echo "$(date '+%m-%d %H:%M:%S') [$TGL] $*" >> "$L"; }
lasts() {       # the latest first start: 18:45 (smoke + tier 1 by 21:30); -75 min per base order still to run
  local m=0
  [ -s "$RES/CARD_lx45g1b_DONE" ] && [ -d "$RES/card_lx45g1b_t1code" ] && head -1 "$RES/CARD_lx45g1b_DONE" | grep -q tier1 || m=$((m + 1))
  [ -s "$RES/CARD_lx45g1b_rev_DONE" ] && [ -d "$RES/card_lx45g1b_t1code_rev" ] && head -1 "$RES/CARD_lx45g1b_rev_DONE" | grep -q tier1 || m=$((m + 1))
  LASTS=$(( LAST0 - m * 4500 ))
}
STATUS="armed"; RPID=""; RSTART=0; VE=""
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
  echo "STK $UPH END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
  say "end (rc $rc): $STATUS; wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
away() { "$AWAY" 2>/dev/null; }
held() {        # 0 = held: Kamen's HOLD, the PAUSE file, Kamen at the Mac (one read; no reading counts as away), or the
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }          # lid shut with no external display (runner 64's lid_shut)
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
> (a quote from the project owner, kept private)
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
back() {        # 0 = Kamen is at the Mac on two reads 30 s apart: a running step yields
  local a; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null || return 1
  sleep "$REREAD"; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null
}
gates() {       # 0 = the start gates hold (ENV set); 1 = not yet; 5 = never
  local m e
  pgrep -f "$RUNPAT" > /dev/null && { WHY="an EvalSuite runner is alive"; return 1; }
  pgrep -f "$G1PAT" > /dev/null && { WHY="G1's chain is alive"; return 1; }
  local IFS_=$IFS a_ mk_ pt_ tm_; IFS=';'
  for a_ in $AFTER; do
    IFS=$IFS_; mk_=${a_%%|*}; pt_=${a_#*|}; tm_=${pt_#*|}; pt_=${pt_%%|*}
    { [ -e "$mk_" ] && [ "$(stat -f %m "$mk_")" -ge "$ARMED" ]; } && continue
    { [ "$(date +%s)" -ge "$(ep "$tm_")" ] && ! pgrep -f "$pt_" > /dev/null; } && continue
    WHY="no $mk_ after the arming$(pgrep -f "$pt_" > /dev/null && echo " (its job runs or waits)" || echo " (none of its job before $tm_)")"; IFS=$IFS_; return 1
  done; IFS=$IFS_
  if [ -n "$VE" ]; then e=$VE                       # the env is read once
  else
    g1dec || { WHY=$DECWHY; return 5; }
    e=$DEC
    lasts                                          # the latest first start, by the base orders still to run
    if [ -z "$e" ]; then
      [ "$(date +%s)" -ge $(( LASTS - 900 )) ] || { WHY="$DECWHY: waiting for G1's decision (stk0 from $(hm $(( LASTS - 900 ))))"; return 1; }
      e=stk0; DECWHY="$DECWHY by $(hm $(( LASTS - 900 ))): stk0 (NewIdeas 07:0x: no G1 verdict by the latest start)"
    fi
  fi
  envof "$e" || { WHY="the env $e is not pinned (CHAIN_STK* settings)"; return 5; }
  [ -e "$KTMK" ] && [ "$(stat -f %m "$KTMK")" -ge "$(stat -f %m "$VFENV" 2>/dev/null || echo 9999999999)" ] \
    || { WHY="no $KTMK newer than $VFENV"; return 1; }
  grep -qE "^$e: PASS" "$KTMK" || { WHY="$KTMK has no '$e: PASS' line ($(grep -E "^$e:" "$KTMK" | head -1 | cut -c1-80))"; return 1; }
  ENV=$e; WHY="env $e ($DECWHY); key test: $(grep -E "^$e: PASS" "$KTMK" | head -1 | cut -c1-80)"
  return 0
}
run() {         # label, the run's TG (for its DONE file), then runner 69's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/stk_chain_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do     # a HOLD mid-run: the finish trap stops the runner and its children
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
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
speed() {       # information beside the verdict (amendment 3a): kept per check, arm vs base
  [ "$DRY" = 1 ] && return
  local o=results/lossygate/speed_$1_$(date +%Y-%m-%d).txt
  .venv/bin/python tools/lossy_speed.py "$1" "$2" > "$o" 2>&1
  say "speed $1 vs $2 (rc $?): $(tail -1 "$o" | cut -c1-160)"
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
  STATUS="B1"; RAN=0; step "B1 base forward (made if G1's chain did not)" lx45g1b 1 card_lx45g1b_t1code "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX"; r=$?
  R="$R B1 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  STATUS="B2"; RAN=0; step "B2 base reverse (made if G1's chain did not)" lx45g1b_rev 1 card_lx45g1b_t1code_rev "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX" CARD_ORDER=rev; r=$?
  R="$R B2 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  STATUS="SMOKE"; RAN=0; step "SMOKE $ENV card tier, 3 keys, both orders (a function check)" "${VT}_smoke" 2 "card_${VT}_smoke_t2code_fwd" "${P[@]}" "${V[@]}" \
    CARD_TIER=2 "CARD_REF=$NAXN" "CARD_KEYSFILE=$SMOKEKF" CARD_RUNSUF=_smoke; r=$?
  R="$R SMOKE rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  if [ "$DRY" = 1 ]; then cardmem                  # the card gate's memory: once per smoke (a newer STK_CARD_MEM is kept)
  elif ! { [ -s results/STK_CARD_MEM ] && [ "$(stat -f %m results/STK_CARD_MEM)" -ge "$(stat -f %m "results/CARD_${VT}_smoke_DONE")" ]; }; then
    cardmem || return 9
  fi
  STATUS="T1F"; RAN=0; step "T1F $ENV forward vs B1" "$VT" 1 "card_${VT}_t1code" "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_REF=card_lx45g1b_t1code CARD_IFEVAL_IDENT=1 "${LATER[@]}"; r=$?
  R="$R T1F rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  [ "$RAN" = 1 ] && speed "$VT" lx45g1b
  if [ "$DRY" != 1 ] && ! verdict "$VT" 1 "card_${VT}_t1code" | grep -q '^LOSSY tier1 (CODE-ONLY mode): PASS'; then ENDS="ended: T1F did not pass, the stack ($ENV) ends (no card gate)"; return 0; fi
  STATUS="T1R"; RAN=0; step "T1R $ENV reverse vs B2, pooled" "${VT}_rev" 1 "card_${VT}_t1code_rev" "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_ORDER=rev \
    CARD_REF=card_lx45g1b_t1code_rev CARD_IFEVAL_IDENT=1 "CARD_FWD_RUN=card_${VT}_t1code" CARD_FWD_REF=card_lx45g1b_t1code "${LATER[@]}"; r=$?
  R="$R T1R rc $r;"; [ "$r" = 0 ] && [ "$RAN" = 1 ] && speed "${VT}_rev" lx45g1b_rev
  [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  ENDS="done: the stack = $ENV ($VT); the both-orders tier-1 verdict in CARD_${VT}_rev_DONE; the card gate's memory in STK_CARD_MEM"; return 0
}
setenv() {      # the env -> the V settings (read once)
  VE=$1; envof "$VE"
  V=(CARD_TAG="$VT" "CARD_FENV=$VFENV" CARD_FBLOB="$VBLOB" "CARD_ENGAGE=$VENG")
}
startwait() {   # wait until the gates hold and nothing holds, on two reads 30 s apart. 0 = start; 1 = no start before the
  local n=0 lasth="" h g w1                        # latest start; 5 = never
> (a quote from the project owner, kept private)
  while :; do
    lasts
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
P=(CARD_SCOPE=code "CARD_TREES=$TREES" CARD_SCREEN_SHA=6039fa8dbea5 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$CUT" CARD_MEM=39)
B=(CARD_TAG=lx45g1b "CARD_FENV=$HOME/BigMoE/data/stack/lx45g1b_env.txt" CARD_FBLOB=2e5b7167af865cc7867c2f1fd8165a5d40fdb9c8)
LATER=(); [ "$DRY" = 1 ] && LATER=(CARD_REF_LATER=1)
R=""; ENDS=""; ENV=""
say "stk_chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): the stack's tier 1 on g1b through $(basename "$RUNNER"), tag $TGL; cutoff $CUT, latest first start $(hm "$LASTS"); after [${AFTER:-none}]; key test $KTMK; the AWAY rule on"
if [ "$DRY" = 1 ]; then
  envof "${CHAIN_DRYENV:?set CHAIN_DRYENV to stk or stk0 for a dry run}" || { say "CHAIN_DRYENV '$CHAIN_DRYENV' is not pinned"; exit 2; }
  ENV=$CHAIN_DRYENV; setenv "$ENV"; steps; exit $?
fi
STATUS="waiting"
startwait; s=$?
[ "$s" = 5 ] && { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
[ "$s" = 1 ] && { STATUS="no start by $(hm "$LASTS"): $WHY"; say "$STATUS"; exit 6; }
say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
setenv "$ENV"; say "the env: $ENV ($DECWHY); $VT, blob $VBLOB; CARD_ENGAGE $VENG"
steps; r=$?
if [ "$r" = 0 ]; then STATUS="${R%;}; $ENDS"; say "$ENDS"; exit 0; fi
if [ "$r" = 75 ]; then STATUS="cut by the cutoff:${R%;} (the rest goes to a re-arm)"; say "$STATUS"; exit 75; fi
STATUS="stopped:${R%;} (rc $r: 9 = no verdict, VOID or INCOMPLETE)"; say "$STATUS"; exit "$r"
