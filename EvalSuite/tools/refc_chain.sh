#!/bin/bash
# refc_chain.sh: the reference contract's Part A (docs/PREREG_refc_2026-11-18.md; CHAIN-16), through runner 70's AR mode
# (tools/day_runner70.sh, CARD_AR=1) on BigMoE's contract snapshot. Built from tools/stk1_chain.sh: the same held / back /
# run / waitfree code, with the AWAY rule ON.
#   Steps: STEP0, the switch-off control (REF2_REFC unset) on the 3 step-0 keys: token-identical to results/ref2tnax, and
#   refc OFF (tools/refc_check.py --off) -> ROWINV, Leap's refc_rowinv.py on the snapshot's own files (REFC_ROOT): "18 of
#   18 cases PASS; modules from ROOT ...: True" and "verdict: PASS" -> AR210, the 210 AR (R57_MTP_K=0, C 192), refc engaged
#   on every run (refc_check.py --on) -> D4X, tools/d4x.py vs ref2tnax -> EXACT, chain-16 (R57_MTP_K=15) on R-56d's 36,
#   refc engaged, and its tokens and kept-row hashes == AR210's (refc_check.py exact). A failed gate (STEP0, ROWINV, an
#   engagement check) is VOID and ends the chain. D4X's STAGE2 (IFEval alone fails) is recorded; stage 2 runs Friday by
#   day under the PREREG, outside this chain.
#   Start gates, read twice 30 s apart: no EvalSuite runner alive (runners 60-70); the markers in CHAIN_AFTER (each
#   written after the arming); the three env files' git blobs are the pinned ones; the PREREG is committed (HEAD) and
#   names all three blobs. No first start after CHAIN_LASTSTART; runner cutoff CHAIN_CUTOFF.
#   Nothing starts while HOLD, PAUSE, a shut lid (no external display) or Kamen at the Mac (away_secs.sh < 900, one
#   read). Mid-run, Kamen at the Mac on two reads 30 s apart makes the step yield; it resumes once nothing holds.
#   EVERY end writes results/REFC_CHAIN_END ("REFC PART A ...: <verdict>"), after the runner and its children exit and
#   peer unlock. Stop it by pid only, with a plain kill (TERM). kill -9 skips the marker.
#   Runs card_c98kctl_ar (STEP0), card_c98kar_ar (AR210), card_c98kch_ar (EXACT). Runner 70 stops on an env file whose
#   comment does not name its tag (a check its dry run skips) or names no data/r57/snap/<dir>/r57_screen*.py, so both are
#   checked at the start, dry runs included, and the three envs must name the same screen.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1); no wait, no gate, no GPU, no marker; ROWINV is skipped.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1] tools/refc_chain.sh
# Pins, set at arming from the PREREG: CHAIN_CTL_ENV / _BLOB, CHAIN_AR_ENV / _BLOB, CHAIN_CH_ENV / _BLOB; CHAIN_TREES /
# CHAIN_SCREEN_SHA / CHAIN_SO_SHA; CHAIN_ROWINV_ROOT (REFC_ROOT) and CHAIN_ROWINV_PY (the snapshot's refc_rowinv.py);
# CHAIN_AFTER (marker|pgrep pattern|time; a semicolon list); CHAIN_LASTSTART, CHAIN_CUTOFF.
# Tests only: CHAIN_QDIR, CHAIN_AWAY, CHAIN_RUNNER, CHAIN_TICK, CHAIN_REREAD, CHAIN_RUNPAT, CHAIN_PREREG.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
TGL=refc; L=results/refc_chain.log; END=${CHAIN_END:-results/REFC_CHAIN_END}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
hm() { date -r "$1" '+%m-%d %H:%M'; }
CUT="${CHAIN_CUTOFF:?pin CHAIN_CUTOFF (YYYY-mm-dd HH:MM)}"; LASTS=$(ep "${CHAIN_LASTSTART:?pin CHAIN_LASTSTART}")
ARMED=$(date +%s); TICK=${CHAIN_TICK:-60}; REREAD=${CHAIN_REREAD:-30}
PREREG=${CHAIN_PREREG:-docs/PREREG_refc_2026-11-18.md}; AFTER=${CHAIN_AFTER-}
RUNPAT=${CHAIN_RUNPAT:-tools/day_runner[67][0-9]\.sh}   # a pgrep -f pattern (runners 60-79)
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner70.sh}
abs() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }   # runner 70 and blob() hash with git -C BigMoE
CTLE=$(abs "${CHAIN_CTL_ENV:?}"); CTLB=${CHAIN_CTL_BLOB:?}; ARE=$(abs "${CHAIN_AR_ENV:?}"); ARB=${CHAIN_AR_BLOB:?}
CHE=$(abs "${CHAIN_CH_ENV:?}"); CHB=${CHAIN_CH_BLOB:?}
RIROOT=${CHAIN_ROWINV_ROOT:?}; RIPY=${CHAIN_ROWINV_PY:?}
K0=results/refc/keys_step0.txt; K210=results/refc/keys_core210.txt; K36=results/refc/keys_r56.txt
say() { echo "$(date '+%m-%d %H:%M:%S') [$TGL] $*" >> "$L"; }
STATUS="armed"; VERDICT="none"; RPID=""; RSTART=0
stoprunner() {
  { [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; } || { RPID=""; return 0; }
  say "stopping runner $RPID and its children"
  pkill -TERM -P "$RPID"; kill -TERM "$RPID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$RPID" 2>/dev/null || break; sleep 10; done
  pkill -KILL -P "$RPID" 2>/dev/null; kill -KILL "$RPID" 2>/dev/null
  wait "$RPID" 2>/dev/null; RPID=""
}
finish() {
  local rc=$?
  stoprunner
  [ "$DRY" = 1 ] && { say "DRY end (rc $rc): no marker"; exit "$rc"; }
  peer unlock > /dev/null 2>&1
  echo "REFC PART A $(date '+%m-%d %H:%M:%S'): $VERDICT ($STATUS)" > "$END"
  say "end (rc $rc): $VERDICT ($STATUS); wrote $END"
}
trap finish EXIT
trap 'STATUS="stopped by a signal during: $STATUS"; exit 143' TERM INT
away() { "$AWAY" 2>/dev/null; }
held() {
  [ -e "$Q/HOLD" ] && { HELD="HOLD"; return 0; }
  [ -e "$Q/PAUSE" ] && { HELD="PAUSE"; return 0; }
> (a quote from the project owner, kept private)
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  HELD="the lid shut, no external display"; return 0
}
back() {
  local a; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null || return 1
  sleep "$REREAD"; a=$(away); [ "${a:-900}" -lt 900 ] 2>/dev/null
}
blob() { git -C "$HOME/BigMoE" hash-object "$1" 2>/dev/null; }
named() {       # runner 70's comment reads (its line 633 + the snapshot regex): each env names its tag and the one screen
  local t_ e_ c_ s_ s0_=""
  for t_ in "c98kctl|$CTLE" "c98kar|$ARE" "c98kch|$CHE"; do
    e_=${t_#*|}; c_=$(grep '^#' "$e_" 2>/dev/null | paste -sd' ' -)
    echo "$c_" | tr -d '-' | grep -qi "${t_%%|*}" || { WHY="$e_'s comment does not name the tag ${t_%%|*}"; return 1; }
    s_=$(echo "$c_" | grep -oE 'data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py' | head -1)
    [ -n "$s_" ] && [ -f "$HOME/BigMoE/$s_" ] || { WHY="$e_'s comment names no BigMoE data/r57/snap/<dir>/r57_screen*.py on disk"; return 1; }
    [ -z "$s0_" ] || [ "$s_" = "$s0_" ] || { WHY="$e_ names screen $s_, not $s0_"; return 1; }
    s0_=$s_
  done
  WHY="the three envs name their tags and the screen $s0_"
}
gates() {       # 0 = the start gates hold; 1 = not yet; 5 = never
  pgrep -f "$RUNPAT" > /dev/null && { WHY="an EvalSuite runner is alive"; return 1; }
  local IFS_=$IFS a_ mk_ pt_ tm_ e_ b_; IFS=';'
  for a_ in $AFTER; do
    IFS=$IFS_; mk_=${a_%%|*}; pt_=${a_#*|}; tm_=${pt_#*|}; pt_=${pt_%%|*}
    { [ -e "$mk_" ] && [ "$(stat -f %m "$mk_")" -ge "$ARMED" ]; } && continue
    { [ "$(date +%s)" -ge "$(ep "$tm_")" ] && ! pgrep -f "$pt_" > /dev/null; } && continue
    WHY="no $mk_ after the arming$(pgrep -f "$pt_" > /dev/null && echo " (its job runs or waits)" || echo " (none of its job before $tm_)")"; IFS=$IFS_; return 1
  done; IFS=$IFS_
  for e_ in "$CTLE|$CTLB" "$ARE|$ARB" "$CHE|$CHB"; do
    b_=$(blob "${e_%%|*}"); [ "$b_" = "${e_#*|}" ] || { WHY="${e_%%|*}'s blob $b_ is not the pinned ${e_#*|}"; return 5; }
    git show "HEAD:$PREREG" 2>/dev/null | grep -q "${e_#*|}" || { WHY="$PREREG at HEAD does not name blob ${e_#*|}"; return 1; }
  done
  WHY="the PREREG filed ($(git log -1 --format=%h -- "$PREREG")); the env blobs pinned"
  return 0
}
run() {         # label, tag, then runner 70's K=V settings. 75 = it yielded to Kamen
  local lab=$1 tg=$2 rc; shift 2
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/refc_chain_runner.out 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do
    [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD during: $STATUS"; say "HOLD: stopping $lab"; exit 8; }
    if [ "$DRY" != 1 ] && back && kill -0 "$RPID" 2>/dev/null; then
> (a quote from the project owner, kept private)
      stoprunner; peer unlock > /dev/null 2>&1; return 75
    fi
    sleep 15
  done
  wait "$RPID"; rc=$?; RPID=""
  local df=results/CARD_${tg}_ar_DONE
  say "$lab: rc $rc; $(if [ -e "$df" ] && [ "$(stat -f %m "$df")" -ge "$RSTART" ]; then cut -c1-220 "$df"; else echo 'no DONE from this run'; fi)"
  return "$rc"
}
waitfree() {
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
rows() {        # label, tag, keys file, env, blob: runner 70's AR mode until its DONE says AR ROWS (resumes after a yield)
  local lab=$1 tg=$2 kf=$3 fe=$4 fb=$5 rc res=""
  if [ -s "results/CARD_${tg}_ar_DONE" ] && head -1 "results/CARD_${tg}_ar_DONE" | grep -q '^AR ROWS'; then say "$lab: already done"; return 0; fi
  while :; do
    [ -e "results/card_${tg}_ar" ] && [ "$DRY" != 1 ] && res=CARD_RESUME=1
    run "$lab" "$tg" "${P[@]}" CARD_TAG="$tg" "CARD_KEYSFILE=$kf" "CARD_FENV=$fe" CARD_FBLOB="$fb" $res; rc=$?
    [ "$rc" = 75 ] || break
    waitfree || return 75
  done
  [ "$DRY" = 1 ] && return "$rc"
  [ "$rc" = 0 ] && head -1 "results/CARD_${tg}_ar_DONE" 2>/dev/null | grep -q '^AR ROWS' && return 0
  [ "$rc" = 6 ] && return 75
  return 9
}
check() {       # label, then the command: its last line is logged; 0 = it passed
  local lab=$1 o rc; shift
  [ "$DRY" = 1 ] && { say "DRY: $lab would run: $*"; return 0; }
  o=$("$@" 2>&1); rc=$?
  say "$lab: $(echo "$o" | tail -1 | cut -c1-240)"
  return "$rc"
}
same0() {       # STEP0's tokens: the control's rows == results/ref2tnax's on every key of K0
  .venv/bin/python - "$K0" <<'S0'
import json, sys
ks = open(sys.argv[1]).read().split()
ids = lambda r: {(x := json.loads(l))["key"]: x["completion_ids"] for l in open(f"results/{r}/decode.jsonl")}
a, b = ids("card_c98kctl_ar"), ids("ref2tnax")
same = [k for k in ks if k in a and a[k] == b.get(k)]
print(f"STEP0 tokens: the switch-off control == results/ref2tnax on {len(same)} of {len(ks)} keys"
      + ("" if len(same) == len(ks) else f" (differ: {[k for k in ks if k not in same]})"))
sys.exit(0 if len(same) == len(ks) else 1)
S0
}
rowinv() {
  [ "$DRY" = 1 ] && { say "DRY: ROWINV would run REFC_ROOT=$RIROOT $RIPY"; return 0; }
  local o=results/refc/rowinv_$(date +%Y-%m-%d_%H%M).txt
  REFC_ROOT="$RIROOT" "$HOME/BigMoE/.venv/bin/python" "$RIPY" > "$o" 2>&1
  local f1 lz; f1=$(grep -m1 "^REFC ROW-INVARIANCE" "$o"); lz=$(tail -1 "$o")
  say "ROWINV: $(echo "$f1" | cut -c1-160) / $lz"
  echo "$f1" | grep -q "18 of 18 cases PASS; modules from ROOT $RIROOT: True" && [ "$lz" = "verdict: PASS" ]
}
steps() {
  local r
  STATUS="STEP0"; rows "STEP0 the switch-off control, 3 keys" c98kctl "$K0" "$CTLE" "$CTLB"; r=$?; [ "$r" = 0 ] || return "$r"
  [ "$DRY" = 1 ] || { check "STEP0 tokens" same0 && check "STEP0 refc off" .venv/bin/python tools/refc_check.py card_c98kctl_ar --off; } \
    || { VERDICT="VOID (STEP0)"; return 9; }
  STATUS="ROWINV"; rowinv || { VERDICT="VOID (ROWINV)"; return 9; }
  STATUS="AR210"; rows "AR210 the contract's AR on the 210" c98kar "$K210" "$ARE" "$ARB"; r=$?; [ "$r" = 0 ] || return "$r"
  check "AR210 refc engaged" .venv/bin/python tools/refc_check.py card_c98kar_ar --on --shapes 7 || { VERDICT="VOID (AR210 not engaged)"; return 9; }
  STATUS="D4X"; local d4; d4=$([ "$DRY" = 1 ] && echo DRY || .venv/bin/python tools/d4x.py card_c98kar_ar ref2tnax 2>&1 | tail -1)
  say "D4X vs ref2tnax: $d4"
  STATUS="EXACT"; rows "EXACT chain-16 on R-56d's 36" c98kch "$K36" "$CHE" "$CHB"; r=$?; [ "$r" = 0 ] || return "$r"
  check "EXACT refc engaged" .venv/bin/python tools/refc_check.py card_c98kch_ar --on --shapes 7 || { VERDICT="VOID (EXACT not engaged)"; return 9; }
  check "EXACT chain-16 == AR" .venv/bin/python tools/refc_check.py exact card_c98kch_ar card_c98kar_ar; local ex=$?
  if [ "$DRY" = 1 ]; then VERDICT="DRY"
  elif [ "$ex" != 0 ]; then VERDICT="FAIL (the loop != its own AR; $d4)"
  else case "$d4" in "D4X: PASS"*) VERDICT="PASS ($d4; exactness PASS)" ;;
                     "D4X: DIFFERS"*) VERDICT="PASS ($d4; exactness PASS)" ;;
                     "D4X: STAGE2"*) VERDICT="STAGE 2 WANTED ($d4; exactness PASS): stage 2 runs Friday by day" ;;
                     *) VERDICT="FAIL ($d4; exactness PASS)" ;; esac
  fi
  return 0
}
startwait() {
  local n=0 lasth="" h g w1
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
P=(CARD_AR=1 CARD_TIER=1 CARD_SCOPE=all CARD_REF=ref2tnax CARD_C=192 "CARD_TREES=${CHAIN_TREES:?pin CHAIN_TREES}" \
   "CARD_SCREEN_SHA=${CHAIN_SCREEN_SHA:?pin CHAIN_SCREEN_SHA}" "CARD_SO_SHA=${CHAIN_SO_SHA:?pin CHAIN_SO_SHA}" "CARD_CUTOFF=$CUT" CARD_MEM=39)
say "refc_chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): the contract's Part A through $(basename "$RUNNER"); cutoff $CUT, latest first start $(hm "$LASTS"); after [${AFTER:-none}]"
named || { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
say "$WHY"
if [ "$DRY" = 1 ]; then steps; exit $?; fi
STATUS="waiting"
startwait; s=$?
[ "$s" = 5 ] && { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
[ "$s" = 1 ] && { STATUS="no start by $(hm "$LASTS"): $WHY"; say "$STATUS"; exit 6; }
steps; r=$?
if [ "$r" = 0 ]; then STATUS="done"; say "REFC PART A: $VERDICT"; exit 0; fi
if [ "$r" = 75 ]; then STATUS="cut by the cutoff (the rest goes to a re-arm)"; say "$STATUS"; exit 75; fi
STATUS="stopped (rc $r)"; say "$STATUS: $VERDICT"; exit "$r"
