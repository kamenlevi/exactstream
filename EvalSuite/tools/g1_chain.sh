#!/bin/bash
# g1_chain.sh: the check-bytes gate G1 (PREREG_lossygate amendments 5 / 5a-5f and G1's arming addendum, dated Tue 01:4x),
# CODE-ONLY, through runner 68 (tools/day_runner68.sh) on BigMoE's snapshot g1b: the base lx45g1b, then the env that
# NewIdeas' 5d pick names (results/G1_PICK). Built from tools/t1c_chain2.sh (822c2fc2db41f880): the same steps and the
# same step / run / held / back / waitfree code, with the AWAY rule ON (4j / 5b / 5f dated lines). A new file and a new
# name, because BigMoE's wrapper matches chain2's.
#   Steps: B0 base tier 0 -> V0 G1 tier 0 vs B0 (KILL ends the gate) -> B1 base forward (+ the unset-path check vs
#   lx45t1c1 forward, information) -> V1 G1 forward vs B1 (IFEval identity; not PASS ends the gate) -> B2 base reverse
#   (+ the unset-path check vs lx45t1c1 reverse, information) -> V2 G1 reverse vs B2, both orders pooled. A step whose
#   verdict is recorded is skipped. A cut tier-1 run resumes (CARD_RESUME=1); a cut tier-0 run is moved aside and
#   starts fresh (amendment 4). After V1 / V2: tools/lossy_speed.py (information).
#   Start gates, all read twice 30 s apart: results/T1C_TUEF3_END written after the arming, with follower3 gone; no
#   EvalSuite runner alive; V2's both-orders verdict (PASS or FAIL) in results/CARD_t1cx45t1c1_rev_DONE; BigMoE's probe
#   re-run marker ~/BigMoE/data/r57/G1P_PROBE_DONE (any end, mtime >= 11-17 01:15:48); and results/G1_PICK, newer than
#   V2's verdict, whose first line "G1 PICK <mm-dd HH:MM:SS>: <env> ..." names one of the eight pinned envs (any other
#   name: no run, rc 5). The pick is read once, at the first start; a later change stops the chain (rc 5).
#   Two windows (NewIdeas 01:23): DAY to CHAIN_CUTOFF (default Tue 23:30). Then NIGHT, which starts only after
#   CHAIN_NIGHT_MK (default BigMoE's NIGHT1117_BIGMOE_END, written on any end of M4 + its capture) is written after the
#   arming, to CHAIN_NIGHT_CUT (default Wed 05:15, for MoonShots' 05:24 cron). No start in a window's last 15 min. A step
#   cut by the day window resumes in the night window. What is left after it goes to Wednesday (a re-arm, in NewIdeas'
#   order).
#   Nothing starts while HOLD, PAUSE, a shut lid (no external display) or Kamen at the Mac (away_secs.sh < 900, one
#   read; no reading counts as away). Mid-run, Kamen at the Mac on two reads 30 s apart makes the step yield (the runner
#   and its children TERM, peer unlock); it resumes once nothing holds. Runner 68 gates each process on NO-SWAP 43 GB
#   (CARD_MEM 39), keeps the +2 GB swap guard, and yields at each process start to another session's run above 5 GB.
#   EVERY end writes results/G1_<TAG>_END (tag g1t), after the runner and its children exit and peer unlock. Stop it by
#   pid only, with a plain kill (TERM). kill -9 skips the marker.
# CHAIN_DRY=1: every runner call as a dry run (CARD_DRY=1, CARD_REF_LATER=1 for references made by earlier steps), on the
# env CHAIN_DRYENV names; no wait, no gate, no GPU, no marker.
#   CHAIN_MYPID=<this session's Claude pid> [CHAIN_DRY=1 CHAIN_DRYENV=g1r] tools/g1_chain.sh
# Tests only: CHAIN_QDIR, CHAIN_AWAY, CHAIN_RUNNER, CHAIN_T1CEND, CHAIN_V2DONE, CHAIN_PROBE, CHAIN_PROBEAFTER, CHAIN_PICK,
# CHAIN_TICK (the wait loops' sleep, default 60 s), CHAIN_REREAD (the second read's delay, default 30 s), CHAIN_F3PAT and
# CHAIN_RUNPAT (the pgrep patterns for follower3 and for an EvalSuite runner).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
DRY=${CHAIN_DRY:-0}; MYPID=${CHAIN_MYPID:?set CHAIN_MYPID to this session s Claude pid}
TGL=${CHAIN_TAG:-g1t}; UPH=$(echo "$TGL" | tr a-z A-Z); L=results/g1_chain.log; END=results/G1_${UPH}_END
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1:00" +%s; }
hm() { date -r "$1" '+%m-%d %H:%M'; }
DCUT="${CHAIN_CUTOFF:-2026-11-17 23:30}"; NCUT="${CHAIN_NIGHT_CUT:-2026-11-18 05:15}"; CUT=$DCUT; GIVEUP=$(( $(ep "$CUT") - 900 ))
ARMED=$(date +%s); TICK=${CHAIN_TICK:-60}; REREAD=${CHAIN_REREAD:-30}
T1E=${CHAIN_T1CEND:-results/T1C_TUEF3_END}; V2D=${CHAIN_V2DONE:-results/CARD_t1cx45t1c1_rev_DONE}
PROBE=${CHAIN_PROBE:-$HOME/BigMoE/data/r57/G1P_PROBE_DONE}; PROBEAFTER=${CHAIN_PROBEAFTER:-$(ep "2026-11-17 01:15")}
PICKF=${CHAIN_PICK:-results/G1_PICK}; NMK=${CHAIN_NIGHT_MK:-$HOME/BigMoE/data/r57/NIGHT1117_BIGMOE_END}
F3PAT=${CHAIN_F3PAT:-tools/t1c_tue_follower[3]\.sh}; RUNPAT=${CHAIN_RUNPAT:-tools/day_runner6[0-9]\.sh}   # pgrep -f patterns
Q=${CHAIN_QDIR:-$HOME/.local/share/quiet}                  # HOLD and PAUSE live here
AWAY=${CHAIN_AWAY:-$HOME/.local/share/quiet/away_secs.sh}; RUNNER=${CHAIN_RUNNER:-tools/day_runner68.sh}
BMS=$HOME/BigMoE/data/stack; EX=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2
TREES=data/r57/snap/g1b=fb0ee18288dae0ac5528006557dfaf8487d63c3b,data/r57/snap/frozen_g1b=2584e36a814ae956ac5c2d9c9090c4d787298502,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
ONE_C=lossy_check:e3+edrop+lmsl_rows; RES=lossy_check:res_opt@run,lossy_check:opt_post@run; COM=lossy_check:com_rows@run,lossy_check:com_opt@run
pickenv() {     # env -> VBLOB / VENG (the pinned blobs, BigMoE a808a932, checked 11-16 19:4x and 11-17 01:1x; 5d's CARD_ENGAGE)
  case "$1" in
    g1)    VBLOB=b736d293176e83b12ecc4c8d18ad665d7c38ed4f; VENG=$ONE_C ;;
    g1r)   VBLOB=e42de6d974f25572644109a299cd690522450b3c; VENG=$ONE_C,$RES ;;
    g1c)   VBLOB=bfd66cedd73b692b4a9cabb62b49f5ac5a345f2c; VENG=$ONE_C,$COM ;;
    g1cr)  VBLOB=78f7932b9c72e6ece3f4af6eedd6651dbbd1e3bb; VENG=$ONE_C,$COM,$RES ;;
    g1cd)  VBLOB=efb0de2be0b1ede610880bc460c3e87efcec440f; VENG=$ONE_C,$COM ;;
    g1cdr) VBLOB=f4eac1d6bd3705417c7d144c056ede62d3a588ae; VENG=$ONE_C,$COM,$RES ;;
    g1l)   VBLOB=1c3e2358e53d42d28421d622475de05392beb619; VENG=$ONE_C ;;
    g1lr)  VBLOB=caab79159e3359af84dcb0e5f26582a93e838253; VENG=$ONE_C,$RES ;;
    *) return 1 ;;
  esac
}
pickof() { head -1 "$PICKF" 2>/dev/null | sed -nE 's/^G1 PICK [0-9]{2}-[0-9]{2} [0-9:]{8}: ([a-z0-9]+)( .*)?$/\1/p'; }
say() { echo "$(date '+%m-%d %H:%M:%S') [$TGL] $*" >> "$L"; }
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
  echo "G1 $UPH END $(date '+%m-%d %H:%M:%S'): $STATUS" > "$END"
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
gates() {       # 0 = G1's start gates hold; 1 = not yet; 5 = never (the pick names no pinned env, or it changed)
  { [ -e "$T1E" ] && [ "$(stat -f %m "$T1E")" -ge "$ARMED" ]; } || { WHY="no $T1E written after the arming"; return 1; }
  pgrep -f "$F3PAT" > /dev/null && { WHY="follower3 still runs"; return 1; }
  pgrep -f "$RUNPAT" > /dev/null && { WHY="an EvalSuite runner is alive"; return 1; }
  head -1 "$V2D" 2>/dev/null | grep -qE '^LOSSY tier1 \(CODE-ONLY mode\): (PASS|FAIL)' || { WHY="no both-orders PASS / FAIL verdict in $V2D"; return 1; }
  { [ -e "$PROBE" ] && [ "$(stat -f %m "$PROBE")" -ge "$PROBEAFTER" ]; } || { WHY="no G1P_PROBE_DONE after $(hm "$PROBEAFTER") yet"; return 1; }
  [ -s "$PICKF" ] || { WHY="no G1_PICK yet"; return 1; }
  [ "$(stat -f %m "$PICKF")" -ge "$(stat -f %m "$V2D")" ] || { WHY="G1_PICK is older than V2's verdict"; return 1; }
  local e; e=$(pickof)
  pickenv "$e" || { WHY="G1_PICK names no pinned env: $(head -1 "$PICKF" | cut -c1-120)"; return 5; }
  [ -z "$VE" ] || [ "$e" = "$VE" ] || { WHY="G1_PICK changed from $VE to $e after the start"; return 5; }
  WHY="T1C_TUEF3_END ($(head -c 70 "$T1E" | tr '\n' ' ')); V2 ($(head -1 "$V2D" | cut -c1-60)); probe ($(head -c 60 "$PROBE" | tr '\n' ' ')); pick $e"
  return 0
}
nightmk() { { [ -e "$NMK" ] && [ "$(stat -f %m "$NMK")" -ge "$ARMED" ]; } || { WHY="no night marker $NMK after the arming"; return 1; }; }
run() {         # label, the run's TG (for its DONE file), then runner 68's K=V settings; the runner is a stoppable child.
  local lab=$1 tg=$2 rc; shift 2                   # 75: it yielded to Kamen (not started, or stopped; nothing left running)
  [ -e "$Q/HOLD" ] && { STATUS="stopped by HOLD before: $STATUS"; say "HOLD: $lab not started"; exit 8; }
> (a quote from the project owner, kept private)
  say "$lab: $*"
  RSTART=$(date +%s)
  env CARD_DRY="$DRY" "$@" "$RUNNER" >> results/g1_chain_runner.out 2>&1 &
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
waitfree() {    # after a yield: until nothing holds (a HOLD ends the chain); 1 = no start left before the window's cutoff
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
verdict() {     # TG, tier, run dir: the step's recorded verdict line (none: rc 1). Tier 0: DONE.t0. Tier 1: DONE (tier1,
  local tg=$1 tier=$2 rd=$3 f                      # VOID or INCOMPLETE) with the tier-1 run dir present, since a tier-0
  if [ "$tier" = 0 ]; then f=results/CARD_${tg}_DONE.t0; else f=results/CARD_${tg}_DONE; fi   # VOID shares the file
  [ -s "$f" ] || return 1
  if [ "$tier" = 1 ]; then { [ -d "results/$rd" ] && head -1 "$f" | grep -qE 'tier1|VOID|INCOMPLETE'; } || return 1; fi
  head -1 "$f"
}
step() {        # label, TG, tier, run dir, then runner 68's settings. 0 = a verdict is recorded; 75 = cut by the window;
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
unset_path() {  # run, reference run: tools/t1c_unset_path.py (amendments 4 / 5: information, flagged if it fails)
  [ "$DRY" = 1 ] && return
  local o=results/lossygate/unset_path_$1_$(date +%Y-%m-%d).txt rc
  .venv/bin/python tools/t1c_unset_path.py "$1" "$2" > "$o" 2>&1; rc=$?
  say "unset-path check $1 vs $2 (information; rc $rc$([ "$rc" = 0 ] || echo ': FLAG')): $(tail -1 "$o" | cut -c1-160)"
}
steps() {       # 0 = the gate ended (done, or a decision: V0 KILL, V1 not PASS); 75 = cut by the window; other = a stop
  local r                                          # (9: no verdict, VOID or INCOMPLETE)
  STATUS="B0"; RAN=0; step "B0 base tier 0" lx45g1b 0 card_lx45g1b_t0code "${P[@]}" "${B[@]}" CARD_TIER=0 "CARD_REF=$EX"; r=$?
  R="$R B0 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  STATUS="V0"; RAN=0; step "V0 $VE tier 0 vs B0" "$VT" 0 "card_${VT}_t0code" "${P[@]}" "${V[@]}" CARD_TIER=0 CARD_REF=card_lx45g1b_t0code "${LATER[@]}"; r=$?
  R="$R V0 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  if [ "$DRY" != 1 ] && verdict "$VT" 0 "card_${VT}_t0code" | grep -q ': KILL'; then ENDS="ended: V0 KILL, G1 ($VE) ends at tier 0"; return 0; fi
  STATUS="B1"; RAN=0; step "B1 base forward" lx45g1b 1 card_lx45g1b_t1code "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX"; r=$?
  R="$R B1 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  [ "$RAN" = 1 ] && unset_path card_lx45g1b_t1code card_lx45t1c1_t1code
  STATUS="V1"; RAN=0; step "V1 $VE forward vs B1" "$VT" 1 "card_${VT}_t1code" "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_REF=card_lx45g1b_t1code CARD_IFEVAL_IDENT=1 "${LATER[@]}"; r=$?
  R="$R V1 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  [ "$RAN" = 1 ] && speed "$VT" lx45g1b
  if [ "$DRY" != 1 ] && ! verdict "$VT" 1 "card_${VT}_t1code" | grep -q '^LOSSY tier1 (CODE-ONLY mode): PASS'; then ENDS="ended: V1 did not pass, G1 ($VE) ends"; return 0; fi
  STATUS="B2"; RAN=0; step "B2 base reverse" lx45g1b_rev 1 card_lx45g1b_t1code_rev "${P[@]}" "${B[@]}" CARD_TIER=1 "CARD_REF=$EX" CARD_ORDER=rev; r=$?
  R="$R B2 rc $r;"; [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  [ "$RAN" = 1 ] && unset_path card_lx45g1b_t1code_rev card_lx45t1c1_t1code_rev
  STATUS="V2"; RAN=0; step "V2 $VE reverse vs B2, pooled" "${VT}_rev" 1 "card_${VT}_t1code_rev" "${P[@]}" "${V[@]}" CARD_TIER=1 CARD_ORDER=rev \
    CARD_REF=card_lx45g1b_t1code_rev CARD_IFEVAL_IDENT=1 "CARD_FWD_RUN=card_${VT}_t1code" CARD_FWD_REF=card_lx45g1b_t1code "${LATER[@]}"; r=$?
  R="$R V2 rc $r;"; [ "$r" = 0 ] && [ "$RAN" = 1 ] && speed "${VT}_rev" lx45g1b_rev
  [ "$r" = 0 ] || [ "$DRY" = 1 ] || return "$r"
  ENDS="done: G1 = $VE (verdicts in the CARD_*_DONE files; the both-orders verdict in CARD_${VT}_rev_DONE)"; return 0
}
setpick() {     # the env -> the V settings (read once)
  VE=$1; pickenv "$VE"; VT=${VE}45g1b
  V=(CARD_TAG="$VT" "CARD_FENV=$BMS/${VT}_env.txt" CARD_FBLOB="$VBLOB" "CARD_ENGAGE=$VENG")
}
startwait() {   # window: wait until the gates (and, by night, the night marker) hold and nothing holds, on two reads 30 s
  local w=$1 n=0 lasth="" h g w1               # apart. 0 = start; 1 = no start before the window's last start; 5 = never
> (a quote from the project owner, kept private)
  while :; do
    h=$(held && echo "$HELD")
    if [ "$h" != "$lasth" ]; then
      if [ -n "$h" ]; then say "held: $h (no start while held)"; else say "no longer held"; fi
      lasth=$h
    fi
    [ "$(date +%s)" -lt "$GIVEUP" ] || { WHY="${h:+held: $h; }$(gates; echo "$WHY")"; return 1; }
    gates; g=$?
    [ "$g" = 5 ] && return 5
    if [ "$g" = 0 ] && [ "$w" = night ] && ! nightmk; then g=1; fi
    if [ -z "$h" ] && [ "$g" = 0 ]; then
      w1=$WHY; sleep "$REREAD"
      if ! held && gates && { [ "$w" != night ] || nightmk; }; then say "start ($w): $w1; read again ${REREAD} s later"; return 0; fi
      say "the start condition did not hold ${REREAD} s later: waiting"
    fi
    n=$((n + 1))
    [ $((n % 30)) = 0 ] && say "waiting: ${h:+held: $h; }$(gates; echo "$WHY")"
    sleep "$TICK"
  done
}
P=(CARD_SCOPE=code "CARD_TREES=$TREES" CARD_SCREEN_SHA=6039fa8dbea5 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$CUT" CARD_MEM=39)
B=(CARD_TAG=lx45g1b "CARD_FENV=$BMS/lx45g1b_env.txt" CARD_FBLOB=2e5b7167af865cc7867c2f1fd8165a5d40fdb9c8)
LATER=(); [ "$DRY" = 1 ] && LATER=(CARD_REF_LATER=1)
R=""; ENDS=""
say "g1_chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY'); session $MYPID): G1 on g1b through $(basename "$RUNNER"), tag $TGL; day cutoff $DCUT, night cutoff $NCUT; the AWAY rule on"
if [ "$DRY" = 1 ]; then
  pickenv "${CHAIN_DRYENV:?set CHAIN_DRYENV to one of the eight envs for a dry run}" || { say "CHAIN_DRYENV '$CHAIN_DRYENV' is not a pinned env"; exit 2; }
  setpick "$CHAIN_DRYENV"; steps; exit $?
fi
for w in day night; do
  if [ "$w" = night ]; then CUT=$NCUT; GIVEUP=$(( $(ep "$CUT") - 900 )); P=(CARD_SCOPE=code "CARD_TREES=$TREES" CARD_SCREEN_SHA=6039fa8dbea5 CARD_SO_SHA=ff69fa24a745 "CARD_CUTOFF=$CUT" CARD_MEM=39); fi
  STATUS="waiting ($w window)"
  startwait "$w"; s=$?
  [ "$s" = 5 ] && { STATUS="no run: $WHY"; say "$STATUS"; exit 5; }
  if [ "$s" = 1 ]; then say "no start in the $w window by $(hm "$GIVEUP") ($WHY)"; R="$R $w: no start;"; continue; fi
  say "free NO-SWAP $(.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))") GB; swap $(sysctl -n vm.swapusage | awk '{print $3}')"
  [ -n "$VE" ] || { setpick "$(pickof)"; say "the pick: $VE ($(head -1 "$PICKF" | cut -c1-160)); blob $VBLOB; CARD_ENGAGE $VENG"; }
  R="$R $w:"; steps; r=$?
  if [ "$r" = 0 ]; then STATUS="${R%;}; $ENDS"; say "$ENDS"; exit 0; fi
  if [ "$r" != 75 ]; then STATUS="stopped:${R%;} (rc $r: 9 = no verdict, VOID or INCOMPLETE)"; say "$STATUS"; exit "$r"; fi
  say "the $w window cut G1 (${R%;}): $([ "$w" = day ] && echo "it resumes in the night window" || echo "the rest goes to Wednesday (a re-arm)")"
done
STATUS="left for Wednesday:${R%;}"; say "$STATUS"; exit 75
