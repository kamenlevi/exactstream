#!/bin/bash
# run_rec1a_c25_slot.sh (Breakthrough, Wed 18 Nov 2026 15:1x): the RE-RUN of the module 1a engine run (rec1a_c25,
# pre-registered in docs/MODULE1A.md 'Tue 06:5x'; the 14:59 run was a not-run, its gate lookup fixed in adacf34), in the
# slot NewIdeas gave at 15:1x. Window 1: from START1 (16:03) to the latest start LATEST1 (18:30). If no run has started
# by then, window 2: from START2 (Thu 07:30) to LATEST2 (Thu 09:10, so a 20-min run ends by 09:30).
# In a window it waits, holding nothing, until ALL of these hold on two reads 30 s apart (checked every minute): no
# PAUSE, the lid open, no HEALTH ACT, no c25 / rec1a Python loop, Kamen away (away_secs >= 900). Then the flag check
# (the real loop up to its model load; once per window), then NO-SWAP (free + purgeable + file-backed, the lower of two
# reads 30 s apart) >= GATE (47 = the measured peak 43.06 + 4, Leap's convention for this harness), then a share lock
# whose reason starts 'MEM 46 GB' (43.06 + the pre-registered 3 GB retained-bytes guard; asked again next minute if
# refused), then NO-SWAP read again 20 s after the lock (below GATE: unlock and wait again).
# The run: a 20-min wall cap; swap +2 GB stops it; HOLD or the lid closing stops it; Kamen back (two reads 30 s apart)
# yields it, and it waits again (same window, else window 2). Stops are TERM only, never -9.
# HOLD (or NewIdeas' BREAK98_HOLD) while waiting: stop. On the FINAL end only (a run of any result, a stop, or no start
# in either window): the lock released, then the END marker data/m1/rec1a_engine/REC1A_END =
# "REC1A END <MM-DD HH:MM:SS>: <why>" (overwritten, so gate on its mtime). Outputs: data/m1/rec1a_engine/rerun_1118/
# (rec1a_c25's .out / .json and c25's report), the runner log, run.txt, env.txt, flagcheck.txt beside it.
# Usage: run_rec1a_c25_slot.sh "<START1>" "<LATEST1>" "<START2>" "<LATEST2>"   (each YYYY-MM-DD HH:MM:SS)
# Env: RJ_TAG (rec1a_1118r), RJ_GATE (47), RJ_MEM (46), RJ_CAP_MIN (20). Simulation: DRY=1 with RJ_DIR (a scratch folder)
# and RJ_END (a scratch END path) REQUIRED; then 1 s waits, gate 1, RJ_JOB in place of the run (default: sleep 3),
# RJ_LOCKCMD (the fake lock, default true), RJ_AWAY_CMD, RJ_NS_CMD, RJ_Q, RJ_CAP_S (the cap in s), RJ_NOFC=1, RJ_SKIPFREE=1.
cd "$HOME/Breakthrough" || exit 2
Q=${RJ_Q:-$HOME/.local/share/quiet}; PY=$HOME/BigMoE/.venv/bin/python
DRY=${DRY:-0}; GATE=${RJ_GATE:-47}; MEMGB=${RJ_MEM:-46}; CAPS=$(( ${RJ_CAP_MIN:-20} * 60 )); TAG=${RJ_TAG:-rec1a_1118r}
D=data/m1/rec1a_engine; END=$D/REC1A_END; OUTD=$HOME/Breakthrough/$D/rerun_1118; ZZ=30; POLL=60; SET=20; GRACE=60
if [ "$DRY" = 1 ]; then
  { [ -n "$RJ_DIR" ] && [ -n "$RJ_END" ]; } || { echo "DRY needs RJ_DIR and RJ_END (scratch paths)"; exit 2; }
  D=$RJ_DIR; END=$RJ_END; OUTD=$RJ_DIR/out; ZZ=1; POLL=1; SET=1; GRACE=2; GATE=${RJ_GATE:-1}; CAPS=${RJ_CAP_S:-$CAPS}
  TAG=${TAG}_dry
fi
mkdir -p "$D" "$OUTD"
L=$D/${TAG}_runner.log; PAT='c25_loo[p][.]py|rec1a_c2[5][.]py'
HOLDS=("$Q/HOLD" "$HOME/NewIdeas/research/week98/BREAK98_HOLD")
[ "$DRY" = 1 ] && HOLDS=("$Q/HOLD")
t() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
S1=$(t "${1:?START1}") && L1=$(t "${2:?LATEST1}") && S2=$(t "${3:?START2}") && L2=$(t "${4:?LATEST2}") || exit 2
LOCKED=0; py=""; WHY=""; RES=""
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
nsr() { if [ -n "$RJ_NS_CMD" ]; then $RJ_NS_CMD; else vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; fi; }
ns2() { local a b; a=$(nsr); sleep $ZZ; b=$(nsr); echo $(( a < b ? a : b )); }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
AWAYCMD=${RJ_AWAY_CMD:-bash $Q/away_secs.sh}
away() { local a; a=$($AWAYCMD 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 0;; *) echo "$a";; esac; }
away_ok() { [ "$(away)" -ge 900 ]; }
back() { [ "$(away)" -lt 900 ] || return 1; sleep $ZZ; [ "$(away)" -lt 900 ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
held() { local f; for f in "${HOLDS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
free_() {   # no PYTHON process runs Leap's c25 loop or rec1a_c25 (a shell whose text names them does not count)
  local p a
  for p in $(pgrep -f "$PAT"); do
    a=$(ps -o args= -p "$p" 2>/dev/null); a=${a%% *}
    case "$a" in *[Pp]ython*) return 1;; esac
  done
  return 0
}
job_env() {   # the run's environment (as the frozen runner's; outputs in OUTD)
  export C25_ARMS=F32s,F32o,F32c6i C25_KEYS=dd__mc_27921 C25_WARM=1 C25_ROUNDS=1 C25_TAG=$TAG
  eval "$(python3 -I -B "$HOME/Leap/scripts/week98/c25/c25_env.py")" || exit 5
  export R57_VTIER= REC1A_OUT="$OUTD" REC1A_TAG=$TAG
  unset C25_PROF C25_CPUSPLIT C25_PASSK
}
flagcheck() {   # the real loop in the run's env up to its model load (no model, no GPU work); 0 = reached it
  ( job_env; export REC1A_FLAGCHECK=1; exec "$PY" -I -B scripts/m1/rec1a/rec1a_c25.py 256 2048 ) > "$D/${TAG}_flagcheck.txt" 2>&1
  local rc=$?
  [ $rc = 97 ] && grep -q "REC1A FLAGCHECK" "$D/${TAG}_flagcheck.txt"
}
lock() {
  if [ "$DRY" = 1 ]; then ${RJ_LOCKCMD:-true} || return 1; LOCKED=1; return 0; fi
  peer lock share $(( CAPS / 60 + 5 ))m --cpu 300 --gpu 60 "MEM $MEMGB GB: Breakthrough module 1a engine re-run (record + replay vs c25's own check, 2 arms; <= $(( CAPS / 60 )) min)" >> "$L" 2>&1 \
    || return 1
  LOCKED=1
}
unlock() { [ $LOCKED = 1 ] || return 0; [ "$DRY" = 1 ] || peer unlock >/dev/null 2>&1; LOCKED=0; }
fin() { unlock; echo "REC1A END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; exit "${2:-0}"; }
onterm() { [ -n "$py" ] && kill -TERM "$py" 2>/dev/null; unlock; say "runner stopped by a signal: no END marker"; exit 143; }
trap onterm TERM INT
trap 'unlock' EXIT
ready() {   # 0 = ready now; else WHY says why. HOLD ends the runner.
  local h; h=$(held) && fin "SKIP: $h present while waiting" 7
  WHY=""
  if [ -e "$Q/PAUSE" ]; then WHY="PAUSE present"
  elif lid_closed; then WHY="the lid is closed"
  elif grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; then WHY="HEALTH ACT"
  elif [ "${RJ_SKIPFREE:-0}" != 1 ] && ! free_; then WHY="a c25 / rec1a loop runs"
> (a quote from the project owner, kept private)
  fi
  [ -z "$WHY" ]
}
run_job() {   # 0 = the run ended (RES set); 2 = it yielded to Kamen
  local lab=$1 sw0 t0 rc s stopped="" res n=0
  sw0=$(swmb); t0=$(date +%s)
  say "running in window $lab (NO-SWAP $(nsr) GB, swap $sw0 MB, wall cap $CAPS s)"
  if [ "$DRY" = 1 ]; then
    JOB=(${RJ_JOB:-sleep 3})
    "${JOB[@]}" >> "$D/${TAG}_run.txt" 2>&1 &
  else
    (
      job_env
      env | sort | grep -E '^(C25_|R57_|BIGMOE_|REC1A_|REF2_|V32_)' > "$D/${TAG}_env.txt"
      exec "$PY" -I -B scripts/m1/rec1a/rec1a_c25.py 256 2048
    ) >> "$D/${TAG}_run.txt" 2>&1 &
  fi
  py=$!
  while kill -0 "$py" 2>/dev/null; do
    sleep $(( DRY == 1 ? 1 : 10 ))
    if [ -n "$stopped" ]; then   # TERM sent: wait it out (TERM again every GRACE s; never -9)
      n=$(( n + 1 )); [ $(( n * (DRY == 1 ? 1 : 10) )) -ge $GRACE ] && { say "still alive after TERM: TERM again"; kill -TERM "$py" 2>/dev/null; n=0; }
      continue
    fi
    s=$(swmb)
    [ "$s" -gt $(( sw0 + 2048 )) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
    [ $(( $(date +%s) - t0 )) -ge "$CAPS" ] && stopped="the wall cap ($CAPS s)"
    { held >/dev/null || lid_closed; } && stopped="HOLD or the lid closed"
    [ -z "$stopped" ] && back && stopped="yield"
    [ -n "$stopped" ] && { say "stop: $stopped (TERM)"; kill -TERM "$py" 2>/dev/null; }
  done
  wait "$py"; rc=$?; py=""
  unlock
  res=$(grep -h "RESULT:" "$OUTD/${TAG}_rec1a.out" 2>/dev/null | tail -1)
  say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0); ${res:-no RESULT line}"
  [ "$stopped" = yield ] && return 2
  RES="run rc $rc${stopped:+ ($stopped)}; ${res:-no RESULT line}"
  return 0
}
window() {   # $1 label, $2 start, $3 latest start. A run ends the runner (fin); returns 1 = no start by the latest start.
  local lab=$1 st=$2 lt=$3 fc=0 last="" tl=0 a b h
  say "window $lab: from $(date -r "$st" '+%a %H:%M'), latest start $(date -r "$lt" '+%a %H:%M')"
  while [ "$(date +%s)" -lt "$lt" ]; do
    if [ "$(date +%s)" -lt "$st" ]; then
      h=$(held) && fin "SKIP: $h present while waiting" 7
      sleep $POLL; continue
    fi
    if ready && sleep $ZZ && ready; then
      if [ $fc = 0 ] && [ "${RJ_NOFC:-0}" != 1 ]; then
        flagcheck || fin "SKIP: the flag check failed ($(tail -1 "$D/${TAG}_flagcheck.txt" | cut -c1-200))" 7
        fc=1; say "flag check: reached the model load"
      fi
      a=$(ns2)
      if [ "$a" -ge "$GATE" ]; then
        if lock; then
          sleep $SET; b=$(nsr)
          if [ "$b" -lt "$GATE" ]; then
            say "NO-SWAP $b GB $SET s after the lock (< $GATE): unlocked, waiting again"; unlock
          else
            run_job "$lab"
            [ $? = 2 ] || fin "$RES" 0
            say "yielded: waiting again"; last=""
          fi
        else
          WHY="the share lock was refused"
        fi
      else
        WHY="NO-SWAP $a GB < $GATE"
      fi
    fi
    if [ -n "$WHY" ] && { [ "${WHY%% *}" != "${last%% *}" ] || [ $(( $(date +%s) - tl )) -ge 600 ]; }; then
      say "waiting: $WHY"; last=$WHY; tl=$(date +%s)
    fi
    WHY=""
    sleep $POLL
  done
  say "window $lab: not started by its latest start"
  return 1
}
say "armed (pid $$, tag $TAG): window 1 $(date -r "$S1" '+%a %H:%M')-$(date -r "$L1" '+%a %H:%M'), window 2 $(date -r "$S2" '+%a %H:%M')-$(date -r "$L2" '+%a %H:%M') (latest starts); MEM $MEMGB GB, NO-SWAP >= $GATE, wall cap $CAPS s; outputs $OUTD; END $END on the final end only"
window 1 "$S1" "$L1"
window 2 "$S2" "$L2"
fin "SKIP: not started in either window" 7
