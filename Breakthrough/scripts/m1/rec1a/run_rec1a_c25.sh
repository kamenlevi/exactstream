#!/bin/bash
# run_rec1a_c25.sh (Breakthrough, Tue 17 Nov 2026 07:2x; MODULE 1a milestone 2): one engine run of rec1a_c25.py,
# pre-registered in docs/MODULE1A.md 'Tue 06:5x' (+ its 07:0x amendments), in a slot NewIdeas gives.
# Waits, holding nothing, until START once no c25 / rec1a loop runs and Kamen is away (away_secs >= 900, read twice
# 30 s apart); never starts after LATEST (= HARD - EST minutes). Then: HOLD / PAUSE / lid / HEALTH ACT checks, the NO-SWAP
# gate on two reads 30 s apart (>= 43 = 35 + 8), a share lock whose reason starts 'MEM 35 GB', a swap guard (+2 GB
# stops it), a yield when Kamen is back (two reads 30 s apart), the run killed at HARD (TERM, KILL 20 s later).
# On ANY end: the lock released, then the marker data/m1/rec1a_engine/REC1A_END = "REC1A END <time>: <why>".
# The run: c25_env.py's exports with C25_ARMS=F32s,F32o,F32c6i, C25_KEYS=dd__mc_27921, C25_WARM=1, C25_ROUNDS=1,
# C25_TAG=<tag>; then R57_VTIER empty; C25_PROF / C25_CPUSPLIT / C25_PASSK unset; rec1a_c25.py 256 2048 -> <tag>_run.txt.
# Usage: run_rec1a_c25.sh "<START YYYY-MM-DD HH:MM:SS>" "<HARD YYYY-MM-DD HH:MM:SS>"
# Env: RJ_EST_MIN (20), RJ_GATE (43), RJ_MEM (35), RJ_TAG (rec1a_<MMDD>); simulation: DRY=1 (START now, fake lock,
# 1 s waits, gate 1, RJ_JOB (default: sleep 3) in place of the run), RJ_AWAY_CMD.
cd "$HOME/Breakthrough" || exit 2
Q=$HOME/.local/share/quiet; PY=$HOME/BigMoE/.venv/bin/python
DRY=${DRY:-0}; GATE=${RJ_GATE:-43}; EST=${RJ_EST_MIN:-20}; MEMGB=${RJ_MEM:-35}; ZZ=30; TAG=${RJ_TAG:-rec1a_$(date +%m%d)}
[ $DRY = 1 ] && { GATE=1; ZZ=1; TAG=${TAG}_dry; }
D=data/m1/rec1a_engine; mkdir -p "$D"
L=$D/${TAG}_runner.log; END=$D/REC1A_END; PAT='c25_loo[p][.]py|rec1a_c2[5][.]py'
job_env() {   # the run's environment (also used by the flag check)
  export C25_ARMS=F32s,F32o,F32c6i C25_KEYS=dd__mc_27921 C25_WARM=1 C25_ROUNDS=1 C25_TAG=$TAG
  eval "$(python3 -I -B "$HOME/Leap/scripts/week98/c25/c25_env.py")" || exit 5
  export R57_VTIER= REC1A_OUT="$HOME/Breakthrough/$D" REC1A_TAG=$TAG
  unset C25_PROF C25_CPUSPLIT C25_PASSK
}
if [ "${RJ_FLAGCHECK:-0}" = 1 ]; then   # the real loop up to its model load, in this env; no gates, no lock, no GPU work
  ( job_env; export REC1A_FLAGCHECK=1; exec "$PY" -I -B scripts/m1/rec1a/rec1a_c25.py 256 2048 ) > "$D/${TAG}_flagcheck.txt" 2>&1
  rc=$?; m=$(grep -h "REC1A FLAGCHECK" "$D/${TAG}_flagcheck.txt")
  echo "flag check rc $rc (97 = reached the model load): ${m:-NO MARKER}"; [ $rc = 97 ] && [ -n "$m" ]; exit $?
fi
# (the same check also runs inside a slot, right after the trigger, unless RJ_NOFC=1: Leap's files may change by then)
START=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?START}" +%s) || exit 2
HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?HARD END}" +%s) || exit 2
[ $DRY = 1 ] && START=$(date +%s)
LATEST=$(( HARD - EST * 60 ))
LOCKED=0
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
fin() { echo "REC1A END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
unlock() { [ $LOCKED = 1 ] && { [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; LOCKED=0; }; }
quit() { unlock; say "$1"; fin "$1"; exit 7; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
stop_checks() {
  [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD present"
  lid_closed && quit "SKIP ($1): lid closed"
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && quit "SKIP ($1): HEALTH ACT"
  if [ -e "$Q/PAUSE" ]; then
    say "PAUSE present ($1): waiting it out until $(date -r "$LATEST" '+%a %H:%M') at the latest"
    while [ -e "$Q/PAUSE" ] && [ "$(date +%s)" -lt "$LATEST" ]; do
      sleep 10
      [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD (during a PAUSE)"
      lid_closed && quit "SKIP ($1): lid closed (during a PAUSE)"
    done
    [ -e "$Q/PAUSE" ] && quit "SKIP ($1): PAUSE still present at the latest start"
    say "PAUSE cleared ($1)"
  fi
}
free_() {   # no PYTHON process runs Leap's c25 loop or my wrapper (a monitor shell whose text names them does not count)
  local p a
  for p in $(pgrep -f "$PAT"); do
    a=$(ps -o args= -p "$p" 2>/dev/null); a=${a%% *}
    case "$a" in *[Pp]ython*) return 1;; esac
  done
  return 0
}
flagcheck() {   # the real loop in the run's env up to its model load (no model, no GPU work); 0 = reached it
  ( job_env; export REC1A_FLAGCHECK=1; exec "$PY" -I -B scripts/m1/rec1a/rec1a_c25.py 256 2048 ) > "$D/${TAG}_flagcheck.txt" 2>&1
  local rc=$?
  [ $rc = 97 ] && grep -q "REC1A FLAGCHECK" "$D/${TAG}_flagcheck.txt"
}
AWAYCMD=${RJ_AWAY_CMD:-bash $Q/away_secs.sh}
away() { local a; a=$($AWAYCMD 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 900;; *) echo "$a";; esac; }
away_ok() { [ "$(away)" -ge 900 ]; }
back() { [ "$(away)" -lt 900 ] || return 1; sleep $ZZ; [ "$(away)" -lt 900 ]; }
trap 'unlock' EXIT
> (a quote from the project owner, kept private)
while true; do
  if [ "$(date +%s)" -ge "$START" ] && free_ && away_ok; then
    sleep $ZZ
> (a quote from the project owner, kept private)
  fi
> (a quote from the project owner, kept private)
  sleep $(( DRY == 1 ? 1 : 60 ))
done
stop_checks "after the trigger"
if [ "${RJ_NOFC:-0}" != 1 ]; then
  flagcheck || quit "SKIP: the flag check failed ($(tail -1 "$D/${TAG}_flagcheck.txt" | cut -c1-200))"
  say "flag check: reached the model load ($(grep -h -o "BIGMOE_POST_SYS='[^']*' BIGMOE_SPIN_COHERENT='[^']*'" "$D/${TAG}_flagcheck.txt"))"
fi
a=$(ns); sleep $ZZ; b=$(ns); a=$(( a < b ? a : b ))
say "NO-SWAP $a GB (lower of two reads; need $GATE)"
while [ "$a" -lt "$GATE" ]; do
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: NO-SWAP $a < $GATE GB at the latest start"
  stop_checks "NO-SWAP wait"
  sleep $(( DRY == 1 ? 1 : 20 )); a=$(ns); sleep $ZZ; b=$(ns); a=$(( a < b ? a : b ))
done
if [ $DRY = 1 ]; then say "DRY: fake lock"; LOCKED=1; else
  lm=$(( ( HARD - $(date +%s) ) / 60 + 5 ))
  peer lock share ${lm}m --cpu 300 --gpu 60 "MEM $MEMGB GB: Breakthrough module 1a engine run (record + replay vs c25's own check, 2 arms), to $(date -r "$HARD" '+%H:%M') at the latest" >> "$L" 2>&1 \
    || quit "SKIP: the share lock was refused"
  LOCKED=1
fi
sw0=$(swmb)
say "running (NO-SWAP $(ns) GB, swap $sw0 MB)"
if [ $DRY = 1 ]; then
  JOB=(${RJ_JOB:-sleep 3})
  "${JOB[@]}" >> "$D/${TAG}_run.txt" 2>&1 &
else
  (
    job_env
    env | sort | grep -E '^(C25_|R57_|BIGMOE_|REC1A_|REF2_|V32_)' > "$D/${TAG}_env.txt"
    exec "$PY" -I -B scripts/m1/rec1a/rec1a_c25.py 256 2048
  ) > "$D/${TAG}_run.txt" 2>&1 &
fi
py=$!
stopped=""
while kill -0 $py 2>/dev/null; do
  sleep $(( DRY == 1 ? 1 : 10 ))
  s=$(swmb)
  [ "$s" -gt $((sw0 + 2048)) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
  [ "$(date +%s)" -ge "$HARD" ] && stopped="hard stop $(date -r "$HARD" '+%H:%M')"
  { [ -e "$Q/HOLD" ] || lid_closed; } && stopped="HOLD or the lid closed"
> (a quote from the project owner, kept private)
  if [ -n "$stopped" ]; then say "stop: $stopped"; kill -TERM $py; sleep 20; kill -9 $py 2>/dev/null; break; fi
done
wait $py; rc=$?
unlock
res=$(grep -h "RESULT:" "$D/${TAG}_rec1a.out" 2>/dev/null | tail -1)
say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0); ${res:-no RESULT line}"
fin "run rc $rc${stopped:+ ($stopped)}; ${res:-no RESULT line}"
