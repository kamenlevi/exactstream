#!/bin/bash
# run_rec_quiet_1118.sh (Breakthrough, Wed 18 Nov 2026 02:3x; MODULE 1a milestone 1's OFFICIAL number, pre-registered in
# docs/MODULE1A.md 'Wed 02:2x' (7345f71)): run_rec_quiet.sh (frozen copy, md5 9f0314e9) inside a quiet lock, in
# NewIdeas' slot (02:1x): after Leap's slot #3 ends (C25_s3_DONE newer than this arm; or past its 11:30 hard end with no
# slot #3 process), Kamen away (>= 900 s, two reads 30 s apart), behind any queued quiet ticket; latest start 14:30.
# Waits out HOLD / PAUSE / BREAK98_HOLD / a closed lid / HEALTH ACT until the latest start; NO-SWAP >= 12 GB (two reads).
# Lock: `peer lock quiet --queue` (to the latest start), reason "MEM 4 GB: ..."; settle 20 s; stop checks + away again.
# Run guards: swap +2 GB, HOLD / lid, Kamen back (two reads 30 s apart) = yield, a 10 min cap. TERM only (never -9),
# then unlock right away. A yield before the latest start goes back to the gates (at most 3 tries).
# On ANY end: unlock, then the marker data/m1/rec1a_quiet/REC_QUIET_END = "REC QUIET END <time>: <why>".
# Env (simulation): DRY=1 (fake lock, short sleeps, NO-SWAP gate 1, RQ_JOB in place of the run), RQ_AWAY_CMD,
# RQ_S3MARK, RQ_S3HARD / RQ_LATEST ("YYYY-MM-DD HH:MM:SS"), RQ_OUT (the .out file the result is read from).
cd "$HOME/Breakthrough" || exit 2
Q=$HOME/.local/share/quiet; DRY=${DRY:-0}
D=data/m1/rec1a_quiet; L=$D/run_1118.log; END=$D/REC_QUIET_END
[ "$DRY" = 1 ] && { L=$D/run_1118_dry.log; END=$D/REC_QUIET_END_dry; }
JOBSH=$D/run_rec_quiet.frozen.sh; OUT=${RQ_OUT:-data/m1/rec1a_rset_layer_fast_quiet.out}
S3MARK=${RQ_S3MARK:-$HOME/Leap/data/week98/c25/C25_s3_DONE}
S3PAT='week98/c25/frozen/(s3[a-z0-9]*_arm|run_c25_s3)'
BH=$HOME/NewIdeas/research/week98/BREAK98_HOLD
S3HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "${RQ_S3HARD:-2026-11-18 11:30:00}" +%s) || exit 2
LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${RQ_LATEST:-2026-11-18 14:30:00}" +%s) || exit 2
GATE=12; ZZ=30; SL=60; ST=20; CAP=600
[ "$DRY" = 1 ] && { GATE=1; ZZ=1; SL=1; ST=1; CAP=20; }
A0=$(date +%s); LOCKED=0; TRIES=0
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
unlock() { [ $LOCKED = 1 ] && { [ "$DRY" = 1 ] || peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"; }; }
fin() { unlock; echo "REC QUIET END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; exit "${2:-0}"; }
trap 'unlock' EXIT
trap 'fin "stopped by a signal" 7' TERM INT HUP
AWAYCMD=${RQ_AWAY_CMD:-bash $Q/away_secs.sh}
away() { local a; a=$($AWAYCMD 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 0;; *) echo "$a";; esac; }
away_ok() { [ "$(away)" -ge 900 ]; }
back() { [ "$(away)" -lt 900 ] || return 1; sleep $ZZ; [ "$(away)" -lt 900 ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
blocked() {   # why the run may not start now ('' = nothing blocks)
  [ -e "$Q/HOLD" ] && { echo HOLD; return; }
  [ -e "$Q/PAUSE" ] && { echo PAUSE; return; }
  [ -e "$BH" ] && { echo BREAK98_HOLD; return; }
  lid_closed && { echo "lid closed"; return; }
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { echo "HEALTH ACT"; return; }
  echo ""
}
s3_ended() {
  if [ -e "$S3MARK" ] && [ "$(stat -f %m "$S3MARK")" -ge "$A0" ]; then echo "marker: $(head -c 120 "$S3MARK" | tr '\n' ' ')"; return 0; fi
  if [ "$(date +%s)" -ge "$S3HARD" ] && ! pgrep -f "$S3PAT" >/dev/null; then echo "no marker, past its hard end, no slot #3 process"; return 0; fi
  return 1
}
late() { [ "$(date +%s)" -ge "$LATEST" ]; }
say "armed (pid $$, DRY $DRY): waits for slot #3's end ($S3MARK newer than $(date -r $A0 '+%H:%M:%S'), or past $(date -r $S3HARD '+%H:%M') with no slot #3 process); then Kamen away, no HOLD / PAUSE / BREAK98_HOLD / lid / HEALTH ACT, NO-SWAP >= $GATE; quiet ticket; latest start $(date -r $LATEST '+%a %H:%M')"
while true; do r=$(s3_ended) && break; late && fin "SKIP: slot #3 had not ended by the latest start"; sleep $SL; done
say "slot #3 ended ($r)"
while true; do
  TRIES=$((TRIES + 1))
  # gates, holding nothing
  while true; do
    late && fin "SKIP: not started by the latest start (try $TRIES; away $(away) s; blocked: $(blocked); NO-SWAP $(ns) GB)"
    b=$(blocked)
    if [ -z "$b" ] && away_ok; then
      sleep $ZZ
      a1=$(ns); sleep $ZZ; a2=$(ns); a=$(( a1 < a2 ? a1 : a2 ))
      if [ -z "$(blocked)" ] && away_ok && [ "$a" -ge "$GATE" ]; then
        say "gates open (try $TRIES): away $(away) s (read twice), NO-SWAP $a GB (lower of two reads; need $GATE)"; break
      fi
    fi
    sleep $SL
  done
  # the quiet ticket, behind any queued one
  qm=$(( (LATEST - $(date +%s)) / 60 )); [ $qm -lt 1 ] && qm=1
  if [ "$DRY" = 1 ]; then say "DRY: fake quiet lock"; LOCKED=1; else
    say "queue the quiet ticket (timeout ${qm}m)"
    peer lock quiet --queue ${qm}m "MEM 4 GB: Breakthrough module 1a milestone 1 official bench (one-layer ICB replay, 200 reps, ~2 min; NewIdeas' Wed slot)" >> "$L" 2>&1 \
      || fin "SKIP: the quiet ticket was refused or timed out (try $TRIES)"
    LOCKED=1
  fi
  say "granted; settle ${ST} s"; sleep $ST
  b=$(blocked)
  if [ -n "$b" ] || ! away_ok; then
> (a quote from the project owner, kept private)
    [ $TRIES -ge 3 ] && fin "SKIP: 3 tries used"; continue
  fi
  sw0=$(swmb)
  say "run (try $TRIES; NO-SWAP $(ns) GB, swap $sw0 MB)"
  if [ "$DRY" = 1 ]; then JOB=(${RQ_JOB:-sleep 3}); "${JOB[@]}" >> "$D/run_1118_dry_job.txt" 2>&1 &
  else bash "$JOBSH" > "$D/run_1118_job.txt" 2>&1 & fi
  py=$!; t0=$(date +%s); stopped=""
  while kill -0 $py 2>/dev/null; do
    sleep $(( DRY == 1 ? 1 : 5 ))
    s=$(swmb)
    [ "$s" -gt $((sw0 + 2048)) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
    [ $(( $(date +%s) - t0 )) -ge $CAP ] && stopped="the ${CAP} s cap"
    { [ -e "$Q/HOLD" ] || lid_closed; } && stopped="HOLD or the lid closed"
> (a quote from the project owner, kept private)
    if [ -n "$stopped" ]; then
      say "stop: $stopped"; pkill -TERM -P $py 2>/dev/null; kill -TERM $py 2>/dev/null
      for _ in $(seq 60); do kill -0 $py 2>/dev/null || break; sleep 1; done
      break
    fi
  done
  wait $py; rc=$?
  unlock
  [ "$DRY" = 1 ] || { j=$(grep -m1 -o '^rc [0-9]*' "$D/run_1118_job.txt" 2>/dev/null); rc="$rc (python ${j:-rc ?})"; }
  res=$(grep -h "RESULT:" "$OUT" 2>/dev/null | tail -1 | cut -c1-300)
  rs=$(grep -h "ICB + residency set:" "$OUT" 2>/dev/null | tail -1 | cut -c1-200)
  say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0)"
  say "  ${res:-no RESULT line}"; say "  ${rs:-no residency-set line}"
  case "$stopped" in
    yield*) [ $TRIES -lt 3 ] && ! late && { say "back to the gates after the yield"; continue; }
            fin "yielded on try $TRIES; no retry left (rc $rc)" 3 ;;
    "") fin "rc $rc; ${rs:-no residency-set line}" ;;
    *) fin "stopped ($stopped); rc $rc" 4 ;;
  esac
done
