#!/bin/bash
# CERTIFY98 FULL capture runner (BigMoE, Wed 18 Nov 2026; the orchestrator 11:2x GO; built from mid_run.sh): gates -> share
# lock -> scripts/dd_capture.py with DLOG_SRC=cert98 (EvalSuite's fresh-window jsonl, C98_JSONL, required) -> per window
# ~/datasets/deepdraft/cert98/<key>.npz (h / wd / tk / lp / ids / gen) + index.jsonl, resumable.
# Gates, polled every 20 s until START_BY ("YYYY-mm-dd HH:MM", required): no HOLD; no PAUSE and the lid open; no other
#   session's MEM >= 16 GB or quiet lock; away_secs >= AWAYMIN (900); NO-SWAP >= GATE (43) on two reads 30 s apart.
# Mid-run: Kamen back (away < AWAYMIN on two reads 20 s apart) or STOP_AT writes the stop file (CERT98.STOP): dd_capture stops
#   at the next prompt boundary (resumable). TERM at STOP_AT + 4 min; the +2 GB swap guard TERMs. dd_capture itself stops on a
#   window below 99.5 % on-path (DLOG-ONPATH-FAIL), on 20 GB written, or below 55 GB free.
# EVERY end writes data/deepdraft/cert98/CERT98_CAPTURE_END: "CERT98 CAPTURE END <time>: PASS (n / n windows) | STOPPED
#   (k / n, resumable) | FAIL | NOT RUN | ERROR (...)". Logs cert98_capture.log + cert98_capture_py.log beside it.
# Sims as mid_run.sh: SIMD, FAKE_AWAYF, FAKE_PEERLS, PYCMD, NOLOCK.
Q=$HOME/.local/share/quiet; W=${SIMD:-data/deepdraft/cert98}; mkdir -p "$W" || exit 2
L=$W/cert98_capture.log; PL=$W/cert98_capture_py.log; MK=$W/CERT98_CAPTURE_END; STOPF=$W/CERT98.STOP
GATE=${GATE:-43}; AWAYMIN=${AWAYMIN:-900}; LOCKMEM=${LOCKMEM:-45}; LOCKMIN=${LOCKMIN:-170}; NOLOCK=${NOLOCK:-0}
C98=${C98_JSONL:?C98_JSONL (the fresh-window jsonl from EvalSuite) is required}; [ -s "$C98" ] || { echo "no $C98" >&2; exit 2; }
OUTD=${OUTD:-$HOME/datasets/deepdraft/certify98}; NPLAN=$(grep -c . "$C98")   # EvalSuite amendment 5: <root>/<set>/<name>.npz (placed at a PASS end)
MSHA=${MANIFEST_SHA256:?MANIFEST_SHA256 (the v2 manifest sha256 from EvalSuite) is required}   # files_sha256.json at the PASS end
export DLOG_SRC=cert98 DLOG_C98_JSONL="$C98" DLOG_OUTD="$OUTD" DLOG_TARGET=1000000000 DLOG_CAP_GB=20 DLOG_MIN_FREE_GB=55 DLOG_BUDGET=${BUDGET:-9000} DLOG_HARD=${HARDS:-9300} DLOG_STOPFILE="$STOPF"
PYCMD=${PYCMD:-.venv/bin/python -B scripts/dd_capture.py 224}
export BIGMOE_EXT_BUILD=${BIGMOE_EXT_BUILD:-build_qom}
case "$AWAYMIN$GATE$LOCKMEM$LOCKMIN" in *[!0-9.]*) echo "AWAYMIN / GATE / LOCKMEM / LOCKMIN must be numbers" >&2; exit 2;; esac
SB=$(date -j -f "%Y-%m-%d %H:%M:%S" "${START_BY:?START_BY (YYYY-mm-dd HH:MM) is required}:00" +%s) || exit 2
SA=0; [ -n "${STOP_AT:-}" ] && { SA=$(date -j -f "%Y-%m-%d %H:%M:%S" "$STOP_AT:00" +%s) || exit 2; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
others_locks() {  # other sessions' lock segments (rows start "- "; mine is "You:"), as g1_follow.sh
  { if [ -n "${FAKE_PEERLS:-}" ]; then cat "$FAKE_PEERLS"; else peer ls 2>/dev/null; fi; } | grep "^- " \
    | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"
}
big_lock() { others_locks | grep -oE "MEM ~?[0-9]+" | grep -oE "[0-9]+" | awk '$1 >= 16 {f=1} END {exit !f}'; }
quiet_held() { others_locks | grep -q "^· benchmarking"; }
away_ok() {
  local a; if [ -n "${FAKE_AWAYF:-}" ]; then a=$(cat "$FAKE_AWAYF" 2>/dev/null); else a=$(bash $Q/away_secs.sh 2>/dev/null); fi
  AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge "$AWAYMIN" ]
}
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%.1f", s*ps/1e9}'; }
gate_ok() { local r1 r2; r1=$(noswap); sleep 30; r2=$(noswap); GR="$r1 / $r2"
  awk -v a="$r1" -v b="$r2" -v g="$GATE" 'BEGIN {exit !(a >= g && b >= g)}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }

STATUS="ERROR (the runner ended early)"; LOCKED=0; PY=""; MADE_STOP=0
fin() {
  trap - EXIT INT TERM
  [ -n "$PY" ] && kill -0 "$PY" 2>/dev/null && { kill -TERM "$PY" 2>/dev/null; wait "$PY" 2>/dev/null; }
  [ "$LOCKED" = 1 ] && { peer unlock > /dev/null 2>&1; say "share lock released"; }
  [ "$MADE_STOP" = 1 ] && rm -f "$STOPF"
  case "$STATUS" in PASS*)                               # place the flat files (key / -> __) at <root>/<set>/<name>.npz
    MV=$(python3 - "$C98" "$OUTD" "$MSHA" <<'PY'
import hashlib, json, os, sys
n, F = 0, {}
for l in open(sys.argv[1]):
    if not l.strip(): continue
    k = json.loads(l)["key"]; src = os.path.join(sys.argv[2], k.replace("/", "__") + ".npz"); dst = os.path.join(sys.argv[2], k + ".npz")
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True); os.rename(src, dst); n += 1
    if os.path.exists(dst):
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        F[k + ".npz"] = h.hexdigest()
json.dump({"manifest_sha256": sys.argv[3], "files": F}, open(os.path.join(sys.argv[2], "files_sha256.json"), "w"), indent=1)
print(f"{n} (files_sha256.json: {len(F)} files)")
PY
)
    STATUS="$STATUS; placed $MV files at <root>/<set>/<name>.npz";;
  esac
  echo "CERT98 CAPTURE END $(date '+%m-%d %H:%M:%S'): $STATUS" > $MK
  say "END: $STATUS (marker $MK)"
}
trap fin EXIT
trap 'STATUS="STOPPED (runner TERM / INT)"; exit 0' INT TERM

rm -f "$STOPF"
say "armed: start by $START_BY${STOP_AT:+, stop at $STOP_AT}; gate NO-SWAP >= $GATE, away >= $AWAYMIN s, lock MEM $LOCKMEM GB for $LOCKMIN min; BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD; cmd $PYCMD"
LASTW=""
while :; do
  w=""
  if [ "$(date +%s)" -ge "$SB" ]; then STATUS="NOT RUN (gates not open by $START_BY: ${LASTW:-?})"; exit 0
  elif [ -e $Q/HOLD ]; then w="HOLD"
  elif [ -e $Q/PAUSE ] || lid_closed; then w="PAUSE or the lid closed"
  elif big_lock; then w="another session's MEM >= 16 GB lock"
  elif quiet_held; then w="another session's quiet lock"
> (a quote from the project owner, kept private)
  elif gate_ok && away_ok; then break
  else w="NO-SWAP $GR GB < $GATE"
  fi
  [ "$LASTW" = "$w" ] || { say "waiting: $w"; LASTW="$w"; }
  sleep 20
done
S0=$(swap_mb)
say "start: gates open (no HOLD / PAUSE / big or quiet lock; lid open), NO-SWAP $GR GB >= $GATE; away_secs $AW >= $AWAYMIN; swap $S0 MB"
if [ "$NOLOCK" != 1 ]; then
  LO=$(peer lock share ${LOCKMIN}m --cpu 200 --gpu 100 "MEM $LOCKMEM GB: BigMoE CERTIFY98 FULL capture (dd_capture.py cert98, ~2.1 h)" 2>&1); LR=$?
  say "share lock rc $LR: $(echo "$LO" | head -n 1 | cut -c1-160)"
  [ $LR = 0 ] && LOCKED=1
fi
$PYCMD > $PL 2>&1 &
PY=$!
say "capture started: pid $PY (output $PL)"
TERMED=""; LOWA=0
while kill -0 $PY 2>/dev/null; do
  sleep 10
  now=$(date +%s)
  if away_ok; then LOWA=0; else LOWA=$((LOWA + 1)); if [ $LOWA -ge 2 ] && [ ! -e "$STOPF" ]; then touch "$STOPF"; MADE_STOP=1; say "Kamen back on the Mac (away_secs $AW): stop file written (the capture stops at the next prompt boundary)"; fi; fi
  if [ "$SA" -gt 0 ] && [ "$now" -ge "$SA" ] && [ ! -e "$STOPF" ]; then touch "$STOPF"; MADE_STOP=1; say "STOP_AT $STOP_AT: stop file written (the capture stops at the next file boundary)"; fi
  if [ "$SA" -gt 0 ] && [ "$now" -ge $((SA + 240)) ] && [ -z "$TERMED" ]; then TERMED="STOP_AT + 4 min"; kill -TERM $PY 2>/dev/null; fi
  S1=$(swap_mb); if awk -v a="$S0" -v b="$S1" 'BEGIN {exit !(b - a > 2048)}'; then [ -z "$TERMED" ] && { TERMED="swap guard: swap $S0 -> $S1 MB"; say "$TERMED"; kill -TERM $PY 2>/dev/null; }; fi
done
wait $PY; RC=$?; PY=""
TAIL=$(grep -E 'DLOG-ONPATH-FAIL|stopping|not saved|Traceback|Error' $PL | tail -n 3 | cut -c1-200 | tr '\n' ' ')
NDONE=$(ls "$OUTD" 2>/dev/null | grep -cE '\.npz$')
if grep -q 'DLOG-ONPATH-FAIL' $PL; then STATUS="FAIL (rc $RC; on-path below 99.5 %: $TAIL; $NDONE / $NPLAN windows on disk)"
elif [ "$RC" = 0 ] && [ "$NDONE" -ge "$NPLAN" ]; then STATUS="PASS ($NDONE / $NPLAN windows; dd_capture's on-path self-check held on each)"
elif [ "$RC" = 0 ] || [ -n "$TERMED" ]; then STATUS="STOPPED ($NDONE / $NPLAN windows, resumable; ${TERMED:-$TAIL})"
else STATUS="ERROR (rc $RC; $TAIL; $NDONE / $NPLAN windows)"; fi
exit 0
