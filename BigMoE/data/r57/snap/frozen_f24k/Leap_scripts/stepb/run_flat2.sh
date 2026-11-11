#!/bin/bash
# FLAT-2 runner (zero session, Thu 5 Nov 2026; ~/FableGod/zero/LOSSY350_ZERO.md section 4). Owner: Leap or BigMoE, in a
# quiet lock (exact timings). Two arms, one process each, ~2-3 min per arm, both at the memory ladder's C:
#   fri  Friday's all-in FAST arm as fri_night.sh / preflight_fri.sh run it today (pqall / ps99 PASS at this HEAD) + the
#        split form R57_QOM_SPLIT=0.9: every lever BigMoE has pre-flighted in this combination. The verdict's arm.
#   all  fri + the exact levers that are wired but off by default and not yet run in this combination: Leap's DKV
#        drafter cache (tokens PASS Thu 08:2x, with the additive mask) and GDNB (bitwise, -0.71 ms per check). If it
#        crashes, fri's number stands; if it runs, its FLAT is the lower bound of today's levers.
# C: the largest of 224 / 208 / 192 / 160 whose lifetime peak + 4 GB fits NO-SWAP (fri_night.sh's ladder; 160 = the
# pre-flight's uniform pools, MEM 36 GB). FLAT does not depend on C (the dense pass, the draft and the commit are the same;
# the pools only change which optional tails QOM keeps, and that term is subtracted).
# Guards as BigMoE's runners: PAUSE / HOLD honoured, swap +2 GB during an arm stops it, alarm 420 s per arm.
# usage: bash run_flat2.sh                        (takes the quiet lock itself, 20 min queue, 4 h wait at most)
#        HAVE_LOCK=1 bash run_flat2.sh            (inside the owner's own lock)
#        FLAT2_DRY=1 bash run_flat2.sh            (prints the ladder, the env and the commands; starts nothing)
#        FLAT2_SMOKE=1 bash run_flat2.sh          (one short pass of each kind per arm, ~40 s: does the lever set run?)
#        FLAT2_ARMS="fri" / FLAT2_C=160 / FLAT2_ALARM=600 / FLAT2_AVAIL=<GB, the ladder's input> as needed
# -> out/flat2_<arm>_<C>_screen.txt (the report; last lines = FLAT-2 VERDICT), _flat2.json (every number), _generated.py
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")" || exit 1
PY=${FLAT2_PY:-$HOME/BigMoE/.venv/bin/python}
OUT="$PWD/out"; mkdir -p "$OUT"; LOG="$OUT/run_flat2.log"
DRY=${FLAT2_DRY:-}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
A=${FLAT2_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=${FLAT2_C:-}
if [ -z "$CC" ]; then for c in 224:44.3 208:42.2 192:40.0 160:36.0; do fits ${c#*:} && { CC=${c%%:*}; break; }; done; fi
[ -z "$CC" ] && { say "NO-SWAP $A GB < 36.0: nothing fits, nothing runs"; exit 0; }
BASE="R57_PRESET=45 R57_REPS=1 R57_REFTOK= BIGMOE_EXT_BUILD=build_qom R57_FAST=QOM R57_D17=1 R57_D7B=code R57_HC8=1"
BASE="$BASE REF2_HC8_FAST=1 R57_DNR=1 R57_HCB=1 REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1 R57_QOM_SPLIT=0.9"
[ "$CC" = 160 ] && BASE="$BASE R57_LAYSZ=0 BIGMOE_TIER_CL=64"      # the pre-flight's uniform pools at C 160
EXTRA="R57_DKV=1 R57_DKV_ADD=1 R57_GDNB=1"
ARMS=${FLAT2_ARMS:-"fri all"}
say "FLAT-2 start (pid $$, NO-SWAP $A GB -> C $CC, arms: $ARMS${FLAT2_SMOKE:+, SMOKE}${DRY:+, DRY})"
if [ -z "$DRY" ] && [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 20m "FLAT-2 (zero): 2-row flat cycle of the 45 GB FAST all-in, 2 arms x ~2-3 min, MEM ~$([ "$CC" = 160 ] && echo 32 || echo 38) GB" >> "$LOG" 2>&1 \
    || { say "no quiet lock: nothing runs"; exit 1; }
  LOCKED=1
fi
rc_all=0
for arm in $ARMS; do
  case $arm in fri) ENV="$BASE";; all) ENV="$BASE $EXTRA";; *) say "unknown arm $arm"; continue;; esac
  tag="flat2_${arm}_${CC}"
  if [ -n "$DRY" ]; then
    say "WOULD RUN [$arm]: env $ENV FLAT2_TAG=$tag $PY flat2.py $CC  (alarm ${FLAT2_ALARM:-420} s, swap guard +2 GB; would take the quiet lock first)"
    continue
  fi
  if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: $arm skipped"; rc_all=1; continue; fi
  SW0=$(swap_mb)
  say "run $tag (swap $SW0 MB)"
  env $ENV FLAT2_TAG=$tag perl -e 'alarm shift; exec @ARGV' ${FLAT2_ALARM:-420} "$PY" flat2.py "$CC" > "$OUT/$tag.log" 2>&1 &
  jp=$!
  while kill -0 $jp 2>/dev/null; do
    sleep 5
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then say "SWAP GUARD: stopping $tag"; T=$(tree $jp); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $jp; rc=$?
  say "done $tag rc=$rc: $(grep -h 'FLAT-2 VERDICT' "$OUT/${tag}_screen.txt" 2>/dev/null | tail -1 | cut -c1-200)"
  [ $rc != 0 ] && rc_all=1
done
[ -n "$LOCKED" ] && [ -z "$KEEP_LOCK" ] && peer unlock >> "$LOG" 2>&1
[ -z "$DRY" ] && { say "FLAT-2 end"; grep -h "FLAT-2 VERDICT" "$OUT"/flat2_*_"$CC"_screen.txt 2>/dev/null; }
exit $rc_all
