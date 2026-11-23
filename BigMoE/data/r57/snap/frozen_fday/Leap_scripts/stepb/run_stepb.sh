#!/bin/bash
# STEP 0 PART B runner (Leap, Fri 2026-11-06 07:5x; from zero's run_flat2.sh). Inside Leap's own quiet lock (HAVE_LOCK=1:
# the lock is taken in a separate command). Arms, one process each, at the memory ladder's C (NO-SWAP + 3 GB margin):
#   fri  today's FAST all-in: FLAT-2's fri arm without R57_QOM_SPLIT (NewIdeas took the split out, Fri 04:1x)
#   nax  fri + FAST-NAX (R57_NAX=1 REF2_NAX_1CQ=1): the same dense pass, NAX6-N's one-call for the experts
#   friks / naxks  + DN-KS (REF2_DN_KS=1, ref2 1851aa5c); fri2 / nax2 = the A-B-A repeats (DN-KS step 2)
# Guards: PAUSE / HOLD honoured, swap +2 GB during an arm stops it, alarm STEPB_ALARM s per arm (default 600).
# usage: HAVE_LOCK=1 bash run_stepb.sh   STEPB_ARMS="fri" / STEPB_SMOKE=1 / STEPB_C=192 / STEPB_DRY=1
# -> ~/Leap/data/stepb/out/stepb_<arm>_<C>_screen.txt, _flat2.json (krec records), _generated.py; log run_stepb.log
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
OUT="$HOME/Leap/data/stepb/out"; mkdir -p "$OUT"; LOG="$OUT/run_stepb.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
[ -z "$HAVE_LOCK" ] && [ -z "$STEPB_DRY" ] && { say "run_stepb.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
A=$(avail_gb)
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n + 3.0 <= a)}'; }
CC=${STEPB_C:-}
if [ -z "$CC" ]; then for c in 224:44.3 208:42.2 192:40.0 160:36.0; do fits ${c#*:} && { CC=${c%%:*}; break; }; done; fi
[ -z "$CC" ] && { say "NO-SWAP $A GB < 39.0 (160's 36.0 + 3): nothing runs"; exit 0; }
BASE="R57_PRESET=45 R57_REPS=1 R57_REFTOK= BIGMOE_EXT_BUILD=build_qom R57_FAST=QOM R57_D17=1 R57_D7B=code R57_HC8=1"
BASE="$BASE REF2_HC8_FAST=1 R57_DNR=1 R57_HCB=1 REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1"
[ "$CC" = 160 ] && BASE="$BASE R57_LAYSZ=0 BIGMOE_TIER_CL=64"
RUNS="FLAT2_OUT=$OUT FLAT2_PASSES=${STEPB_PASSES:-2} FLAT2_AT=${STEPB_AT-8,24} FLAT2_REAL=${STEPB_REAL:-0} FLAT2_REAL_CAP=${STEPB_REAL_CAP:-160} STEPB_KREP=${STEPB_KREP:-3} VECLIB_MAXIMUM_THREADS=4"
[ -n "$STEPB_SMOKE" ] && RUNS="$RUNS FLAT2_SMOKE=1 FLAT2_AT=4 STEPB_KREP=1"
ARMS=${STEPB_ARMS:-"fri"}
say "STEP B start (pid $$, NO-SWAP $A GB -> C $CC, arms: $ARMS${STEPB_SMOKE:+, SMOKE}${STEPB_DRY:+, DRY}; md5 probe $(md5 -q flat2_probe.py | cut -c1-8) k $(md5 -q stepb_k.py | cut -c1-8))"
rc_all=0
for arm in $ARMS; do
  case $arm in fri|fri2) ENV="$BASE";; nax|nax2) ENV="$BASE R57_NAX=1 REF2_NAX_1CQ=1";; friks) ENV="$BASE REF2_DN_KS=1";;
    naxks) ENV="$BASE R57_NAX=1 REF2_NAX_1CQ=1 REF2_DN_KS=1";; friks2) ENV="$BASE REF2_DN_KS=1";;
    friksc) ENV="$BASE REF2_DN_KS=1 STEPB_D7BNAX=$HOME/Leap/data/d17nax/d7b_nax_cand.py";;
    b|b2) ENV="$BASE REF2_DN_KS=1 D7BNAX_32=1";; batt|batt2) ENV="$BASE REF2_DN_KS=1 D7BNAX_32=1 STEPB_ATT_SINGLE=1";;
    bv2) ENV="$BASE REF2_DN_KS=2 D7BNAX_32=2 STEPB_ATT_SINGLE=1";;
    tog) ENV="$BASE REF2_DN_KS=1 D7BNAX_32=2 STEPB_ATT_SINGLE=1";;   # + STEPB_TOGGLE from the caller (in-process A B A B)
    nax0) ENV="$BASE R57_NAX=1 REF2_NAX_1CQ=1";;                     # FAST-NAX as of this morning
    stack) ENV="$BASE R57_NAX=1 REF2_NAX_1CQ=1 REF2_DN_KS=1 D7BNAX_32=2 STEPB_ATT_SINGLE=1 REF2_GDN_LAZY=1";;   # + today's levers
    bfs4) ENV="$BASE REF2_DN_KS=1 D7BNAX_32=1 REF2_FATT_VERIFY=1 REF2_FATT_R8=1";;
    friks32) ENV="$BASE REF2_DN_KS=1 D7BNAX_32=1 STEPB_D7BNAX=$HOME/Leap/data/d17nax/d7b_nax_cand.py";; *) say "unknown arm $arm"; continue;; esac
  tag="stepb_${arm}_${CC}${STEPB_SMOKE:+_smoke}${STEPB_TAGX:+_$STEPB_TAGX}"
  if [ -n "$STEPB_DRY" ]; then say "WOULD RUN [$arm]: env $ENV $RUNS FLAT2_TAG=$tag $PY -B stepb.py $CC"; continue; fi
  if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: $arm skipped"; rc_all=1; continue; fi
  SW0=$(swap_mb)
  say "run $tag (swap $SW0 MB)"
  env $ENV $RUNS FLAT2_TAG=$tag perl -e 'alarm shift; exec @ARGV' ${STEPB_ALARM:-600} "$PY" -B stepb.py "$CC" > "$OUT/$tag.log" 2>&1 &
  jp=$!
  while kill -0 $jp 2>/dev/null; do
    sleep 5
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then say "SWAP GUARD: stopping $tag"; T=$(tree $jp); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $jp; rc=$?
  say "done $tag rc=$rc: $(grep -h -E 'FLAT-2 VERDICT|step B calibration' "$OUT/${tag}_screen.txt" 2>/dev/null | cut -c1-160 | tr '\n' ' ')"
  [ $rc != 0 ] && rc_all=1
done
say "STEP B end (rc $rc_all)"
exit $rc_all
