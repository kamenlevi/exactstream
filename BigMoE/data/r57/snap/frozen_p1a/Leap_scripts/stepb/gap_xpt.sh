#!/bin/bash
# Leap, Fri 2026-11-06 23:5x (the orchestrator 23:5x asks; slot per NewIdeas): one quiet slot, ~6 min, C 224:
#   1. S3 input: XPT (xpt.py) on the real 16-row loop, SB 2 then SB 4: the Python thread's wall and CPU per check in the
#      trunk's layer calls, the expert calls and mx.async_eval (+ the ioreg sampler for the steady-window GPU busy)
#   2. S2 pre-flight: data/s2wire/r57_s2_smoke.py (BigMoE's r57 fa8bee48 + the S2 diff; outputs in ~/Leap) on 2 prompts
#      with SVH + DSUB + R57_SBN=2 + D7BNAX_32=2 on the stack env, R57_SV_CHECK=1 (the SESSION-VOCAB counters)
# WAITFOR=bigmoe: queue now (behind BigMoE's quiet lock, FIFO); WAITFOR=fit12: queue once Fit12's ex16 DONE file appears.
# At the grant: C 224 if NO-SWAP >= 42.3 GB, else the largest C that fits (208 / 192 / 160), else release and stop.
# Read-out -> ~/Leap/data/stepb/gap_xpt.txt. Gives up at Saturday 12:00.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_xpt.txt"
PY=$HOME/BigMoE/.venv/bin/python
WAITFOR=${WAITFOR:-bigmoe}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
D=/Users/kamen/Fit12/data/scr/ex16/DONE
say "gap_xpt start (pid $$, WAITFOR $WAITFOR${HAVE_LOCK:+, HAVE_LOCK: the lock was granted to a separate peer lock command}; NO-SWAP $(avail_gb) GB)"
if [ -z "$HAVE_LOCK" ]; then
  while :; do
    [ "$(date +%Y%m%d%H%M)" -ge 202611071200 ] && { say "Sat 12:00 reached: giving up"; exit 0; }
    case $WAITFOR in
      bigmoe) break;;
      fit12) [ -e "$D" ] && { say "Fit12 DONE: $(cat $D)"; break; };;
    esac
    sleep 30
  done
  PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue 15m "Leap: S3 input (expert path's Python ms per layer, XPT, SB 2 + SB 4) + S2 pre-flight smoke (SVH / DSUB / SBN 2), C 224, ~6 min, MEM ~39 GB" >> "$OUT" 2>&1 \
    || { say "no quiet lock: nothing runs"; exit 1; }
  say "lock granted"
fi
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
sleep 20
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: released, nothing runs"; peer unlock >> "$OUT" 2>&1; exit 2; }
say "NO-SWAP $A GB -> C $C"
S="$HOME/Leap/data/stepb/gpu_util_xpt.txt"; rm -f "$S"
./gpu_util_sampler.sh "$S" & SP=$!
for sb in 2 4; do
  HAVE_LOCK=1 STEPB_C=$C STEPB_SB_N=$sb STEPB_XPT=1 STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="" STEPB_REAL=2 STEPB_REAL_CAP=160 \
    STEPB_FULL_N=0 STEPB_DENSE_N=0 STEPB_KREP=0 STEPB_TAGX=xpt$sb STEPB_ALARM=300 bash run_stepb.sh >> "$OUT" 2>&1
done
kill $SP 2>/dev/null
SB="R57_PRESET=45 R57_REPS=1 R57_REFTOK= BIGMOE_EXT_BUILD=build_qom R57_FAST=QOM R57_D17=1 R57_D7B=code R57_HC8=1 REF2_HC8_FAST=1 R57_DNR=1 R57_HCB=1"
SB="$SB REF2_SMALL_NATIVE=1 R57_D7BNAX=1 REF2_OOO_1C=1 R57_NAX=1 REF2_NAX_1CQ=1 REF2_DN_KS=1 REF2_GDN_LAZY=1 D7BNAX_32=2 R57_FASTTAX=1"
SB="$SB R57_SVH=1 R57_DSUB=1 R57_SV_CHECK=1 R57_SBN=2 R57_PROMPTS=2 R57_TAG=s2smoke VECLIB_MAXIMUM_THREADS=4"
if [ -e "$HOME/.local/share/quiet/PAUSE" ]; then say "PAUSE: smoke skipped"; else
  say "S2 smoke start (C $C)"
  env $SB perl -e 'alarm 300; exec @ARGV' "$PY" -B "$HOME/Leap/data/s2wire/r57_s2_smoke.py" "$C" > "$HOME/Leap/data/s2wire/s2smoke.log" 2>&1
  say "S2 smoke rc=$?"
fi
HAVE_LOCK=1 STEPB_C=$C STEPB_SB_N=2 STEPB_XPT=1 STEPB_XPT_PROF=1 STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="" STEPB_REAL=1 \
  STEPB_REAL_CAP=160 STEPB_FULL_N=0 STEPB_DENSE_N=0 STEPB_KREP=0 STEPB_TAGX=xptp STEPB_ALARM=300 bash run_stepb.sh >> "$OUT" 2>&1
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
for sb in 2 4; do
  J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_xpt${sb}_flat2.json 2>/dev/null | head -1)
  [ -n "$J" ] && "$PY" -B gap_read.py "$J" "$S" >> "$OUT" 2>&1
done
grep -h -E "SESSION-VOCAB|Traceback|Error|error" "$HOME/Leap/data/s2wire/s2smoke.log" | tail -6 >> "$OUT"
tail -4 "$HOME/Leap/data/s2wire/s2smoke.log" >> "$OUT"
say "gap_xpt end"
