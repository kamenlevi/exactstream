#!/bin/bash
# Leap, Sat 2026-11-07 01:3x: S2b step 1 REDONE with the barrier-preserving stub (bar) + its zero-term calibration (cal) + D5 on/off,
# and S2c step 1b (the live check synced per piece vs its block replayed synced right after). One quiet slot (NewIdeas 01:2x),
# C 224 (else the largest C that fits), SB 2 (S2's PASS), the stack env (run_stepb arm bstackF), ioreg sampler on.
# Run with HAVE_LOCK=1 after a separate `peer lock quiet --queue` was granted. Read-out -> ~/Leap/data/stepb/gap_s2b.txt.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_s2b.txt"
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
[ -z "$HAVE_LOCK" ] && { say "gap_s2.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
say "gap_s2b start (pid $$; NO-SWAP $(avail_gb) GB)"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
sleep 15
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: released, nothing runs"; peer unlock >> "$OUT" 2>&1; exit 2; }
say "NO-SWAP $A GB -> C $C"
S="$HOME/Leap/data/stepb/gpu_util_s2b.txt"; rm -f "$S"
./gpu_util_sampler.sh "$S" & SP=$!
HAVE_LOCK=1 STEPB_C=$C STEPB_SB_N=2 STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="8" \
  STEPB_ABL="hcra,cal:hcra,hcrm,cal:hcrm,hcw,cal:hcw,att,cal:att,gdn,cal:gdn,inproj,cal:inproj,moe,cal:moe,d5,dep:hcw" STEPB_ABL_N=12 STEPB_ABL_R=4 \
  STEPB_REAL=2 STEPB_REAL_CAP=160 STEPB_S2C=10 STEPB_S2C_AT=6 STEPB_FULL_N=0 STEPB_DENSE_N=0 \
  STEPB_KREP=0 STEPB_TAGX=s2b2 STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
kill $SP 2>/dev/null
peer unlock >> "$OUT" 2>&1
say "run done, lock released"
J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b2_flat2.json 2>/dev/null | head -1)
[ -n "$J" ] && "$PY" -B gap_read.py "$J" "$S" >> "$OUT" 2>&1
grep -h -E "hook errors|Traceback|probe at check" "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b2.log "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b2_screen.txt 2>/dev/null | tail -5 >> "$OUT"
say "gap_s2b end"
