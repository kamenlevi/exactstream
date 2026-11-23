#!/bin/bash
# Leap, Sat 2026-11-07 01:0x: S2b step 1 (STEPB_ABL: in-situ unsynced price per dense piece) + S2c step 1 (STEPB_S2C: a
# real check's own block alone vs alternating with the previous checks' blocks), one quiet slot (NewIdeas' QUEUE f270dee),
# C 224 (else the largest C that fits), SB 2 (S2's PASS), the stack env (run_stepb arm bstackF), ioreg sampler on.
# Run with HAVE_LOCK=1 after a separate `peer lock quiet --queue` was granted. Read-out -> ~/Leap/data/stepb/gap_s2.txt.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_s2.txt"
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
[ -z "$HAVE_LOCK" ] && { say "gap_s2.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
say "gap_s2 start (pid $$; NO-SWAP $(avail_gb) GB)"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
sleep 15
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: released, nothing runs"; peer unlock >> "$OUT" 2>&1; exit 2; }
say "NO-SWAP $A GB -> C $C"
S="$HOME/Leap/data/stepb/gpu_util_s2.txt"; rm -f "$S"
./gpu_util_sampler.sh "$S" & SP=$!
HAVE_LOCK=1 STEPB_C=$C STEPB_SB_N=2 STEPB_ARMS="bstackF" STEPB_PASSES=1 STEPB_AT="8" \
  STEPB_ABL="hcr,hcra,hcrm,hcw,inproj,att,gdn,moe,ple" STEPB_ABL_N=12 STEPB_ABL_R=4 \
  STEPB_REAL=2 STEPB_REAL_CAP=160 STEPB_S2C=10 STEPB_S2C_AT=6 STEPB_FULL_N=30 STEPB_DENSE_N=30 \
  STEPB_KREP=0 STEPB_TAGX=s2b STEPB_ALARM=480 bash run_stepb.sh >> "$OUT" 2>&1
kill $SP 2>/dev/null
peer unlock >> "$OUT" 2>&1
say "run done, lock released"
J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b_flat2.json 2>/dev/null | head -1)
[ -n "$J" ] && "$PY" -B gap_read.py "$J" "$S" >> "$OUT" 2>&1
grep -h -E "hook errors|Traceback|probe at check" "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b.log "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2b_screen.txt 2>/dev/null | tail -5 >> "$OUT"
say "gap_s2 end"
