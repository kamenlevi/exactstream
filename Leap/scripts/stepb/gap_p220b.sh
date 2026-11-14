#!/bin/bash
# Leap, Sat 2026-11-07 14:4x: PLAN 220 re-price slot PART 2 on r57_screen_fast.py itself (part 1's old screen lacked
# FAST-TAX's att_single and SVH's screened head). One quiet lock, C 224 (else the largest that fits):
#   v1f   v1' dense side (att_single, SVH, SB 2, KEEP, wired): ABL bar / cal per piece incl. ple + head, then the chains
#   v1fg  + R57_GDNB=1 (gdn_body; REF2_GDN_LAZY unset in the probe): base + gdn ABL, then the chains
# Pricing only. Run with HAVE_LOCK=1 after a separate `peer lock quiet --queue`. -> data/stepb/gap_p220b.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_p220b.txt"
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
[ -z "$HAVE_LOCK" ] && { say "gap_p220b.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
say "gap_p220b start (pid $$; NO-SWAP $(avail_gb) GB)"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; exit 0; fi
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: nothing runs"; exit 2; }
say "NO-SWAP $A GB -> C $C"
COMMON="STEPB_C=$C STEPB_PASSES=1 STEPB_AT=8 STEPB_REAL=2 STEPB_REAL_CAP=160 STEPB_S2C=0 STEPB_FULL_N=0 STEPB_DENSE_N=0 STEPB_ALARM=540"
ABLALL="hcra,cal:hcra,hcrm,cal:hcrm,hcw,cal:hcw,att,cal:att,gdn,cal:gdn,inproj,cal:inproj,moe,cal:moe,ple,cal:ple,head,cal:head"
env HAVE_LOCK=1 $COMMON STEPB_ARMS="v1f" STEPB_KREP=0 STEPB_ABL="$ABLALL" STEPB_ABL_N=12 STEPB_ABL_R=4 STEPB_TAGX=p220c bash run_stepb.sh >> "$OUT" 2>&1
env HAVE_LOCK=1 $COMMON STEPB_ARMS="v1fg" STEPB_KREP=0 STEPB_ABL="gdn,cal:gdn" STEPB_ABL_N=12 STEPB_ABL_R=4 STEPB_TAGX=p220c bash run_stepb.sh >> "$OUT" 2>&1
env HAVE_LOCK=1 $COMMON STEPB_ARMS="v1f v1fg" STEPB_KREP=3 STEPB_ABL= STEPB_TAGX=p220d bash run_stepb.sh >> "$OUT" 2>&1
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
for a in v1f v1fg; do
  J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_${a}_*_p220c_flat2.json 2>/dev/null | head -1)
  [ -n "$J" ] && { say "== $a ABL"; "$PY" -B gap_read.py "$J" >> "$OUT" 2>&1; }
  J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_${a}_*_p220d_flat2.json 2>/dev/null | head -1)
  [ -n "$J" ] && { say "== $a chains"; "$PY" -B stepb_read.py "$J" >> "$OUT" 2>&1; }
done
grep -h -E "hook errors|Traceback|Error|FAST-TAX|SVH|GDNB" "$HOME"/Leap/data/stepb/out/stepb_v1f*_p220[cd].log 2>/dev/null | sort | uniq -c | head -12 >> "$OUT"
say "gap_p220b end"
