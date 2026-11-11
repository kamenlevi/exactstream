#!/bin/bash
# Leap, Sat 2026-11-07 14:2x: PLAN 220's re-price slot (NewIdeas QUEUE 8dae435 / 32f2865), one quiet lock, three stepb arms
# + two fcp arms, C 224 (else the largest that fits) for stepb, C 256 for fcp:
#   v1w   v1' + wired, dense side (run_stepb arm v1w, SB 2): ABL bar / cal per piece incl. ple and head (final norm + lm_head)
#   v1wk  the same env: kernel chains at 2 and 16 rows (STEPB_KREP): HC8 k1c / k3c / k2b, gdn_body, in_proj, out_proj ...
#   v1w2k + REF2_DN_KS=2 (in_proj qkv|z at ks 2, 1b): the same chains
#   fcpc / fcpd (scripts/fcp): the partial-commit split on v1' + wired, 6 HumanEval, CAP 128; fcpd adds an eval right after
#   each commit (its GPU time)
# Pricing only (no verdict lines). Run with HAVE_LOCK=1 after a separate `peer lock quiet --queue`. -> data/stepb/gap_p220.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_p220.txt"
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
[ -z "$HAVE_LOCK" ] && { say "gap_p220.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
say "gap_p220 start (pid $$; NO-SWAP $(avail_gb) GB)"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; exit 0; fi
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: nothing runs"; exit 2; }
say "NO-SWAP $A GB -> C $C"
COMMON="STEPB_C=$C STEPB_SB_N=2 STEPB_PASSES=1 STEPB_AT=8 STEPB_REAL=2 STEPB_REAL_CAP=160 STEPB_S2C=0 STEPB_FULL_N=0 STEPB_DENSE_N=0 STEPB_ALARM=540"
env HAVE_LOCK=1 $COMMON STEPB_ARMS="v1w" STEPB_KREP=0 \
  STEPB_ABL="hcra,cal:hcra,hcrm,cal:hcrm,hcw,cal:hcw,att,cal:att,gdn,cal:gdn,inproj,cal:inproj,moe,cal:moe,ple,cal:ple,head,cal:head" \
  STEPB_ABL_N=12 STEPB_ABL_R=4 STEPB_TAGX=p220a bash run_stepb.sh >> "$OUT" 2>&1
env HAVE_LOCK=1 $COMMON STEPB_ARMS="v1w v1w2" STEPB_KREP=3 STEPB_ABL= STEPB_TAGX=p220k bash run_stepb.sh >> "$OUT" 2>&1
HAVE_LOCK=1 bash "$HOME/Leap/scripts/fcp/fcp_run.sh" "fcpc fcpd" >> "$OUT" 2>&1
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_v1w_*_p220a_flat2.json 2>/dev/null | head -1)
[ -n "$J" ] && "$PY" -B gap_read.py "$J" >> "$OUT" 2>&1
for a in v1w v1w2; do
  J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_${a}_*_p220k_flat2.json 2>/dev/null | head -1)
  [ -n "$J" ] && { say "== $a chains"; "$PY" -B stepb_read.py "$J" >> "$OUT" 2>&1; }
done
FCP_PAIRS="" "$PY" -B "$HOME/Leap/scripts/fcp/fcp_read.py" fcpc fcpd > "$HOME/Leap/data/fcp/fcp_read_p220.txt" 2>&1
grep -h -E "hook errors|Traceback|Error" "$HOME"/Leap/data/stepb/out/stepb_v1w*_p220*.log 2>/dev/null | tail -5 >> "$OUT"
say "gap_p220 end"
