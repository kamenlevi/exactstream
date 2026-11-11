#!/bin/bash
# Leap, Sat 2026-11-07 01:3x: S2c step 1c: core placement. Arms: base / STEPB_QOS=1 (decode thread at QOS user-interactive) /
# process each, at C 224 (else the largest C that fits), SB 2, the stack env: base / BIGMOE_SSD_KEEPALIVE_US=0 / R57_VTIER=.
# BIGMOE_IDLE_MS=0.5 (the loader naps when idle) / both. Each: the S2c probe at real check 6 with thread CPU per piece.
# Lines: "extra wait" = moe_pre live - replay wall >= 0.08 ms per layer with |CPU diff| <= 0.03; an arm that removes it
# names the mechanism. Run with HAVE_LOCK=1 after a separate `peer lock quiet --queue`. -> ~/Leap/data/stepb/gap_s2d.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap/scripts/stepb" || exit 1
OUT="$HOME/Leap/data/stepb/gap_s2d.txt"
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
[ -z "$HAVE_LOCK" ] && { say "gap_s2d.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
say "gap_s2d start (pid $$; NO-SWAP $(avail_gb) GB)"
if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: nothing runs"; peer unlock >> "$OUT" 2>&1; exit 0; fi
sleep 10
A=$(avail_gb); C=""
for c in 224:42.3 208:40.2 192:38.1 160:33.9; do ge "$A" "${c#*:}" && { C=${c%%:*}; break; }; done
[ -z "$C" ] && { say "NO-SWAP $A GB < 33.9: released, nothing runs"; peer unlock >> "$OUT" 2>&1; exit 2; }
say "NO-SWAP $A GB -> C $C"
for arm in base qos idle qi; do
  case $arm in base) X="S2C_ARM=base";; qos) X="STEPB_QOS=1";; idle) X="BIGMOE_IDLE_MS=0.5";; qi) X="STEPB_QOS=1 BIGMOE_IDLE_MS=0.5";; esac
  say "arm $arm ($X)"
  env $X HAVE_LOCK=1 STEPB_C=$C STEPB_SB_N=2 STEPB_ARMS="bstackF" STEPB_PASSES=0 STEPB_AT="" STEPB_REAL=1 STEPB_REAL_CAP=160 \
    STEPB_S2C=6 STEPB_S2C_AT=6 STEPB_FULL_N=0 STEPB_DENSE_N=0 STEPB_KREP=0 STEPB_TAGX=s2d_$arm STEPB_ALARM=240 \
    bash run_stepb.sh >> "$OUT" 2>&1
  say "  after $arm: PAUSE $( [ -e "$HOME/.local/share/quiet/PAUSE" ] && echo yes || echo no ); fanwatch: $(tail -1 "$HOME/Library/Logs/fanwatch.log" 2>/dev/null | cut -c1-90)"
done
peer unlock >> "$OUT" 2>&1
say "runs done, lock released"
for arm in base qos idle qi; do
  J=$(ls -t "$HOME"/Leap/data/stepb/out/stepb_bstackF_*_s2d_${arm}_flat2.json 2>/dev/null | head -1)
  [ -n "$J" ] && { echo "=== arm $arm" >> "$OUT"; "$PY" -B gap_read.py "$J" >> "$OUT" 2>&1; }
done
say "gap_s2d end"
