#!/bin/bash
# BigMoE Thu 23:4x: ref2.py stays the pre-19:1x version (md5 04f18c02) through their Fri run, which was pre-flighted on
# it. Once '11-06 ... Fri run end' is in ~/BigMoE/data/fri/fri.log, this puts the FAST-NAX version back atomically
# (md5 eda11b8b; REF2_NAX_1CQ off by default; self_test PASS Thu 23:2x).
# If no end line arrives by Fri 12:00 it gives up, and the re-apply is done by hand. Log: data/fastnax_gpu.log.
LOG="$HOME/Leap/data/fastnax_gpu.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
SRC="$HOME/Leap/data/ref2_fastnax_eda11b8b.py"; R2="$HOME/Leap/scripts/ref2.py"
NEW=eda11b8be7a6e97331df9b61af86a173; OLD=04f18c02b4503618365f44b84dc52e6b
say "ref2 re-apply armed (pid $$): the FAST-NAX version after BigMoE's 'Fri run end'"
until grep -qE '^11-06 [0-9:]{8} Fri run end' "$HOME/BigMoE/data/fri/fri.log" 2>/dev/null; do
  [ "$(date +%Y%m%d%H%M)" -ge 202611061200 ] && { say "no 'Fri run end' by Fri 12:00: ref2 NOT re-applied (stays pre-19:1x)"; exit 3; }
  sleep 20
done
cur=$(md5 -q "$R2")
[ "$cur" != "$OLD" ] && { say "ref2.py md5 $cur is not the pre-19:1x one (changed meanwhile): NOT re-applied"; exit 4; }
if [ "$(md5 -q "$SRC")" = "$NEW" ] && cp "$SRC" "$R2.reapply_tmp" && mv "$R2.reapply_tmp" "$R2" && [ "$(md5 -q "$R2")" = "$NEW" ]; then
  say "RE-APPLIED the FAST-NAX ref2 (md5 eda11b8b, switch off by default) after BigMoE's Fri run end"
else
  say "RE-APPLY FAILED: ref2.py md5 $(md5 -q "$R2")"; exit 5
fi
