#!/bin/bash
# The cf 0.3 chat arm, 4th tonight (the orchestrator 22:0x: chat arm, s7_trl, the Rule C maps, then qom_0.3_live_c147_bonus3g
# only if it can start by 22:50): waits until both the s67 map runner and the Rule C runner have ended, then (before 22:50)
# sets ALONE to now in data/s67_extra.txt (the extra arms no longer wait for s7_trl), adds the arm's line (est 5, start by
# 22:50, twins: the chat arm (primary), qom_0_live_bq_c147_t3), and runs a second s67 runner instance (same guards;
# LASTGO 22:53, hard stop 23:00). A file of its own, so its command line holds no runner name for the pgrep gates.
HERE=$(cd "$(dirname "$0")" && pwd); X=~/Breakthrough/data/s67_extra.txt
while pgrep -f "rulec_slo[t].sh" > /dev/null || pgrep -f "s67_slo[t].sh" > /dev/null; do sleep 20; done
if [ $(date +%s) -ge $(date -j -f "%Y-%m-%d %H:%M" "2026-11-04 22:50" +%s) ]; then
  echo "$(date +%H:%M:%S): too late for the cf 0.3 arm (start by 22:50); Thursday's first map slot"; exit 0
fi
t=$(date +%H:%M); sed -i '' "s/^ALONE 0 .*/ALONE 0 $t/" $X
grep -q "^qom_0.3_live_c147_bonus3g " $X || printf 'qom_0.3_live_c147_bonus3g 5 22:50 qom_0.5_live_c147_bonus5g qom_0_live_bq_c147_t3\n' >> $X
echo "$(date +%H:%M:%S): the cf 0.3 arm queued:"; grep -vE "^#" $X
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-04 22:53:00" S67_DEADLINE="2026-11-04 23:00:00" S67_XALONE="$now" bash $HERE/s67_slot.sh
echo "second runner exit $?"
