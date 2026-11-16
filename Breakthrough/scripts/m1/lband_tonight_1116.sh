#!/bin/bash
# lband_tonight_1116.sh (Breakthrough, Mon 16 Nov 2026 19:5x): NewIdeas 19:3x moves the per-layer lambda band screen
# (docs/CHECK_BYTES_900.md section 6) from Tuesday morning to tonight. No data dependency on Fit12's kill test (its marker
# was only the slot gate). Steps:
#   1. the GO gate, scripts/m1/lband_go_1116.sh, in the background -> $D/GO (Sandbox98's lock gone, no quiet lock, BigMoE's
#      key test first if its gate is open);
#   2. data/m1/lband_map/run_lband_1116.frozen.sh on $D/GO: start from now, latest start 21:00, hard stop 22:50,
#      NO-SWAP >= 32 (MEM 24 + 8), share lock "MEM 24 GB: ..." to the hard stop, self-guard BG_PEAK_MAX 30; log and end
#      marker in $D;
#   3. if any of the 4 arm files is then missing (skipped or cut): the Tuesday fallback, the same runner on Fit12's
#      KILL_DONE with its own defaults (Tue 07:00, latest start 09:00, hard stop 09:25); log in $D2. --skip-done runs only
#      the missing arms.
# Usage: bash scripts/m1/lband_tonight_1116.sh
# Sim overrides: LBT_D, LBT_D2, LBT_O (arm dir), LBT_KD (the fallback's marker), LBT_LATEST / LBT_HARD (epoch), LBT_NS;
# the runner's LB_JOB / LB_Q / LB_PAT / LB_LOCKCMD / LB_UNLOCKCMD and the gate's GO_* pass through from the environment.
cd ~/Breakthrough || exit 9
R=data/m1/lband_map/run_lband_1116.frozen.sh
D=${LBT_D:-$HOME/Breakthrough/data/m1/lband_map/tonight1116}
D2=${LBT_D2:-$HOME/Breakthrough/data/m1/lband_map/tue1117}
O=${LBT_O:-$HOME/FableGod/decision_map/out_code24}
KD=${LBT_KD:-$HOME/Fit12/data/res1c/killtest/KILL_DONE}
NS=${LBT_NS:-32}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
LATEST=${LBT_LATEST:-$(ep "2026-11-16 21:00:00")}
HARD=${LBT_HARD:-$(ep "2026-11-16 22:50:00")}
mkdir -p "$D" "$D2"
GO_BY=$LATEST bash scripts/m1/lband_go_1116.sh "$D/GO" &
gp=$!
LB_D=$D LB_START=$(date +%s) LB_LATEST=$LATEST LB_HARD=$HARD LB_NS_MIN=$NS LB_MEM=24 BG_PEAK_MAX=30 bash "$R" "$D/GO"
rc=$?
kill "$gp" 2>/dev/null   # the gate ends by itself at GO or at the latest start; this only tidies a leftover
echo "$(date '+%a %H:%M:%S') tonight's runner rc $rc: $(cat "$D/LBAND_END" 2>/dev/null)"
miss=""
for a in bt_c6tiers bt_lb0 bt_lb1 bt_lb2; do [ -s "$O/$a.json" ] || miss="$miss $a"; done
if [ -z "$miss" ]; then echo "all 4 arms done tonight: no Tuesday fallback"; exit 0; fi
echo "$(date '+%a %H:%M:%S') missing arms:$miss -> Tuesday fallback on $KD"
LB_D=$D2 LB_NS_MIN=$NS LB_MEM=24 BG_PEAK_MAX=30 bash "$R" "$KD"
echo "$(date '+%a %H:%M:%S') Tuesday fallback rc $?: $(cat "$D2/LBAND_END" 2>/dev/null)"
