#!/bin/bash
# lband_tue_1117.sh (Breakthrough, Mon 16 Nov 2026 21:0x): NewIdeas 20:5x's slot for the per-layer lambda band screen
# (docs/CHECK_BYTES_900.md section 6, amendment (i)) after tonight's run was stopped: Tuesday morning in the gap after
# Fit12's kill test, with a Wednesday fallback for unfinished arms.
#   1. data/m1/lband_map/run_lband_1117.frozen.sh on Fit12's KILL_DONE AND my COMQOM_END (both newer than arming), no map
#      process, Kamen away (away_secs >= 900), all read twice; Tue 07:00 .. latest start 08:45, hard stop 09:25 (Leap's quiet
#      slot #1 at 09:30); NO-SWAP >= 30; share lock "MEM 22 GB" to the hard stop; BG_PEAK_MAX 30; yields when Kamen is back.
#   2. If any of the 4 arm files is then missing: the same runner on a marker written just after its arming (so: after the
#      morning run's end), Wed 06:30 .. latest start 07:15, hard stop 08:15 (before G2's day chain). --skip-done runs only
#      the missing arms. Log and end marker in $D2.
# Usage: bash scripts/m1/lband_tue_1117.sh
# Sim overrides: LBT_D, LBT_D2, LBT_O (arm dir), LBT_KD, LBT_CQ (markers), LBT_START / LBT_LATEST / LBT_HARD and
# LBT_WSTART / LBT_WLATEST / LBT_WHARD (epoch), LBT_NS; the runner's LB_JOB / LB_Q / LB_PAT / LB_LOCKCMD / LB_UNLOCKCMD /
# LB_AWAY_CMD pass through from the environment.
cd ~/Breakthrough || exit 9
R=data/m1/lband_map/run_lband_1117.frozen.sh
D=${LBT_D:-$HOME/Breakthrough/data/m1/lband_map/tue1117}
D2=${LBT_D2:-$HOME/Breakthrough/data/m1/lband_map/wed1118}
O=${LBT_O:-$HOME/FableGod/decision_map/out_code24}
KD=${LBT_KD:-$HOME/Fit12/data/res1c/killtest/KILL_DONE}
CQ=${LBT_CQ:-$HOME/Breakthrough/data/m1/comqom_map/COMQOM_END}
NS=${LBT_NS:-30}
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
mkdir -p "$D" "$D2"
LB_D=$D LB_START=${LBT_START:-$(ep "2026-11-17 07:00:00")} LB_LATEST=${LBT_LATEST:-$(ep "2026-11-17 08:45:00")} \
  LB_HARD=${LBT_HARD:-$(ep "2026-11-17 09:25:00")} LB_AFTER2=$CQ LB_NS_MIN=$NS LB_MEM=22 BG_PEAK_MAX=30 bash "$R" "$KD"
rc=$?
echo "$(date '+%a %H:%M:%S') Tuesday runner rc $rc: $(cat "$D/LBAND_END" 2>/dev/null)"
miss=""
for a in bt_c6tiers bt_lb0 bt_lb1 bt_lb2; do [ -s "$O/$a.json" ] || miss="$miss $a"; done
if [ -z "$miss" ]; then echo "all 4 arms done: no Wednesday fallback"; exit 0; fi
echo "$(date '+%a %H:%M:%S') missing arms:$miss -> Wednesday fallback (start 06:30, latest start 07:15, hard stop 08:15)"
( sleep 3; echo "LBAND MORNING END (for the Wednesday fallback): $(head -1 "$D/LBAND_END" 2>/dev/null)" > "$D2/GO" ) &
LB_D=$D2 LB_START=${LBT_WSTART:-$(ep "2026-11-18 06:30:00")} LB_LATEST=${LBT_WLATEST:-$(ep "2026-11-18 07:15:00")} \
  LB_HARD=${LBT_WHARD:-$(ep "2026-11-18 08:15:00")} LB_NS_MIN=$NS LB_MEM=22 BG_PEAK_MAX=30 bash "$R" "$D2/GO"
echo "$(date '+%a %H:%M:%S') Wednesday fallback rc $?: $(cat "$D2/LBAND_END" 2>/dev/null)"
