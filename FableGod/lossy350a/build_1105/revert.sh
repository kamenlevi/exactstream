#!/bin/sh
# undo apply.sh: restore the two harness files saved as *.pre_xs and remove dm_xs.py
# usage: sh revert.sh [target dir]       default target: ~/FableGod/decision_map
DM=${1:-$HOME/FableGod/decision_map}
[ -f "$DM/dm_common.py.pre_xs" ] && [ -f "$DM/dm_ablations.py.pre_xs" ] || { echo "nothing to revert in $DM (no *.pre_xs)"; exit 1; }
if [ "$DM" = "$HOME/FableGod/decision_map" ] && pgrep -f run_map.py > /dev/null; then echo "NOT REVERTED: a map process is running"; exit 3; fi
mv "$DM/dm_common.py.pre_xs" "$DM/dm_common.py" && mv "$DM/dm_ablations.py.pre_xs" "$DM/dm_ablations.py" && rm -f "$DM/dm_xs.py"
echo "reverted $DM: dm_common.py $(md5 -q "$DM/dm_common.py"), dm_ablations.py $(md5 -q "$DM/dm_ablations.py")"
