#!/bin/sh
# T1 patch (Fable A, 5 Nov 2026): input sparsity arms xs64 / xs48 / xs40 / xs32 for the decision map. See INSTALL.md.
# usage: sh apply.sh [target dir]        default target: ~/FableGod/decision_map
B=$(cd "$(dirname "$0")" && pwd); DM=${1:-$HOME/FableGod/decision_map}
WC=78f789c9dd45f742581f9403205a4132; WA=e5c608ed0b572ad6e2f419e03da5b48f
[ -f "$DM/dm_common.py" ] && [ -f "$DM/dm_ablations.py" ] || { echo "no harness in $DM"; exit 1; }
if [ "$(md5 -q "$DM/dm_common.py")" != "$WC" ] || [ "$(md5 -q "$DM/dm_ablations.py")" != "$WA" ]; then
  echo "NOT APPLIED: $DM is not the HEAD this patch was cut against (or is already patched). See INSTALL.md, 'If HEAD moved'."; exit 2
fi
if [ "$DM" = "$HOME/FableGod/decision_map" ] && pgrep -f run_map.py > /dev/null; then echo "NOT APPLIED: a map process is running"; exit 3; fi
cp -p "$DM/dm_common.py" "$DM/dm_common.py.pre_xs" && cp -p "$DM/dm_ablations.py" "$DM/dm_ablations.py.pre_xs" || exit 4
( cd "$DM" && patch -p1 -i "$B/t1_xs.patch" ) || { echo "patch failed: restoring"; cp -p "$DM/dm_common.py.pre_xs" "$DM/dm_common.py"; cp -p "$DM/dm_ablations.py.pre_xs" "$DM/dm_ablations.py"; exit 5; }
cp "$B/dm_xs.py" "$DM/dm_xs.py" || exit 6
echo "applied to $DM: dm_common.py $(md5 -q "$DM/dm_common.py"), dm_ablations.py $(md5 -q "$DM/dm_ablations.py"), dm_xs.py $(md5 -q "$DM/dm_xs.py")"
echo "inert until DM_XS=1 is set. Undo: sh $B/revert.sh $DM"
