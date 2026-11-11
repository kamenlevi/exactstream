#!/bin/bash
# install_fc.sh (Fable C, Thu 2026-11-05): a COPY of Breakthrough's scripts/dm_s7/install_c3.sh for build_1105's arms; changed
# lines carry "FC" (default source and snapshot dirs, its own scratch dir, test_fc_arms.py beside this file, messages). It is
# for Breakthrough to run: like install_c3.sh it appends ABLATIONS lines to ~/FableGod/decision_map/dm_ablations.py.
# ---- the original header follows ----
# Cloud Fable 3's mask arms into the decision map (the orchestrator 06:5x; Breakthrough, Thu 2026-11-05). For each arm:
#   1. snapshot ~/FableGod/cloud3/live/cloud3/masks/<arm> into ~/FableGod/decision_map/out/c3/<arm> (a fixed copy, so a later
#      change in the cloud3 mirror cannot change what runs) with a sha256 manifest, after check_c3_masks.py passed on it;
#   2. append ABLATIONS["<arm>"] = q1_union("<snapshot dir>") (PERT {rmask, renorm False}; no t3, 4-bit stock path) to a
#      scratch copy of dm_ablations.py and test it (test_c3_arms.py: PERT, the loader's path, 36 windows, the snapshot ==
#      the source byte for byte);
#   3. move it in atomically with no map process. Idempotent per arm (an arm already in dm_ablations is skipped).
# Usage: install_c3.sh <arm> [<arm> ...]. Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SRC=${C3_SRC:-~/FableGod/fablec/build_1105/masks}; SNAP=${C3_SNAP:-$DM/out/fc}   # FC
export C3_SRC=$SRC C3_SNAP=$SNAP   # (Thu 07:2x) another source, e.g. NewIdeas' masks: C3_SRC=~/NewIdeas/data/dm_masks C3_SNAP=$DM/out/ni
SP=~/Breakthrough/data/abl_scratch/fc; HERE=$(cd "$(dirname "$0")" && pwd)   # FC
[ $# -ge 1 ] || { echo "usage: $0 <arm> ..."; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "fc: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test $SNAP && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
for a in "$@"; do
  [ -n "$a" ] && [ -d "$SRC/$a" ] || { echo "fc: no masks for '$a'"; exit 1; }
  t=$(mktemp -d "$SNAP/.$a.new.XXXX") && cp -R "$SRC/$a/." "$t/" && (cd "$t" && shasum -a 256 *.npz > "../$a.sha256") \
    || { echo "fc: snapshot of $a failed"; exit 1; }
  [ -e "$SNAP/$a" ] && mv "$SNAP/$a" "$SNAP/.$a.old.$(date +%s)"   # nothing deleted: an older snapshot is renamed aside
  mv "$t" "$SNAP/$a" || { echo "fc: snapshot of $a failed"; exit 1; }
  grep -qF "ABLATIONS[\"$a\"]" $SP/test/dm_ablations.py || \
    printf 'ABLATIONS["%s"] = q1_union("%s")   # snapshot of %s/%s (sha256 beside it), installed by install_fc.sh\n' "$a" "$SNAP/$a" "$SRC" "$a" >> $SP/test/dm_ablations.py
done
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "fc: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY $HERE/test_fc_arms.py "$@") || { echo "fc: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "fc: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "fc: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "fc: installed $*"
