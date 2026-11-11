#!/bin/bash
# Install the map screens 6 + 7 harness patch (Breakthrough, Wed 2026-11-04 14:5x; the orchestrator 14:3x ROUTE):
# scripts/dm_s7/patch_s7.py applied to decision_map's dm_t3.py / dm_common.py / dm_ablations.py. The patched files are
# built in a scratch folder from today's files, CPU-tested there (test_dm_s7.py cpu: wiring, arms, TRL gate), then moved
# in with no map process (each file atomically; the pre-patch files are kept in $SP/orig). Exit 0 = installed (or already
# in), 1 = failed (nothing changed), 2 = a map process is running (nothing changed).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/s7; H=~/Breakthrough/scripts/dm_s7
FILES="dm_t3.py dm_common.py dm_ablations.py"
grep -q "def routed_kept" $DM/dm_t3.py && grep -qF 'ABLATIONS["s7_trl"]' $DM/dm_ablations.py && grep -q '"t3L"' $DM/dm_common.py \
  && { echo "dm_s7: already installed"; exit 0; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "dm_s7: a map process is running; not installed"; exit 2; }
rm -rf $SP/base $SP/patched; mkdir -p $SP/base $SP/out $SP/orig
for f in $FILES; do cp $DM/$f $SP/base/ || exit 1; [ -e $SP/orig/$f ] || cp $DM/$f $SP/orig/; done
$PY $H/patch_s7.py $SP/base $SP/patched || { echo "dm_s7: patch FAILED; unchanged"; exit 1; }
ln -sf $DM/records3_index_rowmajor.json $SP/patched/records3_index_rowmajor.json
(cd $SP/patched && PYTHONPATH=$SP/patched:$DM DM_OUT=$SP/out $PY $H/test_dm_s7.py cpu) > $SP/install_test.log 2>&1 \
  || { echo "dm_s7: CPU tests FAILED (see $SP/install_test.log); unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "dm_s7: a map process appeared; not installed"; exit 2; }
for f in $FILES; do cmp -s $SP/base/$f $DM/$f || { echo "dm_s7: $f changed since the copy; not installed"; exit 1; }; done
for f in $FILES; do cp $SP/patched/$f $DM/.$f.new && mv $DM/.$f.new $DM/$f || { echo "dm_s7: installing $f FAILED"; exit 1; }; done
echo "dm_s7: installed ($(grep -c '^ok' $SP/install_test.log) CPU checks ok; pre-patch files in $SP/orig)"
