#!/bin/bash
# NewIdeas' DENSE-3 map arm (NewIdeas 16:5x; the orchestrator's OK if it is in by ~19:30) into decision_map's
# dm_ablations.py: the pinned copy dense3_arm.py (= ~/NewIdeas/data/arms/dense3_arm.py at 8fefaa3) pasted after the
# s7_ref3 line. Test first on a scratch copy (test_dense3_arm.py: stack, rewrite, exact restore, env guard, on a mock
# model; CPU), then moved in atomically with no map process. Idempotent.
# Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/dense3
HERE=$(cd "$(dirname "$0")" && pwd)
pgrep -f "run_ma[p].py" > /dev/null && { echo "dense3: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
grep -qF 'ABLATIONS["s7_ref3"]' $SP/base.py || { echo "dense3: no s7_ref3 line (install_late.sh first); unchanged"; exit 1; }
grep -qF 'ABLATIONS["dense3"]' $SP/test/dm_ablations.py || cat $HERE/dense3_arm.py >> $SP/test/dm_ablations.py
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "dense3: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY $HERE/test_dense3_arm.py) || { echo "dense3: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "dense3: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "dense3: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "dense3: installed"
