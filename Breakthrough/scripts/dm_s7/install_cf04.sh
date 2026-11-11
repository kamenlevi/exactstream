#!/bin/bash
# NewIdeas' cf 0.4 chat arm for Thursday (NewIdeas 22:49, QUEUE 22:4x fb39925; the orchestrator: low priority, after DENSE-3,
# only if a share slot is free): qom_0.3_live_c147_bonus4g, the chat arms' q1_union form (rmask = NewIdeas' masks dir,
# renorm off; no t3). Appended to decision_map's dm_ablations.py after a test on a scratch copy (test_cf04_arm.py), moved
# in atomically with no map process. Idempotent. Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/cf04
HERE=$(cd "$(dirname "$0")" && pwd)
L1='ABLATIONS["qom_0.3_live_c147_bonus4g"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.3_live_c147_bonus4g"))   # NewIdeas 22:49 (QUEUE 22:4x fb39925): the chat route, bonus row gated at 0.4, 147-slot pool, 4-bit stock path'
pgrep -f "run_ma[p].py" > /dev/null && { echo "cf04: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
grep -qF 'ABLATIONS["qom_0.3_live_c147_bonus4g"]' $SP/test/dm_ablations.py || printf '%s\n' "$L1" >> $SP/test/dm_ablations.py
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "cf04: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY $HERE/test_cf04_arm.py) || { echo "cf04: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "cf04: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "cf04: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "cf04: installed"
