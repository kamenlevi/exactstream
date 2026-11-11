#!/bin/bash
# NewIdeas' chat route arm for tonight's map slot (the orchestrator 20:1x; NewIdeas QUEUE 20:1x 4fb08ca, masks 4324e09):
# qom_0.5_live_c147_bonus5g = Fable C's 1(c) rule (QOM cf 0.5, the bonus row gated at 0.5) on a 147-slot pool, 1(c)'s
# q1_union form (rmask = NewIdeas' masks dir, renorm off; no t3, 4-bit everywhere on the stock path). Appended to
# decision_map's dm_ablations.py after a test on a scratch copy (test_chat_arm.py), moved in atomically with no map
# process. Idempotent. Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/chat
HERE=$(cd "$(dirname "$0")" && pwd)
L1='ABLATIONS["qom_0.5_live_c147_bonus5g"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.5_live_c147_bonus5g"))   # NewIdeas 20:1x chat route (QUEUE 20:1x 4fb08ca, masks 4324e09): 1(c)'"'"'s rule on a 147-slot pool, 4-bit stock path'
pgrep -f "run_ma[p].py" > /dev/null && { echo "chat: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
grep -qF 'ABLATIONS["qom_0.5_live_c147_bonus5g"]' $SP/test/dm_ablations.py || printf '%s\n' "$L1" >> $SP/test/dm_ablations.py
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "chat: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY $HERE/test_chat_arm.py) || { echo "chat: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "chat: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "chat: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "chat: installed"
