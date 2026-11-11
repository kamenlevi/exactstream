#!/bin/bash
# Install the bits == 2 branch in dm_t3._proj (Breakthrough, Wed 2026-11-04 06:3x; the orchestrator 06:2x): MoonShots'
# to2bit.py (c5af2d3, copied byte for byte into decision_map) for NewIdeas' qom_0_live_bq_t2 (QUEUE.md:1305-1320).
# Run with no map process (map_slot.sh's PRE, in the gap). The patched file is import-tested from a scratch folder, then
# moved in atomically. Exit 0 = installed (or already in), 1 = failed (dm_t3.py unchanged), 2 = a map process appeared.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch
grep -q "import to2bit" $DM/dm_t3.py && { echo "dm_t3: the 2-bit branch is already in"; exit 0; }
cmp -s ~/MoonShots/scripts/to2bit.py $DM/to2bit.py || echo "dm_t3: note: MoonShots' to2bit.py changed since the copy (the pinned c5af2d3 copy is used)"
mkdir -p $SP/t3test; cp $DM/dm_t3.py $SP/dm_t3.py.base
python3 - "$SP/dm_t3.py.base" "$SP/t3test/dm_t3.py" <<'PYEOF' || { echo "dm_t3: patch FAILED; unchanged"; exit 1; }
import sys
s = open(sys.argv[1]).read(); old = "def _proj(h, W, idx, bits):\n    if bits == 3:\n"
assert s.count(old) == 1, "anchor"
new = ("def _proj(h, W, idx, bits):\n"
       "    if bits == 2:   # Wed 06:3x (NewIdeas' qom_0_live_bq_t2, MoonShots' M-34): the MSB part at 2 bits, w = s3 (2 (q3 >> 1) + 0.5) + b3\n"
       "        import to2bit   # decision_map/to2bit.py = ~/MoonShots/scripts/to2bit.py at c5af2d3, byte for byte\n"
       "        return to2bit.proj2(h, W, idx)\n"
       "    if bits == 3:\n")
open(sys.argv[2], "w").write(s.replace(old, new))
PYEOF
(cd $SP/t3test && PYTHONPATH=$DM $PY -c "import inspect, dm_t3, to2bit; assert dm_t3.__file__.startswith('$SP'), dm_t3.__file__; assert 'to2bit.proj2' in inspect.getsource(dm_t3._proj); assert to2bit.__file__.startswith('$DM'), to2bit.__file__; print('dm_t3: import test OK')") \
  || { echo "dm_t3: import test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "dm_t3: a map process is running; not installed yet"; exit 2; }
cmp -s $SP/dm_t3.py.base $DM/dm_t3.py || { echo "dm_t3: changed since the copy; not installed"; exit 1; }
cp $SP/t3test/dm_t3.py $DM/.dm_t3.py.new && mv $DM/.dm_t3.py.new $DM/dm_t3.py && echo "dm_t3: the 2-bit branch is installed" || exit 1
