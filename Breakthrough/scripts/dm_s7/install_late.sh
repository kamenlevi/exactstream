#!/bin/bash
# Late arms for tonight's batch (Breakthrough, Wed 2026-11-04 15:4x), appended to decision_map's dm_ablations.py after an
# import test from a scratch copy, moved in atomically with no map process. Idempotent (each line goes in once).
#   - Fable C's screens 1(b) / 1(c) (the orchestrator 15:3x; ~/FableGod/fablec/BYTES10.md:255-260): q1u_0.9_live_bonus9g,
#     qom_0.5_live_bonus5g = q1_union on the gate-file-cf rebuilds (NEVER the 00:34 _bonus9 / _bonus5).
#   - NewIdeas 14:5x: s7_ref3 = every kept expert of the gated rows at MSB-3 at all 48 layers (lsrc_mask bits 3), the
#     in-frame reference for screen 7 (the engine's FAST pool serves ranks 2-9 at 3 bits; the map's C form at 4).
# Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/late
L1='for _arm in ("q1u_0.9_live_bonus9g", "qom_0.5_live_bonus5g"):   # Fable C screens 1(b) / 1(c) (BYTES10.md:255-260; the orchestrator Wed 15:3x): the gate-file-cf rebuilds, NEVER the 00:34 _bonus9 / _bonus5'
L2='    ABLATIONS[_arm] = q1_union(os.path.expanduser(f"~/FableGod/fablec/masks/{_arm}"))   # union form, as qom_0_live_bonus (1(a)); twins q1u_0.9_live / qom_0.5_live'
L3='ABLATIONS["s7_ref3"] = lsrc_mask(_S7M, {L: 3 for L in range(48)})    # NewIdeas 14:5x: 7-ref3, every kept expert of the gated rows at MSB-3 at all 48 layers (the engine'"'"'s FAST-pool frame)'
pgrep -f "run_ma[p].py" > /dev/null && { echo "late: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
grep -qF 'q1u_0.9_live_bonus9g' $SP/test/dm_ablations.py || printf '%s\n' "$L1" "$L2" >> $SP/test/dm_ablations.py
grep -qF 'ABLATIONS["s7_ref3"]' $SP/test/dm_ablations.py || printf '%s\n' "$L3" >> $SP/test/dm_ablations.py
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "late: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY -c "
import os, dm_ablations as A, dm_common as C
assert A.__file__.startswith('$SP'), A.__file__
for n, t in (('q1u_0.9_live_bonus9g', 'q1u_0.9_live'), ('qom_0.5_live_bonus5g', 'qom_0.5_live')):
    C.clear(); A.ABLATIONS[n](); p = dict(C.PERT); C.clear(); A.ABLATIONS[t](); q = dict(C.PERT)
    assert p['rmask'] == os.path.expanduser('~/FableGod/fablec/masks/' + n) and set(p) == set(q) == {'rmask', 'renorm'}, (p, q)
C.clear(); A.ABLATIONS['s7_ref3'](); p = dict(C.PERT); C.clear()
assert p['t3L'] == {L: 3 for L in range(48)} and p['gate'] == ('t3mask',) and p['t3k'] == 2 and p['rmask'].endswith('qom_0_live_ball_c239_t3'), p
print('late: import + setups OK')
") || { echo "late: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "late: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "late: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "late: installed"
