#!/bin/bash
# Code-only map arms (Fable C's audit AUDIT_1105 sections 2-3 via the orchestrator: reach07, band5, top1 "code only"; Breakthrough,
# Thu 2026-11-05): (1) dm_common.load_windows honours DM_SETS (comma-separated domains, e.g. "code"; unset = every set, as
# before); (2) dm_ablations gets ABLATIONS["code"], a no-op setup, so "<arm>+code" resolves through ablation()'s "+" stack and
# writes out/<arm>+code.json (12 windows) beside the full arm's name. The s67 runner sets DM_SETS=code for "+code" names.
# Test on scratch copies (anchors once each; with DM_SETS=code load_windows gives the 12 code windows and without it the 36;
# ablation("c3_b2d+code") sets exactly c3_b2d's PERT), then cp + mv with no map process. Idempotent.
# Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/codeonly
pgrep -f "run_ma[p].py" > /dev/null && { echo "codeonly: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/new && cp $DM/dm_common.py $SP/base_common.py && cp $DM/dm_ablations.py $SP/base_abl.py || exit 1
grep -q 'DM_SETS' $SP/base_common.py && grep -qF 'ABLATIONS["code"]' $SP/base_abl.py && { echo "codeonly: already in"; exit 0; }
$PY - "$SP" <<'EOF' || { echo "codeonly: patch FAILED; unchanged"; exit 1; }
import sys, os
sp = sys.argv[1]
c = open(os.path.join(sp, "base_common.py")).read(); a = open(os.path.join(sp, "base_abl.py")).read()
o = '    W = []\n    for dom, wf in SETS:\n        z = np.load(os.path.join(TD, wf))\n'
n = ('    W = []; only = [d for d in os.environ.get("DM_SETS", "").split(",") if d]   # (Thu 11-05) e.g. "code": those sets only\n'
     '    for dom, wf in SETS:\n        if only and dom not in only: continue\n        z = np.load(os.path.join(TD, wf))\n')
if 'DM_SETS' not in c:
    assert c.count(o) == 1, ("load_windows anchor", c.count(o)); c = c.replace(o, n)
if 'ABLATIONS["code"]' not in a:
    a += 'ABLATIONS["code"] = lambda: None   # (Thu 11-05) no-op part: "<arm>+code" runs <arm> on the code windows (DM_SETS=code, the s67 runner)\n'
open(os.path.join(sp, "new", "dm_common.py"), "w").write(c); open(os.path.join(sp, "new", "dm_ablations.py"), "w").write(a)
print("codeonly: patched copies written")
EOF
for f in dm_t3.py dm_gate.py to2bit.py run_map.py records3_index_rowmajor.json; do [ -e $SP/new/$f ] || ln -sf $DM/$f $SP/new/$f; done
(cd $SP/new && DM_OUT=$SP/out_test $PY -c "
import os
import dm_common as C, dm_ablations as A
assert C.__file__.startswith('$SP/new') and A.__file__.startswith('$SP/new'), (C.__file__, A.__file__)
W = C.load_windows(12); os.environ['DM_SETS'] = 'code'; Wc = C.load_windows(12); del os.environ['DM_SETS']
assert len(W) == 36 and len(Wc) == 12 and all(w[0] == 'code' for w in Wc) and [w[1] for w in Wc] == [w[1] for w in W if w[0] == 'code'], (len(W), len(Wc))
C.clear(); A.ablation('c3_b2d')(); p = dict(C.PERT); C.clear(); A.ablation('c3_b2d+code')(); q = dict(C.PERT); C.clear()
assert p == q and p.get('rmask', '').endswith('out/c3/c3_b2d'), (p, q)
print('codeonly: load_windows 36 / 12 with DM_SETS=code; c3_b2d+code PERT == c3_b2d PERT')
") || { echo "codeonly: test FAILED; unchanged"; exit 1; }
diff $SP/base_common.py $SP/new/dm_common.py | grep '^[<>]' | cut -c1-120
pgrep -f "run_ma[p].py" > /dev/null && { echo "codeonly: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base_common.py $DM/dm_common.py && cmp -s $SP/base_abl.py $DM/dm_ablations.py || { echo "codeonly: the map's files changed since the copy; unchanged"; exit 1; }
cp $SP/new/dm_common.py $DM/.dm_common.py.new && cp $SP/new/dm_ablations.py $DM/.dm_ablations.py.new \
  && mv $DM/.dm_common.py.new $DM/dm_common.py && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "codeonly: installed"
