#!/bin/bash
# The decision map's scores-only capture (the orchestrator 07:2x, option (i) for Cloud Fable 3's R6 massB_0.5): CAPR "s" records
# each MoE layer's router top-10 scores (softmax fp32 precise, rank-sorted, renormalised over the 10: the xcapture's W) for
# EVERY row, without the hidden states; measure() saves them to <DM_OUT>/scores/<window>.npz (scores [n - 1, 48, 10] float16,
# ids [n - 1, 48, 10] int16, a). run_map.py gets the name "scapture". Run it with DM_OUT set to a scratch dir: a capture also
# writes <DM_OUT>/routing, and the map's own out/routing must stay as is (every mask was built on it).
# Three anchored edits to dm_common.py, one to run_map.py; the default paths are unchanged (CAPR "s" is False unless set).
# Test on scratch copies (import; the edits applied once each; diff = only the new lines), then cp + mv with no map process.
# Exit 0 = in, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/scap
pgrep -f "run_ma[p].py" > /dev/null && { echo "scap: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/new && cp $DM/dm_common.py $SP/base_common.py && cp $DM/run_map.py $SP/base_run.py || exit 1
grep -q 'CAPR.get("s")' $SP/base_common.py && grep -q '"scapture"' $SP/base_run.py && { echo "scap: already in"; exit 0; }
$PY - "$SP" <<'EOF' || { echo "scap: patch FAILED; unchanged"; exit 1; }
import sys, os
sp = sys.argv[1]
c = open(os.path.join(sp, "base_common.py")).read(); r = open(os.path.join(sp, "base_run.py")).read()
E = [  # (old, new) anchored, each must occur exactly once
 ('            if CAPR.get("x"):                             # sketch-on-miss (11-02): the routed experts\' input + router weights\n',
  '            if CAPR.get("x") or CAPR.get("s"):            # sketch-on-miss (11-02): the routed experts\' input + router weights; "s" (Thu 11-05): scores only\n'),
 ('                CAPR["X"].setdefault(li, []).append(x.reshape(-1, x.shape[-1]).astype(mx.float16))\n',
  '                if CAPR.get("x"): CAPR["X"].setdefault(li, []).append(x.reshape(-1, x.shape[-1]).astype(mx.float16))\n'),
 ('            if CAPR.get("x"):                               # out/xcap/<window>.npz: answer rows\' MoE input x [m, 48, 2560] and\n',
  '            if CAPR.get("s"):                               # <OUT>/scores/<window>.npz: every row\'s top-10 scores (Thu 11-05, C3-R6)\n'
  '                S = np.stack([np.array(mx.concatenate(CAPR["W"][L])) for L in range(48)], axis=1)[:len(ids) - 1]\n'
  '                os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)\n'
  '                np.savez(os.path.join(OUT, "scores", nm), scores=S.astype(np.float16), ids=np.stack(R, axis=1)[:len(ids) - 1], a=a)\n'
  '                extra += f"; scores {S.shape}"\n'
  '            if CAPR.get("x"):                               # out/xcap/<window>.npz: answer rows\' MoE input x [m, 48, 2560] and\n'),
 ('CAPR["on"] = False; CAPR["x"] = False\n', 'CAPR["on"] = False; CAPR["x"] = False; CAPR["s"] = False\n'),
]
for o, n in E:
    assert c.count(o) == 1, ("dm_common anchor", o[:70], c.count(o)); c = c.replace(o, n)
R = [('    if n not in ("baseline", "gate", "capture", "xcapture") and not n.startswith("ctx"): ablation(n)\n',
      '    if n not in ("baseline", "gate", "capture", "xcapture", "scapture") and not n.startswith("ctx"): ablation(n)\n'),
     ('        elif name == "xcapture": C.measure(model, windows, "xcapture", lambda: C.CAPR.update(on=True, x=True))   # 11-02 shadows\n',
      '        elif name == "xcapture": C.measure(model, windows, "xcapture", lambda: C.CAPR.update(on=True, x=True))   # 11-02 shadows\n'
      '        elif name == "scapture": C.measure(model, windows, "scapture", lambda: C.CAPR.update(on=True, s=True))   # Thu 11-05: scores only (set DM_OUT)\n')]
for o, n in R:
    assert r.count(o) == 1, ("run_map anchor", o[:70], r.count(o)); r = r.replace(o, n)
open(os.path.join(sp, "new", "dm_common.py"), "w").write(c); open(os.path.join(sp, "new", "run_map.py"), "w").write(r)
print("scap: patched copies written")
EOF
for f in dm_ablations.py dm_t3.py dm_gate.py to2bit.py records3_index_rowmajor.json; do [ -e $SP/new/$f ] || ln -sf $DM/$f $SP/new/$f; done
(cd $SP/new && DM_OUT=$SP/out_test $PY -c "
import ast, sys
for f in ('dm_common.py', 'run_map.py'): ast.parse(open(f).read())
import dm_common as C
assert C.__file__.startswith('$SP/new'), C.__file__
C.CAPR.update(on=True, x=True, s=True); C.clear(); assert C.CAPR['s'] is False and C.CAPR['x'] is False and C.CAPR['on'] is False
print('scap: import + clear() OK')
") || { echo "scap: test FAILED; unchanged"; exit 1; }
diff $SP/base_common.py $SP/new/dm_common.py | grep '^[<>]' | cut -c1-110
diff $SP/base_run.py $SP/new/run_map.py | grep '^[<>]' | cut -c1-110
pgrep -f "run_ma[p].py" > /dev/null && { echo "scap: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base_common.py $DM/dm_common.py && cmp -s $SP/base_run.py $DM/run_map.py || { echo "scap: the map's files changed since the copy; unchanged"; exit 1; }
cp $SP/new/dm_common.py $DM/.dm_common.py.new && cp $SP/new/run_map.py $DM/.run_map.py.new \
  && mv $DM/.dm_common.py.new $DM/dm_common.py && mv $DM/.run_map.py.new $DM/run_map.py && echo "scap: installed"
