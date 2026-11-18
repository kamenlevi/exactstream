#!/bin/bash
# CERTIFY98 amendment 9's contamination check on the FRESH capture (after the freeze, before any scoring): the reported
# sets against t30b's whole lineage. tools/certify98_contam.py, one process at a time (CPU; take a share lock first).
#   bash results/certify98/contam_fresh_2026-11-18.sh > results/certify98/contam_fresh_2026-11-18.txt 2>&1
# - n1112's 3,072 captures (~/FableGod/deepdraft/capture/n1112), with the registered source tags (the default --forbid);
# - b5's 246 windows (the v2 manifest's lineage.b5_windows: ~/Leap/data/l120 and ~/TrainLab/data/d22c), on the span
#   rule alone (--forbid ''): they hold he_ and IFEval sources by construction, handled by name in amendment 9.
# Scored sets: HE+ gated 68, MBPP 300, everyday reported 150. Information: the lineage sets (HE+ 72, IFEval 57) vs b5.
set -u
cd "$HOME/EvalSuite" || exit 1
ROOT=$HOME/datasets/deepdraft/certify98
T=$(mktemp -d)
.venv/bin/python - "$ROOT" "$T" <<'EOF'
import json, os, sys
root, t = sys.argv[1:3]
m = json.load(open("results/certify98/fresh_manifest_v2_2026-11-18.json"))
H = os.path.expanduser("~")
sets = {"heplus_gated": [f"heplus/HumanEval_{i}" for i in m["heplus_gated"]],
        "mbpp": [f"mbpp/mbpp_{i}" for i in m["mbpp_fresh"]],
        "everyday": [f"everyday/{k.replace('/', '_')}" for v in m["everyday_reported"].values() for k in v],
        "heplus_lineage": [f"heplus/HumanEval_{i}" for i in m["heplus_lineage_b5"]],
        "everyday_lineage": [f"everyday/{k.replace('/', '_')}" for k in m["everyday_lineage_b5"]["ifeval"]]}
for s, ks in sets.items():
    ps = [os.path.join(root, k + ".npz") for k in ks]
    miss = [p for p in ps if not os.path.isfile(p)]
    assert not miss, f"{s}: {len(miss)} capture files missing, e.g. {miss[:2]}"
    open(os.path.join(t, s + ".txt"), "w").write("\n".join(ps) + "\n")
fold = {"l120": f"{H}/Leap/data/l120", "d22c": f"{H}/TrainLab/data/d22c"}
b5 = [os.path.join(fold[k.split("/")[0]], k.split("/")[1] + ".npz") for k in m["lineage"]["b5_windows"]]
assert len(b5) == 246 and all(os.path.isfile(p) for p in b5), "b5's windows moved"
open(os.path.join(t, "b5.txt"), "w").write("\n".join(b5) + "\n")
print({s: len(v) for s, v in sets.items()}, "b5", len(b5))
EOF
[ $? = 0 ] || { echo "CONTAM SETUP FAILED"; exit 2; }
N1112="$HOME/FableGod/deepdraft/capture/n1112/*.npz"
rc=0
for s in heplus_gated mbpp everyday; do
  .venv/bin/python tools/certify98_contam.py --test "@$T/$s.txt" --train "$N1112" --label "fresh $s vs n1112 (all 3,072 captures)" || rc=1
  .venv/bin/python tools/certify98_contam.py --test "@$T/$s.txt" --train "@$T/b5.txt" --forbid '' --label "fresh $s vs b5's 246 windows (span rule)" || rc=1
done
for s in heplus_lineage everyday_lineage; do      # information: these are in b5's training by name
  .venv/bin/python tools/certify98_contam.py --test "@$T/$s.txt" --train "@$T/b5.txt" --forbid '' --label "INFORMATION: $s vs b5's 246 windows"
done
rm -rf "$T"
echo "CONTAM FRESH: $([ $rc = 0 ] && echo 'no set VOID' || echo 'a set is VOID (above)')"
exit $rc
