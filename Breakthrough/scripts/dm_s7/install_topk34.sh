#!/bin/bash
# Register step 0b's arms in the decision map (Breakthrough, Thu 2026-11-05 19:5x; PLAN_12GB_90.md step 0b, for Fit12's wider
# pass 1):
#   - topk3 / topk4, the all-rows sources that the cert gate reads (out/topk<k>.json);
#   - explicit entries for cert_top4_p09, cert_top4_p08, cert_top5_p08 and cert_top3_p09.
# The cert arms resolve through dm_ablations' regex already. The s67 runner's installed() check greps for the literal name,
# though, so a regex-only arm would wait forever; each entry is the regex's own gate_k(("cert", "topk<k>", theta, k)).
# Same pattern as install_c3.sh: a scratch copy, a CPU test (each name's PERT; each cert entry equals the regex's), then an
# atomic move with no map process. Each line goes in only if its name is absent.
# Exit 0 = in (or already in), 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/topk34
pgrep -f "run_ma[p].py" > /dev/null && { echo "topk34: a map process is running; unchanged"; exit 2; }
mkdir -p $SP/test && cp $DM/dm_ablations.py $SP/base.py && cp $SP/base.py $SP/test/dm_ablations.py || exit 1
add() { grep -qF "ABLATIONS[\"$1\"]" $SP/test/dm_ablations.py || printf '%s\n' "$2" >> $SP/test/dm_ablations.py; }
add topk4 'ABLATIONS["topk3"] = topk_experts(3); ABLATIONS["topk4"] = topk_experts(4)   # Thu 11-05 19:5x (Breakthrough, PLAN_12GB_90 step 0b): all-rows sources for cert_top3 / cert_top4'
for spec in "4 0.9 p09" "4 0.8 p08" "5 0.8 p08" "3 0.9 p09"; do
  set -- $spec
  add "cert_top$1_$3" "ABLATIONS[\"cert_top$1_$3\"] = gate_k((\"cert\", \"topk$1\", $2, $1))   # Thu 11-05 19:5x (Breakthrough, step 0b): = the cert_top<k>_p<theta> regex, as a literal name for the runner"
done
cmp -s $SP/base.py $SP/test/dm_ablations.py && { echo "topk34: already in"; exit 0; }
(cd $SP/test && PYTHONPATH=$SP/test:$DM $PY -c '
import re, dm_ablations as A, dm_common as C
print("[t] dm_ablations from", A.__file__)
ok = True
for k in (3, 4):
    C.clear(); A.ABLATIONS[f"topk{k}"](); p = dict(C.PERT); C.clear(); ok &= p == {"topk": k, "renorm": False}; print(f"[t] topk{k}: {p}")
for k, th, nm in ((4, 0.9, "p09"), (4, 0.8, "p08"), (5, 0.8, "p08"), (3, 0.9, "p09")):
    a = f"cert_top{k}_{nm}"; C.clear(); A.ABLATIONS[a](); p = dict(C.PERT); C.clear()
    want = {"gate": ("cert", f"topk{k}", th, k), "renorm": False}; ok &= p == want
    C.clear(); A.ablation(a + "+code")(); q = dict(C.PERT); C.clear(); ok &= q == want
    print(f"[t] {a}: {p}; via ablation(+code): {q == want}")
print("ALL OK" if ok else "FAILED"); raise SystemExit(0 if ok else 1)') || { echo "topk34: test FAILED; unchanged"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "topk34: a map process appeared; unchanged"; exit 2; }
cmp -s $SP/base.py $DM/dm_ablations.py || { echo "topk34: dm_ablations.py changed since the copy; unchanged"; exit 1; }
cp $SP/test/dm_ablations.py $DM/.dm_ablations.py.new && mv $DM/.dm_ablations.py.new $DM/dm_ablations.py && echo "topk34: installed"
