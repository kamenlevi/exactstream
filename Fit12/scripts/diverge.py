"""Fit12: first token where each arm's saved output leaves exact B12LL's (ref_nat), per prompt. Usage: diverge.py DIR"""
import glob, json, os, sys

d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Fit12/data/df2")
ref = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in glob.glob(os.path.join(d, "ref_nat_*.json"))}
arms = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(os.path.join(d, "*.json"))} - {"ref_nat"})
for p, r in sorted(ref.items()):
    row = []
    for a in arms:
        f = os.path.join(d, f"{a}_{p}.json")
        if not os.path.exists(f):
            continue
        o = json.load(open(f))
        i = next((j for j, (x, y) in enumerate(zip(r, o)) if x != y), None)
        row.append(f"{a} {'same' if i is None and len(o) == len(r) else i}")
    print(f"{p}: " + ", ".join(row))
