"""Fit12 s1qg_merge.py (Tue 10 Nov 10:4x; EvalSuite 10:4x, route (i)): merge s1qg.sh's chunk outputs for EvalSuite, per order.
READ-ONLY on the chunk files. Writes, in RUN_DIR, for ORDER in (fwd, rev) (chunks qgfNN / qgrNN):
- s1qg_<ORDER>_tokens.json = {key: {"ex": [[ids]]}} (each key's first complete run in that order; a key in two chunks of one
  order must have equal ids, else it is a clash, which EvalSuite treats as VOID);
- s1qg_<ORDER>_runs.json = [{"key": k, "arm": "ex", "cut": false}] for the same keys (a stopped key is in no chunk file, so it
  is absent and re-runs);
- s1qg_<ORDER>_meta.json = the per-key meta (n, cap, finish, pf_s, dec_s, tok_s, checks, paused_s).
Prints per order: keys / 160, finish stop / length, tokens, clashes; and the largest harness lifetime footprint.
Usage: python -I -B s1qg_merge.py RUN_DIR FWD_KEYS_FILE"""
import glob, json, os, re, sys

O = sys.argv[1]
FWD = [k.strip() for k in open(sys.argv[2]) if k.strip()]
life, L = [], []
for t in sorted(glob.glob(os.path.join(O, "qg[fr]*.txt"))):
    life += [float(x) for x in re.findall(r"lifetime ([\d.]+) GB", open(t).read())]
for order, pre, keys in (("fwd", "qgf", FWD), ("rev", "qgr", FWD[::-1])):
    ids, meta, clash = {}, {}, []
    for f in sorted(glob.glob(os.path.join(O, f"{pre}*_ids.json"))):
        if f.endswith("_meta.json"):
            continue
        a = json.load(open(f))
        mf = f[:-len(".json")] + "_meta.json"
        m = json.load(open(mf)) if os.path.exists(mf) else {}
        for k, v in a.items():
            if k in ids:
                if ids[k] != v:
                    clash.append(k)
                continue
            ids[k] = v
            if k in m:
                meta[k] = m[k]
    got = [k for k in keys if k in ids]
    out = {f"s1qg_{order}_tokens.json": {k: {"ex": ids[k]["ex"][:1]} for k in got},
           f"s1qg_{order}_runs.json": [{"key": k, "arm": "ex", "cut": False} for k in got],
           f"s1qg_{order}_meta.json": {k: meta[k] for k in got if k in meta}}
    for name, o in out.items():
        p = os.path.join(O, name)
        json.dump(o, open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
    fin = [meta[k].get("finish") for k in got if k in meta]
    L.append(f"{order}: {len(got)} / {len(keys)} keys (humaneval {sum(k.startswith('humaneval/') for k in got)}, mbpp "
             f"{sum(k.startswith('mbpp/') for k in got)}); finish stop {fin.count('stop')}, length {fin.count('length')}; tokens "
             f"{sum(len(ids[k]['ex'][0]) for k in got)}; clashes {clash if clash else 'none'}")
print(f"s1qg merge ({O}): " + " | ".join(L) + f" | largest harness lifetime {max(life) if life else 'n/a'} GB")
