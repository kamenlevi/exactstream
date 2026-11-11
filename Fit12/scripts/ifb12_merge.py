"""Fit12 ifb12_merge.py (Mon 9 Nov 23:4x; v2 Tue 02:0x, EvalSuite's format 23:4x): merge ifb12.sh's chunk outputs for EvalSuite.
READ-ONLY on the chunk files. Writes, in RUN_DIR:
- ifb12_tokens.json = {key: {"ex": [[ids]]}} (tools/r57rows.py's input: arm "ex", the first run; EOS kept), the gate keys only,
  from the ifb12cNN chunks (each key's first complete run; a key in two chunks must have equal ids, else it is a clash);
- ifb12_runs.json = [{"key": k, "arm": "ex", "cut": false}] for the same keys (a stopped key is in no chunk file, so it is
  absent here and runs again; no key in these files was cut by a time budget);
- ifb12_meta.json = the chunks' per-key meta (n, cap, finish, pf_s, dec_s, tok_s, checks, paused_s): speeds are ROUGH (run
  beside the gate).
- The REVERSE CHECK (EvalSuite): the ifb12rev chunk re-ran the first NREV keys in reverse order; every key's ids must equal the
  first run's, else the arm is VOID. Not merged into the tokens.
Prints: keys merged / the gate keys, finish stop / length, tokens, the largest harness lifetime footprint, the reverse check.
Usage: python -I -B ifb12_merge.py RUN_DIR KEYS_CSV REV_KEYS_CSV"""
import glob, json, os, re, sys

O, KEYS, REV = sys.argv[1], sys.argv[2].split(","), [k for k in sys.argv[3].split(",") if k]
ids, meta, clash, life = {}, {}, [], []
for f in sorted(glob.glob(os.path.join(O, "ifb12c*_ids.json"))):
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
for t in sorted(glob.glob(os.path.join(O, "ifb12*.txt"))):
    life += [float(x) for x in re.findall(r"lifetime ([\d.]+) GB", open(t).read())]
keys = [k for k in KEYS if k in ids]
out = {"ifb12_tokens.json": {k: {"ex": ids[k]["ex"][:1]} for k in keys},
       "ifb12_runs.json": [{"key": k, "arm": "ex", "cut": False} for k in keys],
       "ifb12_meta.json": {k: meta[k] for k in keys if k in meta}}
for name, o in out.items():
    p = os.path.join(O, name)
    json.dump(o, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
rev = "not run"
rf = os.path.join(O, "ifb12rev_ids.json")
if os.path.exists(rf):
    r = json.load(open(rf))
    have = [k for k in REV if k in r and k in ids]
    diff = [k for k in have if r[k]["ex"][0] != ids[k]["ex"][0]]
    rev = (f"{len(have)} / {len(REV)} re-run, {len(diff)} differ {diff}" + (" -> VOID" if diff else
           (" -> OK" if len(have) == len(REV) else " -> INCOMPLETE")))
fin = [meta[k].get("finish") for k in keys if k in meta]
print(f"ifb12 merge ({O}): {len(keys)} / {len(KEYS)} gate keys; finish stop {fin.count('stop')}, length {fin.count('length')}; "
      f"tokens {sum(len(ids[k]['ex'][0]) for k in keys)}; largest harness lifetime {max(life) if life else 'n/a'} GB; "
      f"clashes {clash if clash else 'none'}; REVERSE CHECK: {rev}")
