"""CERTIFY98's scored lists after the contamination check (amendment 9's rule: a window flagged by either check is
dropped; more than 5 % dropped VOIDs a set). Reads the frozen v2 manifest, the frozen files_sha256.json and
contam_fresh_2026-11-18.txt; writes scored_lists_2026-11-18.json: each scored set's capture names (<set>/<name>, the
files_sha256.json keys without .npz) minus its dropped windows, the information sets unchanged.
  python3 results/certify98/scored_lists_2026-11-18.py"""
import hashlib, json, os, re

D = os.path.dirname(os.path.abspath(__file__))
P = {k: os.path.join(D, f) for k, f in (("manifest", "fresh_manifest_v2_2026-11-18.json"),
                                        ("files", "files_sha256_2026-11-18.json"), ("contam", "contam_fresh_2026-11-18.txt"))}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m, f = json.load(open(P["manifest"])), json.load(open(P["files"]))
assert f["manifest_sha256"] == sha(P["manifest"]), "files_sha256.json is not on this manifest"
sets = {"heplus_gated": [f"heplus/HumanEval_{i}" for i in m["heplus_gated"]],
        "mbpp_fresh": [f"mbpp/mbpp_{i}" for i in m["mbpp_fresh"]],
        "everyday_reported": [f"everyday/{k.replace('/', '_')}" for v in m["everyday_reported"].values() for k in v]}
info = {"heplus_lineage_b5": [f"heplus/HumanEval_{i}" for i in m["heplus_lineage_b5"]],
        "everyday_lineage_b5": [f"everyday/{k.replace('/', '_')}" for k in m["everyday_lineage_b5"]["ifeval"]]}
label = {"heplus_gated": "heplus_gated", "mbpp_fresh": "mbpp", "everyday_reported": "everyday"}   # the script's labels
txt = open(P["contam"]).read()
assert txt.rstrip().endswith("CONTAM FRESH: no set VOID"), "the check's last line is not 'no set VOID'"
out, drops = {}, {}
for s, names in sets.items():
    folder = names[0].split("/")[0]
    ends = re.findall(rf"^CONTAM fresh {label[s]} vs .*$", txt, re.M)
    assert len(ends) == 2 and all(": CLEAN" in e or ": FLAGGED" in e for e in ends), f"{s}: {ends}"
    d = sorted({f"{folder}/{x[:-4]}" for e in ends for x in re.findall(r"[\w.-]+\.npz", e.partition("(dropped:")[2])})
    assert set(d) <= set(names) and all(n + ".npz" in f["files"] for n in names), s
    assert len(d) <= 0.05 * len(names), f"{s}: {len(d)} dropped of {len(names)} is VOID"
    drops[s], out[s] = d, [n for n in names if n not in d]
json.dump({"rule": "docs/PREREG_certify98_2026-11-15.md amendment 9 (a flagged window is dropped)",
           "sources": {k: {"path": os.path.relpath(p, os.path.join(D, "../..")), "sha256": sha(p)} for k, p in P.items()},
           "dropped": drops, "scored": out, "information": info},
          open(os.path.join(D, "scored_lists_2026-11-18.json"), "w"), indent=1)
print("SCORED LISTS: " + "; ".join(f"{s} {len(out[s])} of {len(sets[s])} (dropped {', '.join(drops[s]) or 'none'})" for s in sets)
      + "; information " + ", ".join(f"{s} {len(v)}" for s, v in info.items()))
