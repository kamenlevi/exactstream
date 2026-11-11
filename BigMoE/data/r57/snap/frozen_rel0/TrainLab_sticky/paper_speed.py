"""DRAFTER paper, brief rule 2 (MTP Head, Sun 11-08; CPU, existing rows only, no new runs): whole decode, whole request and
first token for the live drafter A/Bs, from the engine's per-prompt rows.
  whole decode   = sum(tokens) / sum(decode s), decode s = n / tok_s (the rows' own tok_s; prefill excluded)
  whole request  = sum(tokens) / sum(decode s + prefill s)   (pf = the row's prefill seconds = the wait for the first token)
  first token    = median pf over the prompts (seconds)
Selections = the read-outs' own: D-20b = data/d20_runs.jsonl at C 224 (only D-20b ran there), not paused, keys with both
arms; D-23 = data/d23_runs.jsonl clean rows; D-23d = data/d23c_runs.jsonl, head b5, preset 45+ooo, C 192, clean, keys
outside D-23c''s 18 (the last clean record per key and arm, as sticky/d23d_read.py). Output: data/paper_speed.txt"""
import json, os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
OLD = set(("humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18,multi/bg1,humaneval/HumanEval/6,"
           "math500/test/intermediate_algebra/1422.json,ifeval/374,agent/edit01,multi/fr6,humaneval/HumanEval/79,"
           "math500/test/algebra/694.json,ifeval/3407,agent/bash19,multi/es9,humaneval/HumanEval/19,"
           "math500/test/number_theory/427.json,agent/edit02").split(","))
L = []
def say(s):
    print(s, flush=True); L.append(s)
def load(f, ok):
    R = {}
    for l in open(os.path.join(TD, f)):
        d = json.loads(l)
        if ok(d):
            R.setdefault(d["key"], {})[d["arm"]] = d
    return R
def stats(R, keys, pol):
    n = sum(R[k][pol(k)]["n"] for k in keys)
    dec = sum(R[k][pol(k)]["n"] / R[k][pol(k)]["tok_s"] for k in keys)
    pf = [R[k][pol(k)]["pf"] for k in keys]
    return n / dec, n / (dec + sum(pf)), float(np.median(pf)), len(keys)
def table(name, R, a, b, groups, switch):
    keys = [k for k in R if {a, b} <= set(R[k])]
    say(f"{name}: {len(keys)} prompts with both arms")
    for gname, gk in groups(keys):
        for arm in (a, b):
            wd, wr, ft, nk = stats(R, gk, lambda k, arm=arm: arm)
            say(f"  {gname:10s} {arm:7s}: whole decode {wd:6.2f} tok/s | whole request {wr:6.2f} tok/s | first token median {ft:6.2f} s | n {nk}")
    wd, wr, ft, nk = stats(R, keys, switch)
    say(f"  {'switch':10s} {'':7s}: whole decode {wd:6.2f} tok/s | whole request {wr:6.2f} tok/s | first token median {ft:6.2f} s | n {nk}")
typ = lambda k: k.split("/")[0]
def by_type(keys):
    out = [("all", keys)]
    for t, names in (("code", ("humaneval",)), ("math", ("math500",)), ("chat", ("ifeval", "multi")), ("agent", ("agent",))):
        gk = [k for k in keys if typ(k) in names]
        if gk:
            out.append((t, gk))
    return out
R20 = load("d20_runs.jsonl", lambda d: d["C"] == 224 and not d["paused"])
table("D-20b (REF2b, C 224, head b3)", R20, "dfl", "mtp7", by_type, lambda k: "dfl" if typ(k) == "humaneval" else "mtp7")
R23 = load("d23_runs.jsonl", lambda d: not (d["cut"] or d["paused"] or d.get("invalid")))
table("D-23 (REF2-T default, C 224, head b3)", R23, "dfl", "mtp7", by_type, lambda k: "dfl" if typ(k) == "humaneval" else "mtp7")
R23d = load("d23c_runs.jsonl", lambda d: d["head"] == "b5" and d.get("pre") == "45+ooo" and d["C"] == 192
            and not (d["cut"] or d["paused"] or d["invalid"]) and d["key"] not in OLD)
table("D-23d (45+ooo default, C 192, head b5, separate passes)", R23d, "dfl", "mtp15e", by_type,
      lambda k: "mtp15e" if typ(k) in ("ifeval", "multi") else "dfl")
open(os.path.join(TD, "paper_speed.txt"), "w").write("\n".join(L) + "\n")
