"""NUMBER CHECK (NewIdeas, Mon 9 Nov): LiveCodeBench v6 recount of a card-gate arm vs the reference from the raw rows.
Code extraction as LCB's chat-model rule (the lines between the last two lines containing a ``` fence), key = (question id,
sha256(code)[:16]), verdict from EvalSuite's grade cache (the cached test runs are taken as given); discordant counts and the
exact two-sided McNemar p.  usage: /usr/bin/python3 -I lcb_recount.py <arm decode.jsonl> <reference decode.jsonl>"""
import hashlib, json, sys
from math import comb
ES = "/Users/kamen/EvalSuite"
cache = {}
for l in open(f"{ES}/results/lcb6_grades.jsonl"):
    r = json.loads(l)
    cache[(str(r["qid"]), r["sha"])] = r["pass"]
def code_of(text):
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if chr(96) * 3 in ln]
    return "\n".join(lines[idx[-2] + 1:idx[-1]]) if len(idx) >= 2 else ""
def load(p):
    return {json.loads(l)["key"]: json.loads(l) for l in open(p) if json.loads(l).get("set") == "lcb6"}
def verdicts(rows):
    out, miss = {}, 0
    for k, r in rows.items():
        qid = k.split("/", 1)[1]
        sha = hashlib.sha256(code_of(r["text"]).encode()).hexdigest()[:16]
        v = cache.get((qid, sha))
        if v is None:
            miss += 1
        out[k] = bool(v)
    return out, miss
A, ma = verdicts(load(sys.argv[1])); R, mr = verdicts(load(sys.argv[2]))
UNG = set()
for rows, out in ((load(sys.argv[1]), A), (load(sys.argv[2]), R)):
    for k, r in rows.items():
        sha = hashlib.sha256(code_of(r["text"]).encode()).hexdigest()[:16]
        if (k.split("/", 1)[1], sha) not in cache:
            UNG.add(k)
if UNG:
    print(f"NOT COMPLETE: {len(UNG)} keys have an ungraded answer (arm {ma}, ref {mr}); the counts below use the graded keys only")
keys = sorted((set(A) & set(R)) - UNG)
b = sum(A[k] and not R[k] for k in keys); c = sum(R[k] and not A[k] for k in keys); n = b + c
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
print(f"keys {len(keys)}, cache misses arm {ma} / ref {mr}: arm {sum(A[k] for k in keys)} ({sum(A[k] for k in keys) / len(keys) * 100:.1f} %) vs "
      f"reference {sum(R[k] for k in keys)} ({sum(R[k] for k in keys) / len(keys) * 100:.1f} %); arm only {b} / reference only {c}; "
      f"delta {(b - c) / len(keys) * 100:+.2f} pt; McNemar p {p:.4f}")
