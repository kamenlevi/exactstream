"""NUMBER CHECK (NewIdeas, Mon 9 Nov): an independent GPQA Diamond recount of a card-gate arm against the reference.
Scoring as specified (simple-evals ANSWER_PATTERN_MULTICHOICE: the first 'Answer: X' match after any think block == the gold
letter); my own reader of the raw decode rows; discordant counts and the exact two-sided McNemar p.
usage: /usr/bin/python3 -I gpqa_recount.py <arm decode.jsonl> <reference decode.jsonl>"""
import json, re, sys
from math import comb
PAT = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
ES = "/Users/kamen/EvalSuite"
gold = {}
for l in open(f"{ES}/data/items/dec_gpqad.jsonl"):
    it = json.loads(l)
    k = it.get("key") or it.get("id")
    g = it.get("answer") or it.get("gold") or it.get("target")
    gold[k] = g
def load(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("set") == "gpqad" or str(r.get("key", "")).startswith("gpqad/"):
            out[r["key"]] = r
    return out
def ok(r):
    t = re.sub(r"(?s)<think>.*?</think>", "", r.get("text", ""))
    m = PAT.search(t)
    k = r["key"]
    g = gold.get(k) or gold.get(k.split("/", 1)[1] if "/" in k else k)
    return m is not None and g is not None and m.group(1).upper() == str(g).strip().upper()[:1]
A, R = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(A) & set(R))
a = [ok(A[k]) for k in keys]; r = [ok(R[k]) for k in keys]
b = sum(x and not y for x, y in zip(a, r)); c = sum(y and not x for x, y in zip(a, r))
n = b + c; p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
print(f"keys {len(keys)} (gold for {sum(1 for k in keys if (gold.get(k) or gold.get(k.split('/',1)[1])) is not None)}): "
      f"arm {sum(a)} ({sum(a) / len(keys) * 100:.1f} %) vs reference {sum(r)} ({sum(r) / len(keys) * 100:.1f} %); "
      f"arm only {b} / reference only {c}; delta {(b - c) / len(keys) * 100:+.2f} pt; McNemar p {p:.4f}")
