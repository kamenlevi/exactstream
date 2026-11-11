"""Where did the repetition guard arm, and on what text? (CPU only.) For a guard-off / guard-on pair saved by rg_check.py
(data/rg/<TAG>_<key>.json), print the first divergence, and the earliest position >= 256 at which some 12-gram of the
guard-off tokens has occurred 4 times within the last 2,048 tokens (BigMoE's guard arming rule, as its repguard.py
docstring states; MTPLX's LoopGuard may differ in details), with that 12-gram decoded.
usage: rg_ngram.py KEY_FILE_PART OFF_TAG ON_TAG   e.g. rg_ngram.py math500_test_prealgebra_1640.json off_a on_a"""
import json, os, sys
from collections import defaultdict

part, off_tag, on_tag = sys.argv[1], sys.argv[2], sys.argv[3]
d = os.path.expanduser("~/Breakthrough/data/rg")
a = json.load(open(os.path.join(d, f"{off_tag}_{part}")))
b = json.load(open(os.path.join(d, f"{on_tag}_{part}")))
ta, tb = a["tokens"], b["tokens"]
div = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
print(f"{part}: off {len(ta)} tokens ({a['finish']}, guard {a['guard']}), on {len(tb)} tokens ({b['finish']}, guard "
      f"{b['guard']}); identical prefix {div}")
N, K, WIN, MIN = 12, 4, 2048, 256
pos = defaultdict(list)
arm, gram = None, None
for i in range(N, len(ta) + 1):
    g = tuple(ta[i - N:i])
    pos[g] = [p for p in pos[g] if p > i - WIN] + [i]
    if i >= MIN and len(pos[g]) >= K and arm is None:
        arm, gram = i, g
        break
if arm is None:
    print("no 12-gram reached 4 occurrences in the guard-off output")
    sys.exit(0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/BigMoE/models/bare"))
print(f"rule met at token {arm} (divergence at {div}): 12-gram {tok.decode(list(gram))!r} at positions "
      f"{[p - N for p in pos[gram]]}")
print(f"guard-off text around the divergence: {tok.decode(ta[max(0, div - 40):div + 20])!r}")
print(f"guard-on  text around the divergence: {tok.decode(tb[max(0, div - 40):div + 20])!r}")
