"""Adds the mc2 capture (the next 500 Magicoder prompts) to data/split.json as 'mc2', in the prompts' seeded order;
train.py's split('pool2') = pool + mc2."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb import core as C
MAGIC = os.path.expanduser("~/datasets/deepdraft/magicoder/prompts_py_v1.jsonl")
keys = []
for i, l in enumerate(open(MAGIC)):
    if i < 540:
        continue
    k = json.loads(l)["key"]
    if os.path.exists(os.path.join(C.DATA, "mc2", k + ".npz")):
        keys.append(k)
    if i >= 540 + 499:
        break
p = os.path.join(C.DATA, "split.json"); s = json.load(open(p)); s["mc2"] = keys
json.dump(s, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
print(f"mc2: {len(keys)} answers; pool2 = {len(s['pool'])} + {len(keys)}")
