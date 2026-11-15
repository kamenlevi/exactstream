#!/usr/bin/env python3
"""1c tiers CPU self-check (LOSSY_CHECK_SPEC patch note 5; Sun 15 Nov 2026). Replays bigmoe/tiers1c.py's rule on the
decision map's captured exact routing (~/FableGod/decision_map/out/xcap: answer rows' top-10 ids + normalised weights)
in 32-row blocks from the first answer row, and compares per (row, layer, rank) with Breakthrough's masks
(~/FableGod/decision_map/out/bt/tiers_1c: keep = 4-bit, keep3 = MSB-3; prompt rows all 4-bit).
- tiers_np (float64): must equal the masks bit for bit.
- tiers_mx (MLX float32, the engine's form, on the CPU): the slots that differ, and how close their I is to a threshold.
usage: .venv/bin/python scripts/tiers1c_selfcheck.py [code|all]  -> data/deepdraft/tiers1c_selfcheck.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from bigmoe import tiers1c as T
import mlx.core as mx
mx.set_default_device(mx.cpu)

M = os.path.expanduser("~/FableGod/decision_map/out")
which = sys.argv[1] if len(sys.argv) > 1 else "code"
files = sorted(f for f in os.listdir(os.path.join(M, "bt", "tiers_1c")) if which == "all" or "_code_" in f)
ROWS = 32
out, ok_np, ok_mx = {}, True, True
for f in files:
    z = np.load(os.path.join(M, "xcap", f)); mk = np.load(os.path.join(M, "bt", "tiers_1c", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"], int(z["a"])
    m, L = ids.shape[0], ids.shape[1]
    keep, keep3 = mk["keep"], mk["keep3"]
    assert keep.shape == (a + m, L, 10), (f, keep.shape, a, m)
    pr_ok = bool(keep[:a].all() and not keep3[:a].any())
    diff_np = diff_mx = slots = 0
    for b0 in range(0, m, ROWS):
        r = slice(b0, min(b0 + ROWS, m))
        for l in range(L):
            tn = T.tiers_np(ids[r, l], w[r, l])
            tm = np.array(T.tiers_mx(mx.array(ids[r, l].astype(np.int32)), mx.array(w[r, l])))
            ref = np.where(keep[a + b0:a + r.stop, l], 0, np.where(keep3[a + b0:a + r.stop, l], 1, 2))
            diff_np += int((tn != ref).sum()); diff_mx += int((tm != ref).sum()); slots += ref.size
    out[f] = {"prompt_rows": a, "answer_rows": m, "slots": slots, "prompt_all_4bit": pr_ok,
              "diff_float64": diff_np, "diff_mlx_fp32": diff_mx}
    ok_np &= diff_np == 0 and pr_ok; ok_mx &= diff_mx == 0
    print(f"{f:34s} answer rows {m:4d}: float64 diffs {diff_np}, MLX fp32 diffs {diff_mx} of {slots} slots; prompt rows all 4-bit {pr_ok}")
v = ("PASS" if ok_np else "FAIL") + f" (float64 rule vs the masks, {len(files)} windows); MLX fp32 form: " + \
    ("bit for bit" if ok_mx else f"{sum(o['diff_mlx_fp32'] for o in out.values())} slots differ")
print("TIERS1C SELF-CHECK:", v)
json.dump({"verdict": v, "windows": out}, open(os.path.expanduser("~/BigMoE/data/deepdraft/tiers1c_selfcheck.json"), "w"), indent=1)
