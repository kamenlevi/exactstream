#!/usr/bin/env python3
"""GPU tests for lmsl.py (V1 = R57_LMSL; Breakthrough, Mon 16 Nov 2026 13:xx), on the 125B's real 4-bit lm_head (only
its three arrays are loaded, ~0.36 GB) and Leap's ref2 SVH (screened_head_install / _add, _screened, _qlin).
  1. the unset path: with lmsl installed and ON off, the (wrapped) SVH call == the original bit for bit; ON on but
     outside the verify phase: the same.
  2. engaged (ON, verify phase): logits finite exactly on (list U session ids U drafts), -inf elsewhere; the finite values
     == a float32 reference (x . dequantized rows) within 2e-2 of the row max; the argmax == the reference argmax over
     the union on every row whose reference top-2 gap is > 0.05.
  3. SVH's feed (ref2.screened_head_add) reaches both states; add_drafts reaches the list state only.
  4. counters: lmsl_rows == the engaged rows; lmsl_fallback == a numpy recount (p1 - p2 <= 0.1 over the union).
  5. ref2._qlin (the head's call) takes the list state when engaged; remove() puts ref2._screened back.
Usage: ~/BigMoE/.venv/bin/python -I -B test_lmsl.py   (GPU; ~1 GB)
"""
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                   # noqa: E402
import mlx.nn as nn                                     # noqa: E402
import numpy as np                                      # noqa: E402

import ref2                                             # noqa: E402
import lmsl                                             # noqa: E402

fails = []


def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails.append(msg)


# the real lm_head (4-bit g64, bf16 scales): only its three arrays, read through the safetensors header's offsets
def tensor(key):
    for f in sorted(glob.glob(os.path.expanduser("~/BigMoE/models/bare/*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        if key in h:
            v = h[key]
            a, b = v["data_offsets"]
            dt = {"U32": np.uint32, "BF16": np.uint16}[v["dtype"]]
            raw = np.fromfile(f, dtype=dt, count=(b - a) // np.dtype(dt).itemsize, offset=8 + n + a).reshape(v["shape"])
            arr = mx.array(raw)
            return arr.view(mx.bfloat16) if v["dtype"] == "BF16" else arr
    raise KeyError(key)


head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
head.weight, head.scales, head.biases = (tensor(f"language_model.lm_head.{k}") for k in ("weight", "scales", "biases"))
mx.eval(head.parameters())
print(f"head: {head.weight.shape} {head.scales.dtype}, {sum(v.nbytes for v in head.parameters().values()) / 1e6:.0f} MB")

ref2.screened_head_install(head, base_n=98304, cap=4096)
e = ref2._SH[id(head["weight"])]
n_ids, mb = lmsl.install(head)
e2 = lmsl._ST["e2"]
check(n_ids == 32768 and e2 is ref2._SH[("lmsl", id(head["weight"]))], f"install: the list state ({n_ids} ids, {mb:.0f} MB) "
      f"registered beside SVH's")
rng = np.random.default_rng(1116)
X = mx.array((rng.standard_normal((16, 2560)) * 1.0).astype(np.float32)).astype(mx.bfloat16)
S, B = head["scales"], head["biases"]
PH = {"p": "decode_verify"}
lmsl._PH[0] = lambda: PH["p"]
orig = lmsl._ORIG["screened"]

# 1. the unset path
lmsl.ON[0] = False
y0 = ref2._screened(e, X, S, B)
check(bool(mx.array_equal(y0, orig(e, X, S, B))), "ON off: the wrapped SVH call == the original, bit for bit")
lmsl.ON[0] = True
PH["p"] = "unknown"
check(bool(mx.array_equal(ref2._screened(e, X, S, B), orig(e, X, S, B))), "ON on outside the verify phase: SVH's own state")
PH["p"] = "decode_verify"

# 3. the feed: session ids reach both states; drafts the list state only
rest = np.setdiff1d(np.arange(248320), np.union1d(np.array(sorted(e["set"])), np.array(sorted(e2["set"]))))
sess = rest[:5].tolist()
ref2.screened_head_add(sess)
check(all(t in e["set"] for t in sess) and all(t in e2["set"] for t in sess), "screened_head_add: the session ids reach "
      "SVH's state and the list state")
outside = [t for t in rng.permutation(248320).tolist() if t not in e2["set"] and t not in e["set"]][:3]
nd = lmsl.add_drafts(outside)
check(nd == 3 and all(t in e2["set"] for t in outside) and not any(t in e["set"] for t in outside),
      "add_drafts: 3 drafts join the list state only")

# 2. engaged
y = ref2._screened(e, X, S, B)
mx.eval(y)
yn = np.array(y.astype(mx.float32))
fin = np.isfinite(yn)
union = np.array(sorted(e2["set"]))
check(all(np.array_equal(np.where(fin[r])[0], union) for r in range(16)), f"engaged: finite exactly on the union "
      f"({union.size} ids: list + {len(e2['extra'])} session / draft ids), -inf elsewhere")
wd = mx.dequantize(head["weight"][mx.array(union)], head["scales"][mx.array(union)], head["biases"][mx.array(union)],
                   group_size=64, bits=4).astype(mx.float32)
ref = np.array(X.astype(mx.float32) @ wd.T)
err = np.abs(yn[:, union] - ref).max(1) / np.abs(ref).max(1)
check(err.max() <= 2e-2, f"engaged: values == the float32 reference within {err.max():.1e} of the row max (line 2e-2)")
srt = np.sort(ref, axis=1)
gap = srt[:, -1] - srt[:, -2]
am_ok = all(union[np.argmax(ref[r])] == int(np.argmax(yn[r])) for r in range(16) if gap[r] > 0.05)
check(am_ok, f"engaged: argmax == the reference's over the union on the {(gap > 0.05).sum()} rows with gap > 0.05")
check(np.isfinite(yn[:, outside]).all(), "engaged: the added drafts are scored (finite)")

# 4. counters
c = lmsl.take()
p = np.exp(yn - (np.log(np.exp(yn - yn.max(1, keepdims=True)).sum(1, keepdims=True)) + yn.max(1, keepdims=True)))
ps = np.sort(p, axis=1)
fb_ref = int(((ps[:, -1] - ps[:, -2]) <= 0.1).sum())
check(c["lmsl_rows"] == 16 and c["lmsl_fallback"] == fb_ref, f"take(): lmsl_rows 16, lmsl_fallback {c['lmsl_fallback']} == "
      f"the numpy recount {fb_ref}; ON off after take ({lmsl.ON[0]})")

# 5. the head's own call, and remove()
lmsl.ON[0] = True
ref2.CFG["screened_head"] = True
ref2._ON[0] = True
yq = ref2._qlin(head, X)
check(bool(mx.array_equal(yq, ref2._screened(e, X, S, B))) and np.isfinite(np.array(yq.astype(mx.float32))[:, outside]).all(),
      "ref2._qlin on the head takes the list state when engaged")
lmsl.ON[0] = False
ref2._ON[0] = False
lmsl.remove()
check(ref2._screened is orig and ("lmsl", id(head["weight"])) not in ref2._SH, "remove(): ref2._screened back, the list "
      "state unregistered")
print("ALL PASS" if not fails else f"{len(fails)} FAIL", flush=True)
sys.exit(1 if fails else 0)
