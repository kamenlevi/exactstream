"""SVH wiring check (Leap, Wed 2026-11-04 16:5x; TABLE_350 'SVH'): ref2's screened verifier head on the 125B's real 4-bit
lm_head, REF2 on, 16-row calls.
 (a) the subset's logits == REF2's full head at those ids, bitwise; the rest -inf; the argmax == the full one whenever
     that is in the subset; counted;
 (b) session ids (screened_head_add) outside the base: their logits == the full head's, bitwise; reset drops them;
 (c) switch off -> == REF2's full head, nothing counted; REF2 removed -> MLX's own head, untouched;
 (d) ref2.self_test PASS; (e) ms per 16-row head call: REF2's full head vs screened (base + 2,048 session rows).
usage: [BigMoE venv] sh_wire_check.py -> ~/Leap/data/sh_wire_check.txt (GPU, MEM ~1 GB; share lock)"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import mlx.nn as nn
import ref2

OUT = os.path.expanduser("~/Leap/data/sh_wire_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nSVH wiring check {time.strftime('%a %H:%M:%S')}")
assert "screened_head" in ref2.CFG and not ref2.CFG["screened_head"], "patch missing or switch not default off"
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
pre = "language_model.lm_head"
sh = mx.load(f"{D}/{wmap[pre + '.weight']}")
head = nn.QuantizedLinear(2560, sh[pre + ".weight"].shape[0], bias=False, group_size=64, bits=4)
head.update({"weight": sh[pre + ".weight"], "scales": sh[pre + ".scales"], "biases": sh[pre + ".biases"]})
del sh
mx.eval(head.parameters())
V = head["weight"].shape[0]
ref2.apply()
mx.random.seed(7)
x = (mx.random.normal((1, 16, 2560)) * 1.5).astype(mx.bfloat16)
mx.eval(x)
f32 = lambda a: np.array(a.astype(mx.float32))
y_full = head(x); mx.eval(y_full)
n = ref2.screened_head_install(head)
ids = np.array(ref2._SH[id(head["weight"])]["ids"])
c0 = ref2.COUNTS["screened_head"]
ref2.CFG["screened_head"] = True
y_s = head(x); mx.eval(y_s)
F, S = f32(y_full)[0], f32(y_s)[0]
mask = np.zeros(V, bool); mask[ids] = True
in_eq = np.array_equal(np.array(y_s.view(mx.uint16))[0][:, mask], np.array(y_full.view(mx.uint16))[0][:, mask])
out_inf = bool(np.all(np.isneginf(S[:, ~mask])))
am_f, am_s = F.argmax(1), S.argmax(1)
inset = mask[am_f]
ok_a = in_eq and out_inf and bool(np.all(am_s[inset] == am_f[inset])) and ref2.COUNTS["screened_head"] == c0 + 1
say(f"(a) base {n} rows: subset logits == full bitwise {in_eq}; outside -inf {out_inf}; argmax == full where the full one "
    f"is in the subset ({int(inset.sum())}/16) {bool(np.all(am_s[inset] == am_f[inset]))}; counted -> {'OK' if ok_a else 'FAIL'}")
outside = np.setdiff1d(np.arange(V), ids)[::97][:300]
k = ref2.screened_head_add(outside.tolist() + outside[:10].tolist())
y_x = head(x); mx.eval(y_x)
X = np.array(y_x.view(mx.uint16))[0]
ok_b = k == len(outside) and np.array_equal(X[:, outside], np.array(y_full.view(mx.uint16))[0][:, outside])
ref2.screened_head_reset()
y_r = head(x); mx.eval(y_r)
ok_b &= bool(np.all(np.isneginf(f32(y_r)[0][:, outside])))
say(f"(b) {k} session ids: their logits == full bitwise {ok_b and True}; reset drops them -> {'OK' if ok_b else 'FAIL'}")
ref2.CFG["screened_head"] = False
c1 = ref2.COUNTS["screened_head"]
y_o = head(x); mx.eval(y_o)
ok_c = bool(mx.all(y_o.view(mx.uint16) == y_full.view(mx.uint16)).item()) and ref2.COUNTS["screened_head"] == c1
ref2.remove()
ref2.CFG["screened_head"] = True
y_m = head(x); mx.eval(y_m)
ref2.CFG["screened_head"] = False
ok_c &= ref2.COUNTS["screened_head"] == c1 and not bool(np.any(np.isneginf(f32(y_m))))
say(f"(c) switch off == REF2's full head, uncounted; REF2 removed -> the plain head: {ok_c}")
ref2.apply()
st = ref2.self_test()
ok_d = all(bool(v) for v in st.values())
say(f"(d) ref2.self_test: {'PASS' if ok_d else 'FAIL ' + str(st)}")
ref2.screened_head_add(outside[:200].tolist() + np.setdiff1d(np.arange(V), ids)[1::53][:1848].tolist())
T = {False: [], True: []}
for r in range(11):
    for on in ((False, True) if r % 2 == 0 else (True, False)):
        ref2.CFG["screened_head"] = on
        mx.eval([head(x) for _ in range(4)])
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([head(x) for _ in range(12)]); mx.synchronize()
        T[on].append((time.perf_counter() - t0) / 12)
ref2.CFG["screened_head"] = False
say(f"(e) ms per 16-row head call: REF2 full {1e3 * np.median(T[False]):.3f}, screened (base + "
    f"{len(ref2._SH[id(head['weight'])]['extra'])} session) {1e3 * np.median(T[True]):.3f}")
ref2.remove()
say(f"SVH wiring: {'PASS' if ok_a and ok_b and ok_c and ok_d else 'FAIL'}")
