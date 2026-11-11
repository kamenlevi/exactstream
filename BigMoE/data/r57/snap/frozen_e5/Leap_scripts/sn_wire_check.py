"""SMALL-N wiring check (Leap, Wed 2026-11-04; TABLE_350 'SMALL-N'): ref2 + glue_fused with ref2_sn_patch.py, random
4-bit weights at the default's small shapes:
 (a) a QuantizedLinear N 512 (k / v), rows 1 / 8 / 16: small_native on -> qmm16n (COUNTS); error vs an fp32 reference
     no worse than MLX's call; row-invariant (a 1-row call == the 16-row call's first row, bitwise);
 (b) R-74's concat through glue_fused.shared_cat on a stand-in block (gate / up N 640, shared_expert_gate N 1): the concat
     takes qmm16n; outputs close to the MLX path (FAST); the down projection unaffected;
 (c) R-83's b | a (N 64 + 64) through _InProjExact: marked and taken;
 (d) switch off: bitwise == today's path, nothing counted; (e) ref2.self_test PASS.
Usage: [BigMoE venv] sn_wire_check.py -> ~/Leap/data/sn_wire_check.txt (GPU, MEM ~0.5 GB; share lock)"""
import os, sys, time, types
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
import mlx.core as mx
import mlx.nn as nn
import ref2
import glue_fused as GF

OUT = os.path.expanduser("~/Leap/data/sn_wire_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nSMALL-N wiring check {time.strftime('%a %H:%M:%S')}")
assert "small_native" in ref2.CFG and not ref2.CFG["small_native"], "patch missing or switch not default off"
ref2.apply()
mx.random.seed(12)
f32 = mx.float32
rel = lambda a, b: float(mx.sqrt(mx.mean((a.astype(f32) - b.astype(f32)) ** 2)).item() / mx.sqrt(mx.mean(b.astype(f32) ** 2)).item())
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
def qlin(N, K):
    lin = nn.Linear(K, N, bias=False)
    lin.weight = (lin.weight * 4.0).astype(mx.bfloat16)        # bf16 weights -> bf16 scales / biases, as the model's
    m = nn.QuantizedLinear.from_linear(lin, 64, 4)
    mx.eval(m.parameters())
    return m
def ref32(m, x):
    w = mx.dequantize(m["weight"], m["scales"], m["biases"], group_size=64, bits=4).astype(f32)
    return x.astype(f32) @ w.T
ok = True
# (a)
m = qlin(512, 2560)
x16 = (mx.random.normal((1, 16, 2560)) * 0.5).astype(mx.bfloat16)
c0 = ref2.COUNTS["small_native"]
ref2.CFG["small_native"] = True
y16 = m(x16); y1 = m(x16[:, :1]); y8 = m(x16[:, :8])
ref2.CFG["small_native"] = False
yo = m(x16)
mx.eval(y16, y1, y8, yo)
r = ref32(m, x16)
a_ok = (ref2.COUNTS["small_native"] == c0 + 3 and eqb(y1, y16[:, :1]) and eqb(y8, y16[:, :8])
        and rel(y16, r) <= rel(yo, r) * 1.05)
ok &= a_ok
say(f"(a) N 512: 3 calls on qmm16n {ref2.COUNTS['small_native'] - c0 == 3}; 1 / 8-row == the 16-row rows {eqb(y1, y16[:, :1]) and eqb(y8, y16[:, :8])}; "
    f"error vs fp32 {rel(y16, r):.2e} (MLX's call {rel(yo, r):.2e}) -> {'OK' if a_ok else 'FAIL'}")
# (b) R-74
blk = types.SimpleNamespace(shared_expert=types.SimpleNamespace(gate_proj=qlin(640, 2560), up_proj=qlin(640, 2560),
                                                                down_proj=qlin(2560, 640)),
                            shared_expert_gate=qlin(1, 2560))
xb = (mx.random.normal((1, 16, 2560)) * 0.5).astype(mx.bfloat16)
g_off, s_off = GF.shared_cat(blk, xb)
c0 = ref2.COUNTS["small_native"]
ref2.CFG["small_native"] = True
g_on, s_on = GF.shared_cat(blk, xb)
ref2.CFG["small_native"] = False
mx.eval(g_off, s_off, g_on, s_on)
b_ok = ref2.COUNTS["small_native"] == c0 + 1 and rel(s_on, s_off) < 2e-2 and rel(g_on, g_off) < 2e-2
ok &= b_ok
say(f"(b) R-74 concat via glue_fused: on qmm16n {ref2.COUNTS['small_native'] - c0 == 1}; shared output vs MLX path rel {rel(s_on, s_off):.2e}, "
    f"gate {rel(g_on, g_off):.2e} -> {'OK' if b_ok else 'FAIL'}")
# (c) R-83
gdn = types.SimpleNamespace(in_proj_qkv=qlin(10240, 2560), in_proj_z=qlin(6144, 2560), in_proj_b=qlin(64, 2560), in_proj_a=qlin(64, 2560))
ref2.CFG["gdn_inproj_x"] = True
f = ref2._InProjExact(gdn)
c0 = ref2.COUNTS["small_native"]
ref2.CFG["small_native"] = True
o_on = f(xb)
ref2.CFG["small_native"] = False
o_off = f(xb)
ref2.CFG["gdn_inproj_x"] = False
mx.eval(o_on, o_off)
c_ok = ref2.COUNTS["small_native"] == c0 + 1 and eqb(o_on[0], o_off[0]) and rel(o_on[2], o_off[2]) < 2e-2
ok &= c_ok
say(f"(c) R-83 b | a marked and on qmm16n {ref2.COUNTS['small_native'] - c0 == 1}; qkv untouched {eqb(o_on[0], o_off[0])}; "
    f"b rel {rel(o_on[2], o_off[2]):.2e} -> {'OK' if c_ok else 'FAIL'}")
# (d) off
c0 = ref2.COUNTS["small_native"]
y2 = m(x16)
mx.eval(y2)
d_ok = eqb(y2, yo) and ref2.COUNTS["small_native"] == c0
ok &= d_ok
say(f"(d) switch off: == today's path, nothing counted: {d_ok}")
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
ok &= st_ok
say(f"(e) ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}")
ref2.remove()
say(f"SMALL-N wiring: {'PASS' if ok else 'FAIL'}")
