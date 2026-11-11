"""DENSE-N wiring check (Leap, Wed 2026-11-04; TABLE_350 'DENSE-N'): ref2 with ref2_dn_patch.py applied.
 (a) _dn_repack (MLX ops, GPU) == qmm26.repack (numpy) bitwise (wn, sn, bn) on the 125B's dense shapes (one matrix each);
 (b) nn.QuantizedLinear modules at those shapes under ref2, rows 1 / 8 / 16: dense_native on == off bitwise; each
     module repacked once (dense_native_repack), every <= 16-row call on the native path (dense_native);
 (c) a raw quantized_matmul on a weight SLICE with the switch on keeps qmm16c (no repack);
 (d) R-83's _InProjExact (gdn_inproj_x on) on a stand-in GDN: on == off bitwise, the concat repacked once;
 (e) nt 2 == nt 1 bitwise; remove() drops the copies; ref2.self_test PASS.
Usage: [BigMoE venv] dn_wire_check.py -> ~/Leap/data/dn_wire_check.txt (GPU, MEM ~1.5 GB; share lock)"""
import os, sys, time, types
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
import ref2
from qmm26 import repack

OUT = os.path.expanduser("~/Leap/data/dn_wire_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
_bits = lambda a: a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
eqb = lambda a, b: bool(a.shape == b.shape and a.dtype == b.dtype and mx.all(_bits(a) == _bits(b)).item())
say(f"\nDENSE-N wiring check {time.strftime('%a %H:%M:%S')}")
assert "dense_native" in ref2.CFG and not ref2.CFG["dense_native"], "patch missing or switch not default off"
ref2.apply()
mx.random.seed(4)
SH = [("in_proj_qkv", 10240, 2560), ("in_proj_z", 6144, 2560), ("out_proj", 2560, 6144), ("q_proj", 12288, 2560),
      ("shared down", 2560, 640), ("lm_head", 248320, 2560)]
def rand_q(N, K):
    hi = mx.random.randint(0, 1 << 16, (N, K // 8)).astype(mx.uint32)   # random 32-bit words (no mx.random.bits)
    lo = mx.random.randint(0, 1 << 16, (N, K // 8)).astype(mx.uint32)
    w = mx.bitwise_or(mx.left_shift(hi, 16), lo)
    s = (mx.random.uniform(0.5, 1.5, (N, K // 64)) * 0.004).astype(mx.bfloat16)
    b = (-8.0 * s.astype(mx.float32) + mx.random.normal((N, K // 64)) * 0.001).astype(mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b
def qlin(N, K):
    m = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    w, s, b = rand_q(N, K)
    m.update({"weight": w, "scales": s, "biases": b})
    mx.eval(m.parameters())
    return m
ok = True
# (a)
for nm, N, K in SH:
    w, s, b = rand_q(N, K)
    g = ref2._dn_repack(w, s, b)
    c = repack(w, s, b)
    same = all(eqb(x, y) for x, y in zip(g, c))
    ok &= same
    say(f"(a) {nm} {N}x{K}: GPU repack == qmm26.repack {same}")
    del w, s, b, g, c
# (b) + (e)
mods = {nm: qlin(N, K) for nm, N, K in SH if nm != "lm_head"}
res = {}
for on in (False, True, "nt2"):
    ref2.CFG["dense_native"] = bool(on)
    ref2.CFG["dense_native_nt"] = 2 if on == "nt2" else 1
    for k in ("dense_native", "dense_native_repack", "qmm16c"):
        ref2.COUNTS[k] = 0
    ys = []
    for rows in (1, 8, 16):
        for nm, m in mods.items():
            mx.random.seed(rows * 7 + len(nm))
            x = (mx.random.normal((1, rows, int(m.weight.shape[1]) * 8)) * 0.5).astype(mx.bfloat16)
            ys.append(m(x))
    mx.eval(ys)
    res[on] = (ys, dict(native=ref2.COUNTS["dense_native"], repack=ref2.COUNTS["dense_native_repack"],
                        qmm16c=ref2.COUNTS["qmm16c"]))
n_calls = 3 * len(mods)
same_on = all(eqb(a, b) for a, b in zip(res[False][0], res[True][0]))
same_nt = all(eqb(a, b) for a, b in zip(res[False][0], res["nt2"][0]))
cnt_ok = (res[False][1] == dict(native=0, repack=0, qmm16c=n_calls) and res[True][1] == dict(native=n_calls, repack=len(mods), qmm16c=0)
          and res["nt2"][1] == dict(native=n_calls, repack=0, qmm16c=0))
ok &= same_on and same_nt and cnt_ok
say(f"(b) modules x rows 1 / 8 / 16: on == off {same_on}; (e) nt 2 == nt 1 {same_nt}; counts off {res[False][1]}, "
    f"on {res[True][1]}, nt2 {res['nt2'][1]} -> {'OK' if cnt_ok else 'WRONG'}")
# (c) a slice keeps qmm16c
ref2.CFG["dense_native"], ref2.CFG["dense_native_nt"] = True, 1
m = mods["in_proj_qkv"]
for k in ("dense_native", "dense_native_repack", "qmm16c"):
    ref2.COUNTS[k] = 0
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
ysl = mx.quantized_matmul(x, m.weight[:4096], scales=m.scales[:4096], biases=m.biases[:4096], transpose=True, group_size=64, bits=4)
ref2.CFG["dense_native"] = False
yoff = mx.quantized_matmul(x, m.weight[:4096], scales=m.scales[:4096], biases=m.biases[:4096], transpose=True, group_size=64, bits=4)
c_ok = ref2.COUNTS["dense_native"] == 0 and ref2.COUNTS["dense_native_repack"] == 0 and ref2.COUNTS["qmm16c"] == 2 and eqb(ysl, yoff)
ok &= c_ok
say(f"(c) a weight slice with the switch on: qmm16c kept, no repack, == off -> {'OK' if c_ok else 'WRONG'} ({dict((k, ref2.COUNTS[k]) for k in ('dense_native', 'dense_native_repack', 'qmm16c'))})")
# (d) R-83 on a stand-in GDN
gdn = types.SimpleNamespace(in_proj_qkv=qlin(10240, 2560), in_proj_z=qlin(6144, 2560), in_proj_b=qlin(64, 2560),
                            in_proj_a=qlin(64, 2560))
ref2.CFG["gdn_inproj_x"] = True
try:
    f = ref2._InProjExact(gdn)
    outs = {}
    for on in (False, True):
        ref2.CFG["dense_native"] = on
        for k in ("dense_native", "dense_native_repack"):
            ref2.COUNTS[k] = 0
        ys = []
        for r in (1, 16):
            mx.random.seed(r)
            ys.append(f((mx.random.normal((1, r, 2560)) * 0.5).astype(mx.bfloat16)))
        mx.eval(ys)
        outs[on] = ([t for y in ys for t in y], ref2.COUNTS["dense_native"], ref2.COUNTS["dense_native_repack"])
    d_ok = all(eqb(a, b) for a, b in zip(outs[False][0], outs[True][0])) and outs[True][1:] == (2, 1) and outs[False][1:] == (0, 0)
except AssertionError as e:
    d_ok = False
    say(f"(d) stand-in rejected: {e}")
finally:
    ref2.CFG["gdn_inproj_x"] = False
ok &= d_ok
say(f"(d) R-83 concat (qkv | z) on == off, repacked once: {'OK' if d_ok else 'WRONG'}")
ref2.CFG["dense_native"] = False
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
ok &= st_ok
say(f"ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}")
ref2.remove()
r_ok = len(ref2._DN) == 0
ok &= r_ok
say(f"(e) remove() drops the copies: {r_ok}")
say(f"DENSE-N wiring: {'PASS' if ok else 'FAIL'}")
