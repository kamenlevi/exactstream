"""DENSE-N-RAM check (Leap, Wed 2026-11-04; TABLE_350 'DENSE-N-RAM'): ref2 with ref2_dnr_patch.py on a stand-in model
(QuantizedLinear modules at the 125B's dense shapes, random 4-bit codes, + a stand-in GDN with R-83):
 (a) _dn_unpack(_dn_repack(w)) == w bitwise (codes, scales, biases) on every shape incl. R-83's concat and the lm_head;
 (b) outputs after dense_native_convert == before, bitwise: rows 1 / 8 / 16 (checks) and 17 / 32 / 256 (prefill), ref2
     on; rows 16 / 64 with ref2 off; the lm_head is left in MLX's layout (head=False);
 (c) R-83: a concat built before convert is converted by it; one built after convert from converted modules; both ==
     the unconverted R-83, on-path (rows 1 / 16) and off-path (rows 17 / 64);
 (d) memory: active memory after convert within +-2 % of before (one copy); MLX's own kernel on a converted weight
     fails loudly;
 (e) dense_native_unconvert restores bitwise; ref2.self_test PASS.
Usage: [BigMoE venv] dn_ram_check.py -> ~/Leap/data/dn_ram_check.txt (GPU, MEM ~2 GB; share lock)"""
import gc, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
import ref2

OUT = os.path.expanduser("~/Leap/data/dn_ram_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nDENSE-N-RAM check {time.strftime('%a %H:%M:%S')}")
assert hasattr(ref2, "dense_native_convert"), "patch missing"
BF = mx.bfloat16
_bits = lambda a: a.view(mx.uint16) if a.dtype == BF else a
eqb = lambda a, b: bool(a.shape == b.shape and a.dtype == b.dtype and mx.all(_bits(a) == _bits(b)).item())
mx.random.seed(44)

def rand_q(N, K):
    hi = mx.random.randint(0, 1 << 16, (N, K // 8)).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, (N, K // 8)).astype(mx.uint32)
    w = mx.bitwise_or(mx.left_shift(hi, 16), lo)
    s = (mx.random.uniform(0.5, 1.5, (N, K // 64)) * 0.004).astype(BF)
    b = (-8.0 * s.astype(mx.float32) + mx.random.normal((N, K // 64)) * 0.001).astype(BF)
    mx.eval(w, s, b)
    return w, s, b

def qlin(N, K):
    m = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    m.update(dict(zip(("weight", "scales", "biases"), rand_q(N, K))))
    mx.eval(m.parameters())
    return m

# (a) the inverse
ok_a = True
for N, K in ((10240, 2560), (6144, 2560), (2560, 6144), (12288, 2560), (2560, 640), (16384, 2560), (248320, 2560)):
    w, s_, b_ = rand_q(N, K)
    back = ref2._dn_unpack(*ref2._dn_repack(w, s_, b_), N, K)
    same = all(eqb(x, y) for x, y in zip(back, (w, s_, b_)))
    ok_a &= same
    if not same:
        say(f"(a) {N}x{K}: unpack(repack) DIFFERS")
    del w, s_, b_, back
say(f"(a) unpack(repack(w)) == w on 7 shapes: {ok_a}")

class GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv, self.in_proj_z = qlin(10240, 2560), qlin(6144, 2560)
        self.in_proj_b, self.in_proj_a = qlin(64, 2560), qlin(64, 2560)

class Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = [qlin(10240, 2560), qlin(6144, 2560), qlin(2560, 6144), qlin(12288, 2560), qlin(2560, 640),
                     qlin(248320, 2560)]
        self.gdn = GDN()

ref2.apply()
ref2.CFG["gdn_inproj_x"] = True
box = Box()
f_pre = ref2._InProjExact(box.gdn)                       # an R-83 concat that exists before convert
object.__setattr__(box.gdn, "in_proj_fused", f_pre)
xs = {r: {K: (mx.random.normal((1, r, K)) * 0.5).astype(BF) for K in (2560, 6144, 640)} for r in (1, 8, 16, 17, 32, 64, 256)}
mx.eval([v for d in xs.values() for v in d.values()])
K_of = lambda m: int(m["weight"].shape[1]) * 8 if m["weight"].ndim == 2 else None
mod_K = [K_of(m) for m in box.mods]
ROWS_ON, ROWS_OFF = (1, 8, 16, 17, 32, 256), (16, 64)

def outs_on():
    y = {}
    for i, m in enumerate(box.mods):
        for r in ROWS_ON if i < 5 else (1, 16):
            y[(i, r)] = m(xs[r][mod_K[i]])
    for r in (1, 16, 17, 64):
        y[("r83", r)] = box.gdn.in_proj_fused(xs[r][2560])
    mx.eval(y)
    return y

def outs_off():
    ref2.remove()
    y = {(i, r): m(xs[r][mod_K[i]]) for i, m in enumerate(box.mods[:5]) for r in ROWS_OFF}
    mx.eval(y)
    ref2.apply()
    ref2.CFG["gdn_inproj_x"] = True
    return y

ref_on, ref_off = outs_on(), outs_off()
keep_sd = [np.array(box.mods[4][k]) if k == "weight" else np.array(box.mods[4][k].view(mx.uint16)) for k in ("weight", "scales", "biases")]
gc.collect(); mx.clear_cache()
_am = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
mem0 = _am()
n, mb = ref2.dense_native_convert(box, head=False)
gc.collect(); mx.clear_cache()
mem1 = _am()
say(f"convert: {n} weights, {mb:.0f} MB; active memory {mem0 / 1e9:.3f} -> {mem1 / 1e9:.3f} GB ({100 * (mem1 / mem0 - 1):+.2f} %)")
ok_d = abs(mem1 / mem0 - 1) <= 0.02 and n == 8      # 5 modules + the gdn's qkv / z modules + f_pre's concat
head_ok = box.mods[5]["weight"].ndim == 2
try:
    y_ = ref2._ORIG["qmm"](xs[16][2560][0], box.mods[0]["weight"], box.mods[0]["scales"], box.mods[0]["biases"],
                           transpose=True, group_size=64, bits=4)
    mx.eval(y_)
    loud = False
except Exception:
    loud = True
ok_d &= loud and head_ok
say(f"(d) one copy (+-2 %) and 8 converted: {abs(mem1 / mem0 - 1) <= 0.02 and n == 8}; lm_head left in MLX's layout: {head_ok}; "
    f"MLX's kernel on a converted weight fails loudly: {loud}")

cnt0 = dict(ref2.COUNTS)
new_on, new_off = outs_on(), outs_off()
ok_b = all(eqb(ref_on[k][j] if isinstance(ref_on[k], tuple) else ref_on[k], new_on[k][j] if isinstance(new_on[k], tuple) else new_on[k])
           for k in ref_on for j in range(len(ref_on[k]) if isinstance(ref_on[k], tuple) else 1))
ok_b &= all(eqb(ref_off[k], new_off[k]) for k in ref_off)
dn = ref2.COUNTS["dense_native"] - cnt0.get("dense_native", 0)
up = ref2.COUNTS["dense_native_unpack"] - cnt0.get("dense_native_unpack", 0)
say(f"(b) after convert == before, bitwise (ref2 on: rows {ROWS_ON}; off: rows {ROWS_OFF}; R-83 rows 1 / 16 / 17 / 64): {ok_b} "
    f"(native calls {dn}, unpacked calls {up})")

# (c) R-83 built after convert, from converted modules
f_post = ref2._InProjExact(box.gdn)
c_conv = ref2._DNR.get(id(f_post.qz[0]), (None,))[0] is f_post.qz[0]
ok_c = c_conv
for r in (1, 16, 17, 64):
    a_, b_ = f_pre(xs[r][2560]), f_post(xs[r][2560])
    mx.eval(a_, b_)
    ok_c &= all(eqb(u, v) for u, v in zip(a_, b_)) and all(eqb(u, v) for u, v in zip(a_, ref_on[("r83", r)]))
say(f"(c) R-83 converted when built before convert and when built after (its concat registered: {c_conv}): == unconverted {ok_c}")

# (e) unconvert
del f_post
nu = ref2.dense_native_unconvert(box)
back_sd = [np.array(box.mods[4][k]) if k == "weight" else np.array(box.mods[4][k].view(mx.uint16)) for k in ("weight", "scales", "biases")]
ok_e = nu >= 8 and all(np.array_equal(a_, b_) for a_, b_ in zip(keep_sd, back_sd))
again = outs_on()
ok_e &= all(eqb(ref_on[k][j] if isinstance(ref_on[k], tuple) else ref_on[k], again[k][j] if isinstance(again[k], tuple) else again[k])
            for k in ref_on for j in range(len(ref_on[k]) if isinstance(ref_on[k], tuple) else 1))
ref2.CFG["gdn_inproj_x"] = False
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
say(f"(e) unconvert ({nu}) restores the weights bitwise and the outputs: {ok_e}; ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}")
ok = ok_a and ok_b and ok_c and ok_d and ok_e and st_ok
say(f"DENSE-N-RAM: {'PASS' if ok else 'FAIL'}")
