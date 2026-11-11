"""GDNB test (Leap, Mon 2026-11-02 19:4x): scripts/gdnbody.py's one-dispatch GDN body against REF2-T on a real layer.
Layer 0's GatedDeltaNet of the bare 125B (MTPLX's class, real 4-bit weights, as r71b), ref2 on (REF2-T: gdn_exact).
A 37-row prefill fills the conv / recurrent caches, then 16 verify rows.
1. Pieces: the fused body vs the reference chain on the same in_proj outputs: q, k, v, y (pre out_proj), conv state,
   recurrent state, bitwise.
2. Layer: the module with the fused body (an in-process patch of GatedDeltaNet.__call__ for B 1, no mask, 1-16 rows) vs
   the unpatched module: the layer output and both caches, bitwise, at S 16, 8, 1.
3. INV: one 16-row fused call == 16 one-row fused calls in sequence (y and states, bitwise).
4. Speed: 36 chained layer forwards (x <- x + gdn(x), one cache each) in one eval, reference vs fused, S 16 and 1,
   median of 15; and kernel-launching nodes of one layer call (R-65's census, views excluded).
Lines (cloud Fable's): bits identical 1-row vs 16-row and vs REF2-T; <= 0.2 ms per layer; KILL > 0.25 ms or any row
dependence. Here "per layer" = the in-situ body (whole layer, fused minus reference, + the reference body's own time).
Usage: [BigMoE venv] gdnbody_test.py -> ~/Leap/data/gdnbody_test.txt"""
import collections, json, os, re, sys, tempfile, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
import gdnbody as GB

OUT = os.path.expanduser("~/Leap/data/gdnbody_test.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"GDNB test {time.strftime('%a %H:%M')}")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg["text_config"])
pre = "language_model.model.layers.0.linear_attn."
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}")
    w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
gdn = Q.GatedDeltaNet(args)
qcfg = cfg["quantization"]
nn.quantize(gdn, group_size=qcfg["group_size"], bits=qcfg["bits"],
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in w)
gdn.load_weights(list(w.items()), strict=False)
gdn.eval()
mx.eval(gdn.parameters())
ref2.apply()
GP = ref2._gdnx()
SCALE = GP._scale()
say(f"layer 0 loaded ({len(w)} tensors); A_log {gdn.A_log.dtype}, dt_bias {gdn.dt_bias.dtype}, norm w {gdn.norm.weight.dtype}, "
    f"eps {gdn.norm.eps}, conv w {gdn.conv1d.weight.shape}")
mx.random.seed(1)
xp = mx.random.normal((1, 37, 2560)).astype(mx.bfloat16)
x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
c0 = ArraysCache(size=2)
mx.eval(gdn(xp, cache=c0), c0.state)
SNAP = (c0[0], c0[1])
u16 = lambda a: np.array(a.view(mx.uint16)) if a.dtype in (mx.bfloat16, mx.float16) else np.array(a.view(mx.uint32))
same = lambda a, b: bool(a.shape == b.shape and (u16(a) == u16(b)).all())

# ---- 1. pieces
qkv = gdn.in_proj_qkv(x16)
z = gdn.in_proj_z(x16).reshape(1, 16, 48, 128)
b = gdn.in_proj_b(x16)
a = gdn.in_proj_a(x16)
qf, kf, vf, ns_ref = GP.fused_gdn_conv_norm_rows(qkv.reshape(16, -1), SNAP[0].reshape(3, 10240), gdn.conv1d.weight)
q_r, k_r, v_r = qf.reshape(1, 16, 16, 128), kf.reshape(1, 16, 16, 128), vf.reshape(1, 16, 48, 128)
out_r, st_r = gated_delta_update(q_r, k_r, v_r, a, b, gdn.A_log, gdn.dt_bias, SNAP[1], None, use_kernel=True)
y_r = gdn.norm(out_r, z)
y_f, q_f, k_f, v_f, cs_f, st_f = GB.gdn_body(qkv, z, a, b, SNAP[0], gdn.conv1d.weight, gdn.A_log, gdn.dt_bias, SNAP[1],
                                             gdn.norm.weight, gdn.norm.eps, SCALE)
mx.eval(y_r, st_r, ns_ref, y_f, cs_f, st_f, q_f, k_f, v_f)
pieces = {"q": same(q_f, q_r), "k": same(k_f, k_r), "v": same(v_f, v_r), "y": same(y_f, y_r),
          "conv state": same(cs_f, ns_ref.reshape(1, 3, 10240)), "recurrent state": same(st_f, st_r)}
say(f"1. pieces vs REF2-T's chain (16 rows): {pieces}")
if not pieces["y"]:
    d = np.abs(np.array(y_f.astype(mx.float32)) - np.array(y_r.astype(mx.float32)))
    say(f"   y: {(u16(y_f) != u16(y_r)).mean() * 100:.3f} % of values differ, max |diff| {d.max():.3e}")

# ---- 2. layer-level, via an in-process patch
ORIG = Q.GatedDeltaNet.__call__
FUSE = [False]
def call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape
    if not (FUSE[0] and B == 1 and mask is None and cache is not None and 1 <= S <= 16 and cache[0] is not None
            and cache[1] is not None and getattr(self, "in_proj_fused", None) is None):
        return ORIG(self, inputs, mask, cache)
    qkv_ = self.in_proj_qkv(inputs)
    z_ = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b_ = self.in_proj_b(inputs)
    a_ = self.in_proj_a(inputs)
    y_, *_, cs_, st_ = GB.gdn_body(qkv_, z_, a_, b_, cache[0], self.conv1d.weight, self.A_log, self.dt_bias, cache[1],
                                   self.norm.weight, self.norm.eps, SCALE)
    cache[0] = cs_
    cache[1] = st_
    cache.advance(S)
    return Q._linear(self.out_proj, y_.reshape(B, S, -1))
Q.GatedDeltaNet.__call__ = call

def layer(fuse, x):
    FUSE[0] = fuse
    c = ArraysCache(size=2)
    c[0], c[1] = SNAP
    yy = gdn(x, cache=c)
    mx.eval(yy, c.state)
    return yy, c[0], c[1]
for S in (16, 8, 1):
    r_ = layer(False, x16[:, :S])
    f_ = layer(True, x16[:, :S])
    say(f"2. layer S {S}: output {same(f_[0], r_[0])}, conv state {same(f_[1], r_[1])}, recurrent state {same(f_[2], r_[2])}")

# ---- 3. INV: one 16-row fused call == 16 one-row fused calls
FUSE[0] = True
c = ArraysCache(size=2)
c[0], c[1] = SNAP
ys = [gdn(x16[:, j:j + 1], cache=c) for j in range(16)]
mx.eval(ys, c.state)
y1 = mx.concatenate(ys, axis=1)
yA, csA, stA = layer(True, x16)
say(f"3. INV fused: 16-row == 16 one-row calls: output {same(yA, y1)}, conv state {same(csA, c[0])}, recurrent state "
    f"{same(stA, c[1])}")

# ---- 4. speed and kernels
VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided", "Flatten",
         "Unflatten", "StopGradient", "Depends", "Split"}
def census(arr):
    fd, p = tempfile.mkstemp(suffix=".dot")
    os.close(fd)
    cnt = collections.Counter()
    try:
        mx.export_to_dot(p, arr)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m and m.group(1) not in VIEWS:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return cnt
NL = 36
def chain(fuse, S):
    FUSE[0] = fuse
    cs = []
    for _ in range(NL):
        cc = ArraysCache(size=2)
        cc[0], cc[1] = SNAP
        cs.append(cc)
    x = x16[:, :S]
    for l in range(NL):
        x = x + gdn(x, cache=cs[l])
    return x
for S in (16, 1):
    res = {}
    for fuse in (False, True, False, True):
        for _ in range(3):
            mx.eval(chain(fuse, S))
        ts = []
        for _ in range(15):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(chain(fuse, S))
            ts.append(time.perf_counter() - t0)
        res.setdefault(fuse, []).append(1e3 * float(np.median(ts)) / NL)
    r_, f_ = np.mean(res[False]), np.mean(res[True])
    FUSE[0] = False
    c = ArraysCache(size=2)
    c[0], c[1] = SNAP
    kr = census(gdn(x16[:, :S], cache=c))
    FUSE[0] = True
    c = ArraysCache(size=2)
    c[0], c[1] = SNAP
    kf = census(gdn(x16[:, :S], cache=c))
    say(f"4. S {S}: whole GDN layer in a 36-layer chain: reference {r_:.4f} ms, fused {f_:.4f} ms per layer "
        f"(runs {', '.join(f'{v:.4f}' for v in res[False])} / {', '.join(f'{v:.4f}' for v in res[True])}); "
        f"saved {r_ - f_:+.4f} ms per layer = {(r_ - f_) * 36:+.2f} ms per check (36 GDN layers); kernels per layer call: "
        f"reference {sum(kr.values())} {dict(kr)}, fused {sum(kf.values())} {dict(kf)}")
Q.GatedDeltaNet.__call__ = ORIG
