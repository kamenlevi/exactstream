"""GDNB timing (Leap, Mon 2026-11-02; quiet lock): the one-dispatch GDN body (scripts/gdnbody.py, bitwise == REF2-T)
against REF2-T's chain, layer 0's real weights, 37-row prefill caches, 16 (and 1) verify rows.
- body: 36 bodies in one eval, each layer's z and a taken from the previous layer's y (so every recurrence and norm sits
  on the dependent path, as in the check); reference = exact prework -> gated_delta_update -> exact gated norm, fused =
  g / beta ops -> one kernel.
- layer: 36 whole GDN layers (in_proj -> body -> out_proj, x <- x + gdn(x)) in one eval.
A / B interleaved 4 times, median of 20 each; ms per layer and x 36 per check.
Usage: [BigMoE venv] gdnbody_time.py -> ~/Leap/data/gdnbody_time.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
import gdnbody as GB

OUT = os.path.expanduser("~/Leap/data/gdnbody_time.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
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
nn.quantize(gdn, group_size=cfg["quantization"]["group_size"], bits=cfg["quantization"]["bits"],
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in w)
gdn.load_weights(list(w.items()), strict=False)
gdn.eval()
mx.eval(gdn.parameters())
ref2.apply()
GP = ref2._gdnx()
SCALE = GP._scale()
mx.random.seed(1)
xp = mx.random.normal((1, 37, 2560)).astype(mx.bfloat16)
x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
c0 = ArraysCache(size=2)
mx.eval(gdn(xp, cache=c0), c0.state)
SNAP = (c0[0], c0[1])
NL = 36


def body_ref(qkv, z, a, b, cs, st):
    S = qkv.shape[1]
    qf, kf, vf, ns = GP.fused_gdn_conv_norm_rows(qkv.reshape(S, -1), cs.reshape(3, 10240), gdn.conv1d.weight)
    out, st2 = gated_delta_update(qf.reshape(1, S, 16, 128), kf.reshape(1, S, 16, 128), vf.reshape(1, S, 48, 128), a, b,
                                  gdn.A_log, gdn.dt_bias, st, None, use_kernel=True)
    return gdn.norm(out, z)


VAR = {"f8": dict(nsg=8), "f16": dict(nsg=16), "f32": dict(nsg=32), "f8_gout": dict(nsg=8, gates_in=False)}
if os.environ.get("GDNB_TIME_G3") == "1":           # G3 (Mon 22:4x): the row-interleaved recurrence next to its layouts
    VAR = {"f8": dict(nsg=8), "f8_g3": dict(nsg=8, g3=True), "f16": dict(nsg=16), "f16_g3": dict(nsg=16, g3=True),
           "f32_g3": dict(nsg=32, g3=True)}
def body_fused(kw):
    def f(qkv, z, a, b, cs, st):
        return GB.gdn_body(qkv, z, a, b, cs, gdn.conv1d.weight, gdn.A_log, gdn.dt_bias, st, gdn.norm.weight, gdn.norm.eps,
                           SCALE, **kw)[0]
    return f


ORIG = Q.GatedDeltaNet.__call__
FUSE = [None]                                     # None = reference, else gdn_body kwargs
def call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape
    if not (FUSE[0] is not None and B == 1 and mask is None and cache is not None and 1 <= S <= 16 and cache[0] is not None
            and cache[1] is not None):
        return ORIG(self, inputs, mask, cache)
    qkv_ = self.in_proj_qkv(inputs)
    z_ = self.in_proj_z(inputs).reshape(B, S, 48, 128)
    y_, *_, cs_, st_ = GB.gdn_body(qkv_, z_, self.in_proj_a(inputs), self.in_proj_b(inputs), cache[0], self.conv1d.weight,
                                   self.A_log, self.dt_bias, cache[1], self.norm.weight, self.norm.eps, SCALE, **FUSE[0])
    cache[0], cache[1] = cs_, st_
    cache.advance(S)
    return Q._linear(self.out_proj, y_.reshape(B, S, -1))
Q.GatedDeltaNet.__call__ = call


def bench(fn, reps=20):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / NL


say(f"GDNB timing {time.strftime("%a %H:%M")} ({os.environ.get("GDNB_COND", "quiet lock")})")
u = lambda a: np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a.view(mx.uint32))
x = x16
qkv = gdn.in_proj_qkv(x); z0 = gdn.in_proj_z(x).reshape(1, 16, 48, 128); a0, b0 = gdn.in_proj_a(x), gdn.in_proj_b(x)
qf, kf, vf, _ = GP.fused_gdn_conv_norm_rows(qkv.reshape(16, -1), SNAP[0].reshape(3, 10240), gdn.conv1d.weight)
out_r, st_r = gated_delta_update(qf.reshape(1, 16, 16, 128), kf.reshape(1, 16, 16, 128), vf.reshape(1, 16, 48, 128), a0, b0,
                                 gdn.A_log, gdn.dt_bias, SNAP[1], None, use_kernel=True)
yr = gdn.norm(out_r, z0)
for name, kw in VAR.items():
    y_, *_, st_ = GB.gdn_body(qkv, z0, a0, b0, SNAP[0], gdn.conv1d.weight, gdn.A_log, gdn.dt_bias, SNAP[1], gdn.norm.weight,
                              gdn.norm.eps, SCALE, **kw)
    say(f"  bits {name}: y == REF2-T chain {bool((u(y_) == u(yr)).all())}, recurrent state {bool((u(st_) == u(st_r)).all())}")
for S in (16, 1):
    x = x16[:, :S]
    qkv = gdn.in_proj_qkv(x)
    z0 = gdn.in_proj_z(x).reshape(1, S, 48, 128)
    a0, b0 = gdn.in_proj_a(x), gdn.in_proj_b(x)
    mx.eval(qkv, z0, a0, b0)

    def body_chain(fn):
        def run():
            y = z0
            for _ in range(NL):
                y = fn(qkv, y, y[..., 0], b0, SNAP[0], SNAP[1])
            return y
        return run

    def layer_chain(kw):
        def run():
            FUSE[0] = kw
            h = x
            for _ in range(NL):
                cc = ArraysCache(size=2)
                cc[0], cc[1] = SNAP
                h = h + gdn(h, cache=cc)
            return h
        return run
    arms = [("ref", body_ref, None)] + [(n, body_fused(kw), kw) for n, kw in VAR.items()]
    rb = {n: [] for n, _, _ in arms}
    rl = {n: [] for n, _, _ in arms}
    for _ in range(4):
        for n, fb, kw in arms:
            rb[n].append(bench(body_chain(fb)))
            rl[n].append(bench(layer_chain(kw)))
    ref_b, ref_l = np.median(rb["ref"]), np.median(rl["ref"])
    say(f"  S {S}: body per layer (ms, 36 chained): " + "; ".join(
        f"{n} {np.median(v):.4f} [{min(v):.4f}-{max(v):.4f}]" + ("" if n == "ref" else f" ({np.median(v) - ref_b:+.4f})")
        for n, v in rb.items()))
    say(f"  S {S}: whole layer (ms, 36 chained): " + "; ".join(
        f"{n} {np.median(v):.4f} [{min(v):.4f}-{max(v):.4f}]" + ("" if n == "ref" else f" ({np.median(v) - ref_l:+.4f})")
        for n, v in rl.items()))
    best = min((n for n in VAR), key=lambda n: np.median(rl[n]))
    say(f"  S {S}: best fused {best}: {36 * (ref_l - np.median(rl[best])):+.2f} ms per check (36 GDN layers, whole-layer chain); "
        f"body {np.median(rb[best]):.4f} ms per layer (line <= 0.2, kill > 0.25)")
Q.GatedDeltaNet.__call__ = ORIG
