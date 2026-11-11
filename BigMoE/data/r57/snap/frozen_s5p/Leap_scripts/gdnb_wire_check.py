"""GDNB wiring re-check (Leap, Wed 2026-11-04 17:0x; TABLE_350 'GDNB-2'): ref2 CFG gdn_body (the one-dispatch GDN body,
Mon 11-02, never bundled) under TODAY's exact config: R-83's exact fused in_proj (gdn_inproj_x), DENSE-N-RAM (in-place
native dense weights), the verify scope (decode_verify / target_verify / exact_verify), real layer 4 (GDN).
 (1) bits: gdn_body on (NSG 8 and 16) == off: the layer output and both caches at S 16 / 8 / 1;
 (2) ms per layer: 36 chained calls (x <- x + gdn(x), one cache each) at S 16, off / NSG 8 / NSG 16, median of 9.
usage: [BigMoE venv] gdnb_wire_check.py -> ~/Leap/data/gdnb_wire_check.txt (GPU, MEM ~1.5 GB; share lock)"""
import os, sys, json, time
os.environ.setdefault("MTPLX_FUSED_GDN_CONVNORM", "1"); os.environ.setdefault("MTPLX_FUSED_CONVNORM_VERIFY", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
import gdnbody as GB
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.cache import ArraysCache
from mtplx.attention_context import attention_phase, model_forward_kind, exact_verify
from mlx.utils import tree_flatten

OUT = os.path.expanduser("~/Leap/data/gdnb_wire_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nGDNB wiring re-check {time.strftime('%a %H:%M:%S')}")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
args = Q.TextArgs.from_dict(json.load(open(f"{D}/config.json"))["text_config"])
LAYER = 4
pre = f"language_model.model.layers.{LAYER}.linear_attn."
class Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Q.GatedDeltaNet(args)
box = Box()
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}"); w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
mod_in = {p_[:-len(".weight")]: int(a_.shape[-1]) for p_, a_ in tree_flatten(box.g.parameters()) if p_.endswith(".weight") and a_.ndim >= 2}
qmap = {}
for m_, in_f in mod_in.items():
    wq, sq = w.get(f"{m_}.weight"), w.get(f"{m_}.scales")
    if wq is not None and sq is not None and wq.dtype == mx.uint32:
        qmap[m_] = {"bits": int(wq.shape[-1]) * 32 // in_f, "group_size": in_f // int(sq.shape[-1])}
nn.quantize(box.g, group_size=64, bits=4, class_predicate=lambda p_, m: qmap.get(p_, False))
box.g.load_weights(list(w.items()), strict=False)
box.g.eval()
mx.eval(box.parameters())
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
n_conv, mb = ref2.dense_native_convert(box)
say(f"layer {LAYER} GDN, quantized modules {sorted(qmap)}; DENSE-N-RAM converted {n_conv} ({mb:.0f} MB)")
g = box.g
mx.random.seed(3)
def fresh():
    c = ArraysCache(size=2)
    x0 = (mx.random.normal((1, 37, 2560)) * 0.7).astype(mx.bfloat16)
    g(x0, None, c)                                        # prefill fills the conv / recurrent caches
    mx.eval(c[0], c[1])
    return c
def scope():
    import contextlib
    st = contextlib.ExitStack()
    st.enter_context(attention_phase("decode_verify")); st.enter_context(model_forward_kind("target_verify"))
    st.enter_context(exact_verify(True))
    return st
bits = lambda a: np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)
ok = True
base_state = fresh()
snap = (base_state[0], base_state[1])
for S in (16, 8, 1):
    x = (mx.random.normal((1, S, 2560)) * 0.7).astype(mx.bfloat16)
    outs = {}
    for arm in ("off", "nsg8", "nsg16"):
        ref2.CFG["gdn_body"] = arm != "off"
        GB.NSG_DEFAULT[0] = 16 if arm == "nsg16" else 8
        c = ArraysCache(size=2); c[0], c[1] = snap
        c0 = ref2.COUNTS["gdnb"]
        with scope():
            y = g(x, None, c)
        mx.eval(y, c[0], c[1])
        outs[arm] = (bits(y), bits(c[0]), bits(c[1]), ref2.COUNTS["gdnb"] - c0)
    ref2.CFG["gdn_body"] = False
    eq = all(all(np.array_equal(a, b) for a, b in zip(outs[k][:3], outs["off"][:3])) for k in ("nsg8", "nsg16"))
    took = outs["nsg8"][3] == 1 and outs["nsg16"][3] == 1 and outs["off"][3] == 0
    ok &= eq and took
    say(f"(1) S {S:2d}: GDNB NSG 8 / 16 == off (y, conv state, recurrent state) bitwise {eq}; GDNB taken {took}")
def chain(S=16, L=36):
    cs = []
    for _ in range(L):
        c = ArraysCache(size=2); c[0], c[1] = snap; cs.append(c)
    x = (mx.random.normal((1, S, 2560)) * 0.7).astype(mx.bfloat16)
    with scope():
        for c in cs:
            x = x + g(x, None, c)
    return x
T = {k: [] for k in ("off", "nsg8", "nsg16")}
for r in range(9):
    for arm in (list(T) if r % 2 == 0 else list(T)[::-1]):
        ref2.CFG["gdn_body"] = arm != "off"
        GB.NSG_DEFAULT[0] = 16 if arm == "nsg16" else 8
        mx.eval(chain())
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain()); mx.synchronize()
        T[arm].append((time.perf_counter() - t0) / 36)
ref2.CFG["gdn_body"] = False; GB.NSG_DEFAULT[0] = 8
b0 = np.median(T["off"])
say("(2) ms per GDN layer at S 16 (36 chained, median of 9): " + ", ".join(
    f"{k} {1e3 * np.median(v):.4f} ({1e3 * (np.median(v) - b0):+.4f}; x36 {36e3 * (np.median(v) - b0):+.2f} ms per check)" for k, v in T.items()))
st = ref2.self_test()
say(f"(3) ref2.self_test: {'PASS' if all(bool(v) for v in st.values()) else 'FAIL ' + str(st)}")
ref2.remove()
say(f"GDNB wiring re-check: {'PASS' if ok else 'FAIL'}")
