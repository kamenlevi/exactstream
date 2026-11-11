"""D5 check (Leap, Wed 2026-11-04 16:4x; TABLE_350 'D5'): ref2 CFG hc_pend_mlp (the MLP write handed to the next HC read as
pending) on real weights, REF2 on, 16-row calls.
 (a) HC reads (layers 3 / 20 / 44, attention- and MLP-side modules, 4 scales, R-73's hc_write off and on): a read of the
     write's output with D5 on (it reads (hyper, pending)) == the read with D5 off: mixed / written / inject bitwise, and
     the fold counted;
 (b) the same on the HC8 path (8-bit mixers via hc8r.hc8_install, CFG hc8_fast: the FAST arm);
 (c) a 4-layer stack of real layers (attention / GDN + both HC modules; the MoE replaced by the same stub in both arms):
     the stack's output with D5 on == off, bitwise, with one fold per layer boundary; and ms per stack call (rough);
 (d) ref2.self_test PASS.
usage: [BigMoE venv] d5_check.py -> ~/Leap/data/d5_check.txt (GPU, MEM ~1 GB; share lock)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/d5_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nD5 check {time.strftime('%a %H:%M:%S')}")
assert "hc_pend_mlp" in ref2.CFG and not ref2.CFG["hc_pend_mlp"], "patch missing or switch not default off"
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg["text_config"])
def load(prefix, mod, skip=()):
    keys = [k for k in wmap if k.startswith(prefix) and not any(k[len(prefix):].startswith(s_) for s_ in skip)]
    w = {}
    for f in sorted({wmap[k] for k in keys}):
        sh = mx.load(f"{D}/{f}")
        w.update({k[len(prefix):]: sh[k] for k in keys if wmap[k] == f})
    from mlx.utils import tree_flatten
    module_in = {p_[:-len(".weight")]: int(a_.shape[-1]) for p_, a_ in tree_flatten(mod.parameters())
                 if p_.endswith(".weight") and a_.ndim >= 2}
    qmap = {}                                        # per-module quant from the packed shapes (as attach_mtp)
    for m_, in_f in module_in.items():
        wq, sq = w.get(f"{m_}.weight"), w.get(f"{m_}.scales")
        if wq is not None and sq is not None and wq.dtype == mx.uint32:
            qmap[m_] = {"bits": int(wq.shape[-1]) * 32 // in_f, "group_size": in_f // int(sq.shape[-1])}
    if qmap:
        nn.quantize(mod, group_size=64, bits=4, class_predicate=lambda p_, m: qmap.get(p_, False))
    mod.load_weights(list(w.items()), strict=False)
    mod.eval()
ref2.apply()
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
mx.random.seed(0)
ok_a = True
n_a = 0
for L in (3, 20, 44):
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = Q.GatedResidual(args)
        load(f"language_model.model.layers.{L}.{name}.", hc)
        mx.eval(hc.parameters())
        for t in range(4):
            sc = (0.5, 1.0, 2.0, 4.0)[t]
            h = (mx.random.normal((1, 16, 4 * 2560)) * sc).astype(mx.bfloat16)
            bo = (mx.random.normal((1, 16, 2560)) * sc).astype(mx.bfloat16)
            inj = (mx.random.normal((1, 16, 4)) * 0.5).astype(mx.bfloat16)
            mx.eval(h, bo, inj)
            for hw in (False, True):
                ref2.CFG["hc_write"] = hw
                ref2.CFG["hc_pend_mlp"] = False
                y = Q._hyper_residual_write(h, bo, inj)
                off = hc(y)
                ref2.CFG["hc_pend_mlp"] = True
                c0 = ref2.COUNTS["hc_pend"]
                y2 = Q._hyper_residual_write(h, bo, inj)
                on = hc(y2)
                ref2.CFG["hc_pend_mlp"] = False
                mx.eval(off, on)
                good = ref2.COUNTS["hc_pend"] == c0 + 1 and all(eqb(a, b) for a, b in zip(off, on))
                ok_a &= good
                n_a += 1
ref2.CFG["hc_write"] = False
say(f"(a) HC reads, layers 3 / 20 / 44 x attn / mlp x 4 scales x hc_write off / on ({n_a}): D5 == off, bitwise, fold "
    f"counted: {ok_a}")

import hc8r
class _Box(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
ok_b = True
for L in (3, 44):
    box = _Box(Q.GatedResidual(args))
    load(f"language_model.model.layers.{L}.attn_hyper_connection.", box.g)
    hc8r.hc8_install(box)
    mx.eval(box.parameters())
    ref2.CFG["hc8_fast"] = True
    for t in range(2):
        h = (mx.random.normal((1, 16, 4 * 2560)) * (1.0 + t)).astype(mx.bfloat16)
        bo = (mx.random.normal((1, 16, 2560)) * (1.0 + t)).astype(mx.bfloat16)
        inj = (mx.random.normal((1, 16, 4)) * 0.5).astype(mx.bfloat16)
        ref2.CFG["hc_pend_mlp"] = False
        off = box.g(Q._hyper_residual_write(h, bo, inj))
        ref2.CFG["hc_pend_mlp"] = True
        c0, c8 = ref2.COUNTS["hc_pend"], ref2.COUNTS["hc8_fast"]
        on = box.g(Q._hyper_residual_write(h, bo, inj))
        ref2.CFG["hc_pend_mlp"] = False
        mx.eval(off, on)
        ok_b &= ref2.COUNTS["hc_pend"] == c0 + 1 and ref2.COUNTS["hc8_fast"] > c8 and all(eqb(a, b) for a, b in zip(off, on))
    ref2.CFG["hc8_fast"] = False
say(f"(b) HC8 path (8-bit mixers, hc8_fast; layers 3 / 44): D5 == off, bitwise, fold counted: {ok_b}")

class StubMoE(nn.Module):
    def __call__(self, x):
        return (x * 0.5).astype(x.dtype)
ple = set(getattr(args, "ple_layer_ids", []) or [])
first = next(i for i in range(4, 48) if all((j + 1) not in ple for j in range(i, i + 4)))
layers = []
for i in range(first, first + 4):
    lay = Q.DecoderLayer(args, i)
    load(f"language_model.model.layers.{i}.", lay, skip=("mlp.",))
    lay.mlp = StubMoE()
    mx.eval(lay.parameters())
    layers.append(lay)
kinds = ["GDN" if l_.is_linear else "attention" for l_ in layers]
def caches():
    return [Q.QSACache(args.indexer_compress_ratio or 4) if not l_.is_linear else Q.ArraysCache(size=2) for l_ in layers]
h_in = (mx.random.normal((1, 16, 4 * 2560)) * 0.7).astype(mx.bfloat16)
mx.eval(h_in)
def stack(cs):
    h = h_in
    for l_, c in zip(layers, cs):
        h = l_(h, input_ids=None, ssm_mask=None, cache=c)
    return h
ref2.CFG["hc_pend_mlp"] = False
y_off = stack(caches()); mx.eval(y_off)
ref2.CFG["hc_pend_mlp"] = True
c0 = ref2.COUNTS["hc_pend"]
y_on = stack(caches()); mx.eval(y_on)
folds = ref2.COUNTS["hc_pend"] - c0
ref2.CFG["hc_pend_mlp"] = False
ok_c = eqb(y_on, y_off) and folds == 3
T = {False: [], True: []}
for r in range(9):
    for on in ((False, True) if r % 2 == 0 else (True, False)):
        ref2.CFG["hc_pend_mlp"] = on
        cs = [caches() for _ in range(8)]
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([stack(cc) for cc in cs]); mx.synchronize()
        T[on].append((time.perf_counter() - t0) / 8)
ref2.CFG["hc_pend_mlp"] = False
say(f"(c) 4-layer stack (layers {first}-{first + 3}: {kinds}; stub MoE): D5 == off bitwise {eqb(y_on, y_off)}, folds {folds} "
    f"(3 expected) -> {'OK' if ok_c else 'FAIL'}; ms per stack call off {1e3 * np.median(T[False]):.3f}, on "
    f"{1e3 * np.median(T[True]):.3f} ({1e3 * (np.median(T[True]) - np.median(T[False])) / 3:+.3f} ms per boundary; rough)")
st = ref2.self_test()
ok_d = all(bool(v) for v in st.values())
say(f"(d) ref2.self_test: {'PASS' if ok_d else 'FAIL ' + str(st)}")
ref2.remove()
say(f"D5 check: {'PASS' if ok_a and ok_b and ok_c and ok_d else 'FAIL'}")
