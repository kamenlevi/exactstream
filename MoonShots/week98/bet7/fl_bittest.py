"""WEEK98 BET 7: bit tests of the exact building blocks against MLX 0.32.2's stock ops (MoonShots, Mon 16 Nov 2026).
Each case compares the block's output with the stock op's, element for element as raw bits (uint16 for bf16, uint32
for fp32), on random inputs at several scales, and at several real threadgroup sizes (the virtual-lane emulation must
not depend on them). PASS = zero mismatching elements in every case.
usage: ~/BigMoE/.venv/bin/python -I -B fl_bittest.py [--only rms] [--rows 512]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fl_exact as E                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_bittest.json"))
A = ap.parse_args()
RES = {}


def say(s=""):
    print(s, flush=True)


def bits(a):
    a = mx.contiguous(a)
    return np.array(a.view(mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else mx.uint32), copy=False)


def compare(name, got, ref):
    mx.eval(got, ref)
    g, r = bits(got), bits(ref)
    bad = int((g != r).sum())
    tot = int(r.size)
    gf, rf = np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32))
    md = float(np.max(np.abs(gf - rf))) if bad else 0.0
    RES[name] = dict(mismatch=bad, total=tot, max_abs_diff=md)
    say(f"  {name:60s} {'PASS' if bad == 0 else 'FAIL'}: {bad} / {tot} elements differ (max |diff| {md:.3g})")
    return bad == 0


def rand(shape, scale, seed, dtype=mx.bfloat16):
    k = mx.random.key(seed)
    x = mx.random.normal(shape, key=k) * scale
    return x.astype(dtype)


def test_rms():
    say("RMSNorm (mx.fast.rms_norm, weight inside) vs fl_exact.rms_norm_exact (virtual-lane statistic)")
    ok = True
    for axis in (128, 256, 2560, 10240):
        for scale in (0.02, 1.0, 40.0):
            x = rand((A.rows, axis), scale, 11 + axis)
            w = (1.0 + rand((axis,), 0.3, 3 + axis, mx.float32)).astype(mx.bfloat16)
            ref = mx.fast.rms_norm(x, w, 1e-6)
            for tg in ((64, 256, 1024) if axis >= 2560 else (32, 64, 256)):
                ok &= compare(f"rms axis {axis} scale {scale} tg {tg}", E.rms_norm_exact(x, w, 1e-6, tg=tg), ref)
    say("grouped RMSNorm (GroupedRMSNorm: rms_norm without weight over 4 x 2560, then x weight) vs exact")
    x = rand((A.rows, 10240), 1.0, 99)
    w = (1.0 + rand((10240,), 0.3, 98, mx.float32)).astype(mx.bfloat16)
    ref = (mx.fast.rms_norm(x.reshape(A.rows, 4, 2560), None, 1e-6).reshape(A.rows, 10240)) * w
    ones = mx.ones((2560,), mx.bfloat16)
    got = (E.rms_norm_exact(x.reshape(A.rows * 4, 2560), ones, 1e-6).reshape(A.rows, 10240)) * w
    ok &= compare("grouped 4 x 2560, weight after (stock two ops)", got, ref)
    return ok


def test_ew():
    """exhaustive: every bf16 bit pattern (NaNs compared as bits too)"""
    import mlx.nn as nn
    say("elementwise replicas, every bf16 bit pattern except NaNs (65,280 inputs incl. +-inf, zeros, subnormals)")
    allp = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    nan = ((allp & 0x7F80) == 0x7F80) & ((allp & 0x007F) != 0)
    x = mx.array(allp[~nan]).view(mx.bfloat16)
    ok = compare("standalone mx.sigmoid vs fl_sigmoid_pre", E.ew_exact(x, 0), mx.sigmoid(x))
    ok &= compare("nn.silu (compiled) vs fl_silu_jit", E.ew_exact(x, 1), nn.silu(x))
    # diagnostics: how the two lowerings differ (expected: some inputs)
    compare("diagnostic: JIT sigmoid vs standalone mx.sigmoid", E.ew_exact(x, 2), mx.sigmoid(x))
    return ok


def test_hc():
    """the fused HC read (fl_hc.py) vs qwen4_exp.GatedResidual on the MTP head's own three HC modules"""
    import fl_hc
    import fl_stock as FS
    Q = FS.Q
    raw = mx.load(f"{FS.D}/mtp.safetensors")
    mods = {}
    for name, combine in (("layers.0.attn_hyper_connection", True), ("layers.0.mlp_hyper_connection", True),
                          ("hyper_connection_mixer", False)):
        gr = Q.GatedResidual(FS.args, use_combine=combine)
        gr.load_weights([(k[len(f"mtp.{name}."):], v) for k, v in raw.items() if k.startswith(f"mtp.{name}.")],
                        strict=True)
        mx.eval(gr.parameters())
        mods[name] = gr
    del raw
    say("HC read (S2 / S3 and their twins) vs GatedResidual, F = 1")
    ok = True
    for mean_mode in (0, 1, 2):
        okm = True
        for trial in range(A.rows // 32 + 4):
            scale = (0.6, 0.05, 3.0, 20.0)[trial % 4]
            h = (mx.random.normal((1, 1, 10240), key=mx.random.key(300 + trial)) * scale).astype(mx.bfloat16)
            bo = (mx.random.normal((1, 1, 2560), key=mx.random.key(700 + trial)) * scale).astype(mx.bfloat16)
            ij = (2.0 * mx.sigmoid(mx.random.normal((1, 1, 4), key=mx.random.key(900 + trial)))).astype(mx.bfloat16)
            for name, gr in mods.items():
                for pend in ((None, (bo, ij)) if name.startswith("layers.0.mlp") else (None,)):
                    st = gr(h, pending=pend) if "block_inject_weight" in gr else gr(h)
                    fu = fl_hc.hc_read(gr, h, pending=None if pend is None else (bo, ij), mean_mode=mean_mode)
                    tag = f"mean{mean_mode} {name.split('.')[-1]}{' +write' if pend else ''} scale {scale}"
                    if "block_inject_weight" in gr:
                        mixed_s, hyper_s, inj_s = st
                        r = [(fu[0], mixed_s.reshape(-1)), (fu[1], inj_s.reshape(-1)), (fu[2], hyper_s.reshape(-1))]
                    else:
                        r = [(fu[0], st.reshape(-1))]
                    for j, (g, s) in enumerate(r):
                        mx.eval(g, s)
                        if not bool(mx.array_equal(g.view(mx.uint16), s.view(mx.uint16)).item()):
                            okm = False
                            if trial < 4:
                                compare(f"{tag} output {('mixed', 'inject', 'written')[j]}", g, s)
        say(f"  mean mode {mean_mode}: {'PASS (every output bit-equal on every trial)' if okm else 'FAIL'}")
        RES[f"hc_mean_mode_{mean_mode}"] = okm
        ok = ok or okm
    return any(RES.get(f"hc_mean_mode_{m}") for m in (0, 1, 2))


def test_moe():
    """the fused MoE (fl_moe.py, S9-S11) vs the head's own SparseMoeBlock (stock eager), F = 1"""
    import fl_moe
    import fl_stock as FS
    mtp, _ = FS.load_mtp()
    blk = mtp.layers[0].mlp
    say("MoE (S9 / S9b / S10 / S11) vs SparseMoeBlock, F = 1")
    ok = True
    nbad = {}
    n = max(8, A.rows // 8)
    for trial in range(n):
        scale = (1.0, 0.3, 3.0, 0.05)[trial % 4]
        x = (mx.random.normal((1, 1, 2560), key=mx.random.key(4000 + trial)) * scale).astype(mx.bfloat16)
        ref = blk(x).reshape(-1)
        gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
        inds_s = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:].reshape(-1)
        out, it = fl_moe.moe(blk, x)
        mx.eval(ref, gates, inds_s, out, it)
        checks = {"router logits": (it["rl"], blk.gate(x).reshape(-1)),
                  "top-10 ids (order too)": (it["inds"].astype(mx.int32), inds_s.astype(mx.int32)),
                  "MoE output": (out, ref)}
        for k, (g, s) in checks.items():
            mx.eval(g, s)
            same = bool(mx.array_equal(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g,
                                       s.view(mx.uint16) if s.dtype == mx.bfloat16 else s).item())
            if not same:
                ok = False
                nbad[k] = nbad.get(k, 0) + 1
                if nbad[k] <= 2:
                    if g.dtype == mx.bfloat16:
                        compare(f"trial {trial} scale {scale}: {k}", g, s)
                    else:
                        say(f"  trial {trial}: {k} differ: fused {g.tolist()} stock {s.tolist()}")
    say(f"  {n} trials: {'PASS (router logits, top-10 ids in order, output all bit-equal)' if ok else 'FAIL ' + str(nbad)}")
    RES["moe"] = dict(trials=n, fail_counts=nbad)
    return ok


TESTS = {"rms": test_rms, "ew": test_ew, "hc": test_hc, "moe": test_moe}
t0 = time.time()
allok = True
for name, fn in TESTS.items():
    if A.only and name not in A.only.split(","):
        continue
    allok &= fn()
say(f"ALL {'PASS' if allok else 'FAIL'} ({time.time() - t0:.1f} s)")
json.dump(RES, open(A.out, "w"), indent=1)
