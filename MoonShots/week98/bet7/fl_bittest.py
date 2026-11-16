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
ap.add_argument("--only_T", default="", help="attn / level tests: comma-separated cache lengths")
ap.add_argument("--levels", type=int, default=21, help="level test: chained levels per run")
ap.add_argument("--driver", default="Fused", help="level test: fl_fused.Fused or Fused2")
ap.add_argument("--control", action="store_true",
                help="level test: NEGATIVE CONTROL, the HC reads' mean in a wrong (float) order; must FAIL")
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


def _bits_same(g, s):
    mx.eval(g, s)
    g, s = g.reshape(-1), s.reshape(-1)
    if g.dtype != s.dtype:
        return False
    if g.dtype in (mx.bfloat16, mx.float16):
        return bool(mx.array_equal(g.view(mx.uint16), s.view(mx.uint16)).item())
    if g.dtype == mx.float32:
        return bool(mx.array_equal(g.view(mx.uint32), s.view(mx.uint32)).item())
    return bool(mx.array_equal(g, s).item())


def test_steps():
    """S1 prep, S4 projections, S6 o_proj, S14 head, S15 tail vs the stock ops, on the head's own weights"""
    import fl_level as L
    import fl_stock as FS
    mtp, _ = FS.load_mtp()
    emb, head = FS.load_emb_head()
    from screened_head import ScreenedHead
    sh = ScreenedHead(head, FS.RANK[:32768], extra_cap=16)
    attn = mtp.layers[0].self_attn
    say("S1 / S4 / S6 / S14 / S15 vs stock, F = 1")
    fails = {}
    n = max(6, A.rows // 16)
    for trial in range(n):
        scale = (0.6, 0.05, 3.0)[trial % 3]
        wid = (mx.random.normal((1, 1, 10240), key=mx.random.key(500 + trial)) * scale).astype(mx.bfloat16)
        tok = mx.array([[int(np.random.default_rng(trial).integers(0, 248000))]], dtype=mx.int32)
        x = (mx.random.normal((1, 1, 2560), key=mx.random.key(600 + trial)) * scale).astype(mx.bfloat16)
        g6 = (mx.random.normal((1, 1, 6144), key=mx.random.key(650 + trial)) * scale).astype(mx.bfloat16)
        sc = (mx.random.normal((1,), key=mx.random.key(670 + trial)) * 5).astype(mx.float32)
        chk = {}
        chk["S1 prep h"] = (L.s1_prep(mtp, emb, wid, tok), mtp._prepare_inputs_eager(wid, emb(tok)))
        q, k, v, ix = L.s4_proj(attn, x)
        chk["S4 q"] = (q, attn.q_proj(x))
        chk["S4 k"] = (k, attn.k_proj(x))
        chk["S4 v"] = (v, attn.v_proj(x))
        chk["S4 indexer qk"] = (ix, attn.indexer.index_qk_proj(x))
        chk["S6 o_proj"] = (L.s6_oproj(attn, g6), attn.o_proj(g6))
        lg_f = L.s14_head(sh, x)
        lg_s = sh.logits(x)                                        # [1, nh] fp32
        chk["S14 head logits (fp32)"] = (lg_f.astype(mx.float32), lg_s)
        lp = lg_s - mx.logsumexp(lg_s, axis=-1, keepdims=True)
        nh = lg_s.shape[-1]
        kidx = mx.argpartition(lp, kth=nh - 8, axis=-1)[:, nh - 8:]
        klp = mx.take_along_axis(lp, kidx, axis=-1)
        cand = (sc[:, None] + klp).reshape(-1)
        sel = mx.argpartition(cand, kth=7)[7:]
        tk = mx.take(sh.ids, mx.take(kidx.reshape(-1), sel))
        o = L.s15_tail(lg_f, sc, sh.ids, K=8)
        chk["S15 kidx"] = (o[0], kidx.astype(mx.uint32))
        chk["S15 klp"] = (o[1], klp)
        chk["S15 cand"] = (o[2], cand)
        chk["S15 sel"] = (o[3], sel.astype(mx.uint32))
        chk["S15 path score"] = (o[4], mx.take(cand, sel))
        chk["S15 token"] = (o[5], tk.astype(mx.int32))
        for name, (g, s) in chk.items():
            if not _bits_same(g, s):
                fails[name] = fails.get(name, 0) + 1
                if fails[name] <= 2:
                    if g.dtype in (mx.bfloat16, mx.float32) and g.size == s.size and g.size > 8:
                        compare(f"trial {trial} {name}", g.reshape(-1).astype(s.dtype), s.reshape(-1))
                    else:
                        say(f"  trial {trial} {name}: fused {g.reshape(-1).tolist()[:8]} stock {s.reshape(-1).tolist()[:8]}")
    for name in ["S1 prep h", "S4 q", "S4 k", "S4 v", "S4 indexer qk", "S6 o_proj", "S14 head logits (fp32)",
                 "S15 kidx", "S15 klp", "S15 cand", "S15 sel", "S15 path score", "S15 token"]:
        say(f"  {name:28s} {'PASS' if name not in fails else 'FAIL'} ({n - fails.get(name, 0)} / {n} trials bit-equal)")
    RES["steps"] = dict(trials=n, fails=fails)
    return not fails


def test_attn():
    """S5 (fused attention) vs the head's own Attention module on a real QSACache: stock attn(x, cache) (decode,
    S = 1, dense) against o_proj(fused gated output), and the cache's new key row against the fused rotated key"""
    import fl_attn
    import fl_level as L
    import fl_stock as FS
    Q = FS.Q
    mtp, _ = FS.load_mtp()
    attn = mtp.layers[0].self_attn
    say("S5 attention (fused) vs Attention.__call__ on a QSACache, F = 1")
    ok = True
    for T0 in [int(t) for t in (A.only_T.split(",") if A.only_T else ("400", "1000"))]:
        cache = Q.QSACache(4)
        xs = (mx.random.normal((1, T0, 2560), key=mx.random.key(T0)) * 0.7).astype(mx.bfloat16)
        mx.eval(attn(xs, cache))                                   # prefill the cache with T0 rows
        bad = 0
        for trial in range(A.rows // 64 + 2):
            x = (mx.random.normal((1, 1, 2560), key=mx.random.key(T0 + 77 + trial)) * 0.7).astype(mx.bfloat16)
            kb, vb = cache.kv.keys, cache.kv.values                  # the backing before this step's write
            mx.eval(kb, vb)
            T = int(cache.offset)
            qf, kf, vf, _ = L.s4_proj(attn, x)
            gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)
            fused_y = L.s6_oproj(attn, gated)
            stock_y = attn(x, cache).reshape(-1)                     # writes row T into the cache
            new_k = cache.kv.keys[0, :, T, :].reshape(-1)
            mx.eval(fused_y, stock_y, new_k, krot)
            s1 = _bits_same(fused_y, stock_y)
            s2 = _bits_same(krot, new_k)
            if not (s1 and s2):
                bad += 1
                if bad <= 2:
                    compare(f"T {T} attention output (after o_proj)", fused_y, stock_y)
                    compare(f"T {T} rotated new key", krot, new_k)
            cache.trim(1)
        say(f"  T0 {T0}: {'PASS' if not bad else 'FAIL'} ({A.rows // 64 + 2 - bad} / {A.rows // 64 + 2} steps bit-equal)")
        ok &= not bad
    return ok


def test_level():
    """deliverable (2): the whole fused level vs the stock level, L chained levels from the same cache state:
    every array a level hands on (h / hh, token, path score, top-K ids and log-probs, candidates, frontier pick)
    and every cache row the L levels wrote (KV keys / values, the indexer's raw keys and pooled blocks)"""
    import fl_fused
    import fl_stock as FS
    L = A.levels
    ok = True
    st = None
    for T0 in [int(t) for t in (A.only_T.split(",") if A.only_T else ("400", "1010", "2000"))]:
        if st is None:
            st = FS.Stock(nh=32768, T0=T0)
        else:                                                       # re-prefill the same object's cache at T0
            st.cache = [FS.Q.QSACache(FS.args.indexer_compress_ratio or 4)]
            rng = np.random.default_rng(5)
            Wh = mx.array((rng.standard_normal((1, T0, FS.HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
            ids0 = mx.array(rng.integers(0, 200000, (1, T0)).astype(np.int32))
            mx.eval(st.mtp.fuse_and_run(Wh, st.emb(ids0), st.cache))
        fu = getattr(fl_fused, A.driver)(st)
        if A.control:
            import fl_hc
            fl_hc.DEFAULT_MEAN_MODE = 1
        c = st.cache[0]

        def snap():
            T1 = int(c.offset)
            arrs = [c.kv.keys[..., T0:T1, :], c.kv.values[..., T0:T1, :], c.raw_keys[:, T0:T1, :]]
            nb = T1 // c.ratio
            arrs.append(c.pooled[:, T0 // c.ratio:nb, :] if c.pooled is not None else mx.zeros((1,), mx.bfloat16))
            # host copies (raw bits): the fused run may write the same cache buffers in place afterwards
            return [np.array(mx.contiguous(a).view(mx.uint16)).copy() for a in arrs]
        for tok0 in (1000, 77777):
            o_s = st.block(L, tok0=tok0, keep=True)
            cs = snap()
            c.trim(L)
            o_f = fu.block(L, tok0=tok0, keep=True)
            cf = snap()
            c.trim(L)
            bad = {}
            for lv in range(L):
                for key in ("hh", "tok", "sc", "kidx", "klp", "cand", "sel"):
                    a, b = o_f[lv][key], o_s[lv][key]
                    if key in ("kidx", "sel", "tok"):
                        a, b = a.astype(mx.int64), b.astype(mx.int64)
                    if not _bits_same(a.reshape(-1), b.reshape(-1)):
                        bad.setdefault(key, []).append(lv)
            for j, nm in enumerate(("KV keys", "KV values", "indexer raw keys", "pooled blocks")):
                if cf[j].shape != cs[j].shape or not np.array_equal(cf[j], cs[j]):
                    bad.setdefault(nm, []).append(-1)
            toks = [int(o["tok"].reshape(-1)[0].item()) for o in o_s]
            tag = f"T0 {T0} (keys {T0 + 1}..{T0 + L}) tok0 {tok0}: {L} levels"
            if bad:
                ok = False
                say(f"  {tag}: FAIL, first differing level per array {({k: v[:3] for k, v in bad.items()})}")
            else:
                say(f"  {tag}: PASS, every array and cache row bit-equal (stock tokens {toks[:6]}...)")
            RES.setdefault("level", []).append(dict(T0=T0, tok0=tok0, L=L, bad={k: v for k, v in bad.items()}))
    return ok


TESTS = {"rms": test_rms, "ew": test_ew, "hc": test_hc, "moe": test_moe, "steps": test_steps, "attn": test_attn,
         "level": test_level}
t0 = time.time()
allok = True
for name, fn in TESTS.items():
    if A.only and name not in A.only.split(","):
        continue
    allok &= fn()
say(f"ALL {'PASS' if allok else 'FAIL'} ({time.time() - t0:.1f} s)")
json.dump(RES, open(A.out, "w"), indent=1)
