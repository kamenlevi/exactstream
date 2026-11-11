"""Unit tests of the screen 7 harness patch (Breakthrough, Wed 2026-11-04 14:5x). Run from the PATCHED folder so the
patched dm_t3 / dm_common / dm_ablations shadow the installed ones:
  cd <patched dir> && PYTHONPATH=<patched dir>:~/FableGod/decision_map DM_OUT=<scratch out> python test_dm_s7.py [cpu|gpu|all]
cpu (no GPU): A1 the wiring: at PERT['t3L'] layers the gated rows' kept experts (ranks 0-1 forced + the mask's 2-9) go to
     routed_kept and the stock path posts nothing for them; other layers stay tbits_mask's (routed_topk K 2, mask ranks
     2-9 on the stock path); A2 the arms resolve and their setups set PERT as specified; A3 the TRL gate refuses a partial
     store and counts it with DM_TRL_FALLBACK=1.
gpu (seconds, a few experts): B1 / B2 routed_kept(src 3 / 2) == routed_topk(bits 3 / 2, K 10, keep) bit for bit;
     B3 routed_kept('trl:mix') vs a dense fp32 reference built from the view (mx.dequantize bits 8); B4 the fallback path
     (missing experts from records3 MSB-3) vs the same reference; B5 the result does not depend on the read group size."""
import os, sys, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = os.getcwd()
assert os.path.exists(os.path.join(HERE, "dm_t3.py")), "run from the patched folder"
import dm_t3 as T
import dm_common as C
import dm_ablations as A
assert T.__file__.startswith(HERE) and C.__file__.startswith(HERE) and A.__file__.startswith(HERE), (T.__file__, C.__file__, A.__file__)
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
bad = []

def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok: bad.append(msg)

class Blk:                                                    # the block surface routed_topk / routed_kept use
    def __init__(s, wg=None):
        s.gate = nn.Linear(2560, 512, bias=False); s.top_k = 10; s.norm_topk_prob = True
        if wg is not None: s.gate.weight = wg
        s.switch_mlp = type("S", (), {"activation": staticmethod(lambda up, gt: nn.silu(gt) * up)})()

M7 = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c239_t3")
TV = T.trl_view()

if MODE in ("cpu", "all"):
    mx.set_default_device(mx.cpu)
    # ---- A1: the wiring on a real window of screen 7's masks ----
    nm = sorted(f for f in os.listdir(M7) if f.endswith(".npz"))[0]
    rm = np.load(os.path.join(M7, nm))["keep"]; nrow = rm.shape[0]
    kk = np.full(nrow, 10, np.int32); kk[~rm[:, :, 0].any(axis=1)] = 0          # measure()'s t3mask rule
    calls = []
    real_kept, real_topk = T.routed_kept, T.routed_topk
    def fake_kept(block, L, xg, src, keep): calls.append(("kept", L, int(xg.shape[0]), src, np.array(keep))); return mx.zeros(xg.shape, xg.dtype)
    def fake_topk(block, L, xg, bits=3, K=2, keep=None): calls.append(("topk", L, int(xg.shape[0]), bits, K, keep)); return mx.zeros(xg.shape, xg.dtype)
    T.routed_kept, T.routed_topk = fake_kept, fake_topk
    blk = Blk()
    for setup_name, S in (("s7_trl", {7: "trl:mix", 18: "trl:mix", 30: "trl:mix", 41: "trl:mix"}), ("s7_cal", {L: 2 for L in (7, 18, 30, 41)})):
        C.clear(); C.PERT.update(rmask=M7, gate=("t3mask",), t3=3, t3k=2, renorm=False, t3L=dict(S))
        C.CUR["k"] = kk; C.CUR["rm"] = rm; C.CUR["kpos"] = {}
        p0, n = 150, 160                                       # one call's rows p0 .. p0 + n - 1 (gated and not)
        G = np.flatnonzero(kk[p0:p0 + n] == 0); NG = np.flatnonzero(kk[p0:p0 + n] != 0)
        for L in (6, 7, 18, 19):
            C.CUR["L"] = L; C.CUR["kpos"][L] = p0; calls.clear()
            x = mx.zeros((n, 2560), mx.bfloat16)
            add = C._t3_add(blk, L, x)
            sc = mx.array(np.random.default_rng(L).random((n, 10)).astype(np.float32))
            out = np.array(C._pert_scores(sc)); s_np = np.array(sc)
            rank = np.argsort(np.argsort(-s_np, axis=1), axis=1)                  # each score's rank
            kept_stock = np.take_along_axis(out != 0, np.argsort(rank, axis=1), axis=1)   # rank-sorted stock kept flags
            if L in S:
                c = calls[0] if calls else None
                ok = (len(calls) == 1 and c[0] == "kept" and c[1] == L and c[2] == len(G) and c[3] == S[L])
                want = rm[p0 + G, L].copy(); want[:, :2] = True
                ok = ok and np.array_equal(c[4], want)
                check(ok, f"A1 {setup_name} L{L}: routed_kept on the {len(G)} gated rows, src {S[L]!r}, keep = ranks 0-1 + the mask's 2-9 "
                          f"(kept pairs {int(want.sum())}, {want.sum() / max(len(G), 1):.2f} per row)")
                check(not kept_stock[G].any() and np.array_equal(kept_stock[NG], rm[p0 + NG, L]),
                      f"A1 {setup_name} L{L}: the stock path posts nothing for the gated rows; ungated rows keep their mask")
            else:
                c = calls[0] if calls else None
                ok = len(calls) == 1 and c[0] == "topk" and c[1] == L and c[2] == len(G) and c[3] == 3 and c[4] == 2 and c[5] is None
                check(ok, f"A1 {setup_name} L{L}: tbits_mask form: routed_topk bits 3, K 2, keep None on the {len(G)} gated rows")
                check(np.array_equal(kept_stock[G], rm[p0 + G, L]) and not kept_stock[G][:, :2].any(),
                      f"A1 {setup_name} L{L}: the stock path posts the mask's ranks 2-9 for the gated rows (ranks 0-1 off)")
            check(C.CUR["kpos"][L] == p0 + n, f"A1 {setup_name} L{L}: the row counter advanced by the call ({C.CUR['kpos'][L] - p0})")
    T.routed_kept, T.routed_topk = real_kept, real_topk
    C.clear()
    # ---- A2: the arms resolve and set PERT as specified ----
    for name in ("qom_0_live_ball_c147_t3", "s7_ref", "s7_trl", "s7_cal", "s7_cal_all"):
        check(name in A.ABLATIONS and A.ablation(name) is A.ABLATIONS[name], f"A2 {name} resolves")
    os.environ["DM_TRL_FALLBACK"] = "1"
    exp = {"qom_0_live_ball_c147_t3": (os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c147_t3"), None),
           "s7_ref": (M7, None), "s7_trl": (M7, {L: "trl:mix" for L in (7, 18, 30, 41)}),
           "s7_cal": (M7, {L: 2 for L in (7, 18, 30, 41)}), "s7_cal_all": (M7, {L: 2 for L in range(48)})}
    for name, (path, t3L) in exp.items():
        C.clear(); A.ABLATIONS[name]()
        P = dict(C.PERT)
        ok = (P.get("rmask") == path and P.get("gate") == ("t3mask",) and P.get("t3") == 3 and P.get("t3k") == 2
              and P.get("renorm") is False and P.get("t3L") == t3L and os.path.isdir(path) and os.path.exists(os.path.join(path, "READY")))
        check(ok, f"A2 {name}: rmask {os.path.basename(path)} (READY), t3mask, t3 3, t3k 2, t3L {('none' if t3L is None else str(len(t3L)) + ' layers ' + repr(sorted(set(t3L.values()), key=str)))}")
    C.clear()
    # ---- A3: the TRL gate ----
    os.environ.pop("DM_TRL_FALLBACK", None)
    cov = {L: TV.coverage(L, range(512), "mix")[0] for L in (7, 18, 30, 41)}
    print(f"     TRL store coverage now (mix): {cov}", flush=True)
    if min(cov.values()) < 512:
        try:
            C.clear(); A.ABLATIONS["s7_trl"](); refused = False
        except RuntimeError as e:
            refused = "missing or partial" in str(e); print("     ->", str(e)[:200], flush=True)
        check(refused, "A3 s7_trl refuses a partial store (no DM_TRL_FALLBACK)")
        os.environ["DM_TRL_FALLBACK"] = "1"; C.clear(); A.ABLATIONS["s7_trl"](); check(True, "A3 with DM_TRL_FALLBACK=1 it runs, counting the fallback")
        os.environ.pop("DM_TRL_FALLBACK", None)
    else:
        C.clear(); A.ABLATIONS["s7_trl"](); check(True, "A3 the store is complete: s7_trl's gate passes")
    C.clear()

if MODE in ("gpu", "all"):
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7)
    # ---- B1 / B2: routed_kept(int src) == routed_topk(bits, K 10, keep), bit for bit ----
    blk = Blk(); mx.eval(blk.gate.parameters())
    xg = mx.array(rng.standard_normal((24, 2560)).astype(np.float32) * 0.5).astype(mx.bfloat16)
    kp = rng.random((24, 10)) < 0.6; kp[:, :2] = True; kp[3] = False; kp[3, :2] = True      # mixed masks, one row ranks 0-1 only
    for b in (3, 2):
        y0 = T.routed_topk(blk, 12, xg, bits=b, K=10, keep=kp); y1 = T.routed_kept(blk, 12, xg, b, kp); mx.eval(y0, y1)
        d = float(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32)).max())
        check(mx.array_equal(y0, y1).item(), f"B{1 if b == 3 else 2} routed_kept(src {b}) == routed_topk(bits {b}, K 10, keep) on 24 rows, layer 12 (max diff {d:.1e})")
    # ---- B3 / B4: the TRL view vs a dense fp32 reference (mx.dequantize bits 8); the fallback path vs MSB-3 dequantized ----
    L = 18
    os.environ["DM_TRL_FALLBACK"] = "1"; T.trl_gate([L], "mix"); os.environ.pop("DM_TRL_FALLBACK", None)
    have = T.TRL_OK[(L, "mix")].copy()                          # the experts whose files are whole (trl_gate)
    enc = np.flatnonzero(have); mis = np.flatnonzero(~have)
    print(f"     layer {L}: {len(enc)} experts encoded so far", flush=True)
    assert len(enc) >= 12, "need >= 12 encoded experts at layer 18"
    def ref_out(blk, xg, keep, src_of):
        """Dense reference from fp32-dequantized weights, rounded to bf16 where the live kernels round (each projection's
        output, the SwiGLU product, the block's bf16 score normalisation), so the remaining gap is ~1 bf16 ulp."""
        gates = mx.softmax(blk.gate(xg), axis=-1, precise=True)            # the routing by the block's own ops (dm_t3's),
        inds = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]          # so bf16 score ties rank the same way and the
        s_ = mx.take_along_axis(gates, inds, axis=-1); s_ = s_ / s_.sum(axis=-1, keepdims=True)   # test isolates the
        o = mx.argsort(-s_.astype(mx.float32), axis=-1)[:, :10]             # weight source and the pair plumbing
        top = np.array(mx.take_along_axis(inds, o, axis=-1)); scs = np.array(mx.take_along_axis(s_, o, axis=-1).astype(mx.float32))
        out = []; bf = lambda a: a.astype(mx.bfloat16)
        for r in range(xg.shape[0]):
            sc = scs[r]; acc = mx.zeros((1, 2560), mx.float32); h = xg[r:r + 1].astype(mx.float32)
            for j in range(10):
                if not keep[r, j]: continue
                Wd = src_of(int(top[r, j]))
                a = nn.silu(bf(h @ Wd["gate_proj"].T)) * bf(h @ Wd["up_proj"].T)               # bf16, as the live activation
                acc = acc + float(sc[j]) * bf(a.astype(mx.float32) @ Wd["down_proj"].T).astype(mx.float32)
            out.append(acc)
        return mx.concatenate(out), top
    def trl_dense(e):
        W = TV.read(L, [e], "mix")
        return {p: mx.dequantize(W[p]["w"][0], W[p]["s"][0].astype(mx.float32), W[p]["b"][0].astype(mx.float32), group_size=64, bits=8) for p in W}
    def msb3_dense(e):
        W = T.read(L, np.array([e]))
        return {p: mx.dequantize(W[p]["w3"][0], W[p]["s3"][0], W[p]["b3"][0], group_size=64, bits=3).astype(mx.float32) for p in W}
    # Tie-free mock router: the picked experts' logits ~ +24 +- 4, every other expert's ~ -24 +- 4 (no zero-logit ties).
    u = rng.standard_normal(2560).astype(np.float32); u /= np.linalg.norm(u)
    def router(pick):
        wg = rng.standard_normal((512, 2560)).astype(np.float32) * 0.06 - 3 * u; wg[pick] += 6 * u
        return Blk(mx.array(wg).astype(mx.bfloat16))
    xg = mx.array((rng.standard_normal((12, 2560)) + 8 * u).astype(np.float32)).astype(mx.bfloat16)
    kp = rng.random((12, 10)) < 0.7; kp[:, :2] = True
    def errs(y, ref):
        d = y.astype(mx.float32) - ref
        return float(mx.sqrt((d ** 2).sum() / (ref ** 2).sum())), float(mx.abs(d).max() / mx.abs(ref).max())
    blk3 = router(enc)                                            # routes onto encoded experts only
    yt = T.routed_topk(blk3, L, xg, bits=3, K=10, keep=kp); ref, top = ref_out(blk3, xg, kp, msb3_dense)
    r0, m0 = errs(yt, ref)
    check(r0 < 3e-2, f"B0 the metric's floor on the live path: routed_topk bits 3 vs the bf16-rounded dense MSB-3 reference: rms rel {r0:.2e}, max rel {m0:.2e} (test_t3's 3-bit tolerance 3e-2)")
    s0 = dict(T.LSRC); y = T.routed_kept(blk3, L, xg, "trl:mix", kp); mx.eval(y)
    ref, top = ref_out(blk3, xg, kp, trl_dense)
    assert have[top].all(), "the mock gate routed onto a missing expert"
    r, m = errs(y, ref); dl = {k: T.LSRC[k] - s0[k] for k in s0}
    check(r < 1e-2 and m < 5e-2 and dl["fallback"] == 0 and dl["trl"] == len(np.unique(top[kp])) and dl["pairs"] == int(kp.sum()),
          f"B3 routed_kept('trl:mix') vs dense fp32 (dequantize bits 8) on 12 rows: rms rel {r:.2e}, max rel {m:.2e}; counts {dl}")
    nxt = {int(e): int(enc[(i + 1) % len(enc)]) for i, e in enumerate(enc)}           # negative control: each expert swapped
    rn, mn = errs(y, ref_out(blk3, xg, kp, lambda e: trl_dense(nxt[e]))[0])
    check(rn > 0.3, f"B3n negative control (each expert's weights swapped for another's): rms rel {rn:.2e}, so the metric separates a wrong expert from rounding")
    if len(mis):
        os.environ["DM_TRL_FALLBACK"] = "1"
        blk4 = router(np.r_[enc[:6], mis[:6]])
        s0 = dict(T.LSRC); y = T.routed_kept(blk4, L, xg, "trl:mix", kp); mx.eval(y)
        ref, top = ref_out(blk4, xg, kp, lambda e: trl_dense(e) if have[e] else msb3_dense(e))
        r, m = errs(y, ref); dl = {k: T.LSRC[k] - s0[k] for k in s0}; used = np.unique(top[kp])
        check(r < 3e-2 and dl["fallback"] == int((~have[used]).sum()) and dl["trl"] == int(have[used].sum())
              and dl["fallback"] > 0 and dl["trl"] > 0,
              f"B4 fallback: missing experts from records3 MSB-3, the rest TRL: rms rel {r:.2e}, max rel {m:.2e}; counts {dl}")
        os.environ.pop("DM_TRL_FALLBACK", None)
    # ---- B5: the read group size does not change the result ----
    g0 = T.LSRC_GROUP; y_a = T.routed_kept(blk3, L, xg, "trl:mix", kp); T.LSRC_GROUP = 3; y_b = T.routed_kept(blk3, L, xg, "trl:mix", kp)
    y_c = T.routed_kept(blk, 12, xg, 2, kp); T.LSRC_GROUP = g0; y_d = T.routed_kept(blk, 12, xg, 2, kp); mx.eval(y_a, y_b, y_c, y_d)
    check(mx.array_equal(y_a, y_b).item() and mx.array_equal(y_c, y_d).item(), "B5 group size 3 vs 128: bit-equal (TRL and MSB-2)")

print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
