"""DRAFT LINE piece 1: SYNC-FREE best-first tree drafting (MoonShots, Tue 17 Nov 2026; the orchestrator 09:4x via
NewIdeas: BigMoE's DEFAULT for the r57 t30b tree-16 loop if this bit test passes on BigMoE's own ops by Wed evening).

dd_tree.py's per-anchor tree (t30b drafts the N highest path-probability nodes, top-K children per node, best-first by
path probability in heapq's (-path_p, push count) order) at B = 1 (one anchor, as an engine drafts), in two forms that
use the SAME MLX ops for every state, logit and probability:
  HS  host-synced: the frontier on the CPU (dd_tree.Forest: float64 path products, heapq) -> one sync per pop;
  SF  sync-free: the frontier on the GPU. One single-thread Metal kernel per pop pushes the popped node's K children
      with their float64 path products computed EXACTLY in integer arithmetic (IEEE round-to-nearest-even, as Python's
      float does), then pops the max by (path_p, earliest push) = heapq's order, ties included. The parent's state is
      gathered by a GPU index; the ancestor rows have a static width per pop (t + 1, padding masked the way tree_attn
      already masks); mx.async_eval per pop, one sync per tree.
Top-K on the GPU in both: argpartition + take (dd_topk.topk_rows' own ops), a stable descending sort of the K (a tiny
kernel; = numpy's stable argsort of -tv), tp = mx.exp(tv - lse). Not numpy's exp: numpy's float32 exp and MLX's GPU exp
(metal::precise::exp) differ in ~1/3 of values by 1-2 ulp [K, 20M-value probe, Tue 09:5x], so NO GPU loop can match a
CPU-exp loop bitwise. REF (dd_tree's own topk_rows, numpy exp, + Forest) runs beside them to count what moving exp to
the GPU changes (trees that differ; top_p ulp gaps).
Line (the bit test): SF == HS bitwise on every node of every tree: parent, token, depth, path_p (float64 bits), top-K
ids and probabilities, the state h. Timing (INDICATIVE under a share lock): ms per pop, HS vs SF vs REF.
usage: ~/BigMoE/.venv/bin/python -I -B sf_tree.py --selftest                (synthetic, seconds, < 1 GB)
       ~/BigMoE/.venv/bin/python -I -B sf_tree.py [--set heplus24|mbppbank] [--windows 0,1] [--nodes 16] [--cut none|Q8H32]
       [--max-anchors 48] [--out <json>]     (MEM ~11-12 GB: share lock with MEM, after `peer ls`)"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/BigMoE/scripts"))                  # read-only imports (run with -I -B: no bytecode written there)
sys.path.insert(0, H("~/MoonShots/week98"))
sys.path.insert(0, H("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402

PAD = -1

HDR = """
// IEEE float64 a (positive, normal) times float32 b (positive, normal), round to nearest even: exactly Python's
// float(a) * float(b). Integer arithmetic only (no float64 on the GPU). bad = 1 on anything outside that domain.
METAL_FUNC ulong f64_mul_f32(ulong a, float b, thread int& bad) {
  uint bb = as_type<uint>(b);
  int ea = int((a >> 52) & 0x7FFul);
  int eb = int((bb >> 23) & 0xFFu);
  if (ea == 0 || ea == 0x7FF || eb == 0 || eb == 0xFF || (a >> 63) != 0ul || (bb >> 31) != 0u) { bad = 1; return 0ul; }
  ulong ma = (a & 0xFFFFFFFFFFFFFul) | (1ul << 52);       // 53 bits
  ulong mb = ulong(bb & 0x7FFFFFu) | (1ul << 23);          // 24 bits
  ulong ml = ma & 0xFFFFFFFFul, mh = ma >> 32;
  ulong p1 = ml * mb;                                      // < 2^56
  ulong p2 = mh * mb;                                      // < 2^45
  ulong lo = p1 + (p2 << 32);
  ulong hi = (p2 >> 32) + ((lo < p1) ? 1ul : 0ul);         // the product hi:lo < 2^77
  int top = ((hi >> 12) & 1ul) != 0ul ? 76 : 75;           // its leading bit
  int s = top - 52;                                        // 23 or 24 bits shifted out
  ulong m = (hi << (64 - s)) | (lo >> s);
  ulong rem = lo & ((1ul << s) - 1ul);
  ulong hwy = 1ul << (s - 1);                              // the halfway point
  if (rem > hwy || (rem == hwy && (m & 1ul) != 0ul)) m += 1ul;
  int e = (ea - 1023) + (eb - 127) + (top - 75);
  if (m == (1ul << 53)) { m >>= 1; e += 1; }
  int be = e + 1023;
  if (be <= 0 || be >= 0x7FF) { bad = 1; return 0ul; }
  return (ulong(be) << 52) | (m & 0xFFFFFFFFFFFFFul);
}
"""

SORT_SRC = """
  // one thread: the K (index, value) pairs sorted by value, descending, stable (= numpy's argsort(-tv, kind="stable"))
  int idx[KK]; float val[KK];
  for (int i = 0; i < KK; ++i) { idx[i] = int(ti[i]); val[i] = tv[i]; }
  for (int i = 1; i < KK; ++i) {
    int ci = idx[i]; float cv = val[i]; int j = i - 1;
    while (j >= 0 && val[j] < cv) { idx[j + 1] = idx[j]; val[j + 1] = val[j]; --j; }
    idx[j + 1] = ci; val[j + 1] = cv;
  }
  for (int i = 0; i < KK; ++i) { tis[i] = idx[i]; tvs[i] = val[i]; }
"""

STEP_SRC = """
  // one thread: push the K children of node t - 1 (the roots at t = 0, path_p 1.0 x tp), then pop node t: the live
  // candidate with the largest float64 path_p, the earliest pushed on ties (heapq's (-p, count) order)
  int t = tt[0];
  int n = nf[0];
  int bad = 0;
  for (int i = 0; i < CAP; ++i) { ok[i] = fk[i]; opar[i] = fpar[i]; otok[i] = ftok[i]; odep[i] = fdep[i]; }
  for (int i = 0; i < NMAX; ++i) { onpp[i] = npp[i]; ondep[i] = ndep[i]; }
  for (int i = 0; i < NMAX * NMAX; ++i) { onanc[i] = nanc[i]; }
  ulong pp = (t > 0) ? npp[t - 1] : 0x3FF0000000000000ul;
  int pd = (t > 0) ? ndep[t - 1] + 1 : 0;
  for (int c = 0; c < KK; ++c) {
    if (n >= CAP) { bad = 2; break; }
    ok[n] = f64_mul_f32(pp, tp[c], bad); opar[n] = t - 1; otok[n] = tis[c]; odep[n] = pd; n += 1;
  }
  int best = -1; ulong bk = 0ul;
  for (int i = 0; i < n; ++i) { if (ok[i] > bk) { bk = ok[i]; best = i; } }
  if (best < 0) { bad = 3; best = 0; }
  ok[best] = 0ul;
  int par = opar[best], d = odep[best];
  onpp[t] = bk; ondep[t] = d;
  for (int j = 0; j < NMAX; ++j) { onanc[t * NMAX + j] = -1; }
  if (par >= 0) {
    for (int j = 0; j < d - 1; ++j) { onanc[t * NMAX + j] = nanc[par * NMAX + j]; }
    onanc[t * NMAX + d - 1] = par;
  }
  onf[0] = n;
  opop[0] = par; opop[1] = otok[best]; opop[2] = d; opop[3] = bad;
"""

_K = {}


def _kern(name, inputs, outputs, src):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, header=HDR, source=src)
    return _K[name]


def sort_desc(ti_raw, tv_raw, K):
    """[K] argpartition indices / values -> [K] int32 ids and float32 values, descending by value, stable"""
    k = _kern("sf_sort", ["ti", "tv"], ["tis", "tvs"], SORT_SRC)
    return k(inputs=[ti_raw.reshape(-1), tv_raw.reshape(-1)], template=[("KK", K)], grid=(1, 1, 1),
             threadgroup=(1, 1, 1), output_shapes=[(K,), (K,)], output_dtypes=[mx.int32, mx.float32])


def topk_gpu(lg, K, idmap=None):
    """lg [1, V] f32 -> top ids [K] int32, values [K] f32, probabilities [K] f32 (exp on the GPU), all lazy.
    The same ops as dd_topk.topk_rows up to its numpy part (argpartition of -l, take, logsumexp). idmap (a screened
    head's row -> token id table) maps the K row indices to token ids."""
    lse = mx.logsumexp(lg, axis=-1)
    ti = mx.argpartition(-lg, kth=K - 1, axis=-1)[:, :K]
    tv = mx.take_along_axis(lg, ti, axis=-1)
    tis, tvs = sort_desc(ti, tv, K)
    tp = mx.exp(tvs[None] - lse[:, None])[0]
    if idmap is not None:
        tis = mx.take(idmap, tis).astype(mx.int32)
    return tis, tvs, tp


class GPUFrontier:
    """the frontier and the popped nodes as GPU arrays; step(tis, tp, t) pushes node t - 1's children (the roots at
    t = 0) and pops node t, all lazily"""

    def __init__(self, K, nodes):
        self.K, self.N = K, nodes
        self.CAP = K * nodes
        self.fk = mx.zeros((self.CAP,), mx.uint64)
        self.fpar = mx.full((self.CAP,), PAD, mx.int32)
        self.ftok = mx.zeros((self.CAP,), mx.int32)
        self.fdep = mx.zeros((self.CAP,), mx.int32)
        self.nf = mx.zeros((1,), mx.int32)
        self.npp = mx.zeros((nodes,), mx.uint64)
        self.ndep = mx.zeros((nodes,), mx.int32)
        self.nanc = mx.full((nodes * nodes,), PAD, mx.int32)

    def step(self, tis, tp, t):
        k = _kern("sf_step", ["fk", "fpar", "ftok", "fdep", "nf", "npp", "ndep", "nanc", "tis", "tp", "tt"],
                  ["ok", "opar", "otok", "odep", "onf", "onpp", "ondep", "onanc", "opop"], STEP_SRC)
        C, N = self.CAP, self.N
        o = k(inputs=[self.fk, self.fpar, self.ftok, self.fdep, self.nf, self.npp, self.ndep, self.nanc, tis, tp,
                      mx.array([t], mx.int32)],
              template=[("CAP", C), ("NMAX", N), ("KK", self.K)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
              output_shapes=[(C,), (C,), (C,), (C,), (1,), (N,), (N,), (N * N,), (4,)],
              output_dtypes=[mx.uint64, mx.int32, mx.int32, mx.int32, mx.int32, mx.uint64, mx.int32, mx.int32, mx.int32])
        (self.fk, self.fpar, self.ftok, self.fdep, self.nf, self.npp, self.ndep, self.nanc, pop) = o
        return pop                                          # [par, tok, depth, bad] of node t

    def arrays(self):
        return [self.fk, self.fpar, self.ftok, self.fdep, self.nf, self.npp, self.ndep, self.nanc]


# ------------------------------------------------------------------------------------------------- selftest (synthetic)
def selftest():
    rng = np.random.default_rng(1117)
    out = {}
    # 1. the float64 product, against Python's float multiply
    n = 200_000
    a = np.exp(rng.uniform(np.log(1e-60), 0, n))                       # float64 path products
    b = np.exp(rng.uniform(np.log(1e-9), 0, n)).astype(np.float32)     # float32 probabilities
    b[:1000] = 1.0
    a[1000:2000] = 1.0
    src = """
      uint i = thread_position_in_grid.x;
      int bad = 0;
      ulong r = f64_mul_f32(a[i], b[i], bad);
      o[i] = r; ob[i] = bad;
    """
    k = mx.fast.metal_kernel(name="sf_mul_test", input_names=["a", "b"], output_names=["o", "ob"], header=HDR, source=src)
    o, ob = k(inputs=[mx.array(a.view(np.uint64)), mx.array(b)], grid=(n, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(n,), (n,)], output_dtypes=[mx.uint64, mx.int32])
    ref = (a * b.astype(np.float64)).view(np.uint64)
    got = np.array(o)
    out["mul_mismatch"] = int((got != ref).sum())
    out["mul_bad"] = int(np.array(ob).sum())
    # 2. the stable descending sort, with ties, against numpy
    K = 8
    mism = 0
    for _ in range(300):
        tv = rng.choice(np.array([-3.0, -1.5, -1.5, 0.0, -0.0, 2.25, 2.25, 7.0], np.float32), K)
        ti = rng.permutation(1000)[:K].astype(np.uint32)
        tis, tvs = sort_desc(mx.array(ti), mx.array(tv), K)
        o_ = np.argsort(-tv, kind="stable")
        mism += int(not (np.array_equal(np.array(tis), ti[o_].astype(np.int32)) and
                         np.array_equal(np.array(tvs).view(np.uint32), tv[o_].view(np.uint32))))
    out["sort_mismatch"] = mism
    # 3. the frontier: random trees with forced ties, the GPU pop order vs dd_tree.Forest
    import dd_tree as DT
    mism, trees = 0, 0
    for nodes in (16, 32):
        for _ in range(60):
            grid = np.array([0.5, 0.25, 0.125, 0.3, 0.2, 0.1, 0.05, 0.6], np.float32)   # few distinct values: many ties
            rt = np.sort(rng.choice(grid, K))[::-1].astype(np.float32)
            ri = rng.permutation(5000)[:K].astype(np.int32)
            F = DT.Forest(ri[None], rt[None])
            G = GPUFrontier(K, nodes)
            tis, tp = mx.array(ri), mx.array(rt)
            ok = True
            for t in range(nodes):
                par, tok, dep, pp, ent = F.pop()
                pop = G.step(tis, tp, t)
                cp = np.sort(rng.choice(grid, K))[::-1].astype(np.float32)
                ci = rng.permutation(5000)[:K].astype(np.int32)
                F.push(t, ci[None], cp[None])
                tis, tp = mx.array(ci), mx.array(cp)
                p_ = np.array(pop)
                anc_row = np.array(G.nanc).reshape(nodes, nodes)[t]
                ok &= (p_[0] == par[0] and p_[1] == tok[0] and p_[2] == dep[0] and p_[3] == 0
                       and int(np.array(G.npp)[t]) == int(np.float64(pp[0]).view(np.uint64))
                       and list(anc_row[:dep[0]]) == ent[0][:-1])
            mism += int(not ok)
            trees += 1
    out["frontier_trees"] = trees
    out["frontier_mismatch"] = mism
    out["PASS"] = out["mul_mismatch"] == 0 and out["mul_bad"] == 0 and out["sort_mismatch"] == 0 and mism == 0
    print(json.dumps(out), flush=True)
    return out


# ------------------------------------------------------------------------------------------------- the trees (model)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--set", choices=["heplus24", "mbppbank"], default="heplus24")
    ap.add_argument("--windows", default="0,1")
    ap.add_argument("--nodes", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cut", default="none")
    ap.add_argument("--run", default="t30b")
    ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--max-anchors", type=int, default=48)
    ap.add_argument("--no-ref", action="store_true", help="skip REF (dd_tree's numpy-exp loop)")
    ap.add_argument("--real", action="store_true",
                    help="the r57 format for real: t30b's 10 Linears + the head's HC mixes as QuantizedLinear 8-bit g64 "
                         "(quantized from bf16, as Q8), logits by Leap's ScreenedHead (FR-Spec top --screen + the "
                         "session extras); no REF")
    ap.add_argument("--compile", action="store_true",
                    help="SF runs the pop's ops under mx.compile (two compiled parts around the KV gathers); HS stays eager")
    ap.add_argument("--gpu-load-mb", type=float, default=0.0, help="diagnostic: add a dependent read of this many MB per SF pop")
    ap.add_argument("--tail", action="store_true", help="SF uses the fused tail kernel (sf_tail.py: top-K + frontier)")
    ap.add_argument("--fta", action="store_true", help="SF uses the fused tree attention kernel (fta_kernel.py)")
    ap.add_argument("--capture", default=None, help="save tree_attn's inputs of the first SF tree (every pop) to this npz")
    ap.add_argument("--capture-finish", default=None,
                    help="save T.finish's inputs / output and moe_train's input / output of the first SF tree to this npz")
    ap.add_argument("--screen", type=int, default=0, help="screen rows (0: the format's own, s22_format.FORMATS)")
    ap.add_argument("--fmt", default="q8h32", help="--real format: q8h32 (today) | c2 (S22 / path A); s22_format.py")
    ap.add_argument("--weights", default=None, help="the head's dd checkpoint (default dd_<run>.safetensors), e.g. S22's")
    ap.add_argument("--extra-cap", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import dd_tree as DT
    import dd_ceiling as CE
    import dd_ttt as DD
    from dd_topk import topk_rows
    import p3_dd_tree as P3
    T, M = DD.T, DD.M
    K, NODES = a.k, a.nodes
    t0 = time.time()
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    P.load_weights(a.weights or os.path.join(DD.CKD, f"dd_{a.run}.safetensors")); mx.eval(P.parameters())
    c = P3.CUTS[a.cut]
    done = P3.apply_cut(mx, c, P, head)
    base = None
    if c is not None and c["nh"]:
        RANK = np.load(P3.RANK_F)
        base = np.zeros(lm.lm_head.weight.shape[0], bool)
        base[np.asarray(RANK[:c["nh"]], np.int64)] = True
    NEGINF = mx.array(float("-inf"), mx.float32)
    emb, layer = lm.model.embed_tokens, head.layers[0]
    att = layer.self_attn
    SHC = None
    if a.real:
        import mlx.nn as nn
        assert a.cut == "none", "--real builds its own format (--fmt: s22_format.FORMATS + the screen)"
        a.no_ref = True

        import s22_format as SF22
        if not a.screen:
            a.screen = SF22.FORMATS[a.fmt]["screen"]
        fmt_done, fmt_orig = SF22.apply_real(mx, nn, P, head, a.fmt)
        done += fmt_done
        n_ok, n_mod, n_bad = SF22.check_vs_fakequant(mx, P, head, fmt_orig, P3.fq)
        done.append(f"FORMAT CHECK {a.fmt}: {n_ok} / {n_mod} modules == M3's fake quant bitwise{' BAD ' + str(n_bad) if n_bad else ''}")
        del fmt_orig
        mx.eval(P.parameters(), head.parameters())
        sys.path.insert(0, H("/Users/kamen/BigMoE/data/r57/snap/frozen_s98c/Leap_scripts"))
        from screened_head import ScreenedHead as SHC     # read-only import
        RANK = np.load(P3.RANK_F)
        done.append(f"head: ScreenedHead FR-Spec top-{a.screen} + {a.extra_cap} extras")
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(DT.OUT, f"topk_{a.run}_{a.set}.npz"))
    d0 = TK["depth"] == 0
    print(f"[sf] loaded in {time.time() - t0:.0f} s; cut {a.cut}{' REAL' if a.real else ''}: {done}", flush=True)
    SH = {"sh": None, "idmap": None}                         # the window's screened head (--real)

    def logits_of(h):
        if SH["sh"] is not None:
            return SH["sh"].logits(head.hyper_connection_mixer(h))
        return T.logits(lm, head, h)[0]

    def _c1(HP, tok2, pos1):
        x = T.fuse(head, P, HP, emb(tok2))
        mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
        q, gate, k, v = T.qkv(att, P, mixed, pos1)
        return hyper, inject, q, gate, k, v

    def _c2(q, gate, hyper, inject, AK, AV, am, ancarr, K1, V1):
        o = DT.tree_attn(mx, att.scale, q, K1, V1, AK, AV, am, ancarr)
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
        h = T.finish(head, P, hyper, inject, blk)
        return h, logits_of(h)
    CC = {}                                                  # compiled per window: the screened head is captured
    CAPT = {"on": False, "calls": []}
    CAPF = {"on": False, "calls": []}
    if a.capture_finish:
        _fin, _moe = T.finish, M.moe_train

        def _cap_moe(moe, P_, x):
            out = _moe(moe, P_, x)
            if CAPF["on"]:
                mx.eval(x, out)
                CAPF["calls"][-1].update(moe_x=np.array(x.astype(mx.float32)), moe_x_dtype=str(x.dtype),
                                         moe_out=np.array(out), moe_out_dtype=str(out.dtype))
            return out

        def _cap_fin(head_, P_, hyper, inject, blk):
            if CAPF["on"]:
                mx.eval(hyper, inject, blk)
                CAPF["calls"].append(dict(hyper=np.array(hyper.astype(mx.float32)), hyper_dtype=str(hyper.dtype),
                                          inject=np.array(inject.astype(mx.float32)), inject_dtype=str(inject.dtype),
                                          blk=np.array(blk.astype(mx.float32)), blk_dtype=str(blk.dtype)))
            h = _fin(head_, P_, hyper, inject, blk)
            if CAPF["on"]:
                mx.eval(h)
                CAPF["calls"][-1].update(h=np.array(h.astype(mx.float32)), h_dtype=str(h.dtype))
            return h
        T.finish, M.moe_train = _cap_fin, _cap_moe
    if a.capture:
        _ta = DT.tree_attn

        def _cap_attn(mx_, scale, q, K1, V1, AK, AV, amask, anc):
            o = _ta(mx_, scale, q, K1, V1, AK, AV, amask, anc)
            if CAPT["on"]:
                mx.eval(q, AK, AV, amask, anc, o)
                CAPT["calls"].append(dict(q=np.array(q), AK=np.array(AK), AV=np.array(AV), am=np.array(amask),
                                          anc=np.array(anc), o=np.array(o), scale=float(scale)))
                if "K1" not in CAPT:
                    mx.eval(K1, V1)
                    CAPT["K1"], CAPT["V1"] = np.array(K1), np.array(V1)
            return o
        DT.tree_attn = _cap_attn

    if a.fta:
        from fta_kernel import tree_attn_fused

    def head_step(HP, tok2, pos1, KS, VS, K1, V1, aidx, am, ancarr, compiled=False, fused_attn=False):
        """dd_tree's per-pop ops for one row: the node's forward -> (h, lg, k, v)"""
        if fused_attn:
            x = T.fuse(head, P, HP, emb(tok2))
            mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
            q, gate, k, v = T.qkv(att, P, mixed, pos1)
            KS.append(k); VS.append(v)
            kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
            o = tree_attn_fused(att.scale, q, K1, V1, kst[:, :, aidx], vst[:, :, aidx], am, ancarr)
            blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, blk)
            return h, logits_of(h)
        if compiled:
            hyper, inject, q, gate, k, v = CC["c1"](HP, tok2, pos1)
            KS.append(k); VS.append(v)
            kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
            return CC["c2"](q, gate, hyper, inject, kst[:, :, aidx], vst[:, :, aidx], am, ancarr, K1, V1)
        x = T.fuse(head, P, HP, emb(tok2))
        mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
        q, gate, k, v = T.qkv(att, P, mixed, pos1)
        KS.append(k); VS.append(v)
        kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
        o = DT.tree_attn(mx, att.scale, q, K1, V1, kst[:, :, aidx], vst[:, :, aidx], am, ancarr)
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
        h = T.finish(head, P, hyper, inject, blk)
        lg = logits_of(h)
        return h, lg

    def tree_hs(R, lg0, K1, V1, anc, maskb, exp_on_gpu=True):
        """HS (exp_on_gpu) or REF (dd_tree's topk_rows, numpy exp): the CPU frontier, one sync per pop"""
        if exp_on_gpu:
            tis, _, tp = topk_gpu(lg0, K, SH["idmap"])
            rti, rtp = np.array(tis)[None], np.array(tp)[None]
        else:
            rti, rtp, _, _, _ = topk_rows(mx, lg0, np.zeros(1, np.int32), K)
        F = DT.Forest(rti, rtp)
        HS, KS, VS = [R], [], []
        rec = dict(par=[], tok=[], dep=[], pp=[], ti=[], tp=[], h=[])
        ancarr = mx.array(np.array([anc], np.int32))
        for t in range(NODES):
            par, tok, dep, pp, ent = F.pop()
            J = int(dep.max()) + 1
            e = ent[0]
            aidx = np.empty((1, J), np.int32); am = np.zeros((1, J), bool)
            aidx[0, :len(e)] = e; aidx[0, len(e):] = t; am[0, :len(e)] = True
            h, lg = head_step(HS[int(par[0]) + 1], mx.array(tok)[None], mx.array((anc + dep + 1).astype(np.int32)),
                              KS, VS, K1, V1, mx.array(aidx), mx.array(am), ancarr)
            if maskb is not None:
                lg = mx.where(maskb, lg, NEGINF)
            if exp_on_gpu:
                tis, _, tp = topk_gpu(lg, K, SH["idmap"])
                ti, tpn = np.array(tis)[None], np.array(tp)[None]
            else:
                ti, tpn, _, _, _ = topk_rows(mx, lg, np.zeros(1, np.int32), K)
            mx.eval(h)
            HS.append(h)
            F.push(t, ti, tpn)
            rec["par"].append(int(par[0])); rec["tok"].append(int(tok[0])); rec["dep"].append(int(dep[0]))
            rec["pp"].append(np.float64(pp[0])); rec["ti"].append(ti[0]); rec["tp"].append(tpn[0]); rec["h"].append(h)
        return rec

    CPUT = {"build": [], "wait": [], "t0": 0.0, "c0": 0.0, "build_cpu": [], "ae_wall": [], "ae_cpu": []}
    DUMMY = {"x": mx.ones((int(a.gpu_load_mb * 1e6 / 4),), mx.float32) if a.gpu_load_mb else None}
    if DUMMY["x"] is not None:
        mx.eval(DUMMY["x"])

    if a.tail:
        import sf_tail

    def tree_sf_tail(R, lg0, K1, V1, anc, maskb):
        """SF with the fused tail: one kernel per pop does node t's top-K and pops node t + 1"""
        G = GPUFrontier(K, NODES)
        pop, _, _ = sf_tail.tail(G, lg0, SH["idmap"], 0)
        HS, KS, VS = [R], [], []
        pops, tis_l, tp_l = [], [], []
        ancarr = mx.array(np.array([anc], np.int32))
        anc1 = mx.array(np.array([anc + 1], np.int32))
        for t in range(NODES):
            par, tok, d = pop[0], pop[1], pop[2]
            pops.append(pop)
            HP = mx.take(mx.stack(HS, axis=0), (par + 1).reshape(1), axis=0).reshape(1, 1, -1)
            jj = mx.arange(t + 1, dtype=mx.int32)
            arow = G.nanc[t * NODES:t * NODES + t + 1]
            aidx = mx.where(jj < d, arow, t).reshape(1, t + 1).astype(mx.int32)
            am = (jj <= d).reshape(1, t + 1)
            h, lg = head_step(HP, tok.reshape(1, 1), anc1 + d.reshape(1), KS, VS, K1, V1, aidx, am, ancarr,
                              compiled=a.compile, fused_attn=a.fta)
            if maskb is not None:
                lg = mx.where(maskb, lg, NEGINF)
            pop, tis, tp = sf_tail.tail(G, lg, SH["idmap"], t + 1)
            HS.append(h)
            tis_l.append(tis); tp_l.append(tp)
            mx.async_eval(h, tis, tp, pop, *G.arrays())
        mx.eval(pops, tis_l, tp_l, HS)
        P_ = np.stack([np.array(p) for p in pops])
        return dict(par=list(P_[:, 0]), tok=list(P_[:, 1]), dep=list(P_[:, 2]), bad=int(P_[:, 3].max()),
                    pp=list(np.array(G.npp)), ti=[np.array(x) for x in tis_l], tp=[np.array(x) for x in tp_l], h=HS[1:])

    def tree_sf(R, lg0, K1, V1, anc, maskb):
        """SF: the GPU frontier, no sync until the tree is done"""
        if a.tail:
            return tree_sf_tail(R, lg0, K1, V1, anc, maskb)
        CPUT["t0"] = time.perf_counter(); CPUT["c0"] = time.process_time()
        G = GPUFrontier(K, NODES)
        tis, _, tp = topk_gpu(lg0, K, SH["idmap"])
        HS, KS, VS = [R], [], []
        pops, tis_l, tp_l = [], [], []
        ancarr = mx.array(np.array([anc], np.int32))
        anc1 = mx.array(np.array([anc + 1], np.int32))
        for t in range(NODES):
            pop = G.step(tis, tp, t)
            par, tok, d = pop[0], pop[1], pop[2]
            HP = mx.take(mx.stack(HS, axis=0), (par + 1).reshape(1), axis=0).reshape(1, 1, -1)
            jj = mx.arange(t + 1, dtype=mx.int32)
            arow = G.nanc[t * NODES:t * NODES + t + 1]
            aidx = mx.where(jj < d, arow, t).reshape(1, t + 1).astype(mx.int32)
            am = (jj <= d).reshape(1, t + 1)
            h, lg = head_step(HP, tok.reshape(1, 1), anc1 + d.reshape(1), KS, VS, K1, V1, aidx, am, ancarr,
                              compiled=a.compile, fused_attn=a.fta)
            if maskb is not None:
                lg = mx.where(maskb, lg, NEGINF)
            tis, _, tp = topk_gpu(lg, K, SH["idmap"])
            HS.append(h)
            pops.append(pop); tis_l.append(tis); tp_l.append(tp)
            extra = []
            if a.gpu_load_mb:
                extra = [mx.sum(DUMMY["x"] * (1.0 + 0.0 * h.reshape(-1)[0].astype(mx.float32)))]   # GPU bytes, 2 primitives
            ta, ca = time.perf_counter(), time.process_time()
            mx.async_eval(h, tis, tp, pop, *G.arrays(), *extra)
            CPUT["ae_wall"].append(time.perf_counter() - ta); CPUT["ae_cpu"].append(time.process_time() - ca)
        CPUT["build"].append(time.perf_counter() - CPUT["t0"])
        CPUT["build_cpu"].append(time.process_time() - CPUT["c0"])
        tw = time.perf_counter()
        mx.eval(pops, tis_l, tp_l, HS)
        CPUT["wait"].append(time.perf_counter() - tw)
        P_ = np.stack([np.array(p) for p in pops])
        return dict(par=list(P_[:, 0]), tok=list(P_[:, 1]), dep=list(P_[:, 2]), bad=int(P_[:, 3].max()),
                    pp=list(np.array(G.npp)), ti=[np.array(x) for x in tis_l], tp=[np.array(x) for x in tp_l], h=HS[1:])

    res = dict(fmt=(a.fmt if a.real else None), screen=(a.screen if a.real else None), weights=a.weights, format=done,
               set=a.set, nodes=NODES, k=K, cut=a.cut, windows=a.windows, trees=0, sf_eq_hs=0, sf_bad=0,
               ref_tree_diff=0, ref_tp_cells_diff=0, ref_tp_cells=0, ref_tp_max_ulp=0, first_fail=None,
               ms_per_pop=dict(HS=[], SF=[], REF=[]))
    wins = [int(x) for x in a.windows.split(",")]
    left = a.max_anchors
    for fi in wins:
        if left <= 0:
            break
        z = np.load(FS[fi])
        ids = np.asarray(z["ids"], np.int32)
        S = len(ids); gen = int(z["gen"]); a0 = max(0, gen - 2)
        sel = d0 & (TK["fi"] == fi)
        ancs = (TK["row"][sel] - 1).astype(np.int64)
        if len(ancs) == 0:
            continue
        step = max(1, len(ancs) // max(1, min(left, len(ancs))))
        pick = ancs[::step][:left]
        mask = mx.array(P3.screen_masks(ids, ancs, base)) if base is not None else None
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        idm = mx.array(ids)[None]
        n1 = S - 1
        mixed, hyper, inject = layer.attn_hyper_connection(T.fuse(head, P, W[:, :n1], emb(idm[:, 1:n1 + 1])).astype(mx.bfloat16))
        q, gate, K1, V1 = T.qkv(att, P, mixed, mx.arange(n1))
        o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")
        n = n1 - a0
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
        h1 = T.finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
        mx.eval(K1, V1, h1)
        del mixed, hyper, inject, q, gate, o, blk
        if a.real:                                          # the window's screened head + its session extras
            sh = SHC(lm.lm_head, RANK[:a.screen], extra_cap=a.extra_cap)
            nx = sh.add(ids[:int(ancs.min()) + 2])
            SH["sh"] = sh
            if "bytes_level" not in res:                     # dense + top-k routed experts + the screened rows, MB
                from mlx.utils import tree_flatten
                E = int(P.router.weight.shape[0])
                eb = sum(v.nbytes for _, v in tree_flatten(head.layers[0].mlp.parameters()) if v.ndim >= 2 and v.shape[0] == E) / E
                dn, ex, _ = SF22.bytes_per_level(P, head, None, int(head.layers[0].mlp.top_k), eb)
                hr = sum(x.nbytes for x in (sh.w, sh.s, sh.b)) + int(nx) * (sh.xw.nbytes + sh.xs.nbytes + sh.xb.nbytes) // sh.cap
                res["bytes_level"] = dict(dense_mb=dn / 1e6, experts_mb=ex / 1e6, head_mb=hr / 1e6, total_mb=(dn + ex + hr) / 1e6)
                print(f"[sf] bytes per level: {res['bytes_level']}", flush=True)
            SH["idmap"] = mx.concatenate([sh.ids, sh.xids]) if sh.n_extra else sh.ids
            mx.eval(SH["idmap"])
            res.setdefault("extras", []).append(int(nx))
        CC["c1"], CC["c2"] = mx.compile(_c1), mx.compile(_c2)
        for anc in pick:
            bi = int(np.flatnonzero(ancs == anc)[0])
            R = h1[:, int(anc - a0):int(anc - a0) + 1]
            lg0 = logits_of(R)
            maskb = mask[bi:bi + 1] if mask is not None else None
            if maskb is not None:
                lg0 = mx.where(maskb, lg0, NEGINF)
            mx.eval(R, lg0)
            arms = ["HS", "SF"] + ([] if a.no_ref else ["REF"])
            if res["trees"] % 2:
                arms = arms[::-1]
            out = {}
            for arm in arms:
                mx.synchronize()
                ts = time.perf_counter()
                if arm == "SF":
                    CAPF["on"] = bool(a.capture_finish) and res["trees"] == 0
                    CAPT["on"] = bool(a.capture) and res["trees"] == 0
                    out[arm] = tree_sf(R, lg0, K1, V1, int(anc), maskb)
                    if CAPF["on"]:
                        CAPF["on"] = False
                        cs = CAPF["calls"]
                        np.savez(a.capture_finish, n=len(cs),
                                 **{f"{k}_{i}": c[k] for i, c in enumerate(cs) for k in c})
                        print(f"[sf] captured {len(cs)} finish calls -> {a.capture_finish}; dtypes "
                              f"{ {k: v for k, v in cs[0].items() if k.endswith('_dtype')} }", flush=True)
                    if CAPT["on"]:
                        CAPT["on"] = False
                        cs = CAPT["calls"]
                        np.savez(a.capture, K1=CAPT["K1"], V1=CAPT["V1"], n=len(cs), scale=cs[0]["scale"],
                                 **{f"{k}_{i}": c[k] for i, c in enumerate(cs) for k in ("q", "AK", "AV", "am", "anc", "o")})
                        print(f"[sf] captured {len(cs)} tree_attn calls -> {a.capture}; dtypes q {cs[0]['q'].dtype} "
                              f"K1 {CAPT['K1'].dtype} {CAPT['K1'].shape} AK {cs[0]['AK'].dtype} o {cs[0]['o'].dtype}", flush=True)
                else:
                    out[arm] = tree_hs(R, lg0, K1, V1, int(anc), maskb, exp_on_gpu=(arm == "HS"))
                mx.synchronize()
                res["ms_per_pop"][arm].append((time.perf_counter() - ts) * 1e3 / NODES)
            hs, sf = out["HS"], out["SF"]
            eq = (hs["par"] == [int(x) for x in sf["par"]] and hs["tok"] == [int(x) for x in sf["tok"]]
                  and hs["dep"] == [int(x) for x in sf["dep"]]
                  and all(int(np.float64(x).view(np.uint64)) == int(y) for x, y in zip(hs["pp"], sf["pp"]))
                  and all(np.array_equal(x, y) for x, y in zip(hs["ti"], sf["ti"]))
                  and all(np.array_equal(np.asarray(x).view(np.uint32), np.asarray(y).view(np.uint32))
                          for x, y in zip(hs["tp"], sf["tp"]))
                  and all(bool(mx.array_equal(x, y).item()) for x, y in zip(hs["h"], sf["h"])))
            res["sf_eq_hs"] += int(eq)
            res["sf_bad"] += int(sf["bad"] != 0)
            if not eq and res["first_fail"] is None:
                res["first_fail"] = dict(fi=fi, anc=int(anc), hs_par=hs["par"], sf_par=[int(x) for x in sf["par"]],
                                         hs_tok=hs["tok"], sf_tok=[int(x) for x in sf["tok"]])
            if "REF" in out:
                rf = out["REF"]
                res["ref_tree_diff"] += int(not (rf["par"] == hs["par"] and rf["tok"] == hs["tok"]))
                for x, y in zip(rf["tp"], hs["tp"]):
                    du = np.abs(np.asarray(x, np.float32).view(np.int32).astype(np.int64)
                                - np.asarray(y, np.float32).view(np.int32).astype(np.int64))
                    res["ref_tp_cells"] += du.size
                    res["ref_tp_cells_diff"] += int((du != 0).sum())
                    res["ref_tp_max_ulp"] = max(res["ref_tp_max_ulp"], int(du.max()))
            res["trees"] += 1
            left -= 1
            if left <= 0:
                break
        del K1, V1, h1
        mx.clear_cache()
    for arm, v in list(res["ms_per_pop"].items()):
        res["ms_per_pop"][arm] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                      p90=float(np.percentile(v, 90)), n=len(v)) if v else None
    if CPUT["build"]:
        res["sf_cpu_build_ms_per_pop"] = float(np.median(CPUT["build"])) * 1e3 / NODES
        res["sf_final_wait_ms_per_tree"] = float(np.median(CPUT["wait"])) * 1e3
        res["sf_build_cpu_ms_per_pop"] = float(np.median(CPUT["build_cpu"])) * 1e3 / NODES
        res["sf_async_eval_wall_us"] = float(np.median(CPUT["ae_wall"])) * 1e6
        res["sf_async_eval_cpu_us"] = float(np.median(CPUT["ae_cpu"])) * 1e6
    res["PASS"] = res["trees"] > 0 and res["sf_eq_hs"] == res["trees"] and res["sf_bad"] == 0
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
