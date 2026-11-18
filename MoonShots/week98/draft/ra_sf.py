"""DRAFT LINE hand-off (BigMoE Tue 17 Nov 17:40; the orchestrator 17:5x): r57's tree drafter on the LIVE MTPLX head.
MoonShots, Tue 17 Nov 2026 (account-1). BigMoE's files are imported read-only (tree16_draft = "Route A").

1. FORMAT on the live head (lm = TextModel, lm.mtp = the head; mtp125.Trainable's map from P names to the head's own
   modules), switchable:
     q8h32  MTP Head's packed artifact ~/datasets/deepdraft/q8h32/t30b_q8h32.safetensors (P.* / head.* names, 8-bit
            g64), or an fp32 dd checkpoint quantized at load; top-10; ScreenedHead FR-Spec 32K + session extras
     c2     an fp32 dd checkpoint (S22: dd_t30qa) quantized at load per s22_format.FORMATS['c2'] (4-bit g64 Linears and
            HC mixes, 8-bit router); top-4; ScreenedHead 8K + session extras
   The draft head (lm._mtp_draft_head_logits) is rebound to the screened rows scattered into a full-vocabulary row
   (-inf elsewhere), so Route A's _topk_np and the loop see vocabulary ids unchanged.
2. SYNC-FREE draft with tree16_draft.draft's signature and output (DFS preorder dict), in a STATIC-WIDTH form: node t's
   forward is Route A's op sequence on the live layer's own modules (fuse, HC reads, q / k / v, norms, RoPE, o_proj,
   gate, MoE, mixer, head), except that the attention reads the committed keys + ALL drafted rows in pop order + its
   own row with an ancestor mask, and the RoPE position Lc + depth - 1 is a GPU value. The MTP cache is not touched
   during the draft. The frontier is sf_tree's GPU heap (float64 path products exact; the (value desc, id asc) top-K
   order of Route A's lexsort); p = exp(v - lse) in fp32 on the GPU (Route A: numpy float64), so path_p is equal to
   ~1e-7, not bitwise.
3. COMPARE on bank windows: Route A (tree16_draft.draft, host-synced) vs SF at the same format: per tree, tokens /
   parents / depth identical?, path_p max relative error, ms per pop. Plus a replica check: SF's forward with the
   exact ancestor rows (no padding, no mask) vs Route A's mtp_forward, bitwise.
usage: ~/BigMoE/.venv/bin/python -I -B ra_sf.py --fmt q8h32|c2 [--weights <dd ckpt>] [--set heplus24] [--windows 0,1]
       [--anchors 24] [--out <json>]     (MEM ~9 GB: share lock)"""
import argparse
import json
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/BigMoE/scripts"))                  # read-only imports
sys.path.insert(0, H("~/MoonShots/week98"))
sys.path.insert(0, H("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402
import mlx.nn as nn                                        # noqa: E402

Q8H32_FILE = H("~/datasets/deepdraft/q8h32/t30b_q8h32.safetensors")
RANK_F = H("~/Leap/data/frspec_rank.npy")
P_TO_LIVE = {                                              # mtp125.Trainable: P name -> the module path on lm.mtp
    "fc_hidden": ("fc_hidden",), "fc_embedding": ("fc_embedding",),
    "q_proj": ("layers", 0, "self_attn", "q_proj"), "k_proj": ("layers", 0, "self_attn", "k_proj"),
    "v_proj": ("layers", 0, "self_attn", "v_proj"), "o_proj": ("layers", 0, "self_attn", "o_proj"),
    "router": ("layers", 0, "mlp", "gate"),
    "se_gate": ("layers", 0, "mlp", "shared_expert", "gate_proj"),
    "se_up": ("layers", 0, "mlp", "shared_expert", "up_proj"),
    "se_down": ("layers", 0, "mlp", "shared_expert", "down_proj"),
    "se_gatew": ("layers", 0, "mlp", "shared_expert_gate"),
}
HC_TO_LIVE = {"attn_hc": ("layers", 0, "attn_hyper_connection"), "mlp_hc": ("layers", 0, "mlp_hyper_connection"),
              "mixer": ("hyper_connection_mixer",)}


def _get(m, path):
    for p in path:
        m = m[p] if isinstance(p, int) else getattr(m, p)
    return m


def _set(m, path, mod):
    parent = _get(m, path[:-1])
    setattr(parent, path[-1], mod)


def _qlin_packed(w, s, b, bits, gs=64):
    out_dims, in_dims = int(w.shape[0]), int(w.shape[1]) * 32 // bits
    ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=gs, bits=bits)
    ql.weight, ql.scales, ql.biases = w, s, b
    return ql


def _qlin_from(wf, bits, gs=64):
    q, s, b = mx.quantize(wf.astype(mx.bfloat16), group_size=gs, bits=bits)
    return _qlin_packed(q, s, b, bits, gs)


class CastLinear(nn.Module):
    """t30b's fp32 se_gatew kept exactly, its output cast to the input's dtype: the live MoE's shared-expert gate must
    not promote the bf16 residual stream to fp32 (fp32 activations make every routed-expert gather_qmm ~40x slower:
    the stored experts' bf16 scales; measured 7 ms vs ~0.2 ms per pop). TrainLab's moe_train casts the MoE output back
    to the stream's dtype the same way (dd_d22_ttt.finish: .astype(hyper.dtype))."""

    def __init__(self, w):
        super().__init__()
        self.weight = w

    def __call__(self, x):
        return (x.astype(mx.float32) @ self.weight.T).astype(x.dtype)


class SharedMLP(nn.Module):
    """mtp125.SharedMLP's call (down(silu(gate(x)) * up(x))); its modules are set by install_format"""

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def install_format(lm, fmt, weights=None, extra_cap=4096, screen_order="rank"):
    """install the format on lm.mtp in place; returns (done, ScreenedHead factory, screen size).
    screen_order: 'rank' (FR-Spec frequency order, the default of every run so far) or 'id' (the base screen sorted by
    token id: screened-row ties then break by token id, as M3's masked full rows and Route A's full-V rows do)."""
    head = lm.mtp
    done = []
    att0 = head.layers[0].self_attn
    if getattr(att0, "qkv_fused", None) is not None:          # a fused q / k / v would keep the old weights
        att0.qkv_fused = None
        done.append("qkv_fused reset (stale weights)")
    if hasattr(head, "_prepare_inputs_eager"):                # a compiled prepare graph bakes the old fc weights in
        object.__setattr__(head, "_mtp_prepare_inputs_impl", head._prepare_inputs_eager)
        done.append("prepare = eager")
    if hasattr(head.layers[0].mlp.shared_expert, "gu_weight"):   # MTPLX's fused gate+up: its call would keep the old
        head.layers[0].mlp.shared_expert = SharedMLP()           # weights; gate / up / down are set below
        done.append("fused shared expert -> gate / up / down")
    if fmt == "q8h32" and weights is None:
        W = mx.load(Q8H32_FILE)
        for name, path in P_TO_LIVE.items():
            if name == "se_gatew":
                _set(head, path, CastLinear(W["P.se_gatew.weight"]))
                done.append(f"{name} fp32 weight (t30b), output in the stream's dtype")
                continue
            _set(head, path, _qlin_packed(W[f"P.{name}.weight"], W[f"P.{name}.scales"], W[f"P.{name}.biases"], 8))
            done.append(f"{name} 8-bit (artifact)")
        for hn, path in HC_TO_LIVE.items():
            g = _get(head, path)
            for sub in ("input_mix_weight_down", "input_mix_weight_up"):
                k_ = f"head.{hn}.{sub}"
                setattr(g, sub, _qlin_packed(W[k_ + ".weight"], W[k_ + ".scales"], W[k_ + ".biases"], 8))
                done.append(f"{hn}.{sub} 8-bit (artifact)")
        top, screen = 10, 32768
    else:
        import s22_format as SF22
        f = SF22.FORMATS[fmt]
        W = mx.load(weights or H("~/datasets/deepdraft/ckpt/dd_t30b.safetensors"))
        for name, path in P_TO_LIVE.items():
            wf = W[f"{name}.weight"]
            if name == "se_gatew":
                _set(head, path, CastLinear(wf))
                done.append(f"{name} fp32 weight, output in the stream's dtype")
                continue
            bits = f["router"] if name == "router" else f["lin"]
            _set(head, path, _qlin_from(wf, bits))
            done.append(f"{name} {bits}-bit")
        for hn, path in HC_TO_LIVE.items():
            g = _get(head, path)
            for sub in ("input_mix_weight_down", "input_mix_weight_up"):
                lin = getattr(g, sub)
                assert not hasattr(lin, "scales"), f"{hn}.{sub} already quantized"
                setattr(g, sub, _qlin_from(lin.weight, f["hc"]))
                done.append(f"{hn}.{sub} {f['hc']}-bit")
        top, screen = f["top"], f["screen"]
    head.layers[0].mlp.top_k = top
    done.append(f"experts top-{top}")
    mx.eval(head.parameters())
    sys.path.insert(0, H("~/Leap/scripts"))
    from screened_head import ScreenedHead
    rank = np.load(RANK_F)[:screen]
    if screen_order == "id":
        rank = np.sort(rank)
    done.append(f"screen order {screen_order}")
    V = int(lm.lm_head.weight.shape[0])
    state = SCREEN                                            # the screen the draft head reads (settable)
    state["sh"] = None

    def new_screen():
        state["sh"] = ScreenedHead(lm.lm_head, rank, extra_cap=extra_cap)
        return state["sh"]

    def draft_head(x):
        sh = state["sh"]
        lead = x.shape[:-1]
        sl = sh.logits(x)                                     # [n, NS] fp32
        if SCREENED["on"]:                                    # the screened row (ids via idmap())
            return sl.reshape(*lead, sl.shape[-1])
        ids = mx.concatenate([sh.ids, sh.xids[:sh.n_extra]]) if sh.n_extra else sh.ids   # used slots only: the unused
        full = mx.full((sl.shape[0], V), -float("inf"), mx.float32)                     # ones hold id 0 (duplicates)
        full[:, ids] = sl[:, :ids.shape[0]]
        return full.reshape(*lead, V)
    object.__setattr__(lm, "_mtp_draft_head_logits", draft_head)
    done.append(f"draft head: ScreenedHead FR-Spec top-{screen} + {extra_cap} extras "
                f"({'screened rows + id map' if SCREENED['on'] else 'scattered to V ' + str(V)})")
    return done, new_screen, screen


SCREENED = {"on": True}                                     # False: the full-vocabulary row (-inf off the screen)
SCREEN = {"sh": None}                                       # the current ScreenedHead (install_format's draft head)


def idmap(sh):
    """the screened row -> token id table (base rows, then the session extras' slots)"""
    return mx.concatenate([sh.ids, sh.xids]) if sh.n_extra else sh.ids


class Head:
    """MTPHead's state (r57_screen_fast.MTPHead): the MTP cache, the root's logits and hidden"""

    def __init__(self, lm):
        self.lm = lm
        self.cache = lm.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        lm = self.lm
        nid = np.asarray(next_ids, np.int32)
        if hidden.shape[1] > 1:
            lm.mtp_update_cache(hidden[:, :-1], mx.array(nid[None, :-1]), mtp_cache=self.cache)
        self.logits, self.h = lm.mtp_forward(hidden[:, -1:], mx.array(nid[None, -1:]), mtp_cache=self.cache,
                                             return_hidden=True)


# --------------------------------------------------------------------------------------- the sync-free static form
SORT_IDX_SRC = """
  // one thread: the K (index, value) pairs by value descending, ties by index ascending (Route A's lexsort)
  int idx[KK]; float val[KK];
  for (int i = 0; i < KK; ++i) { idx[i] = int(ti[i]); val[i] = tv[i]; }
  for (int i = 1; i < KK; ++i) {
    int ci = idx[i]; float cv = val[i]; int j = i - 1;
    while (j >= 0 && (val[j] < cv || (val[j] == cv && idx[j] > ci))) { idx[j + 1] = idx[j]; val[j + 1] = val[j]; --j; }
    idx[j + 1] = ci; val[j + 1] = cv;
  }
  for (int i = 0; i < KK; ++i) { tis[i] = idx[i]; tvs[i] = val[i]; }
"""
_K = {}


def topk_ra(row, K):
    """row [V] (any float) -> ids [K] int32, values [K] f32, p [K] f32: Route A's _topk_np ops on the GPU (the fp32
    logsumexp / argpartition / take, then (value desc, id asc)); p in fp32 (Route A: float64 in numpy)"""
    l = row.astype(mx.float32)
    lse = mx.logsumexp(l)
    if K == 1:                                                # chain drafts: argmax = the lowest index of the max,
        tis = mx.argmax(l).astype(mx.int32).reshape(1)        # the same pick as the total order below
        tvs = mx.take(l, tis)
        return tis, tvs, mx.exp(tvs - lse)
    idx = mx.argpartition(-l, kth=K - 1)[:K]
    val = mx.take(l, idx)
    k = _K.setdefault("ra_sort", mx.fast.metal_kernel(name="ra_sort", input_names=["ti", "tv"],
                                                      output_names=["tis", "tvs"], source=SORT_IDX_SRC))
    tis, tvs = k(inputs=[idx.astype(mx.int32), val], template=[("KK", K)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                 output_shapes=[(K,), (K,)], output_dtypes=[mx.int32, mx.float32])
    p = mx.exp(tvs - lse)
    return tis, tvs, p


def _rope_qk(att, q, k, pos):
    """Route A's RoPE for one row at a GPU position pos [1] int32 (the same path the live Attention takes)"""
    import mtplx.models.qwen4_exp as Q
    if Q.qwen4_opdiet_enabled("rope"):
        cos, sin = Q._shared_rope_cos_sin_half(pos.reshape(()), 1, att._inv_freq, att._rope_attention_scaling)
        return Q._apply_partial_rope_half(q, cos, sin), Q._apply_partial_rope_half(k, cos, sin)
    positions = pos.reshape(()) + mx.arange(1, dtype=mx.int32)
    cos, sin = Q._rope_cos_sin(positions, att._inv_freq, att._rope_attention_scaling)
    return Q._apply_partial_rope(q, cos, sin), Q._apply_partial_rope(k, cos, sin)


def forward_static(lm, h_parent, tok, pos, Kc, Vc, Kd, Vd, mask):
    """node forward: Route A's op sequence on the live layer, attention over [Kc | Kd | self] with `mask`
    (bool [1, 1, 1, T] or None). Returns logits [1, 1, V], h, the node's post-RoPE k / v rows [1, KV, 1, hd]."""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_math import attention_gate
    head = lm.mtp
    L0 = head.layers[0]
    att = L0.self_attn
    assert getattr(att, "qkv_fused", None) is None, "fused qkv: rebuild it for the format first"
    emb = lm.model.embed_tokens(tok.reshape(1, 1))
    x = head._mtp_prepare_inputs_impl(h_parent, emb)
    mixed, hyper, inject = L0.attn_hyper_connection(x)
    B, S = 1, 1
    q, k, v = att.q_proj(mixed), att.k_proj(mixed), att.v_proj(mixed)
    q, gate = mx.split(q.reshape(B, S, att.n_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    k = k.reshape(B, S, att.n_kv_heads, -1)
    v = v.reshape(B, S, att.n_kv_heads, -1)
    q, k = att.q_norm(q), att.k_norm(k)
    q, k = _rope_qk(att, q, k, pos)
    q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    k, v = k.astype(Kc.dtype), v.astype(Vc.dtype)          # the cache stores its rows in its own dtype (bf16)
    parts_k = [Kc] + ([Kd] if Kd is not None else []) + [k]
    parts_v = [Vc] + ([Vd] if Vd is not None else []) + [v]
    Ka, Va = mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)
    out = Q._verify_sdpa(q, Ka, Va, scale=att.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    block_out = Q._linear(att.o_proj, attention_gate(out, gate))
    mixed, hyper, inject = L0.mlp_hyper_connection(hyper, pending=(block_out, inject))
    block_out = L0.mlp(mixed)
    h = Q._hyper_residual_write(hyper, block_out, inject)
    logits = lm._mtp_draft_head_logits(head.hyper_connection_mixer(h))
    return logits, h, k, v


def draft_sf(lm, mtp, root_tok, n=15, k=8, imap=None, rec=None):
    """tree16_draft.draft's contract, sync-free: n pops (n - 1 forwards), the GPU frontier, one sync per tree.
    imap: the screened row -> token id table (screened logits) or None (full-vocabulary logits).
    rec: a list to receive each forward's (logits, h) in pop order (the per-node comparison)."""
    import sf_tree as SFT
    import tree16_draft as T16D
    qc = mtp.cache[0]
    Lc = int(qc.offset)
    Kc, Vc = qc.kv.keys[..., :Lc, :], qc.kv.values[..., :Lc, :]
    G = SFT.GPUFrontier(k, n)

    def topk_ids(row):
        tis_, tv_, tp_ = topk_ra(row, k)
        return (mx.take(imap, tis_).astype(mx.int32) if imap is not None else tis_), tp_
    tis, tp = topk_ids(mtp.logits[0, -1])
    HS, KR, VR, pops = [mtp.h], [], [], []
    Lc_arr = mx.array([Lc], mx.int32)
    for t in range(n):
        pop = G.step(tis, tp, t)
        pops.append(pop)
        if t == n - 1:
            break
        par, tok, d0 = pop[0], pop[1], pop[2]                 # d0: the frontier's depth (root children 0)
        hp = mx.take(mx.stack(HS, axis=0), (par + 1).reshape(1), axis=0).reshape(HS[0].shape)
        pos = Lc_arr + d0.reshape(1)                           # Route A: Lc + (depth - 1), depth = d0 + 1
        if t:
            Kd, Vd = mx.concatenate(KR, axis=2), mx.concatenate(VR, axis=2)
            jj = mx.arange(t, dtype=mx.int32)
            arow = G.nanc[t * n:t * n + t]                     # node t's ancestors (node ids, -1 padded)
            anc_mask = (arow[None, :] == jj[:, None]).any(axis=1)          # drafted row j is an ancestor of node t
            mask = mx.concatenate([mx.ones((Lc,), mx.bool_), anc_mask, mx.ones((1,), mx.bool_)]).reshape(1, 1, 1, -1)
        else:
            Kd = Vd = None
            mask = None
        lg, h, kr, vr = forward_static(lm, hp, tok, pos, Kc, Vc, Kd, Vd, mask)
        HS.append(h); KR.append(kr); VR.append(vr)
        if rec is not None:
            rec.append((lg, h))
        tis, tp = topk_ids(lg[0, -1])
        mx.async_eval(h, kr, vr, tis, tp, pop, *G.arrays())
    mx.eval(pops, *G.arrays())
    P_ = np.stack([np.array(p) for p in pops])
    npp = np.array(G.npp).view(np.float64)
    par = [int(x) for x in P_[:, 0]]
    tok = [int(x) for x in P_[:, 1]]
    dep = [int(x) + 1 for x in P_[:, 2]]
    pth = [float(x) for x in npp[:n]]
    pdr = [pth[i] / (pth[par[i]] if par[i] >= 0 else 1.0) for i in range(n)]
    out = T16D.preorder(int(root_tok), par, tok, dep, pth, pdr)
    out["bad"] = int(P_[:, 3].max())
    return out


# --------------------------------------------------------------------------------------- the comparison harness
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="q8h32")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--set", choices=["heplus24", "mbppbank"], default="heplus24")
    ap.add_argument("--windows", default="0")
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--replica", action="store_true", help="also check SF's forward == Route A's mtp_forward bitwise")
    ap.add_argument("--full-v", action="store_true", help="the full-vocabulary logits row (-inf off the screen) instead "
                    "of the screened row + id map")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import dd_ceiling as CE
    import dd_ttt as DD
    import tree16_draft as T16D
    import dd_tree as DT
    t0 = time.time()
    rt, lm, head = DD.M.load(0.03)
    SCREENED["on"] = not a.full_v
    done, new_screen, screen = install_format(lm, a.fmt, a.weights)
    print(f"[ra] loaded + format {a.fmt} in {time.time() - t0:.0f} s: {done}", flush=True)
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(DT.OUT, f"topk_t30b_{a.set}.npz"))
    res = dict(fmt=a.fmt, weights=a.weights, set=a.set, windows=a.windows, n=a.n, k=a.k, trees=0, same_tree=0,
               same_tokens=0, path_p_max_rel=0.0, bad=0, ms_per_pop=dict(A=[], SF=[]), replica=[], diffs=[],
               screened=not a.full_v, nodes_cmp=0, nodes_logits_eq=0, nodes_h_eq=0, same_pop_order=0)
    left = a.anchors
    for fi in [int(x) for x in a.windows.split(",")]:
        if left <= 0:
            break
        z = np.load(FS[fi])
        ids = np.asarray(z["ids"], np.int32)
        sel = (TK["depth"] == 0) & (TK["fi"] == fi)
        ancs = np.sort((TK["row"][sel] - 1).astype(np.int64))
        step = max(1, len(ancs) // max(1, min(left, len(ancs))))
        pick = ancs[::step][:left]
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        mtp = Head(lm)
        sh = new_screen()
        done_upto = -1
        for s in pick:
            s = int(s)
            sh.add(ids[:s + 2])
            mtp.append(W[:, done_upto + 1:s + 1], ids[done_upto + 2:s + 2])
            done_upto = s
            mx.eval(mtp.logits, mtp.h)
            root = int(ids[s + 1])
            if a.replica and len(res["replica"]) < 3:
                # SF's forward with the exact keys (committed + self, no mask) vs Route A's mtp_forward, a root child
                qc = mtp.cache[0]
                Lc = int(qc.offset)
                c1 = int(mx.argmax(mtp.logits[0, -1]).item())
                lgA, hA = lm.mtp_forward(mtp.h, mx.array([[c1]]), mtp_cache=mtp.cache, return_hidden=True)
                mx.eval(lgA, hA)
                qc.trim(1)
                lgS, hS, _, _ = forward_static(lm, mtp.h, mx.array(c1, mx.int32), mx.array([Lc], mx.int32),
                                               qc.kv.keys[..., :Lc, :], qc.kv.values[..., :Lc, :], None, None, None)
                mx.eval(lgS, hS)
                eqh = bool(mx.array_equal(hA, hS).item())
                eql = bool(mx.array_equal(lgA.astype(mx.float32), lgS.astype(mx.float32)).item())
                res["replica"].append(dict(s=s, h_equal=eqh, logits_equal=eql,
                                           h_maxabs=float(mx.max(mx.abs(hA.astype(mx.float32) - hS.astype(mx.float32))).item())))
                assert int(qc.offset) == Lc
            arms = ["A", "SF"] if res["trees"] % 2 == 0 else ["SF", "A"]
            out = {}
            imap = None if a.full_v else idmap(sh)
            imap_np = None if imap is None else np.array(imap)
            recA, recS = [], []

            def fwd_a(h_, idx_):
                tok_ = int(idx_) if imap_np is None else int(imap_np[int(idx_)])
                lg_, h2_ = lm.mtp_forward(h_, mx.array([[tok_]]), mtp_cache=mtp.cache, return_hidden=True)
                recA.append((lg_, h2_))
                return lg_, h2_
            for arm in arms:
                mx.synchronize()
                ts = time.perf_counter()
                if arm == "A":
                    o_ = T16D.draft(lm, mtp, root, n=a.n, k=a.k, fwd=fwd_a)
                    if imap_np is not None:
                        o_["tokens"] = np.concatenate([o_["tokens"][:1], imap_np[o_["tokens"][1:]]]).astype(np.int32)
                    out[arm] = o_
                else:
                    out[arm] = draft_sf(lm, mtp, root, n=a.n, k=a.k, imap=imap, rec=recS)
                mx.synchronize()
                res["ms_per_pop"][arm].append((time.perf_counter() - ts) * 1e3 / a.n)
                assert int(mtp.cache[0].offset) == s + 1, ("the cache was not restored", arm)
            A, S_ = out["A"], out["SF"]
            same_tok = np.array_equal(A["tokens"], S_["tokens"])
            same = same_tok and np.array_equal(A["parents"], S_["parents"]) and np.array_equal(A["depth"], S_["depth"])
            res["trees"] += 1
            res["same_tokens"] += int(same_tok)
            res["same_tree"] += int(same)
            res["bad"] += int(S_["bad"] != 0)
            if same:
                rel = float(np.max(np.abs(S_["path_p"] - A["path_p"]) / np.maximum(A["path_p"], 1e-300)))
                res["path_p_max_rel"] = max(res["path_p_max_rel"], rel)
                if np.array_equal(A["pop"], S_["pop"]):
                    res["same_pop_order"] += 1
                    for (la, ha), (ls_, hs_) in zip(recA, recS):          # the forwards in pop order
                        res["nodes_cmp"] += 1
                        res["nodes_logits_eq"] += int(bool(mx.array_equal(la, ls_).item()))
                        res["nodes_h_eq"] += int(bool(mx.array_equal(ha, hs_).item()))
            elif len(res["diffs"]) < 5:
                res["diffs"].append(dict(fi=fi, s=s, A_tok=A["tokens"].tolist(), SF_tok=S_["tokens"].tolist(),
                                         A_par=A["parents"].tolist(), SF_par=S_["parents"].tolist()))
            left -= 1
            if left <= 0:
                break
        del W, mtp
        mx.clear_cache()
    for arm, v in list(res["ms_per_pop"].items()):
        res["ms_per_pop"][arm] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                      p90=float(np.percentile(v, 90)), n=len(v)) if v else None
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "diffs"}), flush=True)
    if res["diffs"]:
        print("first diffs:", json.dumps(res["diffs"][:2])[:1500], flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
