"""Roofline engine, M1a: a full 125B forward for M rows (decode / verify) on MTPLX's loaded weights (read-only use) and
BigMoE's expert pools. GDN layers run the engine's fused attention half (rle/gdn.py); QSA layers use the engine's HC
kernels around MTPLX's attention core; the MoE half is router + top-10 + residency (BigMoE ext, synchronous here) + the pool
SwitchGLU + combine + gated shared expert (M1b replaces it with the fused tier GLU); the final mixer and the LM head close it.
Non-exact contract (engine/PLAN.md)."""
import os, sys
import numpy as np
import mlx.core as mx
from . import kernels as KN
from . import gdn as G

K = 10


def qmm(x, w, bits=4):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=bits)


def pack(model):
    """Per-layer engine weights (references into MTPLX's arrays; the HC down rows get the inject rows appended)."""
    tm = model.language_model.model
    P = []
    for i, layer in enumerate(tm.layers):
        d = dict(gdn=layer.is_linear, attn_hc=G.pack_hc(layer.attn_hyper_connection),
                 mlp_hc=G.pack_hc(layer.mlp_hyper_connection), layer=layer)
        if layer.is_linear:
            d.update(G.pack_gdn(layer))
        mlp = layer.mlp
        d["router"] = G.qw(mlp.gate)
        d["router_bits"] = mlp.gate.bits
        P.append(d)
    fin = G.pack_hc(tm.hyper_connection_mixer)
    fin["Wd"] = mx.concatenate([fin["Wd"], mx.zeros((324 - fin["Wd"].shape[0], 10240), fin["Wd"].dtype)], axis=0)
    return P, fin


class Engine:
    def __init__(self, model, ext, moe_mode="sync"):
        """moe_mode: "sync" = exact experts with synchronous residency (tests); "bigmoe" = BigMoE's own MoE block on its
        GPU gate (post_ids/spin, loader thread, look-ahead prefetch, tier pools + MoonShots' 3c GLU with fused combine)"""
        self.moe_mode = moe_mode
        self.model, self.ext = model, ext
        self.tm = model.language_model.model
        self.P, self.fin = pack(model)
        mx.eval([{k: v for k, v in d.items() if k != "layer"} for d in self.P], self.fin)
        self.head = G.qw(model.language_model.lm_head)

    def new_state(self):
        st = {"cache": self.model.make_cache()}
        for i, d in enumerate(self.P):
            if d["gdn"]:
                st[i] = (mx.zeros((3, 10240), mx.bfloat16), mx.zeros((48 * 16384,), mx.float32))
        return st

    def experts(self, i, x, ids):
        """exact routed experts on pool slots (synchronous residency): x [M, 2560], ids [M, K] int -> [M, K, 2560]"""
        sw = self.P[i]["layer"].mlp.switch_mlp
        L = sw._bm["L"]                                             # BigMoE ext layer of this model layer
        idn = np.array(ids)
        flat = [int(v) for v in idn.reshape(-1)]
        self.ext.ensure(L, mx.array(np.array(sorted(set(flat)), np.int32)))
        slots = np.array(self.ext.lookup(L, flat)).reshape(idn.shape)
        assert (slots >= 0).all(), "expert not resident after ensure"
        base = type(sw).__mro__[1]
        return base.__call__(sw, x[:, None, :], mx.array(slots.astype(np.uint32))[:, None, :])[:, 0]

    def moe(self, i, x):
        if self.moe_mode == "bigmoe":
            return self.P[i]["layer"].mlp(x[None])[0]
        if self.moe_mode == "oracle":                               # gate-free hit path (prototype): GPU residency lookup
            return self.moe_oracle(i, x)
        if self.moe_mode == "shared":                               # timing floor: no routed experts, no gate
            mlp = self.P[i]["layer"].mlp
            return mx.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
        d = self.P[i]
        mlp = d["layer"].mlp
        lg = qmm(x, d["router"], d["router_bits"]).astype(mx.float32)
        p = mx.softmax(lg, axis=-1, precise=True)
        top = mx.argpartition(-p, K - 1, axis=-1)[:, :K]
        pk = mx.take_along_axis(p, top, axis=-1)
        o = mx.argsort(-pk, axis=-1)
        ids, pk = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(pk, o, axis=-1)
        w = (pk / pk.sum(-1, keepdims=True)).astype(x.dtype)
        yk = self.experts(i, x, ids)
        y = (yk * w[..., None]).sum(axis=-2)
        sh = mlp.shared_expert(x)
        return y + mx.sigmoid(mlp.shared_expert_gate(x)) * sh

    def build_residency(self):
        """Snapshot GPU-side residency tables (expert id -> slot, -1 = absent) for every layer's base pool and LSB sidecar
        (BigMoE tier mode). Prototype of the gate-free hit path: the real design reads the tables the ext publishes."""
        self.res, self.miss = [], []
        for d in self.P:
            sw = d["layer"].mlp.switch_mlp
            bm = sw._bm
            rm = np.array(self.ext.lookup(bm["L"], list(range(512))), np.int32)
            rl = np.array(self.ext.lookup(bm["tier"]["side"], list(range(512))), np.int32)
            self.res.append((mx.array(rm), mx.array(rl)))

    def moe_oracle(self, i, x):
        if "tierglu" not in sys.modules:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "patches", "planes_engine"))
        from tierglu import tier_glu
        mlp = self.P[i]["layer"].mlp
        t = mlp.switch_mlp._bm["tier"]
        shared = mx.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
        gates = mx.softmax(mlp.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
        sc = mx.take_along_axis(gates, inds, axis=-1)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        o = mx.argsort(-sc, axis=-1)
        inds, sc = mx.take_along_axis(inds, o, axis=-1), mx.take_along_axis(sc, o, axis=-1)
        rm, rl = self.res[i]
        sm, sl = rm[inds], rl[inds[:, :2]]
        self.miss.append((sm < 0).sum() + (sl < 0).sum())
        y = tier_glu(x, t["base"], t["sideP"], mx.maximum(sm, 0), mx.maximum(sl, 0), plan="4433333333", scores=sc)
        return y + shared

    def layer(self, i, H, ids, st):
        """one decoder layer: H [1, M, 10240] -> (H1 after the attention write [M, 10240], H after the MoE write [1, M, 10240])"""
        d = self.P[i]
        layer = d["layer"]
        M = H.shape[1]
        cache = st["cache"]
        if "ple" in layer:
            H = H + layer.ple(H, ids, cache[i])
        h = H[0]
        if d["gdn"]:
            cst, ssm = st[i]
            H1, mixed_m, y_m, cst, ssm = G.attn_half(d, h, cst, ssm, M)
            st[i] = (cst, ssm)
        else:
            y1, mixed = KN.hc_read(h, d["attn_hc"]["Wd"], d["attn_hc"]["Wu"], d["attn_hc"]["nw"], M)
            out = layer.self_attn(mixed[None], cache[i])[0]
            H1 = KN.hc_write(h, y1, out, M)
            y_m, mixed_m = KN.hc_read(H1, d["mlp_hc"]["Wd"], d["mlp_hc"]["Wu"], d["mlp_hc"]["nw"], M)
        return H1, KN.hc_write(H1, y_m, self.moe(i, mixed_m), M)[None]

    def head_logits(self, H):
        _, xf = KN.hc_read(H[0], self.fin["Wd"], self.fin["Wu"], self.fin["nw"], H.shape[1])
        return qmm(xf, self.head)

    def forward(self, ids, st, trace=None):
        """ids [1, M] int -> logits [M, vocab]; st from new_state() (updated in place); trace: list to collect per-layer
        (H after the attention write, H after the MoE write) for debugging"""
        H = mx.tile(self.tm.embed_tokens(ids), (1, 1, 4))           # [1, M, 10240]
        for i in range(len(self.P)):
            H1, H = self.layer(i, H, ids, st)
            if trace is not None:
                trace.append((H1, H[0]))
        return self.head_logits(H)
