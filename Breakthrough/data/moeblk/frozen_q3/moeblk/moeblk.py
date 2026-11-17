"""moeblk.py (Breakthrough, Tue 17 Nov 2026 11:5x; MOE BLOCK proof, docs/MOEBLOCK.md): the MoE block of one layer as a
TEMPLATE. Every model number comes from config/<model>.json; nothing here is 125B-specific.
  load_cfg(path) -> cfg dict
  Layer(cfg, L, slot0)    one layer's router + shared-expert weights (the model's safetensors) and its routed expert pools
                          (BigMoE's loader: base MSB pool + LSB sidecar, all experts resident, as in the lab windows)
  chain_today(lay, x)     TODAY's chain, built from the engine's real components:
      router logits = MLX quantized_matmul (cfg router bits / group size); routing = Leap's router_fused (bitwise ==
      MLX's softmax / argpartition / normalize / argsort chain); 1c tiers + km path + 2-D nolsb = Leap's t1cf.fused
      (c25's C25_T1CF glue); slots = device lookups (the posts' result when every expert is resident; the post kernels
      are outside both arms); routed experts = ooo1cq's one-call through coh1cq (the live COH call; the knob wraps it);
      combine = combine_phase(y, y, 1, scores'); shared = y_s * sigmoid(gate) with y_s = down(silu(gate_p(x)) * up(x))
      (MLX quantized_matmul, cfg shared bits); out = combine + shared.
  block_bytes(lay, ids, nolsb) -> bytes the block must read for this call (the pre-registered floor's numerator)"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("~/BigMoE", "~/Leap/third_party/ms_gdn", "~/Leap/scripts", "~/Leap/scripts/week98"):
    _p = os.path.expanduser(_p)
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("BIGMOE_MAXK", "320")
import mlx.core as mx                                       # noqa: E402
import ref2                                                 # noqa: E402

if not getattr(ref2, "_MOEBLK_APPLIED", False):
    ref2.apply()
    ref2._MOEBLK_APPLIED = True
import bigmoe.gated as BG                                   # noqa: E402
import bigmoe.tier as T                                     # noqa: E402
import coh1cq as CQ                                         # noqa: E402
import router_fused as RF                                   # noqa: E402
import t1cf as T1                                           # noqa: E402


def load_cfg(path=None):
    path = path or os.path.join(HERE, "config", "q125b.json")
    c = json.load(open(path))
    c["model_dir"] = os.path.expanduser(c["model_dir"])
    return c


def _sh4(c):
    """the routed pools' piece shapes for BigMoE's make_pools: (out, in) per projection at 4-bit packing"""
    H, I = c["hidden"], c["expert_inter"]
    gs = c["group_size"]
    sh = []
    for o, i in ((I, H), (I, H), (H, I)):
        sh += [((o, i // 8), mx.uint32), ((o, i // gs), mx.bfloat16), ((o, i // gs), mx.bfloat16)]
    return sh


class Layer:
    def __init__(self, cfg, L, slot0=0):
        self.c, self.L = cfg, L
        c = cfg
        mdir = c["model_dir"]
        wmap = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]
        pre = c["weight_prefix"].format(L=L)
        names = [c["router"]["tensor"]] + c["shared"]["tensors"] + [c["shared"]["gate_tensor"]]
        need = {pre + n + "." + f for n in names for f in ("weight", "scales", "biases")}
        shards = sorted({wmap[k] for k in need})
        W = {}
        for sh in shards:
            d = mx.load(os.path.join(mdir, sh))
            for k in need:
                if k in d:
                    W[k[len(pre):]] = d[k]
            del d
        mx.eval(list(W.values()))
        self.W = W
        self.wbytes = {n: sum(int(W[n + "." + f].nbytes) for f in ("weight", "scales", "biases")) for n in names}
        # routed pools (BigMoE's loader; all experts resident)
        T.PLAN = c["routed_store"]["plan"]
        ext = BG.ext
        d_, ix = T.load_index(c["model_dir"], layout_ok=True)
        r = ix["layers"][str(L)]
        assert r.get("layout") == c["routed_store"]["format"], (r.get("layout"), c["routed_store"]["format"])
        E = c["n_experts"]
        self.base, _ = T.make_pools(ext, slot0, E, E, _sh4(c), d_, r)
        self.Ls, self.side, _ = T.make_sidecar(ext, E, E, _sh4(c), d_, r)
        allids = np.arange(E, dtype=np.int32)
        for j in range(0, E, 64):
            ext.ensure(slot0, mx.array(allids[j:j + 64]))
            ext.ensure(self.Ls, mx.array(allids[j:j + 64]))
        self.sm_of = np.array(ext.lookup(slot0, allids.tolist()), np.int32)
        self.sl_of = np.array(ext.lookup(self.Ls, allids.tolist()), np.int32)
        assert (self.sm_of >= 0).all() and (self.sl_of >= 0).all()
        self.sm_dev, self.sl_dev = mx.array(self.sm_of), mx.array(self.sl_of)
        self.msb_b, self.lsb_b = sum(r["msb_piece_size"]), sum(r["lsb_piece_size"])
        self.region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        self.offs = mx.zeros((3,), mx.int32)
        mx.eval(self.sm_dev, self.sl_dev, self.region, self.status, self.offs)
        self.next_slot = self.Ls + 1

    def qmm(self, x, name, bits):
        W = self.W
        return mx.quantized_matmul(x, W[name + ".weight"], W[name + ".scales"], W[name + ".biases"], transpose=True,
                                   group_size=self.c["group_size"], bits=bits)


def route(lay, x):
    """logits -> (ids [M, K] uint32, scores [M, K] bf16) with Leap's exact fused router"""
    c = lay.c
    logits = lay.qmm(x, c["router"]["tensor"], c["router"]["bits"])
    return RF.router_fused(logits, c["top_k"])


def glue(lay, ids, scores):
    """t1cf: (scores' (dropped -> 0), ids' (dropped -> the row's rank-0 id), nolsb [M, R4] bool)"""
    t = lay.c["tiers_1c"]
    sc, idn, nl, cnt, tier = T1.fused(ids, scores, s=t["s"], lam=t["lam"])
    return sc, idn, nl


def slots(lay, idn, nl):
    """the posts' result with every expert resident: pre_m [M, K], pre_l [M, R4] (-2 where nolsb)"""
    R4 = lay.c["routed_store"]["plan"].count("4")
    pm = mx.take(lay.sm_dev, idn.astype(mx.int32))
    pl = mx.where(nl, mx.array(-2, mx.int32), mx.take(lay.sl_dev, idn[:, :R4].astype(mx.int32)))
    return pm, pl


def shared_expert(lay, x):
    c = lay.c
    s = c["shared"]
    gp, up, dp = s["tensors"]
    g = lay.qmm(x, gp, s["bits"])
    u = lay.qmm(x, up, s["bits"])
    h = (g * mx.sigmoid(g)) * u                              # silu(g) * u
    ys = lay.qmm(h, dp, s["bits"])
    sg = lay.qmm(x, s["gate_tensor"], s["gate_bits"])
    return ys * mx.sigmoid(sg)


def chain_today(lay, x):
    """TODAY's MoE block (see the module doc). x [M, hidden] bf16 -> [M, hidden] bf16"""
    import bigmoe.tier as BT
    c = lay.c
    M, K = x.shape[0], c["top_k"]
    shared = shared_expert(lay, x)
    ids, scores = route(lay, x)
    sc, idn, nl = glue(lay, ids, scores)
    pm, pl = slots(lay, idn, nl)
    y, _, _ = CQ.tier_glu_nax6n_1cq_coh(x, lay.base, lay.side, pm, pl, idn.astype(mx.int32), c["routed_store"]["plan"],
                                        lay.region, lay.offs, lay.status)
    TG = BT._tier_glu_mod()
    yc = TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc.reshape(-1, K))
    return yc + shared


def routing_np(lay, x):
    """host copy of the routing for byte accounting and checks (evaluates; call outside timed regions)"""
    ids, scores = route(lay, x)
    sc, idn, nl = glue(lay, ids, scores)
    mx.eval(ids, scores, sc, idn, nl)
    return np.array(ids), np.array(scores.astype(mx.float32)), np.array(sc.astype(mx.float32)), np.array(idn), np.array(nl)


def block_bytes(lay, idn, nl):
    """the floor's numerator for one call: kept routed experts' MSB-3 planes + LSB planes where a 4-bit use reads them,
    + the router + the shared expert (+ its gate). idn / nl from routing_np."""
    c = lay.c
    R4 = c["routed_store"]["plan"].count("4")
    ex = np.unique(idn)
    lsb = np.unique(idn[:, :R4][~nl])
    routed = len(ex) * lay.msb_b + len(lsb) * lay.lsb_b
    other = sum(lay.wbytes.values())
    return routed + other, routed, other
