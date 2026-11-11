> (a quote from the project owner, kept private)
tonight only" via the orchestrator): DFlash2 L1-20's body, frozen, borrowing one of the 125B's own experts at each layer.
After drafter layer i (i = 0..4, paired with the 125B's tap layer T_i = 5 / 15 / 25 / 35 / 45, the layers whose streams
DFlash2 already reads through fc), every block position's hidden state h is normed and projected (in_i, 2560 x 2560,
identity init) into the 125B's MoE input space at T_i, routed by the 125B's own router at T_i (its 8-bit gate) over a fixed
resident subset (the NRES experts most used at T_i in Breakthrough's expA traces: bulgarian / chat / code / math, top-10 of
800 tokens each; a stand-in for the C 224 pool), top-1, and that frozen 4-bit expert's SwiGLU output is projected back
(out_i, zero init, so the model starts as L1-20 exactly) and added to h. Trained: norm_i, in_i, out_i (65.5M); experts,
routers and the body stay frozen. The expert pick is a hard argmax (no gradient), the expert's computation passes gradient.
Shared by sticky/e5_train.py (training forward) and sticky/d02b_walk.py (TL_E5: the official path, for the walks)."""
import glob, json, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

TAPL = (5, 15, 25, 35, 45)
MD = os.path.expanduser("~/BigMoE/models/bare")
NRES = int(os.environ.get("E5_NRES", "224"))
D, GS = 2560, 64


def resident_sets(n=NRES):
    cnt = {L: np.zeros(512, np.int64) for L in TAPL}
    for f in sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz")):
        z = np.load(f)
        for L in TAPL:
            np.add.at(cnt[L], z[f"L{L}"][:, :10].astype(np.int64).ravel(), 1)
    return {L: np.sort(np.argsort(-cnt[L], kind="stable")[:n]) for L in TAPL}


def load_experts(sets):
    """per tap layer: the subset's stacked 4-bit gate / up / down (weight, scales, biases) and the 8-bit router (rows of
    the subset only), as plain arrays outside any module (never trained, never saved)"""
    idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    keys = {}
    for L in TAPL:
        for p in ("switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj", "gate"):
            for s in ("weight", "scales", "biases"):
                k = f"language_model.model.layers.{L}.mlp.{p}.{s}"
                keys.setdefault(idx[k], []).append(k)
    W = {}
    for shard, ks in keys.items():
        w = mx.load(os.path.join(MD, shard))
        W.update({k: w[k] for k in ks})
        del w
    X = {}
    for L in TAPL:
        sub = mx.array(sets[L].astype(np.int32))
        e = {f"{p}.{s}": mx.take(W[f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{s}"], sub, axis=0)
             for p in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")}
        e.update({f"router.{s}": mx.take(W[f"language_model.model.layers.{L}.mlp.gate.{s}"], sub, axis=0)
                  for s in ("weight", "scales", "biases")})
        X[L] = e
    mx.eval(X)
    del W
    return X


class XBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = [nn.RMSNorm(D, eps=1e-6) for _ in TAPL]
        self.ins = [nn.Linear(D, D, bias=False) for _ in TAPL]
        self.outs = [nn.Linear(D, D, bias=False) for _ in TAPL]
        for m in self.ins:
            m.weight = mx.eye(D)
        for m in self.outs:
            m.weight = mx.zeros((D, D))

    def __call__(self, i, h):
        """h (..., D) -> the branch's addition to h (same shape and dtype)"""
        e = EXP[TAPL[i]]
        shp, dt = h.shape, h.dtype
        x = self.ins[i](self.norms[i](h.reshape(-1, D).astype(mx.float32)))
        xb = x.astype(mx.bfloat16)
        lg = mx.quantized_matmul(xb, e["router.weight"], e["router.scales"], e["router.biases"], transpose=True,
                                 group_size=GS, bits=8)
        j = mx.stop_gradient(mx.argmax(lg, axis=-1)).astype(mx.uint32)[:, None]   # top-1 over the subset
        if STATS is not None:
            mx.eval(j)
            np.add.at(STATS[i], np.array(j[:, 0]).astype(np.int64), 1)
        xi = xb[:, None, None, :]
        g = mx.gather_qmm(xi, e["gate_proj.weight"], e["gate_proj.scales"], e["gate_proj.biases"], rhs_indices=j,
                          transpose=True, group_size=GS, bits=4)
        u = mx.gather_qmm(xi, e["up_proj.weight"], e["up_proj.scales"], e["up_proj.biases"], rhs_indices=j,
                          transpose=True, group_size=GS, bits=4)
        y = mx.gather_qmm(nn.silu(g) * u, e["down_proj.weight"], e["down_proj.scales"], e["down_proj.biases"],
                          rhs_indices=j, transpose=True, group_size=GS, bits=4)
        return self.outs[i](y.reshape(-1, D).astype(mx.float32)).reshape(shp).astype(dt)


EXP = {}                                                       # tap layer -> frozen expert / router arrays (attach())
STATS = None                                                   # optional: [per tap np.int64 counts], expert picks


def attach(xb, sets=None):
    if not EXP:
        EXP.update(load_experts(sets or resident_sets()))
    return xb


_OFF = {}


def install_official(draft, xb):
    """the official path (DFlash2DecoderLayer.__call__ via draft.hidden_states): h + branch after each layer"""
    import dflash_model_mlx as DF
    _OFF.update({id(L): i for i, L in enumerate(draft.layers)})
    _OFF["xb"] = xb
    cls = type(draft.layers[0])
    if not getattr(cls, "_e5_patched", False):
        orig = cls.__call__
        def call(self, *a, **kw):
            h = orig(self, *a, **kw)
            i = _OFF.get(id(self))
            return h if i is None else h + _OFF["xb"](i, h)
        cls.__call__ = call
        cls._e5_patched = True


def load_branch(path):
    """an XBranch from e5_train.py's saved weights (keys 'xb.<...>')"""
    xb = XBranch()
    w = mx.load(os.path.expanduser(path))
    xb.load_weights([(k[3:], v.astype(mx.float32)) for k, v in w.items() if k.startswith("xb.")], strict=True)
    mx.eval(xb.parameters())
    return attach(xb)
