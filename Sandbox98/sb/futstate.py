"""BREAK98 Screen 1 (Fable FAR, ~/FableGod/far/IDEAS_1117_futurestates.md, "Screen 1"): FUTURE-STATE ROWS on the 2B.

The frozen 2B runs [prefix .. anchor s] + K = 8 PLACEHOLDER rows (a chain at positions s+1 .. s+8, no tokens), and the
head reads the target's post-norm state at placeholder offset j as a third input stream at its depth j <= K:
    x_j = fc([norm(emb), norm(h)]) + P(norm_f(W_fut[s, j]))         P zero-initialised (step 0 = the head exactly)
Depth j predicts ids[s + j + 1]; offset j sits at position s + j, whose own lm_head output targets the same token (FAR's
runtime map: the head at depth j, predicting position m + j, reads offset j under row m - 1 = the anchor). Past depth K,
and at anchors without stored states, the input is zero, i.e. the head runs exactly as today. At depth 1 only the
anchors' query rows read it; the keys / values of the true pairs stay future-free (core.depth1).

Variants (sb/futgen.py writes data/fut_V/<set dir>/<key>.safetensors: anc [n] int32, W [n, K, d] bf16, top1 [n, K]):
  A  <|fim_pad|> (id 248063, the 125B's id too)    B  the mean of the input-embedding table    C  <|vision_pad|> (248055)
Levers levers/futA.py / futB.py / futC.py; P and norm_f train on their own Adam (PREFIX 'fut_', MODULE_LR, constant after
the warm-up, like tapmlp / cap2x) while the head continues s30o's Adam."""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sb import core as C

K = 8
PREFIX = "fut_"
MODULE_LR = float(os.environ.get("FUT_LR", "1e-5"))


def fut_path(variant, f):
    """the future-state file for capture file f (data/<dir>/<key>.npz -> data/fut_V/<dir>/<key>.safetensors)"""
    return os.path.join(C.DATA, "fut_" + variant, os.path.basename(os.path.dirname(f)),
                        os.path.basename(f)[:-4] + ".safetensors")


def make(variant, k=K):
    """-> (setup, standin, PREFIX, MODULE_LR) for one lever file; state is per variant (one lever module each)"""
    st = {"f": None, "map": None, "W": None}

    def setup(D, args):
        D.fut_norm = nn.RMSNorm(D.d, eps=1e-6)
        D.fut_norm.weight = mx.array(np.array(D.pre_fc_norm_hidden.weight))   # the W stream's gain (a copy)
        D.fut_fc = nn.Linear(D.d, D.d, bias=False)
        D.fut_fc.weight = mx.zeros((D.d, D.d), mx.float32)

    def _load(f):
        if st["f"] != f:
            p = fut_path(variant, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"no future states for {f} (variant {variant}): {p}")
            z = mx.load(p)
            anc = np.array(z["anc"]).astype(np.int64)
            W = z["W"]
            m = np.full(int(anc.max()) + 2 if len(anc) else 1, len(anc), np.int32)   # anchors without states -> row n
            m[anc] = np.arange(len(anc))
            st.update(f=f, map=mx.array(m), W=mx.concatenate([W, mx.zeros((1,) + W.shape[1:], W.dtype)], 0))  # row n = 0
            mx.eval(st["map"], st["W"])
        return st["map"], st["W"]

    def fut(D, j, anc):
        """mx ops only (it also runs inside ttt_step's mx.grad)"""
        if j > k:
            return None
        m, W = _load(C.CUR["f"])
        idx = m[mx.minimum(anc, m.shape[0] - 1)]                         # past the map's end -> its last entry = row n
        rows = W[idx, j - 1].astype(mx.float32)                          # [n, d]; missing anchors -> zeros -> input 0
        return D.fut_fc(D.fut_norm(rows))[None]

    def standin(j, h, anc):
        return h
    standin._fut = fut
    standin._fut_state = st
    return setup, standin, PREFIX, MODULE_LR
