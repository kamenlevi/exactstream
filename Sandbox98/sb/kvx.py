"""BREAK98c PREFIX EYES (Fable FAR, ~/FableGod/far/IDEAS_1117c_prefix_eyes.md section 1; results/LEVERS_A.md section B98C):
the head reads the frozen trunk's OWN keys / values of the verified prefix.

The head's one decoder layer gains a cross-attention block between its self-attention and its MLP (GLIDE's decoder form;
core.block's xa, core.kvx_in). A row with anchor s at depth j, residual h after the self-attention:
    per trunk layer L in LAYERS (full-attention layers 15 and 19 of 24, around 3/4 of the depth):
      q, gate = split(q_x(norm_x(h)))                 the trunk's gated geometry: 8 query heads, head dim 256, gate half
      q = RoPE(q_norm(q), position s + j - 1)         the head's own position for that row (core.rope_rows)
      a = attention(q, K_L[0..s], V_L[0..s])          the trunk's cached keys (normed, RoPE'd) / values, positions <= s
      h += o_x(a * sigmoid(gate))                     o_x zero-initialised: step 0 = the head exactly
norm_x, q_x (with its gate half) and q_norm start from the trunk layer's own input_layernorm, q_proj (dequantized from
4-bit) and q_norm (data/kvx/init.safetensors, written by sb/kvxgen.py from the live trunk), so the read starts from the
trunk's own attention pattern. The cache: data/kvx/<set dir>/<key>.safetensors, K / V [len(LAYERS), KV, S, hd] bf16, from
one capture-shaped forward per file (sb/kvxgen.py). ctl keeps the native head's own attention over its true pairs (the
trunk's FINAL features of the prefix, FAR's caveat); this block adds only the mid-layer K / V.
Levers levers/kvx3.py / kvx10.py; the block trains on its own Adam (PREFIX 'kvx_', MODULE_LR constant after the warm-up)
while the head continues s30o's Adam."""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sb import core as C

LAYERS = (15, 19)
PREFIX = "kvx_"
SPLIT_CLIP = True        # train_multi: the head clipped apart from the block (B98C amendment, before any data)


def kv_path(f):
    """the K / V file for capture file f (data/<dir>/<key>.npz -> data/kvx/<dir>/<key>.safetensors)"""
    return os.path.join(C.DATA, "kvx", os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4] + ".safetensors")


def init_path():
    return os.path.join(C.DATA, "kvx", "init.safetensors")


def make(module_lr, layers=LAYERS):
    """-> (setup, standin, PREFIX, MODULE_LR) for one lever file"""
    st = {"f": None, "K": None, "V": None}

    def setup(D, args):
        training = args is not None and hasattr(args, "steps") and hasattr(args, "arms")
        ini = mx.load(init_path()) if os.path.exists(init_path()) else None
        if training and ini is None:
            raise FileNotFoundError(f"PREFIX EYES: no trunk init {init_path()} (run sb/kvxgen.py first)")
        for L in layers:
            n = nn.RMSNorm(D.d, eps=1e-6)
            q = nn.Linear(D.d, D.nh * D.hd * 2, bias=False)
            qn = nn.RMSNorm(D.hd, eps=1e-6)
            o = nn.Linear(D.nh * D.hd, D.d, bias=False)
            if ini is not None:                       # read-outs: the saved checkpoint overwrites these right after
                n.weight = ini[f"n{L}"].astype(mx.float32)
                q.weight = ini[f"q{L}"].astype(mx.float32)
                qn.weight = ini[f"qn{L}"].astype(mx.float32)
            o.weight = mx.zeros((D.d, D.nh * D.hd), mx.float32)
            setattr(D, f"kvx_n{L}", n); setattr(D, f"kvx_q{L}", q); setattr(D, f"kvx_qn{L}", qn); setattr(D, f"kvx_o{L}", o)

    def _load(f):
        if st["f"] != f:
            p = kv_path(f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"PREFIX EYES: no trunk K / V for {f}: {p}")
            z = mx.load(p)
            st.update(f=f, K=z["K"].astype(mx.float32), V=z["V"].astype(mx.float32))
            assert st["K"].shape[0] == len(layers), (p, st["K"].shape)
            mx.eval(st["K"], st["V"])
        return st["K"], st["V"]

    def kvx(D, j, h, anc):
        """mx ops only (it also runs inside ttt_step's mx.grad): the cross-attention output [1, n, d] for rows h [1, n, d]
        with anchors anc [n] at depth j"""
        K, V = _load(C.CUR["f"])
        x = C.CUR.get("x")
        S = K.shape[2] if x is None else min(K.shape[2], x["W"].shape[1])    # a training window: keys up to its end
        n = h.shape[1]
        mask = mx.arange(S)[None, :] <= anc[:, None]                          # [n, S]: the verified prefix only
        pos = anc + (j - 1)
        out = None
        for i, L in enumerate(layers):
            xn = getattr(D, f"kvx_n{L}")(h)
            q, gate = mx.split(getattr(D, f"kvx_q{L}")(xn).reshape(1, n, D.nh, 2 * D.hd), 2, axis=-1)
            q = C.rope_rows(D, getattr(D, f"kvx_qn{L}")(q).transpose(0, 2, 1, 3), pos)
            a = mx.fast.scaled_dot_product_attention(q, K[i:i + 1, :, :S], V[i:i + 1, :, :S], scale=D.hd ** -0.5, mask=mask)
            y = getattr(D, f"kvx_o{L}")(a.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate.reshape(1, n, -1)))
            out = y if out is None else out + y
        return out

    def standin(j, h, anc):
        return h
    standin._kvx = kvx
    standin._kvx_state = st
    return setup, standin, PREFIX, module_lr
