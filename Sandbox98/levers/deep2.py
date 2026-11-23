"""Lever 'deep2' (T2's pointer: DEPTH of composition). A second decoder layer after the native one, at every depth and
over the true pairs, so the drafter composes the drafted tokens with the context twice. It starts as a copy of the native
layer with its two output projections (o_proj, down_proj) at zero: a pure residual identity, so step 0 = the base drafter
exactly. It trains on its own Adam (MODULE_LR) while the native head continues the base's Adam."""
import os
import mlx.core as mx

PREFIX, MODULE_LR = "xl", float(os.environ.get("DEEP2_LR", "3e-5"))
_KEYS = ("input_layernorm", "post_attention_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
         "gate_proj", "up_proj", "down_proj")


def setup(D, args):
    from sb import core as C
    L = C.Layer()
    P = dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(D.parameters()))
    w = [(k, mx.array(v)) for k, v in P.items() if k.split(".")[0] in _KEYS]
    L.load_weights(w)
    L.o_proj.weight = mx.zeros(L.o_proj.weight.shape, mx.float32)
    L.down_proj.weight = mx.zeros(L.down_proj.weight.shape, mx.float32)
    D.xl = [L]
    object.__setattr__(D, "_stack", [D, D.xl[0]])
