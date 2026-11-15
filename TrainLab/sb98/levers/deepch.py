"""MTP Head lever 'deepch' (WEEK98 screen 1, depth done properly; REPORT 'WEEK98 S1 DEPTHC'). Loaded by Sandbox98's
sb/train_multi.py and sb/evalx.py through an absolute lever path (their scripts run unmodified).

A second decoder layer after the native one, at every depth and over the true pairs (Sandbox98's deep2 placement), started
as a FULL COPY of the native layer: o_proj and down_proj are NOT zeroed, so every weight of the new layer gets a gradient
from step 0 (deep2's zero start grew to |w| ~0.0025 in 400 steps). The new layer trains on its own Adam at MODULE_LR
(train_multi: constant after the warm-up); the native layer keeps the base schedule (1e-5 cosine to 1e-6).
Step 0 is the native layer applied twice (untied), not the native head."""
import os
import mlx.core as mx
from mlx.utils import tree_flatten

PREFIX, MODULE_LR = "xl", float(os.environ.get("DEEPCH_LR", "3e-5"))
_KEYS = ("input_layernorm", "post_attention_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
         "gate_proj", "up_proj", "down_proj")


def setup(D, args):
    from sb import core as C
    L = C.Layer()
    P = dict(tree_flatten(D.parameters()))
    L.load_weights([(k, mx.array(v)) for k, v in P.items() if k.split(".")[0] in _KEYS])
    D.xl = [L]
    object.__setattr__(D, "_stack", [D, D.xl[0]])
