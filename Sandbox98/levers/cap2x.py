"""Lever 'cap2x' (transfer check 1, CAPACITY; the 125B's answer: BigMoE 24e7758). A zero-init dense SwiGLU branch beside
the head's MLP (same input: post_attention_layernorm(h); WIDTH 6144 = the MLP's own width, so 2x the dense capacity), its
output projection zero, so step 0 is the base exactly. The branch trains on its own Adam at the head's lr (CAPQ's fix of
the 10x-lr spike). Set per drafter instance, so other arms in the same process are untouched."""
import os
import mlx.core as mx
import mlx.nn as nn

WIDTH = int(os.environ.get("CAP_WIDTH", "6144"))
PREFIX, MODULE_LR = "xb_", float(os.environ.get("CAP_LR", "1e-5"))


def setup(D, args):
    D.xb_gate = nn.Linear(D.d, WIDTH, bias=False)
    D.xb_up = nn.Linear(D.d, WIDTH, bias=False)
    D.xb_down = nn.Linear(WIDTH, D.d, bias=False)
    D.xb_down.weight = mx.zeros((D.d, WIDTH), mx.float32)
    base = D.mlp                                           # the class's MLP, bound to this drafter

    def mlp(x):
        return base(x) + D.xb_down(nn.silu(D.xb_gate(x)) * D.xb_up(x))
    object.__setattr__(D, "mlp", mlp)
