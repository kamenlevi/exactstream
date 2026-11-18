"""MTP Head lever 'smoe' (WEEK98 item 9 = GRANDMA_1117 C4, the sparse-dense head; 2B analog, REPORT 'WEEK98 S23 SMOE'):
the head's dense MLP (6,144 neurons) becomes a fine-grained MoE of the SAME total parameters. There are SMOE_E (16)
experts of 6144 / E contiguous neurons each, SMOE_K (4) active per token, i.e. 1/4. They are UPCYCLED from the dense
weights: expert e = neuron slice e of gate / up / down, so no new expert weights.
- Router: Linear(d -> E) on its own Adam at SMOE_LR (1e-4; PREFIX 'moer'), init normal x 0.02 (seed 1117).
- Gates: softmax over the top-K logits, times K: a uniform router sums the selected slices unweighted, the closest
  start to the dense output (init relative difference ~0.87; scaling by E gave 1.88).
- The compute is dense: every expert runs and the unselected ones are masked. This tests kept, not speed.
- At inference the 125B analog would read only the K selected experts' rows (~1/4 of the MLP bytes per level).
setup is idempotent: a read-out (evalx / tree8: setup, then the run's weights with strict=False) rebuilds the router
before its trained weights load."""
import os
import types
import mlx.core as mx
import mlx.nn as nn

E, K = int(os.environ.get("SMOE_E", "16")), int(os.environ.get("SMOE_K", "4"))
PREFIX, MODULE_LR = "moer", float(os.environ.get("SMOE_LR", "1e-4"))


def moe_mlp(self, x):
    h = nn.silu(self.gate_proj(x)) * self.up_proj(x)                      # [.., ffn]
    lg = self.moer(x.astype(mx.float32))                                    # [.., E]
    thr = mx.sort(lg, axis=-1)[..., E - K:E - K + 1]                        # the K-th largest logit
    sel = lg >= thr
    g = mx.where(sel, mx.softmax(mx.where(sel, lg, -1e30), axis=-1), 0.0) * K   # uniform gates = 1 each: the selected slices summed (init rel. diff ~0.87 vs ~1.7-1.9 at x E)
    w = mx.repeat(g, self.gate_proj.weight.shape[0] // E, axis=-1)          # each expert's gate on its neurons
    return self.down_proj(h * w.astype(h.dtype))


def setup(D, args):
    assert D.gate_proj.weight.shape[0] % E == 0
    if "moer" not in D:
        mx.random.seed(1117)
        D.moer = nn.Linear(D.d, E, bias=False)
        D.moer.weight = mx.random.normal(D.moer.weight.shape) * 0.02
    D.mlp = types.MethodType(moe_mlp, D)
    mx.eval(D.parameters())


def usage(D, x):
    """the share of tokens routed to each expert (a read-out check against collapse)"""
    lg = D.moer(x.astype(mx.float32))
    thr = mx.sort(lg, axis=-1)[..., E - K:E - K + 1]
    return (lg >= thr).astype(mx.float32).reshape(-1, E).mean(0)
