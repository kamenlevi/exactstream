"""MTP Head lever 'mlpw' (WEEK98 S20 SMALLHEAD, the draft line): a SMALLER sandbox head. The MLP (6,144 neurons) is cut
to MLPW neurons (default 1536). The kept neurons are native's most important ones, by |down column| x |up row| norm,
in index order (a pruned init, no new weights). Everything else is the native head.
Params: 60.8 M -> 32.5 M (MLPW 1536) or 27.8 M (768). setup is idempotent: an already-cut head is left alone, so a
read-out (evalx / tree8: setup, then the run's own weights with strict=False) rebuilds the cut shape first."""
import os
import mlx.core as mx
import mlx.nn as nn

W = int(os.environ.get("MLPW", "1536"))


def setup(D, args):
    g, u, d = D.gate_proj, D.up_proj, D.down_proj
    if g.weight.shape[0] == W:
        return
    imp = mx.linalg.norm(d.weight.astype(mx.float32), axis=0) * mx.linalg.norm(u.weight.astype(mx.float32), axis=1)
    keep = mx.sort(mx.argsort(-imp)[:W])
    din = g.weight.shape[1]
    ng, nu, nd = nn.Linear(din, W, bias=False), nn.Linear(din, W, bias=False), nn.Linear(W, d.weight.shape[0], bias=False)
    ng.weight, nu.weight, nd.weight = g.weight[keep], u.weight[keep], d.weight[:, keep]
    D.gate_proj, D.up_proj, D.down_proj = ng, nu, nd
    mx.eval(D.parameters())
