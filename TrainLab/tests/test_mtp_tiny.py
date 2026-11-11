"""mtp35.unrolled (training-time test, masks + RoPE offsets) == the head's own cached chain (inference) on a tiny
random MTP head (CPU); gradients flow to fc/attention/router/shared."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.expanduser("~/TrainLab/tests"))
import numpy as np, mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
import mtp35 as M, train35 as T
from mlx_lm.models import qwen3_5
from mlx_lm.models import cache as cache_mod
from test_tiny_cfg import tc
args = qwen3_5.TextModelArgs.from_dict(tc)
mx.random.seed(0)
head = M.SD.MTP(args)
embed = nn.Embedding(512, args.hidden_size)
Tn, K = 40, 5
h = mx.random.normal((1, Tn, args.hidden_size))
tok = mx.array(np.random.default_rng(0).integers(0, 512, (1, Tn)).astype(np.int32))
feats = M.unrolled(head, embed, h, tok, K)
worst = 0.0
for i in (0, 7, 20, 33):
    mc = cache_mod.make_prompt_cache(head)
    if i > 0:
        head(h[:, :i], embed(tok[:, 1:i + 1]), mc)
    x = head(h[:, i:i + 1], embed(tok[:, i + 1:i + 2]), mc)
    for j in range(1, K + 1):
        if i + j + 1 > Tn - 1:
            break
        if j > 1:
            x = head(x, embed(tok[:, i + j:i + j + 1]), mc)
        d = mx.abs(x[0, 0] - feats[j - 1][0, i]).max().item() / (mx.abs(x).max().item() + 1e-9)
        worst = max(worst, d)
        print(f"chain {i:2d} depth {j}: rel diff {d:.2e}")
print("WORST", worst)
# gradients
top = mx.array(np.random.default_rng(1).integers(0, 512, (1, Tn, 8)).astype(np.int32))
tlp = mx.log(mx.softmax(mx.random.normal((1, Tn, 8)), -1)) - 0.1
class LH:  # fake quantized head interface for draft_logits: use a quantized Linear over 512 ids
    pass
q = nn.QuantizedLinear(args.hidden_size, 512, bias=False, group_size=32, bits=8)
M.V = 512
lossf = nn.value_and_grad(head, lambda hd: M.kd_depths(hd, q, M.unrolled(hd, embed, h, tok, 3), top, tlp)[0])
l, g = lossf(head)
from mlx.utils import tree_flatten
nz = [k for k, v in tree_flatten(g) if mx.abs(v).max().item() > 0]
print("loss", l.item(), "nonzero grad tensors", len(nz), "of", len(tree_flatten(g)))
