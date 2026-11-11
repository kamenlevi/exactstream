"""One GDN layer of the tiny model: stock (CPU ops path) vs gdn_train (chunked), same bf16 input."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx, mlx.nn as nn, numpy as np
mx.set_default_device(mx.cpu)
import train35 as T
from mlx_lm.models import qwen3_5_moe
from test_tiny_cfg import tc
for scale in (1.0, 3.0):
    mx.random.seed(0)
    m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
    m.update(nn.utils.tree_map(lambda a: a * scale if a.ndim >= 2 else a, m.parameters()))
    m.set_dtype(mx.bfloat16)
    la = m.language_model.model.layers[0].linear_attn
    x = mx.random.normal((1, 97, 64)).astype(mx.bfloat16)
    a = la(x, None, None); b = T.gdn_train(la, x)
    print(f"scale {scale}: GDN out max |d| {mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item():.3e}, max |out| {mx.abs(a).max().item():.3f}, frac differing {(a != b).astype(mx.float32).mean().item():.3f}")
    ids = mx.array(np.random.default_rng(0).integers(0, 512, size=(1, 97)).astype(np.int32))
    def lp(h):
        lg = m.language_model.lm_head(h)[0].astype(mx.float32); return lg - mx.logsumexp(lg, -1, keepdims=True)
    e = lp(T.forward(m, ids, T.Flow("exact"), False)[0]); t = lp(T.forward(m, ids, T.Flow("exact"), True, ckpt=False)[0])
    p = mx.exp(e); kl = (p * (e - t)).sum(-1)
    print(f"   full model exact: eval vs train path max |dlogp| {mx.abs(e - t).max().item():.3e}, mean KL {kl.mean().item():.2e}")
