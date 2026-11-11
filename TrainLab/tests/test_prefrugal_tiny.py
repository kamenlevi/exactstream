"""Tiny-model smoke test (CPU): pre-routed frugal (prew_drop_w2_C6), floor, sticky + routerfull: eval, two-pass train
steps (maps), checkpointed vs plain gradients."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.expanduser("~/TrainLab/tests"))
import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
import train35 as T
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.switch_layers import SwitchLinear
from test_tiny_cfg import tc
def build():
    mx.random.seed(0)
    m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
    nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, (nn.Linear, nn.Embedding)) and x.weight.shape[-1] % 32 == 0)
    return m
ids = mx.array(np.random.default_rng(0).integers(0, 512, size=(2, 96)).astype(np.int32))
for fl, targets in [("prew_drop_w2_C6", ["gdn", "attn", "shared"]), ("prew_drop_w2_C6_B2-8", ["gdn", "attn", "shared"]),
                    ("floor", ["gdn"]), ("sticky", ["routerfull"])]:
    m = build(); T.prepare(m, targets, 4, 8, [fl]); F = T.Flow(fl)
    top, tlp = T.teacher_top(m, ids)
    T.Cache.stats = np.zeros(4)
    h, _ = T.forward(m, ids, F, False)
    lg = m.language_model.lm_head(h).astype(mx.float32); lp = lg - mx.logsumexp(lg, -1, keepdims=True)
    kl = T.kl_top(mx.take_along_axis(lp, top, -1), tlp).mean().item()
    st = T.Cache.stats; T.Cache.stats = None
    def lossf(ck):
        if F.C is not None:
            T.Cache.record = {}; hh, _ = T.forward(m, ids, F, False); mx.eval(hh)
            T.Cache.maps, T.Cache.record = T.Cache.record, None
        r = nn.value_and_grad(m, lambda mm: T.student_loss(mm, ids, top, tlp, F, 1.0, ckpt=ck)[0])(m)
        T.Cache.maps = None
        return r
    (l1, g1), (l2, g2) = lossf(True), lossf(False)
    a, b = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    err = max((mx.abs(a[k] - b[k]).max() / (mx.abs(b[k]).max() + 1e-9)).item() for k in b)
    opt = T.make_opt(1e-3, 3e-4)
    for step in range(10):
        l, g = lossf(True); opt.update(m, g); mx.eval(m.trainable_parameters(), opt.arrays())
    print(f"{fl:22s} eval KL {kl:.4f} dropped {st[3] / max(st[0], 1):.3f} | loss ckpt {l1.item():.4f} plain {l2.item():.4f} grad err {err:.1e} | after 10 steps {l.item():.4f}")
