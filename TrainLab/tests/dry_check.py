"""(1) train-path loss on a held-out segment == eval-path KLD on it? (2) lower lr: does held-out KLD fall?"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.expanduser("~/TrainLab/tests"))
import numpy as np, mlx.core as mx, mlx.nn as nn
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
# structured data: a repeating-ish Markov text so there is something to learn (random ids are structureless)
rng = np.random.default_rng(0)
P = rng.dirichlet(np.full(512, 0.05), size=512)
def text(n, seed):
    r = np.random.default_rng(seed); x = [0]
    for _ in range(n - 1): x.append(r.choice(512, p=P[x[-1]]))
    return np.array(x, np.int32)
stream = text(30000, 1)
held = mx.array(text(257, 2))[None]
for flow, lr in [("half", 3e-4), ("half", 1e-4)]:
    model = build(); T.prepare(model, ["gdn", "attn", "shared"], 4, 8, [flow]); F = T.Flow(flow)
    T.LoRA.on = False; ref_lp = None
    top_h, tlp_h = T.teacher_top(model, held)
    T.LoRA.on = True
    def held_eval():
        T.Cache.stats = np.zeros(3) if F.C is not None else None
        lp = T.logprobs(model, mx.concatenate([held, held[:, :1]], axis=1), F)
        T.Cache.stats = None
        q = mx.take_along_axis(lp, top_h[0], axis=-1)
        return T.kl_top(q, tlp_h[0]).mean().item()
    def train_path_loss():
        if F.C is not None:
            T.Cache.record = {}; hh, _ = T.forward(model, held, F, False); mx.eval(hh)
            T.Cache.maps, T.Cache.record = T.Cache.record, None
        l = T.student_loss(model, held, top_h, tlp_h, F, 0.0)[1][0].item(); T.Cache.maps = None
        return l
    print(f"{flow} lr {lr}: step0 eval-path KLD {held_eval():.4f}  train-path loss {train_path_loss():.4f}")
    opt = T.make_opt(lr, lr / 3)
    lf = nn.value_and_grad(model, lambda m, ids, top, tlp: T.student_loss(m, ids, top, tlp, F, 1.0))
    for step in range(40):
        o = np.random.default_rng(10 ** 6 + step).integers(0, len(stream) - 129, size=2)
        ids = mx.array(np.stack([stream[a:a + 128] for a in o]))
        top, tlp = T.teacher_top(model, ids)
        if F.C is not None:
            T.Cache.record = {}; hh, _ = T.forward(model, ids, F, False); mx.eval(hh)
            T.Cache.maps, T.Cache.record = T.Cache.record, None
        (loss, (kd, aux)), g = lf(model, ids, top, tlp); T.Cache.maps = None
        opt.update(model, g); mx.eval(model.trainable_parameters(), opt.arrays())
        if step % 10 == 9:
            print(f"   step {step + 1}: train kd {kd.item():.4f}  held eval {held_eval():.4f}  held train-path {train_path_loss():.4f}")
