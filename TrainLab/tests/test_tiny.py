"""train35.py on a tiny random qwen3_5_moe (CPU): exact flow bitwise = stock model; late/half = defer35's patched
dataflows; checkpointed grads = plain grads; a few training steps lower the loss for each flow."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab"))
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
import train35 as T
import defer35 as D
from mlx_lm.models import qwen3_5_moe, qwen3_5, qwen3_next
from mlx_lm.models.switch_layers import SwitchLinear

mx.random.seed(0)
tc = dict(model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
          head_dim=16, vocab_size=512, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
          linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, num_experts=16,
          num_experts_per_tok=4, moe_intermediate_size=32, shared_expert_intermediate_size=32, norm_topk_prob=True,
          rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
          attn_output_gate=True, tie_word_embeddings=False)
def build():
    mx.random.seed(0)
    m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
    # scale weights up so routing and outputs are not trivially uniform
    m.update(nn.utils.tree_map(lambda a: a * float(os.environ.get("TINY_SCALE", "1.0")) if a.ndim >= 2 else a, m.parameters()))
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
    nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, (nn.Linear, nn.Embedding)) and x.weight.shape[-1] % 32 == 0)
    mx.eval(m.parameters())
    return m
model = build()
ids = mx.array(np.random.default_rng(0).integers(0, 512, size=(1, 97)).astype(np.int32))

def lp_of(h):
    lg = model.language_model.lm_head(h)[0].astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)
stock = model(ids)[0].astype(mx.float32); stock = stock - mx.logsumexp(stock, -1, keepdims=True)
mine = lp_of(T.forward(model, ids, T.Flow("exact"), False)[0])
print("exact eval path vs stock: max |dlogp|", mx.abs(mine - stock).max().item())
mine_tr = lp_of(T.forward(model, ids, T.Flow("exact"), True, ckpt=False)[0])
print("exact train path (chunked GDN) vs stock: max |dlogp|", mx.abs(mine_tr - stock).max().item())

# defer35's patched dataflows on the same model
layers = model.language_model.model.layers
for i, l in enumerate(layers): l.mlp._ni_layer = i
D.P.nlayers = len(layers)
qwen3_next.Qwen3NextSparseMoeBlock.__call__ = D.moe_call
qwen3_5.DecoderLayer.__call__ = D.layer_call
for darm, mine_flow in [("defer1_C0_W0", "late"), ("defer1_C0_W1", "late_w1"), ("half_C0_W1", "half"), ("half_C0_W2", "half_w2")]:
    D.parse(darm); D.P.stats = {}
    ref = D.logprobs(model, ids)
    got = lp_of(T.forward(model, ids[:, :-1], T.Flow(mine_flow), False)[0])
    print(f"{mine_flow:8s} vs defer35 {darm}: max |dlogp| {mx.abs(got - ref).max().item():.3e}   (vs exact {mx.abs(ref - stock[:-1]).max().item():.3f})")

import pol35 as PO
for parm, mine_flow in [("C6_w2_lhalf", "half_w2_C6"), ("C4_w1_l1", "late_w1_C4")]:
    PO.Q["q"], PO.Q["stats"] = PO.parse(parm), {}
    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = PO.moe_call
    ref = D.logprobs(model, ids)
    T.Cache.stats = np.zeros(3)
    got = lp_of(T.forward(model, ids[:, :-1], T.Flow(mine_flow), False)[0])
    print(f"{mine_flow:12s} vs pol35 {parm}: max |dlogp| {mx.abs(got - ref).max().item():.3e}  (vs exact {mx.abs(ref - stock[:-1]).max().item():.3f}); late frac {T.Cache.stats[1] / T.Cache.stats[0]:.3f}")
    T.Cache.stats = None
# LoRA + pre-routers; teacher; checkpointed vs plain gradients
flows = ["half_w2_C6", "late", "half", "prew", "pre", "k2", "late_B4-8", "half_w2_B2-8", "k2_n4"]
nl, ntr = T.prepare(model, ["gdn", "attn", "shared", "router"], 4, 8, flows)
print("LoRA modules", nl, "trainable", ntr)
# make LoRA non-zero so gradients flow everywhere
model.update(nn.utils.tree_map(lambda a: a + 0.01 * mx.random.normal(a.shape) if a.ndim == 2 else a, model.trainable_parameters()))
top, tlp = T.teacher_top(model, ids)
for fl in ["late", "half_w2_B2-8", "prew", "k2_n4"]:
    F = T.Flow(fl)
    g1 = nn.value_and_grad(model, lambda m: T.student_loss(m, ids, top, tlp, F, 1.0, ckpt=True)[0])(model)[1]
    g2 = nn.value_and_grad(model, lambda m: T.student_loss(m, ids, top, tlp, F, 1.0, ckpt=False)[0])(model)[1]
    a, b = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    err = max((mx.abs(a[k] - b[k]).max() / (mx.abs(b[k]).max() + 1e-9)).item() for k in b)
    nz = sum(1 for k in b if mx.abs(b[k]).max().item() > 0)
    print(f"{fl:14s} ckpt vs plain grads: max rel err {err:.2e}; nonzero grads {nz}/{len(b)}")
# training steps
for fl in flows:
    model2 = build(); T.prepare(model2, ["gdn", "attn", "shared"], 4, 8, [fl])
    F = T.Flow(fl)
    top, tlp = T.teacher_top(model2, ids)
    opt = T.make_opt(3e-3, 1e-3)
    lf = nn.value_and_grad(model2, lambda m: T.student_loss(m, ids, top, tlp, F, 1.0))
    ls = []
    for step in range(25):
        if F.C is not None:
            T.Cache.record = {}; hh, _ = T.forward(model2, ids, F, False); mx.eval(hh)
            T.Cache.maps, T.Cache.record = T.Cache.record, None
        (loss, (kd, aux)), g = lf(model2)
        T.Cache.maps = None
        opt.update(model2, g); mx.eval(model2.trainable_parameters(), opt.arrays())
        ls.append((float(kd), float(aux)))
    print(f"train {fl:14s} kd {ls[0][0]:.4f} -> {ls[-1][0]:.4f}   aux {ls[0][1]:.4f} -> {ls[-1][1]:.4f}")
