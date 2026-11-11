"""decode35 on the tiny random model (CPU): batched left-padded cached decode == the full-sequence forward's argmax at
every generated position (exact flow and a cache flow), and B=3 == B=1 per row."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.expanduser("~/TrainLab/tests"))
import numpy as np, mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
import train35 as T, decode35 as DC
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.switch_layers import SwitchLinear
from test_tiny_cfg import tc
mx.random.seed(0)
m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
m.set_dtype(mx.bfloat16)
nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, (nn.Linear, nn.Embedding)) and x.weight.shape[-1] % 32 == 0)
rng = np.random.default_rng(3)
items = [{"key": f"k{i}", "prompt_ids": rng.integers(0, 500, n).tolist(), "max_tokens": 12} for i, n in enumerate((9, 23, 40))]
for fl in ("exact", "half_w2_C6", "drop_w2_C6"):
    F = T.Flow(fl)
    batched = DC.decode_batch(m, F, items, print)
    for it in items:
        single = DC.decode_batch(m, F, [it], print)[it["key"]][0]
        gen = batched[it["key"]][0]
        # teacher-forced check: full forward over prompt + generated, argmax at each generated position
        seq = mx.array(it["prompt_ids"] + gen)[None]
        h, _ = T.forward(m, seq, F, False)
        am = np.array(mx.argmax(m.language_model.lm_head(h)[0].astype(mx.float32), axis=-1))
        P = len(it["prompt_ids"])
        tf = am[P - 1:P - 1 + len(gen)].tolist()
        agree_tf = np.mean([a == b for a, b in zip(tf, gen)])
        print(f"{fl:12s} {it['key']} len {P}: batched == single: {gen == single}; batched vs full-forward argmax agree {agree_tf:.2f}")
