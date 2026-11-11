"""Dry run of train35.cmd_train / cmd_eval on the tiny random model (CPU): setup, own reference, evals, teacher cache,
checkpoint save + resume. Usage: dry_train.py SCRATCH_DIR"""
import os, sys, types
sys.path.insert(0, os.path.expanduser("~/TrainLab")); sys.path.insert(0, os.path.expanduser("~/TrainLab/tests"))
import numpy as np, mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
import train35 as T
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.switch_layers import SwitchLinear
from test_tiny_cfg import tc
d = sys.argv[1]
os.makedirs(os.path.join(d, "data"), exist_ok=True); os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
T.TL = d; T.RES = os.path.join(d, "data/results.txt")
np.save(os.path.join(d, "data/train_tokens.npy"), np.random.default_rng(0).integers(0, 512, 20000).astype(np.uint32))
def build(path, lazy=True):
    mx.random.seed(0)
    m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
    nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, (nn.Linear, nn.Embedding)) and x.weight.shape[-1] % 32 == 0)
    return m, None
T.load = build
T.D.to4bit = lambda m: 0
rng = np.random.default_rng(5)
segs = [(n, mx.array(rng.integers(0, 512, 129).astype(np.int32))[None]) for n in ("prose", "gsm", "code", "multi_bg")]
T.heldout = lambda tok: segs
class A: pass
def run(flow, run, tokens):
    a = A(); a.flow, a.run, a.tokens, a.seq, a.batch, a.lr, a.rank = flow, run, tokens, 64, 2, 3e-3, 4
    a.targets, a.eval_every, a.aux, a.time_limit, a.lr_pre, a.duty, a.clip, a.final_evals = "gdn,attn,shared", 1024, 1.0, 0, 1e-3, 1.0, 1.0, ""
    T.cmd_train(a)
T.cmd_eval(["exact", "half_w2_C6", "prew", "k2_n4"])
for flow in ("half_w2_C6", "prew"):
    run(flow, "dry_" + flow, 2048)
    run(flow, "dry_" + flow, 4096)        # resume
print(open(T.RES).read())
print(sorted(os.listdir(os.path.join(d, "ckpt"))), len(os.listdir(os.path.join(d, "data/teacher"))))
