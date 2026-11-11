"""sticky35.py end to end on a tiny random qwen3_5_moe (CPU, 32 experts top-8 like the 35B's k): training, every
evaluation (caps, bonus, budgets, control, betas, bands), the frontier / law / additivity analysis and the outputs, all
in a scratch TrainLab dir (the real data/ and ckpt/ are never touched). Checks: exact zero-shot KLD 0 with per-layer
error 0; budgets / caps respect their unions; the control sits near 0."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.expanduser("~/TrainLab"))
import numpy as np
import mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
import train35 as T
import sticky35 as S
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.switch_layers import SwitchLinear

tc = dict(model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
          head_dim=16, vocab_size=512, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
          linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, num_experts=32,
          num_experts_per_tok=8, moe_intermediate_size=32, shared_expert_intermediate_size=32, norm_topk_prob=True,
          rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
          attn_output_gate=True, tie_word_embeddings=False)


def build():
    mx.random.seed(0)
    m = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
    m.update(nn.utils.tree_map(lambda a: a * 3.0 if a.ndim >= 2 else a, m.parameters()))
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
    nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, (nn.Linear, nn.Embedding)) and x.weight.shape[-1] % 32 == 0)
    mx.eval(m.parameters())
    return m


tmp = tempfile.mkdtemp(prefix="sticky35_tiny_", dir=os.environ.get("TMPDIR"))
os.makedirs(os.path.join(tmp, "data")); os.makedirs(os.path.join(tmp, "ckpt"))
rng = np.random.default_rng(0)
base = rng.integers(0, 512, size=64)                           # a repeating motif: some temporal structure
stream = np.concatenate([base[rng.integers(0, 64, size=40)] if i % 2 else rng.integers(0, 512, size=40)
                         for i in range(600)]).astype(np.int32)
np.save(os.path.join(tmp, "data/train_tokens.npy"), stream)
T.TL = tmp
S.RES = os.path.join(tmp, "data/sticky35_results.txt")
model = build()
T.setup = lambda *a, **k: (model, None)
segs = [(n, mx.array(stream[o:o + 200][None])) for n, o in (("prose", 1000), ("gsm", 3000), ("code", 5000), ("chat", 7000))]
T.heldout = lambda tok: segs
sys.argv = ["sticky35.py", "run", "--betas", "0.44,8,8r", "--train-min", "1.5", "--steps", "150", "--seq", "128"]
S.main()
out = json.load(open(S.RES.replace(".txt", ".json")))
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


check("zero-shot exact: KLD 0, per-layer err 0", out["zero-shot exact"]["kld"] < 1e-6 and out["zero-shot exact"]["err"] == 0)
check("caps: KLD > 0 and U16 rises with m", all(out[f"zero-shot cap{m}"]["kld"] > 0 for m in range(1, 7))
      and out["zero-shot cap1"]["U16"] < out["zero-shot cap3"]["U16"] < out["zero-shot exact"]["U16"])
check("budget U: U16 <= U", all(out[f"zero-shot budget {U}"]["U16"] <= U + 1e-9 for U in (16, 24)))
check("budget4 U: U4 <= U", all(out[f"zero-shot budget4 {U}"]["U4"] <= U + 1e-9 for U in (12, 16, 20)))
check("control at the tiny model's numerics floor (35B floor 0.010)", out["control (fp32 W0) exact"]["kld"] < 0.2)
check("variants evaluated", all(f"{nm} {f}" in out for nm in ("beta 0.44", "beta 8.0", "beta 8.0r")
                                 for f in ("exact", "cap3", "cap2")))
check("bands evaluated", sum(1 for k in out if "band" in k) == 8)
check("routers saved", all(os.path.exists(os.path.join(tmp, "ckpt", f"sticky35_b{b}.safetensors"))
                          for b in ("0.44", "8.0", "8.0r")))
import re
txt = open(S.RES).read()
kl8 = float(re.search(r"beta 8.0: trust KL ([0-9.]+)", txt).group(1))
kl8r = float(re.search(r"beta 8.0r: trust KL ([0-9.]+)", txt).group(1))
check(f"training moves the routers (held-out trust KL beta 8: {kl8:.4f}, 8r: {kl8r:.4f})", kl8 > 1e-3 and kl8r > 1e-4)
print("ALL PASS" if ok else "SOME FAILED", "| scratch dir", tmp)
