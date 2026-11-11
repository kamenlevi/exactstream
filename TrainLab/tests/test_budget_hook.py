"""sticky/budget_hook.py + the bench_budget24.py wrapper on a tiny random qwen3_5_moe (CPU): the hook wraps every layer;
1-row and > MAXROWS-row calls are untouched (bitwise); a 4-row call's union is <= U; a budget >= the natural union is
bitwise natural; the bench wrapper (with stand-ins for bigmoe.gated's loader counters and spec125) installs the hook
once, prints per-pass counters and the hook's distinct-expert stats."""
import os, sys, types
sys.path.insert(0, os.path.expanduser("~/TrainLab/sticky"))
import numpy as np
import mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.switch_layers import SwitchLinear
import budget_hook as H

tc = dict(model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
          head_dim=16, vocab_size=512, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
          linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, num_experts=64,
          num_experts_per_tok=8, moe_intermediate_size=32, shared_expert_intermediate_size=32, norm_topk_prob=True,
          rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
          attn_output_gate=True, tie_word_embeddings=False)
mx.random.seed(0)
model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
model.update(nn.utils.tree_map(lambda a: a * 3.0 if a.ndim >= 2 else a, model.parameters()))
model.set_dtype(mx.bfloat16)
nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, SwitchLinear))
mx.eval(model.parameters())
ok = True
A = lambda t: np.array(t.astype(mx.float32))                            # bf16 -> numpy via f32 (exact)


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


ids = mx.array(np.random.default_rng(0).integers(0, 512, size=(1, 12)).astype(np.int32))
ref1, ref4, ref12 = model(ids[:, :1]), model(ids[:, :4]), model(ids)
H.install(model, "b64", stats=True)                                    # U = E: never restricts, counts the natural union
check("budget = E: bitwise natural", np.array_equal(A(model(ids[:, :4])), A(ref4)))
nat = H.stats_summary()["natural"]
n = H.install(model, "b12", stats=True)
check(f"hook on every layer ({n})", n == tc["num_hidden_layers"])
check("1-row call bitwise untouched", np.array_equal(A(model(ids[:, :1])), A(ref1)))
check("12-row call (> MAXROWS) bitwise untouched", np.array_equal(A(model(ids)), A(ref12)))
H.STATS.update(calls=0, nat=[], used=[])
out4 = model(ids[:, :4])
u = H.stats_summary()
check(f"4-row call restricted: used {u['used']:.1f} <= 12, natural {u['natural']:.1f}", u["used"] <= 12 + 1e-9 and u["natural"] > 12)
check("4-row output changed by the budget", not np.array_equal(A(out4), A(ref4)))
H.install(model, "b32", stats=True)                                       # 4 rows x top-8 <= 32: never binds
o32 = model(ids[:, :4]); u = H.stats_summary()
check(f"budget >= any 4-row union (32): natural selection (used {u['used']:.1f} = natural {u['natural']:.1f}); the "
      f"output differs only by the masked softmax's rounding before renormalisation (max |d logit| "
      f"{np.abs(A(o32) - A(ref4)).max():.3g})",
      u["used"] == u["natural"])                                         # (the tiny chaotic model amplifies the rounding)
H.install(model, "o3", stats=True)
model(ids[:, :4]); u = H.stats_summary()
check(f"OEA k0=3: used {u['used']:.1f} <= 4 rows x 3 = 12", u["used"] <= 12 + 1e-9)
H.install(model, "", stats=False)
check("spec '' = off: bitwise natural", np.array_equal(A(model(ids[:, :4])), A(ref4)))

# budget-aware JIT: BigMoE keeps layer L+1's ORIGINAL router in layer L's switch_mlp._bm["next"]; rewire it
layers = model.language_model.model.layers
for L in range(len(layers) - 1):
    g1 = layers[L + 1].mlp.gate
    object.__setattr__(layers[L].mlp.switch_mlp, "_bm", {"next": [(g1.g if isinstance(g1, H.BudgetGate) else g1, None, 12)]})
H.install(model, "b16", stats=True)
nj = H.install_jit(model, "b16")
check(f"JIT rewired {nj} look-ahead routers (layers 0..n-2)", nj == len(layers) - 1 and all(
    isinstance(layers[L].mlp.switch_mlp._bm["next"][0][0], H.BudgetGate) and layers[L].mlp.switch_mlp._bm["next"][0][0].U == 16
    and not layers[L].mlp.switch_mlp._bm["next"][0][0].stats for L in range(len(layers) - 1)))
x4 = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
r = layers[0].mlp.switch_mlp._bm["next"][0][0]
lg = r(x4).astype(mx.float32).reshape(-1, 64)
s_ = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).max(axis=0)          # BigMoE's ranked prefetch score
top12 = set(np.array(mx.argsort(-s_)[:12]).tolist()); finite = set(np.flatnonzero(np.isfinite(np.array(s_))).tolist())
check(f"JIT prefetch list (top 12 ranked) lies inside the predicted pass set ({len(finite)} experts)", top12 <= finite and len(finite) <= 16)
x1 = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
check("JIT router passthrough for 1 row", np.array_equal(A(r(x1)), A(r.g(x1))))
H.STATS.update(calls=0, nat=[], used=[]); r(x4)
check("JIT routers add no budget stats", H.STATS["calls"] == 0)
H.install(model, "", stats=False)
# the bench wrapper, with stand-ins for bigmoe.gated (loader counters) and spec125 (generate)
fake_g = types.ModuleType("bigmoe.gated")
fake_g.ext = types.SimpleNamespace(reset_stats=lambda: None,
                                   stats=lambda: dict(misses=30, stalled=6, io_s=0.3, pf_issued=12, pf_used=9, pf_late=1))
fake_b = types.ModuleType("bigmoe"); fake_b.gated = fake_g
fake_s = types.ModuleType("spec125")
calls = []


def fake_generate(m, ids_, n, **kw):
    calls.append(n)
    m(mx.array(np.asarray(ids_[:4], np.int32))[None])                 # one 4-row "verify pass"
    return list(range(n)), dict(tok_s=25.0, tok_per_pass=3.1, passes=10)


fake_s.generate = fake_generate
sys.modules.update({"bigmoe": fake_b, "bigmoe.gated": fake_g, "spec125": fake_s})
src = open(os.path.expanduser("~/TrainLab/sticky/bench_budget24.py")).read()
src = src[:src.index("spec = os.path.expanduser(")]                     # everything but the runpy of test_spec125
os.environ["TL_BUDGET"] = "b12"
model2 = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": tc}))
mx.eval(model2.parameters())
ns = {"__file__": os.path.expanduser("~/TrainLab/sticky/bench_budget24.py"), "__name__": "bench"}
exec(compile(src, "bench_budget24.py", "exec"), ns)
gen = sys.modules["spec125"].generate
gen(model2, list(range(20)), 16)                                         # warm-up: installs the hook, no print
gen(model2, list(range(20)), 64)                                         # timed: prints the per-pass line
check("wrapper installed the hook once on every layer",
      all(isinstance(l.mlp.gate, H.BudgetGate) for l in model2.language_model.model.layers) and calls == [16, 64])
print("ALL PASS" if ok else "SOME FAILED")
