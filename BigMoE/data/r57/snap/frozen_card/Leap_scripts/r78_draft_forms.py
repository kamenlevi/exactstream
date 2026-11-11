"""R-78 DRAFT-FORMS (Leap, Sun 2026-11-01; shared GPU, rough): L1-23's bf16 trick on the drafter. For each bf16 projection
of DFlash2 (real L1-20 weights, layer 0 + fc + the selector's hidden_projection) at its live row count, MLX spellings of
x @ W.T: A = nn.Linear's (x @ W.T), B = (W @ x.T).T, C = x @ Wt with Wt = W.T made contiguous once, D = mx.addmm-free
matmul on a [rows, K] copy of x made contiguous. Bitwise vs A over 8 random inputs per shape; ms per call (8 chained calls
per eval, best of 3 medians). A form that is bitwise AND faster is an exact drafter speedup (the drafts stay identical).
Usage: [BigMoE venv] r78_draft_forms.py -> ~/Leap/data/r78_draft_forms.txt"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import dflash_model_mlx as DF

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters())
L0 = draft.layers[0]
a = L0.self_attn
SH = [("q_proj", a.q_proj.weight, 16, 5), ("k_proj ctx", a.k_proj.weight, 8, 5), ("k_proj block", a.k_proj.weight, 16, 5),
      ("v_proj ctx", a.v_proj.weight, 8, 5), ("v_proj block", a.v_proj.weight, 16, 5), ("o_proj", a.o_proj.weight, 16, 5),
      ("mlp gate", L0.mlp.gate_proj.weight, 16, 5), ("mlp up", L0.mlp.up_proj.weight, 16, 5),
      ("mlp down", L0.mlp.down_proj.weight, 16, 5), ("conv kproj", L0.attention_conv.kernel_projection.weight, 16, 10),
      ("fc", draft.fc.weight, 8, 1), ("sel hidden_proj", draft.candidate_selector.hidden_projection.weight, 15, 1)]
b16 = lambda x: np.array(x.view(mx.uint16))
def bench(f, chain=8, n=25):
    for _ in range(3):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / chain
tot = {"A": 0.0, "best": 0.0}
mx.random.seed(0)
for name, W, rows, count in SH:
    Wt = mx.contiguous(W.T)
    mx.eval(Wt)
    forms = {"A": lambda x: x @ W.T,                                       # nn.Linear on the live [1, rows, K]
             "B": lambda x: (W @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], -1),
             "C": lambda x: x @ Wt}
    same = {k: True for k in forms}
    for t in range(8):
        x = (mx.random.normal((1, rows, W.shape[1])) * (0.3 + t)).astype(mx.bfloat16)
        ys = {k: f(x) for k, f in forms.items()}
        mx.eval(ys)
        for k in forms:
            same[k] &= bool((b16(ys[k]) == b16(ys["A"])).all())
    x = (mx.random.normal((1, rows, W.shape[1]))).astype(mx.bfloat16)
    mx.eval(x)
    ms = {k: 1e3 * min(bench(lambda f=f: f(x)) for _ in range(3)) for k, f in forms.items()}
    ok = [k for k in forms if same[k]]
    best = min(ok, key=lambda k: ms[k])
    tot["A"] += count * ms["A"]; tot["best"] += count * ms[best]
    say(f"{name:16s} [{rows} x {W.shape[1]}] x [{W.shape[0]}]: ms A {ms['A']:.4f}, B {ms['B']:.4f} ({'bitwise' if same['B'] else 'DIFFERS'}), "
        f"C {ms['C']:.4f} ({'bitwise' if same['C'] else 'DIFFERS'}); best exact {best} (x{count} per propose)")
say(f"per propose: A {tot['A']:.2f} ms -> best exact forms {tot['best']:.2f} ms ({tot['best'] - tot['A']:+.2f}; shared GPU, rough)")
open(os.path.expanduser("~/Leap/data/r78_draft_forms.txt"), "w").write("\n".join(lines) + "\n")
