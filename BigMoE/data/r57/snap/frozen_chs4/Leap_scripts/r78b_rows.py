"""R-78b (Leap, Wed 2026-11-04 14:1x; the orchestrator's route after MTP Head's DRAFT17): at which block row counts are
draft_fused's R-78 forms bitwise == nn.Linear's x @ W.T? (16 was checked in R-78; 8 and 15 differ.) Real L1-20 weights,
layer 0 (every layer has the same shapes), each form's projections: form C = x @ Wt (Wt = W.T contiguous) on q_proj and
the MLP's gate / up / down; form B = (W @ x.T).T on k / v / o_proj and both conv kernel projections. Rows 1-40, 48, 64;
4 random inputs per (projection, rows), [1, rows, K] as the drafter calls them.
Prints, per row count, whether every form-C and every form-B projection is bitwise; the rows where BOTH hold are the
safe FORM_ROWS (opening them keeps the drafts bitwise; the fallback's per-call weight copy goes away there).
Usage: [BigMoE venv] r78b_rows.py -> ~/Leap/data/r78b_rows.txt (GPU, MEM ~1.5 GB; share lock; ~1 min)"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party"))
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/Leap/data/r78b_rows.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters())
L0 = draft.layers[0]
a = L0.self_attn
C = [("q_proj", a.q_proj.weight), ("mlp gate", L0.mlp.gate_proj.weight), ("mlp up", L0.mlp.up_proj.weight),
     ("mlp down", L0.mlp.down_proj.weight)]
B = [("k_proj", a.k_proj.weight), ("v_proj", a.v_proj.weight), ("o_proj", a.o_proj.weight),
     ("att conv kproj", L0.attention_conv.kernel_projection.weight), ("mlp conv kproj", L0.mlp_conv.kernel_projection.weight)]
b16 = lambda x: np.array(x.view(mx.uint16))
say(f"\nR-78b {time.strftime('%a %H:%M:%S')}: forms vs x @ W.T by rows; C {[(n, tuple(w.shape)) for n, w in C]}; "
    f"B {[(n, tuple(w.shape)) for n, w in B]}")
WT = {n: mx.contiguous(w.T) for n, w in C}
mx.eval(list(WT.values()))
ROWS = [int(r) for r in os.environ["R78B_ROWS"].split(",")] if os.environ.get("R78B_ROWS") else list(range(1, 41)) + [48, 64]
safe, rowsC, rowsB = [], [], []
mx.random.seed(78)
for r in ROWS:
    okC, okB, bad = True, True, []
    for t in range(4):
        for n, w in C + B:
            x = (mx.random.normal((1, r, w.shape[1])) * (0.3 + t)).astype(mx.bfloat16)
            ya = x @ w.T
            yf = x @ WT[n] if n in WT else (w @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], -1)
            mx.eval(ya, yf)
            if not (b16(ya) == b16(yf)).all():
                if n in WT:
                    okC = False
                else:
                    okB = False
                if n not in bad:
                    bad.append(n)
    rowsC += [r] if okC else []
    rowsB += [r] if okB else []
    safe += [r] if okC and okB else []
    say(f"  rows {r:2d}: C {'bitwise' if okC else 'DIFFERS'}, B {'bitwise' if okB else 'DIFFERS'}"
        + (f" ({', '.join(bad)})" if bad else ""))
say(f"R-78b: form C bitwise at rows {rowsC}")
say(f"R-78b: form B bitwise at rows {rowsB}")
say(f"R-78b: both (safe FORM_ROWS) {safe}; 17 {'in' if 17 in safe else 'NOT in'}, 32 {'in' if 32 in safe else 'NOT in'}")
