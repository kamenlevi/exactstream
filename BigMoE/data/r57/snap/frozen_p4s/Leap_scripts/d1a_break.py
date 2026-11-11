"""D1-A0, the HC read's stages in place (Leap, Mon 2026-11-02 23:0x; before D1 kernel A lite): the read as REF2's S4 path with
D1-B's kernel B (leapk.hc_read kernel_b: norm_p -> S4 partials -> kernel B), 96 real mixers, 96-read DEPENDENT chains
(each read's input depends on the previous read's outputs), each stage knocked out in turn (its output replaced by a
fixed pre-computed tensor that depends on the chain through mx.depends, so the chain stays serial):
  full, no norm (S4 + B on a fixed normed / written), no S4 (B on fixed partials), no B (norm + S4 only).
In-place stage cost = full - (the chain without it). Shared GPU unless noted.
Usage: [BigMoE venv] d1a_break.py -> ~/Leap/data/d1a_break.txt"""
import collections, json, os, re, sys, tempfile, time
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import leapk as LK
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/d1a_break.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
ROWS, HC, W, LR = 16, 4, 2560, 320
HCD, BF = HC * W, mx.bfloat16
N, NP = LR + HC, 3
M = os.path.expanduser("~/BigMoE/models/bare")
say(f"D1-A0 HC read stages {time.strftime('%a %H:%M')} ({os.environ.get('D1A_COND', 'shared GPU, rough')})")

# ---------------------------------------------------------------- the 96 mixers (R-09c's setup, without the spare weights)
cfg = json.load(open(os.path.join(M, "config.json")))
EPS = float(cfg.get("text_config", cfg)["rms_norm_eps"])
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
SUBS = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight")
names = [f"language_model.model.layers.{i}.{p}_hyper_connection" for i in range(48) for p in ("attn", "mlp")]
byshard = {}
for n in names:
    for s in SUBS:
        byshard.setdefault(idx[f"{n}.{s}"], []).append(f"{n}.{s}")
Wt = {}
for f, ks in byshard.items():
    w = mx.load(os.path.join(M, f))
    for k in ks:
        Wt[k] = w[k].astype(BF)
    del w
G, WCAT = [], []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt.pop(f"{n}.{s}")) for s in SUBS])
    WCAT.append(mx.concatenate([g.input_mix_weight_down.weight, g.block_inject_weight.weight], axis=0))
    mx.eval(WCAT[-1], g.parameters())
    g.input_mix_weight_down.weight = mx.zeros((LR, 1), BF)    # hc_read reads only its row count; the merged copy is WCAT
    g.block_inject_weight.weight = mx.zeros((HC, 1), BF)
    G.append(g)
mx.eval([g.parameters() for g in G])
EPS_HC = float(G[0].hc_norm.eps)
KH = LK._hc_kernels(HC, W, LR, EPS_HC)
say(f"{len(G)} mixers; up GEMM M {ROWS} x N {HCD} x K {LR}; hc_norm eps {EPS_HC}")


mx.random.seed(9)
H0 = mx.random.normal((1, ROWS, HCD)).astype(BF)
B0 = mx.random.normal((1, ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((1, ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)
def norm(i, hyper, bo, inj):
    return KH["norm_p"](inputs=[hyper.reshape(ROWS, HCD), bo.reshape(ROWS, W), inj.reshape(ROWS, HC), G[i].hc_norm.weight],
                        template=[("T", BF)], grid=(ROWS * HC * 256, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(ROWS, HCD), (ROWS, HCD)], output_dtypes=[BF, BF])
def s4(i, normed):
    return LK.s4_partials(normed.reshape(1, ROWS, HCD), WCAT[i])[0]
def kb(i, cs, normed):
    return LK.hc_kernel_b(cs, KH["tab"], G[i].input_mix_weight_up.weight, normed, HC, W, LR)
# fixed tensors for the knocked-out stages (one per mixer, evaluated once)
FIX = []
for i in range(len(G)):
    wr, nm = norm(i, H0, B0, I0)
    cs = s4(i, nm)
    mx.eval(wr, nm, cs)
    FIX.append((wr, nm, cs))
def chain(skip):
    outs, hyper, bo, inj = [], H0, B0, I0
    for i in range(len(G)):
        dep = [bo, inj, hyper]
        if skip == "norm":
            written, normed = (mx.depends(t, dep) for t in FIX[i][:2])
        else:
            written, normed = norm(i, hyper, bo, inj)
        cs = mx.depends(FIX[i][2], [normed, written]) if skip == "s4" else s4(i, normed)
        if skip == "b":
            mixed = mx.depends(B0, [cs]).reshape(ROWS, W)
            inject = mx.depends(I0, [cs]).reshape(ROWS, HC)
        else:
            mixed, inject = kb(i, cs, normed)
        outs.append((mixed, written, inject))
        hyper, bo, inj = written.reshape(1, ROWS, HCD), mixed.reshape(1, ROWS, W), inject.reshape(1, ROWS, HC)
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs
ARMS = ["full", "norm", "s4", "b"]
T = {a: [] for a in ARMS}
for a in ARMS:
    mx.eval(chain(a))
for r in range(23):
    for a in ARMS[r % 4:] + ARMS[:r % 4]:
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(chain(a))
        mx.synchronize()
        T[a].append(1e3 * (time.perf_counter() - t0))
m = {a: float(np.median(T[a][3:])) for a in ARMS}
say("  ms per 96-read chain (median of 20): " + ", ".join(f"{a if a == 'full' else 'no ' + a} {m[a]:.2f}" for a in ARMS))
say("  in-place cost per 96 reads: " + ", ".join(f"{a} {m['full'] - m[a]:.2f} ms ({1e3 * (m['full'] - m[a]) / 96:.1f} us per read)"
                                                   for a in ARMS[1:]))
# ---- the norm kernel alone, dependent chain (its pure latency) vs an element-wise kernel of the same size
def norm_chain(n=96):
    hyper = H0
    for i in range(n):
        written, normed = norm(i % len(G), hyper, B0, I0)
        hyper = mx.depends(H0, [written, normed])
    return hyper
def copy_chain(n=96):
    h = H0.reshape(ROWS, HCD)
    for i in range(n):
        h = h + B0.reshape(ROWS, W)[:, :1] * 0
    return h
for f in (norm_chain, copy_chain):
    mx.eval(f())
tn, tc = [], []
for r in range(15):
    for f, acc in ((norm_chain, tn), (copy_chain, tc)):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); mx.synchronize(); acc.append(1e3 * (time.perf_counter() - t0))
say(f"  norm kernel alone, dependent chain: {1e3 * np.median(tn) / 96:.1f} us per call; a same-size elementwise add chain "
    f"{1e3 * np.median(tc) / 96:.1f} us per call (the boundary + launch floor)")
