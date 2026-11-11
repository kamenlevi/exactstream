"""HC8-1P screen (Leap, Wed 2026-11-04 14:2x; TABLE_350 'HC8-1P', pre-registered 14:17): the FAST 8-bit one-pass HC read
(scripts/hc8r.py) on the real HC weights of all 96 mixers (48 layers x attention / MLP reads), the pending write on.
 (1) correctness, teacher-forced along the exact chain (each read gets the exact chain's inputs): read() vs emulate()
     (the same FAST math in fp32 MLX ops, the same 8-bit weights): rel rms of mixed / inject per read (max over reads);
     written bitwise == emulate's and == the exact read's (the same pending-write ops);
 (2) deviation vs the exact read (leapk.hc_read, REF2's S4 numerics), teacher-forced: rel rms of mixed / inject, for
     HC8-1P and for HC8-eager (the reference math with MLX's own 8-bit matmuls: the map's hc8w numerics);
 (3) HC8_TIME=1: ms per free-running 96-read chain, arms interleaved, median of 40 after 3: S4g (S4 + R-82), B (kernel B,
     B4p2v), HC8 NP 8 / NP 16.
Lines (TABLE_350): PASS = (1) [rel rms <= 4e-3, written bitwise] AND HC8 <= 5.0 ms per chain; KILL = (1) fails after one
fix, or > 7.0 ms; else BETWEEN.
Usage: [BigMoE venv] [HC8_TIME=1] [HC8_COND=...] hc8r_screen.py -> ~/Leap/data/hc8r_screen.txt (GPU, MEM ~3 GB)"""
import json, os, sys, time
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import leapk as LK
import hc8r
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/hc8r_screen.txt")
TIME = os.environ.get("HC8_TIME") == "1"
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
ROWS, HC, W, LR = 16, 4, 2560, 320
HCD, BF = HC * W, mx.bfloat16
M = os.path.expanduser("~/BigMoE/models/bare")
say(f"\nHC8-1P screen {time.strftime('%a %H:%M:%S')} ({os.environ.get('HC8_COND', 'share lock')}){' + timing' if TIME else ''}")

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
G, WCAT, P8 = [], [], []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt.pop(f"{n}.{s}")) for s in SUBS])
    mx.eval(g.parameters())
    P8.append(hc8r.prep(g))
    WCAT.append(mx.concatenate([g.input_mix_weight_down.weight, g.block_inject_weight.weight], axis=0))
    mx.eval(WCAT[-1])
    g.input_mix_weight_down.weight = mx.zeros((LR, 1), BF)    # hc_read reads only its row count; the merged copy is WCAT
    G.append(g)
mx.eval([g.parameters() for g in G])
say(f"{len(G)} mixers loaded and quantized (8-bit g64); hc_norm eps {P8[0]['eps']}")

f32 = mx.float32
def rel(a, b):
    a, b = a.astype(f32), b.astype(f32)
    return float(mx.sqrt(mx.mean((a - b) ** 2)).item() / max(mx.sqrt(mx.mean(b ** 2)).item(), 1e-30))
eqb = lambda a, b: bool(a.size == b.size and mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())

def exact(i, hyper, bo, inj):
    return LK.hc_read(G[i], WCAT[i], hyper, (bo, inj), rows=ROWS, fuse_gate=True)
def exact_b(i, hyper, bo, inj):
    return LK.hc_read(G[i], WCAT[i], hyper, (bo, inj), rows=ROWS, kernel_b=True)
def fast(NP, v=1):
    def f(i, hyper, bo, inj):
        m, w, j = hc8r.read(P8[i], hyper.reshape(16, HCD), bo.reshape(16, W), inj.reshape(16, HC), NP=NP, v=v)
        return m.reshape(1, 16, W), w.reshape(1, 16, HCD), j.reshape(1, 16, HC)
    return f
def eager8(i, hyper, bo, inj):
    """the reference math (MTPLX's eager GatedResidual chain) with MLX's 8-bit matmuls on the HC8 weights"""
    p = P8[i]
    h = hyper + (bo[..., None, :] * inj[..., :, None]).reshape(*hyper.shape)
    normed = G[i].hc_norm(h)
    q = lambda x, a, s_, b_: mx.quantized_matmul(x, a, s_, b_, transpose=True, group_size=64, bits=8)
    mix = nn.silu(q(normed, p["wq"], p["ws"], p["wb"]) / HC)
    mix = mx.sigmoid(q(mix, p["uq"], p["us"], p["ub"]))
    mixed = mx.mean(mix.reshape(1, 16, HC, W) * normed.reshape(1, 16, HC, W), axis=-2)
    inject = 2.0 * mx.sigmoid((normed @ p["wi"].T) / HC)
    return mixed, h, inject

mx.random.seed(9)
H0 = mx.random.normal((1, ROWS, HCD)).astype(BF)
B0 = mx.random.normal((1, ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((1, ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)

# the exact chain: per read inputs and outputs
ins, ref = [], []
hyper, bo, inj = H0, B0, I0
for i in range(len(G)):
    ins.append((hyper, bo, inj))
    out = exact(i, hyper, bo, inj)
    mx.eval(out)
    ref.append(out)
    hyper, bo, inj = out[1], out[0], out[2]

# (1) + (2), teacher-forced
VARS = [(int(a.split("v")[0]), int(a.split("v")[1])) for a in os.environ.get("HC8_VARS", "16v1,8v1").split(",")]
st = {k: [] for k in ["emu mixed", "emu inject"] + [f"dev {o} {n}" for n in [f"NP{a}v{b}" for a, b in VARS] + ["eager8"]
                                                     for o in ("mixed", "inject")]}
wb_emu = wb_ex = 0
xv = {}                                                     # each variant vs the first one, bitwise (mixed, written, inject)
for i in range(len(G)):
    hyper, bo, inj = ins[i]
    outs = {f"NP{a}v{b}": fast(a, b)(i, hyper, bo, inj) for a, b in VARS}
    em, ew, ej = hc8r.emulate(P8[i], hyper.reshape(16, HCD), bo.reshape(16, W), inj.reshape(16, HC))
    gm, gw, gj = eager8(i, hyper, bo, inj)
    mx.eval(list(outs.values()), em, ew, ej, gm, gw, gj)
    st["emu mixed"].append(max(rel(o[0], em.reshape(1, 16, W)) for o in outs.values()))
    st["emu inject"].append(max(rel(o[2], ej.reshape(1, 16, HC)) for o in outs.values()))
    first = next(iter(outs.values()))
    for nm_, o in list(outs.items())[1:]:
        xv[nm_] = xv.get(nm_, 0) + all(eqb(u, v) for u, v in zip(o, first))
    wb_emu += all(eqb(o[1], ew) for o in outs.values())
    wb_ex += all(eqb(o[1], ref[i][1]) for o in outs.values()) and eqb(gw, ref[i][1])
    for nm, (mm, _, jj) in list(outs.items()) + [("eager8", (gm, gw, gj))]:
        st[f"dev mixed {nm}"].append(rel(mm, ref[i][0]))
        st[f"dev inject {nm}"].append(rel(jj, ref[i][2]))
ok1 = max(st["emu mixed"]) <= 4e-3 and max(st["emu inject"]) <= 4e-3 and wb_emu == len(G) and wb_ex == len(G)
say(f"(1) read() vs emulate(): mixed rel rms max {max(st['emu mixed']):.2e} (median {np.median(st['emu mixed']):.2e}), "
    f"inject max {max(st['emu inject']):.2e}; written == emulate {wb_emu}/{len(G)}, == exact (and eager8) {wb_ex}/{len(G)} "
    f"-> {'OK' if ok1 else 'FAIL'}")
if xv:
    say("    bitwise vs " + next(iter(f"NP{a}v{b}" for a, b in VARS)) + " (mixed, written, inject), of 96: "
        + ", ".join(f"{k} {v}" for k, v in xv.items()))
say("(2) deviation vs the exact read (rel rms, median [max] over 96 teacher-forced reads): " + "; ".join(
    f"{k[4:]} {np.median(v):.2e} [{max(v):.2e}]" for k, v in st.items() if k.startswith("dev")))

if TIME:
    def chain(arm):
        outs, hyper, bo, inj = [], H0, B0, I0
        for i in range(len(G)):
            mixed, written, inject = arm(i, hyper, bo, inj)
            outs.append((mixed, written, inject))
            hyper, bo, inj = written, mixed, inject
            if i % 8 == 7:
                mx.async_eval(outs[-1])
        return outs
    ARMS = {"S4g": exact, "B": exact_b}
    ARMS.update({f"HC8 NP{a}v{b}": fast(a, b) for a, b in VARS})
    for a in ARMS:
        mx.eval(chain(ARMS[a]))
    PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
    order = list(ARMS)
    T = {a: [] for a in order}
    for r in range(43):
        while os.path.exists(PAUSE):
            time.sleep(5)
        for a in order[r % len(order):] + order[:r % len(order)]:
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(chain(ARMS[a]))
            mx.synchronize()
            T[a].append(1e3 * (time.perf_counter() - t0))
    med = {a: float(np.median(T[a][3:])) for a in order}
    say("(3) ms per 96-read chain (median of 40): " + ", ".join(
        f"{a} {med[a]:.2f} [{np.percentile(T[a][3:], 10):.2f}-{np.percentile(T[a][3:], 90):.2f}]" for a in order))
    best = min([a for a in ARMS if a.startswith("HC8")], key=lambda a: med[a])
    v = "KILL" if not ok1 or med[best] > 7.0 else "PASS" if med[best] <= 5.0 else "BETWEEN"
    say(f"HC8-1P verdict: {v} (best {best} {med[best]:.2f} ms per chain; vs S4g {med['S4g'] - med[best]:+.2f}, "
        f"vs B {med['B'] - med[best]:+.2f}; (1) {'OK' if ok1 else 'FAIL'})")
