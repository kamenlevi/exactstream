"""HC8-1P per-kernel breakdown (Leap, Wed 2026-11-04 14:2x; post-hoc after the BETWEEN screen): 96-read chains on the
real mixers, each read's next input made to depend on all of its outputs (mx.depends), so nothing overlaps across reads:
  K1 (chained on written), K1+K2, full (K1+K2+K3), for NP in HC8_NPS (default 16,20,40); median of 30 after 3.
K2 = (K1+K2) - K1, K3 = full - (K1+K2). Rough unless under a quiet lock.
Usage: [BigMoE venv] [HC8_NPS=16,20,40] hc8r_prof.py -> ~/Leap/data/hc8r_screen.txt (appends; GPU, MEM ~2 GB)"""
import json, os, sys, time
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import hc8r
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/hc8r_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
ROWS, HC, W, LR = 16, 4, 2560, 320
HCD, BF = HC * W, mx.bfloat16
NPS = [int(x) for x in os.environ.get("HC8_NPS", "16,20,40").split(",")]
VAR = os.environ.get("HC8_VAR", "")
M = os.path.expanduser("~/BigMoE/models/bare")
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
P8 = []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt.pop(f"{n}.{s}")) for s in SUBS])
    mx.eval(g.parameters())
    P8.append(hc8r.prep(g))
    del g
say(f"\nHC8-1P breakdown {time.strftime('%a %H:%M:%S')} ({os.environ.get('HC8_COND', 'share lock')}){' ' + VAR if VAR else ''}")
mx.random.seed(9)
H0 = mx.random.normal((ROWS, HCD)).astype(BF)
B0 = mx.random.normal((ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)

def stages(NP, upto):
    def f(i, h, bo, inj):
        P = P8[i]
        KP, NT = HCD // NP, (LR + HC + 31) // 32
        V2 = VAR in ("v2", "v4")
        V4 = VAR == "v4"
        part, ssq, hw = hc8r._kern(6 if V4 else 4 if V2 else 1, None)(
            inputs=[h, bo, inj, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
            template=[("T", BF), ("HCD", HCD), ("D", W), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT),
                      ("PEND", 1)],
            grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT if V2 else 32, 1, 1),
            output_shapes=[(NP, 16, NT * 32), (NP, 16), (16, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
        if upto == 1:
            return [part, ssq], hw, bo, inj
        if V4:
            mix1, inject, rinv, _xs = hc8r._kern(7, P["eps"])(
                inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", W)],
                grid=(16 * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
                output_shapes=[(16, LR), (16, HC), (16, HC), (16, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])
        else:
            mix1, inject, rinv = hc8r._kern(2, P["eps"])(
                inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", W)],
                grid=(16 * (LR + HC), 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(16, LR), (16, HC), (16, HC)], output_dtypes=[BF, BF, mx.float32])
        if upto == 2:
            return [mix1, inject, rinv], hw, bo, inject
        m, w_, j = hc8r.read(P, h, bo, inj, NP=NP, v=4 if V4 else 2 if V2 else 1)
        return [m, j], w_, m, j
    return f

def chain(f):
    h, bo, inj, outs = H0, B0, I0, []
    for i in range(len(P8)):
        deps, hw, bo2, inj2 = f(i, h, bo, inj)
        h = mx.depends(hw, deps)
        bo, inj = (bo2, inj2)
        outs.append(h)
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs

ARMS = {f"NP{NP} {nm}": stages(NP, u) for NP in NPS for nm, u in (("K1", 1), ("K1+K2", 2), ("full", 3))}
for a in ARMS:
    mx.eval(chain(ARMS[a]))
order = list(ARMS)
T = {a: [] for a in order}
for r in range(33):
    for a in order[r % len(order):] + order[:r % len(order)]:
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain(ARMS[a])); mx.synchronize()
        T[a].append(1e3 * (time.perf_counter() - t0))
med = {a: float(np.median(T[a][3:])) for a in order}
for NP in NPS:
    k1, k12, full = med[f"NP{NP} K1"], med[f"NP{NP} K1+K2"], med[f"NP{NP} full"]
    say(f"  NP {NP}: ms per 96 reads: K1 {k1:.2f}, K2 {k12 - k1:.2f}, K3 {full - k12:.2f}, full {full:.2f} "
        f"(us per read: {1e3 * k1 / 96:.1f} / {1e3 * (k12 - k1) / 96:.1f} / {1e3 * (full - k12) / 96:.1f})")
