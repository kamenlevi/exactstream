"""HC8-1P wiring check (Leap, Wed 2026-11-04; TABLE_350 'HC8-1P'): ref2 with ref2_hc8_patch.py on real HC weights
(layers 0-3, attention and MLP reads), mixers swapped to 8-bit by hc8r.hc8_install, called through the patched
GatedResidual.__call__ at rows 1 / 8 / 16 with the pending write:
 (a) hc8_fast on: == hc8r.read on the padded inputs, bitwise; row-invariant: a 1-row / 8-row call == the first rows of
     the 16-row call (the same first rows of input), bitwise (SD == AR); COUNTS hc8_fast;
 (b) hc8_fast off, 8-bit mixers: MTPLX's eager chain runs (hc_eager16), close to (a) (rel rms reported);
 (c) bf16 mixers with hc8_fast on: the exact S4 path, bitwise == switch off (the exact mode untouched);
 (d) ref2.self_test PASS; remove() drops the cache.
Usage: [BigMoE venv] hc8_wire_check.py -> ~/Leap/data/hc8_wire_check.txt (GPU, MEM ~1 GB; share lock)"""
import json, os, sys, time
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
import hc8r
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/hc8_wire_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nHC8-1P wiring check {time.strftime('%a %H:%M:%S')}")
assert "hc8_fast" in ref2.CFG and not ref2.CFG["hc8_fast"], "patch missing or switch not default off"
HC, W, LR, BF = 4, 2560, 320, mx.bfloat16
HCD = HC * W
M = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(M, "config.json")))
EPS = float(cfg.get("text_config", cfg)["rms_norm_eps"])
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
SUBS = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight")
names = [f"language_model.model.layers.{i}.{p}_hyper_connection" for i in range(4) for p in ("attn", "mlp")]
def build():
    gs = []
    for n in names:
        g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
        g.load_weights([(s, mx.load(os.path.join(M, idx[f"{n}.{s}"]))[f"{n}.{s}"].astype(BF)) for s in SUBS])
        mx.eval(g.parameters())
        gs.append(g)
    return gs
class Box(nn.Module):
    def __init__(self, gs):
        super().__init__()
        self.layers = gs
G8 = build()
G16 = build()
sw, mb0, mb1, _ = hc8r.hc8_install(Box(G8))
say(f"hc8_install: {sw} linears swapped, {mb0:.1f} -> {mb1:.1f} MB")
_bits = lambda a: a.view(mx.uint16) if a.dtype == BF else a
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(_bits(a) == _bits(b)).item())
f32 = mx.float32
rel = lambda a, b: float(mx.sqrt(mx.mean((a.astype(f32) - b.astype(f32)) ** 2)).item()
                         / max(mx.sqrt(mx.mean(b.astype(f32) ** 2)).item(), 1e-30))
ref2.apply()
mx.random.seed(5)
H = mx.random.normal((1, 16, HCD)).astype(BF)
BO = mx.random.normal((1, 16, W)).astype(BF)
IJ = (2 * mx.sigmoid(mx.random.normal((1, 16, HC)))).astype(BF)
mx.eval(H, BO, IJ)
ok = True
a_ok = inv_ok = True
dev = []
for i, g in enumerate(G8):
    ref2.CFG["hc8_fast"] = True
    ref2.COUNTS["hc8_fast"] = 0
    full = g(H, (BO, IJ))
    mx.eval(full)
    P = ref2._hc8_params(g)
    direct = hc8r.read(P, H.reshape(16, HCD), BO.reshape(16, W), IJ.reshape(16, HC), NP=ref2.CFG["hc8_np"], v=2)
    mx.eval(direct)
    a_ok &= eqb(full[0].reshape(16, W), direct[0]) and eqb(full[1].reshape(16, HCD), direct[1]) \
        and eqb(full[2].reshape(16, HC), direct[2])
    for r in (1, 8):
        part = g(H[:, :r], (BO[:, :r], IJ[:, :r]))
        mx.eval(part)
        inv_ok &= all(eqb(part[k], full[k][:, :r]) for k in range(3))
    cnt = ref2.COUNTS["hc8_fast"]
    a_ok &= cnt == 3
    ref2.CFG["hc8_fast"] = False
    ref2.COUNTS["hc_eager16"] = 0
    eager = g(H, (BO, IJ))
    mx.eval(eager)
    dev.append((rel(full[0], eager[0]), rel(full[2], eager[2]), eqb(full[1], eager[1]), ref2.COUNTS["hc_eager16"]))
ok &= a_ok and inv_ok
say(f"(a) hc8_fast on == hc8r.read bitwise, 3 calls counted each: {a_ok}; 1 / 8-row calls == the 16-row call's rows: {inv_ok}")
b_ok = all(d[2] and d[3] == 1 for d in dev)
ok &= b_ok
say(f"(b) switch off, 8-bit mixers -> eager chain: ran {sum(d[3] for d in dev)} / {len(dev)}; written equal {sum(d[2] for d in dev)}"
    f" / {len(dev)}; FAST vs eager rel rms: mixed max {max(d[0] for d in dev):.2e}, inject max {max(d[1] for d in dev):.2e}")
c_ok = True
for g in G16:
    outs = []
    for on in (False, True):
        ref2.CFG["hc8_fast"] = on
        ref2.COUNTS["hc_read"] = ref2.COUNTS["hc8_fast"] = 0
        o = g(H, (BO, IJ))
        mx.eval(o)
        outs.append((o, ref2.COUNTS["hc_read"], ref2.COUNTS["hc8_fast"]))
    c_ok &= all(eqb(x, y) for x, y in zip(outs[0][0], outs[1][0])) and outs[1][1:] == (1, 0)
ref2.CFG["hc8_fast"] = False
ok &= c_ok
say(f"(c) bf16 mixers with hc8_fast on: the exact path, == off bitwise: {c_ok}")
st = ref2.self_test()
st_ok = all(bool(v) for v in st.values())
ok &= st_ok
say(f"(d) ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}")
ref2.remove()
r_ok = len(ref2._HC8) == 0
ok &= r_ok
say(f"    remove() drops the cache: {r_ok}")
say(f"HC8-1P wiring: {'PASS' if ok else 'FAIL'}")
