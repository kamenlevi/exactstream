"""HC-B ref2 check (Leap, Mon 2026-11-02 21:2x; from r82_hc_gate.py): ref2 CFG hc_b (leapk.hc_read kernel_b, D1-B) vs off and vs
hc_gate, through GatedResidual with REF2 applied, real HC weights of layers 3 / 20 / 44 (attention / MLP reads, with and
without the pending write, 16- and 8-row inputs; 8 rows are padded to 16 by ref2): mixed / written / inject bitwise;
kernels per read; then ref2.self_test with the switch off.
Was: R-82 HC-GATE (Leap, Sun 2026-11-01; shared GPU, real weights, tiny): ref2 CFG hc_gate (leapk.hc_read fuse_gate: the S4
partition sum + gate1 in one kernel) vs off, on the real HC weights of layers 3 / 20 / 44 (attention / MLP hyper-connection
reads, with and without the pending write), REF2 on, 16-row calls: mixed / written / inject bitwise; kernels per read.
Usage: [BigMoE venv] hcb_ref2_check.py -> ~/Leap/data/hcb_ref2_check.txt"""
import os, sys, json, re, tempfile, collections
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg["text_config"])
def load(prefix, mod):
    keys = [k for k in wmap if k.startswith(prefix)]
    w = {}
    for f in sorted({wmap[k] for k in keys}):
        sh = mx.load(f"{D}/{f}")
        w.update({k[len(prefix):]: sh[k] for k in keys if wmap[k] == f})
    mod.load_weights(list(w.items()), strict=False)
    mod.eval()
ref2.apply()
VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided",
         "Flatten", "Unflatten", "StopGradient", "Depends", "Split"}
def census(*arrs):
    cnt = collections.Counter()
    fd, p = tempfile.mkstemp(suffix=".dot"); os.close(fd)
    try:
        mx.export_to_dot(p, *arrs)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return sum(v for k_, v in cnt.items() if k_ not in VIEWS)
b16 = lambda a: np.array(a.view(mx.uint16))
mx.random.seed(0)
fails, kc, n = 0, {}, 0
ARMS = {"off": dict(hc_gate=False, hc_b=False), "gate": dict(hc_gate=True, hc_b=False), "b": dict(hc_gate=False, hc_b=True),
        "b+gate": dict(hc_gate=True, hc_b=True)}
for L in (3, 20, 44):
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = Q.GatedResidual(args)
        load(f"language_model.model.layers.{L}.{name}.", hc)
        mx.eval(hc.parameters())
        for t in range(8):
            sc = (0.5, 1.0, 2.0, 4.0)[t % 4]
            S = 16 if t < 4 else 8
            h = (mx.random.normal((1, S, 4 * 2560)) * sc).astype(mx.bfloat16)
            pend = (((mx.random.normal((1, S, 2560)) * sc).astype(mx.bfloat16),
                     (mx.random.normal((1, S, 4)) * 0.5).astype(mx.bfloat16)) if t % 2 else None)
            mx.eval(h, pend)
            outs = {}
            for arm, cf in ARMS.items():
                ref2.CFG.update(cf)
                o = hc(h, pend) if pend is not None else hc(h)
                if L == 3 and t < 2:
                    kc[(name, t % 2, arm)] = census(*o)
                mx.eval(o)
                outs[arm] = [b16(a) for a in o]
            for arm in ("gate", "b", "b+gate"):
                eq = all((a == b).all() for a, b in zip(outs["off"], outs[arm]))
                fails += not eq
                n += 1
                if not eq:
                    say(f"L{L} {name} t{t} (S {S}) {arm}: DIFFERS")
ref2.CFG.update(hc_gate=False, hc_b=False)
say("kernels per read (layer 3, 16 rows): " + "; ".join(f"{n_.split('_')[0]} pending {bool(p)}: " + ", ".join(
    f"{a} {kc[(n_, p, a)]}" for a in ARMS) for (n_, p) in sorted({(k[0], k[1]) for k in kc})))
say(f"HC-B bits via ref2 (3 layers x 2 reads x 8 inputs (16 / 8 rows, with / without pending) x 3 arms vs off): "
    f"{'PASS' if fails == 0 else f'FAIL ({fails})'} of {n}; hc_read calls {ref2.counts()['hc_read']}, hc_b {ref2.counts()['hc_b']}")
st = ref2.self_test()
say(f"ref2.self_test (switch off): {'PASS' if all(st.values()) else 'FAIL'} {st}")
open(os.path.expanduser("~/Leap/data/hcb_ref2_check.txt"), "w").write("\n".join(lines) + "\n")
