"""INV1 (R-40 REF2 step 2; Leap, pre-registered Wed 2026-10-28 15:50 EET in ~/Leap/TABLE_350.md 'R-40'): INV0 found
that under REF2's padding patches only attention (layer 3, the first QSA layer) still differs between 1-row AR and the
16-row verify: at M = 1 QSA takes the decode gather lane, at M = 16 the verify path (dense selection mask +
_verify_sdpa). INV1 = INV0's REF2 run with one more patch: an attention call with S < 16 rows is padded to 16 rows
(copies of its last row) through the same verify path, then the QSA cache is trimmed back by the padding (KV and pooled
blocks, QSACache.trim) and the real rows' outputs kept.
Pass (for R-40's route): every record and the logits of all 16 rows bit-identical AR vs verify.
Usage: [BigMoE venv] inv1_screen.py -> ~/Leap/data/inv1_screen.txt"""
import os, sys
sys.argv = [sys.argv[0], "inv1"]                         # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
import ca_anat as CA
import leapk as LK
from qmm16 import qmm16c
import numpy as np
import mlx.core as mx
import mlx.nn as nn

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
R = 16
PATCH = [False]

# ---------------------------------------------------------------- REF2's padding patches
def pad16(x2):
    m = x2.shape[0]
    return x2 if m == R else mx.concatenate([x2, mx.zeros((R - m,) + x2.shape[1:], x2.dtype)], axis=0)

ORIG_QMM = mx.quantized_matmul
def qmm(x, w, *args, **kw):
    if not PATCH[0]:
        return ORIG_QMM(x, w, *args, **kw)
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    lead = x.shape[:-1]
    rows = int(np.prod(lead)) if lead else 1
    if not (transpose and mode == "affine" and biases is not None and rows <= R and x.dtype == mx.bfloat16):
        return ORIG_QMM(x, w, *args, **kw)
    K_, N = x.shape[-1], w.shape[0]
    x16 = pad16(x.reshape(rows, K_))
    if bits == 4 and group_size == 64 and N >= 1024 and N % 64 == 0 and K_ % 64 == 0:
        y = qmm16c(x16, w, scales, biases)
    else:
        y = ORIG_QMM(x16, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
    return y[:rows].reshape(*lead, N)
mx.quantized_matmul = qmm

ORIG_LIN = nn.Linear.__call__
def lin(self, x):
    lead = x.shape[:-1]
    rows = int(np.prod(lead)) if lead else 1
    if PATCH[0] and rows < R and x.dtype == mx.bfloat16:
        y = ORIG_LIN(self, pad16(x.reshape(rows, x.shape[-1])))
        return y[:rows].reshape(*lead, y.shape[-1])
    return ORIG_LIN(self, x)
nn.Linear.__call__ = lin

WCAT = {}
ORIG_GR = Q.GatedResidual.__call__
def gr(self, hyper_input, pending=None):
    if not PATCH[0]:
        return ORIG_GR(self, hyper_input, pending)
    lead = hyper_input.shape[:-1]
    rows = int(np.prod(lead)) if lead else 1
    if rows > R:
        return ORIG_GR(self, hyper_input, pending)
    h16 = pad16(hyper_input.reshape(rows, -1)).reshape(1, R, -1)
    p16 = None if pending is None else tuple(pad16(p.reshape(rows, -1)).reshape(1, R, -1) for p in pending)
    if "block_inject_weight" in self:
        if id(self) not in WCAT:
            WCAT[id(self)] = mx.concatenate([self.input_mix_weight_down.weight, self.block_inject_weight.weight], axis=0)
        mixed, written, inject = LK.hc_read(self, WCAT[id(self)], h16, p16, rows=R)
        return (mixed[0, :rows].reshape(*lead, -1), written[0, :rows].reshape(hyper_input.shape),
                inject[0, :rows].reshape(*lead, -1))
    out = ORIG_GR(self, h16, p16)                        # the head mixer: the eager 16-row chain
    return out[0, :rows].reshape(*lead, -1)
Q.GatedResidual.__call__ = gr

ORIG_ATT = Q.Attention.__call__
def att(self, x, cache):
    S = x.shape[1]
    if PATCH[0] and S < R and isinstance(cache, Q.QSACache):
        xp = mx.concatenate([x, mx.repeat(x[:, -1:], R - S, axis=1)], axis=1)
        out = ORIG_ATT(self, xp, cache)                  # the 16-row verify path
        cache.trim(R - S)                                # drop the padding rows' KV and pooled blocks
        return out[:, :S]
    return ORIG_ATT(self, x, cache)
Q.Attention.__call__ = att

# ---------------------------------------------------------------- hooks (forward order)
REC = []
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        REC.append((part, y if isinstance(y, mx.array) else y[0]))
        return y
    klass.__call__ = call
hook(Q.GatedDeltaNet, "gdn")
hook(Q.Attention, "attn")
hook(Q.SparseMoeBlock, "moe")
hook(Q.PLELayer, "ple")
hook(Q.DecoderLayer, "layer")

def run(patched):
    PATCH[0] = patched
    P = 1024
    toks = CA.ids[P:P + R]
    cv, ca = CA.prefill(P), CA.prefill(P)
    # V16: one 16-row verify
    REC.clear()
    snap = CS.snapshot_untrimmable_cache_lazy(cv)
    lgv = CA.forward("A", mx.array([toks]), cv)
    mx.eval(lgv, [r for _, r in REC])
    recv = list(REC)
    model.commit_verified_window(cv, snap.states, keep_tokens=R, verified_tokens=R)
    # AR16: 16 one-token steps through the same verify path, commit keep 1
    reca, lga = [], []
    for j in range(R):
        REC.clear()
        snap = CS.snapshot_untrimmable_cache_lazy(ca)
        lg = CA.forward("A", mx.array([[toks[j]]]), ca)
        mx.eval(lg, [r for _, r in REC])
        reca.append(list(REC)); lga.append(lg)
        model.commit_verified_window(ca, snap.states, keep_tokens=1, verified_tokens=1)
        mx.eval(CA.states(ca))
    PATCH[0] = False
    eqb = lambda a, b: bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())
    tag = "REF2" if patched else "today"
    say(f"  {tag}: {len(recv)} records per forward (V16) vs {len(reca[0])} (AR step)")
    first, nlog = [], 0
    layer_of = []
    li = 0
    for part, _ in recv:
        layer_of.append(li)
        if part == "layer":
            li += 1
    for j in range(R):
        f = None
        for e, ((part, rv), (_, ra)) in enumerate(zip(recv, reca[j])):
            if not eqb(rv[:, j:j + 1] if rv.ndim == 3 else rv[j:j + 1], ra):
                f = (e, part, layer_of[e])
                break
        first.append(f)
        nlog += eqb(lgv[:, j:j + 1], lga[j])
    worst = [f for f in first if f is not None]
    say(f"  {tag}: logits rows equal {nlog} / {R}; rows with any differing record {len(worst)} / {R}")
    if worst:
        e0 = min(worst)
        say(f"    earliest divergence: record {e0[0]} ({e0[1]}, layer {e0[2]}); per row: " +
            ", ".join(f"{j}:{f[1]}@L{f[2]}" if f else f"{j}:=" for j, f in enumerate(first)))
    return nlog, first

say("INV1: 1-row AR (16 steps) vs a 16-row verify, CA's 12-layer slice, 1K context; REF2 + attention padded to 16 rows")
r_ref2 = run(True)
allrec = all(f is None for f in r_ref2[1])
say(f"INV1 verdict: logits equal {r_ref2[0]} / 16, every record equal for every row: {allrec} -> "
    + ("REF2 is batch-invariant with the attention padding (route b holds)" if (allrec and r_ref2[0] == 16)
       else "still diverging (see the earliest record)"))
open(os.path.join(D, "inv1_screen.txt"), "w").write("\n".join(lines) + "\n")
