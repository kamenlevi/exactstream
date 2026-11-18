"""hc4i.py (Leap, Mon 16 Nov 2026 13:1x; V3 = R57_HC4 for EvalSuite's G2, Breakthrough's switch; CHECK25's C5 bytes): an
opt-in installer that sends the HC read of every GatedResidual whose mixers are 4-bit g64 (the V3 copies Breakthrough's
switch puts in place on code items) to hc4r's kernels: 1-16 rows -> hc4r.read on 16 zero-padded rows (ref2's 16-row HC8
form with 4-bit codes), 17-32 rows in the verify phase -> hc4r.read32 (one pass; == two 16-row reads bit for bit). Any
other module (8-bit HC8 mixers, bf16) and any other row count go to the installed chain unchanged (ref2's _gr, v32 /
f32one). Without it, 4-bit mixers make ref2._hc8_params return None and the read falls to MTPLX's stock eager chain.
The per-module arrays (P4) are cached by id(module) and checked by weight identity (as ref2._hc8_params), so a per-item
swap of the QuantizedLinear objects just rebuilds a small dict.
  install() after ref2.apply() (and v32 / f32one, if used); ON[0]; COUNTS; remove()"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/week98"))
import ref2
import hc4r

ON = [True]
COUNTS = dict(hc4_16=0, hc4_32=0, passthru=0)
_ORIG = {}
_P4 = {}
BF = mx.bfloat16


def _p4(g):
    """the 4-bit mixers' arrays of a GatedResidual as hc4r's P, or None (not 4-bit g64 bf16-scaled QuantizedLinear)"""
    d, u = getattr(g, "input_mix_weight_down", None), getattr(g, "input_mix_weight_up", None)
    if not (isinstance(d, nn.QuantizedLinear) and isinstance(u, nn.QuantizedLinear) and "block_inject_weight" in g):
        return None
    e = _P4.get(id(g))
    biw = getattr(g.block_inject_weight, "weight", None)
    if e is not None and e[0] is d["weight"] and e[1] is u["weight"] and e[3] is biw:
        return e[2]
    if not (d.bits == 4 and u.bits == 4 and d.group_size == 64 and u.group_size == 64 and "biases" in d and "biases" in u
            and d["scales"].dtype == BF and u["scales"].dtype == BF):
        return None
    bi = g.block_inject_weight                         # bf16 (HC8's recipe) or a 4-bit QuantizedLinear (V3): the kernels read
    if isinstance(bi, nn.QuantizedLinear):             # it as bf16 [HC, HC*D], so a quantized inject is dequantized once here
        wi = mx.dequantize(bi["weight"], bi["scales"], bi.get("biases"), group_size=bi.group_size, bits=bi.bits).astype(BF)
    else:
        wi = bi.weight.astype(BF)
    P = dict(wq=d["weight"], ws=d["scales"], wb=d["biases"], uq=u["weight"], us=u["scales"], ub=u["biases"],
             wi=wi, wn=g.hc_norm.weight.astype(BF), hc=int(g.hc_count),
             d=int(g.hidden_size), lr=int(d["weight"].shape[0]), eps=float(g.hc_norm.eps))
    mx.eval(P["wi"])
    _P4[id(g)] = (d["weight"], u["weight"], P, biw)
    return P


def _pad(a, R):
    return a if a.shape[0] == R else mx.concatenate([a, mx.zeros((R - a.shape[0],) + a.shape[1:], a.dtype)])


def _gr4(self, hyper_input, pending=None):
    if ON[0] and ref2._ON[0] and not ref2.CFG.get("hc_pend_mlp"):   # (D5's carried write: ref2's own path resolves it)
        P = _p4(self)
        if P is not None:
            lead, rows = ref2._rows(hyper_input)
            import v32
            mid = 16 < rows <= 32 and v32.ST.get("on") and v32._live(rows)
            if (rows <= 16 or mid) and hyper_input.dtype == BF and (pending is None or all(t.dtype == BF for t in pending)):
                R = 16 if rows <= 16 else 32
                h2 = _pad(hyper_input.reshape(rows, -1), R)
                bo = None if pending is None else _pad(pending[0].reshape(rows, -1), R)
                inj = None if pending is None else _pad(pending[1].reshape(rows, -1), R)
                f = hc4r.read if R == 16 else hc4r.read32
                mixed, written, inject = f(P, h2, bo, inj, NP=ref2.CFG["hc8_np"])
                COUNTS["hc4_16" if R == 16 else "hc4_32"] += 1
                return (mixed[:rows].reshape(*lead, -1), written[:rows].reshape(hyper_input.shape),
                        inject[:rows].reshape(*lead, -1))
    COUNTS["passthru"] += 1
    return _ORIG["gr"](self, hyper_input, pending)


def install():
    assert ref2._ORIG, "ref2.apply() first"
    if not _ORIG:
        _ORIG["gr"] = ref2.Q.GatedResidual.__call__
        ref2.Q.GatedResidual.__call__ = _gr4


def remove():
    if _ORIG:
        ref2.Q.GatedResidual.__call__ = _ORIG["gr"]
        _ORIG.clear()
