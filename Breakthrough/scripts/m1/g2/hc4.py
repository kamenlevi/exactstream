"""hc4.py (Breakthrough, Mon 16 Nov 2026 13:xx; V3 = R57_HC4 for EvalSuite's G2, lossygate amendment 5; CHECK BYTES,
~/Breakthrough/docs/CHECK_BYTES_900.md): the HC mixers, block_inject and PLE key / value at 4-bit on CODE items, as an
opt-in per-item switch over HC8 (ref2 / hc8r untouched). Leap's hc4i.py (7c3e149) routes the reads of 4-bit-mixer
GatedResiduals to hc4r's kernels (16 rows, and 17-32 rows in the verify phase).

What changes (FAST, lossy; EvalSuite gates it; adopting it is Kamen's call):
  prepare(model)  BEFORE hc8r.hc8_install (which replaces the bf16 mixers): the 4-bit copies from the bf16 ORIGINALS,
                  built exactly as the map's bt_mix4 (dm_d2._requant: a bf16 nn.Linear -> nn.QuantizedLinear.from_linear(
                  lin, 64 if K % 64 == 0 else 32, 4)):
                    - input_mix_weight_down / up of every non-MTP GatedResidual (194 in the 125B, the final mixer
                      included: without block_inject it takes MTPLX's stock eager read, which serves 4-bit modules);
                    - block_inject_weight of every GatedResidual that has one (96; hc4i dequantizes it once for the read);
                    - the PLE layer's key_proj / value_proj, converted to ref2's DENSE-N-RAM layout (ref2.dense_native_
                      convert over a container that holds only them), like every 4-bit dense weight under R57_DNR.
                  Nothing in the model changes here. Returns (counts, extra MB): the copies sit beside HC8's 8-bit mixers
                  and the bf16 inject / PLE, so MEM grows by that much (~0.3-0.4 GB).
  engage(on)      per item AFTER the prefill (as int8d's ON): on swaps every target module object to its 4-bit copy; off
                  puts back the objects that were there at the first engage (HC8's 8-bit mixers, the bf16 inject / PLE).
                  ref2's and hc4i's per-module caches check weight identity, so a swap only rebuilds small dicts.
  COUNTS / take() hc4 = 1 for an engaged item (EvalSuite's engagement key), swaps; take() disengages.
Unchanged: every item with engage(False) (the original objects, bit for bit), and the prefill of every item (engage
comes after it). Scope: BigMoE's screen engages per item (R57_HC4_SETS, like R57_INT8D_SETS).
  prepare(model) before hc8_install; install() after ref2.apply() and hc4i.install(); engage(); take(); remove()"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
COUNTS = dict(hc4=0, swaps=0, engaged_items=0)
_T = []            # [(module, attr, 4-bit copy)]
_OFF = []          # [(module, attr, original object)], captured at the first engage(True)
_ST = dict(on=False, ref2=None, hc4i=None, mb=0.0)


def _q4(lin):
    """the map's bt_mix4 conversion (dm_d2._requant) of a bf16 nn.Linear, bit for bit"""
    w = lin.weight
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    gs = 64 if in_f % 64 == 0 else 32
    has_b = "bias" in lin
    tmp = nn.Linear(in_f, out_f, bias=has_b)
    tmp.weight = w.astype(mx.bfloat16)
    if has_b:
        tmp.bias = lin.bias
    q = nn.QuantizedLinear.from_linear(tmp, gs, 4)
    mx.eval(q.parameters())
    return q


def prepare(model, gr_cls=None, ple_cls=None, ref2_module=None, dnr=True):
    """the 4-bit copies from the bf16 originals; call BEFORE hc8r.hc8_install. Returns (counts, extra MB)."""
    if gr_cls is None or ple_cls is None:
        from mtplx.models import qwen4_exp as Q
        gr_cls, ple_cls = gr_cls or Q.GatedResidual, ple_cls or Q.PLELayer
    TM = getattr(model, "language_model", model)
    cnt = dict(mix=0, inject=0, ple=0)
    ple_copies = []
    _T.clear()
    _OFF.clear()
    for name, mod in TM.named_modules():
        if name.startswith("mtp") or ".mtp" in name:
            continue
        if isinstance(mod, gr_cls):
            tg = [(a, "mix") for a in ("input_mix_weight_down", "input_mix_weight_up")]
            if "block_inject_weight" in mod:
                tg.append(("block_inject_weight", "inject"))
        elif isinstance(mod, ple_cls):
            tg = [(a, "ple") for a in ("key_proj", "value_proj")]
        else:
            continue
        for attr, kind in tg:
            lin = getattr(mod, attr)
            assert type(lin) is nn.Linear, (f"R57_HC4: {name}.{attr} is {type(lin).__name__}, not the bf16 original: "
                                            f"prepare() must run before hc8r.hc8_install")
            q = _q4(lin)
            _T.append((mod, attr, q))
            cnt[kind] += 1
            if kind == "ple":
                ple_copies.append(q)
    if dnr and ple_copies:
        if ref2_module is None:
            import ref2 as ref2_module
        box = nn.Module()
        box.layers = ple_copies                       # ref2.dense_native_convert walks named_modules: only these
        ref2_module.dense_native_convert(box)
        _ST["ref2"] = ref2_module
    mb = sum(sum(v.nbytes for k, v in q.items() if isinstance(v, mx.array)) for _, _, q in _T) / 1e6
    _ST["mb"] = mb
    return cnt, mb


def install(hc4i_module=None):
    """after ref2.apply() and Leap's hc4i.install(): checks the router is in place (it reads the 4-bit mixers)"""
    if hc4i_module is None:
        import hc4i as hc4i_module
    assert hc4i_module._ORIG, "R57_HC4: hc4i.install() first (the 4-bit mixers' read)"
    _ST["hc4i"] = hc4i_module
    return len(_T)


def engage(on):
    """per item, after the prefill: swap every target to its 4-bit copy (on) or back to its original object (off)"""
    on = bool(on)
    if on == _ST["on"]:
        return 0
    if on and not _OFF:
        _OFF.extend((mod, attr, getattr(mod, attr)) for mod, attr, _ in _T)
    src = _T if on else _OFF
    for mod, attr, obj in src:
        setattr(mod, attr, obj)
    _ST["on"] = on
    COUNTS["swaps"] += 1
    if on:
        COUNTS["engaged_items"] += 1
    return len(src)


def take():
    """this item's counter (hc4 = 1 if it was engaged), then the originals back"""
    was = _ST["on"]
    engage(False)
    return dict(hc4=1 if was else 0)


def remove():
    engage(False)
    _T.clear()
    _OFF.clear()
