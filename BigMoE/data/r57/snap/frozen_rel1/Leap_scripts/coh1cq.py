"""COH-1CQ (Leap, Sun 2026-11-08 01:3x; BigMoE's C4: E + OOO_1C + NAX_1CQ differs run to run, slot reuse ruled out in the ext):
FAST-NAX's one-call (ooo1cq) on the p11 store with every read of the expert POOLS through coherent(system) pointers:
the body's pool pointers (codes, LSB words, p11 scale / bias words) take tierglu_coh's "cn" form, and P11's helpers take
their pool argument as a template parameter (a coherent(system) function parameter does not bind) into coherent struct
members. Activations and slot ids stay plain. S7B-PROBE (TABLE_350 B1-B6): inside one command buffer a CPU rewrite of a
slot the GPU cached earlier is NOT seen by later plain loads; coherent(system) loads see it (B4, B6). The arithmetic is
unchanged, so outputs are bitwise == the plain kernels on static slots (d1c_check D1C_COH=1).
tier_glu_nax6n_1cq_coh(...) has tier_glu_nax6n_1cq's signature; ref2 CFG nax_coh / REF2_NAX_COH=1 routes the 1cq call here."""
import os
import re
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
import ooo1cq as OQ
import tierglu_coh as TC

_K = {}


_P11_PTRS = ("s3p", "b3p", "b2p", "blp")                 # p11 scale / bias words: read only inside P11's helpers


def _body(src):
    """the body's directly read pool pointers to tierglu_coh's "cn" form; the p11 scale / bias pointers stay plain (P11's
    helpers read them through coherent casts, _p11_hdr)"""
    cp = TC.FORMS["cn"]
    names = [n for n in TC._N6_NAMES if n not in ("srow", "brow") + _P11_PTRS
             and re.search(r"const device [A-Za-z_0-9<>, ]+?\*\s*" + n + r"\b", src)]
    types = {}

    def decl(m):
        ty, nm, init = m.group(1), m.group(2), m.group(3)
        if nm not in names:
            return m.group(0)
        types[nm] = ty
        return f"{cp} {ty}* {nm}" + (f" = ({cp} {ty}*)({init});" if init is not None else ";")
    out = re.sub(r"const device ([A-Za-z_0-9<>, ]+?)\*\s*(\w+)(?:\s*=\s*((?:[^;]|\n)+?))?;", decl, src)
    assert set(types) == set(names), set(names) - set(types)
    if "lrow" in names:
        pat = re.compile(r"(?<![\w.])lrow\s*=\s*((?:[^;=]|\n)+?);")
        assert pat.findall(out), "no reassignment of lrow"
        out = pat.sub(lambda m: f"lrow = ({cp} {types['lrow']}*)({m.group(1)});", out)
    for m in re.finditer(r"const device [A-Za-z_0-9<>, ]+?\*\s*(\w+)\s*=\s*((?:[^;]|\n)+?);", out):
        if m.group(1) in _P11_PTRS or out[max(0, m.start() - 20):m.start()].rstrip().endswith("coherent(system)"):
            continue
        used = set(re.findall(r"\w+", m.group(2))) & (TC.POOL_PARAMS | set(names))
        assert not used, f"plain pool pointer left: {m.group(0)[:120]}"
    return out


def _p11_hdr():
    """P11's helpers renamed (P11PC / P11RC / p11pc / p11rc / p11_fillc) with every pool word read through a coherent(system)
    cast; placed AFTER the SYS block, where ooo1c's oc_poll compiles its coherent(system) pointer (before it, the
    qualifier's argument is rejected)"""
    import p11 as P11
    s = P11.P11_METAL
    n1, n2 = len(re.findall(r"\bv\.w\[", s)), len(re.findall(r"(?<![\w.])w\[", s))
    assert n1 == 4 and n2 == 4, (n1, n2)
    s = re.sub(r"\bv\.w\[", "P11C(v.w)[", s)
    s = re.sub(r"(?<![\w.(])w\[", "P11C(w)[", s)
    for a, b in _REN:
        s = re.sub(r"\b" + a + r"\b", b, s)
    return "#define P11C(p) ((coherent(system) device uint*)(p))\n" + s


_REN = (("P11P", "P11PC"), ("P11R", "P11RC"), ("P11T", "P11TC"), ("p11p", "p11pc"), ("p11r", "p11rc"), ("p11_fill", "p11_fillc"))


def _ren(src):
    for a, b in _REN:
        src = re.sub(r"\b" + a + r"\b", b, src)
    return src


def _ka(p11):
    import tierglu_nax6 as N6
    import tierglu_nax6n as NN
    import ooo1c as OC
    import p11 as P11
    assert p11, "COH-1CQ covers the p11 store (the converted records3) only"
    key = ("a", p11)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name="coh1cq_nax6np_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=_ren(_body(OQ.a_one_call_q(NN._A_NP))),
            header=N6._HDR4 + P11.P11_METAL + OC.SYS + _p11_hdr())
    return _K[key]


def _kb(p11):
    import tierglu_nax6 as N6
    import tierglu_nax6n as NN
    import ooo1c as OC
    import p11 as P11
    assert p11, "COH-1CQ covers the p11 store (the converted records3) only"
    key = ("b", p11)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name="coh1cq_nax6np_b",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["y"], source=_ren(_body(OQ.b_drop(NN._B_NP))), header=N6._HDR4 + P11.P11_METAL + OC.SYS + _p11_hdr())
    return _K[key]


def tier_glu_nax6n_1cq_coh(*a, **kw):
    """ooo1cq.tier_glu_nax6n_1cq with the coherent kernels A / B swapped in for this call"""
    ka0, kb0 = OQ._ka, OQ._kb
    try:
        OQ._ka, OQ._kb = _ka, _kb
        return OQ.tier_glu_nax6n_1cq(*a, **kw)
    finally:
        OQ._ka, OQ._kb = ka0, kb0
