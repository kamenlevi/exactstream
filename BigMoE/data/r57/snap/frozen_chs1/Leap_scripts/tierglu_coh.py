"""COH (Leap, Tue 2026-11-03 22:4x; TABLE_350 'COH'; BigMoE's ask (a) after S7B-PROBE): NAX6-N's and 3c-N's main kernels
with every read of the expert POOLS (codes, LSB words, scales, biases; never activations or slot ids) through a
coherent(system) pointer, so a slot the CPU rewrites inside the running command buffer is read fresh (S7B-PROBE B4 / B6).
Built from the LIVE sources by substitution (tierglu_nax6n._A_N / _B_N, tierglu_3cn._SRC["a" / "b"] native): the live
modules are untouched. Two forms: "cn" = `coherent(system) const device` (non-volatile), "cv" = `volatile coherent(system)`.
The arithmetic is unchanged, so outputs are bitwise == the plain kernels (coh_check.py).
tier_glu_nax6n_coh(..., form) / tier_glu_n_coh(..., form): the same signatures as tier_glu_nax6n / tier_glu_n (bf16 pools)."""
import os, re, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_3cn as C3

SYS = """#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
"""
FORMS = {"cn": "coherent(system) const device", "cv": "volatile coherent(system) const device"}
POOL_PARAMS = {"w3g", "w3u", "w3d", "s3g", "s3u", "s3d", "b3g", "b3u", "b3d", "b2g", "b2u", "b2d", "l1g", "l1u", "l1d",
               "bg", "bu", "bd"}


def coherent(src, names, reassigned, cp):
    """every `const device TYPE* NAME = INIT;` / `const device TYPE* NAME;` for NAME in names becomes the coherent type
    (the initializer cast to it), and every reassignment `NAME = EXPR;` for NAME in reassigned is cast likewise"""
    types = {}
    def decl(m):
        ty, nm, init = m.group(1), m.group(2), m.group(3)
        if nm not in names:
            return m.group(0)
        types[nm] = ty
        return f"{cp} {ty}* {nm}" + (f" = ({cp} {ty}*)({init});" if init is not None else ";")
    out = re.sub(r"const device ([A-Za-z_0-9<>, ]+?)\*\s*(\w+)(?:\s*=\s*((?:[^;]|\n)+?))?;", decl, src)
    missing = set(names) - set(types)
    assert not missing, f"no declaration found for {missing}"
    for nm in reassigned:
        pat = re.compile(r"(?<![\w.])" + nm + r"\s*=\s*((?:[^;=]|\n)+?);")
        n0 = len(pat.findall(out))
        assert n0 >= 1, f"no reassignment of {nm}"
        out = pat.sub(lambda m: f"{nm} = ({cp} {types[nm]}*)({m.group(1)});", out)
    # nothing may still read a pool through a plain pointer: no plain `const device` declaration names a pool parameter
    for m in re.finditer(r"const device [A-Za-z_0-9<>, ]+?\*\s*(\w+)\s*=\s*((?:[^;]|\n)+?);", out):
        if m.group(0).startswith(cp.split(" const device")[0]):
            continue
        used = set(re.findall(r"\w+", m.group(2))) & (POOL_PARAMS | set(names))
        pre = out[max(0, m.start() - len(cp) - 2):m.start()]
        assert not used or cp.split(" const")[0] in pre, f"plain pool pointer left: {m.group(0)[:120]}"
    return out


_N6_NAMES = ["w3p", "s3p", "b3p", "b2p", "l1p", "blp", "wrow", "srow", "brow", "lrow", "c"]
_C3_A_NAMES = ["w3", "s3", "l1", "bl", "c", "lb", "b3"]
_C3_B_NAMES = ["w3", "s3", "l1", "bl", "c", "b3"]


def _n6_src(which, form):
    src = NN._A_N if which == "a" else NN._B_N
    return coherent(src, [n for n in _N6_NAMES if re.search(r"const device [A-Za-z_0-9<>, ]+?\*\s*" + n + r"\b", src)],
                    ["lrow", "brow"] if "lrow =" in src else [], FORMS[form])


def _c3_src(kind, form):
    src = C3._SRC[kind][1]
    names = [n for n in (_C3_A_NAMES if kind == "a" else _C3_B_NAMES)
             if re.search(r"const device [A-Za-z_0-9<>, ]+?\*\s*" + n + r"\b", src)]
    return coherent(src, names, [], FORMS[form])


@lru_cache(None)
def _kn6(which, form):
    if which == "a":
        return mx.fast.metal_kernel(name=f"nax6n_coh_a_{form}",
                                    input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u",
                                                 "bg", "bu", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
                                    output_names=["h"], source=_n6_src("a", form), header=N6._HDR4 + SYS)
    return mx.fast.metal_kernel(name=f"nax6n_coh_b_{form}",
                                input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx",
                                             "seg", "nseg"],
                                output_names=["y"], source=_n6_src("b", form), header=N6._HDR4 + SYS)


def tier_glu_nax6n_coh(x, msb_n, lsb_n, msb_slot, lsb_slot, form="cn", plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """tier_glu_nax6n (bf16 native pools) with coherent pool reads: the live function, its two kernels swapped"""
    a0, b0 = NN._kan, NN._kbn
    try:
        NN._kan, NN._kbn = (lambda: _kn6("a", form)), (lambda: _kn6("b", form))
        return NN.tier_glu_nax6n(x, msb_n, lsb_n, msb_slot, lsb_slot, plan=plan, grp=grp, nsg_b=nsg_b, ks=ks, ks_b=ks_b)
    finally:
        NN._kan, NN._kbn = a0, b0


def tier_glu_n_coh(x, msb_n, lsb_n, msb_slot, lsb_slot, form="cn", **kw):
    """3c-N's tier_glu_n with coherent pool reads in kernels A and B (the others are not used by tier_glu_n)"""
    k0 = C3._kern
    def kern(kind, p11=False):
        if kind in ("a", "b") and not p11:
            C3._sync()                                   # as 3c-N's own _kern: the sources follow BigMoE's live hooks
            return _kc3(kind, form)
        return k0(kind, p11)
    try:
        C3._kern = kern
        return C3.tier_glu_n(x, msb_n, lsb_n, msb_slot, lsb_slot, **kw)
    finally:
        C3._kern = k0


def _codes3_coh(form):
    """tg_codes3 from the live 3c header, its pool pointer parameter coherent (an overload: 3c-N passes it w3 + ...)"""
    h = C3.TG._HDR
    a = h.index("inline void tg_codes3(")
    b, depth = h.index("{", a), 0
    for i in range(b, len(h)):
        depth += {"{": 1, "}": -1}.get(h[i], 0)
        if depth == 0:
            break
    fn = h[a:i + 1]
    assert fn.count("const device") == fn.count("const device uint8_t*") == 2, "tg_codes3 reads other pointers"   # row, src
    return "\n" + fn.replace("const device uint8_t*", f"{FORMS[form]} uint8_t*") + "\n"


@lru_cache(None)
def _kc3(kind, form):
    """3c-N kernel `kind` with coherent pool reads: 3c-N's own inputs / output (its _KINDS), the rewritten source, and the
    internals pragma ahead of BigMoE's header (coherent(system) needs it)"""
    _, _, ins, out = C3._KINDS[kind]
    return mx.fast.metal_kernel(name=f"tierglu_n_coh_{kind}_{form}", input_names=ins, output_names=[out],
                                source=_c3_src(kind, form), header=C3.TG._HDR + SYS + _codes3_coh(form))
