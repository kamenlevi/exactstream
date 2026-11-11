"""Lossless exponent-offset pack for qwen4_exp hyper-connection (GatedResidual) mixer weights, plus a fused read that
decodes it in-kernel. Drop-in for MTPLX 2.12.2's `mtplx.kernels.hyper_connection.fused_hyper_read`.

Pack (per bf16 tensor, flat order, groups of 64):
  hi   u8  per weight  : sign bit (bit 7) | 7 mantissa bits
  off  u8  per 2 weights: 4-bit exponent offsets from the group's minimum exponent (low nibble = even index)
  base u8  per group   : the group's minimum exponent
  esc  u16 per group   : 0xFFFF if the group fits, else an index into `raw` (groups whose exponent spread is >= 16)
  raw  u16             : raw bf16 bits of the escape groups, 64 per group
Decode: bits = (hi & 0x80) << 8 | (base + off) << 7 | (hi & 0x7F), i.e. the original bf16 bits, exactly.
About 12.4 bits/weight vs 16 (the 125B's HC mixers: 99.5 % of groups fit).

The kernel is MTPLX's own `_SOURCE` with the three weight loads replaced by the decoder. Every arithmetic op and its
order are unchanged, so with identical decoded weights the outputs are bit-identical (asserted by the test below).
The replacements are checked, so a changed MTPLX source fails loudly instead of silently diverging.

Use at load (no files on disk):
    from hc_pack import pack_bf16, fused_hyper_read_packed
    module._hc_pack = (pack_bf16(module.input_mix_weight_down.weight),
                       pack_bf16(module.input_mix_weight_up.weight),
                       pack_bf16(module.block_inject_weight.weight) if combine else None)
    mixed, inject = fused_hyper_read_packed(x2, module.hc_norm.weight, *module._hc_pack)
then drop the bf16 tensors (keep them only if prefill still needs the eager path).
Test: python hc_pack.py [layer] (reads one layer of ~/BigMoE/models/bare via mmap; needs a GPU slot, ~5 s).
"""
import importlib.util
import json
import os
import struct
import sys
from functools import lru_cache

import numpy as np
import mlx.core as mx

MTPLX_HC = os.environ.get(
    "MTPLX_HC_KERNEL",
    os.path.expanduser("~/.cache/uv/archive-v0/dfadjselx9BIIozk/mtplx/kernels/hyper_connection.py"))
_spec = importlib.util.spec_from_file_location("mtplx_hc_ref", MTPLX_HC)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

G = 64


def pack_bf16(w: mx.array):
    """w: bf16 mx.array (any shape). Returns a tuple of 5 mx arrays (hi, off, base, esc, raw)."""
    u = np.array(w.reshape(-1).view(mx.uint16))
    assert u.size % G == 0, u.size
    ug = u.reshape(-1, G)
    e = ((ug >> 7) & 0xFF).astype(np.int16)
    base = e.min(1)
    fit = (e.max(1) - base) < 16
    off = np.where(fit[:, None], e - base[:, None], 0).astype(np.uint8).reshape(-1)
    hi = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)
    nesc = int((~fit).sum())
    assert nesc < 0xFFFF
    esc = np.full(len(ug), 0xFFFF, np.uint16)
    esc[~fit] = np.arange(nesc, dtype=np.uint16)
    raw = ug[~fit].reshape(-1) if nesc else np.zeros(G, np.uint16)
    offp = (off[0::2] | (off[1::2] << 4)).astype(np.uint8)
    return tuple(mx.array(a) for a in (hi, offp, base.astype(np.uint8), esc, raw))


def unpack_np(p):
    hi, offp, base, esc, raw = (np.array(a) for a in p)
    n = hi.size
    off = np.empty(n, np.uint16)
    off[0::2] = offp & 0xF
    off[1::2] = offp >> 4
    g = np.arange(n) // G
    bits = ((hi.astype(np.uint16) & 0x80) << 8) | ((base[g].astype(np.uint16) + off) << 7) | (hi & 0x7F)
    m = esc[g] != 0xFFFF
    bits[m] = raw[esc[g][m].astype(np.int64) * G + (np.arange(n)[m] % G)]
    return bits


def packed_bytes(p):
    return sum(a.nbytes for a in p)


_DEC = """
inline float hcdec(device const uchar* hi, device const uchar* off, device const uchar* base,
                   device const ushort* esc, device const ushort* raw, uint i) {
    const uint g = i >> 6;
    const ushort e = esc[g];
    uint bits;
    if (e != 0xFFFF) {
        bits = raw[(uint)e * 64 + (i & 63)];
    } else {
        const uint h = hi[i];
        const uint o = (off[i >> 1] >> ((i & 1) * 4)) & 0xF;
        bits = ((h & 0x80) << 8) | (((uint)base[g] + o) << 7) | (h & 0x7F);
    }
    return as_type<float>(bits << 16);
}
"""

_REPL = [
    ("device const T* wrow = wd + (size_t)orow * HCD;", "const uint wbase = (uint)orow * HCD;"),
    ("acc += (float)normed[i] * (float)wrow[i];",
     "acc += (float)normed[i] * hcdec(wd_hi, wd_off, wd_base, wd_esc, wd_raw, wbase + i);"),
    ("device const T* irow = wi + (size_t)sg * HCD;", "const uint ibase = (uint)sg * HCD;"),
    ("acc += (float)normed[i] * (float)irow[i];",
     "acc += (float)normed[i] * hcdec(wi_hi, wi_off, wi_base, wi_esc, wi_raw, ibase + i);"),
    ("device const T* urow = wu + (size_t)pos * R;", "const uint ubase = (uint)pos * R;"),
    ("a += mixv[j] * (float)urow[j];", "a += mixv[j] * hcdec(wu_hi, wu_off, wu_base, wu_esc, wu_raw, ubase + j);"),
]
_NAMES = [f"{w}_{p}" for w in ("wd", "wu", "wi") for p in ("hi", "off", "base", "esc", "raw")]


@lru_cache(maxsize=2)
def _kernel(has_inject: bool):
    src = ref._SOURCE
    for a, b in _REPL:
        assert src.count(a) == 1, f"MTPLX hyper_connection source changed; cannot find: {a}"
        src = src.replace(a, b)
    return mx.fast.metal_kernel(
        name=f"moonshots_hc_read_packed_{'inj' if has_inject else 'mix'}",
        input_names=["x", "gamma"] + _NAMES,
        output_names=["mixed", "inject"],
        header=ref._HEADER + _DEC,
        source=src,
    )


def fused_hyper_read_packed(x2d, gamma, pd, pu, pi=None):
    """Same contract as MTPLX fused_hyper_read(x2d, gamma, wd, wu, wi), with packed weights."""
    S = x2d.shape[0]
    has_inject = pi is not None
    outs = _kernel(has_inject)(
        inputs=[x2d, gamma, *pd, *pu, *(pi if has_inject else pd)],
        template=[("T", x2d.dtype), ("HAS_INJECT", 1 if has_inject else 0)],
        grid=(1024, S, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(S, ref.D_HIDDEN), (S, ref.HC)],
        output_dtypes=[x2d.dtype, x2d.dtype],
    )
    return outs[0], outs[1]


def _load(D, names):
    idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for nm in names:
        f = f"{D}/{idx[nm]}"
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            m = json.loads(fh.read(n))[nm]
        a, b = m["data_offsets"]
        u = np.array(np.memmap(f, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,))).reshape(m["shape"])
        out[nm.rsplit(".", 2)[-2]] = mx.array(u).view(mx.bfloat16)
    return out


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    D = os.path.expanduser("~/BigMoE/models/bare")
    ok = True
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        pre = f"language_model.model.layers.{layer}.{part}."
        w = _load(D, [pre + s for s in ("hc_norm.weight", "input_mix_weight_down.weight",
                                         "input_mix_weight_up.weight", "block_inject_weight.weight")])
        packs = [pack_bf16(w[k]) for k in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")]
        for k, p in zip(("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"), packs):
            rt = np.array_equal(unpack_np(p), np.array(w[k].reshape(-1).view(mx.uint16)))
            ok &= rt
            print(f"{part}.{k}: {w[k].nbytes/1e6:.2f} MB -> {packed_bytes(p)/1e6:.2f} MB "
                  f"({packed_bytes(p)/w[k].nbytes:.0%}), escapes {int((np.array(p[3]) != 0xFFFF).sum())}, round-trip exact: {rt}")
        for S in (1, 3, 4):
            x = (mx.random.normal((S, ref.HCD)) * 2).astype(mx.bfloat16)
            a = ref.fused_hyper_read(x, w["hc_norm"], w["input_mix_weight_down"], w["input_mix_weight_up"],
                                     w["block_inject_weight"])
            b = fused_hyper_read_packed(x, w["hc_norm"], *packs)
            eq = all(mx.array_equal(p, q).item() for p, q in zip(a, b))
            ok &= eq
            print(f"  S={S}: packed fused read == MTPLX fused read, bitwise: {eq}")
    print("ALL EXACT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
