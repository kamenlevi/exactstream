#!/usr/bin/env python3
"""convcache's copies keep every bit (T1c 11-11: the add-of-zero clone turned a -0.0 in row 0 of layer 35's QSA keys into
+0.0 in both branch copies). CPU only, a few MB: the real cache classes (qwen4_exp.QSACache, mlx_lm ArraysCache) with
planted -0.0 and NaN payloads, a source whose live buffers hold more rows than its offset (strided state, as after a
decode), then convcache's own paths: _prefix_state / _restore(copy) and _restore(live), the GDN checkpoint
(_snapshot_gdn) and its restore (_restore_gdn), the MTP cache copy; and the census (_x0msgs) must find the planted values.
Control: the old rule (x + 0, cache_state._clone_tree) must FAIL the same comparison, so the test can see the bug.
usage: ~/BigMoE/.venv/bin/python -I conv/test_bitclone.py"""
import os, re, sys

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)
from mlx_lm.models.cache import ArraysCache
from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convcache as CC

R, CAP, OFF, C = 4, 96, 80, 64                             # ratio, buffer rows, live offset (strided state), restore point
NEG0, NAN1 = 0x8000, 0xFFC1                                # bf16 -0.0, a NaN with a sign and a payload


def bits(x):
    return np.asarray(x.view(CC._UINT[x.dtype.size]))


def planted(shape, dtype, spots):
    """random values of dtype with bit patterns written at spots [(index tuple, bits)]"""
    u = CC._UINT[dtype.size]
    a = bits(mx.random.normal(shape).astype(dtype)).copy()
    for ix, b in spots:
        a[ix] = b
    return mx.array(a).view(dtype)


class Model:
    def make_cache(self):
        return [Q.QSACache(R), ArraysCache(size=2)]

    def make_mtp_cache(self):
        return [Q.QSACache(R)]


def qsa_source():
    q = Q.QSACache(R)
    k = planted((1, 2, CAP, 16), mx.bfloat16, [((0, 1, 0, 5), NEG0), ((0, 0, 0, 7), NAN1), ((0, 0, 9, 3), NEG0)])
    v = planted((1, 2, CAP, 16), mx.bfloat16, [((0, 1, 3, 2), NEG0)])
    raw = planted((1, CAP, 8), mx.bfloat16, [((0, 0, 1), NEG0)])
    pooled = planted((1, CAP // R, 8), mx.bfloat16, [((0, 0, 4), NEG0)])
    q.state = (k, v, raw, pooled)
    q.trim(CAP - OFF)                                      # offset OFF < CAP: state returns strided slices
    mx.eval(q.state)
    return q


def gdn_source():
    g = ArraysCache(size=2)
    conv = planted((1, 3, 24), mx.bfloat16, [((0, 0, 2), NEG0), ((0, 2, 5), NAN1)])
    rec = planted((1, 2, 4, 4), mx.float32, [((0, 1, 2, 3), 0x80000000), ((0, 1, 3, 0), 0xFFC00001)])
    g.state = [conv, rec]
    mx.eval(g.state)
    return g


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(bits(a), bits(b))


def check(tag, got, want, out):
    ok = all(same(x, y) for x, y in zip(got, want)) and len(got) == len(want)
    out.append((tag, ok))
    print(f"{'PASS' if ok else 'FAIL'} {tag}")


def main():
    mx.random.seed(1)
    CC.S["g"] = {"model": Model()}
    out = []
    # the source entry: a QSA layer (strided live state) + a GDN layer; its checkpoint at C taken the way _run_cm takes it
    src = [qsa_source(), gdn_source()]
    mtp = [qsa_source()]
    want_q = [x[..., :C, :] if x.ndim == 4 else x[:, :C if i < 3 else C // R, :] for i, x in enumerate(src[0].state)]
    want_m = [x[..., :C, :] if x.ndim == 4 else x[:, :C if i < 3 else C // R, :] for i, x in enumerate(mtp[0].state)]
    want_g = list(src[1].state)
    snap = CC._snapshot_gdn(src)
    check("checkpoint (_snapshot_gdn) holds the GDN bits", list(snap.states[1]), want_g, out)
    src_bits = [bits(x).copy() for x in src[0].state]
    e = CC.Entry(src)
    e.mtp = mtp
    e.ck = {C: CC.Ckpt(C, snap, None)}
    # the census sees every planted value (as the T1e census line would)
    cen = CC._x0msgs(src, upto=C, gdn=snap)
    print("census:", "; ".join(cen))
    # (an f32 NaN may pass an add of zero unchanged on the CPU: the last count is 1 or 2)
    out.append(("census finds the planted keys rows 0 + 9, values row 3, raw row 0, pooled block 0, the GDN values",
                cen[:5] == ["0[0] rows 0..9 (2 rows, 3 values, first 0xffc1)", "0[1] rows 3..3 (1 rows, 1 values, first 0x8000)",
                            "0[2] rows 0..0 (1 rows, 1 values, first 0x8000)", "0[3] rows 0..0 (1 rows, 1 values, first 0x8000)",
                            "gdn 1[0] 2 values (first 0x8000)"]
                and len(cen) == 6 and re.fullmatch(r"gdn 1\[1\] [12] values \(first 0x80000000\)", cen[5]) is not None))
    print(f"{'PASS' if out[-1][1] else 'FAIL'} {out[-1][0]}")
    # a branch: a new cache from [0, C) of the source, the GDN state from the checkpoint
    cache, m2, ck = CC._restore(e, C, True)
    check("copy: QSA keys / values / raw / pooled below C", list(cache[0].state), want_q, out)
    check("copy: GDN state from the checkpoint", list(cache[1].state), want_g, out)
    check("copy: MTP cache below C", list(m2[0].state), want_m, out)
    # writes into the copy (setitem, as a prefill's KV write) leave the source and the checkpoint as they were
    cache[0].kv.keys[..., 0:1, :] = mx.ones((1, 2, 1, 16), mx.bfloat16)
    g0 = cache[1].state[0]
    g0[0, 0, 0] = 7.0
    mx.eval(cache[0].kv.keys, g0)
    ok = all(np.array_equal(bits(x), b) for x, b in zip(src[0].state, src_bits))
    out.append(("copy: writes into the copy leave the source's bits", ok))
    print(f"{'PASS' if ok else 'FAIL'} {out[-1][0]}")
    cache2, _, _ = CC._restore(e, C, True)
    check("copy again after a write: the checkpoint kept its bits", list(cache2[1].state), want_g, out)
    # the same conversation: trim the live cache to C, GDN from the checkpoint
    cache3, m3, _ = CC._restore(e, C, False)
    check("live: QSA below C after the trim", list(cache3[0].state), want_q, out)
    check("live: GDN state from the checkpoint", list(cache3[1].state), want_g, out)
    # control: the old rule changes the planted bits (else this test could not see the T1c bug)
    old_q = [x + mx.zeros((), dtype=x.dtype) for x in want_q]
    old_g = CS._clone_tree(want_g)
    bad_q = not all(same(x, y) for x, y in zip(old_q, want_q))
    bad_g = not all(same(x, y) for x, y in zip(old_g, want_g))
    out.append(("control: x + 0 and cache_state._clone_tree change the planted bits", bad_q and bad_g))
    print(f"{'PASS' if out[-1][1] else 'FAIL'} {out[-1][0]}")
    ok = all(v for _, v in out)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(v for _, v in out)}/{len(out)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
