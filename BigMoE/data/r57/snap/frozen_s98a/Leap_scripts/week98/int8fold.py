"""int8fold.py (Leap, Mon 16 Nov 2026 21:1x; CHECK25 module 3, the folds in the engine): on top of int8d (installed and
ON), the 4-bit dense calls that share one input run as ONE grouped dispatch (dn32g.dn32gq).
  - The groups are learned, not listed. Within the verify phase, back-to-back int8d-native calls on the same input array
    (object identity) form a run; a run of >= 2 calls becomes a group, keyed by its first weight.
  - From then on, the group's first call computes every member in one dispatch, on int8d's own quantized input (its
    cache). The other members' calls return their outputs from a small cache keyed by (input object, weight object).
  - MODE[0] = "bit" (default): each member keeps int8d's K split (int8d._kd), so outputs == int8d's, bit for bit.
    MODE[0] = "ks1" (FAST): every member at ks 1 (the small calls' dn_ks-off K order); bits change for those calls.
  - A member whose call never comes or that arrives with another input object falls through to int8d. That is
    correct; the folded output is only wasted. COUNTS: folds, hits, learned, bad_plan (a group dn32g cannot plan;
    it then stays unfolded).
  install() after int8d.install(); ON[0]; MODE[0]; reset() forgets the groups and the cache; remove()
usage: [BigMoE venv] int8fold.py test   (synthetic native weights: the hook vs int8d's own path, bit for bit)"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98"))
import int8d as I8
import dn32g as DG

ON = [False]
MODE = ["bit"]
COUNTS = dict(folds=0, hits=0, learned=0, bad_plan=0)
GROUPS = {}            # id(first weight) -> dict(calls=[(w, args, kw)], ok=bool)
_RUN = [None, []]      # the current run: [input object, [(w, args, kw)]]
_CACHE = []            # [(input object, weight object, output)] waiting for their calls
_ORIG = {}


def _close_run():
    x, calls = _RUN
    if x is not None and len(calls) >= 2:
        w0 = calls[0][0]
        g = GROUPS.get(id(w0))
        if g is None or g["calls"][0][0] is not w0 or len(g["calls"]) != len(calls):
            GROUPS[id(w0)] = dict(calls=list(calls), ok=True)
            COUNTS["learned"] += 1
    _RUN[0], _RUN[1] = None, []


def _note(x, w, args, kw):
    if _RUN[0] is x:
        _RUN[1].append((w, args, kw))
    else:
        _close_run()
        _RUN[0], _RUN[1] = x, [(w, args, kw)]


def _fold(x, g, rows, lead, K):
    segs, n0s = [], []
    for (w, args, kw) in g["calls"]:
        nat = I8._native_of(x, w, args, kw)
        if nat is None:
            return None
        wn, sn, bn, N, K_, ks = nat
        if K_ != K:
            return None
        n0s.append(int(w.shape[0]) if not (I8.ref2._DNR and I8.ref2._DNR.get(id(w)) is not None) else N)
        segs.append((wn, sn, bn, N, I8._kd(ks, K) if MODE[0] == "bit" else 1))
    try:
        DG.plan(segs)
    except AssertionError:
        g["ok"] = False
        COUNTS["bad_plan"] += 1
        return None
    ys = DG.dn32gq(*I8._quant(x, rows, K), segs, native=True)
    out = []
    for y, (_, _, _, N, _), n0 in zip(ys, segs, n0s):
        if n0 != N:
            y = y[:, :n0]
        out.append(y.reshape(*lead, y.shape[-1]))
    I8.COUNTS["int8"] += len(out)
    I8.COUNTS["int8_rows"] += rows * len(out)
    return out


def _qmmf(x, w, *args, **kw):
    if ON[0] and I8.ON[0] and I8.ref2._ON[0]:
        lead = x.shape[:-1]
        rows = 1
        for d in lead:
            rows *= d
        if 1 <= rows <= I8.HI and I8._verify_phase() and I8._native_of(x, w, args, kw) is not None:
            for i, (cx, cw, y) in enumerate(_CACHE):
                if cx is x and cw is w:
                    del _CACHE[i]
                    COUNTS["hits"] += 1
                    _note(x, w, args, kw)
                    return y
            g = GROUPS.get(id(w))
            if g is not None and g["ok"] and g["calls"][0][0] is w:
                ys = _fold(x, g, rows, lead, x.shape[-1])
                if ys is not None:
                    _CACHE[:] = [(x, c[0], y) for c, y in zip(g["calls"][1:], ys[1:])] + _CACHE[:4]
                    COUNTS["folds"] += 1
                    _note(x, w, args, kw)
                    return ys[0]
            _note(x, w, args, kw)
            return _ORIG["qmm"](x, w, *args, **kw)
    _close_run()
    return _ORIG["qmm"](x, w, *args, **kw)


def install():
    """after int8d.install() (wraps its wrapper; idempotent); ON[0] switches it, int8d.ON[0] must be on too"""
    assert I8._ORIG, "int8d.install() first"
    if not _ORIG:
        _ORIG["qmm"] = mx.quantized_matmul
        mx.quantized_matmul = _qmmf


def reset():
    GROUPS.clear()
    _CACHE.clear()
    _RUN[0], _RUN[1] = None, []


def remove():
    if _ORIG:
        mx.quantized_matmul = _ORIG["qmm"]
        _ORIG.clear()
    ON[0] = False
    reset()


def test(seeds=2, Ms=(1, 7, 16, 17, 32)):
    """synthetic: int8d's wrapper over the real quantized_matmul, its native lookup and phase patched to synthetic native
    weights; a GDN-like pair and an attention-like quad on one input each, plus a lone call on another input.
    Forward 1 learns (outputs == int8d's), forwards 2-3 fold: MODE bit -> every output == int8d's bit for bit;
    MODE ks1 -> the ks-1 members equal, the others within int8 noise (max rel diff reported)"""
    import numpy as np
    import qmm26 as Q26
    rng = np.random.default_rng(1623)
    K = 2560
    shapes = dict(qkvz=(16384, 1), ba=(96, 10), q=(12288, 1), k=(512, 10), v=(512, 10), idx=(640, 10), lone=(2560, 4))
    W, NAT = {}, {}
    for nm, (N, ks) in shapes.items():
        wf = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(wf, group_size=64, bits=4)
        wn, sn, bn = Q26.repack(wq, s, b)
        mx.eval(wq, s, b, wn, sn, bn)
        W[nm] = (wq, s, b)
        NAT[id(wq)] = (wq, (wn, sn, bn, N, K, ks))
    saved = (I8._native_of, I8._verify_phase, I8.ref2._ON[0], dict(I8._ORIG), mx.quantized_matmul, I8.ON[0])

    def nat_of(x, w, args, kw):
        e = NAT.get(id(w))
        return e[1] if e is not None and e[0] is w and x.dtype == mx.bfloat16 else None
    I8._native_of, I8._verify_phase, I8.ref2._ON[0] = nat_of, (lambda: True), True
    real = mx.quantized_matmul
    I8._ORIG.clear()
    I8._ORIG["qmm"] = real
    mx.quantized_matmul = I8._qmm8
    I8.ON[0] = True
    lines, bad = [], 0
    try:
        install()

        def layer(x1, x2, x3):
            mm = lambda x, nm: mx.quantized_matmul(x, W[nm][0], W[nm][1], W[nm][2], transpose=True, group_size=64, bits=4)
            return [mm(x1, "qkvz"), mm(x1, "ba"), mm(x3, "lone"), mm(x2, "q"), mm(x2, "k"), mm(x2, "v"), mm(x2, "idx")]
        names = ["qkvz", "ba", "lone", "q", "k", "v", "idx"]
        for mode in ("bit", "ks1"):
            MODE[0] = mode
            reset()
            for k_ in COUNTS:
                COUNTS[k_] = 0
            eq = n = 0
            dmax = {nm: 0.0 for nm in names}
            for sd in range(seeds):
                for M in Ms:
                    xs_ = [mx.array(rng.standard_normal((1, M, K)).astype(np.float32)).astype(mx.bfloat16) for _ in range(3)]
                    ON[0] = False
                    ref = layer(*xs_)
                    mx.eval(ref)
                    ON[0] = True
                    for fwd in range(3):
                        out = layer(*xs_)
                        mx.eval(out)
                        for nm, a, r in zip(names, out, ref):
                            n += 1
                            same = bool(mx.array_equal(a, r))
                            eq += same
                            if not same:
                                d = float(mx.max(mx.abs(a.astype(mx.float32) - r.astype(mx.float32)))) / max(
                                    float(mx.max(mx.abs(r.astype(mx.float32)))), 1e-30)
                                dmax[nm] = max(dmax[nm], d)
            if mode == "bit":
                ok = eq == n and COUNTS["folds"] > 0 and COUNTS["hits"] > 0 and COUNTS["bad_plan"] == 0
            else:
                ok = (COUNTS["folds"] > 0 and all(dmax[nm] == 0.0 for nm in ("qkvz", "q", "lone"))
                      and all(dmax[nm] < 2e-2 for nm in names))
            bad += not ok
            lines.append(f"  MODE {mode:3s}: outputs == int8d's {eq}/{n}; counts {COUNTS}; max rel diff per call "
                         + ", ".join(f"{nm} {dmax[nm]:.1e}" for nm in names) + f" -> {'PASS' if ok else 'FAIL'}")
            print(lines[-1], flush=True)
    finally:
        remove()
        I8._native_of, I8._verify_phase, I8.ref2._ON[0] = saved[0], saved[1], saved[2]
        I8._ORIG.clear()
        I8._ORIG.update(saved[3])
        mx.quantized_matmul = saved[4]
        I8.ON[0] = saved[5]
    return bad, lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        b, lines = test()
        p = os.path.expanduser("~/Leap/data/week98/check25/int8fold_test.txt")
        open(p, "w").write("INT8FOLD unit test (synthetic native weights; the hook vs int8d's own path)\n" + "\n".join(lines)
                           + f"\nverdict: {'PASS' if not b else 'FAIL'}\n")
        print("PASS" if not b else "FAIL")
        sys.exit(1 if b else 0)
