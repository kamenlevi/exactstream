"""dn32g.py (Leap, Mon 16 Nov 2026 21:0x; CHECK25 module 3, the folds): several 32-row 4-bit dense calls on ONE input in
ONE dispatch (dn32 / dn32q), each segment's tiles running dn32.py's own kernel text with that segment's K split, so every
tile's arithmetic and K order are the separate call's: grouped == separate, bit for bit (check() proves it per fold).

Why: the folds of SPLIT_1116.md module 3 (b|a into qkv|z; k, v and the indexer into att q). The small calls (3-20 tiles)
cost a launch and a fill / drain each (Fable B: 3-34x their byte floors), plus ~20-40 us of Python per call.

How: one threadgroup = SG simdgroups (32 SG threads). Segment s runs NT_s = SG / KS_s tiles per threadgroup (its K split
KS_s, as in its separate call); a KS 1 segment may end with a part-filled threadgroup (its spare simdgroups return; a
KS 1 tile has no barrier). Threadgroups are laid out segment after segment (offsets TGO_s); the threadgroup memory is one
array sized for the largest segment. NT never changes a tile's bits (each simdgroup owns its tile); KS does, so each
segment keeps its own.
  dn32g(x, segs, native=False) -> [y_s]          segs = [(wx, sx, bx, N, ks)], x [M, K] bf16 (dn32 form)
  dn32gq(xq, xs, xqs, segs, native=False) -> [y_s]                             (dn32q form, int8 activations)
  check() vs separate dn32 / dn32q calls on the folds' shapes (copy and native layouts, M 1..32); bench() rough
usage: [BigMoE venv] dn32g.py check | bench"""
from functools import lru_cache
import os
import re
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98"))
import dn32 as DN

HI = DN.HI
FOLDS = {   # name: [(segment, N, K, ks of its separate call)] (qmm32n.SHAPES; the separate calls' ks = qmm32n.ks_of)
    "GDN qkv|z + b|a": [("qkv|z", 16384, 2560, 1), ("b|a", 96, 2560, 10)],
    "att q + k + v + indexer": [("q", 12288, 2560, 1), ("k", 512, 2560, 10), ("v", 512, 2560, 10),
                                ("indexer qk", 640, 2560, 10)],
}


_RED0 = DN._SRC[DN._SRC.index("    if (KS > 1) {"):DN._SRC.index("    const int n0 = t * 32;")]
_RED1 = r"""    if (KS > 1) {   // dn32's split-K sum in rounds of RB partials (same order: kg 1, 2, .. KS - 1): 1 KB per simdgroup
        threadgroup float* part = (threadgroup float*)tb_all;
        for (int rb = 0; rb < KS - 1; rb += RB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kg > rb && kg <= rb + RB) for (short i = 0; i < 16; i++) part[(((kg - 1 - rb) * NT + nt) * 16 + i) * 32 + lane] = acc[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kg == 0)
                for (int gg = rb; gg < min(KS - 1, rb + RB); gg++)
                    for (short i = 0; i < 16; i++) acc[i] += part[(((gg - rb) * NT + nt) * 16 + i) * 32 + lane];
        }
        if (TWO) {
            for (int rb = 0; rb < KS - 1; rb += RB) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (kg > rb && kg <= rb + RB) for (short i = 0; i < 16; i++) part[(((kg - 1 - rb) * NT + nt) * 16 + i) * 32 + lane] = acc2[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (kg == 0)
                    for (int gg = rb; gg < min(KS - 1, rb + RB); gg++)
                        for (short i = 0; i < 16; i++) acc2[i] += part[(((gg - rb) * NT + nt) * 16 + i) * 32 + lane];
            }
        }
        if (kg != 0) return;
    }
"""
assert _RED0.count("threadgroup_barrier") == 4


def _body(s, rounds=True):
    """dn32's kernel text for segment s: its own constants / buffers, its threadgroup index, no own tb_all; rounds=True:
    the split-K sum in rounds of RB partials (bitwise the same order), so every segment needs only 1 KB per simdgroup"""
    b = DN._SRC
    decl = "    threadgroup uint32_t tb_all[TBW];\n"
    assert b.count(decl) == 1
    b = b.replace(decl, "")
    if rounds:
        b = b.replace(_RED0, _RED1)
    for a, z in ((r"\bN_C\b", f"N_C{s}"), (r"\bKS\b", f"KS{s}"), (r"\bNT\b", f"NT{s}"), (r"\bRB\b", f"RB{s}"),
                 (r"\bwx\b", f"GW{s}"),
                 (r"\bsx\b", f"GS{s}"), (r"\bbx\b", f"GB{s}"), (r"\by\[", f"GY{s}[")):
        b, n = re.subn(a, z, b)
        assert n > 0 or (a == r"\bRB\b" and not rounds), (a, s)
    tg = "threadgroup_position_in_grid.x"
    assert b.count(tg) == 1
    b = b.replace(tg, f"(int(threadgroup_position_in_grid.x) - TGO{s})")
    tl = f"    const int t = (int(threadgroup_position_in_grid.x) - TGO{s}) * NT{s} + nt;"
    i = b.index(tl)
    j = b.index("\n", i)
    b = b[:j + 1] + f"    if (KS{s} == 1 && t >= N_C{s} / 32) return;              // a part-filled KS 1 threadgroup\n" + b[j + 1:]
    return b


@lru_cache(None)
def _k(q8, native, S, rounds=True):
    names = (["xq", "xs", "xqs"] if q8 else ["x"])
    for s in range(S):
        names += [f"GW{s}", f"GS{s}", f"GB{s}"]
    src = "    threadgroup uint32_t tb_all[TBWMAX];\n    const int tgx = int(threadgroup_position_in_grid.x);\n"
    for s in range(S):
        head = "    if" if s == 0 else "    else if"
        cond = f"(tgx < TGO{s + 1})" if s < S - 1 else ""
        src += (f"{head} {cond} {{\n" if s < S - 1 else "    else {\n") + _body(s, rounds) + "    }\n"
    if S == 1:
        src = src.replace("    else {\n", "    {\n", 1)
    src = (src.replace("#if Q8", f"#if {int(q8)}").replace("#if !Q8", f"#if {int(not q8)}")
           .replace("#if NATIVE", f"#if {int(native)}").replace("Q8 ? 0x88888888u : 0u", "0x88888888u" if q8 else "0u"))
    return mx.fast.metal_kernel(name=f"leap_dn32g{'q' if q8 else ''}{'n' if native else ''}{S}{'r' if rounds else ''}",
                                input_names=names,
                                output_names=[f"GY{s}" for s in range(S)], source=src, header=DN._HDR)


def plan(segs, rounds=True):
    """(SG, [(NT_s, TG_s)], TBWMAX) for segs [(..., N, ks)]: SG = the largest ks; a ks > 1 segment needs SG % ks == 0
    and its tiles % (SG / ks) == 0; a ks 1 segment takes NT = SG (its last threadgroup may be part-filled)"""
    SG = max(sg[4] for sg in segs)
    out, tbw = [], (SG * 256 if rounds else 0)
    for (_, _, _, N, ks) in segs:
        tiles = N // 32
        assert N % 32 == 0 and SG % ks == 0, (N, ks, SG)
        nt = SG // ks
        assert ks == 1 or tiles % nt == 0, (N, ks, nt)
        assert DN._tg_ok(ks, nt), (ks, nt)
        rb = max(1, (SG * 256) // (nt * 512)) if rounds else ks
        out.append((nt, -(-tiles // nt), rb))
        if not rounds:
            tbw = max(tbw, max(ks * 256, (ks - 1) * 512) * nt)
    return SG, out, tbw


def _run(q8, xin, segs, native, dtype, rounds=True):
    M, K = xin[0].shape
    assert 1 <= M <= HI and all((K // 64) % sg[4] == 0 for sg in segs)
    SG, pl, tbw = plan(segs, rounds)
    S = len(segs)
    tmpl = [("T", dtype), ("K_C", K), ("M_C", M), ("TBWMAX", tbw)]
    off = 0
    for s, ((_, _, _, N, ks), (nt, tg, rb)) in enumerate(zip(segs, pl)):
        tmpl += [(f"N_C{s}", N), (f"KS{s}", ks), (f"NT{s}", nt), (f"RB{s}", rb), (f"TGO{s}", off)]
        off += tg
    tmpl.append((f"TGO{S}", off))
    ins = list(xin)
    for (wx, sx, bx, _, _) in segs:
        ins += [wx, sx, bx]
    return _k(q8, native, S, rounds)(inputs=ins, template=tmpl, grid=(off * 32 * SG, 1, 1), threadgroup=(32 * SG, 1, 1),
                             output_shapes=[(M, sg[3]) for sg in segs], output_dtypes=[dtype] * S)


def dn32g(x, segs, native=False, rounds=True):
    return _run(False, [x], segs, native, x.dtype, rounds)


def dn32gq(xq, xs, xqs, segs, native=False, dtype=mx.bfloat16, rounds=True):
    return _run(True, [xq, xs, xqs], segs, native, dtype, rounds)


def _mk(rng, N, K):
    import numpy as np
    import qmm26 as Q26
    w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    wn, sn, bn = Q26.repack(wq, s, b)
    wx, sx, bx = DN.layout(wq, s, b)
    mx.eval(wn, sn, bn, wx, sx, bx)
    return (wx, sx, bx), (wn, sn, bn)


def check(seeds=2, Ms=(1, 2, 7, 16, 17, 24, 32)):
    """grouped vs separate (dn32 and dn32q; copy and native layouts) per fold, bit for bit, every segment"""
    import numpy as np
    rng = np.random.default_rng(1621)
    lines, bad = [], 0
    for fold, parts in FOLDS.items():
        K = parts[0][2]
        ws = [_mk(rng, N, K) for (_, N, _, _) in parts]
        ok = n = 0
        for _ in range(seeds):
            X = mx.array(rng.standard_normal((HI, K)).astype(np.float32)).astype(mx.bfloat16)
            for M in Ms:
                x = X[:M]
                xq = DN.quant64(x)
                for native in (False, True):
                    segs = [(*w[1 if native else 0], N, ks) for w, (_, N, _, ks) in zip(ws, parts)]
                    s16 = [DN.dn32(x, *sg[:3], sg[3], sg[4], native=native) for sg in segs]
                    s8 = [DN.dn32q(*xq, *sg[:3], sg[3], sg[4], native=native) for sg in segs]
                    for rounds in (True, False):
                        g16 = dn32g(x, segs, native, rounds=rounds)
                        g8 = dn32gq(*xq, segs, native, rounds=rounds)
                        mx.eval(g16, g8, s16, s8)
                        for a, b in zip(g16 + g8, s16 + s8):
                            n += 1
                            ok += bool(mx.array_equal(a, b))
        bad += ok != n
        SG, pl, _ = plan([(None, None, None, N, ks) for (_, N, _, ks) in parts])
        lines.append(f"  {fold:28s} SG {SG:2d} (NT, TG, RB) {pl}: grouped == separate {ok}/{n} (dn32 + dn32q, copy + "
                     f"native, split-K sum in rounds + in one pass, M {','.join(map(str, Ms))})")
        print(lines[-1], flush=True)
    return bad, lines


def bench(rounds=9, reps=48, M=32):
    """ROUGH (share lock): a chain of `reps` grouped calls vs the separate calls, per fold, int8 native form, us per fold"""
    import time
    import numpy as np
    rng = np.random.default_rng(1622)
    lines = []
    for fold, parts in FOLDS.items():
        K = parts[0][2]
        ws = [_mk(rng, N, K)[1] for (_, N, _, _) in parts]
        segs = [(*w, N, ks) for w, (_, N, _, ks) in zip(ws, parts)]
        X = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(X)

        segs1 = [(*sg[:4], 1) for sg in segs]     # FAST variant: every segment at ks 1 (the small calls' dn_ks-off form)

        def chain(grouped):   # dependent, as in the forward: step i+1's input needs every segment of step i
            x = X
            for _ in range(reps):
                xq = DN.quant64(x)
                if grouped == "g0":
                    outs = dn32gq(*xq, segs, native=True, rounds=False)
                elif grouped == "g1":
                    outs = dn32gq(*xq, segs1, native=True)
                elif grouped == "s1":
                    outs = [DN.dn32q(*xq, *sg[:3], sg[3], 1, native=True) for sg in segs]
                elif grouped:
                    outs = dn32gq(*xq, segs, native=True)
                else:
                    outs = [DN.dn32q(*xq, *sg[:3], sg[3], sg[4], native=True) for sg in segs]
                x = outs[0][:, :K]
                for o in outs[1:]:
                    x = x + o[:, :1] * 0.0
            return [x]
        arms = [True, "g0", False, "g1", "s1"]
        ts = {a: [] for a in arms}
        for r in range(rounds):
            for g in (arms if r % 2 == 0 else arms[::-1]):
                mx.synchronize()
                t0 = time.perf_counter()
                o = chain(g)
                mx.eval(o)
                mx.synchronize()
                ts[g].append((time.perf_counter() - t0) / reps * 1e6)
        md = {a: float(np.median(ts[a][1:])) for a in arms}
        lines.append(f"  {fold:28s} M {M}, us per fold ({len(parts)} -> 1 dispatch): engine ks (bitwise): grouped "
                     f"{md[True]:6.1f} (rounds) / {md['g0']:6.1f} (one pass) vs separate {md[False]:6.1f} "
                     f"({md[True] - md[False]:+.1f} / {md['g0'] - md[False]:+.1f}); all ks 1 (FAST): grouped "
                     f"{md['g1']:6.1f} ({md['g1'] - md[False]:+.1f} vs the engine's separate), separate {md['s1']:6.1f}")
        print(lines[-1], flush=True)
    return lines


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    p = os.path.expanduser(f"~/Leap/data/week98/check25/dn32g_{cmd}.txt")
    if cmd == "check":
        b, lines = check()
        open(p, "w").write("DN32G check: grouped (one dispatch per fold) vs separate calls, bit for bit\n"
                           + "\n".join(lines) + f"\nverdict: {'PASS' if not b else 'FAIL'}\n")
        print("PASS" if not b else "FAIL")
        sys.exit(1 if b else 0)
    if cmd == "bench":
        lines = bench()
        open(p, "a").write("DN32G bench (ROUGH, share lock, int8 native, M 32, chained reps):\n" + "\n".join(lines) + "\n")
