"""S2b item f: the two folds, chained (Leap, Sat 2026-11-07 01:0x; NewIdeas' QUEUE f270dee; Fable B DENSE_REBUILD sec 2).
  GDN:  R-83 (the stack's gdn_inproj_x) runs qkv | z (N 16384) on the dense native path and b | a (N 96) on SMALL-N at
        ks 10: two calls per GDN layer (36 per check). The fold: one call over qkv | z | b | a (+ 32 zero rows, N 16512 =
        a multiple of 64; a zero row gives exactly 0 and is sliced off) at qkv | z's ks.
  ATT:  q (N 12288) + k, v (N 512 each, SMALL-N, ks 10): three calls per attention layer (12 per check); the fold: one
        call over q | k | v (N 13312) at q's ks.
Per form, a dependent chain over distinct weight copies (cold reads, as in the check): layer l's input = x0 + 0 x (the
first value of EVERY output of layer l - 1), so the unfolded calls of a layer stay independent of each other (the
encoder may overlap them, as in situ) and the next layer waits for all of them. Per layer = (chain - links) / n,
round-robin over the forms, median of ROUNDS. ks per shape from ref2._dn_ks with the stack's REF2_DN_KS=1.
Bits: the folded call's rows == the unfolded calls at the SAME ks (the fold itself changes no arithmetic), and the
b | a / k | v rows vs today's ks 10 (max |diff| / max |ref|, the FAST class of DN-KS).
Line (NewIdeas): PASS >= 50 % of the price (GDN -1.6 -> >= 0.8 ms per check; ATT -1.1 -> >= 0.55), KILL < 25 %.
-> ~/Leap/data/fold_bench.txt. Share or quiet lock, MEM ~2.5 GB, ~1-2 min."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import qmm26
import ref2

OUT = os.path.expanduser("~/Leap/data/fold_bench.txt")
ROUNDS = int(os.environ.get("FOLD_ROUNDS", "7"))
M, K = 16, 2560
clock = time.perf_counter
rng = np.random.default_rng(11)


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


ref2.CFG["dn_ks"] = int(os.environ.get("REF2_DN_KS", "1") or "1")
say(f"\nFOLD bench {time.strftime('%a %H:%M:%S')} (MLX {mx.__version__}; {os.environ.get('FOLD_COND', 'share lock')}; "
    f"dn_ks v{ref2.CFG['dn_ks']}; ROUNDS {ROUNDS})")


def ks_of(N):
    return int(ref2._dn_ks(N, K))


def qpack(rows_list):
    """bf16 row blocks -> one native 4-bit g64 pack (rows quantized per row group: the same bits whatever the concat)"""
    w = mx.concatenate(rows_list, axis=0)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    wn, sn, bn = qmm26.repack(q, s, b)
    mx.eval(wn, sn, bn)
    return (wn, sn, bn), int(w.shape[0])


def rnd(N):
    return mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)


def build(kind, nl):
    """per layer: the unfolded packs (with their N, ks) and the folded pack"""
    L = []
    for _ in range(nl):
        if kind == "gdn":
            big, small = rnd(16384), rnd(96)
            pb, nb = qpack([big])
            ps, ns_ = qpack([small])
            pf, nf = qpack([big, small, mx.zeros((32, K), mx.bfloat16)])
            L.append(dict(un=[(pb, nb, ks_of(nb)), (ps, ns_, 10)], fo=(pf, nf, ks_of(nb)), cut=[nb, nb + ns_]))
        else:
            q_, k_, v_ = rnd(12288), rnd(512), rnd(512)
            pq, nq = qpack([q_])
            pk, nk = qpack([k_])
            pv, nv = qpack([v_])
            pf, nf = qpack([q_, k_, v_])
            L.append(dict(un=[(pq, nq, ks_of(nq)), (pk, nk, 10), (pv, nv, 10)], fo=(pf, nf, ks_of(nq)), cut=[nq, nq + nk, nq + nk + nv]))
    return L


def call(x, p, N, ks):
    wn, sn, bn = p
    return qmm26.qmm16n(x, wn, sn, bn, N, ks, 1)


def chain(L, x0, form, reps=5):
    def b(calls):
        prev = mx.zeros((1,), mx.float32)
        for l in L:
            x = x0 + (prev * 0).astype(x0.dtype)
            if not calls:
                prev = x.reshape(-1)[:1].astype(mx.float32)
                continue
            if form == "un":
                ys = [call(x, p, N, ks) for p, N, ks in l["un"]]
            else:
                p, N, ks = l["fo"]
                ys = [call(x, p, N, ks)]
            prev = sum(y.reshape(-1)[:1].astype(mx.float32) for y in ys)
        return prev
    tc, tn = [], []
    for _ in range(reps):
        for calls, acc in ((False, tn), (True, tc)):
            p_ = b(calls)
            t0 = clock(); mx.eval(p_); acc.append(clock() - t0)
    return 1e6 * (float(np.median(tc)) - float(np.median(tn))) / len(L)


x0 = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(x0)
for kind, nl, per_check, price in (("gdn", 36, 36, 1.6), ("att", 12, 12, 1.1)):
    L = build(kind, nl)
    l0 = L[0]
    say(f"  {kind}: unfolded {[(N, ks) for _, N, ks in l0['un']]} vs folded {(l0['fo'][1], l0['fo'][2])}; {nl} distinct layers")
    # bits: the fold's rows vs the unfolded calls at the fold's ks, and the small rows vs today's ks 10
    pf, nf, ksf = l0["fo"]
    yf = call(x0, pf, nf, ksf)
    eq_same, rel_today = [], []
    lo = 0
    for (p, N, ks), hi in zip(l0["un"], l0["cut"]):
        y_same = call(x0, p, N, ksf)
        y_today = call(x0, p, N, ks)
        seg = yf[:, lo:hi]
        eq_same.append(bool(mx.array_equal(seg, y_same).item()))
        den = float(mx.max(mx.abs(y_today.astype(mx.float32))).item()) or 1.0
        rel_today.append(float(mx.max(mx.abs(seg.astype(mx.float32) - y_today.astype(mx.float32))).item()) / den)
        lo = hi
    pad_zero = bool(mx.all(yf[:, lo:] == 0).item()) if yf.shape[-1] > lo else True
    say(f"    bits: fold == unfolded at the same ks {eq_same}; vs today's ks (small calls at 10) max rel diff "
        f"{[f'{r:.1e}' for r in rel_today]}; pad rows zero {pad_zero}")
    chain(L, x0, "un", 2); chain(L, x0, "fo", 2)
    tu, tf = [], []
    for r in range(ROUNDS):
        for form in (("un", "fo") if r % 2 == 0 else ("fo", "un")):
            (tu if form == "un" else tf).append(chain(L, x0, form))
    mu, mf = float(np.median(tu)), float(np.median(tf))
    save = (mu - mf) * per_check / 1e3
    verdict = "PASS" if save >= 0.5 * price else ("KILL" if save < 0.25 * price else "BETWEEN")
    say(f"    per layer: unfolded {mu:.1f} us, folded {mf:.1f} us -> {mu - mf:+.1f} us; x {per_check} per check = {save:+.2f} ms "
        f"(price {price}: {verdict}; rounds unfolded {[round(t, 1) for t in tu]}, folded {[round(t, 1) for t in tf]})")
    del L
    mx.clear_cache()
say("FOLD bench end")
