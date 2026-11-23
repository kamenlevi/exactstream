"""refc_rowinv.py (Leap, Wed 18 Nov 2026 11:4x; CONTRACT YES, NewIdeas CONTRACT_YES_PLAN 899926d section 1 item b):
EvalSuite's VOID-gate self-test for the reference contract at the kernel level. Every contract form must be ROW-INVARIANT:
a row's output bits depend only on that row's input, never on how many other rows are in the call, their order, or which
rows a tree verify packs together. That is what makes tree-16 verify rows == chain-16 rows == AR rows (AR pads to 16)
for the contract's share of the forward. Tree masks act only on attention and the GDN recurrence, which the contract does
not touch (REF2-T's forms); the engine-level tree check (tree-16 == chain-16 == AR on every token) is BigMoE's M4C.
Cases, all bit for bit (uint16 views), random real-scale weights quantized 4-bit g64 in each contract shape:
  dense (mx.quantized_matmul under the contract, 7 shapes incl. lm_head): rows 1..16 vs the 16-row call's rows; 8 random
    tree-shaped row sets (any size 1-16, any order) vs the 16-row call's rows; the 3-D [1, rows, K] call form;
  HC read (leapk.s4_partials / s4_gemm under the contract, 16 padded rows): row permutations and zeroed tails;
  guards: contract OFF, ref2 OFF (r57's drafts) and a weight outside the target set all fall through to today's kernel,
    bit for bit; the contract counters move only when a contract form runs.
usage: [BigMoE venv] [REFC_ROOT=<a snapshot's scripts dir>] refc_rowinv.py   (GPU, ~2 GB; share lock). REFC_ROOT (default
/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts) is where refc, qmm16, leapk (ROOT) and ck, one32 (ROOT/week98) are imported from, so the VOID gate can test a
certified snapshot's own copies; the header prints every module's file and asserts they come from ROOT."""
import os
import sys

import numpy as np
import mlx.core as mx

ROOT = os.path.abspath(os.path.expanduser(os.environ.get("REFC_ROOT", "/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "week98"))
import refc

BF = mx.bfloat16
OUT = os.path.expanduser("~/Leap/data/week98/check25/refc_rowinv.txt")


class _RF:                                    # ref2's two attributes the contract reads (_ON: ref2 on; _DNR: native copies)
    _ON = [True]
    _DNR = {}


def same(a, b):
    a8 = a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32)
    b8 = b.view(mx.uint16) if b.dtype in (mx.bfloat16, mx.float16) else b.view(mx.uint32)
    return a.shape == b.shape and bool(mx.array_equal(a8, b8).item())


def main():
    rng = np.random.default_rng(1118)
    refc.install(_RF)
    import leapk as LK
    lines, bad, n = [], 0, 0

    def rec(ok, s):
        nonlocal bad, n
        n += 1
        bad += not ok
        lines.append(f"  {s} -> {'OK' if ok else 'BAD'}")
        if not ok:
            print(lines[-1], flush=True)

    q = lambda x, w, s, b: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    for (N, K) in refc.CON:
        W = mx.random.normal((N, K), dtype=BF, key=mx.random.key(N + K)) * (K ** -0.5)   # bf16 (lm_head: 1.3 GB)
        wq, ws, wb = mx.quantize(W, group_size=64, bits=4)
        del W
        X = mx.array((rng.standard_normal((16, K)) * 0.8).astype(np.float32)).astype(BF)
        mx.eval(wq, ws, wb, X)
        refc.ON[0], _RF._ON[0] = True, True
        c0 = refc.CNT["c16"]
        Y = q(X, wq, ws, wb)
        mx.eval(Y)
        engaged = refc.CNT["c16"] > c0
        rows_ok = all(same(q(X[:r], wq, ws, wb), Y[:r]) for r in range(1, 17))
        sets_ok = True
        for t in range(8):
            m = int(rng.integers(1, 17))
            idx = mx.array(rng.permutation(16)[:m].astype(np.int32))
            sets_ok &= same(q(X[idx], wq, ws, wb), Y[idx])
        d3_ok = same(q(X[None], wq, ws, wb)[0], Y)
        rec(engaged and rows_ok and sets_ok and d3_ok,
            f"dense {N}x{K}: contract form engaged {engaged}; rows 1-16 == the 16-row call {rows_ok}; 8 tree-shaped row sets "
            f"{sets_ok}; 3-D form {d3_ok}")
        orig = refc._S["qmm"](X, wq, ws, wb, transpose=True, group_size=64, bits=4)
        refc.ON[0] = False
        off_ok = same(q(X, wq, ws, wb), orig)
        refc.ON[0], _RF._ON[0] = True, False
        c1 = refc.CNT["c16"]
        r2off_ok = same(q(X, wq, ws, wb), orig) and refc.CNT["c16"] == c1
        _RF._ON[0], refc.WSET[0] = True, set()
        c2 = refc.CNT["c16"]
        wset_ok = same(q(X, wq, ws, wb), orig) and refc.CNT["c16"] == c2
        refc.WSET[0] = None
        diff_rows = int(sum(not same(Y[i], orig[i]) for i in range(16)))
        rec(off_ok and r2off_ok and wset_ok,
            f"dense {N}x{K} guards: contract OFF == today's kernel {off_ok}; ref2 OFF (drafts) == today's kernel, no contract "
            f"call {r2off_ok}; weight outside the target set == today's kernel {wset_ok}  (info: contract rows != today's "
            f"{diff_rows}/16, a new reference by design)")
        del wq, ws, wb
    HCD, NO = 4 * 2560, 320 + 4
    for sd in range(2):
        A = mx.array((rng.standard_normal((1, 16, HCD)) * 0.8).astype(np.float32)).astype(BF)
        Wh = mx.array((rng.standard_normal((NO, HCD)) * 0.02).astype(np.float32)).astype(BF)
        mx.eval(A, Wh)
        refc.ON[0], _RF._ON[0] = True, True
        g0 = refc.CNT["s4g"]
        Yg = LK.s4_gemm(A, Wh)
        cs, NP = LK.s4_partials(A, Wh)
        mx.eval(Yg, cs)
        perm = rng.permutation(16)
        pi = mx.array(perm.astype(np.int32))
        Yp = LK.s4_gemm(A[:, pi], Wh)
        csp, _ = LK.s4_partials(A[:, pi], Wh)
        perm_ok = same(Yp[0], Yg[0][pi]) and same(csp[:, :16], cs[:, :16][:, pi])
        tails_ok = True
        for m in (1, 5, 12):
            Az = mx.concatenate([A[:, :m], mx.zeros((1, 16 - m, HCD), BF)], axis=1)
            tails_ok &= same(LK.s4_gemm(Az, Wh)[0, :m], Yg[0, :m])
        engaged = refc.CNT["s4g"] > g0
        rec(engaged and perm_ok and tails_ok,
            f"HC read S4 (seed {sd}, 16 padded rows, NP {NP}): contract form engaged {engaged}; row permutation (gemm + "
            f"partials) {perm_ok}; zeroed tails at 1 / 5 / 12 real rows {tails_ok}")
        orig = refc._S["s4g"](A, Wh)
        _RF._ON[0] = False
        r2off_ok = same(LK.s4_gemm(A, Wh), orig)
        _RF._ON[0] = True
        rec(r2off_ok, f"HC read S4 (seed {sd}) guard: ref2 OFF == today's s4_gemm {r2off_ok}")
    class _G:
        pass
    f = lambda *sh, sc: (mx.array(rng.standard_normal(sh).astype(np.float32)) * sc).astype(BF)
    for sd in range(2):                       # REF2_HC_B with the contract: kernel B's sum on the contract's S4 partials
        g = _G()
        g.hc_count, g.hidden_size = 4, 2560
        g.input_mix_weight_down, g.input_mix_weight_up, g.hc_norm = _G(), _G(), _G()
        g.input_mix_weight_down.weight = f(320, HCD, sc=0.02)
        g.input_mix_weight_up.weight = f(2560, 320, sc=0.05)
        g.hc_norm.weight, g.hc_norm.eps = (1.0 + f(HCD, sc=0.1)).astype(BF), 1e-6
        wcat = mx.concatenate([g.input_mix_weight_down.weight, f(4, HCD, sc=0.02)], axis=0)
        hyper = f(16, HCD, sc=0.8)
        pend = (f(16, 2560, sc=0.5), (mx.sigmoid(f(16, 4, sc=1.0)) * 2).astype(BF))
        mx.eval(wcat, hyper, *pend)
        refc.ON[0], _RF._ON[0] = True, True
        _, np_ = LK.s4_partials(hyper.reshape(1, 16, HCD), wcat)
        okb = bool(LK.kernel_b_ok(4, 2560, 320, np_))
        A_ = LK.hc_read(g, wcat, hyper, pend, rows=16, kernel_b=False)
        B_ = LK.hc_read(g, wcat, hyper, pend, rows=16, kernel_b=True)
        mx.eval(*A_, *B_)
        eq = [same(x, y) for x, y in zip(A_, B_)]
        lines.append(f"  INFO (not counted) REF2_HC_B with the contract (seed {sd}): kernel B eligible at NP {np_}: {okb}; (mixed, "
                     f"written, inject) == the contract's S4 + gate1 path {eq}. INCONCLUSIVE at stub level: the same stub with the "
                     f"contract OFF (REF2-T's S4, NP 3) mismatches on mixed too (11-18 11:5x control), so it does not reproduce "
                     f"E14's engine-level bitwise D1-B; HC_B + the contract is settled only by an engine test (M4C with HC_B on)")
    refc.ON[0] = False
    mods = {m: getattr(sys.modules.get(m), "__file__", None) for m in ("refc", "qmm16", "ck", "leapk", "one32")}
    src_ok = all(f is not None and os.path.abspath(f).startswith(ROOT) for f in mods.values())
    bad += not src_ok
    hdr = (f"REFC ROW-INVARIANCE self-test (contract forms, EvalSuite's VOID gate, kernel level): {n - bad} of {n} cases PASS; "
           f"modules from ROOT {ROOT}: {src_ok} {mods}; {refc.report()}")
    print(hdr, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(hdr + "\n" + "\n".join(lines) + f"\nverdict: {'PASS' if not bad else 'FAIL'}\n")
    print("PASS" if not bad else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
