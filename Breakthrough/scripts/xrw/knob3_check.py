"""knob3_check.py (Breakthrough, Tue 17 Nov 2026 21:2x; MOE BLOCK front fold, STACK FORM; gates G2 + G3 of docs/MOEBLOCK.md
"FRONT FOLD, STACK FORM (mode tgl16ef)", pre-registered 1b71659): the FROZEN knob_v3 as Leap's slot #4 will load it
(its modules imported from the knob dir FIRST, asserted by __file__), on the real pools of layers 20 and 44:
  G2  tgl16ef == tgl16e == the original one-call (ON off), bitwise in y, rs_m and rs_l, and tgl16ef twice identical:
      - knob_check's grid: 2 real 32-row blocks per layer with the 1c tiers' marks (ids replaced, 2-D nolsb);
      - the stack's form: 4 real blocks per layer, raw top-10 ids, no marks, 32 rows and their first 16;
      - xrw_robust's grid with no nolsb marks: M 1, 2, 3, 5, 8, 16, 17, 24, 32, a popular expert, QOM drops;
      xfront.COUNTS must show every tgl16ef call folded (no plain fallback).
  G3  knob_v3's tgl16.py / tgl16v.py / w4a8.py / xgrp.py byte-identical to knob_v2's (md5); xknob.py = v2's + the tgl16ef
      line (printed diff); xfront.py == scripts/xrw/xfront.py.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/knob3_check.py [knob_dir]   (share lock, MEM ~4 GB)
Writes data/xrw/knob3_check.txt."""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "data", "xrw", "knob_v3"))
KV2 = os.path.abspath(os.path.join(HERE, "..", "..", "data", "xrw", "knob_v2"))
os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, KD)
import xknob                                                # noqa: E402
import tgl16, tgl16v, w4a8, xgrp, xfront                    # noqa: E402,F401
MODS = (xknob, tgl16, tgl16v, w4a8, xgrp, xfront)
for m_ in MODS:
    assert os.path.dirname(os.path.abspath(m_.__file__)) == KD, (m_.__name__, m_.__file__)
import numpy as np                                          # noqa: E402
sys.path.insert(0, HERE)
import xrw_check as XC                                      # noqa: E402  (helpers only; the knob modules stay the knob dir's)
import xfront_check as FC                                   # noqa: E402  (real_blocks only)
assert XC.XK is xknob and FC.XF is xfront
for m_ in MODS:
    assert os.path.dirname(os.path.abspath(m_.__file__)) == KD, (m_.__name__, m_.__file__)

mx, BG, T, CQ = XC.mx, XC.BG, XC.T, XC.CQ
OUT = os.path.join(HERE, "..", "..", "data", "xrw", "knob3_check.txt")
L_ = []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


def g3():
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    ok = True
    for f in ("tgl16.py", "tgl16v.py", "w4a8.py", "xgrp.py"):
        a, b = md5(os.path.join(KD, f)), md5(os.path.join(KV2, f))
        ok &= a == b
        say(f"  G3 {f}: knob_v3 {a} == knob_v2 {b}: {a == b}")
    d = [ln for ln in difflib.unified_diff(open(os.path.join(KV2, "xknob.py")).read().splitlines(),
                                           open(os.path.join(KD, "xknob.py")).read().splitlines(), lineterm="", n=0)
         if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    minus = [ln for ln in d if ln.startswith("-")]
    plus = [ln for ln in d if ln.startswith("+")]
    exp = ['+    if mode == "tgl16ef":                                  # knob_v3: tgl16e + the front fold (xfront.py), EXACT',
           "+        import xfront", "+        return xfront.tier_glu_nax6n_1cq_tgl16ef"]
    # the docstring: one line of v2's becomes 3 (the knob_v3 note + the same line capitalised)
    xk_ok = (all(e in plus for e in exp) and len(plus) == len(exp) + 3 and len(minus) == 1
             and minus[0].startswith("-with ON[0] True"))
    ok &= xk_ok
    say(f"  G3 xknob.py vs knob_v2's: only the tgl16ef line + the docstring: {xk_ok}; diff:")
    for ln in d:
        say(f"     {ln}")
    a, b = md5(os.path.join(KD, "xfront.py")), md5(os.path.join(HERE, "xfront.py"))
    ok &= a == b
    say(f"  G3 xfront.py: knob_v3 {a} == scripts/xrw {b}: {a == b}; VAR {xfront.VAR}")
    return ok


def main():
    say(f"KNOB_V3 CHECK (G2 + G3): {KD}")
    ok3 = g3()
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    mx.eval(region, status, offs)
    xknob.install()
    rng = np.random.default_rng(1117)
    ok2, n, nf = True, 0, 0
    nxt = 0
    for Lm in (20, 44):
        r = ix["layers"][str(Lm)]
        base, _ = T.make_pools(ext, nxt, 512, 512, XC.SH4, d_, r)
        Ls, side, _ = T.make_sidecar(ext, 512, 512, XC.SH4, d_, r)
        Lb = nxt
        nxt = Ls + 1
        allids = np.arange(512, dtype=np.int32)
        for j in range(0, 512, 64):
            ext.ensure(Lb, mx.array(allids[j:j + 64]))
            ext.ensure(Ls, mx.array(allids[j:j + 64]))
        sm_of = np.array(ext.lookup(Lb, allids.tolist()), np.int32)
        sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)

        def case(tag, x, pm, pl, ids):
            nonlocal ok2, n, nf

            def run(on, mode=None):
                xknob.ON[0] = on
                if mode:
                    xknob.MODE[0] = mode
                y, rm, rl = CQ.tier_glu_nax6n_1cq_coh(x, base, side, mx.array(pm), mx.array(pl),
                                                      mx.array(ids.astype(np.int32)), XC.PLAN, region, offs, status)
                mx.eval(y, rm, rl)
                return y, rm, rl
            f0 = xfront.COUNTS["fold"]
            yo = run(False)
            ye = run(True, "tgl16e")
            yf = run(True, "tgl16ef")
            yf2 = run(True, "tgl16ef")
            xknob.ON[0] = False
            eq = lambda a, b: all(XC.bits_eq(u, v) for u, v in zip(a, b))
            e_o, f_e, f_o, rep = eq(ye, yo), eq(yf, ye), eq(yf, yo), eq(yf, yf2)
            folded = xfront.COUNTS["fold"] - f0 == 2
            good = e_o and f_e and f_o and rep and folded
            ok2 &= good
            n += 1
            nf += xfront.COUNTS["fold"] - f0
            say(f"  L{Lm} {tag}: tgl16e == orig {e_o}; tgl16ef == tgl16e {f_e}; tgl16ef == orig {f_o}; tgl16ef twice "
                f"identical {rep}; folded {folded}")
        for (f, b0, xf, idk, nolsb) in XC.blocks_for(Lm, 2):          # knob_check's grid (the 1c tiers' marks)
            pm = sm_of[idk].astype(np.int32)
            pl = np.where(nolsb, -2, sl_of[idk[:, :XC.R4]]).astype(np.int32)
            case(f"marks {f} rows {b0}-{b0 + 31}", mx.array(xf).astype(mx.bfloat16), pm, pl, idk)
        for (f, b0, xf, idr) in FC.real_blocks(Lm, 4):                 # the stack's form: raw ids, no marks
            for M in (32, 16):
                idm = idr[:M].astype(np.int64)
                case(f"stack {f} rows {b0}-{b0 + M - 1}", mx.array(xf[:M]).astype(mx.bfloat16),
                     sm_of[idm].astype(np.int32), sl_of[idm[:, :XC.R4]].astype(np.int32), idm)
        for M in (1, 2, 3, 5, 8, 16, 17, 24, 32):                     # xrw_robust's grid, no nolsb marks
            pool = rng.choice(512, size=max(12, 3 * M), replace=False)
            ids = np.stack([rng.choice(pool, 10, replace=False) for _ in range(M)]).astype(np.int64)
            if M >= 8:
                ids[:, 0] = pool[0]
            qom = set(rng.choice(pool[1:], size=2, replace=False).tolist())
            pm = np.where(np.isin(ids, list(qom)), -2, sm_of[ids]).astype(np.int32)
            pl = np.where(np.isin(ids[:, :2], list(qom)), -2, sl_of[ids[:, :2]]).astype(np.int32)
            x = mx.array((rng.standard_normal((M, 2560)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
            case(f"robust M {M:2d}", x, pm, pl, ids)
    say(f"  G2: {n} cases, {nf} folded calls; xknob {xknob.COUNTS}; xfront {xfront.COUNTS}")
    ok2 &= xfront.COUNTS["plain_big_p"] == 0 and xfront.COUNTS["plain_src"] == 0
    say(f"KNOB_V3 CHECK: G2 {'PASS' if ok2 else 'FAIL'}; G3 {'PASS' if ok3 else 'FAIL'}")
    open(OUT, "w").write("\n".join(L_) + "\n")
    return 0 if ok2 and ok3 else 1


if __name__ == "__main__":
    sys.exit(main())
