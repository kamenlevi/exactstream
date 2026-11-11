"""NEVER-WAIT read-out with NEVERWAIT_TEST.md 2.5's self-checks and 2.6's / section 4's lines (Fable FAR, Thu 5 Nov 2026).
CPU; reads the map's out/ JSONs and gate_dfl2.json only; prints (and writes nothing).
Self-checks, from the two self-check arms if they ran on the 12 code windows (else: nw_selfcheck.py's own 2-window run):
  (i)   qom_0_live_ball_c208_nwz+code == the twin: top-8 ids and log-probs exactly, every window.         FAIL otherwise
  (ii)  qom_0_live_ball_c208_nw0+code vs the twin: PASS = top-1 differs on <= max(2, 0.4 %) of the rows and the mean
        |d log p1| <= 0.02; FAIL = more than 3 % of the rows or a mean above 0.2 (a gross fault); between: REPORT (the
        arms still count; read nw1 beside nw0).
  (iii) every never-wait run that is present: no harness WARNING on any window, a never-wait record on every window,
        late writes > 0 (0 for nwz).
  A FAIL makes every arm line below INVALID.
Control: nw0 (if it ran on the 12 windows) against the twin is the split write's bf16 re-rounding alone, and nw1 against
nw0 is lateness itself; the engine form has the split write too, so the lines stay on nw1 against the twin.
Arms: for each arm against its own twin, on the 12 code windows' answer rows: all-row top-1 flips vs the baseline argmax
(the map's standing measure, as ~/Breakthrough/scripts/dm_s7/code_read.py), D = arm - twin in points, the paired McNemar
counts, and D split by the draft's cf band (>= 0.9 / < 0.9 / no draft; dm_gate's cf of the draft each row verifies).
Lines (code; T = the twin's flips, D = arm - T):
  nw1 at 208 slots   PASS  <= 1.34 % and D <= +0.40 ("clean" if D <= +0.15)
                     BETWEEN  flips in (1.34, 2.0] or D in (+0.40, +0.80]
                     KILL  flips > 2.0 or D > +0.80, and not clean at the published pool
  pool-conditional PASS: nw1 not PASS at 208 slots but clean (D <= +0.15) at the published pool
  nw1 at 266 slots (tier 1b, if run): nw1's own lines; 266-290 slots is where the engine's by-day base pool sits
  nw1r1 (rank 0 waits): PASS <= 1.34 % and D <= +0.25; KILL D > +0.60; it ships only if nw1 does not PASS
  the stack (split 0.99 + nw1, published pool): PASS <= 1.34 %
Usage (BigMoE venv): python -I -B nw_read.py                    # self-checks + tier 1 (skips what has not run)
                     python -I -B nw_read.py --pair <arm> <twin>  # any pair, e.g. a tier-2 arm (names as run, with +code)
                     NW_OUT=<dir> : another out/ folder for the runs (default ~/FableGod/decision_map/out)"""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
DM = os.path.expanduser("~/FableGod/decision_map"); OUT = os.path.expanduser(os.environ.get("NW_OUT", os.path.join(DM, "out")))
base = json.load(open(os.path.join(DM, "out", "baseline.json")))
CODE = [nm for nm, b in base.items() if b["dom"] == "code"]
_G = {}
def gate():
    if not _G:
        g = json.load(open(os.path.join(DM, "out", "gate_dfl2.json")))
        _G.update({nm: np.array([np.nan if v is None else v for v in x]) for nm, x in g.items()})
    return _G

def raw(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p))
    return R if all(nm in R and "tk" in R[nm] for nm in CODE) else None

def load(name):
    """(flips [rows] bool, cf band [rows]: 0 = cf >= 0.9, 1 = cf < 0.9, 2 = no draft, windows with a warning, nw records) or None."""
    R = raw(name)
    if R is None: return None
    f, band, warn, nw = [], [], 0, []
    for nm in CODE:
        m = min(len(base[nm]["tk"]), len(R[nm]["tk"])); a = base[nm]["a"]
        f.append(np.array(R[nm]["tk"][:m])[:, 0] != np.array(base[nm]["tk"][:m])[:, 0])
        g = gate()[nm][a:a + m]; band.append(np.where(np.isnan(g), 2, np.where(g >= 0.9, 0, 1)))
        warn += "warn" in R[nm]
        if "nw" in R[nm]: nw.append(R[nm]["nw"])
    return np.concatenate(f), np.concatenate(band), warn, nw

def pair(arm, twin, quiet=False):
    """Prints the pair's line; returns (arm flips %, twin flips %, D, z, a warning is present) or None when a run is missing."""
    X, Y = load(arm), load(twin)
    if X is None or Y is None or len(X[0]) != len(Y[0]):
        if not quiet: print(f"  {arm} vs {twin}: " + ("not run yet" if X is None else "the twin has not run (or a different row count)"))
        return None
    x, y, bd = X[0], Y[0], X[1]; n = len(x)
    a, b = int((x & ~y).sum()), int((y & ~x).sum()); z = (a - b) / max(np.sqrt(a + b), 1)
    fx, fy = 100 * x.mean(), 100 * y.mean()
    bands = "; ".join(f"{lab} {int(k.sum())} rows: arm {100 * x[k].mean():.2f} %, twin {100 * y[k].mean():.2f} %, D {100 * (x[k].mean() - y[k].mean()):+.2f} "
                      f"({int((x & ~y & k).sum())} / {int((y & ~x & k).sum())})" for lab, k in (("cf >= 0.9", bd == 0), ("cf < 0.9", bd == 1), ("no draft", bd == 2)) if k.any())
    note = ""
    if X[3]:
        note = (f"; never-wait d {X[3][0]['d']}: {sum(v['late'] for v in X[3]) / max(sum(v['rows'] for v in X[3]), 1):.2f} late terms per answer row, "
                f"{sum(v['writes'] for v in X[3])} late writes")
    if X[2] or Y[2]: note += f"; WARNING: {X[2]} arm / {Y[2]} twin windows carry a harness warning (rows misaligned): the pair is NOT valid"
    print(f"  {arm}: {fx:.2f} % ({int(x.sum())}/{n}) vs {twin} {fy:.2f} % ({int(y.sum())}/{n}): D {fx - fy:+.2f} points; only-arm / only-twin flips {a} / {b}, "
          f"z {z:+.2f}{note}\n      by the draft's cf: {bands}")
    return fx, fy, fx - fy, z, bool(X[2] or Y[2])

T208, NW1, R1 = "qom_0_live_ball_c208+code", "qom_0_live_ball_c208_nw1+code", "qom_0_live_ball_c208_nw1r1+code"
NWZ, NW0 = "qom_0_live_ball_c208_nwz+code", "qom_0_live_ball_c208_nw0+code"
TP, NWP = "qom_0_live_ball+code", "qom_0_live_ball_nw1+code"
T266, NW266 = "qom_0_live_ball_c266+code", "qom_0_live_ball_c266_nw1+code"   # tier 1b: the engine's by-day exposure (ZERO 5d)
TS, STK = "qom_0_split0.99_live_ball+code", "qom_0_split0.99_live_ball_nw1+code"

def selfcheck():
    """Prints the self-check lines; returns 'PASS', 'REPORT', 'FAIL' or None (the self-check arms have not run here)."""
    T, Z, N0 = raw(T208), raw(NWZ), raw(NW0); res = []
    if T is not None and Z is not None:
        ex = all(T[w]["tk"] == Z[w]["tk"] and T[w]["lp"] == Z[w]["lp"] for w in CODE)
        if ex: print(f"  self-check (i)   nwz == the twin exactly (top-8 ids and log-probs, {len(CODE)}/{len(CODE)} windows): PASS")
        else:
            fl = sum(int((np.array(T[w]["tk"])[:, 0] != np.array(Z[w]["tk"])[:, 0]).sum()) for w in CODE)
            dm = max(float(np.abs(np.array(T[w]["lp"])[:, 0] - np.array(Z[w]["lp"])[:, 0]).max()) for w in CODE)
            print(f"  self-check (i)   nwz is NOT the twin: {fl} top-1 flips, max |d log p1| {dm:.2e}: FAIL (nw_selfcheck.py runs the twin twice: it tells "
                  f"harness noise from a patch fault)")
        res.append("PASS" if ex else "FAIL")
    if T is not None and N0 is not None:
        rows = sum(len(T[w]["tk"]) for w in CODE)
        fl = sum(int((np.array(T[w]["tk"])[:, 0] != np.array(N0[w]["tk"])[:, 0]).sum()) for w in CODE)
        mean = sum(float(np.abs(np.array(T[w]["lp"])[:, 0] - np.array(N0[w]["lp"])[:, 0]).sum()) for w in CODE) / max(rows, 1)
        lim = max(2, int(0.004 * rows)); v = "PASS" if (fl <= lim and mean <= 0.02) else "FAIL" if (fl > 0.03 * rows or mean > 0.2) else "REPORT"
        print(f"  self-check (ii)  nw0 vs the twin: {fl} top-1 flips of {rows} rows ({100 * fl / max(rows, 1):.2f} %), mean |d log p1| {mean:.4f}: {v} "
              f"(PASS <= {lim} flips and mean <= 0.02; FAIL > {int(0.03 * rows)} flips or mean > 0.2; between: REPORT)")
        res.append(v)
    bad = []; seen = 0
    for name in (NWZ, NW0, NW1, R1, NWP, STK):
        R = raw(name)
        if R is None: continue
        seen += 1
        for w in CODE:
            if "warn" in R[w]: bad.append(f"{name} {w}: a harness WARNING")
            elif "nw" not in R[w]: bad.append(f"{name} {w}: no never-wait record")
            elif (name == NWZ) != (R[w]["nw"]["writes"] == 0): bad.append(f"{name} {w}: {R[w]['nw']['writes']} late writes")
    if seen:
        print(f"  self-check (iii) {seen} never-wait runs: " + ("no WARNING, a never-wait record on every window, late writes as expected: PASS" if not bad
              else f"FAIL: {len(bad)} windows, e.g. {'; '.join(bad[:3])}"))
        res.append("FAIL" if bad else "PASS")
    return None if not res else "FAIL" if "FAIL" in res else "REPORT" if "REPORT" in res else "PASS"

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--pair":
        print(f"NEVER-WAIT read-out, one pair ({len(CODE)} code windows; all-row top-1 flips vs the baseline argmax)")
        sys.exit(0 if pair(sys.argv[2], sys.argv[3]) else 1)
    print(f"NEVER-WAIT read-out, tier 1 ({len(CODE)} code windows; all-row top-1 flips vs the baseline argmax; lines: NEVERWAIT_TEST.md 2.5, 2.6 and 4)")
    sc = selfcheck()
    if sc is None: print("  self-check arms (nwz, nw0): not run on the 12 code windows (nw_selfcheck.py's 2-window run is the other form)")
    t = load(T208)
    if t is not None:
        ft = 100 * t[0].mean()
        print(f"  the twin at 208 slots ({T208}): {ft:.2f} % ({int(t[0].sum())}/{len(t[0])})" + (" : OVER the 1.34 % line by itself; read every '+' below against it" if ft > 1.34 else ""))
    a = pair(NW1, T208); p = pair(NWP, TP); r = pair(R1, T208); s = pair(STK, TS)
    m = pair(NW266, T266, quiet=True)
    if s is not None: pair(STK, TP, quiet=True)
    c0 = pair(NW0, T208, quiet=True)                             # the control: the split write alone (lateness 0)
    if c0 is not None and a is not None:
        print("  control: nw0 writes each late term straight after its own layer's write, so its D is the split write's bf16 "
              "re-rounding alone; nw1 against nw0 is lateness itself:")
        pair(NW1, NW0, quiet=True)
    V = []
    if sc == "FAIL" or any(x is not None and x[4] for x in (a, p, r, s, m)):
        V.append("INVALID: a self-check failed or a window carries a harness warning (see above); no arm line below counts until it is fixed")
    clean_pub = p is not None and p[2] <= 0.15
    if a is not None:
        f, _, D, _, _ = a
        if f <= 1.34 and D <= 0.40: v = "PASS" + (" (clean: D <= +0.15, it stacks with either split)" if D <= 0.15 else "")
        elif clean_pub: v = "pool-conditional PASS (not PASS at 208 slots, clean at the published pool: ships only where the engine's base-pool misses are <= ~3 per kept token)"
        elif f > 2.0 or D > 0.80: v = "KILL" if p is not None else "KILL at 208 slots (the published-pool arm has not run)"
        else: v = "BETWEEN (nw1r1 decides)"
        V.append(f"nw1 at 208 slots: {v}: {f:.2f} %, D {D:+.2f} (PASS <= 1.34 and D <= +0.40; BETWEEN to 2.0 / +0.80)")
    if p is not None: V.append(f"nw1 at the published pool: {'clean' if clean_pub else 'not clean'}: {p[0]:.2f} %, D {p[2]:+.2f} (clean = D <= +0.15)")
    if m is not None:
        v = "PASS" + (" (clean)" if m[2] <= 0.15 else "") if (m[0] <= 1.34 and m[2] <= 0.40) else "KILL" if (m[0] > 2.0 or m[2] > 0.80) else "BETWEEN"
        V.append(f"nw1 at 266 slots (the engine's by-day base-pool exposure by ZERO's estimate; nw1's lines): {v}: {m[0]:.2f} %, D {m[2]:+.2f}")
    if r is not None:
        f, _, D, _, _ = r
        v = "PASS" if (f <= 1.34 and D <= 0.25) else "KILL" if D > 0.60 else "BETWEEN"
        V.append(f"nw1r1 (rank 0 waits): {v}: {f:.2f} %, D {D:+.2f} (PASS <= 1.34 and D <= +0.25; KILL D > +0.60); it ships only if nw1 does not PASS")
    if s is not None: V.append(f"the stack (split 0.99 + nw1, published pool): {'PASS' if s[0] <= 1.34 else 'FAIL'}: {s[0]:.2f} % (PASS <= 1.34), D {s[2]:+.2f} vs split 0.99")
    if c0 is not None and a is not None:
        V.append(f"control (not a line): of nw1's D {a[2]:+.2f}, the split write alone (nw0) gives {c0[2]:+.2f}; lateness itself {a[2] - c0[2]:+.2f}")
    for v in V: print("VERDICT " + v)
    if not V: print("VERDICT none: no tier-1 arm has run")
