"""Code-only arm reader (Fable C's audit AUDIT_1105 sections 2-3: reach07 / band5 / top1 "code only"; Breakthrough, Thu
2026-11-05). For "<arm>+code" (the 12 code windows; dm_screen_read.py needs all 36): all-row top-1 flips vs the baseline argmax
(out/baseline.json) on the code rows, the audit's lines (PASS <= 1.34 %, BETWEEN <= 2.0 %, KILL above), and a paired McNemar
vs each twin on the same code rows (the twin's full or code-only run). CPU; reads the map's out/ JSONs only; prints.
Usage (BigMoE venv): python code_read.py <arm>+code <twin> [<twin> ...]
NON-TIE READ (Fable C, Thu 2026-11-05, WALL_1105.md section 1): tie rows (floor_rows.json's cls, p1 - p2 <= 0.1) flip 20-45 %
under every arm, so the arm and each twin also get their non-tie flips, the McNemar is repeated on the non-tie rows, and
VERDICT2 = PASS only if all-row <= 1.34 % AND non-tie <= 0.87 % (gate_top2_cf099's own non-tie level on code). The VERDICT
line is unchanged. FLOOR_ROWS = another class file."""
import json, os, sys
import numpy as np
OUT = os.environ["READ_OUT"]   # run_1105: a folder of symlinks (the map's baseline and twins, this run's arms)
base = json.load(open(os.path.join(OUT, "baseline.json")))
CODE = [nm for nm, b in base.items() if b["dom"] == "code"]
_fr = os.path.expanduser(os.environ.get("FLOOR_ROWS", "~/FableGod/floor_rows.json"))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in json.load(open(_fr))} if os.path.exists(_fr) else None   # (window, absolute row) -> sure / mid / tie
NTL = 0.87   # gate_top2_cf099's non-tie flips on code: the 1.34 % level in non-tie terms

def ntmask(name):
    """True on the non-tie code rows, in fv()'s row order (rows without a class count as non-tie)"""
    R = json.load(open(os.path.join(OUT, name + ".json"))); v = []
    for nm in CODE:
        m = min(len(base[nm]["tk"]), len(R[nm]["tk"]))
        v.append(np.array([CLS.get((nm, int(base[nm]["a"]) + j)) != "tie" for j in range(m)]))
    return np.concatenate(v)

def fv(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); v = []
    for nm in CODE:
        if nm not in R or "tk" not in R[nm]: return None
        m = min(len(base[nm]["tk"]), len(R[nm]["tk"]))
        v.append(np.array(R[nm]["tk"][:m])[:, 0] != np.array(base[nm]["tk"][:m])[:, 0])
    return np.concatenate(v)

arm, twins = sys.argv[1], sys.argv[2:]
X = fv(arm)
if X is None: raise SystemExit(f"{arm}: not run or a code window missing")
f = 100 * X.mean(); verdict = "PASS" if f <= 1.34 else "BETWEEN" if f <= 2.0 else "KILL"
print(f"Code-only map screen: {arm}: {len(CODE)} code windows, {len(X)} rows; all-row flips vs the baseline argmax")
NT = ntmask(arm) if CLS is not None else None
if NT is not None and len(NT) != len(X): NT = None
if NT is not None:
    fn = 100 * X[NT].mean()
    print(f"  {arm}: non-tie rows {fn:.2f} % ({int(X[NT].sum())}/{int(NT.sum())}); tie rows {100 * X[~NT].mean():.1f} % ({int(X[~NT].sum())}/{int((~NT).sum())})")
for t in twins:
    Y = fv(t)
    if Y is None or len(Y) != len(X): print(f"  twin {t}: missing or a different row count"); continue
    a, b = int((X & ~Y).sum()), int((Y & ~X).sum())
    print(f"  vs {t} ({100 * Y.mean():.2f} %): only-{arm} / only-{t} flips {a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}")
    if NT is not None:
        a, b = int((X & ~Y & NT).sum()), int((Y & ~X & NT).sum())
        print(f"     non-tie rows ({t} {100 * Y[NT].mean():.2f} %): only-{arm} / only-{t} flips {a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}")
print(f"VERDICT {arm}: code {verdict} {f:.2f} % ({int(X.sum())}/{len(X)}; PASS <= 1.34, BETWEEN <= 2.0, KILL above)")
if NT is not None:
    print(f"VERDICT2 {arm}: code {'PASS' if (f <= 1.34 and fn <= NTL) else 'FAIL'} (all-row {f:.2f} % <= 1.34 AND non-tie {fn:.2f} % <= {NTL:.2f})")
