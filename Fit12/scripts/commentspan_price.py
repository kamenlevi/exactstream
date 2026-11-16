"""commentspan_price.py (Fit12, Mon 16 Nov 2026 13:3x; pre-registered in PLAN "COMMENT-EXACT PRICE"; NewIdeas 13:4x).
Desk, CPU, python -I, read-only inputs. Prices "exact inside comment spans" on 1c-tiers: rows tagged by lossy-45's causal
CodeScan (relax_core.CodeScan, as R57_RELAX_SPAN uses it) skip the 1c tiers and read the base's precision.
Scopes: '#' comments (in_comment()) and '#' comments + docstrings (CodeScan's doc state inside the fence).
"""
import glob
import importlib.util
import json
import os

import numpy as np
from tokenizers import Tokenizer

M = os.path.expanduser("~/FableGod/decision_map/out")
EV = os.path.expanduser("~/EvalSuite/results/card")
OUT = "/Users/kamen/Fit12/data/lossy32/commentspan_price.txt"
TK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
_sp = importlib.util.spec_from_file_location("relax_core_ro", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
RXC = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(RXC)     # read-only import of the engine's tagger
T4_K, T3_K, S, LAM, NL = 2.775, 0.8457, 0.98, 0.016, 48
E4, E3 = 2.135e-3, 2.135e-3 * 0.778
MS_GB = {16: 4.79, 32: 5.12}                 # engine experts ms per REF2-T GB (a082f8c pass K; v2 / v3)
MS_MISS, KEPT, OVH = 0.250, 16.91, 4.0
FIT = (41.5, 0.932)                          # Breakthrough's code law, all rows
ERR = np.array([0.0, 0.08, 1.0])
L = []


def p(x=""):
    print(x); L.append(x)


def tags(tok):
    """per row: (in '#' comment, in docstring) BEFORE the row's token, CodeScan fed the answer text causally. The map's
    answers begin right after the prompt's ```python fence (they start with '\\n' and close with ```): then the scan is
    primed with '```python' so the answer's first newline enters code, as in the engine; the gate's answers carry their
    own fence."""
    head = TK.decode(list(tok[:12])).lstrip()
    prime = "" if head.startswith("```") else "```python"
    sc = RXC.CodeScan(); sc.feed(prime); prev = ""; out = []
    for j in range(len(tok)):
        out.append((bool(sc.in_comment()), bool(sc.code and sc.doc)))
        cur = TK.decode(list(tok[:j + 1]))
        if cur.startswith(prev):
            sc.feed(cur[len(prev):])
        else:                                 # a multi-byte boundary moved: rescan from the start
            sc = RXC.CodeScan(); sc.feed(prime + cur)
        prev = cur
    return np.array(out, bool)


def tiers_np(ids, w):
    rw = S ** np.arange(ids.shape[0])
    u, inv = np.unique(ids.ravel(), return_inverse=True)
    imp = np.bincount(inv, weights=(w ** 2 * rw[:, None]).ravel(), minlength=len(u))
    return np.where(imp > T4_K * LAM, 0, np.where(imp > T3_K * LAM, 1, 2))[inv].reshape(ids.shape)


# ---------------------------------------------------------------- (a) tagged shares: the gate's base rows by set
share = {}
for f in glob.glob(os.path.join(EV, "es_cdlx45t1c1_*_tokens.json")):
    for k, v in json.load(open(f)).items():
        s = k.split("/")[0]
        if s not in ("humaneval", "humanevalx", "mbpp") or k in share.get(s, {}):
            continue
        x = list(v.values())[0]
        tok = x[0] if isinstance(x[0], list) else x
        share.setdefault(s, {})[k] = tags(tok)
p("# COMMENT-EXACT price (Fit12 commentspan_price.py)")
p("(a) tagged-row share, EvalSuite's lossy-45 base rows (tokens of its own answers):")
for s, d in sorted(share.items()):
    allt = np.concatenate(list(d.values()))
    p(f"  {s}: {len(d)} keys, {len(allt)} rows: '#' comment {100 * allt[:, 0].mean():.1f} %, docstring "
      f"{100 * allt[:, 1].mean():.1f} %, either {100 * (allt[:, 0] | allt[:, 1]).mean():.1f} %")

# ---------------------------------------------------------------- the map's code windows
base = json.load(open(os.path.join(M, "baseline.json")))
arm = json.load(open(os.path.join(M, "bt_1c.json")))
TIE = {}
for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))):
    TIE[(r["nm"], int(r["i"]))] = r["cls"] == "tie"
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
mt = []                                       # per row: (comment, doc, flip, tie)
E_rows = {"1c": [], "x_com": [], "x_both": []}
RW = []
acc = {R: {k: [] for k in ("ref_gb", "c1_gb", "xc_gb", "xb_gb", "ref_m", "c1_m", "xc_m", "xb_m", "xcr_m", "xbr_m",
                           "xcr_gb", "xbr_gb")} for R in (16, 32)}
for f in code:
    z = np.load(os.path.join(M, "xcap", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    m = ids.shape[0]
    tok = np.array(base[f]["tk"])[:, 0]
    assert len(tok) == m, (f, len(tok), m)
    tg = tags(tok)
    atk = np.array(arm[f]["tk"])[:, 0]
    tie = np.array([TIE.get((f, a + j), False) for j in range(m)])
    for j in range(m):
        mt.append((tg[j, 0], tg[j, 1], atk[j] != tok[j], tie[j]))
    res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + m]
    t1 = np.load(os.path.join(M, "bt", "tiers_1c", f))
    tier32 = np.where(t1["keep"][a:a + m], 0, np.where(t1["keep3"][a:a + m], 1, 2))
    rank = np.arange(10)
    for R in (16, 32):
        rk = np.broadcast_to(rank, (R, 10))
        for bi in range(m // R):
            sl = slice(bi * R, (bi + 1) * R)
            xc, xb = tg[sl, 0], tg[sl, 0] | tg[sl, 1]
            c = dict.fromkeys(acc[R], 0.0)
            e1 = np.zeros(R); e2 = np.zeros(R); e3 = np.zeros(R)
            for ly in range(NL):
                I, W, Rs = ids[sl, ly], w[sl, ly], res[sl, ly]
                T = tier32[sl, ly] if R == 32 else tiers_np(I, W)
                for e in np.unique(I):
                    se = I == e
                    nres = not Rs[se].all()
                    lsb_any = bool((rk[se] < 2).any())
                    c["ref_gb"] += E4 if lsb_any else E3; c["ref_m"] += nres
                    te = int(T[se].min())
                    if te != 2:
                        c["c1_gb"] += E4 if (te == 0 and lsb_any) else E3; c["c1_m"] += nres
                    for key, X in (("xc", xc), ("xb", xb)):
                        rows_x = np.broadcast_to(X[:, None], (R, 10))
                        used_x = bool((se & rows_x).any())
                        used_nc = te != 2 and bool((se & ~rows_x).any())
                        if not (used_x or used_nc):
                            continue
                        lsb = bool((se & rows_x & (rk < 2)).any()) or (te == 0 and bool((se & ~rows_x & (rk < 2)).any()))
                        c[key + "_gb"] += E4 if lsb else E3; c[key + "_m"] += nres
                        # + RES: an MSB-3-tier expert needed only by untagged rows is dropped when non-resident
                        if used_x or te == 0 or not nres:
                            c[key + "r_gb"] += E4 if lsb else E3; c[key + "r_m"] += nres
                if R == 32:
                    e1 += (ERR[T] * W ** 2).sum(1)
                    Tc = np.where(np.broadcast_to(xc[:, None], (R, 10)), 0, T)
                    Tb = np.where(np.broadcast_to(xb[:, None], (R, 10)), 0, T)
                    e2 += (ERR[Tc] * W ** 2).sum(1); e3 += (ERR[Tb] * W ** 2).sum(1)
            for k_ in acc[R]:
                acc[R][k_].append(c[k_])
            if R == 32:
                E_rows["1c"].extend(e1 / NL); E_rows["x_com"].extend(e2 / NL); E_rows["x_both"].extend(e3 / NL)
                RW.extend(S ** np.arange(R))
mt = np.array(mt, bool)
com, doc, flip, tie = mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3]
p(f"(a') the map's 12 code windows ({len(mt)} rows): '#' comment {100 * com.mean():.1f} %, docstring {100 * doc.mean():.1f} %, "
  f"either {100 * (com | doc).mean():.1f} %")

# ---------------------------------------------------------------- (c) measured 1c flips by tag
p()
p("(c) MEASURED 1c map flips per row (bt_1c top-1 != baseline top-1), by the row's tag:")
for lab, sel in (("all code rows", np.ones(len(mt), bool)), ("'#' comment rows", com), ("docstring rows", doc),
                 ("other (code / prose) rows", ~(com | doc))):
    nt = sel & ~tie
    p(f"  {lab}: {sel.sum()} rows, flips {flip[sel].sum()} ({100 * flip[sel].mean():.2f} %); non-tie {nt.sum()} rows, "
      f"flips {flip[nt].sum()} ({100 * flip[nt].mean():.2f} %)")
for lab, X in (("'#' comments exempt", com), ("comments + docstrings exempt", com | doc)):
    left = flip & ~X; left_nt = flip & ~X & ~tie
    p(f"  if {lab} (upper bound: their flips removed): all-row {100 * left.mean():.2f} % (from {100 * flip.mean():.2f}), "
      f"non-tie {100 * left_nt.sum() / (~tie).sum():.2f} % (from {100 * (flip & ~tie).sum() / (~tie).sum():.2f})")
RW = np.array(RW)
for k_ in ("1c", "x_com", "x_both"):
    E = np.array(E_rows[k_]); Er = float((E * RW).sum() / RW.sum())
    p(f"  law, reach-weighted E {k_}: {Er:.5f} -> predicted all-row flips {FIT[0] * Er ** FIT[1]:.2f} %")

# ---------------------------------------------------------------- (b) bytes, ms and misses per check
p()
p("(b) per check (48 layers): expert GB / expert ms / misses / miss ms")
p("| rows | REF2-T | 1c | 1c + '#' exempt | 1c + '#' + RES | 1c + '#' + doc exempt | 1c + '#' + doc + RES |")
p("|---|---|---|---|---|---|---|")
for R in (16, 32):
    A = {k: float(np.mean(v)) for k, v in acc[R].items()}
    cell = lambda g, mm: f"{g:.2f} GB / {g * MS_GB[R]:.1f} ms / {mm:.1f} / {mm * MS_MISS:.1f} ms"
    p(f"| {R} | {cell(A['ref_gb'], A['ref_m'])} | {cell(A['c1_gb'], A['c1_m'])} | {cell(A['xc_gb'], A['xc_m'])} | "
      f"{cell(A['xcr_gb'], A['xcr_m'])} | {cell(A['xb_gb'], A['xb_m'])} | {cell(A['xbr_gb'], A['xbr_m'])} |")
    acc[R] = A
A32 = acc[32]
p()
p("realistic 32-row check (scoreboard v3's model: FAST dense one pass + experts + 1c glue, max with the CPU, + misses):")
FAST = {"C3": (27.0, 28.6), "C5": (23.8, 26.6)}
for cn, (dl, dh) in FAST.items():
    for lab, gk, mk in (("1c", "c1_gb", "c1_m"), ("+ '#' exempt", "xc_gb", "xc_m"), ("+ '#' exempt + RES", "xcr_gb", "xcr_m"),
                        ("+ '#' + doc exempt", "xb_gb", "xb_m"), ("+ '#' + doc exempt + RES", "xbr_gb", "xbr_m")):
        e = A32[gk] * MS_GB[32]
        lo = max(dl + e + 1.8, 38.5) + A32[mk] * MS_MISS
        hi = max(dh + e + 3.4, 38.5) + A32[mk] * MS_MISS
        p(f"  {cn} {lab}: {lo:.1f}-{hi:.1f} ms -> {1e3 * KEPT / (hi + OVH):.0f}-{1e3 * KEPT / (lo + OVH):.0f} tok/s HE+ at K*")
open(OUT, "w").write("\n".join(L) + "\n")
