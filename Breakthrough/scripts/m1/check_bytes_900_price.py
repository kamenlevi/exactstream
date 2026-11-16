"""check_bytes_900_price.py (Breakthrough, Mon 16 Nov 2026 12:4x; pre-registered in docs/CHECK_BYTES_900.md section 1,
9bed364). CPU only: ~/BigMoE/.venv/bin/python -I -B check_bytes_900_price.py

A. Code 4-bit-eq and predicted code flips (experts only) for the scan options. Runs scripts/m1/dense_cut_price.py from its
   file (output redirected) and reuses its tier rule and its 7-arm calibration. The 4-bit-eq is counted on code blocks only.
B. GB / floor / realistic ms. Runs Fit12's ~/Fit12/scripts/lossy32_price.py (sha256 74f11370dd78) from its file with OUT
   redirected (as e62c405). Rows are added through a copy of its cand() with one extra input: an int8 factor on the dense
   matmul classes. int8 removes a share f of each class's 16 -> 32-row growth term: f = 1 at the realistic low end, and
   1 - 1/1.55 at the high end [est].
C. Leap's bound (~/Leap/scripts/week98/check25_bound.py's method) on code bytes:
   dense max(8.53 ms bf16 or 4.93 int8, bytes / 280) + experts / 280 + cores 1.16.
"""
import contextlib
import hashlib
import io
import json
import os

import numpy as np

H = os.path.expanduser
OUTD = H("~/Breakthrough/data/m1/check_bytes_900")
OUTF = H("~/Breakthrough/data/m1/check_bytes_900_price.out")
os.makedirs(OUTD, exist_ok=True)
L = []


def p(s=""):
    print(s, flush=True)
    L.append(s)


def run(src_path, replace, sha=None):
    raw = open(src_path, "rb").read()
    if sha:
        assert hashlib.sha256(raw).hexdigest().startswith(sha), src_path
    s = raw.decode("utf-8")
    for a, b in replace:
        assert s.count(a) == 1, (src_path, a)
        s = s.replace(a, b)
    g = {"__name__": "__main__", "__file__": src_path}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(s, src_path, "exec"), g)
    return g


# ---------------------------------------------------------------- A. code 4-bit-eq + predicted code flips
DSRC = H("~/Breakthrough/scripts/m1/dense_cut_price.py")
DOUT = 'open(os.path.expanduser("~/Breakthrough/data/m1/dense_cut_price.out"), "w")'
gA = run(DSRC, [(DOUT, f'open({os.path.join(OUTD, "dense_cut_price_rerun.out")!r}, "w")')])
same_dc = open(os.path.join(OUTD, "dense_cut_price_rerun.out")).read() == open(
    H("~/Breakthrough/data/m1/dense_cut_price.out")).read()
GIDX, W2, RD, RDOM, g0, nbl, TIERS3, NL, ROWS = (gA[k] for k in ("GIDX", "W2", "RD", "RDOM", "g0", "nbl", "TIERS3", "NL", "ROWS"))
gdom = np.full(g0, -1, np.int64)
gdom[GIDX.reshape(len(GIDX), -1)] = RDOM[:, None]
assert (gdom >= 0).all()
nbl_code = int((RDOM == 0).sum()) // ROWS * NL
assert sum(int((RDOM == d).sum()) // ROWS * NL for d in range(3)) == nbl


def option(s, lam):
    rw = s ** RD
    S = np.bincount(GIDX.ravel(), weights=(W2 * rw[:, None, None]).ravel(), minlength=g0)
    err = np.array([t[1] for t in TIERS3])
    size = np.array([t[2] for t in TIERS3])
    t = (err[None, :] * S[:, None] + lam * size[None, :]).argmin(1)
    keq_all = size[t].sum() / nbl
    keq_code = size[t][gdom == 0].sum() / nbl_code
    Erow = (err[t][GIDX] * W2).sum((1, 2)) / NL
    sel = RDOM == 0
    Ec = float((Erow[sel] * rw[sel]).sum() / rw[sel].sum())
    return keq_all, keq_code, Ec, gA["predict"](False, "code", Ec), gA["predict"](True, "code", Ec)


DENSE_INC = (1.25 - 1.13, 0.60 - 0.48)          # measured: bt_comb + shortlist minus bt_1c, 12 code windows (d9b2db7)
OPTS = [(0.98, 1.6e-2), (0.98, 1.8e-2), (0.98, 2.0e-2), (0.98, 2.2e-2), (0.935, 1.2e-2), (0.935, 1.6e-2)]
A = {o: option(*o) for o in OPTS}
ka, kc, _, pa, pn = A[(0.98, 1.8e-2)]
scA = (round(ka, 1) == 33.6 and round(kc, 1) == 35.7 and [round(x, 2) for x in pa] == [1.21, 1.06, 1.28]
       and [round(x, 2) for x in pn] == [0.59, 0.52, 0.63])
p("# CHECK BYTES (900 push): C6 / int8 / V4 and the <= 5.5 GB scan, code windows (Breakthrough check_bytes_900_price.py)")
p()
p(f"A. tiers (3 tiers: 4-bit / MSB-3 / drop) on the map's 32-row blocks; dense_cut_price.py re-run output identical to the "
  f"committed one: {same_dc}")
p(f"   SELF-CHECK A: s 0.98 / lambda 1.8e-2 = 33.6 pooled, 35.7 code, flips 1.21 [1.06, 1.28] / 0.59 [0.52, 0.63]: "
  f"{'PASS' if scA else 'FAIL'} (got {ka:.2f} / {kc:.2f}, {pa[0]:.3f} [{pa[1]:.3f}, {pa[2]:.3f}] / {pn[0]:.3f} [{pn[1]:.3f}, {pn[2]:.3f}])")
p("| tiers (s, lambda) | 4-bit-eq pooled | 4-bit-eq code | predicted code flips, experts only: all-row [LOO] / non-tie [LOO] | "
  f"+ measured V1 + V3 increment (+{DENSE_INC[0]:.2f} / +{DENSE_INC[1]:.2f}) [composed] |")
p("|---|---|---|---|---|")
for o in OPTS:
    ka, kc, Ec, pa, pn = A[o]
    p(f"| s {o[0]} lam {o[1]:.1e} | {ka:.1f} | {kc:.1f} | {pa[0]:.2f} [{pa[1]:.2f}, {pa[2]:.2f}] / {pn[0]:.2f} "
      f"[{pn[1]:.2f}, {pn[2]:.2f}] | {pa[0] + DENSE_INC[0]:.2f} / {pn[0] + DENSE_INC[1]:.2f} |")
p("   code lines: all-row 1.34, non-tie 0.87 (the map's verdict levels)")

# ---------------------------------------------------------------- B. Fit12's v2 ladder with rows added
FSRC = H("~/Fit12/scripts/lossy32_price.py")
gB = run(FSRC, [('OUT = "/Users/kamen/Fit12/data/lossy32"', f"OUT = {os.path.join(OUTD, 'fit12_published')!r}")],
         sha="74f11370dd78")
scB_a = open(os.path.join(OUTD, "fit12_published", "lossy32_price.txt"), "rb").read() == open(
    H("~/Fit12/data/lossy32/lossy32_price.txt"), "rb").read()
fast32, F16, GROW, miss32 = gB["fast32"], gB["F16"], gB["GROW"], gB["miss32"]
HC4_X, V1_HEAD_X, V4_GDN_X = gB["HC4_X"], gB["V1_HEAD_X"], gB["V4_GDN_X"]
TIER_RULE, CPU32ONE, C1C_CPU, CPU_EXTRA, MS_PER_GB, BW = (gB[k] for k in ("TIER_RULE", "CPU32ONE", "C1C_CPU", "CPU_EXTRA",
                                                                         "MS_PER_GB", "BW"))
KEPT_HE, KEPT_MB, OVH = gB["KEPT_HE"], gB["KEPT_MB"], gB["OVH"]
MATMUL = ("hc", "attp", "gdnw", "moe", "ple", "head")
F_INT8 = (1.0, 1 - 1 / 1.55)


def int8_extra(i):
    return {k: 1 - GROW[k] * F_INT8[i] / (1 + GROW[k]) for k in MATMUL}


def cand_x(name, dgb, egb, read_share, v1=False, v2=False, v3=False, v4=False, int8=False):
    """Fit12's cand(), plus int8 on the one-pass dense side (today's two-16-row route stays bf16)"""
    gb = dgb + egb
    out = []
    for lo in (True, False):
        i = 0 if lo else 1
        kw = dict(hc_x=HC4_X[i] if v3 else 1.0, ple_x=0.75 if v3 else 1.0, head_x=V1_HEAD_X[i] if v1 else 1.0,
                  gdn_x=V4_GDN_X[i] if v4 else 1.0)
        d = fast32(lo, extra=int8_extra(i) if int8 else None, **kw)
        d_bf = fast32(lo, **kw)
        e = egb * MS_PER_GB + (TIER_RULE[i] if v2 else 0.0)
        gpu = d + e
        cpu = CPU32ONE + (C1C_CPU if v2 else 0.0) + (CPU_EXTRA[i] if (v1 or v3) else 0.0)
        _, ms_m = miss32(read_share)
        tot = max(gpu, cpu) + ms_m + 0.0
        d2 = d_bf * 2 * sum(F16.values()) / fast32(lo)
        tot2 = max(d2 + e, cpu) + ms_m
        out.append((d, e, gpu, cpu, ms_m, tot, tot2))
    return dict(name=name, gb=gb, dgb=dgb, egb=egb, floor=gb / BW * 1e3, lo=out[0], hi=out[1], int8=int8, v4=v4)


dense, GB_EXP, c1_share, c1_map_eq = gB["dense"], gB["GB_EXP"], gB["c1_share"], gB["c1_map_eq"]
D3, D3V4, E_C5 = dense["l45_v1v3"], dense["l45_v1v3v4"], GB_EXP["c1_eng"]


def egb_of(eq):
    return gB["NL"] * eq * gB["E4"] * gB["ENG_OVER_MAP"]


def share_of(eq):
    return c1_share * eq / c1_map_eq


V123 = dict(v1=True, v2=True, v3=True)
ROWSX = []
for v4 in (False, True):
    for nm, eq in (("C5 (1c, lambda 1.6e-2)", None), ("C6 (lambda 1.8e-2)", 35.7)):
        for i8 in (False, True):
            lab = nm + (" + V4" if v4 else "") + (" + int8" if i8 else "")
            if nm.startswith("C6") and i8:
                lab += "  = G2" if v4 else "  = G1"
            egb = E_C5 if eq is None else egb_of(eq)
            sh = c1_share if eq is None else share_of(eq)
            ROWSX.append((cand_x(lab, D3V4 if v4 else D3, egb, sh, v4=v4, int8=i8, **V123), eq, (0.98, 1.6e-2 if eq is None else 1.8e-2)))
for o in OPTS[2:]:
    eq = round(A[o][1], 1)
    for i8 in (False, True):
        lab = f"scan: s {o[0]} lambda {o[1]:.1e} ({eq} eq)" + (" + int8" if i8 else "")
        ROWSX.append((cand_x(lab, D3, egb_of(eq), share_of(eq), int8=i8, **V123), eq, o))

# self-check B (b): rows without int8 equal Fit12's own cand() rows / e62c405
FIT = {c["name"][:2]: c for c in gB["C"]}


def same(a, b):
    return all(abs(x - y) < 1e-9 for x, y in zip(a["lo"] + a["hi"] + (a["gb"],), b["lo"] + b["hi"] + (b["gb"],)))


scB_b = (same(ROWSX[0][0], FIT["C5"]) and same(ROWSX[4][0], FIT["C8"])
         and round(ROWSX[2][0]["gb"], 2) == 5.77 and round(ROWSX[2][0]["floor"], 1) == 20.6)

# ---------------------------------------------------------------- C. Leap's bound on code bytes
DC_BF16, DC_INT8, CORES = 8.53, 4.93, 0.81 + 0.35


def bound(c):
    d = max(DC_INT8 if c["int8"] else DC_BF16, c["dgb"] / BW * 1e3)
    return d + c["egb"] / BW * 1e3 + CORES


b_c5 = (round(bound(ROWSX[0][0]), 1), round(bound(ROWSX[1][0]), 1))
b_c8 = (round(bound(ROWSX[4][0]), 1), round(bound(ROWSX[5][0]), 1))
scC = b_c5 == (23.9, 23.0) and b_c8 == (23.9, 21.8)

p()
p(f"B. Fit12's v2 ladder (its script, sha256 74f11370dd78). SELF-CHECK (a) published output reproduced byte for byte: "
  f"{'PASS' if scB_a else 'FAIL'}; (b) C5 / C8 rows == Fit12's cand(), C6 == e62c405 (5.77 GB, 20.6 ms): "
  f"{'PASS' if scB_b else 'FAIL'}")
p(f"C. Leap's bound. SELF-CHECK (c) C5 {b_c5} == (23.9, 23.0) and C8 {b_c8} == (23.9, 21.8): {'PASS' if scC else 'FAIL'}")
p()
p("| candidate (code windows, 32-row check, engine semantics) | GB (dense + experts) | <= 5.5 GB | floor ms | FAST dense ms | "
  "experts ms | GPU ms | CPU ms | misses ms | realistic ms | today's > 16-row route ms | tok/s HE+ / MBPP at K* | "
  "Leap's bound ms | efficiency for 25 ms | predicted code flips [composed] |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for c, eq, o in ROWSX:
    lo, hi = c["lo"], c["hi"]

    def rng(i):
        a, b = sorted((lo[i], hi[i]))
        return f"{a:.1f}" if abs(a - b) < 0.05 else f"{a:.1f}-{b:.1f}"
    tl, th = sorted((lo[5], hi[5]))
    tok = (f"{1e3 * KEPT_HE / (th + OVH):.0f}-{1e3 * KEPT_HE / (tl + OVH):.0f} / "
           f"{1e3 * KEPT_MB / (th + OVH):.0f}-{1e3 * KEPT_MB / (tl + OVH):.0f}")
    bd = bound(c)
    _, _, _, pa, pn = A[o]
    if c["v4"]:
        q = "V4 = map KILL standalone (RTN 4.43 / 3.62, AWQ 3.86 / 2.99)"
    else:
        q = f"{pa[0] + DENSE_INC[0]:.2f} / {pn[0] + DENSE_INC[1]:.2f}"
        if o == (0.98, 1.6e-2):
            q = "MEASURED 1.09 / 0.62 (24 windows)"
    if c["int8"]:
        q += "; int8: no data"
    today = "n/a (int8 is one-pass)" if c["int8"] else rng(6)
    p(f"| {c['name']} | {c['gb']:.2f} ({c['dgb']:.2f} + {c['egb']:.2f}) | {'YES' if c['gb'] <= 5.5 else 'no'} | "
      f"{c['floor']:.1f} | {rng(0)} | {rng(1)} | {rng(2)} | {rng(3)} | {rng(4)} | **{rng(5)}** | {today} | {tok} | "
      f"{bd:.1f} | {100 * bd / 25:.0f} % | {q} |")
    rows.append(dict(name=c["name"], gb=round(c["gb"], 3), dense_gb=round(c["dgb"], 3), exp_gb=round(c["egb"], 3),
                     floor_ms=round(c["floor"], 2), realistic_ms=[round(tl, 1), round(th, 1)],
                     fast_dense_ms=[round(lo[0], 2), round(hi[0], 2)], bound_ms=round(bd, 2), quality=q))
p()
p("Notes: int8 = int8 activations on the FAST dense matmuls (hc, attp, gdnw, moe, ple, head); model [est], Leap's measured "
  "dn32q / qmm32n ratio replaces it. V4 per Fit12 (dense -0.335 GB, GDN / attention time x0.80-0.85 [est]). Flips "
  "[composed] = the experts-only prediction + the measured V1 + V3 increment on 12 windows; unscreened unless marked.")
json.dump(dict(selfchecks=dict(A=bool(scA), B_a=bool(scB_a), B_b=bool(scB_b), C=bool(scC)),
               scan={f"s{o[0]}_lam{o[1]:.1e}": dict(keq_all=round(A[o][0], 2), keq_code=round(A[o][1], 2),
                                                    flips_all=[round(x, 3) for x in A[o][3]],
                                                    flips_nontie=[round(x, 3) for x in A[o][4]]) for o in OPTS},
               rows=rows), open(os.path.join(OUTD, "check_bytes_900_price.json"), "w"), indent=1)
open(OUTF, "w").write("\n".join(L) + "\n")
