"""Fit12 T2 CHAT WIDTH (Sun 8 Nov 07:4x; pre-registration: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 CHAT WIDTH'): the
reader of /Users/kamen/Fit12/scripts/chw12.sh. Arm ex on chat12's 12 IFEval prompts, CAP 128, two processes:
  A = chw12a: F12_W1=4 (today's 4-row exact calls; the control),  R = chw12r: F12_W1=rand (first call 1-4 rows per check).
1. Tokens: A = chat12's tokens on the first 128 (the harness change is inert, the loop repeats); R = A in tokens AND in the
   kept-row logits hash (lsha) per prompt (the width is exact-preserving).
2. Cost per exact call (R's per-call log; check 0 dropped): ms = F[w] (one intercept per width, first calls) + F4' (later
   calls) + b * GB, least squares; MB and misses per call by width.
3. Pricing (from A's checks: kept and the first draft's cf, the same checks in R when 1. holds): a policy picks the first
   call's width from cf; calls = 1 + ceil((kept - w1) / 4) when kept > w1; each call costs its fitted ms. Reported: the
   exact ms per check, the decode ms per kept token (+ A's measured draft and other ms per check) and the [P] gain.
Usage: python -I -B chw12_read.py [DIR] [TAG_A] [TAG_R] [REF_IDS]"""
import glob, itertools, json, math, os, re, sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/chw12"
TA = sys.argv[2] if len(sys.argv) > 2 else "chw12a"
TR = sys.argv[3] if len(sys.argv) > 3 else "chw12r"
REF = sys.argv[4] if len(sys.argv) > 4 else "/Users/kamen/Fit12/data/scr/chat12/chat12_ids.json"


def arm_lines(tag):
    out = {}
    p_ = os.path.join(D, f"{tag}.txt")
    if os.path.exists(p_):
        for ln in open(p_):
            m = re.match(rf"^\[{re.escape(tag)}\] (\S+) ex theta None .*?\| tok (\d+) sha (\w+) \| lsha (\w+) .*?decode ([\d.]+) s", ln)
            if m:
                out[m.group(1)] = dict(tok=int(m.group(2)), sha=m.group(3), lsha=m.group(4), dec=float(m.group(5)))
    return out


def load(tag):
    r = {}
    for f in sorted(glob.glob(os.path.join(D, f"{tag}_*_ex.json"))):
        kf = os.path.basename(f)[len(tag) + 1:-len("_ex.json")]
        r[kf] = dict(log=json.load(open(f)), ph=json.load(open(f[:-5] + "_ph.json")),
                     toks=json.load(open(f[:-5] + "_toks.json")),
                     calls=json.load(open(f[:-5] + "_calls.json")) if os.path.exists(f[:-5] + "_calls.json") else [])
    return r


A, R = load(TA), load(TR)
la, lr = arm_lines(TA), arm_lines(TR)
ref = json.load(open(REF)) if os.path.exists(REF) else {}
print(f"T2 CHAT WIDTH ({D}): A = {TA} (W1 4), R = {TR} (W1 rand); {len(A)} / {len(R)} prompts")
# 1. tokens
eq_ref = eq_ar = eq_l = 0
for kf in sorted(A):
    key = next((k for k in ref if k.replace("/", "_") == kf), None)
    ta = A[kf]["toks"]
    rr = (ref.get(key) or {}).get("ex", [[]])[0] if key else []
    e1 = rr[:len(ta)] == ta
    e2 = kf in R and R[kf]["toks"] == ta
    ka = next((k for k in la if k.replace("/", "_") == kf), None)
    e3 = ka is not None and ka in lr and la[ka]["lsha"] == lr[ka]["lsha"]
    eq_ref += e1; eq_ar += e2; eq_l += e3
    if not (e1 and e2 and e3):
        print(f"  {kf}: A = chat12 prefix {e1}, R = A tokens {e2}, lsha {e3}")
capn = max((len(v["toks"]) for v in A.values()), default=0)
print(f"TOKENS: A = chat12 (first {capn}) "
      f"{eq_ref} / {len(A)}; R = A tokens {eq_ar} / {len(A)}, lsha {eq_l} / {len(A)}")
# 2. cost per call (R)
rows = [c for v in R.values() for c in v["calls"] if c[0] >= 1]
if rows:
    w = np.array([c[2] for c in rows]); ci = np.array([c[1] for c in rows])
    ms = np.array([(c[8] - c[4]) * 1e3 for c in rows]); gb = np.array([(c[10] - c[6]) / 1e9 for c in rows])
    mi = np.array([c[11] - c[7] for c in rows])
    cols = [((ci == 0) & (w == k)).astype(float) for k in (1, 2, 3, 4)] + [(ci > 0).astype(float), gb]
    X = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(X, ms, rcond=None)
    res = ms - X @ coef
    F = {k: coef[k - 1] for k in (1, 2, 3, 4)}; F4l, b = coef[4], coef[5]
    print(f"COST (R, {len(rows)} calls, checks >= 1): ms = F[w] + {b:.1f} * GB (rms {np.sqrt(np.mean(res ** 2)):.1f} ms; "
          f"{1e3 / b:.1f} GB/s marginal)")
    MB1 = {}
    for k in (1, 2, 3, 4):
        m = (ci == 0) & (w == k)
        MB1[k] = gb[m].mean() * 1e3 if m.any() else float("nan")
        print(f"  first call {k} rows: {m.sum():4d} calls, F {F[k]:5.1f} ms, {MB1[k]:5.0f} MB, {mi[m].mean() if m.any() else 0:4.0f} "
              f"misses, measured {ms[m].mean() if m.any() else 0:5.1f} ms")
    m = ci > 0
    MBl = gb[m].mean() * 1e3 if m.any() else MB1[4]
    F4l = F4l if m.any() else F[4]                  # no later calls in R: price them as a first 4-row call
    print(f"  later calls (4 rows): {m.sum():4d} calls, F {F4l:5.1f} ms, {MBl:5.0f} MB, measured {ms[m].mean() if m.any() else 0:5.1f} ms")
    # 3. pricing on A's checks
    chk = [(r_[2], r_[3] if r_[3] is not None else 1.0) for v in A.values() for kk, r_ in enumerate(v["log"]) if kk >= 1 and r_[1] == 0]
    kept = np.array([c[0] for c in chk]); cf = np.array([c[1] for c in chk])
    assert len(kept), "no exact checks in A"
    oth = []                                          # A's non-exact ms per check (draft + the rest of the check loop)
    for kf, v in A.items():
        ka = next((k for k in la if k.replace("/", "_") == kf), None)
        ex = sum(p[6] - p[2] for p in v["ph"] if p[0] == "exact" and p[1] >= 1) * 1e3
        nc = sum(1 for p in v["ph"] if p[0] == "exact" and p[1] >= 1)
        t0 = min((p[2] for p in v["ph"] if p[1] >= 1), default=None); t1 = max((p[6] for p in v["ph"]), default=None)
        if nc and t0 is not None:
            oth.append(((t1 - t0) * 1e3 - ex) / nc)
    o_ms = float(np.mean(oth)) if oth else 10.0

    def cost(pol):                                   # pol: list of (cf threshold, width), ascending; else 4
        t = 0.0
        for k_, c_ in zip(kept, cf):
            w1 = next((wd for th, wd in pol if c_ < th), 4)
            t += F[w1] + b * MB1[w1] / 1e3
            if k_ > w1:
                t += math.ceil((k_ - w1) / 4) * (F4l + b * MBl / 1e3)
        return t / len(kept)

    base = cost([])
    kp = kept.mean()
    print(f"PRICING on A's {len(kept)} checks (kept {kp:.3f}; other ms per check {o_ms:.1f}): all-4 = {base:.1f} exact ms per check, "
          f"{(base + o_ms) / kp:.1f} ms per kept token")
    ths = (0.3, 0.5, 0.7, 0.9, 0.97, 1.01)
    res_ = []
    for w1 in (1, 2, 3):
        for th in ths:
            res_.append(([(th, w1)], cost([(th, w1)])))
    for (t1, t2) in itertools.combinations(ths, 2):
        res_.append(([(t1, 1), (t2, 2)], cost([(t1, 1), (t2, 2)])))
        res_.append(([(t1, 1), (t2, 3)], cost([(t1, 1), (t2, 3)])))
        res_.append(([(t1, 2), (t2, 3)], cost([(t1, 2), (t2, 3)])))
    res_.sort(key=lambda x: x[1])
    for pol, c_ in res_[:8]:
        g = (base + o_ms) / (c_ + o_ms)
        print(f"  {' '.join(f'cf<{th}:{wd}' for th, wd in pol):24s} exact {c_:6.1f} ms per check, x{g:.3f} decode [P]")
    json.dump(dict(F=F, F4l=F4l, b=b, MB1=MB1, MBl=MBl, other_ms=o_ms, base=base,
                   best=[dict(policy=p_, ms=c_) for p_, c_ in res_[:8]]), open(os.path.join(D, "chw12_report.json"), "w"), indent=1)
else:
    print("COST: no per-call rows in R")
