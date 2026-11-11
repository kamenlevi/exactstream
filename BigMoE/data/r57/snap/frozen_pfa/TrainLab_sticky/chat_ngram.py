"""CHAT150 replays (MTP Head, Sun 11-08 00:xx, CPU): what can raise chat tokens per check, priced with chat's own cycle.
The orchestrator 23:4x (Kamen: chat >= 150 tok/s by Tue / Wed) + NewIdeas PLAN CHAT 150 items CH3 (a faster chat head) / CH4.
Data:
  - D-24's teacher-forced MTP chain, head b5 (the shipped chat head), held-out IFEval (30) + multi (24) answers:
    data/d24/{ifeval,multi}_b5.npz (per anchor and depth 1-15: head top-1 p, true token's rank capped at 3; ids; gen).
  - Chat's cycle: BigMoE N45 (Sat 23:25, 45 GB C 256, exact EB arm, 12 IFEval, both passes), data/r57/nie{1,2}_256_checks.json:
    per check rows, draft ms, and the check-to-check wall (c0 diffs), checks >= 2.
Policies (exact: every proposal is verified, so outputs never change):
  mtp        the shipped chat rule: MTP chain K15, cut at running top-1 p product t 0.3, early exit (d24_read's per_anchor).
  ng(m)      prompt lookup: the longest suffix (m..8 tokens) of the committed text that occurred earlier in the prompt +
             answer; propose up to G tokens after its last occurrence; no match -> no draft (the bonus token only).
  hyb(m)     ng(m) where it matches, else mtp.
  hybg(m)    as hyb, with the other answers of the set as an extra corpus (SuffixDecoding-style, leave-one-out).
  orc        per check the better of mtp and ng(1) (an upper bound for any selector).
  mtp t / K  the cut threshold re-tuned when the head drafts faster (CH3: 2.6 ms per row now; x0.5, x0.25).
Cost per check: wall = F + R x rows + D x drafted steps (fitted below); ng drafts cost ~0.
Usage: [BigMoE venv] python3 -I sticky/chat_ngram.py -> data/chat150_read.txt"""
import json, os, sys
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
BM = os.path.expanduser("~/BigMoE/data/r57")
OUT = os.path.join(TD, "chat150_read.txt")
G = 15
L_ = []


def say(s=""):
    print(s, flush=True); L_.append(s)


# ---------- chat's cycle from N45 (exact EB, 12 IFEval, 2 passes)
rows_, drf_, wall_ = [], [], []
for t in ("nie1", "nie2"):
    for c in json.load(open(os.path.join(BM, f"{t}_256_checks.json"))):
        c0, rw, df = c["c0"], c["rows"], c["drf"]
        for i in range(2, len(c0) - 1):
            rows_.append(rw[i]); drf_.append(df[i]); wall_.append(c0[i + 1] - c0[i])
rows_, drf_, wall_ = map(np.array, (rows_, drf_, wall_))
X = np.stack([np.ones_like(rows_, dtype=float), rows_], 1)
coef, *_ = np.linalg.lstsq(X, wall_ - drf_, rcond=None)   # the draft is serial: wall = F + R x rows + draft ms
F, R = coef; DW = 1.0
dstep = np.polyfit(rows_, drf_, 1)
say(f"CHAT150 replays (CPU; {os.path.basename(OUT)})")
say(f"chat cycle, N45 exact EB 12 IFEval x 2 passes, checks >= 2 (n {len(wall_)}): median wall {np.median(wall_):.1f} ms, "
    f"median rows {np.median(rows_):.0f} (mean {rows_.mean():.2f}), median draft {np.median(drf_):.1f} ms")
say(f"  fit wall = {F:.1f} + {R:.2f} x rows + {DW:.2f} x draft ms; draft = {dstep[1]:.2f} + {dstep[0]:.2f} x rows "
    f"(NewIdeas: 3.0 / 5.6 / 8.2 / 10.8 ms at 1-4 rows)")
by = {r: np.median(wall_[rows_ == r]) for r in range(1, 9) if (rows_ == r).sum() >= 20}
say("  median wall by rows: " + ", ".join(f"{r}: {v:.1f}" for r, v in by.items()))
D_ROW = dstep[0]                                  # ms per drafted MTP step


def cost(rows, steps, dscale=1.0):
    return F + R * rows + DW * (dstep[1] + D_ROW * steps * dscale if steps > 0 else 0.0)


# ---------- the MTP chain (d24_read's rule)
def mtp_anchor(p, r, K=15, t=0.3):
    valid = r >= 0
    L = int(valid.sum())
    if L == 0:
        return 1, 1, 0
    miss = np.where(r[:L] != 0)[0]
    kstar = int(miss[0]) + 1 if len(miss) else L + 1
    cp = np.cumprod(np.nan_to_num(p[:min(K, L)], nan=0.0))
    m = int((cp >= t).sum())
    return min(kstar, m + 1), m + 1, min(m + 1, min(K, L))


# ---------- prompt lookup
class Lookup:
    """last occurrence of every n-gram (n = 1..8) ending at each position; extra corpora searched when local has none"""
    NMAX = 8

    def __init__(self, ids, corpora=()):
        self.ids = ids
        self.last = {}                                  # (n, tuple) -> end index (inclusive) of the latest occurrence
        self.seen = -1
        self.corp = []
        for c in corpora:                               # corpus n-gram -> end index in that corpus (first occurrence kept)
            d = {}
            for e in range(len(c)):
                for n in range(1, self.NMAX + 1):
                    if e - n + 1 < 0:
                        break
                    d.setdefault((n, tuple(c[e - n + 1:e + 1])), e)
            self.corp.append((c, d))

    def advance(self, q):
        """index positions .. q - 1 so that matches end before q (the suffix ending at q looks back)"""
        ids = self.ids
        while self.seen < q - 1:
            self.seen += 1
            e = self.seen
            for n in range(1, self.NMAX + 1):
                if e - n + 1 < 0:
                    break
                self.last[(n, tuple(ids[e - n + 1:e + 1]))] = e

    def propose(self, q, mmin, glob=False):
        """the suffix ending at q (inclusive): the longest earlier match >= mmin -> (match length, proposal tokens)"""
        self.advance(q)
        ids = self.ids
        for n in range(self.NMAX, mmin - 1, -1):
            if q - n + 1 < 0:
                continue
            key = (n, tuple(ids[q - n + 1:q + 1]))
            e = self.last.get(key)
            if e is not None:
                return n, ids[e + 1:min(e + 1 + G, q + 1)]
            if glob:
                for c, d in self.corp:
                    e = d.get(key)
                    if e is not None and e + 1 < len(c):
                        return n, c[e + 1:e + 1 + G]
        return 0, []


def walk(z, i, policy, corpora=(), dscale=1.0, K=15, t=0.3):
    """D-17's walk over one answer: -> tokens, checks, wall ms, rows, ng share"""
    p1, rk, ids, gen = z[f"p1_{i}"], z[f"rk_{i}"], z[f"ids_{i}"].tolist(), int(z["gen"][i])
    a0 = max(0, gen - 2)
    Lw = len(ids) - gen
    lk = Lookup(ids, corpora) if policy[0] != "mtp" else None
    pos, tok, chk, ms, rws, nng = 1, 0, 0, 0.0, 0, 0
    while pos < Lw:
        s = gen + pos - 2
        q = gen + pos - 2                               # the last committed token (the drafts start at ids[q + 1])
        km, rm, sm = (mtp_anchor(p1[s - a0], rk[s - a0], K, t) if 0 <= s - a0 < len(rk) else (1, 1, 0))
        use, k, rows, steps = "mtp", km, rm, sm
        if policy[0] in ("ng", "hyb", "hybg", "orc"):
            n, prop = lk.propose(q, policy[1], glob=policy[0] == "hybg")
            prop = prop[:policy[2]] if len(policy) > 2 else prop       # proposal cap (rows cost on chat)
            if n:
                acc = 0
                for j, x in enumerate(prop):
                    if q + 1 + j < len(ids) and ids[q + 1 + j] == x:
                        acc += 1
                    else:
                        break
                kn, rn = acc + 1, 1 + len(prop)
                if policy[0] == "orc":
                    if kn / cost(rn, 0) > km / cost(rm, sm, dscale):
                        use, k, rows, steps = "ng", kn, rn, 0
                else:
                    use, k, rows, steps = "ng", kn, rn, 0
            elif policy[0] == "ng":
                use, k, rows, steps = "none", 1, 1, 0
        k = max(1, min(k, Lw - pos))
        tok += k; chk += 1; ms += cost(rows, steps, dscale); rws += rows; nng += use == "ng"; pos += k
    return tok, chk, ms, rws, nng


def score(name, sets, policy, **kw):
    out = {}
    for st, z in sets.items():
        T = C = M = RW = NG = 0
        n = len(z["keys"])
        for i in range(n):
            corp = [z[f"ids_{j}"].tolist()[int(z["gen"][j]):] for j in range(n) if j != i] if policy[0] == "hybg" else ()
            t_, c_, m_, r_, g_ = walk(z, i, policy, corp, **kw)
            T += t_; C += c_; M += m_; RW += r_; NG += g_
        out[st] = (T / C, T / M * 1e3, RW / C, NG / C)
    return out


sets = {s: np.load(os.path.join(TD, "d24", f"{s}_b5.npz"), allow_pickle=True) for s in ("ifeval", "multi")}
say(f"answers: ifeval {len(sets['ifeval']['keys'])}, multi {len(sets['multi']['keys'])} (D-24 held-out, head b5)")
base = score("mtp", sets, ("mtp",))
say("")
say("policy                         | IFEval kept/check  tok/s [P]  rows | multi kept/check  tok/s [P]  rows | ng share")
rowfmt = lambda nm, r: (f"{nm:30s} | {r['ifeval'][0]:6.2f} ({r['ifeval'][0] / base['ifeval'][0]:5.3f}x) {r['ifeval'][1]:6.1f} "
                        f"({r['ifeval'][1] / base['ifeval'][1]:5.3f}x) {r['ifeval'][2]:4.1f} | {r['multi'][0]:6.2f} "
                        f"({r['multi'][0] / base['multi'][0]:5.3f}x) {r['multi'][1]:6.1f} ({r['multi'][1] / base['multi'][1]:5.3f}x) "
                        f"{r['multi'][2]:4.1f} | {r['ifeval'][3]:.2f} / {r['multi'][3]:.2f}")
say(rowfmt("mtp K15 t 0.3 (shipped)", base))
for pol in [("ng", 1), ("ng", 2), ("ng", 3), ("hyb", 2), ("hyb", 3), ("hyb", 4), ("hyb", 6), ("hyb", 2, 4), ("hyb", 3, 4),
            ("hyb", 3, 8), ("hyb", 4, 8), ("hybg", 3), ("hybg", 4), ("orc", 1), ("orc", 1, 8)]:
    say(rowfmt(f"{pol[0]}(match >= {pol[1]}{', cap ' + str(pol[2]) if len(pol) > 2 else ''})", score(pol[0], sets, pol)))
say("")
say("CH3, a faster chat head (exact; outputs unchanged): draft ms per step x s, cut t re-tuned on IFEval + multi")
for ds in (1.0, 0.5, 0.25):
    best = None
    for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        r = score("mtp", sets, ("mtp",), dscale=ds, t=t)
        g = np.sqrt(r["ifeval"][1] * r["multi"][1])
        if best is None or g > best[0]:
            best = (g, t, r)
    _, t, r = best
    say(rowfmt(f"mtp draft x{ds} best t {t}", r))
say("")
say("Trees (D-24's leaf rows: the head's rank-1 / rank-2 candidates at kept depths whose top-1 p < tau; a tree-aware GDN")
say("verify is NOT built, priced here at chain-row cost), with today's chat costs; and the head 2x faster:")


def leaf_walk(z, i, K, t, tau, w, dscale=1.0):
    p1, rk, ids, gen = z[f"p1_{i}"], z[f"rk_{i}"], z[f"ids_{i}"].tolist(), int(z["gen"][i])
    a0 = max(0, gen - 2); Lw = len(ids) - gen
    pos, tok, chk, ms = 1, 0, 0, 0.0
    while pos < Lw:
        s = gen + pos - 2 - a0
        if 0 <= s < len(rk):
            r, p = rk[s], p1[s]
            L = int((r >= 0).sum())
            if L == 0:
                k, rows, steps = 1, 1, 0
            else:
                miss = np.where(r[:L] != 0)[0]
                kstar = int(miss[0]) + 1 if len(miss) else L + 1
                cp = np.cumprod(np.nan_to_num(p[:min(K, L)], nan=0.0)); m = int((cp >= t).sum())
                k = min(kstar, m + 1)
                leafd = [j for j in range(1, m + 1) if p[j - 1] < tau]
                if kstar <= m and kstar in leafd and 1 <= r[kstar - 1] <= w:
                    k = kstar + 1
                rows, steps = m + 1 + len(leafd) * w, min(m + 1, min(K, L))
        else:
            k, rows, steps = 1, 1, 0
        k = max(1, min(k, Lw - pos))
        tok += k; chk += 1; ms += cost(rows, steps, dscale); pos += k
    return tok, chk, ms


def leaf_score(K, t, tau, w, dscale=1.0):
    out = {}
    for st, z in sets.items():
        T = C = M = 0
        for i in range(len(z["keys"])):
            a, b, c = leaf_walk(z, i, K, t, tau, w, dscale); T += a; C += b; M += c
        out[st] = (T / C, T / M * 1e3, 0.0, 0.0)
    return out


for ds in (1.0, 0.5):
    for (t, tau, w) in ((0.3, 0.5, 1), (0.3, 0.8, 1), (0.3, 0.8, 2), (0.2, 0.8, 1), (0.15, 0.8, 2)):
        say(rowfmt(f"leaf t {t} tau {tau} w {w} draft x{ds}", leaf_score(15, t, tau, w, ds)))
say("")
say("Ceiling: a PERFECT drafter (every row kept) at r rows, today's chat costs, serial MTP draft vs one parallel pass (5.3 ms):")
for r_ in (4, 8, 16):
    say(f"  {r_:2d} rows: serial {r_ / cost(r_, r_) * 1e3:6.1f} tok/s, parallel {r_ / (F + R * r_ + 5.3) * 1e3:6.1f} tok/s "
        f"(check {F + R * r_ + 5.3:.0f} ms)")
say("")
say("Reading: kept/check from the replay walk ([K] on stored answers); tok/s [P] = tokens / the fitted chat wall.")
open(OUT, "w").write("\n".join(L_) + "\n")
