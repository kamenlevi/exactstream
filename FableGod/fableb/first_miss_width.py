"""first_miss_width.py (Fable B, mission B5): the first-miss predictor as a width policy in MoonShots' M-21 harness
(bwidth45 / wide45 / band45: coupled walks on Breakthrough's capture, the flat cost M0 + 2.0 ms x rows + draft(width),
and the SSD-aware cost with the C 256 + M-17 byte replay, EXACT gate). Reads MoonShots' modules and data in place,
writes only here. Policies: fixed16 (today), fixed8, oracle (hindsight, the ceiling), cfpred (a per-row logistic on
the drafter's cf + depth + history: the information M-21b's stop head had), fmpred (+ token class, copy state, n-gram
novelty, line position of the drafted / real tokens). Both predictors are trained on the real checks' tested rows of
the other fold's windows (2 folds by window) and pick the width whose expected tok/s under the flat cost is highest,
with the t 0.3 cut inside. Code and chat apart. CPU only.
Run: nice -n 15 ~/BigMoE/.venv/bin/python first_miss_width.py > first_miss_width.out  (a few minutes)"""
import os, sys, time
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import bwidth45 as BW
from tokenizers import Tokenizer

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("FM_SEEDS", "2"))
DRAFT, CYC16, ROW = BW.DRAFT, BW.CYC16, BW.ROW
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
CLASSES = ["newline", "indent", "ident_start", "word_cont", "punct", "number", "string", "comment", "other"]
T0 = time.time(); lines = []
_dec = {}


def say(s):
    print(s, flush=True); lines.append(s)


def dec(i):
    i = int(i)
    if i not in _dec:
        _dec[i] = TOK.decode([i])
    return _dec[i]


def classify(s, in_str, in_com, prev_s):
    if in_com:
        return "comment"
    if in_str:
        return "string"
    if "\n" in s:
        return "newline"
    st = s.strip()
    if st == "":
        return "indent"
    c = st[0]
    if c.isdigit():
        return "number"
    if c.isalpha() or c == "_":
        if s[0] == " " or prev_s is None or not (prev_s[-1:].isalnum() or prev_s[-1:] == "_"):
            return "ident_start"
        return "word_cont"
    if c in "\"'`":
        return "string"
    if c == "#" or st.startswith("//"):
        return "comment"
    return "punct"


def window_tables(z):
    """per position of the window: the token seen at that position by a drafter (the real draft where a real check
    covered it, else the true token), its class, line position, copy state (8-gram seen earlier), 3-gram and unigram
    novelty. The state machine runs on the true text (identical to the draft up to each real check's miss)."""
    ids = z["tokens"].astype(int).tolist(); n = len(ids)
    seen = np.array(ids)                         # the token a drafter proposed at each position
    for a, r, k in z["checks_walk"].tolist():
        pass                                     # the capture keeps dcf per check but not the drafted ids: the true token stands in
    cls = [""] * n; pil = np.zeros(n, int)
    in_str = None; in_com = False; p = 0; prev = None
    for i, t in enumerate(ids):
        s = dec(t)
        cls[i] = classify(s, in_str is not None, in_com, prev); pil[i] = p
        for ch in s:
            if ch == "\n":
                in_str = None; in_com = False; p = -1
            elif in_com:
                pass
            elif in_str is not None:
                if ch == in_str:
                    in_str = None
            elif ch == "#":
                in_com = True
            elif ch in "\"'`":
                in_str = ch
        p += 1; prev = s
    g8, g3, g1 = {}, {}, {}
    copy = np.zeros(n, int); nov3 = np.ones(n, int); nov1 = np.ones(n, int)
    for q in range(n):
        if q >= 8:
            key = tuple(ids[q - 8:q])
            copy[q] = 1 if key in g8 else 0
        if q >= 2:
            k3 = tuple(ids[q - 2:q + 1]); nov3[q] = 0 if k3 in g3 else 1
        nov1[q] = 0 if ids[q] in g1 else 1
        # register n-grams ending at q (so later positions can see them, never the current one)
        if q >= 7:
            g8.setdefault(tuple(ids[q - 7:q + 1]), q)
        if q >= 2:
            g3.setdefault(tuple(ids[q - 2:q + 1]), q)
        g1.setdefault(ids[q], q)
    return dict(ids=ids, cls=cls, pil=pil, copy=copy, nov3=nov3, nov1=nov1, n=n)


def feat_rows(tab, conf, a, R, prev_kept, use_content):
    """features of rows a+1 .. a+R-1 (the drafted rows after the anchor's bonus token), the cf from the capture's conf"""
    X = []
    for j in range(1, R):
        pos = a + j
        if pos >= tab["n"] or pos >= len(conf):
            break
        cf = conf[pos] if not np.isnan(conf[pos]) else 0.9
        cf = min(max(cf, 1e-4), 1 - 1e-4)
        f = [np.log(cf / (1 - cf)), j / 15.0, prev_kept / 16.0, 1.0 if prev_kept < 8 else 0.0]
        if use_content:
            c = tab["cls"][pos]
            f += [1.0 if c == k else 0.0 for k in CLASSES[:-1]] + [tab["copy"][pos], tab["nov3"][pos], tab["nov1"][pos], np.log1p(tab["pil"][pos])]
        X.append(f)
    return np.array(X, float)


def fit_logit(X, y, l2=1.0, it=25):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1)
    reg = np.full(d + 1, l2); reg[-1] = 0.0
    for _ in range(it):
        z = Xb @ w; p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + reg * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg)
        w -= np.linalg.solve(H, g)
    return w


def predict(X, w):
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((X.shape[0], 1))]) @ w)))


def train(Zs, tabs, use_content):
    """per-row miss model on the real checks' tested rows"""
    X, y = [], []
    for z, tab in zip(Zs, tabs):
        conf = z["conf"].astype(float); prev = 16
        for a, r, k in z["checks_walk"].tolist():
            fm = k if k < r else 99                           # the miss row index (1-based row after the anchor), 99 = none
            F = feat_rows(tab, conf, a, r, prev, use_content)
            for j in range(F.shape[0]):
                row = j + 1
                if row <= min(fm, r - 1):
                    X.append(F[j]); y.append(1.0 if row == fm else 0.0)
            prev = k
    X = np.array(X); y = np.array(y)
    return fit_logit(X, y, l2=1.0)


def choose_width(tab, conf, a, prev_kept, w, use_content, rates, m0, n):
    """expected tok/s per width under the flat cost from the predicted per-row miss probabilities (rows past the
    capture's 16 use the harness's per-depth rates), with the t 0.3 cut as the harness applies it"""
    best, bestv = 16, -1.0
    for R in (8, 16, 32):
        R_ = min(R, n - a)
        if R_ <= 1:
            continue
        F = feat_rows(tab, conf, a, R_, prev_kept, use_content)
        p = predict(F, w) if F.shape[0] else np.zeros(0)
        # rows beyond the features (none: feat_rows covers a+1..a+R_-1)
        # the cut: rows = 1 + the first index where the running cf product < 0.3 (as check())
        rate = rates[32 if R == 32 else 16]
        c = np.where(np.isnan(conf[a:a + R_ - 1]), rate[:R_ - 1], conf[a:a + R_ - 1])
        below = np.nonzero(np.cumprod(c) < 0.3)[0]
        rows = 1 + int(below[0]) if len(below) else R_
        pr = p[:rows - 1]
        surv = np.cumprod(1 - pr)
        ek = 1.0 + surv.sum()                                  # expected kept: the bonus + each row kept while surviving
        v = ek / (m0 + ROW * rows + DRAFT[R])
        if v > bestv:
            best, bestv = R, v
    return best


def walk(z, tab, pol, rates, u, m0, w=None, use_content=False):
    conf, right, a, end, n = BW.prep(z)
    out, hist = [], []; prev_kept = 16
    while a < end:
        if pol == "oracle":
            R = max((8, 16, 32), key=lambda R: (lambda rk: rk[1] / (m0 + ROW * rk[0] + DRAFT[R]))(BW.check(a, R, conf, right, u, n, rates)))
        elif pol.startswith("fixed"):
            R = int(pol[5:])
        else:
            R = choose_width(tab, conf, a, prev_kept, w, use_content, rates, m0, n)
        rows, kept = BW.check(a, R, conf, right, u, n, rates)
        out.append((a, rows, kept, rows == min(R, n - a), R))
        prev_kept = kept
        a += kept
    return out


def main():
    pool = mp.get_context("fork").Pool(4)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {wn: W4.counts_of(np.load(os.path.join(B.CAPDIR, wn + ".npz"))) for wn in names}
    tm, tl, tn = (sum(C[wn][j] for wn in names) for j in range(3))
    say("B5 FIRST-MISS width policy in M-21's harness (Breakthrough's capture, %d seeds; flat cost M0 + %.1f x rows + draft %s; fixed 16 = %.1f ms; SSD-aware: C 256 + M-17 byte replay, EXACT gate)"
        % (SEEDS, ROW, DRAFT, CYC16))
    for dom, pref, rate_arm in (("code", "tf_code", "base"), ("chat", ("tf_ifeval", "tf_multi"), "base")):
        wins = [wn for wn in names if wn.startswith(pref if isinstance(pref, str) else pref[0]) or (not isinstance(pref, str) and wn.startswith(pref[1]))]
        POP = {"m": [[(tm - C[wn][0])[l] / (tn - C[wn][2]) for l in range(NL)] for wn in wins],
               "l": [[(tl - C[wn][1])[l] / (tn - C[wn][2]) for l in range(NL)] for wn in wins]}
        Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, wn + ".npz")).items()} for wn in wins]
        TAB = [window_tables(z) for z in Z]
        PREW = [W4.prefill_of(z) for z in Z]
        if dom == "code":
            rates = {16: W4.rates("base16"), 32: np.nan_to_num(W4.rates("base32"), nan=0.0)}
        else:
            # chat: per-depth rates from the capture's own uncut walks (M-21b's rates_of), the same for 16 and 32
            cnt = np.zeros(32); hit = np.zeros(32)
            for z in Z:
                for a, r, k in z["checks_walk"].tolist():
                    for j in range(1, r):
                        if j <= k:
                            cnt[j - 1] += 1; hit[j - 1] += 1.0 if j < k or k == r else 0.0
            rr = np.where(cnt > 0, hit / np.maximum(cnt, 1), 0.5)
            rates = {16: rr, 32: rr}
        fold = {i: i % 2 for i in range(len(wins))}
        models = {}
        for k in (0, 1):
            tr = [i for i in range(len(wins)) if fold[i] != k]
            models[("cfpred", k)] = train([Z[i] for i in tr], [TAB[i] for i in tr], False)
            models[("fmpred", k)] = train([Z[i] for i in tr], [TAB[i] for i in tr], True)
        POLICIES = ("fixed16", "fixed8", "fixed32", "cfpred", "fmpred", "oracle")
        T = {p: np.zeros(6) for p in POLICIES}; wid = {p: {8: 0, 16: 0, 32: 0} for p in POLICIES}
        PW = {p: np.zeros((len(wins), 2)) for p in POLICIES}
        for seed in range(SEEDS):
            U = [np.random.default_rng(10_000 * seed + i).random(z["conf"].shape[0] + 1) for i, z in enumerate(Z)]
            base = [walk(z, TAB[i], "fixed16", rates, U[i], 0.0) for i, z in enumerate(Z)]
            r16 = sum(x[1] for wk in base for x in wk) / sum(len(wk) for wk in base)
            m0 = CYC16 - ROW * r16 - DRAFT[16]
            for p in POLICIES:
                if p == "fixed16":
                    wks = base
                elif p in ("cfpred", "fmpred"):
                    wks = [walk(z, TAB[i], p, rates, U[i], m0, models[(p, fold[i])], p == "fmpred") for i, z in enumerate(Z)]
                else:
                    wks = [walk(z, TAB[i], p, rates, U[i], m0) for i, z in enumerate(Z)]
                Qw = [W4.needs_of(z, [x[:4] for x in wks[i]], W4.GATES["EXACT"]) for i, z in enumerate(Z)]
                jobs = [(key, B.SIZES["C_m" if key == "m" else "C_l"][l],
                         [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, c[key][l]) for t, c in enumerate(Qw[i])]
                          for i in range(len(wins))], [POP[key][i][l] for i in range(len(wins))]) for key in ("m", "l") for l in range(NL)]
                mb = 0.0
                for (key, *_), miss in zip(jobs, pool.map(W4.replay_arr, jobs, chunksize=4)):
                    mb += miss.sum() * (MR if key == "m" else LR) / 1e6
                U_gb = sum(sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for q in Qw for c in q) / 1e9
                ck = sum(len(wk) for wk in wks); kept = sum(x[2] for wk in wks for x in wk); rows = sum(x[1] for wk in wks for x in wk)
                flat = sum(m0 + ROW * x[1] + DRAFT[x[4]] for wk in wks for x in wk)
                fixed = sum(W4.dense(x[1]) + 0.35 * x[1] + DRAFT[x[4]] for wk in wks for x in wk) + mb / 12.0
                T[p] += (ck, kept, rows, flat, fixed + U_gb * 1e3 / W4.GPU_GBS, fixed + U_gb * 1e3 / (W4.GPU_GBS * W4.NAX6))
                for i, wk in enumerate(wks):
                    PW[p][i] += (sum(x[2] for x in wk), sum(m0 + ROW * x[1] + DRAFT[x[4]] for x in wk))
                for x in (y for wk in wks for y in wk):
                    wid[p][x[4]] += 1
                del Qw, jobs
            say("  [%s] seed %d done (%.0f s)" % (dom, seed, time.time() - T0))
        say("")
        say("  %s (%d windows): policy | kept / check | rows / check | widths 8 / 16 / 32 (%% of checks) | tok/s flat | SSD-aware today / NAX6 | vs fixed 16 (flat, SSD today, SSD NAX6)" % (dom, len(wins)))
        b = T["fixed16"]
        for p in POLICIES:
            t = T[p]; n = sum(wid[p].values())
            tf, ts, tn6 = (1e3 * t[1] / t[i] for i in (3, 4, 5)); bf, bs, bn6 = (1e3 * b[1] / b[i] for i in (3, 4, 5))
            say("  %-8s | %6.2f | %6.2f | %4.0f / %4.0f / %4.0f | %6.1f | %6.1f / %6.1f | %+5.1f %%, %+5.1f %%, %+5.1f %%"
                % (p, t[1] / t[0], t[2] / t[0], 100 * wid[p][8] / n, 100 * wid[p][16] / n, 100 * wid[p][32] / n, tf, ts, tn6, 100 * (tf / bf - 1), 100 * (ts / bs - 1), 100 * (tn6 / bn6 - 1)))
        for p in ("cfpred", "fmpred"):
            say("  per window, flat tok/s %s vs fixed 16: " % p + ", ".join("%s %+.1f %%" % (wn[3:], 100 * ((PW[p][i][0] / PW[p][i][1]) / (PW["fixed16"][i][0] / PW["fixed16"][i][1]) - 1)) for i, wn in enumerate(wins)))
        say("")
    pool.close(); pool.join()
    say("  total %.0f s" % (time.time() - T0))
    open(os.path.expanduser("~/FableGod/fableb/first_miss_width.out"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
