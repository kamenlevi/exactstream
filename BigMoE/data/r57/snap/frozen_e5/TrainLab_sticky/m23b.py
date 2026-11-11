"""M-23b FIRSTMISS-MTP (row M-23b in ~/NewIdeas/QUEUE.md, pre-registered by MoonShots Tue 11-03 18:5x before any run; this
replay by MTP Head on the orchestrator's routing, Tue 11-03 evening, CPU). Fable B's FIRST_MISS screen (section 5) with the
MTP head as the second opinion. Harness: MoonShots' stop45.py coupled walks on Breakthrough's 36 capture windows (read-only
imports of stop45 / bwidth45 / wide45 / band45 / laysz45), SSD-aware tok/s at C 256 + M-17 (byte replay on the union per
block, EXACT gate), 2 folds by window, logistic acceptance heads, the rule verifies while the predicted survival >= mu (mu
tuned on the train fold; SEEDS seeds on the test fold, 1 for tuning, as stop45).
Heads: A = stop45's features (cf, cumprod, depth, depth^2, the last two kept, cf x depth: M-21b's control); B = A + Fable B's
content signals for the DRAFTED token (class one-hot, 8-gram copy state, 3-gram and unigram novelty in the context, log line
position; first_miss.py's definitions, with the context taken through the anchor token itself, ids[:a + 1]: first_miss.py
used ids[:a]); C_d = B + the MTP head at depths 0..d-1 (the flag DFlash2's draft != the MTP top-1, the MTP top-1 p, the
MTP's p of the drafted token; /Users/kamen/BigMoE/data/r57/snap/frozen_e5/TrainLab_sticky/tree3_feats.py on data/tree3_map) + an availability indicator, d = 1 / 2 / 3
chosen on the train fold. Drafted tokens and MTP features come from tree3_map (the every-position walk of DFlash2 L1-20,
teacher-forced: exact up to and including the first miss; its picks differ from the capture's on ~1-2.5 % of near-ties).
Cost: d x 2.2 ms per check on the critical path for C_d (the MTP chain's steps); a reading with that cost hidden is reported
apart, not judged. LINE (M-23's): code SSD-aware tok/s, C_d vs t 0.3 on the test folds: >= +3 % PASS, < +1 % KILL, else
BETWEEN; chat (IFEval + multi) the same lines, reported apart; C_d vs B reported. Diagnostic: on cf >= 0.99 rows the MTP
flag at depth 0 marks >= 30 % of the first misses while flagging <= 5 % of those rows. Output: data/m23b.txt (appended)."""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import bwidth45 as BW
import stop45 as S
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/TrainLab_sticky"))
import tree3_feats as TF
from tokenizers import Tokenizer

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("M23B_SEEDS", "2"))
MUS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
D16, MTP_MS = 6.3, 2.2
DOMS = ("code", "ifeval", "multi")
TOK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
CLASSES = ["newline", "indent", "ident_start", "word_cont", "punct", "number", "string", "comment", "other"]
_dec = {}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def dec(i):                                                 # Fable B's first_miss.py (copied: that file has no main guard)
    i = int(i)
    if i not in _dec:
        _dec[i] = TOK.decode([i])
    return _dec[i]


def classify(s, in_str, in_com, prev_s):                    # Fable B's first_miss.py, verbatim
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


def advance(state, s):                                      # first_miss.py's line state machine, one token
    in_str, in_com, p = state
    for ch in s:
        if ch == "\n":
            in_str, in_com, p = None, False, -1
        elif in_com:
            pass
        elif in_str is not None:
            if ch == in_str:
                in_str = None
        elif ch == "#":
            in_com = True
        elif ch in "\"'`":
            in_str = ch
    return (in_str, in_com, p + 1)


class Win:
    """one capture window: content + MTP features per anchor for depths 0..14, cached"""
    def __init__(self, name, z, T):
        self.name, self.T = name, T
        self.tok = [int(x) for x in z["tokens"]]
        st, self.state = (None, False, 0), []
        for t in self.tok:
            st = advance(st, dec(t)); self.state.append(st)
        self.g = {}
        for n in (1, 3, 8):
            d = {}
            for q in range(len(self.tok) - n + 1):
                d.setdefault(tuple(self.tok[q:q + n]), q)
            self.g[n] = d
        self.cache = {}

    def feats(self, a):
        if a in self.cache:
            return self.cache[a]
        w = self.T.get(self.name)
        c = None if w is None else w["at"].get(int(a))
        con = np.zeros((15, 12)); mtp = np.zeros((15, 3)); ok = np.zeros(15)
        if c is not None:
            drafts = [int(x) for x in w["own"][c]]
            st, prev = self.state[a], dec(self.tok[a])
            prefix = self.tok[:a + 1]
            for j, dt in enumerate(drafts):
                s = dec(dt)
                cls = classify(s, st[0] is not None, st[1], prev)
                L = len(prefix)
                cp = 1.0 if L >= 8 and self.g[8].get(tuple(prefix[-8:]), 10 ** 9) < L - 8 else 0.0
                nov3 = 0.0 if self.g[3].get(tuple(prefix[-2:] + [dt]), 10 ** 9) <= a - 2 else 1.0
                nov1 = 0.0 if self.g[1].get((dt,), 10 ** 9) <= a else 1.0
                con[j] = [1.0 if cls == k else 0.0 for k in CLASSES[:-1]] + [cp, nov3, nov1, np.log1p(max(st[2], 0))]
                st = advance(st, s); prev = s; prefix = prefix + [dt]
            m = TF.feats(self.T, self.name, a, 15)
            good = ~np.isnan(m[:, 0])
            mtp[good] = m[good]; ok[good] = 1.0
        self.cache[a] = (con, mtp, ok)
        return self.cache[a]


def feat_fn(head, d):
    """-> f(win, conf, a, J, rates, k1, k2) -> [J, n]"""
    def f(win, conf, a, J, rates, k1, k2):
        FA = S.feats(conf, a, J, rates, k1, k2)
        if head == "A":
            return FA
        con, mtp, ok = win.feats(a)
        if head == "B":
            return np.hstack([FA, con[:J]])
        avail = (np.arange(J) < d) * ok[:J]
        return np.hstack([FA, con[:J], mtp[:J] * avail[:, None], avail[:, None]])
    return f


def train_set(Zs, Ws, rates, ff):
    X, y = [], []
    for z, win in zip(Zs, Ws):
        conf, ks = z["conf"].astype(float), [8, 8]
        for a, r, k in z["checks_walk"].tolist():
            J = min(r - 1, 15)
            if J > 0:
                F = ff(win, conf, a, J, rates, ks[-1], ks[-2])
                for dd in range(1, J + 1):
                    if k >= dd:
                        X.append(F[dd - 1]); y.append(float(k > dd))
            ks.append(k)
    return np.array(X), np.array(y)


def walk(z, win, rule, rates, u, w=None, mu=0.3, ff=None):
    """stop45.walk with a pluggable feature function (rule 'learned') """
    conf, right, a, end, n = BW.prep(z)
    out, ks = [], [8, 8]
    while a < end:
        R_ = min(16, n - a)
        J = R_ - 1
        acc = lambda j: (right[a + j] == 1.0) if not np.isnan(right[a + j]) else u[a + j] < rates[j - 1]
        if J <= 0:
            rows = R_
        elif rule == "oracle":
            L = 0
            while L < J and acc(L + 1):
                L += 1
            rows = min(L + 1, R_)
        else:
            if rule == "t":
                c = conf[a:a + J]
                Sv = np.cumprod(np.where(np.isnan(c), rates[:J], c))
            else:
                Sv = np.cumprod(1 / (1 + np.exp(-np.clip(ff(win, conf, a, J, rates, ks[-1], ks[-2]) @ w, -30, 30))))
            below = np.nonzero(Sv < mu)[0]
            rows = 1 + int(below[0]) if len(below) else R_
        kept = rows
        for j in range(1, rows):
            if not acc(j):
                kept = j
                break
        out.append((a, rows, kept, rows == R_))
        ks.append(kept)
        a += kept
    return out


def evaluate(pool, ix, Z, Wn, PREW, POP, policy, rates, seeds):
    """stop45.evaluate with the pluggable walk; policy[i] = (rule, w, mu, ff, extra ms per check); returns (checks, kept,
    rows, ms today incl. the extra, ms today with the extra hidden)"""
    tot = np.zeros(5)
    for seed in range(seeds):
        wks = {i: walk(Z[i], Wn[i], policy[i][0], rates, np.random.default_rng(10_000 * seed + i).random(Z[i]["conf"].shape[0] + 1),
                       policy[i][1], policy[i][2], policy[i][3]) for i in ix}
        Qw = {i: W4.needs_of(Z[i], wks[i], W4.GATES["EXACT"]) for i in ix}
        jobs = [(key, B.SIZES["C_m" if key == "m" else "C_l"][l],
                 [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, c[key][l]) for t, c in enumerate(Qw[i])] for i in ix],
                 [POP[key][i][l] for i in ix]) for key in ("m", "l") for l in range(NL)]
        mbw = {i: 0.0 for i in ix}
        for (key, *_), miss in zip(jobs, pool.map(W4.replay_arr, jobs, chunksize=4)):
            for k, i in enumerate(ix):
                mbw[i] += miss[k] * (MR if key == "m" else LR) / 1e6
        for i in ix:
            ug = sum(sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for c in Qw[i]) / 1e9
            fx = sum(W4.dense(x[1]) + 0.35 * x[1] + D16 for x in wks[i]) + mbw[i] / 12.0
            ms = fx + ug * 1e3 / W4.GPU_GBS
            kept = sum(x[2] for x in wks[i])
            tot += (len(wks[i]), kept, sum(x[1] for x in wks[i]), ms + policy[i][4] * len(wks[i]), ms)
        del Qw, jobs
    return tot


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    TT = {dom: TF.load(dom) for dom in DOMS}
    Wn = [Win(w, Z[i], TT[w.split("_")[1]]) for i, w in enumerate(names)]
    say(f"M-23b FIRSTMISS-MTP on Breakthrough's capture: {len(names)} windows, {SEEDS} seeds, mu grid {MUS}; SSD-aware tok/s today "
        f"(C 256 + M-17, EXACT gate); C_d charged d x {MTP_MS} ms per check")
    # diagnostic: cf >= 0.99 rows, the MTP flag at depth 0 (and at any depth, information)
    for dom in DOMS:
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        dg = np.zeros((2, 4))                                # [depth 0 / any] x [rows, flagged, first misses, marked]
        for i in ix:
            z, win = Z[i], Wn[i]
            conf = z["conf"].astype(float)
            for a, r, k in z["checks_walk"].tolist():
                _, mtp, ok = win.feats(a)
                for dd in range(1, min(r - 1, 15) + 1):
                    if k < dd:
                        break
                    c = conf[a + dd - 1]
                    if np.isnan(c) or c < 0.99 or not ok[dd - 1]:
                        continue
                    fm = k == dd                             # draft dd is the check's first miss
                    for g in ((0, 1) if dd == 1 else (1,)):
                        dg[g] += (1, mtp[dd - 1, 0], fm, fm * mtp[dd - 1, 0])
        for g, lab in ((0, "depth 0 (registered)"), (1, "any depth (information)")):
            rws, fl, fmi, mk = dg[g]
            say(f"  diagnostic {dom} {lab}: cf >= 0.99 rows {int(rws)}, flagged {fl / max(rws, 1):.3f} (<= 0.05); first misses {int(fmi)}, "
                f"marked {mk / max(fmi, 1):.3f} (>= 0.30)")
    res = {}
    for dom in DOMS:
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        rates = S.rates_of([Z[i] for i in ix])
        folds = (ix[0::2], ix[1::2])
        heads = [("A", 0), ("B", 0), ("C1", 1), ("C2", 2), ("C3", 3)]
        pol = {k: {} for k in ["t 0.3", "A", "B", "C1", "C2", "C3", "C_d", "oracle"]}
        for i in ix:
            pol["t 0.3"][i] = ("t", None, 0.3, None, 0.0); pol["oracle"][i] = ("oracle", None, 0.0, None, 0.0)
        notes = []
        for f in (0, 1):
            tr, te = folds[f], folds[1 - f]
            best_c = None
            for hn, d in heads:
                ff = feat_fn(hn[0] if hn in ("A", "B") else "C", d)
                w = S.fit(*train_set([Z[i] for i in tr], [Wn[i] for i in tr], rates, ff))
                ex = d * MTP_MS
                sc = {mu: (lambda t: t[1] / t[3])(evaluate(pool, tr, Z, Wn, PREW, POP, {i: ("learned", w, mu, ff, ex) for i in tr}, rates, 1))
                      for mu in MUS}
                mu = max(sc, key=sc.get)
                pol[hn].update({i: ("learned", w, mu, ff, ex) for i in te})
                notes.append(f"{hn} f{f} mu {mu}")
                if d and (best_c is None or sc[mu] > best_c[0]):
                    best_c = (sc[mu], hn, w, mu, ff, ex)
            pol["C_d"].update({i: ("learned", best_c[2], best_c[3], best_c[4], best_c[5]) for i in te})
            notes.append(f"C_d f{f} = {best_c[1]}")
            say(f"    [{dom} fold {f} tuned, {time.time() - t0:.0f} s]")
        R = {k: evaluate(pool, ix, Z, Wn, PREW, POP, p, rates, SEEDS) for k, p in pol.items()}
        b = R["t 0.3"]
        say(f"  {dom} ({len(ix)} windows; {', '.join(notes)}):")
        for k, t in R.items():
            g, gh = (t[1] / t[3]) / (b[1] / b[3]) - 1, (t[1] / t[4]) / (b[1] / b[4]) - 1
            say(f"    {k:7s} kept / check {t[1] / t[0]:5.2f}, rows / check {t[2] / t[0]:5.2f}, tok/s {1e3 * t[1] / t[3]:6.1f}: {100 * g:+5.1f} % vs t 0.3"
                + (f" (MTP cost hidden: {100 * gh:+5.1f} %)" if k.startswith("C") else ""))
        res[dom] = R
        say(f"    ({time.time() - t0:.0f} s)")
    def gain(R, k, base="t 0.3", col=3):
        return (R[k][1] / R[k][col]) / (R[base][1] / R[base][col]) - 1
    gc = gain(res["code"], "C_d")
    say(f"  LINE code (C_d vs t 0.3, SSD-aware today, MTP cost charged): {100 * gc:+.2f} % -> " + ("PASS" if gc >= 0.03 else "KILL" if gc < 0.01 else "BETWEEN")
        + f"; C_d vs B {100 * gain(res['code'], 'C_d', 'B'):+.2f} %; B vs t 0.3 {100 * gain(res['code'], 'B'):+.2f} %; A vs t 0.3 {100 * gain(res['code'], 'A'):+.2f} %")
    Rc = {k: res["ifeval"][k] + res["multi"][k] for k in res["ifeval"]}
    gch = gain(Rc, "C_d")
    say(f"  chat (IFEval + multi pooled, reported apart): C_d vs t 0.3 {100 * gch:+.2f} % -> " + ("PASS" if gch >= 0.03 else "KILL" if gch < 0.01 else "BETWEEN")
        + f"; C_d vs B {100 * gain(Rc, 'C_d', 'B'):+.2f} %")
    pool.close(); pool.join()
    open(os.path.expanduser("~/TrainLab/data/m23b.txt"), "a").write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
