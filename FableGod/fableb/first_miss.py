"""first_miss.py (Fable B, mission B5): where the drafter's FIRST MISS happens and which cheap pre-check signals see it.
Data [K]: TrainLab's HPD-0 walks (DFlash2 L1-20, block 16, the official loop: per check the drafted tokens dtok, the
drafter's cf per row dcf, the true tokens ttok, the anchor position; the window's ids), Breakthrough's lattices (the
selector's top-16 per depth with probabilities: the top-2 margin and the rank structure; aligned to the walk's checks
where the chain's picks equal the walk's drafts), the model's tokenizer (token classes, line state).
CPU only, numpy. Run: nice -n 15 ~/BigMoE/.venv/bin/python first_miss.py > first_miss.out"""
import os, sys, time
import numpy as np
from tokenizers import Tokenizer

T0 = time.time()
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
WALK = "/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz"
LAT = {"code": "/Users/kamen/Breakthrough/data/trees/lattice_code.npz", "chat": "/Users/kamen/Breakthrough/data/trees/lattice_chat.npz"}
CLASSES = ["newline", "indent", "ident_start", "word_cont", "punct", "number", "string", "comment", "other"]
_dec = {}


def dec(i):
    i = int(i)
    if i not in _dec:
        _dec[i] = TOK.decode([i])
    return _dec[i]


def classify(s, in_str, in_com, prev_s):
    """class of token text s given the line state before it (in_str / in_com) and the previous token's text"""
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
        # a word start if the token has a leading space or the previous token ended a word boundary
        if s[0] == " " or prev_s is None or not (prev_s[-1:].isalnum() or prev_s[-1:] == "_"):
            return "ident_start"
        return "word_cont"
    if c in "\"'`":
        return "string"
    if c == "#" or st.startswith("//"):
        return "comment"
    return "punct"


def line_states(ids):
    """per position: (class, pos_in_line) of the token at that position, with a simple line state machine
    (comments from '#' or '//' to newline; strings between matching quotes, reset at newline)"""
    cls = []; pil = []
    in_str = None; in_com = False; p = 0; prev = None
    for i in ids:
        s = dec(i)
        cls.append(classify(s, in_str is not None, in_com, prev)); pil.append(p)
        # update the state by scanning the token's characters
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
        p += 1
        prev = s
    return cls, pil


def ngram_index(ids, n):
    """first occurrence index of every n-gram"""
    d = {}
    for q in range(len(ids) - n + 1):
        key = tuple(ids[q:q + n])
        if key not in d:
            d[key] = q
    return d


def auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort")) + 1.0
    # ties: average ranks
    order = np.argsort(s); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logit(X, y, l2=1.0, it=25):
    """logistic regression by Newton steps with L2 (not on the bias)"""
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


def load_rows(dom):
    """per tested row of every check: features and the miss label; per check: summary"""
    sets = {"code": ["code"], "chat": ["ifeval", "multi"], "math": ["math"]}[dom]
    rows, checks = [], []
    lat = np.load(LAT[dom], allow_pickle=True) if dom in LAT else None
    lat_by_sample = {}
    if lat is not None:
        for b in range(lat["pick"].shape[0]):
            lat_by_sample.setdefault((str(lat["dom"][b]), int(lat["sample"][b])), []).append(b)
    aligned = 0; total = 0
    for sub in sets:
        W = np.load(WALK % sub, allow_pickle=True)
        names = list(W["names"])
        for wi, name in enumerate(names):
            ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dtok = W[f"{wi}_dtok"]; dcf = W[f"{wi}_dcf"]; ttok = W[f"{wi}_ttok"]
            cls, pil = line_states(ids)
            g8 = ngram_index(ids, 8); g3 = ngram_index(ids, 3); g1 = ngram_index(ids, 1)
            # lattice blocks of this window (same order as the checks when the picks agree)
            key = ("tf_" + sub if dom != "chat" else "tf_" + sub, wi) if dom != "code" else ("tf_code_HumanEval", wi)
            blocks = lat_by_sample.get(key, []) if lat is not None else []
            prev_kept = [16, 16]
            for ci in range(ck.shape[0]):
                kept, nrows, a = int(ck[ci, 0]), int(ck[ci, 1]), int(ck[ci, 2])
                d = dtok[ci]; t = ttok[ci]; c = dcf[ci]
                miss = [j for j in range(nrows) if d[j] != t[j]]
                fm = miss[0] if miss else nrows           # 0-based first miss depth; nrows = none
                lb = blocks[ci] if ci < len(blocks) and np.array_equal(lat["pick"][blocks[ci]], d) else None
                total += 1; aligned += lb is not None
                if lb is not None:
                    lfm = [j for j in range(15) if lat["pick"][lb][j] != lat["true"][lb][j]]
                    lfm = lfm[0] if lfm else 15
                else:
                    lfm = -1
                # draft-prefix text for class / line state: the context up to a, then the drafted tokens
                pre = ids[:a]
                # line state at the anchor from the true context; then walk the drafted tokens
                cls_ctx, pil_ctx = line_states(pre[-64:]) if len(pre) else ([], [])
                in_str = None; in_com = False; p = pil_ctx[-1] + 1 if pil_ctx else 0; prev = dec(pre[-1]) if pre else None
                # recompute the state at the end of the context properly (scan last 64 tokens)
                for i in pre[-64:]:
                    s = dec(i)
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
                    p += 1
                draft_prefix = list(pre)
                for j in range(nrows):                         # every drafted row; tested = j <= fm (fm itself is the miss, if any)
                    s = dec(d[j]); ts = dec(t[j])
                    dcls = classify(s, in_str is not None, in_com, prev)
                    tcls = classify(ts, in_str is not None, in_com, prev)
                    pos = a + j
                    cp = 1 if (len(draft_prefix) >= 8 and g8.get(tuple(draft_prefix[-8:]), 10 ** 9) < len(draft_prefix) - 8) else 0
                    tri = tuple(draft_prefix[-2:] + [int(d[j])]) if len(draft_prefix) >= 2 else None
                    nov3 = 0 if (tri is not None and g3.get(tri, 10 ** 9) < a - 2) else 1
                    nov1 = 0 if g1.get((int(d[j]),), 10 ** 9) < a else 1
                    f = dict(dom=dom, sub=sub, win=wi, check=ci, depth=j, cf=float(c[j]), dcls=dcls, tcls=tcls, pil=p, copy=cp, nov3=nov3, nov1=nov1,
                             prev_kept=prev_kept[-1], prev2_kept=prev_kept[-2], miss=int(j == fm), fm=fm, kept=kept, tested=int(j <= fm))
                    if lb is not None:
                        sp = lat["sel_prob"][lb][j].astype(float)
                        f.update(margin=float(sp[0] - sp[1]), p2=float(sp[1]), ent=float(-(sp[sp > 0] * np.log(sp[sp > 0])).sum()),
                                 ncand=int((sp > 0.01).sum()), lat=1)
                    else:
                        f.update(margin=np.nan, p2=np.nan, ent=np.nan, ncand=-1, lat=0)
                    rows.append(f)
                    # advance the line state with the drafted token
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
                    p += 1; prev = s; draft_prefix.append(int(d[j]))
                checks.append(dict(dom=dom, sub=sub, win=wi, check=ci, fm=fm, kept=kept, lat_fm=lfm, prev_kept=prev_kept[-1]))
                prev_kept.append(kept)
    return rows, checks, aligned, total


def rate_table(rows, key, base, title, order=None):
    vals = order or sorted({r[key] for r in rows}, key=lambda v: (str(type(v)), v))
    out = []
    for v in vals:
        sub = [r for r in rows if r[key] == v]
        if len(sub) < 20:
            continue
        m = sum(r["miss"] for r in sub) / len(sub)
        out.append("%s %.3f (x%.2f, n %d, %.0f%% of misses)" % (str(v), m, m / base, len(sub), 100 * sum(r["miss"] for r in sub) / max(1, sum(r["miss"] for r in rows))))
    print("  %s: " % title + "; ".join(out))


def features(rows, use):
    X = []
    for r in rows:
        f = []
        if "cf" in use:
            cf = min(max(r["cf"], 1e-4), 1 - 1e-4); f += [np.log(cf / (1 - cf))]
        if "depth" in use:
            f += [r["depth"] / 15.0]
        if "class" in use:
            f += [1.0 if r["dcls"] == c else 0.0 for c in CLASSES[:-1]]
        if "copy" in use:
            f += [r["copy"], r["nov3"], r["nov1"], np.log1p(r["pil"])]
        if "hist" in use:
            f += [r["prev_kept"] / 16.0, 1.0 if r["prev_kept"] < 8 else 0.0]
        if "margin" in use:
            m = min(max(r["margin"], 1e-4), 1 - 1e-4); p2 = min(max(r["p2"], 1e-5), 1 - 1e-5)
            f += [np.log(m / (1 - m)), np.log(p2 / (1 - p2)), r["ent"], r["ncand"] / 16.0]
        X.append(f)
    return np.array(X, float)


def eval_sets(rows, allrows, checks, label):
    wins = sorted({(r["sub"], r["win"]) for r in rows})
    fold = {w: i % 2 for i, w in enumerate(wins)}
    y = np.array([r["miss"] for r in rows], float)
    print("\n  == %s: per-row miss AUC (2 folds by window, held-out) and check-level AUCs ==" % label)
    SETS = [("cf", ["cf"]), ("cf + depth", ["cf", "depth"]), ("cf + depth + history", ["cf", "depth", "hist"]),
            ("cf + depth + class", ["cf", "depth", "class"]), ("cf + depth + copy / novelty / line pos", ["cf", "depth", "copy"]),
            ("cf + depth + class + copy + history", ["cf", "depth", "class", "copy", "hist"])]
    has_lat = [r for r in rows if r["lat"] == 1]
    for name, use in SETS:
        pred = np.zeros(len(rows)); pall = np.zeros(len(allrows))
        Xa = features(allrows, use)
        for k in (0, 1):
            tr = np.array([fold[(r["sub"], r["win"])] != k for r in rows]); te = ~tr
            X = features(rows, use); w = fit_logit(X[tr], y[tr], l2=1.0); pred[te] = predict(X[te], w)
            tea = np.array([fold[(r["sub"], r["win"])] == k for r in allrows]); pall[tea] = predict(Xa[tea], w)
        # check-level: P(first miss <= 8 rows) and P(all 15 right) from the per-row predictions over ALL 15 drafted rows
        byc = {}
        for r, p in zip(allrows, pall):
            byc.setdefault((r["sub"], r["win"], r["check"]), []).append((r["depth"], p))
        p_early, y_early, p_all, y_all = [], [], [], []
        for c in checks:
            key = (c["sub"], c["win"], c["check"])
            if key not in byc:
                continue
            ps = [p for d, p in sorted(byc[key])]
            if len(ps) < 15:
                continue                                       # a short last check: no full-width label
            surv = np.cumprod([1 - p for p in ps])
            p_early.append(1 - surv[7]); y_early.append(c["fm"] < 8)
            p_all.append(surv[-1]); y_all.append(c["fm"] >= 15)
        print("    %-42s row AUC %.3f | check: first miss <= 8 AUC %.3f | all right AUC %.3f" % (name, auc(pred, y), auc(p_early, y_early), auc(p_all, y_all)))
    if len(has_lat) > 200:
        yl = np.array([r["miss"] for r in has_lat], float)
        print("  -- on the lattice-aligned rows (n %d, misses %d): the margin's increment --" % (len(has_lat), int(yl.sum())))
        for name, use in (("cf + depth", ["cf", "depth"]), ("margin + rank + depth", ["margin", "depth"]), ("cf + margin + rank + depth", ["cf", "margin", "depth"]),
                          ("everything", ["cf", "margin", "depth", "class", "copy", "hist"])):
            pred = np.zeros(len(has_lat))
            for k in (0, 1):
                tr = np.array([fold[(r["sub"], r["win"])] != k for r in has_lat]); te = ~tr
                X = features(has_lat, use); w = fit_logit(X[tr], yl[tr], l2=1.0); pred[te] = predict(X[te], w)
            print("    %-42s row AUC %.3f" % (name, auc(pred, yl)))


def cut_replay(rows, allrows, checks, label):
    """the per-row CUT (stop verifying at the predicted first miss) at the real anchors, flat cost as M-21's harness:
    cycle = M0 + 2.0 x rows + 6.3 with M0 = 96.8 - 2.0 x 10.6 - 6.3; kept counts the bonus token. Rules: t 0.3 on the
    drafter's cf (today), the cf-only logistic survival >= mu, the full-signal survival >= mu (mu tuned on the training
    fold), and the oracle (rows = the first miss). 2 folds by window."""
    wins = sorted({(r["sub"], r["win"]) for r in rows}); fold = {w: i % 2 for i, w in enumerate(wins)}
    y = np.array([r["miss"] for r in rows], float)
    M0 = 96.8 - 2.0 * 10.6 - 6.3
    byc = {}
    for r in allrows:
        byc.setdefault((r["sub"], r["win"], r["check"]), []).append(r)
    full = [c for c in checks if len(byc.get((c["sub"], c["win"], c["check"]), [])) == 15]
    def score(rule, k=None, w=None, mu=None, use=None):
        kept = rows_ = 0.0
        for c in full:
            if k is not None and fold[(c["sub"], c["win"])] != k:
                continue
            rs = sorted(byc[(c["sub"], c["win"], c["check"])], key=lambda r: r["depth"])
            fm = c["fm"]                                       # 0-based miss row; 15 = none
            if rule == "t03":
                cp = np.cumprod([r["cf"] for r in rs]); below = np.nonzero(cp < 0.3)[0]
                n = 1 + int(below[0]) if len(below) else 15
            elif rule == "oracle":
                n = min(fm + 1, 15)
            else:
                X = features(rs, use); pm = predict(X, w); surv = np.cumprod(1 - pm); below = np.nonzero(surv < mu)[0]
                n = 1 + int(below[0]) if len(below) else 15
            kept += min(fm, n) + 1; rows_ += n
        return kept, rows_
    def tps(kr, nck):
        return 1e3 * (kr[0] / nck) / (M0 + 2.0 * kr[1] / nck + 6.3)
    nck = len(full)
    base = score("t03"); orc = score("oracle")
    print("\n  == %s: the per-row cut at the real anchors (%d full checks; flat cost; tok/s vs t 0.3) ==" % (label, nck))
    print("    %-34s kept/check %.2f rows/check %.2f tok/s %.1f" % ("t 0.3 (today)", base[0] / nck, base[1] / nck, tps(base, nck)))
    print("    %-34s kept/check %.2f rows/check %.2f tok/s %.1f (%+.1f %%)" % ("oracle (rows = first miss)", orc[0] / nck, orc[1] / nck, tps(orc, nck), 100 * (tps(orc, nck) / tps(base, nck) - 1)))
    for name, use in (("cf + depth survival", ["cf", "depth"]), ("cf + depth + history", ["cf", "depth", "hist"]), ("+ class + copy + novelty + line", ["cf", "depth", "class", "copy", "hist"])):
        tot = [0.0, 0.0]; mus = []
        for k in (0, 1):
            tr = np.array([fold[(r["sub"], r["win"])] != k for r in rows])
            X = features(rows, use); w = fit_logit(X[tr], y[tr], l2=1.0)
            # tune mu on the training fold's checks
            best_mu, best_v = 0.3, -1
            for mu in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                kr = score("model", k=1 - k, w=w, mu=mu, use=use); n_tr = sum(1 for c in full if fold[(c["sub"], c["win"])] != k)
                v = tps(kr, n_tr)
                if v > best_v:
                    best_v, best_mu = v, mu
            kr = score("model", k=k, w=w, mu=best_mu, use=use); tot[0] += kr[0]; tot[1] += kr[1]; mus.append(best_mu)
        print("    %-34s kept/check %.2f rows/check %.2f tok/s %.1f (%+.1f %%; mu %s)" % (name, tot[0] / nck, tot[1] / nck, tps(tot, nck), 100 * (tps(tot, nck) / tps(base, nck) - 1), mus))


for dom in ("code", "chat", "math"):
    allrows, checks, aligned, total = load_rows(dom)
    rows = [r for r in allrows if r['tested']]
    n_tested = len(rows); n_miss = sum(r["miss"] for r in rows); base = n_miss / n_tested
    fms = np.array([c["fm"] for c in checks])
    print("\n==== %s: %d checks, %d tested rows, %d first misses (base miss rate per tested row %.4f); kept/check %.2f; lattice-aligned checks %d / %d ===="
          % (dom, len(checks), n_tested, n_miss, base, np.mean([c["kept"] for c in checks]), aligned, total))
    print("  first-miss depth (1-based) histogram: " + " ".join("%d:%d" % (d + 1, int((fms == d).sum())) for d in range(15)) + "  none:%d" % int((fms >= 15).sum()))
    agree = [c for c in checks if c["lat_fm"] >= 0]
    if agree:
        d = np.array([c["lat_fm"] - c["fm"] for c in agree])
        print("  lattice vs walk first miss on aligned checks: equal %.3f, lattice later %.3f, earlier %.3f (n %d); lattice fm >= walk fm %.3f"
              % (np.mean(d == 0), np.mean(d > 0), np.mean(d < 0), len(agree), np.mean(d >= 0)))
    print("  rates per tested row, by the TRUE token's class (what the model had to produce):")
    rate_table(rows, "tcls", base, "true class", CLASSES)
    print("  by the DRAFTED token's class (known before the check):")
    rate_table(rows, "dcls", base, "drafted class", CLASSES)
    for r in rows:
        r["pil_b"] = "0" if r["pil"] == 0 else "1-2" if r["pil"] <= 2 else "3-8" if r["pil"] <= 8 else "9+"
        r["depth_b"] = "1-2" if r["depth"] < 2 else "3-5" if r["depth"] < 5 else "6-10" if r["depth"] < 10 else "11-15"
        r["hist_b"] = "prev<8" if r["prev_kept"] < 8 else "prev 8-14" if r["prev_kept"] < 15 else "prev full"
        r["cf_b"] = "<0.5" if r["cf"] < 0.5 else "0.5-0.9" if r["cf"] < 0.9 else "0.9-0.99" if r["cf"] < 0.99 else ">=0.99"
    rate_table(rows, "pil_b", base, "position in line", ["0", "1-2", "3-8", "9+"])
    rate_table(rows, "depth_b", base, "depth in block", ["1-2", "3-5", "6-10", "11-15"])
    rate_table(rows, "copy", base, "inside a copy span (8-gram seen earlier)", [1, 0])
    rate_table(rows, "nov3", base, "drafted 3-gram novel in the window", [1, 0])
    rate_table(rows, "nov1", base, "drafted token never seen in the window", [1, 0])
    rate_table(rows, "hist_b", base, "after the previous check", ["prev<8", "prev 8-14", "prev full"])
    rate_table(rows, "cf_b", base, "the drafter's cf", ["<0.5", "0.5-0.9", "0.9-0.99", ">=0.99"])
    eval_sets(rows, allrows, checks, dom)
    cut_replay(rows, allrows, checks, dom)
print("\n(%.0f s)" % (time.time() - T0))
