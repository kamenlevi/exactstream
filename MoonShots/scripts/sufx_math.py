"""M-05 SUFX (MoonShots Thu 15:55; exact, CPU): the 125B's own past outputs as a co-drafter for the D98 math gap.
At each check of D-02b's walk (DFlash2 L1-20, block 16, 1 step; per check (accepted a, checkable, anchor); kept = a + 1,
the next anchor is anchor + a + 1; the block starts with the known anchor token and drafts ids[anchor + 1 .. + 15]; fixed
~16:00, the first run drafted from ids[:anchor] and so had to guess the anchor too) a suffix lookup proposes up to 15
tokens: the longest suffix of ids[:anchor + 1] (L <= 16 tokens) found in the datastore, then a majority walk over the continuations of all its
occurrences. Datastore: the other walk samples (prompt + answer, leave-one-out) and the sample's own ids[:p] (prompt
lookup; continuations stop at p). Rule: verify the datastore block instead of DFlash2's when L >= m; m is tuned on even
samples and scored on odd ones, and the other way round. Kept per check on the walk's own trajectory (positions fixed).
PASS (written first): held-out kept per check >= +0.35 on math (7.00 -> 7.35). KILL: < +0.10, or the oracle
max(a, s) < +0.30.
usage: [BigMoE venv] taskpolicy -b python sufx_math.py -> stdout + data/sufx_math.txt"""
import json, os, time
from collections import Counter, defaultdict
import numpy as np

LMAX, NDR = 16, 15
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def load_walk(path):
    w = np.load(path)
    n = len(w["names"])
    return [(w[f"{i}_ids"].astype(np.int64).tolist(), int(w[f"{i}_gen"]), w[f"{i}_checks"].tolist()) for i in range(n)]


def index(docs):
    """L-gram -> list of (doc, end) for L in 1..LMAX (end = index after the match)"""
    ix = defaultdict(list)
    for d, t in enumerate(docs):
        for L in range(1, LMAX + 1):
            for j in range(L, len(t) + 1):
                ix[(L, tuple(t[j - L:j]))].append((d, j))
    return ix


def draft(docs, ix, cur, p, use_ctx, use_ds, keep=None):
    """-> (L, draft tokens) for sample `cur` at check start p"""
    t = docs[cur]
    for L in range(min(LMAX, p), 0, -1):
        cand = []
        for d, j in ix.get((L, tuple(t[p - L:p])), ()):
            if d == cur:
                if use_ctx and j < p:
                    cand.append((d, j))
            elif use_ds and (keep is None or d in keep):
                cand.append((d, j))
        if not cand:
            continue
        out = []
        for k in range(NDR):
            c = Counter()
            for d, j in cand:
                lim = p if d == cur else len(docs[d])
                if j + k < lim:
                    c[docs[d][j + k]] += 1
            if not c:
                break
            tok = c.most_common(1)[0][0]
            out.append(tok)
            cand = [(d, j) for d, j in cand if j + k < (p if d == cur else len(docs[d])) and docs[d][j + k] == tok]
        return L, out
    return 0, []


def score(walk, docs, ix, use_ctx, use_ds, keep_fn=None):
    """-> per check rows (sample, a, L, s)"""
    rows = []
    for i, (t, gen, checks) in enumerate(walk):
        keep = keep_fn(i) if keep_fn else None
        for a, _, anc in checks:
            p = anc + 1                                  # the anchor ids[anc] is known; drafts cover ids[anc + 1 ..]
            L, dr = draft(docs, ix, i, p, use_ctx, use_ds, keep)
            s = 0
            while s < len(dr) and p + s < len(t) and dr[s] == t[p + s]:
                s += 1
            rows.append((i, a, L, s))
    return np.array(rows)


def rule(R, m):
    return np.where(R[:, 2] >= m, R[:, 3], R[:, 1]) + 1


def report(name, R, base):
    ev, od = R[:, 0] % 2 == 0, R[:, 0] % 2 == 1
    ms = range(1, LMAX + 1)
    m_ev = max(ms, key=lambda m: rule(R[ev], m).mean())
    m_od = max(ms, key=lambda m: rule(R[od], m).mean())
    held = np.concatenate([rule(R[od], m_ev), rule(R[ev], m_od)]).mean()
    orc = (np.maximum(R[:, 1], R[:, 3]) + 1).mean()
    mb = m_ev
    use = R[:, 2] >= mb
    say(f"  {name:44s} held-out {held:5.2f} ({held - base:+.2f}); oracle {orc:5.2f} ({orc - base:+.2f}); m (even / odd) "
        f"{m_ev} / {m_od}; at m {mb}: used on {100 * use.mean():4.1f} % of checks, mean s {R[use, 3].mean() if use.any() else 0:5.2f} "
        f"vs DFlash2 a {R[use, 1].mean() if use.any() else 0:5.2f} there")
    return held, orc


t0 = time.time()
for dom, path in (("math", "~/TrainLab/data/d02b_walk_math.npz"), ("code", "~/TrainLab/data/d02b_walk_code.npz")):
    walk = load_walk(os.path.expanduser(path))
    docs = [t for t, _, _ in walk]
    base = np.mean([a + 1 for _, _, ch in walk for a, _, _ in ch])
    say(f"== {dom}: {len(walk)} samples, {sum(len(ch) for _, _, ch in walk)} checks, DFlash2 kept per check {base:.2f} "
        f"(D-02b), answer tokens {sum(len(t) - g for t, g, _ in walk):,}")
    ix = index(docs)
    report("prompt lookup alone (own prompt + prefix)", score(walk, docs, ix, True, False), base)
    report("datastore alone (other samples)", score(walk, docs, ix, False, True), base)
    R = score(walk, docs, ix, True, True)
    report("both (the screened variant)", R, base)
    if dom == "math":
        rng = np.random.default_rng(0)
        for k in (7, 15, 30):
            sub = {i: set(rng.choice([j for j in range(len(walk)) if j != i], k, replace=False).tolist()) for i in range(len(walk))}
            report(f"both, datastore = {k} other samples", score(walk, docs, ix, True, True, lambda i: sub[i]), base)
        ext = []
        for line in open(os.path.expanduser("~/EvalSuite/results/ref/decode.jsonl")):
            r = json.loads(line)
            if r["set"] in ("math", "mgsm"):
                ext.append([int(x) for x in r["completion_ids"]])
        docs2 = docs + ext
        ix2 = index(docs2)
        report(f"both + Eval-suite ref decodes ({len(ext)} answers)", score(walk, docs2, ix2, True, True), base)
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/sufx_math.txt"), "w").write("\n".join(lines) + "\n")
