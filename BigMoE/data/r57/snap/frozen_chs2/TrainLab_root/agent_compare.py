"""TrainLab (CPU): tuned vs stock 125B MTP head on Chat1000's 12 agentic turns, paired.
Stock: ~/Chat1000/data/agent125/<turn>.npz (Chat1000's run); tuned: ~/TrainLab/data/agent125_<run>/<turn>.npz (the same
script with the tuned head). Both: out = exact greedy tokens, mtp[p][j] = the chain's guess for out[p + j].
tau(K) = tokens per pass walking passes with the MTP chain alone (agent_analyze.py's "mtp" policy; K 15 = 16-row pass).
Paired 90 % interval: bootstrap over blocks of 32 positions (passes assigned by start position), only where the two
runs produced the same tokens. usage: agent_compare.py [RUN]   -> stdout + data/agent_compare_<run>.txt"""
import glob, os, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "mtp125n_d4"
SD = os.path.expanduser("~/Chat1000/data/agent125")
TD = os.path.expanduser(f"~/TrainLab/data/agent125_{RUN}")
BLK = 32
out_lines = []


def say(s):
    print(s, flush=True); out_lines.append(s)


def walk(out, mtp, K, T):
    p, res = 0, []
    while p < T:
        d = list(mtp[p][:K]) if mtp[p][0] >= 0 else []
        n = 0
        while n < len(d) and p + n < T and d[n] == out[p + n]:
            n += 1
        res.append((p, n + 1)); p += n + 1
    return res


rows = []
for f in sorted(glob.glob(os.path.join(TD, "*.npz"))):
    name = os.path.basename(f)[:-4]
    sf = os.path.join(SD, name + ".npz")
    if not os.path.exists(sf):
        continue
    t, s = np.load(f), np.load(sf)
    ot, os_ = t["out"], s["out"]
    L = min(len(ot), len(os_))
    same = int(np.argmax(ot[:L] != os_[:L])) if np.any(ot[:L] != os_[:L]) else L
    rows.append((name, ''.join(c for c in name if c.isalpha()), same, L, s, t))
    say(f"{name}: {L} tokens, identical for the first {same}; depth-1 hit stock "
        f"{np.mean(s['mtp'][:same - 1, 0] == os_[:same - 1]):.3f} tuned {np.mean(t['mtp'][:same - 1, 0] == ot[:same - 1]):.3f}")

rng = np.random.default_rng(0)
for K in (8, 15):
    say(f"-- K {K} (tau = tokens per pass, MTP chain alone; paired over the identical prefixes)")
    acc = {}
    for name, cls, same, L, s, t in rows:
        for tag, z in (("stock", s), ("tuned", t)):
            for p, e in walk(z["out"], z["mtp"], K, same):
                for c in (cls, "ALL"):
                    acc.setdefault((c, tag), []).append((name, p // BLK, e))
    for c in sorted({k[0] for k in acc}, key=lambda c: (c == "ALL", c)):
        S, Tn = acc[(c, "stock")], acc[(c, "tuned")]
        tau = lambda P: sum(e for *_, e in P) / len(P)
        blocks = sorted({(n, b) for n, b, _ in S} | {(n, b) for n, b, _ in Tn})
        bi = {k: i for i, k in enumerate(blocks)}
        def per_block(P):
            tok, ps = np.zeros(len(blocks)), np.zeros(len(blocks))
            for n, b, e in P:
                tok[bi[(n, b)]] += e; ps[bi[(n, b)]] += 1
            return tok, ps
        ts, pss = per_block(S); tt, pst = per_block(Tn)
        bs = []
        for _ in range(4000):
            ix = rng.integers(0, len(blocks), len(blocks))
            bs.append((tt[ix].sum() / max(pst[ix].sum(), 1)) / (ts[ix].sum() / max(pss[ix].sum(), 1)))
        lo, hi = np.percentile(bs, [5, 95])
        r = tau(Tn) / tau(S)
        say(f"  {c:6s} stock {tau(S):.2f}  tuned {tau(Tn):.2f}  {100 * (r - 1):+.1f} % [90 % {100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}]"
            f"  ({len(S)} / {len(Tn)} passes)")
for tag, j in (("stock", 4), ("tuned", 5)):
    ch = []
    for row in rows:
        z, same = row[j], row[2]
        m, o = z["mtp"][:same], z["out"][:same]
        ok = np.ones(len(m), bool)
        c = []
        for d in range(15):
            v = np.array([p + d < same and m[p][d] == o[p + d] for p in range(len(m))])
            ok &= v; c.append(ok.sum())
        ch.append(np.array(c, float) / len(m))
    say(f"P(first j drafts right), j = 1..15, {tag}: " + " ".join(f"{v:.2f}" for v in np.mean(ch, 0)))
open(os.path.expanduser(f"~/TrainLab/data/agent_compare_{RUN}.txt"), "w").write("\n".join(out_lines) + "\n")
