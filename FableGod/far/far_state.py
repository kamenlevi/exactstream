"""FAR state proxy (Fable FAR, Wed 2026-11-04): does a SHALLOW hidden state of the future rows carry the deep routes that the
token identity does not? Same frame as far_ceiling.py (M-38's chained cap45 replay, K 266 / 147, H 8, FH policy).
Forecast KNN-s: the route of a future row (all 16 rows, or the kept rows only) at every layer = the true routes of its
nearest earlier row of this prompt (cosine on the 125B's MoE input at layer s, rows of checks <= the decision's check), m
neighbours' union. The future rows' x_s is an ORACLE here (the 125B's own state); a far-ahead drafter's state stands in for
it on the GPU. Also: per-row recall@10 of the nearest neighbour's route vs the token memo's, per layer group.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_state.py [K ...] (one core, < 2 GB, ~2 min) -> far_state.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
import importlib.util
spec = importlib.util.spec_from_file_location("far_ceiling_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E, NI, PRM, PCHK, PEND, BIG, H = F.P, F.NL, F.E, F.NI, F.PRM, F.PCHK, F.PEND, F.BIG, F.H
OUT = os.path.expanduser("~/FableGod/far/far_state.txt")
t0 = time.time()
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# per prompt: rows of the live checks at layer s (16 each), their check index, their ids at every layer [n, 48, 10]
def load_rows(s):
    out = []
    for p, (name, cks, _) in enumerate(P):
        z = np.load(os.path.join(F.CAP, name + "_gate_0.npz"))
        seq, L, rows, xoff = z["seq"], z["L"], z["rows"], z["x_off"]
        xs = z["x"]
        want = {}                                                    # seq -> x rows at layer s (first entry with 16 rows)
        for n, (q, l, r) in enumerate(zip(seq, L, rows)):
            if int(l) == s and int(r) == 16 and int(q) not in want:
                want[int(q)] = (int(xoff[n]), 16)
        X, CHK, IDS, KEPTROW = [], [], [], []
        for t, c in enumerate(cks):
            q = c["seq"]
            o, r = want[q]
            x = bf16(xs[o:o + r])
            X.append(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6))
            CHK += [t] * r
            IDS.append(np.stack([c["rows"][l] for l in range(NL)], 1))   # [16, 48, 10]
            KEPTROW += [j < c["kept"] for j in range(r)]
        out.append((np.concatenate(X), np.array(CHK), np.concatenate(IDS), np.array(KEPTROW)))
    return out


def knn_look(RW, m, kept_only):
    """per-look forecast at decision (l, i): the next H checks' rows (kept only or all) -> their m nearest earlier rows
    (checks <= the decision's check) -> those rows' true experts at layer l"""
    SIM = [X @ X.T for X, _, _, _ in RW]
    def f(l, i):
        p, tdec = PRM[i], PCHK[i]
        X, CHK, IDS, KR = RW[p]
        d = np.full(E, BIG)
        pool = np.nonzero(CHK <= tdec)[0]
        if pool.size == 0:
            return d
        npool = pool[-1] + 1                                         # rows are in check order: the pool is a prefix
        for j in range(i + 1, min(i + H, PEND[i]) + 1):
            rows = np.nonzero(CHK == PCHK[j])[0]
            if kept_only:
                rows = rows[KR[rows]]
            if rows.size == 0:
                continue
            S = SIM[p][rows, :npool]
            nb = np.argsort(-S, axis=1)[:, :m]
            mask = np.zeros(E, bool)
            mask[IDS[nb.ravel(), l].ravel()] = True
            d[mask & (d == BIG)] = j - i
        return d
    return f


def run(K, tag, look, ref):
    S = {"n": 0, "pos": 0, "tp": 0, "fp": 0}
    miss, stall = F.replay(K, "fh", look=look, stats=S)
    m, s = F.pooled(miss), F.pooled_chk(stall)
    fpos = S["tp"] + S["fp"]
    say(f"  K {K} {tag:28s}: {m:6.1f} misses / kept token [{m * F.MSB / 1e6:6.1f} MB], stalls {s:5.1f}, "
        f"{100 * (ref[0] - m) / ref[0]:5.1f} % fewer misses than LRU | recall {S['tp'] / S['pos']:.2f}, precision "
        f"{S['tp'] / max(fpos, 1):.2f} (base {S['pos'] / S['n']:.2f}, forecast-positive {fpos / S['n']:.2f}) ({time.time() - t0:.0f} s)")


say(f"FAR STATE PROXY ({time.strftime('%a %Y-%m-%d %H:%M')}): KNN on the 125B's MoE input at layer s, M-38's frame; K {F.KS}; H {H}")
for s in (0, 1, 3, 8):
    RW = load_rows(s)
    # per-row recall@10 at every layer: nearest earlier row (checks < this row's check) of the same prompt
    rec, n = np.zeros(NL), 0
    rec_k, n_k = np.zeros(NL), 0
    for X, CHK, IDS, KR in RW:
        SIM = X @ X.T
        for r in range(len(CHK)):
            pool = np.nonzero(CHK < CHK[r])[0]
            if pool.size == 0:
                continue
            nb = pool[np.argmax(SIM[r, pool])]
            hit = np.array([np.isin(IDS[r, l], IDS[nb, l]).mean() for l in range(NL)])
            rec += hit; n += 1
            if KR[r]:
                rec_k += hit; n_k += 1
    say(f"s {s}: per-row recall@10 of the row's true top-10 from its nearest earlier row (x_{s} cosine), all 16 rows ({n}) / "
        f"kept rows ({n_k}): " + ", ".join(f"L{a}-{b - 1} {rec[a:b].mean() / n:.2f}/{rec_k[a:b].mean() / max(n_k, 1):.2f}"
                                            for a, b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 48)))
        + f" ({time.time() - t0:.0f} s)")
    for K in F.KS:
        ref = F.LRU_REF[K]
        if s in (1, 3):
            for m in (1, 2, 4):
                run(K, f"KNN-{s} m{m} all 16 rows", knn_look(RW, m, False), ref)
            run(K, f"KNN-{s} m2 kept rows only", knn_look(RW, 2, True), ref)
        else:
            run(K, f"KNN-{s} m2 all 16 rows", knn_look(RW, 2, False), ref)
    del RW
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
