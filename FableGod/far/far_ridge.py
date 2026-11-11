"""FAR ridge (Fable FAR, Wed 2026-11-04): a LEARNED state -> route map (SP-MoE / DraftExpert / LEAD1CORR style, adapted):
ridge from the 125B's MoE input at a shallow layer s of a row to that row's top-10 membership at EVERY layer (48 x 512
binary targets), fit leave-one-prompt-out on the other five prompts' rows (all 16 rows per check). The future rows' x_s is an
oracle stand-in for a far-ahead drafter's state (or a 2-layer pre-pass of the 125B over the far draft). Same frame as
far_ceiling.py (M-38's chained cap45 replay, K 266 / 147, H 8, FH policy); forecast per check = the union of its rows'
predicted top-k (all 16 rows, or the kept rows only).
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_ridge.py [K ...] (one core, < 3 GB, ~3-6 min) -> far_ridge.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E, NI = F.P, F.NL, F.E, F.NI
OUT = os.path.expanduser("~/FableGod/far/far_ridge.txt")
t0 = time.time()
lines = []
ITEM_OF = {(int(F.PRM[i]), int(F.PCHK[i])): i for i in range(NI) if F.PCHK[i] >= 0}


def say(s):
    print(s, flush=True); lines.append(s)


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_rows(s):
    out = []
    for p, (name, cks, _) in enumerate(P):
        z = np.load(os.path.join(F.CAP, name + "_gate_0.npz"))
        seq, L, rows, xoff = z["seq"], z["L"], z["rows"], z["x_off"]
        xs = z["x"]
        want = {}
        for n, (q, l, r) in enumerate(zip(seq, L, rows)):
            if int(l) == s and int(r) == 16 and int(q) not in want:
                want[int(q)] = int(xoff[n])
        X, CHK, Y, KR = [], [], [], []
        for t, c in enumerate(cks):
            o = want[c["seq"]]
            X.append(bf16(xs[o:o + 16]))
            CHK += [t] * 16
            y = np.zeros((16, NL, E), bool)
            for l in range(NL):
                np.put_along_axis(y[:, l], c["rows"][l], True, axis=1)
            Y.append(y)
            KR += [j < c["kept"] for j in range(16)]
        X = np.concatenate(X)
        out.append((X, np.array(CHK), np.concatenate(Y), np.array(KR)))
    return out


def run(K, tag, FC, ref):
    S = {"n": 0, "pos": 0, "tp": 0, "fp": 0}
    miss, stall = F.replay(K, "fh", nxt=F.next_from(FC), stats=S)
    m, s_ = F.pooled(miss), F.pooled_chk(stall)
    fpos = S["tp"] + S["fp"]
    say(f"  K {K} {tag:30s}: {m:6.1f} misses / kept token [{m * F.MSB / 1e6:6.1f} MB], stalls {s_:5.1f}, "
        f"{100 * (ref[0] - m) / ref[0]:5.1f} % fewer misses than LRU | recall {S['tp'] / S['pos']:.2f}, precision "
        f"{S['tp'] / max(fpos, 1):.2f} (base {S['pos'] / S['n']:.2f}, forecast-positive {fpos / S['n']:.2f}) ({time.time() - t0:.0f} s)")


say(f"FAR RIDGE ({time.strftime('%a %Y-%m-%d %H:%M')}): ridge x_s (2560 + bias) -> top-10 membership at all 48 layers, "
    f"leave-one-prompt-out, all 16 rows per check; K {F.KS}; H {F.H}")
for s in (1, 3):
    RW = load_rows(s)
    Xall = [np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) for X, _, _, _ in RW]
    scale = float(np.mean([np.mean(np.sum(X * X, 1)) for X in Xall]))        # mean squared row norm
    for alpha in (0.03, 0.3):
        lam = alpha * scale
        PRED = []                                                            # per prompt: [n, 48, 512] scores
        for p in range(len(P)):
            Xtr = np.concatenate([Xall[q] for q in range(len(P)) if q != p])
            Ytr = np.concatenate([RW[q][2] for q in range(len(P)) if q != p]).reshape(len(Xtr), -1).astype(np.float32)
            G = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
            W = np.linalg.solve(G, Xtr.T @ Ytr)
            PRED.append((Xall[p] @ W).reshape(len(Xall[p]), NL, E))
            del Xtr, Ytr, G, W
        rec, n, rec_k, n_k = np.zeros(NL), 0, np.zeros(NL), 0
        for p in range(len(P)):
            X, CHK, Y, KR = RW[p]
            top = np.argsort(-PRED[p], axis=2)[:, :, :10]
            hit = np.take_along_axis(Y[:, :, :], top, axis=2).mean(2)          # [n, 48] recall@10
            rec += hit.sum(0); n += len(hit); rec_k += hit[KR].sum(0); n_k += int(KR.sum())
        say(f"s {s} alpha {alpha}: held-out per-row recall@10, all rows ({n}) / kept rows ({n_k}): "
            + ", ".join(f"L{a}-{b - 1} {rec[a:b].mean() / n:.2f}/{rec_k[a:b].mean() / n_k:.2f}"
                        for a, b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 48))) + f" ({time.time() - t0:.0f} s)")
        for k in (10, 16):
            for kept_only in (False, True):
                FC = np.zeros((NL, NI, E), bool)
                for p in range(len(P)):
                    X, CHK, Y, KR = RW[p]
                    top = np.argsort(-PRED[p], axis=2)[:, :, :k]
                    for t in range(len(P[p][1])):
                        rows = np.nonzero(CHK == t)[0]
                        if kept_only:
                            rows = rows[KR[rows]]
                        i = ITEM_OF[(p, t)]
                        for l in range(NL):
                            FC[l, i, top[rows, l].ravel()] = True
                for K in F.KS:
                    run(K, f"RIDGE-{s} a{alpha} top{k} {'kept' if kept_only else 'all16'}", FC, F.LRU_REF[K])
        del PRED
    del RW, Xall
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
