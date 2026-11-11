"""NEXTCHECK (MTP Head, Tue 11-03; the orchestrator's 'learned expert-route predictor for prefetch', step 2, CPU): how well can
the information at the end of check p name the experts check p + 1 will MISS, so that their reads go into the SSD-idle
draft window (R-70D)? Data: BigMoE's 45 GB-route capture ~/BigMoE/data/cap45 (R-57 harness, C 160, 6 prompts = the
c45l_256 set; per check and layer the 125B's top-10 ids of every row, every row's MoE input x, miss.bin = the experts
served per (check, layer) with the non-resident flag; routers.npz = layers 1-47). Pools: the measured C 160 misses, and a
per-layer LRU replay of the served lists at C 224 / 256 (warm-up: the first 3 checks of each prompt are not scored).
Predictors (all exact-safe: prefetch only), scored on check p + 1's misses at layers 1-47, one global budget B per check
(the reads with the highest scores across layers; candidates = experts not served in checks p and p - 1):
  last   the layer's router softmax on check p's LAST row's x (the bonus token's predecessor)
  rows   the router softmax averaged over check p's rows
  pop    how often the expert was served at that layer in the prompt so far (no model)
  oracle check p + 1's true misses (the ceiling)
Self-check: the router's top-10 on x reproduces the captured ids. Output: data/nextcheck_pred.txt (appended)."""
import os, glob
import numpy as np
D = os.path.expanduser("~/BigMoE/data/cap45")
TD = os.path.expanduser("~/TrainLab/data")
E, K, NL = 512, 10, 48
BUDGETS = (48, 88, 176)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

RT = {int(k[1:]): v.astype(np.float32) for k, v in np.load(os.path.join(D, "routers.npz")).items()}   # L -> [E, 2560]
raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
SERV, MISS, i = {}, {}, 0                                   # (seq, L) -> served ids / measured C 160 misses (first record)
while i < len(raw):
    s, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2]); v = raw[i + 3:i + 3 + n]
    if L < NL and (s, L) not in SERV:
        SERV[(s, L)] = np.unique(v & 0xFFFF); MISS[(s, L)] = np.unique(v[((v >> 16) & 1) == 1] & 0xFFFF)
    i += 3 + n

def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

prompts = []
chk_n = chk_ok = 0
for f in sorted(glob.glob(os.path.join(D, "*_gate_0.npz"))):
    z = np.load(f)
    by = {}
    for j, (s, L) in enumerate(zip(z["seq"].tolist(), z["L"].tolist())):
        by.setdefault(s, {}).setdefault(L, j)               # the first call per (check, layer)
    seqs = [s for s in sorted(by) if all(L in by[s] for L in range(NL)) and all((s, L) in SERV for L in range(NL))]
    P = []
    for s in seqs:
        rec = {}
        for L in range(1, NL):
            j = by[s][L]; o, r, xo = int(z["row_off"][j]), int(z["rows"][j]), int(z["x_off"][j])
            x = bf16(z["x"][xo:xo + r])
            rec[L] = (x, z["ids"][o:o + r].astype(np.int64))
            if chk_n < 400 and L in (5, 25, 45):            # self-check: the router's top-10 on x == the captured ids
                top = np.argsort(-(x @ RT[L].T), axis=-1)[:, :K]
                chk_n += r; chk_ok += sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(top, rec[L][1]))
        P.append((s, rec))
    prompts.append((os.path.basename(f)[:-11], P))
say(f"NEXTCHECK: cap45 (C 160 capture), {sum(len(p) for _, p in prompts)} full checks in {len(prompts)} prompts; self-check router "
    f"top-10 == captured ids on {chk_ok} / {chk_n} rows")

def lru_misses(P, C):
    """per (check index, L): misses of a per-layer LRU of capacity C replaying the served lists (empty start)"""
    out = []
    pools = {L: {} for L in range(1, NL)}
    for t, (s, rec) in enumerate(P):
        m = {}
        for L in range(1, NL):
            pl, sv = pools[L], SERV[(s, L)]
            m[L] = np.array([e for e in sv if e not in pl], np.int64)
            for e in sv:
                pl.pop(e, None); pl[e] = t
            while len(pl) > C:
                pl.pop(next(iter(pl)))
        out.append(m)
    return out

res = {}
for C in (160, 224, 256):
    tot = {(nm, B): [0, 0] for nm in ("last", "rows", "pop", "oracle") for B in BUDGETS}
    nmiss = nchk = 0
    for name, P in prompts:
        M = [{L: MISS[(s, L)] for L in range(1, NL)} for s, _ in P] if C == 160 else lru_misses(P, C)
        pop = {L: np.zeros(E) for L in range(1, NL)}
        for t in range(len(P) - 1):
            s, rec = P[t]; s1 = P[t + 1][0]
            for L in range(1, NL):
                pop[L][rec[L][1].ravel()] += 1
            if t < 2:                                       # checks 0-2 are warm-up (scored from t + 1 = 3)
                continue
            excl = {L: set(SERV[(s, L)].tolist()) | set(SERV[(P[t - 1][0], L)].tolist()) for L in range(1, NL)}
            true = {L: set(M[t + 1][L].tolist()) for L in range(1, NL)}
            nm_ = sum(len(v) for v in true.values()); nmiss += nm_; nchk += 1
            cand = {}
            for L in range(1, NL):
                x = rec[L][0]
                cand[("last", L)] = softmax(x[-1:] @ RT[L].T)[0]
                cand[("rows", L)] = softmax(x @ RT[L].T).mean(0)
                cand[("pop", L)] = pop[L] / max(pop[L].sum(), 1)
            for nm in ("last", "rows", "pop"):
                sc = []
                for L in range(1, NL):
                    v = cand[(nm, L)]
                    for e in np.argsort(-v)[:max(BUDGETS)]:
                        if int(e) not in excl[L]:
                            sc.append((float(v[e]), L, int(e)))
                sc.sort(reverse=True)
                for B in BUDGETS:
                    hit = sum(1 for _, L, e in sc[:B] if e in true[L])
                    tot[(nm, B)][0] += hit; tot[(nm, B)][1] += min(B, len(sc))
            for B in BUDGETS:
                tot[("oracle", B)][0] += min(B, nm_); tot[("oracle", B)][1] += B
    res[C] = (tot, nmiss / max(nchk, 1), nchk)
    say(f"  C {C} ({'measured flags' if C == 160 else 'LRU replay'}): {nchk} scored checks, {nmiss / max(nchk, 1):.0f} misses per check "
        f"(layers 1-47)")
    for B in BUDGETS:
        say(f"    B {B:3d} reads: " + "; ".join(f"{nm} precision {tot[(nm, B)][0] / max(tot[(nm, B)][1], 1):.3f} recall "
                                             f"{tot[(nm, B)][0] / max(nmiss, 1):.3f}" for nm in ("last", "rows", "pop", "oracle")))
open(os.path.join(TD, "nextcheck_pred.txt"), "a").write("\n".join(lines) + "\n\n")
