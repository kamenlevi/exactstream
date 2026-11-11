"""TREES side analysis (Breakthrough Sun 1 Nov 11:5x; NOT in the registered lines: written after tree_price.py's result): the
cost per kept token of confidence-truncated chains (the live 45 GB rule: drafts kept while the running product of the
selector's confidence >= t; t 0.3 live) and of dynamic first-order trees (paths with probability >= tau, <= 16 rows), per
domain, on the same lattices, under four cost-per-check models (rows = the anchor + drafted nodes):
  GB     BigMoE's rows curve (4-bit, C 160; tree_price.py's)
  45GB   ms = 65.9 + 14.19 x rows: the line through Leap's S45c checks on the 45 GB engine (C 224: 293 ms at 16 rows, 205 ms
         at 9.8; the drafter's time is inside)
  RAM    ms = 15 + 2.88 x rows (Leap's L5 resident model, dfl2_trunc.py)
  SSD    ms = 8.8 + 31.5 x rows^0.7185 (Leap's L5 model of Fit12's SSD-streamed C192 passes)
  D23    ms = F + 7.82 x rows, measured on TODAY's 45 GB default (C 224, gate + REF2-T, t 0.3): 7.82 ms per row is D-23b's
         fixed-effects slope; F = D-23's dfl arm's mean decode ms per check - 7.82 x its mean rows, per type: code 109.8,
         math 80.0 (also mgsm / gsm), chat 64.1 (ifeval / multi), agent 179.2 (added 11:5x, after s45d_replay.py)
Trees are priced at chain rows (as registered) and, as a sensitivity, with a generous discount on their off-chain rows
(0.75x each: MoonShots' M-10 measured whole trees at 0.84-0.91x of a chain's experts; nodes on the chain cost as in a chain). 'plain' = 1 row per token, no drafter (the 45GB line's intercept holds the drafter, so
its plain cost is an upper bound).
Usage: [BigMoE venv] python tree_frontier.py -> stdout + data/trees/frontier.txt"""
import heapq, os
import numpy as np

D = os.path.expanduser("~/Breakthrough/data/trees")
Z = [np.load(f"{D}/lattice_{t}.npz") for t in ("code", "chat", "rest")]
cat = lambda k: np.concatenate([z[k] for z in Z])
CAND, TRANS, PICK, TRUE, NT, SPROB, DOM = (cat(k) for k in ("cand", "trans", "pick", "true", "ntrue", "sel_prob", "dom"))
B, DEP, K = CAND.shape
CAP = np.where(NT == DEP, DEP + 1, NT)
RG, GBP = np.array([1, 2, 4, 8, 12, 16]), np.array([1.40, 2.29, 3.82, 6.29, 8.32, 10.03])
MODELS = {"GB": lambda r: np.interp(r, RG, GBP), "45GB": lambda r: 65.9 + 14.19 * r, "RAM": lambda r: 15 + 2.88 * r,
          "SSD": lambda r: 8.8 + 31.5 * r ** 0.7185}
TS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
TAUS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)
FD23 = {"code": 109.8, "chat (ifeval + multi)": 64.1, "ifeval": 64.1, "multi": 64.1, "agent": 179.2, "mgsm": 80.0,
        "gsm (reference)": 80.0}
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def kbest(ordr, srt, tau):                                             # tree_price.py's k-best, first-order, dynamic size
    out, heap = set(), [(-srt[0, 0, 0], (int(ordr[0, 0, 0]),), 0, 1.0)]
    while heap and len(out) < 15:
        nq, path, r, pp = heapq.heappop(heap)
        if -nq < tau:
            break
        out.add(path)
        d = len(path)
        if d < DEP:
            i = path[-1]
            heapq.heappush(heap, (nq * srt[d, i, 0], path + (int(ordr[d, i, 0]),), 0, -nq))
        if r + 1 < K:
            i = path[-2] if d >= 2 else 0
            heapq.heappush(heap, (-pp * srt[d - 1, i, r + 1], path[:-1] + (int(ordr[d - 1, i, r + 1]),), r + 1, pp))
    return out


KEPT, ROWS, OFF = {}, {}, {}                                           # policy -> [B] (OFF: off-chain tree nodes)
for n in ["chain 16"] + [f"chain t {t}" for t in TS] + [f"tree tau {t}" for t in TAUS]:
    KEPT[n], ROWS[n], OFF[n] = np.zeros(B), np.zeros(B), np.zeros(B)
for b in range(B):
    nt, cap = int(NT[b]), int(CAP[b])
    hit = PICK[b, :nt] == TRUE[b, :nt]
    kc = int(np.argmin(hit)) if not hit.all() else nt
    KEPT["chain 16"][b], ROWS["chain 16"][b] = min(kc + 1, cap), min(16, cap)
    cp = np.cumprod(SPROB[b, :, 0].astype(np.float64))
    for t in TS:
        n = int(np.sum(cp >= t))
        KEPT[f"chain t {t}"][b], ROWS[f"chain t {t}"][b] = min(min(kc, n) + 1, cap), min(n + 1, cap)
    J = np.full(DEP, K)
    for d in range(nt):
        w = np.nonzero(CAND[b, d] == TRUE[b, d])[0]; J[d] = w[0] if len(w) else K
    ch = tuple(int(np.nonzero(CAND[b, d] == PICK[b, d])[0][0]) for d in range(DEP))   # the chain's path (raw indices)
    tr = TRANS[b].astype(np.float64)
    ordr = np.argsort(-tr, axis=-1, kind="stable"); srt = np.take_along_axis(tr, ordr, -1)
    for tau in TAUS:
        T = kbest(ordr, srt, tau)
        d, path = 0, ()
        while d < nt and J[d] < K and path + (int(J[d]),) in T:
            path += (int(J[d]),); d += 1
        KEPT[f"tree tau {tau}"][b], ROWS[f"tree tau {tau}"][b] = min(d + 1, cap), min(len(T) + 1, cap)
        OFF[f"tree tau {tau}"][b] = sum(p != ch[:len(p)] for p in T)

GROUPS = [("code", ("tf_code_HumanEval",)), ("chat (ifeval + multi)", ("tf_ifeval", "tf_multi")), ("ifeval", ("tf_ifeval",)),
          ("multi", ("tf_multi",)), ("agent", ("agent",)), ("mgsm", ("mgsm",)), ("gsm (reference)", ("tf_gsm",))]


def cost(name, m, f, disc=1.0):                                       # cost per kept token (sum of check costs / sum kept);
    r = np.maximum(ROWS[name][m] - (1 - disc) * OFF[name][m], 1.0)    # disc: the cost of an off-chain tree row
    return f(r).sum() / KEPT[name][m].sum()


say(f"TREES frontier (side analysis, not registered): {B} blocks; cost per kept token = sum of check costs / sum of kept tokens; "
    f"GB in GB, the others in ms. 'live' = chain t 0.3 (the 45 GB default's rule).")
for gname, doms in GROUPS:
    m = np.isin(DOM, doms)
    M = dict(MODELS, D23=(lambda F: lambda r: F + 7.82 * r)(FD23[gname]))
    say(f"\n== {gname}: {m.sum()} blocks")
    say("    policy            kept    rows" + "".join(f"{k:>9s}" for k in M))
    for n in ["chain 16"] + [f"chain t {t}" for t in TS] + [f"tree tau {t}" for t in TAUS]:
        say(f"    {n:15s} {KEPT[n][m].mean():6.2f} {ROWS[n][m].mean():7.2f}" + "".join(f"{cost(n, m, f):9.3f}" for f in M.values()))
    say(f"    {'plain (1 row)':15s} {1:6.2f} {1:7.2f}" + "".join(f"{f(1.0):9.3f}" for f in M.values()))
    for k, f in M.items():
        ch = min((cost(n, m, f), n) for n in KEPT if n.startswith("chain"))
        tr = min((cost(n, m, f), n) for n in KEPT if n.startswith("tree"))
        trd = min((cost(n, m, f, disc=0.75), n) for n in KEPT if n.startswith("tree"))
        live = cost("chain t 0.3", m, f)
        say(f"    {k:5s}: best chain {ch[1]} ({100 * (live / ch[0] - 1):+.1f} % tokens per cost vs live t 0.3); best tree {tr[1]} "
            f"{100 * (ch[0] / tr[0] - 1):+.1f} % vs the best chain (off-chain rows at 0.75x: {trd[1]} {100 * (ch[0] / trd[0] - 1):+.1f} %); "
            f"plain (no drafter) {100 * (live / f(1.0) - 1):+.1f} % tokens per cost vs live")
open(f"{D}/frontier.txt", "w").write("\n".join(lines) + "\n")
