#!/usr/bin/env python3 -I
"""replay_c1.py (Fable C, mission C1, Mon 2 Nov 2026, ~23:1x EET). CPU only (numpy, nice 15, 4 threads), recorded data only.
Four measurements on cap45 (BigMoE's 45 GB capture: six prompts, the real 16-row check rows incl. the rejected ones, the
router inputs x per row and layer, res0) and one on Breakthrough's DAY9 routing capture (36 windows, accepted path):
  1. LOOKAHEAD: the recall / precision of R-57's gate-time predictor (layer L+1's router run on layer L's MoE input, one
     ranked unique list per check, top-B) against L+1's true union of top-10s. Calibrates idea C1-1 (the refined look-ahead):
     if the gate-time predictor already recalls >= 0.88, a refinement has nothing to add.
  2. SIDECAR coupling (idea C1-3): the live rule (per layer, LRU + 30 x popularity, strict, prompts chained, res0 forced,
     M-17 per-layer sizes at C 256 / C 320) replayed for both pools, baseline vs
       couple   = when the base pool evicts e, its LSB leaves the sidecar at once (the slot frees);
       deadfirst = couple + the sidecar's victims are the LSBs whose base is not resident, before the LRU + prior order.
     Reported: base / sidecar misses per kept token, SSD MB per kept token, the dead-LSB share of the sidecar at miss time,
     and two side statistics of every sidecar miss: was its base resident before the call (an 'upgrade' miss), and was the
     expert at ranks 2-4 in the previous check of this layer (the 'rank-climb' signal).
  3. HALF (side check): the exclusive expert misses of rows >= s (s = 6 / 8 / 10) under the baseline state, split by
     whether the check accepted >= s rows (kept - 1 >= s), and a cycle model for a split check.
  4. TOKEN (side check, DAY9 capture): per layer, recall@k of a token-id -> expert table (leave-one-window-out, same
     domain) against the true top-10, vs the previous position's own top-10 (Rnd23's 0.28-0.41 baseline).
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I replay_c1.py -> stdout (tee to replay_c1.out)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, pickle, sys, time
from collections import defaultdict, Counter
import numpy as np

T0 = time.time()
NL, E, R4 = 48, 512, 2
MR, LR = 2162688, 770048                          # records3 MSB / LSB parts (bytes per miss)
CAP = os.path.expanduser("~/BigMoE/data/cap45")
PKL = os.path.expanduser("~/MoonShots/data/band45_cap45.pkl")
SZ = {c: json.load(open(os.path.expanduser(f"~/MoonShots/data/laysz45_{c}.json"))) for c in (256, 320)}
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
H = (1, 2, 3, 4, 5)                               # counted prompts (prompt 0 = warm-up), as laysz45
B_PRIOR = 30.0
DMAP = os.path.expanduser("~/FableGod/decision_map/out")


def say(s=""):
    print(s, flush=True)


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- cap45 (rank-sorted ids per check / layer, kept, res0)
# The pickle is MoonShots' own laysz45.load() cache (a sibling session on this Mac, written by scripts/band45.py from
# BigMoE's cap45 npz files), not a download: plain lists / dicts / numpy arrays.
P, agree = pickle.load(open(PKL, "rb"))
assert [p[0] for p in P] == list(ORDER), [p[0] for p in P]
say(f"cap45: {[(n, len(c)) for n, c, _ in P]} checks; router agreement {agree:.4f}; kept per prompt "
    f"{[int(sum(c['kept'] for c in cks)) for _, cks, _ in P]}")
KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
ROWS = np.array([np.mean([len(c["rid"][0]) for c in cks]) for _, cks, _ in P])
say(f"rows per check (as captured): {np.round(ROWS, 2).tolist()}; counted kept tokens {KEPT[list(H)].sum():.0f}")


def needs_of(c, l):
    r = c["rid"][l]
    return frozenset(np.unique(r).tolist()), frozenset(np.unique(r[:, :R4]).tolist())


NEED = [[[needs_of(c, l) for l in range(NL)] for c in cks] for _, cks, _ in P]      # [p][t][l] -> (m, l)

# leave-one-prompt-out popularity (share of checks whose need holds e), per key, prompt, layer
def popularity(key):
    cnt = np.zeros((len(P), NL, E)); n = np.zeros(len(P))
    for p in range(len(P)):
        for t in range(len(P[p][1])):
            n[p] += 1
            for l in range(NL):
                cnt[p, l, list(NEED[p][t][l][key])] += 1
    out = []
    for p in range(len(P)):
        out.append((cnt.sum(0) - cnt[p]) / max(n.sum() - n[p], 1))
    return out


POP = {0: popularity(0), 1: popularity(1)}        # key 0 = base (top-10 union), 1 = sidecar (ranks 0-1)


# ---------------------------------------------------------------- 2. the two-pool replay with coupling arms
def replay_layer(l, Km, Kl, arm, stats):
    """one layer, both pools, prompts chained; arm in ('base', 'couple', 'deadfirst'). Returns per-prompt (miss_m, miss_l)."""
    miss_m = np.zeros(len(P)); miss_l = np.zeros(len(P))
    stm, stl, i = {}, {}, 0
    for p, (name, cks, res0) in enumerate(P):
        popm, popl = POP[0][p][l], POP[1][p][l]
        calls = [(-1, res0[l], frozenset())] + [(t, NEED[p][t][l][0], NEED[p][t][l][1]) for t in range(len(cks))]
        prev_rid = None
        for t, nm, nl in calls:
            i += 1
            counted = t >= 0 and p in H
            # --- side statistics at this call's start (baseline arm only)
            if arm == "base" and t >= 0:
                c = cks[t]; rid = c["rid"][l]; acc = c["kept"] - 1
                for s in (6, 8, 10):
                    if len(rid) > s:
                        lo = set(np.unique(rid[:s]).tolist()); hi = set(np.unique(rid[s:]).tolist()) - lo
                        lol = set(np.unique(rid[:s, :R4]).tolist()); hil = set(np.unique(rid[s:, :R4]).tolist()) - lol
                        xm = sum(1 for e in hi if e not in stm); xl = sum(1 for e in hil if e not in stl)
                        key = (s, "acc" if acc >= s else "rej")
                        if counted:
                            stats["half"][key][0] += xm; stats["half"][key][1] += xl; stats["half"][key][2] += 1 if l == 0 else 0
            # --- base pool
            if counted:
                miss_m[p] += sum(1 for e in nm if e not in stm)
            base_before = {e: (e in stm) for e in nl}
            for e in nm:
                stm[e] = float(i)
            if len(stm) > Km:
                kf = lambda e: stm[e] + B_PRIOR * popm[e]
                victims = heapq.nsmallest(len(stm) - Km, (e for e in stm if e not in nm), key=kf)
                for e in victims:
                    del stm[e]
                if len(stm) > Km:
                    for e in heapq.nsmallest(len(stm) - Km, list(stm), key=kf):
                        victims.append(e); del stm[e]
                if arm in ("couple", "deadfirst"):
                    for e in victims:
                        if e in stl and e not in nl:
                            del stl[e]; stats["purged"] += 1
            # --- sidecar pool
            if t >= 0:
                dead = sum(1 for e in stl if e not in stm)
                stats["dead_num"] += dead; stats["dead_den"] += max(len(stl), 1)
                if counted:
                    for e in nl:
                        if e not in stl:
                            miss_l[p] += 1
                            stats["l_upgrade"] += base_before[e]
                            if prev_rid is not None:
                                stats["l_climb"] += bool(((prev_rid[:, R4:5] == e).any()))
                                stats["l_prev_any"] += bool(((prev_rid == e).any()))
                            stats["l_n"] += 1
            for e in nl:
                stl[e] = float(i)
            if len(stl) > Kl:
                if arm == "deadfirst":
                    kf = lambda e: (0.0 if e in stm else -1e12) + stl[e] + B_PRIOR * popl[e]
                else:
                    kf = lambda e: stl[e] + B_PRIOR * popl[e]
                for e in heapq.nsmallest(len(stl) - Kl, (e for e in stl if e not in nl), key=kf):
                    del stl[e]
                if len(stl) > Kl:
                    for e in heapq.nsmallest(len(stl) - Kl, list(stl), key=kf):
                        del stl[e]
            if t >= 0:
                prev_rid = cks[t]["rid"][l]
    return miss_m, miss_l


say("\n== 2. SIDECAR coupling (cap45, 16-row checks as captured, LRU + 30 x popularity, strict, M-17 per-layer sizes) ==")
HALF = None
for C in (256, 320):
    Cm, Cl = SZ[C]["C_m"], SZ[C]["C_l"]
    say(f"-- C {C}: per-layer C_m sum {sum(Cm)} (mean {np.mean(Cm):.0f}), C_l sum {sum(Cl)} (mean {np.mean(Cl):.0f})")
    for arm in ("base", "couple", "deadfirst"):
        st = {"purged": 0, "dead_num": 0, "dead_den": 0, "l_upgrade": 0, "l_climb": 0, "l_prev_any": 0, "l_n": 0,
              "half": defaultdict(lambda: [0, 0, 0])}
        mm = np.zeros(len(P)); ml = np.zeros(len(P))
        for l in range(NL):
            a, b = replay_layer(l, Cm[l], Cl[l], arm, st)
            mm += a; ml += b
        k = KEPT[list(H)].sum()
        bm, bl = mm[list(H)].sum() / k, ml[list(H)].sum() / k
        mb = (bm * MR + bl * LR) / 1e6
        say(f"  {arm:9s}: base {bm:5.2f} + sidecar {bl:5.2f} misses per kept token = {mb:5.2f} MB per kept token"
            f" (reads {bm + bl:5.2f}); dead-LSB share of the sidecar {st['dead_num'] / max(st['dead_den'], 1):.3f};"
            f" purged {st['purged']}")
        if arm == "base":
            n = max(st["l_n"], 1)
            say(f"    sidecar misses: {st['l_n']} counted; base resident before the call ('upgrade') {st['l_upgrade'] / n:.3f};"
                f" expert at ranks 2-4 in the previous check {st['l_climb'] / n:.3f}; anywhere in the previous check {st['l_prev_any'] / n:.3f}")
            if C == 256:
                HALF = dict(st["half"])
            BASE = (bm, bl, mb)
        else:
            say(f"    vs base: sidecar misses {100 * (bl / BASE[1] - 1):+.1f} %, SSD MB per kept token {100 * (mb / BASE[2] - 1):+.1f} %")

# ---------------------------------------------------------------- 3. HALF-check model (baseline state at C 256)
say("\n== 3. HALF-check side model (cap45 C 256 baseline state; checks as captured = 16 rows, so the tail is overstated vs t 0.3) ==")
nchk = sum(len(P[p][1]) for p in H); kept_tot = KEPT[list(H)].sum()
say(f"  counted checks {nchk}, kept {kept_tot:.0f} ({kept_tot / nchk:.2f} per check); accepted rows (kept - 1) >= s:")
for s in (6, 8, 10):
    acc = HALF.get((s, "acc"), [0, 0, 0]); rej = HALF.get((s, "rej"), [0, 0, 0])
    na, nr = acc[2], rej[2]
    mb_rej = (rej[0] * MR + rej[1] * LR) / 1e6 / max(nr, 1)      # exclusive tail bytes per rejecting check
    mb_acc = (acc[0] * MR + acc[1] * LR) / 1e6 / max(na, 1)
    say(f"  s {s:2d}: accept {na} / reject {nr} checks (P(accept >= s) {na / max(na + nr, 1):.2f}); exclusive tail misses per check:"
        f" reject {mb_rej:6.1f} MB, accept {mb_acc:6.1f} MB")
    for dense, rowt in ((34.0, 3.0), (17.0, 2.0)):
        p_acc = na / max(na + nr, 1)
        save = (1 - p_acc) * (mb_rej / 11.7 + (16 - s) * rowt) - p_acc * dense
        say(f"      model (SSD 11.7 GB/s, dense {dense:.0f} ms, row term {rowt:.1f} ms): {save:+.1f} ms per check")

# ---------------------------------------------------------------- 1. LOOKAHEAD: R-57's gate-time predictor on cap45's x
say("\n== 1. LOOKAHEAD: layer L+1's router on layer L's MoE input (R-57's ranked list), vs L+1's true top-10 union ==")
RW = np.load(os.path.join(CAP, "routers.npz"))
WR = {l: RW[f"w{l}"].astype(np.float32) for l in range(1, NL)}
bands = {"L0-11": range(0, 11), "L12-35": range(11, 35), "L36-46": range(35, 47)}
acc = defaultdict(lambda: np.zeros(6))              # per band: [recall@96 sum, prec@96 sum, recall@U sum, n, union sum, top-10-per-row recall]
nck = 0
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, Lz, rows, roff, xoff, ids, X = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
    checks = {}
    for n_, (s, l, r) in enumerate(zip(seq, Lz, rows)):
        c = checks.setdefault(int(s), {})
        if int(l) in c:
            continue                                # the commit replay
        c[int(l)] = (int(roff[n_]), int(xoff[n_]), int(r))
    for s, c in checks.items():
        if len(c) != NL:
            continue
        nck += 1
        for L in range(NL - 1):
            ro, xo, r = c[L]; ro1, xo1, r1 = c[L + 1]
            true = set(np.unique(ids[ro1:ro1 + r1]).tolist())
            lg = f32(X[xo:xo + r]) @ WR[L + 1].T                     # [rows, 512]
            top = np.argsort(-lg, 1)[:, :10]
            votes = Counter(top.ravel().tolist())
            score = defaultdict(float)
            for rr in range(r):
                for j, e in enumerate(top[rr]):
                    score[int(e)] += 10 - j
            ranked = sorted(votes, key=lambda e: (-votes[e], -score[e]))
            p96 = set(ranked[:96]); pU = set(ranked[:len(true)])
            band = next(b for b, rg in bands.items() if L in rg)
            a = acc[band]
            a[0] += len(p96 & true) / len(true); a[1] += len(p96 & true) / max(len(p96), 1)
            a[2] += len(pU & true) / len(true); a[3] += 1; a[4] += len(true)
            a[5] += np.mean([len(set(top[rr].tolist()) & set(ids[ro1 + rr].tolist())) / 10 for rr in range(min(r, r1))])
say(f"  checks {nck} (all six prompts, 16-row), one ranked unique list per check (votes over rows, then rank score)")
for band, a in acc.items():
    n = a[3]
    say(f"  {band:7s}: true union {a[4] / n:5.1f} experts; recall@96 {a[0] / n:.3f}, precision@96 {a[1] / n:.3f};"
        f" recall@|union| {a[2] / n:.3f}; per-row top-10 recall {a[5] / n:.3f}")
tot = sum(acc[b] for b in acc)
say(f"  ALL    : recall@96 {tot[0] / tot[3]:.3f}, precision@96 {tot[1] / tot[3]:.3f}, recall@|union| {tot[2] / tot[3]:.3f},"
    f" per-row {tot[5] / tot[3]:.3f}   [KILL line for C1-1: recall@96 >= 0.88 means nothing left to refine]")

# ---------------------------------------------------------------- 4. TOKEN table on the DAY9 capture (side check)
say("\n== 4. TOKEN-keyed early-layer predictability (DAY9 capture, accepted path, leave-one-window-out within domain) ==")
base = json.load(open(os.path.join(DMAP, "baseline.json")))
W = []
for f in sorted(os.listdir(os.path.join(DMAP, "routing"))):
    if f not in base:
        continue
    z = np.load(os.path.join(DMAP, "routing", f))
    W.append((base[f]["dom"], base[f]["a"], z["ids"], z["tokens"]))
say(f"  windows {len(W)}: {Counter(w[0] for w in W)}")
res = defaultdict(lambda: np.zeros(5))             # per (dom, band): [rec@10, rec@16, rec@24, prev-pos rec, n]
for dom in ("code", "ifeval", "multi"):
    ws = [w for w in W if w[0] == dom]
    for l in range(NL):
        tot = defaultdict(lambda: np.zeros(E, np.int32)); own = []
        for (_, a, ids, tok) in ws:
            o = defaultdict(lambda: np.zeros(E, np.int32))
            for p_ in range(ids.shape[0]):
                o[int(tok[p_])][ids[p_, l]] += 1
            own.append(o)
            for k_, v_ in o.items():
                tot[k_] += v_
        for wi, (_, a, ids, tok) in enumerate(ws):
            for p_ in range(max(a, 1), ids.shape[0]):
                t_ = int(tok[p_]); cnt = tot[t_] - own[wi].get(t_, 0)
                true = set(ids[p_, l].tolist())
                band = "L0-3" if l < 4 else "L4-11" if l < 12 else "L12-35" if l < 36 else "L36-47"
                r = res[(dom, band)]
                if cnt.sum() > 0:
                    order = np.argsort(-cnt)
                    for j, k in enumerate((10, 16, 24)):
                        r[j] += len(set(order[:k].tolist()) & true) / 10
                r[3] += len(set(ids[p_ - 1, l].tolist()) & true) / 10
                r[4] += 1
for key in sorted(res):
    r = res[key]; n = max(r[4], 1)
    say(f"  {key[0]:6s} {key[1]:6s}: recall@10 {r[0] / n:.3f}, @16 {r[1] / n:.3f}, @24 {r[2] / n:.3f} | previous position's own top-10 {r[3] / n:.3f} (n {int(r[4])})")
say(f"\ndone in {time.time() - T0:.0f} s")
