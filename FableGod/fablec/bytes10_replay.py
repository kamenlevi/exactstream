"""BYTES10 floor replay (Fable C, mission C5, Wed 4 Nov 2026, 00:xx EET). CPU only, one core at nice 15.
CORRECTED Wed 4 Nov 06:5x: the 00:27 version's keep mask sliced the layer axis (keep[sure, :ks]); its outputs are kept as
bytes10_replay_bug.*. The fixed run reproduces NewIdeas' data/ledger/bytes10_replay_fix.out (same code, one line).
Same conventions as NewIdeas' ten_x_ledger.py (plain LRU per layer for the MSB base pool and the LSB sidecar pool,
the prompt's rows warm both, one session per domain carried across the map's windows in walk order, live-cut blocks with
the wrong-draft stand-in row p + 97, kept = min(k, fm - A) + 1), so its 45 GB EXACT row should land near the ledger's.
What it adds:
  OPT    Belady's bound per pool line (the farthest next use is evicted; a block's own experts are pinned while it runs,
         the pool is trimmed to C when the block ends): no demand policy, however clever, reads fewer bytes than this.
  PIN    the cross-request hot set: the top C/2 experts of the layer by the domain's popularity (leave-one-window-out,
         share of the other windows' EXACT live blocks whose union holds the expert) are pinned, the rest is LRU.
  RCP    rank-class pools: experts last needed at ranks 0-1 (the drafted rows' working set) live in an LRU of 0.7 C, the
         rest (ranks 2-9: the exact rows' transient tail) in an LRU of 0.3 C; a rank 0-1 need promotes, A's evictee is
         demoted to B. The bonus and unsure rows' tails then cannot flush the sure rows' working set.
  CMP    the compulsory floor: every distinct expert read once per domain session (infinite pool), and once per request
         (the pool reset at each window, prompt rows warm it = prefill reuse) and once per request without that reuse.
  MIX    one session with the three domains' windows interleaved (code, ifeval, multi, code, ...) sharing the pools: the
         W1B-like mixed protocol; per-domain bytes.
  rules  EXACT; Q1 / QOM at cf 0.5 and cf 0 (sure rows load ranks 0-1 only; QOM's reads equal Q1's by construction) with
         and without 3-bit sure rows (no LSB read); sure rows at top-1 + t3 (the oea_k1 / P1-OEA family); HYB 0.9 u8 + t3;
         HYB-COLD 4/0.05 + t3 (unsure rows drop ranks >= 4 whose expert is cold in the domain's leave-one-window-out
         table); T3ALL (ranks 0-1 at 3 bits on every row: no sidecar; its slots go to the base pool);
         '+bonus' = the bonus row (the block's last row, exact today) gated too: 'next' gives it the drafter's cf of the
         next segment's first draft (a proxy for a one-token-ahead draft), 'all' treats it as sure (the bound).
  kept   synthetic checks of k consecutive answer rows (+ w stand-ins) with the drafter's OWN cf on each drafted row
         (the HPD-0 walk's cf, NaN = never drafted = unsure), beside the ledger's cf 1.0 (a perfect drafter).
Pool lines (the ledger's): 45 GB = laysz45_256 per-layer sizes; 24 GB = C_m 147 + C_l 46; 12 GB = C_m 51 + C_l 32;
12 GB T3ALL = C_m 51 + 32 x 770048 / 2162688 = 62, no sidecar.
Usage (BigMoE venv): nice -n 15 python bytes10_replay.py [quick] -> bytes10_replay.out, bytes10_replay.json"""
import bisect, heapq, json, os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map"))
import dm_gate  # noqa: E402  (load() reads out/gate_dfl2.json only)
OUT = dm_gate.OUT
MR, LR = 2162688, 770048
SZ = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))
POOLS = {"45": (SZ["C_m"], SZ["C_l"]), "24": ([147] * 48, [46] * 48), "12": ([51] * 48, [32] * 48),
         "12t": ([51 + int(round(32 * LR / MR))] * 48, [0] * 48)}
LIVE_T = 0.3
QUICK = "quick" in sys.argv
INF = 10 ** 9
RCP_A = 0.7


def live_segments(k, m, A, dc):
    """dm_q1u_masks / dm_oea_masks' live cut (t 0.3) of one walk check, copied."""
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd)); return segs
        segs.append((L, L, a0, dd[:L]))
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]


def windows():
    base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load(); out = {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"].astype(np.int32)
            out.setdefault(dom, []).append(dict(nm=nm, ids=ids, a=int(base[nm]["a"]), g=np.nan_to_num(G[nm], nan=-1.0),
                                                checks=Z[f"{i}_checks"].tolist(), dcf=Z[f"{i}_dcf"].tolist()))
    return out


def live_blocks(w, bonus=None):
    """[(rows, cfs, kept)]; cf -1 = the bonus row / an undrafted row (always 10 ranks, exact).
    bonus 'next': the block's last row (cf -1) takes the next segment's first draft cf; 'all': 1.0"""
    ids, g = w["ids"], w["g"]; N = len(ids); bl = []
    for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
        for (k, m, A, dc) in live_segments(k0_, m0_, A0_, dc0_):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            if A > N - 1: continue
            rows, cfs = [], []
            for p in range(A, last + 1):
                if p <= fm: rows.append(p); cfs.append(-1.0 if k == m and p == A + m else g[p])
                else: rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
            if rows: bl.append((rows, cfs, min(k, fm - A) + 1, dc[0] if dc else -1.0))
    out = []
    for i, (rows, cfs, kp, dc0) in enumerate(bl):
        cfs = list(cfs)
        if bonus and cfs[-1] < 0:
            cfs[-1] = 1.0 if bonus == "all" else (bl[i + 1][3] if i + 1 < len(bl) else -1.0)
        out.append((rows, cfs, kp))
    return out


def synth_blocks(w, k, wst, cfmode):
    """k consecutive answer rows (the last = the bonus row, cf -1) + wst wrong-draft stand-ins; drafted rows get cf 1.0
    ('one': the ledger's perfect drafter) or the walk's own cf for that row ('walk': today's calibration at k)."""
    ids, g = w["ids"], w["g"]; N = len(ids); bl = []; acc = 0.0
    for s in range(w["a"], N, k):
        real = list(range(s, min(s + k, N)))
        acc += wst; nw = int(acc); acc -= nw
        rows = real + [(p + 97) % N for p in range(s + len(real), s + len(real) + nw)]
        cfs = [1.0 if cfmode == "one" else g[p] for p in real[:-1]] + [-1.0] + [1.0 if cfmode == "one" else 0.9] * nw
        bl.append((rows, cfs, len(real)))
    return bl


# rule -> (thr, ks, ku, t3, cold (r, pth), t3all, bonus): sure rows (cf >= thr) load ks ranks; the others ku; t3: sure rows
# read no LSB; bonus: None / 'next' / 'all' (see live_blocks)
RULES = {"EXACT": (None, 10, 10, False, None, False, None),
         "Q1 0.5": (0.5, 2, 10, False, None, False, None), "Q1 0.5 t3": (0.5, 2, 10, True, None, False, None),
         "Q1 0 t3": (0.0, 2, 10, True, None, False, None), "Q1 0 k1 t3": (0.0, 1, 10, True, None, False, None),
         "HYB 0.9 u8 t3": (0.9, 2, 8, True, None, False, None), "HYB 0 u8 t3": (0.0, 2, 8, True, None, False, None),
         "COLD 4/0.05 t3": (0.9, 2, 10, True, (4, 0.05), False, None), "COLD 0 4/0.05 t3": (0.0, 2, 10, True, (4, 0.05), False, None),
         "T3ALL": (None, 10, 10, False, None, True, None), "Q1 0 T3ALL": (0.0, 2, 10, True, None, True, None),
         "Q1 0.9 t3 +bonus next": (0.9, 2, 10, True, None, False, "next"),
         "Q1 0 t3 +bonus next": (0.0, 2, 10, True, None, False, "next"),
         "Q1 0 t3 +bonus all": (0.0, 2, 10, True, None, False, "all"),
         "Q1 0 k1 t3 +bonus all": (0.0, 1, 10, True, None, False, "all")}
LIVE_RULES = list(RULES) if not QUICK else ["EXACT", "Q1 0 t3", "Q1 0 t3 +bonus next"]
OPT_RULES = ("EXACT", "Q1 0 t3", "Q1 0.5 t3", "Q1 0 k1 t3", "Q1 0 t3 +bonus next", "Q1 0 t3 +bonus all", "T3ALL", "Q1 0 T3ALL")


def popularity(W, dom, skip):
    """leave-one-window-out: share of the domain's other windows' EXACT live blocks whose layer union holds the expert"""
    cnt = np.zeros((48, 512)); n = 0
    for w in W[dom]:
        if w["nm"] == skip: continue
        for rows, cfs, kp in live_blocks(w):
            n += 1
            for l in range(48): cnt[l, np.unique(w["ids"][rows, l])] += 1
    return cnt / max(n, 1)


def block_sets(w, blocks, rule, P):
    """per block: (Ub [48 lists], Ul [48 lists], Ua [48 sets: experts needed at ranks 0-1], kept, nrows)"""
    thr, ks, ku, t3, cold, t3all, _ = RULES[rule]; ids = w["ids"]; seq = []
    for rows, cfs, kp in blocks:
        sure = np.array([thr is not None and c >= thr for c in cfs])
        E = ids[rows]                                                   # [r, 48, 10]
        keep = np.zeros(E.shape, bool); keep[sure, :, :ks] = True; keep[~sure, :, :ku] = True   # FIX Wed 06:5x (found by NewIdeas 03:4x): rank axis, not layers; the 00:27 run sliced layers 0-9
        if cold is not None:
            r, pth = cold
            for l in range(48):
                keep[~sure, l, r:] &= P[l][E[~sure, l, r:]] >= pth
        lsb = np.zeros(E.shape, bool); lsb[:, :, :2] = True
        if t3all: lsb[:] = False
        elif t3: lsb[sure] = False
        Ub = [np.unique(E[:, l][keep[:, l]]).tolist() for l in range(48)]
        Ul = [np.unique(E[:, l][lsb[:, l]]).tolist() for l in range(48)] if not t3all else [[]] * 48
        Ua = [set(np.unique(E[:, l, :2][keep[:, l, :2]]).tolist()) for l in range(48)]
        seq.append((Ub, Ul, Ua, kp, len(rows)))
    return seq


def warm_sets(w):
    return [(np.unique(w["ids"][:w["a"], l]).tolist(), np.unique(w["ids"][:w["a"], l, :2]).tolist(),
             set(np.unique(w["ids"][:w["a"], l, :2]).tolist())) for l in range(48)]


def lru_touch(c, e, C):
    if e in c: c.move_to_end(e); return 0
    c[e] = 1
    if len(c) > C: c.popitem(last=False)
    return 1


class RCP:
    """rank-class pools: A (ranks 0-1 needs, RCP_A x C) and B (the rest); promotion on a rank 0-1 need; demotion of A's
    evictee to B"""
    def __init__(self, C):
        self.A, self.B, self.Ca, self.Cb = OrderedDict(), OrderedDict(), max(int(round(RCP_A * C)), 1), max(C - int(round(RCP_A * C)), 1)

    def touch(self, e, top):
        m = 0
        if e in self.A: self.A.move_to_end(e); return 0
        if e in self.B:
            if not top: self.B.move_to_end(e); return 0
            del self.B[e]
        else: m = 1
        if top:
            self.A[e] = 1
            if len(self.A) > self.Ca:
                d, _ = self.A.popitem(last=False); self.B[d] = 1
                if len(self.B) > self.Cb: self.B.popitem(last=False)
        else:
            self.B[e] = 1
            if len(self.B) > self.Cb: self.B.popitem(last=False)
        return m


class Belady:
    def __init__(self, C, uses):                 # uses: e -> sorted block indices
        self.C, self.uses, self.res, self.h, self.hold, self.curseen = C, uses, {}, [], [], set()

    def nu(self, e, b):
        u = self.uses.get(e)
        if not u: return INF
        i = bisect.bisect_right(u, b)
        return u[i] if i < len(u) else INF

    def touch(self, e, b, cur):
        m = 0 if e in self.res else 1
        self.res[e] = self.nu(e, b); heapq.heappush(self.h, (-self.res[e], e))
        if e in cur: self.curseen.add(e)
        while len(self.res) > self.C and len(self.res) > len(self.curseen) and self.h:
            nu, x = heapq.heappop(self.h)
            if x not in self.res or self.res[x] != -nu: continue          # stale entry
            if x in cur: self.hold.append((nu, x))                         # pinned while its block runs
            else: del self.res[x]
        return m

    def end_block(self):
        for it in self.hold: heapq.heappush(self.h, it)
        self.hold = []; self.curseen = set()
        while len(self.res) > self.C and self.h:
            nu, x = heapq.heappop(self.h)
            if x in self.res and self.res[x] == -nu: del self.res[x]


def simulate(seqs, Cm, Cl, policy, pop=None, doms=None):
    """seqs: the session's (warm, blocks) per window in order; returns sums (per domain when doms is given)"""
    keys = sorted(set(doms)) if doms else ["all"]
    R = {k: dict(mb=0, ml=0, ub=0, ul=0, kept=0, nrow=0, nchk=0, cmb=0, cml=0) for k in keys}
    if policy == "OPT":
        usesb = [dict() for _ in range(48)]; usesl = [dict() for _ in range(48)]; bi = 0
        for warm, seq in seqs:
            for l in range(48):
                for e in warm[l][0]: usesb[l].setdefault(e, []).append(bi)
                for e in warm[l][1]: usesl[l].setdefault(e, []).append(bi)
            bi += 1
            for Ub, Ul, Ua, kp, nr in seq:
                for l in range(48):
                    for e in Ub[l]: usesb[l].setdefault(e, []).append(bi)
                    for e in Ul[l]: usesl[l].setdefault(e, []).append(bi)
                bi += 1
        bp = [Belady(Cm[l], usesb[l]) for l in range(48)]; lp = [Belady(max(Cl[l], 1), usesl[l]) for l in range(48)]
    elif policy == "RCP":
        bp = [RCP(Cm[l]) for l in range(48)]; lp = [OrderedDict() for _ in range(48)]
    else:
        bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    pinb = [set() for _ in range(48)]
    if policy == "PIN":
        for l in range(48):
            top = np.argsort(-pop[l])[:Cm[l] // 2]; pinb[l] = set(int(x) for x in top)
    seen_b = [set() for _ in range(48)]; seen_l = [set() for _ in range(48)]
    bi = 0
    for wi, (warm, seq) in enumerate(seqs):
        r = R[doms[wi]] if doms else R["all"]
        for l in range(48):                                   # the prompt's prefill warms the pools (its reads are not
            for e in warm[l][0]:                              # decode bytes)
                if policy == "OPT": bp[l].touch(e, bi, set())
                elif policy == "RCP": bp[l].touch(e, e in warm[l][2])
                elif e not in pinb[l]: lru_touch(bp[l], e, Cm[l] - len(pinb[l]))
                seen_b[l].add(e)
            for e in warm[l][1]:
                if policy == "OPT": lp[l].touch(e, bi, set())
                else: lru_touch(lp[l], e, max(Cl[l], 1))
                seen_l[l].add(e)
            if policy == "OPT": bp[l].end_block(); lp[l].end_block()
        bi += 1
        for Ub, Ul, Ua, kp, nr in seq:
            for l in range(48):
                if policy == "OPT":
                    cur = set(Ub[l]); curl = set(Ul[l])
                    for e in Ub[l]: r["mb"] += bp[l].touch(e, bi, cur)
                    for e in Ul[l]: r["ml"] += lp[l].touch(e, bi, curl)
                    bp[l].end_block(); lp[l].end_block()
                else:
                    for e in Ub[l]:
                        if policy == "RCP": r["mb"] += bp[l].touch(e, e in Ua[l])
                        elif e not in pinb[l]: r["mb"] += lru_touch(bp[l], e, Cm[l] - len(pinb[l]))
                    for e in Ul[l]: r["ml"] += lru_touch(lp[l], e, max(Cl[l], 1))
                for e in Ub[l]:
                    if e not in seen_b[l]: r["cmb"] += 1; seen_b[l].add(e)
                for e in Ul[l]:
                    if e not in seen_l[l]: r["cml"] += 1; seen_l[l].add(e)
                r["ub"] += len(Ub[l]); r["ul"] += len(Ul[l])
            r["kept"] += kp; r["nrow"] += nr; r["nchk"] += 1; bi += 1
    return R if doms else R["all"]


def compulsory_per_request(seqs):
    """every distinct expert once per request: with the prompt's experts resident (prefill reuse) and without"""
    cb = cl = cb0 = cl0 = kept = 0
    for warm, seq in seqs:
        sb = [set(warm[l][0]) for l in range(48)]; sl = [set(warm[l][1]) for l in range(48)]
        sb0 = [set() for _ in range(48)]; sl0 = [set() for _ in range(48)]
        for Ub, Ul, Ua, kp, nr in seq:
            for l in range(48):
                for e in Ub[l]:
                    if e not in sb[l]: cb += 1; sb[l].add(e)
                    if e not in sb0[l]: cb0 += 1; sb0[l].add(e)
                for e in Ul[l]:
                    if e not in sl[l]: cl += 1; sl[l].add(e)
                    if e not in sl0[l]: cl0 += 1; sl0[l].add(e)
            kept += kp
    return (cb * MR + cl * LR) / kept / 1e6, (cb0 * MR + cl0 * LR) / kept / 1e6


def mbk(r): return (r["mb"] * MR + r["ml"] * LR) / max(r["kept"], 1) / 1e6


def main():
    t0 = time.time(); W = windows(); lines = []; res = {}
    def say(s): print(s, flush=True); lines.append(s)
    say(f"BYTES10 replay: windows {dict((d, len(v)) for d, v in W.items())}, pools {dict((p, (int(np.mean(c[0])), int(np.mean(c[1])))) for p, c in POOLS.items())}; {time.time() - t0:.0f} s")
    POP = {d: {w["nm"]: popularity(W, d, w["nm"]) for w in W[d]} for d in W}
    say(f"popularity tables built ({time.time() - t0:.0f} s)")
    say("")
    say("A. live checks: SSD MB per kept token (MSB part in brackets) by policy | compulsory floors | union GB per kept | kept, rows per check")
    say("| domain | pool | rule | LRU | OPT | PIN | RCP | CMP session | CMP request (prefill reuse / none) | union GB/kept | kept | rows |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    SEQ = {}
    for dom in W:
        for rule in LIVE_RULES:
            bm = RULES[rule][6]
            seqs = [(warm_sets(w), block_sets(w, live_blocks(w, bm), rule, POP[dom][w["nm"]])) for w in W[dom]]
            SEQ[(dom, rule)] = seqs
            cr = compulsory_per_request(seqs)
            for pool in POOLS:
                t3all = RULES[rule][5]
                if (pool == "12t") != t3all: continue
                Cm, Cl = POOLS[pool]
                cells = {}
                pols = ["LRU"]
                if not QUICK and pool != "45" or rule in ("EXACT", "Q1 0 t3"): pols += ["OPT", "PIN", "RCP"]
                if rule not in OPT_RULES: pols = [p for p in pols if p != "OPT"]
                for pol in pols:
                    r = simulate(seqs, Cm, Cl, pol, pop=np.mean([POP[dom][w["nm"]] for w in W[dom]], 0) if pol == "PIN" else None)
                    cells[pol] = mbk(r); cells[pol + "_msb"] = r["mb"] * MR / r["kept"] / 1e6
                    if pol == "LRU":
                        cmp_s = (r["cmb"] * MR + r["cml"] * LR) / r["kept"] / 1e6
                        ug = (r["ub"] * MR + r["ul"] * LR) / r["kept"] / 1e9; kp = r["kept"] / r["nchk"]; rp = r["nrow"] / r["nchk"]
                res[f"{dom}|{pool}|{rule}|live"] = dict(cells, cmp_session=cmp_s, cmp_req=cr[0], cmp_req0=cr[1], union_gb=ug, kept=kp, rows=rp)
                f = lambda p: f"{cells[p]:.1f} ({cells[p + '_msb']:.1f})" if p in cells else "-"
                say(f"| {dom} | {pool} | {rule} | {f('LRU')} | {f('OPT')} | {f('PIN')} | {f('RCP')} | {cmp_s:.1f} | {cr[0]:.1f} / {cr[1]:.1f} | {ug:.2f} | {kp:.2f} | {rp:.2f} |")
        say(f"  ({time.time() - t0:.0f} s)")
    say("")
    say("A2. MIX session (the three domains' windows interleaved, one shared pool per layer; LRU): SSD MB per kept token per domain")
    say("| rule | pool | code | ifeval | multi |")
    say("|---|---|---|---|---|")
    for rule in (["EXACT", "Q1 0 t3"] if QUICK else ["EXACT", "Q1 0.5 t3", "Q1 0 t3", "Q1 0 t3 +bonus next"]):
        order = []; doms = []
        n = max(len(W[d]) for d in W)
        for i in range(n):
            for d in ("code", "ifeval", "multi"):
                if i < len(W[d]): order.append(SEQ[(d, rule)][i]); doms.append(d)
        for pool in ("45", "24", "12"):
            Cm, Cl = POOLS[pool]
            R = simulate(order, Cm, Cl, "LRU", doms=doms)
            for d in R: res[f"{d}|{pool}|{rule}|mix"] = dict(LRU=mbk(R[d]))
            say(f"| {rule} | {pool} | " + " | ".join(f"{mbk(R[d]):.1f}" for d in ("code", "ifeval", "multi")) + " |")
    say(f"  ({time.time() - t0:.0f} s)")
    say("")
    say("B. kept per check (synthetic k consecutive kept rows + w stand-ins; LRU): SSD MB per kept token; 'walk' = the drafter's own cf per row, 'one' = cf 1.0")
    cases = []
    for k in ([4, 8, 12, 14, 16] if not QUICK else [8, 14]):
        for wst in (0.0, 2.0): cases.append((f"k{k}+{wst:.0f}", (k, wst)))
    say("| domain | pool | rule | cf | " + " | ".join(c[0] for c in cases) + " |")
    say("|---|---|---|---|" + "---|" * len(cases))
    krules = [("EXACT", "one"), ("Q1 0 t3", "walk"), ("Q1 0 t3", "one"), ("Q1 0.5 t3", "walk"), ("HYB 0.9 u8 t3", "walk")]
    if QUICK: krules = krules[:2]
    for dom in W:
        for rule, cfm in krules:
            for pool in ("45", "24", "12"):
                Cm, Cl = POOLS[pool]; vals = []
                for nm, (k, wst) in cases:
                    seqs = [(warm_sets(w), block_sets(w, synth_blocks(w, k, wst, cfm), rule, POP[dom][w["nm"]])) for w in W[dom]]
                    r = simulate(seqs, Cm, Cl, "LRU"); v = mbk(r); vals.append(v)
                    res[f"{dom}|{pool}|{rule}|{cfm}|{nm}"] = dict(LRU=v, kept=r["kept"] / r["nchk"], rows=r["nrow"] / r["nchk"],
                                                               union_gb=(r["ub"] * MR + r["ul"] * LR) / r["kept"] / 1e9)
                say(f"| {dom} | {pool} | {rule} | {cfm} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        say(f"  ({time.time() - t0:.0f} s)")
    say("")
    say("C. union per block vs rows (EXACT, no cache): mean distinct experts per layer for blocks of r rows, and the share of the block's pairs that are singletons")
    for dom in W:
        out = []
        for r in (1, 2, 4, 8, 16, 32, 64):
            U = []; S = []
            for w in W[dom]:
                ids = w["ids"]
                for s in range(w["a"], len(ids) - r + 1, r):
                    E = ids[s:s + r]
                    for l in (4, 12, 24, 36, 44):
                        u, c = np.unique(E[:, l], return_counts=True); U.append(len(u)); S.append((c == 1).sum() / (r * 10))
            out.append(f"r {r}: {np.mean(U):.1f} ({100 * np.mean(S):.0f} % singletons)")
        say(f"  {dom}: " + "; ".join(out))
    say("")
    say("D. per-request working sets (means over windows): distinct experts per layer in the prompt (ranks 0-9), in the answer at ranks 0-9 / 0-1, "
        "answer experts not in the prompt (0-9 / 0-1), answer tokens; the share of a live block's expert pairs on its bonus row at cf 0 (Q1 0: 2 ranks per drafted row, 10 on the bonus)")
    for dom in W:
        acc = []
        for w in W[dom]:
            ids, a = w["ids"], w["a"]; N = len(ids)
            dp = np.mean([len(np.unique(ids[:a, l])) for l in range(48)])
            da = np.mean([len(np.unique(ids[a:, l])) for l in range(48)]); da2 = np.mean([len(np.unique(ids[a:, l, :2])) for l in range(48)])
            nn = np.mean([len(set(np.unique(ids[a:, l]).tolist()) - set(np.unique(ids[:a, l]).tolist())) for l in range(48)])
            nn2 = np.mean([len(set(np.unique(ids[a:, l, :2]).tolist()) - set(np.unique(ids[:a, l]).tolist())) for l in range(48)])
            bl = live_blocks(w); nb = sum(1 for rows, cfs, kp in bl if cfs[-1] < 0); nd = sum(len(rows) - 1 for rows, cfs, kp in bl)
            acc.append((dp, da, da2, nn, nn2, N - a, 10 * nb / (10 * nb + 2 * nd)))
        m = np.mean(acc, 0)
        say(f"  {dom}: prompt {m[0]:.0f}; answer {m[1]:.0f} / {m[2]:.0f}; new vs prompt {m[3]:.0f} / {m[4]:.0f}; answer tokens {m[5]:.0f}; bonus-row share of pairs at cf 0: {100 * m[6]:.0f} %")
    say(f"done {time.time() - t0:.0f} s")
    sfx = "" if not QUICK else "_quick"
    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, f"bytes10_replay{sfx}.out"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(here, f"bytes10_replay{sfx}.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
