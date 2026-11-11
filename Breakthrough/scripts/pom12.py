"""R-49 POM12 (NewIdeas Thu 18:3x; NON-exact; a CPU price screen only): MoonShots' M-03 POM (precision on miss) and
M-07 LP1 priced on the 12 GB line, with BigMoE's real file layouts (records3p 903560e for the 2-plane path, records3m).
Engine rules (bt_b12 B12LL, fold-exact): every row's top-10 is used and read on a miss (ranks 0-1 at 4 bits = MSB + LSB
parts, ranks 2-9 at 3 bits = MSB); one call per layer per verify pass with every row; base pool C_m 51, LSB pool C_l 20
per layer. Eviction: v3m / v3n are LRU (every needed expert touched per call, victim = the least recent one this call does
not need); BigMoE's engine is strict LFU (decay 0.99, pom24's rule) - both shown, each against its own baseline.
Bytes per read: full MSB miss 2,048,000 (records3m), 2-plane miss 1,425,408 (records3p seg1), plane-3 upgrade 704,512
(seg2), LSB miss 720,896 (records3m). POM R2: a miss whose best rank in the pass is >= R2 reads seg1 and stays at 2 planes
until a use at rank < R2 reads seg2 (the slot size is unchanged). LP1: an LSB miss whose best rank is 1 reads nothing and
inserts nothing (that use runs at MSB-3); rank-0 misses still read.
Quality (MoonShots' formula, M-03 / M-07): f_e = share of the rank >= R2 gate energy (top-10-normalised weight^2) on kept
rows served at 2 planes, f_1 = share of the rank-1 energy on kept rows served without the LSB; predicted KLD per part =
v3 + f_e x gap(R2) + f_1 x (plan4333333333 - v3), gap(R2) for R2 2..6 interpolated by mean w^2 per rank between the
measured 4433332222 (R2 6) and 4422222222 (R2 2) (M-03's measured run landed under its prediction).
Traces: P12_TRACES=r49 (default): the 12 GB decode's own verify passes from scripts/r49_capture.py (data/r49/*.npz: p8k +
agent rows + a math row; every row of every pass, the kept rows = the tokens emitted), first P12_WARM (10) passes of each
prompt warm the pools; =expA: data/expA/traces125 (800 AR tokens, synthetic P12_R-row passes keeping P12_A; 0-399 warm).
PASS (written first, NewIdeas): some variant -15 % or better SSD bytes / token vs fold-exact + M-01 (R2 10, no LP1) with
predicted KLD inside 1.10 x ar4 on all four parts. KILL: under -8 % for every variant inside the gate.
usage: [BigMoE venv] taskpolicy -b python pom12.py -> stdout + data/r49/pom12_<traces>_<policy>.txt"""
import glob, os, sys, time
from collections import OrderedDict
import numpy as np

FULL, P12, P3, LSB = 2048000, 1425408, 704512, 720896
CM, CL = int(os.environ.get("P12_CM", "51")), int(os.environ.get("P12_CL", "20"))
# SSD busy model (NewIdeas 19:3x: judge by SSD busy time, not bytes; BigMoE b45 at 24 GB: POM -19 % bytes, busy -6.5 %):
# one burst per (pass, layer) call, the 12 GB engine has no prefetch: t = T0 (first read) + bytes / BW + reads x TOP +
# (2-plane reads + upgrades) x TRB / NTH (their CPU rebuild, BigMoE 0.084 ms per projection). BW, T0, TOP from MoonShots'
# ssd_qd (whole reads 14.0 GB/s at QD 4+; QD 1 adds ~0.2 ms; 9 pieces vs 1 read = ~4 us per extra read)
BW, T0, TOP = float(os.environ.get("P12_BW", "14.0e9")), float(os.environ.get("P12_T0", "0.2e-3")), float(os.environ.get("P12_TOP", "4e-6"))
TRB, NTH = float(os.environ.get("P12_TRB", "0.25e-3")), int(os.environ.get("P12_NTH", "6"))
TR = os.environ.get("P12_TRACES", "r49")
NL, DEC = 48, 0.99
v3 = np.array([0.00172, 0.00275, 0.01367, 0.01097])
tgt = np.array([0.00253, 0.00449, 0.02041, 0.02055])
p4333 = np.array([0.00245, 0.00399, 0.01799, 0.01553])
g6, g2 = np.array([0.00095, 0.00158, 0.00786, 0.00765]), np.array([0.00497, 0.00830, 0.03635, 0.03853])
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def load():
    """-> {name: [pass dicts]}: ids [S, 48, 10] int, w [S, 48, 10] (top-10-normalised), kept S_k rows, measured flag"""
    out = {}
    if TR == "expA":
        R, A = int(os.environ.get("P12_R", "4")), int(os.environ.get("P12_A", "3"))
        for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz"))):
            d = np.load(f)
            ids = np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.int32)
            g = np.stack([d[f"G{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.float64)
            g = g / g.sum(-1, keepdims=True)
            out[os.path.basename(f).split("_")[0]] = [dict(ids=ids[s:s + R], w=g[s:s + R], kept=A, meas=400 <= s < 800)
                                                      for s in range(0, ids.shape[0] - R + 1, A)]
        return out
    WARM = int(os.environ.get("P12_WARM", "10"))
    for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/r49/*.npz"))):
        d = np.load(f)
        L, S, E = d["calls_L"].astype(int), d["calls_S"].astype(int), d["calls_E"].astype(int)
        off = np.concatenate([[0], np.cumsum(S)])
        ids20, p20 = d["ids"].astype(np.int32), d["probs"].astype(np.float64)
        passes, i, bad = [], 0, 0
        while i + NL <= len(L):
            if not (L[i:i + NL] == np.arange(NL)).all() or (S[i:i + NL] != S[i]).any():
                bad += 1; i += 1; continue                   # not a full pass (a replay / partial call): skipped
            s = S[i]
            ids = np.stack([ids20[off[i + l]:off[i + l] + s, :10] for l in range(NL)], 1)
            w = np.stack([p20[off[i + l]:off[i + l] + s, :10] for l in range(NL)], 1)
            passes.append(dict(ids=ids, w=w / w.sum(-1, keepdims=True), E=E[i], meas=len(passes) >= WARM))
            i += NL
        n_emit = int(d["tokens"].shape[0])
        for a, b in zip(passes, passes[1:] + [None]):
            a["kept"] = min(a["ids"].shape[0], (b["E"] if b else n_emit) - a["E"])
        name = os.path.basename(f)[:-4]
        out[name] = passes
        say(f"  {name}: {len(passes)} passes ({bad} unaligned calls skipped), rows / pass "
            f"{np.mean([p['ids'].shape[0] for p in passes]):.2f}, kept / pass {np.mean([p['kept'] for p in passes]):.2f}, "
            f"{n_emit} tokens")
    return out


def run(passes, R2, lp1, policy):
    """one prompt, all layers -> (MSB bytes, LSB bytes, tokens) over measured passes, quality sums [m red, m, e red, e, e1 no-lsb, e1]"""
    mb = lb = ntok = 0
    busy = 0.0
    cnt = np.zeros(4)                                        # full, 2-plane, upgrade, LSB reads (measured)
    q = np.zeros(6)
    for l in range(NL):
        res, lsb = OrderedDict(), OrderedDict()              # expert -> planes (2 / 3); LSB residents
        score, lscore, w8 = {}, {}, 1.0
        for P in passes:
            ids, wt = P["ids"][:, l], P["w"][:, l]
            best = {}
            for k in range(10):                              # rank-major, as the ext's flag rows
                for r in range(ids.shape[0]):
                    e = int(ids[r, k])
                    if e not in best:
                        best[e] = k
            w8 /= DEC
            b = 0
            nf = n2 = nu = 0
            for e, k in best.items():
                need3 = k < R2
                if policy == "lfu":
                    score[e] = score.get(e, 0.0) + w8
                if e in res:
                    if policy == "lru":
                        res.move_to_end(e)
                    if need3 and res[e] == 2:
                        b += P3; res[e] = 3; nu += 1
                    continue
                if len(res) >= CM:
                    if policy == "lru":
                        vic = next((v for v in res if v not in best), None)
                    else:
                        cand = [v for v in res if v not in best]
                        vic = min(cand, key=lambda v: score[v]) if cand else None
                    if vic is not None:
                        del res[vic]
                b += FULL if need3 else P12
                nf += need3; n2 += not need3
                res[e] = 3 if need3 else 2
            if policy == "lfu" and len(res) > CM:            # strict: trim to C after the call (pom24)
                keep = set(sorted(res, key=lambda v: -score[v])[:CM])
                res = OrderedDict((e, res[e]) for e in res if e in keep)
            bl = 0
            nl = 0
            for e, k in best.items():
                if k >= 2:
                    continue
                if policy == "lfu":
                    lscore[e] = lscore.get(e, 0.0) + w8
                if e in lsb:
                    if policy == "lru":
                        lsb.move_to_end(e)
                    continue
                if lp1 and k == 1:                           # LP1: rank-1-only LSB miss: no read, no insert
                    continue
                if len(lsb) >= CL:
                    if policy == "lru":
                        vic = next((v for v in lsb if v not in best or best[v] >= 2), None)
                    else:
                        cand = [v for v in lsb if v not in best or best[v] >= 2]
                        vic = min(cand, key=lambda v: lscore[v]) if cand else None
                    if vic is not None:
                        del lsb[vic]
                bl += LSB; nl += 1
                lsb[e] = 1
            if policy == "lfu" and len(lsb) > CL:
                keep = set(sorted(lsb, key=lambda v: -lscore[v])[:CL])
                lsb = OrderedDict((e, 1) for e in lsb if e in keep)
            if not P["meas"]:
                continue
            mb += b; lb += bl
            nr = nf + n2 + nu + nl
            busy += (T0 if nr else 0.0) + (b + bl) / BW + nr * TOP + (n2 + nu) * TRB / NTH
            cnt += (nf, n2, nu, nl)
            if l == 0:
                ntok += P["kept"]
            for r in range(P["kept"]):
                for k in range(10):
                    m = wt[r, k]
                    if k >= R2:
                        red = res.get(int(ids[r, k]), 3) == 2
                        q[:4] += (m * red, m, m * m * red, m * m)
                    if k == 1:
                        q[4:] += (m * m * (int(ids[r, 1]) not in lsb), m * m)
    return mb, lb, ntok, q, busy, cnt


t0 = time.time()
say(f"R-49 POM12 price screen: traces {TR}, pools C_m {CM} / C_l {CL}; bytes full {FULL:,} / 2-plane {P12:,} / upgrade "
    f"{P3:,} / LSB {LSB:,}; busy model per call: T0 {T0 * 1e3:.2f} ms + bytes / {BW / 1e9:.1f} GB/s + {TOP * 1e6:.0f} us / read "
    f"+ 2-plane / upgrade CPU {TRB * 1e3:.2f} ms / {NTH} threads")
DATA = load()
G10 = np.concatenate([np.concatenate([P["w"][:P["kept"]].reshape(-1, 10) for P in ps]) for ps in DATA.values()])
w2 = np.mean(G10 ** 2, 0)
gap = {R: g6 + (g2 - g6) * w2[R:6].sum() / w2[2:6].sum() for R in (2, 3, 4, 5)}
gap[6], gap[10] = g6, np.zeros(4)
say(f"  mean w^2 by rank {np.round(w2, 4).tolist()}; gap R2 5 {np.round(gap[5], 5).tolist()}, R2 4 {np.round(gap[4], 5).tolist()}")
for policy in ("lru", "lfu"):
    base = None
    say(f"policy {policy} ({'v3m / v3n, the 12 GB engine' if policy == 'lru' else 'BigMoE engine rule'}):")
    for R2, lp1 in ((10, False), (10, True), (6, False), (5, False), (4, False), (6, True), (5, True), (4, True)):
        MB = LB = NT = 0
        BZ, CN = 0.0, np.zeros(4)
        Q = np.zeros(6)
        per = []
        for name, ps in DATA.items():
            mb, lb, nt, q, bz, cn = run(ps, R2, lp1, policy)
            MB += mb; LB += lb; NT += nt; Q += q; BZ += bz; CN += cn
            per.append(f"{name} {(mb + lb) / max(nt, 1) / 1e6:.0f}")
        tot = (MB + LB) / NT
        if base is None:
            base, bbase = tot, BZ / NT
        fe = Q[2] / Q[3] if Q[3] else 0.0
        f1 = Q[4] / Q[5] if lp1 and Q[5] else 0.0
        pred = v3 + fe * gap[R2] + f1 * (p4333 - v3)
        inside = (pred <= tgt).all()
        say(f"  R2 {R2:2d}{' + LP1' if lp1 else '      '}: busy {1e3 * BZ / NT:5.2f} ms / token ({100 * (BZ / NT / bbase - 1):+5.1f} %) "
            f"reads / token full {CN[0] / NT:4.1f} 2-plane {CN[1] / NT:4.1f} upgrade {CN[2] / NT:4.1f} LSB {CN[3] / NT:4.1f} | "
            f"{tot / 1e6:6.1f} MB / token ({100 * (tot / base - 1):+5.1f} %) [MSB "
            f"{MB / NT / 1e6:.1f} + LSB {LB / NT / 1e6:.1f}] | f_e {100 * fe:4.1f} % f_1 {100 * f1:4.1f} % | predicted KLD "
            + " / ".join(f"{p:.5f}" for p in pred) + (" INSIDE" if inside else " OVER on " + ", ".join(
                n for n, p, t in zip(("code", "gsm", "ifeval", "multi"), pred, tgt) if p > t))
            + f" | per prompt MB/tok: {', '.join(per)}")
say(f"({time.time() - t0:.0f} s)")
os.makedirs(os.path.expanduser("~/Breakthrough/data/r49"), exist_ok=True)
open(os.path.expanduser(f"~/Breakthrough/data/r49/pom12_{TR}_{CM}_{CL}.txt"), "w").write("\n".join(lines) + "\n")
