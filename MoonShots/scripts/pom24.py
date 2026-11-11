"""M-03 POM, precision on miss (MoonShots Wed 16:04; NON-exact, CPU): SSD bytes come only from misses, quality from every use.
A miss whose best rank in the check is >= R2 reads only the top two code planes (MSB-2; s2 = 2 s3, b2 = b3 + s3 / 2) and
stays resident at 2 planes until a use at rank < R2 reads its third plane; 3-plane residents are untouched.
(1) Plane split of the code rate from records3 (L6 / 24 / 42, 32 experts each): H(top-2 planes) + H(plane 3 | top-2) = H(m),
    scaled to W24-ec's measured 2.7231 b / weight.
(2) Strict LFU replay at the 24 GB stack (wall24 traces, C_m 159 / C_l 44, 16-row checks accepting 10, decay 0.99, pool
    trimmed to C after each check; RAM per slot unchanged = conservative): SSD bytes / token (11-bit scales, 16 KB reads;
    LSB pool as v3), and f = share of the rank >= R2 gate mass (and of its square, the energy) on ACCEPTED rows served at
    2 planes.
(3) Predicted per-part KLD = v3 + f_energy x the measured plan gap (kl125 screen: 4433332222 for R2 6, 4422222222 for R2 2;
    R2 4 = R2 6 + the 2-3 / 4-5 split of the extra gap by mean w^2 per rank).
PASS (written first): some R2 gives >= -10 % SSD bytes / token with predicted KLD inside 1.10 x ar4 on all four parts.
KILL: < -5 % at every R2 inside the gate.
usage: [BigMoE venv] taskpolicy -b python pom24.py -> stdout + data/pom24.txt"""
import fcntl, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def H(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


t0 = time.time()
# (1) plane split
D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
hm, h2 = [], []
for L in (6, 24, 42):
    ix = IX[str(L)]
    fd = os.open(D3 + ix["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    cnt = np.zeros(8)
    for e in sorted(np.random.default_rng(L).choice(512, 32, replace=False).tolist()):
        for p in range(3):
            t = np.frombuffer(os.pread(fd, 614400, e * ix["recsz"] + ix["msb_piece_off"][3 * p]), np.uint8).reshape(-1, 3).astype(np.uint32)
            w = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
            cnt += np.bincount(((w[:, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).ravel(), minlength=8)
    os.close(fd)
    hm.append(H(cnt)); h2.append(H(cnt.reshape(4, 2).sum(1)))
Hm, H2 = float(np.mean(hm)), float(np.mean(h2))
RATE, NW, SB = 2.7231, 4915200, 211248
blk = lambda n: -(-n // 16384) * 16384
P12 = blk(round(NW * RATE * H2 / Hm / 8) + SB)
P3 = blk(round(NW * RATE * (Hm - H2) / Hm / 8))
FULL, LSB = 1900544, 720896
say(f"(1) code planes: H(m) {Hm:.4f} b = H(top-2) {H2:.4f} + H(plane 3 | top-2) {Hm - H2:.4f}; at W24-ec's 2.7231 b: 2-plane read "
    f"{P12:,} B (incl. 11-bit scales), plane-3 read {P3:,} B, full MSB miss {FULL:,} B")

# (2) replay
G = {k: np.stack([d[f"G{l}"] for l in range(W.NL)], 1)[:, :, :10].astype(np.float64) for k, d in W.SETS.items()}
G = {k: g / g.sum(-1, keepdims=True) for k, g in G.items()}
CM, CL = int(os.environ.get("POM_CM", "159")), int(os.environ.get("POM_CL", "44"))
R, A, DEC = int(os.environ.get("POM_R", "16")), int(os.environ.get("POM_A", "10")), 0.99


def strict(seq, C):
    out, Rs, score, w = [], set(), {}, 1.0
    for N in seq:
        w /= DEC
        for e in N:
            score[e] = score.get(e, 0.0) + w
        m = [e for e in N if e not in Rs]
        out.append(len(m))
        for e in m:
            if len(Rs) >= C:
                vic = [v for v in Rs if v not in N]
                if vic:
                    Rs.discard(min(vic, key=lambda v: score[v]))
            Rs.add(e)
        if len(Rs) > C:
            Rs = set(sorted(Rs, key=lambda v: -score[v])[:C])
    return out


def pom(checks, gates, R2):
    """checks: per check (measured, [(best rank dict, rows ids [R, 10])]) for ONE layer -> (bytes list, red mass, mass, red e, e)"""
    res, score, w = {}, {}, 1.0
    byt, rm = [], np.zeros(4)
    for (meas, best, ids, g) in checks:
        w /= DEC
        for e in best:
            score[e] = score.get(e, 0.0) + w
        b = 0
        for e, k in best.items():
            need3 = k < R2
            if e in res:
                if need3 and res[e] == 2:
                    b += P3; res[e] = 3
            else:
                if len(res) >= CM:
                    vic = [v for v in res if v not in best]
                    if vic:
                        del res[min(vic, key=lambda v: score[v])]
                b += FULL if need3 else P12
                res[e] = 3 if need3 else 2
        if meas:
            byt.append(b)
            for r in range(A):
                for kk in range(R2, 10):
                    m = g[r, kk]
                    red = res[int(ids[r, kk])] == 2
                    rm += (m * red, m, m * m * red, m * m)
        else:
            byt.append(0)
        if len(res) > CM:
            keep = sorted(res, key=lambda v: -score[v])[:CM]
            res = {e: res[e] for e in keep}
    return byt, rm


per = {}
lsb_tok = {}
for k, ids in W.IDS.items():
    n = ids.shape[0]
    starts = list(range(0, n - R + 1, A))
    meas = [W.MEAS[0] <= s < W.MEAS[1] for s in starts]
    ntok = A * sum(meas)
    lb = 0
    lay_checks = []
    for l in range(W.NL):
        cl = []
        lsets = []
        for s, f in zip(starts, meas):
            rows = ids[s:s + R, l]
            best = {}
            for r in range(R):
                for kk, e in enumerate(rows[r]):
                    e = int(e)
                    if e not in best or kk < best[e]:
                        best[e] = kk
            cl.append((f, best, rows, G[k][s:s + R, l]))
            lsets.append(frozenset(e for e, kk in best.items() if kk < 2))
        lay_checks.append(cl)
        lb += sum(a for a, f in zip(strict(lsets, CL), meas) if f) * LSB
    lsb_tok[k] = lb / ntok
    per[k] = (lay_checks, ntok)
say(f"(2) traces: {', '.join(per)}; LSB bytes / token (v3 rule, C_l {CL}): " + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in lsb_tok.items()))
gap = {6: np.array([0.00095, 0.00158, 0.00786, 0.00765]), 2: np.array([0.00497, 0.00830, 0.03635, 0.03853])}
w2 = np.mean(np.concatenate([g.reshape(-1, 10) for g in G.values()]) ** 2, 0)
gap[4] = gap[6] + (gap[2] - gap[6]) * w2[4:6].sum() / w2[2:6].sum()
v3 = np.array([0.00172, 0.00275, 0.01367, 0.01097])
tgt = np.array([0.00253, 0.00449, 0.02041, 0.02055])
say(f"    mean w^2 by rank {np.round(w2, 4).tolist()}; plan gaps code / gsm / ifeval / multi: R2 6 {gap[6].tolist()}, "
    f"R2 4 (interp.) {np.round(gap[4], 5).tolist()}, R2 2 {gap[2].tolist()}")
base = None
for R2 in (10, 8, 6, 4, 2):
    tot, rm = 0.0, np.zeros(4)
    for k, (lay_checks, ntok) in per.items():
        b = 0
        for l in range(W.NL):
            bl, r_ = pom(lay_checks[l], None, R2)
            b += sum(bl); rm += r_
        tot += (b / ntok + lsb_tok[k]) / len(per)
    if base is None:
        base = tot
        say(f"    R2 10 (= v3, nothing at 2 planes): {tot / 1e6:.1f} MB / token (wall24 strict at 159 / 44)")
        continue
    fm, fe = rm[0] / max(rm[1], 1e-12), rm[2] / max(rm[3], 1e-12)
    g_ = gap.get(R2)
    pred = v3 + fe * g_ if g_ is not None else None
    ok = "" if pred is None else (" -> predicted KLD " + " / ".join(f"{p:.5f}" for p in pred) +
                                   (" INSIDE the gate" if (pred <= tgt).all() else " OVER the gate on " +
                                    ", ".join(n for n, p, t in zip(("code", "gsm", "ifeval", "multi"), pred, tgt) if p > t)))
    say(f"    R2 {R2}: {tot / 1e6:.1f} MB / token ({100 * (tot / base - 1):+.1f} %); rank >= {R2} gate mass at 2 planes "
        f"{100 * fm:.1f} % (energy {100 * fe:.1f} %){ok}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/pom24.txt" if (R, A, CM, CL) == (16, 10, 159, 44) else
                        f"~/MoonShots/data/pom24_r{R}a{A}_{CM}_{CL}.txt"), "w").write("\n".join(lines) + "\n")
