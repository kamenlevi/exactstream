"""M-04 POMVS, POM with variable slots (MoonShots Thu 15:48; NON-exact, CPU). W24's pool sweep: misses fall faster than RAM
grows (pools x1.1 -> -12 %, x1.4 -> -41 % bytes / token), so bits per RESIDENT expert cost about b^2.4. M-03's POM keeps a
2-plane resident in a full 3-plane slot; here it takes its own size and the freed RAM holds more experts.
  RAM per resident (records3 layout, each projection padded to 16 KB as msb_sz is): 3 planes 2,162,688 B; 2 planes (2-bit
  codes + the same bf16 s / b) 3 x 524,288 = 1,572,864 B (72.7 %).
  SSD reads (W24-ec + 11-bit scales, M-03's plane split, data/pom24.txt): full MSB miss 1,900,544, 2-plane miss
  1,294,336, plane-3 upgrade 606,208; LSB pool as v3 (C_l 44, 720,896 per miss).
  Rules (M-03): a miss whose best rank in the check is >= R2 reads 2 planes and stays a 2-plane resident until a use at
  rank < R2 reads plane 3. Budget per layer: C_m 159 x 2,162,688 B, strict LFU (decay 0.99) trimmed after each check,
  victims = residents this check does not need. DG 0: a victim is evicted. DG 1: a 3-plane victim is DEMOTED (drops plane
  3, stays as a 2-plane resident); a 2-plane victim is evicted.
  Quality: f = share of the rank >= R2 gate energy (mass) on accepted rows served at 2 planes; predicted KLD per part =
  v3 + f x gap(R2) as in M-03 (conservative there: measured 0.00202 vs 0.00221 predicted at f 0.20).
PASS (written first): some variant <= -10 % SSD bytes / token vs POM R2 4 on the narrow path (81.1 MB / token) with
predicted KLD inside 1.10 x ar4 on all four parts. KILL: < -4 % for every variant inside the gate.
usage: [BigMoE venv] taskpolicy -b python pomvs24.py -> stdout + data/pomvs24.txt"""
import heapq, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

lines = []


def say(s):
    print(s, flush=True); lines.append(s)


t0 = time.time()
FULL, P12, P3, LSB = 1900544, 1294336, 606208, 720896
SLOT3, SLOT2V = 2162688, 1572864
CM, CL, DEC = 159, 44, 0.99
G = {k: np.stack([d[f"G{l}"] for l in range(W.NL)], 1)[:, :, :10].astype(np.float64) for k, d in W.SETS.items()}
G = {k: g / g.sum(-1, keepdims=True) for k, g in G.items()}
gap = {6: np.array([0.00095, 0.00158, 0.00786, 0.00765]), 2: np.array([0.00497, 0.00830, 0.03635, 0.03853])}
w2 = np.mean(np.concatenate([g.reshape(-1, 10) for g in G.values()]) ** 2, 0)
gap[4] = gap[6] + (gap[2] - gap[6]) * w2[4:6].sum() / w2[2:6].sum()
v3 = np.array([0.00172, 0.00275, 0.01367, 0.01097])
tgt = np.array([0.00253, 0.00449, 0.02041, 0.02055])


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


def run(checks, R2, slot2, dg, A):
    """one layer -> (bytes per check, [red mass, mass, red energy, energy], mean share of residents at 2 planes, mean count).
    dg 2 (VD, value density): trim by the cheapest SSD bytes at risk per freed RAM byte: demoting a 3-plane resident risks
    its decayed count of uses at rank < R2 x P3 per (SLOT3 - slot2) B; evicting a 2-plane one its decayed count of all uses
    x P12 per slot2 B."""
    res, score, shi, w, used, B = {}, {}, {}, 1.0, 0, CM * SLOT3
    byt, rm, n2, nres, nm = [], np.zeros(4), 0, 0, 0
    sz = {3: SLOT3, 2: slot2}
    for (meas, best, ids, g) in checks:
        w /= DEC
        for e, k in best.items():
            score[e] = score.get(e, 0.0) + w
            if k < R2:
                shi[e] = shi.get(e, 0.0) + w
        b = 0
        for e, k in best.items():
            need3 = k < R2
            p = res.get(e)
            if p is None:
                b += FULL if need3 else P12
                res[e] = 3 if need3 else 2
                used += sz[res[e]]
            elif need3 and p == 2:
                b += P3
                res[e] = 3
                used += SLOT3 - slot2
        if meas:
            byt.append(b)
            for r in range(A):
                for kk in range(R2, 10):
                    m = g[r, kk]
                    red = res[int(ids[r, kk])] == 2
                    rm += (m * red, m, m * m * red, m * m)
            n2 += sum(1 for p in res.values() if p == 2); nres += len(res); nm += 1
        else:
            byt.append(0)
        if used > B and dg == 2:
            hp = [(shi.get(v, 0.0) * P3 / (SLOT3 - slot2), v) if p == 3 else (score[v] * P12 / slot2, v)
                  for v, p in res.items() if v not in best]
            heapq.heapify(hp)
            while used > B and hp:
                c, v = heapq.heappop(hp)
                if res[v] == 3:
                    res[v] = 2
                    used -= SLOT3 - slot2
                    heapq.heappush(hp, (score[v] * P12 / slot2, v))
                else:
                    used -= sz[res.pop(v)]
        elif used > B:
            vic = sorted((v for v in res if v not in best), key=lambda v: score[v])
            for v in vic:
                if used <= B:
                    break
                if dg and res[v] == 3:
                    res[v] = 2
                    used -= SLOT3 - slot2
                else:
                    used -= sz[res.pop(v)]
            for v in vic:
                if used <= B:
                    break
                if v in res:
                    used -= sz[res.pop(v)]
    return byt, rm, n2 / max(nres, 1), nres / max(nm, 1)


def build(R, A):
    """per trace: (per-layer check lists, measured tokens, v3 LSB bytes)"""
    per = {}
    for k, ids in W.IDS.items():
        starts = list(range(0, ids.shape[0] - R + 1, A))
        meas = [W.MEAS[0] <= s < W.MEAS[1] for s in starts]
        lay, lb = [], 0
        for l in range(W.NL):
            cl, lsets = [], []
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
            lay.append(cl)
            lb += sum(a for a, f in zip(strict(lsets, CL), meas) if f) * LSB
        per[k] = (lay, A * sum(meas), lb)
    return per


if __name__ == "__main__":
    for R, A in ((4, 3), (16, 10)):
        per = build(R, A)
        say(f"== {R}-row checks accepting {A} (C_m {CM} x {SLOT3:,} B per layer, C_l {CL}; traces {', '.join(per)})")
        ref = None
        for name, R2, slot2, dg in (("v3 (nothing at 2 planes)", 10, SLOT3, 0), ("POM R2 6, fixed slots (M-03)", 6, SLOT3, 0),
                                    ("POM R2 4, fixed slots (M-03)", 4, SLOT3, 0),
                                    ("POMVS R2 6, DG 0", 6, SLOT2V, 0), ("POMVS R2 6, DG 1", 6, SLOT2V, 1),
                                    ("POMVS R2 4, DG 0", 4, SLOT2V, 0), ("POMVS R2 4, DG 1", 4, SLOT2V, 1),
                                    ("POMVS R2 6, VD", 6, SLOT2V, 2), ("POMVS R2 4, VD", 4, SLOT2V, 2)):
            tot, rm, s2, nr = 0.0, np.zeros(4), [], []
            for k, (lay, ntok, lb) in per.items():
                b = 0
                for l in range(W.NL):
                    bl, r_, f2, n_ = run(lay[l], R2, slot2, dg, A)
                    b += sum(bl); rm += r_; s2.append(f2); nr.append(n_)
                tot += (b + lb) / ntok / len(per)
            if ref is None and R2 == 4:
                ref = tot
            fm, fe = rm[0] / max(rm[1], 1e-12), rm[2] / max(rm[3], 1e-12)
            g_ = gap.get(R2)
            q = ""
            if g_ is not None:
                pred = v3 + fe * g_
                q = (" -> predicted KLD " + " / ".join(f"{p:.5f}" for p in pred) +
                     (" INSIDE" if (pred <= tgt).all() else " OVER on " + ", ".join(
                         n for n, p, t in zip(("code", "gsm", "ifeval", "multi"), pred, tgt) if p > t)))
            vs = f" ({100 * (tot / ref - 1):+.1f} % vs POM R2 4)" if ref is not None and name.startswith("POMVS") else ""
            say(f"  {name:30s} {tot / 1e6:6.1f} MB / token{vs}; residents {np.mean(nr):5.1f} / layer, {100 * np.mean(s2):4.1f} % "
                f"at 2 planes; rank >= R2 at 2 planes: mass {100 * fm:4.1f} %, energy {100 * fe:4.1f} %{q}")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/pomvs24.txt"), "w").write("\n".join(lines) + "\n")
