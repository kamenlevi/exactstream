"""R-54 ECOSPEC12 (NewIdeas Thu 20:47; EXACT by construction: only the number of drafted rows per check changes, verification
is unchanged). In a 4-row MTP check every row's top-10 experts are read on a miss, but the rows after the first rejection
are wasted reads. Screen (CPU only) on R-49's traces (data/r49/*.npz: the 12 GB decode's own verify passes; per pass the
kept rows = the tokens it emitted) with pom12.py's LRU replay and SSD busy model, the 12 GB engine as it ships (v3n + M-01:
C_m 51 MSB / C_l 20 LSB per layer, ranks 0-1 read MSB + LSB, ranks 2-9 MSB; no prefetch, one read burst per (pass, layer)).
(0) Today: every row of every pass. Per pass and row, the MARGINAL reads (experts not resident at the call's start and not
    already needed by an earlier row of the same pass; LSB parts likewise for rank 0-1 uses) and whether the row was kept.
(1) Oracle: each pass keeps only its first `kept` rows (cut after its last accepted row); the replay runs the cut passes,
    so the skipped reads neither fill nor evict the pools (their cache effects are kept).
Also shown, as an approximation only (the real run would re-align its checks): fixed horizons of 3 and 2 rows, tokens per
pass = min(kept, rows).
Busy model (pom12, NewIdeas 19:3x): per call T0 0.2 ms (if it reads) + bytes / 14 GB/s + 4 us per read (no 2-plane rebuilds).
KILL: the oracle saves < 6 % of modeled busy ms per token. PASS: >= 12 % (then step 2: one short capture of the MTP head's
p1 per drafted token, a policy dropping row j when p1(j) < tau and its marginal reads > m). Between: report it (BigMoE runs
the same at 24 GB pools).
usage: [BigMoE venv] taskpolicy -b python scripts/eco12.py -> stdout + data/r49/eco12_<C_m>_<C_l>.txt"""
import glob, os, time
from collections import OrderedDict
import numpy as np

FULL, LSB = 2048000, 720896                                  # records3m MSB part / LSB part
CM, CL = int(os.environ.get("P12_CM", "51")), int(os.environ.get("P12_CL", "20"))
BW, T0, TOP = float(os.environ.get("P12_BW", "14.0e9")), float(os.environ.get("P12_T0", "0.2e-3")), float(os.environ.get("P12_TOP", "4e-6"))
NL, NR = 48, 4
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def load():
    """-> {name: [pass dicts]}: ids [S, 48, 10], kept rows, measured flag (pom12's r49 loader)"""
    out = {}
    WARM = int(os.environ.get("P12_WARM", "10"))
    for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/r49/*.npz"))):
        d = np.load(f)
        L, S, E = d["calls_L"].astype(int), d["calls_S"].astype(int), d["calls_E"].astype(int)
        off = np.concatenate([[0], np.cumsum(S)])
        ids20 = d["ids"].astype(np.int32)
        passes, i, bad = [], 0, 0
        while i + NL <= len(L):
            if not (L[i:i + NL] == np.arange(NL)).all() or (S[i:i + NL] != S[i]).any():
                bad += 1; i += 1; continue
            s = S[i]
            ids = np.stack([ids20[off[i + l]:off[i + l] + s, :10] for l in range(NL)], 1)
            passes.append(dict(ids=ids, E=E[i], meas=len(passes) >= WARM))
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


def run(passes, mode):
    """one prompt, all layers; mode 'all' (today), 'oracle' (first kept rows), or an int h (first h rows, approximate)
    -> bytes, tokens, busy s, reads, per-row marginal stats [row, kept?, (msb reads, lsb reads, bytes)] (mode 'all' only)"""
    by = ntok = nrd = 0
    busy = 0.0
    marg = np.zeros((NR, 2, 3))                              # row j, rejected (0) / kept (1): msb reads, lsb reads, bytes
    nrow = np.zeros((NR, 2))                                 # row-pass counts per layer
    for l in range(NL):
        res, lsb = OrderedDict(), OrderedDict()
        for P in passes:
            S = P["ids"].shape[0]
            use = S if mode == "all" else P["kept"] if mode == "oracle" else min(S, mode)
            ids = P["ids"][:use, l]
            if mode == "all" and P["meas"]:                  # marginal reads per row against the call's start state
                seen, seenl = set(), set()
                for r in range(S):
                    m1 = m2 = 0
                    for k in range(10):
                        e = int(ids[r, k])
                        if e not in res and e not in seen:
                            m1 += 1
                        seen.add(e)
                        if k < 2:
                            if e not in lsb and e not in seenl:
                                m2 += 1
                            seenl.add(e)
                    kp = int(r < P["kept"])
                    marg[r, kp] += (m1, m2, m1 * FULL + m2 * LSB)
                    nrow[r, kp] += 1
            best = {}
            for k in range(10):                              # rank-major, as the ext's flag rows
                for r in range(ids.shape[0]):
                    e = int(ids[r, k])
                    if e not in best:
                        best[e] = k
            b = nr = 0
            for e in best:
                if e in res:
                    res.move_to_end(e)
                    continue
                if len(res) >= CM:
                    vic = next((v for v in res if v not in best), None)
                    if vic is not None:
                        del res[vic]
                b += FULL; nr += 1
                res[e] = 3
            for e, k in best.items():
                if k >= 2:
                    continue
                if e in lsb:
                    lsb.move_to_end(e)
                    continue
                if len(lsb) >= CL:
                    vic = next((v for v in lsb if v not in best or best[v] >= 2), None)
                    if vic is not None:
                        del lsb[vic]
                b += LSB; nr += 1
                lsb[e] = 1
            if not P["meas"]:
                continue
            by += b; nrd += nr
            busy += (T0 if nr else 0.0) + b / BW + nr * TOP
            if l == 0:
                ntok += P["kept"] if mode in ("all", "oracle") else min(P["kept"], mode)
    return by, ntok, busy, nrd, marg, nrow


t0 = time.time()
say(f"R-54 ECOSPEC12 oracle screen: R-49 traces, LRU pools C_m {CM} / C_l {CL} (v3n + M-01); bytes MSB {FULL:,} / LSB "
    f"{LSB:,}; busy per call T0 {T0 * 1e3:.2f} ms + bytes / {BW / 1e9:.1f} GB/s + {TOP * 1e6:.0f} us / read")
DATA = load()
R = {}
for mode in ("all", "oracle", 3, 2):
    tot = np.zeros(4)
    per = []
    M, NRW = np.zeros((NR, 2, 3)), np.zeros((NR, 2))
    for name, ps in DATA.items():
        by, nt, bz, nr, marg, nrow = run(ps, mode)
        tot += (by, nt, bz, nr)
        M += marg; NRW += nrow
        per.append((name, by / nt, bz / nt, nt))
    R[mode] = (tot, per, M, NRW)
base = R["all"][0]
bb, bz = base[0] / base[1], base[2] / base[1]
for mode, (tot, per, M, NRW) in R.items():
    t_b, t_z = tot[0] / tot[1], tot[2] / tot[1]
    lab = {"all": "today (4 rows)", "oracle": "ORACLE (kept rows)", 3: "fixed 3 rows ~", 2: "fixed 2 rows ~"}[mode]
    say(f"  {lab:18s}: busy {1e3 * t_z:6.2f} ms / token ({100 * (t_z / bz - 1):+5.1f} %), {t_b / 1e6:6.1f} MB / token "
        f"({100 * (t_b / bb - 1):+5.1f} %), reads / token {tot[3] / tot[1]:5.1f}, tokens {int(tot[1])} | per prompt busy ms/tok "
        + ", ".join(f"{n} {1e3 * z:.2f} ({100 * (z / [q for q in R['all'][1] if q[0] == n][0][2] - 1):+.1f} %)"
                    for n, _, z, _ in per))
tot, per, M, NRW = R["all"]
say("  today's marginal reads per row (per layer call, measured passes; 'rej' = row after the first rejection = wasted):")
wasted = M[:, 0, 2].sum() / M[:, :, 2].sum()
for r in range(NR):
    n0, n1 = NRW[r]
    f = lambda kp, n: (f"{M[r, kp, 0] / n:5.2f} MSB + {M[r, kp, 1] / n:4.2f} LSB" if n else "    -")
    say(f"    row {r}: kept {100 * n1 / max(n0 + n1, 1):5.1f} % of passes | kept rows {f(1, n1)} | rejected rows {f(0, n0)} | "
        f"share of today's bytes {100 * M[r, :, 2].sum() / M[:, :, 2].sum():5.1f} %")
say(f"  wasted share of today's bytes (marginal reads on rejected rows, no cache effects): {100 * wasted:.1f} %")
osave = 1 - (R["oracle"][0][2] / R["oracle"][0][1]) / bz
say(f"  VERDICT (oracle busy saving {100 * osave:.1f} %; KILL < 6 %, PASS >= 12 %): "
    + ("PASS" if osave >= 0.12 else "KILL" if osave < 0.06 else "BETWEEN"))
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser(f"~/Breakthrough/data/r49/eco12_{CM}_{CL}.txt"), "w").write("\n".join(lines) + "\n")
