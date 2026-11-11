"""TEN-X LEDGER (NewIdeas, Tue 3 Nov 2026 23:5x; CPU only). The orchestrator 23:4x for Kamen: what breaks the SSD wall
(WALL1_SCOREBOARD 'BROKEN': SSD MB per kept token <= 10, exposed ~0). Each multiplicative factor on SSD MB per kept token
is measured on its own, per domain, from existing captures:
- the decision map's 36 windows: the 125B's own rank-sorted routing for prompt + answer, teacher-forced
  (~/FableGod/decision_map/out/routing, baseline.json 'a' = prompt rows);
- MTP Head's HPD-0 walks (DFlash2 cf per draft; ~/TrainLab/data/d02b_walk_hpd0_<dom>.npz), cut at the live engine's
  running cf 0.3 (dm_oea_masks.live_segments), as every live map arm.
Replay (all parts): a plain LRU per layer for the base pool (the MSB part of every routed rank) and the sidecar pool (the
LSB part of ranks 0-1), warmed by the prompt's rows and carried across a domain's windows (one session per domain).
Rows after a check's first miss enter as the map's wrong-draft stand-in (row p + 97), as in every union arm.
Parts:
  A  kept per check. Today's live checks, and synthetic checks of k consecutive answer rows (k = 1..16; the last kept row
     is the bonus row) plus w stand-in rows (w 0, or today's mean waste per check). Per pool line: SSD MB per kept token;
     plus the RAM union bytes per kept token (what the expert kernels read; no cache).
  B  per-request locality: distinct experts per layer over the answer; experts per layer covering 90 / 95 / 99 % of the
     answer's routing mass (mean mass by rank from Fable C's cap45 recomputation, UNION_STACK sec 3: approximate) and
     their GB at 4 bits; the share of that mass on experts the prompt's prefill already routed; the compulsory floor
     (every distinct expert read once per request) vs answer length, with a power-law extrapolation.
  C  FAST reads on today's live checks per pool line: Q1 / QOM at cf 0.5 and cf 0 (sure rows load ranks 0-1 only; QOM's
     reads equal Q1's by construction), with and without 3-bit sure rows (no LSB read for them).
Pool lines: 45 GB = laysz45_256's per-layer sizes (C_m mean 295, C_l 89); 24 GB = C_m 147 + C_l 46 (MoonShots' fast24
params); 12 GB-class = C_m 51 + C_l 32 (Breakthrough's C 51 replays, B12's sidecar). Plain LRU, not the engines' LRU +
popularity / decayed LFU: the 45 GB EXACT row is calibrated against band45's capture (47.0 / 31.1 / 30.7 MB per kept).
Writes ~/NewIdeas/data/ledger/ledger_<part>.txt. Usage (BigMoE venv): python ten_x_ledger.py [A] [B] [C]"""
import json, os, sys, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_oea_masks import live_segments, OUT, dm_gate  # noqa: E402
MR, LR = 2162688, 770048                    # records3 MSB / LSB part bytes = SSD bytes per miss (laysz45.py)
MASS = np.array([0.185, 0.134, 0.112, 0.098, 0.090, 0.083, 0.079, 0.076, 0.073, 0.071])   # mean router mass by rank
DST = os.path.expanduser("~/NewIdeas/data/ledger")
SZ = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))
POOLS = {"45 GB": (SZ["C_m"], SZ["C_l"]), "24 GB": ([147] * 48, [46] * 48), "12 GB": ([51] * 48, [32] * 48)}
W = None
CASES = [(k, 0.0) for k in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16)] + [(10, 2.0), (12, 2.0), (14, 2.0)]


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


def live_blocks(w):
    """today's live checks (route_w.blocks_of, live): [(rows, cfs, kept)]; cf -1 = the bonus row (always 10 ranks)"""
    ids, g = w["ids"], w["g"]; N = len(ids); bl = []
    for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
        for (k, m, A, dc) in live_segments(k0_, m0_, A0_, dc0_):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            rows, cfs = [], []
            for p in range(A, last + 1):
                if p <= fm: rows.append(p); cfs.append(-1.0 if k == m and p == A + m else g[p])
                else: rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
            if rows: bl.append((rows, cfs, min(k, fm - A) + 1))
    return bl


def synth_blocks(w, k, wst):
    """k consecutive answer rows (the last = the bonus row, cf -1; the others drafted and accepted, cf 1) + wst wrong-draft
    stand-ins (cf 1: drafted rows). wst may be fractional: the stand-in count alternates to match the mean."""
    N = len(w["ids"]); bl = []; acc = 0.0
    for s in range(w["a"], N, k):
        real = list(range(s, min(s + k, N)))
        acc += wst; nw = int(acc); acc -= nw
        rows = real + [(p + 97) % N for p in range(s + len(real), s + len(real) + nw)]
        cfs = [1.0] * (len(real) - 1) + [-1.0] + [1.0] * nw
        bl.append((rows, cfs, len(real)))
    return bl


def touch(c, e, C):
    if e in c: c.move_to_end(e)
    else:
        c[e] = 1
        if len(c) > C: c.popitem(last=False)


def replay(task):
    """one (domain, pool line, case, rule): SSD MB per kept token (base + LSB), RAM GB of the union per kept token,
    kept per check, rows per check"""
    dom, pool, case, rule = task
    Cm, Cl = POOLS[pool]; thr, t3 = rule
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = ml = ub = ul = kept = nrow = nchk = 0
    for w in W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):                                  # the prompt's prefill warms both pools
            for l in range(48):
                for e in ids[p, l].tolist(): touch(bp[l], e, Cm[l])
                for e in ids[p, l, :2].tolist(): touch(lp[l], e, Cl[l])
        blocks = live_blocks(w) if case == "live" else synth_blocks(w, case[0], case[1])
        for rows, cfs, kp in blocks:
            sure = [thr is not None and c >= thr for c in cfs]
            for l in range(48):
                Ub, Ul = set(), set()
                for r, s_ in zip(rows, sure):
                    Ub.update(ids[r, l, :2 if s_ else 10].tolist())
                    if not (s_ and t3): Ul.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in Ub if e not in bp[l]); ml += sum(1 for e in Ul if e not in lp[l])
                for e in Ub: touch(bp[l], e, Cm[l])
                for e in Ul: touch(lp[l], e, Cl[l])
                ub += len(Ub); ul += len(Ul)
            kept += kp; nrow += len(rows); nchk += 1
    return task, (mb * MR + ml * LR) / max(kept, 1) / 1e6, (ub * MR + ul * LR) / max(kept, 1) / 1e9, kept / max(nchk, 1), \
        nrow / max(nchk, 1)


def part_B():
    """per-request locality (one row per window, means per domain)"""
    lines = ["PART B: per-request locality (36 map windows; mass = the mean router mass by rank, approximate)",
             "| domain | answer tokens | distinct experts per layer | per layer at 90 / 95 / 99 % of the answer's mass | "
             "95 % set GB at 4 bits | answer mass on experts the prefill routed | distinct experts the prefill routed |",
             "|---|---|---|---|---|---|---|"]
    curves = {}
    for dom, ws in W.items():
        acc = {k: [] for k in ("T", "D", "n90", "n95", "n99", "gb", "pm", "pd")}
        for w in ws:
            ids, a = w["ids"], w["a"]; N = len(ids); T = N - a
            if T < 16: continue
            n90 = n95 = n99 = D = 0; pm = pd = tot = dd = 0.0
            for l in range(48):
                m = {}
                for p in range(a, N):
                    for r, e in enumerate(ids[p, l].tolist()): m[e] = m.get(e, 0.0) + MASS[r]
                v = np.sort(np.fromiter(m.values(), float))[::-1]; cs = np.cumsum(v) / v.sum()
                n90 += int(np.searchsorted(cs, 0.90) + 1); n95 += int(np.searchsorted(cs, 0.95) + 1)
                n99 += int(np.searchsorted(cs, 0.99) + 1); D += len(m)
                fp = set(ids[:a, l].ravel().tolist())
                pm += sum(x for e, x in m.items() if e in fp); tot += v.sum(); pd += sum(1 for e in m if e in fp); dd += len(m)
            acc["T"].append(T); acc["D"].append(D / 48); acc["n90"].append(n90 / 48); acc["n95"].append(n95 / 48)
            acc["n99"].append(n99 / 48); acc["gb"].append(n95 * (MR + LR) / 1e9); acc["pm"].append(pm / tot)
            acc["pd"].append(pd / dd)
            t_list = [t for t in (8, 16, 32, 64, 128, 256, 512) if t <= T] + [T]
            for t in t_list:                                    # compulsory floor: each distinct expert read once
                db = sum(len(set(ids[a:a + t, l].ravel().tolist())) for l in range(48))
                dl = sum(len(set(ids[a:a + t, l, :2].ravel().tolist())) for l in range(48))
                curves.setdefault(dom, {}).setdefault(t if t != T else "T", []).append((t, (db * MR + dl * LR) / t / 1e6, db / 48,
                                                                                       dl / 48))
        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        lines.append(f"| {dom} | {mean['T']:.0f} | {mean['D']:.0f} of 512 | {mean['n90']:.0f} / {mean['n95']:.0f} / "
                     f"{mean['n99']:.0f} | {mean['gb']:.1f} | {100 * mean['pm']:.0f} % | {100 * mean['pd']:.0f} % |")
    lines.append("")
    lines.append("Compulsory floor (no prefill reuse, every distinct expert read once): MB per answer token at t tokens; "
                 "distinct experts per layer in brackets; fit D(t) = c t^b on the fixed t points, extrapolated to 1K / 4K")
    for dom, cv in curves.items():
        pts = sorted((t, float(np.mean([x[1] for x in v])), float(np.mean([x[2] for x in v])),
                      float(np.mean([x[3] for x in v]))) for t, v in cv.items() if t != "T" and len(v) >= 6)
        ts = np.array([p[0] for p in pts], float); dist = np.array([p[2] for p in pts])
        b, lc = np.polyfit(np.log(ts), np.log(dist), 1)
        rl = pts[-1][3] / pts[-1][2]                            # sidecar / base distinct at the longest common t
        ext = []
        for t in (1024, 4096):
            dpl = min(512.0, np.exp(lc) * t ** b)
            ext.append(f"{t}: ~{dpl:.0f} per layer -> ~{dpl * 48 * (MR + min(1.0, rl * (t / pts[-1][0]) ** 0) * LR) / t / 1e6:.1f} MB per token")
        lines.append(f"  {dom}: " + "; ".join(f"t {t}: {mbt:.1f} ({d:.0f})" for t, mbt, d, _ in pts) +
                     f" | fit b {b:.2f}, sidecar / base distinct {rl:.2f} | " + "; ".join(ext))
    lines.append("  (fixed points [K] on the captures; the 1K / 4K points are a power-law extrapolation [P] with the sidecar "
                 "share held at its longest measured value)")
    return lines


def main(parts):
    global W
    t0 = time.time(); W = windows(); os.makedirs(DST, exist_ok=True)
    if "B" in parts:
        L = part_B(); open(os.path.join(DST, "ledger_B.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    tasks = []
    if "A" in parts:
        for dom in W:
            for pool in POOLS:
                tasks.append((dom, pool, "live", (None, False)))
                for case in CASES:
                    tasks.append((dom, pool, case, (None, False)))
    if "C" in parts:
        for dom in W:
            for pool in POOLS:
                for rule in ((None, False), (0.5, False), (0.5, True), (0.0, False), (0.0, True)):
                    tasks.append((dom, pool, "live", rule))
    tasks = list(dict.fromkeys(tasks))
    if not tasks: return
    with mp.get_context("fork").Pool(int(os.environ.get("TXL_PROCS", "3"))) as P:   # <= 4 cores with the mask build
        res = {t: r for t, *r in P.map(replay, tasks, chunksize=1)}
    if "A" in parts:
        L = ["PART A: kept per check (EXACT reads; plain LRU replay; prompt warm-up; carried across a domain's windows)",
             "case = today's live checks; k = k consecutive kept rows per check, no wasted row (a perfect drafter at k); "
             "'k + w wasted' adds w wrong-draft rows (k 14 + 2 = the D98 target, ~14 of 16)",
             "| domain | pool | case | kept per check | rows per check | SSD MB per kept token | RAM union GB per kept token |",
             "|---|---|---|---|---|---|---|"]
        for dom in W:
            for pool in POOLS:
                for case in ["live"] + CASES:
                    mbk, gbk, kpc, rpc = res[(dom, pool, case, (None, False))]
                    nm_ = 'today (live)' if case == 'live' else (f'k {case[0]}' if not case[1] else f'k {case[0]} + {case[1]:.0f} wasted')
                    L.append(f"| {dom} | {pool} | {nm_} | {kpc:.2f} | "
                             f"{rpc:.2f} | {mbk:.1f} | {gbk:.2f} |")
        open(os.path.join(DST, "ledger_A.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    if "C" in parts:
        L = ["PART C: FAST reads on today's live checks (QOM = Q1's reads; 3b = sure rows read no LSB part)",
             "| domain | pool | EXACT | QOM cf 0.5 | QOM cf 0.5 + 3b | QOM cf 0 | QOM cf 0 + 3b |", "|---|---|---|---|---|---|---|"]
        for dom in W:
            for pool in POOLS:
                ex = res.get((dom, pool, "live", (None, False)))
                cells = [f"{ex[0]:.1f}" if ex else "-"]
                for rule in ((0.5, False), (0.5, True), (0.0, False), (0.0, True)):
                    r = res[(dom, pool, "live", rule)][0]
                    cells.append(f"{r:.1f}" + (f" ({100 * (r / ex[0] - 1):+.0f} %)" if ex else ""))
                L.append(f"| {dom} | {pool} | " + " | ".join(cells) + " |")
        open(os.path.join(DST, "ledger_C.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"A", "B", "C"})
