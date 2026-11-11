"""Fit12 GP3 (2026-10-24): split a pipelined B12LL verify cycle with the loader's call log (BIGMOE_CALLLOG rows: L, seq, misses,
paired, t_seen, t_rel, t_wait; L < 48 trunk base rows, 48 the MTP head, 49 + L the LSB sidecar rows (fit_clog.py); seq =
gate.new_pass, which the streamed MTP head also bumps on every draft call).
Per verify seq with all 48 trunk rows: service = sum(t_rel - t_seen) (loader: post seen -> row released), gaps = sum over
L = 1..47 of t_seen(L) - t_rel(L - 1) (GPU between spins: experts of L-1, HC, mixer + router + post of L, plus the post pickup
and release wake latencies), and the cycle = t_seen(0) of the next verify seq - t_seen(0) of this one (verify + drafts + the
CPU work between). Usage: python3 fit_gp3.py clog.csv [more.csv]."""
import sys
from collections import defaultdict


def load(p):
    rows = defaultdict(list)
    for ln in open(p):
        if ln.startswith("L,"):
            continue
        f = ln.strip().split(",")
        if len(f) < 7:
            continue
        L, seq, mis, par = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        rows[seq].append((L, mis, par, float(f[4]), float(f[5]), float(f[6])))
    return rows


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def split(p):
    rows = load(p)
    seqs = sorted(rows)
    ver = [s for s in seqs if len({r[0] for r in rows[s] if r[0] < 48}) == 48 and sum(r[0] < 48 for r in rows[s]) <= 60]
    out = []
    for i, s in enumerate(ver):
        rs = {}
        for r in rows[s]:
            if r[0] < 48 and r[0] not in rs:          # first row per trunk layer (one post per layer per verify call)
                rs[r[0]] = r
        t0 = rs[0][3]
        serv = sum(rs[L][4] - rs[L][3] for L in range(48))
        gaps = sum(rs[L][3] - rs[L - 1][4] for L in range(1, 48))
        mis = sum(rs[L][1] for L in range(48))
        stalled = sum(1 for L in range(48) if rs[L][4] - rs[L][3] > 2e-4)
        nxt = ver[i + 1] if i + 1 < len(ver) else None
        if nxt is None or nxt - s > 8:                # a prompt / warm-up boundary: no cycle
            continue
        cyc = min(r[3] for r in rows[nxt] if r[0] == 0) - t0
        between = [r for q in range(s + 1, nxt) for r in rows[q]]          # draft-call rows (L = 48) up to the next verify
        tail = (min(r[3] for r in between) - rs[47][4]) if between else float("nan")   # last trunk release -> 1st draft post
        dserv = sum(r[4] - r[3] for r in between)
        head = (t0 - max(r[4] for r in rows[s - 1])) if (s - 1) in rows else float("nan")  # prev. release -> L0 post
        out.append(dict(cyc=cyc, serv=serv, gaps=gaps, span=rs[47][4] - t0, mis=mis, stalled=stalled, tail=tail,
                        ndraft=len(between), dserv=dserv, head=head))
    return out


for p in sys.argv[1:]:
    o = split(p)
    if not o:
        print(p, "no complete verify cycles"); continue
    k = ("cyc", "span", "serv", "gaps", "tail", "head", "dserv")
    print(f"{p}: {len(o)} verify cycles; medians (ms): " + ", ".join(f"{x} {1e3 * med([d[x] for d in o]):.1f}" for x in k)
          + f"; misses {med([d['mis'] for d in o])}, stalled layers {med([d['stalled'] for d in o])}, draft rows "
          f"{med([d['ndraft'] for d in o])}")
    gl = sorted(o, key=lambda d: d["cyc"])
    print("   means (ms): " + ", ".join(f"{x} {1e3 * sum(d[x] for d in o) / len(o):.1f}" for x in k if x not in ("tail", "head"))
          + f"; gap per layer {1e3 * sum(d['gaps'] for d in o) / len(o) / 47:.3f}")
