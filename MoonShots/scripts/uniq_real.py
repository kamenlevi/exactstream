"""R-30 UNIQ redo on REAL drafted checks with gate weights (Leap's R-32 dump ~/Leap/data/r32_routes.npz: 628 DFlash2 16-row
checks, 24 held-out code samples, layers 0-11, top-10 ids + softmax gates, accepted drafts per check). Check-level rule: drop
expert e from the check when its summed top-10-renormalized weight over the rows (ranks >= R0 uses only) < theta.
(1) 45 GB line, no cache: per-check per-layer union and (row, expert) pairs, dropped gate mass per row-layer.
(2) 24 GB line: strict LFU per layer (C_m 159 / C_l 44, pool trimmed to C after each check) through each sample's
    consecutive checks, first 5 checks of a sample = warm-up; misses x (1,900,544 / 720,896) per accepted token (keep =
    accepted + 1). Each with all 16 rows read and with the oracle row trim (kept rows only, R-24).
usage: [BigMoE venv] taskpolicy -b python uniq_real.py -> stdout + data/uniq_real.txt"""
import os
import numpy as np

z = np.load(os.path.expanduser("~/Leap/data/r32_routes.npz"))
RT, GT = z["routes"].astype(np.int32), z["gates"].astype(np.float64)
ACC, SMP = z["accepted"].astype(np.int32), z["sample"]
GT = GT / GT.sum(-1, keepdims=True)
_o = np.argsort(-GT, axis=-1, kind="stable")                # the dump's top-10 come in argpartition order, not rank order
RT, GT = np.take_along_axis(RT, _o, -1), np.take_along_axis(GT, _o, -1)
NC, NL = RT.shape[:2]
MS, LS, CM, CL, WARM = 1900544, 720896, 159, 44, 5
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def needs(t, l, r0, th, trim):
    """(msb set, lsb set, union before, pairs before, pairs after, dropped mass, rows) for check t, layer l"""
    rows = ACC[t] + 1 if trim else 16
    ids, w = RT[t, l, :rows], GT[t, l, :rows]
    tot, best = {}, {}
    for r in range(rows):
        for k in range(10):
            e = int(ids[r, k])
            tot[e] = tot.get(e, 0.0) + (w[r, k] if k >= r0 else 9.0)
            best[e] = min(best.get(e, 99), k)
    drop = {e for e, v in tot.items() if v < th}
    keep = {e: k for e, k in best.items() if e not in drop}
    pairs = sum(1 for r in range(rows) for k in range(10) if int(ids[r, k]) not in drop)
    mass = sum(w[r, k] for r in range(rows) for k in range(10) if int(ids[r, k]) in drop)
    return set(keep), {e for e, k in keep.items() if k < 2}, len(tot), rows * 10, pairs, mass, rows


def strict(seq, C, d=0.99):
    out, R, score, wgt = [], set(), {}, 1.0
    for N in seq:
        wgt /= d
        for e in N:
            score[e] = score.get(e, 0.0) + wgt
        m = [e for e in N if e not in R]
        out.append(len(m))
        for e in m:
            if len(R) >= C:
                vic = [v for v in R if v not in N]
                if vic:
                    R.discard(min(vic, key=lambda v: score[v]))
            R.add(e)
        if len(R) > C:
            R = set(sorted(R, key=lambda v: -score[v])[:C])
    return out


say(f"Leap R-32 dump: {NC} checks, {NL} layers, {len(set(SMP.tolist()))} samples, accepted mean {ACC.mean():.2f}")
order = {s: [t for t in range(NC) if SMP[t] == s] for s in sorted(set(SMP.tolist()))}
res = {}
for trim in (False, True):
    for r0, th in ((10, 0.0), (4, 0.06), (4, 0.08), (4, 0.10), (4, 0.12), (2, 0.16)):
        U0 = U1 = P0 = P1 = M = RW = 0
        byt = tok = 0.0
        for s, ts in order.items():
            per_layer = [[needs(t, l, r0, th, trim) for t in ts] for l in range(NL)]
            for l in range(NL):
                for (ms_, ls_, u0, p0, p1, mass, rows) in per_layer[l]:
                    U0 += u0; U1 += len(ms_); P0 += p0; P1 += p1; M += mass; RW += rows
                mm = strict([x[0] for x in per_layer[l]], CM)
                ml = strict([x[1] for x in per_layer[l]], CL)
                byt += sum(a * MS + b * LS for i, (a, b) in enumerate(zip(mm, ml)) if i >= WARM)
            tok += sum(ACC[t] + 1 for i, t in enumerate(ts) if i >= WARM)
        res[(trim, r0, th)] = byt / tok / NL
        say(f"{'kept rows' if trim else 'all 16 rows'}, ranks >= {r0} theta {th:.2f}: union/check-layer {U1 / (NC * NL):.1f} "
            f"(all-rows base {U0 / (NC * NL):.1f}), pairs {P1 / (NC * NL):.1f}, dropped mass {100 * M / RW:.2f} % per row-layer | "
            f"24 GB strict LFU {res[(trim, r0, th)] / 1e6:.3f} MB / token / layer")
b0 = res[(False, 10, 0.0)]
for k, v in res.items():
    say(f"  vs all rows, no drop: {'trim' if k[0] else 'all '} r0 {k[1]} theta {k[2]:.2f}: {100 * (v / b0 - 1):+.1f} % bytes / token at 24 GB")
open(os.path.expanduser("~/MoonShots/data/uniq_real.txt"), "w").write("\n".join(lines) + "\n")
