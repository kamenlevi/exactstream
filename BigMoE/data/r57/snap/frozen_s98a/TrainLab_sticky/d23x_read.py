"""D-23x read-out (MTP Head, Mon 11-02; NewIdeas 09:5x): the first check after the prefill, by pool and layer. CPU only: reads
data/d23x_runs.jsonl (turns/turn75_d23x.sh: the D-23d harness at C 192 with BIGMOE_TRACE=1 D23_TRACE=1). Each record's "tr"
holds the reads since the last call: "pf" = prefill + MTP history, then checks 1-6, each [n base demand, base prefetch, base
joint, sidecar demand, sidecar prefetch, sidecar joint, read seconds, wall span s, per-layer base counts, per-layer sidecar
counts] (base = the MSB-3 pool, every routed expert; sidecar = the LSB-1 part, only experts at 4-bit ranks). Question
(NewIdeas): is the 3-bit (base) pool cold after the 4-bit prefill, so the first check reads it from the SSD? Then a pool
warm-up at the end of the prefill (TRIED: PARKED) is an exact fix. Output: data/d23x_read.txt (appended)."""
import json, os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

R = []
for l in open(os.path.join(TD, "d23x_runs.jsonl")):
    d = json.loads(l)
    if d.get("tr") and not (d["cut"] or d["paused"] or d["invalid"]):
        R.append(d)
say(f"D-23x read-out (C 192, head b5, read trace): {len(R)} clean runs with a trace")
for d in R:
    pc, tr = d["pc"], {str(x[0]): x[1:] for x in d["tr"]}
    say(f"  {d['key']:14s} {d['arm']:6s} prefill {d['pf']:5.1f} s, {d['n']} tok, {d['tok_s']:.1f} tok/s")
    for c in ["pf"] + [str(i) for i in range(1, 7)]:
        if c not in tr:
            continue
        bd, bp, bj, sd, sp, sj, rs, span, lb, ls = tr[c]
        ck = f"check {c}: {pc[int(c) - 1][0]:2d} rows, {pc[int(c) - 1][2]:7.1f} ms" if c != "pf" else "prefill + history     "
        nz = int(np.count_nonzero(np.array(lb)))
        say(f"    {ck}: base {bd:5d} demand / {bp:5d} prefetch / {bj:4d} joint, sidecar {sd:5d} / {sp:5d} / {sj:4d}; "
            f"read s {rs:6.2f}, span {span:5.2f} s; layers with base reads {nz}")
# the first check against checks 2-6: per run and pooled
say("  first check vs checks 2-6 (mean per check):")
agg = {}
for d in R:
    tr = {str(x[0]): x[1:] for x in d["tr"]}
    typ = d["key"].split("/")[0]
    for c in [str(i) for i in range(1, 7)]:
        if c in tr:
            g = agg.setdefault((typ, d["arm"], "1" if c == "1" else "2-6"), [])
            bd, bp, bj, sd, sp, sj, rs, span, lb, ls = tr[c]
            g.append((bd + bj, bp, sd + sj, sp, rs, d["pc"][int(c) - 1][2], d["pc"][int(c) - 1][0]))
for (typ, arm, which), g in sorted(agg.items()):
    a = np.array(g, dtype=float)
    say(f"    {typ:6s} {arm:6s} check {which:3s} (n {len(a):2d}): base demand+joint {a[:, 0].mean():6.0f}, base prefetch "
        f"{a[:, 1].mean():5.0f}, sidecar demand+joint {a[:, 2].mean():5.0f}, sidecar prefetch {a[:, 3].mean():5.0f}; read s "
        f"{a[:, 4].mean():5.2f}; check ms {a[:, 5].mean():6.0f}; rows {a[:, 6].mean():4.1f}")
# per-layer: where the first check's base reads land (agent, pooled over arms), vs checks 2-6
for which in ("1", "2-6"):
    acc, n = None, 0
    for d in R:
        if not d["key"].startswith("agent/"):
            continue
        tr = {str(x[0]): x[1:] for x in d["tr"]}
        for c in (["1"] if which == "1" else ["2", "3", "4", "5", "6"]):
            if c in tr:
                v = np.array(tr[c][8], dtype=float)
                acc = v if acc is None else acc + v
                n += 1
    if acc is not None:
        acc /= max(n, 1)
        q = np.array_split(acc, 4)
        say(f"  agent base reads per layer, check {which} (mean of {n}): layers by quarter {' / '.join(f'{x.sum():.0f}' for x in q)}; "
            f"top layers {', '.join(f'L{i} {acc[i]:.0f}' for i in np.argsort(-acc)[:6])}")
open(os.path.join(TD, "d23x_read.txt"), "a").write("\n".join(lines) + "\n\n")
