"""Expert bytes per check of the layer-wise reuse map (Breakthrough, Mon 2026-11-02 22:5x; Fable A's ASSUMPTION_HUNT.md,
the orchestrator 22:5x; CPU). The FLOOR method (union of expert ids per layer over each 16-row block, summed over 48
layers, 2.2 MB per record, no cache): the HPD-0 walks' blocks (rows A .. A + m), DAY9's routing capture; rows after a
check's first miss hold wrong drafts (row p + 97's routing) with the drafter's cf for those drafts (the walk's dcf).
Schemes: exact (top-10 everywhere), Q1 (top-2 on rows with the drafter's cf >= 0.99), the reuse map (layers 8-11: every
other answer row reuses, layers 12-35: bound_runs(cf >= 0.999, 1) reuses; a reusing row reads nothing at that layer), and
the reuse map + Q1. Usage (BigMoE venv): python dm_pframe_bytes.py -> out/PFRAME_bytes.txt"""
import json, os
import numpy as np
import dm_gate
from dm_common import bound_runs
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
say("Expert bytes per check (GB, union per layer over each 16-row block; ratio to exact and to Q1)")
say("| domain | exact | Q1 (top-2 at cf >= 0.99) | reuse map | reuse map + Q1 | map / exact | map + Q1 / Q1 | reused row-layers (8-11 / 12-35) |")
say("|---|---|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    B = {s: 0.0 for s in ("exact", "q1", "map", "mapq1")}; nchk = 0; ru = [0, 0, 0]
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids); a = base[nm]["a"]
        g = np.nan_to_num(G[nm], nan=-1.0)
        sa = np.zeros(N, bool); sa[a:] = True; s_all = bound_runs(sa, 1)            # layers 8-11 (the map's own phase)
        s_gate = bound_runs(np.r_[np.zeros(a, bool), g[a:N] >= 0.999], 1)          # layers 12-35
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            last, fm = min(A + m, N - 1), A + k; nchk += 1
            rows, cf, sk_all, sk_gate = [], [], [], []
            for p in range(A, last + 1):
                if p <= fm: rows.append(p); cf.append(g[p]); sk_all.append(s_all[p]); sk_gate.append(s_gate[p])
                else:                                           # a wrong draft: stand-in routing, the draft's own cf
                    c = dc[p - A] if p - A < m else -1.0
                    rows.append((p + 97) % N); cf.append(c); sk_all.append((p - a) % 2 == 0); sk_gate.append(c >= 0.999)
            cf = np.array(cf); kq = np.where(cf >= 0.99, 2, 10)
            for l in range(48):
                skip = sk_all if 8 <= l <= 11 else sk_gate if 12 <= l <= 35 else [False] * len(rows)
                for s, kk, use_skip in (("exact", None, False), ("q1", kq, False), ("map", None, True), ("mapq1", kq, True)):
                    u = set()
                    for j, p in enumerate(rows):
                        if use_skip and skip[j]: continue
                        u.update(ids[p, l, :(10 if kk is None else kk[j])].tolist())
                    B[s] += len(u)
                if 8 <= l <= 11: ru[0] += sum(skip); ru[2] += len(rows)
                elif 12 <= l <= 35: ru[1] += sum(skip)
    gb = {s: v * 2.2 / 1e3 / nchk for s, v in B.items()}
    say(f"| {dom} | {gb['exact']:.2f} | {gb['q1']:.2f} | {gb['map']:.2f} | {gb['mapq1']:.2f} | x{gb['map'] / gb['exact']:.2f} | "
        f"x{gb['mapq1'] / gb['q1']:.2f} | {100 * ru[0] / ru[2]:.0f} % / {100 * ru[1] / (ru[2] * 6):.0f} % |")
open(os.path.join(OUT, "PFRAME_bytes.txt"), "w").write("\n".join(lines) + "\n")
