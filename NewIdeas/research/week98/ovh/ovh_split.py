"""Overhead split from BigMoE's s98b loop records (data/r57/bt98/<arm>_checks.json; read-only).
Per key, per check i >= 2: cycle_i = c0[i+1] - c0[i] (ms between check starts) = ck[i] + drf[i+1] + rest_i,
where rest = commit + host-side bookkeeping between the check's end and the next draft's start.
Also ck vs bld + wt (build + wait), and tc (main-thread CPU [build, wait]) when R57_TCPU is on."""
import json, os, statistics as st, sys

D = os.path.expanduser("~/BigMoE/data/r57/bt98")
arms = sys.argv[1:] or ["E1hs", "E2sf", "E3sf", "E4pk", "L1off", "L2on"]


def med(x):
    return st.median(x) if x else float("nan")


for arm in arms:
    p = os.path.join(D, arm + "_checks.json")
    if not os.path.exists(p):
        print(arm, "missing"); continue
    recs = json.load(open(p))
    cyc, ck, drf, rest, bld, wt, tcb, tcw, rows, kept = [], [], [], [], [], [], [], [], [], []
    for r in recs:
        c0, k, d = r.get("c0") or [], r.get("ck") or [], r.get("drf") or []
        b, w, t = r.get("bld") or [], r.get("wt") or [], r.get("tc") or []
        n = min(len(c0), len(k), len(d))
        for i in range(2, n - 1):
            cy = c0[i + 1] - c0[i]
            cyc.append(cy); ck.append(k[i]); drf.append(d[i + 1]); rest.append(cy - k[i] - d[i + 1])
            if i < len(b): bld.append(b[i])
            if i < len(w): wt.append(w[i])
            if i < len(t) and isinstance(t[i], (list, tuple)) and len(t[i]) == 2:
                tcb.append(t[i][0]); tcw.append(t[i][1])
            rows.append((r.get("rows") or [0] * n)[i]); kept.append((r.get("kept") or [0] * n)[i])
    if not cyc:
        print(arm, "no usable checks"); continue
    print(f"{arm}: keys {len(recs)}, checks {len(cyc)} (i >= 2), rows med {med(rows)}, kept mean {sum(kept) / len(kept):.2f}")
    print(f"   median ms: cycle {med(cyc):.2f} = check {med(ck):.2f} + draft {med(drf):.2f} + rest {med(rest):.2f}"
          f"  (means: {st.mean(cyc):.2f} = {st.mean(ck):.2f} + {st.mean(drf):.2f} + {st.mean(rest):.2f})")
    if bld and wt:
        print(f"   check = build {med(bld):.2f} + wait {med(wt):.2f} (+ {med(ck) - med(bld) - med(wt):.2f} other, medians)")
    if tcb:
        print(f"   main-thread CPU per check: build {med(tcb):.2f}, wait {med(tcw):.2f}")
