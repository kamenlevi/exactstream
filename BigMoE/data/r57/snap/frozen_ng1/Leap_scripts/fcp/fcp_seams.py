"""PLAN 220 (4) read-out (Leap, Sat 2026-11-07): the cycle's GPU-idle seams from an FCP_SEAMS run (fcps). Per cycle
(c0 .. next c0), with every mx.eval / async_eval timed:
  A  the draft's last blocking eval -> the check's first submit (after c1): post-draft Python + the first SB group's build
  B  the check's eval (ce) -> the commit's submit (accept logic + the commit build)
  C  the commit sent (c2) -> the next draft's first submit (the GPU runs the commit's replay, if any, then idles)
A blocking eval drains the queue, so A and B are GPU idle; C is idle minus the commit's GPU work (~0.18 ms on a full
accept, ~2.9 on a partial: FCP's commit split). usage: python fcp_seams.py [arm] (default fcps)"""
import collections
import json
import os
import statistics as S
import sys

D = os.path.expanduser("~/Leap/data/fcp")
arm = sys.argv[1] if len(sys.argv) > 1 else "fcps"
P = json.load(open(f"{D}/{arm}_fcp.json"))
C = json.load(open(f"{D}/r57/{arm}_checks.json"))
keys = [c["key"] for c in C]
part = {}                                                 # (run, chk) -> partial accept?
for i, c in enumerate(C):
    for j, (k, r) in enumerate(zip(c["kept"], c["rows"])):
        part[(3 + i, j)] = k < r
ev = sorted(P["seams"], key=lambda x: x[1])
mk = sorted(P["marks"], key=lambda x: x[1])
med = lambda xs: round(S.median(xs), 2) if xs else None
c0s = [m for m in mk if m[0] == "c0"]
R = collections.defaultdict(list)
callers = collections.Counter()
SA = collections.Counter()
SB = collections.Counter()
for i in range(len(c0s) - 1):
    t0, t9 = c0s[i][1], c0s[i + 1][1]
    inner = [m for m in mk if t0 <= m[1] < t9]
    c1 = next((m for m in inner if m[0] == "c1"), None)
    ce = next((m for m in inner if m[0] == "ce"), None)
    c2 = next((m for m in inner if m[0] == "c2"), None)
    if not (c1 and ce and c2):
        continue
    run, chk = c1[2], c1[3]
    if chk < int(os.environ.get("FCP_SEAM_CHK0", "1")) or (run, chk) not in part:
        continue                                          # check 0 (first-check effects) and warm-ups out
    pf = "partial" if part[(run, chk)] else "full"
    dr = [e for e in ev if t0 <= e[1] and e[2] <= c1[1] and e[0] == "mx.eval"]
    sub = [e for e in ev if e[1] >= c1[1]]
    if dr and sub:
        R[("A", pf)].append(1e3 * (sub[0][1] - dr[-1][2]))
        R[("A1 draft end -> c1", pf)].append(1e3 * (c1[1] - dr[-1][2]))
        R[("A2 c1 -> check submit", pf)].append(1e3 * (sub[0][1] - c1[1]))
        R[("D c0 -> draft end", pf)].append(1e3 * (dr[-1][2] - t0))
        rb0 = next((m for m in inner if m[0] == "rb0" and m[1] <= c1[1]), None)
        rb1 = next((m for m in inner if m[0] == "rb1" and m[1] <= c1[1]), None)
        if rb0 and rb1:
            R[("RB the draft's readbacks (GPU busy)", pf)].append(1e3 * (rb1[1] - rb0[1]))
            R[("I1 readback end -> c1 (idle, Python)", pf)].append(1e3 * (c1[1] - rb1[1]))
            R[("IDLE A' = readback end -> check submit", pf)].append(1e3 * (sub[0][1] - rb1[1]))
        for ts, _r, _c, _ph, key in P["samp"]:             # where the main thread is during A (1 kHz Python samples)
            if (rb1[1] if (rb0 and rb1) else dr[-1][2]) <= ts < sub[0][1]:
                SA[(pf, " < ".join(key.split(" < ")[:2]))] += 1
        callers[("draft end", dr[-1][3])] += 1
        callers[("check submit", sub[0][3])] += 1
    cs = [e for e in ev if e[1] >= ce[1] and e[0] == "mx.async_eval"]
    if cs:
        R[("B", pf)].append(1e3 * (cs[0][1] - ce[1]))
        for ts, _r, _c, _ph, key in P["samp"]:             # where the main thread is during B
            if ce[1] <= ts < cs[0][1]:
                SB[(pf, " < ".join(key.split(" < ")[:2]))] += 1
        callers[("commit submit", cs[0][3])] += 1
    nd = [e for e in ev if e[1] >= c2[1] and e[1] < t9 + 0.5]
    if nd:
        R[("C", pf)].append(1e3 * (nd[0][1] - c2[1]))
        callers[("next draft submit", nd[0][3])] += 1
    R[("cycle", pf)].append(1e3 * (t9 - t0))
print(f"== {arm}: GPU-idle seams per cycle (ms, medians; checks >= 1 of the measured prompts {keys})")
for s_ in ("D c0 -> draft end", "A", "A1 draft end -> c1", "A2 c1 -> check submit", "RB the draft's readbacks (GPU busy)",
           "I1 readback end -> c1 (idle, Python)", "IDLE A' = readback end -> check submit", "B", "C", "cycle"):
    print(f"  {s_:5s} partial {med(R[(s_, 'partial')])} (n {len(R[(s_, 'partial')])})   full {med(R[(s_, 'full')])} (n {len(R[(s_, 'full')])})")
n_ = {pf: len(R[("A", pf)]) for pf in ("partial", "full")}
print("  main thread during A (samples per cycle ~ ms): " + "; ".join(f"{k[0]} {v / max(1, n_[k[0]]):.1f} {k[1]}" for k, v in SA.most_common(10)))
print("  main thread during B (samples per cycle ~ ms): " + "; ".join(f"{k[0]} {v / max(1, n_[k[0]]):.2f} {k[1]}" for k, v in SB.most_common(12)))
print("  callers: " + "; ".join(f"{k[0]} @ {k[1]} x{v}" for k, v in callers.most_common(8)))
