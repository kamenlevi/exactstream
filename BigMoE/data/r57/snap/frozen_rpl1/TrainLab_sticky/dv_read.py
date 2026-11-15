"""DIVERSE2 read-out (MTP Head; REPORT 'DIVERSE2'): per member (dv_m0 control, dv_mw x4 on L1-20's misses, dv_mws + live block
starts), teacher-forced at HPD-0's anchors on the held-out code (24 HumanEval+) and math (60 GSM8K): its own per-position
accuracy, q = P(member wrong | L1-20 wrong), either-right, the higher-cf pick; and the joint-chain ensemble with L1-20 kept per
check (d02b_walk_dve_<v>.txt). LINES (code): PASS if mw or mws has q <= 0.50 with own accuracy >= 0.90; KILL if both have
q >= 0.60; BETWEEN otherwise. Output: data/dv_read.txt (appended)."""
import os, re
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
def kept(tag):
    f = os.path.join(TD, f"d02b_walk{tag}.txt")
    if not os.path.exists(f):
        return {}
    return {n: float(v) for n, v in re.findall(r"\s+(code|math) \([^)]*\): \d+ samples, \d+ checks; kept per check ([0-9.]+)", open(f).read())}
def overlap(st, v):
    a, b = os.path.join(TD, f"d02b_walk_hpd0_{st}.npz"), os.path.join(TD, f"d02b_walk_dv_{v}_{st}.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    A, B = np.load(a), np.load(b)
    n = lw = bw = both = ei = pk = 0
    for i in range(len(A["names"])):
        ids, off = A[f"{i}_ids"], int(A[f"{i}_off"])
        assert np.array_equal(A[f"{i}_checks"][:, 1:], B[f"{i}_checks"][:, 1:]), (st, v, i)   # same anchors and m (k differs)
        for c, (k, m, an) in enumerate(A[f"{i}_checks"]):
            for j in range(int(m)):
                t = int(ids[int(an) - off + 1 + j])
                l_, x_ = int(A[f"{i}_ttok"][c, j]), int(B[f"{i}_ttok"][c, j])
                n += 1; lw += l_ != t; bw += x_ != t; both += (l_ != t) and (x_ != t); ei += (l_ == t) or (x_ == t)
                pk += (l_ if A[f"{i}_tcf"][c, j] >= B[f"{i}_tcf"][c, j] else x_) == t
    return dict(n=n, l1=1 - lw / n, own=1 - bw / n, q=both / max(lw, 1), either=ei / n, pick=pk / n)
L1 = kept("")
say("DIVERSE2 read-out (teacher-forced at HPD-0's anchors; ensembles = the joint-chain cf pick with L1-20, official loop)")
res = {}
for v, nm in (("m0", "M2-0 control"), ("mw", "M2-W x4 on misses"), ("mws", "M2-WS + live starts")):
    for st in ("code", "math"):
        o = overlap(st, v)
        if o:
            say(f"  {nm:22s} {st}: own {o['own']:.4f} (L1-20 {o['l1']:.4f}), q {o['q']:.3f}, either {o['either']:.4f}, cf pick {o['pick']:.4f} ({o['n']} positions)")
            if st == "code":
                res[v] = o
    e = kept(f"_dve_{v}")
    if e and L1:
        say(f"  {nm:22s} ensemble kept: code {e.get('code')} (x{e['code'] / L1['code']:.3f}), math {e.get('math')} (x{e['math'] / L1['math']:.3f})")
w = [res[v] for v in ("mw", "mws") if v in res]
if len(w) == 2:
    ok = any(o["q"] <= 0.50 and o["own"] >= 0.90 for o in w)
    v = "PASS" if ok else ("KILL" if all(o["q"] >= 0.60 for o in w) else "BETWEEN")
    say(f"  LINE (code: mw or mws q <= 0.50 with own >= 0.90 PASS; both q >= 0.60 KILL): mw q {w[0]['q']:.3f} / own {w[0]['own']:.4f}, "
        f"mws q {w[1]['q']:.3f} / own {w[1]['own']:.4f} -> {v}")
open(os.path.join(TD, "dv_read.txt"), "a").write("\n".join(lines) + "\n\n")
