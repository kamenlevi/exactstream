"""DUALDRAFT read-out (MTP Head; REPORT 'DUALDRAFT'): kept per check of the joint-chain ensemble (L1-20 + E2's x2) vs L1-20
(9.57 / 7.00) and x2 alone, the self-check, and the teacher-forced overlap at HPD-0's anchors: q = P(x2 wrong | L1-20 wrong),
either-right, and the higher-tcf pick per position. Output: data/dd_read.txt (appended)."""
import os, re
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
def walk(tag):
    f = os.path.join(TD, f"d02b_walk{tag}.txt")
    if not os.path.exists(f):
        return {}, None
    s = open(f).read()
    k = {n: float(v) for n, v in re.findall(r"\s+(code|math) \([^)]*\): \d+ samples, \d+ checks; kept per check ([0-9.]+)", s)}
    sh = re.findall(r"DUALDRAFT: (\d+) of (\d+) reached depths proposed by the second drafter \(([0-9.]+) %\)", s)
    return k, sh
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
L1, _ = walk(""); X2, _ = walk("_e2x2"); SC, shs = walk("_dd_self"); EN, she = walk("_dd_x2")
say("DUALDRAFT read-out (block 16, official loop; joint chain: the more confident drafter's pick per depth)")
say(f"  references: L1-20 code {L1.get('code')} / math {L1.get('math')}; x2 alone code {X2.get('code')} / math {X2.get('math')}")
ok = SC.get("code") == L1.get("code") and SC.get("math") == L1.get("math") and all(float(x[2]) == 0.0 for x in (shs or []))
say(f"  self-check (L1-20 twice): code {SC.get('code')} / math {SC.get('math')}; second-drafter depths "
    f"{[x[2] + ' %' for x in (shs or [])]} -> {'OK' if ok else 'FAIL'}")
if EN:
    say(f"  ensemble L1-20 + x2: code {EN.get('code')} (x{EN['code'] / L1['code']:.3f} vs L1-20, x{EN['code'] / X2['code']:.3f} vs x2), "
        f"math {EN.get('math')} (x{EN['math'] / L1['math']:.3f}); depths from x2: {[x[2] + ' %' for x in (she or [])]}")
    r = EN["code"] / L1["code"]
    v = "VOID (self-check failed)" if not ok else ("PASS" if r >= 1.08 else "KILL" if r < 1.04 else "BETWEEN")
    say(f"  LINE (code ensemble >= x1.08 L1-20 PASS, < x1.04 KILL): x{r:.3f} -> {v}")
for st in ("code", "math"):
    a, b = os.path.join(TD, f"d02b_walk_hpd0_{st}.npz"), os.path.join(TD, f"d02b_walk_dd_x2cf_{st}.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    A, B = np.load(a), np.load(b)
    n = bw = lw = both = ei = pk = 0
    for i in range(len(A["names"])):
        ids, off = A[f"{i}_ids"], int(A[f"{i}_off"])
        for c, (k, m, an) in enumerate(A[f"{i}_checks"]):
            for j in range(int(m)):
                t = int(ids[int(an) - off + 1 + j])
                l_, x_ = int(A[f"{i}_ttok"][c, j]), int(B[f"{i}_ttok"][c, j])
                n += 1; lw += l_ != t; bw += x_ != t; both += (l_ != t) and (x_ != t); ei += (l_ == t) or (x_ == t)
                pick = l_ if A[f"{i}_tcf"][c, j] >= B[f"{i}_tcf"][c, j] else x_
                pk += pick == t
    say(f"  overlap ({st}, teacher-forced at HPD-0's anchors, {n} positions): L1-20 right {1 - lw / n:.4f}, x2 right {1 - bw / n:.4f}, "
        f"q = P(x2 wrong | L1-20 wrong) {both / max(lw, 1):.3f} (hpdU 0.61 / hpdH 0.66 per the orchestrator), either right "
        f"{ei / n:.4f}, higher-cf pick {pk / n:.4f}")
open(os.path.join(TD, "dd_read.txt"), "a").write("\n".join(lines) + "\n\n")
