"""EARLYTAP read-out (MTP Head, Mon 11-02; REPORT 'EARLYTAP'): kept per check (block 16, official loop) of the three
retrained drafters (taps <= 45 control, <= 35, <= 25) from data/d02b_walk_et_<v>.txt; lines on HumanEval+ vs the control.
Output: data/et_read.txt (appended)."""
import os, re
TD = os.path.expanduser("~/TrainLab/data")
R = {}
for v in ("h5", "h4", "h3"):
    f = os.path.join(TD, f"d02b_walk_et_{v}.txt")
    if os.path.exists(f):
        for l in open(f):
            m = re.match(r"\s+(\w+): (\d+) samples, (\d+) checks; kept per check ([0-9.]+)", l)
            if m:
                R.setdefault(m.group(1), {})[v] = float(m.group(4))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("EARLYTAP read-out (DFlash2 retrained with D-09's protocol on L1-20's data; block 16; taps zeroed before the frozen fc)")
for st, A in R.items():
    c = A.get("h5")
    say(f"  {st}: " + ", ".join(f"{v} {A[v]:.2f}" + (f" (x{A[v] / c:.3f})" if c and v != 'h5' else "") for v in ("h5", "h4", "h3") if v in A))
if "code" in R and {"h5", "h4", "h3"} <= set(R["code"]):
    c = R["code"]; r4, r3 = c["h4"] / c["h5"], c["h3"] / c["h5"]
    say(f"  control vs L1-20 itself on code: {c['h5']:.2f} vs 9.57 (this recipe's seed noise)")
    v4 = "KILL" if r4 < 0.93 else ("ALIVE (<= 35 pipeline)" if r4 >= 0.95 else "BETWEEN")
    v3 = "ALIVE (<= 25 pipeline)" if r3 >= 0.90 else "not alive"
    say(f"  LINE (HumanEval+ vs the control; h4 >= 0.95 alive, < 0.93 KILL; h3 >= 0.90 alive): h4 x{r4:.3f} -> {v4}; h3 x{r3:.3f} -> {v3}")
    say(f"  report only (cloud task 15, via the orchestrator): kept >= 0.97 x the control with early taps: h4 x{r4:.3f} "
        f"({'yes' if r4 >= 0.97 else 'no'}), h3 x{r3:.3f} ({'yes' if r3 >= 0.97 else 'no'}); the verdict stays on the lines above")
open(os.path.join(TD, "et_read.txt"), "a").write("\n".join(lines) + "\n\n")
