"""MATCH read-out (MTP Head, Mon 11-02; REPORT 'MATCH'): kept per check of DFlash2 L1-20 with / without Fable's match
override at 16 / 32 rows and the 2.5 length rule (data/d02b_walk_m_<arm>.txt), per set; lines as pre-registered.
Cost model for 2.5: ms(rows) = 88 + 7.82 x rows. Output: data/match_read.txt (appended)."""
import os, re
TD = os.path.expanduser("~/TrainLab/data")
ARMS = ["base16", "match16", "base32", "match32", "len25"]
R = {}
for a in ARMS:
    f = os.path.join(TD, f"d02b_walk_m_{a}.txt")
    if not os.path.exists(f):
        continue
    cur = None
    for l in open(f):
        m = re.match(r"\s+(\w+): (\d+) samples, (\d+) checks; kept per check ([0-9.]+)", l)
        if m:
            cur = m.group(1); R.setdefault(cur, {})[a] = {"kept": float(m.group(4)), "checks": int(m.group(3))}
        m = re.search(r"rows per check ([0-9.]+); drafted positions reached (\d+), with a match (\d+) \(([0-9.]+) %, right ([0-9.]+) %\); changed the draft (\d+) \(override right (\d+), drafter right (\d+)\)", l)
        if m and cur:
            R[cur][a].update(rows=float(m.group(1)), share=float(m.group(4)), prec=float(m.group(5)), chg=int(m.group(6)),
                             chg_ok=int(m.group(7)), drf_ok=int(m.group(8)))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
ms = lambda rows: 88 + 7.82 * rows
say("MATCH read-out (DFlash2 L1-20, official loop; kept per check by arm; ratios vs the same block without the override)")
for st, A in R.items():
    say(f"  {st}: " + ", ".join(f"{a} {A[a]['kept']:.2f}" for a in ARMS if a in A))
    for a in ("match16", "match32", "len25"):
        if a in A and "share" in A[a]:
            say(f"    {a}: positions with a match {A[a]['share']:.1f} %, right {A[a]['prec']:.1f} %; changed {A[a]['chg']} "
                f"(override right {A[a]['chg_ok']}, drafter right {A[a]['drf_ok']}); rows per check {A[a]['rows']:.2f}")
    r16 = A["match16"]["kept"] / A["base16"]["kept"] if {"match16", "base16"} <= set(A) else float("nan")
    r32 = A["match32"]["kept"] / A["base32"]["kept"] if {"match32", "base32"} <= set(A) else float("nan")
    say(f"    kept(16) x{r16:.3f}, kept(32) x{r32:.3f}")
    if {"len25", "match16"} <= set(A) and "rows" in A["len25"] and "rows" in A["match16"]:
        e25 = A["len25"]["kept"] / ms(A["len25"]["rows"]); e16 = A["match16"]["kept"] / ms(A["match16"]["rows"])
        say(f"    2.5 kept per check-ms: len25 {1e3 * e25:.2f} vs match16 {1e3 * e16:.2f} per s -> x{e25 / e16:.3f} "
            f"(PASS >= 1.10, KILL < 1.04)")
        msc = lambda rows: 36.1 + 2.98 * rows            # report only (the orchestrator): C0's SSD-hidden model, measured 11-02
        f25, f16 = A["len25"]["kept"] / msc(A["len25"]["rows"]), A["match16"]["kept"] / msc(A["match16"]["rows"])
        say(f"      (report only, C0 SSD-hidden ms = 36.1 + 2.98 x rows: x{f25 / f16:.3f}; the verdict uses the pre-registered model)")
    R[st]["r16"], R[st]["r32"] = r16, r32
if "code" in R:
    c = R["code"]
    v = "KILL" if c["r16"] < 1.03 else ("PASS" if c["r16"] >= 1.06 and c["r32"] >= 1.25 else "BETWEEN")
    sec = ", ".join(f"{s} kept(16) x{R[s]['r16']:.3f} ({'>=' if R[s]['r16'] >= 1.03 else '<'} 1.03)" for s in ("mbpp", "agent") if s in R)
    say(f"  LINE (HumanEval+: kept(16) >= x1.06 and kept(32) >= x1.25 PASS; kept(16) < x1.03 KILL): "
        f"x{c['r16']:.3f} / x{c['r32']:.3f} -> {v}; secondary: {sec}")
open(os.path.join(TD, "match_read.txt"), "a").write("\n".join(lines) + "\n\n")
