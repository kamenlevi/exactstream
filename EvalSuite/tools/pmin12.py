"""PMIN12 verdict (docs/PREREG_pmin12_2026-11-01.md): b12llff4p (B12_PMIN 0.5) vs b12llff4 (the control: today's 12 GB
default) on decision 5's 239 rows (core 210 + runner 24's 29 agent rows).
PASS = decision 4's rule on math / IFEval / HumanEval+ (tools/d4.py on results/report_b12llff4p_vs_b12llff4.txt, made
by the runner first) AND every b12llff4p process and row <= 11.95 GB.
Information: token-identical rows per set (vs the control; the control vs b12llff = decision 5's rows), verify rows per
pass (the chain cut), tok/s and SSD MB per token per set, the control's footprint.
Usage: .venv/bin/python tools/pmin12.py > results/pmin12.txt (tools/pmin12.py A B: the same on other arms, a dry run)"""
import os, statistics as S, subprocess, sys
from r41c import CORE, LINE, ROOT, first_diff, memory, order, rows

P, C, D5 = (*sys.argv[1:3], "b12llff") if len(sys.argv) > 2 else ("b12llff4p", "b12llff4", "b12llff")   # args: a dry run
SETS = CORE + ("agent",)
keys = order(CORE) + order(("agent",))[:29]
A, B, F = rows(P), rows(C), rows(D5)
rep = os.path.join(ROOT, "results", f"report_{P}_vs_{C}.txt")
miss = {v: [k for k in keys if k not in R] for v, R in ((P, A), (C, B))}
print(f"PMIN12: {P} (B12_PMIN 0.5) vs {C} (control) on decision 5's {len(keys)} rows; missing {P} {len(miss[P])}, "
      f"{C} {len(miss[C])}\n")

print("1. decision 4's rule on the core sets (tools/d4.py)")
d4 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "d4.py"), rep], capture_output=True, text=True)
print("   " + "\n   ".join(d4.stdout.strip().splitlines() or [f"no report: {d4.stderr.strip()[-200:]}"]))

print(f"\n2. footprint (GB; a row's peak = max of its sampled max and the lifetime peak), line {LINE}")
top = memory(P)
print("   the control (information):")
memory(C)

ok = d4.returncode == 0 and top <= LINE and not miss[P] and not miss[C]
print(f"\nPMIN12: {'PASS' if ok else 'INCOMPLETE' if (miss[P] or miss[C]) else 'FAIL'} (decision 4: "
      f"{d4.stdout.strip().splitlines()[-1] if d4.stdout.strip() else 'no report'}; every {P} process <= {LINE} GB: "
      f"{top <= LINE} ({top:.2f}))")

print("\n3. information")
for s in SETS:
    ks = [k for k in keys if k in A and k in B and A[k]["set"] == s]
    same = [k for k in ks if A[k]["completion_ids"] == B[k]["completion_ids"]]
    fd = [first_diff(A[k]["completion_ids"], B[k]["completion_ids"]) for k in ks if k not in same]
    cf = [k for k in keys if k in B and k in F and B[k]["set"] == s]
    print(f"   {s}: {P} == {C} on {len(same)}/{len(ks)} rows" + (f" (first difference at token median "
          f"{S.median(fd):.0f}, min {min(fd)})" if fd else "") + f"; {C} == {D5} (decision 5's rows) on "
          f"{sum(B[k]['completion_ids'] == F[k]['completion_ids'] for k in cf)}/{len(cf)}")
for v, R in ((P, A), (C, B)):
    out = []
    for s in SETS:
        rs = [R[k] for k in keys if k in R and R[k]["set"] == s and "ssd" in R[k]]
        n, ps, vr = (sum(r["ssd"].get(x) or 0 for r in rs) for x in ("n", "passes", "rows"))
        dec = sum(r["ssd"].get("dec_s") or 0 for r in rs)
        if n:
            out.append(f"{s} {n / ps:.2f} tokens / pass, {vr / max(ps, 1):.0f} engine rows / pass, "
                       f"{n / dec:.1f} tok/s decode, {sum(r['ssd'].get('disk_mb') or 0 for r in rs) / n:.0f} SSD MB / token")
    print(f"   {v}: " + "; ".join(out))
print("   (tok/s: shared GPU, rough, not an A/B; Breakthrough's ABBA is the speed number)")
