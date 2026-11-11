#!/usr/bin/env python3 -I
"""budget_c4.py (Fable C, mission C4): the union arms against the 350 budget, per 16-row-block check, from replay_c4.out.
Cycle = chain + exposed SSD; chain = dense + experts + draft + commit + CPU + rest (budget_c3's model, calibrated on
HumanEval/68: 183 modelled vs 187 ms measured).
  experts = 48 x (calls x 0.094 + 0.01015 x groups) x kernel   [EXPERT_KERNEL fit, (expert, tier) groups per layer from the
           replay at live rows; calls = 2 (OOO's two dispatches) or 1; kernel vs NAX6 at S 16 (Leap, layer 20 / per check):
           NAX6 1.0, NAX6-N 0.826 (+21.1 %), NAX6-H 0.917 (+9.1 %), NAX6-NH 0.811 (+23.3 %), 3c 1.17 (per check, DAY9
           11:05; 1.69 at the kernel), 3c-N 1.17 x 1.043]
  SSD transfer = MB per kept token x kept / 11.7 GB/s; the replay's code column is at kept 5.9 per check, scaled to the
           budget's kept with (kept / 5.9) ** -0.5 [G; SSD_WALL's measured x0.965 for +7 % kept]; exposed = (1 - h) x
           transfer, h = 0.38 measured today (c45l_256) / 0.6 target [G].
  terms: expected = dense FAST 17.2 (HC8) / EXACT 23, draft 4.5, commit 2.0, CPU 2.4, rest 1.5 (today: draft 12.0 [K push]);
         targets = dense 15 / 20, draft 2.8, commit 0.5.
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I budget_c4.py | tee budget_c4.out"""
import re

KERN = {"NAX6": 1.0, "NAX6-N": 0.826, "NAX6-H": 0.917, "NAX6-NH": 0.811, "3c": 1.17, "3c-N": 1.17 * 1.043}
SIZES = ["M-17 C256 42.0GB", "joint C256+2 44.1GB", "M-17 C320 50.5GB", "joint C320+2 52.6GB"]
KEPT0 = 5.9            # the replay's code kept per check

# parse replay_c4.out: arm -> size -> dom -> (MB, groups, e/row)
R = {}
arm = None
for line in open("replay_c4.out"):
    if line.startswith("-- "):
        arm = line[3:].strip(); R[arm] = {}
    elif arm and line.startswith("   ") and "|" in line:
        size = line[3:23].strip()
        R[arm][size] = {}
        for dom, mb, g, e in re.findall(r"(code|math|all)\s+([\d.]+) MB \(x[\d.]+;[^)]*\) g\s+([\d.]+) \(x[\d.]+\) e/row\s+([\d.]+)", line):
            R[arm][size][dom] = (float(mb), float(g), float(e))


def experts_ms(groups, kernel, calls=2, tm=1.0):
    return 48 * (calls * 0.094 + 0.01015 * groups * tm) * KERN[kernel]


def cycle(kept, mb, dense, exp_ms, draft, commit, h, cpu=2.4, rest=1.5):
    chain = dense + exp_ms + draft + commit + cpu + rest
    transfer = mb * (kept / KEPT0) ** -0.5 * kept / 11.7
    exposed = transfer * (1 - h)
    return chain, transfer, exposed, chain + exposed, 1000 * kept / (chain + exposed)


print("== A. code, per check: the union arms x RAM points; expert ms (groups at live rows), SSD transfer / exposed, cycle, tok/s ==")
print("   kept 10.0 (today's code) and 12.6 (DFlash3); 'expected' terms (h 0.38, dense 17.2, draft 4.5, commit 2.0) and 'targets' (h 0.6, 15 / 2.8 / 0.5)")
ARMS = [("EXACT", "3c", 2, 1.0, 23.0, 20.0), ("EXACT", "NAX6-N", 2, 1.0, 23.0, 20.0), ("EXACT", "3c-N", 2, 1.0, 23.0, 20.0),
        ("Q1-UNION 0.99", "NAX6-NH", 2, 1.0, 17.2, 15.0), ("Q1-UNION 0.9", "NAX6-NH", 2, 0.85, 17.2, 15.0),
        ("U3 = Q1U 0.9 + t3 sure", "NAX6-NH", 2, 1.0, 17.2, 15.0), ("OEA k8", "NAX6-NH", 2, 1.0, 17.2, 15.0),
        ("HYB 0.9 u8 + t3", "NAX6-NH", 2, 1.0, 17.2, 15.0), ("REACH + t3 sure", "NAX6-NH", 2, 1.0, 17.2, 15.0),
        ("HYB-MASS 0.9 + t3", "NAX6-NH", 2, 1.0, 17.2, 15.0), ("HYB-MISS 4 + t3", "NAX6-NH", 2, 1.0, 17.2, 15.0),
        ("HYB-COLD 4/0.05 + t3", "NAX6-NH", 2, 1.0, 17.2, 15.0), ("HYB-COLD 4/0.05 + t3", "NAX6-NH", 1, 1.0, 17.2, 15.0)]
for size in ("joint C256+2 44.1GB", "M-17 C320 50.5GB"):
    print(f"-- {size}")
    for arm, kern, calls, tm, de, dt in ARMS:
        if arm not in R or size not in R[arm]:
            continue
        mb, g, e = R[arm][size]["code"]
        ems = experts_ms(g, kern, calls, tm)
        for kept in (10.0, 12.6):
            ce = cycle(kept, mb, de, ems, 4.5, 2.0, 0.38); ct = cycle(kept, mb, dt, ems, 2.8, 0.5, 0.6)
            print(f"   {arm:22s} {kern:7s} {'1-call' if calls == 1 else '2-call'}{' tm' if tm < 1 else '   '} kept {kept:4.1f} | groups {g:4.1f} experts {ems:4.1f} ms | MB/kept {mb:4.1f}->{mb * (kept / KEPT0) ** -0.5:4.1f}"
                  f" | expected {ce[3]:5.1f} = chain {ce[0]:4.1f} + exposed {ce[2]:4.1f} (transfer {ce[1]:4.1f}) -> {ce[4]:4.0f} tok/s | targets {ct[3]:5.1f} = {ct[0]:4.1f} + {ct[2]:4.1f} -> {ct[4]:4.0f}")

print("\n== B. what 350 needs on code: cycle <= 28.6 ms at kept 10, 36.0 at 12.6, 38.6 at 13.5; chain floors with the SSD fully hidden ==")
for size in ("M-17 C320 50.5GB",):
    for arm, kern, calls, tm in (("HYB 0.9 u8 + t3", "NAX6-NH", 1, 0.85), ("HYB-COLD 4/0.05 + t3", "NAX6-NH", 1, 0.85), ("HYB-MISS 4 + t3", "NAX6-NH", 1, 0.85)):
        g = R[arm][size]["code"][1]; ems = experts_ms(g, kern, calls, tm)
        ch = 15 + ems + 2.8 + 0.5 + 2.4 + 1.5
        print(f"   {arm:22s} 1-call + tier merge: experts {ems:4.1f}; chain at targets {ch:4.1f} ms -> kept 10: {1000 * 10 / ch:4.0f}, kept 12.6: {1000 * 12.6 / ch:4.0f}, kept 13.5: {1000 * 13.5 / ch:4.0f} tok/s if the SSD hides;"
              f" needs {ch - 28.6:+5.1f} ms at kept 10 / {ch - 36.0:+5.1f} at 12.6")

print("\n== C. chat, per check (IFEval 3.7 rows, kept ~3 [K LIVE_read]): groups at ~4 rows [P: EXACT 16-row 77 -> 4-row ~30], chat bytes from NewIdeas' capture (IFEval / multi vs EXACT) ==")
# chat MB per kept token at C 256 + M-17: band45 capture mode EXACT on chat ~ the math column's level [G]; arms' chat factors [K NewIdeas 2d]
CHAT = {"EXACT": 1.0, "OEA k8": 1 - 0.193, "Q1-UNION 0.99": 1 - 0.095, "Q1 0.9 (BETWEEN on chat)": 1 - 0.18, "HYB 0.9 u8 (BETWEEN on chat)": 1 - 0.324}
for size, mb_exact in (("joint C256+2 44.1GB", R["EXACT"]["joint C256+2 44.1GB"]["math"][0]), ("M-17 C320 50.5GB", R["EXACT"]["M-17 C320 50.5GB"]["math"][0])):
    print(f"-- {size} (EXACT chat proxy = the replay's math column {mb_exact:.1f} MB per kept token at kept 6.2 per check)")
    for arm, f in CHAT.items():
        g = 30.0 * (0.85 if "OEA" in arm else 1.0) * (0.79 if "0.99" in arm else 1.0) * (0.66 if "0.9 (" in arm or "u8" in arm else 1.0)
        kern = "3c" if arm == "EXACT" else "NAX6-NH"; de = 23.0 if arm == "EXACT" else 17.2
        ems = experts_ms(g, kern, 2, 1.0)
        kept = 3.0
        mb = mb_exact * f
        chain = de + ems + 4.5 + 2.0 + 2.4 + 1.5
        transfer = mb * (kept / 6.2) ** -0.5 * kept / 11.7
        exposed = transfer * 0.62
        print(f"   {arm:30s} experts {ems:4.1f} | MB/kept {mb:4.1f} | cycle {chain + exposed:5.1f} = chain {chain:4.1f} + exposed {exposed:4.1f} -> {1000 * kept / (chain + exposed):4.0f} tok/s at kept 3; 350 needs kept >= {0.35 * (chain + exposed):4.1f} per check")
