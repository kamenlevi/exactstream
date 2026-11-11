import os
B = "/Users/kamen/FableGod/far/build_1105/"
def edit(fn, pairs):
    s = open(B + fn).read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, old[:70], s.count(old)); s = s.replace(old, new)
    open(B + fn, "w").write(s); print("edited", fn)
edit("nw_patch.py", [(
'''ABLATIONS["qom_0_live_ball_c208_nwz"] = nw_mask(_NWD + "qom_0_live_ball_c208", 1, zero=True)   # self-check i: the twin's masks through the never-wait path
''',
'''ABLATIONS["qom_0_live_ball_c266"] = q1_union(_NWD + "qom_0_live_ball_c266")   # tier 1b: the twin on 266 uniform slots (the engine's by-day base-pool exposure, ZERO 5d)
ABLATIONS["qom_0_live_ball_c266_nw1"] = nw_mask(_NWD + "qom_0_live_ball_c266_nw1", 1)   # tier 1b: nw1 at 266 slots
ABLATIONS["qom_0_live_ball_c208_nwz"] = nw_mask(_NWD + "qom_0_live_ball_c208", 1, zero=True)   # self-check i: the twin's masks through the never-wait path
''')])
edit("test_nw.py", [(
'''       "qom_0_live_ball_c208_nwz": dict(rmask=S + "qom_0_live_ball_c208", renorm=False, nw=1, nw_zero=True),
''',
'''       "qom_0_live_ball_c266": dict(rmask=S + "qom_0_live_ball_c266", renorm=False),
       "qom_0_live_ball_c266_nw1": dict(rmask=S + "qom_0_live_ball_c266_nw1", renorm=False, nw=1),
       "qom_0_live_ball_c208_nwz": dict(rmask=S + "qom_0_live_ball_c208", renorm=False, nw=1, nw_zero=True),
''')])
edit("check_install.py", [(
'''        "qom_0_live_ball_c208_nwz": ("qom_0_live_ball_c208", 1, True),
''',
'''        "qom_0_live_ball_c266": ("qom_0_live_ball_c266", None, False),
        "qom_0_live_ball_c266_nw1": ("qom_0_live_ball_c266_nw1", 1, False),
        "qom_0_live_ball_c208_nwz": ("qom_0_live_ball_c208", 1, True),
''')])
edit("check_masks.py", [(
'''                  ("qom_0_split0.99_live_ball_c208", [(ORIG, "NewIdeas' builder, unmodified")]),
''',
'''                  ("qom_0_live_ball_c266", [(ORIG, "NewIdeas' builder, unmodified")]),
                  ("qom_0_split0.99_live_ball_c208", [(ORIG, "NewIdeas' builder, unmodified")]),
'''), (
'''        ("qom_0_live_ball_c208_nw1s", "qom_0_live_ball_c208", 1, "s"), ("qom_0_live_ball_nw1", "qom_0_live_ball", 1, ""),
''',
'''        ("qom_0_live_ball_c208_nw1s", "qom_0_live_ball_c208", 1, "s"), ("qom_0_live_ball_nw1", "qom_0_live_ball", 1, ""),
        ("qom_0_live_ball_c266_nw1", "qom_0_live_ball_c266", 1, ""),
''')])
edit("fri_extra_nw.txt", [(
'''# Tier 2, if the night has room (uncomment):''',
'''# Tier 1b, recommended if 4 more GPU minutes exist (uncomment): the same pair on 266 uniform slots, the engine's own
# by-day base-pool exposure by ZERO's estimate (LOSSY350_ZERO.md 5d): it settles a "pass only at the low pool" outcome.
#qom_0_live_ball_c266+code 4 10:00 qom_0_live_ball+code qom_0_live_ball_c208+code
#qom_0_live_ball_c266_nw1+code 4 10:00 qom_0_live_ball_c266+code qom_0_live_ball_c208_nw1+code
# Tier 2, if the night has room (uncomment):''')])
edit("nw_read.py", [(
'''TP, NWP = "qom_0_live_ball+code", "qom_0_live_ball_nw1+code"
''',
'''TP, NWP = "qom_0_live_ball+code", "qom_0_live_ball_nw1+code"
T266, NW266 = "qom_0_live_ball_c266+code", "qom_0_live_ball_c266_nw1+code"   # tier 1b: the engine's by-day exposure (ZERO 5d)
'''), (
'''    a = pair(NW1, T208); p = pair(NWP, TP); r = pair(R1, T208); s = pair(STK, TS)
''',
'''    a = pair(NW1, T208); p = pair(NWP, TP); r = pair(R1, T208); s = pair(STK, TS)
    m = pair(NW266, T266, quiet=True)
'''), (
'''    if p is not None: V.append(f"nw1 at the published pool: {'clean' if clean_pub else 'not clean'}: {p[0]:.2f} %, D {p[2]:+.2f} (clean = D <= +0.15)")
''',
'''    if p is not None: V.append(f"nw1 at the published pool: {'clean' if clean_pub else 'not clean'}: {p[0]:.2f} %, D {p[2]:+.2f} (clean = D <= +0.15)")
    if m is not None:
        v = "PASS" + (" (clean)" if m[2] <= 0.15 else "") if (m[0] <= 1.34 and m[2] <= 0.40) else "KILL" if (m[0] > 2.0 or m[2] > 0.80) else "BETWEEN"
        V.append(f"nw1 at 266 slots (the engine's by-day base-pool exposure by ZERO's estimate; nw1's lines): {v}: {m[0]:.2f} %, D {m[2]:+.2f}")
'''), (
'''    if sc == "FAIL" or any(x is not None and x[4] for x in (a, p, r, s)):''',
'''    if sc == "FAIL" or any(x is not None and x[4] for x in (a, p, r, s, m)):'''), (
'''  pool-conditional PASS: nw1 not PASS at 208 slots but clean (D <= +0.15) at the published pool
''',
'''  pool-conditional PASS: nw1 not PASS at 208 slots but clean (D <= +0.15) at the published pool
  nw1 at 266 slots (tier 1b, if run): nw1's own lines; 266-290 slots is where the engine's by-day base pool sits
''')])
