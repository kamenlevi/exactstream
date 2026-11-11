"""CPU unit test of xpt.attach / _cycle on a synthetic log (exit order, nested expert calls, submits, a commit forward)."""
import math
import sys

sys.path.insert(0, "/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts/stepb")
import xpt

xpt.reset()
stamps, t = [], [100.0]


def at(dt):
    t[0] += dt
    return t[0]


for cyc in range(2):
    stamps.append(at(0.0001))
    xpt.CPUST.append(cyc * 0.05)
    xpt.LOG.append(("ae", at(0.0005), 0.05, 0.04, 0))       # the drafter's submit: before any trunk layer (cycle only)
    for fw, rows in (("check", 16), ("commit", 14)):
        for L in range(48):
            t_lay = at(0.0002)
            t_xp = at(0.0001)
            if fw == "check":                                # exit order: the expert call (inner) first, then its layer
                xpt.LOG.append(("xp", t_xp, 0.30, 0.25, rows))
            xpt.LOG.append(("lay", t_lay, 0.80, 0.70, rows))
            if L % 2 == 1:
                xpt.LOG.append(("ae", at(0.0001), 0.10, 0.09, 0))
        at(0.001)
    xpt.LOG.append(("dlay", at(0.0001), 5.0, 5.0, 16))       # a drafter layer sharing the class: ignored
stamps.append(at(0.0001))
xpt.CPUST.append(0.1)

pc = {"ck": [60.0, 61.0]}
err = xpt.attach(pc, stamps)
ok = not err
for i in range(2):
    ok &= pc["ck_xn"][i] == 48 and math.isclose(pc["ck_lw"][i], 48 * 0.8) and math.isclose(pc["ck_xw"][i], 48 * 0.3)
    ok &= math.isclose(pc["ck_xc"][i], 48 * 0.25) and math.isclose(pc["ck_aw"][i], 24 * 0.1)
    ok &= pc["cm_f"][i] == 1 and math.isclose(pc["cm_lw"][i], 48 * 0.8) and pc["cm_xn"][i] == 0
    ok &= math.isclose(pc["cy_aw"][i], 0.05 + 48 * 0.1) and pc["ck_rows"][i] == 16
    ok &= math.isclose(pc["cy_cpu"][i], 50.0) and pc["cy_wall"][i] > 0
print("err:", err or "-")
print({k: [round(v, 3) for v in pc[k]] for k in sorted(pc)})
print("TEST_XPT", "PASS" if ok else "FAIL")
