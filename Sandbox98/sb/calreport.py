"""Calibration report (results/screens/CALIBRATION.md's rule): per control copy, the paired delta vs the base s30o on
MBPP-50 / HE-30 (bootstrap 95 % CI over windows), seconds per step at depth 12 vs 30, and the fast screen's budget.
usage: calreport.py  (reads results/screens/cal_eval15.json, cal_eval31.json, ckpt/cal12.json, ckpt/cal30.json)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sb import core as C
from sb.compare import paired


def main():
    e15 = json.load(open(os.path.join(C.RES, "screens", "cal_eval15.json")))["per"]
    e31p = os.path.join(C.RES, "screens", "cal_eval31.json")
    e31 = json.load(open(e31p))["per"] if os.path.exists(e31p) else {}
    sps = {}
    for t in ("cal12", "cal30"):
        p = os.path.join(C.CKPT, t + ".json")
        if os.path.exists(p):
            sps[t] = json.load(open(p)).get("s_per_step")
    rng = np.random.default_rng(1114)
    lines = [f"\n## Result ({__import__('time').strftime('%H:%M')})",
             f"Seconds per training step (1 arm, 96 anchors, shared GPU): depth 12 {sps.get('cal12') or float('nan'):.2f}, "
             f"depth 30 {sps.get('cal30') or float('nan'):.2f}" + (f" ({sps['cal30'] / sps['cal12']:.2f}x)" if sps.get('cal12') and sps.get('cal30') else ""),
             "", "| copy | steps | MBPP-50 7-14 delta (CI) | 0-0 | 11-14 | kept/16 | HE-30 7-14 delta (CI) | moves? |",
             "|---|---|---|---|---|---|---|---|"]
    budget = {}
    for tag, steps_list in (("cal12", (100, 200, 300, 500, 700, 1000)), ("cal30", (100, 200, 300, 500))):
        for s in steps_list:
            run = f"{tag}_ctl_s{s}"
            if f"{run}|mbpp|chain" not in e15:
                continue
            d = {}
            for st in ("mbpp", "he"):
                pa, pb = e15[f"s30o|{st}|chain"], e15[f"{run}|{st}|chain"]
                d[st] = {k: paired(pa, pb, k, rng) for k in ("7-14", "0-0", "11-14", "kept")}
            m = d["mbpp"]["7-14"]
            moves = (m["ci95"][0] > 0 or m["ci95"][1] < 0) and abs(m["diff"]) >= 0.01
            if moves and tag not in budget:
                budget[tag] = s
            lines.append(f"| {tag} | {s} | {100 * m['diff']:+.2f} ({100 * m['ci95'][0]:+.2f}..{100 * m['ci95'][1]:+.2f}) | "
                         f"{100 * d['mbpp']['0-0']['diff']:+.2f} | {100 * d['mbpp']['11-14']['diff']:+.2f} | "
                         f"{d['mbpp']['kept']['diff']:+.3f} | {100 * d['he']['7-14']['diff']:+.2f} "
                         f"({100 * d['he']['7-14']['ci95'][0]:+.2f}..{100 * d['he']['7-14']['ci95'][1]:+.2f}) | {'YES' if moves else 'no'} |")
    if e31:
        lines += ["", "Depth-31 read-out of the depth-30 copies vs s30o (MBPP-50): " + "; ".join(
            f"s{s} 23-30 {100 * paired(e31['s30o|mbpp|chain'], e31[f'cal30_ctl_s{s}|mbpp|chain'], '23-30', rng)['diff']:+.2f}, "
            f"kept/32 {paired(e31['s30o|mbpp|chain'], e31[f'cal30_ctl_s{s}|mbpp|chain'], 'kept', rng)['diff']:+.3f}"
            for s in (100, 200, 300, 500) if f"cal30_ctl_s{s}|mbpp|chain" in e31)]
    b12, b30 = budget.get("cal12"), budget.get("cal30")
    lines.append("")
    if b12 is None:
        lines.append("**Decision:** no depth-12 copy measurably moves by 1,000 steps: no fast default; the fast format needs a longer "
                     "budget or a higher lr (calibrate further).")
    else:
        w12 = b12 * (sps.get("cal12") or 0); w30 = (b30 or 0) * (sps.get("cal30") or 0)
        if b30 is not None and w30 and w30 < w12:
            lines.append(f"**Decision:** depth 30 moves sooner in wall time ({b30} steps ~{w30 / 60:.1f} min vs depth 12's {b12} steps "
                         f"~{w12 / 60:.1f} min): the fast format is depth 30 at {b30} steps.")
        else:
            lines.append(f"**Decision:** fast screen = rollout depth 12, read-out depth 15, {b12} steps (~{w12 / 60:.1f} min of training "
                         f"per arm), line band 7-14; winners confirmed at depth 30 (long-screen format)."
                         + (f" Depth 30 first moves at {b30} steps (~{w30 / 60:.1f} min)." if b30 else " Depth 30 did not move by 500 steps."))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
