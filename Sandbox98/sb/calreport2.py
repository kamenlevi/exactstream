"""Calibration v2 report (results/screens/CALIBRATION.md, v2 lines): each budget's own re-annealed control vs s30, paired
on MBPP-50 / HE-30; holds / moves per the lines; the fast budget -> results/screens/calibration_v2.json.
usage: calreport2.py"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sb import core as C
from sb.compare import paired


def main():
    R = os.path.join(C.RES, "screens")
    e15 = json.load(open(os.path.join(R, "cal2_eval15.json")))["per"]
    e31 = json.load(open(os.path.join(R, "cal2_eval31.json")))["per"] if os.path.exists(os.path.join(R, "cal2_eval31.json")) else {}
    rng = np.random.default_rng(1114)
    f = lambda r: f"{100 * r['diff']:+.2f} ({100 * r['ci95'][0]:+.2f}..{100 * r['ci95'][1]:+.2f})"
    lines = [f"\n## v2 result ({time.strftime('%H:%M')})", "",
             "| run | s/step | MBPP-50 7-14 vs s30 (CI) | 0-0 | 11-14 | kept/16 | HE-30 7-14 (CI) | holds | moves |",
             "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for d, budgets in ((12, (100, 200, 400, 800)), (30, (200, 400))):
        for B in budgets:
            tag = f"cal2_d{d}_b{B}"
            run = f"{tag}_ctl"
            if f"{run}|mbpp|chain" not in e15:
                continue
            sps = json.load(open(os.path.join(C.CKPT, tag + ".json"))).get("s_per_step")
            m = {k: paired(e15["s30|mbpp|chain"], e15[f"{run}|mbpp|chain"], k, rng) for k in ("7-14", "0-0", "11-14", "kept")}
            h = paired(e15["s30|he|chain"], e15[f"{run}|he|chain"], "7-14", rng)
            dl = m["7-14"]["diff"]
            holds = dl >= -0.015
            moves = (m["7-14"]["ci95"][0] > 0) and dl >= 0.01
            res[(d, B)] = {"holds": holds, "moves": moves, "s_per_step": sps, "wall_s": (sps or 0) * B}
            lines.append(f"| depth {d}, {B} steps | {sps or float('nan'):.2f} | {f(m['7-14'])} | {100 * m['0-0']['diff']:+.2f} | "
                         f"{100 * m['11-14']['diff']:+.2f} | {m['kept']['diff']:+.3f} | {f(h)} | {'yes' if holds else 'NO'} | {'YES' if moves else 'no'} |")
    if e31:
        lines += ["", "Depth-31 read-out of the depth-30 runs vs s30, MBPP-50: " + "; ".join(
            f"{B} steps: 23-30 {f(paired(e31['s30|mbpp|chain'], e31[f'cal2_d30_b{B}_ctl|mbpp|chain'], '23-30', rng))}, "
            f"kept/32 {paired(e31['s30|mbpp|chain'], e31[f'cal2_d30_b{B}_ctl|mbpp|chain'], 'kept', rng)['diff']:+.3f}"
            for B in (200, 400) if f"cal2_d30_b{B}_ctl|mbpp|chain" in e31)]
    drift = [k for k, v in res.items() if not v["holds"]]
    c12 = sorted(B for (d, B), v in res.items() if d == 12 and v["holds"] and v["moves"])
    c30 = sorted(B for (d, B), v in res.items() if d == 30 and v["holds"] and v["moves"])
    out = {"budget": None, "depth": None}
    if drift:
        lines.append(f"\n**Decision:** drift at {', '.join(f'depth {d} / {B} steps' for d, B in drift)}: the schedule still harms; no default.")
    elif not c12 and not c30:
        lines.append("\n**Decision:** every budget holds, none moves the control by >= 1 pt: s30 doesn't measurably improve with "
                     "more data at these budgets. By the rule, fast screens can't show training levers yet; no default.")
    else:
        w12 = res[(12, c12[0])]["wall_s"] if c12 else None
        w30 = res[(30, c30[0])]["wall_s"] if c30 else None
        if c12 and (w30 is None or w12 <= w30):
            out = {"budget": c12[0], "depth": 12}
        else:
            out = {"budget": c30[0], "depth": 30}
        lines.append(f"\n**Decision:** fast budget = {out['budget']} steps at rollout depth {out['depth']} (the smallest that holds and "
                     f"moves; ~{res[(out['depth'], out['budget'])]['wall_s'] / 60:.1f} min of training per arm).")
    json.dump(out, open(os.path.join(R, "calibration_v2.json"), "w"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
