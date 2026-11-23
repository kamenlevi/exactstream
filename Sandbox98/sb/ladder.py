"""DEPTH_LINES addition 2: each depth-batch arm's full band curve, kept/32, and its 23-30 placed on T2's ladder (the share
of the chain -> full gap it closes, beside T2's rungs K = 0 / 16 / 64 / 256, each as its own share of T2's gap).
usage: ladder.py [results/screens/dp.json]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb import core as C

BANDS = ["0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30"]


def main():
    dp = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.RES, "screens", "dp.json")))
    t2 = json.load(open(os.path.join(C.RES, "t2_trunc.json")))["bands"]
    gt = json.load(open(os.path.join(C.RES, "gate_true.json")))["per"]["s30|mbpp|true"]
    keys = sorted(dp["per"]["s30|mbpp|chain"])                         # the same 50 MBPP windows
    full50 = sum(gt[k]["23-30"][0] for k in keys) / sum(gt[k]["23-30"][1] for k in keys)
    ch50 = dp["res"]["s30|mbpp|chain"]["23-30"]
    rung = {k: (t2[k]["23-30"] - t2["chain"]["23-30"]) / (t2["true"]["23-30"] - t2["chain"]["23-30"]) for k in t2 if k.startswith("K")}
    lines = ["| arm (MBPP-50) | " + " | ".join(BANDS) + " | kept/32 | share of the chain -> full gap at 23-30 | on T2's ladder |",
             "|---" * (len(BANDS) + 4) + "|"]
    for tag, r in dp["res"].items():
        if not tag.endswith("|mbpp|chain"):
            continue
        name = tag.split("|")[0]
        share = (r["23-30"] - ch50) / (full50 - ch50)
        below = [k for k, v in sorted(rung.items(), key=lambda kv: kv[1]) if v <= share]
        above = [k for k, v in sorted(rung.items(), key=lambda kv: kv[1]) if v > share]
        pos = (f"between {below[-1]} and {above[0]}" if below and above else f"above {below[-1]}" if below else f"below {above[0]}")
        lines.append(f"| {name} | " + " | ".join(f"{100 * r[b]:.2f}" for b in BANDS) + f" | {r['kept_per_32']:.2f} | {100 * share:+.1f} % | {pos} |")
    lines.append(f"\nMBPP-50 full-state oracle at 23-30 (gate, same windows) {100 * full50:.2f}; s30's chain {100 * ch50:.2f}. T2 rungs as "
                 "shares of T2's own gap: " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(rung.items(), key=lambda kv: kv[1])))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
