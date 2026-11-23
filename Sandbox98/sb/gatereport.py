"""Applies GATE.md's pre-registered lines to the gate's eval files and writes results/gate.json + a markdown block.
usage: gatereport.py  (reads results/gate_chain.json, gate_true.json, gate_maps.json, maps.json)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb import core as C

BANDS = ["0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30"]


def load(n):
    p = os.path.join(C.RES, n)
    return json.load(open(p))["res"] if os.path.exists(p) else {}


def row(name, s):
    return f"| {name} | " + " | ".join(f"{100 * s[b]:.2f}" for b in BANDS) + f" | {s['kept_per_32']:.2f} |"


def main():
    R = {}
    for n in ("gate_chain.json", "gate_true.json", "gate_maps.json"):
        R.update(load(n))
    maps = json.load(open(os.path.join(C.RES, "maps.json"))) if os.path.exists(os.path.join(C.RES, "maps.json")) else {}
    out, lines = {}, []
    for st in ("mbpp", "he"):
        ch, tr = R.get(f"s30|{st}|chain"), R.get(f"s30|{st}|true")
        if not ch or not tr:
            continue
        drop = ch["0-0"] - ch["23-30"]
        flat = abs(tr["0-0"] - tr["23-30"])
        gain = tr["23-30"] - ch["23-30"]
        out[st] = {"chain_drop_pt": 100 * drop, "true_flat_pt": 100 * flat, "true_gain_pt": 100 * gain,
                   "line1_depth_drop": drop >= 0.08, "line2_fresh_fixes": flat <= 0.03 and gain >= 0.08}
        lines.append(f"### {st.upper()} ({ch['windows']} windows)")
        lines.append("| s30 read-out | " + " | ".join(BANDS) + " | kept/32 |")
        lines.append("|---" * (len(BANDS) + 2) + "|")
        for tag, nm in ((f"native|{st}|chain", "native head, chain"), (f"s30|{st}|chain", "s30, chain (carried stand-in)"),
                        (f"s30|{st}|true", "s30, TRUE W at every depth"), (f"s30|{st}|block:8", "s30, true W every 8 (BLOCK)")):
            if tag in R:
                lines.append(row(nm, R[tag]))
        for m in ("L2", "L12", "L22", "Lall"):
            tag = f"s30|{st}|map:{m}"
            if tag in R:
                lines.append(row(f"s30, map {m} (taps {maps.get(m, {}).get('taps')}, R2 {maps.get(m, {}).get('r2', float('nan')):.3f})", R[tag]))
        for fr in ("0.3", "0.5", "0.7"):
            tag = f"s30|{st}|noise:{fr}"
            if tag in R:
                lines.append(row(f"s30, true W + noise {fr} (R2 ~{1 - float(fr) ** 2:.2f})", R[tag]))
        lines.append("")
        lines.append(f"- Line 1 (23-30 >= 8 pt below depth 0): drop {100 * drop:.2f} pt -> {'HOLDS' if out[st]['line1_depth_drop'] else 'FAILS'}")
        lines.append(f"- Line 2 (true W flat within 3 pt and >= +8 pt over the chain): |0 - 23-30| {100 * flat:.2f} pt, gain "
                     f"{100 * gain:+.2f} pt -> {'HOLDS' if out[st]['line2_fresh_fixes'] else 'FAILS'}")
        mp = [(maps[m]["r2"], R[f"s30|{st}|map:{m}"]["23-30"], m) for m in ("L2", "L12", "L22", "Lall")
              if m in maps and f"s30|{st}|map:{m}" in R]
        if mp:
            mp.sort()
            mono = all(mp[i][1] <= mp[i + 1][1] + 0.005 for i in range(len(mp) - 1))
            early_below = "L2" in maps and f"s30|{st}|map:L2" in R and R[f"s30|{st}|map:L2"]["23-30"] < ch["23-30"]
            out[st]["line3_map_curve"] = {"monotone_in_r2": mono, "early_map_below_chain": early_below,
                                          "points": [[r2, acc, m] for r2, acc, m in mp]}
            lines.append(f"- Line 3 (supporting): 23-30 vs R2 " + ", ".join(f"{m} R2 {r2:.3f} -> {100 * a:.2f}" for r2, a, m in mp)
                         + f"; monotone {'yes' if mono else 'no'}; early map below the chain {'yes' if early_below else 'no'}")
        lines.append("")
    if "mbpp" in out:
        ok = out["mbpp"]["line1_depth_drop"] and out["mbpp"]["line2_fresh_fixes"]
        out["verdict"] = "PASS" if ok else "FAIL"
        lines.insert(0, f"**Verdict (MBPP lines 1 + 2): {out['verdict']}**\n")
    json.dump({"lines": out, "res": R, "maps": maps}, open(os.path.join(C.RES, "gate.json"), "w"), indent=1)
    md = "\n".join(lines)
    open(os.path.join(C.RES, "gate_report.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
