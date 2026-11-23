"""Transfer checks 5 (capacity) and 6 (anchor-only taps): applies TRANSFER.md's pre-registered lines to the xfer batch
(ctl, cap2x, tapsq; read-outs at depth 15 and 31) and fills TRANSFER.md (rows 5 and 6 + a result block).
usage: xferreport.py"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sb import core as C
from sb.compare import paired


def main():
    e15 = json.load(open(os.path.join(C.RES, "screens", "xfer_eval15.json")))["per"]
    e31 = json.load(open(os.path.join(C.RES, "screens", "xfer_eval31.json")))["per"]
    tr = json.load(open(os.path.join(C.CKPT, "xfer.json")))["arms"]
    rng = np.random.default_rng(1114)
    D = lambda E, arm, st, band: paired(E[f"xfer_ctl|{st}|chain"], E[f"xfer_{arm}|{st}|chain"], band, rng)
    f = lambda r: f"{100 * r['diff']:+.2f} ({100 * r['ci95'][0]:+.2f}..{100 * r['ci95'][1]:+.2f})"
    loss = {a: float(np.mean(tr[a]["loss"][-100:])) for a in tr}
    out = [f"\n## Results 5 and 6 ({time.strftime('%a %d %b %H:%M')}; raw: results/screens/xfer_eval15.json, xfer_eval31.json, ckpt/xfer.json)"]
    hold = paired(e15["s30|mbpp|chain"], e15["xfer_ctl|mbpp|chain"], "7-14", rng)
    held = hold["diff"] >= -0.015
    out.append(f"\nIn-batch control vs s30 (must hold): MBPP-50 7-14 {f(hold)} -> {'HOLDS' if held else 'DRIFTS: 5 and 6 are provisional'}")
    # 5. capacity
    b31 = {b: D(e31, "cap2x", "mbpp", b) for b in ("0-0", "7-10", "15-22", "23-30")}
    b15 = {b: D(e15, "cap2x", "mbpp", b) for b in ("0-0", "7-10", "11-14", "7-14")}
    h31 = {b: D(e31, "cap2x", "he", b) for b in ("0-0", "7-10", "15-22", "23-30")}
    d = {b: 100 * r["diff"] for b, r in b31.items()}
    shape = d["23-30"] - d["0-0"] >= 0.3 and d["23-30"] > 0 and d["15-22"] >= d["7-10"] - 0.2
    lr_ = loss["cap2x"] / loss["ctl"]
    lossm = lr_ > 1.0
    v5 = "MATCH" if shape and lossm else ("PARTIAL (shape matches, loss doesn't)" if shape else
                                         "PARTIAL (loss matches, shape doesn't)" if lossm else "MISMATCH")
    out += ["", f"**5. CAPACITY: {v5}.** cap2x minus ctl:",
            "- MBPP-50, depth-31 read-out: " + ", ".join(f"{b} {f(r)}" for b, r in b31.items()) + " (125B: 0 -0.09, 7-10 +0.44, 15-22 +1.00, 23-30 +1.20)",
            "- MBPP-50, depth-15 read-out: " + ", ".join(f"{b} {f(r)}" for b, r in b15.items()),
            "- HE-30, depth-31 read-out: " + ", ".join(f"{b} {f(r)}" for b, r in h31.items()),
            f"- training loss, mean of the last 100 steps: cap2x {loss['cap2x']:.4f} vs ctl {loss['ctl']:.4f} ({100 * (lr_ - 1):+.1f} %; 125B +1.6 %)",
            f"- lines: shape {'yes' if shape else 'no'} (delta 23-30 minus delta 0 = {d['23-30'] - d['0-0']:+.2f}, 23-30 {d['23-30']:+.2f}, "
            f"15-22 {d['15-22']:+.2f} vs 7-10 {d['7-10']:+.2f}); loss worse {'yes' if lossm else 'no'}"]
    # 6. anchor-only taps
    t31, t15 = D(e31, "tapsq", "mbpp", "23-30"), D(e15, "tapsq", "mbpp", "7-14")
    sig = lambda r: r["diff"] >= 0.005 and r["ci95"][0] > 0
    v6 = "MISMATCH (a significant gain)" if sig(t31) or sig(t15) else "MATCH (no significant gain)"
    th = D(e31, "tapsq", "he", "23-30")
    out += ["", f"**6. Anchor-only fused taps: {v6}.** tapsq minus ctl: MBPP-50 23-30 (depth 31) {f(t31)}, windows better "
            f"{t31['wins']} / worse {t31['losses']}; 7-14 (depth 15) {f(t15)}; 0-0 {f(D(e15, 'tapsq', 'mbpp', '0-0'))}; HE-30 23-30 {f(th)}. "
            f"Training loss (information only) {loss['tapsq']:.4f} vs ctl {loss['ctl']:.4f} ({100 * (loss['tapsq'] / loss['ctl'] - 1):+.1f} %). "
            "(125B TAPSQ2: 23-30 -0.63, windows 35 / 37, KL 1.083x: KILL.)"]
    md = os.path.join(C.ROOT, "TRANSFER.md")
    s = open(md).read()
    s = re.sub(r"\| 5 \| CAPACITY.*\n", f"| 5 | CAPACITY: a 2x zero-init branch at the head's lr | small depth-growing gain (0: -0.09 ... 23-30: +1.20), loss +1.6 % | 23-30 {d['23-30']:+.2f}, 0 {d['0-0']:+.2f}, loss {100 * (lr_ - 1):+.1f} % | {v5} |\n", s)
    s = re.sub(r"\| 6 \| anchor-only.*\n", f"| 6 | anchor-only fused taps (TAPSQ2) | KILL, 23-30 -0.63 | 23-30 {100 * t31['diff']:+.2f}, 7-14 {100 * t15['diff']:+.2f} | {v6.split(' (')[0]} |\n", s)
    open(md, "w").write(s + "\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
