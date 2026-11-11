"""KLD-vs-training-tokens table per run from data/results.txt, with the gap closed vs the floor and a log-tokens fit.
usage: curves.py [RUN ...]       (default: every run)"""
import os, re, sys
import numpy as np

RES = os.path.expanduser("~/TrainLab/data/results.txt")
FLOOR = 0.010                      # 35B: bf16 perturbation of the MoE output (defer35 'floor')
pat = re.compile(r"train (\S+) (\S+) tokens (\d+) (\S+) KLD ([\d.]+) \(warm ([\d.]+), P99 ([\d.]+)\) top-1 ([\d.]+) \| (.*)")


def main():
    runs = {}
    for line in open(RES):
        m = pat.search(line)
        if not m:
            continue
        run, flow, tok, tag, kld, warm, p99, top1, rest = m.groups()
        per = dict(re.findall(r"(\w+) ([\d.]+)", rest.split("|")[0]))
        runs.setdefault(run, []).append((int(tok), float(kld), float(top1), {k: float(v) for k, v in per.items()}, flow))
    want = sys.argv[1:] or list(runs)
    for run in want:
        rows = sorted(runs.get(run, []))
        if not rows:
            continue
        z = rows[0][1]
        print(f"== {run} ({rows[0][4]}): zero-shot {z:.4f}, floor {FLOOR}")
        segs = list(rows[0][3])
        print(f"{'tokens':>9} {'KLD':>7} {'gap closed':>10} {'top-1':>6} " + " ".join(f"{s:>8}" for s in segs))
        for tok, k, t1, per, _ in rows:
            closed = (z - k) / max(z - FLOOR, 1e-9)
            print(f"{tok:9d} {k:7.4f} {closed:10.1%} {t1:6.3f} " + " ".join(f"{per.get(s, float('nan')):8.4f}" for s in segs))
        pts = [(t, k) for t, k, *_ in rows if t > 0]
        if len(pts) >= 3:
            t, k = np.array(pts, float).T
            a, b = np.polyfit(np.log10(t), np.log10(np.maximum(k - FLOOR, 1e-6)), 1)
            for target in (1e7, 1e8):
                print(f"   fit (KLD - floor) ~ tokens^{a:.2f}: at {target:.0e} tokens KLD ~ {FLOOR + 10 ** (b + a * np.log10(target)):.4f}")


if __name__ == "__main__":
    main()
