"""Speed model for small-RAM policies on the 125B at 10 % residency (NewIdeas, 2026-10-21), fed by pol35.py lines.
Calibration (Breakthrough, M5 Pro, cold SSD, today's fork engine, eager): frugal 25.3 tok/s with 29.5 waited reads/token,
exact 12.9 with 153; each extra waited read costs ~0.31 ms on the critical path -> t = 30.4 ms + 0.31 ms x waited cold reads.
Late (half-layer) reads: a token-layer with n late reads costs max(0, burst(n) - slack); burst = 0.336 / 0.561 / 0.975 /
1.79 ms for 1 / 2 / 4 / 8 reads (MoonShots r24); slack = resident experts of layer L + layer L+1's attention (SLACK ms).
Bandwidth floor: all reads x 2.77 MB (4-bit; 2.15 with MSB-3 tiers) / 14.6 GB/s.
MTP: 3.2 tokens per 4-row pass; the frugal pass is 80 ms incl. ~20 ms of rank 0-1 waits -> 60 ms + waits of the pass;
cold reads do not amortise across rows (Breakthrough pass_mass_sim), so waited and late reads scale with 3.2 tokens.
35B -> 125B mapping: reads scale with the exact streaming count (35B 131.8 reads/token at C26, 125B 153 at C51).
usage: spdmodel.py [SLACK_ms]   (reads ~/NewIdeas/data/defer35/results.txt 'pol' lines)
"""
import re, sys
import numpy as np

SLACK = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
EX35, EX125 = 131.8, 153.0


def burst(n):
    xs, ys = [0, 1, 2, 4, 8], [0, 0.336, 0.561, 0.975, 1.79]
    return float(np.interp(n, xs, ys))


def model(reads, waited, late_hist, mb=2.77):
    s = EX125 / EX35
    reads, waited = reads * s, waited * s
    p0, p1, p2, p3 = late_hist                       # share of token-layers with 0/1/2/3+ late reads (35B)
    late_cost = 48 * (p1 * max(0, burst(1 * s) - SLACK) + p2 * max(0, burst(2 * s) - SLACK) + p3 * max(0, burst(3.5 * s) - SLACK))
    t_lat = 30.4 + 0.31 * waited + late_cost
    t_bw = reads * mb / 14.6
    eager = 1000 / max(t_lat, t_bw)
    t_pass = 60 + 3.2 * (0.31 * waited + late_cost)
    mtp = 3.2 * 1000 / max(t_pass, 3.2 * t_bw)
    return eager, mtp, t_lat, t_bw


rx = re.compile(r"pol (\S+)\s+KLD ([\d.]+).*?reads/token\s+([\d.]+).*?waited cold/token ([\d.]+)\s+late per token-layer "
                r"0/1/2/3\+: ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)")
print(f"slack {SLACK} ms; columns: arm, 35B KLD, 125B-scaled reads/token, eager tok/s (4-bit), MTP tok/s (4-bit / MSB-3)")
for line in open("/Users/kamen/NewIdeas/data/defer35/results.txt"):
    m = rx.search(line)
    if not m or not line.startswith("seg1365"):
        continue
    arm, kld, reads, waited = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    hist = tuple(float(m.group(i)) for i in range(5, 9))
    e4, m4, tl, tb = model(reads, waited, hist)
    _, m3, _, _ = model(reads, waited, hist, mb=2.15)
    print(f"{arm:24s} KLD {kld:.4f}  reads {reads * EX125 / EX35:6.1f}  eager {e4:5.1f}  MTP {m4:5.1f} / {m3:5.1f}"
          f"  (latency {tl:.1f} ms, bandwidth {tb:.1f} ms per token)")
