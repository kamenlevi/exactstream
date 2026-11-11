"""bonus125.txt rows vs Chat1000's hard caps at equal U(16) (error energy, log-interpolated between cap rows)"""
import numpy as np
cap = {}
for l in open('/Users/kamen/Chat1000/data/switchcap_hard125.txt').read().splitlines()[1:]:
    p = l.split('|'); n, m = p[0].split(); cap.setdefault(n, []).append((float(p[2].split()[0]), float(p[3])))
for l in open('/Users/kamen/TrainLab/data/bonus125.txt').read().splitlines()[1:]:
    p = l.split('|'); n, mode, d = p[0].split(); u, e = float(p[2].split()[0]), float(p[3])
    if float(d) == 0:
        continue
    pts, z = sorted(cap[n]), None
    for (u0, e0), (u1, e1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            z = np.exp(np.log(e0) + (u - u0) / (u1 - u0) * (np.log(e1) - np.log(e0)))
    print(f"{n:9s} {mode:4s} {d}: U16 {u:5.1f} E {e:.4f}" +
          (f"  hard cap at equal U16: E {z:.4f} -> bonus / cap {e / z:.2f}x" if z else "  (outside the caps' U16 range)"))
