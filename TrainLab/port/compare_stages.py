"""PORT diagnosis (MTP Head, Thu 11-05): compare two dumps of the 1-layer forward's intermediates (ref_mlx.py / test_parity.py
evalp --dump): per stage, max |diff| and relative L2 difference. Usage: python compare_stages.py A.npz B.npz"""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in ["h0", "n1", "x1", "k1", "a1", "f1", "h1", "n2", "x2", "k2", "m2", "f2", "h2", "hid", "logits4"]:
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    if x.shape != y.shape:
        print(f"{k:8s} SHAPE {x.shape} vs {y.shape}"); continue
    rel = np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)
    print(f"{k:8s} max|diff| {np.abs(x - y).max():.3e} (max|a| {np.abs(x).max():.3e})  rel-L2 {rel:.3e}")
