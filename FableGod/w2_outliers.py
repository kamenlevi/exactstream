# W2: per-channel outlier statistics of the 125B's widened HC residual stream (d16w: wd rows x 10240 fp16 = 4 streams x 2560)
import numpy as np, glob, os, sys
files = sorted(glob.glob(os.path.expanduser("~/TrainLab/data/d16w/*.npz")))[:3]
for f in files:
    z = np.load(f); wd = z["wd"].astype(np.float32)        # [rows, 10240]
    rows = wd.shape[0]
    rms_ch = np.sqrt((wd ** 2).mean(0) + 1e-12); mx_ch = np.abs(wd).max(0)
    g_rms = np.sqrt((wd ** 2).mean()); ratio = mx_ch / g_rms
    energy = (wd ** 2).sum(0); share = np.sort(energy)[::-1].cumsum() / energy.sum()
    k = int(np.argmax(share >= 0.5)) + 1; k9 = int(np.argmax(share >= 0.9)) + 1
    top = np.argsort(-ratio)[:8]
    per_tok_max = np.abs(wd).max(1); per_tok_rms = np.sqrt((wd ** 2).mean(1)); mass = (per_tok_max / per_tok_rms)
    print(f"{os.path.basename(f)} rows {rows}: channels holding 50% / 90% of energy: {k} / {k9} of 10240; "
          f"max|x|/global RMS: median {np.median(ratio):.1f}, p99 {np.percentile(ratio, 99):.1f}, max {ratio.max():.1f}; "
          f"top channels (stream, ch): {[(int(c // 2560), int(c % 2560)) for c in top]}; "
          f"per-token max/RMS: median {np.median(mass):.1f}, p99 {np.percentile(mass, 99):.1f}")
