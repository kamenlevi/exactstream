"""comqom_masks.py (Fit12, Mon 16 Nov 2026 13:3x): map-arm masks for "1c + comment-QOM" (and + RES), in tiers_1c's format,
for Breakthrough's harness if NewIdeas orders the arm. Desk, CPU, python -I, read-only inputs.
- '#' comment rows (lossy-45's causal CodeScan.in_comment(), primed with '```python' when the answer has no fence, as
  commentspan_price.py) take QOM cf 0's keep (NewIdeas' ~/NewIdeas/data/dm_masks/qom_0_live, the 45 GB pool) at 4 bits;
- every other answer row keeps 1c-tiers (keep = 4-bit tier, keep3 = MSB-3) or, for +RES, masks_1c_res;
- prompt rows unchanged.
Self-check: on non-comment rows the masks equal tiers_1c (or masks_1c_res); on comment rows keep == qom_0_live, keep3 False.
Out: /Users/kamen/Fit12/data/lossy32/masks_1c_comqom[_res]/<w>.npz (36 windows; the tagger only marks code answers)."""
import json, os, sys
import numpy as np
exec(compile(open("/Users/kamen/Fit12/scripts/commentspan_price.py").read().split(
    "# ---------------------------------------------------------------- (a) tagged shares")[0], "head", "exec"))
base = json.load(open(os.path.join(M, "baseline.json")))
O1, O2 = "/Users/kamen/Fit12/data/lossy32/masks_1c_comqom", "/Users/kamen/Fit12/data/lossy32/masks_1c_comqom_res"
os.makedirs(O1, exist_ok=True); os.makedirs(O2, exist_ok=True)
bad, ncom = 0, 0
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f)); a, m = int(z["a"]), z["ids"].shape[0]
    tok = np.array(base[f]["tk"])[:, 0]
    tg = tags(tok)[:, 0] if base[f]["dom"] == "code" else np.zeros(m, bool)
    ncom += int(tg.sum())
    q = np.load(os.path.expanduser(f"~/NewIdeas/data/dm_masks/qom_0_live/{f}"))["keep"].astype(bool)
    for src, out in ((os.path.join(M, "bt", "tiers_1c", f), O1), (f"/Users/kamen/Fit12/data/lossy32/masks_1c_res/{f}", O2)):
        t = np.load(src); k4, k3 = t["keep"].astype(bool).copy(), t["keep3"].astype(bool).copy()
        rows = np.zeros(k4.shape[0], bool); rows[a:a + m] = tg
        k4[rows] = q[rows]; k3[rows] = False
        bad += int((k4[~rows] != t["keep"][~rows]).sum() + (k3[~rows] != t["keep3"][~rows]).sum())
        bad += int((k4[rows] != q[rows]).sum() + k3[rows].sum())
        np.savez_compressed(os.path.join(out, f), keep=k4, keep3=k3)
msg = (f"comqom masks: 36 windows each in {O1} and {O2}; comment rows {ncom}; SELF-CHECK mismatches {bad} "
       f"({'PASS' if bad == 0 else 'FAIL'})")
print(msg); open("/Users/kamen/Fit12/data/lossy32/comqom_masks.txt", "w").write(msg + "\n")
