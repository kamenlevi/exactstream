"""Bit-check of the QOM t3 masks against their 4-bit twins (NewIdeas, Wed 4 Nov 2026 03:3x), before READY.

<arm>_t3 must equal <arm> on ranks 2-9 everywhere, and differ on ranks 0-1 only by switching them off on whole rows
(every layer) that are sure rows of a check. In the 4-bit twin a sure row keeps ranks 0-1 and may drop rank 2-9 experts;
an unsure row, a live bonus row, a prompt row and a row outside the walk keep all 10. So: every row with a rank 2-9 expert
dropped in the twin must have ranks 0-1 off in t3, and no row keeps ranks 0-1 in some layers but not in others.
Also prints, per domain, the rows switched to 3 bits, the twin's sure-looking rows, and the mask files' count (36).
Usage (BigMoE venv): python qom_t3_check.py <dir> <arm>_t3 [<arm>_t3 ...]   -> exit 0 only if every check holds
"""
import os, re, sys
import numpy as np

D = sys.argv[1]; bad = 0
for arm in sys.argv[2:]:
    twin = arm[:-3]; a, b = os.path.join(D, arm), os.path.join(D, twin)
    K = 1 if re.search(r"_k1(_|$)", arm) else 2                    # Wed 04:0x: top-1 arms send rank 0 alone to 3 bits
    fs = sorted(f for f in os.listdir(a) if f.endswith(".npz")); tw = sorted(f for f in os.listdir(b) if f.endswith(".npz"))
    if len(fs) != 36 or fs != tw:
        print(f"{arm}: {len(fs)} files vs the twin's {len(tw)} (names equal: {fs == tw})"); bad += 1; continue
    per = {}
    for f in fs:
        T = np.load(os.path.join(a, f))["keep"]; Q = np.load(os.path.join(b, f))["keep"]
        dom = f.split("_")[1]; s = per.setdefault(dom, [0, 0, 0, 0, 0, 0])
        if T.shape != Q.shape: print(f"{arm} {f}: shape {T.shape} vs {Q.shape}"); bad += 1; continue
        e29 = int((T[:, :, K:] != Q[:, :, K:]).sum())                      # ranks K-9 must be identical
        off = ~T[:, :, :K]; rows_off = off.all(axis=(1, 2)); part = off.any(axis=(1, 2)) & ~rows_off
        twin01 = int((~Q[:, :, :K]).sum())                                 # the twin keeps ranks 0..K-1 everywhere
        dropped = (~Q[:, :, K:]).any(axis=(1, 2))                          # rows with a rank K-9 expert dropped = sure rows
        miss = int((dropped & ~rows_off).sum())                            # a sure row left at 4 bits on ranks 0-1
        s[0] += T.shape[0]; s[1] += int(rows_off.sum()); s[2] += int(dropped.sum()); s[3] += e29; s[4] += int(part.sum()) + twin01
        s[5] += miss
    for dom, (n, r3, sure, e29, part, miss) in per.items():
        ok = e29 == 0 and part == 0 and miss == 0 and r3 >= sure
        bad += not ok
        print(f"{arm} {dom}: rows {n}, ranks 0-{K - 1} at 3 bits on {r3} rows ({100 * r3 / n:.1f} %), twin rows with a rank {K}-9 "
              f"drop {sure} (all covered: {miss == 0}); rank 2-9 diffs {e29}; partial rows {part} -> {'OK' if ok else 'FAIL'}")
print("ALL OK" if bad == 0 else f"{bad} FAILED")
sys.exit(1 if bad else 0)
