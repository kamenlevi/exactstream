"""Small-RAM prefill cost (NewIdeas 2026-10-22, CPU): on TrainLab's 125B prefill capture (cap125est: true top-10 per token
at layers 8/16/24/32/39 over 1,535-token docs), the number of DISTINCT experts a chunk of N consecutive prompt tokens
routes to per layer, the cold share at 10 % / 25 % residency (resident = the doc's most-used experts: optimistic), and
the SSD-bound prefill speed if each cold expert is read once per chunk (2.77 MB records, 14.6 GB/s)."""
import glob
import numpy as np

fs = sorted(glob.glob("/Users/kamen/TrainLab/data/cap125est/*.npz"))
docs = []
for f in fs:
    d = np.load(f)
    tops = {int(k[3:]): d[k].astype(np.int64) for k in d.keys() if k.startswith("top") and int(k[3:]) in (8, 16, 24, 32, 39)}
    if tops: docs.append((f.split("/")[-1], tops))
layers = sorted(docs[0][1])
print(f"{len(docs)} docs, layers {layers}, tokens/doc {docs[0][1][layers[0]].shape[0]}")
print("N tokens/chunk | distinct experts per layer (of 512) | cold at C=51 | cold at C=128 | SSD-bound prefill tok/s at 10 % / 25 %")
for N in (1, 16, 64, 256, 1024, 1535):
    dist = []; c51 = []; c128 = []
    for name, tops in docs:
        for l in layers:
            T = tops[l]; T = T[: (T.shape[0] // N) * N] if N <= T.shape[0] else T
            cnt = np.bincount(T[:, :10].ravel(), minlength=512)
            hot51 = set(np.argsort(-cnt)[:51]); hot128 = set(np.argsort(-cnt)[:128])
            for s in range(0, T.shape[0] - N + 1, N):
                u = set(T[s:s + N, :10].ravel().tolist())
                dist.append(len(u)); c51.append(len(u - hot51)); c128.append(len(u - hot128))
    d, a, b = np.mean(dist), np.mean(c51), np.mean(c128)
    t51 = 48 * a * 2.77e6 / 14.6e9; t128 = 48 * b * 2.77e6 / 14.6e9
    print(f"  {N:5d} | {d:6.1f} | {a:6.1f} | {b:6.1f} | {N / t51:7.0f} / {N / t128:7.0f}  (sweep {t51:.2f} s / {t128:.2f} s per chunk)")
