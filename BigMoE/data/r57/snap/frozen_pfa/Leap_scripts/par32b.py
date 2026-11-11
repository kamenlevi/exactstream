"""L1-32b (Leap, Sun 2026-10-25, CPU): SSD misses per generated token at 24 GB for k parallel streams of one answer vs
one stream, on the Breakthrough routing traces (resid27's LRU pools of C experts per layer). Each sequence (~560-800
tokens) is decoded either as 1 stream, or as 4 streams = its 4 quarters advancing in lockstep, every check reading the
union over all streams' 16 rows; also 4 streams = 4 different sequences (unrelated parts). Pools warm-started with the
C most used experts of the OTHER sequences (no leakage), checks of 16 rows advancing 9.5 tokens per stream; top-10 / 8.
Usage: [BigMoE venv] par32b.py -> ~/Leap/data/par32b.txt"""
import glob, os
import numpy as np
from collections import OrderedDict

NL, E = 48, 512
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
sets = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}
IDS = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
GP = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def usage(names):
    u = np.zeros((NL, E))
    for k in names:
        for l in range(NL):
            np.add.at(u[l], IDS[k][:, l, :10].ravel(), GP[k][:, l, :10].ravel())
    return u

def run(streams, warm, C, k, R=16, Tacc=9.5):
    """streams: list of [n_i, NL, 20] id arrays decoded in lockstep; returns (misses per generated token, mean union)"""
    pools = []
    for l in range(NL):
        od = OrderedDict((int(e), True) for e in np.argsort(-warm[l])[:C])
        pools.append(od)
    pos = [0.0] * len(streams)
    miss, gen, uni, checks = 0, 0, 0, 0
    while all(int(p) + R <= s.shape[0] for p, s in zip(pos, streams)):
        for l in range(NL):
            need = set()
            for p, s in zip(pos, streams):
                need.update(s[int(p):int(p) + R, l, :k].ravel().tolist())
            uni += len(need)
            od = pools[l]
            for e in need:
                if e in od:
                    od.move_to_end(e)
                else:
                    miss += 1
                    od[e] = True
                    if len(od) > C:
                        od.popitem(last=False)
        checks += 1
        for i, p in enumerate(pos):
            gen += int(p + Tacc) - int(p)
            pos[i] = p + Tacc
    return miss / max(gen, 1), uni / max(checks * NL, 1)

say("L1-32b: SSD misses per generated token, 1 stream vs 4 lockstep streams (16-row checks, 9.5 tokens per stream)")
names = sorted(IDS)
for k in (10, 8):
    for C in (192, 288):
        cells = []
        for nm in names:
            warm = usage([o for o in names if o != nm])
            seq = IDS[nm]
            n = seq.shape[0] // 4 * 4
            q = n // 4
            one = run([seq[:n]], warm, C, k)
            four = run([seq[i * q:(i + 1) * q] for i in range(4)], warm, C, k)
            cells.append(f"{nm} {one[0]:.1f} -> {four[0]:.1f} ({four[0] / one[0]:.2f}x; union {one[1]:.0f} -> {four[1]:.0f})")
        warm = usage(names)
        m = min(IDS[o].shape[0] for o in names)
        cross = run([IDS[o][:m] for o in names], warm, C, k)
        one_avg = np.mean([run([IDS[o][:m]], warm, C, k)[0] for o in names])
        say(f"  top-{k}, C {C}: " + "; ".join(cells) + f"; 4 unrelated sequences {one_avg:.1f} -> {cross[0]:.1f} "
            f"({cross[0] / one_avg:.2f}x; warm-start from all, so optimistic for both)")
open(os.path.expanduser("~/Leap/data/par32b.txt"), "w").write("\n".join(lines) + "\n")
