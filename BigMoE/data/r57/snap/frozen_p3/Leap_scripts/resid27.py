"""24 GB with all 512 experts (Leap, Sat 2026-10-24; Kamen rejected pruning ~22:38 via the orchestrator): how many of each
layer's 512 experts must be resident, at what bits, for <= ~16 SSD misses per generated token at 16-row checks?
Breakthrough's 125B routing traces (data/expA/traces125/{code,chat,math,bulgarian}_800_g.npz: router top-20 ids + probs,
800 decode tokens per set, 48 layers). LRU pools of C experts per layer, warm-started with the C most used (top-10 gate
mass) on the first half of each set, measured on the second half. Checks: R rows advancing T accepted tokens (16 / 9.5
DFlash2, 11 / 9.2 truncated, 4 / 3.4 MTP); a check's rows = the next R true tokens. Routing top-10 or top-8.
Then the 24 GB budget: resident experts C x 48 x bytes(bits) + dense (3.44 GB, or 2.1 with the diet) + ~1.5 GB runtime
/ KV, and the SSD side: misses per token x bytes per miss (cold experts stored at b_cold) vs 350 tok/s at 10.6 / 14 GB/s
(the read time fully hidden behind compute; without overlap the reads add to the pass).
Simulator check (prune26): unpruned C 192 at 4-row checks gives 38-51 misses per token vs Fit12's measured ~46-83
(optimistic up to ~1.7x). Usage: python3 resid27.py -> ~/Leap/data/resid27.txt"""
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
        n = IDS[k].shape[0]
        ids, gp = IDS[k][:n // 2, :, :10], GP[k][:n // 2, :, :10]
        for l in range(NL):
            np.add.at(u[l], ids[:, l].ravel(), gp[:, l].ravel())
    return u

def simulate(ids, warm, C, R, Tacc):
    pools = []
    for l in range(NL):
        od = OrderedDict()
        for e in np.argsort(-warm[l])[:C]:
            od[int(e)] = True
        pools.append(od)
    n = ids.shape[0]; pos, miss, gen = 0.0, 0, 0
    while int(pos) + R <= n:
        s = int(pos)
        for l in range(NL):
            od = pools[l]
            for e in set(ids[s:s + R, l].ravel().tolist()):
                if e in od:
                    od.move_to_end(e)
                else:
                    miss += 1
                    od[e] = True
                    if len(od) > C:
                        od.popitem(last=False)
        gen += int(pos + Tacc) - s
        pos += Tacc
    return miss / max(gen, 1)

WARM = usage(list(IDS))
CS = (192, 224, 256, 288, 320, 352, 384, 448)
say("All 512 experts kept; misses per generated token (second halves; LRU warm-started from first halves of all sets)")
res = {}
for k in (10, 8):
    for R, Tacc in ((16, 9.5), (11, 9.2), (4, 3.4)):
        row = {}
        for test in ("code", "chat", "math", "bulgarian"):
            n = IDS[test].shape[0]
            ids = IDS[test][n // 2:, :, :k]
            row[test] = [simulate(ids, WARM, C, R, Tacc) for C in CS]
        res[(k, R)] = row
        say(f"  top-{k}, {R:2d} rows / {Tacc} tokens: " + " | ".join(
            f"C {C}: " + "/".join(f"{row[t][i]:.0f}" for t in ("code", "chat", "math", "bulgarian")) for i, C in enumerate(CS)))
say("  (each cell: code / chat / math / Bulgarian)")
MB = {2.0: 1.229, 2.5: 1.536, 3.0: 1.845, 3.5: 2.152, 4.5: 2.765}     # MB per expert (4.9 M params incl. scales)
say("24 GB budget: max resident C per layer = (24 - dense - 1.5 runtime) / (48 x MB per expert)")
for dense, dl in ((3.44, "full dense"), (2.1, "dense diet")):
    say(f"  {dl}: " + ", ".join(f"{b} b -> C <= {int((24 - dense - 1.5) * 1000 / (48 * MB[b]))}" for b in (2.0, 2.5, 3.0, 3.5)))
say("SSD side for 350 tok/s: misses per token allowed = SSD GB/s / 350 / MB per miss (cold experts stored at b_cold)")
say("  " + ", ".join(f"b_cold {b}: <= {10.6e3 / 350 / MB[b]:.1f} (10.6 GB/s) / {14e3 / 350 / MB[b]:.1f} (14 GB/s)"
                    for b in (2.0, 2.5, 3.0, 3.5)))
open(os.path.expanduser("~/Leap/data/resid27.txt"), "w").write("\n".join(lines) + "\n")
