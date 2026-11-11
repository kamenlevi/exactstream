"""L1-32 screen (Leap, Sun 2026-10-25, CPU): intra-request parallelism on the 125B. If one chat decodes k independent
parts of its answer at once (published: Skeleton-of-Thought, APAR, Hogwild!, PASTA, Multiverse), a check holds k
streams x 16 rows. Its cost in expert bytes is the union of experts over all its rows. This measures that union on the
Breakthrough routing traces (one 125B sequence each: code, math, chat, bulgarian; top-20 ids per layer):
  chain:   16 / 32 / 64 consecutive tokens of one sequence
  same:    k windows of 16 consecutive tokens from one sequence, starts >= 96 tokens apart (parts of one answer)
  cross:   k windows of 16 from k different sequences (unrelated parts; pessimistic)
Per layer union at top-10 and top-8, mean over layers and 200 random draws; experts read per accepted token, with
accepted tokens = 9.5 per 16-row stream (code, L1-20), 10.5 for a 32-row chain (block 32), 11.67 for 64 rows (tree N63).
Usage: [BigMoE venv] par32.py -> ~/Leap/data/par32.txt"""
import glob, os
import numpy as np
NL = 48
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
IDS = {os.path.basename(f).split("_")[0]: np.stack([np.load(f)[f"L{l}"] for l in range(NL)], 1).astype(np.int32)
       for f in sorted(glob.glob(T + "*_800_g.npz"))}
rng = np.random.default_rng(32)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def union(blocks, k):
    """blocks: list of [rows, NL, 20] arrays -> mean over layers of |union of top-k ids|"""
    a = np.concatenate([b[:, :, :k] for b in blocks], 0)            # [rows, NL, k]
    return np.mean([len(np.unique(a[:, l])) for l in range(NL)])

def draw_chain(name, R):
    n = IDS[name].shape[0]
    s = rng.integers(0, n - R)
    return [IDS[name][s:s + R]]

def draw_same(name, k, r=16, gap=96):
    n = IDS[name].shape[0]
    for _ in range(1000):
        st = np.sort(rng.integers(0, n - r, k))
        if k == 1 or np.min(np.diff(st)) >= gap:
            return [IDS[name][s:s + r] for s in st]
    raise RuntimeError("no draw")

def draw_cross(names, r=16):
    out = []
    for nm in names:
        n = IDS[nm].shape[0]
        s = rng.integers(0, n - r)
        out.append(IDS[nm][s:s + r])
    return out

NS = 200
say(f"L1-32 screen: expert union per layer for k parallel streams vs one chain (Breakthrough traces, {NS} draws each)")
for k_top in (10, 8):
    say(f"  top-{k_top}:")
    rows = []
    for name in ("code", "math", "chat", "bulgarian"):
        c16 = np.mean([union(draw_chain(name, 16), k_top) for _ in range(NS)])
        c32 = np.mean([union(draw_chain(name, 32), k_top) for _ in range(NS)])
        c64 = np.mean([union(draw_chain(name, 64), k_top) for _ in range(NS)])
        s2 = np.mean([union(draw_same(name, 2), k_top) for _ in range(NS)])
        s4 = np.mean([union(draw_same(name, 4), k_top) for _ in range(NS)])
        rows.append((c16, c32, c64, s2, s4))
        say(f"    {name:9s} chain 16 / 32 / 64: {c16:5.1f} / {c32:5.1f} / {c64:5.1f}; same-answer streams 2x16 {s2:5.1f}, "
            f"4x16 {s4:5.1f}")
    x2 = np.mean([union(draw_cross(["code", "math"]), k_top) for _ in range(NS)])
    x4 = np.mean([union(draw_cross(["code", "math", "chat", "bulgarian"]), k_top) for _ in range(NS)])
    c16, c32, c64, s2, s4 = [np.mean([r[i] for r in rows]) for i in range(5)]
    say(f"    mean      chain 16 / 32 / 64: {c16:5.1f} / {c32:5.1f} / {c64:5.1f}; same 2x16 {s2:5.1f}, 4x16 {s4:5.1f}; "
        f"cross 2x16 {x2:5.1f}, 4x16 {x4:5.1f}")
    say(f"    experts read per accepted token (code-like acceptance): chain 16 {c16 / 9.5:.2f}, chain 32 {c32 / 10.5:.2f}, "
        f"64-row tree {c64 / 11.67:.2f}; same-answer 2x16 {s2 / 19:.2f}, 4x16 {s4 / 38:.2f}; cross 4x16 {x4 / 38:.2f}")
open(os.path.expanduser("~/Leap/data/par32.txt"), "w").write("\n".join(lines) + "\n")
