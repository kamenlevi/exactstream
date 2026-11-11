"""R-63S STREAMS (MoonShots, Sun 2026-11-01; NewIdeas' row from the orchestrator's night brief): how much of a 16-row check's
expert reads do parallel streams share? (aggregate tok/s across parallel parts; whether that counts is Kamen's call)
A. Independent streams: BigMoE's cap45 live captures (~/BigMoE/data/cap45/*_gate_0.npz, read only: the exact REF2 loop at
   45 GB, real DFlash2 16-row checks; 2 code + 4 math prompts). Each (check, layer) gives the set of experts its rows use
   (prefill and calls over 16 rows skipped). For every 3-prompt combination, checks are overlaid by index: per layer the
   union of the 3 sets vs their sum (no sharing) vs the chance level (3 independent draws of the same sizes from the
   layer's marginal expert popularity over all checks, 64 Monte Carlo samples).
B. Parallel parts of one answer: Breakthrough's expA true-token traces (chat, also code / math / bulgarian; 800 tokens,
   top-10): the answer cut into 8 parts of 100 tokens, 16-token windows at the same offset in each part overlaid; per
   layer the union of k = 1..8 windows; new experts per extra stream = (union(8) - union(1)) / 7.
The row's lines are in true-token units (one stream ~73 experts per layer): PASS = A's union <= ~190 of 3 x 73 = 219
(0.868 of the sum, the chance level) AND B (chat) <= 11 new experts per extra stream; KILL = A's unions at 3 x 73 (no
sharing, ratio ~1.0). Reported as ratios and in 73-units (ratio x 219).
usage: [BigMoE venv] taskpolicy -b python streams45.py (CPU, one core, ~1-2 min) -> stdout + data/streams45.txt"""
import glob, itertools, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
EXPA = "/Users/kamen/Breakthrough/data/expA/traces125/"
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E = 48, 512
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
rng = np.random.default_rng(0)
# ---------------------------------------------------------------- A. independent streams (cap45 live checks)
P = {}
for f in sorted(glob.glob(os.path.join(CAP, "*_gate_0.npz"))):
    z = np.load(f)
    seq, L, rows, off, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    checks = {}
    for s, l, r, o in zip(seq, L, rows, off):
        if r > 16 or l >= NL:
            continue
        checks.setdefault(int(s), {})[int(l)] = frozenset(int(e) for e in np.unique(ids[o:o + r]))
    full = [checks[s] for s in sorted(checks) if len(checks[s]) == NL]
    P[os.path.basename(f).replace("_gate_0.npz", "")] = full[1:]          # drop the first check (cold start)
say("A. cap45 live 16-row checks (exact REF2 loop, C 160): " + ", ".join(f"{k} {len(v)}" for k, v in P.items()))
pop = np.zeros((NL, E))
for v in P.values():
    for ch in v:
        for l in range(NL):
            for e in ch[l]:
                pop[l, e] += 1
pop /= pop.sum(1, keepdims=True)


def draw_union(sizes, l, n=64):
    """mean union of independent popularity-weighted draws without replacement of the given sizes"""
    u = 0
    for _ in range(n):
        s = set()
        for k in sizes:
            s |= set(rng.choice(E, size=k, replace=False, p=pop[l]).tolist())
        u += len(s)
    return u / n


names = list(P)
single = np.mean([len(ch[l]) for v in P.values() for ch in v for l in range(NL)])
say(f"  one stream: {single:.1f} distinct experts per layer per check (live drafts; the row's 73 is true-token rows)")
res = []
for combo in itertools.combinations(names, 3):
    n = min(len(P[c]) for c in combo)
    U = S = C = 0.0
    cnt = 0
    for i in range(0, n, max(1, n // 6)):                                  # ~6 aligned checks per combination
        for l in range(0, NL, 4):                                          # 12 layers: enough for a mean
            sets = [P[c][i][l] for c in combo]
            U += len(frozenset().union(*sets)); S += sum(len(x) for x in sets)
            C += draw_union([len(x) for x in sets], l, n=16)
            cnt += 1
    res.append((combo, U / cnt, S / cnt, C / cnt))
for combo, u, s, c in res:
    say(f"  {' + '.join(x.split('_', 1)[1][:22] for x in combo):70s} union {u:6.1f} of sum {s:6.1f} ({u / s:.3f}); "
        f"chance {c:6.1f} ({c / s:.3f})")
ru = np.mean([u / s for _, u, s, _ in res]); rc = np.mean([c / s for _, _, s, c in res])
code = [r for r in res if sum("humaneval" in x for x in r[0]) == 2]
say(f"  mean over {len(res)} combinations: union / sum {ru:.3f} (= {219 * ru:.0f} of 219 in 73-units), chance {rc:.3f} "
    f"(= {219 * rc:.0f}); with both code prompts: " + ", ".join(f"{u / s:.3f}" for _, u, s, _ in code))
# ---------------------------------------------------------------- B. parallel parts of one answer (expA true tokens)
say("B. expA true-token answers cut into 8 parts of 100 tokens; 16-token windows at the same offset overlaid")
newper = {}
for f in sorted(glob.glob(EXPA + "*_800_g.npz")):
    k = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :10]          # [800, 48, 10]
    curve = np.zeros(9)
    nwin = 0
    for off in range(0, 100 - 16 + 1, 12):
        for l in range(NL):
            u = set()
            for part in range(8):
                w = ids[part * 100 + off: part * 100 + off + 16, l].ravel().tolist()
                u |= set(w)
                curve[part + 1] += len(u)
        nwin += NL
    curve /= nwin
    per = (curve[8] - curve[1]) / 7
    newper[k] = per
    say(f"  {k:9s}: union of k parts " + " ".join(f"{curve[j]:5.1f}" for j in range(1, 9)) +
        f"; new experts per extra stream {per:.1f} (single window {curve[1]:.1f})")
passA = ru <= 0.868
passB = newper.get("chat", 99) <= 11
call = "PASS" if (passA and passB) else ("KILL" if ru >= 0.97 else "BETWEEN")
say(f"Lines: A union / sum {ru:.3f} (PASS <= 0.868 = 190 / 219; KILL ~1.0); B chat {newper.get('chat', float('nan')):.1f} "
    f"new experts per extra stream (PASS <= 11) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/streams45.txt"), "w").write("\n".join(lines) + "\n")
