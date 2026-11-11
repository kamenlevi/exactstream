"""M-10 SIBLINGS (MoonShots, Sun 2026-11-01; decision input for RANKED #10, exact trees): do sibling rows (same prefix,
different token) share experts, so tree branches are cheap in bytes at 45 GB?
Data (read only): BigMoE's cap45 live captures (~/BigMoE/data/cap45/*_gate_0.npz; the exact REF2 loop at 45 GB, real
DFlash2 16-row checks; 2 code + 4 math prompts). Per check (seq): 48 layer calls of 16 rows, then the commit replay of
layer 1 with kept = a + 1 rows (a = accepted drafts). Row 0 is the previous check's bonus token, rows 1-15 the drafts.
Exact sibling pairs: the first rejected row of check k (row a_k + 1, the drafter's token d at position p) and check k+1's
row 0 (the target's token c at p); both see the same prefix.
Measured per layer (then averaged over layers; also by depth band):
  overlap   |sib_d & sib_c| of 10, vs consecutive accepted rows (row j, j+1) and vs chance (two popularity-weighted draws)
  marginal  new experts a sibling row adds to a 16-row check's union: c into check k (which holds d), d into check k+1;
            vs a late chain row: row 15 given rows 0-14 of the same check
  branch    the target branch's first m rows (check k+1, rows 0..m-1) given check k's union, m = 1 / 4 / 8 / 16;
            for checks that accepted everything (a_k = 15) the same quantity is a plain extension (reference)
  hidden    descriptive: cosine of the MoE inputs x of the sibling pair vs of consecutive accepted rows
Lines (QUEUE M-10, written first): PASS = a sibling row adds <= 0.5x a late chain row's new experts (mean over layers, both
directions); KILL = >= 0.85x; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python siblings45.py (CPU, one core, ~1-2 min) -> stdout + data/siblings45.txt"""
import glob, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E, R = 48, 512, 16
BANDS = ((0, 8), (8, 24), (24, 48))
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
rng = np.random.default_rng(0)


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


prompts = []
for f in sorted(glob.glob(os.path.join(CAP, "*_gate_0.npz"))):
    z = np.load(f)
    seq, L, rows, roff, xoff, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"layers": {}, "xo": {}, "replay": None})
        if s not in order:
            order.append(s)
        if l in c["layers"]:                          # second call of a layer in a check = the commit replay
            c["replay"] = (l, r, int(roff[n]))
        else:
            c["layers"][l] = (r, int(roff[n]))
            c["xo"][l] = int(xoff[n])
    good = []
    for s in order:
        c = checks[s]
        if len(c["layers"]) != NL or c["replay"] is None or any(c["layers"][l][0] != R for l in range(NL)):
            continue
        lr, kr, ro = c["replay"]
        r16, o16 = c["layers"][lr]
        if not np.array_equal(ids[ro:ro + kr], ids[o16:o16 + kr]):        # replay rows == the check's first a + 1 rows
            sys.exit(f"replay mismatch {f} seq {s}")
        c["a"] = kr - 1
        c["sets"] = [[frozenset(ids[c["layers"][l][1] + j].tolist()) for j in range(R)] for l in range(NL)]
        good.append(s)
    name = os.path.basename(f).replace("_gate_0.npz", "")
    prompts.append((name, [checks[s] for s in good], z))
    say(f"  {name}: {len(good)} checks, kept {np.mean([checks[s]['a'] + 1 for s in good]):.2f} per check "
        f"(replay rows == check rows: OK)")

# popularity per layer over all check rows (chance model)
pop = np.zeros((NL, E))
for _, cks, _ in prompts:
    for c in cks:
        for l in range(NL):
            for st in c["sets"][l]:
                pop[l, list(st)] += 1
pop /= pop.sum(1, keepdims=True)


def chance_overlap(l, n=4000):
    g = np.log(np.maximum(pop[l], 1e-30))[None, :] + rng.gumbel(size=(2 * n, E))
    top = np.argpartition(-g, 10, axis=1)[:, :10]
    a, b = top[:n], top[n:]
    return float(np.mean([len(set(x) & set(y)) for x, y in zip(a.tolist(), b.tolist())]))


acc = {k: np.zeros(NL) for k in ("ov_sib", "ov_con", "mA", "mB", "mchain", "mmid")}
cnt = {k: np.zeros(NL) for k in acc}
branch = {m: [np.zeros(NL), 0] for m in (1, 4, 8, 16)}
ext = {m: [np.zeros(NL), 0] for m in (1, 4, 8, 16)}
hid = {"sib": np.zeros(len(BANDS)), "con": np.zeros(len(BANDS)), "ns": 0, "nc": 0}
npairs = 0


def add(k, l, v):
    acc[k][l] += v; cnt[k][l] += 1


for name, cks, z in prompts:
    X = z["x"]
    for i, c in enumerate(cks):
        S = c["sets"]
        a = c["a"]
        for l in range(NL):
            U = frozenset().union(*S[l])
            add("mchain", l, len(S[l][15] - frozenset().union(*S[l][:15])))
            for j in range(8, R):
                add("mmid", l, len(S[l][j] - frozenset().union(*S[l][:j])))
            for j in range(a):                                       # consecutive accepted rows (rows 0..a)
                add("ov_con", l, len(S[l][j] & S[l][j + 1]))
        if i + 1 >= len(cks):
            continue
        n = cks[i + 1]
        S2 = n["sets"]
        if a < R - 1:                                               # a rejection: sibling pair at row a + 1 / row 0
            npairs += 1
            for l in range(NL):
                d, t = S[l][a + 1], S2[l][0]
                U1, U2 = frozenset().union(*S[l]), frozenset().union(*S2[l])
                add("ov_sib", l, len(d & t))
                add("mA", l, len(t - U1))                           # target's token added to the drafted check
                add("mB", l, len(d - U2))                           # drafted token added to the next check
                for m in branch:
                    branch[m][0][l] += len(frozenset().union(*S2[l][:m]) - U1)
            for m in branch:
                branch[m][1] += 1
            for b, (lo, hi) in enumerate(BANDS):                    # hidden states: x of the pair, per band
                v = []
                for l in range(lo, hi):
                    xd = bf16(X[c["xo"][l] + a + 1]); xt = bf16(X[n["xo"][l]])
                    v.append(cos(xd, xt))
                hid["sib"][b] += np.mean(v)
                if a >= 1:
                    w = []
                    for l in range(lo, hi):
                        w.append(cos(bf16(X[c["xo"][l] + a - 1]), bf16(X[c["xo"][l] + a])))
                    hid["con"][b] += np.mean(w)
            hid["ns"] += 1
            if a >= 1:
                hid["nc"] += 1
        else:                                                       # full accept: check k+1 extends the chain
            for l in range(NL):
                U1 = frozenset().union(*S[l])
                for m in ext:
                    ext[m][0][l] += len(frozenset().union(*S2[l][:m]) - U1)
            for m in ext:
                ext[m][1] += 1
    if os.path.exists(PAUSE):
        while os.path.exists(PAUSE):
            time.sleep(20)

mean = {k: acc[k] / np.maximum(cnt[k], 1) for k in acc}
chance = np.array([chance_overlap(l) for l in range(NL)])
say(f"M-10 SIBLINGS: {sum(len(c) for _, c, _ in prompts)} live 16-row checks, {npairs} exact sibling pairs "
    f"(rejected draft row vs the next check's row 0, same prefix)")
say("per depth band            layers 0-7  8-23  24-47   all")


def band(v):
    return " ".join(f"{np.mean(v[lo:hi]):6.2f}" for lo, hi in BANDS) + f" {np.mean(v):6.2f}"


say("  overlap of 10: siblings     " + band(mean["ov_sib"]))
say("                 consecutive  " + band(mean["ov_con"]))
say("                 chance       " + band(chance))
say("  new experts:   sibling c into check k (holds d)  " + band(mean["mA"]))
say("                 sibling d into check k+1 (holds c) " + band(mean["mB"]))
say("                 late chain row (row 15 | 0-14)     " + band(mean["mchain"]))
say("                 rows 8-15, each given the earlier   " + band(mean["mmid"]))
say(f"  target branch's first m rows given check k's union ({branch[1][1]} pairs) vs a plain extension after a full "
    f"accept ({ext[1][1]} checks), new experts per layer:")
for m in branch:
    e = (ext[m][0] / ext[m][1]).mean() if ext[m][1] else float("nan")
    say(f"    m {m:2d}: branch {(branch[m][0] / max(branch[m][1], 1)).mean():6.2f}   extension {e:6.2f}")
say("  hidden (cosine of MoE inputs, by band): siblings " +
    " ".join(f"{hid['sib'][b] / max(hid['ns'], 1):.3f}" for b in range(len(BANDS))) +
    "; consecutive accepted rows " + " ".join(f"{hid['con'][b] / max(hid['nc'], 1):.3f}" for b in range(len(BANDS))))
rA = mean["mA"].mean() / mean["mchain"].mean()
rB = mean["mB"].mean() / mean["mchain"].mean()
r = max(rA, rB)
call = "PASS" if r <= 0.5 else ("KILL" if min(rA, rB) >= 0.85 else "BETWEEN")
say(f"Lines: a sibling row adds {rA:.2f}x (c into k) / {rB:.2f}x (d into k+1) a late chain row's new experts "
    f"(PASS <= 0.50 both, KILL >= 0.85) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/siblings45.txt"), "w").write("\n".join(lines) + "\n")
