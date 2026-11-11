"""D-06 PTR analysis (MoonShots Wed 16:0x; CPU, after ptr_capture.py). Can the 125B's own QSA attention serve as a copy pointer
for drafting on agent turns?
Passes are simulated as in Chat1000's agent_analyze.py: the pass at p drafts out[p..p+14] with the native MTP chain
mtp[p] (the stand-in for DFlash2 until Leap's D-02); n = leading matches; the FIRST MISS is out[p+n]; next pass at p+n+1.
At drafting time the last computed target row has input X[L+p-2] (chunk row s = p) and the last known token is the bonus
X[L+p-1]. Pointer j* = the context position (sinks 0-3 and the row's own last two positions excluded) with the highest
attention probability, max over heads, PRIMARY = max over all 12 QSA layers and 24 heads (one pre-registered rule; per-layer
numbers are reported too). Two readings, both pre-registered:
  A (NewIdeas'): the token after the attended position continues the text: draft X[L+p+k] = X[j*+2+k]
  B (induction): the attended token is the next token: draft X[L+p+k] = X[j*+1+k]
Gated: only when the pointer is aligned with the known bonus (A: X[j*+1] == X[L+p-1]; B: X[j*] == X[L+p-1]).
Baseline at the same misses: prompt lookup (longest suffix 8..2 tokens, most recent earlier occurrence; spec125's default).
PASS (NewIdeas' row, written first): >= 25 % of the first misses on agent text recovered; KILL: < 8 %.
Also: block-level tau of PTR alone and of an oracle best-of(MTP, PTR) per pass (the honest drafter-level bound), and the
same-row pointer at positions where the 125B is >= 0.9 sure and the previous token occurs in the context (NewIdeas' proxy;
not available at drafting time).
usage: [BigMoE venv] taskpolicy -b python ptr_analyze.py -> stdout + data/ptr/analyze.txt"""
import glob, os
import numpy as np

D = os.path.expanduser("~/MoonShots/data/ptr/")
SINK, K = 4, 15
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def pointer(top, topv, s, hi, layers=None):
    """highest-probability position <= hi (and >= SINK) at chunk row s over the chosen layers; None if none stored"""
    best, bv = None, -1.0
    for li in (range(top.shape[0]) if layers is None else layers):
        for j, v in zip(top[li, s], topv[li, s]):
            if SINK <= j <= hi and v > bv:
                best, bv = int(j), float(v)
    return best


def lookup(X, P):
    """prompt lookup at known length P: (continuation start, match length) of the longest suffix (8..2) seen earlier"""
    a = np.asarray(X[:P], "<i4").tobytes()
    for n in range(8, 1, -1):
        if P < n + 1:
            continue
        pat = a[-4 * n:]
        end = len(a) - 4
        while True:
            i = a.rfind(pat, 0, end)
            if i < 0:
                break
            if i % 4 == 0:
                return i // 4 + n, n
            end = i + len(pat) - 1
    return None, 0


def run(turns, layers=None, gate=False, reading="A"):
    miss = rec = rec_lk = rec_only = 0
    tau = {"mtp": [], "ptr": [], "best": []}
    for nm, z in turns.items():
        X, L, mtp, top, topv = z["X"], int(z["L"]), z["mtp"], z["top"], z["topv"]
        out = X[L:]
        T = len(out)
        p = 0
        while p < T:
            P = L + p
            j = pointer(top, topv, p, P - 4, layers) if p < top.shape[1] else None   # row s = p, input X[P-2]
            off = 2 if reading == "A" else 1
            if j is not None and gate and X[j + off - 1] != X[P - 1]:
                j = None
            ptr = [int(X[j + off + k]) if j is not None and j + off + k < P else -1 for k in range(K)]
            dm = [int(t) for t in mtp[p][:K]]
            acc = lambda d: next((i for i in range(min(len(d), T - p)) if d[i] != out[p + i]), min(len(d), T - p))
            n = acc(dm)
            na = acc(ptr)
            tau["mtp"].append(n + 1); tau["ptr"].append(na + 1); tau["best"].append(max(n, na) + 1)
            if n < K and p + n < T:
                miss += 1
                hit = ptr[n] == out[p + n]
                e, _ = lookup(X, P)
                lk = e is not None and e + n < P and X[e + n] == out[p + n]
                rec += hit; rec_lk += lk; rec_only += hit and not lk
            p += n + 1                                  # the MTP drafter's own trajectory (passes are MTP's)
    return miss, rec, rec_lk, rec_only, {k: float(np.mean(v)) for k, v in tau.items()}


turns = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(D + "*.npz"))}
say(f"turns: {', '.join(f'{k} (prompt {int(v['L'])})' for k, v in turns.items())}")
nl = next(iter(turns.values()))["top"].shape[0]
for reading in ("A", "B"):
    for gate in (False, True):
        m, r, rl, ro, tau = run(turns, None, gate, reading)
        say(f"reading {reading}{' gated' if gate else ''}, all {nl} layers: first misses {m}, PTR recovers {r} ({100 * r / max(m, 1):.1f} %), "
            f"lookup {rl} ({100 * rl / max(m, 1):.1f} %), PTR-only {ro} ({100 * ro / max(m, 1):.1f} %) | tau per pass: MTP "
            f"{tau['mtp']:.2f}, PTR alone {tau['ptr']:.2f}, best of the two {tau['best']:.2f}")
    per = []
    for li in range(nl):
        m, r, _, ro, _ = run(turns, [li], False, reading)
        per.append(f"L{int(next(iter(turns.values()))['layers'][li])} {100 * r / max(m, 1):.1f}")
    say(f"  reading {reading} per layer (ungated, % recovered): " + ", ".join(per))
for nm, z in turns.items():
    out_ok = []
    for reading in ("A", "B"):
        m, r, rl, ro, tau = run({nm: z}, None, False, reading)
        out_ok.append(f"{reading}: {r}/{m} ({100 * r / max(m, 1):.0f} %), lookup {rl}, tau best {tau['best']:.2f} vs MTP {tau['mtp']:.2f}")
    say(f"  {nm}: " + " | ".join(out_ok))
# NewIdeas' proxy: same-row pointer where the 125B is >= 0.9 sure and the previous token is in the context
for reading in ("A", "B"):
    n = h = 0
    for nm, z in turns.items():
        X, L, top, topv, pm = z["X"], int(z["L"]), z["top"], z["topv"], z["pmax"]
        for t in range(160):
            s = t + 1                                   # row input X[L+t-1] predicts out[t]
            if pm[s] < 0.9 or X[L + t - 1] not in set(X[:L + t - 1].tolist()):
                continue
            j = pointer(top, topv, s, L + t - 3)
            if j is None:
                continue
            n += 1
            h += X[j + (1 if reading == "A" else 0)] == X[L + t]
    say(f"proxy (same row, p >= 0.9, previous token in context), reading {reading}: {h}/{n} = {100 * h / max(n, 1):.1f} %")
m, r, _, _, _ = run(turns, None, False, "A")
mb, rb, _, _, _ = run(turns, None, False, "B")
best = max(r / max(m, 1), rb / max(mb, 1))
say(f"VERDICT D-06 PTR: best reading recovers {100 * best:.1f} % of MTP's first misses on agent text -> "
    f"{'PASS' if best >= 0.25 else ('KILL' if best < 0.08 else 'BETWEEN')} (PASS >= 25 %, KILL < 8 %)")
open(D + "analyze.txt", "w").write("\n".join(lines) + "\n")
