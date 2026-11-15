#!/usr/bin/env python3
"""Close-call (lossy) acceptance under the real decode walk (Breakthrough, Sun 15 Nov 2026 02:4x; the orchestrator: the
direct lossy route to GOAL A; pre-registered docs/CLOSECALL_ACCEPT_PRICE.md c8515d2) [P: teacher-forced upper bounds].
A guess is accepted if it is the 125B's top-1 OR its log-prob is within delta nats of the top-1 (dlp >= -delta).
Cells: t30b's (BigMoE data/deepdraft/ceiling/cells_t30b_{heplus,mbpp100}.npz). The window grid follows EvalSuite's
certify98.windows layout (depth-major per window; a window starts where depth returns to 0); the walk and the strict
counts are certify98's own kept_walk / strict_counts (imported read-only, no bytecode written).
Exposure: the walk's accepted non-top-1 tokens per 1K kept tokens, by the 125B's margin bin and by the target token's
class (code / comment / string / prose outside a code fence) from a character lexer over the decoded answer text.
Output: ~/Breakthrough/data/m1/closecall_walk.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B closecall_walk.py
"""
import glob
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import numpy as np
import certify98 as C98                                     # noqa: E402  (kept_walk, strict_counts, R0, DEPTHS)
from tokenizers import Tokenizer                            # noqa: E402

H = os.path.expanduser("~")
SETS = {"HE+": (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus24.npz", 13.77),
        "MBPP": (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz", 6.92)}
MBPP_FILES = sorted(glob.glob(f"{H}/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz"))[:100]
DELTAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
MBINS = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
CLASSES = ["code", "comment", "string", "prose"]
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
rng = np.random.default_rng(20261115)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def grids(path):
    """per window: rt [n0+1, 31] int8 1/0/-1, mt and dt [n0+1, 31] (margin, dlp), certify98.windows' layout"""
    d = np.load(path)
    dep, right = d["depth"].astype(int), d["right"].astype(int)
    m, dlp = d["m"].astype(float), d["dlp"].astype(float)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out = []
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, C98.DEPTHS), -1, np.int8)
        mt = np.full((n0 + 1, C98.DEPTHS), np.nan); dt = np.full((n0 + 1, C98.DEPTHS), np.nan)
        for j in range(C98.DEPTHS):
            k = dw == j; c = int(k.sum())
            rt[:c, j] = right[s:t][k]; mt[:c, j] = m[s:t][k]; dt[:c, j] = dlp[s:t][k]
        out.append((rt, mt, dt))
    names = list(d["files"]) if "files" in d.files else None
    return out, names


def lex_classes(text):
    """per character: 0 code, 1 comment, 2 string, 3 prose (outside ``` fences when the text has any)"""
    n = len(text); cls = np.zeros(n, np.int8)
    fenced = "```" in text
    infence = not fenced
    i = 0; st = "code"; q = ""; tri = False
    line_start = True
    while i < n:
        if line_start and text.startswith("```", i):     # a fence line: prose, toggles the fence state
            j = text.find("\n", i); j = n if j < 0 else j + 1
            cls[i:j] = 3; infence = not infence; st = "code"; i = j; line_start = True
            continue
        c = text[i]
        if not infence:
            cls[i] = 3
        elif st == "code":
            if c == "#":
                st = "comment"; cls[i] = 1
            elif c in "'\"":
                tri = text.startswith(c * 3, i); q = c; st = "string"
                k = 3 if tri else 1; cls[i:i + k] = 2; i += k; line_start = False
                continue
        elif st == "comment":
            if c == "\n":
                st = "code"
            else:
                cls[i] = 1
        else:                                            # string
            cls[i] = 2
            if c == "\\" and i + 1 < n:
                cls[i + 1] = 2; i += 2; line_start = False
                continue
            if tri and text.startswith(q * 3, i):
                cls[i:i + 3] = 2; i += 3; st = "code"; line_start = False
                continue
            if not tri and (c == q or c == "\n"):
                st = "code"
        line_start = c == "\n"
        i += 1
    return cls


def token_classes(ids, gen):
    """class of each answer token ids[gen:] (by its first non-space character), keyed by absolute position"""
    pieces = [TOK.decode([int(t)]) for t in ids[gen:]]
    text = "".join(pieces); cls = lex_classes(text)
    out = {}; off = 0
    for k, p in enumerate(pieces):
        j = next((off + i for i, ch in enumerate(p) if not ch.isspace()), off)
        out[gen + k] = int(cls[j]) if j < len(cls) else 0
        off += len(p)
    return out


def walk_exposure(rt, rtd, mt, a0, tcls):
    """the walk on rtd from row R0: kept per check, and its accepted non-top-1 cells (margin, class)"""
    kept, acc = [], []
    p = C98.R0
    while p < rtd.shape[0]:
        k = 0
        while k < 31 and rtd[p, k] == 1:
            if rt[p, k] == 0:                           # accepted by the close-call rule, not the 125B's top-1
                row = a0 + p + k + 1; acc.append((mt[p, k], tcls.get(row + 1, 0)))
            k += 1
        kept.append(k + 1); p += k + 1
    return kept, acc


def boot(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (10000, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


RES = {}
NODLP = set()
for sname, (path, published) in SETS.items():
    W, names = grids(path)
    _z = np.load(path); _w = ~_z['right']
    if _w.any() and np.isnan(_z['dlp'][_w]).all():      # (02:46) the HE+ capture left dlp / rank empty on the misses
        NODLP.add(sname)
        say(''); say(f'== {sname}: NOT PRICEABLE from these cells: dlp is NaN on all {int(_w.sum())} wrong guesses (the capture did '
            'not record the guess\'s log-prob gap); its rows below are the exact walk only')
    meta = []                                            # per window: (a0, token classes)
    for wi in range(len(W)):
        if sname == "HE+":
            z = np.load(f"{H}/datasets/deepdraft/bank/{names[wi]}")
        else:
            z = np.load(MBPP_FILES[wi])
        ids, gen = np.asarray(z["ids"]), int(z["gen"])
        meta.append((max(0, gen - 2), token_classes(ids, gen)))
    say("")
    say(f"== {sname}: {len(W)} windows; published t30b kept/32 {published}")
    say("| delta (nats) | kept/32 [95 % window CI] | vs exact | walk-strict 0-14 % | accepted non-top-1 per 1K kept | "
        "code | comment | string | prose |")
    say("|---|---|---|---|---|---|---|---|---|")
    for dl in DELTAS:
        knum, kden, expo = [], [], []
        RTD = []
        for (rt, mt, dt), (a0, tc) in zip(W, meta):
            rtd = rt.copy()
            if dl > 0:
                rtd[(rt == 0) & (dt >= -dl)] = 1
            RTD.append((rtd, None))
            kp, acc = walk_exposure(rt, rtd, mt, a0, tc)
            knum.append(sum(kp)); kden.append(len(kp)); expo += acc
        k, lo, hi = boot(knum, kden)
        C = C98.strict_counts(RTD, "walk", 31)
        num, den = sum(n for n, _ in C), sum(d_ for _, d_ in C)
        ws = 100 * num[:15].sum() / den[:15].sum()
        tot = sum(knum); E = np.array(expo, dtype=float).reshape(-1, 2) if expo else np.zeros((0, 2))
        per = lambda sel: 1000 * int(sel.sum()) / tot
        bycls = [per(E[:, 1] == c) for c in range(4)] if len(E) else [0, 0, 0, 0]
        base = RES.get((sname, 0.0), (k,))[0]
        RES[(sname, dl)] = (k, lo, hi, ws, 1000 * len(E) / tot, bycls,
                            [per((E[:, 0] >= a) & (E[:, 0] < b)) if len(E) else 0 for a, b in MBINS])
        say(f"| {dl:g} | {k:.2f} [{lo:.2f}, {hi:.2f}] | {k / base - 1:+.1%} | {ws:.2f} | {1000 * len(E) / tot:.1f} | "
            + " | ".join(f"{v:.1f}" for v in bycls) + " |")
    say("accepted non-top-1 per 1K kept tokens by the 125B's margin (m = top-1 minus top-2 nats): "
        + " ".join(f"[{a:g},{b:g})" for a, b in MBINS))
    for dl in DELTAS[1:]:
        say(f"  delta {dl:g}: " + " ".join(f"{v:.1f}" for v in RES[(sname, dl)][6]))
    say(f"SELF-CHECK delta 0 vs published {published}: {RES[(sname, 0.0)][0]:.2f} -> "
        f"{'PASS' if abs(RES[(sname, 0.0)][0] - published) < 0.01 else 'FAIL'}")

say("")
best = None
for dl in [d for d in DELTAS if 0 < d <= 1.0]:
    he, mb = RES[("HE+", dl)], RES[("MBPP", dl)]
    up_he, up_mb = he[0] / RES[("HE+", 0.0)][0] - 1, mb[0] / RES[("MBPP", 0.0)][0] - 1
    code_exp = max(he[5][0], mb[5][0])
    ok = up_he >= 0.15 and up_mb >= 0.15
    say(f"delta {dl:g}: lift HE+ {up_he:+.1%}, MBPP {up_mb:+.1%}; code exposure per 1K HE+ {he[5][0]:.1f}, MBPP {mb[5][0]:.1f}"
        f" -> {'lift met' if ok else 'lift not met'}{', code exposure <= 10' if code_exp <= 10 else ', code exposure > 10'}")
    if ok and code_exp <= 10 and best is None:
        best = dl
SETS_OK = [s for s in SETS if s not in NODLP]
say(f'sets with miss gaps: {SETS_OK}; without: {sorted(NODLP)}')
lift_any = any(RES[("HE+", d)][0] / RES[("HE+", 0.0)][0] >= 1.15 and RES[("MBPP", d)][0] / RES[("MBPP", 0.0)][0] >= 1.15
               for d in DELTAS if 0 < d <= 1.0)
v = f"PASS (delta {best:g})" if best is not None else ("BETWEEN" if lift_any else "KILL")
if NODLP:
    mb_ok = [d for d in DELTAS if 0 < d <= 1.0 and RES[("MBPP", d)][0] / RES[("MBPP", 0.0)][0] >= 1.15 and RES[("MBPP", d)][5][0] <= 10]
    v = (f"UNDETERMINED: {sorted(NODLP)} has no miss gaps in its cells. MBPP alone: "
         + (f"lift >= +15 % with code exposure <= 10 per 1K first at delta {mb_ok[0]:g}" if mb_ok else "no delta <= 1 meets both MBPP lines"))
say(f"VERDICT (pre-registered: PASS = some delta <= 1 lifts kept/32 >= +15 % on both sets with <= 10 code non-top-1 per 1K; "
    f"KILL = no delta <= 1 lifts both >= +15 %): {v}")
say("Caveat: teacher-forced UPPER bounds (after a changed token the cells still follow the 125B's own path; MoonShots' J(A): "
    "changed tokens change the following code about half the time). EvalSuite's code gate in real decoding decides.")
open(os.path.expanduser("~/Breakthrough/data/m1/closecall_walk.out"), "w").write("\n".join(lines) + "\n")
