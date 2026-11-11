"""price_all.py (Fable FULL, LOSSY350, for the orchestrator's 16:4x ask): the copy pass (A), the comment-free mode (B),
the split form and rule C priced together, in the ENGINE's convention, on the 12 code map windows.
Machinery: Fable C's bytes10_replay (windows, live_blocks at the t 0.3 cut, POOLS at 45 GB: plain LRU per layer, prompt
rows warm the pool, one session over the windows in walk order) and audit_split_replay's rules:
  QOM0BQ  today's preset: drafted rows load ranks 0-1 and keep ranks 0-1 + members + residents; no-draft rows 10 ranks.
  SPLIT99 / SPLIT90: the same loads; rows at cf >= 0.99 / 0.9 strict (ranks 0-1 + members).
Arms of the block list:
  live    the live blocks as recorded.
  A32/A64 the usable prompt-copy run (after an 8-token trigger) goes through wide blocks of 32 / 64 copied rows (cf 1.0,
          no draft call); every other live block as recorded.
  +B      comment-free, lower bound: rows whose input token is a comment token or a comment-ending newline are dropped,
          comment tokens are not counted as kept, blocks left with no kept token are dropped; blocks are NOT re-formed.
Per arm and rule: kept tokens, live blocks and wide blocks, their tiles (distinct kept experts per block over 48 layers;
wide blocks also in padded 16-row tiles), loads that miss.
Priced [P] with today's terms per 16-row check [K FULL section 1, Wed morning: dense 36, experts 23, draft 10, glue 5, SSD
21 = 95 ms]: live check = 72 + TAU x tiles, TAU = 23 ms / today's mean tiles per live block; wide pass = dense 36 x (1.24
at W 32, 1.70 at W 64 [K D32, linear in rows]) + glue 5 + SSD 21 x (1.2, 1.45) [G] + TAU x padded tiles, no draft.
The npz is the project's own walk (allow_pickle for its names array only). CPU, ~20 s.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I price_all.py > price_all.out"""
import importlib.util, math, os, sys
from collections import OrderedDict
import numpy as np
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/FableGod/fablec")
from bytes10_replay import windows, live_blocks, POOLS, MR  # noqa: E402
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", f"{H}/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
WK = np.load(f"{H}/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
NAMES = [str(x) for x in WK["names"]]
TRIG = 8
def prep(w):
    wi = NAMES.index(w["nm"]); ids = WK[f"{wi}_ids"].astype(int).tolist(); ck = WK[f"{wi}_checks"]; N = len(w["ids"])
    sc = rc.CodeScan(); fl = []
    for i in ids:
        s = TOK.decode([i]); before = bool(sc.code and sc.com); anyc = False
        for ch in s:
            sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
        fl.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
    g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:][:max(0, N - g0)]
    pos = {}
    for i, t in enumerate(P): pos.setdefault(t, []).append(i)
    q = 0; run = None
    while q < len(A) and run is None:
        best = 0
        for i in pos.get(A[q], ()):
            L = 0
            while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
            best = max(best, L)
        if best >= 8: run = (g0 + q + TRIG, g0 + q + best)      # token positions [s, e) of the usable copy
        q += 1
    w["fl"] = fl; w["run"] = run
def strict(rule, c): return (rule == "SPLIT99" and c >= 0.99) or (rule == "SPLIT90" and c >= 0.9)
def simulate(ws, rule, arm, nocom):
    Cm = POOLS["45"][0]; pools = [OrderedDict() for _ in range(48)]
    out = dict(kept=0, live=0, live_tiles=0, wide=0, wide_tiles=0, wide_pad=0, wide_rows=0, miss=0, copy_kept=0, copy_live=0, copy_live_tiles=0)
    Wd = {"A32": 32, "A64": 64}.get(arm)
    def block(E, cf, kind):
        R = len(E); t = 0; pad = 0; m = 0
        k0 = [10 if c < 0 else 2 for c in cf]; st = [strict(rule, c) for c in cf]
        for l in range(48):
            c = pools[l]; loads = set()
            for r in range(R): loads.update(E[r, l, :k0[r]].tolist())
            for e in loads:
                if e in c: c.move_to_end(e)
                else:
                    m += 1; c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
            cnt = {}
            for r in range(R):
                row = E[r, l].tolist(); kk = k0[r]
                for e in row[:kk]: cnt[e] = cnt.get(e, 0) + 1
                for e in row[kk:]:
                    if e in loads or (not st[r] and e in c): cnt[e] = cnt.get(e, 0) + 1
            for e in cnt:
                if e in c: c.move_to_end(e)
            t += len(cnt); pad += sum(math.ceil(v / 16) for v in cnt.values())
        out["miss"] += m
        return t, pad
    for w in ws:
        ids = w["ids"]; a = w["a"]; fl = w["fl"]; run = w["run"]
        for l in range(48):
            c = pools[l]
            for e in ids[:a, l].ravel().tolist():
                if e in c: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
        done_wide = False
        for rows, cfs, kp in live_blocks(w, bonus="next"):
            A = rows[0]; first = A + 1
            in_run = run is not None and run[0] <= first < run[1]
            if in_run and Wd:
                if not done_wide:
                    done_wide = True
                    for b in range(run[0], run[1], Wd):
                        # the rows whose INPUT tokens are the copied tokens b-1 .. (the row at p verifies token p + 1)
                        lo, hi = b - 1, min(b - 1 + Wd, run[1] - 1)
                        E = ids[lo:hi]
                        if len(E) == 0: continue
                        t, pad = block(E, [1.0] * len(E), "wide")
                        out["wide"] += 1; out["wide_tiles"] += t; out["wide_pad"] += pad; out["wide_rows"] += len(E); out["kept"] += len(E); out["copy_kept"] += len(E)
                continue
            keep_idx = list(range(len(rows))); kept = kp
            if nocom:
                com_rows = {i for i in range(min(kp, len(rows))) if rows[i] < len(fl) and fl[rows[i]] != 0}
                kept = kp - sum(1 for p in range(A + 1, A + kp + 1) if p < len(fl) and fl[p] != 0)
                keep_idx = [i for i in keep_idx if i not in com_rows]
                if kept <= 0 or not [i for i in keep_idx if i < kp]: continue
            E = ids[[rows[i] for i in keep_idx]]; cf = [cfs[i] for i in keep_idx]
            t, pad = block(E, cf, "live")
            out["live"] += 1; out["live_tiles"] += t; out["kept"] += kept
            if in_run: out["copy_kept"] += kept; out["copy_live"] += 1; out["copy_live_tiles"] += t
    return out
W = windows()["code"]
for w in W: prep(w)
base = simulate(W, "QOM0BQ", "live", False)
TAU = 23.0 / (base["live_tiles"] / base["live"]); T0 = 95.0 * base["live"] / base["kept"]
print(f"12 code map windows, 45 GB pools, one session. Today (live blocks, QOM): {base['live']} checks, kept {base['kept']} ({base['kept'] / base['live']:.2f} per check), "
      f"tiles per check {base['live_tiles'] / base['live']:.0f}; in the usable copy run: {base['copy_live']} checks, {base['copy_kept']} tokens ({100 * base['copy_kept'] / base['kept']:.1f} %), "
      f"{base['copy_kept'] / max(1, base['copy_live']):.2f} per check, tiles per check {base['copy_live_tiles'] / max(1, base['copy_live']):.0f}. TAU = {1000 * TAU:.2f} us per tile.")
print("arm      rule     | kept | live checks (tiles each) | wide passes (rows, tiles, padded tiles each) | misses per kept token | ms per kept token | tok/s vs today")
res = {}
for arm in ("live", "A32", "A64"):
    for nocom in (False, True):
        for rule in ("QOM0BQ", "SPLIT99", "SPLIT90"):
            r = simulate(W, rule, arm, nocom); res[(arm, nocom, rule)] = r
            ms_live = r["live"] * 72.0 + TAU * r["live_tiles"]
            dn, ss = {"A32": (1.24, 1.2), "A64": (1.70, 1.45)}.get(arm, (1.0, 1.0))
            ms_wide = r["wide"] * (36.0 * dn + 5.0 + 21.0 * ss) + TAU * r["wide_pad"]
            per = (ms_live + ms_wide) / r["kept"]
            r["per"] = per
            wl = f"{r['wide']:3d} ({r['wide_rows'] / r['wide']:.1f}, {r['wide_tiles'] / r['wide']:.0f}, {r['wide_pad'] / r['wide']:.0f})" if r["wide"] else "  -"
            print(f"{arm + ('+B' if nocom else ''):8s} {rule:8s} | {r['kept']:4d} | {r['live']:3d} ({r['live_tiles'] / max(1, r['live']):5.0f}) | {wl:28s} | {r['miss'] / r['kept']:.2f} | {per:5.2f} | x{T0 / per:.3f}")
