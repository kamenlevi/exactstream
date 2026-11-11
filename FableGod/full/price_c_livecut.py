"""price_c_livecut.py (Fable FULL, LOSSY350): rule C in the ENGINE's convention (live-cut blocks at t 0.3), on the 12 code
map windows, to read its overlap with the comment-free mode where the engine actually runs.
A live block ends either at the cut (no miss among its verified rows) or at a miss on a verified row. Rule C can only act
on the second kind, and only when the missed row is a comment row it forgives. Per block ended by a miss: is the missed
row a comment row; does rule C forgive it (MoonShots' span_relax_c acc on the 125B's top-8, baseline.json).
Gain bounds for rule C: each forgiven miss adds at least 1 token (the accepted draft) and at most the block's remaining
verified rows. For scale: the blocks whose kept tokens are all comment tokens (what the comment-free mode removes).
CPU, seconds. Run: nice -n 15 ~/BigMoE/.venv/bin/python -I price_c_livecut.py > price_c_livecut.out"""
import importlib.util, os, sys
import numpy as np
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/FableGod/fablec")
from bytes10_replay import windows, live_blocks  # noqa: E402
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("span_relax_c_ro", f"{H}/MoonShots/scripts/span_relax_c.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
RC = S.RCORE
POS, nwin = S.build("code", f"{H}/FableGod/decision_map/out/baseline.json")
byrow = {(q["win"], q["p"]): q for q in POS}
fl_cache = {}
def flags(q):
    if q["win"] not in fl_cache:
        sc = RC.CodeScan(); fl = []
        for t in q["ids"]:
            s = S.dec(t); before = bool(sc.code and sc.com); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            fl.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
        fl_cache[q["win"]] = fl
    return fl_cache[q["win"]]
W = windows()["code"]
tot_blocks = tot_kept = 0; end_miss = end_cut = 0; miss_com = miss_forg = miss_nolook = 0; add_lo = add_hi = 0
com_only = 0; com_only_kept = 0; com_kept = 0
for w in W:
    key = w["nm"][:-4] if w["nm"].endswith(".npz") else w["nm"]
    anyq = next((q for q in POS if q["win"] == key), None)
    if anyq is None: continue
    fl = flags(anyq)
    for rows, cfs, kp in live_blocks(w, bonus="next"):
        A = rows[0]; tot_blocks += 1; tot_kept += kp
        ktok = [p for p in range(A + 1, A + kp + 1) if p < len(fl)]
        nc = sum(1 for p in ktok if fl[p] != 0); com_kept += nc
        if ktok and nc == len(ktok): com_only += 1; com_only_kept += len(ktok)
        if len(rows) > kp:                       # stand-in rows exist: the block ended at a miss on a verified row
            end_miss += 1; p = A + kp - 1        # the row that verified the missed draft (target p + 1)
            q = byrow.get((key, p))
            if q is None: miss_nolook += 1; continue
            if p + 1 < len(fl) and fl[p + 1] != 0: miss_com += 1
            if (not q["right"]) and S.acc(q, "C"):
                miss_forg += 1; add_lo += 1; add_hi += len(rows) - kp
        else: end_cut += 1
print(f"12 code map windows: live blocks {tot_blocks}, kept tokens {tot_kept} ({tot_kept / tot_blocks:.2f} per block); comment tokens among the kept {com_kept} ({100 * com_kept / tot_kept:.1f} %)")
print(f"blocks ended at the cut (no miss among the verified rows): {end_cut} ({100 * end_cut / tot_blocks:.0f} %); ended at a miss on a verified row: {end_miss} ({100 * end_miss / tot_blocks:.0f} %; not in the walk's tested rows {miss_nolook})")
print(f"of the misses: on a comment row {miss_com}; forgiven by rule C {miss_forg}")
print(f"rule C under the live cut: +{add_lo} .. +{add_hi} kept tokens = x{1 + add_lo / tot_kept:.3f} .. x{1 + add_hi / tot_kept:.3f} kept per check (no-cut profile on the same windows: x1.189)")
print(f"blocks whose kept tokens are all comment tokens: {com_only} ({100 * com_only / tot_blocks:.0f} % of the checks) for {com_only_kept} tokens ({com_only_kept / max(1, com_only):.2f} per check)")
