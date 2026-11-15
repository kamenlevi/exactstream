"""WEEK98 K-next scorer (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md: BigMoE's verify-dump
branches (kdump_<set>.jsonl: per check-end miss, the 125B's top-8 at the rows past the drafter's wrong token d_k).
- The 125B's row 0 (after d_k): M1 = P(argmax == y | t30b's fresh depth 0 wrong at the next check), M0 = the same given
  right, P(argmax == y | argmax != t30b's fresh depth 0), top-1/2/3/5/8 containment, and by the target margin bin.
- The drafter's own d1 (its depth k+1 draft after d_k): N1, N0, P(d1 == y | d1 != t30b's fresh depth 0).
- Drift: does row 1 (after d_k, d1) predict y2 when d1 == y?
t30b's fresh depth-0 draft at the next check: the cells' depth-0 cell whose target row is t (= y if right; else
tk[t][rank], rank < 32; unknown if rank 99).
LINES: LIVE if M1 (or N1) >= 50 % on MBPP AND the precision when it disagrees with t30b >= 60 %; derails if M1 < 25 %.
usage: ~/BigMoE/.venv/bin/python -I kdump_score.py heplus24|mbppbank"""
import json, os, sys
import numpy as np

H = os.path.expanduser
st = sys.argv[1]
L = [json.loads(l) for l in open(H(f"~/BigMoE/data/deepdraft/ceiling/kdump_{st}.jsonl"))]
cells = H("~/BigMoE/data/deepdraft/ceiling/" + ("cells_t30b_heplus24.npz" if st.startswith("heplus") else "cells_t30b_mbpp100.npz"))
Z = np.load(cells, allow_pickle=True)
fresh = {}
if "fi" in Z.files and "row" in Z.files:
    files = [os.path.basename(str(f)) for f in Z["files"]]
    for f, r, d, ok, rk in zip(Z["fi"], Z["row"], Z["depth"], Z["right"], Z["rank"]):
        if d == 0:
            fresh[(files[int(f)], int(r))] = (bool(ok), int(rk))
else:                                                    # MBPP: rows by the margin alignment
    sys.path.insert(0, H("~/MoonShots/week98")); import chooser as CH
    c = CH.mbpp_cells()
    for f, r, d, ok, rk in zip(c["fi"], c["row"], c["depth"], c["right"], Z["rank"]):
        if d == 0:
            fresh[(c["files"][int(f)], int(r))] = (bool(ok), int(rk))
BANK = H("~/datasets/deepdraft/bank")
tkc = {}


def fresh_tok(e):
    k = (e["file"], int(e["t"]))
    if k not in fresh:
        return None
    ok, rk = fresh[k]
    if ok:
        return int(e["y"])
    if not (0 <= rk < 32):
        return -1
    if e["file"] not in tkc:
        tkc[e["file"]] = np.asarray(np.load(os.path.join(BANK, e["file"]))["tk"])
    return int(tkc[e["file"]][int(e["t"])][rk])


rows = []
for e in L:
    am = int(e["top8_ids"][0][0]); y = int(e["y"]); f = fresh_tok(e)
    rows.append(dict(am=am, y=y, wrong=not e["t30b_next_d0_right"], fresh=f, d1=e.get("d1"), mt=e["m_target"],
                     top=[int(x) for x in e["top8_ids"][0]], r1=(int(e["top8_ids"][1][0]) if e.get("branch_rows", 1) > 1 and len(e["top8_ids"]) > 1 else None),
                     y2=int(e["y2"]) if e.get("y2") is not None else None))


def rate(sel, hit):
    n = sum(1 for r in rows if sel(r)); h = sum(1 for r in rows if sel(r) and hit(r))
    return dict(pct=round(100 * h / max(n, 1), 1), n=n)


res = {"entries": len(rows), "fresh_known": sum(r["fresh"] is not None for r in rows)}
res["125B row0: P(argmax==y)"] = rate(lambda r: True, lambda r: r["am"] == r["y"])
res["125B row0: M1 (t30b fresh d0 wrong)"] = rate(lambda r: r["wrong"], lambda r: r["am"] == r["y"])
res["125B row0: M0 (t30b fresh d0 right)"] = rate(lambda r: not r["wrong"], lambda r: r["am"] == r["y"])
res["125B row0: P(==y | argmax != t30b fresh d0)"] = rate(lambda r: r["fresh"] not in (None, -1) and r["am"] != r["fresh"], lambda r: r["am"] == r["y"])
for k in (1, 2, 3, 5, 8):
    res[f"125B row0: y in top-{k}"] = rate(lambda r: True, lambda r, k=k: r["y"] in r["top"][:k])
for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 1e9)):
    res[f"125B row0: P(argmax==y) target margin {lo}-{hi if hi < 1e9 else 'inf'}"] = rate(lambda r, lo=lo, hi=hi: lo <= r["mt"] < hi, lambda r: r["am"] == r["y"])
has_d1 = lambda r: r["d1"] is not None and r["d1"] >= 0
res["drafter d1: P(d1==y)"] = rate(has_d1, lambda r: r["d1"] == r["y"])
res["drafter d1: N1 (t30b fresh d0 wrong)"] = rate(lambda r: has_d1(r) and r["wrong"], lambda r: r["d1"] == r["y"])
res["drafter d1: N0 (t30b fresh d0 right)"] = rate(lambda r: has_d1(r) and not r["wrong"], lambda r: r["d1"] == r["y"])
res["drafter d1: P(==y | d1 != t30b fresh d0)"] = rate(lambda r: has_d1(r) and r["fresh"] not in (None, -1) and r["d1"] != r["fresh"], lambda r: r["d1"] == r["y"])
res["drift: row1 argmax==y2 | d1==y"] = rate(lambda r: has_d1(r) and r["d1"] == r["y"] and r["r1"] is not None and r["y2"] is not None, lambda r: r["r1"] == r["y2"])
m1 = res["125B row0: M1 (t30b fresh d0 wrong)"]["pct"]; p1 = res["125B row0: P(==y | argmax != t30b fresh d0)"]["pct"]
n1 = res["drafter d1: N1 (t30b fresh d0 wrong)"]["pct"]; q1 = res["drafter d1: P(==y | d1 != t30b fresh d0)"]["pct"]
res["verdict 125B row0"] = "LIVE" if m1 >= 50 and p1 >= 60 else ("DERAILS (M1 < 25 %)" if m1 < 25 else "BETWEEN")
res["verdict drafter d1"] = "LIVE" if n1 >= 50 and q1 >= 60 else ("NOT LIVE" if n1 < 25 else "BETWEEN")
for k, v in res.items():
    print(f"{k}: {v}")
json.dump(res, open(H(f"~/MoonShots/week98/kdump_score_{st}.json"), "w"), indent=1)
