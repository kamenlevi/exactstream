"""PK v2 QUIET timing (pre-registered M-46 / ALL_LEVERS item 1): PK vs SF interleaved on the same real windows and
anchors (tree-16, k 8), the first 2 rounds dropped; ms per pop and per tree-16 block (15 pops), the byte floor
(178.8 MB at 292.7 GB/s = 0.611 ms) -> f = pop - floor. Run under `peer lock quiet` only (MoonShots, Wed 18 Nov 2026).
usage: ~/BigMoE/.venv/bin/python -I -B pk_timing.py [--rounds 8] [--out pk_timing_quiet.json]"""
import argparse
import json
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
import ra_sf as RA  # noqa: E402
import r57_draft as R  # noqa: E402
import pk_level as PKL  # noqa: E402
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402
import dd_tree as DT  # noqa: E402

FLOOR_MS = 178.8e6 / 292.7e9 * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="pk_timing_quiet.json")
    a = ap.parse_args()
    rt, lm, head = DD.M.load(0.03)
    R.install(lm, "q8h32")
    pk = PKL.PK(lm)
    states = []                                                    # (view, root, sh, imap) per anchor, built once
    for set_, FS, wins in (("heplus24", CE.HEP, (2, 3)), ("mbppbank", CE.MBB, (4, 5))):
        TK = np.load(os.path.join(DT.OUT, f"topk_t30b_{set_}.npz"))
        for fi in wins:
            z = np.load(FS[fi])
            ids = np.asarray(z["ids"], np.int32)
            W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
            sel = (TK["depth"] == 0) & (TK["fi"] == fi)
            ancs = np.sort((TK["row"][sel] - 1).astype(np.int64))
            for s in [int(x) for x in ancs[::max(1, len(ancs) // 3)][:3]]:
                R.begin_window(lm, ids[:int(z["gen"])])
                R.observe(lm, ids[:s + 2])
                mtp = RA.Head(lm)
                mtp.append(W[:, :s + 1], ids[1:s + 2])
                mx.eval(mtp.logits, mtp.h)
                sh = R.ST["sh"]
                imap = RA.idmap(sh)
                rowv = R._screened_row(mtp.logits[0, -1], sh)
                mx.eval(rowv)
                states.append((R._Root(mtp.cache, rowv[None, None, :], mtp.h), int(ids[s + 1]), sh, imap, set_))
    res = dict(floor_ms=FLOOR_MS, rounds=a.rounds, trees=len(states), sf=[], pk=[], bitwise=0, checked=0)
    RA.SCREENED["on"] = True
    try:
        for r in range(a.rounds):
            for arm in (("sf", "pk") if r % 2 == 0 else ("pk", "sf")):
                mx.synchronize()
                t0 = time.perf_counter()
                outs = []
                for view, root, sh, imap, _ in states:
                    if arm == "sf":
                        RA.SCREEN["sh"] = sh                       # the state's own screen for the draft head
                        outs.append(RA.draft_sf(lm, view, root, n=15, k=8, imap=imap))
                    else:
                        outs.append(pk.draft(view, root, sh, imap))
                mx.synchronize()
                dt = (time.perf_counter() - t0) * 1e3 / (15 * len(states))
                res[arm].append(dt)
                if arm == "pk" and r == a.rounds - 1:
                    sfo = []
                    for v, rt_, sh_, im, _ in states:
                        RA.SCREEN["sh"] = sh_
                        sfo.append(RA.draft_sf(lm, v, rt_, n=15, k=8, imap=im))
                    for o1, o2 in zip(sfo, outs):
                        res["checked"] += 1
                        res["bitwise"] += int(o2 is not None and all(np.array_equal(o1[k], o2[k]) for k in
                                                                    ("tokens", "parents", "pop", "path_p", "p")))
    finally:
        RA.SCREENED["on"] = False
    for arm in ("sf", "pk"):
        v = sorted(res[arm][2:])
        res[arm + "_med"] = v[len(v) // 2]
    res["f_pk"] = res["pk_med"] - FLOOR_MS
    res["gain_ms_per_pop"] = res["sf_med"] - res["pk_med"]
    res["gain_ms_per_block16"] = 15 * res["gain_ms_per_pop"]
    print(f"[pk_timing] ms per pop (median of {a.rounds - 2} rounds, {len(states)} trees): SF {res['sf_med']:.3f} PK "
          f"{res['pk_med']:.3f} (-{res['gain_ms_per_pop']:.3f}, -{res['gain_ms_per_block16']:.2f} ms per tree-16 block); "
          f"f = {res['f_pk']:.3f} vs floor {FLOOR_MS:.3f}; bitwise {res['bitwise']} / {res['checked']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
