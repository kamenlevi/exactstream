"""r57_draft's own checks before BigMoE's (MoonShots, Wed 18 Nov 2026), on bank windows with MTPHead's state (RA.Head):
install + check_install; per anchor: sync-free vs host-synced (Route A) tree identity, the sync-free draft twice
(bitwise), the MTP cache untouched, ms per pop; with --topk (fmt 'stored', k 1): each chain's drafts up to its first
miss vs dd_topk's TOPK top-1 at (fi, row, depth) (BigMoE's test iii).
usage: ~/BigMoE/.venv/bin/python -I -B r57_draft_test.py --fmt q8h32|c2|stored [--weights <ckpt>] [--set heplus24]
       [--windows 2,3] [--anchors 12] [--n 15] [--k 8] [--no-observe] [--topk] [--out <json>]"""
import argparse
import json
import os
import sys

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
import r57_draft as R  # noqa: E402
import ra_sf as RA  # noqa: E402
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402
import dd_tree as DT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--set", default="heplus24")
    ap.add_argument("--windows", default="2,3")
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-observe", action="store_true", help="add only the roots (r57's flow): exercises the new-extra path")
    ap.add_argument("--topk", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rt, lm, head = DD.M.load(0.03)
    prov = R.install(lm, a.fmt, a.weights)
    chk = R.check_install(lm)
    print(f"[r57_draft_test] check_install: {sum(chk.values())} / {len(chk)} {'' if all(chk.values()) else chk}", flush=True)
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(DT.OUT, f"topk_t30b_{a.set}.npz"))
    res = dict(fmt=a.fmt, prov=prov, check=chk, set=a.set, windows=a.windows, n=a.n, k=a.k, observe=not a.no_observe,
               trees=0, sf_eq_hs=0, sf_twice_bitwise=0, cache_kept=0, bad=0, ms_pop=dict(SF=[], HS=[]), diffs=[],
               topk_cells=0, topk_eq=0, topk_bad=[])
    for fi in [int(x) for x in a.windows.split(",")]:
        z = np.load(FS[fi])
        ids = np.asarray(z["ids"], np.int32)
        gen = int(z["gen"])
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        sel = (TK["depth"] == 0) & (TK["fi"] == fi)
        ancs = np.sort((TK["row"][sel] - 1).astype(np.int64))
        pick = ancs[::max(1, len(ancs) // a.anchors)][:a.anchors]
        mtp = RA.Head(lm)
        R.begin_window(lm, ids[:gen])
        done_upto = -1
        for s in [int(x) for x in pick]:
            if not a.no_observe:
                R.observe(lm, ids[:s + 2])
            mtp.append(W[:, done_upto + 1:s + 1], ids[done_upto + 2:s + 2])
            done_upto = s
            mx.eval(mtp.logits, mtp.h)
            root = int(ids[s + 1])
            L0 = int(mtp.cache[0].offset)
            o1 = R.draft(lm, mtp, root, n=a.n, k=a.k, sync_free=True)
            o2 = R.draft(lm, mtp, root, n=a.n, k=a.k, sync_free=True)
            oh = R.draft(lm, mtp, root, n=a.n, k=a.k, sync_free=False)
            res["trees"] += 1
            res["cache_kept"] += int(int(mtp.cache[0].offset) == L0)
            res["bad"] += int(o1["bad"])
            tw = all(np.array_equal(o1[key], o2[key]) for key in ("tokens", "parents", "depth", "pop", "path_p", "p"))
            res["sf_twice_bitwise"] += int(tw)
            same = np.array_equal(o1["tokens"], oh["tokens"]) and np.array_equal(o1["parents"], oh["parents"])
            res["sf_eq_hs"] += int(same)
            if not same and len(res["diffs"]) < 6:
                d = int(np.argmax((o1["tokens"] != oh["tokens"]) | (o1["parents"] != oh["parents"]))) \
                    if len(o1["tokens"]) == len(oh["tokens"]) else -1
                res["diffs"].append(dict(fi=fi, s=s, first_node=d, sf=o1["tokens"].tolist(), hs=oh["tokens"].tolist()))
            res["ms_pop"]["SF"].append(o2["ms"] / a.n)
            res["ms_pop"]["HS"].append(oh["ms"] / a.n)
            if a.topk:                                        # chain (k 1): the drafts up to the first miss vs TOPK top-1
                toks = o1["tokens"]
                for d in range(1, len(toks)):
                    m_ = (TK["fi"] == fi) & (TK["row"] == s + d) & (TK["depth"] == d - 1)
                    if not m_.any():
                        break
                    t1 = int(TK["top_ids"][m_][0, 0])
                    res["topk_cells"] += 1
                    if int(toks[d]) == t1:
                        res["topk_eq"] += 1
                    elif len(res["topk_bad"]) < 8:
                        res["topk_bad"].append(dict(fi=fi, s=s, depth=d, mine=int(toks[d]), topk=t1,
                                                    topk_p=[float(x) for x in TK["top_p"][m_][0, :2]]))
                    if int(toks[d]) != int(ids[s + 1 + d]):
                        break
    for arm in ("SF", "HS"):
        v = res["ms_pop"][arm][2:] or res["ms_pop"][arm]
        res["ms_pop"][arm + "_med"] = float(np.median(v))
    print(f"[r57_draft_test] {a.fmt} {a.set} w {a.windows} n {a.n} k {a.k}: trees {res['trees']}, SF == HS "
          f"{res['sf_eq_hs']}, SF twice bitwise {res['sf_twice_bitwise']}, cache kept {res['cache_kept']}, bad "
          f"{res['bad']}; ms per pop SF {res['ms_pop']['SF_med']:.3f} HS {res['ms_pop']['HS_med']:.3f}"
          + (f"; TOPK top-1 {res['topk_eq']} / {res['topk_cells']}" if a.topk else ""), flush=True)
    for d in res["diffs"][:3]:
        print("  diff", d, flush=True)
    for d in res["topk_bad"][:4]:
        print("  topk", d, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
