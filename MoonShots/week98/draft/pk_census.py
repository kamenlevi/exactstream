"""PK v2 census (MoonShots, Wed 18 Nov 2026): what one live sync-free draft (ra_sf.draft_sf, r57's live MTP head with
the format installed) dispatches and reads. Per tree and per forward: dispatches per stock pipeline (mtlprof), command
buffers, GPU busy time; per pop: the bytes each module reads. -> the PK v2 phase list and its byte floor.
usage: python pk_census.py [q8h32 | c2 [weights]]   (share lock; writes pk_census_<fmt>.json)"""
import json
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Leap/third_party/mtlprof"))
import mtlprof as MP  # noqa: E402
MP.load(2)
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
import ra_sf as RA  # noqa: E402
import sf_tree as SFT  # noqa: E402
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402

fmt = sys.argv[1] if len(sys.argv) > 1 else "q8h32"
weights = sys.argv[2] if len(sys.argv) > 2 else None
rt, lm, head = DD.M.load(0.03)
done, new_screen, screen = RA.install_format(lm, fmt, weights)
z = np.load(CE.HEP[0])
ids = np.asarray(z["ids"], np.int32)
W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
s = 182
mtp = RA.Head(lm)
sh = new_screen()
sh.add(ids[:s + 2])
mtp.append(W[:, :s + 1], ids[1:s + 2])
mx.eval(mtp.logits, mtp.h)
imap = RA.idmap(sh)
root = int(ids[s + 1])
for _ in range(4):
    RA.draft_sf(lm, mtp, root, imap=imap)


def census(fn, reps=6):
    out = []
    for _ in range(reps):
        mx.synchronize()
        p0, c0 = MP.psos(), MP.counts()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        t1 = time.perf_counter()
        c1, p1 = MP.counts(), MP.psos()
        cb = MP.cbs(c0["ncb"], c1["ncb"])
        rd = cb[cb["ready"] == 1]
        busy = MP.busy(cb, float(rd["start"].min()), float(rd["end"].max())) if len(rd) else 0.0
        out.append(dict(wall_us=(t1 - t0) * 1e6, gpu_busy_us=busy * 1e6, ncb=int(c1["ncb"] - c0["ncb"]),
                        disp=int(c1["disp"] - c0["disp"]), pso=MP.pso_delta(p0, p1)))
    med = {k: float(np.median([o[k] for o in out[1:]])) for k in ("wall_us", "gpu_busy_us", "ncb", "disp")}
    med["pso"] = out[-1]["pso"]
    return med


qc = mtp.cache[0]
Lc = int(qc.offset)
Kc, Vc = qc.kv.keys[..., :Lc, :], qc.kv.values[..., :Lc, :]
tok = mx.array(int(mx.argmax(mtp.logits[0, -1]).item()), mx.int32)
pos = mx.array([Lc], mx.int32)


def one_fwd():
    lg, h, k, v = RA.forward_static(lm, mtp.h, tok, pos, Kc, Vc, None, None, None)
    mx.eval(lg, h, k, v)


row = mtp.logits[0, -1]
G = SFT.GPUFrontier(8, 15)


def one_tail():
    tis, tv, tp = RA.topk_ra(row, 8)
    mx.eval(tis, tp)


def one_step():
    tis, tv, tp = RA.topk_ra(row, 8)
    mx.eval(G.step(mx.take(imap, tis).astype(mx.int32), tp, 0), *G.arrays())


res = dict(fmt=fmt, weights=weights, Lc=Lc, screen=screen, done=done)
res["tree"] = census(lambda: RA.draft_sf(lm, mtp, root, imap=imap))
res["forward"] = census(one_fwd)
res["topk"] = census(one_tail)
res["topk_step"] = census(one_step)

# bytes per pop: the head's parameters as the static forward reads them (experts: top_k of n slices; the indexer and
# any module the static form skips are listed but not counted), the screened head, the embedding row, the prefix KV
top = int(head.layers[0].mlp.top_k)
rows, tot = [], 0.0
for k, v in tree_flatten(head.parameters()):
    nb = float(v.nbytes)
    used = "indexer" not in k
    if "switch_mlp" in k:
        nb = nb / v.shape[0] * top
    rows.append(dict(name=k, shape=list(v.shape), dtype=str(v.dtype), mb=nb / 1e6, used=used))
    tot += nb if used else 0.0
shb = {n: float(getattr(sh, n).nbytes) for n in ("w", "s", "b", "xw", "xs", "xb")}
et = lm.model.embed_tokens
emb_row = sum(float(getattr(et, n).nbytes) / getattr(et, n).shape[0] for n in ("weight", "scales", "biases")
              if hasattr(et, n) and getattr(et, n) is not None)
kv = float(Kc.nbytes + Vc.nbytes)
tot_all = tot + sum(shb.values()) + emb_row + kv
res["bytes"] = dict(head_params_mb=tot / 1e6, screened_head_mb={k: v / 1e6 for k, v in shb.items()},
                    emb_row_kb=emb_row / 1e3, prefix_kv_mb=kv / 1e6, total_mb=tot_all / 1e6,
                    floor_292_us=tot_all / 292.7e3, floor_260_us=tot_all / 260e3, top_k=top, params=rows)
json.dump(res, open(H(f"~/MoonShots/week98/draft/pk_census_{fmt}.json"), "w"), indent=1)
b = res["bytes"]
print(f"{fmt}: bytes per pop {b['total_mb']:.1f} MB (head params {b['head_params_mb']:.1f}, screened head "
      f"{sum(b['screened_head_mb'].values()):.1f}, prefix KV {b['prefix_kv_mb']:.2f}) -> floor {b['floor_292_us']:.0f} us "
      f"@292.7, {b['floor_260_us']:.0f} us @260 GB/s")
for nm in ("tree", "forward", "topk", "topk_step"):
    r = res[nm]
    print(f"{nm}: wall {r['wall_us']:.0f} us | GPU busy {r['gpu_busy_us']:.0f} us | cbs {r['ncb']:.0f} | dispatches {r['disp']:.0f}")
print("forward pipelines:")
for k, v in sorted(res["forward"]["pso"].items(), key=lambda x: -x[1]):
    print(f"  {v:4d}  {k}")
print("tree pipelines not in the forward:", {k: v for k, v in res["tree"]["pso"].items() if k not in res["forward"]["pso"]})
for r in rows:
    if r["mb"] > 0.05 or not r["used"]:
        print(f"  {r['mb']:8.2f} MB {'' if r['used'] else '(unused) '}{r['name']} {r['shape']} {r['dtype']}")
