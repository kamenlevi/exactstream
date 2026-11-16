"""WEEK98 BET 7 deliverable (3): the fused level's fixed cost and ms per block, against the stock level (MoonShots,
Mon 16 Nov 2026; protocol pre-registered in LEVERS_A "BET 7", 6bb1a0c).

Arms (interleaved, order alternating per round, the first 2 rounds dropped): stock (fl_stock.Stock.level) and fused
(fl_fused.Fused.level), F = 1, K = 8, L = 21 levels per block, async_eval per level and one sync per block (M1's
protocol), at two screen sizes over the same weights (nh = 32,768: 177.6 MB per level = C0; nh = 8,192: 142.2 MB),
and at two cache lengths (T0 = 400: the one-pass attention; T0 = 1,500: the two-pass form).
Reported per arm: wall ms per level (median [p10-p90]), CPU ms per level (graph building + async_eval);
f = ms - MB / 292.7 GB/s (the pre-registered definition, M2's BW) and each arm's own two-point fit over the screens;
ms per block = levels x ms per level at tree-16 (12.6) and tree-32 (20.8) [P]; the ~26 MB and C1 (95.1 MB) points as
projections [P] = levels x (f + MB / BW). Self-check: the fused tokens equal the stock tokens in every timed block.
Timing under a share lock beside other GPU jobs is INDICATIVE; the verdict uses a quiet slot.
usage: ~/BigMoE/.venv/bin/python -I -B fl_time.py [--rounds 20] [--out <json>]   (MEM ~3 GB)"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402

import fl_fused                                             # noqa: E402
import fl_stock as FS                                       # noqa: E402
from screened_head import ScreenedHead                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--L", type=int, default=21)
ap.add_argument("--T0s", default="400,1500")
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_time.json"))
A = ap.parse_args()
BW = 292.7
MB = {32768: 177.6, 8192: 142.2}


def say(s=""):
    print(s, flush=True)


def gpu_busy():
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                           text=True, timeout=5).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


t00 = time.time()
T0S = [int(t) for t in A.T0s.split(",")]
st = FS.Stock(nh=32768, T0=T0S[0])
heads = {32768: st.sh, 8192: ScreenedHead(st.head, FS.RANK[:8192], extra_cap=16)}
fu = fl_fused.Fused(st)
fu2 = fl_fused.Fused2(st)
say(f"built at {time.time() - t00:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; GPU busy {gpu_busy()} %")


def set_T0(T0):
    """re-prefill the cache at T0 rows (P2's synthetic states)"""
    st.cache = [FS.Q.QSACache(FS.args.indexer_compress_ratio or 4)]
    rng = np.random.default_rng(5)
    Wh = mx.array((rng.standard_normal((1, T0, FS.HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    ids0 = mx.array(rng.integers(0, 200000, (1, T0)).astype(np.int32))
    mx.eval(st.mtp.fuse_and_run(Wh, st.emb(ids0), st.cache))


def run(arm, nh, tok0):
    st.sh = heads[nh]
    lvl = {"stock": st.level, "fused": fu.level, "fused2": fu2.level}[arm]
    h, tok, sc = st.h0, mx.array([[tok0]], dtype=mx.int32), mx.zeros((1,), mx.float32)
    prep = None
    t_cpu = 0.0
    toks = []
    t0 = time.perf_counter()
    for _ in range(A.L):
        a = time.perf_counter()
        o = lvl(h, tok, sc, 1, 8, prep=prep) if arm == "fused2" else lvl(h, tok, sc, 1, 8)
        h, tok, sc = o["h"], o["tok"], o["sc"]
        if arm == "fused2":
            prep = o["prep"]
            mx.async_eval(h, tok, sc, o["kidx"], o["klp"], prep[0], prep[1])
        else:
            mx.async_eval(h, tok, sc, o["kidx"], o["klp"])
        t_cpu += time.perf_counter() - a
        toks.append(tok)
    mx.eval(toks, h, sc)
    t1 = time.perf_counter()
    st.cache[0].trim(A.L)
    return (t1 - t0) / A.L, t_cpu / A.L, [int(t.reshape(-1)[0].item()) for t in toks]


res = {"rounds": A.rounds, "L": A.L, "bw": BW, "arms": {}, "busy": {}, "selfcheck": {}}
for T0 in T0S:
    set_T0(T0)
    arms = [(a, nh) for nh in (32768, 8192) for a in ("stock", "fused", "fused2")]
    W = {k: [] for k in arms}
    C = {k: [] for k in arms}
    busy = []
    same = True
    for r in range(A.rounds + 2):
        busy.append(gpu_busy())
        order = arms if r % 2 == 0 else arms[::-1]
        toks = {}
        for k in order:
            w, c, tk = run(k[0], k[1], 1000 + r)
            toks[k] = tk
            if r >= 2:
                W[k].append(1e3 * w)
                C[k].append(1e3 * c)
        for nh in (32768, 8192):
            same &= toks[("stock", nh)] == toks[("fused", nh)] == toks[("fused2", nh)]
    res["selfcheck"][T0] = same
    res["busy"][T0] = busy
    say(f"T0 {T0}: {A.rounds} rounds; fused tokens == stock tokens in every block: {same}; GPU busy at round start "
        f"{busy}")
    for k in arms:
        x = np.array(W[k])
        res["arms"][f"{T0} {k[0]} {k[1]}"] = dict(
            ms=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
            p90=round(float(np.percentile(x, 90)), 4), cpu=round(float(np.median(C[k])), 4), mb=MB[k[1]],
            f=round(float(np.median(x)) - MB[k[1]] / BW, 4))
    for a in ("stock", "fused", "fused2"):
        r32, r8 = res["arms"][f"{T0} {a} 32768"], res["arms"][f"{T0} {a} 8192"]
        bw = (MB[32768] - MB[8192]) / (r32["ms"] - r8["ms"]) if r32["ms"] > r8["ms"] else float("nan")
        f2 = r32["ms"] - MB[32768] / bw if bw == bw else float("nan")
        res["arms"][f"{T0} {a} fit"] = dict(bw=round(bw, 1), f=round(f2, 4))

say("")
say("ms per level, median [p10-p90] | CPU ms per level | f = ms - MB / 292.7")
for k, v in res["arms"].items():
    if "fit" in k:
        say(f"  {k:24s} two-point fit over the screens: BW {v['bw']} GB/s, f {v['f']:.4f} ms")
    else:
        say(f"  {k:24s} {v['ms']:.4f} [{v['p10']:.4f}-{v['p90']:.4f}] | CPU {v['cpu']:.4f} | MB {v['mb']} | "
            f"f {v['f']:.4f}")
say("")
say("ms per block (F = 1 chain shape; levels x ms per level) [P]:")
for T0 in T0S:
    for a in ("stock", "fused", "fused2"):
        ms = res["arms"][f"{T0} {a} 32768"]["ms"]
        f = res["arms"][f"{T0} {a} 32768"]["f"]
        line = (f"  T0 {T0} {a:6s}: full head (177.6 MB) tree-16 {12.6 * ms:.2f} / tree-32 {20.8 * ms:.2f} ms (measured "
                f"per level); C1 95.1 MB {12.6 * (f + 95.1 / BW):.2f} / {20.8 * (f + 95.1 / BW):.2f}; ~26 MB "
                f"{12.6 * (f + 26 / BW):.2f} / {20.8 * (f + 26 / BW):.2f} [P]")
        res.setdefault("blocks", {})[f"{T0} {a}"] = line
        say(line)
json.dump(res, open(A.out, "w"), indent=1)
say(f"wrote {A.out} ({time.time() - t00:.0f} s)")
