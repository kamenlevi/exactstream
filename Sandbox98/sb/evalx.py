"""Per-depth read-out (dd_ttt.py's eval, ported): the teacher-forced chain from every anchor of each held-out answer
(anchors gen - 2 .., depth 31), bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 over all valid cells, the strict
conditional rate per depth (right at j given right at 1..j-1), and kept per 32-row check (walk from the answer's first
anchor). Per-window counts are saved for paired comparisons (compare.py).

Modes (--mode, comma-separated; each is scored on every run x set):
  chain          the drafter's own carried stand-in (the real chain)
  true           the stand-in replaced by the TRUE W (the target's post-norm final hidden) at every depth >= 2 (oracle)
  block:K        true W only every K depths (j = 1 + m K), the carried stand-in in between
  map:NAME       the stand-in replaced by a fitted map's prediction (ckpt/maps/NAME.npz, from fitmap.py) at depth >= 2
  noise:F        the true W plus Gaussian noise of F x its per-dim std (R2 ~ 1 - F^2): the R2 -> accuracy curve
Sets: mbpp (100), he (60), mcval (the 40 held-out Magicoder answers); --n N takes the first N windows of each.
usage: evalx.py --runs native,s30 --sets mbpp,he --mode chain,true [--n 100] [--lever NAME] --out results/x.json"""
import argparse, json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C


def set_files(st, n):
    if st == "mcval":
        fs = [os.path.join(C.DATA, "mc", k + ".npz") for k in json.load(open(os.path.join(C.DATA, "split.json")))["val"]]
    else:
        fs = C.files(st)
    return fs[:n] if n else fs


def replacement(mode, f):
    """[1,S,d] rows that replace the stand-in, or None (chain)"""
    if mode == "chain":
        return None, 1
    z = np.load(f)
    W = C.load_W(z).astype(mx.float32)
    if mode == "true":
        return W, 1
    if mode.startswith("block:"):
        return W, int(mode.split(":")[1])
    if mode.startswith("noise:"):
        fr = float(mode.split(":")[1])
        sd = mx.sqrt(((W - W.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True))
        key = mx.random.key(zlib.crc32(os.path.basename(f).encode()))
        return W + fr * sd * mx.random.normal(W.shape, key=key), 1
    if mode.startswith("map:"):
        from sb.fitmap import apply_map
        return apply_map(mode.split(":", 1)[1], f)[None], 1
    raise ValueError(mode)


def cond_counts(rt):
    """one window's walk-strict counts per depth, as in C.summarize's cond: r[j] right at j with right at 0..j-1,
    v[j] the valid cells still alive at j"""
    r, v = np.zeros(rt.shape[1], np.int64), np.zeros(rt.shape[1], np.int64)
    alive = np.ones(rt.shape[0], bool)
    for j in range(rt.shape[1]):
        ok = alive & (rt[:, j] >= 0)
        r[j], v[j] = int((ok & (rt[:, j] == 1)).sum()), int(ok.sum())
        alive &= rt[:, j] == 1
    return r, v


def run_eval(D, fs, mode, Dp=31, fn=None, pc=None):
    """pc: a dict to fill with each window's walk-strict counts {window: [r per depth, v per depth]} (sb/driftg.py's
    paired bootstrap); their sums must equal the summary's cond_n"""
    tabs = []
    for f in fs:
        C.wait_pause()
        R, K = replacement(mode, f)
        rt, gen, a0 = C.chain_table(D, f, Dp=Dp, R=R, K=K, fn=fn)
        tabs.append((os.path.basename(f)[:-4], rt, gen, a0))
        mx.clear_cache()
    s, per = C.summarize(tabs)
    if pc is not None:
        for key, rt, _, _ in tabs:
            r, v = cond_counts(rt)
            pc[key] = [r.tolist(), v.tolist()]
        tot = np.zeros(len(s["cond_n"]), np.int64)
        for r, v in pc.values():
            tot[:len(v)] += v
        assert tot.tolist() == s["cond_n"], "per-window walk-strict counts do not add up to cond_n"
    return s, per


def fmt(s):
    return (" ".join(f"{b}:{100 * s[b]:.2f}" for b in s if "-" in b and not b.startswith("cond") and s[b] is not None)
            + f" | kept/{s.get('kept_max') or 32} {s['kept_per_32']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="native"); ap.add_argument("--sets", default="mbpp,he")
    ap.add_argument("--mode", default="chain"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--depth", type=int, default=31); ap.add_argument("--lever", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--mem", type=float, default=4.5)
    ap.add_argument("--window", type=int, default=0, help="eval-only: cap the head's attention to the last n true pairs")
    ap.add_argument("--nm", type=int, default=0, help="MBPP windows (0: --n)"); ap.add_argument("--nh", type=int, default=0, help="HumanEval windows (0: --n)")
    a = ap.parse_args()
    C.limit_memory(a.mem)
    C.WINDOW["n"] = a.window
    C.load_target(); C.drop_trunk()
    t0 = time.time()
    out = {"args": vars(a), "res": {}, "per": {}, "pcond": {}}
    names = [sp.partition("@")[0] for sp in a.runs.split(",")]
    for spec in a.runs.split(","):
        run, _, lev = spec.partition("@")                   # 'run@lever': a per-run lever (multi-arm batches)
        rtag = spec if names.count(run) > 1 else run        # one drafter read out under several levers: tag by lever
        lev = lev or a.lever
        D = C.load_drafter(run, strict=not lev)
        fn = None
        if lev:
            from sb.train import load_lever
            L = load_lever(lev)
            if hasattr(L, "setup"):
                L.setup(D, a)
                D.load_weights(os.path.join(C.CKPT, run + ".safetensors"), strict=False) if run != "native" else None
            fn = getattr(L, "standin", None)
        for st in a.sets.split(","):
            fs = set_files(st, {"mbpp": a.nm, "he": a.nh}.get(st) or a.n)
            for mode in a.mode.split(","):
                pc = {}
                s, per = run_eval(D, fs, mode, a.depth, fn, pc=pc)
                tag = f"{rtag}|{st}|{mode}"
                out["res"][tag] = s; out["per"][tag] = per; out["pcond"][tag] = pc
                print(f"[ev] {tag} ({len(fs)}): {fmt(s)}  [{time.time() - t0:.0f}s]", flush=True)
    out["wall_s"] = time.time() - t0; out["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
