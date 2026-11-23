"""SIB mix oracle, accuracy form (after sb/sib_oracle.py): the frozen s30 head with its carried stand-in at the sibling
slots replaced by a mix of its own carried state and the 2B's sibling state, m = unit(w * unit(sib) + (1 - w) *
unit(own)) at the sibling row's RMS (the post-norm scale the head reads W at), applied online (the chain carries on
from the mixed state). w = 0 is the own arm and w = 1 the sib arm of the main read-out; their cells match it.
Walk-strict and marginal accuracy per band on the main read-out's cells, G = (mix - own) / (true - own) with own / true
from results/sib_s30.json.  usage: sb/sib_mixacc.py --ws 0.1,0.2,0.3,0.4 --out results/sib_mixacc.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def mix_fn(mx, NSIB, rep, msk, w):
    def fn(j, h, anc):
        if j < 2 or j > NSIB + 1:
            return h
        m = msk[anc, j - 2]; r = rep[anc, j - 2]; hh = h[0].astype(mx.float32)
        rms = lambda x: mx.maximum(mx.sqrt((x * x).mean(-1, keepdims=True)), 1e-6)
        rr = rms(r)
        v = w * r / rr + (1 - w) * hh / rms(hh)
        v = v / rms(v) * rr
        return mx.where(m[:, None], v, hh)[None]
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--ws", default="0.1,0.2,0.3,0.4"); ap.add_argument("--sets", default="he,mbpp")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    import mlx.core as mx
    from sb import core as C, sibstate as SB
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "3")))
    C.ARGMAX_ROWS = 256
    C.load_target(); C.drop_trunk()
    D = C.load_drafter("s30")
    ws = [float(x) for x in a.ws.split(",")]
    names = ["own"] + [f"w{w:g}" for w in ws] + ["sib"]
    main_res = json.load(open(SB.H("results", "sib_s30.json")))["res"]
    t0 = time.time(); res = {}
    for st in a.sets.split(","):
        fs = SB.set_files(st, a.n)
        cnt = {nm: np.zeros((5, SB.NSIB + 1), np.int64) for nm in names}
        for f in fs:
            key = os.path.basename(f)[:-4]
            o = np.load(SB.H("data", "sib", st, key + ".npz"))
            if len(o["starts"]) == 0:
                continue
            z = np.load(f); S = len(z["ids"]); a0 = int(o["a0"])
            rep_s, msk, _ = SB.rep_rows(C, mx, o, S, "sib")
            rows = o["starts"] - a0
            fns = {"own": SB.make_fn(mx, None, msk, []), "sib": SB.make_fn(mx, rep_s, msk)}
            fns.update({f"w{w:g}": mix_fn(mx, SB.NSIB, rep_s, msk, w) for w in ws})
            for nm in names:
                cnt[nm] += SB.counts(C.chain_table(D, f, Dp=SB.NSIB + 1, fn=fns[nm])[0][rows])
            mx.clear_cache()
        mr = main_res[f"s30|{st}"]
        r = {}
        for kind, (di, ni) in (("strict", (0, 1)), ("marginal", (2, 3))):
            r[kind] = {}
            for b, (lo, hi) in SB.BANDS.items():
                own_m, true_m = mr[kind][b]["own"], mr[kind][b]["true"]
                row = {}
                for nm in names:
                    v = float(cnt[nm][ni, lo:hi + 1].sum() / max(cnt[nm][di, lo:hi + 1].sum(), 1))
                    row[nm] = {"acc": v, "G": (v - own_m) / (true_m - own_m) if abs(true_m - own_m) > 1e-9 else None}
                row["main_own"], row["main_true"], row["main_sib"] = own_m, true_m, mr[kind][b]["sib"]
                r[kind][b] = row
        res[st] = r
        for b in SB.BANDS:
            x = r["strict"][b]
            print(f"[mixacc] {st} walk-strict {b}: " + ", ".join(f"{nm} {100 * x[nm]['acc']:.2f}" for nm in names) +
                  f" | main own {100 * x['main_own']:.2f} true {100 * x['main_true']:.2f} sib {100 * x['main_sib']:.2f} | "
                  "G " + ", ".join(f"{nm} {x[nm]['G']:+.3f}" for nm in names if nm != "own"), flush=True)
        print(f"[mixacc] {st} done, {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    json.dump({"ws": ws, "res": res, "wall_s": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"[mixacc] DONE -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
