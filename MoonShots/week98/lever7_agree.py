"""WEEK98 lever 7 (and lever 1 context), MoonShots Sun 15 Nov 2026: how often does Qwen3.5-2B's own greedy token equal the
125B's greedy token on code answers? Teacher-forced on the bank's val files (40 n1112 files t30b never saw), every answer
row; the bank's answers are the 125B's greedy, so ids[r + 1] is the 125B's pick (PROXY.md: 100 % agreement checked).
Split by the 125B's own margin (top-1 minus top-2 log-prob, the bank's lp), as CEILING did.

LINE (written first): if the 2B agrees on >= 90 % of the 125B's SURE rows (margin >= 4 nats), it is a usable auditor for
lever 7 (price the cut rule next); if < 75 %, KILL lever 7.
Read-only: ~/datasets/deepdraft/bank, the proxy's rows_va.json, Sandbox98's sb/core.py. Writes ~/MoonShots/week98/ only.
Also on the eval sets (--set mbpp / heplus: every answer row, reference = the 125B's runtime top-1, tk[:, 0]).
usage: ~/BigMoE/.venv/bin/python -I lever7_agree.py [--set val|mbpp|heplus]"""
import json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98"))
BANK = H("~/datasets/deepdraft/bank")
OUT = H("~/MoonShots/week98")


def main():
    from sb import core as C
    import mlx.core as mx
    C.limit_memory(6)
    t0 = time.time()
    st = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "val"
    by_file = {}
    if st == "val":
        for nm, r in json.load(open(os.path.join(BANK, "proxy", "rows_va.json"))):
            by_file.setdefault(nm, []).append(r)
    else:
        for nm in json.load(open(H("~/BigMoE/data/deepdraft/bank/split.json")))[st]:
            z = np.load(os.path.join(BANK, nm)); n, gen = z["raw"].shape[0], int(z["gen"])
            by_file[nm] = list(range(gen - 1, n))
    dump = {"file": [], "row": [], "top1": [], "margin2b": [], "tk0": [], "margin125": []}
    bins = {"all": [0, 0], "m<0.25": [0, 0], "0.25-1": [0, 0], "1-4": [0, 0], "m>=4": [0, 0]}
    for k, (nm, rr) in enumerate(by_file.items()):
        C.wait_pause()
        z = np.load(os.path.join(BANK, nm))
        ids, lp, tk = np.asarray(z["ids"]), np.asarray(z["lp"], np.float32), np.asarray(z["tk"])
        _, post, _, _ = C.target_states(ids, taps=(), logits=False)
        rr = np.array([r for r in rr if r + 1 < len(ids)])
        top, mg = [], []
        for i in range(0, len(rr), 256):
            lg = C.tm().embed_tokens.as_linear(post[mx.array(rr[i:i + 256])]).astype(mx.float32)
            ls = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            t2 = mx.topk(ls, 2, axis=-1)                     # an unsorted pair: the 2B's own margin = |a - b|
            top.append(np.asarray(mx.argmax(lg, axis=-1))); mg.append(np.abs(np.asarray(t2[:, 0] - t2[:, 1])))
        top, mg = np.concatenate(top), np.concatenate(mg)
        hit = top == tk[rr, 0]                            # the 125B's own runtime top-1 (= ids[r + 1] on greedy answers)
        m = lp[rr, 0] - lp[rr, 1]
        dump["file"] += [nm] * len(rr); dump["row"].append(rr); dump["top1"].append(top); dump["margin2b"].append(mg)
        dump["tk0"].append(tk[rr, 0]); dump["margin125"].append(m)
        for key, sel in (("all", np.ones_like(hit)), ("m<0.25", m < 0.25), ("0.25-1", (m >= 0.25) & (m < 1)),
                         ("1-4", (m >= 1) & (m < 4)), ("m>=4", m >= 4)):
            bins[key][0] += int((hit & sel).sum()); bins[key][1] += int(sel.sum())
        if k % 10 == 0:
            print(f"{k}/{len(by_file)} files {time.time() - t0:.0f}s", flush=True)
    res = {k: dict(agree=v[0] / max(v[1], 1), rows=v[1]) for k, v in bins.items()}
    for k, v in res.items():
        print(f"[agree] {k}: 2B top-1 == 125B greedy on {100 * v['agree']:.2f} % of {v['rows']} rows", flush=True)
    np.savez_compressed(os.path.join(OUT, f"rows2b_{st}.npz"), fname=np.array(dump["file"]),
                        **{k: np.concatenate(v) for k, v in dump.items() if k != "file"})
    json.dump(dict(res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, f"lever7_agree_{st}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
