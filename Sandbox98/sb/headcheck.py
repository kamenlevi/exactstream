"""Head check (one-off, ~2 min): greedy answers for a few MBPP prompts, the capture forward, and the native MTP head's
teacher-forced accuracy at depths 1-8 under the 4 input conventions (W = pre / post final norm) x (fc order [emb, hid] /
[hid, emb]). The best convention becomes core.CFG. -> results/headcheck.json"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.gen import greedy_batch, chat_ids


def main():
    C.limit_memory(6)
    m, tok = C.load_target()
    items = [json.loads(l) for l in open(os.path.expanduser("~/EvalSuite/data/items/dec_mbpp.jsonl"))][:6]
    prompts = [chat_ids(tok, it["user"]) for it in items]
    t0 = time.time()
    outs = greedy_batch(prompts, 256)
    ntok = sum(len(o) for o in outs)
    print(f"[hc] generated {ntok} tokens in {time.time() - t0:.1f}s", flush=True)
    os.makedirs(os.path.join(C.DATA, "hc"), exist_ok=True)
    agree, mx_abs = [], 0.0
    fs = []
    for it, p, o in zip(items, prompts, outs):
        ids = np.array(p + o, np.int32)
        res, post, tp, lg = C.target_states(ids)
        am = np.array(mx.argmax(lg, -1))
        g = len(p)
        agree.append(float((am[g - 1:-1] == ids[g:]).mean()))
        mx_abs = max(mx_abs, float(mx.abs(res).max()))
        f = os.path.join(C.DATA, "hc", it["key"].replace("/", "_") + ".npz")
        np.savez(f, ids=ids, gen=g, res=np.array(res.astype(mx.float16)), asst=(np.arange(len(ids)) >= g).astype(np.uint8),
                 tk=np.zeros((len(ids), 1), np.int32), lp=np.zeros((len(ids), 1), np.float32))
        fs.append(f)
    print(f"[hc] capture argmax == greedy token on answer rows: {np.mean(agree):.4f}; max |res| {mx_abs:.1f}", flush=True)
    D = C.load_drafter("native")
    out = {"agree": agree, "max_abs_res": mx_abs, "tokens": ntok, "conv": {}}
    for wsrc in ("post", "pre"):
        for order in ("ew", "we"):
            C.CFG.update(wsrc=wsrc, order=order)
            tabs = []
            for f in fs:
                rt, gen, a0 = C.chain_table(D, f, Dp=8)
                tabs.append((os.path.basename(f), rt, gen, a0))
            s, _ = C.summarize(tabs)
            dep = [float((np.concatenate([t[1][:, j] for t in tabs]) == 1).sum() /
                         max((np.concatenate([t[1][:, j] for t in tabs]) >= 0).sum(), 1)) for j in range(8)]
            out["conv"][f"{wsrc}/{order}"] = {"depth1_8": dep, "kept": s["kept_per_32"]}
            print(f"[hc] W {wsrc}, fc order {order}: depth 1-8 " + " ".join(f"{100 * v:.1f}" for v in dep)
                  + f"; kept/32 (8 deep) {s['kept_per_32']:.2f}", flush=True)
    os.makedirs(C.RES, exist_ok=True)
    json.dump(out, open(os.path.join(C.RES, "headcheck.json"), "w"), indent=1)
    print(f"[hc] peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
