"""Capture (resumable): greedy answers of the frozen target + one full forward over prompt + answer per item, stored per
item in data/<set>/<key>.npz:
  ids [S] int32, gen (prompt length), asst [S] uint8 (1 = answer token), res [S, 2048] fp16 (final residual, PRE-norm;
  the head reads final_norm(res)), tk [S, 32] int32 + lp [S, 32] fp16 (top-32 log-probs of row t, predicting ids[t+1]),
  am [S] int32 (the capture forward's argmax), stop (1 if the answer ended with an end token);
and data/<set>/<key>.taps.npz: taps [5, S, 2048] fp16 = the residual after layers core.TAPS (2, 7, 12, 17, 22).
Sets: mbpp (EvalSuite dec_mbpp, 100), he (EvalSuite dec_humaneval, 60), mc (Magicoder prompts_py_v1 in its seeded order:
the same first prompts the 125B's n1112 capture took; --n of them).
usage: capture.py --sets mbpp,he,mc --n 540 [--max-tokens 512] [--batch 24]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.gen import greedy_batch, chat_ids

H = os.path.expanduser
ITEMS = H("~/EvalSuite/data/items")
MAGIC = H("~/datasets/deepdraft/magicoder/prompts_py_v1.jsonl")


def items_of(st, n):
    if st == "mbpp":
        rows = [json.loads(l) for l in open(os.path.join(ITEMS, "dec_mbpp.jsonl"))]
        return [(r["key"].replace("/", "_"), r["user"]) for r in rows]
    if st == "he":
        rows = [json.loads(l) for l in open(os.path.join(ITEMS, "dec_humaneval.jsonl"))]
        return [(r["key"].replace("/", "_"), r["user"]) for r in rows]
    if st == "mc":
        out = []
        for l in open(MAGIC):
            r = json.loads(l)
            out.append((r["key"], r["problem"]))
            if len(out) >= n:
                break
        return out
    if st == "mc2":                                        # 22:3x: fresh screen-pool answers, the prompts after mc's 540
        out = []
        for i, l in enumerate(open(MAGIC)):
            if i < 540:
                continue
            r = json.loads(l)
            out.append((r["key"], r["problem"]))
            if len(out) >= n:
                break
        return out
    raise ValueError(st)


def save(st, key, prompt, ans, eos):
    ids = np.array(prompt + ans, np.int32)
    res, post, tp, _ = C.target_states(ids, logits=False)
    tks, lps, ams = [], [], []
    for c0 in range(0, len(ids), 256):                 # the 248K-wide logits in row chunks (memory)
        lg = C.lm_logits(post[c0:c0 + 256])
        lp_all = lg - mx.logsumexp(lg, -1, keepdims=True)
        t32 = mx.argpartition(-lp_all, kth=31, axis=-1)[:, :32]
        v32 = mx.take_along_axis(lp_all, t32, -1)
        o = mx.argsort(-v32, axis=-1)
        tks.append(mx.take_along_axis(t32, o, -1)); lps.append(mx.take_along_axis(v32, o, -1)); ams.append(mx.argmax(lg, -1))
        mx.eval(tks[-1], lps[-1], ams[-1])
    tk, lp, am = mx.concatenate(tks), mx.concatenate(lps), mx.concatenate(ams)
    taps = mx.stack([tp[i] for i in C.TAPS], 0)
    mx.eval(res, tk, lp, am, taps)
    d = os.path.join(C.DATA, st)
    f = os.path.join(d, key + ".npz")
    np.savez(f + ".tmp.npz", ids=ids, gen=len(prompt), asst=(np.arange(len(ids)) >= len(prompt)).astype(np.uint8),
             res=np.array(res.astype(mx.float16)), tk=np.array(tk).astype(np.int32), lp=np.array(lp.astype(mx.float16)),
             am=np.array(am).astype(np.int32), stop=int(len(ans) > 0 and ans[-1] in eos))
    np.savez(os.path.join(d, key + ".taps.npz.tmp.npz"), taps=np.array(taps.astype(mx.float16)))
    os.replace(os.path.join(d, key + ".taps.npz.tmp.npz"), os.path.join(d, key + ".taps.npz"))
    os.replace(f + ".tmp.npz", f)                      # the main file last: its presence means the item is complete
    g = len(prompt)
    return float((np.array(am)[g - 1:-1] == ids[g:]).mean()) if len(ids) > g else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="mbpp,he,mc"); ap.add_argument("--n", type=int, default=540)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=512); ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    C.limit_memory(7)
    m, tok = C.load_target()
    eos = set(tok.eos_token_ids)
    for st in a.sets.split(","):
        os.makedirs(os.path.join(C.DATA, st), exist_ok=True)
        todo = [(k, u) for k, u in items_of(st, a.n) if not os.path.exists(os.path.join(C.DATA, st, k + ".npz"))]
        print(f"[cap] {st}: {len(todo)} to do", flush=True)
        t0, ntok, agree = time.time(), 0, []
        for b0 in range(0, len(todo), a.batch):
            part = todo[b0:b0 + a.batch]
            prompts = [chat_ids(tok, u) for _, u in part]
            t1 = time.time()
            outs = greedy_batch(prompts, a.max_tokens, batch=a.batch)
            tg = time.time() - t1
            for (k, _), p, o in zip(part, prompts, outs):
                agree.append(save(st, k, p, o, eos))
                ntok += len(o)
            mx.clear_cache()
            print(f"[cap] {st} {b0 + len(part)}/{len(todo)}: {ntok} answer tokens, gen {sum(len(o) for o in outs) / tg:.0f} tok/s, "
                  f"capture argmax agreement {np.mean(agree):.4f}, {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB",
                  flush=True)
    print("[cap] DONE", flush=True)


if __name__ == "__main__":
    main()
