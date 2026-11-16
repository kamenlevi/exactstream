"""TrainLab decode test on the 35B (Kamen 17:10: the small-RAM bar is task results): the Eval suite's decode items
(dec_math 50 = MATH-500 + GSM8K, dec_ifeval 100; their chat-templated non-thinking prompt_ids), greedy, under a
dataflow with or without a trained run's LoRA. Deployed semantics: one LRU per sequence and layer that warms over the
prompt and keeps its state through the generation. Batched with left padding (mlx_lm ArraysCache / BatchKVCache);
finished rows leave the batch. Rows -> data/decode/<arm>.jsonl (resumable); score with score_decode.py (Eval suite
venv: its IFEval checkers and math normaliser), incl. answer agreement with the exact arm.
usage: decode35.py ARM FLOW [--run NAME] [--sets math,ifeval] [--n 100] [--batch 16] [--time-limit S]
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train35 as T

ITEMS = os.path.expanduser("~/EvalSuite/data/items")
EOS = {248044, 248046}


def make_caches(model, pads):
    from mlx_lm.models.cache import ArraysCache, BatchKVCache
    return [ArraysCache(size=2, left_padding=pads) if l.is_linear else BatchKVCache(left_padding=pads)
            for l in model.language_model.model.layers]


def filter_caches(caches, keep):
    idx = mx.array(keep)
    for c in caches:
        c.filter(idx)


def decode_batch(model, flow, items, log):
    """items: list of dicts with prompt_ids, max_tokens, key. Returns {key: (completion ids, finish)}."""
    B = len(items)
    lens = [len(it["prompt_ids"]) for it in items]
    Tm = max(lens)
    pads = [Tm - n for n in lens]
    ids = np.zeros((B, Tm), np.int32)
    for b, it in enumerate(items):
        ids[b, pads[b]:] = it["prompt_ids"]
    caches = make_caches(model, pads)
    T.Cache.persist, T.Cache.seq_ids = {}, [it["key"] for it in items]
    T.Cache.valid = np.arange(Tm)[None, :] >= np.array(pads)[:, None]
    h, _ = T.forward(model, mx.array(ids), flow, False, caches=caches)
    nxt = mx.argmax(model.language_model.lm_head(h[:, -1:]).astype(mx.float32), axis=-1)       # [B, 1]
    mx.eval(nxt)
    T.Cache.valid = None
    live = list(range(B))
    outs = {b: [] for b in range(B)}
    done = {}
    while live:
        tok = np.array(nxt)[:, 0]
        keep = []
        for j, b in enumerate(live):
            t = int(tok[j]); outs[b].append(t)
            if t in EOS or len(outs[b]) >= items[b]["max_tokens"]:
                done[items[b]["key"]] = (outs[b], "eos" if t in EOS else "length")
            else:
                keep.append(j)
        if not keep:
            break
        if len(keep) < len(live):
            filter_caches(caches, keep)
            live = [live[j] for j in keep]
            tok = tok[keep]
        T.Cache.seq_ids = [items[b]["key"] for b in live]
        h, _ = T.forward(model, mx.array(tok[:, None].astype(np.int32)), flow, False, caches=caches)
        nxt = mx.argmax(model.language_model.lm_head(h).astype(mx.float32), axis=-1)
        mx.eval(nxt)
    T.Cache.persist = T.Cache.seq_ids = None
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm"); ap.add_argument("flow")
    ap.add_argument("--run"); ap.add_argument("--sets", default="math,ifeval")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--time-limit", type=float, default=0)
    a = ap.parse_args()
    T.limit_memory()
    t0 = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    flow = T.Flow(a.flow)
    if a.run:
        st = json.load(open(os.path.join(T.TL, "ckpt", a.run + ".json")))
        model, tok = T.setup(targets=st["targets"].split(","), r=st["rank"], alpha=2 * st["rank"],
                             pre_flows=[a.flow, st["flow"]])
        model.load_weights(os.path.join(T.TL, "ckpt", a.run + ".safetensors"), strict=False)
    else:
        model, tok = T.setup(pre_flows=[a.flow])
    items = []
    for s in a.sets.split(","):
        items += [json.loads(l) for l in open(os.path.join(ITEMS, f"dec_{s}.jsonl"))][:a.n]
    out = os.path.join(T.TL, "data/decode"); os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, f"{a.arm}.jsonl")
    have = {json.loads(l)["key"] for l in open(fp)} if os.path.exists(fp) else set()
    todo = sorted([it for it in items if it["key"] not in have], key=lambda it: len(it["prompt_ids"]))
    print(f"{a.arm}: flow {a.flow} run {a.run}: {len(todo)} items to decode ({len(have)} done)", flush=True)
    ntok, t1 = 0, time.time()
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > limit - 240:
            print("time limit: stopping (resumable)", flush=True)
            break
        batch = todo[i:i + a.batch]
        res = decode_batch(model, flow, batch, print)
        with open(fp, "a") as f:
            for it in batch:
                ids, fin = res[it["key"]]
                ntok += len(ids)
                f.write(json.dumps({"key": it["key"], "set": it["set"], "flow": a.flow, "run": a.run,
                                    "completion_ids": ids, "finish": fin, "text": tok.decode(ids)}) + "\n")
        print(f"  {i + len(batch)}/{len(todo)} items, {ntok / (time.time() - t1):.0f} tok/s, peak "
              f"{mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
