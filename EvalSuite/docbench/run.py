#!/usr/bin/env python3
"""docbench runner (mlx_lm 0.31.3, ~/BigMoE/.venv324): one config over set/docs.jsonl with greedy batched decoding
(mlx_lm BatchGenerator, non-thinking chat template, identical prompts), outputs to
results/docbench/<config>/outputs.jsonl + meta.json. --speed: batch-1 prefill and decode tok/s and peak memory on a
fixed prompt (the longest document's prompt), for a quiet lock.
Configs that quantize in memory follow mlx_lm.convert exactly (load lazy -> quantize_model with the same predicate),
evaluated one decoder layer at a time so the BF16 weights are never all resident beside the quantized ones."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, statistics, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema  # noqa: E402

LMS = os.path.expanduser("~/.lmstudio/models/lmstudio-community")
Q95 = "mlx-community/Qwen3.5-9B-MLX-bf16"
CONFIGS = {
    "smoke-4b-q4": dict(path="mlx-community/Qwen3.5-4B-MLX-4bit", note="pipeline smoke test only"),
    "9b-bf16": dict(path=Q95, note="reference"),
    "9b-q8": dict(path=Q95, bits=8, note="in memory from BF16, affine 8-bit g64"),
    "9b-q4": dict(path=Q95, bits=4, note="in memory from BF16, affine 4-bit g64 (the common MLX / LM Studio 4-bit)"),
    "9b-q4mixed": dict(path=Q95, bits=4, recipe="mixed_4_6",
                       note="in memory from BF16, mlx_lm mixed_4_6 (6-bit v_proj / down_proj in some layers, lm_head)"),
    "27b-q4": dict(path="mlx-community/Qwen3.5-27B-4bit", note="as published (affine 4-bit g64)"),
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
    "38-27b-q4": dict(path=f"{LMS}/Qwen3.8-27B-MLX-6bit", requant=4,
                      note="in memory from the 6-bit: dequantize, then affine 4-bit g64 (one extra rounding step "
                           "compared with a 4-bit made from BF16)"),
}
MAX_TOKENS = 640
OUT = os.path.join(os.path.dirname(HERE), "results", "docbench")


def footprint_gb(peak=False):
    class RU(ctypes.Structure):                   # rusage_info_v4: uuid, then 64-bit counters
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]
    ru = RU()
    if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) != 0:
        return float("nan")
    return (ru.u28 if peak else ru.u7) / 1e9     # lifetime max / current phys_footprint


def load_config(name):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.utils import quantize_model
    from mlx_lm.convert import mixed_quant_predicate_builder
    c = CONFIGS[name]
    t = time.time()
    model, tok, config = load(c["path"], return_config=True, lazy=True)
    if c.get("bits"):
        pred = mixed_quant_predicate_builder(c["recipe"], model, 64) if c.get("recipe") else None
        model, config = quantize_model(model, config, 64, c["bits"], mode="affine", quant_predicate=pred)
    if c.get("requant"):
        n = 0
        for _, m in model.named_modules():
            if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)) and m.bits != c["requant"]:
                w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
                m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=c["requant"])
                m.group_size, m.bits = 64, c["requant"]
                n += 1
        del w                                     # (the last dequantized matrix, e.g. lm_head: memory only)
        print(f"requantized {n} modules to {c['requant']}-bit g64", flush=True)
    for layer in model.layers:                    # one layer at a time: bounded peak while converting
        mx.eval(layer.parameters())
        mx.clear_cache()
    mx.eval(model.parameters())
    mx.clear_cache()
    weights = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    bits = {}
    for _, m in model.named_modules():
        if hasattr(m, "bits") and isinstance(getattr(m, "bits"), int):
            bits[m.bits] = bits.get(m.bits, 0) + 1
    info = dict(config=name, path=c["path"], note=c["note"], load_s=round(time.time() - t, 1),
                weights_gb=round(weights / 1e9, 3), active_gb=round(mx.get_active_memory() / 1e9, 3),
                footprint_gb=round(footprint_gb(), 3), quantized_modules_by_bits=bits)
    print(json.dumps(info), flush=True)
    return model, tok, info


def prompt_ids(tok, doc):
    ids = tok.apply_chat_template([{"role": "user", "content": schema.prompt(doc["type"], doc["text"])}],
                                  add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


def quality(name, docs, limit=0, batch=16, pbatch=8, pstep=2048):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    model, tok, info = load_config(name)
    if limit:
        per = {}
        docs = [d for d in docs if per.setdefault(d["type"], []).append(d) or len(per[d["type"]]) <= limit]
    prompts = [prompt_ids(tok, d) for d in docs]
    rendered = hashlib.sha256("\x00".join(tok.decode(p) for p in prompts).encode()).hexdigest()
    order = sorted(range(len(docs)), key=lambda i: len(prompts[i]))      # length-sorted batches: little padding
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    mx.reset_peak_memory()
    gen = BatchGenerator(model, stop_tokens=[[t] for t in tok.eos_token_ids], max_tokens=MAX_TOKENS,
                         completion_batch_size=batch, prefill_batch_size=min(pbatch, batch),
                         prefill_step_size=pstep)
    uids = gen.insert([prompts[i] for i in order], [MAX_TOKENS] * len(order))
    toks = {u: [] for u in uids}
    fin = {}
    t0 = time.time()
    with gen.stats() as st:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    toks[r.uid].append(r.token)
                if r.finish_reason is not None:
                    fin[r.uid] = r.finish_reason
    gen.close()
    wall = time.time() - t0
    rows = []
    for u, i in zip(uids, order):
        d = docs[i]
        rows.append(dict(id=d["id"], type=d["type"], n_prompt=len(prompts[i]), n_gen=len(toks[u]),
                         finish=fin.get(u), text=tok.decode(toks[u])))
    rows.sort(key=lambda r: r["id"])
    with open(os.path.join(out_dir, "outputs.jsonl" if not limit else "outputs_limit.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(info, docs=len(docs), docs_sha256=hashlib.sha256(open(os.path.join(HERE, "set", "docs.jsonl"), "rb")
                                                                  .read()).hexdigest(),
                schema_sha256=hashlib.sha256(open(os.path.join(HERE, "schema.py"), "rb").read()).hexdigest(),
                prompts_sha256=rendered, max_tokens=MAX_TOKENS, batch=batch, prefill_batch=pbatch,
                prefill_step=pstep, wall_s=round(wall, 1),
                prompt_tokens=st.prompt_tokens, prompt_tps=round(st.prompt_tps, 1),
                gen_tokens=st.generation_tokens, gen_tps=round(st.generation_tps, 1),
                peak_gb=round(mx.get_peak_memory() / 1e9, 3), footprint_peak_gb=round(footprint_gb(True), 3),
                length_stops=sum(1 for r in rows if r["finish"] == "length"), finished=time.time())
    json.dump(meta, open(os.path.join(out_dir, "meta.json" if not limit else "meta_limit.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("config", "docs", "wall_s", "prompt_tps", "gen_tps", "peak_gb",
                                           "length_stops")}), flush=True)


def speed(name, docs, n_gen=256, reps=3):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    model, tok, info = load_config(name)
    doc = max(docs, key=lambda d: len(d["text"]))
    ids = mx.array(prompt_ids(tok, doc))
    greedy = lambda x: mx.argmax(x, axis=-1)
    for _ in zip(range(8), generate_step(mx.array(prompt_ids(tok, docs[0])), model, max_tokens=8, sampler=greedy)):
        pass                                       # warm-up (kernels, wired memory)
    res = []
    for _ in range(reps):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        g = generate_step(ids, model, max_tokens=n_gen, sampler=greedy)
        next(g)
        t1 = time.perf_counter()
        n = 1 + sum(1 for _ in g)
        t2 = time.perf_counter()
        res.append(dict(prefill_tps=ids.size / (t1 - t0), decode_tps=(n - 1) / (t2 - t1),
                        peak_gb=mx.get_peak_memory() / 1e9))
    out = dict(info, prompt_tokens=ids.size, gen_tokens=n_gen, reps=reps,
               prefill_tps=round(statistics.median(r["prefill_tps"] for r in res), 1),
               decode_tps=round(statistics.median(r["decode_tps"] for r in res), 2),
               peak_gb=round(max(r["peak_gb"] for r in res), 3), footprint_peak_gb=round(footprint_gb(True), 3),
               all=res, finished=time.time())
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, name, "speed.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("config", "prompt_tokens", "prefill_tps", "decode_tps", "peak_gb",
                                          "weights_gb")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(CONFIGS))
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N documents of each type (a check)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pbatch", type=int, default=8)
    ap.add_argument("--pstep", type=int, default=2048)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(os.path.join(HERE, "set", "docs.jsonl"))]
    if a.speed:
        speed(a.config, docs)
    else:
        quality(a.config, docs, a.limit, a.batch, a.pbatch, a.pstep)


if __name__ == "__main__":
    main()
