"""Smaller-model check for 350 (SMALL35_PREREG.md; Leap, Sat 2026-10-24): Qwen3.6-35B-A3B 4-bit fully in RAM, greedy.
Plain (mlx_lm's stream_generate) vs z-lab's DFlash loop at block 16 / 8 (SpecDecode's vendored copy, third_party/sd35),
on HumanEval+ code prompts (EvalSuite's tf_code items, re-rendered with the 35B's own chat template) and Kamen's agentic
samples (SpecDecode's samples.jsonl, rendered by its render.py). Plain runs first for every prompt, because the DFlash
loop patches the model to capture hidden states. Decode tok/s counts from the first generated token; DFlash also
reports accepted drafts per block and the first position where its output leaves plain decode (near-ties only).
Usage: [SpecDecode venv] small35_bench.py [N_CODE] [AGENT_IDS]  -> ~/Leap/data/small35.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/sd35"))
import dflash35_model_mlx as dfm
from render import render
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
N_CODE = int(sys.argv[1]) if len(sys.argv) > 1 else 12
AGENT = (sys.argv[2] if len(sys.argv) > 2 else "bash17,edit00").split(",")
MAXT = {"code": 400, "agent": 256}
dfm.snapshot_download = lambda *a, **k: os.path.expanduser("~/SpecDecode/data/dflash")   # the local drafter
t0 = time.time()
model, tok = dfm.load(MODEL)
draft = dfm.load_draft("z-lab/Qwen3.6-35B-A3B-DFlash")
mx.eval(model.parameters(), draft.parameters())
print(f"[s35] loaded in {time.time() - t0:.0f}s, active {mx.get_active_memory() / 1e9:.1f} GB", flush=True)
EOS = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

items, same = [], 0
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if not r["key"].startswith("tf_code") or len([x for x in items if x[0] == "code"]) >= N_CODE:
        continue
    text = tok.decode(r["prompt_ids"])
    user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    same += ids == list(r["prompt_ids"])
    items.append(("code", r["key"], ids))
samples = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.expanduser("~/SpecDecode/data/samples.jsonl"))}
for sid in AGENT:
    ids, _ = render(samples[sid], tok)
    items.append(("agent", sid, list(ids)))
print(f"[s35] {len(items)} prompts; code prompts identical to the 125B's rendering: {same}/{N_CODE}", flush=True)

def plain(ids, n):
    out, t_first = [], None
    for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=n, sampler=make_sampler(0.0)):
        if t_first is None:
            t_first = time.perf_counter()
        out.append(int(r.token))
        if r.token in EOS:
            break
    return out, time.perf_counter() - t_first

def dflash(ids, n, b):
    out, t_first, acc = [], None, []
    for r in dfm.stream_generate(model, draft, tok, ids, block_size=b, max_tokens=n, temperature=0.0):
        if t_first is None:
            t_first = time.perf_counter()
        out += [int(t) for t in (getattr(r, "tokens", []) or [])]
        a = getattr(r, "accepted", None)
        if a is not None:
            acc.append(a)
    return out, time.perf_counter() - t_first, acc

plain(items[0][2], 16)                                 # warm-up (kernels, caches)
res, ref = {}, {}
for cls, key, ids in items:
    out, dt = plain(ids, MAXT[cls]); ref[key] = out
    res.setdefault(("plain", cls), []).append((len(out) - 1, dt, 0, 0))
    print(f"[s35] plain {cls} {key}: {len(out)} tok, {(len(out) - 1) / dt:.1f} tok/s", flush=True)
dflash(items[0][2], 32, 16)                            # warm-up of the DFlash path
for b in (16, 8):
    for cls, key, ids in items:
        out, dt, acc = dflash(ids, MAXT[cls], b)
        div = next((i for i, (x, y) in enumerate(zip(out, ref[key])) if x != y), None)
        res.setdefault((f"dflash{b}", cls), []).append((len(out) - 1, dt, len(acc), sum(acc)))
        print(f"[s35] dflash{b} {cls} {key}: {len(out)} tok, {(len(out) - 1) / dt:.1f} tok/s, "
              f"{sum(acc) / max(1, len(acc)):.2f} accepted/block, first divergence from plain: {div}", flush=True)
lines = [f"Qwen3.6-35B-A3B 4-bit (Youssofal body) in RAM, greedy; peak {mx.get_peak_memory() / 1e9:.1f} GB"]
for (mode, cls), rows in res.items():
    n = sum(r[0] for r in rows); dt = sum(r[1] for r in rows); nb = sum(r[2] for r in rows); na = sum(r[3] for r in rows)
    lines.append(f"{mode:8s} {cls:5s}: {n / dt:6.1f} tok/s over {len(rows)} prompts, {n} tokens"
                 + (f", {na / max(1, nb):.2f} accepted + 1 per block" if nb else ""))
for l in lines:
    print(l, flush=True)
open(os.path.expanduser("~/Leap/data/small35.txt"), "w").write("\n".join(lines) + "\n")
