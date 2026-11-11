"""L1-15 end to end (Leap, Sat 2026-10-24): the 35B's DFlash loop (z-lab, block 16) on 12 HumanEval+ prompts with the GDN
kernel program in the verify passes (DFL35_GDNK=1) vs z-lab as is, and with L1-14's truncation (t = 0.3) on top.
Reports decode tok/s from the first token, accepted + 1 per block, and how far the outputs agree with z-lab's (the new
GDN projections round differently, so outputs may leave the stock ones at near-ties).
Usage: [SpecDecode venv] small35_gdnk.py  -> ~/Leap/data/small35_gdnk.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/sd35"))
import dflash35_model_mlx as dfm

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
dfm.snapshot_download = lambda *a, **k: os.path.expanduser("~/SpecDecode/data/dflash")
model, tok = dfm.load(MODEL)
draft = dfm.load_draft("z-lab/Qwen3.6-35B-A3B-DFlash")
mx.eval(model.parameters(), draft.parameters())
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

P = []
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"].startswith("tf_code") and len(P) < 12:
        user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
        P.append((r["key"], list(ids if isinstance(ids[0], int) else ids["input_ids"])))

def run(gdnk, trunc):
    os.environ["DFL35_GDNK"] = gdnk; os.environ["DFL35_TRUNC"] = trunc
    n_tok, dt, nb, na, outs = 0, 0.0, 0, 0, []
    for key, ids in P:
        toks, t_first, acc = [], None, []
        for r in dfm.stream_generate(model, draft, tok, ids, block_size=16, max_tokens=400, temperature=0.0):
            if t_first is None:
                t_first = time.perf_counter()
            toks += list(getattr(r, "tokens", []) or [])
            a = getattr(r, "accepted", None)
            if a is not None:
                acc.append(a)
        n_tok += len(toks) - 1; dt += time.perf_counter() - t_first; nb += len(acc); na += sum(acc)
        outs.append(toks)
    return n_tok / dt, na / max(1, nb) + 1, n_tok, outs

say("35B DFlash block 16, 12 HumanEval+ prompts, max 400 tokens (decode tok/s from the first token)")
res = {}
for gdnk, trunc in (("0", "0"), ("1", "0"), ("0", "0.3"), ("1", "0.3"), ("0", "0"), ("1", "0")):
    tps, apb, n, outs = run(gdnk, trunc)
    tag = f"GDN kernels {'on ' if gdnk == '1' else 'off'}, truncation {trunc}"
    res.setdefault((gdnk, trunc), []).append((tps, outs))
    say(f"  {tag}: {tps:6.1f} tok/s, {apb:.2f} tokens per block, {n} tokens")
base = res[("0", "0")][0][1]
for (gdnk, trunc), runs in res.items():
    outs = runs[0][1]
    same = sum(a == b for a, b in zip(outs, base))
    first = [next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) for a, b in zip(outs, base)]
    say(f"  outputs vs z-lab as is (GDN {gdnk}, trunc {trunc}): {same} / 12 identical; first difference at token "
        f"{sorted(first)} (prompt lengths of output {[len(o) for o in outs]})" if (gdnk, trunc) != ("0", "0") else
        f"  z-lab as is, rerun identical to first run: {sum(a == b for a, b in zip(runs[-1][1], outs))} / 12")
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/small35_gdnk.txt"), "w").write("\n".join(lines) + "\n")
