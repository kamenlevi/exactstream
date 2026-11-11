"""L1-15 (Leap, Sat 2026-10-24): the 35B DFlash loop with the GDN rollback replay submitted on its own (DFL35_ROLLEVAL=1)
vs left lazy inside the next verify pass (z-lab as is), stock vs the GDN kernel program; plus L1-14's truncation (t = 0.3)
on the best combination. 12 HumanEval+ prompts, block 16, decode tok/s from the first token, warm (a full warm-up pass
of every configuration first). Normal loop (no phase syncs).
Usage: [SpecDecode venv] small35_rolleval.py  -> ~/Leap/data/small35_rolleval.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/sd35"))
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
        P.append(list(ids if isinstance(ids[0], int) else ids["input_ids"]))
def run(gdnk, roll, trunc="0"):
    os.environ["DFL35_GDNK"] = gdnk; os.environ["DFL35_ROLLEVAL"] = roll; os.environ["DFL35_TRUNC"] = trunc
    os.environ["DFL35_PROF"] = "0"
    n_tok, dt, nb, outs = 0, 0.0, 0, []
    for ids in P:
        toks, t_first = [], None
        for r in dfm.stream_generate(model, draft, tok, ids, block_size=16, max_tokens=400, temperature=0.0):
            if t_first is None:
                t_first = time.perf_counter()
            toks += list(getattr(r, "tokens", []) or [])
            if getattr(r, "accepted", None) is not None:
                nb += 1
        n_tok += len(toks) - 1; dt += time.perf_counter() - t_first; outs.append(toks)
    return n_tok / dt, n_tok / max(1, nb), outs
CFG = [("0", "0", "0"), ("0", "1", "0"), ("1", "0", "0"), ("1", "1", "0"), ("0", "0", "0.3"), ("1", "1", "0.3")]
for cfg in CFG:
    run(*cfg)                                                  # warm-up
say("35B DFlash block 16, 12 HumanEval+ prompts (warm), decode tok/s")
for g, r, t in CFG:
    tps, tpb, outs = run(g, r, t)
    say(f"  GDN kernels {'on ' if g == '1' else 'off'}, rollback eval {'separate' if r == '1' else 'lazy    '}, truncation {t:3s}: "
        f"{tps:6.1f} tok/s ({tpb:.2f} tokens per block)")
open(os.path.expanduser("~/Leap/data/small35_rolleval.txt"), "w").write("\n".join(lines) + "\n")
