"""L1-15 check (Leap, Sat 2026-10-24): where the 35B DFlash loop's block time goes, z-lab as is vs the GDN kernel program
(the verify pass is 44.0 -> 37.0 ms in isolation, test_pass16.txt, yet tok/s barely moved). Phase timers in the loop
(third_party/sd35, DFL35_PROF=1): drafter graph build, target graph build, wait for the drafts, wait for the target pass,
acceptance + rollback build + yield. 12 HumanEval+ prompts, block 16, no truncation; a warm-up pass of all prompts first.
Usage: [SpecDecode venv] small35_prof.py  -> ~/Leap/data/small35_prof.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
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
        P.append(list(ids if isinstance(ids[0], int) else ids["input_ids"]))

def run(gdnk, prof):
    os.environ["DFL35_GDNK"] = gdnk; os.environ["DFL35_TRUNC"] = "0"
    os.environ["DFL35_PROF"] = os.environ.get("PROF_MODE", "1") if prof else "0"
    dfm._PROF.clear()
    n_tok, dt = 0, 0.0
    for ids in P:
        toks, t_first = [], None
        for r in dfm.stream_generate(model, draft, tok, ids, block_size=16, max_tokens=400, temperature=0.0):
            if t_first is None:
                t_first = time.perf_counter()
            toks += list(getattr(r, "tokens", []) or [])
        n_tok += len(toks) - 1; dt += time.perf_counter() - t_first
    return n_tok / dt, dict(dfm._PROF)

say(f"35B DFlash block 16, 12 HumanEval+ prompts: per-block phase times (ms, mean over blocks); PROF_MODE "
    f"{os.environ.get('PROF_MODE', '1')} (2 = drafter finished before the target is built), matmul {os.environ.get('DFL35_QMM', 'c')}")
for gdnk in ("0", "1"):
    run(gdnk, False)                                           # warm-up: kernel variants compiled, caches filled
    tps, pr = run(gdnk, True)
    nb = max(1, pr.pop("blocks", 1))
    tot = sum(v for k, v in pr.items() if not k.startswith(" ")) / nb * 1e3
    say(f"  GDN kernels {'on ' if gdnk == '1' else 'off'}: {tps:6.1f} tok/s; block {tot:5.2f} ms = " +
        ", ".join(f"{k} {v / nb * 1e3:5.2f}" for k, v in pr.items()) + f"  ({nb} blocks)")
open(os.path.expanduser("~/Leap/data/small35_prof.txt"), "a").write("\n".join(lines) + "\n")
