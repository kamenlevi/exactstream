"""L1-15 check (Leap, Sat 2026-10-24): is the -4.5 % tokens per block with the GDN kernel program chance (outputs left
stock's path on 6 / 12 prompts) or numerics? Per prompt: output tokens, blocks, accepted per block, stock vs DFL35_GDNK=1
(block 16, no truncation). On prompts whose outputs are identical in both modes the text is the same, so any block-count
difference comes from the drafts (the drafter reads the target's hidden states, which the new rounding moves).
Usage: [SpecDecode venv] small35_acc.py  -> ~/Leap/data/small35_acc.txt"""
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_third_party/sd35"))
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
def run(gdnk):
    os.environ["DFL35_GDNK"] = gdnk; os.environ["DFL35_TRUNC"] = "0"
    res = []
    for key, ids in P:
        toks, acc = [], []
        for r in dfm.stream_generate(model, draft, tok, ids, block_size=16, max_tokens=400, temperature=0.0):
            toks += list(getattr(r, "tokens", []) or [])
            a = getattr(r, "accepted", None)
            if a is not None:
                acc.append(a)
        res.append((toks, acc))
    return res
A, B = run("0"), run("1")
say("per prompt: stock vs GDN kernel program (block 16, no truncation): tokens, blocks, tokens per block")
same_blocks = [0, 0]; diff_blocks = [0, 0]; same_tok = [0, 0]; diff_tok = [0, 0]
for (key, _), (ta, aa), (tb, ab) in zip(P, A, B):
    same = ta == tb
    first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    say(f"  {key:24s} {'SAME text' if same else f'differs at {first:3d}'}: stock {len(ta):3d} tok / {len(aa):3d} blocks "
        f"({len(ta) / max(1, len(aa)):5.2f}), kernels {len(tb):3d} / {len(ab):3d} ({len(tb) / max(1, len(ab)):5.2f})"
        + ("" if not same else f"; accepted lists equal: {aa == ab}"))
    if same:
        same_blocks[0] += len(aa); same_blocks[1] += len(ab); same_tok[0] += len(ta); same_tok[1] += len(tb)
    else:
        diff_blocks[0] += len(aa); diff_blocks[1] += len(ab); diff_tok[0] += len(ta); diff_tok[1] += len(tb)
say(f"identical-output prompts: tokens per block stock {same_tok[0] / max(1, same_blocks[0]):.3f} vs kernels "
    f"{same_tok[1] / max(1, same_blocks[1]):.3f} ({same_blocks[0]} vs {same_blocks[1]} blocks for {same_tok[0]} tokens)")
say(f"diverging prompts: stock {diff_tok[0] / max(1, diff_blocks[0]):.3f} ({diff_tok[0]} tok) vs kernels "
    f"{diff_tok[1] / max(1, diff_blocks[1]):.3f} ({diff_tok[1]} tok)")
open(os.path.expanduser("~/Leap/data/small35_acc.txt"), "w").write("\n".join(lines) + "\n")
