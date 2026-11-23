"""Greedy generation with the frozen target (mlx_lm BatchGenerator, argmax sampler) and chat formatting."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb import core as C


def chat_ids(tok, user):
    """the user turn in Qwen3.5's chat template, thinking off (the template's default), generation prompt added"""
    msgs = [{"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return list(tok.encode(s, add_special_tokens=False))


def greedy_batch(prompts, max_tokens, batch=16, on_done=None):
    """prompts: lists of ids -> answers (lists of ids, the stop token INCLUDED when the answer stopped)"""
    from mlx_lm.generate import BatchGenerator
    m, tok = C.load_target()
    eos = set(tok.eos_token_ids)
    outs = [None] * len(prompts)
    for b0 in range(0, len(prompts), batch):
        C.wait_pause()
        gen = BatchGenerator(m, stop_tokens=[[t] for t in eos], completion_batch_size=batch, prefill_batch_size=min(8, batch))
        part = prompts[b0:b0 + batch]
        uids = gen.insert(part, [max_tokens] * len(part))
        res = {u: [] for u in uids}
        while rs := gen.next_generated():
            for r in rs:
                res[r.uid].append(int(r.token))
        gen.close()
        for i, u in enumerate(uids):
            outs[b0 + i] = res[u]
            if on_done:
                on_done(b0 + i, res[u])
    return outs
