"""Pipelined greedy decode for the streamed 125B (mlx-lm generate_step pattern) instead of MTPLX's synchronous
generate_ar: token t+1's id stays a lazy on-GPU argmax, is async-evaluated, and token t+2's graph is built from
it right away. MTPLX's PLE stage (layer 1) needs the id on the CPU and blocks there; with BIGMOE_ASYNC_EVERY=1
the streamed MoE of layer 0 hands its graph to the GPU first, so layer 0 overlaps the wait and the n-gram gather.
Same model, same kernels: tokens must equal MTPLX's generate_ar (checked by bench.py mode ar2 vs ar)."""
import os
import mlx.core as mx


def prefill_lmp4pd(model, prompt_ids, cache, step=2048):
    """BIGMOE_PREFILL=lmp4pd (PrefillKernels): layer-major prefill with PLE precompute (nsp_lmp4) under MTPLX's
    "prefill" attention phase with the dense-band SDPA lane; exact (same cache hash / sha as chunk-major), 16K in
    23.7 s. FRESH caches only (its PLE plan builds the n-gram history from the prompt). Needs BIGMOE_PF_PREFILL=1 and
    BIGMOE_PF_FULLSCR=1 set before bigmoe is imported. Fills all but the last prompt token, like the stock loop."""
    import sys
    assert os.environ.get("BIGMOE_PF_PREFILL") == "1" and os.environ.get("BIGMOE_PF_FULLSCR") == "1", \
        "lmp4pd needs BIGMOE_PF_PREFILL=1 BIGMOE_PF_FULLSCR=1 at import"
    for p in ("~/PrefillKernels/patches/nosync_prefill", "~/MoonShots/patches/layer_major_prefill"):
        p = os.path.expanduser(p)
        if p not in sys.path:
            sys.path.insert(0, p)
    import nsp_lmp4
    from mtplx.attention_context import attention_phase
    os.environ["MTPLX_QSA_DENSE_BAND_SDPA"] = "1"
    with attention_phase("prefill"):
        nsp_lmp4.prefill_qwen4_lmp4(model, list(prompt_ids), cache, step=step)


def greedy(model, prompt_ids, max_tokens, on_token=None, prefill_chunk=2048):
    if os.environ.get("BIGMOE_RESET_GEN", "1") == "1":   # LC1: no QSA state carried over from an earlier generation
        from .fuse import reset_generation_state
        reset_generation_state(model)
    cache = model.make_cache()
    if os.environ.get("BIGMOE_PREFILL") == "lmp4pd" and len(prompt_ids) > int(os.environ.get("BIGMOE_LMP4_MIN", "2048")):
        # long prompts only: the pool warm-up after layer-major prefill costs ~3 s, chunk-major is as fast below ~2K
        prefill_lmp4pd(model, prompt_ids, cache, prefill_chunk)
    else:
        ids = mx.array([list(prompt_ids)[:-1]])
        for i in range(0, ids.shape[1], prefill_chunk):           # prefill all but the last prompt token (as MTPLX),
            lg = model(ids[:, i:i + prefill_chunk], cache=cache)  # which then runs as an ordinary 1-row decode step
            mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    y = mx.array(int(prompt_ids[-1]))
    first = True
    out = []
    for _ in range(max_tokens):
        nxt = model(y.reshape(1, 1), cache=cache)
        nxt = nxt[0] if isinstance(nxt, tuple) else nxt
        y2 = mx.argmax(nxt[0, -1].astype(mx.float32))
        mx.async_eval(y2)
        if not first:                       # the first y is the prompt's last token, not output
            tok = int(y.item())
            out.append(tok)
            if on_token: on_token(tok)
        first = False
        y = y2
    out.append(int(y.item()))
    if on_token: on_token(out[-1])
    return out[:max_tokens]
