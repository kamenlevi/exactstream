# Layer-major, double-buffered prefill for the streamed 125B (BigMoE)

## Why prefill is slow today
- Chunk-major prefill (MTPLX: 2,048-token chunks; each chunk runs all 48 layers) streams, per (chunk, layer), the union of the
  experts its tokens use. For ≥ ~300 tokens with top-10 of 512 that union is nearly all 512 experts. So every chunk re-reads every
  non-resident expert: at 50 % resident that's ~256 × 2.77 MB × 48 ≈ **34 GB per chunk**.
- Measured (BigMoE FINDINGS): one short prompt takes 6.5–10 s of prefill, i.e. ~34 GB at ~4–5 GB/s effective. Each layer's union is loaded
  synchronously before the layer runs, so SSD and compute alternate.

## Two changes, both exact
1. **Full-layer prefetch (all prompt lengths).** In prefill, layer L+1 will need ~all of its non-resident experts, so no prediction
   is needed. While layer L computes, read *all* of layer L+1's non-resident experts into a second 1.4 GB scratch pool
   (double buffering, all reads in flight). The SSD then runs at ~13 GB/s, overlapped with compute.
   Same expert bytes, same kernels → bit-identical.
2. **Layer-major order (prompts > 1 chunk).** `for layer: for chunk:` instead of `for chunk: for layer:`. Each layer's experts are
   loaded **once per prompt** instead of once per chunk. Each (layer, chunk) step gets exactly the same inputs as in chunk-major order:
   its own hidden state, and the layer's KV/GDN cache after chunks 0..c-1. Same kernels, same chunk boundaries → **bit-identical KV, state
   and logits** to the current reference. No option-2 revalidation needed.
3. *(Optional, reference-v2 batch)* inside layer-major only one layer's activations are live, so chunks can grow to 8–16K tokens. The 125B MoE
   layer then costs 5.3 / 4.8 µs per token instead of 8.4 at 2,048 (`data/layered_prefill.txt`: 11.7 → 18.5 → 20.5 TFLOPS). Different
   chunk boundaries change bits, so validate once under option 2.

## Estimates (125B, 50 % resident, SSD 13 GB/s; compute ~1 ms/token at 2,048-token chunks: MoE 0.40 ms measured + dense ~0.45 + attention ~0.15, estimated)
| prompt | today (chunk-major, ~6.7 s SSD per chunk + compute) | full-layer prefetch only | + layer-major | + 8K chunks (v2) |
|---|---|---|---|---|
| 327 tokens | ~7–10 s | **~2.8 s** (SSD-bound 2.6 s) | same | same |
| 4K | ~17 s | ~6 s | **~4.5 s** | ~4 s |
| 16K | ~70 s | ~25 s | **~16 s** | ~13 s |
| 32K | ~140 s | ~50 s | **~32 s** | ~26 s |
Today's long-prompt numbers are extrapolated from the one-chunk measurement; compute is the uncertain part. Measure it first:
resident-only prefill of 2 × 2,048 tokens with all experts forced resident on a layer subset, or time compute separately in the current run.

## Driver (replaces `Qwen4ExpTextModel._forward`'s loop for prefill; MTPLX 2.12.2 qwen4_exp.py ~L6455)
```python
def prefill_layer_major(tm, ids, cache, chunk=2048):          # tm = Qwen4ExpTextModel
    chunks = [ids[:, i:i + chunk] for i in range(0, ids.shape[1], chunk)]
    hs = [mx.tile(tm.embed_tokens(c), (1, 1, tm.args.hc_count)) for c in chunks]
    masks = [None] * len(chunks)          # B=1, no padding: create_ssm_mask -> None (assert this per chunk)
    for L, (layer, c) in enumerate(zip(tm.layers, cache)):
        streamer.ensure_layer_resident(L)                 # waits for the double-buffered full-layer read of L
        streamer.prefetch_full_layer(L + 1)               # issue every non-resident expert of L+1 now (all in flight)
        for k, x in enumerate(chunks):
            if L == tm._ple_stage_idx:                    # the n-gram stage belongs to the PLE layer's chunk step
                layer.ple.ple_embedding.stage(x, c, layer.ple.NGRAM_IDX)
            hs[k] = layer(hs[k], input_ids=x, ssm_mask=masks[k], cache=c)
            mx.eval(hs[k], *state_arrays(c))              # bound the graph; keeps peak memory at one layer
        streamer.release_layer(L)                         # scratch pool of L goes back (MLX clear_cache, as BigMoE does)
    tm._last_widened = hs[-1]
    return tm.hyper_connection_mixer(hs[-1][:, -1:])      # logits for the last token; MTP history from hs if needed
```
Memory: hyper states for all chunks (T × 10,240 bf16 = 0.33 GB at 16K, 0.66 GB at 32K) + 2 × 1.4 GB expert scratch.
Check before relying on it: the n-gram stage's per-chunk history (`cache[state_idx]`) must advance chunk by chunk at the PLE layer.
It does in this order, because the PLE layer sees chunks 0, 1, 2… in sequence.

## Exactness test (one command)
Prefill the same 6K-token prompt both ways (chunk-major reference vs `prefill_layer_major`, same chunk size). Compare bitwise:
every layer's KV/GDN cache, the final hidden state, and the first 64 decoded tokens' sha. Expected: identical.
