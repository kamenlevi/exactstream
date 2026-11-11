R4 (a), KV/state exactness: summary
- Marconi (2411.19379, CUDA) / SGLang MambaRadixCache: checkpoint the GatedDeltaNet state exactly at radix branch points. SD checkpoints at fixed 4K/16K intervals plus prompt and generation ends, so a branch inside a block re-prefills ≤4K (≤16K) tokens, ~2–8 s per branched turn. Lossless, ~1 day of work → SpecDecode.
- oMLX (MLX) SSD tier for KV/state survives restarts: extending SD's persisted prefixes to whole sessions saves 30–90 s on resumes. Lossless → SpecDecode.
- SplitZip bf16-exponent coding of the SSD tier: 1.3x storage, small.
- Position-independent reuse (CacheBlend/EPIC): impossible on GatedDeltaNet layers (state can't be stitched); LMCache lacks hybrid support. Open research.
- Lossy: kv_bits=8 (+20–25% decode at 100K), Quest-style sparse attention on the 10 full-attn layers (+35–45% at 100K).
