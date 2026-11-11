# PHYSICS: the only page you think from (orchestrator, Tue 3 Nov 2026 ~00:1x)

Kamen: think first, read after. Generate from this page and first principles. Open project files ONLY to kill-test an idea, or to grep ~/TRIED/TRIED.md for repeats.

**The model:** Qwen3.8-Flash-Next, a 125B mixture-of-experts.
- 48 layers: 36 linear-attention (GDN, a fixed 3.1 MB state per layer) + 12 softmax attention.
- Hidden size 2,560. 512 routed experts per layer, each token picks its top 10. One expert = 3 x 2,560 x 640 = 4.9 M weights.
- A shared expert, 4 hyper-connection residual streams, a 248,320-word vocabulary, and a built-in multi-token-prediction head.
- Per token the model touches ~2.4 B expert weights (10 of 512 experts x 48 layers) plus a ~3.4 GB dense part.
- As stored now: experts at ~3 bits (~2.15 MB each, plus a 0.77 MB extra-precision part for each token's top 2), ~53 GB of experts in all.

**The Mac (Kamen's MacBook Pro, M5 Pro):**
- 64 GB unified memory at ~260-290 GB/s.
- Internal SSD: 11-13 GB/s reads; it pays +50-65 us to wake after 0.25-1 ms idle.
- GPU: 20 cores with matrix units, ~14 T MAC/s bf16, int8 ~1.6x that.
- A Neural Engine (it runs a 0.47B drafter in 12.4 ms per batched call).
- 18 CPU cores (~1 TFLOPS), idle most of the time.
- Video encode/decode engines, idle.

**The goal:** 350 tokens/s of generated text (code measured) with the full model: first at ~45-50 GB of RAM, later 24 and 12 GB. A long-term line aims at 1000 tok/s.

**Fixed, never question:**
- the full 125B with all 512 experts (no smaller model, no pruning)
- this MacBook only, no added hardware
- quality: EXACT = bit-identical to the model's own greedy output; FAST = the same scores on math / code / instruction / multilingual tests
Everything else is open.

**Today:** 63.6 tok/s, exact. A small drafter guesses 16 tokens, and the 125B checks all 16 in one pass and keeps the right prefix: ~10.5 kept per check on code, ~2.4 on chat.

**The three walls, as measured:**
1. **Bytes:** ~62 MB must come from the SSD per kept token at today's cache. 350 needs <= ~33.
2. **GPU time per 16-token check:** the dense part ~17-23 ms plus the experts ~20-40 ms. 350 needs ~36 ms in all.
3. **The drafter:** it agrees with the 125B on ~93-96 % of code tokens, much less on chat. Better affordable drafters flatten out at ~12-13 kept per check.

**Bookkeeping that kills ideas:** price bytes and matrix-unit tiles on the UNION of experts per check. An expert read once serves every row that uses it. Never price per row.
