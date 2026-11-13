**What is new**

- **A faster first token at 45 GB (P2).** With `--config chat --mem 45`, a prompt of 256 tokens or fewer now skips the prefill and runs through the decode's check path, as it has at 24 GB since 1.2.0: on 12 IFEval prompts the first token takes 0.47 seconds instead of 2.44 (Strata: 0.83), and the whole request runs at 43.2 instead of 27.6 tokens per second. P2 is lossy: 2 of the 12 answers are the same as before. It passed its own same-scores gate at 45 GB on IFBench and on GPQA Diamond's prompts of 256 tokens or fewer. Longer prompts take the prefill as before, and `exact` and `lossy` are unchanged.

**What else changed**

- `chat` at 45 GB runs a new decode loop, `engine/loop/r57_screen_c45p2.py`: the loop the 45 GB P2 gate ran, which is 1.2.0's 45 GB loop plus P2. Its other modules are 1.2.0's. The gate ran one of them, pflm.py (the prefill for longer prompts), in a newer copy that does the same operations; RELEASE_CHECKS_SENTENCE. Every other engine file is unchanged from 1.2.0.
- `chat` at 45 GB uses QOM on three more benchmark sets (LiveCodeBench v6, IFBench, GPQA Diamond), as the gate ran it. Prompts you type (`--kind chat`, `code` or `math`) are handled as in 1.2.0.
- With the conversation cache on, `serve.py` no longer reports the previous request's cache numbers for a P2 prompt, which bypasses the cache. This changes statistics only, never tokens.
- LIMITS.md adds the 45 GB lifetime peaks on long prompts, the 32 GB agent test's 36.50 GB and the 45 GB clean-clone test's peaks.

The drafter files stay on the v1.0.0 release. The papers are unchanged.
