# PF-A vs the default prefill on a short last chunk (Wed 11 Nov 01:46-01:50)

**Question (from T1, data/agent_cc_1111.md):** the conversation cache's signature A showed that a short tail run
chunk-major and the same tail run inside PF-A's sweep give different bits. If so, PF-A (R57_PFLM=1, layer-major) and the
default chunk-major prefill should differ whenever a prompt's last 1,024-token chunk is short. The 1.1.0 text says PF-A
gives "the same tokens as the default prefill" (INSTALL.md:45) and run.py's --env help calls it "exact". The earlier
bitwise records never had a short last chunk (30K: 292 tokens; 6,980: 836).

**Setup:** the v110 clean clone (= the 1.1.0 candidate), `run.py --config exact --mem 32 --max-tokens 64 --env
R57_LOGHASH=1`, R57_PFLM=1 vs R57_PFLM=0, one process per run. Prompts cut from data/prefill_1109/prompt_30k.txt to
P = 5 x 1,024 + a tail (12 template tokens included): pfa_tail_1111/prompt_t116.txt, prompt_t300.txt. Runner:
data/agent_cc_1111/t1b_pfa.sh (step 2).

| prompt | last chunk | tokens (64) | verify-logits hashes equal | first token default / PF-A |
|---|---|---|---|---|
| P 5,235 | 115 tokens | **differ from the 36th** | **0 of 20** (first difference at check 1) | 42.03 / 13.42 s |
| P 5,420 | 300 tokens | equal | 20 of 20 | 41.98 / 13.56 s |

Peaks 32.95 / 33.25 GB (115) and 32.96 / 33.35 GB (300).

**Verdict: CONFIRMED. PF-A is bitwise-equal to the default prefill only when the prompt's last 1,024-token chunk is
long enough; with a 115-token last chunk the bits differ from the first check and the answer diverges at token 36.**
With T1 / T1b the boundary sits between 197 (differs) and 208 (equal) tokens at 32 GB. Cause: in PF-A mode 1 every MoE
call runs on a full-E scratch (512 weight rows) with the original ids, the default on a compact scratch of the call's
unique experts; pflm.py:15-17 already warns that this changes MLX's gather kernel choice on short calls. Both paths are
deterministic; they are two different, valid roundings. Not measured: the 24 GB loop (w24) and 45 GB (the tier call is
the same mechanism there, so likely the same).

**Engine fix to make PF-A exact on every prompt (proposal for the next release, not built):** take mode 2's path
(a compact sorted copy of the call's experts out of the full-E scratch, "bitwise by construction") for calls under ~256
rows, i.e. a short last chunk; no extra SSD reads, ~1.5 GB transient and a GPU copy per layer for that chunk only.
Test: tails 115 / 197 / 208 / 300 at 32 and 24 GB with R57_LOGHASH.

## Drafted wording (held for the orchestrator's review; release/src is NOT edited)

- **run.py --env help** (src/run.py:162-164). Now: "... the prefill paths: R57_PFLM=1 PF-A and R57_PFD=1 PF-D, both
  exact; R57_PFLM=3 PF-B, which can differ when the prompt's last chunk is short". Draft: "... the prefill paths:
  R57_PFD=1 PF-D, exact; R57_PFLM=1 PF-A, exact except when the prompt's last 1,024-token chunk is under about 200
  tokens; R57_PFLM=3 PF-B, which can differ when the prompt's last chunk is short".
- **INSTALL.md:44-45.** Now: "At `--mem 32` and `--mem 24` it prefills long prompts layer-major (the same tokens as the
  default prefill, a 2.4-3.8x faster first token, 0.3-3.2 GB more memory)." Draft: "At `--mem 32` and `--mem 24` it
  prefills long prompts layer-major (a 2.4-3.8x faster first token, 0.3-3.2 GB more memory). Its tokens are the default
  prefill's, except when a prompt runs less than about 200 tokens past a multiple of 1,024: there the arithmetic
  rounds differently and an answer can differ."
- **chat.py docstring** (src/chat.py:17). Now: "(PF-A: the same tokens, a faster first token, a little more memory)".
  Draft: "(PF-A: a faster first token, a little more memory; the same tokens unless the prompt runs less than ~200 tokens
  past a multiple of 1,024)".
- **chat.py comment** (src/chat.py:263-264). Now: "the same tokens as the default prefill (bitwise at 30K tokens;
  identical answers at 32 and 24 GB)". Draft: "bitwise with the default prefill unless the last 1,024-token chunk is
  under ~200 tokens (Leap 11-11: 115 differs, 300 equal)".

## Paper lines that call the layer-major prefill exact (for the Editor; ~/Papers, not edited)

- MAIN.md:42 "prefill (layer-major, exact)".
- MAIN.md:236 table row "layer-major prefill | 3 agent prompts, 1.6K-7.9K tokens | decode tokens equal; logits hash equal
  on 8 / 8, 28 / 28, 22 / 22" (a measurement at 45 GB; check those three prompts' last-chunk sizes).
- MAIN.md:522 "the layer-major exact prefill of section 6.2"; MAIN.md:532 column head "layer-major prefill (exact)".
- MAIN.md:546-548 (the v1.0.1 erratum text): "the layer-major prefill's hashes equal the default prefill's on 26 of 26"
  (true at 30K, whose last chunk is 292 tokens; worth saying so).
- MAIN.md:588 "layer-major 1.9-3.4x at 1.6K-7.9K tokens, exact (built, not shipped)".
- MAIN.md:943 and PATH_DRAFT.md:24 "exact layer-major prefill | 45 GB, 1.6K-7.9K-token prompts".
- SMALL_RAM.md:710 "The exact layer-major prefill ([MAIN] section 6.2) cuts that 2.4-2.7x".
- RELEASE.md:73-74 (v1.0.1 release notes): PF-A listed beside PF-B with "(exact E; ...)"; PF-B's short-last-chunk caveat
  is there, PF-A's is not.
- KERNELS.md:249 and R35 (PF-B's fix measured against PF-A on a 292-token tail): unaffected (PF-A equals the default at
  300), but the reference's own caveat could be noted.
- Internal notes (ERRATUM_v1.0.1_sources.md:18, EDITOR_NOTES.md:386) state measurements and stay true.
