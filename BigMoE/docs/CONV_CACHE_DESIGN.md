# Conversation cache: keep a conversation's model state between requests (design)

Design Tue 10 Nov 2026 20:41 EET. **Plan compressed Tue 23:0x** (Kamen, via the orchestrator: full Claude Code support by
Thu 12 Nov 17:00): B0 Tue night, B1 + B2 Wed, T2 Wed night, RAM tier only; B3 (SSD) and T3 after Thursday (section 8).
B0 is done (section 11). Goal (Kamen, 10 Nov 20:3x, via the orchestrator): the model usable from Claude Code and Codex. Leap builds the server (API, templates, tool calls); this is
the engine part: a new turn prefills only its new tokens. Tags: [K] measured, or read from code / config (source given);
[P] computed from shapes or projected; [G] guess.

## 1. Why

Claude Code resends its system prompt and tool schemas (~15-20K tokens) plus the whole conversation every turn. Today every
request starts from `model.make_cache()` and prefills from token 0 in 1,024-row chunks (`prefill()`,
data/r57/snap/w24/r57_screen_w24.py:1013, "r57" below). At 24 GB the first token took 6.02 s for a 726-token prompt and
14.57 s for 2,801 tokens [K data/w24/w24e_report.txt, pc1024]. At that rate a 20K-token prompt takes about 1.4 min [P].
On the 32 GB exact config, Leap's PF-B (with the SEGALL fix) prefills 30K tokens at 555 tok/s [K Leap, 10 Nov], so 18K
tokens take about 32 s [P]. With the cache
a turn prefills three things only: the tail since the last kept checkpoint (at most 1,023 tokens), the previous reply, and
the new tokens (the tool result or the user message).

## 2. What a conversation's state is

Config: models/bare/config.json. Cache classes: .venv/lib/python3.12/site-packages/mtplx/models/qwen4_exp.py.
- The model has 48 layers: 12 QSA attention layers (every 4th) and 36 Gated-DeltaNet (GDN) layers.
- QSA attention: 2 KV heads × 256, plus an indexer (4 heads × 128, compress ratio 4, budget 2,048).
- GDN: 16 K heads and 48 V heads at 128 × 128; conv kernel 4.
- `make_cache()` gives one `QSACache` per attention layer and one `ArraysCache` (conv, recurrent; 4 entries on PLE layers)
  per GDN layer.

| piece | where it lives | trim to a shorter prefix? | size |
|---|---|---|---|
| attention K/V, 12 layers | `QSACache.kv`, the stock KVCache (bf16, 256-row steps) | yes (offset) | 24,576 B / token [P] |
| QSA indexer: raw keys, pooled blocks, fp32 mirror | `QSACache.raw_keys / pooled / pooled_f32_t`, positional on `kv.offset` | yes (MTPLX made them trimmable) | ~5.4 KB / token [P]; capacity doubles |
| GDN recurrent state, 36 layers | `ArraysCache[1]`, fp32 [1, 48, 128, 128] | **no** | 113.2 MB per position [P] |
| GDN conv state | `ArraysCache[0]`, [1, 3, 10240] | **no** | 2.2 MB [P] (MTPLX: 115.6 MB per checkpoint with the PLE tails [K checkpoint_anchors.py]) |
| positions | implicit in `kv.offset` (text-only RoPE, partial rotary 0.25) | n/a | one int |
| DFlash2 drafter context | rebuilt per prompt from taps `T[P-2047:P]` (r57:1095-1101). Taps: layers 5/15/25/35/45, each the 4-way mean of the 10,240-wide hyper-connection stream (2,560 values), fp16 | n/a | 52.4 MB per checkpoint (2,047 × 12,800 × 2 B) [K B0; drafter hidden 2,560, sliding_window 2048]. (This doc said 13.1 MB until B0: wrong width.) |
| DFlash2 rotating KV (5 sliding layers) | rebuilt from the taps at the first propose | not stored | (41.9 MB if it were) [P] |
| MTP head (the chat switch's drafter) | `make_mtp_cache()`: one QSACache fed (hidden_i, token_i+1) pairs | yes | ~2.5 KB / token [P], plus the widened hidden row at c-1 (the pending pair) |
| U3C prompt footprint (FAST cold arms) | `FG["cap"]`, 48 × 512 bool, accumulated over the prefill (r57:1071) | n/a | 24.6 KB [K] |

**Not stored:**
- Expert residency (pools, sidecars, QOM keep-sets, gate state). It is process state, and a restore leaves it alone.
- SESSION-VOCAB and the copy gate's k-grams. They are rebuilt from the prompt ids, as today.
- A6 pools / warm. They come from the last prefill chunk, which a warm run also runs.
- Last logits. A restore always leaves at least 1 token to prefill.

**Per conversation [K B0, section 11]:** 29,952 B per token logical (KV 24,576 + indexer 5,376), allocated 1-8 % more
(KV in 256-row steps, indexer by doubling), plus the MTP head's cache (2,496 B per token, live and trimmable), plus per kept
checkpoint 168 MB on the DFlash2 path (GDN 115.5 + taps 52.4 MB) or 115.5 MB on the MTP path.

| conversation | checkpoints | size (DFlash2 path) |
|---|---|---|
| 12.3K tokens (the recorded Claude Code turn) | 2 | 0.77 GB |
| 20K tokens | 2 | 1.07 GB |
| 64K tokens | 3 | 2.6 GB |
| 128K tokens | 4 | 4.9 GB |

(Before B0 this table said 0.86 / 2.35 / 4.4 GB: the taps were 4× too small.)

Cross-check: Strata parks the same model at ~19.5 KB per token, and writes 430 MB to disk for a 20K-token system prompt
[K Strata docs/how-it-works.md].

## 3. Where to snapshot: the 1,024 grid, exact by construction

**Rule.** Checkpoints are taken only at multiples of PFCHUNK (1,024), during prefill.
- A warm request restores the highest kept checkpoint c with c ≤ min(matched prefix, N - 1). It then prefills [c, N) in
  1,024-row chunks starting at c.
- Every call it makes is a call the cold prefill of the whole prompt also makes: the same chunk, the same tokens, the same
  KV and GDN state before it.
- So the logits and states are bitwise those of a cold run. No new "exact" label is needed, and the cold path stays the
  gated one, byte for byte.

**Never reuse state that decode wrote.** The reply's KV and GDN states came from verify calls of at most 16 rows, not from
prefill chunks. Templates also re-render the reply (Qwen-style templates drop earlier turns' reasoning). The reply is
re-prefilled as part of the next turn's suffix.

**Which edges to keep.** This is decided before the prefill starts, since N and the hints are known. Only those edges hold
a copy, so the prefill's extra peak is at most 3 × 168 MB. The rules follow MTPLX's anchors:
1. **Prompt end:** the last edge ≤ N - g, where g is the length of the generation prompt. This is where the next turn
   diverges.
2. **Stable prefix:** the last edge ≤ the end of system + tools. Leap's server sends it; the fallback is the longest common
   prefix of the last two requests. Sub-agents and new sessions start here.
3. **Restore point:** the edge this request restored from (a rewind lands there).
4. **Coarse grid:** one edge per 8,192 tokens, thinned by halving when the budget is short (MTPLX GRID_TOKENS).
5. **Live KV:** the live cache's KV is kept whole (it is trimmable).

**Later option, not in the first build.** A checkpoint at the exact prompt end P_n saves up to 1,023 re-prefilled tokens
per turn. But its result equals a cold run split at P_n, not today's prefill (W24e: other chunk edges change tokens), so it
would need its own label. At 24 GB a short suffix costs about one chunk pass whether it is 300 or 1,000 tokens, so the
saving is small there.

## 4. Matching a request to a stored prefix

**Entry key.** Each entry is keyed by:
- the model, and the config's env-file git blob (e.g. card_w24d_env.txt e06f495e);
- the engine snapshot commit and the chat-template hash;
- PFCHUNK and the prefill mode (plain / PF-A / PF-B with SEGALL; PF-D);
- the drafter (DFlash2 / MTP).

Any mismatch is a miss (MTPLX's MODEL / TEMPLATE / POLICY_MISMATCH reasons).

**Lookup.**
- Use a block-hash chain on the grid: h_k = sha256(h_(k-1) ‖ ids[1024(k-1) : 1024k]). Each checkpoint stores its h_k, and
  the new prompt's chain is computed on the host in µs.
- The deepest k with a stored h_k wins. Order: the live entry first, then RAM, then SSD.
- Then ids [0, c) are compared exactly; a hash alone is never trusted.

**Token ids only, never text.** The server's hints are only hints:
- stable_prefix_end;
- a session key, used for eviction priority only (Codex's prompt_cache_key if it sends one; Claude Code sends none, so the
  server may derive one);
- a side-request flag, so title or summary calls never evict the main session.

**Engine call for Leap's server:**

    cache, last_logits, info = conv.prefill(ids, hints={"stable_end": int, "session": str | None, "side": bool})
    # info: matched, c, source (live / ram / ssd / none), prefilled tokens, miss_reason, restore_ms, prefill_ms

**Miss log.** Every miss logs its reason and the first divergent token, with ±20 decoded tokens of context (as MTPLX's
`_debug_prefix_divergence` does), so render drift shows up on Leap's side. Strata lost the whole prefix on every step to
clients that merge a turn's tool calls into one assistant message, and fixed it in the server (patch 13) [K].

## 5. Memory: RAM tier, SSD tier, and the three lines

**Budget** = the line minus the config's measured lifetime peak (phys_footprint) at the session's length. The live
conversation's KV is already inside that peak. The cache adds checkpoints (168 MB each on the DFlash2 path) and parked conversations.

**24 GB: no room.**
- w13b peaks at 23.910 GB with 481-token prompts, 24.014 GB at 520 and 25.112 GB at 2,801 [K docs/WED_TESTS.md,
  long-prompt peak].
- A 20K-token Claude Code prompt adds ~0.6 GB of KV on top [P].
- So the cache cannot make 24 GB hold agent-length prompts. That needs a long-context 24 GB config with fewer resident slots
  (one base-slot row ≈ 0.1 GB [G, from the expert shape], so ~8 rows for a 0.77 GB conversation), priced by a W24-style A/B.
- Cache policy at 24 GB: the live session and 1 prompt-end checkpoint in RAM; anything parked goes to SSD.

**32 GB** (Leap's exact config, where PF-B's 555 tok/s was measured; its peak at agent lengths is measured in B4): reserve
~2.5 GB, enough for a 64K main session plus a 20K sub-agent or side conversation. The rest goes to SSD.

**45 GB:** reserve ~6 GB, about 150K tokens of conversations including checkpoints. (Strata: 3 GB in 8 slots holds ~160K
tokens at 19.5 KB per token.)

**SSD tier, every line** (after Strata's patches 17, 19 and 20):
- Park a conversation after the engine has been idle ~2 s: write its KV rows and checkpoints in 16 MiB blocks with
  F_NOCACHE (macOS's O_DIRECT). Stop if a request arrives.
- If the file is at most 8,192 tokens behind, append only the new rows.
- Restore in two passes: validate the checksums, then copy block by block into the cache arrays, with no whole-file buffer.
  1.07 GB (a 20K conversation) reads in ~0.1 s [P].
- Cap the directory at 16 GB and delete it at exit. Keep entries across R57_HARD restarts (keyed by config blob + commit).
- No spill writes below 110 GB free disk (the health floor).

## 6. Interactions with QOM, residency, PF-A, PF-B, PF-D and the drafters

**QOM and residency.**
- Prefill runs `set_arm("slot")`, the stock path with all 10 experts (r57:1061). PF-D's rebuilt rows are bitwise the SSD
  rows (R57_PFD_CHECK). So prefill bits do not depend on residency, and the hash check holds on every line.
- Decode on FAST (R57_FAST=QOM, the 24 GB card config) does depend on pool residency and timing (EvalSuite: "drafts carry
  history"). A warm turn reads fewer experts during prefill, so its pools differ and its tokens can differ from a cold
  turn's, just as they already differ between any two prompt histories.
- The cache does not snapshot residency. The card label stays a scores label: run the gate's agent-like subset with the
  cache on before claiming the card for it.
- Speed: a warm prefill is 1-3 chunks. The decode-start miss burst (A6, from the last chunk's routing) is the same work as
  cold, and PF-D gets more hits because the conversation's experts are resident.

**Prefill modes.**
- **PF-A** (R57_PFLM=1, layer-major over the same chunks): each chunk's work per layer is unchanged, so a restore plus a
  layer-major suffix should equal a cold layer-major run [P; T1 checks it].
- **PF-B** (R57_PFLM=3): the grouped router and the shared expert span chunks, and a short last chunk changes bits (Leap,
  10 Nov). So PF-B works with the cache only with SEGALL (R57_PFLM_SEGALL=1, per-chunk slices; its port is due after the
  release). Without SEGALL the cache refuses PF-B.
- **PF-D** (R57_PFD): exact by construction. It falls back as today: on LMC, grouped prefill, no sidecar, or layout ≠ 2.
- **PFNOLOG** (no vocab head on cache-only chunks): unchanged, since the warm path's last chunk is the cold path's last
  chunk.

**Drafters.**
- **DFlash2:** its context at decode start is the taps of the last 2,047 positions. The warm path takes [c, N) from its own
  prefill and [N - 2047, c) from the checkpoint's taps. The bits are the same, because both came from the same chunks.
- **MTP head:** trim its cache to c, re-add the pending (hidden_(c-1), token_c) pair from the stored row, then feed it the
  suffix's widened rows as today.

## 7. Exactness check (the gate before Leap's server uses the cache)

**T1, lossless restore** (unit test, ~30 min, one process).
- Prompts of 3K, 20K and 60K tokens. For each kept edge c, compare a cold prefill of [0, N) with a restore at c plus a
  prefill of [c, N).
- Compare the per-chunk logits hashes (R57_LOGHASH), the last row's logits bytes, and the sha256 of every cache-state array
  after the prefill. Pass = all equal.
- Run on the 24 GB card config and the 45 GB exact config; with plain prefill, PF-A, PF-B with SEGALL and PF-D; restoring
  from RAM and from SSD.

**T2, turn replay** (~1-2 h per line, a night job over 5 GB).
- Recorded Claude Code and Codex sessions, at least 2 of each, 10-30 turns with tool calls, rendered by Leap's server.
- Each turn runs warm vs cold (cache bypassed, same process order).
- Pass: prefill hashes equal on every turn. On the 45 GB exact config, decode tokens are also equal (CAP 128).
- On FAST, report token agreement and time to first token (info only).
- Also measure: lifetime peak under the line, and SSD bytes written per turn.

**T3, stress.** Interleaved main, sub-agent and side requests; a rewind; a system-prompt edit; eviction to SSD and back; an
R57_HARD restart in the middle.

A hash mismatch is a bug until shown otherwise.

## 8. Build plan and size

| step | what | size |
|---|---|---|
| B0 | Measure the real state bytes (KV, indexer, GDN, taps, MTP) at 2K / 20K / 60K tokens on both configs | ½ day, short GPU runs |
| B1 | Capture at the chosen edges in `prefill()` (chunk-major, PF-A, PF-B with SEGALL); restore and suffix prefill; T1 | ~250 lines, 1 day |
| B2 | Store: entries, hash chain with exact compare, anchors, byte budget, LRU with FLOP-aware keeping (Marconi), miss reasons; the engine call for Leap; T2 | ~350 lines, 1 day + a night |
| B3 | SSD tier: idle spill, streamed restore, checksums, cap, survival across restarts; T3 | ~200 lines, ½-1 day + a night |
| B4 | Per-line budgets; time to first token and peak at 24 / 32 / 45 GB | measurements, nights |

**Total:** ~800 lines, 3-4 days of build, 2 nights of GPU tests. The build reuses:
- `mtplx.cache_state` snapshot and restore (the engine already uses it for verify rollback, r57:1187);
- the anchor rules of `mtplx.checkpoint_anchors`.

It does not take MTPLX's SessionBank wholesale: that is 5,192 lines wired to MTPLX's own runtime.

**Compressed plan (Tue 23:0x, target Thu 12 Nov 17:00).** The shipped configs use no PF-A / PF-B (chat-24 / chat-32 /
exact-32: plain 1,024-row chunks; chat-45: PF-D), so B1 covers plain + PF-D only; PF-B waits for the SEGALL port.
- Tue night (CPU + light GPU): B0, done (section 11).
- Wed: B1 = `engine/loop/convcache.py`, installed by serve.py; T1 on the real engine (chat-32 and chat-24, both drafter
  paths). Then B2: the RAM store, matching, the hints and the info record for Leap (section 12).
- Wed night: T2 on the GPU, replaying Leap's recorded requests (agent/sessions_real_1110.jsonl, 45 requests) warm vs cold.
- After Thursday: B3 (SSD tier), T3, B4.

## 9. Prior art

**MTPLX**, the library this engine runs on (.venv/.../mtplx) [K code]:
- SessionBank: exact token-prefix entries; restores cloned state and forwards only the suffix. Its doc: "the benchmark gate
  compares the warm result against a cold full prefill".
- checkpoint_anchors.py: boundary records at chunk edges, 115.6 MB each on Flash-Next, and the anchors above.
- engine_session.py: the longest-prefix session lookup and eviction. It also spills idle sessions to SSD.
- Defaults: 24 entries, 24 GiB, 8 GiB per session, 1 h idle TTL, 256-token prefix blocks, 512-token minimum match.

**Strata** (github.com/ExTV/strata-5090-4070, docs/how-it-works.md, read 10 Nov) [K their doc]:
- Conversation parking in RAM: ~19.5 KB per token, 3 GB in 8 slots, 0.6-1.2 s switches between a 30K and a 15K chat.
- A checkpoint at each turn boundary.
- A system-prompt prefix cache on disk (#960): a 20K-token system prompt takes 0.63 s instead of 11.65 s. Restore is 250 ms
  from disk and 145 ms from RAM; the state is 430 MB per 20K tokens.
- SSD spill (patch 17), streamed restore (#1269) and idle-time writes (patch 20).
- The server fix for merged tool calls (patch 13).

**Hybrid-model caches** [K search summaries, not read in full]:
- SGLang's MambaRadixCache / Unified Radix Cache: a recurrent state can be reused only at an exact checkpoint (LMSYS blog,
  11 Aug 2026).
- vLLM RFC #55697 (Sep 2026): application-directed prefix checkpoints for hybrid GDN / Mamba models. Markers declare
  boundaries and the scheduler splits prefill chunks there; up to 7.6× on L40S.
- Marconi (arXiv 2411.19379): admission plus FLOP-aware eviction for hybrid prefix caches, 4.5-34.4× better hit rates.
- LMCache does not support GDN hybrids yet.

**Attention-only prefix caching** (vLLM APC block hashes, SGLang RadixAttention, llama.cpp slot save / restore): the source
of the hash-chain matching above.

## 10. Where we lose, and risks

1. 24 GB does not hold agent-length prompts, with or without the cache.
2. Each turn re-prefills up to 1,023 tokens plus the reply.
3. FAST decode tokens depend on history.
4. A client that rewrites its system prompt every turn falls back to a cold prefill.

## 11. B0 results (Tue 10 Nov 23:08; data/conv/b0.json, scripts/conv_b0.py)

One QSA attention layer and one GDN layer of qwen4_exp (mtplx, the engine's model code) with random weights, each with its
cache from the classes `make_cache()` uses. The inputs are a function of the token ids. MLX 0.32.2, peak 4.6 GB, 42 s,
beside Kamen's chat.

**Mechanics: PASS.** This replays the recorded Claude Code shape (Leap's agent/sessions_real_1110.jsonl: a 12,302-token
turn whose system + tools end at 10,019):

| case | restored at | prefilled | result |
|---|---|---|---|
| turn 2 (12,496 tokens) on the live cache, after decode-style writes (16-row verify, rollback, commit 5; 150 tokens) | 12,288 | 208 | the chunk output and all 9 state arrays are bitwise those of a cold prefill |
| a new conversation (10,159 shared tokens) on the same live cache | 9,216 | 3,084 | equal |
| control: the stable checkpoint restored again into a fresh container | 9,216 | 3,084 | equal (the checkpoint survived its reuse) |
| negative control: the stable edge's GDN state under the prompt-end edge's KV | 12,288 | 208 | caught: the chunk output and the recurrent state differ |

So on the stock layer code, the mechanics hold: `QSACache.trim` on the live cache (KV, raw keys, pooled blocks, fp32 mirror)
plus `restore_cache` of eager GDN clones. The engine's patched paths (ref2, the BigMoE ext, QOM) are for T1 on Wednesday.

**Sizes [K]:**

| piece | measured |
|---|---|
| attention KV | 24,576 B/token logical; allocated in 256-row steps (12,544 rows for 12,302 tokens) |
| QSA indexer (raw keys, pooled blocks, fp32 mirror) | 5,376 B/token logical; allocated by doubling (16,384 raw rows for 12,302 tokens) |
| all 12 QSA layers, allocated (logical) | 0.061 (0.061) GB at 2,048 tokens; 0.396 (0.368) at 12,302; 0.679 (0.613) at 20,480; 1.862 (1.840) at 61,440 |
| GDN state per checkpoint, 36 layers | 115.46 MB (conv [1, 3, 10240] bf16 + recurrent [1, 48, 128, 128] fp32 = 3.21 MB per layer) |
| DFlash2 taps tail per checkpoint | 52.4 MB (2,048 rows × 12,800 fp16; 2,048 because A6's dwin path reads T[P-2048:P-1]) |
| the loop's T during a cold prefill (taps of every position) | 25,600 B/token: 0.31 GB at 12.3K tokens, 1.57 GB at 61K |
| MTP head cache | 2,496 B/token [P: one QSA layer] |

**Costs** [K; measured beside Kamen's chat, so upper bounds]:
- GDN clone 1.24 ms and restore 1.21 ms for all 36 layers (median of 7);
- taps tail clone 14 ms;
- the block-hash chain over 61,440 tokens 0.16 ms.

Capture and restore are noise next to one 1,024-row chunk.

**Finding (a memory lever, separate from the cache).** Today's cold prefill keeps the taps of every prompt position, but
the drafter reads only the last 2,048. At 12.3K tokens that is 0.31 GB of the request's peak (twice that while `taps()`
concatenates), and at 61K it is 1.57 GB. Keeping only the tail would be exact and would help long prompts on every line,
24 GB most. A warm turn already holds only [c, N) plus the stored tail.

## 12. The engine call for Leap's serve.py (proposal, Tue 23:3x)

**Where it hooks.** serve.py runs the loop unchanged (runpy), and requests enter through `next_item()`. The loop's
`prefill(ids, want_w, chunk)` (r57_screen_rel1.py:986, the same function as r57_screen_w24.py:1013) and `MTPHead` are
`__main__` globals, looked up at call time. `convcache.install(G())` swaps two of them:
- `prefill` becomes the cached prefill:
  - It runs the loop's per-chunk body (the same calls, PFNOLOG included).
  - It clones the GDN state, the taps tail and the U3C footprint at the chosen edges.
  - On a hit it first trims the live cache to c and restores the checkpoint, then runs the chunks from c.
  - PF-A / PF-B prompts go to the loop's own function: a miss, not stored.
- `MTPHead` becomes a subclass that adopts the live MTP cache trimmed to c (the chat path), and registers it for the next
  turn.
- T comes back as a view that slices by absolute position over [c - 2048, N).

**Per request, from serve.py:**
- Before the item is returned: `CC.hint(sys_end=..., lane=..., side=..., conv=...)`. These are Leap's render meta, already
  in the recording.
- After the request: `CC.last()` returns `{hit, c, src (live / ram / none), prefilled, miss, restore_ms, prefill_ms,
  entries, bytes}`, for the done line and the render log.
- Off switch: `CONV_CACHE=0` makes `install()` a no-op, which leaves today's cold path.

**What the renderer must keep:**
- The same ids for the same history: no per-turn timestamps or ids.
- The previous prompt stays an exact prefix of the next one. In the recording it is: each Claude Code / Codex turn starts
  with the whole previous prompt.
- Side requests are flagged (`side`), so they never evict the main conversation.

## 13. B1 code: conv/convcache.py (Tue 10 Nov 23:2x), checked on a stand-in loop

The module implements section 12 plus two changes agreed with Leap at 23:2x.

**PF-A from c, in one sweep.** serve.py runs PF-A (R57_PFLM=1) at 32 and 24 GB: a 12K Claude Code prompt takes ~28 s
against ~95 s chunk-major [K Leap]. So the cached prefill runs PF-A from c:
- It replicates `pflm.prefill_layer_major` modes 1 / 2 on the restored cache.
- Each (layer, chunk) step is the chunk-major step. So a layer's GDN / PLE state and its U3C row, taken right after the
  chunk that ends at an edge, are that edge's checkpoint for that layer. They are cloned as the sweep passes: one sweep, no
  extra expert reads.
- PF-B (R57_PFLM=3) still takes the loop's own prefill, uncached, until the SEGALL port.

**The taps lever.** Only the rows the drafter reads (the last SWIN = 2,048) and each edge's tail are kept. Every other
chunk's taps are dropped as they are made, and kept rows are evaluated at once, so no tap holds its layer's output alive.
This is exact. `CONV_CACHE_TAPS=all` keeps them all.

**The stand-in-loop test.** conv/test_mock.py drives `install()`, `hint()`, `prefill()` and `MTPHead` like the loop does,
with two sequential layers (QSA + GDN, real classes, random weights), the loop's TAP wrapper, a U3C hook, and decode-style
writes. Every request is checked twice: against `CONV_CACHE_CHECK`'s cold prefill, and against an external cold run.
Results, 7/7 requests PASS in each run:

| run | data |
|---|---|
| chunk-major, the recorded lengths | data/conv/b1_mock.json |
| chunk-major, 8,000 tokens shorter | data/conv/b1_mock_small_pflm0.json |
| PF-A against Leap's real `pflm.prefill_layer_major`, 8,000 tokens shorter | data/conv/b1_mock_small_pflm1.json |

The requests:
- turn 2 restored on the live cache;
- a new conversation copied from the stable edge;
- the chat path's MTP head taking over its trimmed cache;
- a one-chunk side request, not stored;
- a return to the first conversation.

**Known limit: the drafter.** An MTP-head request restores only from an entry an MTP-head request built. With
`--drafter auto`, a chat-path turn right after a DFlash2 turn is a cold prefill. Fix, if needed: feed the entry's MTP cache
in DFlash2 requests too (one MTP-layer pass per prompt row).

T1 on the real engine is next: serve.py `--mem 32` with the hook, `CONV_CACHE_CHECK=1`, replaying Leap's 45 recorded
requests.

## 14. T1 on the real engine (Wed 11 Nov 01:01-01:13, Leap's run): FAIL, 5 of 12 hits bitwise; cause of 4 found

**Setup.**
- Engine: Leap's cache clone, i.e. pkg_110f prep plus the agent files plus conv/convcache.py from 7eb3401, unchanged.
- Server: serve.py `--mem 32` (chat-32, C 160, R57_PFLM=1, chunk 1,024), drafter dfl, CONV_CACHE_GB 2.5, CONV_CACHE_CHECK=1.
- Load: Leap's replay of the 45 recorded requests.
- Data: data/conv/t1/ (verdict.json, convcache.log, replay.out). The prediction is data/conv/t1_predicted.json.

**What the cache did = the prediction, exactly.**
- All 45 requests returned HTTP 200, and there are 45 cache lines.
- 12 hits (6 Claude Code, 6 Codex), all at the predicted restore points, live or copy as predicted.
- 33 misses: 31 one-chunk requests plus the 2 first turns.
- Prefilled 50,607 of 167,343 prompt tokens (30.2 %). Restore 3.1-15.4 ms. The cache stayed at or below 2.07 GB (budget 2.5).

**Exactness: 5 of 12 hits PASS.** A PASS means bitwise equal last logits, all 122 state arrays, the taps tail, the widened
rows and the U3C footprint. The 7 FAILs come in two kinds:

| kind | hits | new tokens | what differs |
|---|---|---|---|
| A | N 12,404 at c 12,288 (twice); N 8,388 and 8,389 at c 8,192 | 116, 116, 196, 197 | last logits, 119 of 122 states, taps tail |
| B | N 12,300 copied at c 9,216 (twice, PF-A from c); N 12,627 at c 12,288 | 3,084, 3,084, 339 | 1 of 122 states; logits and taps equal |

**Kind A cause (found by Leap, confirmed in the code).** The module chose PF-A by the new part (`N - c > chunk`). The loop
chooses it by the whole prompt (r57_screen_rel1.py:987, `len(ids) > chunk`).
- So a warm tail of at most one chunk ran chunk-major. Its MoE calls used tier.prefill_call's compact scratch, where E =
  the call's unique experts.
- The cold prompt ran the same tail in PF-A mode 1, on the full-E scratch: E = 512, original ids.
- pflm.py:15-17 warns that E changes MLX's gather kernel on short calls. That fits the boundary seen here: tails of 116-197
  new tokens differ, tails of 208-425 match.
- 119 of 122 differing means the change starts at the first MoE call. Layer 0's GDN states run before it, and one PLE slot
  depends only on the ids.
- Fix: choose by the whole prompt. When `PFLM and N > chunk`, run PF-A from c, even for a one-chunk tail.
- The rule in section 7 gains a clause: prefill [c, N) in the same chunks **and the same prefill mode the cold prompt
  takes**. The mode depends on N, not on N - c.

A confound remains. All 4 kind-A hits also had N below their entry's previous prompt plus output, which leaves stale QSA rows
past N in the live buffers; every PASS had N above it. The rerun with the fix uses the same order, so it separates the two.

**Kind B: not yet known.** The FAIL line does not name the leaf. Leap's rerun names it: cache index, class, leaf number, and
warm vs cold shapes.

**Info only, because CHECK runs a cold reference beside the warm path.**
- Peak: 34.83 GB on the cold 12.3K prompt; up to 37.47 GB with the checks. The CHECK-off cached test gives the real number.
- Warm prefill alone (prefill_ms): 4.0-7.8 s for 116-425 new tokens on 8-12K contexts, against a cold 21.4 s (8,390
  tokens) and 29.0 s (12,302). The copy at 9,216 with 3,084 new tokens took 12.2-12.4 s. The warm path ran with the check's
  experts resident, so these times are optimistic.

**Next.**
1. Leap's rerun with the fix and the leaf names (~02:00).
2. The kind B fix.
3. The cached agent tests (32 / 45 GB, CHECK off).

## 15. T1b (Wed 11 Nov 01:3x-01:45, Leap's run): kind A fixed; kind B = layer 35's attention keys after the first copy

**Setup.** Same as T1. The module was Leap's copy, now conv/convcache.py (sha256 00488b23): 7eb3401 plus two changes:
- PF-A is chosen by the whole prompt;
- the FAIL line names each differing leaf.

The stand-in test passes on it: 4 PASS / 0 FAIL [K Leap].

**Results.** Data: data/conv/t1b/.
- 12 hits, as predicted. 7 PASS, 5 FAIL.
- **Kind A is gone.** All 6 Codex hits pass, including the 196- and 197-token tails, and the 116-token Claude Code tails no
  longer differ in logits, taps or the other 119 states. That also rules out the stale-rows confound of section 14.
- **Kind B is one leaf.** All 5 FAILs show `35:QSACache[0]`, the attention KEYS of layer 35 (a QSA layer), with equal
  shapes.
  - Values, raw and pooled at layer 35 are equal, and so are the other 121 leaves, the last logits and the taps tail.
  - So the differing key rows are rows the attention never selected.
- **Pattern.** In the Claude Code conversation, the live hit before any copy passes. The first copy (from 9,216, 3,084 new
  tokens through PF-A) fails, and so does every later hit:
  - live on the main entry from 12,288 (116 new, twice);
  - the second copy;
  - live with 339 new.
  The Codex conversation never copies, and all 6 of its live hits pass, including trims below a decoded reply.
- **Warm prefill alone:**

  | case | new tokens | warm prefill |
  |---|---|---|
  | live tail | 116 | 2.8 s |
  | live tail | 196-425 | 3.8-5.6 s |
  | copy | 3,084 | 12.1 s |
  | cold | 12,302 / 8,390 (whole prompt) | 28.0 s / 21.5 s |

  The check runs first, so the experts are resident: these times are optimistic until the CHECK-off test.

**Reading so far.**
- Keys are stored after RoPE and values are not. So something rewrote layer-35 keys, or wrote them at a wrong position, in
  rows that attention never selected.
- The main entry passed its check right after its prefill, and the copy taken from it failed. So its layer-35 keys changed
  between its check and the copy: during its decode, or in the copy's read.
- Leap finds no decode-time write to a QSA layer's kv.keys outside update_and_fetch in the loop or REF2:
  - the key buckets pad copies;
  - COPY = g2 is a host-side draft gate;
  - SVH, SB1, SBN and S0CARRY do not touch the KV.
- Not yet ruled out: MTPLX's own QSA verify / commit path.
- Layer 35 is one of the drafter's tap layers (5, 15, 25, 35, 45), but layer 15 is also a QSA tap and is clean.

**The drift diagnostic (CHECK-only, for T1c).**
- Every QSA leaf gets a per-row print: two exact integer sums of the row's raw bits, so equal rows print equal whatever
  the strides or capacity.
- The prints are recorded at the end of each prefill (rows [0, N)) and re-checked at four points:
  1. at the next request's start;
  2. after the cold check and the restore;
  3. on the restored rows below c, against the source's prefill;
  4. on the other entries, after the prefill.
- A FAIL line now also gives the differing rows, warm vs cold.
- Unit checks: equal rows at different capacities print equal; one changed value is caught at its row; the stand-in loop
  stays 7/7.
- Reading the result: drift at the request's start means the decode changed stored rows. A restore mismatch means the
  copy's read. Rows that differ only in [c, N) mean the warm prefill itself.

## 16. T1c (Wed 11 Nov 03:57-04:09, Leap's run): the copy's clone rewrote one value; the fix = bitwise copies

**Setup.** As T1b, with the drift diagnostic of section 15 (134e811, sha256 e3434b13). The stand-in test passed first,
4 PASS / 0 FAIL [K Leap].

**Results.** Data: data/conv/t1c/.
- The same 12 hits as predicted. 7 PASS, 5 FAIL: the same 5 as T1b.
- **No drift.** 0 drift lines. No stored entry's QSA rows changed between the end of its prefill and the next request,
  nor across the cold check and the restore. The decode does not touch stored rows.
- **The copy changes one row.** Both copy restores (from 9,216) log `restore (copy at 9216): rows below c differ from the
  source's prefill: 35[0] rows 0..0 (1 of 9216)`. Right after the copy, before any prefill, row 0 of layer 35's keys
  differs from the source's.
- Every FAIL is that row (`QSA rows warm vs cold: 35[0] rows 0..0`). The failing live hits are on the copy-made entries
  and carry the row from their copy.

**Cause.**
- The copy's only arithmetic is its clone, `x + 0` (cache_state._clone_tree's rule, borrowed so that a copy owns its
  buffer).
- An add of zero is not a bit copy for floats. -0.0 + 0.0 = +0.0, and NaN bits are rewritten. Checked in MLX: 0x8000
  becomes 0x0000; 0x7fc1 and 0xffc0 become 0x7fc0.
- -0.0 equals +0.0 as a number, so attention, logits and taps stay equal and only the bits differ, as T1b saw.
- So row 0 of layer 35's keys holds a value the add rewrites. A -0.0 is the likely one: a NaN there would have broken
  every output.
- Where it comes from is open. No key-norm gain is zero (all 13 QSA layers, min |gain| 0.98), so it is not a dead
  dimension. T1e's census line prints the value's bits.

**The fix (conv/convcache.py).**
- `_clone` copies bits: an integer add of zero on the same-size unsigned view (bf16 as uint16, f32 as uint32), viewed
  back. It is exact for every bit pattern, strided slices included.
- The GDN checkpoints went through the same rule twice: cache_state.snapshot_untrimmable_cache (chunk-major) or `_clone`
  (PF-A) when taken, and cache_state.restore_cache on every restore, live hits included.
  - T1's live hits passed because no checkpointed GDN value was a -0.0 or a NaN. That was luck, not design.
  - convcache now takes and restores the checkpoints itself with the bitwise clone (`_snapshot_gdn`, `_restore_gdn`),
    using cache_state's container logic.
- The checkpoints' taps tails use `_clone`, so they are bitwise now too.
- No add-of-zero copy is left on the cache path.
- A CHECK-only census line at each restore lists the values an add of zero would rewrite, in the source rows below c
  and in the GDN checkpoint.

**Tests.** conv/test_bitclone.py runs on the CPU in a few MB, on the real QSACache and ArraysCache classes with planted
-0.0 and NaN payloads and a strided live state. 10/10 PASS:
- the copy restore, the live restore, the GDN checkpoint and its restore, and the MTP copy are all bit-exact;
- writes into a copy leave the source and the checkpoint unchanged;
- the census finds every planted value;
- control: the old rule changes the planted bits, so the test can see the bug.

The stand-in loop (conv/test_mock.py, MOCK_SMALL=1) passes on the fix in both modes: PF-A and chunk-major, 7/7 cases,
4/4 checks each (data/conv/b1_mock_small_pflm{1,0}_bitclone.json). Its census finds nothing, as expected with random
weights.

## 17. T1d (Wed 11 Nov 04:10-04:23, Leap's run): the fallback passes, 10 of 10 hits bitwise

**Setup.** b855883 with CONV_CACHE_BRANCH=0: a request that would copy from an entry runs a cold prefill instead and
starts a new entry. Otherwise as T1. The prediction for this setting is data/conv/t1_predicted_branch0.json (10 hits).

**Results.** Data: data/conv/t1d/.
- 10 hits as predicted. 10/10 PASS: logits, all 122 state arrays, the taps tail, the widened rows and the U3C footprint
  are bitwise cold. 0 drift lines, 0 server errors.
- The 2 would-be copies ran cold (27.5 s and 28.0 s for 12,300 tokens).
- 41.3 % of the prompt tokens were prefilled (69,039 of 167,343). With branches on it was 30.2 %.
- Live hits restore in 58-93 ms and prefill 116-425 new tokens in 2.9-5.7 s (the check runs first, so the experts are
  resident).
- One cold miss took 135 s instead of 28 s: a heat freeze of the server, 04:15:35-04:17:16 [K Leap], not the module.

**Reading.** Every hit of the fallback is bitwise on the real engine, and live hits never copy a QSA array. It ships if
the copy fix of section 16 does not pass its own T1 (T1e).

## 18. T1e (Wed 11 Nov 05:08-05:20, Leap's run): the copy fix passes, 12 of 12 hits bitwise with branches on

**Setup.** As T1c, on 3c068d7 (conv/convcache.py sha256 55631cb8): branches on, CONV_CACHE_CHECK=1, the same 45
requests. serve.py was the pinned copy the cached agent test will run (T1c's plus BIGMOE_IDLE_MS=50, a timing-only
setting) [K Leap]. The stand-in test passed first, PF-A 4 PASS / 0 FAIL and chunk-major 4 / 0.

**Results.** Data: data/conv/t1e/.
- 12 hits as predicted (data/conv/t1_predicted.json), 12/12 PASS: logits, all 122 state arrays, the taps tail, the
  widened rows and the U3C footprint are bitwise cold. 0 drift lines, 0 restore lines, 0 server errors.
- Both copies pass (3,084 new tokens from 9,216 in 12.0 s and 12.3 s), and so do the live hits on the copy-made
  entries (116 new tokens in 2.9-3.0 s, 339 in 4.8 s).
- 30.2 % of the prompt tokens were prefilled (50,607 of 167,343), against 41.3 % with branches off (T1d).
- **The census confirms the cause.** All 12 restores report one value an add of zero rewrites: `35[0] rows 0..0
  (1 rows, 1 values, first 0x8000)`, a -0.0 in row 0 of layer 35's keys. Nothing in any GDN checkpoint or the other leaves.
  - It shows up in both conversations. Position 0 is the same `<|im_start|>` token in each, so this is the model's
    layer-35 key for that token.
  - The Codex hits passed before the fix only because they never copied.

**Reading.** The copy path is bitwise on the real engine, so branches stay on. The cached agent test and the 45 GB T1
run on 3c068d7 [K Leap]. The CONV_CACHE_BRANCH=0 fallback stays in the module as a switch.

## 19. The cached 32 GB agent run (Wed 11 Nov 07:01-07:21, Leap's run 32c_1111): 117 of 118 requests hit, tasks 2.8x faster

**Setup.** EvalSuite's agent test (the 30-case probe, then 12 Claude Code tasks) against serve.py --mem 32 on Leap's
cache clone, as registered in EvalSuite's amendment 3 part 2 [K Leap]:
- serve.py = cachepin d6b44ed0 plus the hook; engine/loop/convcache.py = 3c068d7 (sha256 55631cb8);
- branches on, CONV_CACHE_GB 2.5 (the 32 GB default);
- CONV_CACHE_CHECK off, so there was no cold compare per hit.

**Results.** Data: data/conv/agent32c/. That folder holds convcache.log and summary.json, which recomputes every number
below from Leap's convcache.log and serve.log (Leap 72b6601).
- **117 of 118 requests hit: 106 live and 11 copies.** The one miss is the first probe request (8,738 tokens, 20.4 s
  cold).
- **The 11 copies are the first requests of tasks t02-t12.** The gaps between them equal the task walls. Each new
  conversation starts from the previous task's entry at 9,216 tokens, the shared system-prompt edge (as T1e's copies),
  and prefills 3,286-3,364 new tokens.
- **Prompt size:** median 13.0K tokens, max 30.2K.
- **New tokens prefilled per request:** median 664, p90 3,293, max 11,248. In total that is 9.3 % of all prompt tokens
  (141,318 of 1,519,622).
- **First token:** median 6.3 s, p90 12.7 s, max 36.4 s (the 11,248-token prefill). The cold rerun's median was 31.9 s
  [K Leap]. Restores take 2.7 ms median and 10.6 ms max.
- **Memory:** lifetime peak 37.07 GB (the cold rerun peaked at 36.69 GB [K Leap]). The cache held at most 2 entries and
  2.58 GB, and ended with 1 entry of 2.58 GB.
- **Agent results [K Leap]:** probe 29/30, tasks 11/12 (t09_refactor fails in every run), 0 server errors. The tasks
  took 916 s in all, against 2,595 s cold (2.8x).

**Reading.** The cache holds up through a whole Claude Code session at 32 GB: nearly every request is a hit, copies
included, and memory stays within the line. This run checked no hit against a cold prefill; the module's exactness on
the real engine is T1e's (section 18: same module, 12/12 bitwise, both copies included). Next: T1-45 (chunk-major at
45 GB) on 3c068d7 [K Leap].

## 20. T1-45 (Wed 11 Nov 07:21-07:45, Leap's run): 12 of 12 hits bitwise at 45 GB, on the chunk-major path

**Setup.** As T1e, at --mem 45 [K Leap]:
- the cache clone: serve.py = cachepin d6b44ed0, engine/loop/convcache.py = 3c068d7;
- branches on, CONV_CACHE_GB 2.5, CONV_CACHE_CHECK=1, the same 45 recorded requests.

serve.py sets R57_PFLM only at 32 and 24 GB; at 45 GB the engine refuses PF-A and has its own prefill, PF-D. So at
45 GB every prefill takes convcache's chunk-major path over PF-D, and convcache takes the GDN checkpoints itself
(`_snapshot_gdn`). This is that path's first run on the real engine; T1 through T1e ran PF-A at 32 GB. The stand-in
passed first: chunk-major (the gate) 4 PASS / 0 FAIL, PF-A 4 / 0.

**Results.** Data: data/conv/t1_45/: verdict.json from conv/t1_verdict.py, convcache.log and replay.out (Leap 3671b2e).
- **12 hits as predicted, 12/12 PASS: every hit is bitwise cold at 45 GB.** 0 drift lines, 0 restore lines, 0 server
  errors.
- The prefill share is T1e's: 30.2 % (50,607 of 167,343 tokens).
- **The census lines are identical to T1e's.** All 12 restores report `35[0] rows 0..0 (1 rows, 1 values, first
  0x8000)`: 2 copies at 9,216, 4 live hits at 12,288 and 6 at 8,192. The -0.0 belongs to the model; it does not
  depend on the memory mode or the prefill path.
- **Timing** (CHECK on, so information only):
  - Short prefills match 32 GB: live hits prefill 116-425 new tokens in 3.0-5.8 s (T1e 2.9-6.5 s).
  - Long ones run at PF-D's speed. The misses (8,390 and 12,302 tokens) took 61.4 and 88.7 s, about 137-139 tok/s;
    T1e did them with PF-A in 22.7 and 27.7 s (369-444 tok/s). The copies' 3,084 new tokens took 22.9-23.2 s here,
    12.0-12.3 s in T1e.
  - serve.py's lifetime peak was 50.02 GB, and the cache held at most 2.067 GB.

**Reading.** The module is bitwise on both of the real engine's prefill paths, at 32 and 45 GB. Next is Leap's cached
45 GB agent run. Its misses and copies will be slower than the 32 GB run's, because long prefills at 45 GB run at
PF-D's ~138 tok/s, about a third of PF-A's speed at 32 GB. That gap is known: N45 measured 98-124 tok/s on Saturday.
