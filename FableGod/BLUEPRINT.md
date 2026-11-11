# BLUEPRINT — a mixture-of-experts model designed for one user, one token at a time, on a laptop with unified memory and an SSD
FableGod, Sun 1 Nov 2026, written 22:40-22:50 EET from a day's measurements on a 125B MoE (512 experts top-10, 48 layers, GDN + attention hybrid, 4-bit) on an M5 Pro MacBook with 64 GB. Marks: [K] measured here, [P] arithmetic on measurements, [G] judgement.

## Abstract
Today's open MoE models are shaped by two things that do not exist on a laptop: expert parallelism across many GPUs, and batches of thousands of tokens. Served to one user on unified memory at 300 GB/s with the rest of the weights on a 13 GB/s SSD, the same models spend most of their bytes moving expert knowledge that does not change the next token, hide their decisions in a deep dense trunk that must run sequentially, route in a way the SSD cannot predict, keep mixers in bf16 that nothing downstream needs, and carry a recurrent state that makes speculation expensive to roll back. We lay out what a model built for this machine would look like, why each choice follows from first principles and from our measurements, what it would run at on this MacBook against today's 125B at equal quality, and which of its properties an inference engine can imitate on existing models now.

## 1. The machine and the cost model
- One sequence, one token at a time. Each emitted token requires the model's computation at its position; speculative decoding lets a check of R draft positions share weight reads, but the reads it shares are only those rows have in common.
- Unified memory: ~300 GB/s for everything the GPU reads; ~45 GB usable for weights and cache. A 4-bit 125B does not fit (68 GB of experts + 3.3 GB dense), so 40 % of the experts live on the SSD at 13 GB/s and 0.149 ms per 2.9 MB read [K].
- The costs that bind, measured on the 125B at 16-row checks: 3.3 GB of dense weights per pass (11 ms at the floor, 27 ms today); ~0.5-0.6 GB of expert bytes per row, because 16 rows touch 72 of 512 experts per layer and the union grows as N^0.8 [K]; the expert kernel ALU-bound at 6.5 us per (row, expert) pair at 1-2 rows per expert [K]; 128 MB of SSD reads per kept token on code at a 45 GB cache, 60 % of a code check [K]; a drafter that keeps 10.5 of 16 on code and 2 of 16 on chat [K]; ~3,900 kernels per check [K]; a GDN state of 36 x 3.1 MB that speculation must snapshot [K].
- Where the decisions are: the trunk with the hottest 10 % of experts reproduces 98 % of code tokens and 90 % of chat tokens; ranks 0-1 of 10 are load-bearing, ranks 2-9 pass at 3 bits; dropping to top-8 costs 1.24-1.45x the KLD gate [K]. An untrained exit at layer 40 of 48 agrees 5.7 %; nothing is readable at layer 12 [K]. The HC mixers are lossless at 8 bits and flip 6 % of tokens at 4 bits [K]; their input stream has a consistent ~1 % of channels at 11-35x the RMS [K, measured tonight].

## 2. Principles
P1 Separate knowledge from decision, and charge them differently. Decisions form in the dense trunk and in the top-ranked experts; the long tail of experts is vocabulary. Everything about the design should make the trunk cheap to run per step and the tail cheap to skip, read in bulk, or share.
P2 Bytes per token must be sharable across speculative rows. The union of experts a block touches, not the active parameters per token, sets the cost at batch 1 with speculation.
P3 Routing must be predictable at least a layer group ahead, in units the SSD and the cache can manage.
P4 Every dense tensor quantises to 4 bits losslessly by construction; nothing stays in bf16 for stability's sake.
P5 Depth is latency and kernel count; width is bandwidth. At batch 1, prefer width.
P6 The model drafts itself: a head trained to chain, on the trunk's own features, is part of the model.
P7 The state is checkpointable for free.
P8 The vocabulary serves the head, not the tokeniser's convenience.

## 3. The design
3.1 Experts: fine-grained, but routed by slab. Keep small experts (they minimise the union's bytes per row: 10 GB per 16-row pass against ~38 GB for coarse 100M-parameter experts touching half a layer [P]). Constrain each token's top-k to at most 2 of 16 slabs per layer (DeepSeek's node-limited routing, re-purposed from device locality to storage locality). Slabs of 32 experts are ~90 MB records: SSD reads become long and sequential at full bandwidth, the cache manages 768 slabs instead of 24,576 experts, and prefetch predicts a slab, which is far easier than an expert. Quality cost: node-limited routing was trained into DeepSeek-V3 at no reported loss [P].
3.2 Routing: one router per group of 4 layers, with the group's slab choice made at the group's first layer. Routing is then known 3 layers ahead at zero cost, which is the lead the SSD needs (the oracle at lead 3 was worth +33-48 % at 24 GB here [K]); within the group, expert choice stays per layer. A stickiness term in the routing loss (consecutive tokens prefer the same slabs) raises the 36 % consecutive overlap we measure today [K] and shrinks the union per 16 rows, which is the only lever on bytes per row that does not touch precision [G].
3.3 Shared expert: large, not tiny. The always-on expert is read once per pass and shared by every row; at 16 rows its bytes are amortised 16x. Put common knowledge there (20 % of expert parameters), rare knowledge in the routed tail [P].
3.4 The trunk: 24 layers, hidden 4096, instead of 48 at 2560. Same dense parameters, half the sequential depth and kernel count, matvecs twice as wide for the matrix units, half the recurrent state. Plain pre-norm residual stream, or hyper-connections trained with quantisation-aware mixers at 4 bits; no bf16 anywhere in the trunk (P4). The 1.27 GB bf16 tax and the 4-stream residual go away [K for the tax].
3.5 Mixers: attention everywhere with 2 KV heads and a 128-dim head, sparse only beyond 32K. At batch 1 the KV read is 12 KB per token per pass, 0.2 GB at 16K, under a millisecond per check; rollback for speculation is a length, and trees cost nothing but rows. GDN's prefill advantage is worth little to one user [P]. If a recurrent mixer is kept, its state must be small and its update logged so rollback is free (P7).
3.6 Precision designed in: routed experts trained so that 3 bits are lossless at ranks >= 2 and 4 bits at ranks 0-1, matching what we found by accident with the rank tiers [K]; the dense trunk at 4 bits (P4).
3.7 A chaining draft head, trained with the model: a small head (no experts) on multi-depth trunk features, trained with the training-time test (depth j sees its own chain) so that chained drafts hold ~0.9+ per token over 16 positions, and a mid-depth exit at layer 12 trained with layer dropout so a self-draft with half the trunk is honest [P: the native head trained for one step reaches 0.64 at depth 1 on chat, DFlash2 0.93 on code; EAGLE-3-style training lifted the head's code chain 26 % here].
3.8 Vocabulary 128K: half the head bytes, the exact two-stage top-k for drafting, and tokenisers that keep Bulgarian and Arabic near 1.5 tokens per word so chains do not break every 1.6 tokens there [P, from the coverage data].

## 4. What it would run at on this MacBook
Against the 125B at equal quality (same active knowledge, same trunk parameters), at the engine floor [P/G]:
- dense per pass 2.0 GB (6.7 ms) instead of 3.3 (11 ms);
- expert bytes per row ~0.35 GB with sticky slab routing (1.2 ms) instead of 0.5-0.6 (1.7-2 ms);
- SSD off the critical path at 45 GB: slab reads at full bandwidth with a 3-layer lead;
- a drafter at ~0.93-0.95 per token over 16 on code, ~0.8 on chat, at ~2 ms per block: kept ~12.5 on code, ~5 on chat;
- check floor 6.7 + 16 x 1.2 + 2 = 28 ms: ~450 tok/s on code, ~180 on chat; with today's-engine-class overheads (x1.5), ~300 and ~120. The 125B's corresponding floor numbers are ~260 and ~100 [P, missions 1-8].
Not 1,000: that needs tokens no version of the model processes at all.

## 5. What an engine can imitate today on existing models
- Slab caching and prefetch: group experts offline by co-routing statistics into contiguous records, manage residency and prefetch per group; exact, +latency only (the SSD's 13 ms per-check constant), ~+8 % chat / math [P].
- The knowledge / decision split: the trunk plus the resident experts as the drafter (D2), low-rank experts dropped on drafter-sure rows inside the check within accepted noise (quality-defined), if Monday's map allows [P].
- HC at 8 bits or 4 bits with hot columns kept (mission 11 W1 / W2): -5..-8 % of a floor check, a reference change [P].
- The chaining head: train the existing MTP head the way it chains (done once, +26 % on code) and keep its experts resident [K].
- Rollback: one snapshot and a replay of accepted rows for the GDN state [P].
- Width for depth, sticky routing, the shared-expert split, the vocabulary: training-time choices, not imitable.

## 6. What would prove this blueprint wrong
Monday's decision map: if top-2 flips > 5 % on sure code positions, the knowledge / decision split (P1) is false for this family and 3.3 / 5 above lose their basis; if flips are flat across layers, the trunk cannot be made cheap and 3.4 buys only kernel count; if the HC stream's hot channels do not explain HC4, 3.4's "no bf16" rests on quantisation-aware training alone. The slab-routing claim (3.1) needs one training run nobody here can do; its engine imitation (slab caching) is the cheap proxy.
