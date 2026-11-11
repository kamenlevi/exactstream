# CLOUD TASK 17: the clean-sheet engine. Design what you'd build from scratch for 350+.

A closed thinking task. Don't use the web; think and calculate. Everything you need is below, measured unless marked.

**Why.** Your ladder (task 15) improves today's engine piece by piece: ~225 tok/s halved, ~330-350 at targets. Tonight's measurements showed several pieces have hard limits:
- **The exact dense pass floors at ~19-21 ms per 16-row check** (FAST ~16-17; ~14-15 with HC8). The cost is ~2,000 small dependent kernels, not launches: the GPU idles only ~2.4 ms of an 87 ms check.
- **The matrix-unit expert kernel (NAX6) is compute-bound, not byte-bound,** on padded 16-row tiles. Only ~2.1 real rows sit in each (expert, tier) group.
  - Peak ~14 T MAC/s; bitwise rewrites were slower; a SIMD-ALU hybrid can't be bitwise. As FAST it's ~-4 ms per check.
- **Your sketch-on-miss with plain rank-32 SVD is near-worthless:** the experts' spectrum is flat, rank 32 keeps 12-16 % of the energy, relative error 0.92-0.94. An activation-aware version is untested.
- **Tree branching in the GDN is cheap but doesn't pay:** state 3.15 MB per layer, 113 MB per node (not 2.4 GB); a fork ~1 ms, a leaf ~0. After a rescued hard token the run continues only ~1.1 tokens on chat and ~3.2 on code.
- **The draft can be half hidden, not fully:** a pre-drafted candidate is usable when both its anchor and its bonus guess are right, which is 0.59 code / 0.22-0.32 chat with 5 candidates. Exposed draft floors at ~2.3-3.2 ms on code.
- **New:**
  - The SSD charges +50-65 us per read burst after 0.25-1 ms of idle; 4 KB keepalive reads every 50 us remove it, ~3 ms per check.
  - Q1, the gate keyed on drafter certainty, is measured on code: expert bytes per check x0.72 (the union per 16-row block), SSD bytes x0.60-0.72 behind the C 256 cache, flips 0.09 % of gated rows.
  - Rows per expert group (code, the union per block): 2.33 at 16 rows, 3.12 at 32, 3.75 at 48, 4.29 at 64. Consecutive checks' accepted lengths correlate 0.60 on code.
  - Reusing the previous position's MoE output (layers 8-11 on all rows, 12-35 on sure rows) saves only x0.91 of expert bytes, because reads are a union per block.

**The facts**
- **Model:** 48 layers (36 GDN + 12 attention), hidden 2,560, 4 hyper-connection streams (low-rank 320, input-dependent mixing), 512 experts top-10 (intermediate 640), and an MTP head.
- **Weights:**
  - experts at 3-bit tiers: ~2.15 MB per distinct expert, plus a 0.77 MB 1-bit sidecar for ranks 0-1
  - dense trunk ~3.44 GB
- **The Mac:** RAM 258-287 GB/s; SSD 11.3-13 GB/s; matrix unit ~14 T MAC/s; 20 GPU cores; Neural Engine 12.4 ms per batched 3-candidate draft (GPU slows x1.02); CPU ~1 TFLOPS.
- **Today:** 63.6 tok/s exact at ~42 GB. A day check takes ~166 ms:
  - draft 6.3-12 ms
  - dense ~34 ms GPU
  - experts 43-50 ms
  - SSD exposed 39-65 ms
  - kept ~10.5 per 16-row check on code, ~2.4 on chat
- **The drafter:** the knowledge wall puts affordable drafters at ~0.96-0.967 per token. Kept tops out ~12-12.5 exact and ~13 FAST on code. More training flattens: +0.2 kept per doubling, then +0.06.
- **Quality contract:** EXACT = bitwise to the engine's own greedy output; FAST = same scores on math / code / instruction / multilingual tests.

**Deliver, in this order**
1. **The clean-sheet design** (<= 500 words): if you built the engine from zero for this model on this Mac, what is the data flow of one check, with ms per piece? Keep or throw away any of today's pieces (16-row checks, the two-pass idea, the gate, OOO, the drafter), and say why.
2. **Where it beats the ladder** (<= 250 words): tok/s halved and at target, on code and chat, EXACT and FAST, and which assumption of today's design it drops.
3. **The three biggest unknowns** (<= 200 words), each with a 10-minute test and a pre-registered line.
4. **One radical option** (<= 150 words) that changes the problem rather than the engine, without a smaller model or pruning (all 512 experts stay), plus its odds.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number given / arithmetic / judgement. Halve your own gains.
- Price expert bytes and matrix-unit tiles on the union per check (an expert read or tiled once serves every row that uses it), never per row.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- Cut depth, not sections.
- Print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval. Don't change or commit anything in any repository, and don't install or download anything.
