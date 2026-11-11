# Bit-exact speculative decoding of a 125B mixture-of-experts model on one 64 GB laptop

*Batch-invariant kernels on Apple's GPU and its matrix units, SSD-streamed experts at 45 GB and 12 GB, and a lossy mode
certified on task scores.*

Authors: [[AUTHORS: to be chosen by the project owner]]

Alternative titles: (a) "Exact 16-token speculative decoding of a streamed 125B MoE on Apple silicon"; (b) "What it
takes to verify 16 tokens at once, bit for bit, on a model three times larger than RAM".

---

## 0. Draft status (delete before release)

- **v1.1, Fri 6 Nov 2026, 21:00-21:40, for Sat 7 Nov 14:00.** Structure, methods, measurements and negative results
  are in. The final result table is not: it waits for the weekend's gated runs and the config freeze.
- **Owner confirmations are in (Fri 6 Nov, 21:2x-21:3x).** The owners of the engine, the kernels, the 12 GB line and
  the inventory answered the open questions of v1; Appendix A lists the answers. The largest correction: the 12 GB
  line is verified greedy decoding, not bit-identical to its own one-token decoding (section 2.8).
- **Marks.** [K] = measured, with the source beside it. [P] = arithmetic or a CPU replay on measured data. [G] = a
  judgement. A [P] or [G] is never a result; the text keeps the mark wherever the number goes.
- **Placeholders.** `[[TBD-n: ...]]` marks a number that does not exist yet. Appendix A lists every one, with its
  owner and the file expected to fill it.
- **Status checks done for v1.** Every number below was read from the file named beside it on Fri 6 Nov 21:00-21:30,
  and its heading was checked for WITHDRAWN / CORRECTED. Two headline numbers were withdrawn this week (40.8 tok/s
  "12 GB exact", 157.9 tok/s "45 GB FAST stack"); both appear only in section 7, as negative results.
- **Not verified for v1.** (1) No external paper or repository was opened: every citation is copied from the
  sessions' records (section 8) and must be re-opened before release. (2) Tonight's runs (the FAST stack without
  CD16, its same-scores gate, MBPP) had not finished when this was written.
- **Source keys.**

  | key | file |
  |---|---|
  | T*n* | a stable label for one entry of the ledger `~/TRIED/TRIED.md`. *n* was the entry's line number in the merge of Fri 6 Nov ~21:00; the ledger has been re-merged since. Appendix D maps each label to the entry's ID, and `check_sources.py` finds entries by content |
  | INV | `~/NewIdeas/PUBLISH_INVENTORY.md` (draft, Fri 6 Nov 20:4x) |
  | NOV | `~/TRIED/NOVELTY.md` (refreshed Fri 6 Nov 21:3x) |
  | PA | `~/NewIdeas/PRIOR_ART.md` (30 Oct) |
  | Q | `~/NewIdeas/QUEUE.md`, head (Fri 6 Nov 21:0x) |
  | EV:*name* | `~/EvalSuite/docs/<name>.md` |
  | LT:'*heading*' | a section of `~/Leap/TABLE_350.md`; LL = `~/TRIED/leap.md` |
  | F:*slug* | a note in `~/FableGod/findings/` |
  | BM:*file* | a file in `~/BigMoE/` |
  | F12 | `~/Fit12/PLAN_12GB_90_fit12.md`, 'Lever 3 RESULT' |
  | RF | `~/MoonShots/research/results_first/RESULTS_FIRST.md` (4 Nov; its 12 GB row is the withdrawn 40.8) |
  | KP / KW | `~/TrainLab/K_PLAN_1106.md` / `~/FableGod/goal/K_WALL_1106.md` |
  | L350 | `~/FableGod/full/LOSSY350_FULL.md` |

---

## Abstract

We decode a 125B-parameter mixture-of-experts language model (48 layers, 512 experts per layer, 10 routed per token)
on one 64 GB laptop, with the routed experts streamed from the internal SSD at 3 to 4 bits per weight. Our
contribution is exactness: a speculative check that verifies 16 drafted tokens in one pass and is bit-identical, in
tokens and in logits, to decoding the same model one token at a time. Stock kernels do not give this, because a
1-row call and a 16-row call run different code. We obtain it with batch-invariant kernels on the GPU and on its
matrix units, fixed attention key buckets, and a load-time self-test, and we audit it over 36 full-length prompts
(15,690 of 15,690 kept rows bitwise on the current reference, 16,305 of 16,305 on its predecessor). Every later
exact optimisation is then gated by token equality, which is cheap to test.

On that base the engine decodes at [[TBD-1: 45 GB EXACT tok/s per workload]] within a 45 GB memory line (49.9 tok/s
[K] on a 13-prompt mixed set today). A second configuration runs within an 11.4 GB lifetime footprint at 18.4-18.9
tok/s [K] on HumanEval-type code. Every token it commits is verified by the full 12 GB computation, but it is not
bit-identical to its own one-token decoding (section 2.8). That speed is about 12 times the fastest exact run at
12 GB or less that we found published (1.48 tok/s; the exactness class, precision, hardware and prompts all differ,
section 6.3) [[TBD-18: the project owner decides whether this comparison is printed]]. A lossy FAST mode, accepted
on task scores rather than bits, reaches [[TBD-2: 45 GB FAST tok/s, workload named, gate verdict]] on code; chat and
agent turns stay exact, because FAST failed their gates.

We also report methods that should carry to other MoE models (an SSD wake penalty on Apple's internal SSD and a
keepalive that removes it; 3-bit top-ranked experts on rows the drafter is sure of; relaxed acceptance restricted to
code comments), per-kernel measurements on the M5 GPU, and the negative results, including two headline numbers we
withdrew. By our own prior-art checks every lever in the shipped configurations is published or adapted; what is
ours is making them bit-exact together, at this size, on this hardware.

---

## 1. Introduction

### 1.1 The setting

- **The model.** Qwen3.8-Flash-Next is an open-weight MoE with a hybrid trunk: 36 linear-attention (Gated DeltaNet,
  "GDN") layers and 12 full-attention layers, four residual streams joined by low-rank "hyper-connection" (HC)
  mixers, one shared expert and 512 routed experts per layer with 10 routed per token, and a multi-token-prediction
  (MTP) head (table in section 3.1; BM:FINDINGS.md section 1).
- **The machine.** One MacBook Pro, M5 Pro, 64 GB of unified memory, internal SSD. All numbers are single-user decode.
  The M5 GPU has a Neural Accelerator in each core (Apple's term). We call these the matrix units and program them
  through Metal 4's Metal Performance Primitives tensor operations, a 2-D matmul on cooperative tensors (source: the
  kernels' owner, 6 Nov; the wording is to be checked against Apple's pages, [[TBD-17]]).
- **The problem.** At MLX 4-bit the routed experts alone are 67.9 GB (BM:FINDINGS.md section 1), so they cannot all
  be resident. Each token reads about 2.46 GB of dense weights and 1.33 GB of routed experts [K, same source]. A token
  therefore costs a full pass over the dense trunk whatever the cache does, and decoding one token at a time is slow.
- **Speculative decoding** fixes the second cost: a small drafter proposes a block of tokens, and one pass of the
  large model verifies up to 16 of them. The dense trunk is read once per check instead of once per token.
- **Why exactness is not free.** A verify pass computes 16 positions in one batched call; one-token decoding computes
  each position in a 1-row call. On stock kernels those are different programs with different rounding. In our first
  screen, all 16 rows already differed from the one-token results at the first matmul of layer 0 (section 2.2). A
  model that greedy-decodes then drifts at near-ties, and "speculative decoding is lossless" stops being true.

### 1.2 Contributions

1. **Exact 16-row speculative checks at 125B on Apple silicon (section 2).** A reference kernel set under which a
   16-row verify equals 16 one-token steps bit for bit, including attention past 512 and 1,024 keys, on the GPU and
   on its matrix units; a live-loop audit; and a rule for changing the reference (certify once on scores, then demand
   token equality).
2. **A streamed engine at two memory lines (section 3).** 45 GB (exact) and 12 GB (verified, section 2.8), with the
   levers that carry the speed: a sync-free expert gate, out-of-order expert execution, rank-tiered records,
   per-layer pools, and SSD handling.
3. **A lossy mode and a way to certify it (sections 3.6-3.7).** FAST changes which experts a row computes, and is
   accepted only on task scores, low-probability "blunders", drift along long answers, and two cache histories. The
   method rejected several of our own fastest configurations.
4. **Methods that carry to other models (section 4)**, each tagged published / adapted / new with its evidence.
5. **Measurements on the M5 GPU (section 5):** what one dependent dispatch costs, where a check's time goes, and why
   narrow matrix-vector kernels, not kernel count, were the loss.
6. **Negative results (section 7).** 608 of the 1,345 ideas in the project's ledger ended in KILL [P: sum of the
   ledger's section tallies on 6 Nov]. We report the ones that teach something.

### 1.3 What we do not claim

- **Not a speed record for this model.** Public engines run the same model at 56-61 tok/s with 2.4-bit experts and
  a 16 GB expert store on the same chip class (RF, rows "Kapitsa", "Splash 1.5.0"; self-reported, total process
  memory unknown). Our lines use about 3.2 bits per expert weight; the 45 GB line is exact to its reference and the
  12 GB line is verified (section 2.8).
- **Not new methods.** By the project's own web checks, every lever in the shipped defaults is published or adapted
  (NOV, "Every lever in the live defaults is published or adapted"; INV section 1). Output-exact SSD streaming is
  itself published: Memory-Sovereign Inference streams an 80B model with byte-identical logits (NOV, 3 Nov entry).
- **Not exact to the original 4-bit weights.** "Exact" always means exact to a stated reference (section 2.1).

---

## 2. Exactness

### 2.1 Definition

A decode mode is **exact** when its output tokens, and the logits row behind every kept token, are bit-identical to
one-token greedy decoding under the same **reference**. A reference is a fixed set of weights, kernels and call
shapes. Changing a kernel changes rounding, so it makes a new reference. A new reference is accepted once, on task
scores (section 2.6); after that, every optimisation must reproduce its tokens exactly.

The references used in this paper:

| name | what it is | status |
|---|---|---|
| REF2 | batch-invariant kernel set at 4 bits (section 2.3) | adopted 28 Oct (T1210) |
| REF2b | REF2 + key buckets, slot call, commit phase, SDPA classes (section 2.4) | adopted 31 Oct; the 4-bit exact mode (T1219) |
| REF2-T | REF2b on rank-tiered experts: ranks 0-1 at 4 bits, ranks 2-9 at 3 bits | 45 GB default since 1 Nov (T1214, T307) |
| NAX6 / NAX6-N | REF2-T with the routed experts on a matrix-unit kernel; -N is the same on a native record layout | certified 5 Nov as the 45 GB exact reference (EV:RESULT_nax6_2026-11-05) |
| the 12 GB line | not a reference in this sense: verified greedy decoding in calls of 2-4 rows (section 2.8) | deterministic; not bit-identical to its one-token decoding |

### 2.2 Why stock kernels are not batch-invariant

- **The first screen (INV0).** Sixteen one-token steps were compared with one 16-row verify on identical caches, per
  layer and per row, bitwise. With the shipped kernels all 16 rows diverged at the first record, layer 0's GDN block:
  the 1-row and 16-row matmuls are different kernels. Logits were equal on 0 of 16 rows [K, LT:'R-40 REF2', INV0].
- **Three kinds of break were found, in this order.**
  1. Matmuls and HC reads: the framework picks a kernel by row count (it switches at 6 rows, T1225).
  2. Attention: at 1 row versus inside 16 rows it runs other code and other key blocking (INV0 with the matmul
     patches: layers 0-2 equal, the first divergence at layer 3, the first attention layer).
  3. Long context: past 512 keys the attention kernel's split depends on the key count, and below 128 keys a second
     kernel class applies (section 2.4).
- **What was invariant as shipped:** the GDN recurrence (its state is fp32 and the step kernel is sequential) and the
  framework's gather-style expert call [K, INV0 / INV1]. A chunked GDN verify would not be: it changes the arithmetic
  (T1244, KILL on paper for exact modes).

### 2.3 REF2: one kernel shape for decode and verify

The principle is the published one (batch-invariant kernels with fixed split sizes; Thinking Machines Lab, cited in
PA). Our port to MLX / Metal:

- **Pad to one shape.** Every quantized matmul, bf16 linear and HC read with fewer than 16 rows is padded to 16, so
  a row's bits never depend on how many other rows are in the call.
  - 4-bit, group-64 projections with N >= 1,024 run on one 16-row matrix-unit kernel (qmm16c).
  - Other matmuls use the framework's own 16-row call, which is padding-independent for every shape we tested (T478).
  - HC reads use a fused 16-row form.
- **Routed experts on one 16-row tile kernel,** whose per-row sums do not depend on the row count, used for decode
  and verify alike (T1195, T479).
- **Attention through the verify path.** A call with fewer than 16 rows is padded to 16 with copies of its last row,
  run through the 16-row path, and the attention cache is trimmed back afterwards (T1209).
- **A load-time self-test** checks the invariances before any token is produced.

Results on a 12-layer slice at 1K context [K, LT:'R-40 REF2']:

- INV1: the logits of all 16 rows and every one of the 37 records per forward are bit-identical between 16 one-token
  steps and the 16-row verify.
- Speed: the check is faster than with the shipped kernels: 39.60 -> 26.78 ms per 12-layer check at 1K context and
  41.88 -> 27.36 ms at 8K [K, paired]. Scaled to 48 layers that is -51 and -60 ms per check [P: x4]. The padded
  matrix-unit kernels beat the chunked small-row calls: the 4-bit dense part fell from 56.8 to 27.4 ms per 16-row
  check (T478).
- The price: plain one-token decoding now runs 16-row kernels and slows to x0.75 (11.65 vs 15.50 tok/s, T1210). The
  speculative loop is the product; one-token decoding is only the definition.

### 2.4 What broke in the live loop, and the fixes (REF2b)

REF2 passed the slice and then failed the first full-length audit: its 16-row loop was token-equal to its own
one-token decoding on only 28 of 36 prompts, math 4 of 12 (T1259, KILL of REF2 as it was). The causes were all in
attention at longer context, and they are MLX-specific findings of this project (PA, row 1):

- **Key buckets.** Past 512 keys, every 16-row attention call runs at the next multiple of 512 keys, zero-padded and
  masked. Rows whose one-token step would land in another class are recomputed there. With buckets on, the live loop
  is bit-exact across the 512 boundary (160 of 160 rows on one prompt, 220 of 220 on another); with buckets off, 2 of
  160 (T1208).
- **SDPA classes.** Rows of a 16-row call whose one-token step had at least 128 keys while the call has fewer are
  recomputed, padded and masked to 128. With it, 1,477 of 1,477 tokens and 441 of 441 logits rows are bitwise on 6
  prompts; the control still breaks (216 of 256) (T1218).
- **The commit phase and the slot call** complete the set (T1219; the commit-phase sigmoid is named in PA).

### 2.5 The audits

Each audit decodes full-length prompts twice, once with the 16-row speculative loop and once one token at a time, and
compares every kept row (tokens and a hash of the logits row).

| reference, arm | prompts exact | tokens | kept rows bitwise | long context | source |
|---|---|---|---|---|---|
| REF2b, plain | 36 / 36 | 16,341 | 16,305 / 16,305 | 15 / 15 past 512 keys, 6 / 6 past 1,024; longest 2,056 keys | EV:RESULT_r56b_2026-10-31 |
| REF2b + the sync-free gate (160 ids) | 36 / 36 | 16,341 | 16,305 / 16,305 | same | same |
| REF2-T + out-of-order experts + fused combine | 36 / 36 | - | its one-token rows equal the previous audit's on 15,489 / 15,489 | - | T490 |
| NAX6-N + out-of-order experts, on the gate | 36 / 36 | 15,726 | 15,690 / 15,690 | 15 / 15 past 512, 6 / 6 past 1,024 | EV:RESULT_nax6_2026-11-05, Part 2 |

- The 36 prompts are 12 code, 12 math and 12 instruction prompts.
- In the NAX6-N audit the fast paths were confirmed engaged: the out-of-order path ran on every tier call (753,120 of
  753,120 in one-token decoding, 147,036 of 147,036 in the loop).
- The rank-tier reference was first checked at the kernel level: 16 of 16 rows and every layer equal at 383 and 536
  keys (T1214).

### 2.6 Changing the reference: certified once, on scores

A new reference moves bits, so it is judged on tasks. The rule ("decision 4") is fixed in advance: on each of three
sets, the paired score difference must be above -5 points **and** the exact McNemar p must be at least 0.05. The sets
are math (n 50), IFEval (n 100, strict, prompt level) and HumanEval+ (n 60, pass@1): 210 rows.

| new reference vs old | math | IFEval | HumanEval+ | verdict | source |
|---|---|---|---|---|---|
| REF2 vs the previous reference (ref.r2) | +8.0 (p 0.12) | -3.0 | 0.0 | PASS | T1210; EV:RESULT_ref2_2026-10-28 |
| REF2b vs ref.r2 | +6.0 (p 0.25) | -3.0 (p 0.38) | 0.0 (p 1.00) | PASS | T1219 |
| REF2-T vs REF2b (3-bit ranks 2-9) | 0.0 (p 1.00) | +3.0 (p 0.38) | +3.3 (p 0.50) | PASS | T307 |
| NAX6 vs REF2-T | 0.0 (88.0 vs 88.0) | -3.0 (87.0 vs 90.0, p 0.375) | -3.3 (91.7 vs 95.0, p 0.50) | PASS | EV:RESULT_nax6_2026-11-05, Part 1 |

Stated honestly:

- **These are same-score tests on 210 items, not proofs of equal quality.** For NAX6, pooled over the three sets, NAX6
  alone is right on 2 items and REF2-T alone on 7 (two-sided p 0.18). The rule is per set and every set passes; the
  lean is reported with it (EV:RESULT_nax6_2026-11-05).
- **REF2 itself tripped one information line:** 4 loop rows against 1 for its baseline (EV:RESULT_ref2_2026-10-28).
- **A reference can fail.** A further bundle of faster kernels (REF2c-T) failed decision 4 on IFEval (-5.0) and was
  not adopted (BM:HANDOVER.md, 'Key numbers').
- **The tiers are not exact to 4 bits.** REF2-T moves tokens relative to REF2b (T307, "exact: no"); it is its own
  reference. All ranks at 3 bits fails (section 7.9).

### 2.7 What exactness buys

Once a reference exists, an optimisation is tested by token equality on a handful of prompts, in minutes, with no
quality argument. Every lever of the 45 GB exact default was accepted this way: "every token == REF2-T" is part of
each line in section 3.4. The same property exposed two bugs that speed numbers alone would have hidden: a run that
never engaged its fast path (T18, "0 OOO calls") and a kernel that was silently bypassed (F:flat2 caveat).

### 2.8 The 12 GB line: verified, not yet batch-invariant

As corrected by the line's owner on 6 Nov against its code and records (F12; `~/Fit12/PLAN.md`, the T3t and T3R
results and the width tests WV1-WV3).

- **What it computes.** The loop drafts 15 tokens per check with the block drafter and verifies them with the 12 GB
  model in calls of 2 to 4 rows, nearly always 4. In those calls all ten routed experts are read and waited for;
  ranks 0-1 run at 4 bits and ranks 2-9 at 3. Every committed token is the greedy argmax of that computation.
- **What holds.** The output is deterministic and independent of the cache state: Friday's run reproduced Thursday's
  on 16 of 16 prompts after a different cache history. Calls of width 2, 3 and 4 are bitwise equal to each other.
- **What does not hold.** The output is not bit-identical to one-token decoding. The line's own one-token path is a
  different computation: it defers cold ranks (section 3.5), so its bits depend on the cache. Even with all ranks
  waited for, a width-1 forward differs from widths 2-4 at rounding level. So this line does not meet the definition
  of section 2.1. It gives what speculative decoding gives without the kernel work of sections 2.3-2.4: every
  committed token is the full model's argmax for its prefix, computed at a width of 2-4 rows.
- **A second caveat.** At near-ties the committed tokens also depend on the draft pattern: the engine replays one
  layer at width 1 when a small call keeps a single row. On Friday a drafter-side change kept 14 of 16 prompts
  token-equal, and both divergences followed this mechanism. A fix (a 2-row replay) made the tokens independent of
  the draft policy in October on another drafting path (12 of 12 policy cells). It was never validated on this path
  and is not planned before the freeze (ledger entry PR1 in `~/TRIED/fit12.md`; the owner, 6 Nov).
- **Not exact to the 45 GB reference or to 4 bits** in any case: the precision differs.
- **What would make it exact:** the one-shape discipline of section 2.3 on the 12 GB kernels, the replay fix, and a
  same-scores run. None is done ([[TBD-3]]).

---

## 3. The system

### 3.1 Model, store and hardware

| part | parameters | notes |
|---|---|---|
| routed experts | 120.8 B | 48 layers x 512 experts, top-10; one expert = gate/up [1280 x 2560] + down [2560 x 640] |
| linear attention (GDN) | 2.09 B | 36 of 48 layers |
| full attention + indexer | 0.62 B | 12 layers, 24 query heads, 2 key/value heads, head dim 256 |
| hyper-connections | 0.64 B | 4 residual streams, low-rank mixers |
| embedding / output head / shared expert / router | 0.64 / 0.64 / 0.24 / 0.06 B | one shared expert per layer |
| MTP head | 2.61 B | one layer with its own 512 experts |
| n-gram tables | 51.2 B | layer 1 only; 16 rows read per token, from the SSD |

Source: BM:FINDINGS.md section 1 (config and safetensors headers). The rows above the MTP head sum to 125.7 B [P];
about 6 B are touched per token [P: 2.36 B routed + ~3.65 B dense]. The records quote "~5B" and "~6B" active
[[TBD-4: one figure]].

- **Records.** Each expert is one contiguous, 16 KB-aligned record, read with one call and no page cache
  (F_NOCACHE). A record has an MSB part (the 3 most significant bits of every weight plus scales and biases,
  2,162,688 bytes) and an LSB sidecar (the 4th bit plus the original bias, 770,048 bytes). The 4-bit weights are
  rebuilt bit-exactly from the two parts. The store is 48 x 512 records, 72.1 GB (BM:DESIGN_3bit.md).
- **Rank tiers.** A token's two highest-scoring experts (ranks 0-1) run at 4 bits and need both parts; ranks 2-9 run
  from the MSB part alone. Expert bytes fall about 20 % (`~/TRIED/TRIED_SHORT.md`, "3-bit experts by routing rank").
- **Native layout and scale packing.** For the matrix-unit kernel the records are stored in the kernel's own
  fragment order (+18.7 % at 16 rows, T527), with scales packed to 11 bits (bitwise, +1.2 to +2.2 % cost, T334).
- **The SSD.** Cold 2.77 MB reads reach 14.0 GB/s at queue depth 16; one read at queue depth 1 takes 0.45 ms (median)
  [K, BM:FINDINGS.md section 2]. In the engine, the SSD time per check follows a line: 10-16 ms plus 0.15-0.17 ms per
  missed expert, which is the drive's streaming rate (12.7-14.4 GB/s) above a per-check floor [K, T288].

### 3.2 Expert cache and loader

- **Pools.** Each layer has a pool of C expert slots in GPU memory ("C 224" means 224 slots per layer on average).
  Slot counts differ per layer at equal RAM: -17.2 % SSD bytes per kept token and +8.3 % tok/s, exact (T12, M-17;
  adapted from AdapMoE).
- **Eviction** is LRU at 45 GB (about 9 % fewer misses than LFU there) and decayed LFU at 24 GB (`TRIED_SHORT.md`).
- **Loading.** All misses of a layer load in parallel (T7). On the converted store the reads are aligned and land in
  slot memory without a bounce copy (INV, M-29). A part of the pool is a volatile tier that the operating system may
  reclaim under memory pressure (toggle cost under 1 us, T1088).

### 3.3 The check loop

1. **Draft.** A block drafter (DFlash2, the L1-20 checkpoint) proposes 16 positions in one pass. For chat-type
   prompts the model's own MTP head is chained instead, chosen up front by a keyword rule (T782; PA row 5). Code and
   agent turns stay on the block drafter.
2. **Cut.** Drafts are kept only while the running product of the drafter's confidences stays at or above 0.3. Live
   checks therefore verify 9.7 rows on code, 3.7 on IFEval and 3.0 on multilingual chat [K, T993]. The cut is exact
   and gave +19 % (T309).
3. **Verify.** One padded 16-row pass of the 125B computes the rows. Drafts are accepted up to the first mismatch,
   and the row after the last accepted draft gives one more token for free (the bonus row).
4. **Commit.** The recurrent and attention states are advanced by the kept rows only.

Tokens kept per check decide the speed, and they depend on the workload (section 5.4).

### 3.4 The 45 GB exact default

Every item was adopted on "every token equal to the reference" plus a speed line written before the run.

| lever | what it does | measured effect | tag | source |
|---|---|---|---|---|
| sync-free expert gate | expert ids stay GPU data; the GPU waits on slots the loader publishes, with no CPU-GPU sync per layer; up to 160 ids per layer | warm check + commit 232.6 -> 117.3 ms; live +35 % (23.7 -> 32.0 tok/s) | adapted (anemll-flash-mlx, SlotStream) | T489; T476: 120-140 ms of a 200 ms check had been syncs |
| confidence cut at 0.3 | section 3.3 | 42.8 -> 50.9 tok/s (+19 %), tokens equal | published (EVICT, DraftExpert) | T309 |
| out-of-order experts + fused combine | resident experts compute first, missed ones as their reads land, in a fixed bitwise order | +13 % steady (rough, T14); +18.7 % in an A-B-A run (`TRIED_SHORT.md`) | adapted (MoE-Infinity, ProMoE, Fiddler) | T14, T490 |
| exact-safe kernel fusion | router, MoE tail, RoPE, residual | about -1,100 of 3,944 kernels per check, +2.4 % | engineering | `TRIED_SHORT.md`, K-73 |
| per-layer pool sizes | section 3.2 | +8.3 % | adapted | M-17 |
| drafter by task | section 3.3 | chat +24.5 % on 23 fresh prompts, faster on 19 of 19 chat prompts, every token equal; agent turns do not gain and stay on the block drafter | adapted (WhiFlash) | T782 |
| SSD keepalive, 50 us | section 4.1 | about -3 ms per check on clean prompts | new finding, known mechanism | BM:docs/NIGHT_1104.md:37 |
| loader-to-GPU handoff by a coherent spin | section 5.5 | handoff ~23-80 us -> ~3-4 us | adapted | NOV; T76 |
| volatile expert tier | section 3.2 | memory headroom, exact | adapted | T1088 |
| matrix-unit expert kernel on native records (NAX6-N) | section 2.1 | +6.7 % over the previous exact kernel (49.86 vs 46.75 tok/s, 13 prompts, C 192) | adapted | LL, 'NATIVE-FRI run 2' |

- Four levers adopted together on 4 Nov (the drafter switch, the keepalive, the volatile tier and a bundle of bitwise
  kernel merges) gave +12.1 %, and the loader-path work (joint read issue, copy split, coherent spin, system-scope
  post) +6.4 % more, all exact. SSD megabytes per kept token fell from 172.5 to 120.2 [K, T10; T565].
- **Speed today:** 49.9 tok/s [K] steady, geometric mean over 13 prompts of a mixed 14-prompt set, quiet machine,
  C 192, peak 38.1 GB (LL, 'NATIVE-FRI run 2'; INV section 2). Per domain, from the exactness audit and not an A/B
  (shared GPU): HumanEval+ 66.8, math 42.6, IFEval 21.0 tok/s, 7.20 rows kept per check
  (EV:RESULT_nax6_2026-11-05). [[TBD-1]] replaces both.

### 3.5 The 12 GB line

- **Two paths.** One-token decoding at 12 GB defers experts: a token's cold ranks 2-9 are added half a layer late, so
  their reads overlap the next half layer's compute. We found this independently in October; KTransformers published
  it first as Expert Deferral (NOV; T29). It is the accepted 12 GB default for plain decoding (T29, T1057). The
  drafted loop reported here does not defer: its multi-row verify calls read and wait for all ten ranks (the line's
  owner, 6 Nov).
- **The loop** drafts 15 tokens per check with the same block drafter and verifies in calls of 2-4 rows. Its check is
  bytes-bound: 4.2 GB of SSD reads and 522.5 ms per check, of which 510.7 ms is the verify pass and 11.2 ms the
  draft, for 9.71 tokens kept [K, F12, arm "ex", 14 clean prompts].
- **Workload.** 16 HumanEval prompts, greedy, a 256-token cap, full outputs, pools of 47 base and 32 sidecar slots
  per layer.
- **Speed [K, F12, with the owner's labels].** 18.58 tok/s over the 14 clean prompts, under a shared GPU lock (rough);
  18.38 over all 16, two of which ran beside another GPU job. Quiet-machine runs exist only on subsets: 18.9 on 4
  prompts and 18.42 on 8. The paper quotes the range 18.4-18.9. A quiet run of the loop alone over all 16 prompts
  does not exist ([[TBD-12a]]).
- **Memory.** 11.39 GB is the lifetime peak of one process that ran six arms in turn; the loop never ran alone. An
  earlier quiet run peaked at 11.354 GB. Reads bypass the page cache. The footprint is the engine's own on a 64 GB
  machine, not a limit enforced by the operating system (F:12gb-exact-40tps, caveats).
- **A two-pass FAST form** (a cheap top-2 pass that certifies rows, a full pass for the rest) exists and is not
  releasable as measured (section 7.4).

### 3.6 The FAST mode

FAST keeps the loop and changes what a verify row computes. Every part is lossy, so none is judged by bits.

| part | what it changes | evidence so far | tag |
|---|---|---|---|
| QOM-KEEP | on a drafted row, ranks 0-1 always run. A rank 2-9 expert is kept if the check loads it anyway (the union rule) or if it is already resident; otherwise it is not read and drops out at weight 0. No rerouting, no renormalisation. The bonus row stays exact unless DRAFT17 gives it a draft. One switch, set per prompt set (the code sets in the current stack) | decision map at the 45 GB pool: code 0.51 % of rows flip, chat 4.03 / 3.26 % (lines 1.34 / 5.42 / 3.69) [K, F:cloud3; T552]; SSD bytes -60 / -37 / -33 % at certainty 0.5 (NOV) | adapted (cache-conditional routing; OEA) |
| U3 | every kept expert of a drafted row runs from the MSB part (3 bits): ranks 0-1 lose their 4th bit, and no LSB sidecar is read for them | section 4.2 | new signal, published mechanism |
| DRAFT17 | one extra draft position, so every bonus row has a draft and can be gated | code SSD bytes per kept token x0.706 at 24 GB [P]; map 1.07 % flips, unchanged [K] (F:draft17) | adapted (extends our own certainty-gated bonus row) |
| HC8 | 8-bit HC mixers read in one pass on the matrix units | 0.54x the exact read's time (5.22 vs 9.67 ms per 96-read chain) [K, F:hc8]; decision 4 PASS alone (EV:RESULT_hc8_2026-11-02); fails chat combined with relaxed acceptance (section 7.6) | published (8-bit weights); the one-pass read on the matrix units is ours |
| FAST-NAX | the QOM / U3 forms on the matrix-unit expert kernel | +16.8 % over the previous FAST preset (113.9 vs 97.5 tok/s, 12 HumanEval, C 224), kept -1.8 % [K, Q] | adapted |
| DN-KS | a split of K per shape for narrow dense matrix-vector calls (section 5.2) | -4.19 ms of fixed cost per check [K, LL] | published (split-K) |
| att_single | one attention call per layer instead of 12 head chunks | -4.06 ms [K, LL]; bitwise below 512 keys | engineering |
| 17 to 32-row drafter projections on the matrix units | the drafter's block no longer falls back to slow calls | draft -2.1 ms, then -0.75 ms more [K, LL] | engineering |
| lazy GDN commit | exact; small | about -0.2 ms per check [K, LL] | engineering |
| RELAX B (code and math answers) | (1) the union gate: a row the drafter is sure of (certainty >= 0.9) keeps ranks 0-1, plus any rank 2-9 expert that the check's other rows load anyway, with no renormalisation; (2) the copy gate: a row whose last 8 input tokens occur in the prompt is treated as sure; (3) exact acceptance, except the comment rule of section 4.3 on code answers; (4) HC8 | tok/s x1.49 HumanEval+, x1.32 math against the exact control [K, paired, C 160]; expert bytes per check -41 % on code; decision 4 PASS (section 3.7). On the map with live-length checks the gate passes code (1.25 %) and is BETWEEN on chat (T993): a code and math lever | adapted (OEA; PSR) |

- **A plain statement about "drafter certainty".** As shipped, the switch runs at certainty threshold 0. "Sure" then
  means every drafted row, for QOM-KEEP, for the union rule inside it and for U3 (BM:scripts/r57_screen_fast.py, the
  QOM arm; confirmed by the owners of the engine and of the inventory, 6 Nov). Thresholds up to 0.3 are identical bit
  for bit on the live map, because the cut of section 3.3 already removes rows below a running confidence of 0.3
  (T1011). The drafter's certainty therefore enters only through the cut. The gate's live effect is "skip what would
  miss", which depends on the cache state.
- **Composition of RELAX B** is taken from its pre-registration (EV:PREREG_relax_2026-11-02, revision 3, "Arms").
- **Speed** is in section 6. The valid measured FAST number today is 113.9 tok/s on 12 HumanEval prompts, pending
  its gate.

### 3.7 How FAST is certified

No single test is trusted. A FAST configuration passes four layers, each with lines written before the run.

1. **The decision map (a screen, minutes).** The exact reference's own answers are replayed with the lossy rule
   applied, on 36 windows (12 code, 12 IFEval, 12 multilingual), and we count rows whose top token changes. Lines:
   1.34 / 5.42 / 3.69 % of rows. These are the measured all-row flip rates of the first gate approved for an
   end-to-end same-scores test (top-2 experts on rows with drafter certainty of at least 0.99). The rule therefore
   reads "no more flips than the form already sent to the task-level gate". The union gate at 0.9 replaced that form
   because it flipped less in all three domains (0.98 / 4.92 / 3.04 %), and the mode built on it passed decision 4
   (EV:PREREG_relax_2026-11-02, revision 3; the inventory owner's wording, 6 Nov).
   - **The tie floor.** Rows where the top two probabilities are within 0.1 (1.5 / 8.5 / 7.0 % of rows) flip 20-45 %
     under any perturbation. Quantizing the attention weights to 4 bits alone flips 0.92 / 4.53 / 3.43 % of all rows.
     The lines sit only 0.4 / 0.9 / 0.3 points above that, so the map now reports non-tie flips beside all-row flips
     [K, T1049].
   - The map predicted a later same-scores failure correctly (section 7.4) and passed one lever that then failed on
     speed (section 7.1). It is a quality screen only.
2. **Same scores (decision 4, hours).** The rule of section 2.6 against the exact reference, on the 210 rows.
3. **Three bars on the committed tokens.** Each committed token is re-scored by the exact reference along the mode's
   own prefix.
   - Blunders: tokens to which the reference gives probability below 0.01. Bar: at most 1.0 per 10,000 tokens on math
     and code.
   - The agent bar: no blunder at all in agent turns (tool calls and file writes).
   - Drift: on answers of 512 tokens or more, the last quarter's KL divergence must be at most 1.5x the first
     quarter's.
4. **Two cache histories.** A mode whose output depends on the cache state is certified under two prompt orders, and
   must pass in both (EV:PREREG_stack_2026-11-06).

Results so far [K]:

| mode | math | IFEval | HumanEval+ | decision 4 | bars | outcome |
|---|---|---|---|---|---|---|
| RELAX ship (B on code + E on chat) | 90.0 vs 88.0 | 87.0 vs 90.0 (p 0.25) | 93.3 vs 95.0 | PASS | 1 agent blunder; drift 6.04 on 4 long rows | BETWEEN, then routed: B ships for code and math only |
| B's drift re-check | - | - | - | - | pooled 0.82 on 16 long rows (line 1.5) | PASS as registered; see below |
| B + chat arm with HC8 (ship8) | 90.0 vs 88.0 | 84.0 vs 90.0 (p 0.03) | 93.3 vs 95.0 | FAIL | - | KILL: chat stays exact |
| plain FAST on 17 long agent turns, 2 orders | - | - | - | - | 85 blunders in 19,749 tokens (bar 0) | FAIL: agent turns stay exact |
| the FAST stack, 2 orders | [[TBD-7]] | (chat exact) | [[TBD-7]] | [[TBD-7]] | [[TBD-7]] | pending (the night of 6 Nov) |

Sources: EV:RESULT_relax_2026-11-05, EV:RESULT_relax_drift_2026-11-06, EV:RESULT_agentfast_2026-11-06.

- **The pooled drift PASS hides a per-source reading.** On HumanEval-type answers the divergence does grow along the
  answer (x4.07 on 6 new long rows, x6.04 on the first 4), from very small values (median first-quarter KL 0.0006).
  The pooled ratio passes because agent rows have a high, flat divergence. The registered line was the pooled one; we
  report both (EV:RESULT_relax_drift_2026-11-06, section 1).
- **Other arms of the same run failed decision 4** and are closed (EV:RESULT_relax_2026-11-05): the gate alone,
  without HC8 and the comment rule (math 82.0 vs 88.0); the gate with relaxed acceptance on every code row
  (HumanEval+ 90.0 vs 95.0); a wider chat rule (IFEval 85.0 vs 90.0); the gate on chat (IFEval 84.0 vs 90.0); all
  ranks at 3 bits (HumanEval+ 88.3 vs 95.0).
- **A limit of the rule itself.** The gate alone failed on math (82.0) and the same gate with HC8 passed (90.0), on
  the same 50 items. Four items moved the verdict. Decision 4 guards against large regressions; it does not measure
  small ones. That is why the map and the bars sit beside it, and why sample sizes should grow before release
  [G].
- **Speed per task, not only per token.** Relaxed chat rules lengthen answers (+15 to +43 % tokens per task), so the
  run also compares seconds per task. Every arm was faster per task (x0.80-0.86), but the per-token gains overstate
  it (same source, "The per-task line").

---

## 4. Methods that carry to other models

Tags follow the project's novelty record (NOV, INV section 1): **published** (used as is), **adapted** (a published
idea changed in a way that matters), **new** (no prior art found in our web checks), **engineering**. "Carries":
MD = any MoE with a drafter, SD = any speculative-decoding LLM, MAC = Apple's SSD or Macs, AGPU = MoE on Apple's GPU.
None of these is specific to this model's GDN or HC layers (INV).

| method | tag | status here | carries | evidence |
|---|---|---|---|---|
| batch-invariant 16-row kernel set + key buckets | adapted (Thinking Machines Lab) | live, exact | SD, AGPU | section 2 |
| SSD wake penalty + keepalive | new finding on a known mechanism | live, exact | MAC | section 4.1 |
| wake-ahead ping | adapted | passed in microbenchmarks, not live | MAC | T47, T98 |
| aligned zero-copy reads into GPU slots | engineering | live in the store | MAC | NOV, M-29 |
| loader-to-GPU handoff by a coherent spin | adapted | live, exact | MAC, AGPU | T76 |
| OS-reclaimable (volatile) expert tier | adapted | live, exact | MAC | T1088 |
| rank-tiered records (4 bits for ranks 0-1, 3 for 2-9) | adapted (HOBBIT, MC-MoE) | live, own reference | MD | T307 |
| out-of-order expert execution in a fixed bitwise order | adapted | live, exact | MD | T14 |
| cold ranks half a layer late | published (KTransformers) | live at 12 GB | MD | T29 |
| QOM: drafted rows skip the cold experts that would miss | adapted | FAST | MD | NOV |
| 3-bit kept experts on drafted rows (U3; "drafter-sure rows" in its tested form) | new signal on a published mechanism | FAST | MD | section 4.2 |
| union rule keyed on drafter certainty (Q1-UNION) | adapted (OEA; PSR) | RELAX B | MD | T103, T993 |
| relaxed acceptance only inside code comments | adapted | RELAX B | SD | section 4.3 |
| copy gate: rows inside a verbatim copy of the prompt count as sure | adapted (SAM-Decoding) | RELAX B | SD, MD | EV:PREREG_relax_2026-11-02 |
| one extra draft position for the bonus row | adapted (extends our own certainty-gated bonus row) | FAST | MD | F:draft17 |
| per-shape K split for narrow matrix-vector calls (DN-KS) | published (split-K) | FAST | AGPU | section 5.2 |
| the request's own prompt footprint as the "cold" table | new signal on a published mechanism | passed on code, not live | MD | NOV; T312 |
| matrix-unit kernels for padded expert tiles | published practice; the M5 form is ours | live | AGPU | NOV |

The count behind the tags: 309 adopted or passed levers, tagged as 36 published, 91 adapted, 5 new and 177
engineering (NOV, the refresh of Fri 6 Nov 21:3x; the tags are judgements on the records, with no new web check). No
lever in a release configuration is tagged "new". The one fully new idea (reusing the previous position's MoE
output) was killed on bytes (NOV).

### 4.1 The SSD wake penalty and the keepalive (closest to new)

- **Finding.** On this machine's internal SSD, the first read after an idle gap is slower. After gaps of 0.25-1 ms it
  costs +50-65 us per burst; after 20 ms, +100-120 us [K, T93]. A streamed MoE produces exactly such gaps: one burst
  of expert reads per layer, with GPU work between bursts. If every layer pays, that is about 3 ms per check at 45 GB
  [P, T93].
- **A per-call decomposition** puts the fixed cost of one loader call at 187 us of the engine's 283 us: first byte
  about 119 us, wake 44, dispatch 8, and about 96 us outside the I/O path [K, T97].
- **Fix.** A trickle of 4 KB reads keeps the drive awake. A real spacing of at most 270 us removes the whole +44 us;
  400 us or more leaves half [K, T44]. A busy CPU core does nothing, so the effect is in the SSD path [K, T93].
- **In the engine:** a 50 us keepalive, about -3 ms per check on clean prompts [K, BM:docs/NIGHT_1104.md:37], adopted
  in the exact default as part of the +12.1 % bundle (T10). No isolated A/B of the keepalive exists on the final
  engine (the engine's owner, 6 Nov), so its share of the bundle is known only from that figure.
- **A cheaper form** is a single 4 KB read about 200 us before a predicted burst. It removes 97.5 % of the penalty in
  microbenchmarks (100 us ahead is too late, 400 us too early), and 87-94 % at the engine's real gaps [K, T47, T98].
  It is not live.
- **What does not work:** pinging the record's own first block to pre-warm it (+0.6 to +3.2 us, T202); splitting one
  record into parallel reads (each piece pays its own request latency, T199); the page cache (+49 us per call, T201).
- **Prior art (as recorded).** Idle-state exit latency of NVMe drives is published. We found no sub-millisecond
  numbers for Apple silicon's internal SSD and no LLM offload work using it (NOV). The mechanism is known; the
  measurement and its use in an MoE loader are ours.

### 4.2 Three-bit top-ranked experts on rows the drafter is sure of (as shipped: on every drafted row)

- **Background.** Ranks 0-1 at 4 bits are load-bearing: all ranks at 3 bits fails both the divergence gate and the
  same-scores test (`TRIED_SHORT.md`; EV:RESULT_relax_2026-11-05, T3ALL: HumanEval+ 88.3 vs 95.0).
- **The signal.** On rows that verify a draft the drafter was confident about, the LSB sidecar is not read for
  experts that only such rows use. The rationale [G]: on those rows the large model's top token rarely depends on
  the 4th bit of its top experts. The evidence is the flip rates below.
- **Evidence [K, decision map].** With the union rule at certainty 0.9: 1.19 / 5.06 / 3.24 % flips on full 16-row
  checks, under the lines (T103). On the QOM form at certainty 0.5: 0.65 / 4.42 / 3.49 %, PASS on all three; at
  certainty 0 (the shipped setting): 0.71 / 4.45 / 4.14 %, so code and IFEval pass and multilingual chat is BETWEEN
  (T106). SSD megabytes per kept token: -46.3 % against -34.9 % for the union rule alone [P, T103].
- **As shipped.** The certainty threshold is 0, so every drafted row counts as sure and every kept expert of a
  drafted row runs at 3 bits; the bonus row stays exact unless it has a draft. The drafter's certainty enters only
  through the cut that decides which rows are drafted at all (section 3.6). The "certainty 0" numbers above are the
  shipped setting's.
- **Status.** Live in the FAST stack on the code prompt sets. Its task-score gate is the stack's ([[TBD-7]]).
- **Prior art (as recorded).** Per-token mixed precision exists keyed on the router's rank, and per-expert bit widths
  keyed on importance; none found keys precision on the drafter's certainty (NOV, row "3-bit records on drafter-sure
  rows").

### 4.3 Relaxed acceptance restricted to code comments, and its failure when widened

- **The observation.** In code answers, comment text is a small share of positions and a large share of the
  drafter's misses: 13.5 % of positions and 44.7 % of misses on one walk (T1350); on another, comments carry 52 % of
  the first misses in a check [P on K, L350 section 1]. A wrong comment word does not change what the code does.
- **The rule** (EV:PREREG_relax_2026-11-02). On a row whose draft sits inside a '#' comment, the draft is kept if it
  ranks first or second on its row and has at least a quarter of the top token's probability. Everywhere else
  acceptance is exact.
  - Only inside the answer's code fence; never in its first two lines; never in docstrings, strings or code.
  - Never on a comment that is or could become a tool directive (type comments, linter pragmas, an encoding line, a
    shebang).
  - The row must already be inside the comment before its draft, and a draft that starts with a newline stays exact.
  - The span is found by an incremental scan of the answer's text, token by token, so it needs no lookahead.
- **In the engine it is safe so far, and small.** As part of RELAX B it passed decision 4 and the blunder bar on
  code (0 blunders in 25,656 new HumanEval+ tokens), but it adds only 1.8 % to tokens kept per check on HumanEval+
  [K, EV:RESULT_relax_2026-11-05]. A replay had promised x1.07 to x1.16 (T1350). The live confidence cut explains the
  gap: checks that would have continued through a comment are already cut short [P, L350 section 8.2]. A second
  version is in tonight's run [[TBD-10]].
- **Widened, it fails.** "CD16" keeps the drafter's whole comment and docstring text without checking it, and checks
  only code. On 12 MBPP tasks it kept x1.45 tokens per check and ran +23.7 % faster, and the tasks' tests passed on
  8 of 12 against 11 of 12 without it, in both passes: 3 flips against a line of 2 [K, T855; F:45gb-fast-stack]. The
  answers show comment loops and code that drifts after the drafter's own comments. HumanEval did not show it,
  because its docstrings copy the prompt. CD16 was dropped the same day, with the headline number that contained it
  (section 7.3).
- **The lesson for others:** text that "does not matter" still conditions everything after it. Relaxing the choice
  among the model's own near-top candidates is a different operation from accepting text the model never scored.
- **Tag.** Relaxed and lossy verification is published; the comment-only span was found in neither of two web
  checks (NOV; INV section 1): adapted.

---

## 5. Measurements on the Apple M5 GPU

All timings are from one M5 Pro; "quiet" means no other GPU job. Method notes that matter are stated, because two of
our own earlier timing methods were wrong (sections 5.1, 7.8).

### 5.1 One dependent dispatch costs about 1.6 us

- **Measurement.** A chain of 200 dependent kernels that move almost no bytes: 1.24 us per kernel at 32 threads, 1.40
  to 1.56 us at 1,024 to 20,480 threads, 4.13 us when 1 MB is moved [K, LT:'STEP 0 PART B RESULT'].
- **Consequence.** A check launches about 3,000 kernels at 16 rows (3,900 at 2 rows). At 1.6 us each that is 2-6 ms,
  not the 20 ms an earlier fit assumed. Reducing the kernel count is not the main lever.
- **Method note.** Timing each kernel with its own synchronisation is not usable: the results swing 2-3x between
  processes. Only dependent chains, whole-piece splits and counts were used (same source). Even a split with one
  synchronisation per piece overstates absolute time (section 5.4).

### 5.2 The time is lost in narrow matrix-vector kernels

Per call in a dependent chain, at the engine's real shapes [K, same source]:

| kernel | tiles | time per call | bytes | GB/s | calls per check |
|---|---|---|---|---|---|
| GDN input projection (wide) | 512 | 113 us | 23.6 MB | 210 | 36 |
| attention query projection (wide) | 384 | 75 us | 17.7 MB | 243-247 | 12 |
| GDN / attention output projection | 80 | 112 us | 8.9 MB | 81 | 48 |
| shared expert gate/up | 41 | 48 us | 2.0 MB | 42 | 48 |
| attention key and value projections | 16 | 48 us | 0.8 MB | 18 | 24 |
| GDN a/b projections | 3 | 45 us | 0.22 MB | 5 | 36 |

- **Reading.** Wide kernels already stream near the machine's rate (about 250 GB/s is possible; F:flat2). Narrow
  ones run at 5 to 81 GB/s. The reference kernel walks the whole K dimension with one SIMD group per 32-column tile,
  so with 3 to 80 tiles the GPU is mostly empty and the call costs about 1.1-1.2 us per K group whatever its bytes.
- **Fix (DN-KS).** Split K per shape so that at least about 320 SIMD groups run. The output projection falls from
  115 to 41 us (214 GB/s), a/b from 47 to 10, key/value from 47 to 9, shared gate/up from 48 to 15 [K, LL, 'DN-KS step
  1']: -7.74 ms per 2-row check in the microbenchmark and -4.19 ms of measured fixed cost in the engine [K, LL,
  'DN-KS step 2']. The engine sees about half the chained gain, because narrow calls overlap other work (LT, 'DN-KS
  v2 by TOGGLE').
- **Its cost.** The split changes the order of summation, so it is not bitwise to the reference: the relative error
  is at most 2e-3 per projection, and most projections are at 0 or below 5e-6. The split is row-invariant, so a check
  would still equal one-token decoding under a reference built with the same split. That reference would need its own
  certification, and none is planned (the kernels' owner, 6 Nov). A bitwise alternative for the exact line, folding
  the narrow projections into the wide calls' tile streams, is proposed at about -2.7 ms per check and is not built
  [G].

### 5.3 Single-call attention and the drafter's block

- **Attention.** The 16-row attention call fell back to 12 head chunks per layer, each a matmul / softmax / matmul
  triple: 144 calls per check at about 22 us and 20 GB/s. One call per layer saves 4.06 ms of fixed cost and is
  bitwise below 512 keys [K, LL, 'SDPA-1'].
- **Drafter.** With 17 draft positions the drafter's block exceeded its 16-row fast path, and every projection fell
  back to a generic call at 52-57 GB/s (5.9 of 8.8 ms). A 17-32-row form on the matrix-unit kernel took 2.1 ms off
  the draft, and a true 32-row kernel 0.75 ms more [K, LL, 'D17-NAX', 'QMM32'].
- **Running total.** These brought the 2-row fixed cost from 55.2 to about 42-44 ms in one day [K, LL, 'SDPA-1'; the
  kernels' owner, 6 Nov]. On the real 16-row blocks of one HumanEval prompt the cycle fell from about 77 to about
  68 ms, -11 % [K, LT, 'NO-WAIT CEILING on the engine's own blocks']. That prompt keeps 14.42 tokens per check; the
  12-prompt average is about 9.4-10.

### 5.4 Where a check's time goes (FLAT-2)

- **Definition.** The fixed cost of a check = the steady 2-row cycle minus twice the warm per-row slope.
- **Result.** 53.7 ms with every FAST lever owned on 6 Nov morning [K, F:flat2; LT:'FLAT-2 RESULT']. The
  pre-registered lines were PASS at 30 ms and KILL above 36 ms: a KILL.
- **Per piece, a warm 2-row check, each piece timed with its own synchronisation:** GDN 13.2 ms, experts 11.1 (about
  4.6 of it fixed), HC reads 8.6, attention 6.4, router and shared expert 5.7, MoE tail 1.1, HC write 0.8 [K, same].
  The CPU builds the graph in 13.4 ms, hidden behind GPU work: the check is GPU-bound.
- **Read that split as proportions, not as time.** Each synchronised piece pays a fresh command-buffer start. Measured
  in place and unsynchronised, after the day's kernel changes, the whole dense pass at 16 rows takes 25 ms of wall
  time and about 22 ms of GPU time (the GPU is 87-89 % busy), where the synchronised split of the same pass sums to
  32-39 ms. The real gap over the 19.6 ms of chained kernels is 3-5 ms. An all-resident 16-row check is then about
  51 ms: about 25 dense and about 26 on the expert path [K, LT:'ENGINE GAP HUNT'; small pools, shared GPU].
- **Only the experts grow with rows:** +3.0-3.3 ms per added row on the previous expert kernel, all experts resident;
  +0.98 ms per row on the matrix-unit FAST kernel (60 vs 25 ms at 16 rows) [K, LT:'STEP 0 PART B RESULT'].
- **What it caps [P].** At 12 tokens kept per check, a 54 ms fixed cost alone caps code near 220 tok/s even with free
  experts. Speed on code is `kept / (fixed + per-row x rows + SSD wait)`; all three terms and the kept count must
  move together.
- **Kept tokens depend on the workload [K, KP].** 10.36 per check on the engine's 12 HumanEval prompts; on exact
  walks 9.57 (HumanEval+), 5.94 (MBPP), 7.00 (GSM8K) and 3.01 (agent turns). On the 12 HumanEval prompts at a 256
  token cap, 67.8 % of generated tokens are one verbatim copy of the prompt's signature and docstring [K,
  F:lossy350-thinking-hour; L350 section 1]; MBPP, math and IFEval answers copy 1-3 %. A code tok/s number without its
  workload is not comparable.

### 5.5 CPU-GPU visibility on unified memory

- A GPU kernel that spins on shared memory sees CPU writes only after 0.5-1,000 ms with relaxed or device-scope
  atomics; visibility is guaranteed only at command-buffer boundaries [K, BM:FINDINGS.md section 4]. Small inputs
  (under 4 KB) are copied at encode time, so a kernel spins on a snapshot forever.
- Two forms work: the CPU releases a layer by writing a fresh region that the GPU polls line by line (never-read
  lines cannot be stale; about 69 us per gated layer against about 230 us for a full synchronisation, T495), and
  system-coherent loads, which bring the loader-to-GPU pickup to a median of 2.9 us [K, T76].
- In the other direction, device-scope GPU stores reach the CPU only on flush, about 20-23 us after the next dispatch
  [K, T76].
- Read-ahead hints on the SSD slowed the GPU by 73 % in another group's report, and look-ahead prefetch did not pay
  at 45 GB here (section 7.9): on this hardware SSD traffic and the GPU share the memory system.

### 5.6 The model's own structure

- **Experts are independent functions.** On the real weights (three layers): no shared base, no common subspace, no
  near-duplicate neurons; routed expert outputs on one input are near-orthogonal (mean cosine +0.02). Substituting a
  resident expert for a missing one equals dropping it [K, F:lossy350-thinking-hour].
- **A union law.** If every row of an R-row check is kept, distinct (expert, layer) groups per kept token fall only
  about 25 % per doubling of R: 276 / 218 / 163 on code at R 8 / 16 / 32 [P on K routing, F:union-law]. Wider checks
  cannot make experts cheap by sharing.
- **Chat is an acceptance problem.** At equal check length chat routes fewer experts per token than code. Its cost
  comes from short checks (about 2-4 rows) and rejected rows [P on K, same].

---

## 6. Results

### 6.1 The final table (placeholders)

All rows: single user, greedy decoding, quiet machine, steady-state tok/s as a geometric mean over the named prompt
set, lifetime peak footprint beside it, the store at about 3.2 bits per expert weight. FAST rows are measured under
two prompt orders.

The cache size of the frozen configurations is not fixed yet. FAST is measured at C 224, and at C 256 only if that
gains at least 3 % with a lifetime peak of at most 45.0 GB (the engine's owner, 6 Nov). The C of the frozen EXACT
configuration is undecided.

| line | mode | HumanEval-type | MBPP | math | IFEval (chat) | lifetime peak | quality gate |
|---|---|---|---|---|---|---|---|
| 45 GB | EXACT (NAX6-N) | [[TBD-1a]] | [[TBD-1b]] | [[TBD-1c]] | [[TBD-1d]] | [[TBD-11]] | exact; reference certified (section 2) |
| 45 GB | FAST (the stack + RELAX B) | [[TBD-2a]] | [[TBD-2b]] | [[TBD-2c]] | runs EXACT | [[TBD-11]] | [[TBD-7]] |
| 12 GB | verified loop | 18.4-18.9 [K] ([[TBD-12a]]) | [[TBD-12b]] | [[TBD-12c]] | [[TBD-12d]] | 11.39 GB [K] | verified; not bit-identical to one-token decoding (section 2.8) |
| 12 GB | FAST (top-2 pass, theta 0.95) | [[TBD-13]] | [[TBD-13]] | - | - | [[TBD-11]] | [[TBD-13]] |

### 6.2 What is measured today (6 Nov, to be replaced)

| line | mode | workload | tok/s | status | source |
|---|---|---|---|---|---|
| 45 GB | EXACT | 13 of a mixed 14-prompt set, C 192 | 49.9 [K] | stands | LL, 'NATIVE-FRI run 2' |
| 45 GB | FAST (FAST-NAX all-in) | 12 HumanEval, C 224 | 113.9 [K] (passes 112.7 / 115.0) | gate pending | Q; BM:data/fnx/fnx_report.txt |
| 45 GB | FAST stack without CD16 | 12 HumanEval, C 224 | ~128-137 [P] | measured in tonight's run | Q, INV |
| 45 GB | FAST + RELAX B | HumanEval+ / math, C 160 | x1.49 / x1.32 against the exact control [K, paired] | decision 4 PASS | EV:RESULT_relax_2026-11-05 |
| 45 GB | FAST | 12 MBPP, C 224 | 71.3 [K] | no MBPP gate yet | INV |
| 12 GB | verified loop | 4 to 16 HumanEval prompts over four runs (section 3.5) | 18.4-18.9 [K] | stands, with the two caveats of section 2.8 | F12 |
| 12 GB | FAST (theta 0.95) | HumanEval | 35.9 [P, a calibrated replay] | not measured, not gated, not scheduled | T1265; the line's owner, 6 Nov |

Ratios against the exact control are not absolute speeds: that run used a smaller cache (C 160) than the release
setting.

### 6.3 Against published runs

Single-user decode of the same model, streamed from SSD, as collected on 4 Nov (RF; each row's source link is in
that file; self-reports are marked there). Not re-checked for this draft.

| who | expert precision | exact? | memory | hardware | tok/s |
|---|---|---|---|---|---|
| this work, 12 GB line | ~3.2 bits | verified; not bit-identical to its one-token decoding | 11.4 GB lifetime, whole process | M5 Pro, 64 GB machine | 18.4-18.9 (code prompts) |
| SlotStream | 4 bits | exact | 10.7 GB process target | Mac mini M2, 16 GB | 1.48 |
| Splash 1.4.0 | 2.4 bits | exact routing, verified drafting | 8.8 GB expert store; total unknown | MacBook Air M5, 24 GB | 28.8 |
| a Splash fork | 2.4 bits | no (cache-biased routing) | same | same | 48.8 |
| this work, 45 GB line | ~3.2 bits | exact to REF2-T / NAX6 | [[TBD-11]] | M5 Pro, 64 GB | [[TBD-1]] |
| SlotStream | 4 bits | exact | 33 GB ceiling | M5 Pro, 48 GB | 23.69 |
| Kapitsa | 2.4 bits | exact routing, verified drafting | 16 GB expert store; total unknown | M5 Pro, 48 GB | 61.3 |

- **The 12 GB comparison, with its caveats.** 18.4-18.9 against 1.48 tok/s is about 12x. The two runs differ in four
  ways. Exactness class: the published run is exact; ours is verified and deterministic, but not bit-identical to its
  one-token decoding (section 2.8). Precision: about 3.2 bits against 4. Hardware: M5 Pro against M2. Prompts: ours
  are code-heavy. Our 12 GB is also the engine's own footprint on a 64 GB machine, not an enforced limit. Faster
  published runs near 12 GB use 2.4-bit experts or lossy routing, or report only part of their memory (RF;
  F:12gb-exact-40tps). Whether this comparison is printed is the project owner's decision ([[TBD-18]]).
- **No clean first on speed.** At 2.4 bits, public engines run this model faster in less memory than our 45 GB line
  (RF, "Bottom line"). A fair claim needs the same machine, prompts and caps, a hard memory limit, and a quality
  comparison between the stores; none of these is done [[TBD-14: decide whether to run them or to state the gap]].

### 6.4 Stated limits

- The 12 GB line is verified, not exact in the sense of section 2.1 (section 2.8).
- Agent turns (tool calls, file edits) run EXACT. FAST fails their bar (section 3.7).
- Chat runs EXACT. The chat form of the relaxed rule passed alone and failed when combined with the 8-bit HC mixers
  that the shipped build turns on globally (section 7.6).
- Every code number names its workload. HumanEval-type prompts are flattered by prompt copying (section 5.4).
- FAST outputs can differ between runs at near-ties, because several FAST parts depend on the cache state and on
  the drafts (section 7.5). FAST quality is therefore certified under two prompt orders.
- The memory line is the lifetime peak of the process footprint, not the framework's own peak counter.

---

## 7. Negative results that teach something

### 7.1 Never waiting for the SSD: passes the quality screen, loses in the engine

- **Idea.** A missing top-2 expert is read in the background and its term is added one layer late, so the GPU never
  waits (the deferral family, here for the 45 GB FAST mode).
- **Quality screen: PASS.** Code flips 0.68 % against 0.74 % for its twin at a 208-slot pool (T50).
- **Engine: KILL of the first form.** -2.8 % (153.5 vs 157.9 tok/s in the same run), although kept tokens rose 4.6 %.
  The check grew from 68.8 to 75.3 ms (T209).
- **Why.** The form did not cut the wait: the gate service went from 14.0 to 15.6 ms per check, and about 6.5 ms of
  late work was added in the per-layer pipeline (T209c; the engine's owner, 6 Nov). Its own calls still waited on the
  loader's release for cold optional experts and cold sidecars (LL, 'NEVER-WAIT v3').
- **Why the price was wrong.** The prediction (+10-15 %) treated 13.5-15 ms of gate service per check as removable
  waiting. It is a spin bound that out-of-order execution already overlaps with compute.
- **Status.** A revised form that drops the cold optional reads is built and bit-checked. It needs its own quality
  arm before a speed test, so the family is not closed.
- **Note.** Both arms ran on a stack that still contained CD16 (section 7.3; confirmed by the engine's owner). The
  comparison is within one run.

### 7.2 Fewer expert groups on sure rows ("split"): the tile price was 2-3x too high

- **Idea.** On rows the drafter is sure of, compute only ranks 0-1 plus experts the check loads anyway.
- **Quality screen.** PASS on code (0.92 % and 1.31 % flips at certainty 0.99 and 0.9, line 1.34 %), KILL on chat
  (T553, T552).
- **Engine.** Computed expert groups fell to x0.63 and x0.49, yet check + commit fell only 6 and 5 %, and kept tokens
  fell 4.2 and 1.7 %: net about 0 to +1.5 % against a priced +11-19 % (T610; F:flat2). On the matrix-unit kernel:
  -1.3 %, with kept tokens -5.7 % (T744).
- **Lesson.** On this GPU the expert time is mostly per-row work, not per-group reads (11.1 ms at 2 rows, 57.0 at 16,
  all resident). Counting tiles, groups or bytes overpredicted the gain two to three times. A lossy cut also lowers
  acceptance, and that loss must be in the price.

### 7.3 Wholesale acceptance of comments (CD16), and a withdrawn 157.9 tok/s

- The stack measured 157.9 tok/s on 12 HumanEval prompts (passes 157.2 / 158.1) and was written up as a headline.
  It included CD16. CD16 then failed quality on MBPP (section 4.3), and the number was withdrawn before publication
  (F:45gb-fast-stack, WITHDRAWN).
- **Lesson.** A lever must be gated on a workload that can expose it. HumanEval could not: its docstrings are copies
  of the prompt, so accepting them unchecked changes nothing. Guarded forms are being tested [[TBD-15]].

### 7.4 The 12 GB FAST mode failed same-scores

- **The mode.** A cheap first pass with only the top-2 experts certifies a token when its top probability is at
  least theta; other rows are recomputed with the full computation.
- **Result.** At theta 0.8 on code and 0.95 on chat: HumanEval+ 80.0 vs 91.7 (0 / 7, p 0.02) and IFEval 84.0 vs 90.0
  FAIL; math passes; memory passes (at most 10.84 GB) (EV:RESULT_t0_2026-11-05). Truncations and loops rose from
  13 / 2 to 23 / 7.
- **What it teaches.**
  - The decision map had predicted it: the same setting flips 1.63 % of code rows, over the 1.34 % line (T54).
  - The certificate itself is token-accurate (0-1 divergences in about 3,400 tokens at every theta). The damage comes
    through the context: the cheap pass also writes the attention and recurrent state that later tokens read (T1265).
    A certificate on the token says nothing about the state it leaves behind.
- **Status.** A code-valid point (theta 0.95, map 1.22 %) is modelled at 35.9 tok/s [P, a calibrated replay]. It is
  neither measured nor gated, and no run is scheduled ([[TBD-13]]).

### 7.5 A withdrawn "12 GB exact at 40.8 tok/s"

- A public note claimed 40.8 tok/s, bit-exact, in 11.35 GB. A re-check found that the measured arm was a FAST mode:
  it commits rows certified by the cheap pass without the full check. It matched the verified loop on 5 of 16
  prompts, and the quality screen scores its certificate (top-2 experts, theta 0.8) a code KILL at 1.63 %. The
  verified loop runs 18.4-18.9 tok/s. The note now carries a dated correction (F:12gb-exact-40tps; F12).
- The same week showed that the verified loop itself is not bit-identical to one-token decoding (section 2.8). The
  word "exact" had been used for three different properties: bit-identical to one-token decoding, verified by the
  full model, and reproducible.
- **Lesson.** An arm's name is not its definition. The exactness check had been run on 4 of the 16 prompts. An audit
  over every prompt, as in section 2.5, would likely have caught it before the note went out [G].

### 7.6 Agent turns, and two lossy parts that compound on chat

- **Agent turns.** Plain FAST wrote 85 low-probability tokens in 19,749 on 17 long agent turns (about 43 per
  10,000); 75 of them are inside the written file content. Short edits are clean; long file writes carry the failure
  (EV:RESULT_agentfast_2026-11-06).
- **Cause.** With QOM-KEEP off, blunders fall from 21 to 4 (forward order) and 18 to 0 (reverse): the cache-dependent
  keep carries them. Other parts do not move the count (EV:RESULT_agentbisect_2026-11-06).
- **Run-to-run dependence is wider than one lever.** With QOM-KEEP off, the two prompt orders still differ within
  the first tokens. The check logs show the drafted row counts differing about 27 tokens before the committed tokens
  do, so the drafts carry history. If any FAST kernel's bits depend on a check's row count, even exact acceptance
  then depends on the drafts (same source, addendum). The exact base must be identical in both orders
  [[TBD-16: the result of that control]].
- **Chat.** Relaxed acceptance alone scores IFEval 87, 8-bit HC mixers alone 89, both together 84 against 90 for the
  reference: a failure neither shows alone (EV:RESULT_relax_drift_2026-11-06, section 3).

### 7.7 A data-aware 2.4-bit trellis store did 1.6-2.5x the damage its error predicted

- Priced from its weight-error ratio against 3-bit truncation, the store measured much worse on the decision map:
  1.45 / 9.54 / 10.44 % flips against lines of 1.34 / 5.42 / 3.69 (F:trellis_mse_miss).
- Two independent derivations agree that these errors cost about 1.7-3.5x more KL divergence per unit of
  expert-output error than truncation errors do. The mechanism is open.
- **Lesson.** For this model, weight or output error cannot rank a data-aware code against plain truncation without
  a divergence screen on real prompts.

### 7.8 A fixed cost that did not fall, and a timing method that was wrong

- The fixed cost per check stayed at 54 ms with every FAST lever on (section 5.4). 350 tok/s on code at 12 kept
  tokens would need about 25 ms [P, F:flat2].
- An earlier fit had put a dependent dispatch at 12-20 us, which pointed at fewer, fused kernels. The direct
  measurement is 1.6 us (section 5.1); the loss was in narrow kernels. Two rebuilt kernels from that programme (a
  row-parallel GDN step and a replacement HC kernel) were bit-correct and slower than what they replaced (LL,
  'GDN-CHUNK', 'HC-K2C').
- A split that was thought live was inert for a night because another fast path bypassed it (F:flat2, caveat). The
  engaged-path counters of section 2.7 exist for this reason.

### 7.9 From the ledger: ideas an outside reader would likely propose

One line each; numbers and sources in `~/TRIED/TRIED_SHORT.md` and the full ledger.

- **Look-ahead expert prefetch:** pays at 24 GB (+9-10 %), KILL at 45 GB (wrong guesses evict).
- **mmap or the page cache instead of direct reads:** KILL (mmap about 5x slower).
- **Cache-aware routing or substituting resident experts:** KILL; not lossless, and small. Expert independence
  (section 5.6) explains why.
- **Fewer experts per token (top-8):** KILL at the strict gate, with or without renormalisation.
- **All experts at 3 bits; any 2-bit tier:** KILL.
- **Draft trees:** KILL; the best tree is -10 to +2 % against the best chain at equal cost.
- **Wider (32-row) checks:** KILL at today's per-row cost (-1.3 %); they pay only below about 0.9 ms per row (KP).
- **Self-speculation, layer skip, early exit:** KILL; exiting after layer 40 agrees with the full model on 5.7 % of
  code positions.
- **Exact speculative sampling instead of greedy matching:** KILL; acceptance +0.003 to +0.014.
- **Drafter retrains:** on-policy data +0.13 kept per doubling; 2x capacity +0.18; one reseed alone moves code by
  0.16. Only the data slope is above noise (KP).
- **Reusing the previous position's MoE output:** the one fully new idea; KILL on bytes (union only x0.91) (NOV).
- **Matrix units for experts at short checks:** KILL at the 1 Nov operating point (+1.8 %); adopted later, once
  checks were longer and the records were in the kernel's native layout (T675, T527).
- **Compiling the verify graph:** KILL (-0.7 ms; the graph build was not the cost then).
- **Several streams in one pass:** aggregate throughput only; per stream it is slower than solo (F:union-law).

---

## 8. Prior art and references

Citations are copied from the sessions' records (NOV "Sources", PA, RF, KP). None was re-opened for this draft:
identifiers, venues and claims must be checked before release [[TBD-17]]. Items are grouped by what we took from
them.

**Batch invariance and determinism.** Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Sep 2025;
batch-invariant RMSNorm / matmul / attention, fixed split sizes); vLLM and SGLang deterministic modes (CUDA only);
LLM-42 (arXiv 2601.17768). Our port and the three MLX-specific breaks are section 2. Our early notes did not cite
this work; they should have (PA, row 1).

**SSD-streamed MoE inference.** Memory-Sovereign Inference (arXiv 2608.23805): output-exact NVMe streaming of an 80B
model with byte-identical logits. SlotStream, anemll-flash-mlx (slot banks, expert ids as data), Flash-MoE, Kapitsa,
Splash (public engines; RF, PA). Apple's "LLM in a flash" (2023). KTransformers (SOSP '25): Expert Deferral.

**Expert caching and prefetch.** Pre-gated MoE (2023); ProMoE (2410.22134); MoE-Infinity (2401.14361); AdapMoE
(2408.10284); HOBBIT (2411.01433); FlashMoE (2601.17063); SwapMoE (2308.15030); MoE-SpeQ (2511.14102); SP-MoE
(2510.10302); Fiddler.

**Mixed precision for experts.** HOBBIT; MC-MoE (2410.06270); token-level precision restoration (2512.17073); DynaExq
(2511.15015); DyMoE (2603.19172); SliceMoE (2512.12990); Matryoshka Quantization (2502.06786); QTIP-style trellis
codes.

**Fewer or shared experts under speculation.** OEA, Opportunistic Expert Activation (2511.02237); PSR,
Prefetch-Steered Routing (2610.34721); AcceptMoE (2608.02989); MoE-Spec (2602.16052); EVICT (2605.00342); DraftExpert
(2607.24434); EcoSpec (2607.12696); Utility-Driven SD for MoE (2506.20675); Lynx (2411.08982); Mixture of
Cache-Conditional Experts (2412.00099).

**Drafters and verification.** Leviathan et al. 2023 (speculative decoding, the cost ratio); DFlash (2602.06036) and
its checkpoint for this model family; DFlare (2606.02091); EAGLE-3 (2503.01840); PARD-2 (2605.08632); WhiFlash
(2606.07710, drafter by task); TriSpec (2601.23180); SAM-Decoding (2411.10666); prompt-lookup decoding.

**Relaxed and lossy verification.** MARS (2601.15498); DIVERSED (2604.07622); "Revisiting Lossy Verification"
(2607.26627); AdaptiveSpec (2609.02897); the answer-lengthening effect of relaxed rules (2610.23818).

**Kernels.** MonoMoE (2609.04244, padding waste in decode MoE matmuls); ONNX Runtime's MoE small-row path; QuaRot /
SpinQuant (folding a norm weight into the next linear).

**Storage.** NVMe autonomous power state transitions and published wake-up tests; anticipatory device wake (US patent
11789658; "Software Engagement with Sleeping CPUs", HotOS '15); direct I/O into device-visible buffers (GPUDirect
Storage, BaM).

**Benchmarks.** HumanEval+ and MBPP (via EvalPlus), IFEval, GSM8K [[TBD-17: standard citations]].

---

## Appendix A. Placeholders and open confirmations

| id | what is missing | who has it | expected source |
|---|---|---|---|
| TBD-1 (a-d) | 45 GB EXACT quiet tok/s on HumanEval, MBPP, math, IFEval at the release C, certified kernel | BigMoE | the freeze runs (Mon / Tue night per INV section 4) |
| TBD-2 (a-c) | 45 GB FAST tok/s per workload, two prompt orders | BigMoE | tonight's palindrome (`data/` report), then the freeze runs |
| TBD-3 | the 12 GB line made exact: one-shape kernels, the replay fix, a same-scores run. Not planned before the freeze, so both caveats are printed | Fit12 | - |
| TBD-4 | one figure for active parameters per token (~5B vs ~6B in the records) | BigMoE | config arithmetic |
| TBD-5 | (resolved in v1: RELAX B's rule list, from its pre-registration) | - | - |
| TBD-6 | (resolved in v1.1: the map's anchor arm, section 3.7) | - | - |
| TBD-7 | the FAST stack's same-scores verdict under two orders, with its bars | EvalSuite | `docs/RESULT_stack_*` (pre-registered 6 Nov) |
| TBD-8 | (resolved in v1.1: the novelty record was refreshed on 6 Nov) | - | - |
| TBD-9 | (resolved in v1.1: no isolated keepalive A/B exists; the text says so) | - | - |
| TBD-10 | the second comment-rule version's engine result | MoonShots / BigMoE | tonight's run |
| TBD-11 | lifetime peak footprint of every released config | BigMoE / Fit12 | the night's report lists every arm's lifetime peak; then the freeze runs |
| TBD-12 (a-d) | the 12 GB verified loop: (a) one quiet run alone on all 16 HumanEval prompts with its own lifetime peak, about 5 minutes of GPU; (b-d) MBPP, math, chat | Fit12 | unscheduled |
| TBD-13 | the 12 GB FAST point at theta 0.95: measured speed and same-scores. Unscheduled; the map arms of the union form (theta 0.8 / 0.9 / 0.95) run on the night of 6 Nov | Fit12 / Breakthrough / EvalSuite | `~/Breakthrough/SCOREBOARD.md` |
| TBD-14 | same-machine comparison with the 2.4-bit engines, a hard memory limit, store-quality comparison: run, or state the gap | project owner | needs downloads: an owner decision |
| TBD-15 | guarded comment acceptance (G16 / D16) on MBPP: quality, then speed | BigMoE | tonight's run |
| TBD-16 | the exact base in two prompt orders (must be identical) | BigMoE / EvalSuite | - |
| TBD-17 | every citation re-opened; benchmark citations added; Apple's wording for the GPU's Neural Accelerators and Metal 4's tensor operations checked | NewIdeas | - |
| TBD-18 | whether the "about 12x" comparison at 12 GB is printed, given that the 12 GB line is verified and not exact | project owner | the orchestrator is asking |

Owner confirmations, asked and answered on Fri 6 Nov (21:2x-21:3x):

| question | owner | answer |
|---|---|---|
| Is the 12 GB line "exact to its own reference"? | Fit12 | No as worded. It is verified greedy decoding in 2-4-row calls with all ten ranks waited for; deterministic; not bit-identical to its one-token decoding; near-tie tokens depend on the draft pattern. The replay fix is not planned before the freeze. Sections 2.8 and 3.5 follow the owner's wording |
| Which 12 GB numbers to quote? | Fit12 | The 14-prompt figures, with the labels of section 3.5. The FAST point at theta 0.95 is 35.9 [P], unscheduled |
| Is 49.9 tok/s at C 192 the standing 45 GB EXACT row? | Leap | Yes, until the freeze runs |
| The release C | BigMoE | FAST at C 224, or C 256 if it gains at least 3 % with a lifetime peak of at most 45.0 GB. The EXACT C is undecided |
| Is U3 live in the stack? | BigMoE, NewIdeas | Yes, as the 3-bit part of the QOM switch at certainty threshold 0, on the code sets. "Sure" means every drafted row |
| Is an exact form of the K split planned? | Leap | No. A bitwise alternative is proposed and not built |
| A public name for the matrix units | Leap | Apple's term: a Neural Accelerator in each GPU core, programmed through Metal 4's Metal Performance Primitives. To be checked against Apple's pages |
| Did the never-wait A/B run on a stack with CD16? | BigMoE | Yes, both arms, in one run. The late work is in the per-layer pipeline |
| Why does that arm anchor the map's lines? | NewIdeas | It was the first gate approved for the end-to-end test (section 3.7) |
| The novelty counts | NewIdeas | 309 tags; none "new" in a release configuration (section 4) |

Still the project owner's decisions, raised by the orchestrator: whether any "first" is claimed (TBD-14), and whether
the 12 GB "about 12x" comparison is printed (TBD-18).

Where the weekend's result files land (the orchestrator, Fri 6 Nov 21:2x):

| what | where |
|---|---|
| the stacked palindrome (TBD-2, TBD-10, TBD-15) | `~/BigMoE/data/sat/` (the night script's report); the chosen code profile in `~/BigMoE/data/stack/` |
| the stack's gate, the long-code drift top-up, MBPP coverage (TBD-7) | `~/EvalSuite/results/` (newest `*report*.txt`) and `~/EvalSuite/docs/RESULT_*.md` |
| the GPU-idle check and Saturday's kernel A/Bs (section 5) | `~/Leap/TABLE_350.md`, newest sections |
| the map arms | `~/Breakthrough/SCOREBOARD.md` |
| the plan and the verdict summary | `~/NewIdeas/QUEUE.md` (the "300" line and the GPU order) |

## Appendix B. Suggested figures

1. The check loop (draft, cut, verify, commit) with the two memory lines. Schematic.
2. Where rows first diverge, per layer, stock kernels against REF2 (data: `~/Leap/data/inv0_screen.txt`,
   `inv1_screen.txt`).
3. A check's time by piece at 2 and 16 rows (data: `~/Leap/data/flat2/`, `data/stepb/`).
4. GB/s against tiles per kernel, before and after the K split (data: `~/Leap/data/dnks_bench.txt`).
5. SSD first-read penalty against idle gap, with and without the keepalive (data: `~/MoonShots/data/ssdmicro*.txt`).
6. Tokens kept per check by workload, and the copied share of HumanEval answers (data: KP, KW, L350).

## Appendix C. How the results were produced

- Several sessions worked in parallel on one machine, each owning a part (the engine, kernels, the drafter, the
  evaluation suite, SSD measurements, the 12 GB line, idea screening). GPU time was serialised by a lock; timings that
  are compared were taken on a quiet machine.
- Every test has PASS and KILL lines written before it runs; verdicts come from results only. The evaluation suite's
  pre-registrations and results are dated files (`~/EvalSuite/docs/PREREG_*`, `RESULT_*`).
- Every idea, with its verdict and deciding number, is one line in a ledger: 1,345 merged ideas on 6 Nov, of which
  142 adopted, 267 passed, 608 killed [P: sums of the section tallies in `~/TRIED/TRIED.md`]. The ledger and the
  novelty record are the release's appendix (INV section 3).

## Appendix D. Ledger keys

A key T*n* is a stable label for one entry of the ledger `~/TRIED/TRIED.md`; *n* was the entry's line number in the
merge of Fri 6 Nov ~21:00. To find an entry, search the ledger for its ID. The last column is the entry's line when
this table was generated (Fri 6 Nov, after the 21:2x re-merge); `check_sources.py --keys` regenerates the table.

| key | ledger entry ID | line |
|---|---|---|
| T7 | Concurrent miss loads | 7 |
| T10 | Item 5 | 10 |
| T12 | M-17 LAYSZ45 | 12 |
| T14 | OOO + TIER_COMBINE | 14 |
| T18 | R-69b OOO-LIVE | 18 |
| T29 | B12LL | 29 |
| T44 | M-28 SSDMICRO keepalive spacing (4 KB pread + usleep(n), BigMoE's loop | 44 |
| T47 | M-29 SSDMICRO2 wake-ahead ping | 47 |
| T50 | NW-BUILD (Fable FAR; the build of NEVERWAIT_TEST.md v2; adapted, RE-OP | 50 |
| T54 | PLAN-12GB-90 (a plan, not an idea; the orchestrator Thu 19:4x, with Fi | 54 |
| T76 | W1-PICKUP | 76 |
| T93 | M-20 SSDWAKE | 93 |
| T97 | M-28 SSDMICRO per-call cost decomposition + JOINT in micro | 97 |
| T98 | M-29 follow-up | 98 |
| T103 | Q1U-T3 | 103 |
| T106 | QOM-T3 | 106 |
| T199 | M-28 SSDMICRO pieces | 199 |
| T201 | M-28 SSDMICRO page cache instead of F_NOCACHE for demand records | 201 |
| T202 | M-31 HEADPING | 202 |
| T209 | NW-ENGINE | 209 |
| T209c | NW-ENGINE | 210 |
| T288 | FAR-WAIT (Fable FAR, recorded by MoonShots) | 289 |
| T307 | REF2-T (R-58) | 308 |
| T309 | t 0.3 truncation | 310 |
| T312 | 5b U3C len | 313 |
| T334 | P11 | 335 |
| T462 | U-32 | 463 |
| T476 | GATE16 sync-free 16-row gate | 477 |
| T478 | L1-23 batch-invariant 16-row dense kernels | 479 |
| T479 | L1-25 experts_nax | 480 |
| T489 | R-55 GATE16 | 490 |
| T490 | R-56d | 491 |
| T495 | Spin gate | 496 |
| T527 | N7-13 NAX6-N | 528 |
| T552 | SPLIT-ARMS-CHAT | 553 |
| T553 | SPLIT-ARMS-CODE | 554 |
| T565 | Item 2 | 566 |
| T610 | SPLIT-ENGINE-OFF | 611 |
| T675 | K-62j NAX6 | 676 |
| T744 | SPLIT-NAX | 745 |
| T782 | D-23d CHATSWITCH | 783 |
| T855 | CD16-DROP | 856 |
| T993 | LIVE-CUT AUDIT | 994 |
| T1011 | CF03-EQ-CF0 | 1012 |
| T1049 | WALL-TIEFLOOR | 1050 |
| T1057 | Fold-fast with the memory fix (b12llff3, decision 5) | 1058 |
| T1088 | PURGE-1 | 1089 |
| T1195 | Batch-invariant reframing | 1196 |
| T1208 | R-40 STEP 4b / 4c -> REF2b key buckets | 1209 |
| T1209 | R-40 INV1 | 1210 |
| T1210 | R-40 REF2 | 1211 |
| T1214 | R-58 REF2-T | 1215 |
| T1218 | REF2 sdpa_classes | 1219 |
| T1219 | REF2b | 1220 |
| T1225 | MMA-tile exactness check (Leap's claim) | 1226 |
| T1244 | Chunked (WY / UT) GDN verify | 1245 |
| T1259 | R-56 REF2 live loop | 1260 |
| T1265 | 12 GB lever 1 | 1266 |
| T1350 | COMMENT-LOOPHOLE (Fable mission 25, recorded by MoonShots) | 1351 |
