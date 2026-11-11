# GOAL.md — where the missing 2–3x hides

Independent read of MISSION_GOAL.md only (no other folder read). Marks: **[K]** = measured, from the brief; **[P]** = projected
by arithmetic from [K]; **[G]** = guess / assumption I could not derive.

Written 2026-11-04 by the `goal` session.

---

## 0. The arithmetic everything below uses

**One expert.** 3 matrices × 2560 × 640 = 4,915,200 weights. At ~3.75 bpw that is **2.30 MB** [P] (brief: "~2–3 MB" [K]).
At 3.0 bpw 1.84 MB, at 2.8 bpw 1.72 MB, at 2.0 bpw 1.23 MB. All experts: 512 × 48 = 24,576 experts × 2.30 MB = **56.5 GB** [P]
— so even the 45 GB line cannot hold them all (≈ 42 GB is available for experts after dense ≈ 1.3 GB, drafter, KV/GDN state
and headroom [G]) → residency ≈ 75 % and the SSD is in play at 45 GB.

**One check (16 rows).** 16 rows × 10 experts = 160 (row, expert) pairs per layer. "~2 of the 16 rows are real per expert" [K]
→ **~80 unique experts per layer**, **3,840 expert tiles per check**, **8.8 GB of expert bytes per check** [P].
At 82 % of 286 GB/s (FAST kernel [K]) that is **37.7 ms**; at 77 % (EXACT) **40.1 ms** [P].
Dense part ≈ 2.6 B params (36 GDN layers ≈ 1.5 B, 12 attention ≈ 0.2 B, shared experts 0.24 B, lm_head 0.64 B) → ≈ 1.3 GB
at 4-bit → 6 ms at 74 % of the byte floor [P]. That leaves ≈ 6 ms of the brief's "~50 ms compute" [K] for small kernels,
router/gather/scatter, GDN scans and per-layer CPU↔GPU syncs [P].

**FAST check today (code, 45 GB), ms** [K unless marked]:

| item | ms | share | floor | what sets it |
|---|---|---|---|---|
| drafter | 5.4 | 6 % | ~2 [P] | 16 sequential steps; lm_head over 248k vocab [G] |
| **SSD wait** | **24** (16–60) | **27 %** | **0** | 69 misses/check (= 15 MB/kept × 10.5 = 158 MB) [P]; 12.6 ms at 12.5 GB/s if streamed, so it is **latency-bound**: ~1.4 misses per layer, each a per-layer round trip |
| **expert tiles** | **~38** [P] | **42 %** | 31 at 100 % BW | 3,840 tiles; the kernel is at the compute/byte knee because the matrix unit does 16 rows per tile for 2 real rows [K] |
| dense | ~6 [P] | 7 % | 4.5 | 74 % of byte floor [K] |
| small kernels / syncs | ~6 [P] | 7 % | ~2 | ~700 dispatches, 48 CPU syncs [G] |
| commit | 5 | 6 % | 0 (hideable) | serial after verify [K] |
| **total** | **≈ 85–90** | | | 90 / 10.5 kept = 8.6 ms per kept token → 117 tok/s (brief: ~110) ✓ |

**EXACT check today**: 83 tok/s → 126 ms per check at 10.5 kept [P]. Same compute (+2.4 ms from the 77 % kernel), same
drafter and commit → the extra ≈ 36 ms is **SSD wait ≈ 55–60 ms = ~45 % of the EXACT check** [P] (EXACT loads every miss,
FAST does not; and/or EXACT keeps a larger copy, so lower residency). Whether that wait is latency-bound (as in FAST) or
bandwidth-bound (≈ 350 misses × 2.3 MB = 0.8 GB → 64 ms at 12.5 GB/s, which is suspiciously close) the brief cannot tell;
the oracle-replay test in Lever 1 settles it in 10 minutes.

**The budgets.**
- 300 tok/s FAST = **3.33 ms per kept token, all-in** = 35 ms per check at 10.5 kept, 39 at 11.7, 41 at 12.4, 43 at 12.9.
- 200 tok/s EXACT = 5.0 ms per kept token = 52 ms per check at 10.5 kept, 60 at 12.

**Five structural facts that follow:**

1. **Expert streaming alone is 3.6 ms per kept token — already more than the whole 3.33 ms FAST budget.** 300 FAST is
   impossible without cutting *expert tiles per kept token*; no amount of SSD, drafter or commit work gets there alone.
2. **At 45 GB, bits do not buy speed.** The kernel sits at the compute/byte knee (16-row tiles, 2 real rows) [K]. Halving
   bytes (2-bit) leaves the same 3,840 tiles → ≈ the same ~38 ms [P]. Bits buy *SSD bytes* (24/12 GB lines) and *residency*
   (≤ 2.8 bpw → 42 GB → the 45 GB line becomes SSD-free). Only **fewer tiles** cuts the 38 ms. (5-minute check of this
   inference: time the kernel on one expert set at 4-bit vs 3-bit, and at 2 vs 16 real rows. If time tracks bytes, I am
   wrong and a 2.8-bpw body is worth ~25 % directly; if it tracks tiles, everything below stands.)
3. **The SSD wait is a latency problem, not a bandwidth problem (FAST).** 158 MB/check would stream in 12.6 ms and hide
   entirely under 50 ms of compute; it costs 24 ms because each layer waits for its own 1–2 reads. Nothing about the SSD
   needs to change; the *issue time* does.
4. **Kept-per-check is capped at ~12–13 by every known means** (perfect pick 12.4, relaxed 11.7–12.9, 2–4B drafter 12–12.5
   [K]) → at most +15–25 % on code. **It is not where the 2–3x is on code.** On chat (3 kept) it is the whole game.
5. **~40 % of rows are wasted on code, ~85 % on chat.** Kept 10.5 = 9.5 accepted drafts + 1 bonus → 6.5 of 16 rows sit after
   the rejection point and their experts are loaded for nothing; with chat's ~3 kept, 14 of 16 rows are wasted. Rows are
   free inside a tile [K]; the *experts those rows pull in* are not.

---

## 1. Where the 2–3x hides — ranked by size (per check, code, 45 GB)

| rank | bucket | FAST ms (share) | EXACT ms (share) | removable | line |
|---|---|---|---|---|---|
| 1 | **SSD wait on the critical path** | 24 (27 %) | ~58 (45 %) | **all of it** | EXACT-safe |
| 2 | **expert tiles** (unique experts × 48) | 38 (42 %) | 40 (32 %) | 30–45 % by confidence-tiered top-k (FAST); 10–15 % by adaptive draft length (EXACT); **0 % by bits** | FAST / EXACT |
| 3 | **serial tail**: drafter 5.4 + commit 5 + syncs/small kernels ~6 | ~16 (18 %) | ~16 (13 %) | ~60 % (commit ∥ drafter, one command buffer, fused small kernels, FR-Spec vocab for the drafter) | EXACT-safe |
| 4 | kept per check | — | — | +10–20 % (relaxed acceptance FAST; better drafter EXACT) | both |
| 5 | dense | 6 (7 %) | 6 (5 %) | ~0 | — |

**Stacks** [P]:

*FAST 45 GB, code:* 90 → L1 (SSD 24→4, syncs 6→4) **62** → L2 (tiles −35 %: 38→24) **48** → L3 (commit hidden, drafter
3 with FR-Spec, tiles −10 %) **≈ 38–40 ms** at ~10.2 kept → **≈ 260 tok/s**; + relaxed acceptance 1.2 % (11.5 kept) →
**≈ 290–300**; 2.4 % (12.6) → ≈ 320. **300 FAST = L1 + L2 + L3 + relaxed acceptance.** No single lever does it.

*EXACT 45 GB, code:* 126 → L1 (fit-in-RAM: SSD→0; or prefetch: →~10) **68–78** → L3 (−5 commit, −3 syncs, −4 tiles) **56–66** →
kernel 77→82 % (−2.5) **54–64** → **155–195 tok/s** at 10.3 kept; a 12-kept drafter → **180–225**. **EXACT 200 = L1 as
"fit" (or near-perfect prefetch) + L3 + a better drafter.** L1 is half of the whole gap.

*24 GB SSD bytes per kept token (FAST):* 19.5 [K] → adaptive draft length (−25 %: the wasted unsure rows' misses) **14.6**
→ 2-bit copy (×0.53) **≈ 7.7 MB** → goal met. Without the 2-bit copy the goal is missed (14.6 > 10). Tiering does *not*
help this line (resident experts cost no SSD bytes, and sure rows already skip their misses [K]).

*12 GB, code:* 41 tok/s EXACT [K] ≈ 290 ms/check ≈ 3.5 GB of SSD per check [P], SSD-bandwidth-bound. Target 120–150 tok/s
needs ≤ 0.8 GB per check fully overlapped: 2-bit (×0.53) × adaptive length (×0.75) × tiering (×0.5) → ≈ 0.7 GB →
≈ 55–60 ms/check if overlapped (L1 machinery) → ≈ 170 theoretical, **≈ 120 realistic** [P]. All FAST. The hardest line;
it needs every lever.

*Chat / mixed:* kept ~3 with 16 rows. Adaptive length alone (rows 16→~5, U 80→~40) ≈ ×1.6–2 on chat bytes and time
[P]; an EAGLE-3-style feature-conditioned drafter is the only known route from 3 to ~5 kept.

---

## 2. Top 3 levers

### Lever 1 — Take the SSD off the critical path at 45 GB  (EXACT-safe)

**Mechanism.** Reads are issued one layer at a time, so 1–2 misses per layer each cost a round trip. Move the issue time
to *before* the check:

- **A. Cross-check prefetch from the drafter.** The drafter's 16 tokens and hidden states exist 5–50 ms before the 125B
  needs the experts. Apply the 125B's own 48 routers (each 2560×512, ~1.3 M weights — negligible) to a learned linear
  alignment of the drafter's per-token state (SP-MoE's construction, 89 % top-1 expert accuracy from draft states), or a
  distilled 48-way head; take the union over the 16 rows (a missed expert needed by 2 rows gets 2 chances); issue every
  non-resident predicted expert as one QD-16 batch of `pread`s at draft time. Over-fetch by 20 % (top-12) — bandwidth is
  not the constraint (158 MB in a 50 ms window is 3 GB/s).
- **B. In-check stride prefetch for the residue.** At layer *l*, apply router *l+2* to *h_l* (ProMoE); the residual stream
  changes slowly, so recall on top-10 is high; 2 layers ≈ 2 ms ≥ one 2.3 MB read at QD ≥ 2.
- **C. One command buffer per check.** With reads issued ahead, the GPU no longer returns to the CPU per layer: wait on
  `MTLSharedEvent`s signalled by I/O completion inside the buffer. This also removes most of the ~6 ms of per-layer sync.
- **Permanent variant: fit.** An expert body at ≤ 2.8 bpw (1.72 MB) is 42 GB → all experts resident at 45 GB → SSD wait 0,
  no machinery, and the EXACT line is *re-based* on that model (it must pass the FAST bar once; after that EXACT is
  bit-identical to its own token-by-token decode). Same tooling as the planned 2-bit trellis copy [K]
  ([QTIP, NeurIPS 2024, arXiv 2406.11235](https://arxiv.org/abs/2406.11235): trellis-coded 2–3 bpw with compute-based
  decode, no lookup table). If Appendix check 1 says bytes (not tiles) set the kernel time, a *lossless* entropy code of
  the existing codes is the exact-safe ~15 % byte cut ([DFloat11, arXiv 2504.11651](https://arxiv.org/abs/2504.11651) is
  the GPU-decoder pattern); if tiles set it, skip that.

**Expected gain** [P]. FAST: 90 → 62–66 ms/check (**+36–45 %**) and the 16–60 ms variance disappears. EXACT: 126 →
68–80 ms (**+60–85 %**) if latency-bound or fit; if EXACT's wait is bandwidth-bound, prefetch only overlaps it with compute
(126 → ~75) and **fit is the real fix**. Largest, surest item on the board; nothing numerical changes.

**Line.** EXACT (and FAST). The HOBBIT variant — use the 2-bit copy of a missed expert instead of waiting — is FAST.

**Prior art.** Published, close: [SP-MoE, arXiv 2510.10302](https://arxiv.org/abs/2510.10302) (draft-model states → target
routers → prefetch, 89 % top-1), [MoE-SpeQ, arXiv 2511.14102](https://arxiv.org/abs/2511.14102) (on-device draft model
predicts future experts; host→device prefetch), [DraftExpert, arXiv 2607.24434](https://arxiv.org/abs/2607.24434)
(router-agreement distillation for prefetch hit rate), [ProMoE, arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
(stride prediction, llama.cpp integration, 1.34× decode), [Pre-gated MoE, ISCA'24, arXiv 2308.12066](https://arxiv.org/abs/2308.12066),
[AdapMoE, arXiv 2408.10284](https://arxiv.org/abs/2408.10284) (later-layer gates as prefetch signal),
[HOBBIT, arXiv 2411.01433](https://arxiv.org/abs/2411.01433) (low-precision stand-in on a miss). New here only in the
combination with a 16-row speculative check on one Mac, where the prediction window is a whole check.

**Cheapest test (≤ 10 min).** *Oracle replay:* log the routing of 20 checks; re-run them issuing every check's SSD reads at
t = 0 (QD 16, explicit `pread`, no page-cache faults). Measure SSD wait and compute.
- **PASS:** wait ≤ 5 ms/check, compute unchanged → the issue order is the whole problem; build A+B+C.
- **KILL:** wait > 12 ms with oracle issue → the read path itself (page faults, per-read latency, QD 1) is the problem or the
  wait is bandwidth-bound; fix the path / go to "fit" before any predictor.
- *Second 10-min test, CPU only:* on 2k logged tokens (drafter states + 125B top-10 per layer), fit the linear alignment;
  **PASS** ≥ 70 % recall on the experts that actually *missed* (not on all experts), **KILL** < 40 %.

### Lever 2 — Confidence-tiered top-k: cut tiles, not bytes  (FAST)

**Mechanism.** The 38 ms wall is 3,840 tiles, and the kernel cannot be made cheaper per tile [K]. Rows are free inside a
tile; experts are not. So reduce the *experts the rows pull in*: a **sure** row (drafter top-1 p ≥ τ, ~75–85 % of code
rows [G]) runs with k = 4 (gate renormalised over the top-4) and the shared expert; an **unsure** row keeps k = 10.
Selections per layer: 16 × (0.8 × 4 + 0.2 × 10) = 83 vs 160 → unique experts ≈ 45–55 vs 80 [P] → **tiles −35–45 %**.
The verify decision on sure rows is unaffected in practice (they accept at ~99 %); what degrades is the *state* they leave
for later rows and the KV/GDN state — the published evidence says greedy outputs tolerate this: half-k keeps greedy
accuracy on fine-grained MoEs (Expert-Sample), upper layers tolerate fewer experts (HookMoE), and plain static-k reduction
is itself a strong baseline ("Don't Skip the Baseline"). Tiering is strictly gentler than static-k because every unsure row
keeps k = 10. The brief's own 24 GB rule — skip sure rows' *missing* experts and keep benchmark scores [K] — is the
special case; the new step is to skip *resident* ones too, because at 45 GB the wall is tiles, not bytes.
Optional refinement: k_sure per layer (4 in upper layers, 6 in lower), since PCoMoE shows the damage is uneven by layer.

**Expected gain** [P]. Experts 38 → 22–25 ms/check. Alone: 90 → 75 (+20 %). On top of L1 + L3: **48 → 38–40 ms/check →
≈ 260 tok/s at 10.2 kept; ≈ 300 with relaxed acceptance 1.2 %.** This is the lever that makes 300 FAST arithmetically
reachable; without it the FAST floor is ≈ 48 ms/check ≈ 220–240 tok/s. It also cuts SSD misses at 12 GB (fewer experts
needed at all, ×0.5 on the sure rows' share).

**Line.** FAST only (changes states; same-benchmark-scores bar).

**Prior art.** Published: [AdapMoE, arXiv 2408.10284](https://arxiv.org/abs/2408.10284) (per-token adaptive expert count,
−25 % activations, 1.35×, no accuracy loss claimed), [Lu et al., ACL 2024, arXiv 2402.14800](https://arxiv.org/abs/2402.14800)
(dynamic expert skipping), [CALM, NeurIPS 2022, arXiv 2207.07061](https://arxiv.org/abs/2207.07061) (confidence-gated
compute per token with calibrated thresholds — the right calibration recipe for τ),
[Expert-Sample, arXiv 2602.02443](https://arxiv.org/abs/2602.02443) (greedy accuracy stable at half top-k on Qwen3-class
fine-grained MoEs), [HookMoE, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1610) (>50 % fewer activated experts,
upper layers tolerate fewer), [LExI, arXiv 2509.02753](https://arxiv.org/abs/2509.02753),
[PCoMoE, arXiv 2609.01024](https://arxiv.org/abs/2609.01024) (per-layer damage is uneven — the warning).
New: gating k by the *drafter's* confidence, available before the check, instead of by the row's own gate weights.

**Cheapest test (≤ 10 min).** Greedy, 40 problems of the code benchmark, τ = 0.95, k_sure = 4, k_unsure = 10, shared
expert on. Record score and tiles/check.
- **PASS:** score within 1 problem of baseline **and** tiles/check ≤ 0.70 × baseline → ship it, then calibrate τ CALM-style.
- **KILL:** at k_sure = 6 the score drops ≥ 3 problems → the state error compounds across 48 layers; the lever is dead at
  45 GB (keep it only in the 24 GB form).
- *Zero-GPU pre-test:* from existing logs, the fraction of code rows with drafter p ≥ 0.95 and their 125B agreement. If
  sure rows < 60 % of rows, the expected tile cut is < 20 % → skip the GPU test.

### Lever 3 — Stop paying for wasted rows and the serial tail  (EXACT-safe)

**Mechanism.**
- **a. Adaptive draft length.** Stop drafting at the first token whose acceptance estimate falls below θ (drafter top-1 p,
  or AdaEDL's entropy lower bound — training-free), floor 6, cap 24 (longer when the drafter is on a confident run: at
  p = 0.935 a 24-row draft keeps 12.5, and confident runs are exactly where the cap binds). Expected rows/check on code
  ≈ 12–13 at ≈ 10.2 kept → tiles −10–15 %; on chat rows ≈ 5 at ≈ 2.9 kept → tiles −45 % and SSD misses per kept −40 %.
  Any draft sequence is exact under standard verification.
- **b. Drafter ∥ commit.** The drafter needs only the accepted token ids; commit needs the 125B's states. Different
  queues → −5 ms/check.
- **c. One command buffer, fused small kernels** (norm + router + gather; the GDN chunk pieces): −3–4 ms of the ~6 ms of
  dispatch/sync. Falls out of L1-C.
- **d. FR-Spec for the drafter.** The 248k-vocab lm_head is likely most of the drafter's 5.4 ms [G]; drafting over a
  frequency-ranked 32k subset keeps verification exact → ≈ −2–3 ms.

**Expected gain** [P]. Code: ×1.15–1.2 on both lines; chat/mixed: ×1.6–2 (the 14-prompt mixed EXACT 51 → ~80); 24 GB bytes
per kept: 19.5 → ≈ 14.6 (and ≈ 7.7 with the 2-bit copy, meeting the ≤ 10 MB goal).

**Line.** EXACT (a–d all preserve the output distribution).

**Prior art.** Published: [SpecDec++, COLM 2025, arXiv 2405.19715](https://arxiv.org/abs/2405.19715) (acceptance head +
threshold stopping policy, +7–11 %), [AdaEDL, arXiv 2410.18351](https://arxiv.org/abs/2410.18351) (training-free entropy
bound, +10–57 % over static length), [FR-Spec, arXiv 2502.14856](https://arxiv.org/abs/2502.14856) (reduced drafting
vocabulary, exact), [EAGLE-3, arXiv 2503.01840](https://arxiv.org/abs/2503.01840) (the chat-kept route); the brief's "gate
the last row with a 17th draft" [K] is a one-row version of (a). New: nothing — the MoE twist is only that a wasted row
costs *bytes and tiles*, so the published +10 % becomes +15–20 % on code and ~2× on chat.

**Cheapest test (≤ 10 min, CPU only).** Replay logged drafts + acceptance outcomes for 2k code tokens and 2k chat tokens
through the stopping rule at θ ∈ {0.5, 0.7, 0.8, 0.9}; count rows/check and kept/check (rows per kept token is the tile
proxy; if routing logs exist, count unique experts directly).
- **PASS:** some θ gives rows-per-kept −20 % on code at ≤ 3 % kept loss, and −40 % on chat.
- **KILL:** every θ that saves ≥ 15 % rows costs ≥ 8 % kept on code → the drafter's confidence is uninformative about
  rejection; keep the fixed 16 and drop (a).

---

## 3. One bet

**Lever 1 — get the SSD off the critical path at 45 GB.** It is the largest single bucket on both lines (27 % of a FAST
check, ~45 % of an EXACT check), it is the only EXACT-safe lever worth more than 30 %, it is pure scheduling (the bytes
already stream in a quarter of the time they currently cost), and its 10-minute oracle-replay test cannot be fooled:
either the wait collapses when reads are issued at t = 0, or the read path is broken and that is the finding.
Build it as drafter-state → 125B routers → one QD-16 batch at draft time, with stride prefetch for the residue and the
check as one command buffer; if the 2-bit trellis tooling matures, make it permanent with a ≤ 2.8 bpw body that fits.

Then the only wall left is expert tiles — and the arithmetic says bits will not move it, only tiering (Lever 2) will.

**What I would not spend on:** kernel redesign #13; a bigger standalone drafter (a 2B drafter at 4-bit reads ~1 GB per
draft step → 16 GB per check ≈ 56 ms, dead on arrival unless it is EAGLE-style on the 125B's features); tree drafts (rows
are free but their experts are not); 2-bit *for speed* at 45 GB (2-bit is for SSD bytes and residency).

---

## Appendix — the small checks that decide which branch of the arithmetic is true

1. **Tiles vs bytes** (5 min): expert kernel on one fixed expert set at 4-bit and 3-bit, 2 and 16 real rows. Time tracks
   bytes → bits help directly (≈ +25 % at 2.8 bpw), tiering helps less; time tracks tiles → this file stands as written.
2. **Latency vs bandwidth of the EXACT SSD wait** (the oracle replay above). Latency → prefetch; bandwidth → fit.
3. **Sure-row fraction on code** (logs only): sets Lever 2's ceiling; < 60 % sure rows → tiering < 20 %.
4. **Wasted-row share** (logs only): rows after the rejection point per check; code expected 6.5/16, chat 13–14/16; sets
   Lever 3's ceiling.
