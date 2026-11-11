# Drafter plan for 300 tok/s FAST on code: decision note (MTP Head, Wed 11-04 ~15:45)

> **CORRECTED Wed 11-04 20:5x: three cost lines below were too optimistic.** The PyTorch port's FLOP count is in
> notes/PORT_PLAN.md §3; Leap's measured extraction is in REPORT 'P-OFF amendment 2'.
> - Prefill extraction (§2): measured 56 tok/s on short windows and ~230 tok/s on 4.5K-token windows, not 300-690.
>   20M tokens is ~1-4 days of Mac time, not ~1 night.
> - Training (§3): ~225 GFLOP per anchor at L1-20's recipe. One epoch over 20M tokens is ~35-45 h on 2 x T4, not
>   2-3 h. 4B on a TPU v3-8 is ~15 h per 20M tokens, not 4.5 h per 100M.
> - Earliest live (§4): best case ~Nov 20-25 for 95 % per token (was ~Nov 13-18), if a probe passes and anchor
>   subsampling (~4x cheaper, untested) holds. Without it, add ~1-2 weeks.
> The lines are kept below as written, marked [CORRECTED].

**My pick:** don't start a cloud drafter build yet. First run two one-night local probes. They decide whether *any* drafter
we can train gets past ~12 kept per 16-row check. Meanwhile, ship today's drafter-side time cuts. The cloud build starts only
if a probe passes.

## 1. Target, and what the measurements say about it
- Today: DFlash2 L1-20 keeps **9.57** per 16-row check on code (24 held-out HumanEval+, the exact walk), ~10.5 in the
  engine (FAST, C 224). 300 tok/s needs ~14 kept at a ~50 ms cycle, or 10.5 at ~35 ms (the orchestrator's math).
- Asked-for target: **13-14 kept per 16-row check.** On today's evidence no drafter we can train reaches that:
  - A **perfect pick** among DFlash2's 16 candidates keeps 12.37 (RERANK-0). DFlash2 + the native MTP chain, perfect
    union: 11.35. The realistic MTP-assisted pick keeps 10.51, at a cost of ~20 ms of sequential MTP steps (MTPSCREEN).
  - **More data at DFlash2's size** adds +0.13 kept per doubling on code (5 points, D-09 + E2/D2). 13 would need ~23
    doublings. Data alone is not the route.
  - **Capacity** (a 2-4B "DFlash3") is the untested input. The estimate on file (DRAFTER_98) is ~0.94 per position, i.e.
    12-12.5 kept at 16 rows and 14-16 at 32 rows.
- **Realistic drafter target:** ~11.5-12.5 per 16-row check (+20-30 %). 14 looks reachable only with 32-row checks, and
  those pay only if extra rows get cheap (FAST row gating / NAX32). The rest of the way to 300 has to come from the cycle.

## 2. Data: what a bigger drafter would need, and the shortcuts
- **On-policy** (the 125B's own answers, decoded at ~50-110 tok/s): 1M tokens ~3-6 h; 5M ~2-4 nights; 20M 2-5 days of
  nonstop decoding. We have ~0.3M answer tokens on disk.
- **Distillation shortcut: off-policy prefill.** Run a public code corpus through the 125B's prefill at ~300-690 tok/s
  [K: 370 Leap, 613-692 BigMoE], saving the frozen-fc features (5 KB per token) and the 125B's top-32 soft labels.
  20M tokens is ~1 night; 100M is ~2-4 days. [CORRECTED: ~1-4 days per 20M] It needs a corpus download, which needs Kamen's OK (name, source, size).
  Unknown: whether off-policy tokens move kept as much as on-policy ones (+0.13 per doubling).
- **A smaller dataset** doesn't help on its own: the data slope is the limit, not volume per se.
- **The MTP chain avoids the data problem** (it is the target's own co-trained head; ~0.97 per depth on code), but it is
  sequential. Even with Leap's screened head, a step costs 1.35 ms (the layer alone 1.28 ms). As a drafter it keeps 9.97
  at 16 rows and 12.55 at 32 rows. **New, untested idea (MTP-BLOCK):** train the native MTP layer as a one-pass block
  drafter (DFlash-style) instead of a chain. That keeps the target-class weights, needs no new data, and trains locally
  like D-22.
- **Upload limit:** Kaggle caps private datasets at 200 GB, about 40M tokens at 5 KB (~73M at fp8). Uploading takes 1-3 days.

## 3. Free compute (MoonShots ~/MoonShots/research/FREE_GPU.md; Kamen creates the accounts; quotas to confirm in the UI)
- **Drafters up to ~0.6-1B** (DFlash2 class, 2x deeper or wider): Kaggle 2 x T4, 30 h/week, 12 h sessions. Needs a PyTorch
  port of our MLX trainer (~1 day of work). One epoch over 20M tokens is ~2-3 h. [CORRECTED: ~35-45 h]
- **2-4B (DFlash3-XL):** Kaggle TPU v3-8 (128 GB HBM, 20 h/week, 9 h sessions); needs a JAX / torch_xla port and sharded
  Adam. 4B over 100M tokens is ~4.5 h. [CORRECTED: ~15 h per 20M]
- Overflow: Modal's $30/month credits; AMD Developer Cloud's ~50 MI300X hours, after approval. No paid GPUs.

## 4. The probes, and the earliest live date
- **P-CAP (1 night, Mac, ~15-20 GB):** a 2x DFlash2 (10 layers) on L1-20's exact data and recipe vs L1-20. If kept moves by
  >= +0.3 at fixed data, capacity is the lever and the cloud build is worth it. If it moves < +0.1, the drafter route to 12+
  is closed for this model.
- **P-OFF (1-2 nights, needs the corpus OK):** prefill ~2M tokens of public code, continue L1-20 on it, and compare the gain
  per doubling with on-policy's +0.13.
- **MTP-BLOCK (1 night, Mac):** the MTP layer as a block drafter on the ~0.3M tokens we have; kept vs DFlash2.
- **If a probe passes:** data 2-4 nights + trainer port 1-2 days + Kaggle training 1-2 sessions + engine integration 1 day.
  The earliest a new drafter could be live is **~Nov 13-18** [CORRECTED: ~Nov 20-25 at best]. Expected kept on code: ~11.5-12.5 at 16 rows (~13-14 at 32).
- **If all three fail:** the drafter side is capped near today's 10.5. The 300 route is then cycle-side (BigMoE / Leap) at ~35 ms.

## 5. Drafter-side pieces that ship now (no new training)
- **R57_D7B=code:** the 4-bit drafter for code, 10.39 -> 6.30 ms per check, code kept +0.00 % (Leap R-85b; already in the
  engine, needs turning on for FAST code). Math loses 3.4 % kept. A 4-bit-aware fine-tune is quick (~1 h of GPU at ~11 GB),
  but first try two no-training variants (group size 32, or one bf16 layer; ~10 min each at ~3 GB).
- **R57_D17 + QOM:** one more draft for the check's last row at no cost (D-17-1: right on 79.8 % of full blocks). BigMoE has
  the patch steps.
- **Leap's screened MTP head:** halves each MTP step (2.57 -> 1.35 ms), which speeds up chat drafting through the chat switch.
- **R57_DA (draft-ahead):** exact; only for the SSD-bound 24 GB line (tomorrow, BigMoE).
