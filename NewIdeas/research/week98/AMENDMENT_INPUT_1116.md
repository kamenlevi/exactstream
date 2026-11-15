# Input for EvalSuite's pre-capture CERTIFY98 amendment (NewIdeas, Goal A lead; prepared Sun 15 Nov 22:1x)

The orchestrator approved these lines Sun 21:5x, and K* / T8 22:0x-22:1x. EvalSuite files them Monday morning as one
pre-capture amendment. A short follow-up Tuesday evening pins Leap's measured c32 and R32/R16 (E2) and confirms K*.

Every number below is a dev replay (= real decoding per kept_walk) unless marked LAB (true state, not deployable).

## 0. Conventions
- **Walk-strict bands count TARGETS ONLY:** a walk cell with no target is not a miss (EvalSuite 151d997).
  - kept_walk is unchanged.
  - certify98.py's identity check stays on the old counts as a code check.
- **Bands:** gate bands 0-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 (amendment 4). Band 0 alone is used where stated.
- **LB:** ratio estimate minus t(K-1, 0.95) x the window-clustered CR1 SE, one-sided 95 %.
- **Sets:** HE+ 140 is gated. MBPP 300 is gated only where stated; otherwise it is reported against its own ceiling.

## 1. LAB: the knowledge line (true state; NOT deployable)
- **What:** the head's accuracy with the 125B's true state at every depth.
  - At depth d from anchor a, the head reads the trunk at row a + d, so its guess IS the depth-0 guess of anchor a + d.
  - No build: it comes from the capture's depth-0 cells.
  - Cross-check on dev: the shortcut reproduces BigMoE's REPLACE oracle, kept/32 20.77 / 12.72 vs 20.90 / 12.76
    (research/week98/lab_truestate.*).
- **Metric:** MARGINAL accuracy per band (all cells): 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30.
- **Line (HE+):** every band point >= 97.5 AND LB >= 97.0, with each band's point quoted against Kamen's 97.8.
  - Dev: 97.95 / 97.93 / 97.91 / 97.88 / 97.84 / 97.79 / 97.72.
  - Rough power at 140 windows: 0.71. All points >= 97.8 has P = 0.41.
- **MBPP:** reported. Dev 94.0-93.6 flat: knowledge-limited.
- **Information:** its walk-strict bands. HE+ dev 93.2 / 95.3 / 96.4 / 97.8 / 98.8 / 99.2 / 99.3 (0 / 1-2 / ... /
  23-30): the 125B's near-ties after a miss.

## 2. C1: tree coverage at band 0
- **Candidates:** t30b + EAGLE-2-16 (pinned, amendment 6 / 6a) and tree-32 at K* (section 4).
- **Metric:** walk-strict COVERAGE at band 0 (targets only).
- **Line (HE+):** point >= 97.8 AND LB >= 97.0.
  - Dev: tree-16 98.6, tree-32 98.47. Rough power 1.00.
- **MBPP:** reported (tree-16 97.8, tree-32 98.61).

## 3. C2: t30b + frozen 2B audit (exact; replaces NewIdeas' 21:3x line, which failed on dev)
- **Auditor:**
  - mlx-community/Qwen3.5-2B-MLX-4bit, revision 93760be4f1f69842a46bc13dbdc0f19e291392a3;
  - model.safetensors sha256 713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5;
  - no training.
- **Rule (MoonShots', frozen Sun 01:3x):**
  - The 2B runs one teacher-forced pass over the bonus + drafts.
  - At the first row where its top-1 differs from the draft and its margin >= tau(depth), its token replaces the draft
    and the head re-drafts the rest of the block. tau = 4 nats at depth 0, 2 at 1-2, 1 at 3-6, 0.5 at 7-30.
  - Headline P = 2 audit passes per check; P = 1 reported.
- **Head:** dd_t30b.safetensors, sha256 790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502.
- **Scorer:** MoonShots' join (join_heplus.py / checkstart.py) on the capture's cells, plus the 2B's teacher-forced
  rows (GPU minutes, 2-4 GB, by day).
  - NewIdeas' tree_audit.audit_chain reproduces MoonShots' kept exactly: 14.456 / 15.399 / 7.427 / 7.713.
  - MoonShots re-derived the bands to the digit (walkstrict_bands.json).
- **PRIMARY (HE+):** walk-strict bands 15-22 and 23-30, each point >= 97.8 AND LB >= 97.0.
  - Dev P = 2: 98.32 [LB 97.5] / 98.28 [97.5]. Rough power 1.00.
  - Dev P = 1: 97.63 / 98.21. Power 0.20; reported.
- **CO-PRIMARY:** kept/32, paired window-clustered LB of (C2 - t30b) > 0. Dev +8.9 % (P = 2), +5.0 % (P = 1).
- **Reported:** bands 0-14 (dev P = 2: 92.7 / 91.0 / 93.2 / 94.8 / 97.1), and MBPP (dev P = 2: 15-22 / 23-30
  91.5 / 89.9, knowledge-limited).
- **No speed claim:** the 2B pass is untimed (1.06 GB read per pass).

## 4. The tree-K line: top-8 children (T8) at K* = 32 rows (31 best-first nodes + the bonus)
- **Basis:** BIGGER_TREES.md.
  - Leap's R(K) (05eea11) x MTP Head's S11 kept and real dumps to 31 nodes: K* = 32 on HE+, 16 on MBPP. Trees of
    48-128 nodes never pay.
  - The 127-node top-16 dump (T16) is information.
- **T32/32 = tree-32 kept / chain-32 kept** (the same 32 rows; no R dependence).
  - PASS: LB >= c32 AND point >= c32 x 1.03. c32 is Leap's measured tree overhead, pinned Tue (first cut 1.03-1.05).
  - **Gated on HE+ AND MBPP.** Dev 1.181 [LB 1.126] / 1.304 [1.283].
- **T32/16 = tree-32 kept / chain-16 kept** (amendment 6's statistic at K*).
  - PASS: LB >= rho AND point >= rho x 1.03, where rho = R32/R16 x c32. R32/R16 comes from Leap's MEASURED 32- vs
    16-row engine check (E2), pinned Tue evening, before the capture.
  - **Gated on HE+ only.** Dev 1.531 [LB 1.470].
  - For orientation, the bar is 1.454 at the floor, 1.489 at the contract and 1.74 at today's G32T.
  - MBPP is information: 1.371 [1.351]. It fails at every cost, because MBPP-like text prefers tree-16.
- **Band-0 coverage at K*:** section 2, gated on HE+.
- **The sentence names K* per set:** 32 for HE+-like code, 16 for MBPP-like.

## 5. Conditional candidates (named now; pinned only if their screens pass)
- **C2b, the distilled auditor** (Sandbox98 L1CF, Mon from 07:00): 1c's recipe, one epoch over 3,032 n1112 files.
  - If PASS (HE+ agreement >= +1.0 at the epoch end, val not falling): the final LoRA's sha256 replaces the 2B in C2's
    rule, with tau re-picked on MBPP 100 and checked on HE+ 24 before the capture.
  - Same lines as C2.
- **C3, t30c** (MTP Head S10 PASS2, Monday): t30b continued on the unseen n1112 files, with the recipe chosen by
  p2f / p2s / p2b.
  - Only if the Tue-night run PAYS on dev by Wed ~15:00: HE+ 23-30 marginal >= t30b + 1.0, no band below t30b - 0.3.
  - Then it replaces t30b in C1 / C2 / tree-K as a separate, named arm. t30b's lines stand regardless.

## 6. What the certificate can and cannot say (for the Saturday story)
- **Can (if the lines pass):**
  - "Exact. The first guess after a check is covered ~98.5 % (tree-32); deep guesses 15-30 are right ~98 % (2B audit,
    P = 2).
  - The head's knowledge, given the true state, is ~97.8 % flat (LAB).
  - Tree-32 keeps x1.18 more tokens per check than the chain at the same 32 rows."
- **Cannot:**
  - 97.8 % at walk-strict bands 1-14: every deployable lever is at 90.6-97.1, and even the true state at 93.2-96.4 for
    0-6.
  - 450 exact: it needs flat 98.0-98.4 % with the check at its byte floor (TARGETS_450_900.md).
