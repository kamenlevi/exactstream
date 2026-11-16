# 1c-tiers: the lossy check variant for EvalSuite's CODE-ONLY gate (WEEK98 pillar 2), build + gate spec

Fit12, Mon 16 Nov 2026 08:5x. **APPROVED** by the orchestrator (08:3x); NewIdeas coordinates the owners and orders the
slot.
- Name: **1c-tiers**. ("C3" is CERTIFY98's t30c candidate.)
- Sources:
  - Breakthrough ~/FableGod/deepdraft/LOSSY_CHECK_SPEC.md (V2);
  - BigMoE's HANDOVER (Sun 15 01:4x plan; semantics confirmed by Breakthrough 01:59, reading (a));
  - EvalSuite ~/FableGod/deepdraft/LOSSY_GATE.md section 10 and docs/PREREG_lossygate_2026-11-15.md (amendments 1, 1c,
    1d, 2a, 3a);
  - the price: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md "LOSSY32 RESULT".

## 1. What it is
- **Base:** lossy-45 exactly as shipped (Leap pkg_120b configs/lossy-45.env: C 256, HC8, SVH, QOM keep tables on
  R57_QOM_SETS=humaneval,mbpp, relaxed span accept, ...).
- **Variant:** base + V2 only, the 1c expert tiers in the check.
  - No V1 (head shortlist), no V3 (HC4), no V4 (3-bit dense), no close-call.
- **Env delta** (the names in LOSSY_CHECK_SPEC; BigMoE implements):
  `R57_TIERS=1c R57_TIERS_LAM=0.016 R57_TIERS_S=0.98 R57_TIERS_SETS=humaneval,humanevalx,mbpp`
- **Where:** verify checks only, in the check's MoE layers after the router. Unchanged:
  - prefill;
  - both drafters (DFlash2 / MTP);
  - the commit replay;
  - the dense side.

## 2. The rule, per check and per MoE layer
- **Importance:** I_e = sum over the check's rows d of s^d · w(d, e)^2.
  - d = 0 is the first verified row.
  - w is the row's top-10 routing weights, normalised as the block does.
  - s = 0.98, lambda = 0.016.
- **Tier per expert e of the check's union:**
  - 4-bit if I_e > 2.775·lambda (0.0444);
  - MSB-3 if 0.8457·lambda (0.01353) < I_e <= 0.0444;
  - else drop.
- **Engine semantics** (reading (a)):
  - **4-bit tier:** keep the base's precision (REF2-T 4433333333: rank 0-1 uses read the LSB plane, rank 2-9 uses read
    MSB-3).
  - **MSB-3 tier:** every use reads only the MSB-3 plane (no LSB read, even for rank 0-1 uses).
  - **drop:** every use of e at weight 0 (the km path: score 0, index -> the row's rank-0 id). No renormalisation.
- **Reference code:** BigMoE bigmoe/tiers1c.py (c1c217b): tiers_np (float64) and tiers_mx (MLX, the engine's form).
  Both return 0 / 1 / 2 per (row, rank) slot.
  - Fit12's copy reproduces Breakthrough's tiers_1c masks on all 1,520,640 slots of the 12 map code windows
    (/Users/kamen/Fit12/scripts/lossy1c_16row.py).
- **QOM:** on the R57_TIERS_SETS sets, 1c REPLACES QOM's keep tables in the check. They are never stacked; the base's
  other levers stay.

## 3. Build (owners; NewIdeas orders and times it)
- **Leap, ref2 one-call (~ref2.py:915-990):**
  - t["nolsb"] becomes a 2-D [M, R4] mask, R4 = 2 (tier[:, :R4] == 1); the where()s broadcast as they are.
  - The CPU dup pick (`un = np.where(~nl_np)`) becomes a GPU pick: the id of the first tier-0 slot of rank < R4.
  - Under 1c, eprep_k's fused per-row glue is bypassed.
  - When: by day today / Tue, after Leap's WEEK98 items, never delaying E14 (Tue 00:30) or M4 (23:05).
- **BigMoE, r57_screen_fast.py _fast_call (FG["t1c"]):**
  - tier = tiers1c.tiers_mx(ids, scores) on the GPU, with no host sync;
  - MSB-3 through Leap's 2-D nolsb;
  - drop through the km path (needs t["scores"]: lossy-45's preset sets BIGMOE_TIER_COMBINE=1);
  - the set scope from R57_TIERS_SETS (the R57_QOM_SETS plumbing);
  - the counter (section 4);
  - **a NEW snapshot** with the 1c code, plus two env files on it: the lossy-45 base and 1c-tiers.

## 4. The counter (exact key names; EvalSuite's registered parsing reads them)
runs.json per prompt:
`"lossy_check": {"checks": n, "e4": n, "e3": n, "edrop": n, "lmsl_rows": 0, "lmsl_fallback": 0}`
- **e4 / e3 / edrop:** the distinct union experts per check-layer by tier, summed over the prompt's checks and layers.
- **lmsl_rows / lmsl_fallback:** keep the keys at 0 (no V1). No `hc4` key (no V3).
- **Outside the sets:** the field is ABSENT (or every count 0) on prompts outside R57_TIERS_SETS.
- **EvalSuite gates `CARD_ENGAGE=lossy_check:e3+edrop+lmsl_rows`** (amendment 1d): > 0 on every code run, 0 on every
  IFEval run. `checks` and `e4` are not in the sum.
- **Sanity reference** (the map's 12 code windows, Fit12 lossy1c_16row.py), per check-layer means:

  | check | union | e4 | e3 | edrop | dropped share | e3 + edrop per check (48 layers) | expert GB per check, REF2-T -> 1c |
  |---|---|---|---|---|---|---|---|
  | full 16-row check (what the gate runs) | 76.6 | 8.7 | 20.5 | 47.5 | 61.9 % of the union (36.2 % of the row-rank slots) | ~3,260 | 6.54 -> 2.51 (x0.38) |
  | 32-row block (the map's PASS) | 114.2 | 15.3 | 30.7 | 68.2 | 59.7 % | ~4,750 | 9.81 -> 3.99 (x0.41) |

  - Checks shorter than 16 rows (early stops, MBPP's short drafts) give smaller unions and other shares.
  - Per check-layer, e4 + e3 + edrop must equal the union.

## 5. CODE-ONLY: what routes a prompt
- **In the gate: the benchmark set.**
  - A key's set is its prefix (humaneval/..., humanevalx/..., mbpp/...).
  - The lossy path engages only for sets in R57_TIERS_SETS. Everything else (IFEval, chat) runs the base path,
    token-identical to the base.
- **Asymmetry to know:** lossy-45's QOM runs on humaneval and mbpp only, so humanevalx (104 of HE+'s 164) runs exact
  expert reads in the base.
  - On humanevalx, 1c-tiers adds lossiness where the base has none.
  - On humaneval / mbpp it replaces QOM.
  - The gate covers all 264 code keys this way, as Breakthrough's spec intends.
- **In the package (deployment, later; NOT tested by this gate):** run.py's `--kind code` maps a user prompt to the set
  "humaneval" (KIND_SET), so the 1c sets engage. `--kind chat` (the default) maps it to "multi", the base path.
  - So CODE-ONLY in use means the caller's declared kind. There is no classifier.
  - A chat prompt sent as `--kind code` would take the lossy path; the gate does not measure that misroute.

## 6. Checklist before EvalSuite arms (BigMoE; results to EvalSuite with the pins)
1. **CPU self-check:** tiers_np / tiers_mx == Breakthrough's tiers_1c masks bit for bit on the 12 map code windows
   (done in c1c217b). Re-run it on the new snapshot's code.
2. **Unset path:** with R57_TIERS unset, the new snapshot's lossy-45 is token-identical to the old snapshot's lossy-45
   on a few code and IFEval keys (inert unless set). EvalSuite also checks this against cached rows.
3. **Scope:** with R57_TIERS=1c:
   - an IFEval key's tokens equal the base's, with no lossy_check field (or 0);
   - a code key engages (e3 + edrop > 0), with e4 + e3 + edrop = the union per check-layer, near section 4's reference.
4. **The arming rule (after last night's REPLAY crash):** each env (the base and 1c-tiers) decodes one scored HumanEval
   key and one IFEval key end to end on the REAL loop: rc 0, tokens written, the counter as in check 3. The head test
   stops at the model load and would not catch an error at the first scored key.
5. **Limits:** 1c only lowers the experts read and never adds ids, so the gate's MAXK and C pool are unchanged. Say so
   if the code shows otherwise.
6. **Pins for EvalSuite's second amendment:** the snapshot tree, the screen sha256, the .so, both env blobs.

## 7. The gate (EvalSuite registers it; summary for the owners)
- **Arms:** BASE = lossy-45 on the new snapshot (R57_TIERS unset); VARIANT = 1c-tiers on the same snapshot.
  - Same keys, order and process split; each arm in its own processes from the same cold start (amendment 1c).
- **Tier 0 (mini-panel KILL line), then tier 1 CODE-ONLY:**
  - pooled code (HE+ 164 + MBPP 100) above -3 pt with McNemar p >= 0.05;
  - per-set decision-4 guards;
  - the RUNAWAY guard (amendment 3a);
  - IFEval 100 in its own processes: tokens identical to the base (CARD_IFEVAL_IDENT=1) and engagement 0;
  - forward, then reverse; both orders pooled.
- **Check length: <= 16 rows** (lossy-45's loop). A PASS licenses the 16-row form in CODE-ONLY mode. The 32-row form's
  quality rests on the map's 32-row PASS until a 32-row lossy loop exists.
- **Time:** ~4.6 h at 45 GB (base ~2.3 h + variant ~2.3 h, both orders).
- **Slot (NewIdeas):** Tue by day, after night1117 and BigMoE's 127-node dry run; or Wed by day, before the capture.
  The drafter line wins any collision.

## 8. What it should do (information; the gate decides quality, never speed)
- **Quality, the map** (1c alone, 32-row blocks, vs the map's all-4-bit base; Breakthrough d9b2db7): code flips 1.13 %
  all-row / 0.48 % non-tie, against the lines 1.34 / 0.87. On the engine's REF2-T base the increment should be at or
  below the map's.
- **Bytes:**
  - at 16-row checks, 2.51 vs 6.54 GB of experts per check (x0.38);
  - at 32 rows, 3.99 vs 9.81 GB.
- **Time at 32 rows [P]:** 53.0-55.3 ms per check with one-pass forms, against lossy-45's 90.3-92.0
  (/Users/kamen/Fit12/data/lossy32/lossy32_price.txt). The 16-row speed effect is for the variant's own A/B, not this
  gate.
