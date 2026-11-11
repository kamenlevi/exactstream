# FAST track: ideas held back only because they are not bitwise (NewIdeas, Mon 2 Nov 22:0x; updated Tue 06:4x)
BREAKTHROUGH PUSH (17:06): the scoreboard is in RANKED_45GB.md 'BREAKTHROUGH PUSH SCOREBOARD'. FAST bets in it: Q1-ON-MISS
(pre-registered 554dfe0; sure rows drop only rank 2-9 experts that would miss: Q1's SSD bytes at 94-98 % of the experts),
FLOOR night 1, and the 12 GB P1-K1 stack. Prior art for the push: research/PUSH_PRIOR_ART.md.

> (a quote from the project owner, kept private)
lines whose reason is bits, a new reference or "not exact", not quality. Gains are each line's own measured numbers unless
marked (model). Every item needs a same-scores test before a FAST preset ships.
Novelty (Mon 22:xx, ~/TRIED/NOVELTY.md, prior-art web checks): NEW = 45 #2 (the verifier's experts keyed on drafter
certainty), 45 #7 (P-frame reuse) and 45 #4's comment loophole (as a span rule); adapted = COPYGATE (SAM-Decoding), the
12 GB two-pass (TriSpec, with a self-proxy at top-2), 3-anchor ANE drafting (SpecBranch / PEARL / Mirror-SD); published
practice = the NAX hybrid for padded tiles (MonoMoE, ORT's MoE GEMV path). Every live-default lever is published or
adapted; the new ones get their first same-scores test this week.
Tue 06:4x: Q1-UNION's union step is published (OEA, Opportunistic Expert Activation, arXiv 2511.02237, ICML 2026), so
45 #2 is now 'adapted + a new gate': setting each row's base expert count by the drafter's certainty is the new part.
Tue 23:1x (MoonShots' sweep): PSR §7 (arXiv 2610.34721, 26 Oct) already weights the verify block's expert demand by the
drafter's confidence (one set per block, rerouted), so the certainty gate is adapted too. Ours: a per-row count, no
reroute, no renormalisation.
One new bet in flight per line: 45 GB Q1-UNION at cf 0.9 on code (RELAX A / B / C, Thu 00:30; with live-length checks: code PASS, chat BETWEEN), +
3-bit sure rows (FLOOR at scale Wed, live-cut unions asked); 24 GB Q1 at the 24 GB pools (M-18d: x0.658 PASS on
option A; option B PASS x0.572 with the MTP chain's own gate, Tue 11:1x); 12 GB P1-OEA (pass 1 at top-1 + the block's union members: certificate PASS Tue 10:18 at x0.465 pass-1 reads; mixed
runs PASS; timing re-test T3R 19:15: x1.065 = BETWEEN but void (a heat PAUSE inside the run); 3-bit stack x1.227 INFO;
clean 16-prompt re-run T3R2 PASS Wed 13:1x: x1.148 pooled on shared text over 16 / 16 prompts, 57.3 vs 49.9 tok/s at
12 GB; 3-bit stack x1.268 INFO; next EvalSuite same-scores at 12 GB behind RELAX + T0); T0 same-scores held to Thu.
P2-PREFETCH (exact add-on) KILLED Wed 13:1x: x0.901 (late reads fully exposed; pass 2 runs on few checks).

## 45 GB, ranked by gain x readiness
> (a quote from the project owner, kept private)
   bsolutely fine with all of this"); a FAST mode may depend on the cache state. Conditions: EvalSuite certifies it under
   >= 2 different prompt orders (histories); EXACT stays the default and history-independent; the logical-cache variant
   (PSR, repeatable per request) later.
   - Rule: drafter-sure rows skip only the rank 2-9 experts that would miss; everything resident is used. Reads = Q1's.
   - Map (Tue 22:59, live-cut checks): cf 0.5 flips 0.36 / 2.68 / 1.63 % (lines 1.34 / 5.42 / 3.69), cf 0.9 0.48 / 2.24 /
     1.44 %: PASS / PASS / PASS, the lowest flips of any FAST arm. Details under item 2.
   - Price [P] (capture, C 256): cf 0.5 SSD MB per kept token -60 / -37 / -33 %; with 3-bit sure rows -81 / -53 / -45 %.
   - Steps: cf 0 (~05:00) -> qom_0.5_live_t3 (pre-registered) and the C 160 arm (the masks assume C 256 pools; day runs
     at C 160 drop more) -> BigMoE's engine path (U3's 3-bit sure-row path + the resident keep) -> EvalSuite under two
     prompt orders.
   - EvalSuite's certification template (23:22, docs/PREREG_TEMPLATE_cachedep.md, 099ce88): two histories (canonical + a
     seeded shuffle, one process each), PASS required in both; a repeat of H1's first 50 rows must match; run in
     BigMoE's engine at the shipped C.
1. The expert kernel (Fable mission 27): NAX6 42-43 ms vs 3c 50-56 ms per 16-row check. NAX7-F is KILLED, so exact NAX
   work ends at NAX6 (a new reference: -8..-13 ms, one same-scores test). FAST design C, int8 x int8 on the matrix units
   (M27-C), is a rough PASS at 1.55-1.73x bf16, i.e. the expert call ~42 -> ~25-27 ms (~-15 ms per check, model). Leap
   builds design C next (row-invariant), then one same-scores test covers NAX6 + H-1 + design C.
1b. H-1, the NAX hybrid for padded tiles (Leap TABLE_350 "H-1", af57ce6; published practice: MonoMoE, ORT's MoE GEMV
   path): its compact arm is +10.4 % on the expert call at S 16 (+15 % at S 8, which matters for chat's short checks; +6 %
   at S 32), ~-4 ms per 16-row check. Bits differ, so EvalSuite judges it with the FAST arms. It needs the small-group split
   on the GPU first (gpu_grp, no host sync). Bundle it into item 1's same-scores test: one reference change, one test.
1c. F1, the long-context attention kernel (FAST-K; ranked Tue 21:39, the orchestrator 21:3x). Leap's fs4 = FATT16 (R-66's
   fused flash-attention-2 on the matrix units) in place of REF2-T's chunked SDPA in the 12 QSA layers.
   - Batch- and N-invariant by construction (INV PASS on every form). More accurate than MLX vs fp64 (mean 1.75e-3 vs
     2.29e-3), but not bitwise to REF2-T: a FAST-K reference change.
   - Speed [K B6, layer level, 16-row calls]: 1.77x at 2K, 1.70x at 10K, 1.53x at 20K. Attention per check 23.5 -> ~13
     ms at 10K. Fable B: attention is the only check term that grows with context, ~1.37 ms per 1K keys.
   - Price [P]: code at ~2K keys ~-5 ms per check (~+5 %); agent turns at ~10K -10.5 ms (~+13 % decode); 16-20K -13..-15
     ms (~+15 %). It matters most for Kamen's agent work.
   - KV8 on top: BETWEEN (Leap 21:3x: -0.8 ms at 10K; K-outlier risk), not pursued.
   - Status:
     - the kernel is built;
     - the ENGINE SWITCH: Leap's Wed by-day item (fs4 in the QSA verify path, a default-off flag; BigMoE's WED_TESTS
       item 7 'when Leap asks');
     - then a token / INV smoke;
     - then EvalSuite's FAST-K same-scores, PRE-REGISTERED Tue 21:53 (63cd47e): decision 4 on the core 210 vs ref2t +
       RULER 24 at 16K (<= 1 net lost), agent 29 as flags, AT14 timing; ~3.5 h GPU, earliest Fri after the conversion
       checks.
2. Q1, top-2 experts on drafter-sure rows (cf >= 0.99), kept weights never renormalised: code SSD MB per kept token x0.60,
   implied 63.6 -> 70-76 tok/s; sure rows at 3 bits (no LSB read) x0.46 -> 72-79 (Breakthrough's arm tonight). 0.09 % of
   gated code rows flip. Test: EvalSuite RELAX arm A (Thu 00:30).
   Q1-UNION (Tue 03:19, SCOREBOARD fc32173): a sure row also keeps its rank 2-9 experts that the check loads anyway
   (the union of the unsure rows' 10 ranks and the sure rows' top-2), so bytes = Q1's at the same cf. At cf 0.9 it is a
   PASS: all-row flips code 0.98 / IFEval 4.92 / multi 3.04 % vs Q1 at 0.99's 1.34 / 5.42 / 3.69 (lower everywhere);
   gated code 0.26 % (7/2678). Union per 16-row check (NewIdeas CPU replay of the map's blocks, masks bit-equal to
   Breakthrough's): code x0.618 at cf 0.9 vs x0.720 at 0.99 (IFEval x0.903 / x0.963, multi x0.934 / x0.973). SSD MB per
   kept token on cap45's live checks (MoonShots band45, C 256 + M-17): code x0.595 at 0.9 vs x0.681 at 0.99; all six
   prompts -34.9 % -> implied 72.2 tok/s SSD-only / 84.3 with the GPU side (model). RELAX rev 3 APPROVED (orchestrator
   06:4x): Q1-UNION 0.9 + copy in A / B / C. The 3-bit sure-rows arm PASS (1/2211 gated code at 3
   bits = the 4-bit control's): 3-bit records on sure rows are free; t3_all's damage sits on unsure rows.
   MAP Tue 09:55 (routed 10:16): Q1-UNION 0.9 + 3-bit sure rows (q1u_cf09_t3) PASS / PASS / PASS: all-row flips code
   1.19 / IFEval 5.06 / multi 3.24 % (paired vs Q1 at 0.99: z -0.78 / -0.88 / -1.33; vs Q1-UNION 0.9 at 4 bits +7 / +5 /
   +7 flips net). SSD MB per kept token: cap45 -46.3 % (76.4 / 91.8 tok/s, model) vs -34.9 %; map capture code / IFEval
   / multi -67.1 / -27.8 / -21.7 % vs -50.3 / -19.7 / -16.4 %; 24 GB x0.536 vs x0.591. The best arm that passes every
   domain: in Wed's FLOOR at scale. Not a RELAX arm this week: EvalSuite needs a per-row 'no LSB' mask in the
   engine's tier call first (orchestrator 10:4x; rev 3 stays for Thursday).
   LIVE-LENGTH CHECKS (11:01; map 10:55): the morning's unions used all 16 walk rows. Live checks are cut at running cf 0.3
   (rows per check: code 9.7, IFEval 3.7, multi 3.0). q1u_0.9_live = 1.25 / 6.45 / 3.77 % = PASS / BETWEEN / BETWEEN
   (paired vs the 16-row arm z +2.2 / +4.3 / +2.6; vs Q1 at 0.99 z -0.5 / +2.5 / +0.3). So Q1-UNION 0.9 is a CODE
   lever live. Chat needs cf 0.99 (Q1 at 0.99 is the line; the union only adds experts): chat SSD -10.5 / -8.4 %.
   RELAX rev 3's A / B / C run code, math and agent sets only. SSD prices are live and stand.
   Q1-ON-MISS (QOM; pre-registered 554dfe0; map Tue 22:59, Breakthrough's prefix): a sure row keeps ranks 0-1, its union
   members AND every rank 2-9 expert already resident in the base pool after the check's loads; only the rank 2-9 experts
   that would MISS are dropped. Reads = Q1's at the same cf. On live-cut checks:
   - cf 0.9: 0.48 / 2.24 / 1.44 % all-row flips = PASS / PASS / PASS;
   - cf 0.5: 0.36 / 2.68 / 1.63 % = PASS / PASS / PASS (vs q1u_0.9_live z -4.6 / -8.9 / -6.7).
   The lowest flips of any FAST arm on the map; keeping the resident experts fixes Q1-UNION's chat problem.
   - Price at cf 0.5 [P] (capture, C 256): SSD MB per kept token -60.4 / -36.9 / -33.4 %; SSD-only model, code HE68 /
     HE91, 84.7 / 62.1 -> 138.9 / 83.2 tok/s (halve). GPU work stays near exact's union: an SSD lever, not a compute one.
   - Cache-state dependent: Kamen's decision 2 (research/PUSH_SHORTLIST.md).
   - Next: qom_0.5_live_t3 (pre-registered 23:0x: the 3-bit bound for QOM-full, reads -80.7 / -52.6 / -44.9 %); cf 0 at
     ~05:00; the engine = U3's 3-bit sure-row path + the resident keep (BigMoE), then EvalSuite's same-scores.
2b. Sketch-on-miss: KILL (Tue 02:1x, Breakthrough SCOREBOARD 7a79270). Whitened shadows on held-out real activations have
   output error 0.92 at rank 32 and 0.89 at rank 64 (a drop is 1.0; the GO bar was 0.5), and plain SVD 0.99 / 0.97: the
   expert spectrum is too flat for a low-rank stand-in. Gain 0.
2c. Sidecar drop (MoonShots M-26): every rank at 3 bits (plan 3333333333) and the freed sidecar RAM spent on base pools:
   code SSD MB per kept token at C 256 46.3 -> 24.5 (-47 %), with Q1 27.8 -> 11.8 (-58 %, under FLOOR's 33 MB target); C 320
   8.1 with Q1; chat 30.2 -> 17-18. Implied steady 63.6 -> 75.8, 85.3 with Q1: the largest priced FAST SSD lever. Quality
   is the whole question: all-3-bit failed the strict KLD gate before (x1.24-1.83). Breakthrough's t3_all / t4_all map runs
   Tue (second in its list), with the sure-rows-only 3-bit arm right after as the fallback.
   MAP RESULT (Tue 02:29, SCOREBOARD 8b3d8e5): code BETWEEN (+0.59 points of all-row flips over the 4-bit control; sure
   rows 0 / 2,922), chat worse (+4-5 points: mid rows flip ~5 % vs ~0.4 %). Next: EvalSuite's same-scores (IFEval first;
   proposed as an extra RELAX arm Thu), and tonight's 3-bit sure-rows arm, which keeps chat's mid rows at 4 bits.
2d. The union stack (NEW today; OEA is published, the certainty gate is ours): every row keeps a base top-k0 and adds
   any lower-ranked expert the check already loads. Union per check vs exact (CPU, the map's blocks): OEA k8 on every
   row x0.81-0.83 on all domains (= top-8's bytes, but rows keep ~9.0 of 10 experts vs 8.0); sure rows k0 2 + unsure
   k0 8: code x0.527, IFEval x0.756, multi x0.762; unsure k0 6: code x0.435, chat x0.60. SSD MB per kept token on cap45
   (C 256 + M-17; model tok/s SSD-only / + GPU): OEA k8 -18.6 % (67.9 / 72.1), sure 2 + unsure 8 -44.4 % (75.0 / 91.4),
   unsure 6 -52.7 % (77.7 / 99.1), sure 2 at 3 bits + unsure 8 -55.8 % (79.6 / 100.2). Chat, on the map capture's
   IFEval / multi windows (band45 capture mode, same pools; SSD MB per kept token): Q1 0.9 -19.7 / -16.4 %, OEA k8
   -17.8 / -20.8 %, HYB 0.9 u8 -32.4 / -32.4 %, HYB 0.9 u6 -43.2 / -45.2 %, HYB 0.9 u8 3b -40.5 / -37.7 % (code there:
   Q1 0.9 -50.3 %, HYB 0.9 u8 3b -73.0 %). On chat the unsure-row cut is the lever. OEA k8 needs no drafter
   signal, so it also fits the chat switch (MTP head drafting chat, no DFlash2 cf): -18 / -21 % on chat by itself. MAP RESULT
   (Tue 09:34-09:55; per domain all-row flips <= Q1 at 0.99's 1.34 / 5.42 / 3.69 % = PASS, <= 1.5x BETWEEN; rows 3,366 /
   3,397 / 3,553): oea_k8 0.62 / 3.83 / 3.07 % PASS everywhere; oea_k6 0.71 / 6.06 / 5.38 PASS / BETWEEN / BETWEEN;
   hyb_0.9_u8 1.34 / 6.18 / 4.08 PASS / BETWEEN / BETWEEN; hyb_0.9_u6 1.28 / 8.01 / 5.71 PASS / BETWEEN / KILL;
   hyb_0.9_u8_t3 1.19 / 6.68 / 4.39 PASS / BETWEEN / BETWEEN; q1u_cf09_t3 PASS everywhere (item 2; all on 16-row unions,
   optimistic: see item 2's live-length result). REACH-CUT live (10:59): code 1.28 % PASS, a tie with Q1-UNION live
   (z +0.2) at a 13 % smaller code union; chat BETWEEN: a code-only add-on, small in SSD bytes (cap45 code x0.925 of
   Q1 0.9's, capture x0.97; union -6 %; model +3 % SSD-only / +6 % with the GPU side). Gated code flips 0-1
   of 2,211 in all. On code every arm ties Q1 at 0.99 (paired z -0.3 to -0.9), so a code-only stack is open:
   hyb_0.9_u8_t3 code SSD MB per kept token -60.2 % on cap45, -73.0 % on the capture, 24 GB x0.422. On IFEval the
   unsure-row cut costs (u8_t3 z +2.8, u6 z +5.4 vs Q1 at 0.99), so chat keeps unsure rows at 10. OEA k8 alone passes
   everywhere with no drafter signal: the chat-switch candidate (-18 / -21 % chat bytes). Wed 00:30 FLOOR at scale:
   q1u_cf09_t3 (Breakthrough's run (4), 'the best one'), and hyb_0.9_u8_t3 on the 24 HumanEval windows as the code-only
   arm if the night has room.
   What each part buys on code (wide45's cost model, 16-row checks, model tok/s today): EXACT 86.3, Q1 0.9 130.2 (+51 %),
   + 3-bit sure rows 146.6 (+70 %), HYB 0.9 u8 139.9 (+62 %), both 158.9 (+84 %). So 3-bit sure rows (map PASS at cf
   0.99 / 0.999 only) are worth more than OEA's unsure-row cut; two optional 3-bit map arms proposed (06:5x).
3. REF2c (att_single -5 ms + padded GDN in_proj -0.5 ms; Leap's estimate ~-6 ms per check, ~7 %, model): STOP PASS on
   IFX (EvalSuite Tue 06:37): 220 new IFEval prompts 84.1 vs 83.2 % (+0.9 [-1.5, +3.3] pt, 6/4, p 0.75); pooled with
   Part 1 -0.9 pt PASS. Part 1's 85 vs 90 reads as the core items' luck. A new reference (bits differ from REF2-T):
   Kamen's call, FOR_KAMEN decision 8. The live-loop gain is not measured yet (BigMoE's combination measures it).
4. Near-tie acceptance on chat (R-61N at rho 0.25: chat +8..+23 % per pass; exact-loop M-15 +14 %) + the '#'-comment
   loophole: RELAX arms B / E / F (Thu 00:30).
   Re-price the comment loophole when RELAX arm B runs: Fable B's B5 found that 49 % of code first misses are comment tokens
   (2.9x their base rate), so its kept gain may be well above the x1.06 estimate.
5. HC8 (D4): ~-2.5 ms more on the HC read; 1.44 % of rows flip, 0.00 % of sure rows; EvalSuite decision 4 PASS. In the fast
   preset (B mode).
6. bmm16 HC (L1-29, new rounding): HC mix-down 5.11 -> 2.59 ms per check (-2.5 ms). Needs a REF3 bundle; price it on top of
   HC8 / D1, which touch the same read.
7. Middle-layer MoE reuse (P-frame): KILL on bytes (the layer-wise map: union x0.91 only). New as an idea, dead as a lever.
8. 32-row checks + Q1: KILL as a global switch (M-18c: pooled code -1.3 % today / -0.6 % with NAX6; always-32 -14..-17 %).
   'Deep rows sure' +4.3 / +3.8 % (BETWEEN); only cap-bound prompts gain (HumanEval/136 +13 %). Fable B's B3 agrees: 32
   rows only as the copy-driven block inside long copies (NEXT_DRAFTER). U-32 (Tue 06:5x): the union stack doesn't
   change it: 32 vs 16 rows -1.3 % (deep sure +3.8 %), KILL.
9. Small kernel items held only for bits: L1-31x fused GDN prework (1.51 -> 1.09 ms), chunked (WY / UT) GDN verify (the GDN
   body is only ~3 ms per check, so <= ~-1.5 ms), FUSED_CONVNORM_VERIFY (+2.7 % on the old path). Bundle them only if 1 or
   3 opens a reference change anyway.
10. One answer in 4 parallel parts (L1-32): 0.59x expert reads per accepted token, ~2x (model), but it changes the answer's
    form (Skeleton-of-Thought style): highest quality risk, last.
Closed tonight: seeded exact argmax (FEXIPRO's method, published; ~0.1 ms at 16 rows: low priority, exact);
predict-then-correct EXACT KILL (FAST predict-and-skip BETWEEN, P ~15 %); learned stop rules M-21 / M-21b KILL on code.
Output-exact already (exact track, not FAST): the drafter gate + up merge (-1.6 ms draft, kept x0.9955, ~+1.5 % net).
Dead on value, not on bits: exact trees (TREE_WALL: best +1.6 % halved), MTPLX fused MoE kernels (slower), MLX 0.32.4 on
the old path (no gain; a re-test on today's path needs a download OK).

## Near misses of the strict KLD gate (quality-gated, not bit-gated; FAST judges by same-scores instead)
- Top-8 per verify row at 45 GB (M-09d, no renorm): -16 % experts, over the strict 1.10x gate; at 12 GB top-8 passed all 5
  rules (+17-20 %).
- 2-bit tails 4433332222 (TAILS2): code KLD 0.00267 vs line 0.00253 (5 % over); MSB-2 for the lowest ranks est. +20-25 %.
One same-scores arm each, after items 1-5.

## 24 GB
1. FM1 (budget b16 + budget-aware JIT prefetch + fixed 3:3:0 drafts): +33 % geomean (code 35.8 -> 48.8, edit 37.2 ->
   56.5); Eval run 1: 5/5 rules vs v3, 4/5 vs the 4-bit ref (IFEval -5.0). Needs run 2; Kamen OK'd a non-exact 24 GB
   version on 10-29.
2. POM / LPOM (cold ranks >= 6 read as 2 code planes): -12.4 % bytes, then -9.6 % more (predicted inside the gate).
3. R-30 UNIQ (drop single-use low-rank experts): bytes -17.6 % at 2.3 % gate mass; quality never measured.
4. Q1 at 24 GB (M-18d, NEW bet; option A with DFlash2's cf as the proxy): exact v3 verify 175.0 -> 115.2 SSD MB per kept
   token (x0.658, PASS, but within 0.05 of the line, so MTP Head's per-row MTP confidences re-run it as option B). On b16
   it is KILL (x1.079): the budget already caps the union at 16, and Q1 only changes which 16 fill it. b16 stays lower in
   bytes (81.8), so Q1 is the quality-safer alternative to FM1's budget (0.09 % of gated code rows flip vs b16's IFEval
   -5.0 in run 1), not an add-on to it.
   The union stack at 24 GB (Tue 06:5x, NewIdeas' copy of q1_24, same windows and pools, 2 seeds; SSD MB per kept token):
   Q1 0.9 103.4 (x0.591), + 3-bit sure rows 93.8 (x0.536), HYB 0.9 u8 83.5 (x0.477), + 3-bit sure rows 73.9 (x0.422).
   The hybrid reads about b16's bytes without capping any row below its top-8: the quality-safer route to FM1's speed,
   if 4-row checks keep quality (fewer union members come back at 4 rows than at the map's 16). MoonShots told.
   CAVEAT (11:07, from the live-length result): with ~4 rows per check the members nearly vanish, so quality falls
   toward plain Q1 at that cf. Plain Q1 0.9 is BETWEEN on code and KILL on chat on the map (1.78 / 8.86 / 7.60 %); even
   with live members, Q1-UNION 0.9 is BETWEEN on chat. So the realistic 24 GB version is Q1 at 0.99 (x0.658, M-18d's
   PASS on bytes, the map's line itself), or 0.9 on code only.

## 12 GB (FOR_KAMEN decisions 1, 2 and 6 become FAST-preset items)
1. Top-8 per verify row (TK1 / TK2): +17.4-19.8 %, 5/5 quality rules (decision 2). OEA k8 reads the same union and
   gives rows back ~55 % of the experts top-8 drops: the quality-safer form at equal bytes (the 45 GB map arm tells).
2. PMIN12 (MTP cut at p_min 0.5): chat +21 %, scores identical; waits for Fit12's HT1 trim (-0.5..-0.65 GB) (decision 6).
3. Sparse prefill lane (L2-10): 64K prefill 1.47x, peak -0.9 GB, answers identical at 32K / 60K (decision 1).
4. The two-pass certification (T0, new): same-scores Tue after IFX (EvalSuite wiring it 06:4x).
   Domain-adaptive certificate (Breakthrough 03:1x): PASS on its line (code certified 0/2863), but it does not beat
   cf 0.8 as the working point (IFEval CIs overlap); chat runs one exact pass anyway.
   T3 TIMING (Fit12, Tue 02:29-02:38): the 2-rank pass 1 gave the same tokens and logits as the 10-rank one on all 7 prompts;
   code 56.2 tok/s vs 18.9 for the exact 16-row check (3x), but under its pre-registered KILL line of 70. Pass 1 is 145 of the
   211 ms per check, so pass 1 is what must come down; math 1.24x the exact check, IFEval 0.85x.
   NEW BET (Tue 09:09, P1-OEA, pre-registered in QUEUE): pass 1 = each row's top-1 + its ranks already in the block's
   union (OEA). Code pass-1 reads 0.708 -> 0.329 GB per check at C 51 (54 -> 25 ms at 13 GB/s), same mean gate mass as
   top-2 (0.357 vs 0.358, estimate). Top-2 + union: today's reads, mass 0.53. MAP RESULT (10:23): P1-K1 PASS on code at
   theta 0.8: blocks needing pass 2 52 vs 53 % (paired z -0.5), certified flips 7/2997 = the control's 7/3019, at x0.465
   pass-1 reads; chat slightly weaker (+1.4 / +3.1 points of pass-2 blocks). P1-K2 PASS at today's reads: pass 2 in 48
   vs 53 % of code blocks (z -3.4), multi 66 vs 81 %. END TO END (mixed runs 10:44-10:51): P1-K1 two-pass certified
   flips on code 7/2997 = the control's 7/3019, all code rows 1.52 vs 1.96 % (PASS). TIMING (Fit12 14:39-14:46): KILL
   as registered (summed tok/s x0.849), confounded: a FAST arm changes the text, and on /91 the control stopped early,
   skipping 80 hard tokens. On the 630 shared tokens: P1-K1 x1.100 (the line), 3-bit top-2 x1.12, 3-bit P1-K1
   x1.16-1.20. Pass 1 145.9 -> 103.9 ms per check (bytes x0.485). T3P: pass 1 = 57 % SSD wait + 43 % GPU gap.
   Shared-text re-test (8+ prompts, ~10 min) pre-registered 14:52 (QUEUE).
   EXACT add-on (Tue 14:56, CPU model): P2-PREFETCH reads pass 2's experts in pass 1's GPU gaps for rows with drafter
   cf < 0.99 (6 per layer): -12.6 ms per code check (oracle -20). With 3-bit P1-K1 that models ~70-73 tok/s, at the 70
   line. Engine arm proposed to Fit12 (exact gate vs t80ck1u; line >= 1.05x).
5. Budget b20 (+24 % geomean, speed only so far); DFlash2 at 12 GB (+14 % code / +23 % GSM, +0.35 GB); C_m 56 (+7-14 %,
   short context only).

## Test order beside the exact queue (big runs weekdays 00:30-15:00; BigMoE's day screens and nightly combined run first)
- Tue night, done: Breakthrough's map (shadows STOP, t3_all BETWEEN, 3-bit sure rows PASS, domain-adaptive PASS,
  Q1-UNION 0.9 PASS), Leap's screens, Fit12's T3 timing, IFX STOP PASS (06:37). BigMoE's 45 GB arms did not run
  (NO-SWAP 34-41 GB vs 47-55 needed; the orchestrator's memory plan is Kamen's 07:15 decision).
- Tue by day (to 15:00, one MEM job at a time): T0 same-scores (12 #4) -> RELAX smoke; the 6 union-stack map arms
  DONE 09:34-09:55 (45 #2 / 2d: q1u_cf09_t3 passes everywhere; the hybrids pass code only); MTP Head's chain (D-23x, EARLYTAP, DUALDRAFT) by its latest starts.
- Wed 00:30: BigMoE combined -> FLOOR (~10 h). Thu 00:30: BigMoE combined -> RELAX night (the 45 GB FAST preset).
- Next free night: one same-scores test for NAX6 + H-1 + design C (45 #1 / 1b), FM1 run 2 (24 #1), the 12 GB FAST preset (top-8 +
  PMIN12 + sparse lane as one same-scores arm).
