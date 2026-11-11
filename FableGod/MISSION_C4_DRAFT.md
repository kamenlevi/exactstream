# MISSION C4 (Fable C): THE UNION STACK. Price the cut that halves the experts a 16-row check touches.

(Draft by the orchestrator for Kamen, Tue 3 Nov 2026 06:0x; revised 06:5x with NewIdeas' counts and 10:2x with the map's verdicts. Ready to send on Kamen's go.)

**Why.** Your ROUTE_50 puts FAST 350 at 50 GB 12-20 ms per check short. Two of the three gaps are the same quantity, the union of experts a 16-row block pulls per layer:
- the expert term needs ~6 ms (19.6 -> ~14: "union x0.5 or a +30 % kernel; nothing measured gives it");
- the SSD needs 6-10 ms of exposed transfer hidden, and its bytes are the union's misses.

The night and this morning measured the cut:
- **Q1-UNION at theta 0.9: PASS** (Breakthrough 03:20). Gated code flips 0.26 % vs Q1's 0.52 %; all rows, fewer flips than Q1 at 0.99 everywhere (code 0.98 vs 1.34 %). Bytes = Q1 at 0.9's: code SSD MB per kept token x0.595 at C 256 (MoonShots band45). Union per 16-row block x0.618 on the map's blocks (NewIdeas).
- **Prior art:** the union step is published as OEA, Opportunistic Expert Activation (arXiv 2511.02237): tokens piggyback on experts the batch already loads. Ours is OEA inside one verify check, with k0 per row from the drafter's certainty; that gate is new.
- **NewIdeas' stack (union per check vs exact, code, the map's blocks):** OEA k8 on every row x0.83 (top-8's bytes, rows keep 9.0 of 10 experts); OEA k6 x0.65; hybrid (sure rows k0 2, others k0 8) x0.53, IFEval x0.76, multi x0.76; hybrid u6 x0.44, chat ~x0.60.
- **Breakthrough's map, Tue 09:31-10:18 (SCOREBOARD b70716c):** all-row flips per domain (code / IFEval / multi) against Q1 at 0.99's 1.34 / 5.42 / 3.69 %. PASS = at or under; BETWEEN = up to 1.5x; KILL = above.
  - **PASS in all three domains:**
    - oea_k8: 0.62 / 3.83 / 3.07 % (bytes x0.83; gated code flips 0/2211).
    - q1u_cf09_t3 (Q1-UNION 0.9 + 3-bit sure rows): 1.19 / 5.06 / 3.24 %.
  - **PASS on code only:**
    - oea_k6: 0.71 / 6.06 / 5.38 %.
    - hyb_0.9_u8: 1.34 / 6.18 / 4.08 %.
    - hyb_0.9_u8_t3: 1.19 / 6.68 / 4.39 %.
    - hyb_0.9_u6: 1.28 / 8.01 / 5.71 % (KILL on multi).
  - So the hybrids need a domain switch (code on, chat off). FLOOR night 1 measures q1u_cf09_t3 and oea_k8 at scale (extra_arms.txt).
- **Correction at live check lengths (NewIdeas 11:00, ~/NewIdeas/data/LIVE_read.txt):**
  - Live checks stop at running cf 0.3. They hold 9.7 / 3.7 / 3.0 rows (code / IFEval / multi) vs the map's 15.7, so sure rows find fewer union members.
  - q1u_0.9_live: 1.25 / 6.45 / 3.77 % = PASS on code, BETWEEN on chat (IFEval z +2.47 vs Q1 at 0.99).
  - Every 0.9-gated union arm is therefore code-only. Chat needs Q1-UNION at 0.99, which about halves chat's SSD saving (-10.5 / -8.4 %).
  - oea_k8 (no drafter gate) is unaffected.
  - REACH-CUT (reach_0.9_0.5_live): unsure rows unlikely to be reached keep only top-2 + union. It ties on code (1.28 %) with a 13 % smaller code union; chat BETWEEN. Adapted from AcceptMoE (arXiv 2608.02989).
- **NewIdeas' SSD prices for these arms** (band45 capture mode, C 256 + M-17, the map's windows; SSD MB per kept token vs EXACT, code / IFEval / multi):
  - Q1 0.9: -50.3 / -19.7 / -16.4 %.
  - Q1 0.9 + 3-bit: -67.1 / -27.8 / -21.7 %.
  - OEA k8: -21.0 / -17.8 / -20.8 %.
  - HYB 0.9 u8 + 3-bit on code: -73.0 %.
  - wide45 model on code: EXACT 86.3, Q1 0.9 +51 %, + 3-bit +70 %, HYB u8 +62 %, both +84 % (model).
- Also: 3-bit ranks 0-1 on sure rows is free (PASS, adds nothing over Q1); t3_all is BETWEEN on code, unsafe on chat.

**Read first:** your ROUTE_50.md and INVENT.md; Breakthrough's SCOREBOARD (Q1-UNION, the four OEA / hybrid arms, t3_all, the sure-rows arm); MoonShots' data/band45.txt; NewIdeas' masks in ~/NewIdeas/data/dm_masks; TRIED lines on union, top-k, routing and gates; DAY9_2026-11-02.md from 02:00 on.

**Deliver**
1. **The arms that pass at live check lengths, priced on the union per live block:**
   - code: Q1-UNION 0.9 (+3-bit sure rows), the hybrids, REACH-CUT;
   - chat: oea_k8 and Q1-UNION 0.99;
   - with the domain switch that implies. experts per layer, the expert ms at 16 rows, MB per kept token at C 320 and at 45 GB, and the exposed SSD. Say how much of the 12-20 ms each closes, for code and chat.
2. **New union cuts of your own** beyond OEA and the certainty gate (your methods), each priced the same way and tagged published / untested / new.
3. **The best stack against the 350 budget** at 50 and 45 GB, on both tracks where it applies (a union cut is FAST; say if any part can be exact).
4. **One pre-registered follow-up screen** (<= 10 min) for the next question.

**Rules**
- CPU only. Writes only in ~/FableGod/fablec/. No downloads. Both tracks first-class. Halve gains. Mark every number [K] / [P] / [G].
- Price on the union per 16-row block.
- Within ~40 minutes, write a complete first draft to UNION_STACK.md, then refine it, and stop.

**Push addendum (Tue 17:0x, BREAKTHROUGH PUSH; read ~/FableGod/push/BREAKTHROUGH_PUSH.md):**

The 350 budget now also has to price the levers that passed or were built today:
- the volatile expert tier: built (BigMoE 2a99aa7), smoke test tonight; per NewIdeas' RANKED 33, C 320 roughly halves code MB per kept token;
- the native layout NAX6-N: +21 %, with 3c-N as the exact fallback at -4.3 %; the conversion is prepared;
- NAX6-H: +9.1 %;
- warm pools: mission 20, tonight;
- the U3 engine mode (= q1u_cf09_t3, code): first measurement tonight.

In deliverable 3, give two stacks at 45 GB and at 50 GB:
- the best FAST stack on code, and on chat (OEA k8 / Q1-UNION 0.99);
- the best EXACT stack.

For each, say what is still short of 350 and which wall it hits (SSD, verify row, dense, attention). Fable A (the SSD wall) and Fable B (the verify-row wall) are working in parallel in ~/FableGod/push/. Don't duplicate them; take their walls as inputs once their files exist.
