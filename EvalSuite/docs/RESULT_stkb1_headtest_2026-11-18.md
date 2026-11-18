# RESULT: blob 1's head test through runner 69 is a function PASS; the xrw check is VALID in both orders, and t30b's tree drafts chat too (Wed 2026-11-18 16:5x)

PREREG_lossygate amendment 6 (filed 787590b, Wed 16:4x), section 2's first bullet: a by-day head test of runner 69 on
blob 1, before the smoke. It is a function check, not a verdict. Its records are information.

## What ran
- **When:** Wed 16:46:36-16:51:26, Kamen away (away_secs 65,171 at the start; he was back at ~16:54, after the end).
  BigMoE's key test had passed at 16:43:49, and the GPU was free.
- **How:** runner 69 on results/lossygate/stkb1_env.txt (git blob 4be579d0, tag stkb1, C 256) with amendment 6's pins
  (trees s98q b0d02d33, frozen_s98q 8245a18b, bigmoe d92cdb4b; screen 94cdd2a6323c; build_qom .so ff69fa24a745), CARD_MEM 39.
  - CARD_TIER=2, CARD_SCOPE=code, tools/stk_b1head_keys.txt (lcb6/3692, ifbench/124) with CARD_SETS=ifbench,lcb6, both
    orders, CARD_RUNSUF=_b1head, no engagement terms, against NAX6-N (ref2tnaxn.r6).
  - The wrapper applied the AWAY rule (no start below 900 s; a stop on two reads 30 s apart). It never had to stop.
- **Records:** results/CARD_stkb1_b1head_DONE, results/day69_stkb1_b1head_runner.log,
  results/lossygate/b1head_wrap_2026-11-18.log, results/card/es_cdstkb1_b1head_181646_{fwd1,rev1}_*,
  results/card_stkb1_b1head_t2code_{fwd,rev}/decode.jsonl. The dry run before it: results/day69_dry_stkb1_b1head.log.

## Results
- **Function: PASS.** The runner returned rc 0 and wrote a "(tier 2" DONE line that is not VOID or INCOMPLETE. Every
  pin held at the start (BigMoE's HEAD had moved to 43da7aa7; the pinned trees are unchanged there).
- **The drafter is t30b on every prompt.** Both processes installed r57_draft's q8h32 weights (sha256 b5388cc4...), and
  every record says drafter "mtp" with a tree block of 15 nodes and k 8.
  - The process log also prints "chat switch: MTP head d22_b5 ... (chat -> MTP, the rest -> DFlash2)". The records
    contradict it: the chat key has a tree block with sib_accepted > 0, which a K15 chain head cannot produce. I read the
    line as a stale label (asked of BigMoE).
- **Engagement**, per key and order:

| order | key | type | kept / check | checks | tok/s | tree sib_accepted | close calls accepted | xrw cand (orig, fallback) |
|---|---|---|---|---|---|---|---|---|
| fwd | ifbench/124 | chat | 5.09 | 403 | 26.4 | 134 | 0 | 19,729 (0, 0) |
| fwd | lcb6/3692 | code | 6.22 | 329 | 37.0 | 96 | 58 | 16,067 (0, 0) |
| rev | lcb6/3692 | code | 6.61 | 310 | 45.4 | 118 | 64 | 15,142 (0, 0) |
| rev | ifbench/124 | chat | 5.42 | 378 | 37.3 | 125 | 0 | 18,503 (0, 0) |

  - Close calls appear on code only, as registered. The tree and tgl16eu act on chat too. So amendment 6 keeps
    tree.sib_accepted and xrw.cand out of runner 69's CODE-ONLY engagement rule: as runner terms they would VOID every
    IFEval process.
- **The xrw check (tools/xrw_check.py, amendment 6's process-wide rules): VALID in both orders.**
  - It ran on scratch copies of each order's run with a ledger naming that order's runs.json; a tier-2 run without
    engagement terms writes no ledger, while tier 1 does.
  - fwd: orig 0, fallback_not_p11 0, cand 35,796, tree sib_accepted 230. rev: orig 0, fallback 0, cand 33,645, sib 243.
    (In this check, ifbench counts as "code", since only ifeval/ keys count as chat.)
  - Both modules manifests load xknob from the pinned path, /Users/kamen/BigMoE/data/r57/snap/frozen_s98q/
    Breakthrough_xrw_knob_v2/xknob.py, at sha 6474eab8d8f5. The same check with a wrong sha reads VOID, as it should.
- **Memory:** lifetime peak 42.6 GB in each process (C 256), with swap 9,753 -> 9,697 MB. Monday's ccx45cc1 tier 1 ran
  at the same C 256 and MEM 39: 57 processes peaked at 42.4-43.1 GB (median 42.9), with no swap stop. So tier 1's
  CARD_MEM 39 (gate 43 GB NO-SWAP) stays.
- **Lengths:** both answers run to the item's cap in both orders (ifbench/124 1,536 tokens, lcb6/3692 2,048), and so do
  NAX6-N's reference rows. That is the items, not a runaway.
- **Scores (n = 1 per set and order; information, not a verdict):** lcb6/3692 fails in the arm and the reference.
  ifbench/124 fails in the arm and passes in the reference, in both orders (McNemar p 1.00). The arm's tokens first
  differ from the reference at token 1 (ifbench) and token 37 (lcb6). A FAST stack can differ that early.

## What it means for tonight
- Tier 1 #1's chain (tools/stk1_chain.sh, 33f01fa: BREAK98_HOLD added, as amendment 6 section 7 says) was ARMED at 16:55
  (pid 9694) on these pins, with no wait for Breakthrough's slot (NewIdeas 16:4x). Kamen came back at ~16:54, so it
  holds under the AWAY rule and starts once he has been away 15 min, latest first start Thu 00:30.
- The smoke repeats this check on 3 keys at the chain's start. Its lifetime peak sets the card gate's memory by section
  2's rule. With a peak near 42.6 GB that would be gate 44, CARD_MEM 40 (a prediction, not a result).
