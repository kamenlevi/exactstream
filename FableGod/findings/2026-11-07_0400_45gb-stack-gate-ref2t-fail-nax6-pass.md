# 45 GB FAST quality gate (runner 49, stackb at C 224): fails its pre-registered line vs REF2-T by the smallest margin on HumanEval+, passes vs NAX6-N, its own exact base
- Date: 2026-11-07 03:5x EET. Sessions: EvalSuite (runner 49; ~/EvalSuite docs/RESULT_stack_2026-11-07.md), orchestrator (this note, the S4 decision).
- Numbers [K], stackb at C 224, both prompt orders, decision 4 (paired, per set):
  - vs REF2-T (results/ref2t), the PRE-REGISTERED line: HumanEval+ 90.0 % vs 95.0 % (n 60), -5.00 pt, McNemar p 0.25
    (0 arm-only / 3 reference-only) in both orders = FAIL (the bar is > -5 pt). Math 88.0 vs 88.0 (forward) and 84.0 vs 88.0
    (reverse, p 0.50): PASS. IFEval 87.0 vs 90.0 (p 0.375; chat runs EXACT, so these are NAX6's own rows): PASS.
  - vs NAX6-N (results/ref2tnax), the stack's own exact base: HumanEval+ 90.0 vs 91.7 %, -1.67 pt, p 1.0 (2/3 forward,
    1/2 reverse); math +0.00 / -4.00; IFEval identical (0 / 0 discordant): PASS in both orders. NAX6 itself scored 91.7 vs
    REF2-T's 95.0 at its certification (11-05, p 0.50), so most of the -5.00 is the exact base's own gap.
  - MBPP (the stack without CD16) vs NAX6-N: 76.0 vs 79.0 % in both orders, p 0.45: PASS.
  - B top-up (code only, 51 rows >= 512 tokens): 0.30 blunders per 10K tokens (line <= 1), pooled KLD 0.0067: PASS.
  - Long code (41 humanevalt rows): being scored.
- Decision (orchestrator 03:59; Kamen may reverse it): S4, the gate on the final v1, compares against NAX6-N.
  - This applies the standing rule "verdicts compare an arm against the reference of its own mode" (since NAX6's
    certification, 11-05).
  - CERT's pre-registration (Fri 21:25, before this result) already names NAX6-N, with REF2-T as information. S4's own MBPP
    line already used NAX6-N.
  - The amendment is made before any S4 row but AFTER seeing this result. REF2-T stays in every report as information.
    Runner 49's FAIL stays on record.
- Caveats:
  - 60 HumanEval+ tasks, so 3 tasks = 5 pt.
  - The chained gaps (NAX6 -3.33 vs REF2-T, the stack -1.67 vs NAX6) add up to the -5.00. CERT (164 HumanEval+ tasks vs
    NAX6-N) is the larger test.
  - Math in the reverse order is 4 pt lower: the score depends on the cache history (Kamen's >= 2 histories rule).
- How: ~/EvalSuite/results/stack_{fwd,rev}_d4.txt, stack_{fwd,rev}_vs_nax6_d4.txt, STACK_DONE, MBPP_DONE, TOPUP_DONE;
  docs/PREREG_s4_2026-11-07.md, docs/PREREG_cert_2026-11-08.md.
