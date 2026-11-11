# 12 GB chat mode (P2') passes the IFBench gate on its loose metric; strict is 7 points below (Tue 10 Nov 10:01)
- **What:** the 12 GB build's chat mode (verified greedy loop + the P2' first-call width rule, lossy at rounding level).
  Run on the card gate's 150 IFBench keys at a 1,536-token cap, one order, plus a reverse-order re-run of 10 keys.
- **Gate [K, EvalSuite]:** IFBench loose 40.0 % vs 44.0 % for the reference: -4.00 pt, McNemar p 0.21. Decision 4 PASSES.
  The re-run was identical 10 / 10. Peak 11.22 GB.
- **Information only [K]:** IFBench strict 32.7 % vs 40.0 %, -7.33 pt, p 0.013. The 24 / 45 GB configs scored 36.7 % strict
  on the same keys, so "same instruction following" cannot be claimed from this.
- **Cap hits:** 16 outputs hit the 1,536-token cap (reference: 10). 5 of the 16 end in near-pure repetition loops (share of
  repeated 8-grams in the last 512 tokens >= 0.87); 4 more are partly repetitive.
- Sources: ~/EvalSuite/docs/RESULT_card_ifb12_2026-11-10.md; /Users/kamen/Fit12/data/scr/ifb12;
  /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'IFB12 RUN'.
