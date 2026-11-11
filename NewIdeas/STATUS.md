# STATUS (NewIdeas = chief researcher) - round report 2026-10-22 16:49 (end of the account-1 run; history in git)

Kamen's standing decisions: small-RAM line judged by task results; no external GPUs ever; main track = 24 GB setting; speed
goal ~350 tok/s one chat (1,000 after, never closed); fast modes only as opt-in, exact unchanged; tag every lever
published / published-untested-here / new, keep one new bet in flight, show the arithmetic.

## Per target: biggest factor -> mechanism -> evidence (today)
- **A, main track 24 GB exact:** SSD waiting, not compute (all-hit 4-row verify 38 ms, near the floor). Shipped: bit-identical
  prefill patch (TTFT -24 % at 327 tokens). Closed: fast mode dh_w4 (substitutes for JIT prefetch), 2-bit-heavy tiers (gate),
  bit-identical dequant-once kernel (slower). Measured 16:45: per-pass shared-expert-set budget b16 +22-32 % tok/s (non-exact),
  quality 1.6-1.8x over the strict targets on gsm / ifeval: not passing yet (next: b20 / b22 combined with v3, budget-aware JIT). Correctness: the default loop guard overrides ~18 % of long math answers; fix R1 (math
  masked like tool calls) measured on replay; BigMoE's check on TASKS.md.
- **B, 12 GB line:** B12 fits 12 GB with real prompts; prefill 426-474 tok/s exact. Verdict FAILED 3/5 (14:03) but is
  BORDERLINE once the guard and B12's run-to-run noise are separated (PLAN_B.md 16:02). Eval suite: like-for-like rerun + two
  runs or per-row pool reset for history-dependent variants. Next quality headroom: 4-bit pools / LATE3 (Breakthrough).
- **C350 / C1000, one chat:** exact closed on every class; agentic / prose closed for any drafter; code / math need three
  non-exact pieces (MMA kernel, per-pass budget, drafter tau16 ~11-12.6). Native MTP (4.00 agentic) stays the best drafter
  measured; code-chat night runner ready at Chat1000 (Kamen's decision).
- **D, Mac mini / E, 4 GB:** unchanged (D ~21-24 tok/s modelled for B12 + MTP; E needs a different model).

## Today's mechanisms
Alive: per-pass budget (+22-32 %, quality not yet inside the gate), D1i decoupled prefill (adopted on both lines in its exact parts), guard fix R1,
paired history-free A/B runs for frugal variants. Dead today (DEAD.md): dh_w4 fast mode, dequant-once bit-identical kernel,
DFlash retarget step 1, layer-local trained sticky routing, SuffixDecoding on agentic, frugal prefill of tool outputs, D1k
look-ahead, guard variant R2 (misses loose loops).
Lessons: replay a "loop-only" guard over normal outputs before shipping it; test a stricter detector on the positives it was
built for; single runs of history-dependent variants are noisy (11 of 98 IFEval rows flip between two B12 runs); timestamps
from `date` only.

## Open risks
The Eval suite's like-for-like B12 rerun and B8 verdict; BigMoE's guard check (math-only loops would no longer arm); the
budget's quality (combined check at b20 / b22, main account's week).
