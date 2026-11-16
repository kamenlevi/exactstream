# THINK_1116c: the DRIFT wall only (Mon 16 Nov ~18:3x -> ~19:45, desk sprint; orchestrator)

Why: NewIdeas' 18:2x correction (tree_frontier_t16.txt, TRIED FRONTIER#2). At quality, the lossy acceptance frontier
is FLAT (guarded comment rules x0.997-1.019). So the kept x1.43-1.62 that 775 lossy needs, and that exact 400+ needs,
can come only from the DRAFTER: a per-guess walk-strict ~97.8 % to 30 deep (kept ~23 per 32-row check vs 16.9 today).
> (a quote from the project owner, kept private)

## The facts (records: WEEK98.md, PLAN900.md 1b, ~/TRIED/TRIED.md; quote with their headings)
> **CORRECTED 18:3x (Fable C's audit, IDEAS_1116c_drift.md):** "97.8 % flat to 30" and the 89.65 / 94.32 / 95.29 band
> 23-30 figures are MARGINAL (every-anchor) rows, not walk-strict. Walk-strict, the zero-drift head reads 93.2 / 95.3 /
> 96.4 / 97.8 / 98.8 / 99.2 / 99.3 by band; t30b walk-strict 23-30 is 97.7. So the walk-strict drift gap on code is
> 4.0 / 3.5 / 2.2 pt at bands 3-6 / 7-10 / 11-14, not ~8. The +51 % (20.77 vs 13.77) is NewIdeas' LAB-TRUESTATE shortcut
> on the chain-32 cells; the real tree-64's 20.77 is a coincidence. New premise [K]: on HE+, drift is sequence-specific
> and persistent across blocks (92 % of the per-window drift-gap variance).
- Given the 125B's TRUE state at every position (LAB), the t30b head holds 97.8 % flat to 30 on HE+ (MBPP 93.6 %,
  knowledge-limited). In real drafting it carries its OWN state, and accuracy decays: HE+ band 23-30 is 89.65 (t30b),
  94.32 (+2B audit), 95.29 (+4B audit); the head-or-2B oracle is 96.29.
- Ties cap the shallow bands even with zero drift (true-state chain 93.2 / 95.3 / 96.4 / 97.8 at bands 0 / 1-2 / 3-6 /
  7-10); trees cover ties; bands 3-14 stay 93-96 % even at K127, because there ties AND drift both act.
- In the 2B sandbox (SIB read-out), the carried state's direction error to the true state is 0.85-0.99 by depth (HE);
  a state from a different (sibling) sequence is worse (0.95-1.10).
- Real 127-node trees (HE+): kept 16.26 / 20.77 / 23.70 at 32 / 64 / 128 rows. PLAN900 1b books the LAB true-state
  ceiling as +51 % kept (20.77 vs 13.77 own-state); Fable C, please verify which run each figure comes from.

## Kill list for drift (no re-proposal without a NEW mechanism)
SIB sibling states (KILL 15:2x; the w 0.2 mix is a parked micro-lever); t30c, more budget / data (S10 FAIL); tapmlp /
shallow taps (CLOSED 18:1x: the fix needs mid-depth taps the 125B can't give); capacity x2 (no effect); zero-start depth
(no help in the 2B sandbox); carryac state supervision (+0.6-0.8 pt at 400 steps); span tree / copies; any smaller
verifier; any acceptance-rule route (flat at quality).

## The question
Find a mechanism that closes >= half of the own -> true gap at depths 3-30 on code, i.e. moves walk-strict bands 3-14
from 93-96 % toward 97.5 %+, at a drafting cost that still fits ~3-7 ms per block, by Saturday or with a certified
method by Saturday. Look outside LLM drafting too (state estimation / Kalman-style correction, control, codecs,
weather-model data assimilation, biology). For each idea give:
- the mechanism, and why it attacks drift (not ties);
- the number it should move, with a bound from our records;
- a Sandbox98 screen (2B + s30 analog, <= 5 GB, <= 30 min) with a PASS / KILL line, or a <= 10-min CPU screen on the
  existing dumps;
- what it costs per block on the 125B, and the quality risk (exact methods preferred).

Lenses: Lunatic = the information limits of drift; ZERO = from first principles; FAR = drafting shapes and training;
Fable C = audit the drift facts above, then invent; Fable B = what the 125B's check pass already computes that the next
block could use.

Write IDEAS_1116c_drift.md in your own folder. Then send ONE line (best idea, its number, its screen) to NewIdeas
(`peer send ojox-28 "..."`) and the orchestrator (session message).
