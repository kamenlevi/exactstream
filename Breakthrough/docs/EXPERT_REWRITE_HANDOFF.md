# EXPERT REWRITE -> BigMoE's r57 stack: the tgl16eu flag (handoff, Breakthrough, Tue 17 Nov 2026 11:1x)

**UPDATE 11:2x: wire knob_v2** (~/Breakthrough/data/xrw/knob_v2, md5s in knob_v2.md5). Its outputs are BITWISE identical
to knob_v1's (24 real calls) and its upgrade grouping is O(1): v1's grouping cost ~0.5 ms per 16-row check. In the
lines below, read knob_v2 for knob_v1.

## The decision (orchestrator, 11:1x)
- tgl16eu goes into Saturday's stack as a flag in BigMoE's r57, BEFORE Thursday's tier 1.
- The condition: Leap's c25 A/B confirms by Wed evening about -2.5..-3 ms at 16 rows, no extra expert GB per check, and
  determinism.
- Quality is judged INSIDE the stack's gate (tier 1 Thu + card gate Fri night). If the stack fails, the first bisect
  turns tgl16eu off.
- w4a8u (int8 activations) is NOT in.

## What tgl16eu is (full record: docs/EXPERT_REWRITE.md)
It replaces the lossy expert one-call (`ooo1cq.tier_glu_nax6n_1cq`, reached through `ref2._ooo_sel` on the FAST-NAX
path; COH routes through it too):
- The same grid, grouping head, QOM drops, late-slot polling and posts. Leap's own transforms build it:
  `ooo1cq.a_one_call_q`, `ooo1cq.b_drop`, and coh1cq's coherent(system) pool reads.
- The accelerator's left operand (the activation rows) is staged in threadgroup memory, with an analytic-layout epilogue.
  This exact part is bitwise == ooo1cq.
- The TIER-4 UPGRADE: a 3-bit pair joins its expert's 4-bit segment when that expert has an eligible 4-bit pair in the
  same call. That expert's LSB plane is read anyway, so there are no extra bytes and those pairs get the 4-bit weights.
  It gives -19 % segments at 16 and 32 rows.
- Quiet kernel numbers (11:05): 0.830x of today's call at 16 rows, 0.843x at 32 rows.
- Unchanged pairs are bitwise == today. Upgraded pairs are at ooo1cq's accuracy against the 4-bit fp32 reference.
- The output is deterministic for the same call content, but a pair's tier depends on the other rows of the call (as
  today's 1c tiers already do). Leap accepted this on the FAST track.

## Wiring it into r57 (BigMoE owns r57; these are the lines; the code is FROZEN)
After `import ref2` in scripts/r57_screen.py (ooo1cq is importable from ~/Leap/scripts by then), behind a flag of your
choice, e.g. `R57_XRW=tgl16eu`:

```python
if os.environ.get("R57_XRW"):
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/data/xrw/knob_v1"))
    import xknob
    xknob.install()                         # wraps ooo1cq.tier_glu_nax6n_1cq once (COH on and off)
    xknob.MODE[0] = os.environ["R57_XRW"]   # "tgl16eu" for Saturday ("tgl16e" = the exact half only)
    xknob.ON[0] = True                      # False = today's call, bitwise (the bisect switch)
```

- **Scope: the check only.** r57 removes REF2 around each propose, and the one-call is reached only through
  `ref2._ooo_sel`, so drafts are untouched. Nothing in ref2 reassigns `ooo1cq.tier_glu_nax6n_1cq` (checked); coh1cq
  only swaps ooo1cq's inner kernels during its own call.
- **Precondition:** the FAST-NAX one-call path must be on (R57_TIERS=1c forces REF2_NAX_1CQ; the p11 / native store).
  Smoke check: after a few checks, `xknob.COUNTS["cand"] > 0` and `xknob.COUNTS["fallback_not_p11"] == 0`. A store that
  is not p11 falls back to today's call, and the count shows it.
- **Frozen:** ~/Breakthrough/data/xrw/knob_v1, md5s in ~/Breakthrough/data/xrw/knob_v1.md5. Do not edit it; any change
  ships as knob_v2 with a new release check. Release check: scripts/xrw/knob_check.py PASS.
- **Memory:** no new buffers. Compile is ~0.8 s at the first call (kernels are cached per process).

## The checks behind it (all in docs/EXPERT_REWRITE.md)
- a) ON off == today, bitwise.
- b) 5 repeats bitwise; rs_m / rs_l == today's.
- e) Leap's m32 scenarios: posts cover every used expert 24 / 24; late release with no hang.
- Unchanged pairs bitwise == ooo1cq (layers 20 / 44).
- The grouping kernel == its numpy replica in 80 / 80 random calls, and == ooo1cq's grouping with the upgrade off.
- c) (32 rows == two 16-row halves) does NOT hold, by design.
- Engine A/B (Leap's names, 11:12): F16gu / F32c6u = tgl16eu, F16gk / F32c6k = tgl16e (exact), F16gx = G1 + GDN16 +
  tgl16eu. **The deciding number is F16gu - F16g.** Each arm's line carries the expert GB per check; pass K re-runs
  F16gu / F16gx for determinism.

Contact: Breakthrough (account-1). Leap reviews.

## EvalSuite's needs (11:2x; runner 69 checks them; due THURSDAY MIDDAY)
EvalSuite pins tgl16eu in its stack amendment draft (results/lossygate/stack_amendment6_DRAFT_2026-11-17.md). After a
FAIL, a bisect (tgl16eu off first) is a new config with its own amendment and gates.

1. **The flag name / value as it appears in BigMoE's stack env.** Proposed: `R57_XRW=tgl16eu`. BigMoE confirms the exact
   name to EvalSuite directly.
2. **A per-run engagement field in runs.json.** The frozen knob counts calls (`xknob.COUNTS`: cand = calls run by the
   candidate, orig = calls sent to today's kernel, fallback_not_p11). The suggested r57 lines:

   ```python
   c0 = dict(xknob.COUNTS)                                                     # at run start
   runs_json["xrw"] = {"mode": xknob.MODE[0], **{k: xknob.COUNTS[k] - c0[k] for k in xknob.COUNTS}}   # at run end
   ```

   Runner 69's check: `xrw:cand@run > 0` over the run's code answers, plus orig == 0 and fallback_not_p11 == 0 with the
   flag on.
   - A count of UPGRADED pairs would need a new grouping output, i.e. knob_v2 (a lazy per-run counter, one eval at run
     end). It ships only if EvalSuite asks for that proof; knob_v1 stays frozen under Leap's A/B.
- **EvalSuite 11:2x: the call count is ENOUGH (no knob_v2).**
  - Engagement: `xrw:cand@run > 0` over the run's code answers.
  - Validity, checked from runs.json before the verdict (a line in the amendment): xrw.orig == 0 and
    xrw.fallback_not_p11 == 0, summed over the run, with the flag on, and xrw.mode == "tgl16eu". Any non-zero means part
    of the run took today's path, and the run is VOID.
  - So r57 must keep `xknob.ON[0] = True` for the WHOLE run. Never toggle it around phases (prefill, drafts, etc.):
    drafts don't reach the one-call anyway, and every other one-call counts.
  - The flag R57_XRW=tgl16eu goes into the draft as proposed, final on BigMoE's confirmation.
