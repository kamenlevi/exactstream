# NEVER-WAIT engine form: the r57 <-> ref2 interface (BigMoE, Fri 6 Nov 09:3x; the orchestrator's split)

The spec is ~/FableGod/far/NEVERWAIT_TEST.md section 6. It lives in ref2's one-call (REF2_OOO_1C, and FAST-NAX's
REF2_NAX_1CQ), the tier call the FAST all-in ships. Leap builds the ref2 side and BigMoE the r57 side. Both are off by
default; off = today's code, bitwise.

**Stop rule (the orchestrator):** if tonight's map arm KILLs on code, or tonight's snw ceiling KILLs (< 0.3 S), the build
stops where it is and we re-read.

## 1. Flags

| flag | side | meaning |
|---|---|---|
| `REF2_NEVERWAIT=1` | ref2 | the one-call's gate posts a cold DEMAND expert's read and releases at once; the call runs without that group (pending) |
| `R57_NEVERWAIT=1` | r57 | sets REF2_NEVERWAIT, arms the per-check bookkeeping below, and asserts the combination (FAST only, one-call on) |
| `REF2_NW_FORCE=<spec>` | ref2, test only | marks chosen (layer, expert) groups late whatever their residency, for the bit checks in section 4 |

R57_NEVERWAIT asserts:
- R57_FAST is set and REF2_OOO_1C=1. It never runs on the two-phase path or on exact arms.
- R57_QOM_SPLIT is unset or composes with it (open point 1).

## 2. What ref2 keeps (inside the one-call; no host sync, no second expert call inside a layer)

For each MoE layer call L that leaves late groups, ref2 keeps:
- the late groups' ids;
- the MoE input rows they need (the call's x rows, as given);
- their routing weights (the gate scores the combine would have used);
- layer L's MLP-side inject gates (16 x 4 per row, under HC8).

At layer L+1's call:
- the late groups whose bytes are now in run on their saved input;
- their weighted sum is ONE separate tensor, written after L+1's own MoE write through L's saved inject gates (one extra
  write per late layer, never folded into L+1's write);
- groups still not in wait for the call after (nw2 / nw3 behaviour per the spec's section 5; the first build may cap at one
  layer and wait at L+1: open point 2).

Layer 47 (the last MoE layer):
- wait for a cold rank-0 group (at most one stall);
- drop every other late group (weight 0, counted).

End of check: nothing pending survives into the next check. ref2 asserts its late state is empty after layer 47.

## 3. What r57 does (BigMoE)

1. Per check, before the verify pass: `ref2.NW_RESET()` (or a CFG field), so no stale state survives a rolled-back
   warm-split pass or an aborted check.
2. The rows: the verify pass's rows are the check's rows (after D17 / U3's live cut, `len(dr) + 1`). ref2 sees exactly
   those rows in x. Rows the live rule cut are never in the call, so they carry no pending. Open point 3: confirm ref2
   needs no per-row drop list from r57.
3. Counters per check, from ref2 to r57, summed over the 48 layer calls (like Leap's QCNT_1C: a list ref2 appends lazily,
   r57 reads after the check's eval):
   - late groups;
   - late (row, expert) uses;
   - late mass (sum of the dropped routing weights);
   - groups landed one / two / three+ layers late;
   - layer-47 drops and layer-47 stalls;
   - stalled layer calls (should be ~0 with NW on, except layer 47).

   r57 writes them per check into checks.json (`nw` field) and per run into runs.json.
4. The warm-split and C3-R8 counters keep working. R57_QCNT1C's groups count the groups each call computed, late ones in
   the call that ran them.

## 4. Bit checks (who, against what)

| # | setting | must equal | who |
|---|---|---|---|
| 1 | REF2_NEVERWAIT unset | today's one-call, bitwise, on QOM / FAST-NAX / stack arms | Leap (ref2.self_test + a live arm) |
| 2 | REF2_NEVERWAIT=1, everything resident (a warm pass: no cold demand group) | today's output, bitwise (nothing is late) | Leap |
| 3 | REF2_NEVERWAIT=1 + REF2_NW_FORCE (chosen groups late at chosen layers) | a reference built from ordinary calls: layer L without the forced groups, then the forced groups' expert MLP on L's saved input, written through L's gates after L+1's MoE write | Leap (reference in MLX, the same kernels) |
| 4 | R57_NEVERWAIT=1 on a 2-prompt CAP-32 arm | runs clean; counters plausible (late groups > 0 cold, 0 warm); layer 47 ends empty | BigMoE (pre-flight) |

Exactness: an NW arm is NOT exact (late terms change the outputs). Adoption follows the map arm's quality line (spec
section 2.6) and EvalSuite's same-scores, as for every FAST arm.

## 5. The speed A/B (Sat night, if neither stop rule fires)

Spec section 6.5: 12 HumanEval, A-B-A-B, the stack vs stack + R57_NEVERWAIT=1.
- PASS: the median check is >= 0.5 x S shorter, at kept within 2 %, with S = the stack's median gate service per check.
- Pre-registered in its script before arming.

## 6. Open points for Leap (answer in this file or by message)

1. QOM-KEEP's resident keep vs a late group: a late group is a cold DEMAND expert (not an optional use), so QOM's optional
   uses are unaffected. Confirm that the one-call's demand / optional split gives this directly.
2. The first build: one layer late only (a group not in by L+1 waits there), or the full nw2 / nw3 carry?
3. Whether ref2 needs any per-row information from r57 beyond the call's rows (I believe not).
4. Where the inject gates live under HC8-1P v4 (ref2's HC8 path), so the extra write uses L's gates, not L+1's.
5. How a late group interacts with FAST-NAX's 1CQ head (the fixed rs_l writes per pair).
