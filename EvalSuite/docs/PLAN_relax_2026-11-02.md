# Relaxed acceptance R1 + Q1's gate, free-running: scope (Mon 2026-11-02 16:3x; for the orchestrator)

Request (orchestrator 16:15, from Fable's RELAX_350 section 2): scope mission 7's free-running harness + decision 4 for
the first relaxed mode, R1 (accept the draft while log p(d) >= log p1 - 0.5 nat) with Q1's gate; CPU prep today;
propose a night. Not exact: adoption is Kamen's call. This file becomes the PREREG once the CPU prep is done, before
any row.

## 1. Arms
- **`r1g99`**: the 45 GB default's numerics with the rule and the gate on top.
  - Base config: REF2-T + OOO + fused combine, R-56d's `ref2to_gate16` (snap/leap-faff8e5, tiers 4433333333 / 3c /
    C_l 112, sync-free gate MAXK 160).
  - Drafts: DFlash2 L1-20 (Leap's l120_ft, live16b's chain) in 16-row checks.
  - Q1's gate: a row whose drafter cf for the draft it verifies is >= 0.99 takes top-2 experts at the model's own
    weights (mass, no renorm), as the decision map ran Q1.
    - cf = the selector's softmax over its 16 candidates at the pick: MTP Head's chain_cf, the HPD-0 walk's number.
    - The bonus row after 15 accepted drafts stays exact.
  - Acceptance R1(0.5): draft j is kept while logit(d_j) >= max logit - 0.5 on the gated row; then the bonus token is
    that row's argmax.
- **`g99`**: the same with exact acceptance (Q1 alone). It is Q1's own same-scores test (DAY9 #1's open step), and it
  splits the combined result into gate and rule.
- **Threshold 0.99 or 0.999 (the orchestrator's call; one flag):**
  - 0.99 is the operating point of Q1's bytes PASS and of RELAX_350's speed table.
  - RELAX_350's "~2.4 % on code" uses the 0.999 gate (1.25 % + R1's 1.19 %).
  - In the map, 0.99 costs +0.1 pt on code and +1.5 pt on IFEval (all rows 1.34 vs 1.25 %, 5.42 vs 3.97 %).
  - 0.999 is the fallback either way (mission 7's outcome B).

## 2. The harness: mission 7's emulation on the engine's own loop
- **Base loop:** live16b's loop (Leap's STEP 3k harness), commit keep k+1 unchanged. With the rule off, R-56d proved it
  bit-exact against REF2-T's AR at full length: 36/36 prompts, 15,489/15,489 kept rows, OOO + fused combine on.
- **Two caches in lockstep:**
  - G is the mode under test. Its gated rows and the rule decide k, and its taps feed the drafter.
  - E is exact REF2-T on the same committed prefix: the same 16 input rows, top-10 on every row, the same k+1
    committed.
- **Per committed token (from E's row and G's row):**
  - E's argmax: the deviation along the emulation's own prefix.
  - E's p of the committed token: a blunder if below 0.01.
  - Full-vocabulary KLD(E || G): the drift measure.
  - The cause of each deviation:
    - rule: the draft is not G's argmax;
    - gate: G's argmax differs from E's on a gated row;
    - context: the same on an ungated row.
- **The gate's mechanism:** one hook in `model.block`, the per-row expert count for the current 16-row call (None
  leaves the code path unchanged).
  - Ranks >= k are zeroed after the rank sort. The dropped experts are still computed at weight 0, a bit-exact zero
    contribution, as `dm_common._pert_scores` does.
  - No bytes are saved, so tok/s is not measured.
  - Kept per check is measured, paired per check: the rule's k against exact acceptance's k on the same gated rows.
- **Step 0 (VOID check) in every process:** HumanEval/27, with the gate and the rule off, must reproduce
  results/ref2t token for token, and G's logits rows must equal E's bitwise at every check.
- **Cost:** two verify passes per check. The second one hits the cache the first one warmed, since the masked experts
  are still read. So R-56d's single-pass speeds roughly halve (code 52 / math 39 / instructions 18.5 tok/s, quiet,
  C 192).
- **No engine work from other sessions.** The gate is emulated in EvalSuite's block the way the map emulated it. The cf
  formula is the HPD-0 walk's.

## 3. Sets
- **Decision 4's 210 answers:** math 50 (34.3K REF2-T tokens), IFEval 100 (29.9K), HumanEval+ 60 (13.7K).
- **Information:**
  - multi 24: deviation on chat-like text; no gold.
  - agent 6 (bash17, bash18, bash19, edit00, edit01, edit02): 9-12K prefill, max_tokens 1,024; long-context turns and
    tool calls.
- **Drift:** every completion of >= 512 tokens (45 of REF2-T's 210 rows), plus the agent turns; first quarter vs last
  quarter.

## 4. Lines (to pre-register before any row)
- **Verdict:** decision 4 against REF2-T, as HC8 and REF2c-T were judged: per set, paired delta > -5 pt and exact
  McNemar p >= 0.05 (tools/d4.py).
- **Mission 7's bars on top:**
  - Blunders: E's p of the committed token < 0.01. The bar is <= 1 per 10K tokens on math + code, and none inside an
    agent tool call.
  - Drift: mean KLD(E || G) over the last quarter <= 1.5x the first quarter, pooled per set.
- **Outcomes:**
  - PASS: decision 4 passes and both bars pass.
  - BETWEEN: decision 4 passes and a bar fails.
  - KILL at this point: decision 4 fails. The next step is then gate 0.999 or R1(0.25), one more night.
- **Information:**
  - Deviation by set and cause, against RELAX_350's first-order numbers: code ~2.5 %, IFEval ~7-9 %, multi ~6 % with
    the 0.99 gate.
  - Kept-per-check gain against the predicted x1.107 code, x1.084 math, x1.17 IFEval.
  - The standard table.
  - Every must-fix deviation listed: a number, identifier, path, tool argument or code token.
- **What it cannot say:**
  - No t 0.3 cut (as R-56d and the walks): 2-4 % of rows sit in a different check than in the engine.
  - Chat uses DFlash2, not the deployed MTP chain. The chat rule R2(top-2, 0.25) is a later night, on its own
    drafter.
  - No tok/s.

## 5. Prep and the night
- **Today (CPU, after the 19:25 wake):**
  - `pending/relax.py`, the loop. It moves into evalsuite/ only after runner 38 ends, because runner 38's pin hashes
    evalsuite/*.py.
  - `tools/relax_report.py`: the deviation, blunder, drift and kept tables, tested on synthetic logs.
  - The PREREG.
  - Runner 40, drafted.
- **Tue 3 Nov by day, after runner 38 ends (08:30 at the latest):**
  - Wire `model.block`'s hook and `evalsuite gpu relax`.
  - GPU smoke under a share lock, ~20 min, ~30 GB: step 0, plus 3 prompts per arm (gated rows and rule accepts counted;
    E = G bitwise with the gate and the rule off).
  - Tue night belongs to FLOOR: nothing queued.
- **Proposed night: Wed 4 Nov.**
  - `r1g99` first: ~2-2.5 h quiet at C 128, ~2.5-3 h shared. Then `g99`, the same length.
  - ~4.5-5.5 h in all, peak ~30-33 GB (R-56d: 33 GB at C 192). It fits beside the IFX leftovers if NewIdeas orders
    them first.
  - Verdict Thu morning.

## 6. Revision Mon 18:4x: Fable's three arms adopted (cloud task 5, "the 350 mode"; relayed by the orchestrator)
Fable's point: `r1g99` is a stretch mode (~2.4 % of code tokens deviated). Test the mode that could ship first: the
int4 twin's level, <= ~1.5 % of code and <= 5 % of chat tokens. Adopted. Sections 1, 2 and 5 change as below; the lines
in section 4 stay.

- **Arms.** Each arm runs in its own process. The thresholds are on DFlash2's cf, as in section 1.
  - **A, the twin's level** (runs first):
    - HumanEval+, math, multi and agent: the cf gate at >= 0.999 plus the copy gate, with exact acceptance.
    - IFEval: R1(0.5) and no gate.
    - Fable's prediction: ~1.4 % deviated tokens on code, ~3.7 % on IFEval.
  - **B, the stretch** (about `r1g99`, layered on A):
    - HumanEval+ and math: the cf gate at >= 0.99 plus the copy gate, with R1(0.5).
    - Its chat setting is A's, so B's IFEval rows are A's.
    - Fable's prediction: ~2.2-2.5 % on code.
  - **C, the chat gate alone:** IFEval with the cf gate at >= 0.999 plus the copy gate, exact acceptance.
  - **Control** (first in the night; it is also step 0):
    - HumanEval+ 60, plus the first 10 math and the first 20 IFEval prompts, with the gate and the rule off.
    - It must reproduce results/ref2t token for token, and it is the tok/s denominator.
  - **HC8 is left out** (Fable: all 5 of its changed answers were losses). No arm carries it.
- **Three decision-4 verdicts from the three arms:**

  | Mode | math | HumanEval+ | IFEval |
  |---|---|---|---|
  | A | A | A | A |
  | B | B | B | A |
  | the gate alone (0.999 + copy) | A | A | C |

  The gate-alone mode is Q1's same-scores test (DAY9 #1's open step), and it replaces `g99`.
- **The copy gate:**
  - A row is inside a copy when the 8 tokens ending at its input token occur earlier in the window. The window is the
    prompt, the committed answer and this check's earlier inputs.
  - It applies to the rows the cf gate leaves exact. SAFARI 2.2 (b): the 125B is sure on 99.7 % of such rows.
  - The log records which gate fired on each row, so deviations split by cause: cf gate or copy gate.
  - If the copy gate is not ready by Wednesday, A runs without it (Fable's fallback).
- **Live tok/s, bytes only:**
  - On a gated row, ranks 2-9 point at the row's own rank-0 expert at weight 0, instead of at their own experts at
    weight 0.
  - The numerics are those of Q1's masking: every weight-0 slot adds an exact zero.
  - But the 16-row id union posted to the loader shrinks, so fewer experts stream from SSD.
  - Compute does not shrink, because the fused GLU still runs 10 slots per row. This is Fable's "bytes only" case.
  - Tuesday's smoke checks two things:
    - (a) this routing equals the masking bitwise on one prompt;
    - (b) fewer loads per check (the loader's counters) and a shorter check.
  - tok/s is paired against the control in the same harness at the same C. This harness is live16b's loop, not the
    engine's K-73 / per-layer-sizes build, so the ratio is the number that transfers.
- **The exact reference moves out of the timed run:**
  - Each arm runs G alone. Per check, it logs the 16 input ids, cf, the gate flags, G's argmax per row, k with exact
    acceptance, k under the rule, and G's top-64 log-probs on the committed rows.
  - A REF2-T replay then feeds each logged check's 16 rows to an exact cache and commits the same k+1, with no drafter
    and no timing.
  - From E's argmax, its p of the committed token and its top-64, the replay gives:
    - the deviation;
    - the blunders;
    - the cause of each deviation;
    - KLD(E||G), with the tails outside both top-64s lumped together.
  - Step 0 extends to the replay: on the control, its logits rows must equal G's bitwise (row hashes).
- **Night (Wed 4 Nov, quiet lock, C 256 = the default's cache, peak ~42 GB):**
  - Control ~15 min, A ~60 min, B ~25 min, C ~30 min: ~2.5 h with the model loads.
  - The replays (~2 h, C 192, ~33 GB) follow if the ~5 h slot allows; otherwise they run Thu by day under a share lock.
  - If the slot runs short, drop C first, then B.
  - Decision 4 comes Thu morning; mission 7's bars come after the replays.
- **Information, from Fable** (modelled gains halved, for the engine rather than this harness): mode A ~74 tok/s by day
  and ~80 at night; mode B ~84 and ~91.
- **Prep:**
  - At the next wake (CPU): `pending/relax.py` (the G-only loop and its log), `pending/replay.py`,
    `tools/relax_report.py` (tested on synthetic logs), the PREREG and runner 40.
  - Tue by day after runner 38: the wiring, then a ~30 min smoke under a share lock covering:
    - step 0 on HumanEval/27;
    - the replay bitwise;
    - the rank-0 routing equal to the masking bitwise, with the loads dropping;
    - 3 prompts per arm.

## 7. Addendum Mon 19:5x: live bytes per check, arm D, Kamen's new laptop schedule
- **The orchestrator (19:5x):** "please also log live expert bytes per check (and GB/s), next to tok/s".
  - Cloud Fable's task 9 (~/FableGod/CLOUD_ANSWER_EXACT350.md) finds no exact 350.
  - A 16-row check moves ~10.8 GB (expert union 7.38 + dense 3.44).
  - The gate is the only lever that removes bytes, and its first decisive measurement is "Wednesday's gate test, with
    live bytes per check".
- **Logged per check, every arm and the control:**
  - Expert union bytes: per layer, the unique ids the 16 rows post (the gated rows' dropped ranks fold into their rank
    0), times each tier's bytes per expert from the pools (3-bit base for every rank, plus the 1-bit side for ranks
    0-1). The ids come from model.block's existing capture.
  - SSD bytes: this process's disk reads (`diskread_bytes()`, rusage), plus the loader's misses and io_s (`st.stats()`).
  - The draft, verify and commit times.
  - GB/s, two ways: (expert union + dense) / verify time, and SSD bytes / verify time.
  - Dense bytes = the non-expert weight bytes, summed once at load.
  - The report gives these per set, per arm against the control, and per kept token.
- **Arm D, added:**
  - HumanEval+ and math: the cf gate at >= 0.99 plus the copy gate, exact acceptance, no R1.
  - This is task 9's lever (gate 0.99; "R1 stays out").
  - It makes a fourth decision-4 mode, "gate only": D's math and HumanEval+, plus C's IFEval (0.999, exact acceptance).
  - ~25 min G-only plus ~25 min replay.
- **Drop order if the slot runs short:** B first (R1, which task 9 keeps out), then C, then D. A and the control always
  run.
- **Kamen's schedule (RULES.md, 19:3x):** big tests Mon-Fri 00:30 until at least 15:00, small tests only from then
  until 00:00.
  - "Wednesday night" becomes Thu 00:30-15:00.
  - Tuesday's smoke fits Tue 08:30-15:00, after runner 38's 08:30 deadline.
  - If the smoke passes and the queue allows, the arms could instead take Wed ~10:30-15:00, after FLOOR (Wed 00:30 to
    ~10:30). NewIdeas orders the queue.
  - Runner 38 is unchanged: it starts at GO (BigMoE's, after 01:00) or at 02:00, both inside the window; its deadline
    is 08:30.

## 8. CPU prep done (Mon 21:05): the PREREG is docs/PREREG_relax_2026-11-02.md
- The copy gate's source is narrowed to the prompt (COPYGATE's rule 1: copies of the answer itself are where precision
  drops).
- Agent rows keep their items' 128 tokens.
- Code:
  - `pending/relax.py`, `pending/replay.py`, `pending/relax_core.py`; `tools/relax_wire.sh` moves them into evalsuite/
    after runner 38.
  - `tools/relax_report.py`; its `--selftest` passes.
  - `pending/test_relax.py`: the rules, the bytes, the KLD, and route == mask on a toy combine. All pass.
  - `tools/relax_smoke.sh` + `tools/relax_smoke_check.py`.
  - `tools/day_runner40.sh` (a draft).
- Keys: `results/relax/keys_<arm>.txt` and `smoke_keys_<arm>.txt`.

## 9. Merged with cloud task 14 (Mon 21:3x; orchestrator 21:22: "reconcile it with PREREG_relax's arms and send NewIdeas the merged order")
Cloud 14 (~/FableGod/CLOUD_ANSWER_FASTMODE.md), in short:
- The code fast mode is the gate, plus the comment loophole and HC8 riding along; R1 goes last.
- Chat gets R1 / R2, and the gate stays code-only.
- Its test plan: A gate alone (the control), B the mode, C + R1; chat D (8-row, exact: BigMoE's), E + R1, F + R2,
  G + gate (optional). Drop G, then C.
- The contract is task results; deviation is a diagnostic.

**Merged arms** (they replace sections 6-7's A / D / C / B before any row; slot Thu 00:30, after BigMoE's nightly
combined run):

| arm | sets | setting |
|---|---|---|
| control | HumanEval+ 60, 10 math, 20 IFEval | exact; step 0, and the tok/s and bytes denominators |
| A, the gate alone (cloud A) | HumanEval+, math, agent | gate 0.99 + copy, exact acceptance (was D) |
| B, the mode (cloud B) | HumanEval+, math, agent | A + HC8, + the comment loophole on HumanEval+ |
| E (cloud E) | IFEval, multi | R1(0.5), no gate |
| F (cloud F) | IFEval, multi | E + R2 (R1 or R2 accepts) |
| C (cloud C) | HumanEval+, math | B + R1(0.5) on every code token |
| G (cloud G) | IFEval, multi | E + the gate (0.999 + copy) |

- **The comment loophole** is Fable mission 25's engine rule (COMMENT_LOOPHOLE.md): R2(top-2, 0.25 p1) on rows whose
  draft sits inside a '#' comment of the answer's python fence. Never on docstrings, strings or code, never on a
  tool-prefixed comment, never in the code's first two lines.
  - My two cuts make it stricter: the row must already be inside the comment before its draft (so a drafted '#' is
    never relaxed), and a draft starting with a newline stays exact.
- **The drop order** is G, then C. HC8 is in B and C only, as cloud 14 has it. If HC8 ships globally, chat with HC8 is
  untested; an E8 arm (E + HC8, ~30 min) can be added on request.
- **The ship check** is B (HumanEval+, math) + E (IFEval), by decision 4 against REF2-T per set. Information modes:
  A + E, C + E, B + F, B + G.
- **What changes in the harness:**
  - HC8 per arm: `hc8_fake` before `ref2.apply`, and step 0 compares against results/ref2th8.
  - R2 needs the draft's rank on its row.
  - Per-row rules: exact, R1, R2, the comment rule.
  - An incremental fence + comment scanner.
  - Arms with no gate and no HC8 (E, F, the control) need no replay: their rows are exact, so G = E.
- **Time:** G arms ~3 h, replays of A, B, C and G ~1.6 h; ~4.6 h in all.
