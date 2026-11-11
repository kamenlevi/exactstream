# PREREG: RELAX, quality-defined modes on the 16-row live loop (Mon 2026-11-02 21:0x; revision 1 at 21:3x, revision 2 Tue 02:44, revision 3 Tue 07:00; before any row)

Source: the orchestrator's requests of 16:15, 18:3x, 19:5x and 21:22, built on:
- Fable's RELAX_350 and mission 25 (COMMENT_LOOPHOLE.md);
- cloud task 5 ("the 350 mode"), task 9 (exact 350: the gate is the deciding lever) and task 14 (the fast-mode test
  order).

The plan is docs/PLAN_relax_2026-11-02.md sections 6-9. **Revision 1** merges cloud task 14's order: the arms and modes
below replace the first version's A / D / C / B, the gate stays code-only except in the optional arm G, and the
contract is task results (deviation is a diagnostic). No row existed at either version. This file fixes the arms, the
rules and the lines before any row exists. None of these modes is exact, so adopting one is Kamen's call.

## Question
Does cloud 14's code fast mode keep REF2-T's scores (decision 4) without blunders or drift? So do its chat rules, and
what does each buy in kept tokens per check, expert bytes per check and tok/s?
- **The code fast mode:** the gate (top-2 experts on drafter-sure rows, plus the copy gate), the comment loophole and
  HC8.
- **The chat rules:** R1, R2 and, optionally, the gate on chat.

## Arms
One process per arm. Code: `pending/relax.py`, which moves to `evalsuite/relax.py` once runner 38 ends, unchanged
except for its location. Keys: `results/relax/keys_<arm>.txt`.

| arm (cloud 14) | sets (keys) | gate | acceptance | HC8 |
|---|---|---|---|---|
| control | HumanEval+ 60, the first 10 math, the first 20 IFEval, AT14 (104) | off | exact | off |
| A, the gate alone (A) | HumanEval+ 60, math 50, agent 6 (116) | Q1-UNION, cf >= 0.9 + copy | exact | off |
| B, the mode (B) | HumanEval+ 60, math 50, AT14 (124) | Q1-UNION, cf >= 0.9 + copy | exact, + the comment rule on HumanEval+ | on |
| E (E) | IFEval 100, multi 24 (124) | off | R1(0.5) | off |
| F (F) | as E | off | R1(0.5) or R2(top-2, 0.25 p1) | off |
| C (C) | HumanEval+ 60, math 50 (110) | Q1-UNION, cf >= 0.9 + copy | R1(0.5), + the comment rule on HumanEval+ | on |
| G (G, optional) | as E | cf >= 0.999 + copy | R1(0.5) | off |
| T3ALL (NewIdeas) | IFEval 100, HumanEval+ 60, math 50 (IFEval first) | off | exact | off; tiers 3333333333 |

- "math" is decision 4's 50 (math500 25 + gsm8k 25).
- The agent set is bash17-19 and edit00-02, with 9-12K-token prompts and the items' own max_tokens of 128. Every other
  prompt runs to its item's max_tokens.
- The order is control, A, B, E, T3ALL, F, C, G. If the slot runs short, drop G first, then C, then F.
- **T3ALL, revision 2** (NewIdeas 02:42, "your call"): plan 3333333333, every rank at 3 bits (MoonShots' M-26 sidecar
  drop). It is the largest priced FAST SSD lever: code 24.5 MB per kept token against 46.3, and 11.8 with Q1.
  - Quality only; no freed-RAM speed is part of the arm.
  - Its plan has no exact rows, so its step 0 checks that the plan took effect (bigmoe.tier PLAN 3333333333, no 4-bit
    ranks). Its replay checks itself on the control's step-0 rows, and E is REF2-T (4433333333).
  - In this harness the rank sort stays on (the combine's slot order); BigMoE's own uniform path keeps the argpartition
    order. That differs in float order only, so it doesn't matter for a quality arm.
- Cloud 14's D (8-row chat, exact) is BigMoE's measurement, not an arm here.

## The harness
- **Base:** R-56d's `ref2to_gate16`, which was bit-exact against REF2-T's AR (36/36 prompts, 15,489/15,489 rows):
  - snap/leap-faff8e5;
  - tiers 4433333333 / 3c / C_l 112;
  - BIGMOE_TIER_COMBINE=1 and BIGMOE_OOO=1;
  - BIGMOE_MAXK=160 on build_next.

  The loop, drafter and commit are live16b's: DFlash2 L1-20 + Leap's l120_ft, 16-row checks, keep k+1.
- **cf:** the drafter's selector softmax over its 16 candidates at the pick (MTP Head's d02b_walk `chain_cf`). The
  picks are unchanged.
- **The cf gate:** row j verifies draft j and uses cf[j]. A gated row keeps ranks 0-1 at the model's own weights, with
  no renormalization: Q1's "mass".
  - Its ranks 2-9 point at its own rank-0 expert at weight 0.
  - Numerically this is the same as Q1's masking (every weight-0 slot adds an exact zero), but the 16-row id union the
    loader posts shrinks.
  - The mechanism is a wrapper on the streamed switch's `__call__`, active only inside the 16-row verify.
  - Row 15, the bonus after 15 kept drafts, is always exact.
- **Revision 3, Q1-UNION** (NewIdeas' proposal; approved by the orchestrator 06:4x; in A, B and C):
  - It replaces Q1 at cf >= 0.99 with Q1-UNION at cf >= 0.9; the copy gate is unchanged. G keeps plain Q1 at 0.999.
  - Per layer of a check, U = every rank of the unsure rows + ranks 0-1 of the sure rows (cf-sure or copy-gated). A
    sure row keeps a rank-2-9 expert at its own weight iff the expert is in U; otherwise that slot points at the row's
    rank-0 expert at weight 0. No renorm. The loader posts U, which is Q1's union at the same threshold.
  - Rule: Breakthrough's ~/FableGod/decision_map/dm_q1u_masks.py; the union step is OEA's (arXiv 2511.02237).
  - Unit-tested (07:00, pending/test_relax.py): the graph's mask equals a numpy port of that rule on 200 random checks.
    It keeps every slot Q1 keeps, and it posts exactly Q1's expert union.
  - The map (03:19, SCOREBOARD fc32173), all-row flips, Q1-UNION 0.9 against Q1 0.99: code 0.98 vs 1.34 %, IFEval 4.92
    vs 5.42 %, multi 3.04 vs 3.69 %. Gated code rows flip 7/2678 = 0.26 % against 2/2211.
  - Bytes: the expert union per 16-row check on code is x0.618 against Q1 0.99's x0.720 (NewIdeas' CPU replay). SSD MB
    per kept token on code is -40.5 % (band45's price).
- **The copy gate:** a row the cf gate leaves exact takes top-2 when its last 8 input tokens occur in the prompt. Those
  8 tokens are the committed tokens plus this check's earlier drafts.
  - The source is the prompt only. This narrows the plan's "window". The reason is COPYGATE's rule 1 and its precision
    table (~/FableGod/copygate/GATE.md 1.2 c): copies of the answer itself are where precision drops (MBPP 75 harms in
    455 hits, HumanEval+ 6 in 26), while prompt copies carry the coverage (HumanEval+ 3,572 hits).
  - The 99.7 % "sure inside copies" is SAFARI H2's.
- **The rules on G's row (gated or not):**
  - R1(delta): draft j is kept while logit(d_j) >= max logit - delta.
  - R2(top-2, 0.25 p1): draft j is kept while it ranks first or second on its row (strictly larger logits counted) and
    p(d_j) >= 0.25 p1.
  - "R1 or R2" keeps a draft either rule keeps.
  - The bonus is row k's argmax. Every rule keeps every draft the exact rule keeps, and the log keeps both k.
- **The comment loophole (B, C; HumanEval+ only):** R2 on a row whose draft sits inside a '#' comment. This is Fable
  mission 25's engine rule (COMMENT_LOOPHOLE.md):
  - It is checked by an incremental scan of the answer's text, token by token. The scan agreed with mission 25's
    `regions()` on all 1,439 comment tokens of REF2-T's HumanEval+ answers, apart from my two cuts.
  - Only inside the answer's ``` fence, never in its first two lines, and never in docstrings, strings or code.
  - Never on a comment whose text, with the draft or with the row's argmax, is or could become a tool prefix: `# type:`,
    `# noqa`, `# pragma`, `# fmt:`, `# pylint`, `# mypy`, `# -*- coding`, a shebang.
  - My cuts: the row must already be inside the comment before its draft (a drafted '#' stays exact), and a draft
    starting with a newline stays exact.
- **HC8 (B, C):** Q3's 8-bit hyper-connection mixers and PLE, `hc8_fake` as model.load applies it for ref2th8, before
  ref2.apply.
- **Logged per check** (`results/relax_<arm>/checks/<key>.jsonl.gz`):
  - the 16 input ids, cf, the gate kind per row, G's argmax per row, k under the rule and k exact;
  - logit(d) - max and the draft's rank per row;
  - the rows open to the comment rule;
  - G's top-64 log-probs on the committed rows;
  - the expert-union bytes, as run and for the same rows exact (per layer: unique ids x the pool's bytes per expert);
  - the process's SSD bytes, the loader's misses and io_s;
  - the draft, verify and commit ms.
- **Step 0, in every G process:** humaneval/HumanEval/27 with the gate and the rule off must equal its reference row
  token for token: results/ref2t, or results/ref2th8 for the HC8 arms. Its row hashes go to `step0/`. A failure is VOID
  (exit 5).
- **The exact reference (`pending/replay.py`):** REF2-T replays each arm's logged checks on its committed prefix. It
  uses the same 16 input rows, top-10 on every row, the same k+1 commit, no drafter and no timing.
  - Per committed token: E's argmax, E's log-prob of the committed token, and KLD(E||G) against G's top-64. The tail
    outside G's top-64 is lumped into one term.
  - Its step 0 replays a G process's step-0 rows and must match them bitwise. A failure is VOID.
  - It runs for the arms with a gate or HC8 (A, B, C, G). In the control, E and F, G runs REF2-T's own numerics on its
    own prefix, so E = G: E's argmax is G's, the committed token's log-prob is G's (row max + logit gap), and KLD is 0.

## Modes and verdicts
| mode | math | HumanEval+ | IFEval |
|---|---|---|---|
| ship (cloud 14: "ship B and E Friday if they pass") | B | B | E |
| A_E (the gate alone, B's control) | A | A | E |
| C_E (+ R1 on code) | C | C | E |
| B_F (chat + R2) | B | B | F |
| B_G (chat + the gate) | B | B | G |
| T3ALL (all ranks at 3 bits) | T3ALL | T3ALL | T3ALL |

- **Decision 4 (the contract)** against REF2-T on the mode's 210 rows: `tools/relax_report.py --score` builds
  `results/relax_mode_<m>`, then runs `evalsuite report --variant relax_mode_<m> --decode --ref ref2t` and
  `tools/d4.py`. Per set the paired delta must be > -5 pt and the exact McNemar p >= 0.05. Each set's line also reads on
  its own: math and HumanEval+ judge the code arm, IFEval the chat arm.
- **Mission 7's bars, from the mode's arms:**
  - Blunders: E's p of the committed token < 0.01. The bar is <= 1 per 10K committed tokens on math + HumanEval+
    pooled, and none on the agent rows of A / B. Any agent blunder counts as inside a tool call; the rows are tool-call
    turns.
  - Drift: on rows of >= 512 tokens, mean KLD(E||G) in the last quarter of positions <= 1.5x the first quarter, pooled
    per set. A set with KLD 0 in both quarters (no gate and no HC8 there, so G = E bitwise) passes. KLD 0 in the first
    quarter but not the last fails.
- **Verdict per mode:**
  - PASS: decision 4 passes and both bars pass.
  - BETWEEN: decision 4 passes and a bar fails.
  - KILL: decision 4 fails.
  - VOID: a step 0 fails. The arm reruns.
- Deviated tokens are a diagnostic, not a line (cloud 14).
- T3ALL's code-only fallback (chat stays at 4433333333, NewIdeas) reads T3ALL's math and HumanEval+ lines: the chat
  rows would then be REF2-T's own.

## Predictions (information, not lines)
- **Deviated tokens along G's own prefix (diagnostic):**
  - Code: A ~1.0-1.3 % (map: Q1-UNION 0.9 at 0.98 % all-row flips); B about the same plus HC8's and the comment
    rule's share (mission 25: R2 in comments 1.37 % of positions, all in comments); C ~2.5 % (cloud 14's R1 share on
    top).
  - Chat: E ~3.7 % (RELAX_350's R1 on IFEval); F ~7 % (R2's 7.08 %, above the int4 twin's ~5 %); G ~E + 3-4 points.
- **Kept per check, real** (first-order halved):
  - E on IFEval x1.05-1.10;
  - F x1.10-1.18;
  - the comment rule on HumanEval+ ~x1.03-1.06;
  - C's R1 on code ~x1.05.
- **Expert union per check on code:** x0.618 with Q1-UNION 0.9 (NewIdeas' replay), plus a few points from the copy
  gate.
- **Copy gate coverage** (measured on CPU on REF2-T's own answers, Mon 21:06): rows inside a prompt copy are 53.8 % of
  HumanEval+ rows, 2.3 % of math and 1.4 % of IFEval. Most of these are cf-sure already. SAFARI's estimate for the
  added coverage on HumanEval+ is +3-8 points.
- **tok/s against the control** (bytes only, this harness, halved): A on code x1.08-1.15; B a few % more (HC8 ~-2-3 ms
  per check); C +~5 %.
- **Cloud 14's engine numbers** (by day / night): gate 0.99 ~96 / ~110 tok/s; + R1 ~100 / ~114.
- **HC8's own run** (decision 4 against REF2-T): math -4.0, IFEval -1.0, HumanEval+ -3.3, all 5 changed answers losses.
  B and C carry that on top of the gate.
- **T3ALL** (Breakthrough's map, Mon night):
  - code BETWEEN: +0.59 pt all-row flips over the 4-bit control, 0/2922 on sure rows;
  - chat +4-5 pt (mid rows ~5 % against 0.4 %, ~10x), so IFEval is the risk.
- **My P(PASS):** ship 50 %, A_E 65 %, C_E 35 %, B_F 30 %, B_G 40 %, T3ALL 30 %; T3ALL's code-only lines 60 %.

## Information
- **AT14, the agent-turn read-out** (NewIdeas ~01:00: "read-out only, no pass line"): dec_agent's 14 turns with
  9.1-12K-token prompts (results/relax/keys_at14.txt), in the control (exact) and in B (the mode). B's agent set is
  AT14; A keeps its 6.
  - Per turn: prefill s, first-check ms, decode s, answer tokens, turn tok/s = answer tokens / (prefill + decode),
    prefill share, and TTFT. TTFT equals the prefill time, because the first token comes from the prefill's logits.
  - Answers cap at the items' 128 tokens; real agent answers run longer. The same keys go to BigMoE for its engine.

For each arm and set:
- deviated tokens by cause: rule; rule in a comment; gate (cf); gate (copy); other (an ungated row whose context
  differs; with HC8, every row);
- the must-fix list (numbers, and any code or agent token);
- the gated share of rows;
- kept per check against the exact rule on the same checks;
- union and SSD GB per check, and per kept token;
- GB/s, defined as (union + dense) / verify;
- tok/s, paired against the control on the keys both ran;
- the standard table of the decode report.

## What it cannot say
- **No t 0.3 cut**, as in R-56d and the walks: 2-4 % of rows sit in a different check than in the engine.
- **Chat runs DFlash2's drafts in 16-row checks,** not cloud 14's MTP chain in 8-row blocks, so the R1 / R2 decisions
  on IFEval are DFlash2's: a per-token proxy.
- **Chat runs without HC8.** If HC8 ships globally with B, chat + HC8 is untested; an E8 arm (E + HC8, ~30 min) can
  be added on request. HC8's own run covered chat: IFEval -1.0, 1/0.
- **tok/s is EvalSuite's harness**, without the engine's K-73 / per-layer sizes, and bytes only: compute does not
  shrink. The ratio against the control is the number that transfers.
- **Agent rows stop at 128 tokens.** Drift comes from the 45 REF2-T rows of >= 512 tokens, plus whatever G writes long.

## Runs
- **Smoke** (share lock, ~45 min, ~33 GB at C 192 with "MEM 34 GB" in the reason), after runner 38 ends and
  `tools/relax_wire.sh` has moved the code into evalsuite/.
  - Keys: `results/relax/smoke_keys_<arm>.txt`, with output tag `smoke`. Every arm runs; A runs again with
    EVALSUITE_RELAX_GATE=mask (tag `smokemask`); then the replays of the control, A and B.
  - Checks (`tools/relax_smoke_check.py`):
    - step 0 in each process (B and C against REF2-T + HC8);
    - A's route rows equal its mask rows bitwise (EVALSUITE_RELAX_HASH=1 logs every committed row's hash);
    - the replay's step 0 matches bitwise, and the control replays bitwise on every row;
    - the gate shrinks the union on A, B, C, G;
    - HC8 is on in B and C.
  - NewIdeas: Tue by day after IFX and BigMoE's small screens, if it can start by ~14:20; else Wed by day.
- **Night** (quiet lock, C 256): Thu 00:30, after BigMoE's nightly combined run (GO from BigMoE).
  - Control, A, B, E, T3ALL, F, C, G: ~4.3 h with loads (AT14 included).
  - Then the replays of A, B, T3ALL, C and G at C 192: ~2.8 h.
  - If the slot runs short, drop G first, then C, then F.
- **Report:** `tools/relax_report.py --score`, and a RESULT doc.

## Smoke result (Tue 10:21-11:11): PASS (results/relax_smoke.txt)
- **Step 0** equalled its reference in every G process: REF2-T, or REF2-T + HC8 for B and C. T3ALL's plan was in
  effect.
- **The rank-0 routing equals the masking bitwise** (A against Amask: 271/271 committed rows).
- **The replay:** step 0 bitwise in every replay; the control replayed bitwise on every row (162/162) with no deviation.
  - Replay B first went VOID: B's own gate-off rows are HC8, not REF2-T. The replay now checks on the control's (fixed
    11:0x); the re-run gave 57/57.
- **Gates engaged:** the union shrank on every gated check of A, B, C and G. HumanEval/45: union 8.61 -> 4.06 GB per
  check, misses 1018 -> 436 (Q1-UNION 0.9).
- **Rules engaged:** R1 / R2 kept more than the exact rule (E 7, F 38, C 3 checks); the comment rule was open on 25 / 4
  rows of B / C; HC8 was on in B and C.
- **Information from 2-3 prompts per arm** (not a result): F's R2 turned ifeval/1675 into a 967-token answer that kept
  offering alternatives (REF2-T 45, E 126). The night's standard table (length, hesitation) will show whether that is a
  pattern.
- **T3ALL's bytes and speed in this harness are not representative:** the sidecar pool stays allocated but unused, so
  the base pool is smaller (SSD 2.56 against 1.71 GB per check). T3ALL is a quality arm, as registered.

## Amendment (Wed 2026-11-04 15:18): code changed after the smoke; RELAX's own modules did not; the night moves to Wed 23:30
- **RELAX's own modules are unchanged:** relax.py, replay.py and relax_core.py.
- **Three other evalsuite/ files changed after the smoke** (7d219cd, f66d69d):
  - model.py: new variants on snap/leap-52cf2c9, and an hc8r hook that only runs for those variants;
  - decode.py: the switch guard maps tier_nax_ooo to NAX6's counts;
  - ifeval/instructions.py: a non-letter letter_frequency character is counted as given. That affects 2 IFX prompts and
    none of the core 100 that RELAX's IFEval rows use.
  - None of these touch RELAX's arms. Every RELAX process's step 0 (HumanEval/27 == its reference row) still guards it.
- **The night:** runner 40 starts Wed 23:30 (the orchestrator), not Thu 00:30. No BigMoE run precedes it now, so
  EvalSuite touches results/relax/GO at the orchestrator's order. BigMoE's Thu-morning ticket follows RELAX_DONE.
- **18:05:** evalsuite/live16b.py gained Part 2's arm, a rows run per arm, the gate-ext rule for snapshots after Nov 4,
  and NAX6's count names. RELAX does not import live16b (relax.py, relax_core.py and replay.py have their own copy
  of the loop), so none of it is on RELAX's path. relax.py's build_next assert is unchanged.

## Amendment (Wed 2026-11-04 21:48, before any night row): a per-task line (the orchestrator 21:5x)
- **Why:** arXiv 2610.23818 reports relaxed verification giving 1.34x more accepted tokens per round but 1.54x longer
  completions. The orchestrator: "A relaxed arm that is faster per token but slower per task is a KILL for FAST."
- **Read-out, every arm:** on the keys it shares with the control (unpaused in both), per set and pooled:
  - tokens per task;
  - seconds per task (prefill s + decode s);
  - decode seconds per token;
  - all as ratios to the control, plus tokens per task against REF2-T's stored rows (results/ref2t) on the keys both
    hold. That last one has no timing noise.
- **The line,** for the relaxed arms (B: the comment rule; C, E, F, G): KILL for FAST when, pooled over the shared keys,
  seconds per token are below the control's (faster per token) AND seconds per task are more than 5 % above it.
  - The 5 % is the cross-process timing band, since every arm is its own cold-start process.
  - A mode with such an arm is KILL whatever decision 4 says. tools/relax_report.py prints it per arm and in the
    mode verdicts.
- **Shared keys with the control:** HumanEval+ 60 and the first 10 math (A, B, C), the first 20 IFEval (E, F, G), and
  AT14 (B).
- Nothing else changes.

## Amendment (Wed 2026-11-04 22:54, before any night row): C 160, and the start moves up
- **Memory:** NO-SWAP is 36 GB with nothing big running (the orchestrator 22:51). C 256 (gate 46 GB) and C 192 (38 GB)
  do not fit. BigMoE cancelled its 23:00 ticket for the same reason, and the orchestrator said "Start RELAX as soon as your
  memory gate passes ... use a lower C, your call."
- **So the arms and the replays run at C 160:** --need 30, gate 34 GB, BigMoE's by-day C.
  - Tokens do not depend on C: the gate decisions (cf, copy) and the numerics are the same.
  - The timing read-outs and the per-task line compare every arm with the control in the same run, at the same C. So
    they stay valid as ratios; they are not comparable with C 256 numbers.
- **Runner 40 now waits for memory** (NO-SWAP >= need + 4) before each chunk, so a memory wait never counts as a failed
  chunk.
- **The start moves to now (22:54),** because the GPU is free (no BigMoE 23:00 ticket). END stays Thu 08:10, so no arm
  chunk starts after 06:40 and no replay chunk after 07:10.
