# IDEAS_1117c_channels: take the commentary off the critical path (Lunatic, BREAK98c, Tue 17 Nov 11:1x-12:1x)

> **ADDENDUM 12:5x: the 2B pre-screen ran (Sandbox98 CHAN, 0609c6d). Quality FAILS by the 3-pt rule. The speed side
> holds. The faithful form is KILLED by its own comment-consistency count.** [K, ~/Sandbox98/results/LEVERS_A.md
> section CHAN result]
> - **Quality.** Pass@1 pooled 36.74 -> 33.71 (-3.03 pt, 25 / 33 discordant, McNemar p 0.36); HE+ -4.88, MBPP +0.00.
>   - 7 of HE+'s lost passes no longer parse: with the docstring opener masked, the 2B writes the docstring it copies
>     from the prompt as bare code.
>   - A guard (the trunk keeps a copied prompt docstring) would likely remove that part (HE+ ~16 / 17 discordant without
>     those 7). This is post hoc and untested.
> - **Speed.** Confirmed in real decoding:
>   - check-ending misses per code token x1.04 (the line was 1.5);
>   - x1.72 fewer sweeps (HE+ x1.71, MBPP x1.74);
>   - the 2B's own miss split matches the 125B's: 66.7 % comment + docstring;
>   - but rows per sweep are 48.6 (p90 64), and the answers are 11 % shorter (68.7k vs 76.8k tokens).
>   - In tok/s that is x1.54 per sweep at equal cost. Priced as in 3.4, it is x1.29 at this week's costs and x1.35 at
>     the bound [P], not the x1.7 the sweep count suggests.
> - **The blocker is the form, not pass@1.**
>   - Comment consistency is 25.6 % ('#' 14 %, docstrings 37 %): having written the comment, the model would write a
>     different next line in ~3/4 of slots.
>   - The code is identical to the coupled answer's in only 32 / 264 answers (line match 0.49).
>   - So the commentary steers the code. Splitting the channels does not take a side decision off the chain. It writes
>     a different program and attaches comments meant for the program the model would have written.
>   - Correction to 3.3: the head-side miss rates after comments were the wrong proxy. They measure drafting
>     difficulty, not the model's dependence on its comments. This also explains INERT's code drift.
> - **Verdict.**
>   - KILL for the faithful form (comments that describe the code they sit in), on the 2B. No 125B run is recommended.
>   - The only rescue left is comments written by infill AFTER their code (FIM-style, reading the trunk's next lines).
>     It is untested, it costs suffix rows plus a one-check lag per comment, and it ships comment-blind code. That
>     makes it an owner's call, not a lever to queue.

Lens: assumptions 5 + 6 (kept = per-position acceptance of one chain / tree; the answer is produced left to right, one
token per position). Marks: [K] measured here or in a named record, [P] priced from measured parts, [G] judgement.
Data: BigMoE's t30b cells + topk dumps (HE+ 24 answers, 6,375 tokens; MBPP 100 answers, 15,641 tokens), the same
chain-32 / chain-16 walks and lexer as lim800_slack.py. CPU only, ~1 min per screen.

## 0. The answer in seven lines

1. **Two laws bound every decode shape that keeps the full 125B as the per-token verifier [P]** (section 1).
   - Row law: every token that later text reads costs one 125B row.
   - Decision law: one sweep of the weights resolves at most one decision that later text depends on and the drafter
     did not foresee, plus what the sweep's own rows pre-cover (a tree).
   So assumptions 5 and 6 bind only through the answer's DEPENDENCY CHAIN. On that chain, best-first trees are already
   optimal (BREAK98b), and reusing rows after a miss is measured dead (section 4).
2. **The chain is not the whole answer.** Comments and docstrings hold most of the misses that end checks [K,
   shape_names.out]:
   - exact: 65.6 % (HE+) / 74.1 % (MBPP); CC(1): 63.6 / 72.5 %.
   - Comments are 13 % of HE+ tokens but 55 % of its check-ending misses.
   Today they sit on the critical path only because the code after them reads them.
3. **Idea 1, CODE || COMMENTARY.**
   - The code is one causal channel that never reads the answer's comments or docstrings.
   - Each comment / docstring is a branch forked from the code at its slot, decoded in the same sweeps.
   - Every token is still the 125B's, verified exactly within its own channel.
4. **Number** [K cells, P costs], under CC(1):
   - Sweeps: x1.67-1.87 fewer on HE+, x1.39-1.47 on MBPP. In the frame's units (chain-32), kept per sweep goes
     16.87 -> 28.2-31.6 on HE+ and 8.53 -> 11.9-12.4 on MBPP, at ~47 / ~36 rows per sweep.
   - Full cycle at this week's check costs: about x1.5 on HE+ and x1.3 on MBPP.
   - At the C6 + int8 bound with a 3 ms draft: ~860-960 HE+ / ~395-410 MBPP, against 568 / 288 coupled (chain-32).
5. **What it is not.**
   - Not COMMENT-FREE: comments stay, and the 125B writes them.
   - Not INERT: there is no head-written wording, so no comment loops, and the code can't drift on wording it never reads.
   - Not SoT / PAR-CHAT: the split is code vs commentary, priced on measured decision shares.
   - Not Hogwild: branches fork the code's state, so no state has interleaved writers.
6. **Decisive test: one paired quality run of the comment-blind code with branch comments** (section 3.5). The speed side
   is priced. The open questions are quality and the output form, and the form (the code never reads its own
   comments) is a product decision.
7. **Killed today [K]:**
   - Fresh-name / tokenization-equivalent acceptance: 0.4-0.8 % / 2.9-4.0 % of check-ending misses, x1.002-1.027.
   - Reusing post-miss rows (Jacobi, resync, renaming transfer).
   - The exact cousin of Idea 1: it doubles the rows for no net saving [P].

## 1. The two laws (why most 5/6-breaking shapes cannot pay) [P]

- **Row law.** A token that later text reads must be a 125B row:
  - its KV entry at every attention layer and its update of every GDN state need the full residual stream through
    that layer;
  - its verification needs the 125B's output on the row before it.
  No shape escapes this: not trees, Jacobi, out-of-order or parallel segments. A shape only changes which rows a sweep
  holds.
- **Decision law.** Call a decision unforeseen when the drafter's guess at that position is wrong.
  - The 125B makes an unforeseen decision on the row before it, given the exact prefix.
  - Text after that decision depends on it, so its rows can't be valid before the decision is known.
  - Only rows that pre-cover the 125B's choice can carry the text on within the same sweep, and that is a tree.
  So the number of sequential sweeps is at least the number of unforeseen decisions on the dependency chain, minus the
  covered ones. Two DEPENDENT decisions per sweep cost a second weight stream: >= 0.57 pass (BREAK98b). Per-sweep
  coverage is optimal under best-first: the head is calibrated, so E[kept] = sum(path_p) over the tree.
- **The escape the laws leave.** A decision that no later text reads is not on the chain.
  - Today every decision is on the chain, because every token is read by everything after it.
  - The laws do not require that. Cut a read, and the decisions behind it can run in parallel, under the row law and
    the decision law alike.
  - The question is which reads an answer can lose at quality. For code answers, section 2 says: the commentary.

## 2. Where the sequential decisions are [K, shape_names.out; token shares from lim800_slack.out]

| set, rule | check-ending misses | in COMMENT | in DOC | in CODE (+ STR) | tokens: COMMENT / DOC / CODE |
|---|---|---|---|---|---|
| HE+ exact | 337 | 54.6 % | 11.0 % | 34.4 % | 13.1 / 56.5 / 29.7 % |
| HE+ CC(1) | 250 | 48.8 % | 14.8 % | 36.4 % | |
| MBPP exact | 2,144 | 32.4 % | 41.7 % | 25.8 % | 19.6 / 37.7 / 41.5 % |
| MBPP CC(1) | 1,712 | 32.0 % | 40.5 % | 27.5 % | |

Check-ending misses per 100 tokens, exact:
- HE+: comments 22, code 6, docstrings 1. The HE+ docstring re-emits the prompt, so it's nearly free.
- MBPP: comments 23, docstrings 15 (a paraphrase of the task), code 9.

The 125B's wording inside comments is the costliest thing it writes, per token, by 2.6-3.7x over code. Under the decision
law, that wording costs sequential sweeps only because the code after it reads it.

## 3. Idea 1: CODE || COMMENTARY (the code never reads the commentary)

### 3.1 Mechanism

- **Trunk = the code channel.** The usual tree / chain + CC(1) decode, over a context of the prompt plus the answer's
  code only.
  - A comment slot is where the 125B's argmax on the trunk is a comment opener (`#` after code or at a line start, or a
    docstring opener).
  - At a slot, the trunk masks the opener and takes its best non-comment token.
- **Branches = the commentary.** At each slot, fork a branch that starts with the opener.
  - Its state is a copy-on-fork of the trunk's KV + GDN state at the slot.
  - It decodes the comment / docstring to its end (newline or closing quotes) with its own chain + CC(1), sharing every
    sweep with the trunk: one weight stream, more rows.
  - A branch reads the prompt plus the code before its slot. In the base form it does not read earlier comments.
- **Output.** Splice each branch into the code at its slot. Comments sit where the 125B wanted them, in its words.
- **Drafter.** One batched draft per sweep over the trunk plus the active branches (the per-pop fixed cost and the
  head's bytes are shared).
- **Engine.** Persistent branches across sweeps (tree support plus a branch table) and a row scheduler (trunk first,
  then branches by age).
  - Each live branch keeps its own GDN state + KV: ~0.1 GB per branch [G], 2-4 live at a time. That memory must fit
    the 45 GB line's margin.

### 3.2 Assumptions broken

- **6:** two or more positions of one answer advance per sweep, at different places in the text, each causal in its own
  channel.
- **5:** kept per check becomes the sum of several independent chains. A miss in a comment no longer ends the code's
  check, and the reverse.

### 3.3 Why today's walls don't apply

- **Kept** (own cap 16.26 per 32-row tree; the head lacks the 125B's decision; no cheap readout before the last
  layers). The idea makes no claim on per-chain kept: every decision is still a full-depth 125B decision. It takes
  about 2/3 of the decisions off the critical path; it does not make them cheaper.
- **Rows** (>= 0.15-0.25 ms per marginal row). Branches add +12-75 % rows per sweep (means 22-47 vs 16 / 32 coupled; table below). That is
  priced in at both the bound's 0.25 ms per row and this week's 16 -> 32-row slope of 0.75 ms per row.
- **Closed list.**
  - "Two decisions per pass" assumed dependent decisions. These are made independent by construction.
  - Partial channels, early readouts, repair, future-state rows and landmarks all try to predict a decision cheaply.
    This idea predicts nothing.
- **Prior kills.**
  - COMMENT-FREE (ruled cheating): comments stay.
  - INERT wholesale (quality KILL, comment loops + code drift): no head wording, and the code never reads comment wording.
  - PAR-CHAT / L1-32p (SoT, < 1.25 on chat; f(code) < 20 %): those split prose into points; this split is measured at
    2/3-3/4 of the code answers' decisions.
  - Hogwild (GDN can't take interleaved writers): each branch owns its forked state.
- **Plan transfer** (CORRECTED by the 12:5x addendum: the head-side numbers below were the wrong proxy; in real
  decoding, comments steer ~3/4 of the 2B's next lines). Do comments carry the code's next decision, so that cutting the read would add code misses? The
  head-side record says no [K, shape_channels.out]. On the first code line after a comment line, the head misses more,
  not less:
  - HE+ depth-0 3.3 % vs 1.2 % elsewhere (depth 0-15: 10.4 vs 8.7 %);
  - MBPP 3.3 vs 2.5 % (14.7 vs 12.7 %).

### 3.4 The number [K: shape_channels.out]

The coupled run is today's single walk. Channels: the critical path max(code chain, fork time + span checks), with rows
summed per sweep.

Two mappings of the code chain onto the coupled cells bracket the truth:
- strict: the coupled depth from the anchor; a chain stops if it would pass depth 30. Conservative across comment gaps.
- matched: the same drafting depth for the same target token.

| set | rule | walk | coupled checks (kept) | channel sweeps | fewer sweeps | rows / sweep mean (p90) | check-only tok/s at the bound | at this week's check |
|---|---|---|---|---|---|---|---|---|
| HE+ | CC(1) | chain-32 | 378 (16.87) | 226 / 202 | x1.67 / x1.87 | 47 (80) | 669 -> 994 / 1,107 | 351 -> 471 / 525 |
| HE+ | CC(1) | chain-16 | 527 (12.10) | 302 / 293 | x1.75 / x1.80 | 28 (49) | 630 -> 917 / 952 | 336 -> 466 / 483 |
| HE+ | exact | chain-32 | 460 (13.86) | 254 / 226 | x1.81 / x2.04 | 47 (84) | 550 -> 887 / 989 | 289 -> 421 / 467 |
| MBPP | CC(1) | chain-32 | 1,833 (8.53) | 1,317 / 1,262 | x1.39 / x1.45 | 36 (64) | 339 -> 463 / 485 | 178 -> 227 / 239 |
| MBPP | CC(1) | chain-16 | 1,971 (7.94) | 1,394 / 1,345 | x1.41 / x1.47 | 22 (38) | 413 -> 535 / 556 | 220 -> 271 / 281 |
| MBPP | exact | chain-32 | 2,249 (6.95) | 1,549 / 1,502 | x1.45 / x1.50 | 36 (64) | 276 -> 392 / 405 | 145 -> 193 / 199 |

(Pairs are strict / matched. Check costs: the bound is the C6 + int8 32-row 25.2 ms, 0.25 ms per row above 32 and
19.2 ms at 16 rows. This week's check is 36 ms at 16 rows plus 0.75 ms per row, i.e. ~48 at 32.)

**Full cycle** (the draft is per sweep, so fewer sweeps also cut the draft):
- This week, chain-16 + CC(1), with a 16.6 ms batched draft + 1.5 ms per sweep:
  - HE+ 224 -> 333-345 tok/s (x1.49-1.54);
  - MBPP 147 -> 189-196 (x1.28-1.33).
  - If branches add 20 % to each draft: x1.41-1.46 / x1.21-1.26.
- At the bound, chain-32 + CC(1), with a 3 ms draft + 1.5 ms:
  - HE+ 568 -> ~860-960;
  - MBPP 288 -> ~395-410.

Trees are not simulated here [G]. Today ~56-69 % of EAGLE-2-31's side rows cover comment wording (lim800_strict.out, HE+ / MBPP). In
the split, the trunk's tree spends them on code instead, so the gain should mostly survive on trees.

### 3.5 The decisive test (one; the speed side is priced)

**A paired quality run of CODE || COMMENTARY in plain decode.** It needs no engine work: run the trunk with comment
openers masked, run one short decode per slot for the branches, splice, then score.
- Pass line: EvalSuite's gate (CODE-ONLY tier 1, both orders, vs the same reference as CC(1)).
- The same run logs two numbers:
  - the code channel's check-ending misses per code token vs coupled; KILL the speed claim if > 1.5x;
  - comment consistency: in each branch, teacher-force the next code line and count the slots where the 125B, having
    written the comment, would not write that line.
- Cheap pre-screen first: the same run on the 2B in Sandbox98 (<= 5 GB, ~30 min). It predicted d2's fail (-6.06
  pooled).

### 3.6 Quality risk

- **Comment-blind code.** The code never reads its own commentary. For HE+, the docstring it skips is in the prompt
  anyway. Inline comments may hold a plan, and the head-side evidence (3.3) says they don't make the next line easier;
  whether the 125B's code needs them is what the run measures.
- **Comment / code mismatch.** A comment is written from the code before it, and the code after it never saw it. At
  plan near-ties the two can disagree, and a misleading comment is a real defect even when tests pass.
  - Guard: teacher-force the next code line in the branch (~10 rows per comment, same sweeps). Regenerate or drop
    the comment where the 125B disagrees.
- **Comments don't read earlier comments** (base form), so numbering or "step 2"-style continuity can break.
  - Variant: a branch waits for earlier branches, trading some parallelism.
- **Output form.** The text differs from the coupled greedy answer, so this is lossy and gated. And because the code
  never reads its own comments, it is a product decision, not only a gate.

## 4. Killed today, with numbers

- **FRESH-NAME acceptance** (accept the head's new local name where the 125B wants another new name: alpha-equivalent,
  so tests can't see it). 0.4-0.8 % of check-ending misses: x1.002-1.006 [K, shape_names.out]. KILL.
- **TEXT-PREFIX acceptance** (same characters, another tokenization). 2.9-4.0 %: x1.008-1.023, x1.011-1.027 with the
  above [K]. KILL.
- **Reusing rows after a miss** (Jacobi tails, resync drafting, a renaming map from the rejected branch). The tail is
  rarely right:
  - after a wrong token: 16 % / 12 % at the next position (NewIdeas' swap-only repair, WEEK98 07:4x);
  - after a near-tie divergence: 27.9 % of all later tokens (lu_resync.out).
  KILL for this lens.
- **The exact cousin of Idea 1** (the decoupled decode as the 125B's own draft, then a coupled verification of the
  spliced answer). Every token becomes a row twice. Matches end at the first comment-dependent token (resync 27.9 %
  after it). At the bound the verification pass costs about what the parallelism saves [P]. KILL by arithmetic.
- **MAP-window close calls** (accept a miss when the 125B's verdicts after it repay the deficit). The close-call band is
  small: R1(2) is only +2.4 % kept over R1(1), and d2 FAILed (-4.17 pt pooled, MBPP -6.00). A semantic error is followed by verdicts as
  confident as a stylistic one, so there's no new quality signal [G]. Not pursued.

## 5. What changes

- **Kept side:** the commentary is the largest lever in this lens.
  - x1.2-1.5 at this week's costs (HE+ x1.41-1.54, MBPP x1.21-1.33), x1.4-1.7 at the bound.
  - It is the only one found today that is not a +5-25 % information channel.
  - It composes with the check line (Leap) and the draft line (MTP Head + MoonShots), because it cuts the NUMBER of
    sweeps, check and draft alike.
- **550:** HE+ clears it at the bound with room. MBPP does not (~400): MBPP's code chain itself (9 check-ending
  misses per 100 code tokens) is the next wall.
- **It needs two things before any engine work:**
  - the owner's call on the form (comments kept, the code doesn't read them);
  - one paired quality run.

## 6. Reproducibility

- `~/BigMoE/.venv/bin/python3 -I shape_names.py` -> shape_names.out (~33 s, 0.19 GB peak).
- `~/BigMoE/.venv/bin/python3 -I shape_channels.py` -> shape_channels.out (~1 min, < 0.3 GB).
- Both exec lim800_slack.py's loader, lexer and walk (read-only), on BigMoE's t30b cells + topk dumps and the bank's
  ids (prompt rows aligned 24/24 and 100/100). shape_channels' coupled walk starts at anchor 0 (HE+ exact 13.86 vs
  13.77 from certify98's start).
- Costs:
  - the bound: C6 + int8 (BREAK98b);
  - this week's check: 36 ms at 16 rows / 48 ms at 32 (BREAK98c "realistic this week"), extended linearly;
  - draft: sync-free Q8H32 16.6 ms per tree-16 block (BREAK98c).
- Not modelled:
  - trees;
  - the expert-union growth from mixing code and commentary rows in one sweep [G: a few % of the check];
  - branch-table overheads.

## 7. TRIED lines (ready to paste; Lunatic, 2026-11-17)

- SHAPE-NAMES (fresh-name / tokenization-equivalent acceptance at misses: alpha-renaming and same-text tokenizations, which tests can't see) | wall: kept | line: 45 GB lossy | exact: no (semantics-preserving) | verdict: KILL: 0.4-0.8 % / 2.9-4.0 % of check-ending misses, x1.002-1.027 kept over exact or CC(1) (HE+ / MBPP chain-32) | 2026-11-17 | ~/FableGod/lunatic/shape_names.out | [lunatic]
- DECISION-SHARE (where the sequential 125B decisions sit in code answers) | wall: kept | line: all | exact: n/a | verdict: MEASURED: comments + docstrings hold 65.6 / 74.1 % of exact check-ending misses (HE+ / MBPP; CC(1) 63.6 / 72.5); comments 22-23 per 100 tokens vs code 6-9 | 2026-11-17 | ~/FableGod/lunatic/shape_names.out | [lunatic]
- CODE || COMMENTARY (the code channel never reads the answer's comments / docstrings; each is a branch forked at its slot, decoded in the same sweeps; all tokens 125B-verified in their channel) | wall: kept / critical path | line: 45 GB lossy | exact: no (a different answer; gated) | verdict: KILL (faithful form) on the 2B pre-screen (Sandbox98 CHAN 0609c6d): pass@1 -3.03 pt pooled (p 0.36; 7 HE+ parse failures from the masked copied docstring), speed confirmed (x1.04 misses per code token, x1.72 fewer sweeps, ~x1.3 tok/s after rows), comment consistency 25.6 % (comments steer ~3/4 of next lines; code identical in 32 / 264) | pre-screen numbers before it: [K cells] CC(1) x1.67-1.87 fewer sweeps HE+, x1.39-1.47 MBPP; [P] full cycle this week x1.41-1.54 HE+ / x1.21-1.33 MBPP, at the bound + 3 ms draft ~860-960 / ~395-410 vs 568 / 288 | decisive: a paired plain-decode quality run (2B pre-screen first) + the owner's call on the form | 2026-11-17 | ~/FableGod/lunatic/IDEAS_1117c_channels.md | [lunatic]
