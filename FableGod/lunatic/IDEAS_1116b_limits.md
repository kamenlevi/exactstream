# IDEAS_1116b_limits: where is the slack in LOSSY kept? (Lunatic, THINK_1116b)

> **CORRECTED Mon 16 Nov 18:5x (Lunatic, after NewIdeas' 18:2x correction): this file's INERT is the WHOLESALE arm, a
> recorded quality KILL** (TRIED CD16-DROP 11-06: MBPP12 8/12 vs 11/12, comment loops + code drift). The quality-safe
> guarded arms (T16 / TC16) are flat: x0.997-1.019 (TRIED FRONTIER#2).
> - **Void:** every kept and tok/s figure below that includes INERT (22.37 / 23.88 HE+, 17.13 / 18.08 MBPP, the 655-850
>   projections, INERT-WIDE). **LAT (Idea 1) is void:** under T16 a comment sibling is not waste.
> - **Still valid:** the exact and CC-only rows of Section 0, the answer-end ceiling (30.50 / 28.08), the
>   7.4 ms-draft point, and the Idea 4 / 5 kills.
> - Follow-up: IDEAS_1116c_drift.md (the drift wall).

Mon 16 Nov 17:5x-18:3x. Brief: ~/FableGod/deepdraft/PUSH800.md section 3, the limits lens. Desk + six tiny CPU screens
on the existing dumps (each < 0.3 GB, one process, 1-3 min); outputs are the `.out` files beside each script:
- `lim800_slack.py`: kept under acceptance rules; where the misses that end checks sit;
- `lim800_strict.py`: the same with the frame's strict span ends; tree headroom; where EAGLE-2's side rows sit;
- `lim800_spanend.py`: a strict vs lenient span-end check;
- `lim800_rerank.py`: two zero-row re-rankers; headroom with lenient ends;
- `lim800_wide.py`: wide checks over INERT spans, priced at Leap's C6 + int8 bound;
- `lim800_treewaste.py`: side rows by span, lenient.

Data: BigMoE's t30b cells + TOPK + tree dumps (`~/BigMoE/data/deepdraft/ceiling/`, HE+ heplus24 = 24 windows / 6,375
answer tokens, MBPP mbppbank = 100 windows / 15,641), the certify98 chain-32 walk, the 125B's tokenizer.

Marks: [K] measured on those dumps (dev replay, the same data and walk as the frame's 16.91), [P] arithmetic on [K],
[G] judgement.

Terms:
- **INERT** = the frame's #13: any draft accepted inside comment / docstring / prose spans; **span ends** (the newline
  that ends a comment, the closing triple quote) and code rows stay strict. A small lexer tags each answer token's span
  from the decoded text (checked by eye on four answers).
- **CC d1**: a code-row draft within 1 nat of the 125B's top-1 is accepted.

**Caveats:**
- Dev data overstates INERT: TRIED 1066 has x1.47 tok/s on dev HE+ vs x1.30 fresh, at 32 rows. I use ~0.89x for
  fresh-like [G].
- After an accepted miss, the walk follows the text's own cells, the same approximation as every close-call replay.

## The answer in six lines
1. **The frame's own lossy rules already move kept a lot on dev:** CC d1 + INERT (span ends strict) keeps **22.37** per
   32-row check on HE+ and **17.13** on MBPP [K]. That is x1.32 / x1.80 over the frame's 16.91 / 9.50, and ~655 / ~500
   tok/s at 85 % of the C6 + int8 bound with a 3 ms draft [P].
2. **What saturates next is not entropy.** After those rules, the misses that end a check are 74 % (HE+) / 82 % (MBPP)
   CODE rows plus 25 % / 17 % span ends [K]. About half of the code ones sit where the 125B is >= 4 nats sure: drift on
   forced code tokens, not ties. Widening the close call can't buy it at quality: CC d2 + INERT 24.90 / 19.31 even with
   lenient ends, CC d4 alone 20.50 / 12.75 [K].
3. **The 32-row ceiling is set by answer ends:** 30.50 (HE+) / 28.08 (MBPP) with every miss fixed [K]. The frame's
   QAT-head column (kept 27.6-30.8 at a 7.4 ms draft) is at 90-101 % of that ceiling. At 32 rows, 775 needs the
   <= 3 ms draft (kept 24.2-26.5, 79-87 % of the ceiling).
4. **The lossy rules turn most of the tree into waste:** 56 % (HE+) / 69 % (MBPP) of EAGLE-2 tree-32's side rows at
   walk starts are alternative WORDING inside comments / docstrings, where INERT accepts any draft [K]. Spending them on
   code rows and span ends has a kept ceiling of **x1.20-1.28 (HE+) / x1.30-1.45 (MBPP)** [K upper bound]. That is
   Idea 1, the best one.
5. **INERT spans need no verification, so they can be checked wide.** Span-sized checks of <= 128 rows give x1.16
   (HE+) / x1.07 (MBPP) tok/s at the bound [P, lenient ends] (Idea 2, COPYRUN generalised).
6. **Killed today** [K]: a scope re-rank and a syntax veto on the head's top-8 (zero rows, zero bytes) buy nothing; the
   head never mis-closes a bracket.

**No single lever I found gives x1.3 at quality on both sets.** Idea 1's ceiling reaches it on MBPP only. The stack Idea 1
x Idea 2 projects HE+ ~805-850 tok/s on dev and ~715-760 fresh-like, at 85 % with a 3 ms draft [P x G]; MBPP stays
<= ~650.

## 0. The lossy kept map (what bounds saturate) [K, lim800_slack.out + lim800_spanend.out + lim800_strict.out]
Chain-32 walk, kept per 32-row check:

| rule | HE+ | MBPP |
|---|---|---|
| exact | 13.77 | 6.92 |
| CC d1 / d2 / d4 / d6 | 16.78 / 18.21 / 20.50 / 21.61 | 8.52 / 10.14 / 12.75 / 15.00 |
| INERT alone (lenient span ends) | 22.21 | 16.73 |
| CC d1 + INERT, lenient span ends | 23.88 | 18.08 |
| **CC d1 + INERT, span ends strict (the frame's rule)** | **22.37** | **17.13** |
| CC d2 + INERT (lenient) | 24.90 | 19.31 |
| + accept whitespace-only code misses / string contents (lenient) | 23.97 / 23.97 | 18.19 / 18.17 |
| CC d1 + INERT strict + fix code misses where m >= 4 | 24.61 | 20.39 |
| CC d1 + INERT lenient + fix code misses where m >= 8 / 4 / 2 / 1 | 25.10 / 26.79 / 28.59 / 29.93 | 19.67 / 21.63 / 24.48 / 27.01 |
| every miss fixed (the answer-end ceiling) | 30.50 | 28.08 |

**Span shares** (answer tokens):
- HE+: CODE 29.4 %, COMMENT 13.1 %, DOC 56.5 %. The answer re-emits the prompt's docstring (TRIED 1471).
- MBPP: CODE 41.0 %, COMMENT 19.6 %, DOC 37.7 %.

**Misses that end a check after CC d1 + INERT strict:**
- HE+ (n 128): CODE 74.2 %, span ends 25.0 %, strings 0.8 %.
- MBPP (n 581): CODE 81.6 %, span ends 17.4 %, strings 1.0 %.
- Code misses by class (lenient run):
  - HE+: identifiers 44 %, punctuation / operators 39 %, newline / indent 10 %, numbers 6 %. Where the 125B is
    >= 4 nats sure: 50 % / 46 % / 50 % / 100 %. Median depth 13-16.
  - MBPP: identifiers 48 %, punctuation 38 %, newline 9 %, numbers 3 %. >= 4 nats sure: 41 % / 48 % / 33 % / 47 %.
    Median depth 7-10.

**Readings:**
- On dev HE+, 775 at 85 % with a 3 ms draft needs kept ~26.5. CC d1 + INERT strict + a drafter that never misses forced
  code tokens (m >= 4) reaches 24.61; span ends and code misses at 1-4 nats must also go (trees, Idea 1).
- MBPP is out of reach this week.
- The slack the lens asked about is **forced code tokens the head gets wrong at depth, plus where comments end**, not
  the entropy of code positions. CC d1 already takes the near-ties.

## Idea 1 (best): LOSSY-AWARE TREE (LAT): spend side rows only where the lossy rules can't help
**Mechanism:**
- EAGLE-2 places side rows (a second node at a depth = an alternative token) where the head is unsure. On code answers
  that is mostly comment wording.
- Under INERT, a wording row inside a comment / docstring is accepted whatever is drafted, so a sibling there can never
  add kept. Under CC d1, a sibling at a code near-tie mostly duplicates the close call. The frame measured that
  overlap: tree-32 adds only 16.91 vs 16.78 over chain-32 with CC d1 (tree_cc32, additivity 0.57).
- LAT selects the 31 nodes with two changes to EAGLE-2:
  1. no side node, and no subtree, for alternative wording inside a comment / docstring / prose span. Siblings stay
     allowed at span-end decisions (a candidate that ends or continues the span) and on code rows.
  2. optionally, no side node where the head's top-2 are within ~1 nat.
- The freed rows go to the next-best pops: code-row siblings, span-end siblings, deeper main-chain nodes.
- The span tag is a CPU lexer state the INERT rule needs anyway.

**Numbers [K, lim800_strict.out]:**
- Side rows at walk starts: 36 % (HE+) / 54 % (MBPP) of EAGLE-2-31's nodes. Of those, **56.1 % / 68.8 % are INERT
  wording** (waste), 11.1 % / 6.9 % span-end decisions, 32.4 % / 24.0 % code. That is ~6 (HE+) / ~11.5 (MBPP) of the
  31 tree rows per check.
- Headroom, CC d1 + INERT strict as the base: rescue every code + span-end miss whose 125B token is the head's 2nd
  choice and kept goes 22.37 -> 26.90 (x1.20) on HE+ and 17.13 -> 22.28 (x1.30) on MBPP. With a top-4 sibling:
  x1.28 / x1.45. With a top-8: x1.32 / x1.53.
  - This is an upper bound: it assumes a sibling where each miss happens, with its continuation.
  - [G] LAT captures a third to a half of it: **+7-14 % kept HE+, +10-22 % MBPP.**

**Tag:** adapted (EAGLE-2 dynamic trees + typical / close-call acceptance). New here, as far as TRIED shows: tree
selection conditioned on what the acceptance rule already covers (INERT span state, CC). The CC overlap was measured
(tree_cc32); re-spending the freed rows was not. Greps: lossy-aware, tree x inert, side rows in comments.

**Lossy / exact:** lossy mode (the tree itself is exact). **Quality risk: none new.** LAT changes which drafts the 125B
checks, never what is accepted. Its quality is CC d1 + INERT's, which EvalSuite gates already.

**What kills it:** the head's confidence doesn't locate code / span-end misses, so the re-spent rows land where nothing
misses and the gain is < +3 %.

**Screen (<= 10 min CPU, NewIdeas' tree_cc32 machinery):**
- Data: BigMoE's `tree_t30b_{heplus24,mbppbank}_n127k8.npz` (127 pops per anchor) + cells / TOPK. Span states from
  `lim800_slack.load(tag)["S"]` (per anchor x depth; import it). Span-end tokens by `lim800_strict.ends()`.
- Arms, each re-selecting 31 of the 127 pops as a closed subtree:
  - EAGLE-2-31 (path_p order);
  - LAT-31: path_p, with INERT-wording side nodes and their subtrees scored 0;
  - LAT-31b: also drop side nodes where the head's top-2 are within 1 nat.
- Walk with CC d1 on code rows + INERT on INERT rows (span ends strict), through tree_cc32's tree + CC walk. Report
  kept/32 on HE+ and MBPP.
- **PASS:** LAT >= +8 % over EAGLE-2-31 under the same rules on both sets. **BETWEEN:** +3-8 %. **KILL:** < +3 %.

## Idea 2: INERT-WIDE: an INERT span is a prefill, so check it wide (COPYRUN, TRIED 1471, generalised)
**Mechanism:**
- Under INERT, the rows inside a comment / docstring span are accepted whatever is drafted. They only fill KV, like
  prefill.
- A check that starts inside an INERT span can be sized to the span: span + 32 rows, at most 128. It takes the whole
  span, then goes on into code rows with CC d1.
- Per-check fixed costs (the dense read 7.6 ms, cores, overhead) are paid once per span instead of once per 32 rows.

**Numbers [P, lim800_wide.out]:**
- Lenient span ends; strict ends lower both arms by ~5-6 %, so the ratio holds [G].
- Cost model: Leap's C6 + int8 bound; dense = max(7.6, 4.9 x R/32); experts by the code union law; draft 3 ms per 31
  levels, one per row (copies NOT made free); overhead 1.5.

| width cap | HE+ kept / check | HE+ tok/s at 100 % / 85 % | MBPP kept / check | MBPP tok/s at 100 % / 85 % |
|---|---|---|---|---|
| 32 (frame) | 23.88 | 804 / 699 | 18.08 | 609 / 530 |
| 48 | 28.21 | 877 / 763 (x1.09) | 19.75 | 637 / 554 (x1.05) |
| 64 | 29.93 | 886 / 772 (x1.10) | 20.34 | 640 / 557 (x1.05) |
| 128 | 34.46 | 934 / 814 (x1.16) | 21.37 | 652 / 568 (x1.07) |

Check at the bound: R32 25.2, R48 30.2, R64 36.8, R128 61.3 ms.

- On HE+ most of the gain is the re-emitted prompt docstring. Drafting it from the prompt (lookup) adds more; that is
  COPYRUN's own estimate of x1.4-1.6.
- **COPYRUN is not in PUSH800's frame.** In TRIED it is UNTESTED; T-A1 is its test.

**Tag:** COPYRUN (TRIED 1471, FULL / GOAL) for prompt copies. New: the same for every INERT span under the INERT rule.

**Lossy / exact:** the copy part can be exact (copy rows verified); model-written spans are lossy (INERT).

**Quality risk:**
- Copies: none.
- Model-written comments: small but new. The head drafts up to ~96 comment tokens without re-anchoring, so comments
  drift further from the 125B's. A relaxed comment changes the next code 48 % of the time (2B, TRIED); runaways sit in
  comments (TRIED 1581).
- Gate: EvalSuite, as for INERT. A safe first form is WIDE on docstring copies only.

**What kills it:**
- A real 96-128-row pass costs > ~2.9x a 32-row check (the break-even at R128).
- Or fresh-data INERT runs are much shorter than dev's.

**Screen:** the CPU part is `lim800_wide.py` (1 min). Two things remain open:
- on fresh HE+, the same walk on the fresh windows;
- the engine's real cost of a 64 / 128-row pass vs 32 rows, from any measured prefill or C64 timing on Leap's
  scoreboard.

## Idea 3: the drafter line for code rows (what any drift fix must buy; a target, not a lever)
- Under the frame's lossy rules, kept is decided by confident code misses and span ends (Section 0).
- For dev HE+ ~26.5 (775 at 85 %, 3 ms draft): **no misses on forced code tokens (m >= 4) gives 24.61; the rest must
  come from trees on span ends + code misses at 1-4 nats (Idea 1).**
- **For S13 / S14 and any head screen, add one line next to kept:** confident code-row misses per 1,000 answer tokens.
  Today it is ~8 on HE+ dev (about half of the ~100 code misses that end checks, per 6,375 tokens). The 775 line is ~0.
- KILL / revise rule: the drafter project can't promise 775 on code with the frame's rules unless a head screen cuts
  this number by >= 75 %.

## Idea 4 (killed today): zero-row CPU re-rankers on the head's top-8 [K, lim800_rerank.out, lenient ends]
- **SCOPE:** promote an in-scope identifier over an unseen top-1.

  | p ratio | HE+ (vs 23.88) | MBPP (vs 18.08) | HE+ cells wrong -> right / right -> wrong |
  |---|---|---|---|
  | 0.3 | 23.88 | 18.23 (x1.008) | 131 / 147 |
  | 0.1 | 23.88 | 18.27 | 195 / 342 |
  | 0.03 | 23.88 | 18.27 | 253 / 587 |

  **KILL:** it breaks as much as it fixes. New names and builtins are often right as top-1.
- **SYNTAX VETO:** replace a top-1 that closes the wrong bracket. 0 cells changed on HE+, 2 on MBPP. **KILL:** the head
  never mis-closes brackets, so code misses are not syntax errors.
- **WHITESPACE-ONLY / string-content acceptance:** +0.4-0.6 %. Too small to price.

## Idea 5 (desk, killed): bigger fixed checks everywhere at lossy acceptance
- At an effective ~98 % per position (CC d1 + INERT on HE+): dense turns compute-bound at ~49 rows and the union grows
  x1.28 / x1.50 at 48 / 64 rows.
- So 48 / 64-row checks project only +2-6 % over 32 rows with a 3 ms draft [P]. Not a x1.3 lever.
- Idea 2 is the form of this that pays: width only where rows are certain.

## What this changes for PUSH800
- **Recount kept under the stacked rules, not lever by lever.** On dev, CC d1 + INERT (strict ends) gives x1.32 HE+ /
  x1.80 MBPP over 16.91 / 9.50: **HE+ ~655 tok/s at 85 % of the C6 + int8 bound with a 3 ms draft.** Fresh data is
  weaker (TRIED 1066).
- **With WIDE + LAT on top:**
  - HE+ dev ~805-850 at 85 % [P x G], fresh-like ~715-760.
  - So 775 on HE+ hangs on INERT passing EvalSuite, the engine at 85 %, a <= 3 ms draft, LAT + WIDE screening as
    projected, AND fresh data holding near dev.
  - Of these, LAT is the only free one: zero bytes, no new quality risk.
- **MBPP does not reach 775 by Saturday on any lever here** (<= ~650 at 85 %). The demo should name the benchmark and
  disclose the docstring re-emission share on HE+ (56.5 % of answer tokens).
- **Drop the 7.4 ms-draft column as a route to 775 at 32 rows:** it needs kept at 90-101 % of the answer-end ceiling
  (30.50 HE+ dev).

## Reproducibility
`cd ~/FableGod/lunatic && ~/BigMoE/.venv/bin/python3 -I lim800_slack.py`. Then, in any order: `lim800_strict.py`,
`lim800_spanend.py`, `lim800_rerank.py`, `lim800_wide.py`, `lim800_treewaste.py`.
- Read-only on BigMoE's dumps and EvalSuite's certify98.
- Peak 0.2 GB; 40 s to 3 min each.

## TRIED lines (ready to paste; Lunatic, 2026-11-16)
- LIM800-SLACK: lossy kept map on the t30b dumps (chain-32 walk, dev) | wall: kept | line: 45 GB lossy | exact: no | verdict: MEASURED: CC d1 + INERT (span ends strict) 22.37 HE+ / 17.13 MBPP (lenient ends 23.88 / 18.08); misses that end checks after them: code 74 / 82 %, span ends 25 / 17 %; ~half of code ones at 125B margin >= 4 (drift on forced tokens); CC d2 + INERT 24.90 / 19.31; answer-end ceiling 30.50 / 28.08 | ~/FableGod/lunatic/lim800_slack.out, lim800_strict.out | 2026-11-16 | [lunatic]
- LAT (lossy-aware tree): no tree side rows for wording inside INERT spans | wall: kept | line: 45 GB lossy | exact: no (tree exact; rules lossy) | verdict: UNTESTED; [K] 56 % HE+ / 69 % MBPP of EAGLE-2-31's walk-start side rows are INERT wording; code + span-end headroom x1.20-1.28 HE+ / x1.30-1.45 MBPP (rank <= 1 / <= 3, upper bound); screen: tree_cc32 walk on n127k8, EAGLE-2-31 vs LAT-31, PASS >= +8 % both | ~/FableGod/lunatic/IDEAS_1116b_limits.md | 2026-11-16 | [lunatic]
- INERT-WIDE (COPYRUN generalised to all INERT spans) | wall: fixed cost per check | line: 45 GB lossy | exact: copies yes, comments no | verdict: [P] span-sized checks <= 128 rows x1.16 HE+ / x1.07 MBPP tok/s at the C6 + int8 bound (draft 3 ms / 31 levels, copies not free, lenient ends); open: fresh windows + the engine's 64 / 128-row cost | ~/FableGod/lunatic/lim800_wide.out | 2026-11-16 | [lunatic]
- SCOPE-RERANK + SYNTAX-VETO on the head's top-8 | wall: drafter | line: all | exact: n/a | verdict: KILL: scope x1.000 HE+ / x1.008-1.011 MBPP (breaks as many cells as it fixes); syntax veto changes 0 / 2 cells (the head never mis-closes brackets) | ~/FableGod/lunatic/lim800_rerank.out | 2026-11-16 | [lunatic]
