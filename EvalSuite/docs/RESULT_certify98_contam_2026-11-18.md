# RESULT: CERTIFY98's FULL capture frozen; the contamination check on t30b's lineage: no set VOID, 5 windows dropped (Wed 2026-11-18 14:5x)

After the capture and the freeze, before any candidate pass (docs/PREREG_certify98_2026-11-15.md, amendments 9, 10
and 10a).

## The capture and the freeze
- **The capture PASSED** 647 / 647 windows (BigMoE, received 14:41): on-path self-check held on every window; 12.03 GB;
  peak 34.8 GB; C 160 after the 11:43 stop (amendment 10a's dated line).
- **The freeze** (tools/certify98_freeze.sh, armed 11:40): commit 69b304d at 14:41:31, results/certify98/
  files_sha256_2026-11-18.json, sha256 0064c6a63604ff4223f85506ed2ad89f37e42d4629161fcd75cce36ff01c57da.
  - FREEZE CHECK PASS: manifest_sha256 = the v2 manifest's (12e2bf9a...); the 647 listed windows; every file re-hashed
    (12.03 GB in 4 s).
  - results/certify98/FILES_FROZEN names the commit; the engine count check (iii) waits for it.

## The contamination check (amendment 9)
results/certify98/contam_fresh_2026-11-18.txt (results/certify98/contam_fresh_2026-11-18.sh, 26a141f; 316 s, one CPU
process). Each scored set ran against both halves of t30b's lineage:
- n1112's 3,072 captures with the registered source tags: every src is dlog-magicoder, none forbidden.
- b5's 246 training windows on the span rule alone.

| set | vs n1112 (3,072) | vs b5 (246) | 13-gram overlap, mean (n1112 / b5) |
|---|---|---|---|
| HE+ gated 68 | FLAGGED 2 (HumanEval 94, 143) | CLEAN | 3.03 % / 4.58 % |
| MBPP 300 | FLAGGED 3 (mbpp 152, 180, 367) | CLEAN | 1.92 % / 0.60 % |
| everyday 150 | CLEAN | CLEAN | 0.00 % / 0.38 % |

- **No set is VOID.** The flagged windows are dropped, as registered: 2.9 % of HE+ and 1.0 % of MBPP, under the 5 %
  line.
- **Where the shared runs sit:** four of the five lie in the 125B's answer, past the prompt (HumanEval 94 at token 696,
  gen 361; HumanEval 143 at 360, gen 193; mbpp 152 at 280, gen 199; mbpp 180 at 225, gen 175). mbpp 367's run
  straddles the prompt's end (76, gen 85).
  - HumanEval 94 and 143 share their run with the same capture at the same place (dd__mc_3362 at 345). Both problems
    call for a prime test.
  - These are the drafter's own training tokens inside the answer it would be scored on, which is what the span rule
    removes.
- **Information, the lineage sets vs b5** (out of every gated and reported set by name, amendment 9):
  - HE+ 72: 72 / 72 flagged; 13-gram mean 91.28 %.
  - IFEval 57: 51 / 57 flagged; 13-gram mean 44.47 %. The span rule misses 6 that b5 holds by name, which is why
    amendment 9 excludes by name.

## The scored sets
results/certify98/scored_lists_2026-11-18.json (written by scored_lists_2026-11-18.py from the frozen manifest, the
frozen files_sha256.json and the check's output; sha256 c7ba538610817fcf...):
- HE+ 66 (the gated 68 minus HumanEval 94 and 143);
- MBPP 297 (the 300 minus 152, 180 and 367);
- everyday 150;
- information, unchanged: HE+ lineage 72, IFEval lineage 57.

The count check's 24 windows (amendment 9) contain neither dropped HE+ window.

## Power at the scored counts (information; no line changes)
results/certify98/power_k66_2026-11-18.txt: amendment 9's K 68 run (the pinned certify98_lines.py, 579a60ac; seed 1116,
2000 draws) at HE+ K 66 and MBPP K 297. Every line moves by 0.02 at most:
- LAB: 0.49 (K 68: 0.49).
- C1 HE+: tree-16 1.00, tree-32 0.70 (1.00, 0.70). MBPP (reported): tree-16 0.50, tree-32 1.00 (0.51, 1.00).
- T32/32: 1.00 on both sets (1.00).
- T32/16 HE+ at E2's R32/R16 1.433 x c32 1.04: 0.45 (0.46).

## Next
- BigMoE's engine count check (iii) on the frozen files, chain-16 (amendment 10a), in its post-capture sequence now.
- Thursday's single look dumps the scored lists above (and the information sets as registered).
- Before scoring: pins.json committed, tools/certify98_dumpcheck.py PASS (amendment 10, section 3).
