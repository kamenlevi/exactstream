# RESULT: CERTIFY98's dev rehearsal on chain-16: tokens per check inside +-3 % (ratio 0.985); the draft-identity line is not readable from r57's records; every flip is a drafter near-tie (Wed 2026-11-18 15:1x)

Amendment 10a's dev rehearsal (docs/PREREG_certify98_2026-11-15.md): BigMoE's chain-16 engine run on the dev bank's 24
HE+ windows (heplus24) against kept_walk at cap 15 from the dev TOPK dump. Information before the fresh count check
(iii); no line here gates anything.

## What ran
- **The engine** (BigMoE's post-capture sequence, Wed 14:5x; data/r57/post1118/c98dev_{runs,checks,tokens}.json): r57
  on s98n, NAX6-N, R57_DRAFT=t30b in the stored form (weights sha256 790c42db..., the PREREG's pin), R57_TREE_K=1
  R57_TREE_NODES=15 (chain-16), cap 512.
- **The replay:** ~/BigMoE/data/deepdraft/ceiling/topk_t30b_heplus24.npz, the dev TOPK dump. Its cells give the same walk
  as the dev ceiling cells (checked: identical counts).
- **The reader:** tools/certify98_countcheck.py (985d054d...; selftest PASS). Output, with every input's sha256:
  results/certify98/rehearsal_c16_2026-11-18.txt.
  - The engine's count is put on kept_walk's own convention. Each check's kept is clipped at the answer's end. A check
    whose bonus token is the answer's last is followed by the replay's terminal check of 1.
  - r57's own 'kept' field counts drafts accepted past the end of the answer (HumanEval_1 in (iii): 247 over 21 checks
    for a 244-token answer). Near +-3 %, a check either way matters.

## Results
- **Windows:** 11 of 24 are usable.
  - 10 answers are identical to the bank's; HumanEval_129 is identical up to the bank's 400-token cap and is read to
    there.
  - The other 13 answers differ from the bank's, which was captured on another build (BigMoE: "another path"). The first
    differences fall between tokens 75 and 330.
- **Tokens per check:** engine 12.349 vs replay 12.542 at cap 15. That is a ratio of 0.985 (-1.5 %), inside the +-3 %
  line, on 2,408 tokens with 195 engine checks against 192 replay checks.
- **Draft identity, as registered** (>= 99 % of draft positions up to each check's first miss equal to dd_topk's top-1):
  NOT READABLE. r57's records hold each check's kept count, not its draft tokens.
- **At the acceptance level:** 187 of 195 checks keep the same count from the same position. The 8 that differ all part
  at a near-tie of the stored-form drafter:

| window | check (position) | engine / replay kept | replay's draft there | drafter top-1 / top-2 | log margin |
|---|---|---|---|---|---|
| HumanEval_10 | 18 (197) | 15 / 11 | wrong | 0.586 / 0.314 | 0.625 |
| HumanEval_10 | 23 (226) | 13 / 12 | wrong | 0.386 / 0.341 | 0.125 |
| HumanEval_129 | 18 (263) | 7 / 16 | right | 0.531 / 0.469 | 0.125 |
| HumanEval_135 | 8 (105) | 10 / 16 | right | 0.527 / 0.465 | 0.125 |
| HumanEval_135 | 11 (130) | 10 / 16 | right | 0.472 / 0.417 | 0.125 |
| HumanEval_158 | 18 (164) | 7 / 11 | right | 0.400 / 0.400 | 0.000 |
| HumanEval_47 | 7 (96) | 4 / 11 | right | 0.598 / 0.362 | 0.500 |
| HumanEval_96 | 11 (159) | 8 / 6 | wrong | 0.562 / 0.437 | 0.250 |

  - The target's token is the drafter's top-1 or top-2 at every one of them. Five go against the engine and three go
    for it.
  - None parts where the drafter is confident. A difference in how r57 loads t30b (a cast, the head) would show there
    too.
  - My reading: the engine's live features and the bank's stored ones differ slightly (another build), and that flips
    drafter near-ties. Not verified; the drafts would settle it.

## What it means for the fresh check (iii)
- (iii)'s engine records are in: 22 of the first 24 gated windows are identical to the capture's (HumanEval 1 and 26
  differ). The comparison reads those 22 once Thursday's TOPK dump exists.
- On the fresh set the capture and the engine share NAX6-N. Only the 16-row verify's ulp-level feature differences can
  flip a near-tie, so I expect fewer flips than on the bank. The rehearsal's -1.5 % net effect is the size to expect at
  most. This is a prediction, not a result.

## Open
- To read the registered identity line, r57 would log each check's drafts. BigMoE could re-run the 24 bank windows that
  way (~6 min). That is BigMoE's call; nothing gates on it.
