# 12 GB chat +9.6 % from a confidence-adaptive verify width: the first exact call reads 2 rows instead of 4 when the first draft looks weak (rounding-level, EvalSuite gate pending)
- Date: 2026-11-08 10:3x EET. Session: Fit12.
- Claim: on chat, the 12 GB verified-greedy loop keeps only ~2.3 tokens per 16-draft check, but every check's first exact
  call verifies 4 rows. So ~47 % of the rows read are wasted, and each row costs ~290 MB of expert reads from the SSD.
  The drafter's own confidence in its first draft (cf0) predicts the short checks: below 0.5, 79 % of checks keep 1 token.
  - Reading 2 rows first when cf0 < 0.9 (later calls stay 4 rows) cuts SSD bytes per kept token 13 % on chat.
  - It raises chat whole decode 9.6 %, and leaves code within 1 %.
- Cost model [K] (CPU audit of chat12 + a random-width run, 12 IFEval): one exact call = 42 / 44 / 54 / 59 ms fixed for
  1 / 2 / 3 / 4 rows + 84 ms per GB read (11.9 GB/s); 300 / 531 / 868 / 1,158 MB per first call. The rule was priced at
  x1.17 [P] on chat before it was run.
- Number [K], 12 GB line (tier plan 4433333333, C_m 47 / C_l 32, engine v3p bt_b12 8db71cc537, DFlash2 L1-20 at 4 bits),
  share lock with an idle GPU and no other job running, each arm alone in its own process, rc 0:

  | arm (+ PR1, no 1-row tail chunk) | prompts | whole decode, geo (pooled) | SSD per kept token | kept per check |
  |---|---|---|---|---|
  | 4-row first call (reference) | 12 IFEval, CAP 128 | 12.68 (12.56) tok/s | 564 MB | 2.35 |
  | cf0 < 0.9: 2 rows, else 4 | 12 IFEval, CAP 128 | 13.89 (13.97) tok/s, x1.096 | 491 MB | 2.33 |
  | 4-row first call (reference) | 16 HumanEval, CAP 256 | 18.68 (18.69) tok/s | 441 MB | 10.10 |
  | cf0 < 0.9: 2 rows, else 4 | 16 HumanEval, CAP 256 | 18.50 (18.47) tok/s, x0.990 | 439 MB | 10.38 |

  - A 2-or-3-row variant (cf0 < 0.9: 2, else 3) measured x1.108 on chat; it was not code-checked.
  - Lifetime peaks 10.96-11.16 GB.
- Exactness: NOT bit-exact. Tokens match the reference on 5 / 12 chat and 12 / 16 code prompts.
  - On this engine path the committed state depends on call widths, even with PR1 (the 2-row replay of keep-1 commits)
    and no 1-row calls. The first divergence can come after two checks.
  - It is a rounding-level change of a loop that is already "verified greedy, not bit-exact": every committed token is
    still the full model's own verify argmax.
  - Adoption therefore needs EvalSuite same-scores (FAST-track rule): IFEval + HumanEval at 12 GB.
- How: /Users/kamen/Fit12/scripts/t3_twopass.py (F12_W1 = 4 | rand | rand:a,b | cf:C:W[:C2:W2]; F12_W1TAIL; per-call log),
  fit_t3.py (F12_PR1 via scripts/fit_pr1.py), runners chw12.sh / chw13.sh / chw14.sh, readers chatw_audit.py,
  chw12_read.py, chw13_read.py; data /Users/kamen/Fit12/data/scr/chw12, chw13, chw14.
- Prior art: adapted. Adaptive speculation length / verify width from drafter confidence is standard in speculative
  decoding (and this project's MTP-path "tau" widths, Oct). The finding is the 12 GB SSD-bound pricing: the fixed cost per
  call is large (42-59 ms) but shrinks with width, so 2-row first calls pay on weak drafts.
- Status: measured, rough (share lock, texts differ between arms); not adopted.
- Next: EvalSuite's gate when NewIdeas queues it. Bigger 12 GB chat levers remain kept per check (a chat drafter: MTP
  Head's F2a) and the 42-59 ms fixed cost per exact call.
- Update 2026-11-08 16:44: the gate is placed as a 12 GB IFBench arm (EvalSuite, Mon >= 15:00, scored vs the 45 GB card reference). The frozen gate harness passed its smoke test, and the loop is history-independent on 13 prompts (bit-identical tokens + logits hash in another order): one order is enough.
