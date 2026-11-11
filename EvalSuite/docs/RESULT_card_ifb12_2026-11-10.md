# RESULT: the 12 GB chat mode (Fit12's P2', ifb12) on the card's IFBench vs NAX6-N (Tue 2026-11-10 09:58) — PASS on IFBench loose; strict drops 7.3 pt

Pre-registered in docs/PREREG_card_gates_2026-11-08.md, amendment 13 (37a03ca, 02:08:53), before any row of this arm. Its
dated start line is 482d602. Scope: IFBench only, one order, plus a 10-key reverse check. This is not a full card gate.

**Verdict [K]:** IFBench loose, the card gate's metric, gives 40.0 % vs 44.0 %: delta -4.00 pt, McNemar p 0.2101 (arm
only 5 / reference only 11). Decision 4 passes. The reverse check is OK: 10 / 10 keys re-run, 0 differ.

**Read with it [K], information only:** IFBench strict gives 32.7 % vs 40.0 %: delta -7.33 pt, McNemar p 0.0127 (arm
only 3 / reference only 14). The pre-registered metric is loose, so this does not change the verdict. But on strict
instruction following the 12 GB mode is significantly below the reference, and no "equal instruction following" claim
can rest on this arm.

**The arm:** Fit12 ran it in its own session and lock (route (i)); EvalSuite scored it.
- Frozen harness ~/Fit12/snap/chw1108/fit_t3.py (sha256 7a9ffd8375ed). Runner ~/Fit12/scripts/ifb12.sh (2143f25c7ff7).
  Merge ~/Fit12/scripts/ifb12_merge.py (ac65fec294ec).
- Env record ~/Fit12/data/gate/card12_chw_env.txt (blob da1eb63570ee): F12_T3ARMS=ex, F12_CAP=item, F12_CMCL=47,32,
  F12_W1=cf:0.9:2, F12_W1SEED=1, F12_W1TAIL=1, F12_PR1=1.
- All four hashes were rechecked after the run and are unchanged.
- Keys: the 150 gate IFBench keys in card_keys order, 20 per process (8 processes). Prompts are the items' own
  prompt_ids. The cap is 1,536.

**Checks [K]:**
- Each of the 150 gate keys was in exactly one chunk file, so there are no clashes and no extra keys.
- The ifb12rev chunk re-ran the first 10 keys (ifbench/116 ... ifbench/10) and matched all 10 token for token.
- ifb12_tokens.json equals the chunk runs. ifb12_runs.json has 150 runs, none cut.
- tools/r57rows.py turned them into results/card_ifb12/decode.jsonl: 150 rows, 134 ending in EOS.

**Process notes:**
- Chunk 1 ran beside the w24 card gate, 07:14:32-07:41:51, in neighbour mode. Then the arm held at EvalSuite's request,
  because of the heat pauses.
- Chunks 2-8 ran alone, 08:41:43-09:53:48, and the reverse chunk 09:53:49-09:58:01. Between chunks the arm let two other
  sessions' short quiet runs go first.
- Every chunk ended with rc 0. The run ended at 09:58:01 with rc 0.
- The release came five hours after the amendment because EvalSuite's release note reached Fit12, an idle session, late
  (amendment 13's dated line).

## The line (vs NAX6-N)
| Line | Result | Verdict |
|---|---|---|
| IFBench, one order (n 150) | 40.0 % vs 44.0 %, delta -4.00 pt, McNemar p 0.2101 (arm only 5 / reference only 11) | PASS |
| Reverse check (10 keys) | 10 / 10 re-run, 0 differ | OK |

The rule: a paired delta above -5 points and an exact McNemar p of at least 0.05 (decision 4). The p was recomputed by
hand from its discordant counts (two-sided exact binomial).

## Information (no verdict)
- **Strict:** 32.7 % vs 40.0 % (delta -7.33 pt, p 0.0127; 3 / 14). The 24 GB and 45 GB configs scored 36.7 % strict on
  the same keys (RESULT_card_w24_2026-11-10.md, RESULT_card_lcoh_2026-11-09.md).
- **At the cap (1,536):** 16 rows, against 10 for both the reference and the 24 GB config.
- **Repetition at the cap [K]:** measured as the share of repeated 8-grams in a row's last 512 tokens (Fit12's measure,
  rechecked here).
  - 5 of this arm's 16 capped rows are near-pure loops, at 0.87 or more: ifbench/181, /214, /182, /192 and /229. Four
    more are partly repetitive (/210 0.56, /239 0.54, /277 0.42, /159 0.33). The other 7 score 0.21 or less, and this
    arm's stop rows of 512+ tokens average 0.043.
  - The reference's 10 capped rows include 1 such loop, and the 24 GB config's 10 include 3.
  - On the 5 loop keys:
    - ifbench/181 loops in all three configs, the reference included.
    - ifbench/214 loops here and at 24 GB, while the reference stops after 44 tokens.
    - ifbench/182, /192 and /229 loop only at 12 GB. The reference stops after 532, 652 and 310 tokens; on /182 it
      was already repetitive (0.86) before stopping.
  - So the 12 GB mode falls into loops more often. On these 5 keys its loose verdicts equal the reference's, so the
    loops do not move the gate's line.
- **Token-identical:** 21 rows to the reference, 20 to the 24 GB config's rows, 20 to the 45 GB L + COH rows.
- **Against the 24 GB config** (40.7 % loose, forward): the same verdict on 131 of 150 rows; 9 rows pass only here and
  10 only there.

## Memory and speed
- **Memory [K]:** the harness's own lifetime footprint per chunk was 11.056-11.218 GB, the largest in chunk 5
  (ifb12_merge.txt and the chunk logs). That is under the 12.0 GB line in every chunk.
  - The runner's live 10 s probe read exactly 10 GiB (10.737 GB) in all 9 chunks: it printed whole GiB. As Fit12 notes,
    its 12.0 GB stop would have fired only somewhere above ~12.3 GB. The line holds on the harness's figures, not on that
    probe.
  - IFBench prompts are short (at most 206 tokens), so this run does not test long prompts at 12 GB.
- **Speed (rough, not a speed claim; ifb12_meta.json, unpaused keys, sum n / sum decode s):**

  | Chunks | Unpaused keys | Paused | Whole decode |
  |---|---|---|---|
  | 2-8, alone | 129 | 1 | 16.6 tok/s |
  | 1, beside the gate | 12 | 8 | 8.2 tok/s |

  This is a quality arm; Fit12 owns the 12 GB speed figures.

## Known limits
- IFBench only and one order. GPQA Diamond and LCB v6 were not run at 12 GB, so this is not a card gate. The
  one-order design rests on Fit12's history-independence data (chw18) and this run's reverse check.
- Power: a PASS bounds the loss at n 150; it does not show there is none. The loose delta of -4.00 pt is 1 point inside
  the rule, and strict is significantly lower.

## Data
- results/card_ifb12/decode.jsonl: the rows.
- results/card_ifb12_ifbench_d4.txt: the line.
- results/card_ifb12_src/: copies of Fit12's merged tokens, runs, meta and merge report, the chunk ids, meta and logs,
  the DONE file, ifb12.sh, ifb12_merge.py, the env record, and Fit12's summary.txt lines for this run. Fit12's folder
  is not under version control.
- Reference: results/ref2tnaxn.r6.
