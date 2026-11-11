# PREREG: a 12 GB code gate on Fit12's S1q (HumanEval+ and MBPP vs the 45 GB exact references) (Tue 2026-11-10 10:4x; before any row)

Written and committed before any row of this arm.

- **Request:** NewIdeas (the queue owner), 10:31: today's one big job while the GPU is idle. Fit12 proposed EvalSuite's
  HumanEval+ / MBPP code gate on S1q, "a valid 12 GB code mode". This gate would give SMALL_RAM a gated 12 GB code row.
- **The arm (s1q):** Fit12's S1q is QOM0 at C_m 51 / C_l 32 with a GPU residency probe (~/Fit12/data/scr/s1q,
  s1q_read.txt: 41.77 tok/s whole decode pooled over 16 HumanEval prompts). Fit12 runs it in its own session and lock
  (route (i), as with ifb12 in PREREG_card_gates amendment 13); EvalSuite scores it.
  - **The arm's files** (Fit12 10:3x; EvalSuite checked every hash at 10:3x):
    - Frozen harness in ~/Fit12/snap/s1q1110/ (sha256):
      - fit_onecall.py 6fd6d2951273. It equals the S1q run's copy (914e1ea452de, ~/Fit12/data/scr/s1q/MANIFEST.txt)
        except line 77, its sys.path line, which loads the snapshot's own folder (diff checked).
      - t3_onecall.py 24c21d9902a7, t2_twopass.py 36dd222c77a9 and fit_pr1.py abe5e79915c8, all equal to the S1q run's.
      - The engine: pinned bt_b12.py 8db71cc5375a (as in that MANIFEST).
    - Env record ~/Fit12/data/gate/code12_s1q_env.txt (git blob 524dec45a1e2). The runner passes these values:
      BIGMOE_MAXK=92, F12_T3ARMS=q0r, F12_CAP=item, F12_CMCL=51,32, F12_W1=4, F12_WL=4, F12_W1SEED=1, F12_W1TAIL=1,
      F12_PR1=1.
    - Against the measured S1q run (~/Fit12/scripts/s1q.sh), the cap is each item's own (768 / 1,024) instead of 256.
      The logging variables F12_PHLOG and F12_GSTAT are left out. So is F12_RERUN, which re-runs a prompt once after a
      PAUSE for clean timing.
    - Runner ~/Fit12/scripts/s1qg.sh (9cc5e78ed4fc); merge ~/Fit12/scripts/s1qg_merge.py (e0abf895b0b7). The key file
      ~/Fit12/data/gate/s1qg_fwd_keys.txt is byte-identical to results/card/code12_keys.txt (96e7e92dc06c).
    - Output: ~/Fit12/data/scr/s1qg (s1qg_<order>_tokens / _runs / _meta.json, and the merge's report). At 10:38 it held
      only WAIT, so no row exists.
    - **Outside code:** the frozen harness also imports modules from other sessions' live trees. Its sys.path covers
      ~/Breakthrough/scripts, ~/Breakthrough/v3p (bigmoe_v3f, hc8), ~/SpecDecode, ~/Scout, ~/BigMoE (patches, bigmoe),
      the mtplx package in BigMoE's venv, ~/Fit12/scripts and Leap's d7b_live / third_party. S1q's measurement loaded
      from the same trees.
      - results/card/code12_outside_manifest_pre.txt hashes them at this commit: tools/code12_outside_manifest.py, 507
        entries, digest 9435e36075a8704c.
      - The RESULT hashes them again after the run and reports any change. A change during the run in a module the
        harness loads makes the affected order's verdict void.

- **Started (dated line, 10:4x):** Fit12 released the arm at 10:40:43, 18 s after this PREREG's commit (89d6938,
  10:40:25). Process qgf01 (20 keys from humaneval/HumanEval/68) started at 10:41:10, with NO-SWAP 50 GB and a swap
  base of 6,683 MB. Fit12 asked Leap to leave d7b_live.py and third_party/dflash_model_mlx.py alone until the run ends.

## The sets, keys and references
| Set | Keys | Cap | Scored by | Reference (45 GB exact) |
|---|---|---|---|---|
| HumanEval+ | the 60 keys of data/items/dec_humaneval.jsonl | 768 (max_tokens) | evalplus base + plus tests (evalsuite.decode_report) | NAX6's AR, results/ref2tnax (60 rows) |
| MBPP | the 100 keys of data/items/dec_mbpp.jsonl | 1,024 (max_tokens) | tools/mbpp_score.py (3 asserts, isolated child, 10 s) | NAX6-N's AR, results/ref2tnaxn.r2 (100 rows) |

- **Prompts:** the items' own prompt_ids (longest 367 and 323 tokens). The reference rows' n_prompt equals the items' for
  60 / 60 and 100 / 100.
- **Why two references:** NAX6-N has no HumanEval+ rows (results/ref2tnaxn.r1 holds 1). ref2tnax is the only 45 GB exact
  run with these 60 rows. MBPP uses NAX6-N's rows, as the 11-07 MBPP gate did (PREREG_mbpp_2026-11-06.md).
- **The scoring path was dry-run on the references themselves:** HumanEval+ 60 / 60 token-identical, PASS; MBPP 79.0 vs
  79.0 %, PASS.

## The orders
- **Both orders, no reverse check.** The forward order is results/card/code12_keys.txt (sha256 96e7e92dc06c): the 60
  HumanEval+ keys in items-file order, then the 100 MBPP keys in items-file order. The reverse order is that list reversed.
- **Why both:** QOM rows were order-dependent in the w24 card gate (LCB forward vs reverse: 9 / 131 token-identical;
  RESULT_card_w24_2026-11-10.md). A one-order design would need history-independence, which is not shown for QOM0 with the
  residency probe.
- 20 keys per process, one process each, as in ifb12.

## Rows
- Per order, Fit12's merge writes tokens ({key: {"ex": [[ids]]}}, each key's first complete run) and runs
  ([{key, arm "ex", cut}]).
- EvalSuite turns them into results/card_s1q_fwd/decode.jsonl and results/card_s1q_rev/decode.jsonl with
  tools/r57rows.py (arm ex; cut after the first EOS or at max_tokens).
- A run cut by a time budget or a stop is not a row, and its key runs again. A key with two different complete runs in
  one order (a clash) makes the arm VOID.

## The decision
- **Decision 4 per set and order:** a paired delta above -5 points AND an exact McNemar p of at least 0.05. DIFFERS
  (HIGHER) passes.
  - HumanEval+: `tools/set_d4.py humaneval card_s1q_<order> ref2tnax`.
  - MBPP: `tools/mbpp_score.py d4 card_s1q_<order> ref2tnaxn.r2`.
- A set passes only in both orders; the gate passes when both sets do.
- With fewer than 60 + 100 rows in either order, there is no verdict. The arm may resume in a later slot under these
  rules.

## Memory, guards and time
- **The 12.0 GB line:** on the harness's own lifetime footprint per process, as in ifb12. The RESULT reports every
  process's figure against 12.0 GB.
- **Guards (NewIdeas 10:31):** NO-SWAP >= 20 GB at the start; stop on swap above the run's start + 0.5 GB or on a
  footprint above 12.0 GB; yield at chunk ends to any quiet ticket.
- **The runner's guards** (s1qg.sh):
  - A chunk starts only with no WAIT, PAUSE or HOLD, at least 20 GB NO-SWAP, no quiet ticket waiting, and no other
    session's MEM > 5 GB lock. Its lock reads "MEM 12 GB".
  - It stops the run on a phys_footprint_peak above 12.0 GB, read in exact bytes (footprint -f bytes), or on swap above
    the run's start + 0.5 GB.
- **Time:** chunks start by 14:40, and a chunk still running at 14:50 is stopped (the orchestrator: apps reopen at
  15:00). That is stricter than the 15:00 window.
- **Priced [P]:** 160 prompts x 2 orders at ~211 tokens per prompt (s1qh) and ~42 tok/s is ~30 min of decode, ~1-1.5 h
  with prefill and process loads.

## Information (no verdict)
Per set and order:
- rows token-identical to the reference, and between the two orders;
- rows at the cap;
- repetition at the cap (the repeated-8-gram share in the last 512 tokens, as in RESULT_card_ifb12);
- speed, rough (Fit12's meta).
