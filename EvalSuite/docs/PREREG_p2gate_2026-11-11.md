# PREREG: the P2 same-scores gate on the 24 GB card (written Wed 2026-11-11 14:3x, before any P2 gate row)

**Asked by:** BigMoE, with the orchestrator's go, Wed 14:xx.

**What P2 is** (BigMoE: bc04e37, ff0215d, c706283, f5ad618; docs/FIRST_TOKEN.md):
- It is a first-token path for prompts of at most R57_P2 tokens. Such a prompt skips the prefill: it runs in
  R57_P2_ROWS-row chunks as plain forwards under the decode arm (R57_P2_VERIFY=0).
- Longer prompts take the card's prefill unchanged.
- **BigMoE's measurements (information; W24FT, 12 IFEval prompts):**
  - median first token 3.08 s -> 1.12 s;
  - a peak of 23.0 GB;
  - tokens equal to the exact prefill's on 2 of 12, so it is LOSSY, which is why it needs this gate.

## The arm and the reference
- **The arm:** the card as gated plus P2.
  - The card is w13b: BigMoE's data/stack/card_w24d_env.txt, git blob e06f495e.
  - P2 adds R57_P2=256, R57_P2_ROWS=13 and R57_P2_VERIFY=0.
  - It runs on BigMoE's snapshot w24ft: data/r57/snap/w24ft/r57_screen_w24ft.py at c706283, sha256 0e2cc385e83e....
    That is the card's snapshot w24 plus two patches: the PF-D fold (off unless R57_PFD=1 and R57_PFD_FOLD=1), and P2,
    with a p2 field in runs.json.
  - BigMoE's W24FT control ran w24ft with P2 unset: its tokens and logits equal w24's.
  - BigMoE commits the arm's env file, and its blob goes into the dated line before the start.
- **The reference:** the card as gated, i.e. its rows from the w24 card gate (results/card_w24_fwd;
  docs/RESULT_card_w24_2026-11-10.md).
  - On IFBench and GPQA Diamond the card's forward and reverse orders are token-identical, 150 / 150 and 100 / 100
    (neither set uses QOM).
  - So the arm runs one order, and its rows are compared with the card's forward rows.

## The keys (tools/p2_keys.json, sha256 c874dbc8d0aca774...)
The screen applies P2 when the prompt's length is at most R57_P2 (r57_screen_w24ft.py, line 1123). The keys are the
card gate's keys (tools/card_keys.json) whose prompt is at most 256 tokens, by n_prompt in the card's rows:
- **IFBench:** all 150 (n_prompt 28-206).
- **GPQA Diamond:** 67 of 100.
- **LCB v6:** none, because its shortest prompt is 275 tokens. P2 never runs there, so LCB is not run.
- **Controls:** the 5 shortest GPQA Diamond prompts above 256 tokens (258, 258, 264, 265, 269). They run in their own
  run, results/card_p2_ctl, so they stay out of the GPQA line.
- **IFEval is not in the verdict.** The card's own benchmarks decide (Kamen's 11-08 rule), and the card as gated has no
  IFEval rows. An IFEval line would need a same-run card arm, which doubles its cost.

## How it runs
- **Settings:** as the card gate: non-thinking, greedy, the card gate's caps (IFBench 1,536, GPQA 2,048), C 100,
  R57_HARD 1200, and 20 prompts per process.
- **Order:** the card gate's forward order (sorted keys), restricted to these keys.
- **The runner:** tools/day_runner62.sh, the card runner's (runner 61's) arm stage for these keys.
  - It keeps runner 61's code checks: the BigMoE pin, bm_pin on the snapshot, and per-process manifests against the
    first process's.
  - It drops the reference stage and the second order, and adds the P2 checks below.
  - Its sha256 goes into the dated line before the start.
- **Schedule:** the day's one job over 5 GB, under a share lock with MEM 24 GB in the reason, from the dated line.
  - The cutoff is Thu 00:15, so the night queue starts clean. If it is not done, it resumes Thursday after the night
    runs, until 14:00.
  - Estimate: 2-2.5 h, since the card's rows on these keys hold 159K tokens (about 1.5 h at 31 tok/s) plus process
    starts.

## The lines (the card gate's rule, decision 4; tools/set_d4.py)
- **Per set:** compared with the card's rows on the same keys, the paired delta must be above -5 points and the exact
  two-sided McNemar p at least 0.05. A gain with p < 0.05 is DIFFERS (HIGHER), which passes.
  - IFBench on its 150 keys (loose prompt accuracy, as in the card gate).
  - GPQA Diamond on its 67 keys (SET_D4_N=67).
- **The control:** all 5 control rows are token-identical to the card's. If any differs, the gate FAILS: P2 changed a
  prompt it should not touch, so the gate did not test its real scope.
- **The gate PASSES** when both sets pass, the control holds, and the run is valid.

## Validity (otherwise void: fix, then a new run)
- The env file's blob equals the dated line's, and the runner's code checks pass.
- P2 engaged: runs.json p2 > 0 on the kept run of each of the 217 short keys, and p2 = 0 on the 5 controls.
- Every key has exactly one uncut run.

## Information (no verdict)
- token identity with the card, per set;
- first-token time on the short prompts, P2's purpose, against the card's where its runs.json records one;
- whole decode;
- the lifetime peak (the 24 GB line);
- heat pauses.

## Known limits
- **Power:** GPQA's 67 pairs give McNemar little power, and the -5 point bar is the card gate's.
- **One order:** this rests on the card's order identity on these sets, and on P2's path using no QOM. The gate does not
  test P2's own order dependence.
- **Scope:** it covers only the card's benchmarks. Other short-prompt use (chat) is not separately gated.

## Amendment 1 (Wed 2026-11-11 14:3x): the dated line before the start
EvalSuite checked each item below against the files.
- **The arm's env file:** BigMoE's data/stack/card_w24p2_env.txt (346a691), git blob 6ff0296ff0c13dc9....
  - Its env line is card_w24d_env.txt's line (blob e06f495e) plus exactly R57_P2=256, R57_P2_ROWS=13 and
    R57_P2_VERIFY=0: 45 keys.
  - Its comment names tag p2, C 100, MEM 24 and the w24ft screen.
- **The snapshot:** data/r57/snap/w24ft/r57_screen_w24ft.py, sha256 0e2cc385e83e... (git blob fa92662a80f2), with
  pflm.py at git blob a602d0027fe4.
  - Nothing in the snapshot changed between c706283 and BigMoE's HEAD, 4fa9149, and nothing there is uncommitted.
- **The prompt length:** the screen's len(ids) is the item's prompt_ids, the card rows' n_prompt (BigMoE, from the card's
  runs.json). One GPQA prompt is exactly 256 tokens, and P2 includes it (<=).
- **The runner:** tools/day_runner62.sh, sha256 dd9b0399c1b739df..., with tools/p2_keys.json c874dbc8d0aca774... and
  tools/set_d4.py 2df3260cc8978dcf....
- **The dry run (14:35):**
  - C 100, MEM 24, the .so ff69fa24a745 (the card gate's), and code pin 2d66c4634f77;
  - bm_pin OK, 217 keys plus 5 controls, and the reference card_w24_fwd holding all 222.
- **The start:** attached (run_in_background), with CARD_TAG=p2, CARD_FENV = that file and CARD_CUTOFF 2026-11-12 00:15.
  It runs as soon as the runner's gates allow.
- **Amendment 1a (Wed 14:4x, before any GPU step):** the runner was started at 14:35:44 (pid 20000), and it waited
  without end for the env file.
  - BSD `date -j -f "%Y-%m-%d %H:%M"` keeps the current seconds. So its "written after 14:33" test read 14:33:44,
    while the env file was written at 14:33:43.
  - It was stopped at 14:41, by pid with its child, then `peer unlock`. It had made no row.
  - The fix, in two lines only, puts :00 seconds into its EARLIEST and CUTOFF.
  - The runner is now tools/day_runner62.sh, sha256 dbdc426dfb785962.... Its dry run at 14:41:43 matches 14:35.
  - Nothing else changes.
