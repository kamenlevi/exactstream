# PREREG: the model card's benchmarks as fast-vs-exact gates (Sun 2026-11-08 12:0x; before any row)

Written before any row of any arm or reference.
> (a quote from the project owner, kept private)
  Not something new".
  - Scope: IFBench, LiveCodeBench v6 and GPQA Diamond. The card (models/bare/README-upstream-qwen.md, Qwen3.8-Flash-Next)
    gives 81.3, 91.9 and 91.7, in thinking mode with sampling. HLE is skipped (gated).
  - Arms: L + COH (the config we publish) vs NAX6-N first, then M1-B and the stack as time allows.
- **Placement:** NewIdeas 11:3x. The 45 GB arms run in the evening window (~4 h, ending by 22:45); the subsets are
  EvalSuite's choice.

## The sets (tools/card_items.py)
Items are in data/items and sources in data/raw; neither is pushed. All are non-thinking chat template and greedy, as
every dec_ set.
- **IFBench** (allenai/IFBench cd189a9c2889, 300 prompts).
  - The prompt is the user turn; max 1,536 tokens.
  - Grader: the official one, vendored verbatim in evalsuite/ifbench (only the package imports made relative).
  - Metric: prompt-level loose, the paper's headline. Its README generates at temperature 0. Strict is information.
  - Check: on the repo's sample outputs, EvalSuite's scorer equals the official run_eval.py flow on 300 of 300 prompts
    (strict and loose). The repo's own eval files differ on 3 of 288, from checker changes made after they were written.
- **GPQA Diamond** (198, data/raw/gpqa_diamond.csv), generative.
  - openai/simple-evals 652c89d0ca9d: QUERY_TEMPLATE_MULTICHOICE, with the options in its first-repeat permutation
    (random.Random(0), one rng.sample(range(4), 4) per row in file order).
  - Max 2,048 tokens. Score: the first ANSWER_PATTERN_MULTICHOICE match equals the gold letter.
- **LiveCodeBench v6** (livecodebench/code_generation_lite rev 0fe84c3912ea, test6.jsonl, sha256 bb4c364f7192...).
  - The window is 2025-02-01 <= contest_date < 2025-05-01, the Qwen cards' "v6 (25.02-25.05)"; this card names no window.
  - 131 problems: AtCoder 82 (easy 18, medium 19, hard 45) and LeetCode 49 (13 / 20 / 16).
  - Prompt: LCB's OpenAIChat messages (SYSTEM_MESSAGE_GENERIC + the generic question template, lcb_runner 28fef95ea8c9).
    Max 2,048 tokens.
  - Grader: LCB's run_test, vendored verbatim (evalsuite/lcb), on the public + private tests. It allows 6 s per test, uses
    the official global timeout, and passes a row only when every test passes.
  - EvalSuite's driver (evalsuite/lcb/grade.py) adds:
    - `python -I` in its own process group, in an empty temporary directory;
    - a kill above 3 GB resident memory;
    - a verdict cache keyed by (question, code sha), so identical code always gets the same verdict;
    - private tests unpickled with no globals allowed.
  - Dry run: 7 of 7 hand-written cases as expected (stdin and call-based pass, wrong answer, timeout, memory kill, no code).
- **Gate keys** (tools/card_keys.json, 381): IFBench 150 and GPQA Diamond 100 (random.Random(1108).sample of the sorted
  keys), plus all 131 LCB problems.
  - Why subsets: one arm in two orders must fit one evening window. That is ~450K tokens, at NewIdeas' 38-45 tok/s plus
    the per-process start (the chat gate ran 23-26 tok/s whole wall on ~320-token rows).

## Typing (the screens' rule())
Checked on the gate keys; the result is the same in scripts/r57_screen.py and the chs4 / chs5 / fu snapshots.
- IFBench: chat 141, math 6, code 2, agent 1.
- GPQA Diamond: chat 67, math 33.
- LCB: code 131.
- So M1-B's chat lever (MTP drafts gated by QOM) acts on IFBench and two thirds of GPQA, wherever its env puts QOM on
  those sets. Unlike MGSM (all math), these sets test it.
- **No screen's SETOF maps ifbench, gpqad or lcb6 yet.** Each arm's env must name a snapshot that does; runner 61 stops
  otherwise (dry run 11:52: it stopped on chs4).

## Reference: NAX6-N's AR -> results/ref2tnaxn.r6 (runner 61, CARD_STAGE=ref)
- The decode stage on the converted store, in the same environment as CERT's references and runner 60's stage 0.
- C is chosen by free memory: 192, 160 or 128 (need 36 / 32 / 27 GB). ref2tnax's rows were made at C 192.
- **Line check before any chunk below C 192:** 12 IFEval rows decoded at that C (results/ref2tnaxn.r<C>) must be
  token-identical to ref2tnax's. If they are not, that C is not used and the stage waits for 40 GB. A pass lets the
  reference run by day in Kamen's ~36 GB.
- Cost, estimated from NAX6-N's own rows (11-14 tok/s per row on IFEval / math): ~225K tokens (range 190-300K), ~5-6 h.
  It resumes across windows.

## Arms (runner 61, CARD_STAGE=arm), in this order
1. **L + COH (lcoh):** the config we publish.
2. **M1-B (m1b):** chat_env + COH.
3. **The stack:** as time allows.
- **Env files.** BigMoE writes one per arm, naming a snapshot whose SETOF maps the three sets. The env applies the
  per-set levers as the shipped config would apply them to these prompts: R57_QOM_SETS gains lcb6 wherever it holds the
  code sets, and ifbench / gpqad wherever it holds the chat sets. BigMoE states the mapping in the env file's comment.
- **Engagement:** every row of each set in the env's R57_QOM_SETS must show QOM optional uses, else the arm is VOID
  (runner 60's check).
- **Two orders:** forward is the keys sorted, then reverse. 20 prompts per process, the screen's CAP 2,048, each row cut
  to its item's max_tokens (r57rows).
- **Cost:** ~450K tokens (range 380-600K) for both orders, ~3.6 h (range 3-4.6 h). An arm that runs out of window
  resumes in the next one (CARD_RESUME=1).
- Runner 61 keeps runner 60's checks: code pin, manifests, the snapshot pin, memory, swap and the day rule.

## The lines (vs NAX6-N, per arm)
1. **Decision 4 per benchmark and order:** paired delta > -5 pt AND exact McNemar p >= 0.05 on the gate keys, no stage
   2. DIFFERS (HIGHER) passes. A benchmark PASSES only in both orders.
2. **Card gate PASS = all three benchmarks PASS.** A FAIL names its benchmarks and orders.
3. **Reading:**
   - A FAIL with McNemar p >= 0.05 (the delta alone) is reported as "FAIL (delta only)". It is still a FAIL.
   - VOID: the code checks or engagement fail.
   - INCOMPLETE: a "no free slot" abort or missing rows.
- **M1-B:** a PASS is a verdict on known benchmarks in its own right (the orchestrator). The typing above means it
  covers M1-B's chat lever.

## Information (no verdict)
- Scores per benchmark for the arm and NAX6-N: IFBench strict; LCB by difficulty and platform; GPQA by domain.
- Rows token-identical to NAX6-N, order agreement, cap hits, rough tok/s.
- The card's numbers, for orientation only (see the anchor below).

## Known limits
- **Non-thinking greedy.** The card's numbers are thinking mode with sampling. FAST levers on long thinking traces are
  not tested here.
- **Power at these N.** Take a lossless arm whose rows differ from NAX6-N on 10-15 % of keys, in random directions. Then
  Decision 4's -5 pt delta fails it in one order with these chances:
  - ~5 % at 10 % discordance (N 100);
  - ~15 % at 15 % discordance (N 100);
  - ~7 % at 15 % discordance (N 150).
  - Over 3 benchmarks x 2 orders, a lossless arm fails somewhere with a real chance (~25-50 %). That is the price of one
    evening per arm. The full sets (IFBench 300, GPQA 198) cut it to ~2-5 % per order. They cost ~+236K tokens per arm
    (~+1.7 h) and ~+118K on the reference (~+2.5 h).

## The card anchor (the orchestrator's question; not a gate)
- **Cost of exact in the card's mode.** That means thinking mode, sampling (temperature 1.0, top_p 0.95, top_k 20) and
  long outputs. Assumed lengths, typical for thinking models and not measured here: GPQA 6-10K tokens, LCB 12-25K,
  IFBench 1-3K.
  - The full sets, one sample each: ~4.6M tokens.
  - NAX6-N's AR at ~14 tok/s: ~90 h of 45 GB GPU. A FAST engine at ~40 tok/s: ~32 h, but that is the lossy engine.
  - The card may average several samples per item; it does not say.
- **It cannot be compared with the card at fine resolution:**
  - **The weights differ.** NAX6-N is our 45 GB tiered 3/4-bit quantization; the card's numbers come from the vendor's
    own serving stack. A card comparison measures quantization plus harness, not lossy vs exact.
  - **The settings are not stated.** For these rows the card gives no sampling settings, sample count, max length,
    prompt templates or LCB window.
  - **Sampling noise.** One sample per item leaves 95 % half-widths of ±3.8 pt (GPQA, 198 at 91.7 %), ±4.4 pt (IFBench,
    300 at 81.3 %) and ±4.7 pt (LCB, 131 at 91.9 %).
  - At best a full replication says "within ~±4-5 pt of the card": it can show gross breakage, not parity.
- **Recommendation: no card-mode run now.** If Kamen wants a sanity anchor anyway, the cheapest informative one is
  IFBench in thinking mode on NAX6-N: ~0.6M tokens, ~12 h of AR, ±4.4 pt, as information.

## Amendment 1 (Sun 12:0x, before any row): the extension rule, and the reference after Strata
- **The orchestrator 11:5x, verbatim:** "if an arm fails a single order only on the -5 pt delta (McNemar p >= 0.05), that
  benchmark is extended to its full set in both orders before it counts as a fail. A McNemar p < 0.05 stays a fail at
  once." The subsets stay as registered.
- **Reading, fixed now:**
  - **PENDING:** a benchmark where exactly one order FAILS with McNemar p >= 0.05 and the other order passes.
  - **FAIL at once:** both orders FAIL (even on the delta alone), or any order FAILS with p < 0.05.
  - **The full sets** (tools/card_keys.json "extend"):
    - IFBench: 300 (+150).
    - GPQA Diamond: 198 (+98).
    - LCB: the whole test6 file, 175 (+44 problems dated 2025-01-04 to 01-26). The gate's 131 are already the whole
      25.02-25.05 window.
  - **The extension:**
    1. NAX6-N's rows on the added keys (CARD_STAGE=ref CARD_EXTEND=<sets>).
    2. The arm's rows on the added keys in both orders (sorted, then reverse), appended to the arm's two runs
       (CARD_RESUME=1 CARD_EXTEND=<sets>).
    3. The benchmark's verdict: Decision 4 per order on the full set (set_d4 with SET_D4_N), PASS = both orders. It is
       final: there is no further extension.
  - **The card gate:**
    - PASS: all three benchmarks PASS.
    - FAIL: any benchmark FAILs.
    - PENDING: an extension is due and no benchmark FAILs. When another benchmark already FAILs, no extension runs,
      since the gate's verdict cannot change.
  - **Cost of an extension** (both orders, then the reference):
    - IFBench: ~100K tokens (~0.8 h), reference ~1.3 h.
    - GPQA: ~137K (~1 h), reference ~1.6 h.
    - LCB: ~70K (~0.5 h), reference ~0.8 h.
- **The reference waits (the orchestrator 11:5x; NewIdeas' order 11:5x).** NewIdeas' order is Strata -> BigMoE's PFB ->
  Breakthrough's two short screens -> the card reference (by day) -> an evening arm, if one fits before 22:45. Mon 07:05
  takes the arm work left over.
  - Runner 61's reference stage starts only when all three hold:
    1. Strata has ended: ~/StrataEval/h2h/logs/waiter.log's last line is WAITER END, or
       ~/StrataEval/h2h/results/DRIVER_DONE exists.
    2. BigMoE's PFB waiter (CARD_AFTER_PIDS, pid 71508: turn45.sh running scripts/pfb.sh) has exited, or 60 min have
       passed since Strata's end (CARD_AFTER_MAXWAIT: PFB had the free 47 GB first; a PFB that could not start does
       not hold the reference all day).
    3. A 10 min grace for the queue's short jobs has passed.
  - Each chunk also waits while another session's MEM job runs, as before.
- **The LCB item file** now holds all 175 test6 problems, each with a "window" flag. The gate keys and the other two item
  files are byte-identical to 9af1abb's (checked).
- **Verified, dry:**
  - The verdict block on synthetic runs: a p < 0.05 FAIL beats a pending extension, and a single-order delta-only miss
    (-6 pt, p 0.18) reads "PENDING: extend gpqad".
  - Runner 61's dry runs:
    - the reference stage shows 381 keys, or 523 with gpqad + lcb6 extended;
    - the arm refuses CARD_EXTEND without CARD_RESUME=1;
    - the arm still stops at the SETOF check.

## Amendment 2 (Sun 12:1x, before any row): the reference runs now, in short chunks, and yields to Strata
- **The orchestrator 12:0x** replaces amendment 1's "after Strata": "start the card reference now, in 10-prompt chunks,
  once Fit12's T2 is done". About 36 GB is free with Chrome open; one > 5 GB job by day.
  - Fit12's T2 (chw15p) ended 12:04; its report is written.
- **Yield rule, the orchestrator's:** before each chunk, if `pgrep -x "Google Chrome"` finds nothing, Strata is about to
  take the Mac, so no new chunk starts.
  - Resume needs Strata's waiter.log last line WAITER END (or DRIVER_DONE), then BigMoE's PFB waiter (pid 71508; at most
    60 min), then no quiet lock held.
  - Once Strata has ended, Chrome is not checked again.
- **Runner 61's reference stage:**
  - Chunks of 10 rows: `--max-rows 10`. The 12-row line check runs as one chunk.
  - Before each chunk:
    - the yield check;
    - no quiet lock held;
    - no other MEM > 5 GB shared run;
    - swap <= its value at the start + 2 GB, else the stage stops;
    - the C / line-check rule unchanged.
  - Cutoff Sun 22:45; it resumes on a rerun.
- **The reference's rows do not depend on this schedule.** Every chunk is the same exact AR decode stage. Only a C
  below 192 has its line check.
- Mock-tested: the yield function returns at once while Chrome runs or Strata has ended. With Chrome gone it waits for
  Strata, then the PFB waiter, then quiet locks, then resumes.
- **A failed start, 12:06:54 to 12:07:3x, with no row.**
  - What happened: the first line check's run name (ref2tnaxn.lc160) is not a name the decode stage takes, which is
    `<variant>.r<digits>`. The stage exited in 5 s before loading the model, and the runner kept retrying.
  - I stopped it by pid (the runner and its sleep) and ran peer unlock; no lock was left. Its log is
    results/day61_ref_runner.lcname_1206.log.
  - Fix (fdab3dd): the line-check runs are ref2tnaxn.r<C>, and a line-check decode that fails with no row now stops the
    runner.

## Amendment 3 (Sun 14:5x; the orchestrator 14:4x): the reference waits for Strata's end; scheduling only

- **Why:** Kamen asked for the Strata comparison as soon as possible. Strata starts with Chrome open, and the reference's
  decode (29 GB) was the one thing standing in the way of the Strata waiter's 47 GB gate.
- **Stop, 14:46:44:** runner 61 and its children were stopped by pid (10437, 51276, 51277, 51290), then peer unlock.
  - Rows are appended as they finish, so only the row in flight was lost. It is decoded again.
  - Kept: 185 of 381 rows (IFBench 150/150, GPQA Diamond 35/100, LCB 0/131).
- **Re-armed at 14:52 with `CARD_WAIT_STRATA=1`:** no chunk starts until Strata has ended (WAITER END or DRIVER_DONE),
  whether Chrome runs or not.
  - Then, as in amendment 2: the PFB waiter's pid (at most 60 min), 2 min, and no quiet lock.
  - The swap guard's base is re-read when a yield ends. Swap that others added while the runner waited is not its own.
  - Mock-tested: with the flag it waits even while Chrome runs; without it, it returns at once.
- **Leftover:** rows not done by tonight's 22:45 cutoff run first in the Mon 07:05 slot, then the lcoh arm (the
  orchestrator 14:5x). There is no night block: MTP Head's F2a has the GPU from 23:00 to about 07:30.
- **Unchanged:** keys, row counts, settings and the verdict rules.

## Amendment 4 (Sun 15:2x; the orchestrator 15:2x): only a finished Strata releases the reference; scheduling only

- **What happened:** Strata's waiter ended at 15:19:38 with "WAITER END: INCOMPLETE after 3 attempts" (three heat
  pauses; 2 of 8 passes done). Runner 61 read any WAITER END as Strata's end, so at 15:20:11 it moved on to the PFB wait.
  - No chunk ran. Its 60-minute PFB cap would have run out at about 16:20.
  - The orchestrator re-armed the waiter (12 attempts) at 15:2x.
- **Fix:** I stopped the runner at 15:23:16 by pid (57598 and its sleep), then peer unlock.
  - Strata now counts as ended only on DRIVER_DONE, or when the last waiter.log line says "WAITER END: DONE".
  - An INCOMPLETE or GAVE UP end no longer releases it. `touch results/card/STRATA_RELEASE` releases it by hand, only on
    the orchestrator's word.
  - It is re-armed in the first gap with no quiet lock (the peers hook blocks starting programs during one).
- **Monday's slot (NewIdeas 15:25, correcting its first 09:00):** 07:05 to 15:00, Kamen's weekday big-test window. The
  reference leftover runs first, then lcoh, then whichever next arm fits. The last chunk or arm must END by 15:00.
  - The runner starts a chunk only if its longest chunk so far (at least 35 min) fits before the cutoff.
- **PFB's waiter has a new pid, 79192** (BigMoE 15:28; 71508 is gone). BigMoE re-armed it with the same Strata release
  as this amendment. The re-arm uses `CARD_AFTER_PIDS=79192`.
- **Unchanged:** keys, row counts, settings and the verdict rules.

## Amendment 5 (Sun 16:5x; the orchestrator 16:4x): a hold for Strata at the 24 and 12 GB lines; scheduling only

- **Why:** Kamen wants the low-RAM results published, and they need Strata measured at the 24 and 12 GB lines. Its
  quiet-lock waiter queued at 16:48 (about 35-50 min of runs).
- **Ask:** after the current chunk, start no chunk until ~/StrataEval/h2h/results_low12/DRIVER_DONE exists, or for 75
  minutes, whichever comes first.
- **What happened:**
  - The chunk started 16:43:08 at C 192. It stopped at its next row for Strata's queued ticket: rc 75 at 16:50:25,
    190/381 rows.
  - The runner then waited for a share lock. I stopped it by pid at 16:51: the runner (80531), its chunk subshell
    (35419), and the decode client that was waiting for the next chunk's share lock (35420). Then peer unlock.
  - Lesson: killing the main loop orphans the per-chunk subshell and the waiting client. Both need their own kill.
- **Restart:** at results_low12/DRIVER_DONE or 18:05, whichever comes first, either directly or through
  tools/card_hold.sh. That script stops a running runner after its chunk, holds for a file or a time, then re-arms the
  same reference command.
- **Unchanged:** keys, row counts, settings and the verdict rules.

## Amendment 6 (Sun 18:3x): no decode starts during the quiet PAUSE; operations only

- **What happened:**
  - At 18:05 the hold's 75-minute cap ended. A chunk ran 18:06-18:10 (rc 75, yielding to a 13-min quiet ticket):
    191/381 rows.
  - The next chunk's decode client waited for its share lock. When Strata released its quiet lock for a heat pause
    (PAUSE at 18:31), the client got the lock and started a decode at 18:32:25, about 30 s into the PAUSE.
  - The runner checks PAUSE before each chunk, but a client already waiting for its lock does not.
- **Action:** I stopped the client (7362) and its 19-second-old loading stage (96140) by pid, then peer unlock. No row was
  in flight. The runner logged rc 143 and went back to waiting for PAUSE to clear.
- **Fix without editing the running runner or the code it imports:** tools/card_pause_guard.sh runs beside it.
  - While PAUSE exists, it stops any decode client still waiting for its lock, and any stage younger than 120 s (still
    loading, no row yet), then runs peer unlock.
  - A stage already decoding rows is left alone: fanwatch freezes it, and a quiet ticket stops it at its next row.
  - The proper fix (the client rechecks PAUSE after its lock is granted) waits until no runner 61 is running.
- **Unchanged:** keys, row counts, settings and the verdict rules.

## Amendment 7 (Sun 18:3x-19:0x; the orchestrator 18:3x and 18:5x): queue order after Strata's low-RAM run; scheduling only

- **Order after Strata low12:**
  - First: the reference was to follow a 7-min BigMoE diagnostic, then the reference.
  - The runner was in its PAUSE wait, with no chunk running. I stopped it by pid at 18:37:08 so it could not start
    first. Rows stay at 191/381.
- **18:57:**
  - Strata low12 wrote DRIVER_DONE. Its exact arm was dropped after repeated heat stops.
  - BigMoE's diagnostic gave up without running: two grants found only 44 GB no-swap.
  - The orchestrator put a 6-min Strata chat arm (12 IFEval, cache 1 GB, quiet lock) next.
- **The reference's gate:** ~/StrataEval/h2h/results_fit12/DRIVER_DONE, or 19:45 if it never appears.
- **tools/card_hold.sh** can now also chain behind another session's job: HOLD_AFTER_LOG / HOLD_AFTER_PAT /
  HOLD_AFTER_MAX.
  - It waits for that job's "done" line, written since the hold began, after the file appears.
  - Mock-tested three ways: the done line after the file, the done line before the file, and no file until the cap.
- **Unchanged:** keys, row counts, settings and the verdict rules.

## Amendment 8 (Sun 19:3x; ~/Papers/MONDAY.md, NewIdeas 19:2x-19:3x): Monday's chain; scheduling plus one memory threshold

- **Order, with no gap after F2a stage 1:**
  - The trigger is MTP Head's STAGE1_DONE file, newer than the chain's start, or a new night.log line that is exactly
    "[Day HH:MM:SS] F2a night DONE".
  - The backstop is 03:40.
  - Then: the card reference's leftover (cutoff 15:00), then lcoh, then m1b, then zs32 (cutoffs 14:30). The last process
    must END by 15:00.
  - zs32 is the 32 GB chat arm: M1-B + COH + SB1 at C 160, MEM 35 GB, with the card sets in its QOM sets
    (~/BigMoE/data/stack/card_zs32_env.txt, BigMoE 6717aa3).
  - Anything not gated by 15:00 is labelled "not gated".
- **tools/card_chain.sh:**
  - It waits for tonight's runner first. It stops if the reference is incomplete.
  - After an arm with rc 0, 2 or 6, it goes on to the next arm. Any other rc stops it.
  - A PAUSE guard runs beside the reference step (amendment 6).
  - Mock-tested trigger: the armed line, another new line and an old STAGE1_DONE wait; the anchored line and a fresh
    STAGE1_DONE start.
- **The memory threshold, the one non-schedule change:**
  - The MEM 47 arms (lcoh, m1b) start at 48 GB NO-SWAP instead of MEM + 4 = 51.
  - Basis (NewIdeas): M1-B's measured lifetime peak is 42.2-42.4 GB up to 1,536 tokens. Add about 1.5 GB for the
    card's longer rows, plus 4.
  - Runner 61 reads CARD_MEM_FREE. That edit is made after tonight's runner exits and before the chain calls runner 61.
  - The swap guard (start + 2 GB) still stops a run. zs32 keeps MEM + 4 = 39.
- **Unchanged:** keys, row counts, settings and the verdict rules.
- **Update (19:5x; the orchestrator 19:4x):** apps close 23:00 -> Mon 17:00, and gates may run to 16:00.
  - The chain was stopped at 19:50 by pid (it was only waiting) and re-armed with cutoffs 16:00 (reference) / 15:30 (arms).
  - Re-armed again at 19:51 with the backstop at 03:55 instead of 03:40 (NewIdeas 19:5x: F2a may end at 03:30 + its 10-min
    guard). Pid 53863.
  - The 48 GB memory threshold is dropped: the arms keep runner 61's MEM + 4. Runner 61 is not edited.
  - At 19:32 the pause guard stopped a 54-second-old loading stage when a heat PAUSE began. No row was in flight.

## Amendment 9 (Mon 06:4x, before any new arm row): a run cut at the time budget is not a row; operations only

- **What happened:**
  - The card screen (BigMoE snapshot data/r57/snap/card, HARD = 600 s) stops a run mid-decode when its process's 600 s
    budget runs out, and flags the run `cut` in <tag>_runs.json.
  - tools/r57rows.py did not read that flag. It kept each cut run as a row ending in "length" below the cap.
  - lcoh's forward order kept 10 such rows, one in each process that ran out of time. GPQA: 1, 502, 24, 1,234 and 1
    tokens. LCB: 1,165, 1,093, 1,679, 1,369 and 643 tokens. These were the only rows ending in "length" below the cap.
  - At 06:37 the reverse order's first process cut an LCB run after 1 token. That run had no QOM use, so the engagement
    check voided the arm (rc 5) and the chain stopped. Nothing from that process was kept. The VOID done-file is kept as
    results/CARD_lcoh_DONE.void_0637.
- **The fix (EvalSuite only):**
  - r57rows leaves out a key whose first run is `cut`, so the key re-runs in the next process. Keys the screen "stopped
    before" have always worked that way.
  - The engagement check counts kept runs only, and names the cut ones.
  - The 10 cut rows were removed from results/card_lcoh_fwd. The full file is in git at 1eeceb6.
  - The code pin changes (r57rows.py is pinned): 5680b8a9731b -> 8ecc1604bd56. The decode code, the screen, BigMoE
    b37f36b, the ext .so and the module manifest are unchanged.
- **The restart:** tools/card_chain2.sh. lcoh resumes with CARD_RESUME=1, so its processes are checked against the 03:42
  manifest. It decodes the 10 cut keys first, then the whole reverse order. Then m1b and zs32 follow, with the cutoffs of
  amendment 8.
- **The decision rule is unchanged.** At 06:3x I had scored lcoh's forward order for information with the cut rows in it.
  That look is void. The verdict uses kept rows only.
- **The reference is not affected.** It ran through EvalSuite's own decode path, and none of its rows ends in "length"
  below the cap.
- **Earlier arms that went through r57rows are audited separately.** A cut row is usually scored as wrong, but the
  direction is not fixed: a cut IFBench reply can still meet a length limit.
- **Audit (06:5x):** 40 runs in results/ hold screen rows written by r57rows. Only two have rows ending in "length" below
  the item's cap:
  - card_lcoh_fwd: the 10 rows above, now removed.
  - agentfast_fwd_v1: 1 of 11 rows. That run was set aside as information on 11-06 (RESULT_agentfast) and fed no verdict.
  - No earlier verdict used a cut row. A run cut after its item's cap is harmless: r57rows keeps the first max_tokens
    ids, which an uncut run shares.

## Amendment 10 (Mon 10:2x, while m1b's forward order runs, before its verdict; NewIdeas 07:22): m1b's cutoff may extend to 15:45; scheduling only
- **Why:** m1b started at 10:07 with the arms' 15:30 cutoff (amendment 8: the last process starts by 15:15). lcoh took
  5.8 h of decoding (RESULT_card_lcoh_2026-11-09). m1b's first GPQA process decoded about 13 % more tokens than lcoh's
  GPQA processes, so its end is projected near 15:15, at the cutoff.
- **The queue owner's OK (NewIdeas 07:22):** m1b may run to a 15:45 cutoff (last start 15:30, verdict by ~15:55). Nothing
  is booked 15:30-16:00. A verdict not out by 16:00 stays "not gated" (~/Papers/MONDAY.md).
- **How:** tools/card_extend_cutoff.sh m1b "2026-11-09 15:45" waits beside chain2. If m1b's step ends rc 6 on the 15:30
  cutoff (the runner's "does not fit before the cutoff" or "no memory before the cutoff" stop, which writes no DONE file),
  it resumes m1b at once: CARD_RESUME=1, CARD_CUTOFF 15:45. The resume checks its processes against m1b's first manifest.
  Any other end starts nothing. No process is stopped to do this.
- chain2 may meanwhile start zs32; that runner's first gate stops on the same 15:30 cutoff before any lock or process.
- **Unchanged:** the decision rule, the keys, the code pin (8ecc1604bd56), the config, and the other arms' cutoffs.

## Amendment 11 (Mon 12:4x, while m1b's forward order runs, before its verdict; the orchestrator 12:43): m1b's cutoff may extend to 16:20; scheduling only
- **Why:** amendment 10's projection was wrong. It came from m1b's first processes. Measured over the forward order, m1b
  runs at lcoh's pace: 309 kept rows at +122.4 min against lcoh's 308 at +121.1 (both runner logs, per process). Decoding
  is projected to end near 15:57 (+-10 min), so under 15:45 (last start 15:30) m1b would end INCOMPLETE, one or two
  reverse GPQA processes short.
- **The OK (the orchestrator 12:43; it owns ~/Papers/MONDAY.md):** m1b may run to 16:20. Kamen's window runs to 17:00,
  the package's clean-clone run passed at 12:13, and Fit12's chw19 goes after m1b. The verdict, or "not gated", goes to
  the editor by 16:30. zs32 stays "not gated". This replaces my 12:27 ask to the queue owner (a 16:00 cutoff).
- **How:** amendment 10's waiting extension had started nothing; it was stopped at 12:45 and re-armed as
  tools/card_extend_cutoff.sh m1b "2026-11-09 16:20" (results/card_extend.log). If m1b's step ends rc 6 on the 15:30
  cutoff, it resumes m1b at once with CARD_CUTOFF 16:20: the last process starts by 16:05 and ends by about 16:17 (a
  process takes at most about 11.7 min). No process is stopped.
- **Unchanged:** the decision rule, the keys, the code pin (8ecc1604bd56), the config, and zs32's 15:30 cutoff.
- **Correction (13:1x):** the longest process so far took 12.7 min (m1b forward process 20, 12:47-13:00; lcoh's longest
  was 11.7), so a process that starts at 16:05 ends by about 16:18. The runner's 15-min start gate still covers it.

## Amendment 12 (Mon 9 Nov 19:0x, before any W24 env file or row): a card-gate arm for the 24 GB config (w24)
- **Why:** the queue owner (NewIdeas, 18:56) puts one gate first: a same-scores gate on the model card's benchmarks for
  BigMoE's 24 GB config (W24: L + COH below a 24.0 GB lifetime peak on the fast path, BigMoE scripts/w24.sh), done before
  the Wednesday publish.
- **The arm:** tag w24; the config BigMoE names in data/stack/card_w24_env.txt, written and committed in BigMoE after this
  amendment (its comment names w24, the arm's C, its MEM and the screen data/r57/snap/w24/r57_screen_w24.py; its env line
  is the chosen W24 arm's keys with R57_QOM_SETS including lcb6, as in card_lcoh_env.txt).
- **Unchanged from the other arms:** the 381 gate keys, caps 1,536 / 2,048 / 2,048, non-thinking greedy, the reference
  NAX6-N (results/ref2tnaxn.r6), the decision rule (decision 4 per benchmark and order; a benchmark passes in both orders;
  the gate passes when all three do), both orders, PERPROC 20, amendment 9's cut-run rule, the engagement and module
  checks, resumes under CARD_RESUME=1.
- **What differs, by design:** C and MEM come from the env comment (W24 runs at C 100 with its own base / sidecar slots;
  runner 61 now accepts C 100, tested by a dry run on a mock env); the screen is W24's snapshot (rel1's screen + the
  R57_ROWCAP row cap), so the code pin differs from 8ecc1604bd56; the per-process time budget is whatever HARD that
  snapshot holds when the arm starts (600 s today; BigMoE may set 1,200 s before the first row, never after).
- **Slot and cutoff (queue owner 18:59):** start Tue 10 Nov ~01:30 if the env file is committed by then, with CARD_CUTOFF
  2026-11-10 15:00; otherwise the Tue-night slot. An arm stopped at a cutoff resumes in the next slot (CARD_RESUME=1).
- **Priced length [P]:** ~9-12 h for both orders (central ~10 h): lcoh kept 867,659 tokens in 5.8 h at 45 GB; at 24 GB
  the card rows' rate is assumed x0.5-0.7, and the 600 s budget fits ~7 long rows per process (~80 processes). Re-priced
  from W24's own IFEval rate and the arm's first process.

## Amendment 12a (Mon 9 Nov 21:1x, after the env file, before any w24 row): HARD 1,200 s and the process price; operations only
- **The env file (BigMoE 8f76fc3, 20:01):** arm w13t of W24: 130 base + 26 sidecar slots, rows <= 13, prefill trim
  (BIGMOE_TIER_PF_GROUP=64, R57_CACHE_GB=0.25), QOM on humaneval, mbpp and lcb6, and R57_HARD=1200 (the screen cuts the
  running row and ends the process after 1,200 s of running time; BigMoE 788f0e1). Its comment names w24, C 100, MEM 24 GB
  and the w24 screen. It is the config of W24b's w13t rows plus HARD and lcb6 in the QOM sets; BigMoE's own runner
  (scripts/w24.sh through data/r57/snap/c45_runner_rel1.sh) passes R57_PRESET=45, R57_REPS=1, R57_WARM=1 and C 100, as
  runner 61 does.
- **The change (runner 61):** a process takes up to HARD plus its load, the cut row and its teardown (762 s at most at
  HARD 600, Mon 11-09). Runner 61 now reads R57_HARD from the env line (600 when it is unset or empty, the screen's
  default). It prices each process at HARD + 300 s before the cutoff: 900 s at 600, as before; 1,500 s at 1,200. The
  shared lock's lease is twice that: 30 min at 600, as before; 50 min at 1,200. A value that is not whole seconds stops
  the runner. Dry runs: the w24 env reads 1,200 s, 1,500 s and 50 min; card_m1b_env.txt reads 600 s, 900 s and 30 min; a
  bad value stops it (rc 2); an empty one gives 600 s. The share-lock test passes. The 01:25 launcher
  (tools/card_w24_start.sh) was stopped before the edit and re-armed after the tests; it starts nothing before 01:25.
- **Priced length, re-done [P]:** the estimate comes from a budget sim on lcoh's rows. A row takes first token + n / tps; a
  process is cut at HARD + 40 s of row time and costs 65 s of load and teardown. The sim gives 6.1 h for lcoh at 45 GB
  (5.8 h measured). The 24 GB rates are scaled per set from BigMoE's CAP-256 pairs on the same 12 prompts:
  - ifbench and gpqad (exact reads): x0.63, from IFEval at 28.4 tok/s (w12, 24 GB) vs 45.4 (rel rows, 45 GB).
  - lcb6 (QOM): x0.71 from HumanEval (116.5 vs 164.0) and x0.80 from MBPP (90.5 without its one stalled row, vs 112.7).
  - first token: x1.2.

  | Rates (exact reads / lcb6 / first token) | Processes per order | Both orders |
  |---|---|---|
  | central: x0.63 / x0.75 / x1.2 | 20 | ~8.1 h |
  | slow: x0.5 / x0.65 / x1.5 | 20 | 9.9 h |
  | slowest: x0.4 / x0.6 / x2.0 | 21 | 11.9 h |
  | central at HARD 600 | 25 | 8.6 h |

  None of these reaches the runner's limit of 30 processes per order. This replaces amendment 12's "~80 processes",
  which assumed ~7 long rows per 600 s; most rows are short (IFBench averages 384 tokens).
- **Unchanged:** the keys, the reference, the decision rule, both orders, PERPROC 20, the slot (from 01:25, latest start
  05:00) and the 15:00 cutoff. From a 01:25 start the central case ends near 09:30 if memory is free.

## Amendment 12b (Mon 9 Nov 23:2x, before any w24 row and before any W24c data): a 02:10 start, a pinned env, a switch rule fixed in advance, one neighbour; operations only
- **Start (the orchestrator 23:2x, with Kamen's approval):** the arm starts at 02:10:00 Tue, not 01:25, so Leap's 32 GB
  tests fit before it. The latest start (05:00:00) and the 15:00 cutoff are unchanged. Times now carry seconds: BSD
  `date -j -f` without them keeps the current seconds, so the old start could fall up to 59 s late. From 02:10 the priced
  lengths of amendment 12a end at 10:17 (central), 12:04 (slow) and 14:04 (slowest), all before 15:00 if memory is free.
- **The env is pinned (tools/card_w24_start.sh):** the launcher starts runner 61 only when the env file is committed in
  BigMoE and its blob (`git hash-object --no-filters`) is one that was dry-run. Tonight that is only w13t's 757b768c
  (card_w24_env.txt, BigMoE 8f76fc3). Any other content makes it wait; it says so once and starts nothing by 05:00.
  Runner 61 reads a frozen copy, results/card_w24_env_run.txt (cp -p, the same blob), so a later edit of BigMoE's file
  cannot reach the running arm. Tests:
  - the w13t env starts to the second;
  - the W24c env with only w13t's blob allowed waits, then starts nothing (rc 3);
  - with its own blob allowed it starts;
  - runner 61's dry run on the frozen copy passes.

  The launcher was stopped before the edit and re-armed at 23:20:43 (pid 65794).
- **The switch rule, fixed before W24c's data:** sources are the orchestrator (23:1x via NewIdeas 23:07, and 23:2x) and
  BigMoE's docs/W24_DECISION.md, section W24c (ddc90f5, with its 23:2x amendment). The arm runs W24c's 14-row config
  instead of w13t only if NewIdeas tells EvalSuite by 02:05 that W24c passed:
  - HumanEval whole decode >= 120.0 tok/s, pooled over both 14-row repeats;
  - no bad rows, and every lifetime peak below 24.000 GB;
  - BigMoE's other conditions;
  - NewIdeas' number check (~/NewIdeas/scripts/numcheck/w24c_check.py, fixed before the result).

  On that word EvalSuite re-arms the launcher on card_w24c_env.txt (BigMoE 8aecd3c, blob 5d2b765e), with that blob alone.
  Nothing is copied over card_w24_env.txt. Otherwise w13t runs as committed.
  - **The candidate:** it differs from w13t in exactly 4 keys:
    - BIGMOE_TIER_CM 130 -> 140;
    - BIGMOE_TIER_CL 26 -> 28;
    - R57_ROWCAP 13 -> 14;
    - R57_D7B code -> all: only the 4-bit DFlash2 drafter, so GPQA and IFBench prompts typed math or agent draft on it too.
      The screen accepts "all" (line 149) and then loads only that drafter (line 862).
  - **Its dry run (23:13) passes:**
    - 42 keys, none repeated;
    - C 100 and MEM 24;
    - the same snapshot, screen (6ea91764f855), .so (ff69fa24a745) and code pin (27ac500bd0e0) as w13t;
    - 1,500 s per process and 381 keys per order.

  The choice depends only on W24c's own HumanEval speed and memory, never on card-gate rows. The decision rule is
  unchanged. The RESULT names the config that ran, from the launcher's "env ready" line and the frozen copy.
- **One neighbour (Kamen via the orchestrator 23:2x; NewIdeas' rules 23:13):** one job of at most 12 GB may run beside
  the arm.
  - **The yield:** at each process start, runner 61 yields to another session's shared MEM run (the day rule: any run
    above 5 GB). The threshold is now CARD_YIELD_GB, and the w24 launcher passes 12: the arm yields only to a run above
    12 GB, or to two runs above 5 GB. Other arms keep the default of 5. Tests: a unit test (11 cases); card_m1b_env.txt's
    dry run is unchanged; a value that is not whole GB stops the runner.
  - **The arm's own guards are unchanged:** 28 GB NO-SWAP free before each process, a stop if swap is more than 2 GB
    above its base, and the code pins.
  - **Tokens:** within a process the arm's tokens do not depend on load (BigMoE 22:31: w12 vs w12r 24/24, w12t 24/24). A
    slowed process is cut sooner, and the cut row restarts cold in the next process, as in any run.
- **Margin with a neighbour [P]:** the budget sim of amendment 12a, at central rates, with every row and load slowed by a
  factor s. The neighbour's real slowdown is not measured.

  | s | Processes per order | Both orders |
  |---|---|---|
  | x1.25 | 20 | 10.2 h |
  | x1.5 | 22 | 12.5 h |
  | x2 | 26 | 17.4 h |
  | x2.5 | 31 (past the stop at 30) | - |

  The longest row (256 s at central rates) stays under the 1,240 s cut up to about x4.8. The clock binds first: from
  02:10 the central case passes 13:30 at a sustained x1.37, and the slow case at x1.14. A watcher, not the runner, writes
  results/w24_gate_eta.txt after each process. The neighbours stop if its projected finish passes 13:30 (NewIdeas' rule).
- **Unchanged:** the keys, the reference, the decision rule, both orders, PERPROC 20 and the 15:00 cutoff.

## Amendment 12c (Mon 9 Nov 23:3x, before any w24 row): an earlier start once Leap's tests are done and the config is settled; scheduling only
- **Why (NewIdeas 23:30, the queue owner):** Kamen handed the Mac over at 23:2x and closed Chrome (NO-SWAP 41 GB at
  23:30), so tonight's chain (W24c, then q1u_0_live with S1p, then Leap's tests) may end well before 02:10. 02:10 was
  chosen so that Leap's tests fit first (the orchestrator 23:2x). At the slowest pricing a 02:10 start ends at 14:04, past
  the neighbours' 13:30 line even with no neighbour.
- **The rule (tools/card_w24_start.sh):** the arm still starts at 02:10:00 at the latest. It may start earlier, but not
  before 01:25:00, once all three hold:
  - both of Leap's done files (~/Leap/data/drafter_1109.done and prefill_1109.done) are newer than 23:30:00 and 300 s
    old;
  - results/w24c_verdict.txt exists. EvalSuite writes it when NewIdeas' W24c verdict arrives, either way, so the config
    (amendment 12b) is settled before the first row.

  With no verdict, 02:10 stands. Leap's tests still run first, and the 5 min after its files lets them exit. A test runs
  the launcher in dry mode on fake files (7 cases, all as intended): the path off; a missing verdict file; files and the
  verdict present; one file missing; a file older than 23:30; the floor in the future; a 15 s delay. The launcher was
  stopped before the edit and re-armed at 23:34:37 (pid 61821).
- **Unchanged:** the config rule, the env pin, the neighbour rule, the keys, the reference, the decision rule, both
  orders, the latest start (05:00) and the 15:00 cutoff. From 01:25 the priced lengths end at 09:31 (central), 11:19
  (slow) and 13:19 (slowest).

## Amendment 12d (Mon 9 Nov 23:5x, before any w24 row): W24c's outcome, a second candidate (W24d), one final word before the start; operations only
- **W24c's outcome [K] (BigMoE docs/W24_DECISION.md, bd719b4, written after the data):**
  - BigMoE's conditions 1-3 passed: HumanEval whole decode pooled over the two 14-row repeats 126.11 tok/s; 0 bad rows;
    peak 23.808 GB.
  - Condition 4 failed: the 14-row config was +4.1 % over the same-run w13t controls, against a bar of > +4.6 %.
  - The orchestrator overrode condition 4 and chose the 14-row config for the gate, subject to NewIdeas' number check
    and relayed by NewIdeas. Its stated reasons, both measured in that run:
    - w13t peaked once at 24.006 GB (w13d, HumanEval/68) with tokens identical to w13c (23.916 GB);
    - the run's repeats agree within 0.4 %.
  - The card gate's own rule is untouched: the config choice used only W24c's HumanEval speed and memory, never
    card-gate rows. Until NewIdeas' word arrives, the launcher stays on w13t.
- **A second candidate (BigMoE 87dd1a3; the orchestrator's "option B"; W24_DECISION.md section W24d, 498691a):**
  data/stack/card_w24d_env.txt (blob e06f495e) is w13b: 141 base + 26 sidecar slots, 13 rows, R57_D7B=all.
  - **When it replaces the 14-row config:** only under section W24d. That section runs a same-run ABBA on HumanEval and
    requires all of:
    - more than 1 % faster, pooled;
    - every w13b peak below 24.000 GB;
    - no bad or cut rows;
    - everything landed by 01:30.
  - **Key differences:** 3 keys from card_w24c_env.txt (CM 140 -> 141, CL 28 -> 26, ROWCAP 14 -> 13) and 2 from w13t
    (CM 130 -> 141, D7B code -> all).
  - **Its dry run (23:51) passes:**
    - 42 keys, none repeated;
    - C 100 and MEM 24;
    - the same snapshot, screen 6ea91764f855, .so ff69fa24a745 and code pin 27ac500bd0e0;
    - bm_pin OK at 87dd1a3;
    - 1,500 s per process and 381 keys per order;
    - the launcher's dry run on a frozen copy with its blob.
- **One final word:** the arm runs the config that NewIdeas names in its final word: w13t, the 14-row config
  (card_w24c_env.txt, 5d2b765e) or w13b (card_w24d_env.txt, e06f495e). EvalSuite re-arms the launcher on that file with
  that blob alone. Only then does it write results/w24c_verdict.txt, amendment 12c's early-start condition; a partial
  word never writes it. With no final word by 02:05, w13t runs at 02:10. The RESULT names the config and reports every
  lifetime peak against the 24.0 GB line.
- **The final word (Tue 10 Nov 01:3x, before any w24 row): w13b**, data/stack/card_w24d_env.txt, blob e06f495e (BigMoE
  87dd1a3).
  - **Who gave it:** the orchestrator, in NewIdeas' place. NewIdeas could not answer before 02:05 (its account's limit
    ran until 02:00). The orchestrator states that it did the number check from the raw rows and
    data/w24/w24d_verdict.txt. Its word reached EvalSuite at 01:33, after section W24d's 01:30 for NewIdeas' check
    (condition 4). The orchestrator ordered W24d; this deviation is recorded here on its word.
  - **W24d's outcome [K] (BigMoE docs/W24_DECISION.md "W24d OUTCOME", f0c8d2a; data/w24/w24d_verdict.txt, 23:56:44):**
    w13b pooled 130.33 tok/s against the 14-row config's 125.91 (+3.51 %, bar > +1 %); w13b peaks 23.819 / 23.781 GB;
    0 bad or cut rows in the four arms.
  - **EvalSuite's recheck (01:34), from ~/BigMoE/data/r57/{w14c,w13e,w13f,w14d}h_100_runs.json:** the same numbers.
    Each arm has 12 rows, with the same keys in the same order. Whole decode: w14c 126.35, w13e 130.20, w13f 130.46,
    w14d 125.48. Peaks (max of mem[1], mem[-1]): 23.828, 23.819, 23.781 and 23.782 GB. No bad or cut rows.
  - **The env:** card_w24d_env.txt is still blob e06f495e, tracked, and equal to BigMoE's HEAD (last change 87dd1a3,
    23:50:55). Its dry run (23:51) passed.
  - **The re-arm:** the w13t launcher (pid 61821) was stopped at 01:35, before any start (no runner, no q61.pid). It
    was re-armed at 01:35:23 (pid 78825) with W24S_ENV=card_w24d_env.txt and W24S_BLOBS=e06f495e alone.
    results/w24c_verdict.txt is written after this commit, so amendment 12c's early start can apply (Leap's two done
    files are from 00:18:12).
  - No w24 row existed. The config choice used only W24d's HumanEval speed and memory.

## Amendment 13 (Tue 10 Nov 02:1x, before any row of this arm): a 12 GB IFBench arm beside the w24 gate (ifb12); IFBench only
- **Why:** the orchestrator (23:2x) asked for 12 GB work beside the w24 gate. NewIdeas (23:30, route (i)) placed Fit12's
  12 GB chat mode (P2', the chat-width rule) there as the gate's one neighbour. This arm gates that mode on IFBench alone.
- **Who:** Fit12 runs it in its own session and lock; EvalSuite pre-registers and scores it. The files, all checked by
  EvalSuite at 02:0x:
  - runner ~/Fit12/scripts/ifb12.sh (sha256 2143f25c7ff7);
  - merge ~/Fit12/scripts/ifb12_merge.py (ac65fec294ec);
  - frozen harness ~/Fit12/snap/chw1108/fit_t3.py (7a9ffd8375ed);
  - env record ~/Fit12/data/gate/card12_chw_env.txt (git hash-object blob da1eb63570ee). The runner passes its seven
    values: F12_T3ARMS=ex, F12_CAP=item, F12_CMCL=47,32, F12_W1=cf:0.9:2, F12_W1SEED=1, F12_W1TAIL=1, F12_PR1=1.
- **Keys and prompts:** the 150 gate IFBench keys (the ifbench/ lines of results/card/card_keys.txt), in that order, 20
  keys per process. Prompts are the items' own prompt_ids (data/items/dec_ifbench.jsonl); their lengths equal the NAX6-N
  reference rows' n_prompt for 150 of 150. The cap is the item's max_tokens, 1,536.
- **One order, plus a reverse check:** one order only, on Fit12's history-independence data
  (~/Fit12/data/scr/chw18/chw18r_ids.json). After the 150 keys, the first 10 run again in reverse order in their own
  process (ifb12rev: ifbench/116 ... ifbench/10).
  - Any token difference among those 10 makes the arm VOID.
  - So does a key with two different runs (the merge's "clash").
  - With fewer than 10 re-run, there is no verdict.
- **Rows:** ifb12_merge.py writes ifb12_tokens.json ({key: {"ex": [[ids]]}}, each key's first complete run) and
  ifb12_runs.json (cut flags). EvalSuite turns them into results/card_ifb12/decode.jsonl with tools/r57rows.py: arm ex,
  cut after the first EOS or at max_tokens. A cut run is not a row.
- **Decision:** `tools/set_d4.py ifbench card_ifb12 ref2tnaxn.r6`, decision 4: a paired delta above -5 pt and an exact
  McNemar p of at least 0.05, on all 150 keys. With fewer than 150 rows there is no verdict; the arm may resume later
  under these rules. The RESULT says it gates the 12 GB chat mode on IFBench only: one order, not a full card gate.
- **Memory and speed:** the RESULT reports the largest lifetime footprint against the 12.0 GB line. The runner stops any
  process above 12.0 GB, and that process's keys are not rows. Speeds are rough, because the arm runs beside the gate.
- **Beside the w24 gate (amendment 12b, one <= 12 GB neighbour):**
  - Its lock reads "MEM 12 GB", so the gate does not yield to it.
  - A process starts only when results/w24_gate_eta.txt is at most 30 min old, says "neighbour may start: yes" and does
    not say "passes 13:30: yes". The first process also needs Breakthrough's beside_gate.done.
  - It stops on a fresh "passes 13:30: yes", swap above its start + 0.5 GB, or a footprint above 12.0 GB.
  - Once the gate has ended, a process runs only when no other session holds a MEM > 5 GB lock and no quiet ticket waits.
- **Start:** held (Fit12's WAIT file) until this amendment is committed. Fit12's start time goes below as a dated line.
- **Started (Tue 10 Nov 07:1x):** Fit12 set GO at 07:14:09 (~/Fit12/logs/summary.txt). Process ifb12c01 (20 keys from
  ifbench/10) started at 07:14:32 in neighbour mode. Its first row file is from 07:14:55.
  - The files are as registered (rechecked 07:1x): fit_t3.py 7a9ffd8375ed, ifb12.sh 2143f25c7ff7, ifb12_merge.py
    ac65fec294ec, env blob da1eb63570ee.
  - At its start the w24 gate had fwd 381/381 and rev 300/381, with a lifetime peak of 25.112 GB.
  - The summary.txt lines at 02:04-02:05 are Fit12's dry tests: a fake harness with 10 s chunks, and the 12 GB stop
    path (rc 126). The arm's folder held only WAIT at 02:08:33, so no row predates this amendment.
  - The start came five hours after the amendment because EvalSuite's 02:09 release note reached Fit12, an idle session
    on another account, only at 07:1x.
