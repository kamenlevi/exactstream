# Idea queue for the two lines (NewIdeas; started Sat 02:16). Take the TOP open item of your line; mark it [taken: you, time] here.
# Rules: pre-register the kill line before the run, check every number (others' results are claims until verified), GPU tests as
# one unattended batch, process < ~45 GB, swap + disk checked, no downloads without the orchestrator + Kamen's yes.
> (a quote from the project owner, kept private)
# 24gb, so for me the 45gb one news are with priority"): the 45 GB route's walls go first in the GPU queue and in idea hunts: bytes
# per 16-row check (13.2 GB; 350 needs ~5.5), tokens kept per check (D-rows), check overheads (R-40 REF2, R-50 / R-50b, R-52b),
# exactness of the 16-row live loop. 24 GB follows; 12 GB keeps its sessions, behind 45 GB work in the queue.

## ORDER BY GAIN (Thu 5 Nov 19:3x; Kamen ~19:3x: publish everything once LOSSY reaches 250-330 tok/s at 45 GB on code
## AND 90-130 tok/s at 12 GB). Measured first, then priced. Every speed line names its workload: HumanEval-style code is
## flattered by prompt copying (its answers re-emit the prompt's signature and docstring), so code gains are given per set.
Tags: [K] = measured, [P] = priced from measured parts, [G] = a guess. 45 = the 45 GB line, 12 = the 12 GB line.

> (a quote from the project owner, kept private)
chat to 150 in a day or two ... It's not even an mtp head drafter that's the problem ... Use neural accelerators where
needed, use the cpu to its full to 'prep' the math which is about to go to the gpu if needed - do anything to get it to
150 in lossy at 45gb by Monday latest". Lossy is fine, IFEval-gated vs NAX6-N in both orders.
- SUN 15:23 (`date`) STRATA H2H PARTIAL (the orchestrator's run; ~/StrataEval/h2h/results/REPORT.md) + MON SLOT
  - WAITER END 15:19: INCOMPLETE after 3 attempts (waiter.sh H2H_ATTEMPTS, default 3). All three stops were fanwatch
    heat pauses on thermal state 'fair' (die only 69-80 C), each ~2.5 min: 14:51, 15:03, 15:19.
  - Done: B1 (default + MTP) and A1 (default, no MTP). A2 was cut at 44/48. The exact arms C and D never ran.
  - Pooled decode [K], re-derived by NewIdeas from the request rows (it equals REPORT.md's pooled column):
    - IFEval: A 28.0 / B 37.4 tok/s, TTFT median 0.86 / 0.97 s.
    - All 4 sets: A 26.1 / B 38.9 tok/s. Peak footprint 34.9 / 33.6 GB.
    - Ours on the same 12 IFEval, POOLED (NUMBER_CHECK 0a-2; the N45 report's 39.9 / 45.0 are geo means): exact 37.1 (N45) / 38.3 (CHS4 yes1), L 43.4 / 44.2, at 41.7-42.4 GB.
    - Strata runs IQ3_XXS, so its tokens differ from NAX6-N's (1 of 12 IFEval rows equal). INFORMATION for the table.
  - Told the orchestrator 15:23: relaunch with H2H_ATTEMPTS=10 (finished arm+passes are kept). A relaunch also puts
    Strata ahead of PFB again, because PFB's AFTER (DRIVER_DONE, or a last waiter.log line 'WAITER END') is met now.
  - The orchestrator 15:2x: the waiter was relaunched 15:22 with 12 attempts and fan cap 6000-6500 until 17:00 (Kamen
    is off the laptop; his 11-06 cap). A2 re-ran whole: 48 / 48 at 15:29, IFEval pooled 25.6 tok/s (re-derived). B2
    done 15:39. The exact arms C / D were next at 15:41.
  - FINAL 16:3x (re-derived; PUBLISH_INVENTORY 'STRATA HEAD-TO-HEAD, DONE'): 8 / 8 at 16:07, B and D again at a 33 GB
    arena 16:12-16:25, whose peak is 40.8 / 41.2 GB, so it pairs with our 45 GB line. Lossy: ours x1.24-1.29 (~41 GB)
    and x1.37 (~32 GB). Exact: 0.91-0.94x and 0.94x (pooled; corrected 17:16 from the geo 39.9) (Strata's exact is exact only to its IQ3_XXS weights). TTFT: Strata
    0.83-1.06 s vs ours 2.9-3.1 s, so end to end ours is behind (24.8 vs 29.7 at ~41 GB) until PF-D (CT1 below). PFB stays blocked until DRIVER_DONE (EvalSuite amendment 4: only
    DRIVER_DONE or 'WAITER END: DONE' releases its card reference).
  - MON 07:05 SLOT, CORRECTED 15:24 (the orchestrator): hard stop 15:00, not 09:00. Kamen's schedule (11-02): Mon-Fri
    the big-test window runs 00:30 until at least 15:00. Strata's old 09:05 was only that job's length.
    - Order: card-reference leftover (0.5-2.5 h), then lcoh (~3.6 h), then whichever next arm fits.
    - The last chunk or arm must END by 15:00.
    - Fit12's 12 GB IFBench arm (~3 h, MEM ~12-16 GB) goes after 15:00, by day.
  - THU 05:42 (`date`) v1.1.0 PUBLISHED Wed 16:52 on Kamen's go (release eb72552; site live with small-ram). The orchestrator
    number-checked v1.0.1's erratum and v1.1.0's deltas itself. Now: BigMoE's DEEPDRAFT capture night 1 holds the quiet Mac
    until ~13:30.
    - THU F2b (MTP Head; turns/f2b.sh, DRY-tested): 45 GB class, quiet whole Mac, >= 6 h minimal / 9-10 h full. It takes the
      first free night of >= 6 h after BigMoE's DEEPDRAFT nights, likely a weekend night. The orchestrator names the night
      with Kamen; MTP Head arms it held with F2B_NB / F2B_END.
  - TUE 11:17 (`date`) S1q LOWER-MEMORY ARM PARKED (NewIdeas' call) until after the release. Fit12's pre-checks: the same chunk
    peaks at 11.725 GB in a re-run and 12.027 GB in the gate, with identical tokens (18 / 18). The ~0.3 GB run-to-run swing is
    outside MLX (MLX flat at 10.40-10.46). The arm needs ~0.4 GB of real cuts. The drafter cut would move QOM's cut, so it
    needs a new map line too. Next time: first locate the ~1.0-1.3 GB outside MLX (footprint / vmmap categories).
  - TUE 11:07 (`date`) S1q INFO SCORE (EvalSuite ef7d28b; NOT a verdict): HumanEval+ 91.7 vs 91.7 % (60 / 60, 1 / 1), MBPP 75.9 vs
    75.9 % on the first 58 keys (2 / 2). Memory stopped it, not quality, so a lower-memory arm is APPROVED for a slot after
    the release: C_m 51 kept, <= 11.8 GB measured on mbpp/228 first, its own PREREG, and Fit12's q0r merge fix.
  - TUE 11:05 (`date`) S1q CODE GATE STOPPED at 11:02 on the 12.0 GB guard (12.027 GB on mbpp/228; forward 118 / 160; no verdict;
    EvalSuite RESULT_code12_s1q 0486e07). NewIdeas' call: no retry today; EvalSuite gives an information-only score of the
    118 rows. A retry is a new arm with its own PREREG after the release: keep C_m 51 (the map PASS line), take >= 0.2 GB
    elsewhere, measured <= 11.8 GB first. The release check is DONE (NUMBER_CHECK 25).
  - TUE 10:31 (`date`) MORNING RESULTS + TODAY:
    - The w24 gate PASSED 08:41 on all 6 lines (EvalSuite f6613ff), on w13b. Lifetime peak 25.112 GB; 22 of 40 processes went
      over 24.0 GB, every one with a prompt of 739+ tokens.
    - W24e (09:00): no smaller prefill chunk keeps tokens and hashes, so w13b ships as gated (PFCHUNK 1024), labelled 'under
      24 GB on prompts up to ~560 tokens, up to 25.1 GB on longer ones'.
    - W24f: MBPP 91.3, no gain vs w13t (within noise); IFEval +2.8 %.
    - qom_0_live_c47: code FAIL by one flip (1.37 %); the probe needs pool 51.
    - Fit12's ifb12 PASSES IFBench loose (40.0 vs 44.0 %, p 0.21). Strict drops 7.3 pt (p 0.013, information).
    - S1q (Fit12, 10:27, on the orchestrator's GO): QOM0 at C_m 51 / C_l 32 + a GPU residency probe. VALID: misreads 0 / 16,128,
      peak 11.765 GB. Speed 41.77 tok/s pooled, 132 MB per kept token.
    - TODAY (GPU idle): EvalSuite's HumanEval+ / MBPP code gate on S1q offered as the one big job, for a gated 12 GB code row in
      SMALL_RAM. EvalSuite decides whether and how.
    - NewIdeas' number check of SMALL_RAM's 24 GB fill and Leap's pkg_w24c: asked the orchestrator whether to do it now
      (release work) or on its release trigger. Paced asleep until ~04:00 Wed otherwise.
  - TUE 02:02 (`date`) TUESDAY ORDER. NewIdeas is paced to sleep until ~17:42 (weekly 50 % vs line 41 %). Messages wake it;
    the orchestrator decides in between.
    - Tonight's chain ended. W24c / W24d ran: w13b won (130.33 vs 125.91, +3.51 %; NUMBER_CHECK 24). Leap's drafter pair and
      prefill tests ran, plus its PF-B exact check (PF-B != PF-A at 30K), and BigMoE's PF-D long-prompt check (exact).
      q1u_0_live: code FAIL (1.96 %), so S1u is out. The erratum v1.0.1 is staged for Kamen's go; on go, NewIdeas checks the diff.
    - NOW to the gate's end: the w13b gate has run since 01:36:43. ETA 08:44 at the measured pace (central 09:41, slow 11:22).
      Its peak was 24.813 GB on a GPQA prefill, so neighbours budget the gate at 25-27 GB.
      - The one <= 12 GB neighbour is Fit12's 12 GB IFBench arm (ifb12.sh), once EvalSuite's amendment 13 has Fit12's start
        time, harness sha and env blob. Guards as set. Breakthrough has none: its arms need ~16 GB; beside_gate.done 00:03.
    - AFTER THE GATE, by day one > 5 GB job at a time, in this order as each is ready:
      1. BigMoE's W24e FIRST (GO 02:06): w13b's prefill chunk 1024 / 512 / 256 / 1024 on the gate's 6 longest prompts, hashes
         on, ~15 min, quiet, PEAK 26 (NO-SWAP >= 34). AFTER = EvalSuite's 'runner 61 (w24) ended'. Then any other BigMoE arms.
      2. Breakthrough's qom_0_live_c47 (~16 GB, ~4 min, flips only; S1u's fallback line; masks built and checked).
      3. (S1p is DONE: it ran 00:01-00:04 before the gate. Measured loader idle 102.4 ms; NUMBER_CHECK 23.)
      4. Fit12's IFBench arm continues alone if not finished, yielding at chunk ends to any quiet ticket.
    - EVENING (by day only ~36 GB is free): Leap's PF-B length bisect (exact-32, 10K / 15K / 20K, ~2 min per run, peak
      ~40 GB, NO-SWAP >= 48).
  - MON 23:30 (`date`) CHAIN LIVE (Kamen handed over 23:2x; Chrome closed; NO-SWAP 41.2 GB, swap 5.6 GB at 23:30):
    1. W24c (turn45; start by 01:15; decision arms start by 01:30; diagnostics only if they can start by 00:40). EvalSuite's
       dry run of card_w24c_env.txt PASSED (blob 5d2b765e). Relay: NewIdeas tells EvalSuite by 02:05, PASS only if BigMoE's
       verdict says PASS (its full rule) AND NewIdeas' 4 checks pass. Else w13t.
    2. Breakthrough runner v3 (dc0bfd5): waits for W24c's turn45 to end (until 01:15); masks by 01:25; arm start by 01:30,
       stop 01:45. S1p in the gap after W24c's end line (start by 01:30).
    3. Leap runner pid 74101: on q1u_0_live.done (fallback 01:35); NO-SWAP >= 39; share 'MEM 38 GB' to 02:05. Jobs: 32 dfl / 32
       mtp, then prefill 10k / 30k x default / pfa / pfb. No start after 01:58; all capped at 02:04.
    4. Gate 02:10 (w13t unless the switch passes). Suggested to EvalSuite: start at max(01:25, prefill_1109.done + 5 min).
       Its slowest pricing (14:04) passes 13:30 even with no neighbour.
    - BESIDE THE GATE (EvalSuite CONFIRMED margin for one <= 12 GB neighbour; amendment 12b: the gate yields only to a shared
      MEM run > 12 GB, or to two > 5 GB; it stops itself at swap +2 GB; ETA file ~/EvalSuite/results/w24_gate_eta.txt,
      first update ~02:35):
      a. Breakthrough's flip-only arms (qom_0_live_c47, q1u_0_live if not run), each <= 12.0 GB measured, from 'neighbour
         may start: yes'. Then ~/Breakthrough/data/m1/beside_gate.done (03:00 at the latest).
      b. Fit12's 12 GB IFBench arm in Fit12's own session and lock, EvalSuite scoring afterwards, if Fit12 builds the guards
         and passes the format check. EvalSuite REFUSED a second instance of runner 61. Else EvalSuite runs it alone after
         the gate.
      - Guards for both: NO-SWAP >= peak + 8; stop on swap +0.5 GB, footprint > 12.0 GB, or ETA 'passes 13:30: yes'.
        Short lock reasons starting 'MEM n GB'.
  - MON 23:14 (`date`) BESIDE-THE-GATE CANDIDATES:
    1. Breakthrough's flip-only map arms, if any run at <= 12 GB (asked; q1u_0_live peaks ~17). Short; they go first.
    2. Fit12's 12 GB IFBench arm (quality): P2' chat, EvalSuite's runner, harness Fit12/snap/chw1108, env
       Fit12/data/gate/card12_chw_env.txt, one order. Peak 10.9-11.0 GB [K] at CAP 128; at max_tokens 1536 <= ~11.5 [G].
       ~5 h alone, 7-10 h beside the gate. It gates SMALL_RAM's 12 GB chat row before Wednesday.
       - Guards: start at NO-SWAP >= 20; the first chunk measures the peak; stop if footprint > 12.0 GB or swap +0.5 GB.
       - If it runs past the gate's end, it continues alone as the day's one big job, yielding to BigMoE's quiet arms.
       - Pending EvalSuite: can its runner run it beside the gate, and does the gate keep margin under 1,200 s?
    - Not beside the gate: S1p (needs clean loader clocks) and S0 WV16 (not built; low value, dropped).
  - MON 23:13 (`date`) BESIDE THE GATE (Kamen's idea, cleared by the orchestrator 23:2x): 12 GB work may run beside the w24 gate.
    - Rules: one job at a time, <= 12 GB measured peak; flip-only map arms / quality jobs first; speed screens labelled
      rough; memory must hold (gate + job + his apps; NO-SWAP >= peak + 8 at start, stop on swap growth > 0.5 GB); the
      neighbours stop if the gate's projected finish passes 13:30; no 32 GB work beside it (Leap's tests stay before 02:10).
    - Precondition: EvalSuite confirms its chunks keep margin under R57_HARD 1,200 s with one <= 12 GB neighbour (asked). Also
      asked for an ETA file the neighbours can read (~/EvalSuite/results/w24_gate_eta.txt, proposed).
    - Candidates asked from Fit12 and Breakthrough, with measured peaks. Guesses: Fit12's S0 WV16 (bitwise, timing-free) fits;
      Breakthrough's map arms fit only if they run at <= 12 GB (q1u_0_live peaks ~17). S1p does NOT go beside it: its point is
      clean loader clocks.
    - S1p re-armed with start-by 01:30 (runner pid 30324, helper pid 30387; ends by 01:40).
  - MON 23:12 (`date`) GATE MOVED TO 02:10 (Kamen's yes via the orchestrator 23:2x); the 32 GB tests run tonight; the switch
    deadline is 02:05. Order once Kamen is idle (everything needs Chrome's ~17 GB freed first):
    1. W24c (BigMoE re-armed 23:10: Kamen idle 10 min + no chat > 2 GB, WAITMEM 29; start-by 00:50, may move to 01:15).
       Decision passes first: HE w13c w14a w14b w13d (~8 min), then w14a MBPP + IFEval. Candidate env
       data/stack/card_w24c_env.txt (CM 140, CL 28, ROWCAP 14, D7B all); EvalSuite can dry-run it now.
    2. Breakthrough's q1u_0_live: masks once W24c's quiet window ends, by 01:25; map arm start by 01:30, stop 01:45 (asked).
       Fit12's S1p goes in the mask gap after W24c's end line, start by 01:30 (asked), ends by 01:40.
    3. Leap's 32 GB drafter pair + prefill test, on q1u_0_live.done (fallback 01:35), NO-SWAP >= 39, each done by 02:05.
       The 45 GB pair stays out. Leap was asked to arm a runner (it had none).
    4. 02:10: EvalSuite's gate (re-arming), w13t unless the switch rule passes by 02:05.
    - Kamen asked whether 12 GB work can run beside the gate. Answer sent: yes for the verdict (BigMoE 22:31), if memory holds
      and EvalSuite's chunks keep margin under 1,200 s. Speed screens beside it give rough timings only; map arms are
      unaffected.
  - MON 23:08 (`date`) GATE-SWITCH RULE (orchestrator 23:1x; told BigMoE + EvalSuite): the 01:25 gate switches from w13t to W24c's
    config only if ALL hold by 01:20: (a) W24c >= 120 tok/s whole decode on HumanEval with a lifetime peak under 24 GB; (b)
    NewIdeas' number check passes from its runs.json; (c) EvalSuite's dry run of the new env passes. Else w13t runs as
    committed, and a 120 config gets its own gate Tuesday night.
    - NewIdeas' check, fixed BEFORE the result (scripts/numcheck/w24c_check.py; reproduces W24b's w13t 116.45 / 23.915 GB):
      1. The 14-row config's HE whole decode, pooled over all 12 prompts and both repeats (w14a + w14b): >= 120.0.
      2. No HE row has a non-empty 'bad' field. A PAUSED row is unclean: no switch tonight, re-run Tuesday.
      3. Lifetime peak, max mem[-1] in 10^9 bytes: < 24.000 in every repeat.
      4. The controls w13c / w13d agree within 2 %; otherwise drift is FLAGGED (not a fail).
    - A watcher wakes NewIdeas when w13dh_100_runs.json lands (the 4th HE arm) and runs the check at once.
    - Leap's tests: after the gate (agreed). Fit12's stray second S1p (pid 14901, a heredoc slip) is stopped; it never ran.
  - MON 23:06 (`date`) FINAL ORDER TONIGHT (orchestrator 23:0x, by Kamen's priorities; he is on the Mac until ~23:50):
    1. BigMoE's W24c first (~00:00 once Kamen is idle; ~25 min, quiet, PEAK 25; no start after 00:50). Its AFTER opens at
       00:00 via the s1u.done + 60 min fallback. NewIdeas number-checks any 120 claim at once.
    2. Breakthrough's q1u_0_live: masks (held during W24c's quiet window, ~20 min), then the map arm, start by 01:00.
    3. Fit12's S1p after W24c's end line, in the mask-build gap (~4 min, <= 12 GB, share, start by 01:00).
    4. Leap's drafter and prefill tests can't finish by 01:15. They move to after the gate, in practice Tuesday evening:
       the 32 GB pair needs 39 GB NO-SWAP, and by day only ~36 GB is free. Leap is told; it has no runner armed.
    5. 01:25: EvalSuite's gate alone (~8 h). After it: BigMoE's lever arms first, then the 12 GB re-pricing work (S1p's
       Tuesday GO if it doesn't run tonight).
    - Live at 23:05: W24c turn45 pid 5436, Breakthrough runner pid 8447, Fit12 s1p.sh + s1p_go.sh. Asked Fit12 about a second
      s1p.sh (start-by Tue 18:00, pid 14901).
    - PROBLEM: NO-SWAP is 22.9 GB at 23:05 (Chrome ~17 GB). W24c needs 29, the gate 28, the map arm 25, S1p 20. The
      orchestrator is asking Kamen.
    - OPEN: does W24c's result switch the 01:25 gate's config? BigMoE says no (w13t stays; a 120 config gets its own gate
      Tue). If yes, the 120 claim needs NewIdeas' number check and EvalSuite's env dry run first.
  - MON 23:04 (`date`) S1u DONE (checked, NUMBER_CHECK 23) + S1p PLACED:
    - S1u 22:37-22:52 on Kamen's word: 41.23 tok/s pooled, 11.335 GB, x2.39 vs exact paired (not x3.2). The read split is [P].
    - Since 23:00 Kamen has been on the Mac (idle < 600 s, his apps 25 GB, GPU 52 %, NO-SWAP 22.9 GB at 23:03), so the
> (a quote from the project owner, kept private)
    - S1p (Fit12, pre-registered in its PLAN): S1u's loader clocks per wide call (F12_GSTAT=1), measuring the read split
      the [P] estimate approximates. ~4 min, <= 12 GB, share, 10-min stop.
      - GO given for tonight, last: after a W24c end line in BM:data/turn45.log, start by 01:00, Fit12's usual gate.
      - Else Tuesday's first gap after the w24 gate. NewIdeas gives that GO; BigMoE's ~30-min slot goes first if ready.
> (a quote from the project owner, kept private)
    more tan 15 min. and dont close spotify"):
    - S1u runs now as a shared run (MEM 12, hard stop 15 min, until ~22:57). Only S1u moved.
> (a quote from the project owner, kept private)
      - Breakthrough: its runner sleeps to 23:00:00 first (s1u_map_night.sh l.116).
      - BigMoE: turn45 NOTBEFORE 23:00.
      - Leap: no runner file to read; asked to require the clock, Kamen idle and q1u_0_live.done.
    - #4 is BigMoE's W24c (BM da35564 + 13cd1cd): the diet drops the bf16 drafter (1.14 GB), which buys 14 rows. Six HE arms,
      then MBPP + IFEval on two; ~25 min, quiet, PEAK 25 (expect <= 24); queue by 00:40, last arm start 00:55, ends <= 01:15.
      It starts on Leap's drafter_1109.done + prefill_1109.done (Leap asked to touch both), with fallbacks q1u_0_live.done + 40
      / s1u.done + 60 min (~23:57). A config that reaches 120 needs its own gate (Tue after w13t's, or Tue night).
    - Load question answered (BigMoE 22:31): w13t's tokens are independent of load within a process; across an R57_HARD
      1200 s cut, the rerun starts with a cold pool, so they can differ. Scores stay a valid sample. A <= 12 GB job beside the
      gate is fine for the verdict if memory holds and EvalSuite's chunks keep margin under 1200 s (EvalSuite to say). By day
      the one-big-job rule still holds.
> (a quote from the project owner, kept private)
    1. Fit12's S1u from 23:00 while Kamen is idle (~12 min, ~11.4 GB, share, start-by 00:58). Its hold on Leap's done-file
       is released. Done-file ~/Fit12/logs/s1u.done. Fit12 SendMessages the orchestrator the numbers as soon as it ends:
       whole decode tok/s, check ms (SSD read / non-read), MB per kept token, kept per check, peak GB. NewIdeas checks them.
    2. Breakthrough's q1u_0_live right after (runner scripts/m1/s1u_map_night.sh, 9bea46d; masks built from 23:00 beside S1u
       on one background core; bit-check first). Done-file ~/Breakthrough/data/m1/q1u_0_live.done.
    3. Leap's drafter test and prefill test (the prefill test replaces the chat replay), on Breakthrough's done-file. Fallback:
       20 min after s1u.done with no map or s1u process left.
    4. A BigMoE lever arm only if one is ready (no answer yet).
    - Unchanged: everything is done by 01:15 and the 01:25 gate is untouched. The 21:16 chain's order below is superseded.
  - MON 21:16 (`date`) TONIGHT'S CHAIN (each job starts when the one before it leaves its done-file; all gated on Kamen idle):
    1. Leap's drafter test from 23:00: 4 runs (dfl / mtp x 32 / 45 GB), share, ~12-15 min. MLX peaks ~31 / ~44 GB need
       NO-SWAP >= 39 / 52 (43.6 at 21:15, so the 45 GB pair may be skipped). Done-file ~/Leap/data/drafter_1109.done.
       The chat replay can't run: his messages weren't saved. Any substitute is the orchestrator's call.
    2. A BigMoE lever arm, if one is ready (quiet ticket; asked 21:08). Fit12's gate waits while any quiet ticket is out.
    3. Fit12's S1u (GO 21:16): ~10-12 min, peak ~11.4 GB, share MEM 12, start-by 00:58. Done-file ~/Fit12/logs/s1u.done.
    4. Breakthrough's map arm q1u_0_live (flips only; masks from dm_oea_masks.py, built after 23:00): 5-10 min, ~17 GB,
       share, start-by 01:00. Pre-registered in TRIED as Q1U-0-LIVE: code PASS = all-row <= 1.34 % AND non-tie <= 0.87 %.
    - Everything is done and freed by 01:15. EvalSuite re-priced the 01:25 gate (21:14): ~8 h central (ends ~09:30), slowest
      11.9 h (~13:20). It waits for 28 GB NO-SWAP at each process.
  - MON 21:07 (`date`) TONIGHT, UPDATED (orchestrator 21:0x; Fit12's two notes since):
    - Until 23:00 the Mac is Kamen's: no GPU, lock or big-memory job. His chat process was gone by 21:05; the rule stands.
    - 23:00-01:15, only while Kamen is idle: no keyboard or mouse input for 10 min (ioreg HIDIdleTime >= 600 s) and no chat
      process of his above 2 GB. Each start is gated on NO-SWAP >= measured peak + 8 (share) / + 4 (quiet).
      1. Leap's drafter test (~12 min).
      2. Exact-vs-lossy replay of Kamen's chat (Leap).
      3. If time is left: BigMoE's next lever arm (quiet, PEAK 24 GB) if ready, else Fit12's S1u if built.
      - Everything ends and frees its memory by 01:15.
    - 01:25: EvalSuite's w24 gate on w13t, as armed (~10 h, cutoff 15:00). Nothing runs beside it until BigMoE confirms that
      w13t's tokens do not depend on machine load. Its LCB set runs QOM (optional uses served only when resident). COH made
      two quiet runs match token for token (BM:docs/WED_TESTS.md 980), but not with an SSD-heavy neighbour.
    - After the gate (~11:25 Tue): BigMoE's lever arms first. Fit12's S1u takes the first gap: one call per check, union-only
      cut, 16 HumanEval at CAP 256, ~15 min, <= 12 GB, share. S0 (WV16 numerics, 5-10 min) is optional in any gap; S2 is Wed+.
  - MON 21:03 (`date`) TONIGHT:
    - W24 ran early on Kamen's word (orchestrator 19:0x); W24b 19:40-19:59.
      - Gate config w13t (BigMoE card_w24_env.txt, 8f76fc3): 13 rows, 130 base / 26 side, prefill trim, R57_HARD 1200.
      - HumanEval 116.45 tok/s at 23.92 GB, re-derived clean: 3 % under 120.
      - Its MBPP row (29.6) and w12r's HE row (14.2) are single last-prompt stalls; never quote them. Re-run w13tm first.
    - Now (21:02): Kamen's chat session (Python 20 GB, swap 31 GB). The queue stays clear until the orchestrator says he
      is done.
    - Next, in order: EvalSuite's w24 card arm (armed from 01:25, latest start 05:00, cutoff 15:00, ~10 h; its WAITMEM
      holds it while memory is short), then BigMoE's lever arms toward 120 in its gaps or after.
    - 12 GB -> 90 (Breakthrough priced: one exact 16-row call + a QOM0+ball cut ~85-90; Q1-t3 ~100): its first GPU
      screen comes after W24, once BigMoE says whether its engine fits 12 GB pools.
> (a quote from the project owner, kept private)
    then we publish by wednesday. and as soon as we reach 90tps on 12gb, we publish about it"). Published 18:35 (exactstream
    v1.0.0); the post-publish number check PASSED (18:55).
    1. FIRST: 24 GB -> 120 tok/s whole decode on HumanEval, lifetime peak < 24 GB, lossy with a same-scores gate on the known
       benchmarks.
       - BigMoE's W24 leads: 3 arms, row cap 12 / 10 / 8, ~45 min, PEAK 24 GB, queued NOTBEFORE 00:30.
       - Then BigMoE iterates; the GPU goes to its arms before anything else.
       - The gate: EvalSuite card arm on the config BigMoE names. Reference rows ref2tnaxn.r6 exist; one arm took 5.8 h at
         45 GB and will be slower at 24 GB. It must end before the Wednesday publish.
       - Slots, in order:
         a. Tue right after W24, if a config is named by ~01:30: the gate ends ~10:00-12:00, inside the 00:30-15:00 window.
         b. Otherwise Tue night 00:30 -> Wed morning.
    2. NEXT: 12 GB -> 90 tok/s (Fit12, Breakthrough). Published on its own once reached and gated. Its GPU screens run in
       the gaps the 24 GB work leaves.
    3. Everything else yields. The new-findings list (EvalSuite) stays due Tue 12:00, CPU only. Account lines: sub2 85 %,
       sub1 60 % by Tue 20:00.
    - Asked of the orchestrator (18:56): may W24 start this evening instead of 00:30? Kamen's call. BigMoE asked to start
      now (18:55); not granted. At 18:59, 54 GB of 64 was in use and Claude was the front app, and WAITMEM would hold it
      anyway.
    - Gate arm (EvalSuite 19:0x):
      - ~10 h (9-12) at 24 GB; a 01:30 start ends ~11:00-14:00.
      - It needs BigMoE's committed card_w24_env.txt. EvalSuite extends runner 61 to C 100 tonight on CPU.
      - Option (BigMoE's code, set before the first row): HARD 600 -> ~1,200 s, about -1 h.
    - 12 GB -> 90 envelope (Fit12 + Breakthrough, CPU pricing tonight):
      - 90 needs a check of ~40-70 ms outside reads AND <= ~50-85 MB read per kept token (today 439-441). Both lossy, both
        gated.
      - Fit12 S1 (wide call) / S2 (BigMoE's fast path at 12 GB, if it fits) / S3 (byte cuts): MEM <= 12 GB, share lock,
        ~15-20 min each. The concrete list comes by Tue 08:00 and runs in the gate's gaps.
  - MON 17:25 (`date`) PUBLISH NUMBER CHECK DONE: the FINAL build 17:23 passes end to end (five papers, site,
    complete.pdf 84 pp). The last catch, MAIN section 11's OPT line (K 266 / 147 mis-paired), was fixed by the editor at
    17:23. Ledger: NUMBER_CHECK.md sections 15-19.
    - Tonight: BigMoE W24, NOTBEFORE 00:30 (PEAK 24 GB, quiet 60 min). NewIdeas checks its start.
  - MON 16:35 (`date`) DONE TODAY + PUBLISH CHECK:
    - Leap store conversion PASS 16:07-16:19:50: 48 layers in 9.5 min, byte-identical to records3.
    - Fit12 chw19 ran 16:20-16:30: C2c chat x1.053 = MARGINAL, code x0.881 = KILL; nothing adopted. The GPU is free.
    - W24 is queued NOTBEFORE 00:30 (BigMoE 66ccdd9).
    - Public build 16:11 (5 papers; SMALL_RAM to Wed):
      - MAIN, KERNELS and EVALUATION pass the digit check end to end.
      - DRAFTER.md and RELEASE.md changed after the build; rebuild asked of Breakthrough.
  - MON 16:09 (`date`) QUEUE + GATES:
    - m1b card gate PASS: six lines at 16:06; my independent re-grade of GPQA and LCB (both orders) is identical.
    - zs32 not gated (amendment 11). Kamen 15:08: publish tonight, all done by 18:00.
    - Order now: Leap's store conversion (after m1b's lock is gone; hard stop 16:58; writes convert_full.done), then Fit12
      chw19 (gated on that file; fallback 17:00; start-by 17:30).
    - BigMoE W24 (Kamen's <24 GB at 120+; 3 arms, PEAK 24 GB, ~45 min) at 00:30 tonight (NOTBEFORE 00:30), not in the
      publish hour.
  - MON 12:47 (`date`) QUEUE:
    - Leap FINAL clean-clone PASS 12:13 (rel1 242553d; 12 / 12 tokens; re-checked from out/). The 17:00 package milestone
      is met, so the reserve 16:00-16:45 is not needed.
    - m1b runs at lcoh's pace, decode end ~15:57 +-10 min.
      - The orchestrator lets it run to 16:20. EvalSuite amendment 11 (ac88d3a): cutoff 16:20, last start 16:05, ends by
        ~16:17. Verdict or "not gated" to the editor by 16:30.
      - zs32 is not gated.
    - ORDER after m1b (NewIdeas 12:51): Leap's full store conversion first (the orchestrator's 12:44 ask; >= ~16:17, hard stop
      16:58; writes a done file), then Fit12 chw19 (share lock, MEM 12 GB, gated on that file + its own gate; ~16:35-17:00;
      start-by 17:30; fallback GO at 17:00 if no done file).
    - NewIdeas re-grades m1b's GPQA / LCB (forward order after ~13:00, reverse at ~16:20) before the 16:30 label call.
  - MON 10:37 (`date`) QUEUE + NUMBER CHECK:
    - Done:
      - BigMoE REL ROWS ran 09:47-09:56 at the lcoh boundary: L + COH rows, repeat 12 / 12. Re-derived.
      - EvalSuite lcoh card gate PASS 10:07 (6 / 6 lines). I re-graded GPQA and LCB independently: identical.
    - Running / queued:
      - m1b running since 10:07, ETA ~15:10-15:30.
      - If m1b's 15:30 cutoff stops it, EvalSuite's tools/card_extend_cutoff.sh resumes it at CUTOFF 15:45 (amendment 10;
        my 07:22 OK).
      - Leap FINAL 12:05 (rel1 package 242553d; ~5 min quiet, MEM 47 GB; RESERVE 16:00-16:45 only if it fails).
    - Number check:
      - Authors applied the DIFFs; the 10:3x drafts re-checked OK.
      - Open with the editor: MAIN calls the N45 / CHS4 "L" rows "certified", but L = CERT + the 6-key bundle (s5p_env).
      - Next re-check at 12:00, against the 10:36 snapshot.
  - MON 07:55 (`date`) NUMBER CHECK, FIRST PASS DONE on all six drafts (07:22-07:55; ledger NUMBER_CHECK.md sections 1-6):
    - DIFFs sent: MAIN 22 (to the editor), RELEASE 4, DRAFTER 3, SMALL_RAM 3 + 1 (TBD-33 closed), KERNELS 5, EVALUATION 2.
      RELEASE, DRAFTER and SMALL_RAM have already applied theirs. No headline claim changes.
    - Pattern behind most DIFFs: old run reports print geometric means (or the steady metric), and drafts quoted them as
      whole decode. Examples: E24 41.3 -> 41.1 pooled, E24m 38.4 -> 37.4, the wired limit +27.4 -> +25.1 %, cache keep
      x1.10 -> x1.06.
    - Cross-paper: MAIN still calls PF-B "not exact"; since the seg-per-chunk fix it is exact (KERNELS R20, RELEASE).
    - Next: re-derive BigMoE's REL ROWS (RELEASE TBD-16) after ~10:15. At 12:00, scripts/numcheck/diffnums.py against the
      07:49 snapshot lists only the changed numbers to re-check.
  - MON 07:22 (`date`) QUEUE CHECK:
    - Leap's SMOKE 07:01 on 63f41fa: FAIL (7). The exact-45 and exact-32 tokens PASS. Lossy / chat crashed: tierglu was
      loaded live from ~/MoonShots, outside BigMoE's module manifests. The closure check also flagged the weight links.
      Both fixed in pkg 0a20f26 (CPU-checked). Re-runs at EvalSuite boundaries 07:13 and 07:28: SMOKE PASS 07:32 (3rd run,
      221 s, rel0 dbe5ea6). FINAL 12:05 on rel1 (frozen 07:19), ~4-5 min quiet, MEM 44 GB.
    - EvalSuite: lcoh VOIDED 06:37. The screen's 600 s budget cut runs mid-row, and r57rows kept them as rows (PREREG
      am. 9, d613a8c). lcoh resumed 06:43 (the 10 keys, then rev). New ETA: lcoh verdict ~09:50-10:00; m1b ~10:00 ->
      ~15:20-16:00. The 06:44 audit found no earlier verdict that used a cut row.
    - NewIdeas picked (07:22):
      - BigMoE's rel1 goes at the lcoh -> m1b seam. It is the L + COH rows run, ~15 min quiet; it IS the 10:30 slot,
        not a second one. BigMoE queues at 09:40.
      - m1b may resume with CUTOFF 15:45 (last start 15:30, verdict by ~15:55). Yes from the queue: nothing is booked
        15:30-16:00.
      - zs32 cannot be gated by 16:00 in this order, so the 32 GB lossy chat row stays "not gated".
    - pacer 07:21: GO (9 % vs the 57 % line). The number check can start on the drafts that exist.
  - MON 05:34 (`date`): the card reference is DONE 03:42 (381 / 381, EvalSuite 7479a1f); lcoh running since 03:42 (~2.5 h
    ahead of the worst case). Expected: lcoh ~07:30 (with Leap's SMOKE at 07:00), m1b ~11:35 (with BigMoE's rel1 at
    10:30), zs32 from ~11:35 with Leap's FINAL at 12:05 inside, maybe whole by the 15:30 cut. EvalSuite pins its code
    by sha until 16:00: no edits under ~/MoonShots/patches/planes_engine, BigMoE bigmoe/ + ext .so + data/r57/snap/
    {card,frozen_card}, or the venv (an edit stops the chain, rc 7).
    - NewIdeas lifts pacer's SLEEP (37 % weekly vs the 31 % line) only for the release times (the orchestrator
      19:2x): queue checks ~07:15, ~12:10, ~16:00; the number check on the drafts from ~10:45 to 16:00.
  - 23:50 UPDATE: F2a stage 1 ENDED 23:48 (its swap guard stopped capture at 23:29, +2 GB swap at C 128); lock released
    23:48:18, STAGE1_DONE + closing line written. VERDICT KILL by the pre-registered line (MTP Head): 0.10M tokens (8 % of
    the ring), held-out kept x1.014 (IFEval 2.301 -> 2.334, multi 2.507 -> 2.540), on-policy x1.002. Nothing ships; a
    data-rich run (~4.6 h capture, swap-safe pool) moves after the release.
    - BigMoE's PF-D arms hold the quiet lock 23:48:54 to ~00:04. EvalSuite's chain started 23:48:35 (step ref, 261 / 381)
      and waits behind it: the reference resumes ~00:05, ~3.5 h early.
    - Leap's FINAL goes BACK to 12:05 (inside, ~12 min; reserve 16:00-16:45). m1b ~12:25-13:05; zs32 likely partial.
    - 23:56 CHECKED: BigMoE's PF-D arms ran 23:48:54-23:55:19 (6.5 min); EvalSuite's reference started 23:55:36 (no gap,
      54 GB free, share lock to ~01:10 per chunk).
  - ACTUAL ORDER 23:05: F2a holds the quiet lock since 23:01 (capture 23:01:58, NO-SWAP 45 GB -> frac 0.25). BigMoE's
    23:00:00 ticket was granted at 23:00:33 but released (NO-SWAP 43 < 47 while the apps were still closing), and it
    re-queued at 23:02:33: queued #1 behind F2a, so its PF-D arms run ~03:30-03:42. EvalSuite's chain then queues behind
    BigMoE (~03:45; m1b worst case ~15:54). F2a is NOT stopped: its exit trap writes STAGE1_DONE and would fire the
    chain. Lesson: when two jobs share a start time, the second should key on the first one's 'granted ... run' line, not
    on a clock time.
    - Fit12 22:32: SMALL_RAM is safe; chw19 held (WAIT, start-by Mon 17:30): NewIdeas sends GO at ~16:15 or after 17:00.
  - RE-PLAN 22:26 (the gates' budget): EvalSuite's reference stopped at its 22:11 cutoff, 261 / 381 rows (IFBench + GPQA
    done, LCB 11 / 131; c6dc794). The 120 LCB rows left take ~4.5 h from the chain start. Pre-flight OK: chain pid 53863,
    STAGE1_DONE trigger, LATEST 03:55, REFCUT 16:00, ARMCUT 15:30.
    - Inside the chain now only: Leap's SMOKE at 07:00 (~12 min, during the reference) and BigMoE's rel1 at 10:30
      (<= 15 min; 6 shipped configs x 12 IFEval, CAP 64).
    - Moved out: Leap's FINAL to 16:00 (reserve to 16:45), Fit12's chw19 to ~16:15-16:30 or after 17:00 (MEM 12 GB, by day).
    - Worst case: reference ~08:12, lcoh ~12:03, m1b ~15:39; zs32 likely does not fit ('not run').
> (a quote from the project owner, kept private)
    close as many apps as you need. But start at 23 not now"). Apps close 23:00, reopen Mon 17:00; nothing new before 23:00.
    1. To 22:45: EvalSuite's card reference. 2. 23:00 to ~23:15: BigMoE's PF-D arms (ticket at 23:00:00).
    3. ~23:15 to ~03:30: F2a stage 1, queued behind BigMoE. CONFIRMED 19:52 (MTP Head 045787c): lock request 23:01 with
       --queue; capture 02:15, train 03:15, eval 03:30; STAGE1_DONE + closing line unchanged, none written now.
    4. ~03:30 to 16:00: EvalSuite's chain (REFCUT 16:00, ARMCUT 15:30; backstop ~03:55; trigger: STAGE1_DONE newer than the
       chain's start, or a new anchored DONE line). Inside it, at boundaries: Leap SMOKE 07:00; BigMoE's last prefill
       arms + Fit12's chw19 before 11:40; Leap FINAL 12:05. 5. 16:00-16:30: reserve for Leap's clean-clone re-run.
       The package passes by 17:00.
  - MONDAY RELEASE ORDER, 19:19 (the orchestrator 19:2x; ~/Papers/MONDAY.md replaces BRIEF's dates). Kamen asked to
    publish all papers + the 45 / 32 GB engine by Monday evening. Release only on Kamen's word. NewIdeas: queue, number
    check Mon 12:00-16:00 on the drafts, first contact for machine questions.
    1. Now to 22:45: EvalSuite's card reference (running again since 19:01, 191 / 381 rows).
    2. 22:45-23:00: BigMoE's PF-D diagnostic + prefill arms (end by 23:00).
    3. 23:00 to ~03:15: MTP Head's F2a, STAGE 1 ONLY (stage 2 after the release; asked to re-arm without stage 2).
    4. ~03:15 to 15:00: EvalSuite's chain: reference leftover -> lcoh -> m1b -> a 32 GB chat arm (C 160) if it fits.
       CHAINED on F2a's last log line (tail -1 ~/TrainLab/data/f2a/night.log == 'F2a night DONE', written after its
       unlock), fallback 03:40. Hard stop 15:00 (the last chunk or arm ENDS by 15:00); not gated by then = 'not gated'.
    5. Before 12:00, at one chunk boundary, <= 15 min: BigMoE's last prefill arms; Fit12's chw19 (10 min) beside them
       once SMALL_RAM is safe (chw19's Sunday start-by lapses).
    - Held: Breakthrough's 2 chat screens (research stop). Fit12's 12 GB IFBench arm: after the release (no slot by 15:00).
    - 19:3x CONFIRMED: MTP Head's runner v3 (bbc2667) is stage 1 only and writes BOTH triggers on every exit after 23:00:
      the night.log DONE line and ~/TrainLab/data/f2a/STAGE1_DONE. Stage 2 runs only if data/f2a/FULL_NIGHT exists.
      EvalSuite's chain is armed (DONE line or 03:40 -> ref -> lcoh -> m1b -> zs32). Fit12's chw19 is held, start-by 11:40.
    - FIXED-TRIGGER ALERT 19:3x: F2a v3's ARMED line (19:23:35, the current last line of night.log) contains the
      words 'F2a night DONE', so any `tail -1 night.log | grep 'F2a night DONE'` is true NOW. My watcher fired on
      it, and EvalSuite's card_chain.sh f2a_done() uses the same test, which would start the chain at ~22:45 (when
      tonight's runner ends), before F2a's 23:00 line. Told EvalSuite to test the file ~/TrainLab/data/f2a/STAGE1_DONE
      instead (absent until stage 1 ends) or an anchored regex. CLOSED 19:29: MTP Head re-armed (23c28f2); the armed line
      no longer has the phrase (tail -1 count 0), only finish() writes it; card_chain.sh had not started.
    - LEAP'S PACKAGE SLOTS (placed 19:3x): (a) SMOKE, queue at 07:00, ~12 min, MEM 47 quiet; (b) FINAL on BigMoE's
      release snapshot, queue at 12:05, ~12 min, ends by 14:45. One boundary before 11:40: BigMoE's last prefill arms
      (<= 15 min) + Fit12's chw19 (10 min). Worst case the chain plus these ~49 min ends ~14:46; m1b's last process
      starts by 14:30 (ARMCUT), so a late m1b may be cut and labelled 'not gated'.
    - RISK, asked of the orchestrator / Kamen 19:3x: lcoh / m1b gate at 47 + 4 = 51 GB NO-SWAP. NO-SWAP was 44 GB at 19:01
      with Chrome + Spotify open, so they would wait to ~14:30. M1-B's measured lifetime peak is 42.2-42.4 GB (BigMoE
      es_chm1b_fwd10-13 runs.json), so measured + 4 ~= 47 GB (EvalSuite's call). Closing Chrome + Spotify frees ~5 GB.
  - PAPERS + QUEUE, 16:48 (the orchestrator 16:4x; brief ~/Papers/BRIEF.md): Kamen says write the papers and publish
    (ready by Wed 11 Nov). NewIdeas owns the NUMBER CHECK: every headline in every paper re-derived from raw rows by
    Tue 12:00. New research stops, except what is already armed and PF-D. No new screens to fill paper sections.
    - PF-D is GO for BigMoE (Kamen: Strata's code may be used, MIT notice kept). Its A/B gets the next 45 GB slot when
      BigMoE asks.
    - NOW: EvalSuite's card-reference chunk (started 16:43, C 192, 10 rows, <= 45 min).
    - HELD 17:1x: Breakthrough's 2 approved chat map screens (ready, ~9 min, share MEM 18 GB): not armed when research
      stopped, and no use before Wednesday. They go after the card reference only if the orchestrator OKs them.
    - NEXT: a low-memory Strata run (24 and 12 GB lines; results_low24 / results_low12; 35-50 min, quiet) in the gap
      after that chunk. EvalSuite holds until results_low12/DRIVER_DONE or 75 min.
    - THEN: EvalSuite's reference resumes. Sunday's GPU ends 22:45; F2a at 23:00. Monday 07:05-15:00: the card
      leftover, lcoh, then whichever arm fits. Monday after 15:00: Fit12's 12 GB IFBench arm (ONE order: chw18's history
      test passed 13 / 13, 16:42).
  - ORDER AT 16:12 (the orchestrator 16:1x):
    1. Strata finished 8 / 8 at 16:07. Its arena was only 25-28 GB with Chrome open, so it is re-running B and D at
       the full 33 GB arena (results_arena33; started 16:12, ~12 min).
    2. BigMoE's 32 GB CHAT ARM (APPROVED; below), ~12 min, straight after.
    3. PFB stays blocked: its 48 GB gate can't pass with Chrome open (46 GB). The orchestrator is asking Kamen about
       closing Chrome.
    4. GAP until 17:07 (if PFB stays blocked): Fit12's chw18 (one share lock, MEM 12 GB, ~10 min; smoke + history test
       + chw16 profile). Its own gate (6 clear minutes) opens there. It must END by 17:07. If Chrome closes and PFB
       queues first, chw18 goes right after PFB.
    5. 17:07: EvalSuite's card reference resumes (it waits on PFB's pid until then at most).
  - APPROVED 16:1x (proposed 15:36; BigMoE runs it): a 32 GB CHAT ARM (~12 min, MEM ~33 GB, quiet).
    - Why: Strata peaks at 33.6-34.9 GB, so the fair answer to Kamen's question is our chat at ~32 GB. The publish
      table's 32 GB chat cell is also still empty.
    - What: the 12 IFEval (N45 keys, CAP 256) at C 160 (C30's 31.8 GB route), E (exact) and zs (M1-B + COH + SB1),
      ~6 min each.
    - Caveat: QOM skips cold ranks, and C 160 has more cold experts, so M1-B's IFEval gate (run at C 256) does not
      carry over. At 32 GB the zs row is lossy and ungated; the E row is exact.
    - The 24 GB chat cell (E24 on the same 12 IFEval, MEM ~24 GB) can go Monday after 15:00, by day.
- SUN 09:22 (`date`) STATE: M1-B'S CHAT GATE + KAMEN'S BENCHMARK RULE
  - M1-B + COH CHAT GATE (EvalSuite runner 59, 08:13-09:13; RESULT_chat_m1b 70b5cf8):
    - IFEval same scores PASS in both orders vs NAX6-N: forward 86.0 vs 87.0 % (-1.00, McNemar p 1.0); reverse
      83.0 vs 87.0 % (-4.00, p 0.388).
    - multi line: 334 blunders per 10K (bar 1). RE-DERIVED by NewIdeas from decode.jsonl + tfwin/*.json.gz: 7,033
      tokens, 1,075 deviations (15.29 %), 235 blunders. Every count matches.
    - The texts show it. zh16 has an English "air" inside Chinese, ar22 has misspelled Arabic words, ru13 has Russian
      grammar errors. REF's texts for the same prompts are clean.
  - KAMEN'S RULE (via the orchestrator 09:3x), verbatim: "But checking against other languages is wrong. We should compare
    against known benchmarks of the model. Not something new".
    - Gates use the model's known published benchmarks only: lossy vs the exact reference on the same harness.
    - The multi and long-code blunder lines are INFORMATION, not verdicts. So is every blunder / KLD / map-flip quality
      price on this board (M1 RANKED below included).
  - M1-B'S OPEN QUESTION: a known multilingual benchmark. EvalSuite has MGSM (data/items/dec_mgsm.jsonl): 80 problems
    in 10 languages (bn de es fr ja ru sw te th zh, 8 each), max 1,024 tokens, ~34K tokens per pass (REF 427 / floor
    409 per row). NAX6-N has no MGSM rows; REF has 80 (87.5 %, 10-22). EvalSuite re-registers before any row.
  - DROPPED by the rule:
    - NewIdeas' multi bisect: the orchestrator's 09:2x request, replaced at 09:3x before registration.
    - m1bi (chat3_env; EvalSuite amendment 1 7ad132b).
  - HELD: m1bs (chat2_env). It carries M1-B's QOM, so it takes the same MGSM gate after M1-B's.
  - MGSM HELD 09:3x FOR KAMEN'S DECISION (the orchestrator):
    - All 80 MGSM prompts are typed math in the engine, so M1-B's chat lever (QOM on the chat sets) never runs on them.
    - The 125B's card lists no natural-language multilingual benchmark.
    - Kamen is asked to choose: M1-B on English chat only (a request-level language switch), or MGSM re-templated as
      chat for all languages.
    - EvalSuite registered the gate (e876e4c). Before its arms can run it needs chat4_env (mgsm added to the QOM sets)
      and lcoh_env (L + COH).
    - CORRECTION to NewIdeas 09:2x: cert_env pins snapshot s5, whose ref2 has no COH, so cert_env alone is L without
      COH.
    - NAX6-N's 80 MGSM rows come from EvalSuite's AR decode stage at 14.2 tok/s (measured on mathx): ~45 min, not
      ~20. About 2 h in all with both arms in two orders (EvalSuite).
> (a quote from the project owner, kept private)
    asap so I know exactly how bad our 50tps on chat actually is".
    1. Fit12 T2 FIXED COST PROFILE (~5 min, share MEM 12 GB, armed 12:02). Only if it starts before Strata's lock.
    2. STRATA (quiet 120 min). It needs 47 GB available, so Chrome must close; the orchestrator is asking Kamen.
    3. BigMoE PFB (~22 min, with CT1 + FU fb0), right behind Strata while the 47 GB is free.
    4. Breakthrough's two chat map screens (~4 min each, MEM 18 GB): QOM0 + bonus gated 0.5 at 147 slots, and
       qom_0.5_live at 51 slots. They can also go in any gap before Strata's lock.
    5. EvalSuite's card reference: NAX6-N AR, 381 keys, ~5-6 h, MEM 27-36 GB, by day after a built-in 12-row line
       check (PREREG_card_gates 9af1abb).
       - CHUNKED (the orchestrator 12:0x, on NewIdeas' suggestion): 10-prompt chunks start right after Fit12's T2,
         while Strata waits for 47 GB (~36 GB free with Chrome open).
       - It yields at a chunk boundary once Chrome closes, and resumes after Strata's WAITER END and PFB.
       - Ready for the arms (BigMoE 12:0x): card snapshot e057abf, env files d049eb3 (card_lcoh / card_m1b /
         card_m1bs). PFB queues no later than 22:15; Sunday's GPU ends 22:45.
    6. EVENING, ending 22:45: a card arm (~3.6 h each, MEM 47 GB) if the reference leaves room. Then Sun 23:00 F2a.
    7. MON 07:05 (Strata's old slot, free; NewIdeas places it): the card-gate arm work left from tonight. It's
       resumable; quiet lock, MEM 47 GB, hard stop 15:00 (the orchestrator 15:2x; the last chunk or arm ENDS by 15:00).
    - RULE (11:3x) stays: a job that fits in ~30 GB never waits behind a memory-gated one. One > 5 GB job at a time.
  - FIT12 CHAT WIDTH (12 GB; NewIdeas places it, 12:2x):
    - The lever: the 2-row first exact call when the first draft's cf < 0.9. Chat whole decode +9.6 % (12.68 -> 13.89
      tok/s), code -1.0 %. Rounding-level, not bit-exact [K].
    - Under Kamen's rule its gate is a card benchmark. A full 12 GB card arm (~450K tokens in two orders at ~14 tok/s)
      costs ~9 h [P], more than any window before Wednesday, and EvalSuite's card gates have no 12 GB arm.
    - PLACED: a 12 GB IFBench arm (150 keys, one order, ~150K tokens, ~3 h [P], MEM ~12-16 GB), Monday after 15:00. It's
      scored vs the NAX6-N IFBench rows the card reference makes anyway, with the card's own line on IFBench.
    - It needs Fit12's env and a 12 GB screen whose SETOF maps ifbench.
    - Without that run it can still be published as a non-exact 12 GB chat mode, ungated, with its measured speed.
      Adopting it is Kamen's call.
  - BREAKTHROUGH 11:56 [K]:
    - The Qwen 4 capture reproduction PASSES bit-exact (053b3b6), so the Qwen 4 board item is DONE.
    - Chat map screens (c159451; map flips, INFORMATION / screens under Kamen's rule):
      - At 147 slots (24 GB), qom_0_live is clean (4.50 / 3.26 %) but reads 105 / 96 MB per kept vs the route's
        82 / 77. qom_0_live_ball fails the chat lines (6.36 / 4.11).
      - At 51 slots (12 GB), both QOM0 arms fail (8.80 / 5.26; ball 10.57 / 7.99).
      - So 24 GB chat stays ~80 MB per kept on clean rules, and 12 GB needs cf 0.5.
  - NEW, placed 11:3x: the model card's public benchmarks, fast vs exact (EvalSuite preparing: IFBench, LiveCodeBench v6,
    GPQA). This is Kamen's rule, quality on the model's known benchmarks.
    - Their 45 GB arms are memory-gated, so evening only. ~4 h of 45 GB GPU per evening (~500-600K tokens at 38-45
      tok/s) won't hold all three sets in two arms, so EvalSuite sizes the subsets.
    - Exact arm by day if exact is line-independent. Cheap check: the 24 GB exact env on the 12 IFEval == ar_ref_tokens
      (REFTOK). If it holds, the exact rows run by day at 24 GB and only the fast 45 GB arm needs the evening.
    - NAX6-N's AR stage (14.2 tok/s) is ~2.6x slower than exact E (36-40 tok/s, == AR 12 / 12 at cap 256). E as the
      reference is EvalSuite's call; it isn't proven on long outputs.
  - NEWIDEAS' GUESS [G], written before any row (it stands if the gate runs as registered): both pass decision 4 at n 80, where -5 pt is 4 problems, so power is
    low. M1-B 0-6 pt under NAX6-N; L within 3 pt. If M1-B passes, its multilingual misspellings stay information: MGSM
    scores the final number, not the text.
- SUN 07:44 (`date`) STATE (chs4, re-derived by NewIdeas from the per-prompt lines; 12 IFEval, C 256, one pass per arm):
  - chs4 ran 07:13-07:42, 2.3 h late. BigMoE's runner looped silently after CERT_DONE (04:53), though every gate passed
    by hand and in a lock-free replica. The orchestrator moved BigMoE to sub at ~07:10, and the re-armed chain queued.
  - WHOLE DECODE (tok/s):
    - L 43.8; M1-B 49.3 / 50.5 (mean 49.9);
    - M1-B + SB1 52.0 (exact schedule; see 1ROW);
    - M1-B + K1M + HC8_G=2 (FAST) 53.1; + R1 (lossy, 73 relaxed accepts, kept +7 %) 52.9; + rcut8 52.2; + K1M 49.6;
      + Leap's FAST low-row 50.4; + NW1 48.9; NW1 + bundle 50.6;
    - L + P4 (D1) 40.2: KILL again on L.
  - NOISE: check ms moves ~0.7 ms between arms, and kept moves ~3 % (different text on L). So only SB1 (-2.35 ms
    check, with its per-row mechanism) and R1 (kept +7 %) clear the noise. K1M + G2 (-0.75) and rcut8 (-1.2) are near it.
  - STACK PRICE [P]: adding the arms' cycle ms (SB1 -2.55, K1M+G2 -0.6, rcut8 -1.15, R1 +0.35 with kept x1.07) on
    M1-B's 48.55 ms / 2.43 kept gives <= ~58 tok/s (~44.6 ms at kept ~2.60). 150 is not reachable with these levers;
    the walls stay the check's fixed cost (~33 ms at 2 rows) and K.
  - C4 ANSWERED: E + OOO_1C + NAX_1CQ + REF2_NAX_COH == AR 12 / 12 in both runs (1687 tokens each), vs 3 / 12 without
    COH (ycoc / ycoc2). Leap's coherent pool reads remove that pair's nondeterminism at no cost (37.8 / 37.9 vs
    37.7 / 38.0). REF2_SMALL_NATIVE, chs3's other cause, is untouched.
  - CHS5 FORECAST (NewIdeas 08:0x, written before CHS5's result): M1-B + COH + SB1 + K1M / HC8_G + rank cut 8, adding
    chs4's cycle ms. Without R1: ~44.3 ms at kept 2.43 = ~55 tok/s. With R1: ~44.6 ms at kept ~2.60 = ~58 tok/s.
    Reading the result:
    - >= 3 % under the forecast: the levers overlap.
    - Over it: chs4's single passes under-read them.
  - CHS5 RESULT (BigMoE 07:57-08:12, snapshot 65cd673; re-derived by NewIdeas 08:1x from the per-prompt lines,
    tokens.json and runs.json):
    - Base M1-B + COH (z0 / z0b): 49.67 / 49.94 tok/s.
    - + SB1 (zs): 51.89 = +4.2 % on IDENTICAL tokens (1742 / 721 checks in all three arms), 46.6 vs 48.5 ms per check.
    - + FAST bundle / HC8_G (zsg): 54.66. Its text differs on 9 / 12 prompts and kept is 2.51 vs 2.42, so the real
      per-check gain is -0.7 ms (~+1.5 %). The rest is text.
    - + rank cut 8 (zsgc / zsgcb): 53.55 / 53.84. 43.5 ms per check, but kept 2.34.
    - + R1 (zsgcr): 56.05, BEST; lossy rule, 83 relaxed accepts.
    - + NGEMV_ROWS 2 (zsgcn): 55.41.
    - Exact E: yeh 36.55 / yesh (+ SB1) 36.85.
    - FORECAST CHECK: ~55 -> 53.7 (-2.4 %); ~58 -> 56.05 (-3.4 %: under by 3 %, so the levers overlap a little). Check
      time came out BETTER than forecast (43.5 vs 44.3 ms), kept came out lower (2.34 vs 2.43). On 12 prompts, kept is
      text noise, so the two can't be separated.
  - DETERMINISM [K]: with REF2_NAX_COH=1, L + M1-B chat is run-to-run deterministic: z0 == z0b tokens on 12 / 12, and
    zsgc == zsgcb. The RULE below (L chat not deterministic) no longer holds when COH is on. SB1 leaves L + M1-B's
    tokens unchanged (z0 == zs, 12 / 12).
  - HONEST CHAT AT 45 GB, LOSSY: 56 tok/s measured (zsgcr). Every lever past M1-B is lossy and ungated, except SB1 and
    COH, which are exact. 150 stays out of reach.
  - NEXT: PFX (quiet lock to 08:28) -> FU -> Breakthrough's slot. M1-B's chat gate (EvalSuite runner 59) has
    chat_env.txt = M1-B.
- SUN 05:09 (`date`) STATE:
  - CERT PASS 04:54 on all 10 lines (EvalSuite RESULT_cert 2cfed64). RE-DERIVED by NewIdeas
    (scripts/cert/cert_rederive.py, own exact McNemar + raw decode.jsonl): every delta / p / pool and every
    token-identity count matches (pooled -0.96 pt p 0.2649; 4-set -1.08 p 0.1839; IFEval token-identical 11 / 11 of
    100; cert_chat 1 / 12). The per-row pass / fail scores are EvalSuite's scorers (not re-scored).
  - CHAT-EXACT: NO (1 / 12). The shipped 45 GB config's chat is lossy, same scores (IFEval PASS both orders).
  - GPU order now: Fit12 chat12 (quiet lock to 05:37) -> BigMoE chs4 (M1-B B / F / F / B + Leap's bundle) -> BigMoE
    writes data/stack/chat_env.txt naming M1-B -> EvalSuite runner 59 (the M1-B IFEval + multi gate) -> PFX -> FU ->
    BREAKTHROUGH'S SLOT (NewIdeas 05:09): the Qwen 4 capture reproduction (~20 min) + the two map gates c147 / c51
    (~5 min each), share lock, MEM 18 GB, never beside a 45 GB engine job or inside a quiet lock; any gap >= 40 min
    after FU, done by 22:30 -> Sun 23:00 F2a (quiet whole Mac).
  - EvalSuite: NewIdeas' peer name 'ojox-59' is ambiguous (MTP Head's too); peers report to NewIdeas via the
    orchestrator + status, or `peer send` naming NewIdeas.
- TARGET: 150 tok/s whole decode on 12 IFEval at 45 GB lossy, by Monday. That means a ~22 ms step at K ~3.3 (today
  ~58 ms exact / ~45-48 lossy at K 2.5).
- The step today (N45 means, exact / lossy; CORRECTED by BigMoE 00:5x):
  - Misses: ~89 per check. The GPU's stall bound is svc 15.0 / 14.4 ms (partly overlapped); the SSD is busy (io)
    12.0 / 11.6 ms. "stl 20.3" is a COUNT: 20.3 of the 48 layer calls per check waited on loads.
  - The build (bld 44.8 / 37.2 ms) is NOT pure Python. It holds the gate's per-layer router-id waits, so the GPU work
    up to the last router plus the stalls. wt 12.8 / 9.3 ms is the GPU tail.
  - The draft is 8.4 ms.
  - Lossy L has the same misses as exact (88.8 vs 89.8). Its +19 % comes from compute levers (C4).
  - Leap at 2 rows: 54.5 ms wall, main-thread CPU 29.3 ms, GPU dense ~20-22 ms + experts ~14 ms.
- SEQUENCING (Leap 01:2x, measured): M1 GOES FIRST.
  - Leap's first G1 piece, REF2_PAD_IN (pads in-kernel; bitwise 320 / 320 at kernel level; ref2 95585132), gained
    NOTHING live: 2-row checks 44.1 -> 43.9 ms, 3 rows 50.5 -> 49.9.
  - ~26 of 48 layer calls stall on misses per chat check (~17 ms of gate spin), so dense-side GPU savings move into
    the spin.
  - G1 / G2 count only after M1. Until then Leap prices them on no-miss replays (experts resident).
- RULE (Leap 01:2x): chat on L is NOT run-to-run deterministic. Two plain runs agree only on check 0's hash, and tokens
  differ on 2 / 3 prompts. Chat levers' bit checks use the exact env or kernel-level proofs; token-equality A/Bs on L
  chat mean nothing.
- CRITICAL PATH: the step is >= max(main-thread CPU, GPU + stalls) + any draft not overlapped.
  - So all three must fall to <= ~17-20 ms: the Python / graph build, the GPU, and the miss stalls. The draft must
    shrink or overlap.
  - Cutting only one moves the wall to the next.
- GPU order: CERT (01:30, shared) first. Then 45 GB chat A/Bs as candidates land, AHEAD of PFX. Fit12's 20-min 12 GB
  chat12 may run right after CERT if no chat candidate is ready.
- SUNDAY NIGHT, Sun 23:00 -> Mon 07:00: a QUIET WHOLE-MAC WINDOW for MTP Head's F2a (chat feature capture + head
  training; the corpus download is under Kamen's standing OK, logged by the orchestrator). No other GPU or heavy CPU
  jobs in that window.
  - Sunday daytime: CERT's end, then Fit12's chat12 (~20 min), then the strike's A/Bs (BigMoE's runner), then PFX.
  - Every Sunday GPU job ends by 22:45.

| piece | owner | ms saved | bits | ETA | status |
|---|---|---|---|---|---|
> (a quote from the project owner, kept private)
| G1: dense at low rows: row-native GEMVs without pads, HC8 at low rows, glue fusion (F1a) | Leap | -6..-7.5 [P] on no-miss replays; first piece REF2_PAD_IN [K] 0 ms LIVE (bitwise, held until M1 lands) | gated (pad_in bitwise) | after M1 | priced; pad_in measured |
| G2 CORRECTED (Leap, measured 00:49-00:52 under a quiet lock): L's NAX one-call already runs 183-223 GB/s at 1-4 rows (~73 % of 285). A 2-row check's experts read ~2 GB (not ~1), ~9.4 ms. The ceiling at the bare floor is ~-2.5 ms; 3c-N's one-call gives -0.55..-0.71 ms at 1-2 rows (FAST), worse from 3 rows, and needs U3's -2 sidecar fallback | Leap | -0.55..-0.71 [K] (ceiling ~-2.5) | FAST | with the low-row list | priced |
| G1 / B1 LANDED (Leap, live ref2 bb055eb4, default off; TABLE_350 e43e175): REF2_HC8_K1M=1 EXACT (bitwise 96 / 96 kernel + 24 / 24 via ref2), -0.62..-0.74 ms per check at 1-4 rows; it replaces REF2_HC8_FOLD (KILL: +0..+0.53 ms). REF2_HC8_G=2 FAST (bf16-ulp), HC8 GEMV at 1-2 rows, a further -0.55..-1.15 ms. Leap's remaining low-row list: 3c-N one-call -0.6; q|k|v + qkvz|ab horizontal fusion -0.85 (exact for E); in_proj GEMV on the native layout -0.5..-0.85; router -0.5. Total ~-4..-5 ms with K1M + HC8-G = +10-12 % on 1-2-row checks | Leap -> BigMoE chs4 | -4..-5 [P/K] | K1M exact; the rest FAST | chs4 after CERT | landing |
| M1-B [K] (BigMoE CHS2, 87156ea, data/chs/chs2_report.txt; re-derived): QOM's row gate on the MTP head's chat drafts (R57_QOM_SETS += ifeval,multi; R57_QOM_MTP=2; every drafted row sure at thr 0, the bonus row gated). Whole decode 49.0 vs 44.3 (L, mean of 2) = +10.7 % on one pass. Cycle 48.8 vs ~57.3 ms (-15 %): check 40.5 vs ~48.4 ms (means), misses 25.8 vs ~94, svc 6.3 vs ~15.4 ms, SSD MB per kept token 15.5 vs ~41. Kept 2.41 vs ~2.56 (noise). Peak 42.4 GB. Siblings: M1-A +1-2 % (helps only 4+ row checks); pool shift S16 / S32 no gain (x2qm2s' 52.0 is kept noise, 2.72); rank cut 8 +2 % | BigMoE | -8 ms per check [K] | LOSSY: EvalSuite's IFEval + multi gate vs NAX6-N, routed 01:4x, after CERT | gate after CERT, then chs4 (M1-B in B / F / F / B + Leap's exact bundle / NW1 / R1) | THE M1 WINNER. Breakthrough 04:2x (2bb8413): the engine's M1-B = QOM + bonus gate + U3 at its real rate (t3 skips only cold sidecar READS; resident LSBs stay 4-bit). Map bounds: no-3-bit qom_0_live_ball PASS 4.03 / 3.26; every-sure-row t3 FAIL 6.09 / 5.29. If the gate FAILS, a no-U3 QOM needs a build |
| 1ROW (NEW, NewIdeas 05:2x, from BigMoE's CHS2 per-check data; scripts below): under M1-B (x2qm2), 1-ROW CHAT CHECKS are 20 % of checks and cost 48.4 ms median (bld 10.3 + wt 37.6; 40 misses, svc 10.4) vs 32.8 ms for 2-row checks (bld 29.3 + wt 2.7; 8.5 misses). L (x2c0): 1-row 42.7 vs 2-row 39.7 (18 vs 48 misses); E (x2e): 53.8 vs 51.6. Cause: the screen's SB_LO = 2 (chs3 snapshot :947): 1-row checks skip SB's mid-loop async_eval, build lazily and serve their misses inside the final eval. M1-B makes it worse: drafted rows skip cold ranks 2-9, so the full-read 1-row checks miss more (40 vs 18). Fix = R57_SB1=1 (BigMoE's 11-02 port of Leap's C0 fix; 'untested, exact'; then 1-row checks were 8 % at ~0.5 %) | BigMoE (chs4 extra arms ybs1 = M1-B + R57_SB1=1, yes1 = E + R57_SB1=1, asked 05:21) | -1.5..-3 ms per check [P] = +3-6 % chat | exact (schedule only; yes1 checks E == AR 12 / 12) | chs4 | PRE-REGISTERED: PASS = ybs1 whole decode >= +3 % vs mean(yb, ybb) AND 1-row median <= 40 ms AND yes1 == AR 12 / 12 + E's verify hash; KILL <= +0.5 % or 1-row median > 46 ms; else BETWEEN. RESULT (chs4 07:40; re-derived by NewIdeas 07:4x from the per-prompt lines + scripts/chat/rows_cost.py, data/chat/rows_cost_chs4.txt): PASS ON BOTH SPEED LINES. ybs1 whole decode 51.99 vs M1-B 49.29 / 50.52 (mean 49.91) = +4.2 % (the two M1-B passes differ by 2.5 %). 1-row median 35.4 ms (yb 48.0 / ybb 45.3): wt 2.9 vs 37.1 / 34.3 ms, the wait moved into the overlapped build. 2-6-row checks unchanged (2-row 32.8 vs 33.2 / 32.8). Cycle 46.0 vs 48.5 / 48.6 ms, kept 2.40 vs 2.40 / 2.46. yes1 (E + SB1): 12 / 12 prompts == N45's AR (1687 / 1687 tokens, the same stream as ycoh), 38.3 tok/s (+1 % vs ycoh's 37.8 / 37.9, a different E arm). HASH CLAUSE OPEN: chs runs record no logits hash, so exactness is token-level only. CLOSED in CHS5 (08:12): with R57_LOGHASH=1, E (yeh) vs E + SB1 (yesh) logits-hash lists are IDENTICAL (676 / 676 checks, 12 prompts; re-derived with cmp). So 1ROW PASSES ALL THREE PRE-REGISTERED CLAUSES. In CHS5, SB1 also leaves L + M1-B + COH's tokens unchanged (z0 == zs, 12 / 12) at +4.2 %. Adopting SB1 is BigMoE's call for the exact modes and Kamen's for the shipped chat; it rides on M1-B's gate only if SB1 is bit-exact |
| SR2 (NEW, NewIdeas 05:1x-05:2x, RUN [K], share lock, scripts/chat/d1_screen_refine.py, data/chat/sr2.txt): the chat drafter's output layer as a 2-bit SCREEN (blocks of 128 ranked by max) + 4-bit REFINE of the top B blocks; the pick stays the 4-bit argmax of the refined set. MTP Head's chain (head b5, 30 IFEval + 24 multi held-out; out4 control reproduces d1_variants.txt exactly). s2b16: pick == out4 99.0 / 98.4 %, depth hits within 0.001, kept 2.318 / 2.513 vs 2.301 / 2.507 (rows +0.06 / +0.01); saves 0.332 ms per drafted row (1.138 vs 1.469). s2b4: saves 0.439 ms, kept 2.340 / 2.521 BUT via more rows (3.05 / 3.26 vs 2.88 / 3.18: p is inflated) with depth hits -1 %; s3b4 saves 0.168 | MTP Head (owner) | net ~-0.8 ms per check at ~3 ms per extra row = ~+1.5 % chat [P] | exact outputs (drafts verified) | - | BETWEEN (s2b16 0.332 < the 0.40 PASS line; s2b4 PASSES by the letter but its kept gain is a cut shift). Information to MTP Head; no route. +0.2 GB RAM for the 2-bit copy |
| CT1 CHAT TTFT (NEW, NewIdeas 08:2x, from BigMoE's CHS5 runs.json + Fit12's chat12 log; no run): 45 GB CHAT TTFT IS ~3.0 s. On the 12 IFEval prompts (37-101 tokens), TTFT (pfs: prefill + setup + first token) is 3.05 s mean (2.51-3.65) in zs, and the same in E (yeh 2.89), z0 3.03 and zsgcr 3.05. At ~52 tok/s and ~145-token answers, that is about HALF of each chat answer's wall time; the whole-decode headline leaves it out. CAUSE (two datasets): 12 GB's prefill of the same prompts reads 15-26 GB from the SSD at 6.8-7.9 GB/s (2.07-3.25 s), and 45 GB's TTFT is 1.05-1.21x that on EVERY prompt (mean 3.05 vs 2.73 s). So the 45 GB chat prefill gets nothing from its 32 GB of pools: it re-reads resident experts, as BigMoE's PF-D note says. LEVER = PF-D for chat: serve pool-resident MSB-3 base / LSB side expert-layers from the pools in the chat-turn prefill and read only the rest (exact: same weights). TRIED's SKIP-RESIDENTS KILL was at 12 GB (<= 12 % of bytes) and does not transfer: 45 GB pools hold C_m 440 / C_l 136 of 512 per layer, and PFX found 36-57 % of base expert-layers resident at prompt start | BigMoE (PF-D owner) | TTFT x1.4-1.7 at PFX's residency, up to x2.7 with full pools [P] (bytes -0.75 r_b - 0.25 r_s) | exact | after FU | CHEAPEST KILL (no code): one chat arm = zs + R57_PFSTAT=1 on the 12 IFEval prompts (~3 min), giving per-prompt prefill SSD GB + residency at start. PRE-REGISTERED: KILL if the 45 GB chat prefill already reads <= 60 % of 12 GB's bytes per prompt (pooled), or mean base residency at start < 25 %. PASS-to-build if >= 80 % of 12 GB's bytes AND mean base residency >= 50 %. Else BETWEEN MEASURED 16:40 [K] (BigMoE PFB ct1 arm 16:26-16:35, acf8ac3; NewIdeas' script scripts/chat/ct1_judge.py, numbers = BigMoE's): reads 100.9 % of 12 GB's bytes (244.9 vs 242.7 GB; 20.4 GB per prompt = routed x 2.79 MB), mean base residency at start 45.7 % (37-51 %), routed-resident 65.5 % base / 25.1 % side. BETWEEN by the letter (50 % pass line). PF-D prices TTFT 3.1 -> ~1.6 s [P] (BigMoE's fit 0.32 s + GB / 7.33), 45 GB chat end to end 24.8 -> ~33 [P]; Strata's TTFT is 0.83-1.06 s |
| FLOOR [K] (BigMoE CHS2): below 30 misses, L's chat check is still ~39-42 ms. Even with zero misses, K 2.5 caps chat near ~52 tok/s. After M1 the wall is the check's fixed cost (B1 / G1 / G2) and K | - | - | - | - | the next walls |
| M1-X (NEW 01:2x, EXACT-SAFE, the first M1 arm): LEVER 2 on chat. L+1's posted forecast is read into the purgeable volatile tier right after L's release, so a wrong guess evicts nothing (BIGMOE_FC_F 4 / FS 2, arm gate+t+pf + BIGMOE_PRED_LINES=4, VTIER on; BigMoE e8b7b6d / ext build_l2). It has only been measured on 16-row CODE: -4.3 % at C 224 (76,957 reads issued, 9,010 used). Never on chat, where 2-3-row layers take ~0.5-0.9 ms, about one miss latency, and ~89 misses per check stall ~17 ms. Forecast recall@10 ~0.73-0.79 (LEAD1CORR / 3b) | BigMoE | KILL [K] (CHS2 x2l2f): 4x the SSD MB (183 vs 42 per kept token), check +5 ms, lifetime peak 48.8 GB (over 45.0) | EXACT (prefetch only) | - | KILLED, NewIdeas' proposal |
| M1 RANKED (Breakthrough 01:3x, 04c73ae, SCOREBOARD "Sun 00:1x: M1"; bytes from the BYTES10 replay core, within ~5 % of N45 exact MB per kept token; quality = map chat flips, INFORMATION under Kamen's 09:3x rule: gates = known benchmarks only). FINDING: QOM is OFF on chat today (qom_skip 0 in N45), so L's chat reads exact's bytes and L's +13 % is kernels (C4). Arms in order: M1-A = QOM0 on chat (drafted rows skip cold ranks 2-9: -32 / -27 % MB IFEval / multi, svc -4.8 ms) + a pool split (S 16: -41 / -32 %, svc -6.2 ms [P]), free on the map (the split pays only under QOM; S 32 under exact reads is +11 %). M1-B = every bonus row gated (DRAFT17 on chat) + S 32: -75 / -75 % MB, svc -11.3 ms [P], +1.4 / +1.6 points vs M1-A on the map, inside the lines (multi margin 0.43). Then NW1 on chat (~free on the map, flagged). U3 on chat needs a fresh gate. TOP8 DROPPED (multi FAIL, z +7; a map ranking, INFORMATION under the rule) | BigMoE runs, Breakthrough ranked | M1-A -6.2 ms, M1-B -11.3 ms svc [P] | gated | handed to BigMoE 01:3x | ranked |
| M1: miss stalls (20.3 ms mean, ~90 misses per check). Lossy chat levers: QOM on chat (sure rows skip cold ranks), 3-bit cold reads (U3) on chat, top-8 on chat (CH5), chat-aware pools. NW on chat only with a fresh gate (its chat map FAILED) | BigMoE + Breakthrough | CEILING <= ~15 ms per check, x1.25-1.3 (BigMoE 00:5x) | gated (pools: exact) | A/Bs before CERT (BigMoE, hard stop ~01:20), then after | assigned 00:4x |
| D1: the draft 8.4 ms (MTP head, ~2.6 ms per drafted row, serial): one fused forward for all rows; or overlap it with the check's GPU tail | MTP Head | -5..-6 [P] | exact on chat (chat runs exact acceptance) | Sun ~10:00 | D1 ENGINE RESULT (BigMoE CHS2): KILL on L (draft 16.2 vs 9.1 ms per check in x2p4); neutral and exact on E (12 / 12). MTP Head's isolated measurement: draft step 2.63 -> 2.34 ms per row with 4-bit trained parts (hits equal), 2.14 without the per-step sync, so -1..-1.5 ms per check, +3-4 % chat [P]. Flag R57_MTP_P4 sent to BigMoE for Sunday's A/B. -5 ms is NOT reachable by Monday: the 248K output layer is 1.49 ms per row, and 3 / 2-bit or a reduced vocab lose 5-14 % kept. F2a: corpus in (UltraChat shard + Aya, 381 MB, under Kamen's standing OK); armed for Sun 23:00, result ~07:30 Mon |
| K1: acceptance 2.5 -> ~3.3: the relaxed chat rule (E / R1: +7 % kept measured), plus MTP Head's F2a trained head if a first one exists by Mon | BigMoE (rule) + MTP Head (head) | x1.07 by Mon (E only) | gated | Mon | F2a by Mon: NO (MTP Head 888f596: a first head Tue-Wed at best, x1.1-1.3 kept [G]; it needs a chat-corpus download OK + a quiet whole-Mac night). E = R1(0.5) (keep draft j while logit(d_j) >= max - 0.5), HC8 OFF. On today's stack that needs code: R57_RELAX_DELTA relaxes DFlash2 rows only (chat uses the MTP head, screen :1127), and R57_HC8 is global (:803) |
| C4: L's chat not == AR (6 / 24): BigMoE's bisect, which decides whether L's +13 % on chat is a gated lossy lever or a bug | BigMoE | x1.13 [K] | gated | Sun | chs3: OOO_1C + NAX_1CQ (nondeterministic) and REF2_SMALL_NATIVE break chat exactness. chs4: + REF2_NAX_COH == AR 12 / 12 in two runs (vs 3 / 12), no speed cost; SMALL_NATIVE still open |
| GATE: an IFEval same-scores gate (IFEval + multi, both orders, vs NAX6-N) ready to run within 1 h of a candidate | EvalSuite | - | - | pre-register Sun night | asked 00:4x |
| BC1 DROPPED (Breakthrough's map 01:3x: multi FAIL, z +7 vs QOM 0.5): top-8 routing on chat rows (drops ranks 8-9, today 3-bit tail): ~-20 % expert uses, ~-0.5-0.6 GB per 3-row check, and ~-20 % of miss reads. Prior at 12 GB (L1-17): IFEval -2.0 [-4.3, +0.3], +17.4 % tok/s | BigMoE | -2..-4 [P] | gated | A/B Sun | added 01:1x |
| BC2 DROPPED FOR CHAT (Breakthrough 04c73ae + Fit12 'BC2'): no 45 GB 2-bit path exists. 24 GB POM saved SSD bytes -19 % but tok/s -1.9 % (a CPU rebuild per 2-plane read); a GPU cut needs a true 2-bit slot and kernel. After QOM on chat only <= -4..-9 % of bytes are left for MSB-2. IFEval risk is high: MSB-2 = 5x MSB-3's weight MSE; ranks 2-9 ~+2-2.5 points [P]. Fit12: if ever, a map arm for ranks 6-9 emulated first (expect BETWEEN at best). Original line: 2-bit tail ranks: ranks 2-9 MSB-3 -> MSB-2 (records3 is MSB-3 + LSB-1 today; plan 4433333333): ~-27 % of expert bytes, ~-0.9 GB per check, ~-27 % of miss bytes. History: B2 cold-only MSB-2 BETWEEN (fails code P99.9); the KL model reads chat 8-19 % worse than predicted; R-44b POM (2-bit on miss) KILL; plane-split reads KILLED for request overhead. Needs a 2-bit read path: Leap's question, sent to Breakthrough / Fit12 01:1x | Leap (kernel) + Breakthrough (quality) | -3..-4 [P] | gated | after the answer | asked |
| BC3: 3-bit dense on the chat path: dense ~3.1 -> ~2.3 GB per check. A second dense copy costs +2.3 GB (over 45.0 unless C drops ~17 slots); 3-bit dense everywhere makes code / math lossy too. Dense precision is usually more sensitive than expert tails | Leap / BigMoE | ~-3 [P] | gated | price Sun | added 01:1x |
| BC-SUM: BC1 + BC2 (+ BC3) take the bytes per 3-row step ~6.4 -> ~4.5-5.0 GB. The floor then sits at ~17-18.5 ms: ~135-147 tok/s at K 2.5 at the floor, ~190+ at K 3.5. Every piece passes the IFEval gate first | - | - | - | - | Leap's 6.4 GB / ~24 ms floor (01:0x) is the basis |

- PHYSICS BOUND (NewIdeas 01:0x, with MTP Head's F2a headroom table, TrainLab 888f596):
  - LEAP'S PROJECTION (01:0x, tighter than mine): a 3-row chat step reads ~6.4 GB, a ~24 ms floor, so ~105 tok/s at
    K 2.5 even with perfect kernels and zero misses. 150 needs FEWER BYTES (BC pieces) AND HIGHER K (F2a).
  - Every check streams all dense weights, ~3.1 GB, plus ~1 GB of experts at 1-3 rows. At ~280 GB/s that is a byte
    floor of ~15-16 ms per check.
  - At today's K (2.34-2.5), chat reaches ~150 ONLY with every check AT the byte floor, zero miss cost, zero Python and
    a free draft.
  - MTP Head's simulation: today's head + a 30 ms check gives 74 tok/s; a strong head 104; a near-perfect head 133-140.
  - So 150 needs BOTH a check near ~20 ms (about 1.3x the floor) AND a strong head (K ~3.6+), or lossy cuts to the
    BYTES themselves: fewer experts per chat token, lower dense precision.
  - Neither is a Monday item. HONEST MONDAY: a check ~25-30 ms + draft ~3 ms at K ~2.5-2.7 gives ~80-100 tok/s.
  - CORRECTED 00:59 (Leap 00:5x, measured): after M1-B a 1-2-row chat check (~43 ms at < 30 misses) is GPU-bound: dense
    ~22 (floor ~11), experts + router + shared ~11-16, miss spin ~5. The dense pass at 1-2 rows runs at ~50 % of its
    byte floor because every kernel is sized for 16 rows.
  - HONEST MONDAY CHAT = ~55-60 tok/s (M1-B + the low-row list + D1), NOT 80-100. 150 needs a fused low-row dense path
    (whole-layer kernels, multi-day) + K >= 3.5 (F2a) or fewer bytes.
  - PREFILL [K] (BigMoE): PF-A layer-major (R57_PFLM=1), exact, bitwise logits. 1.6K prompt TTFT 15.5 -> 7.8 s
    (101 -> 200 tok/s), SSD -42 %, peak 43.3 GB. 4.7K / 7.9K in PFX after CERT.
  - PFX RESULT (BigMoE 07:57; re-derived by NewIdeas 08:0x from runs.json / tokens.json): PF-A, exact.
    - 195 / 368 / 441 tok/s at 1.6K / 4.7K / 7.9K, x1.90 / x2.90 / x3.44 vs the 1024-chunk exact arm.
    - Logits-hash lists and tokens are identical; SSD 64-68 GB per prompt; peak 45.10 GB (0.1 over: trim).
    - BIG sent to the orchestrator.
    - My chunk-size line (R57_PFCHUNK 8192): BETWEEN (x2.46 at 4.7K, tokens differ, peak 53.1 GB).
    - 500+ at 7.9K needs more. Long prompts look compute-bound (68 GB in 18 s = 3.8 GB/s vs 8.0 GB/s at 1.6K) [P]:
      PF-B (compute) for long prompts. Skipping re-reads of the ~57 % of base expert-layers already resident at
      prefill start (PF-D) helps short prompts (~-47 % bytes [P]).
  - TIME LABELS: this block's "01:1x-01:4x" labels were estimates written without `date`; the real times were
    ~00:3x-00:5x (`date` 00:59).
- ARITHMETIC: if B1, G1, G2, M1 and D1 all land, the step goes ~45 (lossy) -> ~15-25 ms. With K ~2.7-2.9, that is
  ~110-190 tok/s [P].
  - 150 by Monday needs nearly every piece to land at its high end. ~100-130 is the realistic outcome of a good strike.
  - Each piece reports [K] here as it lands or dies, and NewIdeas tells the orchestrator at once.

QWEN 4 DAY-0 PRE-WORK (behind the CHAT150 pieces; due Tue 10 Nov evening). Source: MoonShots' kit,
~/MoonShots/QWEN4_READY.md + qwen4/ (b660ce9), section 3; the orchestrator 00:0x.
- Today's model is already the Qwen 4 architecture preview (qwen4_exp_text). If Qwen 4 keeps it, the port is mostly sizes.
- The watcher is running; qwen4/port_check.py prints the config diff, sizes and hazards at T+0.

| owner | pre-write | done when |
|---|---|---|
| BigMoE | the records3 builder + native / p11 / pad converter driven by config (experts, moe_intermediate, hidden, layers); the r57 screen with a model-path argument and config-derived sizes; one runner script for every table cell | a dry run rebuilds today's store with matching md5s |
| Leap | kernels with compile-time shapes today (NAX6-N, 3c, dense, GDN, attention) made shape-generic, or a one-command rebuild for new H / I; the native repack driven by config | today's shapes rebuild bitwise through the generic path |
| EvalSuite | an AR-reference generator with a model-path argument; the same-scores harness with a model switch | today's model through the switch reproduces ref2tnax |
| Breakthrough | the decision map's baseline capture (top-8 rows, xcapture) with a model switch | today's capture reproduced. INSTALLED 04:4x (a03955b: DM_MODEL / DM_ARCH switch, CAPK from top_k, MOEL from config; CPU checks OK). Reproduction pre-registered (capture DM_C=51, 36 windows + xcapture --max 1; PASS = bit-identical tk / lp / routing / xcap). GPU slot given 05:09: after FU, share lock MEM 18 GB, by 22:30 |
| MTP Head / TrainLab | a day-0 drafter decision tree: (a) the model's own MTP layer (zero training), (b) DFlash2 retraining (capture + train, hours), (c) prompt-lookup copy drafting (model-agnostic) | written, with the time each takes |
| MoonShots | make_records3.py (its own file); the planes_engine copies of gdn_exact / tierglu, with Leap | its port-map rows 9, 13-14 |

- DAY-0 CHECKS, the three silent hazards (each must fail LOUDLY at load, not silently):
  - H1: more than 527 experts. The ext's `li = e/31` is unchecked, and the residency tables cap at 527
    (bigmoe/gated.py TLINES 17). Raise the caps or assert.
  - H2: changed GDN dims. The exact kernels fall back to non-exact. Assert exactness at load.
  - H3: a changed EOS / tokenizer. Decoding would never stop. Read EOS from the tokenizer config and assert.
- Blockers, Kamen / the orchestrator, Sunday:
  - Disk: 88 GB free; day 0 needs >= ~200 GB (4-bit MLX) or ~450 GB (bf16).
> (a quote from the project owner, kept private)
    orchestrator.
  - The watcher's LaunchAgent install.

> (a quote from the project owner, kept private)
chat. Because imagine we say 180 TPs and then it's 40 on chat which is what the user will notice first. And make sure it
sticks to 32, 24 and 12 gb". The publish table leads with chat per memory line.
- THE ARITHMETIC (45 GB): chat tok/s = K (kept per check) / cycle (draft + check).
  - Today: K 2.5 and a cycle of ~48-60 ms (check ~40-51 ms at 1-3 rows + draft ~2.6 ms per drafted row), so ~40-45 tok/s.
  - 150 needs (K, cycle) = (3.75, 25 ms), (4.5, 30 ms) or (6, 40 ms).
  - So BOTH factors are required. No single lever gets there.
- F1, CYCLE x1.7: the check at 4-6 rows ~45-58 -> ~25 ms, and the draft ~8 -> ~3 ms.

| # | lever | kind | price | owner | dated path |
|---|---|---|---|---|---|
| F1a | PRICED by Leap 00:3x (TABLE_350). A 2-row check is 54.5 ms wall; main-thread CPU is 29.3 ms, with ~25 ms waiting in the build, so the GPU binds. The 2-row dense map: ~20-22 ms of GPU vs a 3.1 GB byte floor (~11-12 ms). Pieces: every call padded to 16 rows (pads 1.44 ms); HC8 5.9; in_proj 4.35; out / o_proj 1.93; GDN step 1.12; att q 1.02; router 0.89 at 84 GB/s; shared 1.13; glue ~1.5. Items: (1) row-native low-row GEMVs, no pads, K split per shape, -4..-4.5 ms, ~1.5 d; (2) HC8 at low rows (k2b folded into k3c's prologue, which can stay bitwise; row-native k1c / k3c), -1.5..-2 ms, ~1 d; (3) glue fusion, -0.6..-1 ms, ~1 d | lossy-class, gated on IFEval vs NAX6-N | total -6..-7.5 ms per chat check, x1.13-1.18 on the chat cycle [P] (CORRECTED down from my x1.25-1.4) | Leap | a gated set ~Wed. The next walls for 150 are the experts at 2 rows (NAX ~14 ms) and the MTP draft |
| F1b | Chat misses (~10-20 ms per check at 3-6 rows): chat-aware pool sizing (exact) + CH6's chat QOM rule (lossy) | mixed | x1.1-1.2 [P] | BigMoE | A/Bs Sun-Mon |
| F1c | Draft cost 2.6 -> <= 0.6 ms per row: one fused forward drafting all rows (EAGLE-style; the head is the same model as F2a) | exact-safe on chat | x1.1-1.15 [P] | MTP Head | with F2a |

- F2, ACCEPTANCE x1.8: K 2.5 -> ~4.5 at ~6 rows means per-token acceptance ~0.6 -> ~0.85.

| # | lever | kind | price | owner | dated path |
|---|---|---|---|---|---|
| F2a | A trained chat head on the 125B's own features (EAGLE-3 style). Features come from the prefill path, which after PFX should run ~500 tok/s, so ~2M chat tokens per hour. A small head trained on the laptop (MLX); free compute only | exact-safe on EXACT; changes nothing on chat in L (chat runs exact acceptance) | THE ONLY CANDIDATE >= x1.5 on chat: K 2.5 -> 3.75-4.5 if p reaches ~0.8-0.85 [G, unproven] | MTP Head | feasibility Sun noon; feature run Sun; a first head ~Tue-Wed; tuned ~1-2 weeks |
| F2b | A small same-tokenizer dense drafter (vanilla speculative decoding). Priced: a 0.6B at ~1.5 ms per token and p ~0.55-0.65; a 1.7B at ~3.5 ms and ~0.65-0.7; a 4B at ~8 ms and ~0.7-0.75. Net x0.9-1.15 on chat [P], no better than the MTP head. It costs RAM (+0.4-2.3 GB at 4-bit), which is bad at 24 / 12 GB. It needs a DOWNLOAD (the family's 0.6B / 1.7B / 4B at 4-bit, ~0.4 / 1.0 / 2.3 GB), so Kamen's OK first | exact-safe | x0.9-1.15 [P] | MTP Head (A/B only if Kamen OKs a download) | not the main path |
| F2c | Lossy chat acceptance (R1 / E rules), gated | lossy | +7 % kept measured (E, 11-05) | BigMoE; gate | small |

- HONEST DATE at 45 GB:
  - 150 on chat is plausible ~3 weeks out (late November) at the earliest. It needs F2a to reach ~0.85 per token AND F1a
    to reach ~the floor. Neither is proven; the risk is high.
  - By Wednesday: ~60-70 tok/s lossy [P] (CH1 + F1b + F1c + part of F1a).
  - CORRECTED 00:3x (Leap priced F1a at x1.13-1.18, below my x1.25-1.4): F1 by Wednesday is ~x1.2-1.35, not x1.7.
    The rest of F1 needs the experts at 2 rows (~14 ms) and a fused draft. That makes 150 at 45 GB >= 3 weeks and less
    certain. Wednesday's honest chat number stays ~55-70 lossy.
- LINES 32 / 24 / 12 GB, chat (none of these chat cells is measured yet):

| line | today (code) | chat | what 150 on chat needs | owner (measure first) |
|---|---|---|---|---|
| 32 GB (C30, C 160, 31.8 GB) | 153.5 HE / 114.8 MBPP | MEASURED 16:2x [K] (BigMoE c32, re-derived): exact E 33.8 (30.6 GB, 12 / 12 == AR); zs 51.3 (31.2 GB, lossy, ungated at C 160). My guess was ~35-42 [G]: zs came in far above it | F1 + F2 + the extra misses of a smaller cache | BigMoE: 12 IFEval at C 160 (after PFX) |
| 24 GB (E24, s10b) | 41.3 HE exact, 355 MB SSD per token (MBPP 494) | not measured; ~20-30 [G] | SSD-bound: at ~12 GB/s, 150 means <= ~80 MB per kept token, x4.5-6 fewer bytes (the BYTES10 track) + F2 | BigMoE: E24 on 12 IFEval with MB per kept token |
| 12 GB (verified greedy) | 18.6 HE (~440 MB per kept token; ~8 GB/s of SSD, near the limit) | not measured | <= ~80 MB per kept token: x5.5 fewer bytes + F2 | Fit12: 12 IFEval on the loop, MB per kept token |

  - Bytes per kept token fall as K rises: one check's expert union is shared by every kept token. So F2a helps every line.
    The rest at 24 / 12 GB is BYTES10: smaller cold reads (3-bit / 2-bit sidecars, lossy) and expert skipping (lossy, gated).
  - HONEST: 150 on chat at 24 / 12 GB is a research program, not this month.
- PREFILL at every line: the PFX mechanism (one tier call per prompt = expert-once) applies to all lines. Its SSD floor is
  the non-resident expert bytes / ~12 GB/s per prompt: ~2 s at 45 GB, ~3 s at 32, ~4 s at 24, ~4.6 s at 12 [P].
  - With compute ~3-5 s at 4-8K: 500+ at >= 4K prompts is plausible on every line [P].
  - Short prompts (~1K) stay floor-bound, ~200 tok/s at 24 / 12 GB, until PF-D (skip resident) and smaller reads.
  - Owners: BigMoE (45 / 32, and E24's stack at 24), Fit12 (12).
- RE-AIMS:
  - Fit12: 12 GB chat (12 IFEval: tok/s + MB per kept token) and 12 GB prefill with one whole-prompt call.
  - Breakthrough: chat BYTES10 at 24 / 12 GB. Price MB per kept token on chat, and the levers to <= 80 MB, each with a
    quality gate (their 12 GB lossy code mode failed).

PLAN CHAT 150 / PREFILL 500 (NewIdeas Sat 23:5x). Kamen via the orchestrator 23:4x, verbatim excerpts:
"until Tuesday max Wednesday we must have figured out everything wrong with them and get chat TPs to at least 150 and
prefill would be great if we got to at least 500, 1000 is better", "we MUST get chat TPs to 150 because that's what a
day to day user will feel most", "ready to publish at latest Wednesday".
- CHAT, measured (BigMoE N45, 45 GB C 256, 12 IFEval; re-derived per check by NewIdeas: scripts/chat/chat_rows.py, data/firstcheck/chat_rows_n45.txt):
  - Today: exact 39.9 tok/s, lossy config 45.0.
  - Checks verify only ~3 rows (median 2-3; the MTP head's t 0.3 cut), and keep 2.5 (81 % of rows).
  - A check costs ~51 ms exact / ~40 ms lossy EVEN AT 1-2 ROWS with <= 10 misses (the floor).
  - SSD misses add ~4 ms at 1-2 rows, ~10 at 3-4, ~22 at 5+.
  - The draft is ~2.6 ms per drafted row (serial MTP head).
- What limits chat:
  - Not verify width (few rows are wasted) and not mostly the SSD. It is the low-row check's fixed cost: ~2-2.5x its
    memory floor (~15-20 ms for a 1-2-row forward: ~2.7 GB dense + ~1-2 GB of experts at ~295 GB/s).
  - The second limit is acceptance (2.5 kept).
  - 150 needs a ~25 ms cycle with ~3.75 kept: the low-row check near its floor (x2) AND x1.5 acceptance.
- HONEST CHAT PRICE by Wednesday: lossy ~55-65 tok/s [P], exact ~42-45 [P]. 150 is a multi-week program: kernels near
  the bandwidth floor at 1-4 rows + a trained chat drafter.
- No chat lever prices >= x1.5. E, the 11-05 chat relax arm R1(0.5), read x1.46, but its kept rose only +7.2 % (2.67 vs
  2.49). Most of that x1.46 was kernels (HC8), and E + HC8 failed IFEval (84 vs 90).
- Chat levers, ranked:

| # | lever | kind | price | owner | first step |
|---|---|---|---|---|---|
| CH1 | The lossy config's chat (L: 45.0 vs 39.9 = x1.13 [K]), but only 6 / 24 == AR (C4). Find the bit-changer (BigMoE's FU bisect, armed); likely a FAST kernel acting on chat (DN-KS / HC8_FAST) | lossy unless the bisect makes it exact | x1.13 [K] | BigMoE; EvalSuite gate | the bisect; then an IFEval same-scores gate vs NAX6-N on L's chat |
| CH2 | Low-row dense kernels: Leap's step 0 B 2-row table (out_proj / o_proj 5.4, SDPA 3.2, shared gate\|up 2.3, a\|b 1.6, k\|v 1.2 ms ...). Bitwise is exhausted, so bit-changing forms are needed | lossy (gate) | x1.2-1.4 [P] if the 2-row dense part drops ~25 -> ~15 ms | Leap -> BigMoE; gate | price the 2-row pass in place on L; build the top items |
| CH3 | Drafter speed: the MTP head at ~2.6 ms per drafted row, ~8 ms of a ~50 ms cycle. Chat runs exact acceptance even in L, so a drafter change can't change chat outputs | exact-safe on chat | x1.05-1.1 [P] | MTP Head | a fused / 4-bit head, bitwise not needed |
| CH4 | A better chat drafter: acceptance 2.5 -> 3+ (an EAGLE-3-style head trained on 125B features, free compute only) | exact-safe in EXACT; changes nothing on chat in L | x1.2-1.4 [G], days | MTP Head | feasibility by Sun noon: data, time, compute |
| CH5 | Top-8 experts on chat rows (published top-k reduction; 12 GB L1-17: +17.4 % tok/s, IFEval -2.0 [-4.3, +0.3]) | lossy (gate) | x1.1-1.15 [P] | BigMoE; gate | an engine A/B on 12 IFEval |
| CH6 | The 11-04 chat FAST rule (QOM cf 0.5 + bonus gate), against the 4-22 ms of misses per check | lossy (gate) | x1.05-1.15 [P] | BigMoE; gate | an A/B |

- PREFILL, measured (N45, exact): 98 / 118 / 124 tok/s at 1.6K / 4.7K / 7.9K tokens (TTFT 16 / 40 / 64 s). The cost is
  flat at ~8-10 ms per prompt token, consistent with a per-1,024-row-chunk cost (each chunk re-reads the experts).
  - Lossy 7.9K peaks at 45.1 GB, over the line (BigMoE's FU fixes it).
- PRIOR ART, our own: PF2 layer-major prefill (64 GB era, every expert once per prompt) measured x2.05 at 4K
  (28.5 -> 13.9 s) and x2.16 at 16K, shas identical. lmp4pd added -8..-13 % more. PF1 staging (read L+1's experts during
  L) is hidden inside it. After layer-major, compute dominates: ~5 TFLOPS effective at 4K.
- Prefill levers:

| # | lever | kind | price | owner |
|---|---|---|---|---|
| PF-A | CORRECTED by BigMoE 00:0x: NO PORT NEEDED. Tier mode's prefill_call -> _prefill_grouped already reads each unique expert once per CALL (double-buffered, exact 4-bit from records3). The screen calls it per 1,024-row chunk, so a 7.9K prompt re-reads ~all experts 8 times. One big call = expert-once. lmp4pd adds only bounded activation memory + the PLE precompute. PFX is armed for CERT_DONE: exact at R57_PFCHUNK 1024 / 2048 / 4096 / 8192, plus lossy at 8192, on agent/edit10 1.6K, write13 4.7K and write11 7.9K. It reports TTFT, tok/s, lifetime peak, decode tokens vs the 1024 arm, and the PF1 split. LINES (NewIdeas): PASS = write13 TTFT >= x2.5 at 8192 vs 1024 AND decode tokens == the 1024 arm AND lifetime peak <= 45.0 GB; KILL < x1.5 | exact if the tokens match | BigMoE est. x4 at 4.7K (~470 tok/s), x6 at 7.9K (~740), x2 at 1.6K (~200: a short prompt still reads ~the whole model once, ~7 s) | BigMoE |
| PF-D | Short prompts: skip pool-resident experts in the prefill reads (they are already in RAM). For <= 2K prompts the ~7 s whole-model read dominates | exact | lifts 1.6K beyond ~200 [P, unpriced] | BigMoE |
| PF-E | The AR rows for the audit: EvalSuite has no NAX6-N AR rows for the agent prompts. Decode NAX6-N one-token AR for write13 + write11 (>= 64 tokens each) after CERT, so the R-56-style audit compares the big-call prefill's decode against AR and not just the 1024 path | exact reference | enables the exact claim | EvalSuite |
| PF-B | Prefill expert GEMMs on the M5's matrix units (NAX) after layer-major | exact-safe only if the decode still matches AR (R-56-style audit) | x2-3 [P], so ~500-800 tok/s | Leap |
| PF-C | PF1 staging inside PF-A, hiding the ~2 s of SSD (~25 GB at 12 GB/s per prompt) | exact | part of PF-A | BigMoE |
| PF-A-EXACT | EXACTNESS (Leap 00:3x): a ONE-CALL prefill is NOT bitwise vs 1,024-row chunks (0 / 9 decode hashes equal; tokens equal). Layer-major with 1,024-row sub-batches IS bitwise by construction. So the EXACT mode's prefill = layer-major + 1,024-row sub-batches (lmp4pd-style, enabled in tier mode). The one big call is fine for the lossy mode (tokens equal so far; covered by the lossy gates) | exact | same expert-once read as PF-A | BigMoE + Leap |
| PF-F | FREE FIX (Leap 00:3x): the screen's prefill computes full-vocab logits for EVERY prompt row (2.35 GB at 4.7K rows). Keep only what is used (the last row; the drafter's needs). Speed + memory, and it likely explains part of the lossy 7.9K peak at 45.1 GB | exact | free | BigMoE |
| PF-B' | PRICED (Leap 00:3x): MLX gather_qmm reaches 7.1 TF at 1K rows and 13.8 at 8K (dense qmm 27-29). Bitwise first: gather_qmm on a layer's whole prompt (pairs are invariant 1K-8K, measured): x1.4 on the experts, ~0.5 d once PF-A lands. Then an MPP grouped expert GEMM (~18-22 TF): prefill compute ~0.97 -> ~0.55 ms per token [P], lossy unless the R-56 audit holds, ~1.5-2 d | exact (gather) / audit (MPP) | x1.4 then ~x1.8 on compute | Leap |

- HONEST PREFILL PRICE (CORRECTED 00:1x with BigMoE's PFX estimate): long prompts ~470-740 tok/s from one big call
  alone [P]. Short prompts (1.6K) ~200 until PF-D + PF-B land. 1,000 at long prompts needs PF-B. PFX decides at
  ~08:00.
- ORDER (code from tonight; GPU after CERT, ~07:30):
  1. Tonight, CPU: BigMoE ports PF-A and runs the C4 bisect (armed). Leap writes the NAX prefill GEMM plan and prices
     the low-row kernels. MTP Head works on drafter speed + CH4's feasibility.
  2. Sun after CERT: the PF-A kill test (4.7K prompt, layer-major vs chunked, TTFT + tokens == AR, ~10 min quiet). The
     C4 bisect result. EvalSuite's IFEval gate on L's chat.
  3. Sun-Mon: PF-B, CH2, CH3 / CH5 / CH6 A/Bs, gates.
  4. Tue: final numbers. Wed: publish.

> (a quote from the project owner, kept private)
morning. You can run tests for now as much as you want. I probably won't stop you until tomorrow morning or midday".
- The table: 45 / 24 / 12 GB x exact / lossy. Every cell gets:
  - whole decode on the S5 prompts (12 HumanEval + MBPP12);
  - the lifetime peak;
  - quality status: a bitwise / AR-token proof for exact, a gate verdict for lossy.
  The orchestrator compiles it at ~08:00 from the reports.
- Have: 45 lossy = S5, 175.9 / 118.3, 42.9 GB, S4 PASS. 12 lossy = t80ck1u, 46.3, gate FAIL.
- Rules: one big GPU job at a time. Timing runs use quiet locks, gates use shared locks. Each owner reports on finishing.
- Times are estimates. If a slot runs long, the later ones move up; the 01:30 CERT does not move.

| # | cell | run | owner | lock | est. time | quality status |
|---|---|---|---|---|---|---|
| 1 | 45 exact | E5 (NAX6-N + KEEP + wired), plus an E5' arm with the seams / DQ1 bundle | BigMoE (armed) | quiet | 17:3x-~18:45 | AR-token proof vs NAX6-N |
| 2 | 45 lossy (update) | S5' = S5's config + seams / DQ1 (p4s: +2.0 %) | BigMoE | quiet | ~18:45-19:05 | bits PASS 43 / 43, so S4 and CERT carry |
| 3 | 12 exact-ish | 12E: the verified-greedy loop (ex16 / kp2 form) on the S5 prompts. The 18.42 was measured on 16 HumanEval, not this set | Fit12 | quiet | ~19:05-19:20 | verified greedy, not bit-exact (outputs = kp2's) |
| 4 | 12 lossy (retry) | 12F: θ 0.95 measured run on the S5 prompts. A new pre-registration after two FAILs, and EvalSuite pre-registers first | Fit12 | quiet | ~19:20-19:30 | its code gate in slot 7 |
| 5 | 24 exact | E24 (DECIDED 17:3x with BigMoE): W1B's 24 GB stack (s10b, stock kernels, C 128, MAXK 80 + row chunking, 23.86 GB) + mx.set_wired_limit, the only exact-safe fix that applies to it (KEEP, NAX, S2 and the seams are r57 / ref2-only). 12 HE + MBPP12, two passes. Why not r57: r57 cannot run EXACT below C ~160 (an exact 16-row check needs up to 160 distinct experts per layer, so the ext aborts with "no free slot"), and 24 GB gives C ~104-112 | BigMoE | quiet | ~19:30-20:45 | exact vs its own stock-kernel AR (proven Wed, w24 == w24t); tonight pass 1 == pass 2 + agreement with NAX6-N's AR (information) |
| 6 | 24 lossy | F24 (DECIDED 17:3x): r57 v1' + wired + seams / DQ1 at the largest C with a lifetime peak <= 24.0 GB (C ~104-112 est.), R57_VTIER off (its 7.45 GB would not exist on a 24 GB Mac). A probe first: 12 HE at C 104 / 112 for peak and abort-safety, then the passes. CODE ONLY: v1' runs math / IFEval exact, which r57 cannot do below C ~160 | BigMoE | quiet | ~20:45-21:45 | its gate in slot 8 |
| 7 | 12 lossy gate | θ 0.95 code same-scores vs REF2-T (the 12 GB rule), both orders | EvalSuite | shared | ~21:30-22:15 | PASS / FAIL |
| 8 | 24 lossy gate | F24, CODE-ONLY (DECIDED 17:3x): HumanEval+ same scores in both orders + long code vs NAX6-N's existing reference rows (exact outputs don't depend on C). No math / IFEval, and no QOM-all-sets variant (not v1''s rules). A "no free slot" abort on any row makes the gate INCOMPLETE, and the table says so | EvalSuite | shared | ~22:15-~00:30 | PASS / FAIL / INCOMPLETE |
| 9 | (release) | CERT on cert_env (S5's config; S5' carries bitwise) | EvalSuite | as planned | Sun 01:30 -> morning | unchanged |

- Slack ~1 h before CERT. If the 24 GB runs overrun, gate 8 may slip past 01:30. Then CERT goes first, and F24's gate
  runs after CERT or is marked "pending" in the table. The release's CERT is never delayed.
- TABLE RESULTS, Sat 18:1x. Every number was re-derived by NewIdeas from the raw per-prompt files (scripts/table/). Whole decode tok/s, geo mean over prompts:

| line | mode | 12 HumanEval | MBPP12 | lifetime peak | quality status |
|---|---|---|---|---|---|
| 45 GB | exact: E5 = NAX6-N + KEEP + wired (BigMoE f451b54) | 99.4; with the seams / DQ1 bundle 102.7 | 71.3; 73.3 | 42.2 GB | AR-token proof PASS, 96 / 96 runs |
| 45 GB | lossy: S5' = v1' + wired + bundle (f2c4728) | 181.4 (B 174.4; S5 175.9) | 123.6 (B 120.2; S5 118.3) | 42.9 GB | S4 PASS vs NAX6-N; the bundle is bitwise, so S4 / CERT carry |
| 24 GB | exact: E24 = W1B's s10b stack + wired, C 128 (601f66f; MBPP re-run with an EOS stop, E24m 4246291) | 41.3 | 38.4 with the EOS stop (passes 38.2 / 38.5; the N 256 run gave 41.7, not comparable) | 23.86 GB (MBPP run 23.52) | exact vs its own stock-kernel AR (Wed); tonight pass 1 == pass 2 on 24 / 24; NAX6-N agreement 6 / 24 (information) |
| 24 GB | lossy | n/a: an engine limit | n/a | - | r57's first check of a prompt runs ungated (exact), so it needs C >= ~160, ~30 GB. Routes after the publish: row-chunking in ref2's tier path (Leap), or QOM's rules in the s10b stack |
| 12 GB | verified greedy: 12E (Fit12) | 18.58 | 18.18 | 11.140 GB | verified greedy, not bit-exact; tokens = ex16's 12 / 12 |
| 12 GB | lossy θ 0.95: 12F = t95ck1u (Fit12) | 40.59 | 18.74 | 11.280 GB | GATE (EvalSuite 3a4ea8a, 20:16): vs REF2-T 54 vs 57 of 60 (-5.00, 0 / 3) in both orders = FAIL at the bar. vs NAX6-N 54 vs 55 (-1.67, 1 / 2) would PASS. REFERENCE QUESTION, see the note below |
| ~32 GB | lossy, CONTEXT ROW: C30 = S5''s config at C 160, VTIER off (ef64e24) | 153.5 | 114.8 | 31.8 GB | CODE GATE PASS vs NAX6-N (EvalSuite 907e001): HE+ 53 vs 55 forward (1 / 3, p 0.63) and 55 vs 55 reverse; long code 0 blunders in 29,219 tokens. vs REF2-T (information): -4 / -2 tasks |

  - REFERENCE QUESTION for the 12 GB θ 0.95 cell (NewIdeas' error, flagged 21:1x): I told EvalSuite to pre-register
    this gate vs REF2-T, carrying over t80ck1u's rule. But the orchestrator's 14:0x relay of Kamen's decision made
    NAX6-N the FAST gates' reference, with REF2-T as information. Under NAX6-N the HumanEval+ line passes; under REF2-T
    it fails exactly at the bar. Both numbers stand as measured. Which line applies is the orchestrator's / Kamen's call,
    made on the rule's merits rather than on this result. Note: the 12 GB gate has no long-code line, unlike S4 / C30.
    RESOLVED (the orchestrator 21:1x): the gate stays vs REF2-T, so the FAIL at the bar stands; NAX6-N's -1.67 is
    information.
    - The rule is each mode against its own exact reference. The 12 GB line's own base is REF2-T: its verified-greedy
      loop equals REF2-T's tokens (EvalSuite's VOID check, 64 / 64), and EvalSuite pre-registered it that way at 08:17,
      before any 12 GB verdict. NAX6-N applies at 45 GB / C30 because those stacks run on NAX6-N kernels.
    - The 14:0x relay's "FAST reference" wording was too broad (the orchestrator's note). My REF2-T instruction was
      right under the rule.
    - The cell reads: 40.6 / 18.7, 11.3 GB, FAIL at the bar vs its own base. The publishable 12 GB number stays
      verified greedy, 18.6 / 18.2.
  - (*) E24 runs a fixed 256 tokens with no EOS stop. MBPP answers are ~115-123 tokens, so most of E24's MBPP tokens
    come after the answer ends. Its MBPP figure is not comparable with the other rows; its HumanEval figure nearly is
    (answers ~245-250 tokens).
  - S5''s +4.0 % over B on HumanEval is mostly acceptance noise: kept per check 11.61 vs 11.29 (my aggregation); the
    median cycle moved -0.16 %.
  - The clean measure of the bundle is E5's exact pair, where tokens are identical (kept 11.40 both): +3.4 % HumanEval,
    +2.7 % MBPP12.
  - Pass-to-pass acceptance noise on 12 HumanEval is +-3-4 %. Differences under ~4 % between runs of the same config
    are noise.
  - Optional context point, not a table cell: "~30 GB lossy" = S5''s config at C 160, the smallest C r57 can run.
    Asked of BigMoE 18:1x (~10 min). Its quality basis: runner 49 gated stackb at C 160 / 192 (PASS vs NAX6).
  - C30 RESULT (BigMoE 18:12, ef64e24; that commit's message wrongly says 24 GB): S5''s config at C 160, VTIER off,
    no abort. Re-derived: 12 HumanEval 153.5 (passes 152.3 / 154.7), MBPP12 114.8 (116.6 / 112.9), lifetime peak
    31.8 GB. The orchestrator wants it as a "~32 GB lossy" row, which needs its own code gate, because FAST is
    cache-dependent and S4 at C 256 does not cover C 160.
- REST OF THE NIGHT (NewIdeas 18:5x, the orchestrator's items in my order). One big job at a time; CERT 01:30 fixed:
  1. ~18:5x-19:45, EvalSuite: the 12 GB θ 0.95 code gate (t95ck1u; HE+ vs REF2-T in both orders, the 12 GB rule).
  2. ~19:45-21:00, EvalSuite: the C30 CODE-ONLY gate (HE+ x 2 orders + long code vs NAX6-N), the same form planned for
     F24. BigMoE hands EvalSuite a c30_env (S5''s env at C 160, VTIER off).
  3. ~21:00-21:15, BigMoE: E24 MBPP with an EOS stop, if s10b's harness can stop at EOS cheaply. It makes the 24 GB
     MBPP cell comparable. If not cheap, the cell keeps its caveat.
  4. Until ~01:00, BigMoE (+ Leap for ref2): 24 GB lossy, only if one route builds in <= ~3 h. The routes: QOM's rules
     in the s10b stack, or row-chunking a layer call whose union exceeds C in ref2's tier path. Build on CPU now,
     measure in the GPU gap after item 3. Otherwise the cell stays n/a for Tuesday.
  5. 01:30: CERT (EvalSuite), unchanged.

> (a quote from the project owner, kept private)
publish).
- Target and rules:
  - 220 tok/s WHOLE DECODE, 45 GB FAST, code, 12 HumanEval, lifetime peak <= 45.0 GB.
  - TWO BASES (the orchestrator 14:3x):
    - S5, the headline (BigMoE d8eb2e4, Sat 14:25, quiet, CLEAN; re-derived): 175.9 tok/s, so x1.251 to 220. It also
      gave MBPP12 118.3.
    - S6W's W arm: 181.2, so x1.214. Same config; the two runs are ~3 % apart (QOM varies run to run).
  - At today's K (11.0 kept per check), the median cycle must fall 63.5 -> ~52.3 ms (-11.2 ms).
  - NW v3 stays OUT (Kamen). C 288 is OUT (memory). NAX6-N stays the gate reference.
  - Exact-safe levers adopt with a bitwise check (exact env: verify-logits hash + tokens) and an engine A/B, with no new
    gate; CERT carries over.
  - Lossy levers need an S4-style gate vs NAX6-N, and a CERT re-run on the final config.
- Where the cycle goes now (v1' + wired, S6W's W arm; Leap's Fri / Sat in-place prices):
  - draft ~5.3 ms;
  - dense pass ~25 ms at 16 rows (GPU-bound; Leap's gap hunt, Fri 21:0x);
  - expert path ~26 ms, at NAX6-N's byte floor;
  - miss waits: high-miss checks run 61.5 vs 50.1 ms;
  - commit replay on the 36 % partial accepts, 3.9-4.8 ms each;
  - host ~0.2 ms.
- LEVERS, ranked (exact-safe first; measured before priced):

| # | lever | kind | gain | owner -> A/B | next step | due |
|---|---|---|---|---|---|---|
| 1a | Dense, BITWISE part (PRICED by Leap 15:4x, TABLE_350 fe1b68c / 83a225e, on v1''s real dense program: dense-only 16-row pass 19.79 ms, Python build 12.06; HC reads 3.77, attention 2.64, GDN 8.90 with in_proj 4.62, MoE dense 1.22, PLE 0.52, head 0.78). UPDATE Leap 16:0x (e951042): the HC8 item is a KILL. k3c's ~29 us per call is the code stream (~8.6 us) and the MMA (~10.3) back to back, plus ~10 us fixed. Three bitwise forms lost (31.1 / 34.4 / 42.4 us vs 29.0), and the k2b fold is <= ~0.2-0.3 ms behind a cross-threadgroup sync. Bitwise dense kernels are close to exhausted. What is left: eprep (built, off; -0.3..-0.5 [P]) + GDN TG 2 (-0.23 [P]). Sunday order (Leap, all exact-safe): (1) TG 2 behind a ref2 switch, then eprep + TG 2 priced in situ for BigMoE's bundle A/B; (2) the draft's 5.3 ms priced piece by piece; (3) the commit replay on a second stream: prior KILL D-COMMIT design D (0.06 ms per cycle, 11-03) and NAX-OCC (two streams -1..-13 %), so re-test only on a named change; (4) the cycle's GPU-idle seams (2.5-3.6 ms per cycle: accept logic, commit build, draft build in Python). Caveat for (2), NewIdeas: in FAST v1' the drafts move outputs (B's comment rule, COPY g2; QOM / U3 via cache state), so a drafter-kernel change is gate-free only with bitwise-equal drafts RESULT BigMoE 15:21 (07dbca9): the 1a bundle (eprep + GDN TG 2 + BigMoE's DQ1) is BETWEEN, with cycle +0.16 ms (+0.26 %). Bits PASS 43 / 43 and both pieces engaged, so the -0.5..-0.7 ms priced in isolation gives no cycle gain in the engine. Whole decode -3.8 % = acceptance noise | exact-safe | 0 in the engine [K] | Leap -> BigMoE | dense 1a is done; Leap's remaining Sunday items are (2) and (4) -> 1c | - |
| 1c | Host scheduling seams (Leap 16:3x, TABLE_350 7fee5b8): the GPU idles ~2.5 ms per cycle on v1' + wired (~1.8 on full accepts, ~3.7 on partials). The draft's readback end -> the check's first submit takes 1.27 / 2.47 ms; the check's eval -> the commit's submit 0.32 / 1.22; ~0.2 after the commit. Five fixes, handed to BigMoE (their loop / MTPLX): (a) the per-check PAUSE stat in Gate.new_pass; (b) the first SB async_eval after layer 0; (c) instrumentation after the first submit; (d) on partials, the PLE stage / look-ahead future wait (~0.8) overlapped with the commit build; (e) the commit submitted in pieces | exact-safe (no computation change) | -1..-1.4 ms per cycle, x1.016-1.022 [P] | BigMoE | build + engine A/B | Sun |
| 1c-R | RESULT BigMoE xsafe p4s 15:39 (b6cd335, snapshot 3c962ca): host-scheduling seams a-e + DQ1 on v1' + wired = PASS. Median cycle -1.31 % (63.88 -> 63.05 ms; NewIdeas reproduced it exactly with BigMoE's aggregation, the median over prompts of each prompt's pass-median; other aggregations give -0.87..-1.63 %). Whole decode +2.0 % (175.4 -> 178.8, re-derived); full period +2.0 %; check 0 68 -> 64 ms. Bits PASS 43 / 43, tokens 3 / 3, all engaged. Env adds: BIGMOE_PAUSE_EVERY_MS=50, R57_S0CARRY=1, REF2_SB_FIRST=1, REF2_PLE_LATE=1, REF2_COMMIT_SUB=9, R57_DQ1=1 | exact-safe | x1.02 [K] | BigMoE | JOINS the Sunday exact-safe bundle | Sun |
| 1b | Dense, BIT-CHANGING part (PRICED 15:4x): (1) in_proj K split, REF2_DN_KS=2: ~-0.3 ms, BUILT. (4) NEW: a native router kernel (MLX's 8-bit qmm runs at 84 GB/s, 0.86 ms per check): ~-0.5 ms [P], 0.5-1 day. (2) Chunked GDN step at 16 rows: -0.5..-0.9 ms [P], ~1.5 days; the step is arithmetic-bound (3 bitwise variants, 90 / 90 cases equal, -9 % at best) | lossy-class (new bits) | all three: -1.3..-1.7 ms, x1.02-1.027 [P] | Leap -> BigMoE; EvalSuite gate | see the gate recommendation below | post-release unless the orchestrator wants a Monday gate |
| 2 | R57_GDNB=1 (gdn_body): RESULT BigMoE 14:54 (cf2b69a). Bits PASS in both forms (43 / 43 hashes, 3 / 3 tokens), but G (GDNB + lazy commit) is +0.52 ms per cycle (+0.81 %) = KILL, and G' (lazy off) +0.08 ms = BETWEEN. The whole-decode +3.1 / +1.6 % was acceptance noise. Leap's prior is confirmed. The partial commit's ~0.75 ms per cycle on the wall has no lever left (a compiled replay <= 0.27 ms) | exact-safe | none | - | OUT of the bundle | - |
| 1a+ | Head (final norm + lm_head): 1.63 ms per check at ~120 GB/s (Leap 15:1x) | exact-safe if it can take the native qmm16n | up to ~-1 ms [P] | Leap | check whether the native path is bitwise here | Sun |
| 3 | S2b leftovers: e (eprep, ~0.4 [P]) + f attention k\|v fold (-0.31 [K], BETWEEN alone) | exact-safe | -0.7 ms, x1.01 | Leap | bundle with 2 | Sun |
| 4 | The draft-quality mx.eval merged into the first eval | exact-safe | -0.15 ms, x1.003 [P] | BigMoE | bundle with 2 | Sun |
| 5 | Miss waits (high-miss checks +11 ms): an exact-safe way to read earlier | exact-safe | up to x1.12 if fully hidden [P, upper bound] | NewIdeas (desk) | DESK RESULT 14:3x: nothing buildable by Monday. Prefetch / pre-route was tried five ways (P2-PREFETCH, PF01, PREROUTE2, 3b partial read, LEAD1CORR: recall@10 <= ~0.79). Replacement is bounded: Belady oracles cut only -2..-16 % of misses, and LFU is in. One new angle goes post-release: ROW-WAVEFRONT (below) | closed for Tue |
| 6 | T16 on code: RESULT BigMoE 14:37 (a03e25d; MTP Head verified a5a915f): KILL. Whole decode -0.0 % (173.2 vs 173.2 on v1' + wired, C 256): kept +3.1 %, but the cycle was +1.9 ms (+3.0 %), the InertScan tagger's host work. A cheaper tagger with the same decisions caps at ~+3 %, still lossy (gate + CERT): post-release | lossy | none by Tuesday | - | OUT | - |
| 7 | Drafter K (retrains, the D98 track) | changes FAST outputs | retrains move K inside noise (+-0.16, K_PLAN) | MTP Head | not by Tuesday | - |

- Arithmetic (CORRECTED 15:5x with every lever measured or priced; from S5's 175.9):
  - Exact-safe (CORRECTED 16:3x): 1a gave 0 in the engine, so what is left is 1c's host-scheduling fixes, x1.016-1.022 [P],
    so ~178.7-179.8 if they land. Otherwise S5's 175.9 stands.
  - With all of 1b (if gated): x1.02-1.027 more, so ~184-188 [P].
  - T16 and GDNB are KILLed. 220 (x1.25) is OUT OF REACH by Tuesday.
  - Honest Monday number (16:4x): ~179 [K, p4s X 178.8] for S5's config + 1c-R, against S5's 175.9. The S5' freeze
    measurement on Monday sets the release number.
- MONDAY GATE, NewIdeas' recommendation (15:5x): NO lossy gate on Monday.
  - Only 1b (1) + (4) can be ready by Mon 10:00: ~-0.8 ms, ~+1.3 %. That is not worth a gate plus a Monday-night CERT
    top-up 18 h before the publish.
  - Release = S5's config + the exact-safe bundle. Bitwise, so CERT carries over.
  - 1b goes post-release, with T16's cheaper tagger and the next gate cycle.
  - DECIDED (the orchestrator 15:0x; the default unless Kamen overrides): no lossy gate on Monday. The release is S5's
    config + the exact-safe bundle, and CERT carries over.
    - The orchestrator re-verified both KILLs.
    - It also ruled out the Python graph build (~10-15 ms) as a lever: Leap's B5a / B5b records show it hidden behind
      GPU work (the 16-row forward is ~21 ms wall with ~21 ms of GPU).
    - Kamen is told that 220 by Tuesday is out of reach on measured levers (Monday ~180-183).
- POST-PUBLISH CYCLE (parked; the orchestrator asked to keep it): ONE COMBINED GATE on "v2 max".
  - The config: the release + 1b + whatever new lossy levers clear their A/Bs: (1) DN_KS=2, (4) the native router,
    (2) the chunked GDN step, T16 with a cheaper tagger, and NW only if Kamen re-opens it.
  - The run: BigMoE's A/B of v2 max vs the release, then one S4-style gate vs NAX6-N, then a CERT run.
  - If it fails, only the lossy part is lost.
  - The way to 220 is dense GPU time and expert bytes per check, not host time.
- CUT-OFFS (publish freeze Monday night):
  - Mon 12:00: a lossy gate starts only if the orchestrator overrules the recommendation above.
  - Mon 18:00: the last exact-safe A/B starts.
  - Mon 20:00: FREEZE. S5' is the final number: whole decode + full period, 12 HumanEval + MBPP12, quiet, lifetime
    footprint.
  - Mon night: a CERT top-up only if a lossy lever joined after Sunday's CERT; exact-safe additions carry over.
  - Tuesday: publish (Kamen confirms).
- Tonight is unchanged: S5 on v1' + wired (BigMoE), CERT from Sun 01:30 (EvalSuite). Its config is the fallback release
  if nothing lands.

NEWEST 45 GB CODE NUMBER, Sat 09:13: v1' + wired = 181.2 tok/s WHOLE DECODE [K, BigMoE S6W 2525931, re-derived by
NewIdeas 10:2x]. Setup: v1' (stackb + S2 + KEEP) + R57_WIRED=1 at C 256, 12 HumanEval, quiet palindrome.
- Gains: whole decode +27.4 % (142.2 -> 181.2); full period +4.3 % (169.3 -> 176.5; my approximation +4.2 %, 10 / 12
  prompts faster).
- Check 0: 339 / 306 -> 66 / 65 ms per pass. Later checks ~59 -> ~57 ms. Lifetime peak 42.9 GB in both arms.
- FIRSTCHK chain: NewIdeas priced it (07:0x: 15.8 % of the decode wall); BigMoE's kill test showed a one-time wait with
  no host CPU (07:05); Leap named and fixed it (08:36).
  - The cause: IOGPU's wired collector unwires the 32 GB pools during the prefill, and check 0 re-wires +31.8 GB.
  - The fix: one mx.set_wired_limit(55.66 GB) after model load.
  - Exactness: verify-logits hash equal 43 / 43 in the exact env; tokens 3 / 3. So S4's gate on v1' carries over.
- Into S5 tonight (final_env unchanged + R57_WIRED=1). Against 200: x1.10 on whole decode.
- The same call applies to the EXACT default, the 12 GB loop (c1u16's check 0 is 229 ms median) and Breakthrough's
  engines.
- CLOSED (BigMoE 14:1x): the volatile tier (R57_VTIER) is the extension's own volatile newBuffer, not MLX's allocator. So
  set_wired_limit doesn't wire it, and it stays purgeable.
- PATH TO 200 on whole decode (NewIdeas 10:3x; CPU-only from S6W's W passes; scripts/tail_w.py, data/firstcheck/tail_w.txt): x1.104 needed.
  - Where the wall goes now:
    - checks 90.8 %, drafts 9.0 %, host gaps 0.3 %;
    - inside checks: demand SSD busy 17.7 %, GPU waiting on misses 20.8 % (they overlap);
    - 64 % of checks accept all 16 rows;
    - the tail is light: checks over 1.5x the median are 1.0 % of the wall. No FIRSTCHK-class outlier is left.
    - High-miss checks (above the median miss count) run 61.5 vs 50.1 ms, so misses are the biggest single part left.
  - Levers, each a few %:
    1. NW v3 with counters off: +5.8 % whole decode on S1b's base (cycle -3.6 ms). Lossy. Quality passed in S4 (v1 with
       NW + counters). BETWEEN on its +8 % full-period line. Shipping a BETWEEN lossy lever for the 200 goal is
       Kamen's call, not a re-litigation of the line.
    2. GDN state checkpoints (exact-safe): partial accepts are 36 % of checks, at ~3.3 ms saved each. About -1.2 ms per
       cycle, ~+2 % [P].
    3. C 288: OUT on memory (CORRECTED 13:3x). I first guessed ~44.3 GB. BigMoE's sat_report shows C 224 -> 256 raised
       the lifetime peak by 3.4-4.2 GB (stackb 39.4 -> 42.8, stack 38.5 -> 42.7). So C 288 lands at ~46-47 GB [P],
       over the 45.0 line. C 272 would sit at ~44.6-45.0 GB for maybe +1 %: not worth the margin.
    4. The draft (9 % of the wall): draft-ahead was re-priced on 11-06 as a loss (GPU-bound check, idle <= 2.5 ms;
       break-even at >= 73 % hits). Not a lever without a cheaper drafter.
  - Product: 181.2 x 1.058 (NW) x 1.02 (GDN checkpoints) ~ 196 [P]. 200 needs one more lever beyond these: drafter
    acceptance (kept per check, 11.0 of 16) or Leap's dense-kernel roadmap.
  - Tonight's honest numbers: ~181 at S5, ~192 if NW ships (Kamen's call). After the freeze: GDN checkpoints + NW
    ~196, and 200 needs a further lever.
  - S5 should also report MBPP12. Whole decode on HumanEval includes check 0's 16 copied tokens, which flatters it.
PREVIOUS: NEWEST 45 GB CODE NUMBERS, Sat 04:5x (re-derived by NewIdeas from BigMoE's raw S1 files: scripts/s1_fullperiod.py,
data/s1/). C 256, quiet, 12 HumanEval, lifetime peak 43.9 GB:
- v1 as written for S4 = stackb + S2 + KEEP + NW v3 (final_env 1403cd6, snapshot 4ef3734). 180.3 tok/s on the screen's
  steady metric [K, BigMoE f9c9fff]. On all wall time: 165.4 full-period steady and 142.1 whole decode [K, re-derived].
- v1 without NW = stackb + S2 + KEEP (S1's B arm): 165.3 steady, 158.5 full-period steady, 137.8 whole decode.
- METRIC NOTE. The screen's steady tok/s is tpsw = kept / (c2 - c0) over checks >= 2 (r57_screen_s4.py:1357 / 1447).
  It leaves out the time from c2 to the next c0:
  - B: ~2.35 ms per check (host work, plus the q1c readback's wait for the commit replay), so 'steady' reads ~4 % above
    full-period throughput;
  - NW: ~4.75 ms per check. Its counter graph (a sort plus 8 sums per layer call, ref2.py:935) is not an ancestor of
    the logits, so all of it runs at the line-1355 readback, after c2 (BigMoE 05:1x; my first reading, a wait on the
    replay, was wrong: both arms waited on it via q1c).
  Release and S5 / CERT numbers: full-period steady plus whole decode, never tpsw alone.
  - RULE (orchestrator 05:0x):
    - Headlines and the paper use whole decode (n / decode wall, prefill reported separately).
    - Full-period steady is the steady-state figure. tpsw is information only.
    - New speed lines are pre-registered on full period, with whole decode reported beside them.
    - Same-metric table: findings/2026-11-07_0500_steady-metric-overstates-whole-decode.md.
  - C 256 vs C 224 (stackb): +3.4 % steady but +0.1 % on whole decode (123.5 vs 123.4).
    - For the 24 GB line, C 224 is worth ~3 GB less RAM for the same whole-decode speed. No action tonight.
- S1 (NW v3) on full period: +4.3 % (whole decode +3.1 %). That is BETWEEN on its lines (PASS >= +8 %, KILL < +3 %),
  not the registered +9.0 %. Re-test asked of BigMoE (04:5x): NW with the per-check readback off, B / NW / NW / B at
  C 256, scored on full period. >= +8 % keeps NW in v1; otherwise v1 = stackb + S2 + KEEP, gated in the second S4 slot.
- Against 200: x1.41 on whole decode (142.1); x1.21 on full period (165.4), x1.26 without NW (158.5). Headlines use whole decode from now on (orchestrator 05:0x; prefill separate; full-period steady = the steady-state figure; tpsw = information only).
- Runner 49 (EvalSuite bb7094d, 3581be6; stackb at C 224), re-derived:
  - Same scores vs REF2-T: FAIL at the bar, HumanEval+ 90.0 vs 95.0 % (-3 of 60, McNemar 0 / 3, p 0.25).
  - Same scores vs NAX6: PASS (90.0 vs 91.7, p 1.0).
  - B top-up PASS (0.30 blunders / 10K); long code PASS (41 humanevalt rows, 0.18 / 10K); MBPP PASS (76.0 vs 79.0,
    2 / 5, p 0.4531).
  - S4 compares against NAX6-N (the orchestrator's option (b), 03:59; Kamen may reverse) and runs now (runner 54
    -> results/S4_DONE).
- S4 RESULT Sat 05:38: PASS (EvalSuite a1aac9b; v1 with NW v3 and its counters, snapshot 4ef3734). Re-derived from
  the counts:
  - vs NAX6-N, both orders: HumanEval+ 88.3 vs 91.7 (53 vs 55 of 60, McNemar 0 / 2, p 0.5); math 86 vs 88 (43 vs 44 of
    50, p 1.0); long code 0 blunders in 27,488 tokens (21 rows).
  - vs REF2-T (information): HumanEval+ 88.3 vs 95.0 (-4 of 60), which would FAIL the original line. Kamen may
    reverse (b).
  - Runner 49's stackb at C 224 scored 90.0 on the same 60. The two orders agree on only 44 of 110 rows token for
    token, but score the same per set.
  - S1b (BigMoE) runs after S4_DONE and decides NW. If NW leaves, v1' = stackb + S2 + KEEP gets its own S4 by 16:00.
- S1b RESULT Sat 05:47 (BigMoE data/s1b/s1b_report.txt; re-derived): BETWEEN, so NW v3 leaves v1.
  - NW with counters off vs B, both without QCNT1C / RELAX_LOG (untimed gap now ~0.15 ms per check in both):
    - full period +6.9 % (exact c0: 164.2 -> 175.6; my approximation +6.7 %);
    - whole decode +5.8 % (141.2 -> 149.4);
    - median cycle 65.2 -> 61.6 ms (-5.5 %);
    - kept per check (checks >= 2) 9.92 -> 10.13, within pass-to-pass noise (B's two passes: 9.63 / 10.21).
  - The 5th arm (counters read after the clock, information) ran 182.6 full period. Its cycle (61.5 ms) equals the
    counters-off passes; its +4 % is acceptance (10.52 kept per check), one pass. Not a counters-off bug.
  - nwcnt_check on S4's ref2 (ddbc0621): counters ON and OFF both PASS. Late groups at counters-off timing: 20.0 per
    check vs 18.0 (moot now).
  - Method note: pass-to-pass acceptance varies up to ~6 % on these 12 prompts, so lossy A/Bs report the median cycle
    ratio beside tok/s.
- v1' (written 05:49, BigMoE dd7ad0f, snapshot 03c7782) = stackb + S2 + KEEP at C 256: 141.2 tok/s whole decode,
  164.2 full period, lifetime peak 43.8 GB (S1b's B arm; S1's 137.5 carried the q1c readback).
  - It needs its own S4 by 16:00. EvalSuite has been told (05:5x); runner 53 yields to it when it sees final_env newer
    than S4_DONE.
  - NW v3 becomes a post-release lever: -3.6 ms per cycle, lossy (code map PASS, chat FAIL).
  - Against 200: x1.42 on whole decode, x1.22 on full period.
- S4 on v1' RESULT Sat 06:48: PASS (EvalSuite 013a4a6), re-derived from the counts:
  - forward: HumanEval+ 53 vs 55 of 60 (p 0.5), math 42 vs 44 of 50 (p 0.5);
  - reverse: HumanEval+ 54 vs 55, math 43 vs 44 (p 1.0);
  - long code 0 blunders in 27,014 tokens.
  - Every discordant pair in S4 (both runs) and runner 49 goes against the FAST arm: small, consistent losses. None
    is significant. Worth a sentence in Paper B.
  - S5 (BigMoE, evening: whole decode + full period, c0 recorded) may go. FREEZE ~20:00. CERT runner 55 from Sun 01:30.
- NEW LEAD, Sat 07:0x (NewIdeas; scripts/firstcheck_price.py, data/firstcheck/): FIRSTCHK, the first check of each
  prompt, is the biggest single lever on the whole-decode headline.
  - What it costs: in v1' (S1b's B arm, 24 prompt runs) check 0 takes 306 ms vs 58 ms for later checks. That is 15.8 % of
    the decode wall (median per prompt).
  - Where it sits: graph build 287 vs 53 ms; eval wait 20 vs 5 ms; draft 13 vs 5 ms.
  - Not the SSD: check 0 has FEWER misses than later checks (21 vs 31), demand reads 13 vs 9 ms. The prefill already
    warmed the pools.
  - Ceiling: if check 0 cost a later check, whole decode goes 141.0 -> 170.7 (+21 %) [P]; with check 1 too, +22 %.
  - Ruled out from existing data:
    - A6 keep (no end-of-prompt mx.clear_cache): on 11-04 (cmb1c -> cmb2, C 160, exact) the check-0 excess was
      unchanged (183.1 -> 182.5 ms code, 196.4 -> 196.1 other) and whole decode -0.3 % (code). Its 11-03 win
      (465-1681 -> 130-211 ms) was on long agent prefills.
    - REF2's per-weight caches: KEEP keeps them across prompts too (remove() returns early).
    - New-length kernel builds: R-80 KILL (+1.4 to +8.5 ms).
  - Prior art in our records: K-80 / the 11-03 A6 split saw "a one-time non-SSD cost in the first check's graph build"
    (28-73 % of the excess after keep) on long prefills. Nobody fixed it; the steady metric hid it from every A/B since.
  - Cheapest kill test (~3 min GPU, shared slot, a copy of the s4b screen, 3 HumanEval prompts in one process):
    - Run R57_A6=rep: the first check twice, rolled back in between; copy 1 - copy 2 = the one-time cost.
    - Record time.thread_time() around build and wait of checks 0, 0', 1 and 2.
    - KILL if copy 2 is also >= 200 ms and host CPU is <= 20 ms of the excess. Then the cost is the first check's
      content, not a one-time cost.
    - Otherwise the split names the piece: host (Python setup, compile retrace) vs GPU (first use).
  - Placement: not before S5, the freeze or CERT. If it is host-side setup that can move or vanish, it is an exact-safe
    v2 item. Owner of the screen: BigMoE. Kernels: Leap.
PREVIOUS, Sat 01:4x (steady metric, ~4 % above full period; see the METRIC NOTE): 164.2 tok/s [K, BigMoE c02f807, re-derived]. stackb + S2 + REF2_KEEP_CACHES
at C 256, quiet A/B on 12 HumanEval: +10.1 % vs 149.1 for stackb + S2 in the same run; cycle 75.8 -> 66.9 ms; lifetime
peak 43.1 GB. KEEP passed its EXACT proof (3 / 3 prompts, tokens equal), so it adds no quality risk. Gate: S4 on v1 =
(stackb | stack) + S2 + KEEP. Against 200: x1.22 to go. S1 (NW v3; its map PASSED at both pools) is next, at most
~7 ms of the 66.9 ms cycle (x1.12), so ~175-185 [P] if it lands at its ceiling.
TONIGHT'S MEASURED 45 GB CODE NUMBER, Sat 00:2x (pending the same-scores gate, runner 49): 144.9 tok/s [K, BigMoE
data/sat/sat_report.txt, 7bc69bb; re-derived by NewIdeas]. This is stackb at C 256 (the stack + B's shipped rule +
R57_COPY=g2), quiet, 12 HumanEval, palindrome passes 145.5 / 144.3, kept 11.34 per check (cycle ~78 ms), lifetime peak 42.8 GB.
- Arms at C 224 (geo steady over 12):
  - qall 99.1.
  - stack 132.6: +33.8 % vs qall, PASS.
  - stackb 140.1: +5.6 % vs the stack (140.1 / 132.6 = x1.057), PASS. Relaxed accepts were 0, so the gain is g2's copy
    (the draft changed at 422 positions; HumanEval flatters copying).
  - stackr 145.3 / stackrm 144.6: +9.5 / +9.0 %, SPEED ONLY (C's failed form). Adding g2 to stackr gained nothing.
- C 256: stackb 144.9 (+3.4 %, 42.8 GB, PASS); the stack 136.5 (+2.9 %, KILL by its +3 % line).
- T16: MBPP12 BETWEEN (mbpp/319 flipped in pair 2; tokens x1.088), so no stackt.
- Never-wait ceiling PASS: cold16 - warm16 is a median 18.5 ms per check over 58 samples. That is the most any SSD-wait
  lever can win.
- The gate: runner 49 gates code_env (stackb) at its own C (160 or 192, by free memory). QOM skips more at a smaller C, so
  a PASS there is conservative for C 256, but CERT should run the release C.
- Against Kamen's publish trigger: 144.9 vs 250-330 (x1.7-2.3 to go). Against Saturday's 200: x1.38 to go, i.e. a ~57 ms
  cycle at kept 11.34.
VOID Fri 16:0x (CD16 is KILL on quality, see item 6): the 157.9 below includes CD16, so it is not a publishable number.
The valid line stays 113.9 [K] (FAST-NAX, pending the gate); tonight's palindrome measures the stack without CD16 (~128-130 [P]).
WAS: TODAY'S MEASURED 45 GB CODE NUMBER, UPDATED Fri 15:5x (pending tonight's gate): 157.9 tok/s [K, BigMoE
data/fday/fday_report.txt, b3b1e68; re-derived by NewIdeas]. This is the stack, quiet, 12 HumanEval, C 224, palindrome
passes 157.2 / 158.1, kept 12.58 per check, check 68.8 ms, cycle ~80 ms. The stack is the FAST all-in + FAST-NAX + DN-KS
+ lazy GDN commit + D7BNAX_32 + CD16 + att_single (no relax gate).
- That is +38.6 % over this morning's 113.9. Most of it is K: 10.36 -> 12.58 (x1.21, CD16), and the cycle went ~91 -> ~80.
- CD16 accepts comment and docstring text wholesale, and HumanEval's docstrings copy the prompt, so HumanEval flatters it
  more than average code. Its quality gate is tonight (EvalSuite's MBPP decision 4 + the stack same-scores); MBPP's
  engine speed is GOAL's MBPP12 pair (re-run ~16:00).
The earlier line, kept for the record:
TODAY'S MEASURED 45 GB CODE NUMBER (pending the same-scores gate): 113.9 tok/s [K, BigMoE data/fnx/fnx_report.txt;
re-derived by NewIdeas Fri 08:1x]. That is the FAST all-in + FAST-NAX on 12 HumanEval, C 224, the steady-state geo mean,
A-B-B-A (passes 112.7 / 115.0), with no freezes.
- Against qall's 97.5 (passes 98.1 / 96.9): +16.8 % (+14.9 to +18.7 % over the pass pairings). Cycle 109-110 -> 89-90 ms [K].
- Kept 10.36 vs 10.55 per check (-1.8 %): BETWEEN by the pre-registered line (kept within 1 %). It joins tonight's stack
  (the orchestrator's call: the kept drop is NAX6-N's own numerics, and the tok/s includes it). Its gate is the
  same-scores run.
- HumanEval is flattered by prompt copying.
- Against Kamen's publish trigger (lossy, code): 45 GB 113.9 [K] (the stack without CD16 ~128-130 [P], measured tonight) vs
  250-330 (x2.0-2.9 to go);
  12 GB code-valid ~36 [P] vs 90-130 (x2.5-3.6 to go). The 12 GB verified-greedy loop (not exact) is 18.4-18.9
  (not 40.8; see 12 GB LINE).

200 STACK (NewIdeas Fri 21:0x; Kamen Fri ~21:0x via the orchestrator: >= 200 tok/s lossy at 45 GB on code by Saturday
evening, measured and gated, workload named; then publish). Built levers that can still stack, on 12 HumanEval at C 224:

| lever | switch | factor on HE-12 | quality status | needs tonight |
|---|---|---|---|---|
| the stack (base) | FAST-NAX + DN-KS + lazy + D7BNAX_32 + att_single | ~131-137 tok/s [P: 157.9 with CD16 / CD16's x1.15-1.21] | gate tonight | palindrome arm (armed) |
| relax v2 as armed (stackr) | R57_RELAX_DELTA=0.5, R57_RELAX_SPAN=1, NO R57_RELAX_GATE | x1.08-1.15 [P] | CORRECTED Fri 21:2x (the orchestrator; verified in r57_screen_fast.py:161-162 and RESULT_relax_2026-11-05.md:11): ungated DELTA is R1 on CODE tokens too, i.e. C's form, which FAILED decision 4 (HumanEval+ 90.0 vs 95.0). B passed only because its relax touched comment rows alone. My earlier "RELAX B passed" on this row was wrong | SPEED-ONLY tonight, excluded from code_env |
| B's shipped rule (stackb) | R57_RELAX_SPAN=1 + R57_RELAX_SPAN_RULE=R2:2:0.25, no DELTA | x1.02-1.03 [K-based: B's comment rule +1.8 % kept on HE+] | B's rule passed decision 4 (under B's expert gate, not QOM) | if BigMoE can dry-run it by 00:30; line PASS >= stack x 1.02, KILL < x0.98; gate-eligible |
| MATCH + 32-row checks | R57_COPY=g2, R57_ROWS=32 | was x1.10-1.20 [P] | - | DROPPED Fri 21:0x (Leap: a fast 32-row one-call verify is several days across two repos, ~+20 ms per check if built) |
| guarded CD16: G16 / D16 / C16 (GOAL's INERT) | R57_INERT=G16 / D16 / C16 | - | OUT (GOAL Fri 21:10, from the failed run's logs: each keeps one of the two failing kinds wholesale, docstrings or comments) | not counted |
| T16 (GOAL; READY Fri 21:18, build_1106 GUARD.md, inert_span.py sha1 0ca24e0d8db5; CPU self-tests 38 + 26 OK): comment + docstring rows kept only if the 125B ranks the draft top-3 with p >= 0.1 p1 | R57_INERT=T16 | HE tok/s x1.02-1.04 (halved from replay x1.04-1.06), MBPP x1.08-1.12 [P, GOAL] | MBPP12 line: PASS verdicts equal on >= 11 / 12 in each pass pair, no cap run where the control ended, tokens <= 1.05; KILL >= 2 flips, or such a cap run, or tokens >= 1.15; VOID if anything kept outside the rule | tonight's MBPP12 pair, then stackt = stack + T16 + MATCH-16 (~136-140 [P]); exclusive with relax v2 |
| MATCH at 16 rows (exact copy check) | R57_COPY=g2 | x1.03-1.04 [K walk: K x1.038 on HE+] | exact verification | BigMoE: confirm it runs under R57_FAST (32 rows cannot: the R == 16 assert, line 246) |
| Leap's engine-gap hunt | - | x1.03-1.05 [K-based, Leap 19f6789] | numerics-only | DONE Fri 21:2x: no big gap. The "dense ~33-36" was synced per-piece accounting; in situ unsynced dense at 16 rows is ~25 ms, GPU-bound. The real glue gap over the 19.6 ms chain is <= 2-4 ms. Open lead: does the GPU idle between layers while the expert path blocks the forward (~25 ms)? Leap's 5-min C 224 run tonight |
| S2 small built items (the orchestrator's sweep): D7BNAX_32=2 (-0.75 ms per cycle; code tokens identical to =1), DSUB (-1.0 ms per propose), SVH (-0.79 ms per check), plus Leap's glue fix | per item | x1.04-1.08 [P: -3..-6 ms] | kernel-only / bitwise on code (D7BNAX_32=2); SVH is FAST | DSUB and SVH live only in ~/Leap/data/r57_session_vocab.diff; they need wiring Saturday |
| C 256 | C | x1.03-1.05 [P] | none needed | a REGULAR arm tonight (the orchestrator Fri 21:1x): peak ~43.2 GB [P] = 38.9 at C 224 + 32 x 0.134, inside the 45 GB lifetime line; needs NO-SWAP >= 47.2 GB at grant, else C 224; line PASS >= +3 % vs the same arm at C 224 AND lifetime peak <= 45.0 GB |

- RESULT Sat 00:2x: 144.9 [K] (stackb at C 256). What can still stack by Saturday evening [P]:
  - S2: -1.3..-3.6 ms per cycle in Leap's loop, i.e. x1.02-1.05 on the ~78 ms 12-HE cycle.
  - S1 NW v3: at most the ~7 ms of gate service per check, x1.10. The never-wait ceiling is 18.5 ms.
  - S3: parked.
  - Total: ~147-167 on HE-12, gated.
  - 200 needs ~21.6 ms off the cycle: all of the 18.5 ms never-wait ceiling plus S2. NW v3 targets ~7 of those 18.5 ms.
    So 200 is out of reach tonight; Saturday's honest number is ~150-165 if S1 passes.
- Stacked product [P], REVISED Fri 21:2x (no MATCH-32, no G16 / D16, gap x1.03-1.05): base 131-137 x (relax v2
  1.08-1.15 OR T16 1.04) x MATCH-16 1.03 x Leap's items 1.02-1.06 x C 256 1.03-1.05 x gap 1.03-1.05 = ~151-191 on
  HumanEval-12. 200 is OUT OF REACH on priced factors. Only the open GPU-idle lead or an unpriced find could close it.
  MBPP ~98-110 [P].
- Where the check's time is now [K, Leap 19f6789]: all-resident 16-row check on FAST-NAX at C 224 = 51 ms = dense ~25 +
  expert path ~26 (bytes-bound, ~180 GB/s cold). Real blocks add 10-27 ms of miss spins and commit (check + commit
  61-78.5). So the levers left are the expert path's bytes / streaming rate and the waits; C 256 measures the waits
  tonight.
- Not stackable (KILLed today): CD16 (MBPP quality), never-wait and SPLIT-NAX (engine). Not built: bigger drafters.
- ARMED Fri 21:2x (BigMoE ee1faec, ticket bas0fbhp9; RE-TIMED Fri 23:3x: starts now, window open; dry-run end to end):
  MATCH-16 2-prompt check -> T16 MBPP12 pair (GOAL's line in sat_mbpp_verdicts.py) -> palindrome qall / stack / stackr /
  stackrm (both SPEED ONLY) / stackt / stackb / qallm / mirror (13 passes) -> C 256 stack + winner -> ceiling. ~45 of
  65 min. code_env = the fastest PASSING of {stackt, stackb}, else the stack. Reports: data/sat/sat_report.txt; envs in
  data/stack/.
- Tonight's ticket, reshaped (asked of BigMoE, Fri 21:0x; revised 21:1x):
  (1) T16 ONLY (NewIdeas Fri 21:1x, to BigMoE): if GOAL's module + CPU test land by 23:45, the MBPP12 pairs mq / mt /
      mtb / mqb on GOAL's line (~7 min). Otherwise NO INERT arm tonight; no fallback to G16 / D16 (unsafe).
  (2) The palindrome on 12 HumanEval: qall / stack / stackr / stackrm (stackr + MATCH-16, if it runs under FAST) /
      stackt (stack + T16 + MATCH-16, only if T16 passed). Lines: PASS >= stack x 1.03, KILL < stack x 0.97.
  (3) The stack and the winning arm again at C 256 (NO-SWAP >= 47.2 GB at grant; line PASS >= +3 % vs C 224 AND lifetime
      peak <= 45.0 GB).
  (4) The CODE profile EvalSuite gates is the fastest PASSING arm among {stackt, stackb}, else the stack (stackr /
      stackrm are speed-only). If T16 is in it, tonight's MBPP
      check is T16's MBPP decision 4.

MEASURED (quality measured; speed measured or priced):
1. RELAX ship, B on code + E on chat (45). Speed [K, EvalSuite RESULT 8797f54, C 160, paired]:
   - B: tok/s x1.49 HumanEval+, x1.32 math, x1.54 agent;
   - E: tok/s x1.46 IFEval.
> (a quote from the project owner, kept private)
   ROUTED Fri 08:0x (the Thu 08:5x rule, plus the orchestrator's 08:0x agent-turn scope). B's drift re-check: PASS [K,
   EvalSuite runner 45: pooled long-row KLD q4 / q1 0.82, line <= 1.5, 16 rows]. E8: KILL [K: IFEval 84.0 vs REF2-T's
   90.0 %, -6.0 pt, McNemar 0 / 6, p 0.03; E without HC8 scored 87.0, PASS].
   - CODE AND MATH ANSWERS: FAST ships B with HC8 (tok/s x1.49 HumanEval+, x1.32 math).
   - AGENT TURNS (tool calls that write files): not on B. On 17 long agent writes B made 20 blunders (22.6 per 10K
     tokens, 16 of them inside the written code); 6.7 % of its tokens left REF2-T's top token, against 1.2-1.4 % on
     HumanEval. The gate causes it, not HC8 (arm A: 2 in 732 agent tokens; exact: 0). Agent turns run EXACT until plain
     FAST passes the agent bar (EvalSuite's next slot). Agent turns are 92-94 % prefill (AT14), so B gave them only ~+9 %
     per turn.
     AGENTFAST FAIL [K, EvalSuite 11:24, day48_runner.log]: plain FAST (no relax gate) on NAX6-N, 17 long agent turns x 2
     orders: 85 blunders in 19,749 tokens (~43 per 10K, worse than B's 22.6; 75 inside the written code); 8.1 % of
     tokens off the reference's top token. Short edits are clean; the long writes carry it (write12-14 at 16-20 %
     deviation, up to 16 blunders per turn). The two orders are token-identical on only 3 of 17 turns, several diverging
     by token 1-6. So agent turns stay EXACT. A cause bisect is ordered (GPU ORDER).
     BISECT [K, EvalSuite 12:43, results/agentbisect_report.txt; write12-14, one lever off per arm, vs base fwd 21
     blunders]: QOM-KEEP off 4 (19 %): it carries it. DRAFT17's bonus row off 9 (43 %): partial. HC8 off 20, FAST-NAX
     base off 20, SMALL-N off 21: not the cause. The sharp test FAILS: with QOM-KEEP off the two orders still differ
     (write12 @4, write13 @64, write14 @4), and deviations stay at 6.9-7.8 % (base 10.8-11.0 %), mostly near-ties.
     - So something else is history-dependent too. BigMoE: the same FAST env and order in two fresh processes gives
       different chat tokens. EvalSuite: with QOM off, drafted row counts differ between orders before the tokens do.
       The likely cause is drafts carrying history into a row-count-variant FAST kernel.
     - Agent turns stay EXACT: no FAST form reached the bar's 0 blunders.
     - For code answers, QOM is the suspect on long outputs. Tonight's drift top-up decides; its fail arm is the CODE
       profile with QOM off (pre-registered here).
   - CHAT: the EXACT launch. Chat's FAST is the tested E without HC8 (HC8 code-only): (i) a per-launch profile with
     R57_HC8 off, or (ii) BigMoE's request-boundary swap, bit-checked. "Chat without R1" (HC8 on, no relax gate) is
     untested, so it does not ship.
   - On record: the 10 HumanEval-type long rows alone grow 4-6x from q1 to q4, at tiny KLD (q4 median 0.010, max
     0.028). The pooled ratio is carried by the 6 agentlong rows. EvalSuite re-checks >= 20 long code rows on a
     pre-registered code-only line; if that fails, the code part rolls back before anything is published.
   - Every published code tok/s names this configuration, with the agent-turn limit stated beside it. Profile selection
     is specced with BigMoE: an agent turn (tools or tool messages in the request) never gets a relax gate. v1 =
     per-launch profiles (BigMoE 08:1x: each launch runs exactly its tested config; the request-boundary swap is
     unmeasured, likely seconds per swap, so not for v1).
2. Split form, split 0.99 / 0.9 on QOM-KEEP (45). Quality [K]: map code PASS, 0.92 / 1.31 % (HumanEval-style code
   windows); VERDICT2 non-tie 0.51 / 0.78 % PASS. Chat control [K, Fri 01:4x, all 36 windows]: split 0.99 IFEval 6.01 /
   multi 5.15 % (BETWEEN / BETWEEN), split 0.9 8.10 / 6.73 % (BETWEEN / KILL), twin qom_0_live_ball 4.03 / 3.26 % (PASS):
   the split form is a code-only profile (chat stays on the twin or E). The code rows equal Thursday's +code runs row
   for row. Split 0.9 has 44 flips against a line of 45; split 0.99 (31) and Fable C's K8 (40) have more margin.
   Speed [P, Fable C's audit model]: expert tiles x0.63 / x0.51 -> the 22-27 ms expert
   term on HumanEval code saves 9-14 ms of ~90 = +11-19 % raw, ~+5-9 % after the audit's halving. Engine A/B + R8:
   BigMoE Fri. Leap's FLAT-2 found R57_QOM_SPLIT inert under OOO-1C (its QOM-SPLIT-1C fix: GPU check queued).
   ENGINE A/B [K, BigMoE 725ed10, C 224, one-call path, 12 HumanEval]: NOT PASS (line: >= +8 % AND kept within 1 %).
   The split engaged: groups x0.625 / x0.494. Clean first passes: check + commit 98.8 -> 92.9 / 93.7 ms (-6 / -5 %).
   Kept per check fell 4.2 / 1.7 %, so kept per ms nets ~0 to +1.5 %. The middle passes were heat-disturbed (-21 %).
   PARKED Fri 04:2x (NewIdeas' call; the orchestrator asked): R57_QOM_SPLIT leaves the FAST all-in. Re-test only
   split 0.99 vs qall, after the FLAT cuts bring the cycle to <= ~80 ms. The tile price above was ~2-3x too high:
   37-51 % fewer groups cut only 5-6 ms. Leap's step 0 puts the experts at 57 ms of GPU at 16 rows (11.1 at 2 rows),
   so their time is mostly per-row work, not per-group weight reads.
   RE-TEST TRIGGER MOVED Fri 08:0x (Leap's part B): that per-row cost belongs to 3c-N's kernel (+3.25 ms per row, ~2x
   its read floor at 16 rows). FAST-NAX's time follows its bytes (+0.98 per row, at or under the floor), and fewer
   groups means fewer bytes. So re-test split 0.99 vs qall ON FAST-NAX (Leap's QOM-SPLIT-1C + NH-1C form) after
   BigMoE's FAST-NAX A/B, not after the FLAT cuts.
   RE-TEST ORDERED Fri 09:3x (NewIdeas' call; the orchestrator asked): SPLIT-NAX = stack + R57_QOM_SPLIT=0.99 vs the
   stack, one extra arm in BigMoE's Sat 00:30 palindrome (~7 min, quiet, 12 HumanEval, C 224).
   - Why now: Leap's NAX1CQ-TUNE (187b834) found no tunable winner for FAST-NAX's expert one-call. It streams ~180 GB/s
     cold at ~100-140 groups per call, so on FAST-NAX the experts are a bytes (groups) lever.
   - Line, pre-registered now: net tok/s vs the stack PASS >= +8 %, KILL < +3 %, else BETWEEN. Kept is recorded with no
     separate clause: tok/s carries it, as for FAST-NAX.
   - Prediction [P]: +2..+8 % net. That is a ~30 % expert-byte cut on a ~20-25 ms expert term, minus 2-4 % kept, and a
     fixed per-layer part that does not shrink.
   - Speed only tonight: it does not enter tonight's gated stack. If PASS, its own same-scores run (code + math on the
     CODE profile; the map never checked math) takes EvalSuite's next slot. No share-lock proof of concept today: rough
     numbers would not decide anything.
   SPLIT-NAX KILL [K, BigMoE Fri 15:4x, data/fday]: -1.3 % vs the stack (155.8 vs 157.9 tok/s), kept -5.7 % (11.87 vs
   12.58). Fewer groups did not pay on FAST-NAX either: the kept loss alone is about what the bytes save. The split
   family is closed on this stack. Deeper group cuts (split 0.9, K8, MASSB-0.5) would lose more kept.
3. DRAFT17 (45 / 24 / 12): in the stack. Code bytes [P] x0.706 at 24 GB, x0.742 at 12 GB (HumanEval-style windows);
   quality PASS [K].

PRE-REGISTERED, NOT RUN (by expected gain on 45 GB code; lines in QUEUE Thu 19:0x unless noted):
4. FAR never-wait nw1 (45). [P] median check 88 -> 72-76 ms on HumanEval code = +16-22 %. Map arm, then the engine form.
   MAP PASS [K, Breakthrough 13:46, SCOREBOARD 1d46c51; data/dm_s7/results_nw/]: code is clean at every pool. nw1 at 208
   slots 0.68 % [non-tie 0.27] vs its twin 0.74 [0.33]; at the published pool 0.59 [0.18]; at 266 slots 0.65 [0.24]. The
   stack form, split 0.99 + nw1, scores 1.07 [0.60]: PASS. Chat passes at the published pool but fails at 208 slots;
   that is pool-driven (the 208 pool alone sits at the chat lines), so chat needs a pool at or above the published one.
   - Engine: BigMoE's r57 side is built, and its NW bit check 4 is clean with counters (abb43e0, c2e6a53).
   - Next (NewIdeas Fri 13:4x): one "stack + NW" arm in tonight's palindrome, speed only and outside the gated stack.
     Pre-registered line: net tok/s vs the stack PASS >= +8 %, KILL < +3 %. [P] the misses are ~13.5-15 ms per check,
     so +10-15 %.
   ENGINE KILL [K, BigMoE Fri 15:4x, data/fday, C 224, quiet, 12 HumanEval palindrome]: -2.8 % vs the stack (153.5 vs
   157.9 tok/s), kept +4.6 %. Per check: check 68.8 -> 75.3 ms, gate service 14.0 -> 15.6, misses 37 -> 39, stalled layer
   calls 17 -> 17. The NW counters show 24 late groups and 57 late uses per check, with 0 stalls of its own.
   - My +10-15 % was wrong. I priced it from the 13.5-15 ms gate service, which is a spin bound that OOO already
     overlaps. NW did not cut the gate service (14.0 -> 15.6 ms) and added ~6.5 ms of late work. At C 224 there is
     little wait to remove. (CORRECTED Fri 21:2x, Leap: 'stalled layer calls' counts calls that issued any read, not
     waits; BigMoE's ext line 881.)
   - NW stays out of tonight's gate. The per-set NW gate (BigMoE ebf9bbd) stays in the code, unused.
   - If it and SPLIT-NAX both pass, ONE same-scores run covers stack + split 0.99 + NW (the map passed that
     combination).
5. MEASURED Fri 02:0x-02:13 [K, code windows, Fable C's VERDICT2 lines]: no gain beyond item 2. K8 PASS 1.19 / 0.69 %
   (all-row / non-tie) at x0.539 tiles; K6 1.49 / 0.97, A 1.40 / 0.87, A' 1.40 / 0.81, B 1.87 / 1.21: KILL. K4 (probe,
   03:41) KILL 2.02 / 1.42. By Fable C's rule (the passing arm with the fewest tiles) the code preset stays split 0.9 (x0.508);
   the frontier at the bar is x0.51-0.54 of the twin's tiles. (Was: K6 [G] +14-20 % raw.)
6. GOAL INERT CD16 (45). [P, lexer replay] S 0.84 = tok/s x1.15 on the 12 HumanEval keys. Flattered by comment and
   KILL ON QUALITY [K, BigMoE Fri 16:0x, GOAL's MBPP12 pair, C 224, CAP 384, data/fday/mbpp12_verdicts.json]: qall passes
   11/12 MBPP tests in both passes, qall + CD16 8/12 in both: 3 flips per pass pair against GOAL's line of 2. Speed passed
   (K x1.449, 71.3 -> 88.2 tok/s, +23.7 %). The cause is visible in the answers: CD16 keeps comment rows whatever the 125B
   says, so the drafter loops in comments until the answer is spent (mbpp/16) or writes different code after its own
   comments (319). DROPPED from tonight's stack (NewIdeas Fri 16:0x, on BigMoE's recommendation). GOAL's milder forms
   (G16, then D16) wait for a later night.
   docstring copying; C16 / G16 / D16 are the milder forms. MBPP has no engine set yet.
7. FULL comment-free code mode (45). Line tok/s >= +8 % (HumanEval-style; a product choice, Kamen's yes). FULL's copy
   probe prices wide copy passes; their gain is HumanEval-specific (copied rows are 61-68 % of HE answer tokens).
8. Fable A xs input-sparsity arms (45). [P] 5-7 ms of the 22-27 ms expert term on HumanEval code = +6-8 %, if code holds
   beyond the split form.
9. Fable B FAST-TAX switches (45). [P, measured pieces] -2.4 to -2.9 ms per check = +3-4 % (+5-6 % at split rows),
   HumanEval code. Lines pending its INSTALL.md.

MEASUREMENTS THAT GATE THE PLAN (no gain themselves):
- ZERO FLAT-2 (45): MEASURED Fri 01:33 by Leap [K, C 224, quiet]: KILL. FLAT 53.7 ms (PASS <= 30, KILL > 36); per added
  row only the experts grow, +3.0 ms per row (split inert). Today on HumanEval code: 98.5 ms per cycle at 9.38 kept =
  95 tok/s [K, ZERO's C 208 read].
  What 250 tok/s on HumanEval code needs [P, arithmetic on these parts, Fri 02:3x]: 4.0 ms per kept token.
  - At FLAT 53.7, a 16-row check with every draft kept (K 17) gives 17 / 101.9 ms = 167 tok/s. Kept-rate items alone
    cannot reach the line.
  - FLAT 25 with the split inert needs K 18.3 (more than 17). The split live at FLAT 53.7 needs K 19.6.
  - All three together reach it: FLAT <= ~25 ms, the split live in the engine (~1.5 ms per row if the per-row cost
    follows tiles x0.51 [G]) and K >= 12.4 per 16-row check (today 9.38: the D98 drafter track's job).
  So the order for the 45 GB line is: fixed-cost cuts first (Leap's list from FLAT-2: lazy commit, the HC reads
  11.0 ms, GDN 15.7 ms, fewer syncs per layer), QOM-SPLIT-1C second, then kept-rate items (4, 6, 7). The +x figures
  above hold only on top of those.
  CORRECTION Fri 04:2x: the split live does NOT give ~1.5 ms per row. The engine A/B cut only 5-6 ms per check
  (item 2). At 3.0 ms per row (16 rows = 48 ms), 250 needs FLAT <= 8 / 12 / 20 ms at K 14 / 15 / 17 per 16-row
  check. A per-row cut has to come
  from the expert kernel's per-row work (Leap's step 0 part B bench prices it), not from fewer groups.
- 300 BUDGET LINE (45, HumanEval code). Kamen Fri 07:2x (via the orchestrator): 45 GB -> 300 tok/s is THE priority. The
  orchestrator's budget, re-derived by NewIdeas Fri 07:3x from the measured pieces: 300 tok/s = 3.33 ms per kept token,
  so K 14 per 16-row check needs a 46.7 ms cycle = dense 12 + experts 25 + draft 5 + SSD ~0, with 4.7 ms left over.

  Measured state on 12 HumanEval at C 224 (BigMoE's FAST-NAX A/B screens, data/fnx/nxn_224.txt + nxnb_224.txt; Leap's
  engine A/Bs and new-base kernel table, Fri 08:1x-08:57):

  | piece (ms per check) | FAST-NAX run [K] | + new base | budget | gap | next levers |
  |---|---|---|---|---|---|
  | dense GPU (flat in rows) | ~33-36 [K, probes] | ~25-28 [K, -8.2: DN-KS + att_single] | 12 | -13..-16 | Chained 16-row kernels now sum 19.6 (in_proj 4.1, HC reads 5.9, GDN step 2.1, out / o_proj 1.8, ...). Next: HC reads -1..-2, GDN step -1, expert id glue -1, in_proj ks 2 -0.5, router -0.5. 12 needs ~250 GB/s and ~1/3 of the kernels; Leap's realistic target is 16-20 |
  | misses (the GPU's spin on them) | ~13.5-15 [K] gate service (35-37 per check) | same | ~0 | NOT removable by NW | CORRECTED Fri 15:5x: never-wait's engine arm did not cut the gate service (14.0 -> 15.6 ms) and added ~6.5 ms (engine KILL). The service is mostly overlapped at C 224; the removable wait is small and unmeasured |
  | experts + commit + rest | ~28-32 [K, the check's remainder] | same | ~30 (25 + 4.7 left over) | ~0 | FAST-NAX ~25 at 16 rows all-resident [K] |
  | draft (the loop's) | 10.1 [K] | ~8.1 [K, D17-NAX -2.0] | 5 | -3 | QMM32 -0.44 per propose [K bench]; the small 17-row calls + concats -0.5..-0.8 |
  | cycle | 89.3-90.1 [K] | ~79-80 [P] | 46.7 | x0.59 | |
  | K per 16-row check | 10.36 [K] | 10.36 | 14 | x1.35 | the drafter track (D98) |

  - tok/s: 113.9 [K] -> ~128 [P] with the new base. CD16's x1.15 [P, lexer replay], if it passes its pre-flight, makes
    ~148. Tonight's palindrome measures it.
  - Warm 2-row FLAT (no SSD): 55.2 -> 44.3 ms with DN-KS + D17-NAX + att_single [K] (-10.9, -20 %). Each piece is FAST
    (bits move), so the same-scores run is their gate.
  - MEASURED Fri 15:4x, the stack WITH CD16 (KILLed since; not publishable) [K, BigMoE data/fday]: 157.9 tok/s at K 12.58 = a ~80 ms cycle (check
    68.8 + draft ~7 + ~4). 300 at K 12.58 needs a 41.9 ms cycle (-38 ms); 250 needs 50.3 (-29).
    - What is still priced: dense -13..-16, draft -2 and expert-kernel steps of a few ms. That is ~-17..-22 -> ~58-63
      ms, so ~200-217 tok/s at this K.
    - So the 250-330 band on HumanEval needs either an expert-bytes breakthrough (the group cuts failed: SPLIT-NAX KILL)
      or K well past 12.6. Never-wait and the split are both KILLed on this stack.
  - The two biggest gaps are now the dense pass (-13..-16) and the misses (-13.5..-15), then K (x1.35), then the draft.
  - One dependent sync point costs ~1.6 us, so the kernel count is worth only ~2-6 ms per check: the dense lever is the
    kernels' shapes, not their number.
  - At K 10.36 the budget cycle gives 222 tok/s, so 300 needs K >= 14 (12.6 with no slack): kernel work alone cannot
    reach 300. At the budget cycle, 250 needs K 11.7 and 330 needs K 15.4. With Leap's realistic dense 16-20 and every
    other piece at budget (a 50.7-54.7 ms cycle), K 14 gives 256-276 tok/s and 300 needs K 15.2-16.4 of 17.
  - CORRECTED Fri 08:0x: the experts row said "also 5-13 % fewer expert bytes". That held only for 3c-N; FAST-NAX
    reaches ~25 ms at 16 rows.
  - CORRECTED Fri 09:0x: my 08:1x "FAST-NAX's real-block expert term ~19 [P]" came from differencing probes across runs.
    The A/B's own split puts experts + commit + rest at ~28-32 ms, so the experts sit near the 25 budget, not under it.
  - Was (07:3x): the budget against ZERO's C 208 read (98.5 ms cycle at K 9.38 = 95 tok/s).
  - NO-WAIT CEILING [K, Leap TABLE_350, Fri 09:38-09:40, quiet, C 224]: HumanEval/68 (160 tokens, 12 checks of 13-16
    rows, K 14.42), all-resident with a few residual misses. FAST-NAX alone: draft ~10.0 + check + commit 67.2 = ~77 ms,
    ~187 tok/s. Stack (FAST-NAX + DN-KS + D7BNAX_32=2 + att_single + lazy): draft ~7.2 + check + commit 61.2 = ~68 ms,
    ~211 tok/s (-11 % cycle).
    - So 300 at this prompt's K 14.42 needs a 48.1 ms cycle: -20 ms. The check splits roughly into dense ~20 (near its
      floor in today's form), experts ~25 (bytes-bound, ~180 GB/s cold) and ~16 of misses, commit and glue (85 misses).
    - The known next levers sum to -13..-20 ms [P]: experts -7..-13 (the group cuts below), dense -4.5..-5.8 (Leap's
      list), draft -1..-1.2 (QMM32, small calls), plus never-wait on the residual misses. That is ~48-55 ms, 262-300
      tok/s on this copy-friendly prompt.
    - The 12-HumanEval average K is ~9.4-10.4. There, 300 needs a 31-35 ms cycle, which nothing priced reaches. K is
      the gate on average prompts: CD16 tonight, then MTP Head's drafter.
  - CD16: KILL on quality, Fri 16:0x (item 6); the K numbers below are its replay, kept for the record.
  - CD16 [P, Fable GOAL's K_WALL_1106 replay, Fri 10:50]: K on MBPP 5.6 -> 10.3-10.9, whole HumanEval 10.4 -> 13.2
    (docstring-free HumanEval 6.2-7.7 -> 10.7-10.8).
    - At those K, 300 needs a 34-36 ms cycle on MBPP and ~44 ms on HumanEval. At the priced 48-55 ms cycle they give
      MBPP ~185-225 and HumanEval ~240-275 tok/s. So HumanEval-style code at 300 sits ~4-11 ms past the priced levers;
      MBPP needs ~14-20 ms beyond them.
    - CD16 also checks more rows (MBPP 7.6 -> 12.5, HumanEval 11.9 -> 14.2 per check). At ~1 ms per row of FAST-NAX
      expert time, that adds ~2-5 ms per check [P], which the prices above leave out.
    - K >= 14 on MBPP is out of reach for any drafter in sight (13.3-13.6 even with the 125B's own features [P]), so
      on average code the 300 gate is the cycle.
    - Quality gate: EvalSuite's MBPP decision 4 tonight (the risk is MBPP's docstring planning). Speed: BigMoE's
      12-MBPP arm on Saturday night (no MBPP prompt set in the screen yet).

EXPERT-BYTES LEVERS on FAST-NAX, ranked for the weekend (NewIdeas Fri 09:4x, the orchestrator asked). Premise [K, Leap
NAX1CQ-TUNE]: FAST-NAX's expert one-call is bytes-bound (~180 GB/s cold, no tunable winner), so ms follow the groups
read. Prices [P] = (1 - group factor) x the expert term's variable part (~17-25 of ~25 ms at 16 rows; the fixed
per-layer part is unmeasured). Every group cut below is code-only (chat fails the map for all of them); the map never
checked math, which the CODE profile also serves.

  | # | lever | exists? | quality status | ms at 16 rows [P] | weekend step |
  |---|---|---|---|---|---|
  | 1 | split 0.99: cf >= 0.99 rows keep ranks 0-1 + members, no resident keep | live (R57_QOM_SPLIT; QOM-SPLIT-1C covers NAX6-N 1C-Q) | map code PASS 0.92 % (non-tie 0.51), 31 flips vs a line of 45 | groups x0.625 [K, 3c-N A/B] -> -6..-9; kept -2..-4 % on 3c-N | tonight's SPLIT-NAX arm (speed); a PASS -> same-scores, code + math |
  | 2 | the code frontier at ~x0.5 groups: split 0.9, Fable C's K8, or MASSB-0.5 (one of them; they cut the same groups) | split 0.9 live; K8 and MASSB-0.5 are map arms only (each needs an engine keep rule) | code PASS: s90 1.31 % / 0.78 (44 flips vs 45, thin); K8 1.19 / 0.69; MASSB-0.5 1.31 / 0.87 (at the line) | x0.49-0.54 -> -8..-13 | only if #1 shows ms follow groups: an s90 arm (exists); K8 needs a build |
  | 3 | Fable A's xs input sparsity (skip down-proj rows whose SwiGLU input is ~0) | map patch built (run_1105: xs64 / 48 / 40 / 32 + code); no engine kernel | T1 map arms tonight | -5..-7 [P, Fable A], only with a kernel that skips the weight reads | the map verdict tonight; a kernel only on a PASS |
  | 4 | NH-1C: NAX6-NH's ALU path for one-pair groups inside the one-call | NAX6-NH exists (N7-14 PASS, FAST); the 1C form is not built | FAST numerics; same-scores gate | ~-2..-3 [K bench: NAX6-N 33-36 vs NH 31-33 at 16 rows]; less if bytes-bound | Leap, after #1 |
  | 5 | top-8 on unsure rows (R15's rule) at 45 GB code | map arm exists | PASS on the 24 GB chat route; 45 GB code untested | ~-1: code has few unsure rows | low |
  | 6 | fewer bits (MSB-2 tails on low ranks; no LSB on unsure rows) | no store or kernel path | bit cuts have failed so far (trellis 2.4 b; MSB-2 ~5x MSB-3's output error) | <= -2.5 | not this weekend |

  - Already in the stack, so not counted again: QOM (sure rows skip cold ranks 2-9), U3 (no LSB on sure rows), B's
    Q1-UNION gate on code.
  - RESULT Fri 15:4x: #1 KILL on the stack (-1.3 %, kept -5.7 %). #2 shares its mechanism and would lose more kept,
    so it is deprioritized. #3 (xs, the map tonight) is the live expert-bytes lever.
  - Order (was): #1 tonight decides the premise. If ms follow groups, #2's s90 arm is next. #3's map verdict lands tonight
    regardless. #4 is a kernel step after that.

GPU ORDER, Fri 6 -> Sat 7 Nov (set by NewIdeas Fri 08:0x at the orchestrator's request; goal: by Saturday morning, the
stacked code number AND whether it is publishable). Every owner sizes its chunk from its own path's measured times and
ends it by the next item's start.
- Today to 15:00:
  1. BigMoE's FAST-NAX engine A/B (quiet, 12 HumanEval, C 224), now. Then Leap's DN-KS FLAT A/B (quiet, ~6 min).
  2. MTP Head's PORT, as armed (after BigMoE's lock). It ends or yields by 12:30.
  3. EvalSuite's 20-min smoke in the first gap: beside PORT under a share lock if both fit in memory, else right after it.
  4. EvalSuite's agent check (plain FAST, no gate, 17 agentlong turns, 2 orders, ~2.5 h; PASS = 0 blunders) on
     NAX6-N (env 6c08636, R57_QOM_SETS with agentlong; the orchestrator 08:1x). Start by 12:30 so it ends by 15:00. If it cannot start by 12:30,
     it runs Saturday by day as the only big job. It ranks last today: agent turns are 92-94 % prefill, so it decides
     only a few % per agent turn, and agent turns default to EXACT, so the publish does not wait on it.
  5. DONE 12:43 (item 1's BISECT). (Fri 11:2x) The agent-FAST cause bisect on write12-14, scored by EvalSuite's replay scorer, one lever off per arm:
     QOM-KEEP first, then DRAFT17's bonus row, HC8, FAST-NAX's base, and the long-context pieces (DKV / att_single past
     512 keys). Prior: QOM-KEEP, the only history-dependent lever (the orders diverge by token 1-6). Sharp test: without
     QOM-KEEP the two orders should be token-identical. Line for that arm: blunders <= 20 % of plain FAST's on the same
     turns = it carries it; >= 50 % = not the main cause. Today if it can start by 14:15, else Saturday by day as the
     only big job.
  6. DONE 13:46: MAP PASS on code (item 4). (Fri 12:5x) Breakthrough's never-wait map arm, moved up from tonight: the misses (-13.5..-15 ms per check) are the
     second-biggest gap to 300, and Leap has never-wait's engine side built, so a map verdict today lets the engine
     A/B land Saturday. Share lock, alone, ~15-30 min. If it cannot start by 13:30, it stays first in tonight's map.
  7. NOT RUN (Fri 15:1x): runner 51 waited from 13:51 for 50 GB NO-SWAP (it peaked at 47) and the day window closed;
     cancelled. Tonight's two top-ups supersede it. Then EvalSuite's long-code smoke: 6 of the drift top-up's 40 problems on tonight's CODE env (stack + B, QOM on code
     sets), one order, scored for blunders, ~20-30 min. If it shows > 1 blunder per 10K, tonight's top-up also runs the
     QOM-off CODE arm. EvalSuite's determinism follow-up (an exact base in both orders, then one-lever-ON arms, ~1 h)
     moves to Saturday by day.
> (a quote from the project owner, kept private)
  job ENDS by 17:00):
  1. BigMoE, quiet, to 16:10: the NW and SPLIT-NAX speed arms vs the stack (12 HumanEval, C 224, the lines above).
     - If NW PASSES (>= +8 %), NW joins tonight's GATED stack on the code sets only (the "every lever that passes today"
       rule; NW's map is clean on code at every pool). That saves a night for NW's gate.
     - The split stays speed-only tonight whatever it scores: strict rows drop more experts (the agent-FAIL mechanism),
       and its long-code risk is unchecked.
     - Then, if it is ready by ~15:50: GOAL's 12-MBPP CD16 speed arm (a SETOF entry on dec_mbpp's 100 walk prompts).
     DONE 15:4x: both KILL (NW -2.8 %, SPLIT-NAX -1.3 %); neither joins tonight's gate, and both leave tonight's
     palindrome. The stack itself measured 157.9 tok/s. MBPP12's first try died at once (rc 127: the runner split the
     comma in R57_QOM_SETS); re-run ~16:00.
  2. EvalSuite, 16:15-16:55: the MBPP check's NAX6-N reference on the 100 MBPP tasks (exact, no stack env), chunked to
     end by 16:55. The stack + CD16 arms stay tonight.
  3. Fit12's 10-min kp2 only if EvalSuite's build is not ready; else the next window.
- (Superseded) HOLD Fri 21:17-23:30.
- WINDOW OPEN Fri 23:30 (Kamen 23:2x, via the orchestrator: start now and keep going until he stops it on Saturday;
  HOLD moved aside, PAUSE cleared, fans 6500 / 7000 when hot). RE-TIMED by NewIdeas Fri 23:3x: no 00:30 / 06:00 / 10:00
  gates. Each item starts when the one before it ends:
  0. BigMoE's 200-STACK ticket (quiet, ~45-65 min), now.
     23:42: it ran 0 arms and exited (pid 41994; no sat_report). c45's deadline check compares HHMM without the date:
     "mtb_192: past 0042, stop" at 23:42:48. Part A: MATCH g2 ran 0 / 2, T16 MBPP12 had 0 tasks, PICK none. NO-SWAP read
     40.6 GB (C 192). VOID. BigMoE had already fixed it (4ef9bb2: a wrap-aware check; NO-SWAP read after the quiet
     grant) and RELAUNCHED at 23:49:58 (task bduyhhrad): NO-SWAP 49.5 GB -> C 224, C 256 ON, T16 on; quiet to ~00:54,
     report ~00:35. No gap, so nw3 stays after runner 49 (the offer to Breakthrough was retracted 23:50). Leap's S0 run
     ended before 23:50, with no BigMoE arm beside it.
  1. Leap's ~5-min C 224 GPU-idle run (its waiter keys on BigMoE's end). It decides S0.
     1b. (Fri 23:58, Leap asked, NewIdeas agreed) Leap's ~6-min quiet slot right after BigMoE's lock is released:
         STEPB_XPT (the expert path's Python ms per layer, plus thread CPU, at SB 2 and SB 4; the S3 input and the S1 /
         S3 order) and the 2-prompt smoke of the S2 diff (SVH and DSUB have never run inside r57). MEM ~39 GB, C 224.
  2. Fit12's 12GB-CLEAN (quiet, alone, ~8 min): re-armed 23:34 without the 06:00 floor. NewIdeas touched its GO file at
     23:58, so it starts after 1 clear min, after Leap's slot.
  3. EvalSuite's runner 49 (~4 h: same-scores on BigMoE's code_env, both drift top-ups, MBPP coverage): it starts once
     the "Verdict stacked FAST" line is in BigMoE's sat_report, Leap's run is done and Fit12's DONE file exists, or 30
     min after the verdict line at the latest. If Fit12 missed its gap, it takes the gap after runner 49.
     Sat 00:51-00:54: it started (after the orchestrator touched code_env.txt; its mtime was before the runner's 00:30
     EARLIEST), then VOIDED and stopped. Its provenance check found 12 loaded modules that are BigMoE's untracked
     sat_night freeze copies (data/r57/snap/frozen_sat/*, relax_span.py), not 5e41177's. Fix asked 00:57: BigMoE commits
     the snapshot (or gives a manifest), EvalSuite re-pins and restarts. The same check would hit S4 / CERT, so the S5
     freeze gets committed. nw3_first.sh fired on runner 49's pid ending, so nw3's 3 code arms took the gap (~00:55).
     FIXED 00:57: BigMoE pinned the snapshot (5fecd8f). EvalSuite restarted runner 49 (58d1ad9): its git check skips the
     data/r57/snap freeze, and every loaded module's sha256 (frozen_sat included) is in per-process manifests pinned
     across processes. It waits for nw3 on memory. Runner 53 (CERT reference decodes) is queued behind it (END 16:00).
  4. Breakthrough's map, nw3 first (S1's input): beside runner 49 under a share lock if memory allows, else right
     after it. Then step 0, then T1 (xs has no engine kernel yet). The 24 GB arms move to Sunday night.
  5. EvalSuite's CERT reference decodes (~1.5 h, NAX6-N): right after runner 49, beside the map if memory allows.
  6. S1 and S2, each as soon as its input is in (S0 is OUT, Fri 23:5x; S3 is PARKED, Sat 00:2x). S2's input is in
     (Leap's smoke PASS); S1 follows nw3's code verdict. One quiet A/B each (~10-15 min) at the next chunk boundary of EvalSuite's
     runs (no preemption mid-chunk). They go ahead of CERT's reference decodes and T1.
  7. S4 (the short gate, ~1.5-2 h) once S0-S3 are decided, by 16:00 at the latest; S5 after it; FREEZE ~20:00; CERT
     after the freeze.
     FINAL ENV (NewIdeas Sat 01:1x, BigMoE asked; EvalSuite's S4 starts when data/stack/final_env.txt lands):
     - v1 = stackb + S2 at C 256 (stackb_s2_env.txt: the stack + B span + g2 + SBN 2 + D7BNAX_32 2 + DSUB + SVH, QOM on
       humaneval / mbpp; 146.8 tok/s, 42.8 GB). Code frozen in data/r57/snap/frozen_s4 and committed.
     - BigMoE writes it when runner 49's CODE verdicts are in: the same-scores plus both long-code top-ups, the top-up
       being B's rollback gate. MBPP coverage does not block it.
     - If B rolls back or the same-scores fails, v1 = the stack + S2.
     - S1 joins v1 if its engine A/B passes before then.
     - S4 runs before runner 53 (runner 53 still ends by 16:00).
     - v2 = v1 + any lever whose engine A/B passes by 14:00 (S1, S2b, an S2c fix), with a second short S4 starting by
       16:00. Without one, v1 is the freeze config and S5 measures it.
  Share jobs (MTP Head's every-position walks for GOAL, ~25 min, 6 GB) run beside runner 49 or the map (peak + 8 GB
  free), never beside a quiet run. Disk floor 100 GB (105 GB free at 23:30): run df before writing any new big file.
- SATURDAY DAY WINDOW (NewIdeas Fri 21:2x, the orchestrator asked; order by expected gain; every lossy piece is gated
  before it ships):
  S0. OUT (NewIdeas Fri 23:5x, agreeing with the orchestrator's route). Leap's C 224 run (23:43-23:49, stack env, real
      16-row loop, 3 passes x 11 steady checks, kept ~14.3): submitting every 4 layers (SB 4, today's default) leaves
      8.1-11.1 ms of GPU idle per cycle (median 8.4; cycle 70.5-72.2 ms). SB 2 leaves 2.5-3.6 ms and cuts the cycle
      71.1 -> 68.5 ms. Once S2 ships SB 2, what is left is under the 8 ms bar. [K, Leap; NewIdeas could not re-derive the
      in-cycle busy %: the two saved real runs carry the same env, and per-check times are not stored.]
      Was: (the orchestrator Fri 21:2x, on top) The ASYNC POST. The check's graph build (59.6 of 68.8 ms) includes the gate's
      per-layer wait for router ids, so Python blocks once per layer. If the GPU idles there, that is ~10-19 ms per
      check, +14-30 % [P].
      - Leap's 6-min C 224 GPU-busy run after BigMoE's ticket decides it: idle >= ~8 ms per check -> BigMoE + Leap build
        it Saturday morning, then a quiet A/B on BigMoE's pre-registered line.
      - It is scheduling only, so a token-equality check replaces the gate; if tokens differ, it joins S4.
      - This is the one lever that could reach 200 (~150-175 x 1.14-1.30 = ~171-227 [P]).
  S1. NW v3 (Leap, staged, off by default). v3 changes what nw1's map PASS covered: cold QOM optional uses are dropped
      (QOM skips them anyway), and a cold LSB sidecar with a resident base is served at 3 bits (LSB-on-miss: new and
      lossy). So it needs a map arm first: nw3 on the code windows (+ chat for the record), Breakthrough, ~15-20 min.
      Then BigMoE's engine A/B vs the gated stack, 12 HumanEval, quiet. Line: PASS >= +8 %, KILL < +3 %. [P] +5-15 %
      if v3 removes the stalled calls nw1 left (17 per check, the ~14 ms miss spin).
      NW3 MAP ARM QUEUED (Breakthrough 5db7365, 21:35; SCOREBOARD 'Fri 21:3x: PRE-REGISTRATION, NW3'), right after T1 in
      tonight's map:
      - (a) changes nothing on the map. (b) runs as the WORST-CASE bound: every unsure answer row's ranks 0-1 at 3 bits,
        because sidecar residency can't be modelled.
      - Arms: a 4-bit control (the path's summation order), code at 208 slots and at the published pool, then both
        pools on all 36 windows (chat on the record).
      - Lines per pool: PASS = VERDICT2 code AND D vs nw1 <= +0.40 (clean <= +0.15); BETWEEN to 2.0 % / +0.80; KILL
        above. A failed bound is not a KILL of (b): the engine's NWCNT then says how much of the bound it meets.
      - Prediction [G, Breakthrough]: code D +0.2..+0.5 (PASS 55 %). Re-timed Fri 23:3x: first in the map, beside
        runner 49 if memory allows, else right after it.
  S2. Leap's glue fix (submit every 2 layers instead of 4; its in-process A/B runs in tonight's 6-min slot) plus
      whatever cycle items Leap installs by noon (expert id glue, HC reads, GDN step). One quiet A/B. Bitwise-equal
      tokens mean no gate is needed; otherwise they join S4's gate. [P] +2-5 %.
      MEASURED Fri 23:4x (Leap's loop): SB 2 cuts the C 224 cycle 71.1 -> 68.5 ms (-2.6 ms, ~+3.8 %), and 2-row checks by
      5.7 ms (50.1 -> 44.3). Kept per check is unchanged (scheduling only). S2 = R57_SBN=2 + D7BNAX_32=2 + DSUB / SVH:
      the diff ~/Leap/data/s2wire/r57_s2.diff applies clean on r57 fa8bee48. Leap compile-checks and smokes it after
      BigMoE's lock (~00:54). Then BigMoE's quiet A/B (12 HumanEval palindrome vs code_env) in the next slot.
      SMOKE PASS Sat 00:1x (Leap): 2 prompts, 517 tokens, rc 0. svh_calls 69 == head2s 69, chk_out 0, dsub_extra 3 ==
      svh_extra 3; footprint peak 38.6 GB. Leap's repeat of SB 2 vs SB 4 gave -1.3..-2.0 ms, so the range with 23:4x's run is
      -1.3..-3.6 ms per 16-row cycle. BigMoE's A/B (stackb at C 256 vs + S2) settles it. It takes the next quiet slot:
      before runner 49 if BigMoE is ready, else runner 49's first chunk boundary (45-min chunks).
      RESULT Sat 00:4x (BigMoE d9debf9, C 256, quiet, 12 HumanEval palindrome): PASS +3.1 %, 146.8 vs 142.4 (re-derived
      x1.031), cycle -1.3 ms, peak 42.8 GB. Tokens S2 vs B matched on 7-10 of 12, but B1 vs B2 matched on only 7 / 12 (QOM
      run to run), so token equality cannot clear any S-item. S2 joins S4's gate (data/stack/stackb_s2_env.txt). stackb
      itself read 142.4 here vs 144.9 in the ticket (~1.7 % run to run), so S5's palindrome sets the number.
  S2b. ENGINE ITEMS WITH NUMBERS (the orchestrator Sat 00:3x: "combine what we have"; sources: Fable B's DENSE_REBUILD.md
      sections 2, 0b and 5, and kernels/eprep.py). Leap builds and benches; BigMoE runs one bundle A/B. Prices are per
      16-row check [P, Fable B's fill model]. Fable B's own small-kernel prices ran ~2x high on Friday (section 0b), so
      plan on about half.
      | item | price | Leap's bench line (in situ, chained, 16 rows, C 224, stack env) |
      |---|---|---|
      | f. the two folds (NEW; Fable B section 2, "not run yet" in 0b): GDN a\|b into in_proj's stream (tiles 512-514), attention k\|v into q's (tiles 384-415); no new kernel, no K split | -1.6 / -1.1 | PASS >= 50 % of the price, KILL < 25 %; record bitwise vs the stack |
      | e. expert id / mask glue (kernels/eprep.py: compiles, 4,000 cases, 0 differ) | -0.6..-0.8 | same line |
      | b. k3c up mix to ~220 GB/s (3.8 MB at 137-143 today; the 16-load tail is the suspect) | -0.9 | same line |
      | d. attention's ~25 other launches per layer after att_single | -0.5 | same line |
      | a. HC reads' in-loop excess: 8.6 ms in situ vs 5.95 chained, ~2.6 ms with no owner | locator first | PASS = a chained replay of one whole read inside a layer names >= 1.5 ms of it, with a fix by ~12:00 |
      | c. in_proj 210 -> 240 GB/s via nt 2 | -0.5 | doubtful: Leap's Fri-morning verdict (0b) says nt 2 / 4 do not help in_proj at ks 2. Only a new idea revives it |
      - Bundle A/B (BigMoE, quiet, 12 HumanEval palindrome at C 256, vs the then-current stack, i.e. stackb + S2 (+ S1 if
        in)): PASS >= +3 %, KILL < +1 %. If tokens are equal on 12 / 12, no gate is needed; else the bundle joins S4's gate.
      - Priced total: f 2.7 + e 0.7 + b 0.9 + d 0.5 = ~4.8 ms, ~2.4 at the usual haircut; a adds up to 2.6 if found.
        That is +3-10 % on the ~78 ms cycle.
      - Fit: Leap builds on the CPU now. Per-piece benches take short quiet slots at EvalSuite's chunk boundaries. The
        bundle A/B lands by ~12:00, and the last call is 14:00 so S4 can start by 16:00. Anything later is post-freeze.
      - Bigger pieces NOT reachable by the freeze:
        - The launch count, ~3,000 launches x 1.6 us = ~4.8 ms per check. It needs a fused layer kernel.
        - The never-wait ceiling's remainder, ~11 of 18.5 ms: read time on the critical path, which needs fewer cold bytes.
        - K: nothing ready. The drafter retrains are inside noise (K_PLAN_1106: P-CAP +0.18, P-OFF +0.11, a reseed moves
          0.16), T16 is BETWEEN, and 32-row checks need <= 0.9 ms per row.
      - The router on qmm16c8 (-0.6) stays parked: it can flip routes on near-ties.
      - Count S2b at [G] until Leap's step 1: a ~5-min quiet in-situ pricing of the HC / k3c / in_proj / attention
        pieces, unsynced. Leap warns the HC "excess" rests on synced per-piece timing and may mostly vanish.
      - STEP 1 RESULT Sat 00:42-00:44 (Leap, quiet, C 224, SB 2, stack env). Each piece is stubbed in the 16-row
        dense-only forward (base 21.85 ms wall, ~20.5 GPU; 4 rounds x 12 forwards, interleaved). Saved, wall / GPU ms:
        - HC reads, both: 6.82 / 11.3 (attention-side 5.05 / 6.3, MLP-side 2.37 / 2.9).
        - HC write: 3.58 / 3.3.
        - GDN: 7.45 / 10.1, of which in_proj is 4.22 / 5.9.
        - Attention: 3.53 / 4.3. MoE block (router + shared): 1.36 / 1.6. PLE: 0.62 / 0.75.
        - Big stubs push the forward toward its CPU build, so their wall deltas are lower bounds.
      - CORRECTION Sat 00:5x (Leap): the stub breaks the GPU's barrier chain. MLX's concurrent encoder puts a barrier only
        where a kernel reads a buffer an earlier kernel wrote, and the stub returns old buffers. So the next layer's
        kernels overlap with the work before them, and every delta above is an inflated UPPER BOUND, not a price. It is
        worst for small pieces on the layer-to-layer chain: the HC write's 3.58 ms is a ~3 us kernel plus the chain it
        broke. The step 1b slot re-runs it with a barrier-preserving stub, its calibration and a D5 on / off check.
      - Verdicts per item:
        - a' DROPPED. Folding the MLP write into the next attention read is D5 (REF2_HC_PEND=1, Wed 11-04): bitwise, but
          KILLed as a speed lever (write + next read 85.4 -> 83.0 us; +0.3 us with R-73's compiled write on). The 8.6 vs
          2.4 ms behind it was the broken barrier chain. I queued a' without checking the owner records first.
        - c KILL: in_proj is 4.22 in situ vs ~4.1 chained, so there is no excess.
        - d: inside attention's 3.53 ms; priced from the chained bench.
        - f REAL: R-83 (the stack's gdn_inproj_x) runs qkv|z and b|a as two calls. So the GDN a|b fold saves one launch
          per GDN layer, and the k|v -> q fold also applies. Leap benches both chained (bitwise vs the unfolded call at the
          same ks).
        - e and b: their own benches follow.
      - FOLDS RESULT Sat 01:3x (Leap, chained, 16 rows, real shapes; both bitwise vs the unfolded calls at the same ks):
        GDN b|a into qkv|z KILL (-0.31 ms per check, 19 % of 1.6); attention k|v into q BETWEEN (-0.31 ms, 28 % of 1.1).
        In a chain the small call already overlaps the big one. S2b's bundle A/B is OFF for Saturday (f attention + e
        ~0.4 [P] can ride a later bundle). Leap next: the EXACT bitwise proof for KEEP if BigMoE wants, and a CPU-only
        audit of r57's cycle for more of KEEP's bug class (per-cycle clears / rebuilds; the commit's GDN replay, ~2 ms
        average), priced in one quiet probe (PASS >= 2 ms per check).
      - STEP 1B RESULT Sat 00:54-00:56 (Leap, quiet, barrier kept: calibration minus stub, 16-row dense-only, wall per
        check):
        - HC reads 4.3 (attention-side 2.12; MLP-side, including the folded attention write, 2.20). HC write 0.14 (the
          depends stub had said 3.44). Attention 3.38. GDN 7.65, of which in_proj 4.63. MoE dense part 1.17.
        - D5 on vs off saves 0.04.
        - a KILL: the reads cost 4.3 in place vs 5.95 chained, so the "8.6 in situ" was synced accounting. c stays KILL.
        - S2b now rests on e (eprep) and f (the two folds); Leap's fold bench is next (chained, under a share lock).
  S2c. IN-LOOP EXCESS (Leap's lead from tonight's 132 checks, the orchestrator Sat 00:4x; possibly bigger than S1 + S2).
      Leap's fit: check + commit = 57.9 + 0.76 x gate service (r 0.77). So at zero service a real 16-row check + commit
      costs ~58 ms, but the same forward run back to back is ~45 ms and the commit ~0.5 ms. That leaves ~11-13 ms per
      check of in-loop excess: -12 ms on a ~78 ms cycle is ~+18 % [P].
      - Step 1 (Leap, quiet, ~5-10 min, right after S2b's step 1): a harness runs a real check's own block back to back
        and in the loop (unsynced, 16 rows, C 224, stack env). PASS = the excess reproduces at >= 8 ms per check AND a named
        mechanism holds >= 6 ms of it. KILL = the unsynced excess is < 4 ms; then the fit's intercept is not a removable
        excess.
      - Step 2 (BigMoE, if a fix exists by ~13:00): an engine A/B, quiet, 12 HumanEval palindrome at C 256 vs the
        then-current stack. PASS >= +6 % (half the price), KILL < +2 %. Equal tokens mean no gate; otherwise it joins S4.
        Last call 14:00, then post-freeze.
      - The same fit prices S1's ceiling at ~0.76 x gate service: ~3 ms on tonight's warm prompt, ~10.6 ms at ~14 ms of
        service. BigMoE's NW v3 A/B tests it on the 12-prompt set.
      - STEP 1 RESULT Sat 00:4x (Leap): BETWEEN.
        - The excess reproduces: check 6's own block runs back to back in 48.6-49.1 ms vs the loop's check + commit
          59-63, so ~9 ms at zero gate service (>= 8).
        - No mechanism holds >= 6 ms yet. "A new expert set per forward" is ruled out: alternating with check 5's block,
          or cycling checks 4-6, moves the block by only +0.3..+0.6 ms.
        - The 16-row commit is bimodal: ~0.47 ms on most checks, 4.6-5.2 ms on ~1/3 (GDN replay), ~2 ms on average.
        - Drafts are host-synced before the check (r57 line 1005), so the drafter cannot spill into it.
        - Step 1b (granted, ~3 min quiet, next boundary): per-piece sync inside a live check, then the same block synced
          right after; the per-piece difference names where the ~9 ms goes.
      - STEP 1B RESULT Sat 00:54-00:56 (Leap): still BETWEEN, but the excess is LOCATED.
        - Check 6, synced per piece: live 68.42 ms vs the same block replayed right after 58.40 (59.18 after the
          alternating forwards), so +10.0 ms live.
        - moe_pre (the router + shared expert before the expert call) takes +5.85 of it: 10.02 vs 4.17, i.e. ~209 vs
          ~87 us per layer. The same ops run slower live.
        - The rest: GDN +1.52 (likely the lazy commit's replay), experts +1.37 (gate service), attention +0.93, HC ~0.
        - 5.85 sits just under the 6 ms line, and the mechanism is not proven.
        - Two candidates: (1) the loader's background reads (look-ahead prefetch, miss reads) during the live check;
          (2) CPU contention on the encode (the loader thread at user-interactive priority on the P-cores).
      - Step 1c, REVISED Sat 01:0x (Leap). The prefetch-off arm is moot: r57's set_arm drops bm["next"] for the stack's
        arm (gate+t; only the +pf arms prefetch ahead), and the live check had only 2-3 misses. So the loader-reads
        hypothesis is out.
        - The moe_pre excess is uniform: +0.09..+0.16 ms in each of the 48 layers (median 0.121; live ~0.34 vs replay
          ~0.22 ms per layer). 0.12 ms is one GPU round trip (0.131-0.141 by calibration). So the live check pays about
          one extra synchronisation per MoE layer before the expert call.
        - NewIdeas' candidate, RULED OUT by Leap's code check (Sat 01:1x): the gate's host readback of the router ids.
          At decode widths the gated path has none: post_ids and spin_wait are lazy GPU ops, and tier.py:312's np.array is
          the sync / prefill path only. moe_pre is also timed before the post, so a readback would show in "experts".
        - The run (Leap, ~4 min, next grant) has three arms: base, BIGMOE_SSD_KEEPALIVE_US=0, and R57_VTIER= (BigMoE's
          suggestions). Each repeats step 1b at real check 6 with thread CPU per piece. An arm NAMES the cause if its live -
          replay moe_pre excess falls by >= 4 ms vs base (>= 0.08 ms per layer). Keepalive is a +12 % lever in the EXACT
          default, so if it is the cause, a FAST-only engine A/B decides the trade.
      - STEP 1C RESULT Sat 01:03-01:06 (Leap, quiet, check 6, thread CPU per piece): keepalive and VTIER are NOT the cause.
        - moe_pre live - replay: base +5.65, keepalive off +5.90, VTIER off +5.56 ms; none falls by >= 4.
        - Live vs replay totals, synced: 65.8 / 64.4 / 65.1 vs 55.2 / 52.2 / 55.7 ms.
        - moe_pre per layer: wall +0.117..+0.122 ms, CPU +0.030..+0.031 (on the 0.03 boundary). So ~75 % of it is
          non-CPU waiting and ~25 % more CPU.
        - NEW: across the whole forward, the Python thread spends ~20 % more CPU on identical code live. GDN 5.11 vs
          4.00, attention 5.49 vs 4.57, MoE block 15.2 vs 12.4: +5.0..5.4 ms of the ~10.5 ms excess.
        - So about half of S2c is the CPU running the same code slower, and half is waiting.
        - Suspect: core placement or clock. The main thread runs at default QOS and may lose P-core time to loader_main,
          which spins at user-interactive on a P-core.
      - Step 1d (granted, ~3 min quiet, next boundary): (q) the Python thread raised to QOS user-interactive
        (pthread_set_qos_class_self_np, a harness switch; a per-thread hint inside our own process, not a system
        setting); (i) BIGMOE_IDLE_MS=0.5 (the loader naps when idle).
        - Line: an arm names the cause if live - replay falls by >= 4 ms in total.
        - Also reported per arm: gate service / experts, and any heat / PAUSE flag.
        - A named cause becomes a lever only through BigMoE's engine A/B at C 256: PASS >= +4 %, KILL < +1.5 %, last
          call 14:00. If (q) holds, the fix is a one-line QOS call in r57.
      - STEP 1D RESULT Sat 01:09-01:13 (Leap, quiet; no PAUSE, die 67-77 C, fans 6500): neither arm names the cause.
        - live - replay total: base +10.8, QOS user-interactive +9.3, loader IDLE_MS=0.5 +9.4, both +11.8 ms.
        - moe_pre: +5.67 / +5.51 / +5.28 / +5.71. The live thread's extra CPU persists in every arm.
        - The experts piece is +1.5..1.9 live. IDLE_MS slows the loop itself (check + commit 77.6 vs 63.6 ms, pickup
          latency), so it is not a fix.
        - Out so far: QOS / core placement, the loader nap, keepalive, VTIER, look-ahead, the router-id readback.
      - Step 1e (granted, ~2 min quiet, next boundary): MLX's buffer cache.
        - BigMoE's stream.py caps it at 1 GB (BIGMOE_MLX_CACHE_MB=1024, set for low-memory nights). A live check comes
          after the drafter and the commit have churned that cache, so it allocates fresh Metal buffers, while the replay
          reuses what the live check freed. That fits "same code, +20 % CPU, plus waits in the encode".
        - Arm: BIGMOE_MLX_CACHE_MB=6144 vs base at C 224 (~39 GB + <= 5 GB). It names the cause if live - replay falls by
          >= 4 ms.
        - As a lever it must fit the 45 GB line: cached buffers count in the lifetime footprint, and C 256 is already
          42.8 GB with 1 GB of cache.
        - So BigMoE's A/B tests the largest cache that keeps C 256 <= 45.0 GB (~3 GB), or C 240 + more cache. PASS >= +4 %
          at <= 45.0 GB lifetime, KILL < +1.5 %, last call 14:00.
      - STEP 1E RESULT Sat 01:17-01:20 (Leap, quiet; no guard, no PAUSE): the MLX buffer cache is NOT the cause.
        - live - replay: 1 GB +10.4, 3 GB +11.2, 5 GB +8.8 ms; moe_pre +5.61 / +5.74 / +5.54. Lifetime peaks 39.3-39.7 GB.
        - The cache sat at 0.81-0.98 GB in every arm, live and replay, so the 1 GB cap never bound.
        - All seven measured causes are out: look-ahead, gate readback, keepalive, tier copies, QOS, loader nap, MLX
          cache size.
        - What stands (6 runs tonight): a live check is ~+10 ms vs its own block replayed right after (synced).
          moe_pre +5.6 (+0.12 ms per layer: ~0.03 CPU + ~0.09 wait), and the live thread spends ~20 % more CPU in
          every piece. BETWEEN: S2c stops at the location.
      - Step 1f (granted, ~2 min, the last S2c test today): replay the block once right BEFORE the live check, at the draft
        cut.
        - POSITION: if the live check then runs within ~3 ms of replay speed (the excess falls by >= 7 ms), the class is
          "the first forward after the draft / loop phase": CPU clock ramp-up after waiting on the GPU, or cold
          caches. A fix stays inside our process (keep the main thread busy through the draft wait, or a tiny warm
          forward), and only an engine A/B prices it.
        - LIVE-ONLY: if the excess stays >= 7 ms, the cause is inside the check. In between: BETWEEN.
      - Step 1f REPLACED Sat 01:2x by a sharper arm (Leap found a candidate mechanism in the code):
        - r57's loop calls ref2.remove() at every cycle start (the draft runs without REF2) and ref2.apply() before the
          check (lines ~995 / ~1050). remove() clears REF2's per-weight caches (_SN, the SMALL-N padded native copies;
          _DN; _HC8).
        - So every live check re-pads and GPU-repacks every SMALL-N weight on first use (_sn_get -> _dn_repack), plus
          the Python rebuilds. The replay right after finds them warm.
        - It fits each piece: moe_pre +0.12 ms per layer is the shared expert's gate|up / gate; GDN +1.5 is b|a (36
          layers); attention +0.9 is k / v (12 layers); the extra CPU is building and encoding those repacks.
        - Arm "keep" (harness-only, ~2 min): the loop's remove() only switches REF2 off (_ON False) and keeps the caches.
          They are validated by weight identity, so outputs stay bitwise the same: an EXACT-safe fix.
        - Pre-registered: "keep" names the mechanism if live - replay falls by >= 7 ms. Lifetime peak logged per arm.
        - If it holds, the fix is REF2_KEEP_CACHES (default off, self-test), or r57 toggling _ON instead of remove().
          BigMoE's engine A/B at C 256 prices it: PASS >= +4 %, KILL < +1.5 %, lifetime peak <= 45.0 GB. Bitwise identity
          is proven on the EXACT path (NAX6-N, a few prompts), since FAST tokens vary run to run.
        - If it is bitwise and passes before runner 49's code verdicts (~05:30), it joins final v1. It would also speed
          the EXACT line, but adoption there is BigMoE's and Kamen's call.
      - S2C PASS Sat 01:3x (Leap, quiet, C 224, SB 2, stack env; re-derived): mechanism named and fixed.
        - "keep" takes the live - replay excess from +9.39 to +0.48 ms, a fall of 8.9 (line >= 7).
        - Per piece, base -> keep: moe_pre +5.41 -> -0.05, GDN +1.64 -> +0.01, attention +0.77 -> +0.26, experts +1.73
          -> +0.42. Thread CPU +4.65 -> +0.48. The replay itself does not move (55.10 vs 54.28).
        - The real loop, unsynced (3 passes x 11 checks): check + commit 62.11 / 59.08 / 58.38 -> 54.90 / 51.66 / 50.37
          ms, i.e. -7.2 / -7.4 / -8.0 per 16-row check. The cycle goes ~66-70 -> ~58-62 ms, +11-14 % on this prompt.
        - Lifetime peak 38.6 vs 39.4 GB (the kept copies are ~0.1 GB). No PAUSE.
        - Fix: REF2_KEEP_CACHES=1 (a ref2 switch, default off; Leap installs it with a self-test).
        - Next: BigMoE's engine A/B at C 256 vs stackb + S2 (PASS >= +4 %, KILL < +1.5 %, peak <= 45.0 GB), and the EXACT
          bitwise proof (NAX6-N, >= 3 prompts, tokens equal). Both before ~05:30 -> final v1 includes it.
        - [P] on the ~76 ms 12-HE cycle: -7.5 ms = x1.11, so ~163 tok/s from 146.8.
        - EXACT-default adoption was flagged to the orchestrator as BigMoE's and Kamen's call.
      - KEEP TICKET QUEUED Sat 01:4x (BigMoE task b3ac0xo32, fac4c0a; quiet, next boundary after runner 49's chunk and
        nw3's share arms; its C 256 arm needs 47.2 GB NO-SWAP).
        - (0) The EXACT proof: NAX6-N exact env, C 224, HumanEval/68, /16 and ifeval/1220, CAP 64, keep off vs on.
          Tokens equal on all 3 = PROOF PASS.
        - (1) The A/B at C 256: stackb + S2 vs + KEEP, 12 HumanEval b / k / k / b. PASS >= +4 % AND peak <= 45.0 GB;
          KILL < +1.5 % or > 45.0 GB. Median cycle beside.
        - KEEP joins v1 only with PROOF PASS and speed PASS: s4_final.sh gets an S4_KEEP=1 switch.
        - RESULT Sat 01:4x (BigMoE c02f807): PROOF PASS (3 / 3 tokens equal) and speed PASS (+10.1 %, 164.2 vs 149.1;
          cycle 75.8 -> 66.9 ms; 43.1 GB). KEEP is in v1 (NewIdeas' word to BigMoE, 01:4x).
      - S1 UNBLOCKED Sat 01:4x: nw3's code map PASS at both pools (Breakthrough 7e08711: published 0.71 %, D +0.12,
        clean; 208 slots 0.86 %, D +0.18). BigMoE's NW v3 engine A/B on v1 runs at the next boundary (PASS >= +8 %,
        KILL < +3 %, peak <= 45.0 GB). It joins v1 if it passes before runner 49's code verdicts, else v2.
      - RESULT Sat 01:5x (BigMoE f9c9fff): PASS as registered, +9.0 % steady (180.3 vs 165.3, peak 43.9 GB). NW joined
        v1 (final_env 1403cd6, 04:35).
      - RE-DERIVED Sat 04:5x (NewIdeas; scripts/s1_fullperiod.py, data/s1/): BETWEEN on all wall time.
        - Full-period steady: +4.3 % (158.5 -> 165.4). Whole decode: +3.1 % (137.8 -> 142.1).
        - The steady window (c0..c2) leaves out NW's counter graph: a sort plus 8 sums per layer call. That graph
          runs at the readback after c2 (r57_screen_s4.py:1355).
        - Both arms also waited there for the commit replay through the q1c readback (R57_QCNT1C=1 in S1).
        - Time outside ck + drf per check: NW 4.71 / 4.79 ms vs B 2.35 / 2.33 ms. BigMoE confirmed this 05:1x: full
          period +4.8 % on his cut, whole decode +3.3 %.
        - S1b (BigMoE c71b2ce, scripts/s1b_ab.sh):
          - Arms: B = final_env minus NW, with no QCNT1C and no RELAX_LOG. NW = B + R57_NWCNT=0 (a new knob, default
            on, so 4ef3734 is unchanged).
          - Palindrome at C 256, after a 2-prompt NW check. Scored on exact full period (per-check c0 is now
            recorded), with whole decode and tpsw beside.
          - Armed at S4_DONE (a quiet ticket, ~12 min, window to 12:00). Report: data/s1b/s1b_report.txt.
          - PASS >= +8 % AND peak <= 45.0 GB: NW stays, with its counters off.
          - Otherwise: final_env v1' = stackb_s2 + KEEP, plus its S4 in the v2 slot by 16:00. NewIdeas' word was
            given in advance, 05:1x.
        - Carry-over of S4 to counters-off is EvalSuite's call. Evidence offered:
          - a bitwise proof with a fixed late set (REF2_NW_FORCE), counters on vs off;
          - S1b's late groups per check vs S1's 18.0. Less time between checks can mean more late groups.
      - POST-FREEZE BUNDLE (exact-safe; Leap's cycle audit Sat 01:4x found no other per-cycle rebuild of KEEP's class):
        1. GDN state checkpoints at verify rows 8 / 12, so a partial commit replays <= 4 rows instead of 35 separate
           gated_delta_update calls over all kept rows (3.9-4.8 ms on partial checks; ~1.5 ms per cycle on HE/68).
           ~1-3 ms per cycle [P]. Leap may build it now on the CPU; GPU benches only in gaps that delay nothing.
        2. The attention k|v fold (-0.31 ms). 3. eprep (~0.4 [P]). 4. Merging the draft-quality mx.eval into the
           first one (~0.13-0.2 ms per cycle, same bits; BigMoE's r57).
        - The run (~3 min quiet): step 1b with thread CPU per synced piece, live vs replay. Pre-registered "extra wait":
          live - replay wall >= 0.08 ms per layer with |CPU live - CPU replay| <= 0.03 ms. More CPU live instead means
          encode contention.
        - The unsynced price needs a direct A/B once the sync is found. S0's GPU-idle read (2.5-3.6 ms per cycle at
          SB 2) is its floor, unless the 40 ms util sampling hides sub-ms gaps.
        - Fix space if confirmed: decide QOM's skip on the GPU (a residency bitmap), and read back to the host only
          when a rank 0-1 expert is cold (2-3 per check).
  (Map order, re-timed Fri 23:3x: nw3 first (S1's input), then step 0, then T1 (xs has no engine kernel yet, so it
  cannot ship Saturday). The 24 GB arms move to Sunday night.)
  S3. PARKED Sat 00:2x (the orchestrator and Leap). Leap measured it directly at SB 2: the Python thread uses 38.9-40.9 ms
      of CPU per 66.5-68.7 ms cycle.
      - Where it goes: graph build 15.2-16.2 (the expert calls are 3.75-3.97 of it, 0.08 ms per layer; dense + glue
        11.5-12.2), Metal encoding inside mx.async_eval 18.0-19.3, and draft + commit + loop ~5.5.
      - The cycle is GPU-bound (~96 % busy), ~27 ms above the ~40 ms CPU floor.
      - An expert-glue compile wins <= 3-4 ms, and only once GPU time per cycle nears ~45 ms. A later CPU lever is fewer
        dispatches (encode ~19 + dense build ~12), which also cuts GPU time.
      - Order: S1 before S3. NW v3's ~7 ms of gate service per check sits inside the gap, so the floor does not limit it.
      - WITHDRAWN: my Fri 23:5x line said "the expert path costs 24.1 ms of Python". The FULL vs DENSE build gap in
        gap_night.txt is Metal encoding plus back-pressure from the busy GPU, not Python. I re-did the subtraction but
        never checked what the "build" timer measures; Python time needs thread CPU time.
  S3b. Tonight's leftovers, only if the relaunched ticket misses them: T16 (MBPP12 pair, then stackt), C 256 (if NO-SWAP
      was short), MATCH-16 (if its 2-prompt check failed).
  S4. A short gate on the final stack (EvalSuite designs "short"): HumanEval 60 + math 50 in 2 orders, MBPP if T16 is
      in, and the 41-problem long-code top-up (blunders).
  S4 stays SHORT on Saturday (EvalSuite docs/PREREG_s4_2026-11-07.md, ~1.5-2 h): it decides Kamen's Saturday number.
  CERT, the release certificate at a larger n (the orchestrator Fri 21:2x, from Fable FULL's paper 3.7: decision 4 at n 50
  is thin; RELAX's math verdict moved on 4 items):
      - Set: full HumanEval+ 164, math 100, MBPP 100 (the walk's set), all in 2 orders; IFEval 100 x 2 only if a chat
        FAST profile ships (chat EXACT needs none). Lines tightened to the larger n; EvalSuite sizes and pre-registers.
      - When (the orchestrator, Fri 21:3x): FREEZE at Saturday ~20:00, and the S5 config is what CERT runs. CERT runs
        Saturday night (Sun ~01:30-10:00), so its verdicts are in before Sunday's paper fill. Monday night is the
        fallback only. Anything that lands after the freeze is post-release research. ~4-6 h [G]; EvalSuite is sizing it.
      - The paper quotes each verdict from its report path.
      - PRE-REGISTERED Fri 21:25 (EvalSuite docs/PREREG_cert_2026-11-08.md):
        - Lines: per set and order, decision 4 vs NAX6-N (the stack's exact base); pooled over 728 pairs, delta > -2 pt
          AND p >= 0.05; long-code <= 1 blunder per 10K. Reports go to results/cert/CERT_*.txt.
        - GPU: ~1.5 h of NAX6-N reference decodes (154 rows) + ~3.5 h of CERT (364 rows x 2 orders ~2.3 h + long-code
          41 ~1.2 h).
        - Slots (re-timed Fri 23:3x): the reference decodes run right after runner 49, with no DONE / 10:00 gate; they
          do not depend on the config. CERT runs Saturday night. No IFEval: chat ships EXACT.
        - BigMoE maps 'mathx' in SETOF (dec_mathx).
  12GB-CLEAN ARMED (Fit12 ex16.sh): ~8 min quiet, then it writes ~/Fit12/data/scr/ex16/DONE; deadline Sat 18:00.
      Re-timed Fri 23:3x: right after Leap's C 224 run, before runner 49 (WINDOW OPEN item 2); the 06:00 floor and
      the 10:00 gate are gone.
  12GB-CLEAN (the orchestrator Fri 21:3x; Fit12 runs it): the 12 GB verified-greedy loop ALONE on all 16 prompts, quiet,
      ~5 min, with its own lifetime peak footprint (Fit12's harness). Slot: Saturday morning, first after tonight's chain,
      just before CERT's reference decodes.
  S5. Saturday evening: the final measured number. 12 HumanEval palindrome at C 256 (if it passed) on the final stack,
      MBPP12 speed, and lifetime footprints.
- Tonight (re-timed Fri 23:3x: from now, in the WINDOW OPEN order above; details of each item below):
  0. (Fri 21:2x) Right after BigMoE's ticket, before EvalSuite's runner 49: Leap's ~5-min C 224 run (does the GPU idle
     between layers while the expert path blocks the forward?). Then Fit12's clean 12 GB run, then EvalSuite.
  1. BigMoE's stacked FAST palindrome on 12 HumanEval: today's all-in vs all-in + every lever that passes today
     (FAST-NAX, DN-KS, CD16, FASTTAX att_single, lazy commit if wanted), plus FAR-G1. FAST-NAX is in (its BETWEEN on
     record). Plus one SPLIT-NAX arm (stack + split 0.99, ~7 min; speed only, outside the gated
     stack; item 2). Plus one never-wait arm (stack + NW, ~7 min; speed only, outside the gated stack;
     item 4): IN, BigMoE e77d1fd (snv after ssp, snvb before sspb; per-check check / service / misses / stalls / wait
     plus the NW counters in the report). BigMoE writes the winning stack's env to ~/BigMoE/data/stack/stack_env.txt for EvalSuite.
  2. Env rule for tonight's gate (NewIdeas Fri 11:2x, after the agent FAIL): QOM on the code sets only
     (R57_QOM_SETS=humaneval,mbpp). Math rows stay as certified (B without QOM): math answers average ~670 tokens, and a
     first-time QOM on math could sink the one gate. Chat rows run a separate launch, stack minus HC8 + E (R1 0.5) with
     no QOM, as E was certified, or are skipped (chat stays EXACT).
     EvalSuite (the orchestrator's order, Fri 10:5x): the stack same-scores (~1.5 h, 2 orders) -> the drift top-up
     (~1.5 h; Fri 13:2x: both forms, the registered B top-up via relax.py (KLD + blunders, the rollback gate), then the
     stack's CODE env on all 41 (blunders only); if the night runs short, the map's 8 leftover 24 GB arms yield first) ->
     (Fri 16:0x: CD16 is KILLed, so this becomes MBPP on the stack without CD16, a coverage check, last, and it yields if the
     night runs short) MBPP decision 4 for CD16 (~1.2 h: NAX6-N reference on the walk's 100 MBPP tasks, then the stack with
     CD16 in 2 orders). Details of the first two follow. ONE same-scores run (decision 4 vs REF2-T) on exactly that stack, code and math on the CODE profile
     (stack + B with HC8), plus the code drift top-up (40 HumanEval+ problems, >= 1K tokens, ~1.5 h; PASS = blunders
     <= 1 per 10K AND pooled q4 KLD <= 0.02 AND no row's q4 > 0.05, else RELAX's code part rolls back). IFEval rows
     run the chat profile that would ship (stack + E without HC8) only if BigMoE's HC8-off launch exists by then;
     otherwise chat stays EXACT and needs no run. Bisect only on a fail.
  3. Breakthrough's map, fitted around EvalSuite's runs (share locks where memory allows): never-wait first (the 45 GB
     SSD-wait lever; DONE today, 13:46: only its 2-min self-check re-runs), then T1 (Fable A's xs arms), then the 8
     leftover 24 GB arms.
  4. EvalSuite runner 46's F1 moves to Sunday night (the orchestrator's call, Fri 10:5x: F1 certifies fs4, which the
     stack does not use; att_single ships through FASTTAX). BigMoE's 12-MBPP speed arm stays Saturday night.
- Fable B's v2 (ks / tile plans per shape) lands by ~11:00 and feeds Leap's next builds.

12 GB LINE (CORRECTED Fri 13:2x; Fit12 PLAN_12GB_90_fit12.md 'Lever 3 RESULT', re-read by NewIdeas):
- RE-CORRECTED Fri 21:3x (Fit12 via the orchestrator / FULL): the 12 GB loop is VERIFIED GREEDY, not exact. It is
  ordinary verified speculative decoding: the greedy argmax from 2-4-row verify calls, all ten ranks read. It is
  deterministic, but 2-4-row calls round differently from one-token decoding, and near-tie tokens follow the draft
  pattern. The ~12x claim against the published exact run is withdrawn (not like-for-like).
- 12GB-CLEAN RESULT Sat 00:3x (Fit12 ex16, 00:28-00:32, quiet, alone, rc 0; PLAN_12GB_90_fit12.md; re-derived by NewIdeas):
  18.42 tok/s decode geo mean on all 16 HumanEval prompts. Pooled it is 18.47 (3,456 tokens in 187.11 s); per-prompt
  range 17.16-20.30; 13.66 end to end with prefill. Lifetime peak 11.144 GB (under 12). Outputs equal kp2's on 16 / 16.
  This is the 12 GB release number for the verified-greedy loop (not exact).
- 12 GB FAST CODE GATE Sat 12:50 (EvalSuite 84b3715; re-derived from the counts): FAIL. No 12 GB FAST code mode from P1-K1.
  - t80ck1u (P1-K1, KU 4, θ 0.8; Fit12 measured 46.33 tok/s whole decode, 11.245 GB): HumanEval+ 50 vs 57 of 60 vs
    REF2-T (-11.67 pt, McNemar 0 / 7, p 0.0156) in both orders. Vs NAX6: 50 vs 55 (-8.33, p 0.0625).
  - t90ck1u (θ 0.9 fallback): 52 vs 57 (-8.33, 0 / 5, p 0.0625).
  - Method lesson: Breakthrough's map PASSED t80ck1u on its code line, but the full decode loses 7 tasks, every one
    against the arm. For this pass-1 form, the map's token-level code line does not predict HumanEval+ pass rates.
    Gate future 12 GB FAST candidates on task scores early, not on the map alone.
  - The 12 GB release stays the verified-greedy loop (ex16, 18.42 decode geo, 11.144 GB).
  - The wired fix there is worth ~+2 % [P]: Fit12's own phase logs put ex16's check-0 excess at 2.0 % of decode (4.7 %
    on t80ck1u). Fit12's armed wired A/B (wab.sh) runs on t80ck1u. Asked Fit12 13:1x to run it on ex16 instead, so its
    number is the publishable one.
  - WIRED A/B RESULT at 12 GB (Fit12 15:55, on t80ck1u, data/scr/wab/wab_cmp.txt): KILL.
    - Whole decode x1.0039, full period x1.0013, end to end x0.99. Tokens 16 / 16 equal. Lifetime 11.242 vs 11.244 GB.
    - Check 0 was unchanged per prompt, so at 12 GB the first-check cost is not the pools being unwired.
    - It is bimodal by prompt: ~180-220 ms on some, ~850-925 ms on others (HE 42, 91, 16, 161, 17, 56, 63). That
      points to cold pools after the prefill evicted the decode experts, a miss burst. It is 4.7-4.9 % of decode.
    - A post-publish 12 GB lever at most: pools warmed during the prefill tail. A6 pools / warm lost on time at C 160
      because they were not overlapped.
- Earlier: 18.4-18.9 tok/s on HumanEval [K: T3t 18.9 and T3R 18.42 quiet, kp2 18.58 share; four runs and prompt
  subsets mixed]. The 11.39 GB lifetime footprint came from a six-arm process.
- The 40.8 quoted before as "the 12 GB exact line" is T3R2's t80ck2, which is FAST. It commits pass-1-certified rows
  (theta 0.8, cf 0.9) without the exact check (= the exact loop on 5 of 16 prompts in kp2), and the map scored that
  setting a code KILL (1.63 % vs the 1.34 % line).
- The code-valid FAST point is top-2 at theta 0.95 (map 1.22 %, PASS): ~36 tok/s [P, Fit12's model]. The 90-130 publish
  line is x2.5-3.6 away.
- Fit12's lever 3 (rough, share, 14 clean HumanEval): Leap's switches add ~+5 % (pass-1 dense port x1.054; att_single
  x1.013, BETWEEN; the drafter port -3 ms per check). The 12 GB check is bytes-bound (~1.47 GB of SSD per check), so the
  12 GB lever is bytes per check (BYTES10), not kernels.
- Breakthrough + Fit12 own the line. Items 2, 4 and 9 carry over to 12 GB with unpriced pool effects; DRAFT17
is in.

NOT ON EITHER LINE (24 GB chat bytes; kept for 24 GB): R15 / R15b (Fri 00:30), the 0.4 gate, the chat route itself
(PASS, ~88 MB).
- R15 [K, Fri 03:45]: PASS by the amended lines. Flips 0.65 / 4.39 / 2.76 % vs the route's 0.71 / 4.30 / 2.73;
  paired p 0.83 / 0.84 / 1. Unsure rows at top-8 join the 24 GB chat route. Bytes [P, r15_price.txt]: x0.88 / 0.78 /
  0.73 (Cloud3 x0.93 / 0.81 / 0.82). R15b qualified (multi 2.76 <= 3.19) but the runner hit its 14-run cap: Saturday.

## RAPID SCREENS (Kamen Mon 10-26 ~14:30; RULES.md 'Rapid screening + your hardware only'). Read this first.
Protocol: every idea gets ONE screen of <= 10 min of machine time (load included) under `peer lock quiet --queue 10m`. Write
its PASS / KILL line in this table BEFORE the run, log the number after, then take the next idea whatever the result. Size to fit
(3 paired rounds, a few layers, a trace subset, 30-50 items, a microbench). Only a PASS earns a longer confirmation; adoption
gates are unchanged (exact modes stay bit-exact; adopting a non-exact mode is Kamen's call after an Eval-suite verdict).
Before every pacing sleep, queue your open screens as ONE detached runner (nohup, ppid 1, PAUSE-aware). Hardware: Kamen's
MacBook only (no external drives: the second-SSD idea is rejected). All 512 experts stay: no pruning, no merging.
Tags: [new] = not tried anywhere here as far as NewIdeas knows (check your own ledger, skip and say if done); [port] = measured on
another engine, untried on this one. Anyone may add a row; mark [taken: you, time].
| id | owner | idea (one line) | PASS / KILL (written first) | result |
|---|---|---|---|---|
| R-01 | Fit12 | CONF (a confirmation, not a screen; one detached runner): ALLWAIT + MOV + PFM on, C_m(n) re-derived from the new peaks; 5 prompts x 2.6K / 8.5K / 16K / 32K, each also after another prompt in the same process | PASS: every hash (logits, 256 tokens, kv, head) = C_m 51 + ALLWAIT on the same path, fresh = warm, peak <= 11.95 GB, decode >= +4 % at 8.5K. KILL: any mismatch or peak > 11.95 | **KILL by its line, on 32K memory only (Fit12 22:12, PLAN.md 'R-01 CONF RESULT')**: exact: every hash (logits, sha256, kv, head, lpre) = the C_m 51 reference at all 6 positions at 2.6K / 8.5K / 16K; fresh = warm. Warm speed: 8.5K C_m 60 +4.8 % (line +4 % met), 2.6K C_m 64 +7.7 %, 16K C_m 56 +2.7 %; test-arm peaks <= 11.81 GB. But the 32K C_m 51 REFERENCE reaches 12.24 GB clean at prompt 5 of 6 (lifetime 12.35): per-prompt growth in one process, outside MLX's active memory (+0.15 / 0.29 / 0.40 / 0.90 GB at 2.6K / 8.5K / 16K / 32K). NewIdeas 22:13: pre-existing, not a fold blocker; fold uses multi-prompt C_m 64 / 60 / 56 / 47; Fit12's 10-min growth screen (harness vs PFR vs engine) GO. **Eval-suite fold verdict, interim (45 of 239 rows per arm; runner 24 stopped Wed 08:34 after the night PAUSE froze chunk 4, no rows lost)**: tokens identical 44/45; agent/write16 (8,277-token prompt) differs from token 32 (late 0 / over 0 both arms; C_m 60 vs 51; ALLWAIT calls 65 vs 0); fold-off = the saved B12LL run on 45/45, fold-on 44/45. On peaks 12.00 GB (edit10 the only row > 11.95), off 13.15 (28 rows over); agent prefill -28 %, decode +3.6 %, math decode +8.9 %. **Flip held** until the cause is known: PREREG re-run per arm + a spec-off decode of write16 + the logit margin at token 32 (EvalSuite), and whether the fold changes MTP drafts by design (Breakthrough; asked Wed ~09:1x). **Breakthrough (Wed ~14:1x): bits change by design, only through ALLWAIT**: in the MTP path's keep-1 commit replays (layer 1 re-run alone on the kept row; 65 in write16) the shipped late rule never adds cold ranks 2-9 (stashed for layer 2, which the replay doesn't run, then cleared), and fold-on adds them. So the carried state differs after the first such replay, and a near-tie can flip; the verify itself is width-invariant (WV1-3). The 4-bit reference can't settle it (on write16 it leaves both arms' path at token 11). **Split (NewIdeas ~14:2x)**: exact part = R-41 (my call); ALLWAIT + per-prompt C_m (+4-9 % decode, closer to the full model) = a non-bit-exact default change -> Kamen's call on the 239-row task scores (PREREG rule 2; the EXACT line is expected to fail on near-ties). Decisive arm added to EvalSuite's write16 re-run: fold OFF + B12_ALLWAIT=1 (v3l) should give fold-on's tokens. **CORRECTION (Breakthrough ~15:0x): write16's late counter is 0 in both arms, so no replay dropped a rank and ALLWAIT gives the same slots and math there; the ALLWAIT story did not happen on write16.** Spec pass counts are equal on/off in all 44 other rows (26 long unaligned agent prompts), only write16 differs (55 vs 49): prompt-specific. MLX bm-64 gather bug ruled out (PFR chunks 512-padded, 8,704 x 10 rows = 64 x 1,360). Suspects: PFR on the MTP path (one 8,704-row chunk vs 4096s), the QSA rows target, MOV, C_m 60 vs 51; Breakthrough bisects them on write16's own prompt (R-41) |
| R-02 | Fit12 | KVC [new]: pre-allocate the QSA KV to n + max_new at prefill (no growth copies). Does it explain MOV64's ~0.5 GB? | PASS: peak -0.3 GB at 32K or 64K, hashes equal. KILL < 0.1 GB | **BETWEEN 20:1x (Fit12, PLAN.md 'Batch 2 results')**: qsa_rows_target cuts the 32K decode-phase clean peak 11.35 -> 11.23 GB (-0.12; PASS -0.30), exact (all hashes equal), speed unchanged. NewIdeas 20:22: free and exact, so it joins the ALLWAIT / MOV / PFM fold if R-01 passes |
| R-03 | Fit12 | WARM [new]: PFR leaves the pools empty for decode; fill them with the experts the last 64 prompt tokens used most (counts kept during prefill) | PASS: first 128 decode tokens >= +10 % faster at 8.5K, hashes equal (ALLWAIT path). KILL < +3 % | **KILL 20:1x (Fit12)**: oracle-warm 2nd decode vs cold, first 128 tokens 16.24 -> 16.64 tok/s (+2.5 %, line +10 %). No fill to build |
| R-04 | Fit12 | GAP [port, MoonShots gapclock.py]: does the GPU clock down during the 12 GB engine's SSD waits? One layer's expert kernels after 0 / 2 / 10 ms idle | lever if post-gap >= 1.2x back-to-back; KILL < 1.05x | **KILL 20:1x (Fit12)**: one MoE layer after a 10 ms idle runs 1.13x (line 1.20x); a 48-layer token after 2-10 ms gaps 1.07-1.10x = +0.5-0.65 ms per stall (< 1 % of a 60 ms step). No clock-down lever |
| R-05 | Leap | SD1 resized to the protocol: W vs U from the L1-20 checkpoint, <= 5 min of training each, same 32-row eval | PASS W/U >= 1.05 (then the 50-min run). KILL < 1.02 | KILL 19:58 (Leap, data/r05_report.txt): W / U 0.970 at b32 (CI 0.954-0.986), 0.976 at b16. Both arms fell below L1-20 (U 9.68 vs 10.23): 21 more steps of the recipe already cost ~5 %. The R_j weights pile onto depths already right |
| R-06 | Leap | DR as one <= 10-min microbench: 8-bit drafter + fused head top-k | PASS draft <= 5 ms at b16, tokens within 1 %. KILL > 8 ms or -3 % | KILL 19:16 (Leap, data/dr_screen.txt): best arm within 1 % = 4-bit drafter + two-stage top-16, 9.64 ms at b16 (8-bit 9.98, bf16 10.92). Fewer weight bytes barely help: the forward is launch-bound |
| R-07 | Leap | DR-v [port: FR-Spec 2025, applied to the drafter's head]: top-16 over the 32K most frequent tokens only | PASS head <= 1.5 ms at b16, accepted tokens >= 99 % of full. KILL < 98 % | KILL 19:16 (Leap): head 0.85 ms, but tokens x0.951-0.960 |
| R-08 | TrainLab | S10c-1: frozen M1a vs the plain router on b33's late + demand misses (CPU) | PASS M1a >= router + 10 points. KILL < +5 | **PASS Tue 19:48 (TrainLab REPORT 6471c98, data/s10b_fit_b33ld.txt)**: frozen M1a (S10b's own b32 models, no refit) on b33's late + demand misses at lead 2, shipped top-K rule, 4.8 reads per pass-layer: held-out 52.5 % vs router 33.7 % (+18.8), leave-one-out 54.3 vs 35.4 (line +10). Without interpolation M1a's top-16 catches 51.2 % with 4.34 reads vs the shipped lead-1 policy's 43.4 % with 5.23. Models exist for the 8 traced layers only. Next: BigMoE's engine A/B as ONE screen on those 8 layers (NewIdeas 19:51); ceiling at fixed reads = the SSD's idle time per pass (KP: <= 7 / 10 / 2 % tok/s for code / prose / edit) unless fewer wasted reads also cut bytes |
| R-09 | Leap | HCF [new, Leap's]: fused hyper-connection read / write at 16 rows (MTPLX's bit-exact 3-kernel read as two 8-row calls, and a 16-row port) vs the eager chain (~11 kernels per read). CA (Mon ~19:10): HC = 31-34 ms of the 143-150 ms 16-row check | PASS: >= 15 ms saved per 16-row check AND bitwise = the eager chain. KILL: < 8 ms. Fast but not bitwise: logged as a reference change (priced like NAX; BigMoE's / Kamen's call) | BETWEEN 20:52 (Leap, data/r09_screen.txt): exact 5-kernel read (down + inject as ONE matmul, MLX's own GEMMs) = bitwise over 96 reads, 24.37 -> 14.70 ms (-9.67). Elementwise fusion alone -1.8. MTPLX 2x8 read -13.85 but not bitwise. Next: R-09b (same-K-order GEMM, more threadgroups) |
| R-09b | Leap | HCF2 [new, Leap's; prior art: MTPLX calls the library's own GEMM template so its sums match]: the merged down + inject projection (M 16, N 324, K 10,240) runs on only ~11 output tiles; same per-output K order with ~4x more threadgroups (narrower N tiles, no split-K) | PASS >= 15 ms saved per check vs E AND bitwise. KILL < 11 ms | STEP 1 19:19 (Leap, data/r09b_screen.txt): build it. The 16-row merged GEMM is MLX's regular GEMM (bitwise = rows of a 32-row GEMM). Rule 9.67 + (9.87 in-chain - 2.5 floor) = 17.0 >= 15; a second estimate (K walk like the N = 4 inject) gives ~15.1. Knockouts are not additive (removing a stage lets the chain halves overlap). Building: MLX's K order by bitwise candidates, then BN 8 tiles; BUILD BETWEEN 20:40 (data/r09c_screen.txt): every arm bitwise (96/96 projections, chains = X5 = E). S4 (MLX's NAX split-K loop, 33 one-simdgroup tiles, loads 3 steps ahead) 9.82 ms per 96 reads = -14.68 vs E (X5 -9.81; the PASS line is 15). The replica alone -12.22; fused partitions lose. Prefetch keeps the bits, so R-13c's premise holds |
| R-10 | Leap | DCOMP [new]: compile the drafter (mx.compile of its per-layer blocks at block 16, the two-stage head, and the 16-step selector chain as one function); R-06 shows weight bytes are not its cost | PASS: draft <= 6.5 ms at b16 (from 9.64-10.92), tokens per check within 1 %. KILL > 8.5 ms | KILL 19:26 (Leap, data/r10_screen.txt): compiled drafter bitwise = eager but 10.50 ms at b16 (eager 11.24); 4-bit compiled 9.37 (tokens x0.994). Python builds the graph in 0.46 ms: the drafter is GPU-bound, not CPU-bound |
| R-11 | Leap | ONE-SUBMIT [port: Splash 1.0 'submits drafting, verification, acceptance and state updates together'; our A7 (+12.9 % on the 35B loop) is not yet on the 125B]: one GPU submission per cycle, acceptance decided on the GPU, no CPU round trip between draft, check, accept and commit | screen: in the CA harness, GPU idle time between the four stages at 16 rows. Lever if >= 8 ms per cycle; KILL < 3 ms | LEVER by its line 20:07 (Leap, data/r11_screen.txt): 48-layer idle ~11.5 ms per cycle = 2.9 after the acceptance read + 8.6 after the draft read (PLE staging 3.4 ms per check + 4 x check build, CPU-held). BUT the no-sync bound O beats S by only 1.63 ms of 5.75 at 12 layers (G1 0.24; SB, the check async/4 + commit alone, 1.36): the evals look CPU-bound (MLX's encoding). R-11b measures that |
| R-11c | Leap | SB [exact; from R-11]: mx.async_eval of h + midloop states every 4 layers in the 16-row verify forward, and of the cache states right after the acceptance read so the commit replay goes out alone (two lines) | PASS: >= 2 ms per 12-layer cycle AND bitwise. KILL: < 0.5 ms or any bit differs | **PASS 21:00 (Leap, TABLE_350)**: bitwise (drafts + tokens of 43 cycles + final caches, both contexts); paired medians S 51.86 -> SB 47.82 ms at 1K, 53.96 -> 50.09 at 8K per 12-layer cycle (-4.0 / -4.2 ms), 3x R-11's estimate. At 48 layers ~-11..-16 ms per check (the commit-alone part doesn't scale); a 48-layer engine confirmation is R-37 |
| R-12 | Leap | COMMIT-SELECT [new here; Splash snapshots GDN state]: keep the check rows' GDN states (all of them, or every 4th row plus a <= 3-row replay) and select the accepted one, instead of the 9.7 ms commit replay | PASS: commit <= 5 ms per check AND bitwise = the replay. KILL > 8 ms or not bitwise | KILL 19:45 (Leap, data/ca_commit.txt; TABLE_350 'R-12 step 1'): the 48-layer commit is ~4.7 ms at keep 10 (CA's 9.7 counted the PLE layer's replay x4). Select loses on bytes: all rows 1.81 GB (7.2 ms) or every 4th 0.45 GB + a 3-row replay (~4.3) vs a 3.3 ms GDN replay. Spin-off (exact): the 35 layers' replays are independent and launch-bound (~0.07 ms fixed each): one batched kernel ~-2.5 ms per check |
| R-13 | Leap | OCC [new]: occupancy audit of every skinny GEMM in the 16-row check (R-09 found the N = 4 inject matmul at ~83 us per call: one threadgroup walking K = 10,240). List each shape per layer (shared-expert gate, GDN b / a, router, HC, ...), time each alone vs its bandwidth floor, sum the excess over 48 layers, HC excluded (R-09b) | lever if the excess >= 10 ms per check; KILL < 4 ms. Fix = narrower N tiles with the same per-output K order (bitwise) | LEVER, from data (Leap 22:35, no GPU; TABLE_350 'R-13'): kern23's dense set at 16 rows = 29.0 ms over floors per check (shipped chunked 36.4 vs floor 7.4; in the check ~18 ms after overlap). Biggest: in_proj_qkv 7.3, o_proj+out_proj 5.4, lm_head 4.4, in_proj_z 4.0, q_proj 2.7; the skinny ones (shared gate, a/b) only ~0.9. L1-23's batch-invariant set already recovers ~21 (a reference change); a bitwise-vs-shipped fix waits on R-09b step 1 |
| R-13b | Leap | QMVW [exact half of R-13, family 3]: qmv_wide at M = 16 runs 4 vector groups, so weights are dequantized and read 4 times (shared gate/up/down, k/v, GDN a/b, shared-expert gate: ~4.7 ms vs 0.5 floor). One pass with 16 vectors; vecs_per_tg never enters the arithmetic | PASS: >= 2 ms saved per check (standalone, kern23's cold shapes and counts) AND bitwise vs the MLX 16-row call. KILL: < 1 ms or not bitwise || KILL 20:32 (Leap, data/r13b_screen.txt): NV4/8/16 all bitwise = MLX (transcription works; vecs_per_tg never enters the arithmetic), but one 16-vector pass is 2x SLOWER (10.17 vs 4.76 ms per check): MLX's 4 vector groups hit cache on the same small weights, and NV16 loses latency hiding. Best exact: NV8 -0.26 ms per check |
| R-13c | Leap | NAXP [family 1, only if R-09b's prefetch keeps the bits]: the R-09b prefetched-load move on the NAX qmm family (qkv, q_proj, lm_head: 6.96 ms vs 3.94 floor) | PASS: >= 1.5 ms saved per check AND bitwise. KILL: < 0.5 ms or not bitwise || KILL 20:56 (Leap, data/r13c_screen.txt): all 5 arms bitwise (MLX's qmm_t_nax replica, 16-row tiles, double-buffered weights), none faster: MLX 7.20 vs best C2 7.06 ms per check (-0.14); double buffering costs +0.8-1 ms. The quantized NAX kernel is bound by its dequantize / NAX work, not load latency, unlike R-09b's bf16 kernel |
| R-13d | Leap | SPK16 [family 2, the biggest exact piece]: non-NAX qmm_t_splitk at BM = BK = BN = 32 wastes half of each 32-row tile at M = 16 and writes bf16 partials + a separate sum launch (84 per check; 13.2 ms vs 2.7 floor). MLX's own template at BM = 16 with the partition sum fused in, same per-output k-step order | PASS: >= 4 ms saved per check AND bitwise. KILL: < 2 ms or not bitwise || PASS 20:57 (Leap, data/r13d_screen.txt): MLX's own qmm_t_impl at BM 16 + mx.sum(axis=0) (MLX's own reduction) is bitwise = MLX's 16-row call on z and o/out_proj and saves 4.52 ms per check (13.15 -> 8.64; z 5.67 -> 3.73, o/out 7.48 -> 4.90). The replica at BM 32 is also bitwise; double buffering adds nothing. Confirmation next: in CA's real check, with S4 |
| R-14 | Fit12 | HC12 [port of R-09 to the 12 GB engine]: knock out bt_b12's HC reads / writes to price them per decode pass (1-3 rows), then Leap's bitwise X5 merge (down + inject as one matmul; MLX sums each output in the same K order whatever N) | PASS >= +3 % decode at 8.5K with equal hashes. KILL < +1 % | **PRICED 3.2 %, DROPPED 20:1x (Fit12; NewIdeas 20:22)**: every HC8 read + write run twice: 17.07 -> 16.54 tok/s at 8.5K (pairs 2.8 / 3.6 / 3.1 %), all hashes equal. Even a free HC would give ~3 %, and X5 isn't bitwise on HC8: not worth a bitwise HC8 rewrite now |
| R-15 | Leap | DSK [new]: the drafter on fast NON-bitwise kernels. R-10 shows its ~500 kernels are GPU time (~21 us each; ovh27: small matmuls 16-25 us each), not CPU. The drafter is outside exactness (it only proposes tokens), so any faster kernel is allowed: split-K / NAX skinny GEMMs, fused attention over the context, fused norm + matmul. Step 1: kernel-time breakdown of one b16 draft by kernel class (~2 min); step 2: swap the top class or two | PASS: draft <= 6.5 ms at b16 (from 10.5), tokens within 1 %. KILL: > 8.5 ms after swapping the top two classes || STEP 1 20:28 (Leap, data/r15_screen.txt): 4-bit draft 10.93 ms = dense 4.66 + head 2.27 + attn 0.93 + conv 0.84 + select 0.75 + norm 0.20 + rest 1.60 (bf16: dense 5.99). GPU-bound (eval 21 % CPU). Dense is slow for R-13's reasons: qmv_wide x4 vector groups (q/k/v/o) and non-NAX split-K (MLP). Step 2 after R-13b's NV16 timing; at floors dense ~2 + head ~1.3 -> ~7.3, so a PASS also needs the small ops fused; STEP 2 BETWEEN 22:35 (Leap, data/r15b_screen.txt): 4-bit drafter at b16 9.93 -> 7.93 ms (gate|up merged on NAX qmm + down via D2), tokens x0.9955 (b32 10.31, x0.9984). D2 alone is bit-identical: 8.47 ms. The rest (head 2.3 + small ops ~4.3) blocks 6.5 |
| R-16 | MoonShots | W24-ec [exact, after W24-sp step 2]: entropy-code the MSB-3 codes on the SSD copy (your floor: 2.72 of 3 bits) and decode on the GPU, which idles ~60 % of a 24 GB pass (b28). Screen = a GPU rANS / tANS decode microbench on real code bytes from a few layers, many independent streams per expert | PASS: decode >= 25 GB/s of codes on the M5 GPU AND code bytes <= 2.80 b per weight over the sampled layers (~-7 % bytes / token, exact, stacks on W24-sp). KILL: < 10 GB/s, or > 2.90 b | **PASS (MoonShots Tue 19:45, scripts/w24ec.py, data/w24ec.txt)**: interleaved rANS (10-bit table per layer, 65,536-symbol chunks per simdgroup) on 96 real experts (L0 / L24 / L47): 2.7231 b/weight all-in (-9.23 % code bytes), GPU decode bit-exact at 46.2 GB/s. Caveats: ~ms per-chunk latency (prefetch pipeline needed), variable record sizes (MSB miss ~116 blocks vs 126), pools stay decoded. CONFIRMATION (MoonShots 20:00; data/w24ec_replay*.txt, data/w24ec_ch*.txt): with W24-sp in-kernel killed, the live stack is W24-ec + 11-bit at load (pools unchanged): 24 GB 117.4 -> 104.0 MB / token (-11.4 %; W24-ec alone -7.3 %); 12 GB C_m 51 / C_l 20: v3 206.0, bt_b12 pieces 210.2 -> 182.3 (-11.5 % / -13.3 %; W24-ec alone 190.6, -7.5 % vs v3; my replay's absolute v3 is below Fit12's K1 256.4, relative cuts transfer). Chunk sweep (L24, bit-exact): 16K 2.773 b, 32K 2.748, 64K 2.736; ENGINE CHOICE 32K (1,024 dependent steps per lane, same 116-block record as 64K), 16K if demand-miss latency binds (-10.8 %). Decode must batch a layer's misses (thousands of chunks in flight: 27-46 GB/s); ~1 GB of misses per 24 GB check = ~25 ms of GPU decode inside the SSD-bound idle. EXACT 24 GB STACK RESTATED (20:04): RAM +1.25 GB (C_m 159 / C_l 44) + W24-ec + 11-bit at load = 94.4 MB / token (-19.6 %, ~150 tok/s SSD cap), replacing the old 96.6. 12 GB re-checked with STRICT LFU (data/w24ec_12gb_strict.txt): relative cuts unchanged (-11.7 % vs v3, -13.5 % vs bt_b12, W24-ec alone -7.7 %, at 16/10 and 4/3); strict absolute v3 at 51 / 20: 359.6 (16/10) / 316.5 (4/3) MB / token **24 GB PORT KILL (BigMoE b42, Wed 16:12-16:19, ~/BigMoE/data/b42/ab.txt; logged by NewIdeas 18:4x)**: MoonShots' GPU rANS decode of real MSB-3 codes paced to 6 GB/s beside the engine (6.00 GB/s held, 0 mismatching bytes, every sha equal) cost the engine -7.1 % tok/s (code 34.50 -> 32.06, prose -9.1 %, edit -5.3 %; verify +9-12 ms per pass): the decode competes with the verify GPU work, so a ~7 % cost eats most of what -11.4 % bytes could buy. May still pay on the more SSD-bound 12 GB line (R-19). |
| R-17 | MoonShots | T3-45 [pricing, ~5 min]: TRL-3 (3INST, ~3.07 b, 0.50x MSB-3 error, K2 passed) as a NON-exact 45 GB fast mode: experts' bytes per 16-row check x ~0.72 in your par45 model, at today's and at the all-levers non-byte time | lever if the tok/s ceiling rises >= 20 % at the all-levers non-byte time (then: GPU encoder cost + an Eval-suite verdict, Kamen's call); KILL < 10 % | **KILL (MoonShots Tue 19:4x, scripts/t3_45.py, data/t3_45.txt)**: bytes model x0.72 gives +11.8 % today, +17..+22 % at all levers (x0.68: +14 % / +20..+25 %), but resident experts at 16 rows are decode-bound: B10(b) 3INST trellis 1.257 ms/layer vs 3c 4-bit 1.240 at 0.45x the bytes, so -0.5 % today / -0.9 % all levers. Revives only with a trellis kernel at <= ~0.72x of 4-bit's time |
| R-18 | BigMoE | PAR24 [new for 24 GB; par45 priced streams only at 45 GB]: do lockstep parallel answer streams (SoT-style, independent KV / GDN state each) cut SSD bytes per token at 24 GB? Streams of the same answer share topic, so a miss read for stream A is soon a hit for B. CPU replay through MoonShots' wall24.py LFU at the 24 GB split: k = 1 / 2 / 4 streams interleaved per token, taken from different segments of one long trace (same answer), plus a control of streams from unrelated prompts | PASS: 2 same-answer streams cut bytes / token >= 20 % vs 1 stream (then 4 streams, and MoonShots folds it into its stack). KILL: < 8 % | **KILL Tue 19:41** (BigMoE, scripts/par24.py, data/par24/par24.txt, 34 s CPU): 2 same-answer streams +3.4 % bytes / token at 16-row checks (116.1 -> 120.0 MB; wall24's own LFU baseline 117.4), -2.3 % at 4-row checks; 4 streams +7.4 / -3.1 %. Control (unrelated prompts) +50 % (2) / +130 % (4): same-answer streams share experts, but only enough to break even with one stream, since the pools (147 / 46 per layer) must hold k working sets. Capped LFU (= run_pool where no call overflows; 4 x 16 rows needs up to 343 experts in one call) |
| R-19 | Breakthrough | W24-12 [port pricing of MoonShots' exact packs to the 12 GB line]: price W24-sp (11-bit scale / bias) + W24-ec (rANS MSB-3 codes, R-16 PASS: 2.723 b, bit-exact GPU decode 46 GB/s) on bt_b12 from your own pass trace: MSB pieces 2,211,840 -> ~1,900,544 B (-14 %), LSB 720,896 unchanged, plan 4433333333; predicted decode gain = SSD-busy share x byte cut, at 2.6K and 8.5K | PASS: >= +8 % decode at 8.5K (then the port, after MoonShots' bytes / token confirmation). KILL: < +3 % | **BETWEEN (Breakthrough, SCOREBOARD fe244c1)**: SSD = 56 % of the 8.5K AR step (~0.10 ms/MB); at -13.3 % bytes the gain estimate is +6.3 % (bandwidth) to +8.1 % (upper), under the +8 % line; rANS decode ~7.2 ms/token eats it unless overlapped. Waits for BigMoE's decode-while-waiting screen |
| R-20 | Breakthrough | KV8M [after dx run 3]: does 8-bit KV actually save memory at long context with PFR on? (dx ran lane-on, no PFR: kv8's prefill peak was HIGHER, 12.75 vs 11.90 GB at 32K.) One 64K prompt, PFR on, kv:64 vs bf16: prefill and decode peaks | PASS: kv8 lowers the 64K peak by >= 0.3 GB (room for ~3 more C_m slots). KILL: < 0.1 GB | **KILL (Breakthrough, fe244c1)**: kv:64 RAISES the 64K peak, 12.69 vs 11.80 GB (prefill transient +1.3 GB; after decode -0.54 GB) and decode is -26 %. 8-bit KV is not a 12 GB lever. Also AW-AR: ALLWAIT on the plain AR path -3.6 % at 8.5K (+3.8 % at 2.6K) -> folded on the MTP path only |
| R-21 | Leap | GDNB [exact spin-off of R-12]: the commit's GDN replay is launch-bound (~0.07 ms fixed per layer; keep1 / keep15 = 0.64) and the 35 layers' replays are independent: ONE batched gated-delta launch over stacked states, each layer's arithmetic unchanged | PASS: commit -2 ms or more per check AND bitwise. KILL: < 1 ms | KILLED Thu 18:16 (Leap, ~/Leap/data/r21_screen.txt, TABLE_350 'R-21 GDNB step 1'): even an ideal batched replay (one launch per op on pre-stacked inputs) saves <= 0.1 ms GPU + 0.22 ms CPU build per commit; bitwise equal. The 35 replays are at their bytes floor (220 MB of fp32 state read + written, 1.19-2.04 ms at keep 1-15, ~185 GB/s), not launch-bound: R-12's 0.07 ms/layer was an 8-layer-slice artifact. Naive stacking costs +1.1 ms |
| R-22 | TrainLab | M1aB [CPU, after the export]: M1a's read-budget curve on b33 late + demand misses: catch rate vs prefetch reads per pass-layer (top-K over non-resident, lead 2), against the shipped policy's 43.4 % at 5.23 reads. At 24 GB the SSD is 90-98 % busy, so wasted prefetch reads (precision ~29 %) cost real bytes; fewer reads at equal catch is the part of S10c NOT capped by the SSD's idle time | PASS: M1a matches 43.4 % at <= 3.0 reads per pass-layer (BigMoE then also runs its A/B at that budget). KILL: needs >= 4.5 reads | **PASS 20:02 (TrainLab, REPORT 'R-22 M1aB'; a read-out of the S10c-1 run's budget sweep, data/s10b_fit_b33ld.json, no new run)**: held-out, M1a matches 43.4 % at 2.81 reads per pass-layer (K=12: 43.8 % with 2.85, 1.60 wasted, precision 37.5 %) vs the shipped policy's 5.23 (4.06 wasted, 16.3 %); LOO 2.66 vs 4.88. -46 % reads, -61 % wasted at equal catch. A/B budget: K=12 (~2.85 reads); K=16 = 51.2 % at 4.34 |
| R-23 | MoonShots | W24-cx [exact, CPU, added + run Tue 20:04]: code the MSB-3 codes / LSB bits below order-0 with context a decoder has: (xc) against the best other expert of the layer at the same position, (gu) up given gate, (gc) per-group variance class (+3 b / 64), (lc) LSB bit given the MSB-3 code. scripts/xc_entropy.py, layer 24, 32 experts | xc PASS median <= 2.40 b, KILL >= 2.65; gu PASS <= 2.60, KILL >= 2.69; gc PASS <= 2.62, KILL >= 2.68; lc PASS <= 0.90, KILL >= 0.97 | **ALL KILL (20:04, data/xc_entropy_L24.txt)**: order-0 2.7267; xc 2.7267 (gain 0.0000 b: the experts share no code structure, not an upcycled model), gu 2.7232, gc 2.7539 (worse than order-0), lc 0.9870 b. W24-ec's ~2.72 b is the lossless floor for these codes |
| R-24 | MoonShots | ROWTRIM [exact, oracle bound, added + run 20:06]: read experts only for the rows a check will accept + 1 (an approximate resident-only pass would predict the stop row) instead of all 16 drafted rows | PASS oracle <= -15 % bytes / token at 24 GB; KILL > -7 % | **KILL on the available traces (data/rowtrim24.txt)**: wall24 replay -1.0 / -1.3 % (accept 8 / 10, v3 split; exact stack -0.5 / -1.3 %), because the true-token traces reuse the 'rejected' rows' experts in the next check. Real rejected drafts route differently, so the true bound lies between -1 % and ~-20 % (union(11) vs union(16)); pricing it needs a capture of real 16-row drafted checks with every row's routing (none exists: BigMoE's b33 engine verifies ~1.1-1.25 rows per token, Leap's par32 uses true tokens). Parked unless someone logs 16-row routing anyway **PRICED ON REAL DRAFTS 20:32-20:35 (Leap's ~/Leap/data/r24_routes.npz: 20 consecutive DFlash2 16-row checks, layers 0-11, one code sample, ctx 184, accepted mean 9.40; data/rowtrim_leap.txt)**: rejected-only experts = 28.2 % of the per-layer union (22-32 % by layer, falling with depth), 35.0 % of the (row, expert) pairs; 44 % of them return in the next check's kept rows, 75 % within 5. 24 GB strict LFU (C_m 159, warm-up 5 checks): reading only kept rows -30.3 % misses (C 100: -30.0, C 220: -28.8): the ORACLE PASSES the -15 % line. Mid-check cut from layer k with perfect stop-row prediction (layers uniform): k = 12 -> 75 % reachable (union -21.1 %, pairs -26.2 %, 24 GB misses -22.7 %), k = 24 -> 50 % (-14.1 / -17.5 / -15.2), k = 36 -> 25 % (-7.0 / -8.8 / -7.6). Open: how well acceptance can be predicted at layer k (logit lens on the verify forward's hidden states; Leap) |
| R-25 | MoonShots | DYNK [NON-exact, bytes side; added + run Tue 20:09]: drop a rank-8 / rank-9 expert only when its top-10-renormalized gate weight < tau (instead of fixed top-8, which fails strict). scripts/dynk24.py, wall24 LFU at the exact stack (C_m 159 / C_l 44, W24-ec + 11-bit at load) | PASS: some tau keeps >= 60 % of fixed top-8's byte cut while dropping <= 40 % of its gate mass (then a TrainLab strict / KLD screen there); KILL: every tau reaching 50 % of the cut drops >= 70 % of the mass | **BETWEEN (data/dynk24.txt)**: the top-10 gate is flat (ranks 8-9 average ~6 % each), so thresholds separate little. Exact stack 94.4 MB / token: tau 0.05 -6.3 % (20 % of rank 8-9 uses, 1.7 % mass), tau 0.06 -13.2 % (47 % of uses, 4.7 % mass = 51 % of top-8's cut for 39 % of its mass), tau 0.07 -19.8 % (8.5 % mass), fixed top-8 -25.7 % (70.2 MB / token, 12.1 % mass). If anything, a TrainLab strict screen at tau 0.06 (82.0 MB / token, ~173 tok/s SSD cap); NewIdeas' call |
| R-27 | MoonShots | LAYSZ [exact, CPU replay, added + run Tue 20:13-20:16]: static NON-uniform per-layer pool sizes (same RAM): per-layer LFU miss curves on the wall24 traces, a Lagrangian picks each layer's C_m / C_l on calibration sets, scored on the held-out set (leave one of the 4 sets out). scripts/laysz24.py | PASS >= -8 % bytes / token held-out vs uniform; KILL < -3 % | **CORRECTED 20:28 (MoonShots): 24 GB PASS, 12 GB KILL.** My first numbers were inflated by a replay flaw: wall24's LFU keeps experts staged beyond C (when a check's union exceeds C) as FREE cache, so starved layers (C 1-59) and the 12 GB uniform 51 (< the ~66-expert 16-row union) held more than their RAM. STRICT replay (pool trimmed to C after every check; scripts/laysz24.py LAYSZ_POL=lfus, data/laysz_*_lfus.txt), held-out LOO means: 24 GB (uniform 159 / 44, 94.4) 16/10 -10.3 % (bulgarian -15.8, chat -16.8, code -4.1, math -4.7; allocation 79-319, layers 0-2 ~2x, 15-16 / 31-32 ~80), 4/3 -8.9 % (-13.6 / -13.6 / -4.0 / -4.3): PASS on the mean line, weak for code / math. 12 GB (uniform 51 / 20): 16/10 -2.2 %, 4/3 -1.7 %: KILL, agreeing with Fit12's PLA KILL (the earlier -20.7 % was the flaw). Unaffected: every uniform 24 GB number (C_m >= 147 > union) and R-16's relative cuts **ENGINE A/B KILL (BigMoE b36, 20:5x; ~/BigMoE/data/b36/ab.txt)**: per-layer sizes at the shipped 147 / 46 budget (4/3, strict prior -8.6 %): SSD bytes / token -1.5 % (A/A noise 1.0 %), tok/s -0.6 %; late + demand misses -3.3 %, on-time hits +9.8 %: the engine's JIT prefetch + JOINT already cover what bigger early-layer pools would hold, so the LFU-only replay does not carry. Lesson: replay-only pool levers need an engine A/B before they count |
| R-30 | MoonShots | UNIQ [id was R-29; renumbered by NewIdeas 20:32, R-29 is TrainLab's LAYSZ-ON] [NON-exact, bytes side; added + run Tue 20:30; kin: Lynx batch-aware expert selection]: in a 16-row check, drop a row's rank >= R0 expert only if NO other row of the check routes to it (and its renormalized weight < tau): a single-use, low-weight expert costs a whole read for little. scripts/uniq24.py, wall24 LFU at the exact stack (159 / 44 > union) | PASS: >= 15 % bytes cut at <= 3 % gate mass dropped per row-layer; KILL: nothing beats DYNK's bytes-per-mass by >= 1.5x | **PASS (bytes; data/uniq24.txt)**, 94.4 MB / token base: ranks >= 6 & w < 0.06 -17.6 % at 2.26 % mass; ranks >= 8 any weight -23.6 % at 4.0 % (fixed top-8: -25.7 % at 12.1 %); ranks >= 6 & w < 0.08 -35.3 % at 6.3 %; ranks >= 4 & w < 0.08 -40.7 % at 7.5 %; ranks >= 4 any -50.3 % (47.0 MB / token) at 11.0 %: 5.2-7.9 % bytes per % of mass vs top-8's 2.1 and DYNK's 2.8. With R-27 (-10 %) the aggressive end reaches ~42 MB / token (~336 tok/s cap) IF quality holds. Quality = TrainLab (strict / KLD on real checks at 2-3 settings, renormalized vs not). Note: a row's experts then depend on the other drafted rows, so outputs depend on the drafts (non-exact, deterministic per run); Kamen's call after a verdict ADDENDUM 20:34 (data/uniq24_mass.txt, UNIQ_MODE=mass): check-level rule = drop expert e from the whole check when its SUMMED weight over the check's rows (ranks >= R0 uses) < theta. Same frontier as single-use where they overlap (theta 0.08: -41.0 % at 7.6 % mass) but it reaches further: ranks >= 4 theta 0.06 -19.0 % at 2.4 %; 0.10 -51.0 % at 11.0 %; 0.12 -53.5 % (43.9 MB / token) at 12.0 % = top-8's mass (top-8: -25.7 %); ranks >= 2 theta 0.16 -65.7 % (32.4) at 19.6 %. With R-27 (-10 %) theta 0.12 is ~39.5 MB / token, under 350's 40.6, IF quality holds at 12 % mass (top-8 at 12.1 % failed strict; here the dropped mass is concentrated on single-use low-weight experts, a different distribution) 12 GB / 4-row checks (C 51 / 20, data/uniq_mass_51_20_r4a3.txt): weaker, theta 0.06 -17.3 % at 4.2 % mass, 0.08 -41.7 % at 13.9 % (3-4 % bytes per % mass vs 5-8 at 16 rows): with 4 rows most experts are single-use, so the rule's edge grows with check width; 24 GB 16-row checks are where it pays [RETRACTED 20:51: my bug, Leap's dumps list each row's top-10 in argpartition order, not rank order, so 'ranks >= R0' was applied to the wrong experts; file renamed data/uniq_realdraft_INVALID_unsorted.txt] CAVEAT 20:39: on Leap's REAL drafted checks (20 x 16 rows, layers 0-11, ids only) the weight-free single-use rules cut less of the per-check union than on the true-token traces: ranks >= 8 8.4 % vs 16.2 %, >= 6 18.8 % vs 30.0 %, >= 4 30.8 % vs 41.5 % (real union 89.3 vs 68.9: rejected rows add top-rank experts the rule never drops). So the bytes numbers above are optimistic by ~1.3-1.9x; one code sample, no cache, no weights 45 GB LINE (all resident, data/uniq45.txt; true tokens, no cache, 48 layers): per 16-row check, ranks >= 4 theta 0.06 union -11.6 % / pairs -4.9 % (2.4 % mass); 0.08 -29.2 / -12.3 % (7.6 %); 0.10 -38.5 / -16.4 % (11.0 %); 0.12 -40.8 / -18.0 % (12.0 %); ranks >= 2 0.16 -55.5 / -27.3 % (19.6 %); top-8 for reference: union -17.4 %, pairs -20 % at 12.1 %. So at 45 GB it cuts bytes (union), not work (pairs); same real-draft caveat REDO ON REAL DRAFTS, RANKS SORTED BY GATE (20:50; Leap's R-32 dump: 628 DFlash2 checks, 24 code samples, layers 0-11, gates; scripts/uniq_real.py, data/uniq_real.txt): bytes cuts close to the true-token replay, dropped mass ~1.3-1.4x higher. 24 GB strict LFU (per-sample consecutive checks, C_m 159 / C_l 44), all 16 rows: theta 0.06 -17.0 % at 3.4 % mass; 0.08 -37.0 % at 10.0 %; 0.10 -49.0 % at 15.4 %; 0.12 -51.1 % at 16.7 %. With the oracle row trim (R-24): trim alone -28.3 %, trim + theta 0.08 -54.0 % (11.6 % mass of kept rows), trim + 0.06 -39.7 %. 45 GB union per check-layer 87.1 -> 75.7 / 60.7 / 51.0 / 49.0 at theta 0.06 / 0.08 / 0.10 / 0.12 |
| R-33 | MoonShots | LFUT [id was R-32; renumbered by NewIdeas 20:39, R-32 is Leap's STOPPRED] [exact, CPU replay, added + run Tue 20:36]: tune the engine's LFU at 24 GB: decay d per check (today 0.99) and row-count scores (rows using e in the check) vs presence. scripts/lfut24.py, exact stack 159 / 44, 16/10 and 4/3 | PASS >= -5 % at both shapes; KILL < -2 % | **KILL (data/lfut24.txt)**: today's LFU is at the optimum: best -0.4 % (d 0.98, 16/10); d 0.95-0.999 within -0.4..+0.4 % at 16/10 and +0.2..+4.9 % at 4/3; row-count scores worse everywhere (+1.6..+5.4 %) |
| R-31 | TrainLab | UNIQ-Q [quality side of R-30; NON-exact; after R-26]: drop a row's rank >= R0 expert only when no other row of the 16-row check uses it (and weight < tau). Strict / KLD at MoonShots' check-level rule (20:3x addendum: drop expert e from the whole check when its summed weight over the rows using it, ranks >= 4, is < theta): theta 0.06 (2.4 % mass, -19 % bytes), 0.08 (7.6 %, -41 %), 0.12 (12.0 %, -53.5 %: head-to-head with top-8's failed strict at the same mass); renormalized vs not. Rows depend on the other drafted rows: use real drafted 16-row checks if your harness has them, else 16 consecutive true tokens as a proxy (say which) | PASS: the theta 0.08 point passes the strict gate that fixed top-8 (12.1 %) failed; then EvalSuite task tests (Kamen's bar). KILL: even the 4.0 % point fails it. Note the per-check union budget (XShare / OEA kin) failed IFEval by 5-7 earlier | **NO DATA 20:55 (TrainLab, REPORT 'R-31 UNIQ-Q', 2846e87)**: the suite's reference harness computes v3's tiers row by row; at 16 rows it costs >= 0.27 s per position-arm (>= 3.5x 4-row) and the first sequence didn't finish in 9.5 min. The run outlived its lock by ~1 min (swap 5 -> 14 GB; Leap's guard stopped its screen); run_locked.sh now kills the process group. NewIdeas 20:57: next = (b) a <= 10-min profile of the 16-row path with RSS + swap watched; if a cheap fix brings 16-row near 4-row cost, theta 0.08 first; else wait for a real-drafted 16-row quality harness (no 40-min lock on the true-token proxy) |
| R-32 | Leap | STOPPRED [exact; decides R-24's mid-check row cut]: predict each check's stop row from the target's own hidden states at layer 12 / 24 (logit lens on the drafted next token, or a small linear acceptance probe; fit on half the checks, score on the rest), on real DFlash2 16-row checks with accept lengths. Baseline that must be beaten: the same cut at layer 0 from the drafter's confidence alone (pre-check truncation, dfl2_trunc-style). Rows from the first predicted reject are dropped from layer k on; kept rows stay exact; a wrong reject only costs tokens | PASS: at layer 12, realized union cut >= 10 % at <= 2 % fewer accepted tokens per check AND >= 5 points better than the layer-0 baseline at that token loss. KILL: < 5 % union cut at <= 2 % token loss, or no better than the layer-0 baseline || BETWEEN by the letter 20:43, KILL in substance (Leap, data/r32_screen.txt; 628 real checks): layer-12 probe 11.1 % whole-check union cut at 2.0 % token loss vs layer-0 drafter-features probe 11.0 % (oracle 23.7 %). The layer-12 logit lens alone gives 0 %. The signal is the drafter's own confidence, so truncate the DRAFT (rows leave all 48 layers): ~14.7 % union / ~18.8 % pairs at ~2 % tokens. Routes + gates for MoonShots: data/r32_routes.npz |
| R-34 | MoonShots | RAM24 [exact; 'RAM freed elsewhere', your RAM curve ~8 MB/token per GB at 24-28 GB]: audit every non-pool RAM user of the 24 GB engine (dense weights, drafter, MTP head, n-gram cache, KV + GDN state, MLX / Metal / Python runtime) from BigMoE's config and a memory trace, and price exact reclaims. Candidates nobody priced at 24 GB: the drafter at 4-bit (it only proposes tokens; R-06: tokens x0.994 at 45 GB), the MTP head's experts sharing the expert pools, runtime / allocator slack | PASS: >= 0.75 GB reclaimable without changing output (>= ~-6 % bytes/token). KILL: < 0.25 GB || **PASS on paper for the DFlash2 (350-plan) config, slack pending (MoonShots 20:44; data/ram24.txt)**: replay at the exact stack: ~6.5 MB / token per GB reclaimed (-7 % per GB). New exact reclaims: DFlash2 drafter at 4-bit instead of bf16 (1.2 -> ~0.34 GB; R-06 tokens x0.994; output unchanged) +0.86 GB -> 88.7 MB / token (-6.1 %); MTP head off when DFlash2 drafts (dense 0.10 + 64-slot expert pool ~0.20) +0.30; embed_tokens on the SSD (0.36 GB at 4-bit, one row per token; BigMoE keeps it resident today) +0.36; HC lossless pack (LEDGER MEM, slower) +0.28 -> all four 1.80 GB -> 82.7 MB / token (-12.5 %). Today's MTP-drafted engine: embed + HC pack ~0.64 GB (-4.2 %), under the line without slack. Biggest unknown = allocator / activation slack (LEDGER M24: MLX peak 2.5-4.5 GB over active after load): asked BigMoE for a v3 memory trace (active / peak / cache by phase, live set at the decode peak, phys footprint, n-gram hot cache accounting); 1 GB of slack would be -7 % more **TRACE IN (BigMoE b36, ~/BigMoE/data/b36/ram24.json; shipped v3 147 / 46, code prompt 0) -> PASS (exact reclaim >= 0.75 GB)**: after load active 21.06 GB; spec decode 400 tok peak 24.36, active 21.35, cache 0.74; AR 128 tok incl. prefill peak 22.16; spec prefill alone peaks 24.26: the ~2.2 GB over AR is spec125's PREFILL path, not verify / decode. So the biggest exact reclaim is that transient: give spec125 the AR prefill path or chunk it, bit-exact by shas (spec == ar2 already holds), ~2.1 GB, plus embed_tokens on the SSD 0.318 GB (untied; lm_head 0.318 stays) -> ~2.4 GB = pools +~14 % at ~6.5 MB / token per GB (replay; R-27 showed replay pool levers can shrink in the engine, so the engine A/B decides). Sizes as loaded: pools 17.07, HC mixers 1.26, embed 0.318 + lm_head 0.318, GDN / attention projections 1.31, routers 0.063, PLE key 0.052, MTP experts 0.156, shared experts 0.117. The shipped config already peaks over 24 GB (24.36), so the prefill fix is first needed just to sit inside the budget |
| R-35 | Leap | TRUNC2 [exact; follow-on of R-32: the stop signal sits in the drafter, so cut the DRAFT, dfl2_trunc-style, with R-32's layer-0 logistic probe on drafter features instead of drafter p alone]: on 628 real checks the probe cuts ~14.7 % of the union / ~18.8 % of expert pairs at ~2 % fewer accepted tokens (drafter p alone: 9.0 % at 1.6 %). NewIdeas' pricing: experts ~58.7 of ~156 ms per cycle and pairs-proportional -> ~-11 ms per cycle, ~+5 % tok/s today; ~+10 % once experts sit near the NAX floor (~37 ms of a ~58 ms cycle). Real-loop A/B vs block 16, tau from samples 0-11, scored on 12-23 | PASS: >= +4 % tok/s on the real loop, output identical. KILL: < +1.5 % (or no better than DFL2E's earlier truncation result) | **RE-SCOPED + PARKED 20:4x (Leap / NewIdeas)**: the probe version is dead: DFL2E's running-product rule (Sat, dfl2_trunc.py; SpecDec++ / AdaEDL kin) already does better: t 0.3 = -26.7 % rows at -1.8 % tokens (probe: -18.8 % at ~2 %). Cost model with pairs-proportional experts: ~+9 % tok/s at 45 GB. Never decoded: it needs a loop that verifies DFlash2 16-row checks with variable row counts. Parked until such a loop exists; BigMoE asked whether v3 can take it as a 10-min A/B at t 0.2 / 0.3 / 0.4 |
| R-36 | Leap | INCHECK [confirmation of R-09b S4 + R-13d]: both swaps inside CA's real 12-layer check (GatedResidual's 16-row read -> S4; split-K matmuls -> R-13d's D2 via a quantized_matmul shim), bitwise logits + every cache state vs the unpatched engine, paired timing at 1K and 8K | PASS: every bit equal AND in-check saving >= 80 % of the standalone sum (19.2 ms per 48 layers, so >= 15.4). KILL: any bit differs || PASS 22:12 (Leap, data/r36_screen.txt): every bit equal (logits of 43 checks + all cache states, S4 / D2 / both, 1K and 8K). Per 12-layer check base 36.09 -> both 31.51 ms (1K), 38.30 -> 33.79 (8K): -4.60 / -4.55, x4 = -18.4 / -18.2 ms per 48-layer check (standalone sum 19.2). In-check factors S4 0.97 / 0.91, D2 1.43 / 1.30. SB (R-11c) is separate and should stack |
| R-37 | BigMoE | SB-24 [exact; port of R-11c to v3's 16-row-capable verify forward]: mx.async_eval(h, *Q._midloop_state_arrays) every 4 layers, and mx.async_eval(*cache states) right after the acceptance read. At 24 GB the SSD caps GPU-side savings (KP: <= 4-6 ms per pass), but an earlier start may also feed the read pipeline. 10-min A/B, same prompts and seeds | PASS: >= +2 % tok/s AND all shas equal. KILL: < +0.5 % or any sha differs | |
| R-38 | BigMoE | PF24 [exact; from MoonShots' R-34 PASS]: the shipped 24 GB config peaks at 24.36 GB (b36 trace), all from spec125's PREFILL transient (~2.2 GB over AR). Run spec prefill through the AR path or chunked (spec == ar2 already holds by shas), and put embed_tokens on the SSD (0.318 GB). Then give the ~2.4 GB to the pools and A/B | PASS: peak <= 23.9 GB with all shas equal, AND bytes / token -5 % or better at the bigger pools. KILL: still > 24 GB, or bytes better than -2 % only (replay said ~-15 %; R-27 showed replay pool gains can vanish in the engine) | **bytes PASS on MLX peak; footprint re-size pending (BigMoE b41b: runs Tue 23:45-23:52, A/B read Wed 08:16; data/b41b/ab.txt, ram24.txt)**: step 1 BIGMOE_MTP_PF_GROUP=48 (grouped prefill of the MTP layer only): spec prefill MLX peak 24.17 -> 21.97 GB (= trunk only), same prefill time, all shas equal; BigMoE makes it the shipped default. S = step 1 + pools C_m 159 / C_l 49 (18.41 vs 17.07 GB) vs shipped A, 4 domains x 400 tok, ABBA, every sha equal: SSD MB/tok 353 -> 334 (-5.4 %, noise 1.1), gate wait -4.6 %, late + demand -7.2 %, tok/s +2.7 % (noise 1.6; code / prose / edit / math +0.8 / +14.4 / -3.5 / +2.9 %); MLX peak S 23.79 (code prompt 0: 23.60, shipped 24.36). **Budget defined (NewIdeas, Wed ~09:1x): 24 GB = the process's LIFETIME PEAK phys_footprint (RUSAGE_INFO_V4, as the 12 GB line), not the MLX peak.** S's footprint is already 24.25 GB at the end of decode (A 22.91; MLX cache 0.76 GB + ~0.8 GB non-MLX on top of active), so b41c probes A0 (old shipped) / A1 (+ step 1) / S re-sized to ~23.85 GB lifetime peak on code prompt 0, then the bytes A/B A1 vs S (when the lid opens). Footnote: n-gram rows come through numpy memmaps (clean page cache, outside footprint, tiny); expert reads are F_NOCACHE. **FINAL (BigMoE b41c / b41d, Wed 15:3x-16:0x): A0 (old shipped) lifetime peak 25.36 GB (over budget in the spec prefill); A1 = + MTP_PF_GROUP=48 = 23.16 GB, exact, ADOPTED. Pools re-sized to fit (C_m 152 / C_l 47, peak 23.63 GB, exact): bytes / token -0.9 % (noise 1.6) -> pool half KILL; pools stay 147 / 46. Every shipped decode number ran at 147 / 46, so the 24 GB tok/s figures stand** |
| R-39 | Leap | STACK [confirmation before hand-off]: S4 + D2 + SB together in CA's real 12-layer check, bitwise vs base at 1K and 8K | PASS: every bit equal AND paired saving >= 6.5 ms per 12-layer cycle (R-36's 4.6 + R-11c's 4.0, minus overlap margin). KILL: any bit differs || PASS 22:25 (Leap, data/r36b_screen.txt): every bit equal (logits of 43 checks + all states; SB, S4D2, all3; 1K and 8K). Per 12-layer check base 36.46 -> all3 29.98 ms (1K, -6.63; sum of parts 6.80) and 39.65 -> 31.76 (8K, -7.73; 7.69): x4 = -26.5 / -30.9 ms per 48-layer check, 18-20 % of the check. SB here is checks-only (2.1-3.2); with the drafter (R-11c) 4.0-4.2. Hand-off: ~/Leap/HANDOFF_EXACT16.md |
| R-40 | Leap -> EvalSuite | REF2 [prior art, cited Fri 18:3x (PRIOR_ART.md): Thinking Machines Lab, 'Defeating Nondeterminism in LLM Inference' (H. He, Sep 2025: batch-invariant RMSNorm / matmul / attention with fixed split sizes; repo batch_invariant_ops); vLLM VLLM_BATCH_INVARIANT, SGLang --enable-deterministic-inference (CUDA only); LLM-42 (arXiv 2601.17768). Ours: the MLX / Metal port and the MLX-specific breaks] [Kamen APPROVED decision 4, Tue 22:40]: build the new exact reference for 45 GB from the faster kernels: batch-invariant dense (qmm16c), NAX experts (qmmg25), plus the bit-identical S4 / D2 / SB, as one patch module with a load-time self-test. Step 1 (Leap): batch invariance (M = 1 result == every row of M = 16, bit for bit) and the 12-layer check time vs today. Step 2 (EvalSuite): same-scores verdict REF2 vs today's reference on its math / code / instruction sets, AR decoding, Leap's module applied | Step 1 PASS: bitwise batch-invariant AND check time <= today - 40 ms (48-layer equivalent). Step 2 PASS (Kamen's same-scores rule): no task score differs beyond the suite's noise; only then does REF2 replace today's reference | **INV0 (Leap, Wed 15:48; TABLE_350 'R-40')**: 16 one-token AR steps vs one 16-row verify, CA's 12-layer slice, 1K context. Today's kernels: every row diverges at layer 0's GDN block (logits equal 0/16). With REF2's patches (< 16-row matmuls and HC reads padded to 16; qmm16c for big 4-bit, MLX's 16-row calls for the rest; S4 reads): layers 0-2 are bit-identical for all 16 rows (GDN incl. the recurrence, PLE, MoE gather experts, HC). First divergence: layer 3's attention (the first QSA layer), all rows. Open: QSA's indexer block selection (may depend on the total key count) and / or the SDPA kernel (vector for 1 query, full for 16). Next (Leap, KERNELS): localize inside QSA, then a batch-invariant attention path or a 16-row padded decode for attention only. **INV1 (Leap, Wed 16:08): REF2 is batch-invariant bit for bit**: 16 one-token AR steps vs one 16-row verify, 12-layer slice at 1K: logits of all 16 rows and all 37 records per forward identical, with < 16-row matmuls / HC reads padded to 16 (qmm16c for 4-bit g64 N >= 1024, MLX's 16-row call otherwise, S4 HC reads) and attention with < 16 rows padded to 16 through the verify path (QSA cache trimmed back); GDN and MLX's gather experts are invariant as shipped. So spec decoding against REF2's own AR output is exact by construction on this slice. Next: scripts/ref2.py (patches + experts_nax + SB + a load-time self-test), R-40's timing screen (PASS -40 ms per 48 layers) with the invariance re-checked under experts_nax, then the Eval-suite hand-off. Cost: REF2's AR step runs at 16 padded rows, so the Eval suite's REF2 AR runs are slower than today's; INV1 16:07: with attention padded to 16 rows (+ QSA cache trim) every record and all 16 logits rows bit-identical AR vs verify. **R-40 PASS 16:28 (Leap, data/r40_screen.txt)**: ref2.py (+ experts_nax via MTPLX_FUSED_GATE_UP=1) batch-invariant (logits 16/16, 0 differing records) and REF2 + SB vs today per 12-layer check 39.60 -> 26.78 ms (1K) / 41.88 -> 27.36 (8K): x4 = -51.3 / -59.6 ms per 48-layer check. Hand-off to EvalSuite: ~/Leap/REF2_HANDOFF.md; STREAMED 16:52 (Leap, data/ref2_stream_check.txt): after EvalSuite's catch (streamed pools, no fused gate|up) ref2 runs the experts as two qmmg calls on BigMoE's pools; in EvalSuite's exact REF harness (C=192) all 145 records x 48 layers and 16 logits rows bit-identical AR vs verify; experts 48 per AR token. EvalSuite green-lit **STEP 2 PASS (EvalSuite R-40, runs ended Wed 20:56; docs/RESULT_ref2_2026-10-28.md, 84ebfba; table checked by NewIdeas 21:2x in results/report_ref2_vs_ref.r2.txt)**: REF2 vs ref.r2 (today's stock, re-decoded because step 0 found 5 of 6 rows identical to the 10-20 ref): math 90.0 vs 82.0 % (+8.0, McNemar p 0.12, lost 0 / won 4, all four ref.r2 cap hits), IFEval 87 vs 90 % (-3.0, p 0.38, 4 / 1), HumanEval+ 91.7 = 91.7 %; floor yardstick +6.0 / -2.0 / +3.3; also passes vs the 10-20 ref. Flags (information): 4 loop rows vs 1 (one is REF2's own, HumanEval/93, which the baseline fails too); IFEval agreement 95 vs the floor's 97 %; the standard table's truncation / loop line says FAIL (it fails for ref.r2 vs ref too). Counters exact on all 210 rows; one frozen code version per arm. Plain AR 11.65 vs 15.50 tok/s (x0.75: every token runs as 16 padded rows); peak 35.6 vs 35.0 GB. **Decision 4's condition is met.** Baselines from now on (NewIdeas 21:2x): REF2 rows for 45 GB verdicts, ref.r2 for stock-anchored ones; results/ref (10-20, BigMoE 7dfe9d1+dirty; 10 of 210 rows differ on today's code, scores equal) is history only. **STEP 3 routed 21:2x to Leap + BigMoE: the 48-layer streamed A/B of REF2 + SB with drafting at 45 GB. PASS: real 16-row checks bit-identical to REF2's AR AND >= 40 ms saved per 48-layer check vs today's shipped exact check (also prefill s, tok/s on code + math). KILL: < 20 ms (the slice gain does not survive the engine). GPU after MTP Head's D-09.**  **STEP 3 first attempt (Leap Thu 01:57-02:06, TABLE_350 'R-40 STEP 3'; stopped by run_locked's 9-min limit, no verdict file): NOT exact in the live loop**: 0 / 48 logits rows bit-identical to in-process ref2.ar_step on both prompts; tokens = the suite's REF2 AR rows 256 / 256 on HumanEval/68 but 199 / 256 on prealgebra/1640 (contradicts the 16:52 stream check: C 192, P 1024, no SB, no drafter). In this streamed engine a 16-row check takes ~3.9 s (16 x 10 ids > MAXK 40 -> BigMoE's _sync_call: per-layer id sync, synchronous SSD loads, scratch gather, clear_cache); REF2 saved 80 / 46 ms, inside the noise. STEP 3x (pre-registered 02:12, ~3 min: V0-V4 localize the break; warm checks priced, PASS >= 40 ms / KILL < 20 ms) waits for the end of Kamen's full stop. REF2 is not adopted until the break is explained (NewIdeas 02:1x). **STEP 3y (Leap 17:46-17:48, ~/Leap/HANDOVER.md, TABLE_350)**: the break is a stale PF1 staging entry, not REF2 and not the stock kernels. A commit keep >= 5 replays the PLE layer (zero-based 1) alone with > MAXK ids; that leaves pending[2], and the next forward's layer 2 builds its scratch from buffer 0 after layer 0 restaged it, i.e. on layer 0's expert weights (measured: the next prefill differs from record 14, layer 2's MoE; a gated call clears it). EvalSuite: no suite verdict exposed (PF1 runs only on the AR ref family). BigMoE's fix 7ac3192 (kick-time drop + buffer layer tags); Leap's 3y-fix screen (runner 21, data/r40s3y_screen_fix.txt), then the step 3 re-run. **STEP 3k PASS 00:29 (Leap, ~/Leap 23abf98, TABLE_350 'R-40 STEP 3k'): REF2 is bit-exact in the live DFlash2 loop on 6 / 6 prompts: 1,477 / 1,477 tokens == EvalSuite's REF2 rows, 441 / 441 logits rows bitwise, Part D states exact; the control (SDPA classes OFF, 694) breaks (216 / 256, 14 / 54) as in 3q.** **STEP 4a (Leap 02:13, data/r40s4a_screen.txt): the full-length drift is a keys-past-512 effect: for a 16-row q, MLX 0.32.2 takes its full NAX attention kernel, whose row bits depend on the call's key length (rows whose AR call has N or N+1 keys match, N+2 and up do not); first differences in QSA layers 3 / 7 / 11 / 23 / 35. STEP 4b next: the class map + a fix test (pad both calls' keys to a 512-bucket). NewIdeas 02:33: if it holds, REF2b = the padded fix, EvalSuite regenerates the REF2 rows and re-runs step 2's same-scores test (inside Kamen's approval); keep-the-AR (~8 attention calls per chunk) only as the fallback.** **STEP 4c PASS 02:49 (Leap, data/r40s4c_screen.txt, ~/Leap 3c48315): key buckets (every 16-row attention call with >= 512 keys runs at the next multiple of 512): 702 160 / 160 kept rows bitwise at positions 520-680, HumanEval/68 220 / 220 across 512; the control without buckets breaks (2 / 160). Merged into ref2.py as REF2b (CFG key_buckets default on; off reproduces REF2's AR); a merge microtest is queued. EvalSuite's plan (05:49): (1) regenerate the 65 of 210 core rows that reach position 497 (~72 min AR), copy the other 145 with a spot check; (2) decision 4's same-scores test REF2b vs ref.r2; (3) the R-56 re-run on REF2b, every token + every kept row bitwise. Keep-the-AR fallback estimated (not measured) at ~45-55 ms per check (+15-20 %); REF2b's own padding cost is measured in S45c.** Fixes in ref2.py: the commit runs in the verify phase (MTPLX's attention_gate lowers sigmoid differently outside decode_verify), and rows that straddle MLX's 128-key SDPA mode switch are recomputed in the AR's class. S45 (a measurement): the same exact loop at C 192 gives code 25.1 / math 19.2 tok/s, median cycle 320 ms (drafter ~13, check + commit 266-397), kept 8.27, lifetime peak 36.3 GB. **Caveat 01:46 (EvalSuite R-56 interim + Leap 01:4x): exact only to 256 tokens so far: the full-length audit has 16 / 21 prompts equal; 5 leave the stored REF2 rows at 1-ULP near-ties (gap 0.125; one exact tie at intermediate_algebra/607 token 669) at positions 498 / 515 / 573 / 724 / 910; ar_bad 0 on the first 48-97 rows everywhere. EvalSuite's bitwise diagnostic (~02:30) decides loop drift vs a moved reference; if the loop, Leap bisects the first bad check.** **STEP 3 re-run (3z + 3x / 3y; Leap 18:48, ~/Leap/data/r40s3z_screen.txt, r40s3y_screen.txt; TABLE_350 'R-40 STEP 3x/3y/3z'; bdf9f5e): exactness FAIL, speed PASS-size.** Speed: on BigMoE's default check path (PF1 only for prefills) REF2 + SB saves 137.5 ms per 48-layer check live (median; /68 160.3, prealgebra/1640 114.7; line 40), 166.9 warm; live checks cost 1.2-1.9 s at C 176 (450-880 expert misses per check, io 130-230 ms, 48 id syncs + scratch gathers); tok/s code 8.6 -> 9.6, math 3.3 -> 3.3. Step 3's 3.9-4.3 s checks were EvalSuite's forced PF1 staging, not the engine path. Exact: REF2 checks equal REF2's AR after a prefill and after full accepts (/68: 256 / 256 tokens, 48 / 48 rows) but break after PARTIAL-accept commits (math, kept ~5: 40 / 48 rows, tokens 55 / 256); REF2's AR never runs that path, and the stock arm leaves ITS AR at the same token 55, so the suspect is MTPLX's commit path: the PLE layer's lone replay at S = keep rows takes the small-S GDN path (fused conv + norm, not bitwise; L1-31). Next: STEP 3w bisection (~2 min, after 3y-fix). R-21 GDNB KILLED (Leap): the commit's 35 GDN replays sit at their bytes floor. **LIVE-LOOP EXACTNESS PASS (Leap, Fri 00:24, STEP 3k, ~/Leap 23abf98):** with ref2's defaults (commit in the verify phase: MTPLX's PLE-gate sigmoid differs outside decode_verify; SDPA 128-key classes: MLX's vector SDPA switches mode at 128 keys; R-50 slot calls), REF2 is bit-exact inside the live DFlash2 loop: 6/6 prompts, 1477/1477 tokens == results/ref2 rows, 441/441 logits rows bitwise, Part D states exact, control (SDPA classes off) breaks. (The small-S GDN suspect above was wrong.) S45 at C 192: code 25.1 / math 19.2 tok/s exact, cycle 320 ms, misses 413-779 per check = 111-194 ms SSD. |
| R-41 | Breakthrough | FOLDX [exact; from the fold verdict's write16 mismatch, Wed ~14:2x]: 'fold-exact' = MOV + PFR on the MTP path + R-02 + the FR-Spec fix, with C_m 51 and ALLWAIT off (ALLWAIT is the only fold piece that moves bits). Bitwise screen vs shipped B12LL on 2-3 long prompts (8.5K / 16K / 32K, each after another prompt in the same process), plus its lifetime-peak footprint on the long agent rows (fold-on peaks 12.00 at C_m 60-64, shipped 13.15 in the eval's multi-prompt chunks; estimate ~11.1) | PASS: every hash (logits, kv, state, tokens) equal to shipped AND lifetime peak <= 11.95 GB -> shipped default on NewIdeas' word (fixes shipped's 12 GB overshoot with identical tokens). KILL: any hash differs | **Must include write16's own prompt (Breakthrough ~15:0x): bisect fold-exact, then PFR only, then rows target only, in a scratch bt_b12 copy via BT_DIR; live bt_b12 / v3m untouched**. **Bisect DONE (Breakthrough ~16:0x, data/w16/, 9 fresh-process arms on write16's prompt): the culprit is PFR on the MTP path.** ON tokens: fold-exact, PFR only, PFR + C_m. OFF tokens: rows target only, MOV only, ALLWAIT only. Mechanism: shipped s_moe 4096 splits 8,276 prompt tokens into 4096 + 4096 + 84 rows; PFR's balanced grid makes one 8,276-row chunk; the sorted gather runs short calls (the 84-row tail) through another kernel, so the last prompt rows' bits differ -> a near-tie flip at token 32. Earlier PFR A/Bs never had a small tail (282 / 3,560 rows). Fix under test: B12_PFR_GRID=keep (the 4096 grid; bitwise by construction, keeps the memory saving, loses the big-chunk prefill speed, which becomes a non-exact option for Kamen). NewIdeas' calls: if the AR path's PFR also differs on write16, the AR default goes to PFR-keep (restores exactness); the Eval suite holds runner 24 until the rebuilt fold passes. **PART 3 PASS (Breakthrough ~16:5x, SCOREBOARD ac04dd8, data/fx/)**: AR PFR-keep = PFR off at 8,474 and 65,000 tokens (every hash; prefill 28.2 / 269.5 s vs balanced 18.9 / 208.1; prefill peaks 9.46 / 11.25 GB vs balanced 9.27 / 11.08, PFR off 11.05 / 13.54); balanced differs at token 7 on write16. Fold-exact (MOV + PFR-keep on MTP prefills > 4096 + rows target, C_m 51, no ALLWAIT) is bitwise to shipped on p2k / p8k / p16k / p32k in one process and on write16; lifetime peaks 11.09 / 11.09 / 11.09 / 11.46 GB vs shipped 12.07 at 16K and 13.03 at 32K; same speed. **New bt_b12 defaults (NewIdeas' calls): B12_PFR_GRID=keep, B12_FOLD=1 exact pieces only, engine v3m (bitwise = v3l, adds drain()). Opt-ins for Kamen (non-exact): the balanced grid (AR prefill -31 % at 8.5K, -21 % at 65K vs PFR off; vs keep, the new default: 28.2 -> 18.9 s = -33 %, 269.5 -> 208.1 s = -23 %) and ALLWAIT + C_m per prompt (+4-9 % decode, old fold interim: b12llfon vs b12llfoff decode tok/s +3.6 % agent (29 rows), +8.9 % math (16 rows), recomputed by NewIdeas 18:4x from results/*/decode.jsonl ssd.dec_s)** **CONFIRMATION PLAN R-41C (NewIdeas 21:2x, sent to EvalSuite for its PREREG)**: A, the exact default: b12llfx (31b63af defaults) on runner 24's 45 rows, the 29 agent rows in ONE process. PASS: 45 / 45 token-identical to b12llfoff (= b12ll on all 45) AND lifetime peak <= 11.95 GB in every process; a differing row is re-decoded with b12llox (old envs, new code) to split code from drift. B, decision 5: b12llff (+ B12_FOLD_AW=1 B12_FOLD_CM=1 B12_PFR_GRID=balanced; Breakthrough c39b9a9: reproduces the old fold-on) on all 239 rows, its first 45 in chunks alternating with A (same-day speed A/B, information). Baseline b12ll's core rows; any core row where b12llff differs is re-decoded with b12llfx, which replaces b12ll there if they differ. PASS (same scores): R-40's rule 1 on math 50 / IFEval 100 / HumanEval+ 60 AND lifetime peak <= 11.95 GB in every process (the old fold-on had one row at 12.00). **R-41C RESULT (EvalSuite, rows ended Wed 23:07; docs/RESULT_r41c_2026-10-28.md, d165d3d): part A PASS**: fold-exact (b12llfx) token-identical to b12llfoff (= b12ll) on 45 / 45; one-process peaks 11.51 GB (math 16) and 11.58 GB (agent 29, the creep test; old fold-off reached 13.15 in a 17-row process); decode x0.993, prefill x0.970 (unpaired). **Part B cannot pass: memory over the line** (fold-fast b12llff: 13.20 GB math 16, 12.23 GB agent 29). Math: PFR resizes the pools C_m 51 -> 64 on short prompts (6.0 -> 7.34 GB, ~the 1.3 GB gap), then ~0.1-0.2 GB per row creep (fold-exact creeps the same way, lower); agent = the old fold-on's own footprint over a longer process. Tokens = old fold-on 45 / 45, = b12ll 44 / 45 (write16). Speed A/B (ABBA, information): decode x1.062, agent prefill x0.731, SSD 449.6 -> 404.2 MB / token (-10 %). B's 194 core rows still run Thu 07:30 (NewIdeas 23:2x: kept; a memory fix that keeps the tokens keeps the scores). Decision 5 on hold until fold-fast fits. **B call (NewIdeas Thu 15:4x)**: option (a) + a memory follow-up, no restart. EvalSuite's runner 27 decodes B's other 194 core rows on the c3f957cba6 pin (scores; PREREG amendment 2, 8778a3a). Breakthrough's FFX screen (fold-fast + fix, C_m capped at 60 to 8.6K; SCOREBOARD 0050c97; PASS tokens = b12llff and peak <= 11.85 GB). Then EvalSuite (GO 15:4x): (1) b12llff2 = fold-fast + fix + the capped points on runner 24's 45 rows in the same two processes, PASS 45/45 = b12llff AND <= 11.95 GB (only on an FFX PASS); (2) fold-exact + fix on the same 45 rows, PASS 45/45 = b12llfx AND <= 11.95 GB. Bits-neutral, so B's scores carry over. **FFX PASS (Breakthrough 16:13, SCOREBOARD eb4625a)**: fold-fast + B12_FRS_NOCYCLE with C_m capped at 60 up to 8.6K: tokens = b12llff on 8 / 8 math rows; lifetime peak 11.68 GB (line 11.85); 8-row creep 0.227 vs 0.821 GB unfixed; speed not comparable (Kamen's game was on the GPU). b12llff2's env is with EvalSuite for R-41D (1). **R-41C part B DONE 19:16 (EvalSuite, results/r41c_B.txt, report_b12llff_vs_base_r41c.txt; docs/RESULT_r41c_2026-10-28.md)**: scores PASS, memory FAIL, so B is not PASS. Fold-fast (b12llff) = b12ll on 210 / 210 core rows: math 86.0 = 86.0 %, IFEval strict 87.0 = 87.0 %, HumanEval+ 91.7 = 91.7 %, McNemar 0 / 0, p = 1, no drift; agent 28 / 29 identical (write16 from token 32, not exact). Memory over 11.95 GB in all 10 processes (13.20 / 12.23 / 13.10 / 14.94 / 13.73 / 13.61 / 13.37 / 13.90 / 15.60 / 15.09 GB; each starts at 9.67 and creeps to a last-row start of up to 14.73 GB; 214 of 239 rows over), on c3f957cba6 (no R-45 fix). Speed on the 45 paired rows vs fold-exact: decode x1.062 (18.57 -> 19.88 tok/s), agent prefill x0.731, SSD 449.6 -> 404.2 MB / token. Breakthrough swapped bt_b12 to ce6d4c63d2 (fix on) at 19:18; R-41D starts with b12llff2, which inherits this score PASS if its 45 rows = b12llff and <= 11.95 GB. The ~20 MB / request native residue (R-45S; v3o fix after R-45T) is not in ce6d4c63d2. **R-41D (1) b12llff2 FINAL 21:20 (EvalSuite, results/b12llff2/decode.jsonl, tools/r41d.py): tokens PASS, memory NOT PASS.** Tokens 45 / 45 = b12llff (agent 29 in one process, then math 16). Lifetime phys_footprint: agent process 12.47 GB (over 11.95), math 11.73 GB (under; R-41C's b12llff 13.20: the fix + cap took 1.47 GB off, the math creep mostly gone, last row starts 11.05 vs 11.96). The agent peak is a transient inside agent/write14 (n_prompt 1,945): start 11.65, sampled max 12.37, lifetime 12.47, back to 11.68 at the row's end; fold-exact jumps the same in that row (b12llfx +0.68, b12llfx2 +0.72 GB) but sits ~0.9 GB lower; without the spike the process would still touch 12.00 (write16's own max). Agent creep left: 9.67 -> 11.80 at the last row's start (b12llff 12.00). (2) b12llfx2: 28 / 29 agent rows identical so far, peak 11.52; its math process queues behind the 45 GB jobs. **NewIdeas 21:5x: decision 5 not settled. Next (Breakthrough, behind 45 GB work in the GPU queue): v3o (the read-buffer leak, ~0.6 GB over 29 rows) + a fold-fast C_m point for prompts under ~4K that budgets write14's ~0.7 GB prompt-reading spike (or a smaller prefill group there); then one agent-process re-run (b12llff3). PASS: 29 / 29 = b12llff AND lifetime <= 11.95 GB.** |
| R-42 | Fit12 (account-2 slack, orchestrator 15:47) | GROWFIX [exact, memory only; from the fold run: the SHIPPED 12 GB setup peaks at 13.15 GB on the eval's long agent prompts in one process (28 of 45 rows > 11.95), from the growth outside MLX that Fit12's GROW found Tue (+0.2-0.3 GB per prompt, levelling off after 4-5)]: between prompts, call macOS malloc_zone_pressure_relief(NULL, 0) (ctypes); separately, mx.clear_cache() + any trim the C++ ext offers. 5 long prompts (c32k or the eval's agent rows) in one process, footprint after each prompt, tokens hashed | PASS: growth over prompts 2-5 cut by >= 70 % with identical tokens. KILL: < 20 % (then it is live memory, i.e. a leak hunt, not a trim) | **KILL (Fit12 Wed 19:48; ~/Fit12/PLAN.md 'R-42 RESULT', data/scr/gfix/{s,t}.txt)**: 4 x c32k in one process on the 31b63af defaults: growth 0.152 GB stock vs 0.147 trimmed (-3 %), every sha equal, same time per prompt. The trim empties MLX's 0.07-0.10 GB cache, but the next prompt takes it back; malloc relief frees < 0.5 MB, so the 0.62-0.86 GB outside MLX is not allocator slack. Creep proper ~0.03 GB per prompt (part of G is prompt length); lifetime peak 11.443 GB < 11.95. Fit12's optional c32k_1 x 4 run declined (NewIdeas 21:2x): R-41C A runs the 29 agent prompts in one process and measures creep at scale. |
| R-43 | BigMoE | DFL24 [exact; NewIdeas Wed 16:2x; the 24 GB line's biggest open lever]: the shipped 24 GB engine drafts with the MTP head (~3.5 tokens per 2-7-row pass) and reads ~355 MB / token (b41b); with DFlash2 block-16 checks the S10 replays put v3 at ~116 MB / token. Pricing screen from measured parts, no build: (a) one 16-row verify pass in this engine at 24 GB (GPU ms, SSD MB, gate wait) on DFlash2-like rows, e.g. replayed D-02b drafts or 16 rows of real text; (b) kept per check from D-02b (code 9.57, math 7.00; prose / agent from MTP Head's D-02a when it lands); (c) drafting 12.5 ms (7.9 with R-15) -> predicted tok/s per domain vs shipped A (b41b: code 25.1, prose 19.1-21.8, edit 28.4, math 19.9). If PASS: rough hours and risks for the build (DFlash2 at 4-bit in RAM, the R-35 variable-row rule, Leap's exact-16 hand-off where it applies) | PASS (build it): predicted >= +25 % on code and math, and >= 0 % on prose. KILL: < +10 % on code | **KILL (BigMoE b43, Wed 16:32-16:34, data/b43/price.txt)**: the gate row holds <= 92 expert ids (~9 rows), so a gated 16-row check would be a build; BigMoE swept fixed widths instead (traced, n-gram drafts on copy prompts). Per pass on code text, 3.76 -> 6.74 rows: gate wait 61 -> 97 ms, GPU 29 -> 44, commit 7.6 -> 13.9, expert reads 533 -> 856: each extra row costs ~12 ms of gate wait (~108 reads), ~5 ms GPU, ~2 ms commit, and rejected rows cost what kept rows do. At 16 rows with D-02b's kept: code +7..+9 % or -9..-8 % (two fits), math -1..+1 % or -16..-15 %. Break-even kept per check: code ~8.9-10.5, math ~7.0-8.3; +25 % on code needs ~11-13 kept of 16. **NewIdeas' error: my premise compared a replay's 116 MB / token with the engine's rusage 355; the same replay gives ~122 for the shipped 4-row engine.** So at 24 GB wide checks pay only with a drafter keeping ~12+ of 16 (the D98 track) or with more RAM (45 GB) |
| R-44 | BigMoE (MoonShots' formats) | REPACK24 [a confirmation build: M-01 and M-03 both passed; NewIdeas Wed 16:4x]: one re-pack of the 24 GB expert records. (a) M-01's fixed-width scale / bias fields (sign + 2-bit exponent with escape + 7 raw mantissa bits; b as a 3-bit residual), a plain unpack at load with no rANS: exact. (b) M-03's plane-separated records (top-2 planes / plane 3 / LSB) + a tier-2 path in the expert kernel + per-slot plane state and upgrade reads, behind a switch: non-exact. Check free disk first (stop at 110 GB free) | (a) PASS -> default on NewIdeas' word: every sha equal AND rusage SSD bytes / token -2.5 % or better; KILL: any mismatch. (b) PASS -> to the Eval suite's strict verdict, then Kamen: rusage bytes / token -12 % or better at equal pools with tok/s not worse; KILL: < -6 % | **(a) PASS 17:18 (BigMoE, ~/BigMoE/data/b44/ab.txt, 700e969; one quiet lock 17:01-17:08: A1 S1 S2 A2, prompt 4 of each domain, 400 tokens, one binary ext/build_m01, equal pools C_m 147 / C_l 46, 17.07 GB)**: every sha equal in all 4 runs (loader bitwise on L0 / L23 / L47, 96 experts each; the converter checks every expert); rusage SSD bytes / token -4.8 % (353.1 -> 336.2 MB; A1 vs A2 0.1 %), gate wait -5.0 %, SSD busy -4.1 %, tok/s +2.0 % (code +2.0, prose +2.7, edit +2.2, math +1.2); lifetime peak 23.10 GB (shipped 23.16). records3m (68 GB): MSB part 132 -> 125 blocks, LSB 47 -> 44; s3 = sign + 3-bit exponent window + 7 mantissa bits, b3 = 2-bit exponent offset to s3 + mantissa (its sign is always opposite s3's), b = sign + 3-bit window + mantissa. Deviation from the row: b is coded alone, since each tier pool reads one part (a cross-part residual would add only ~0.5 %). **NewIdeas 17:22: default on the pre-registered lines (shipped env + BIGMOE_M01=1); records3 stays on disk while R-48 checks whether the 12 GB engine can read records3m. (b) GO, with M-07's LP1 as a second level behind the same switch** (a rank-1 use whose LSB is not resident runs on the tier-3 path; that LSB miss reads nothing, no insert; rank 0 still reads): one A/B at equal pools, A P L L P A on b44's prompts, A = the M-01 default, P = + POM R2 4, L = + POM R2 4 + LP1. P: the lines above. L vs P: PASS a further -6 % or better rusage bytes / token with tok/s not worse; KILL a saving under 3 %. Log the share of rank-1 uses served without the LSB (replay 26 %) and of rank >= 4 uses at 2 planes (20 %). On a PASS each goes to the Eval suite's strict verdict, then Kamen. **(b) b45 DONE 19:25 (BigMoE, ~/BigMoE/data/b45/ab.txt, ce9a5e8; A P L L P A, equal pools 17.07 GB, prompt 4 of each domain, 400 tokens): P BETWEEN, L BETWEEN.** P vs A: rusage bytes / token 342.2 -> 277.3 MB (-19.0 %) but tok/s -1.9 % (A1-A2 noise 3.0 %), gate wait -5.8 %; L vs P: bytes -4.5 % more, tok/s -0.6 %; L vs A: bytes -22.6 %, tok/s -2.5 %. Shares: rank >= 4 uses at 2 planes 31 % (replay 20 %), rank-1 uses without the LSB 37.5 % (replay 26 %), upgrades 30 % of 2-plane reads. Why no speed: SSD busy ms / token fell only 28.7 -> 26.8 (-6.5 %): seg1 1.43 MB + seg2 0.70 MB pieces replace one 2.05 MB read, upgrades sit on the critical path, and 2-plane prefetch reservations were 1.5-1.6x the demand 2-plane reads. Footprint equal. **NewIdeas 19:29: POM stopped at 24 GB (BigMoE's call, agreed); judge byte cuts by SSD busy time from now on (b44's M-01: bytes -4.8 % -> busy -4.1 % with the same reads; b45: +65 % reads, my fit, eat most of -19 %).** Thread-local bounce buffers (pom_read :266, read_expert :298) leak per retired GCD worker: BigMoE's pool fix GO now (exact). |
| R-45 | Fit12 (account-2 slack; design: Breakthrough Wed 23:2x) | CREEPMAP [exact, memory only; from R-41C: both arms creep ~0.1 GB per row with long decodes (fold-exact 10.12 -> 10.95 GB over rows 2-9, then -0.90 at row 10; fold-fast 11.46 -> 12.61, then -1.14 at row 13). Fold-exact's 11.51 / 11.58 leave 0.37 GB, which a longer process (a chat) could cross; and the creep is memory an exact bigger expert cache could use (fold-fast's C_m 64: -10 % SSD MB / token, x1.06 decode, for +1.34 GB)]. Suspects, both outside MLX: (a) the n-gram hot-row LRU (MTPLX_NGRAM_HOT_MB, default 1024): numpy rows filled by decode-sized gathers, each a view into its whole fetch batch (qwen4_exp.py ~5570), so real bytes can exceed the cap; (b) cyclic garbage freed only by gen-2 GC (NewIdeas: less likely, since fold-fast's PFR runs gc.collect() at every prefill and still creeps, and R-42 ran gc.collect() per prompt). Screen (<= 10 min; env + harness only, so Breakthrough's code freeze holds): one process, fold-exact (bt_b12 c3f957cba6 defaults), 8 of the suite's math rows decoded to EOS as the suite does; per row: footprint, MLX active, hot-cache bytes (rows and pinned batch arrays), then gc.collect() and the footprint again; tokens hashed (= EvalSuite's b12llfx rows) | PASS: one suspect explains >= 80 % of the row-start creep -> the fix (exact; Breakthrough, after the freeze): a real cap on the hot cache (e.g. 256 MB, rows copied out of their batches; check the hit rate) or gc.collect() between requests; then re-derive the C_m points of fold-exact and fold-fast from the bounded footprint; if (a), BigMoE checks the 24 GB engine's hot cache too. KILL: neither does -> C_m points that budget the creep (short prompts ~C_m 52), giving back most of fold-fast's math decode gain | **PASS (b) Thu 00:37 (Fit12; /Users/kamen/Fit12/PLAN.md 'R-45 RESULT', data/scr/creep/c1.txt)**: cyclic garbage, not the hot cache. 8 R-41C math rows in one process on b12llfx (tokens = the suite's on all 8), gc.collect() after each row: creep C 0.885 GB (suite 0.833 over the same row starts); gc frees B 0.757 GB = 86 %; n-gram hot cache A +0.030 GB = 3 % (56K rows of 100 B); 0.098 GB unexplained. Each row's garbage is one mtplx.frspec_draft._full_vocab_head.<locals>._FullVocabDraftHead: a class defined per call (frspec_draft.py:42, built at :228 with the FR-Spec draft head), in a cycle via super()'s __class__ cell, whose closure cells keep the pruned head's 4 MLX arrays (0.095 GB) alive until a gen-2 GC (MLX active 9.427 -> 9.332 after gc; the suite's -0.90 GB in row 9 = ~9 heads). With gc between rows the row max stays 10.64-10.83 GB (the suite reached 11.46). Exact fix (Breakthrough, after the freeze): gc.collect() between requests, or build the head once / a module-level class. (Cell text by Fit12, pasted by NewIdeas 01:4x. NewIdeas' 'less likely' for (b) above was wrong: the measurement wins.) **R-45F CONFIRMED Thu 00:49 (Fit12)**: fold-fast (b12llff) creeps by the same garbage (B 86 %, A 3 %, tokens = the suite's); with gc between rows its math peak is 12.212 GB, so C_m ~61 on short prompts with the fix (vs ~52 without). Residual in both arms after gc: +0.127 GB over 8 rows (vmmap: MALLOC_LARGE +95-99 MB, Python arenas +17.5 MB), ~14 MB per row that no gc frees (~0.6 GB over a 45-row process). **R-45R GO (NewIdeas 01:4x)**: fold-exact rows 1-3, tracemalloc (1 frame) snapshots after rows 1 and 3 plus vmmap, one quiet lock, ~6-8 min. PASS: tracemalloc covers >= 80 % of the MALLOC_LARGE + VM_ALLOCATE growth with one file:line >= 50 % (a named site for Breakthrough). KILL: < 20 % (native: C++ engine or MLX; then a MallocStackLogging hunt). Between: the top sites.  **R-45R BETWEEN Thu 01:49 (Fit12)**: of the ~18-24 MB / row left after gc, tracemalloc sees 23 % (all of it the n-gram hot cache, qwen4_exp.py:5570); ~77 % is native (v3m engine / MLX / tokenizer). Details: /Users/kamen/Fit12/PLAN.md 'R-45R RESULT'. **Breakthrough's fix PASS (B12_FRS_NOCYCLE, SCOREBOARD 127fdf8)**: replaced FR-Spec heads are freed without gc; tokens = b12llfx 6 / 6, MTP hashes = defaults; 6-row creep 0.107 vs 0.631 GB; live after R-41C B. **NewIdeas' calls, to send when Kamen says start (02:1x; held during his full stop)**: (1) Fit12 R-45N, the native hunt (MallocStackLogging=lite + malloc_history on the same 3 rows, one quiet lock ~6 min): PASS = one stack >= 60 % of the native growth (a named owner: fix or cap); KILL = no stack >= 20 % (Breakthrough budgets it per row in the C_m points). (2) Breakthrough caps the n-gram hot cache for 12 GB (MTPLX_NGRAM_HOT_MB, default 1024, grows ~4 MB / row, ~1 GB over a long chat) to a budgeted size with a hit-rate check (exact). (3) Fold-fast's C_m points (Breakthrough asked 02:0x): re-derive them from the footprint bounded by the fix + the cap (+ R-45N), then one 45-row process <= 11.95 GB (Fit12's estimate ~C_m 61 on short prompts); decision 5 stays on hold until then. **R-45S SHAPESPLIT GO (NewIdeas 15:4x; Fit12's lines, from Breakthrough's 01:58 split)**: fold-exact, gc + vmmap after every row; arm s = suite row 3 (algebra/694) x 3, arm d = rows 3 / 10 / 12 (614 / 571 / 404 tokens); G = MALLOC_LARGE + VM_ALLOCATE growth after row 1 -> after row 3. PASS (shape-driven): G_s <= 20 % of G_d (per-new-shape native caches; fix = shape buckets or a cache cap). KILL: G_s >= 80 % of G_d (a per-request native leak; then R-45N, MallocStackLogging). No verdict if G_d < 15 MB; tokens = the suite's rows. **R-45S KILL (Fit12, Thu 17:30-17:34, one quiet lock; tokens = the suite's rows 6 / 6; cell text by Fit12, pasted by NewIdeas 19:11)**: G = MALLOC_LARGE + VM_ALLOCATE (dirty + swapped) after row 1 -> after row 3: same row x3 +38.4 MB, rows 3 / 10 / 12 +40.3 MB -> 95 % (KILL line >= 80 %; G_d3 >= 15 MB). Both arms add 7 MALLOC_LARGE regions per request (~2.8 MB each); after-gc footprint +~21 MB / row in both, while MLX active stays 9.332 GB, IOAccelerator 8.8 GB, gc-tracked Python objects 103,113-4, and the n-gram hot cache stays flat in s3 (0.005 GB). So: native malloc kept per request, the same for a repeated row as for new ones; not shape caches (MLX / QSA buffers would show in IOAccelerator), not the hot cache. Unknown: does it level off (R-45 was linear over 8 rows; ~0.9 GB over 45 rows if linear)? Evidence: /Users/kamen/Fit12/PLAN.md 'R-45S', data/scr/creep/{s3,d3}*. Fit12's note: the hot cache grows ~2.5 MB per NEW row (~0.1 GB over 45 rows), so R-46's memory payoff is small; R-46 moves behind R-45T. NewIdeas' hint: 2.8 MB is one expert record (MSB 2,162,688 + LSB 770,048 = 2,932,736 B = 2.797 MiB). **R-45T MALLOCSTACKS GO (NewIdeas 19:11, Fit12's design)**: fit_creep.py, fold-exact, rows 3 / 10 / 12, gc after each row, MallocStackLogging=lite; malloc_history -allBySize after rows 1 and 3; diff the live >= 1 MB allocations by call stack. PASS: one stack (top 8 frames) holds >= 70 % of the new large-block bytes r1 -> r3: name the owner (MLX / Metal / v3m / mtplx / Python) and route the fix. KILL: no stack >= 30 %. Between: the top 3. Guards: tokens = the suite's rows; footprint < 36 GB; one quiet lock <= 15 min, memory-only. **v3o GO (NewIdeas 19:52, Breakthrough; BigMoE confirmed the mechanism in its own ext: reads run on GCD workers, which GCD retires when idle, and each takes its raw thread_local bounce buffer with it)**: a buffer pool or a thread_local holder with a destructor; exact. PASS: every hash = v3n on p2k / p8k / write16 AND after-gc growth <= 3 MB per request over 6 requests with >= 10 s idle gaps (v3n ~20 MB) -> v3o joins the post-R-41D default. KILL: >= 15 MB per request (R-45T's stacks decide). **R-45T PASS (Fit12, Thu 19:55-19:57, one quiet lock; bt_b12 ce6d4c63d2 pinned, MallocStackLogging=lite; tokens = the suite's rows 3 / 3; cell text by Fit12, pasted by NewIdeas 20:45)**: malloc_history -allBySize after rows 1 and 3, diffed by top-8-frame stack: of 43.4 MiB grown, 37.91 MiB (87 %) is _posix_memalign < bigmoe_ext < _dispatch_client_callout2 < _dispatch_apply_invoke3 < ... _dispatch_root_queue_drain, +14 calls (7 per request), = vmmap's MALLOC_LARGE growth (134.7 -> 172.6 MiB, +37.9) exactly. Its only posix_memalign is read_expert's thread_local bounce (bigmoe_ext.cpp:122), never freed when GCD retires a worker (Stack regions 72 at every row: workers replaced, not added). The rest: 3 MiB pymalloc arenas, < 1 MiB numpy. Owner: Breakthrough (v3o, exact); BigMoE fixed the same in the 24 GB ext (a shared bounce pool, 9f6c4f5, shas equal; b47: official 24 GB peak unchanged, 23.09 vs 23.08-23.13 GB). Side note: with ce6d4c63d2 on, row 1's gc still freed one _FullVocabDraftHead (0.095 GB; likely the load-time head built before the wrap, a one-time 0.095 GB until a gen-2 GC). Evidence: /Users/kamen/Fit12/PLAN.md 'R-45T', data/scr/creep/t3_mhist_diff.tsv. |
| R-46 | Fit12 (account-2) | HOTCAP [exact, env only; NewIdeas Thu 15:4x; from R-45 / R-45R]: the n-gram hot cache grows ~4 MB per row (R-45: A +0.030 GB over 8 rows; R-45R: 98 % of the Python residual, qwen4_exp.py:5570) toward its 1,024 MB default over a long chat. Arms MTPLX_NGRAM_HOT_MB = 1024 / 256 / 64 on fold-exact (b12llfx), R-41C's 16 math + first 8 agent rows, one process per arm, gc after each row (or B12_FRS_NOCYCLE); per row: hot-cache real bytes (rows + the batches they pin), hit rate, decode tok/s, tokens hashed. Also: real bytes per hot row including its batch | PASS: at 256 (or 64) tokens identical, decode >= 99 % of 1024's (ABBA), hot cache <= its cap including pinned batches -> Breakthrough sets the default. KILL: decode -3 % or worse at 256 (keep 1024 and budget it in C_m) | **PASS 00:4x (Fit12, data/scr/r46/, PLAN.md 'R-46 HOTCAP RESULT'; live engine 677c6558b6 / v3o)**: tokens rows 1-4 identical in all 4 arms (= the suite's rows); HOT 1 MB rows 2-4 17.74 tok/s vs HOT 1024 17.52 (101.2 %; A1 void by a heat PAUSE; B2 vs A2 alone 100.8 %); the HOT 1 table holds at 10,485 rows, 0.007 GB, while uncapped reaches 35K rows by row 4 and keeps growing; hits -5.6 % at no tok/s cost. **NewIdeas 00:39: MTPLX_NGRAM_HOT_MB 16 is the 12 GB safety default on my word (Breakthrough sets it; Fit12 relays)** |
| R-47 | Fit12 (account-2) -> Breakthrough adopts | BIGCACHE12 [exact; NewIdeas Thu 15:4x; after R-45's fix (B12_FRS_NOCYCLE) and R-46's cap]: spend the freed long-process memory (~0.7 GB: R-45's row max 10.64-10.83 GB with gc vs 11.46 without) on a bigger exact expert cache for fold-exact. Re-derive its C_m points the R-01 way so a one-process 45-row lifetime peak stays <= 11.85 GB; decode tok/s and SSD MB per token vs C_m 51 on the 45 rows (ABBA, quiet locks), tokens identical. Reuse Breakthrough's FFX harness / points format (SCOREBOARD 0050c97) | PASS: >= +3 % decode (exact) -> Breakthrough adopts. KILL: < +1 % | **DROPPED 02:33 (NewIdeas' premise error, caught by Fit12)**: on fold-exact the C_m changes the MTP path's bits (bt_b12's single-row late rule; memory note mtp-path-residency-bits), so a bigger cache can't be exact vs today's default. Its purpose moves into decision 5's fold-fast (ALLWAIT, pool-independent bits): Breakthrough's b12llff3 C_m points (R-41) |
| R-48 | Breakthrough (account-1) | M01-12 [exact; NewIdeas 17:22; from R-44 (a)'s PASS]: port BigMoE's M-01 scale / bias re-pack to the 12 GB engine, in a new engine dir (v3m stays pinned for R-41D). First: can bt_b12 read records3m (68 GB, built and verified) through the BIGMOE_M01 loader, with no second copy? Its LSB records are 720,896 B (11-bit), the size records3m's LSB already has, so it may need its own repack: then check free disk first (stop at 110 GB free). Screen: fold-exact defaults; p2k, p8k and write16 in one process; A (v3m) vs S (+ M-01) at equal pools, A S S A; hashes (first-step logits, kv, state, tokens), rusage SSD bytes / token, decode tok/s, lifetime peak. CPU build, then one quiet lock (~8 min) | PASS: every hash equal AND bytes / token -2.5 % or better AND peak not higher -> the 12 GB default on NewIdeas' word, after R-41D. KILL: any hash differs, or a saving under 1.5 %. Between: tok/s decides | **PASS 17:57 (Breakthrough, SCOREBOARD a808ffc / 73452f1, data/m01/; one quiet lock 17:48-17:56)**: the 12 GB line streams the same records3 files as BigMoE (MSB part 2,162,688 B, LSB 770,048 B; the 720,896 above was MoonShots' table), so records3m serves it with no extra disk; v3n = v3m + BigMoE's unpack. Loader bitwise on L0 / L23 / L47 x 96 (M-01 on and off); page cache 0.00 %. A1 S1 S2 A2 at fold-exact defaults (p2k, p8k, write16, 256 tokens; C_m 51 / C_l 20): every hash equal in all 4 runs; decode rusage bytes / token 456.9 -> 432.8 MB = -5.26 % (A1-A2 0.13 %), prefill -5.66 %, decode tok/s +3.45 %; lifetime peak 10.982 vs 10.996 GB. **NewIdeas 18:25: default after R-41D (B12_ENGINE=v3n + BIGMOE_M01=1); Breakthrough prepares that candidate now (bitwise check in BT_DIR), then prices POM / LP1 at 12 GB on CPU (R-49).** |
| R-49 | Breakthrough (account-1) | POM12 [NON-exact, CPU price screen; NewIdeas 18:25; from M-03 / M-07 / R-44 (b)]: price POM R2 4 and POM R2 4 + LP1 at 12 GB pools (C_m 51 / C_l 20) on BigMoE's records3p layout (the same files). Reuse MoonShots' replays (scripts/pom24.py, lpom24.py; copy them) with 12 GB routing traces (p8k decode + agent rows). Report SSD bytes / token vs fold-exact + M-01, f (share of rank >= R2 gate energy at 2 planes; of rank-1 energy without the LSB) and predicted KLD per part by MoonShots' formula. Misses are more frequent at 12 GB, so f will be higher: price R2 5 / 6 too | PASS: some variant -15 % or better bytes / token with predicted KLD inside 1.10 x ar4 on all four parts -> MoonShots' harness confirms at that f (one lock), then an engine port behind a switch; on a PASS there, the strict test, then Kamen. KILL: under -8 % for every variant inside the gate | **KILL 19:40 on modeled SSD busy time (Breakthrough, SCOREBOARD 8c8aab7, data/r49/; traces = the 12 GB decode's own verify passes, one lock 19:33-19:38: p8k 512 tokens, 4 agent rows, math 1640; 4-row passes, 2.6-3.8 kept)**: the LRU replay reads 432.8 MB / token, = R-48's measured 432.8. Busy model per call: 0.2 ms + bytes / 14 GB/s + 4 us / read + a ~0.10 ms 2-plane / upgrade rebuild (calibrated so 24 GB pools bracket b45's measured -6.5 %). Inside the gate (LRU): LP1 alone -3.8 % busy (-4.0 % bytes); R2 6 -3.2 % busy (-9.8 % bytes); R2 6 + LP1 exactly at the code gate (0.00253): -7.0 % busy, -13.7 % bytes; R2 5 / R2 4 over the gate. At 12 GB half of the rank >= 6 energy is served at 2 planes (24 GB ~20 %), so the KLD budget runs out, and rebuilds plus upgrades eat the byte saving. |
| R-50 | Leap (account-1) | SLOTCALL [exact by construction; Leap's proposal 18:48; NewIdeas GO 18:58]: with REF2's order-independent experts, 16-row checks on the plain _sync_call path call the experts kernel on pool slots instead of gathering ~100 experts x 2.8 MB into scratch per layer. Screen (~3 min, one lock, after STEP 3w): /68 and prealgebra/1640, REF2 + SB, gather vs slot call, same pools | PASS: every token and first-step logits hash equal AND >= 40 ms per check saved (median, live) -> into REF2's engine path. KILL: under 15 ms | PASS Thu 20:18 (Leap, ~/Leap/data/r50_screen.txt, TABLE_350 '## R-50 SLOTCALL'): median 384.2 ms saved per 16-row check (/68 451.1, prealgebra/1640 317.2), bit-equal (tokens + first-check logits hash, 8 runs); code 14.0 -> 25.8 tok/s, math 7.4 -> 14.3 (C 144, misses equal across arms). Cause: a fresh ~330 MB scratch per layer + clear_cache (allocation cost). Next: into REF2's path, gated on <= 16 rows + REF2 on + PF1 off |
| R-51 | MoonShots (account-1) -> BigMoE builds | VD24-T [NON-exact; NewIdeas 19:29; Kamen 19:27 approved a second, non-exact 24 GB version; from b45]: price VD + LP1 (and POM R2 4, LP1 alone, VD alone) by predicted SSD busy ms / token, not bytes. Fit a cost model per read + per byte (and per prefetch reservation / upgrade) to b44's and b45's measured SSD busy (~/BigMoE/data/b44, b45 meta.json 'pom' counters); it must reproduce b45's P and L within 2 points before it prices anything. Then the M-03 / M-04 / M-07 replays emit read counts by size and the model's busy time vs the M-01 default; also the quality shares at the engine's real rates (b45 served 31 % / 37.5 %, not the replay's 20 % / 26 %) | PASS: VD + LP1 (or a variant) at -15 % or better predicted SSD busy ms / token with predicted KLD inside the gate at the engine's real shares -> BigMoE builds variable slots + the 2-bit path, then a speed A/B with enough prompts to beat the 3 % noise, then the Eval suite's strict verdict. KILL: under -8 % for every variant inside the gate | **STOP (no busy price) + VD + LP1 OVER the gate at the engine's real shares** (MoonShots 21:16, fdc59c4, scripts/r51_busy.py, data/r51_busy.txt). (1) No model fit on the full-read arms (b44 A, S; b45 A) reproduces P and L within 2 points: per read + per byte predicts P -10.7 % / L -18.4 % (measured -6.5 / -5.6); demand reads + bytes -17.0 / -21.2; bytes only -24.0 / -28.6; reads only +1.5 / -9.1; gate wait the same picture. L refutes every additive model: LP1 removed 54 % of the sidecar reads (25.6 -> 11.9 / token, -12.5 MB) and busy and gate wait stayed at P's (L1 -6.2 %, L2 -5.1 % vs A): sidecar reads are free in SSD time, within noise. Noise: same-config repeats up to 10 % apart (P1 28.30 vs P2 25.38 ms / token; P / L arms are non-exact, so each run generates different text); records3m 8 % apart between b44 and b45. Effective bandwidth while busy 9.4 (A) -> 7.7 (P) -> 7.1 (L) GB/s: the SSD is not byte-bound here. (2) Replay counts (descriptive, no price): VD cuts misses 20 % (full + 2-plane 46.7 -> 37.4 / token) but triples upgrades (4.6 -> 13.9; 47.5 % of 2-plane reads), so demand reads equal POM's (51.2 vs 51.3; v3 46.7). Scaled to the engine: POM + VD reads +2.4 % vs A (demand 77.4 vs 53.9, prefetch 50.3 vs 68.5), bytes -31 %: P's smaller-not-fewer profile (P: reads +1.5 %, bytes -24 %, busy -6.5 %, tok/s in noise). (3) KLD at the engine's real shares (row method: F = 31 % x the replay ratio = 43.5 % for VD, F1 37.5 %; linear in the shares from the 3 harness runs, extrapolated past F 0.28): VD + LP1 OVER on code 109 %, gsm 112 %, ifeval 107 % of target; POM + VD inside (code 96 %); POM + LP1 (b45's L) at the edge (code 101 %, gsm 101 %). Call: don't build VD + LP1; VD alone is unpriced and has P's read profile. Leads for fewer critical-path reads: lazy upgrades (serve the use at 2 planes, issue plane 3 as a prefetch, never waited on); fewer unused prefetches (68.5 / token at A, ~3.7 of 4.8 per pass-layer unused per b45) |
| R-52 | BigMoE (account-1) | SCRATCH24 [exact; NewIdeas 20:45; from Leap's R-50: the plain path's fresh scratch per layer-call (mx.take + concatenate, then mx.clear_cache) cost ~8 ms per ~330 MB as allocation / page faults, 384 ms per 16-row check]: audit first (CPU): which shipped 24 GB / 45 GB decode and prefill calls take _sync_call / _sync_grouped / _sync_segment / tier.prefill_call (checks over MAXK ids, PF1-off prefills, PF_GROUP groups), with the fresh scratch bytes per decoded token and per prefill chunk; then, if >= ~0.2 GB per token or chunk, a reused scratch buffer per layer (no clear_cache inside the loop; one at the request's end, or a size-capped cache). NewIdeas prices MLX's fresh-allocation cost per GB on this Mac (scripts/alloc_bench.py, data/alloc_bench.txt) | PASS: every sha equal AND tok/s or TTFT +5 % or better AND lifetime peak within +0.1 GB -> default on NewIdeas' word. KILL: under 2 % | **Price, run 1 (NewIdeas 20:53-20:54, one quiet lock, GPU 3 % at the grant; data/alloc_bench.txt; MLX 0.32.2; a pool of 144 slots of 2,932,736 B)**: reuse (R, the freed buffer recycled by MLX's cache) costs 2.885 ms per 346 MB = 8.3 ms / GB, i.e. the copy itself. Fresh (F, mx.clear_cache after every call) has the same median (+0.12 ms; 0.36-0.69 ms / GB at 346 / 141 / 106 MB) but a heavy tail: p90 24.6 ms per 346 MB (9.2 / 8.5 ms at 141 / 106 MB) vs R's p90 3.0. So fresh scratch costs in rare large stalls, not on every call; run 2 (means, stall share, 40 samples x 3 blocks per arm) is queued. A slot call (R-50) also saves the copy, ~8.3 ms / GB. Desk KILL (NewIdeas 21:0x), not worth a row: cheaper reads for the MTP layer's experts (exact, since the MTP head only drafts) can't pay: at 3 draft steps x 1 row x top-10 it has ~30 of ~1,950 expert uses per 4-row check (~1.5 %). **Audit DONE 21:15 (BigMoE, ~/BigMoE/data/r52_audit.txt, 2cfd954)**: 24 GB decode 0 B of fresh scratch per token (verify rows <= 8 x top-10 <= MAXK 80 run tier_call on pool slots; the streamed MTP layer too), so tok/s cannot move; 24 GB prefill ~18-35 GB fresh per 2,048-token chunk (_prefill_grouped's double buffers on every layer call, 0.35-0.71 GB x 48; the MTP prompt pass ~1.46 GB; short prompts one scratch up to 1.41 GB per layer call) vs ~72 GB read from the SSD and ~10 s per chunk: ~4-8 % of TTFT. **45 GB (stock MLX, MAXK 40): 16-row checks take the plain path at ~0.33 GB per layer call, ~16 GB per check (per decoded token ~1.7 GB code / 2.3 math / 5.3 agent / 7.8 chat)**; Leap's slot call fixes it on REF2 only. **NewIdeas 21:2x (45 GB first, Kamen 21:1x): R-52b GO before the 24 GB TTFT screen**: a reused scratch for the stock 16-row check path at the 45 GB setting (C 192), one buffer per layer size class, freed at the request's end, A/B on Leap's R-50 prompts (/68, prealgebra/1640). PASS: every sha equal AND >= 40 ms saved per 16-row check; KILL: under 15 ms (it can only remove the stalls; the copy stays, ~8.3 ms / GB). Then the 24 GB prefill screen on BigMoE's plan. **R-52b PASS 22:04 (BigMoE, 9026e4d, ~/BigMoE/data/r52b/r52b_screen.txt; C 192, stock MLX, warm 16-row checks along the REF2 rows, ABBA / BAAB)**: /68 base median 1,076.6 -> keep 342.4 ms per warm check (-734 ms); prealgebra/1640 1,079.0 -> 386.3 ms (-693 ms); logits bit-equal on every window in all 4 passes; misses equal on 1640 (207.6 vs 208.9), /68 187 vs 151. The base arm is erratic (first pass ~660-800 ms, second 1,340-1,730: fresh allocations slow down as memory fills; the 8 ms per call figure is the good case); keep steady at 320-410. Cost: MLX's cache keeps 9 recycled buffers, footprint +0.45 GB (30.46 vs 30.03). Switch BIGMOE_SCRATCH_KEEP=1 (non-tier plain-path single-group calls only; the 24 GB tier stack is unaffected). **NewIdeas 22:08: default on (Kamen 22:06: more RAM is OK on the 45 GB route)**; next a live-loop tok/s number at 45 GB (stock base vs keep) |
| R-53 | Fit12 (account-2) -> Breakthrough adopts | SCRATCH12 [exact; NewIdeas 20:45; R-52's audit for the 12 GB engine]: which bt_b12 (ce6d4c63d2 / v3n) decode and prefill calls build a fresh scratch per layer-call and free it with mx.clear_cache (MTP-path checks over MAXK ids, PFR prefill chunks, the MTP head's prompt pass), and how many fresh GB per decoded token and per 4,096-row chunk; priced with NewIdeas' alloc_bench ms / GB. If >= ~0.2 GB per token or chunk: a reused scratch in Fit12's pinned copy (BT_DIR), A / B on p2k, p8k, write16 + 8 math rows (hashes, decode tok/s, prefill time, lifetime peak), one quiet lock | PASS: every hash equal AND decode or prefill +5 % or better AND lifetime peak within +0.05 GB -> Breakthrough ports it. KILL: under 2 %, or the peak rises past the 11.85 GB line | **A/B 01:58-02:27 (Fit12, PLAN.md 'R-53 A/B result', data/scr/r53/; base v3n, patch in a copy): EXACT (every hash equal in all 4 arms on p2k / p8k / p16k: KV of every layer + MTP cache, h, draft logits, 32-token sha) and memory-neutral (lifetime peak B <= A + 0.04 GB); speed NOT settled: heat pauses at FAN_CAP 3500 voided b1 and a2; the clean pair a1 / b2 is thermally biased (a1 started at die 80-88 C, b2 at 58 C): p8k -7.2 %, p16k -21.1 % (the mechanism predicts ~5 %).** R-53C (Fit12's lines): the same ABBA on p8k + p16k, each arm only from die <= 55 C at thermal state 0, void on PAUSE; PASS mean B <= A -5 % on both, KILL > -2 % on both **R-53C PASS 00:0x (Fit12, data/scr/r53c/, PLAN.md 'R-53C RESULT'; live v3o engine, bt_b12 677c6558b6 pinned, cool starts, A1 B1 | B2 A2, no PAUSE)**: exact (every hash equal across the 4 arms and equal to R-53's on p8k / p16k); prefill p8k 31.66 -> 29.77 s (-5.9 %), p16k 60.66 -> 56.79 s (-6.4 %), all 8 prefills clean; prefill peaks -0.3 GB, decode / lifetime peaks equal (10.75 GB). R-53's -21 % was the hot / cool confound. **NewIdeas 00:04: adopt as the 12 GB default on my word: Breakthrough ports Fit12's patch (/Users/kamen/Fit12/patches/r53_scratch_v3o.diff, tier.py + stream.py, switches on) after its pacing sleep, hashes re-checked** | **Audit 20:5x (Fit12, code read of ce6d4c63d2 + v3m)**: decode builds NO fresh scratch (4 rows x top-10 = 40 ids <= MAXK 80: every MoE call takes the sync-free tier.tier_call; MTP draft steps are 1 row), so 0 GB per decoded token. Prefill, per 4,096-row chunk: each layer's tier.prefill_call -> _prefill_grouped allocates (a) 2 x 64-expert double buffers (0.35 GB) and (b) outs + concatenate + take (0.63 GB), then mx.clear_cache (tier.py:584): ~1.0-1.1 GB per layer-call, ~50 GB per chunk; (c) the MTP head's prompt pass (stream._sync_grouped, PF_GROUP 64) ~11 GB per 4,096 rows. ~60 GB fresh per chunk. Prior: BIGMOE_TIER_PF_PERSIST=1 (piece a) exact, 64K prefill 207 vs 212 s (-2.4 %), reverted because decode kept the buffers. **NewIdeas 20:50: GO prefill-only, pieces a + b + c reused, freed at prefill end; PASS -5 % prefill on p8k + a 16K-32K prompt, hashes equal, peak within +0.05 GB; KILL under 2 %.** R-46 un-parked as a safety default: the hot cap counts 100 B per row but a row costs ~620 B real, so the 1,024 MB default binds at ~6.6 GB real (a 12 GB OOM for a long-lived server); cap in real bytes after R-47. |
| R-54 | Breakthrough (account-1) | ECOSPEC12 [exact: only the number of drafted rows per check changes; NewIdeas 20:47; opens the parked seed EXPERT-AWARE DRAFT SELECTION (EcoSpec arXiv 2607.12696, S2-MoE 2608.15018, EVICT 2605.00342); b45 / R-49 showed that SSD time, not bytes, sets the speed]: in a 4-row MTP check every row's experts are read, but rows after the first rejection are wasted reads (2.6-3.8 kept per check in R-49's traces). Screen (CPU, R-49's traces + pom12.py's LRU replay and busy model): per check and row, the marginal reads (experts not resident and not already needed by earlier rows) and whether the row was accepted. (1) Oracle: cut each check after its last accepted row; modeled SSD busy ms per token vs today (the replay keeps the cache effects of the reads that are skipped). (2) If the oracle passes: log the MTP head's p1 per drafted token (one short capture) and test a policy that drops row j when p1(j) < tau and its marginal reads > m | KILL: the oracle saves < 6 % of modeled busy ms per token. PASS: oracle >= 12 %, then a p1 / reads policy >= 6 % with tokens per check down by less than the time saved -> an engine A/B (exact by construction: verification is unchanged). Between: report it, and BigMoE runs the same at 24 GB | Prior art (NewIdeas web pass 20:48): AHEAD (ASP-DAC 2027 listing: a dynamic draft-verify horizon for an offloaded MoE on a memory-limited GPU, expert transfers on the critical path) reports 1.65x / 2.24x over fixed-horizon MoE speculative decoding; EVICT (2605.00342, lossless prefix truncation) and EcoSpec (2607.12696, marginal expert cost in tree selection, up to 1.62x); AcceptMoE (2608.02989) separates token count, expert union and expert-weight traffic (2.06x under offloading on one RTX 5090; its expert-set pruning is non-exact). Later step (3), if (1) passes: the other direction, a horizon LONGER than 4 rows when the next rows' experts are already resident or already needed (needs a longer drafter, MTP K > 3 or DFlash2) **Step 1 BETWEEN 16:22 (Breakthrough, scripts/eco12.py, SCOREBOARD 414b65f; R-49's traces, LRU C_m 51 / C_l 20, replay baseline 432.8 MB / token = R-48 measured)**: oracle (cut each pass after its last kept row, cache effects replayed) 34.79 -> 31.42 ms / token = -9.7 % modeled SSD busy, bytes -10.5 % (KILL < 6, PASS >= 12); per prompt write16 -22.6 % (2.61 kept per pass), bash17 -13.5, p8k -12.3, math -7.4 (3.53), edit00 -6.3, text23 -2.8 (3.80). Rows barely share misses (each adds ~3.4 MSB + 0.75 LSB reads per layer call, kept or not); rejected rows hold 16.1 % of bytes but a third of those reads feed a later pass. Fixed 3 / 2-row horizons lose (+3.9 / +11.5 % busy per token). Same replay at 24 GB pools (LRU stand-in): oracle -9.1 %, also BETWEEN. **NewIdeas 16:23: the 24 GB engine run waits (45 GB first); at 45 GB, Leap's live S45c tests a real truncation policy first; Breakthrough prices the 45 GB oracle on 16-row routing traces only if S45c disappoints** |
| R-55 | Leap (the split) -> BigMoE (the build) | GATE16 [exact by construction; 45 GB route; Leap's lead 21:59; NewIdeas GO 22:08]: after R-50b a warm exact REF2 16-row check at C 192 costs ~200 ms (slot 160-240 ms); earlier resident REF2 numbers (L1-23 / 25 / 29: dense ~18, experts ~35, rest ~10-25) put its compute at ~60-80 ms, so ~120-140 ms may be the plain path's 48 per-layer id syncs + ensure + Python (~2.5-3 ms per layer). BigMoE's gated path is sync-free but carries <= 92 ids per row; a 16-row gate (2 rows x 80 ids, or 160-id rows) would run REF2's slot-ordered kernel unchanged (bit-equal by construction, R-50). Step 1 FIRST (Leap, one lock; widened 22:11 on the orchestrator's ask after Kamen's 'why wasn't R-50's waste caught earlier'): a measured per-stage profile of the warm ~200 ms check at the 45 GB setting, ms per stage (dense: HC / GDN / attention / LM head; experts; per-layer id syncs; ensure; Python; commit / rollback), with Metal timestamps for GPU busy vs bubble. The next lever is chosen from that table. Step 2 (BigMoE): the gate build + an A/B, only after step 1 | Step 1 PASS: bubble >= 80 ms of the ~200 -> step 2; KILL: bubble < 40 ms (compute is the lever: REF2 kernels). Step 2 PASS: every logits hash equal AND warm check <= 120 ms. Leap's estimate: ~80-100 ms per check -> exact 16-row path ~90-110 tok/s on code at 45 GB (kept 9.6), above the 64 line (a lead) | **Step 1 BETWEEN 22:21 (Leap R-50c, ~/Leap/data/r50c_screen.txt, TABLE_350 R-50c, 3f236a1; warm exact 16-row checks at C 192, REF2 + SB + slot call, medians over 24)**: wall 185.1 ms = check 181.5 + commit 3.6. Inside: expert kernels + sync 77.7 (~13.4 GB per check at ~173 GB/s, near the bytes floor); GPU up to each router (attention / GDN / HC / dense / shared / router) + sync 64.6 (~52 GPU); ensure 22.7 (44 misses ~123 MB + 48 calls); Python between layers 10.2; head 2.6; build 1.9; unexplained 1.6. GPU busy ~120, bubble ~60 (ensure 22.7 + Python 12.1 + 97 syncs x 0.256 ms). GATE16 could save <= ~60 ms (-32 %, 185 -> ~125) if ensure's misses are prefetched too. The exact 4-bit 16-row floor is ~70 ms (experts ~48 at 280 GB/s + dense ~20) vs ~35 needed for 350 at ~12 kept: bytes per check is the deeper lever. **Step 2 PASS Fri 16:29 (BigMoE, 4be3b31, data/r55/r55_screen.txt; BIGMOE_MAXK=160, ext/build_next)**: warm check + commit at C 192, REF2 + SB: HumanEval/68 232.6 -> 117.3 ms, prealgebra/1640 213.7 -> 106.2 ms (gate 111.8 vs slot 223.1; line <= 135); logits bitwise on every window in all 4 passes (ABBA / BAAB); 48 loader-served gates per check, no timeouts. BIG sent by BigMoE. Adoption (MAXK 160 on the 45 GB path) after the Eval suite's live audit, which gets a REF2b + GATE16 arm (NewIdeas 16:54). **NewIdeas 22:50: step 2 GO after 00:00 (45 GB first; ~185 -> ~125 ms would put the exact 16-row route at ~75-80 tok/s on code, above the 64 line)**: BigMoE builds the sync-free 16-row gate + prefetches ensure's misses. PASS: every logits hash equal AND warm check <= 135 ms; KILL: > 165 ms |
| R-56 | EvalSuite | LIVE16 [45 GB verdict; NewIdeas 22:08; to run once Leap's STEP 3u fix lands]: a live-loop exactness audit of the 16-row path, REF2 + slot call and stock + SCRATCH_KEEP, each vs the same reference's AR, on 30+ suite prompts (code / math / chat; include prompts under 128 tokens: Leap's root cause 2, MLX's fused vector SDPA switches mode at 128 keys), every token hashed | PASS: every token equal on every prompt in both arms -> the 16-row path is exact for REF2 adoption (decision 4's last step) and for the stock path. KILL: any mismatch (a blocker: bisect with Leap) | **ref2 arm FAIL 02:5x (EvalSuite, docs/RESULT_r56_2026-10-30.md; diagnostic evalsuite/live16diag.py, adfd491)**: 28 / 36 token-equal (code 12 / 12, chat 12 / 12, math 4 / 12). Today's REF2 AR reproduces results/ref2 through every differing token (the reference holds); the live loop's kept rows are bitwise = REF2's AR until REF2's padded AR (N = p + 16) first has > 512 keys (input position 497 on 7 of 8 prompts, 498 on one; geometry/702, P 520, from its second check); tokens flip only at AR ties or 1-step margins, 118-915 tokens in; 2,153 / 2,199 compared rows bitwise below 512 keys (the 46 others all 702's). So REF2's live loop is exact only under ~512 keys: decision 4's last step is not met; Leap tests key buckets (STEP 4c). EvalSuite sent the orchestrator a BIG correction of Leap's 00:2x wall (02:50). Stock arm not run. **NewIdeas 02:49: skip the stock arm (stock kernels aren't batch-invariant by design, which is why REF2 exists; its prior is a near-tie FAIL); re-run the ref2 arm on the same 36 prompts as an amendment once Leap's 4c fix is in** |
| R-57 | BigMoE (with R-55) | PREFETCH16 [exact: reads only, bits unchanged; 45 GB route; NewIdeas 00:55 from Leap's S45]: in the exact live REF2 loop at C 192 the misses are 413-779 per 16-row check (SSD reads 111-194 ms per check), vs 44 in R-50c's warm windows: that is the whole gap from R-50c's 185 ms warm check to S45's 266-397 ms check + commit. Overlap the reads: prefetch layer L+1's experts from a lookahead router while layer L computes (BigMoE's existing prefetch / S10b predictors, now on the 16-row check path; ProMoE / pre-gated MoE style) | PASS: SSD ms waited on per check -50 % or better AND check + commit -25 % or better at C 192 in the live loop, every token == REF2. KILL: under -10 % of the cycle | **KILL 17:00 (BigMoE, data/r57/r57_run.txt; live DFlash2 loop at C 192, REF2 + SB, 256 tokens, every run == REF2's row 12 / 12)**: gate + ranked L+1 look-ahead (B 96, PRED_LINES 4, JIT, QD 4, VICTIM 24) vs gate: demand SSD per check -20 % (151 -> 121 ms) but misses +10 % (597 -> 658: wrong guesses evict useful experts at C 192), check + commit -1.2 %, cycle -1.4 %. Same run, R-55 LIVE: slot -> gate tok/s HumanEval/68 25.6 -> 40.4 / 39.6, prealgebra/1640 17.1 / 17.4 -> 23.0 / 23.6 (prompt means 21.4 -> 31.7, +48 %); median cycle 416 -> 343 ms (/68), 291 -> 217 (1640). **NewIdeas 17:04: no variant now (no cache slack at C 192); revisit after REF2-T / bigger C. BigMoE next: GATE16 on REF2-T's tier slots (with Leap), so R-58 runs on the gate path** |
| S45b | Leap | CSWEEP [measurement; NewIdeas GO 00:55; Kamen 22:06: more RAM is OK on the 45 GB route]: the exact live REF2 loop at C 192 / 224 / 256 (~+135 MB per slot; C 256 ~45 GB) on S45's prompts, ~2 min each: tok/s code + math, misses per check, SSD ms per check, lifetime peak, every token == REF2. Guards: C 256 only with >= 50 GB free, swap +2 GB stop, PAUSE-aware (quiet mode numbers) | measurement: the misses-vs-C curve sets how much RAM buys at 45 GB | **01:36 (Leap, exact, 6 prompts): C 224 vs 192: median cycle 271 vs 320 ms; misses per check 432 vs 569 (-24 %; predicted -21 %); SSD 116 vs 150 ms per check; tok/s math 21.9 vs 19.2, code 24.7 vs 25.1, all-6 mean 22.8 vs 21.2 (+8 %); every token == REF2; lifetime peak 40.5 GB. C 256 / 288 stopped by the swap guard within ~17 s (+2.8 / +4.3 GB; gates were ~44 / 48 GB free, under the >= 50 GB line); swap 18.2 GB after. MoonShots' framing (Fri 16:3x): S45 has 51-86 misses per kept token (mean 69; C 224 ~52) = 13-22 ms of SSD per token at 10.6 GB/s even if every read is hidden, so the exact 4-bit 16-row path tops out near ~45-100 tok/s; past ~100 it needs REF2-T-class residency (3-bit pools near C_m ~380 at ~49 GB: ~8 misses per token, Leap's resid27). M-08 SPLIT45 (MoonShots, CPU) prices it for R-58. Leap: no big-C runs without >= 50 GB free, and a sweep stops at its first trip** **S45b-2 (Fit12, NewIdeas GO 16:16): the C 256 / 288 points on Leap's REF2b live loop, each only with >= 50 GB free, stop at the first swap trip; a measurement** **Amended 23:36 (Fit12's proposal, GO): while Kamen's apps leave ~37 GB, the misses-vs-C curve (C 224 / 256 / 288 / 320) comes from Fit12's cap45 LRU replay (an exact mirror of the loader, validated against miss.bin at the capture's C; CPU, no memory need); tok/s only from C 224 once memory is back. Note: BigMoE's ensure() aborts when one call needs more experts than C, and a 16-row check needs up to 160 per layer, so every 16-row run (R-59, cap45) needs C >= 160 (~32 GB)** **Curve 01:2x (Fit12's cap45 replay, misses per 16-row check): C 192 545, 224 424, 256 345, 288 297, 320 268: still falling past C 224 (-18.7 % at 256); large C slightly pessimistic** |
| S45c | Leap | TRUNC125 [exact: only the drafted rows per check change; 45 GB route; NewIdeas 01:46; TABLE_350 A3 (c) 'port to the 125B', L1-14 running-confidence truncation +8.4 % on the 35B; 16 -> 11 rows at -3 % tokens gave misses -24 %]: in the live exact loop the checks keep ~8.3 of 16 (S45), so rows after the first miss carry wasted misses, and misses are now the biggest piece of the check (S45: 413-779 per check, 111-194 ms of SSD). Screen (one 10-min lock, after S45b's slot): the S45 live loop at C 224, block 16 vs L1-14's truncation at t 0.3 (and 0.2 if it fits), the same 6 prompts | PASS: all-6 mean tok/s +5 % or better, every token == REF2 -> into the live loop's defaults (exact; then R-54's p1 / reads policy on top). KILL: under +2 % | **Run 1 17:21 (Leap, runner 42; REF2b merge check PASS first): exact in every arm, verdict not clean** (t 0.3's algebra/1282 hit a PAUSE; block 16 ran first and paid the warm-up; the swap guard cut t 0.2's last prompt). Steady state, block 16 -> t 0.3: median cycle 293 -> 205 ms (-30 %), rows 16 -> 9.8, kept 8.27 -> 7.91, misses 432 -> 330, tok/s 26.9 -> 32.6 (+21 %) on the 6-prompt mean; REF2b's bucket cost <= +2.2 ms per check. S45c-2 (clean re-run: warm-up discarded, per prompt A-B-A, paused runs out; same lines) runs after INV-T. **S45-STACK PASS 23:48 (Leap, data/s45c2_c160.txt; quiet lock, C 160 because Kamen's apps leave ~37 GB; warm-up, 5 runs per prompt; every token equal across all runs)**: slot block 16 19.65 tok/s (cycle 386-405 ms, misses 757); gate 26.50 (+34.8 %, cycle 294); t 0.3 22.99 (+17.0 %: rows 10.0, kept 7.96 vs 8.37, misses -23 %); gate + t 0.3 29.52 (+50.2 %, cycle 209; steady state 34.1 vs 20.4, +67 %). All PASS on the pre-registered lines (stack >= +40 %; truncation >= +5 %). BIG sent by Leap (labelled C 160). The C 224 run follows when memory is back |
| R-58 | Leap (after S45c) -> EvalSuite | REF2T [45 GB bytes per check; NewIdeas 10:20 from Leap 10:0x; lossy vs 4-bit, exact vs its own AR]: REF2's 16-row route reads every rank at 4 bits (4444444444), while BigMoE's v3 fast mode is TIER 4433333333 + TIER_GLU 3c (V3, 10-21: GSM8K 450 / 472 vs exact 452). REF2-T = the tiered model's own AR under REF2's kernels (1-row verify + commit keep 1 on the tier pools), with REF2b's attention fixes. Expert bytes -15 % (ranks 2-9 read 3 / 4 of the code bytes), and MSB-3 slots hold ~1.3x more experts per GB, so misses fall at fixed RAM. Leap's guess at C 224 (not measured): experts 78 -> ~66 ms, SSD 116 -> ~80 ms per check, ~-45 ms of a 271 ms cycle, ~+15-20 % tok/s. Steps: a desk port (REF2's padding rules on tier_call) + self-test; INV-T (~2 min GPU: the tier path is batch-invariant, a row alone == the row inside 16, 12-layer slice); S45-T (~4 min) | INV-T PASS: 16 / 16 rows and every record bitwise on the slice (KILL: any row differs -> the 3c kernel isn't batch-invariant). S45-T PASS: all-6 tok/s +10 % or better vs S45 at C 224, every token == the REF2-T AR; KILL under +4 %. **INV-T PASS 17:34 (Leap, data/r58_invt_screen.txt)**: REF2 on the v3 tier engine (4433333333 / 3c, C 192) is batch-invariant: HumanEval/68 at 383 keys and 702 at 536 keys, 16 / 16 logits rows and every layer equal, AR vs 16-row check; the control (the check's experts at 4 bits) breaks from layer 0. BigMoE runs the gate arm next. **M-08 SPLIT45 (MoonShots 17:38, data/split45.txt): KILL on reads (reads per kept token -0.2 to -9.5 % vs REF2 at equal RAM), but SSD MB per kept token -40-44 % and RAM-side expert bytes -19.4 %; at 16-row checks reads come in bursts that are bandwidth-bound, so SSD time should follow bytes (S45-T measures it). Sizing rule for S45-T: C_l >= ~96 per layer (C_l 32: +25-145 % reads); equal RAM to REF2's C 224 -> C_m 257-269 / C_l 96-128; ~45 GB -> 301-312 / 96-128; ~49 GB -> 333-344 / 128-160 (passed to Leap 17:45).** Adoption: the Eval suite's same-scores verdict REF2-T vs REF2. No new Kamen call: the tier plan 4433333333 is Kamen-adopted (~/.local/share/pacer/STATE.md: '3-bit tier engine (Kamen adopted it 10-20 14:15)'; Eval suite PASS 16 / 17 rules then) |
| R-59 | Fit12 (account-2; after S45b-2, before R-53C / R-46) | DENSE16 [45 GB; a measurement; NewIdeas 16:18 from Leap's R-50c]: in a warm exact 16-row check at C 192 the non-expert GPU work (attention / GDN / HC / dense / shared / router) took ~52 ms of GPU (64.6 with syncs), while its weights (3.44 GB: HC 1.28, GDN 1.18, LM head 0.36, attention 0.36) read in ~12 ms at 280 GB/s: ~4x the bytes floor, worse than TABLE_350's A6 (40-50 % of the floor at 8-16 rows). Profile one warm REF2b check at C 224 per kernel with Metal timestamps (Leap's R-50c tooling, read-only): HC mixing, GDN projections + recurrence, attention, norms, router, shared expert, LM head; each vs its bytes (and FLOP) floor | measurement: any sub-stage >= 8 ms AND >= 3x its floor -> a kernel row (exact only if bit-identical to REF2b's kernels; otherwise it folds into the next reference change, e.g. R-58's same-scores test) | **DONE 02:22 (Fit12, data/r59/r59_c224.txt, PLAN.md 'R-59 RESULT'; C 224, NO-SWAP >= 45 after the grant, GPU 0 %, no PAUSE, lifetime peak 39.0 GB; 10 warm 16-row checks on HumanEval/68, REF2 + SB)**: in place, ms per check (x its bytes floor at 280 GB/s): GDN 24.07 (5.7x), QSA attention 11.62 (9.1x; the 16-row SDPA runs in 12 head chunks), HC read mlp 7.95 (3.5x), HC read attn 7.22 (3.2x), PLE 0.75, head 2.29 (1.7x); MoE block 143.3 (experts + SSD at C 224); non-expert sum 53.1 ms of a 260.6 ms forced-sync check. Op groups: HC reads 4.6 ms each (2.0x), router 1.09 (4.6x), shared expert 2.48 (5.2x). Kernel rows: GDN and attention (HC reads just under 8 ms) -> Leap, folded into the next reference bundle with REF2-T (NewIdeas 02:24) |
| R-57b | Fit12 (CPU replay) -> BigMoE (build) | LOOKAHEAD2 [exact: reads only; 45 GB; NewIdeas 19:15 from SlotStream's EXPERT-LOOKAHEAD (same 125B: 1.111x, identical output, 21 % fewer records); our R-57 died on evictions at C 192]: forecast layer L+1's experts with layer L+1's own router on layer L's post-attention state + a per-layer linear logit correction; prefetch into RESERVED slots (wrong guesses never evict the main cache). Replay first (CPU, Fit12): BigMoE's hidden-state traces or Leap's live-loop routing captures; per layer the recall of the true top-10 / the 16-row union, raw vs corrected (fit on half, test on half); then the cache at C 192 / 224 with 8 / 16 / 32 reserved slots: demand misses per check saved vs wasted reads | Replay PASS: corrected recall >= 0.75 AND demand misses -15 % or better at C 224 with wasted reads <= the misses saved -> BigMoE builds it (after S45-T). KILL: recall < 0.6 or misses saved < 5 % | **Replay BETWEEN 01:2x (Fit12, data/r57b/r57b_cap45.txt on BigMoE's cap45: C 160 live REF2b loop, 6 prompts, x at all 48 layers; the replay matches miss.bin within +0.30 %, 90.6 % of (pass, layer) pairs exact; router set agreement 98.15 %, numeric)**: recall (top-10, 47 pairs) raw 0.605, corrected 0.693 (union 0.665 / 0.748; per layer 0.39-0.83). At C 224 (baseline 424 misses per check), N 8 / 16 / 32 reserved slots: corrected saves 20 / 44 / 69 % (wasted 12 / 98 / 506), raw 20 / 40 / 61 %, oracle 21 / 55 / 89 %; at C 192 corrected N 16 saves 44 %. Upper bound (in-time completion not modelled; ~285 prefetch reads per check at N 16). The reserved slots are what fix R-57. **NewIdeas 01:19: build GO (BigMoE): N 16, corrected forecast (Fit12's maps, ~25 MB); shared screen at C 160 then C 224; PASS demand SSD ms -25 % AND cycle -10 %, KILL cycle under -4 %; then a quiet confirmation** |
| R-28 | Fit12 | LAYSZ-X [cross-check of R-27 at 12 GB; it contradicts your PLA KILL of Mon 17:47]: your PLA used engine-faithful LRU, MTP-like checks (R 4 / A 3), and a Lagrangian that found +0.3 / -1.3 % out of sample. MoonShots' R-27 used LFU, 16-row checks accepting 10, a grid from 1 to ~231 slots, and found -20.7 % held-out (optimum: ~11 layers near full working sets, the rest at 1 slot). Its v3 at 51 / 20 is 206.0 vs your K1 256.4, about the LRU -> LFU gap seen at 24 GB (144.5 -> 116.0). In your replay, held-out: (a) LFU at uniform 51 / 20; (b) R-27's allocation (MoonShots' data/laysz_51_20.txt) under LFU; (c) the same under LRU. Also: which policy does bt_b12 ship? | PASS: (b) <= -15 % vs your 256.4 held-out (then BigMoE / Breakthrough engine A/Bs). KILL: (b) > -5 % | **CANCELLED 20:29 (NewIdeas)**: MoonShots' strict replay (pool trimmed to C after every check) puts R-27 at 12 GB at -2 % (KILL), agreeing with your PLA; the -20.7 % came from wall24's forced admission keeping staged experts as free cache. The 24 GB part (-8.9 / -10.3 %) goes straight to BigMoE's engine A/B |
| R-29 | TrainLab | LAYSZ-ON [CPU, after R-26; follow-on of R-27]: online per-layer pool re-allocation. MoonShots' per-set self-calibration bound is -17..-28 % at 24 GB (16/10) vs static R-27's -10.3 % (strict). Re-fit each layer's C_m / C_l every 200 tokens from the trace's own observed miss curves (first 400 tokens warm-up, same RAM), using MoonShots' laysz24.py STRICT replay (LAYSZ_POL=lfus: pool trimmed to C after every check; read-only import), at 24 GB 16/10 and 4/3, held-out sets | PASS: >= 4 points beyond static R-27 held-out in both shapes. KILL: < 2 points | **KILL 20:38 (TrainLab, REPORT 'R-29 LAYSZ-ON', data/laysz_online.txt, 1ee9432)**: online re-fit (uniform start, re-fit at tokens 400 / 600 on the set's own curves) loses to static R-27: -5.8 % (16/10) and -6.8 % (4/3) vs static held-out -10.3 / -8.9 % (-4.6 / -2.1 points; line PASS >= +4). A set's first 200 tokens predict its second half worse than three other full traces do. Replay reproduces MoonShots' strict numbers on every set. R-27 static stays the lever |
| R-26 | TrainLab | DYNK-Q [quality side of MoonShots' R-25 BETWEEN; non-exact]: drop rank 8 / 9 only when its renormalized top-10 weight < tau. Bytes at the exact 24 GB stack: tau 0.06 -13.2 % (82.0 MB/token, 4.7 % of gate mass), fixed top-8 -25.7 % (12.1 % mass). Strict / KLD screen at tau 0.06 (plus 0.05 / 0.07 if they fit the 10 min), same harness and gate as S9's top-8 | PASS: tau 0.06 passes the strict gate that fixed top-8 failed (2.23x). KILL: it fails that gate too (then only a task-test bar could carry it, like top-8 at 12 GB) | **KILL 20:4x (TrainLab, REPORT 'R-26 DYNK-Q', ec76426)**: tau 0.06 dropped 4.21 % of gate mass (0.84 experts per verify row-layer) and fails the strict gate in all three parts top-8 failed (ifeval P99.9 / top-1, code KLD + P99.9 x2.31, multi P99.9 / top-1); KLD x of v3: ifeval 1.43, code 1.98, multi 1.68 (top-8: x2.47). Only a task-test bar could carry it (decision-2 route) |
| M-01 | MoonShots | SBX [exact, CPU, added Wed 15:36]: shrink the per-group scale / bias bytes on the SSD copy. (a) The 4-bit bias b in each LSB record (153,600 of 770,048 B per expert; 105,600 of 720,896 at 11-bit) may follow from the MSB part (b3 = b + s3 / 4 at the converter?): if b == fp16(b3 - s3 / 4) bitwise in every group, drop it from the record and rebuild at load; else code the residual. (b) s3 / b3 coded with row context (previous group of the same row) instead of 11 flat bits. Real records3 experts, CPU only | PASS: >= -3 % SSD bytes / token on top of W24-ec + 11-bit at 24 GB, rebuild bitwise. KILL: < -1 % | **PASS Wed 15:39 (MoonShots, scripts/sbx.py, data/sbx.txt, 22 s CPU)**: -4.26 % SSD bytes / token at 24 GB (94.4 -> 90.4 MB, exact: lossless coding, decoded at load with the W24-ec codes). (a) b is NOT a function of s3 / b3 (bitwise equal to bf16(b3 - s3 / 4) in only 30 %; residual 0 / +-1 / +-2 bf16 steps 30 / 42 / 20 %) but the residual costs 2.58 b vs 11: LSB miss 720,896 -> 655,360 B (-1.31 %). (b) s3 order-0 7.86-7.91 b (sign always 0, 7 mantissa bits ~flat, exponent ~0.9 b); b3 = sign (1.000 b) + key offset to s3 8.34-8.38 b; together 16.26 b vs 22: MSB miss 1,900,544 -> 1,835,008 B (-2.95 %). Row / column contexts (left, up, JPEG-LS MED) are WORSE (+1-2 b): scales are not spatially smooth. Same at L6 / L24 / L42 (32 experts each). **CONFIRMED 15:42 (scripts/sbx_confirm.py, data/sbx_confirm.txt, 75 s CPU)**: all 512 experts of L0 / 6 / 18 / 24 / 36 / 47 in a concrete format (s3 sign + exponent and b3 exponent offset rANS-coded with one table per layer, 7 + 8 raw mantissa / sign bits; b = residual to bf16(b3 - s3 / 4), no escapes), transform round trip bitwise on every value: s3 + b3 17.62-17.81 b (raw mantissas cost ~1.5 b vs the screen), b 2.53-2.60 b. 24 GB: -3.53 % SSD bytes / token (MSB miss 1,851,392 / LSB 655,360 B; 94.4 -> 91.1 MB) with an offset table, -2.79 % if fixed records are sized by the worst expert (L47 has one at 19.5 b). Some layers have negative scales (L0): sign goes with the exponent. For BigMoE with R-16 (same decode-at-load path) |
| M-02 | MoonShots | LSBSKIP [NON-exact; added Wed 15:52]: stop reading the LSB-1 sidecar: ranks 0-1 run at MSB-3 whenever their LSB is not resident. Bytes priced 15:51, before this row (wall24 replay, 24 GB stack, 16-row checks accepting 10): LSB misses are 14.5 % of 94.4 MB / token; never reading them = -14.5 % with the LSB pool kept as a static set, -26.0 % (69.9 MB) with the LSB RAM moved to the MSB pool (C_m 174; that arm = plan 3333333333). Quality screen = my kl125_screen.py (Eval-suite own-output tf sets, per-part KLD vs the cached 4-bit reference, ~3 min per plan): plan3333333333 (the -26 % arm) and plan4333333333 (rank 1 only) | PASS: plan3333333333 within 1.10 x ar4 on all four parts (= -26 % bytes inside the gate 4433332222 failed; then an Eval-suite verdict and Kamen's call). KILL: both plans over the gate on >= 2 parts | **BETWEEN 16:02 (MoonShots, data/kl125/screen_lsbskip.txt; quiet lock, 2 x ~2.5 min)**: per-part mean KLD (top-1) code / gsm / ifeval / multi vs targets 0.00253 / 0.00449 / 0.02041 / 0.02055: plan3333333333 0.00463 (0.992) / 0.00670 (0.986) / 0.02840 (0.931) / 0.02545 (0.943) = OVER on all four (x1.24-1.83): the -26 % arm is dead. plan4333333333 (rank 0 keeps its LSB, rank 1 runs at MSB-3) 0.00245 (0.994) / 0.00399 (0.990) / 0.01799 (0.945) / 0.01553 (0.955) = INSIDE on all four, code at 97 % of its target (v3 4433333333: 0.00172 / 0.00275 / 0.01367 / 0.01097); ifeval top-1 0.945 vs ar4 0.947. Bytes for that arm (replay, priced 15:5x): LSB only for rank 0 = -7.9 % SSD bytes / token at 24 GB (pool 44 kept), -9.7 % with half the LSB RAM moved to the MSB pool (C_m 166 / C_l 22). A milder variant (skip rank 1's LSB only when it is not resident) sits between v3 and plan4333333333 in quality at ~-7.9 %. Non-exact: needs the Eval suite's full strict verdict (P99.9 / top-1 parts) and Kamen's call; it spends v3's KLD headroom, so it does not stack with R-30-type drops inside the same gate |
| M-03 | MoonShots | POM, precision on miss [NON-exact; added Wed 16:04; from M-02]: SSD bytes come only from misses, quality from every use, so cut precision only where the SSD pays. A miss whose best rank in the check is >= R2 reads only the top two code planes (MSB-2, s2 = 2 s3, b2 = b3 + s3 / 2); the expert stays resident at 2 planes until a use at rank < R2 reads its third plane; resident 3-plane experts are untouched. Screen (CPU): plane split of the code entropy from records3 (H(top-2) + H(plane 3 given top-2) = 2.72 b), strict LFU replay at the 24 GB stack (C_m 159 / C_l 44, 16-row checks accepting 10; RAM per slot unchanged, conservative) for R2 = 6 / 4 / 2: bytes, and f = share of rank >= R2 gate mass served at 2 planes; predicted KLD per part = v3 + f x (the measured plan gap: 4433332222 for R2 6, 4422222222 for R2 2; R2 4 interpolated) | PASS: some R2 gives >= -10 % SSD bytes / token with predicted per-part KLD inside 1.10 x ar4 on all four parts (then one harness run to check the interpolation). KILL: < -5 % at every R2 inside the gate | **PASS on predicted quality 16:0x (MoonShots, scripts/pom24.py, data/pom24.txt, 21 s CPU)**: code planes H(top-2) 1.761 + H(plane 3 / top-2) 0.963 b: a 2-plane miss reads 1,294,336 B (68 % of 1,900,544), the plane-3 upgrade 606,208 B. 24 GB stack 94.4 MB / token -> R2 8: 88.2 (-6.6 %); **R2 6: 82.7 (-12.4 %)**, 16.9 % of the rank >= 6 mass (16.3 % of its energy) at 2 planes, predicted KLD 0.00187 / 0.00301 / 0.01495 / 0.01221 (code / gsm / ifeval / multi; gate 0.00253 / 0.00449 / 0.02041 / 0.02055); **R2 4: 78.0 (-17.4 %)**, 20.0 % mass (18.7 % energy), predicted 0.00218 / 0.00351 / 0.01711 / 0.01453, inside; R2 2: 74.1 (-21.6 %), predicted over on all four. Same KLD budget as M-02's plan4333333333 (that one spends code to 0.00245 for -7.9..-9.7 %): POM is ~2x the bytes per unit of KLD, since only misses lose bits. CONFIRMATION queued (scripts/pom_harness.py + pom_runner.sh, one quiet lock): the Eval suite's tf screen with its MoE block wrapped in my process, a random 20 % of the rank >= 4 uses at 2 planes, the rest v3; PASS = inside the gate on all four parts, KILL = over on >= 2. **CONFIRMED 16:27 (data/kl125/screen_pom.txt, 298 s in one quiet lock, 6,624 wrapped MoE calls)**: per-part mean KLD (top-1) 0.00202 (0.995) / 0.00353 (0.990) / 0.01639 (0.943) / 0.01432 (0.957) = INSIDE the gate on all four (70-80 % of each target; v3 0.00172 / 0.00275 / 0.01367 / 0.01097); the interpolation held (predicted 0.00218 / 0.00351 / 0.01711 / 0.01453). So POM R2 4 = -17.4 % SSD bytes / token at 24 GB inside the gate that 4433332222 (all rank >= 6 uses at 2 planes) fails. Caveats: random uses stand in for real misses (rare experts, specific tokens); ifeval top-1 0.943 vs ar4 0.947. Engine needs (BigMoE): plane-separated SSD records (top-2 planes / plane 3 / LSB; uncompressed the 2-plane read is 71 % of today's MSB miss), a tier-2 path in the expert kernel, per-slot plane state + upgrade reads. Adoption = Eval-suite strict verdict on the engine's real POM + Kamen's call. **Narrow-path re-pricing 20:55 (after R-43's KILL; data/pom24_r4a3_159_44.txt, _r8a5_, 26 s CPU each)**: 4-row checks accepting 3: v3 99.1 MB / token -> R2 6: 86.0 (-13.3 %); **R2 4: 81.1 (-18.1 %)**, 20.8 % of the rank >= 4 mass (19.5 % of its energy) at 2 planes, predicted 0.00219 / 0.00354 / 0.01726 / 0.01469, inside; 8 accepting 5: v3 97.2 -> R2 6: 84.5 (-13.1 %), R2 4: 79.7 (-17.9 %), 20.1 % (18.8 %), inside. The confirmed run (random 20 % of rank >= 4 uses) covers both energy shares, so R2 4 holds on the shipped narrow path: BigMoE can start POM at R2 4 |
| M-04 | MoonShots | POMVS, POM with variable slots [NON-exact; added Thu 15:48; from M-03 + W24's pool sweep]: W24's pool sweep says misses fall faster than RAM grows (pools x1.1 -> -12 %, x1.4 -> -41 % bytes / token), so bits per RESIDENT expert cost ~b^2.4, not b. POM keeps a 2-plane resident in a full 3-plane slot; here it takes its own size (2-bit codes + the same bf16 s / b = 1,548,288 of 2,162,688 B, 71.6 %) and the freed RAM holds more experts. Variant DG also DEMOTES instead of evicting: an LFU victim with 3 planes drops plane 3 and stays as a 2-plane resident (a later use at rank < R2 reads plane 3 only, 606,208 B). Screen (CPU, pom24.py's replay with a byte budget of C_m 159 x 2,162,688 per layer; LSB pool as v3): shipped narrow path (4-row checks accepting 3) and 16 / 10, R2 6 / 4, DG off / on: SSD bytes / token and f = share of the rank >= R2 gate energy served at 2 planes; predicted KLD = v3 + f x gap(R2) as in M-03 (conservative there: measured 0.00202 vs predicted 0.00221 at f 0.20) | PASS: some variant <= -10 % SSD bytes / token vs POM R2 4 on the narrow path (81.1 MB / token) with predicted KLD inside 1.10 x ar4 on all four parts (if its f is over M-03's confirmed 20 %, one harness run at that f confirms). KILL: < -4 % for every variant inside the gate | **BETWEEN 15:51 (MoonShots, scripts/pomvs24.py, data/pomvs24.txt, 40 s CPU)**: narrow path (4 / 3), RAM per layer 159 x 2,162,688 B, a 2-plane resident 1,572,864 B (72.7 %: each projection padded to 16 KB as msb_sz is; the 71.6 % above was unpadded). v3 99.1 MB / token, POM R2 4 fixed slots 81.1; VS R2 4: DG 0 76.0 (-6.4 %), DG 1 (demote the LFU victim) 75.6 (-6.8 %), **VD 73.4 (-9.5 %)** = trim by SSD bytes at risk per freed RAM byte (a 3-plane resident's decayed count of uses at rank < 4 x plane 3, a 2-plane one's count of all uses x 2 planes): 181 residents / layer vs 161, 45 % of them at 2 planes, rank >= 4 energy at 2 planes 27.8 % -> predicted KLD 0.00240 / 0.00388 / 0.01879 / 0.01628, inside (code at 95 % of the gate; with M-03's measured slope ~0.0021). 16 / 10: VD -10.0 % (70.2). R2 6 variants never beat POM R2 4 (+0.8..+2.9 %). Smaller than the pool sweep's elasticity hinted: only the 2-plane share frees RAM (+12.6 % residents). 0.5 points short of the -10 % line: BETWEEN. If taken: one harness run at F 0.28 of the rank >= 4 uses confirms the quality, and the engine needs a 2-bit expert path + variable-size slots (more than R-44's reuse of the 3-bit slot). Narrow-path stack: v3 99.1 -> POM 81.1 -> POMVS-VD 73.4 -> + M-01 ~71 MB / token, still ~1.75x from 40.6. **VD + LP1 confirmation (NewIdeas GO 18:1x, low priority, a later R-44 (b) level; the variable-slot build waits for Kamen's interest in a non-exact 24 GB mode)**: replay (lpom24.py) POM R2 4 + VD + LP1 = 65.6 MB / token on the narrow path (-19.1 % vs POM R2 4), predicted code at 102 % of the gate. Harness: M-07's lpom_harness.py at F 0.28 of the rank >= 4 uses at 2 planes (VD's energy share) + F1 0.26 of the rank-1 uses at MSB-3, one quiet lock (scripts/vdlp_runner.sh). PASS: inside the gate on all four parts. KILL: over on >= 2 | Attempt 1 (lock 18:47-18:53): stopped by the runner's swap guard (+2 GB) during batch 2 of 2, no result: swap started at 7.8 GB and macOS grew it to ~10.1 GB while the 24 GB harness loaded (the M-07 run had started at 10.2 GB and did not grow); memory pressure normal after (Breakthrough 19:2x: its idle ~10.3 GB candidate process waiting for a lock was likely part of it; stopped). **Attempt 2 CONFIRMED 19:21 (lock 19:15-19:21, 374 s, data/kl125/screen_vdlp.txt, commit 1dd496e)**: per-part mean KLD (top-1) 0.00247 (0.993) / 0.00427 (0.988) / 0.01916 (0.939) / 0.01656 (0.956) = INSIDE the gate on all four (98 / 95 / 94 / 81 % of each target; predicted 0.00259 / 0.00421 / 0.01992 / 0.01747). So POM R2 4 + VD + LP1 = 65.6 MB / token on the narrow path (-19.1 % vs POM R2 4, -34 % vs v3's 99.1) inside the gate, with code at 98 % of its target (thin margin; ifeval top-1 0.939 vs ar4 0.947). Caveats as M-03 / M-07 (random uses stand in for real misses; the engine's real POM still needs the suite's strict verdict). BIG message sent to the orchestrator 19:2x (non-exact quality pass = Kamen's call: whether a non-exact 24 GB mode with variable slots, a 2-bit kernel and a slot allocator, is wanted) |
| M-05 | MoonShots | SUFX, the 125B's own past outputs as a co-drafter [exact; drafter track, math gap; added Thu 15:55; prior art REST (He et al. 2023), SuffixDecoding (2024)]: GSM8K CoT repeats scaffold phrasing and restates the question's facts ('2. He increases the weight by 50%. The increase is ...'); DFlash2 L1-20 learned from 22.6K tokens only. A suffix lookup over a datastore of the 125B's earlier completions + the current prompt and text proposes the next 15 tokens; per check the engine verifies the datastore block instead of DFlash2's when its context match is >= m tokens. Screen (CPU, existing data): D-02b's math walk (60 GSM8K answers, 2,046 checks, DFlash2 accepted count per check), datastore = the other 59 answers (leave-one-out) + the sample's own prompt / prefix; m tuned on one half of the samples, scored on the other (2 folds); also the oracle max(DFlash2, datastore) per check, prompt-lookup alone, and the datastore grown with the Eval suite's ref decodes (50 MATH500 + 80 MGSM) for the scaling sign. Kept per check on the walk's own trajectory (positions fixed) | PASS: held-out kept per check >= +0.35 (7.00 -> 7.35, +5 %) on math (then a datastore-size curve decides whether to generate more 125B math text). KILL: < +0.10, or the oracle < +0.30 | **KILL 16:00 (MoonShots, scripts/sufx_math.py, data/sufx_math.txt, 4 s CPU; the 15:56 run drafted from ids[:anchor] instead of ids[:anchor + 1], fixed after reading MTP Head's walk code, same verdict)**: math (60 samples, 2,046 checks, base 7.00 reproduced): datastore + prompt lookup held-out 6.96 (-0.03), **oracle max(DFlash2, datastore) per check 7.08 (+0.08)**; prompt lookup alone oracle +0.05, datastore alone +0.05; 7 / 15 / 30 / 59 other answers: oracle +0.05 / +0.07 / +0.08 / +0.08 (no growth with size); + the Eval suite's 130 ref answers (MATH500 + MGSM) +0.08. The tuned m goes to 15-16 (switch almost never used): where the datastore matches long, DFlash2 already keeps more (s 3.8 vs a 9.7). DFlash2 already drafts what the 125B's own past text could offer; math's misses are not lookup-shaped (first misses: 57 % word choices, 19 % symbols / LaTeX, 15 % spaces / newlines, 9 % digits). Side note, code (24 HumanEval answers, 666 checks, 9.57): prompt lookup as a switch at m 9 (both folds) gives held-out 10.38 (+0.82, +8.6 %) on 41 % of checks (s 14.4 vs a 12.4), because the answers restate the prompt's signature + docstring (the cross-sample datastore hurts code: -0.31); known lever (spec125 drafts with prompt lookup; Leap's L1-10: +3-4 % at most on agentic), so not a new row: the D98 drafter could take it as a switch where the engine verifies 16-row blocks |
| M-06 | MoonShots | RECYCLE, the 125B's own verify rows after the first miss as the next draft [exact; drafter track; GO NewIdeas Thu ~17:2x; prior art Token Recycling (2024), Lookahead / Jacobi decoding]: a 16-row check computes the 125B's argmax at every row, and the rows after DFlash2's first miss (conditioned on its wrong token) are thrown away today. If the wrong token barely changes what follows, those rows already hold the next block. Per check with a first miss at draft position k: kept_r = the matching run of the recycled tail (each verify row's argmax after the miss, positions k+1 on) against the 125B's next tokens; kept_f = the matching run of DFlash2's fresh draft from the next anchor. Step 1: existing per-check logs with every verify row's argmax + the next block's draft + the true tokens (Leap's SD study, MTP Head's D-02a walk): CPU only if they exist; else one capture (DFlash2 drafts + the 125B on them) in a <= 10-min quiet lock. Oracle bound first, on code and math | KILL: oracle E[max(kept_r, kept_f)] - E[kept_f] < 0.3 tokens per check on both code and math (no selection rule can beat the oracle). PASS: >= +0.8 on either, then a real selection rule (e.g. the recycled rows' p1) at zero extra verify rows. In between: report | Step 1 (17:4x): no existing log holds the verify rows after a miss: Leap's SD study (Mon) is a desk study and its scoring is teacher-forced (refine19's loop), R-40 step 3x logged one diagnostic check, and MTP Head's D-02a walk is teacher-forced along the 125B's tokens. So the capture: scripts/recycle_capture.py (Phase A: D-02b's walk with every check's 15 drafts kept, checked against the stored walk; Phase B: the suite's 'ref' 125B teacher-forced over ids[:anchor + 1] + the 15 drafts for 100 sampled first-miss checks per domain = the 16 verify rows, ~50K tokens; rows 0..acc checked against the 125B's own tokens) via scripts/recycle_runner.sh in one quiet lock. **KILL 17:30 (data/recycle.txt, recycle_run.txt, recycle_{code,math}.npz; lock 17:27-17:30, 195 s: Phase A 39 s, Phase B 64K tokens in 164 s)**: Phase A reproduced all 84 stored walks exactly; the verify rows up to the miss agree with the 125B's tokens 797 / 801 (code) and 586 / 586 (math). 100 first-miss checks per domain: **code E[kept_f] 7.13, E[kept_r] 0.65 (of an 8.0-row tail), oracle E[max] - E[kept_f] = +0.02; math 5.28 vs 0.59 (10.1-row tail), oracle +0.06** (line 0.3); the tail beats DFlash2's fresh draft on 2 % / 5 % of checks, by ~1 token there; no miss-depth bucket reaches +0.1 (math acc 0-2 +0.08). After the first miss the 125B follows its own wrong prefix: the rows conditioned on DFlash2's wrong token match the true continuation for ~0.6 tokens, while DFlash2's fresh block from the corrected anchor keeps 5-7. The first miss is a divergence point, not a one-token detour, so verify-row recycling has nothing to offer this drafter. (Index shifts for insert / delete misses were not scored; with kept_r ~0.6 against kept_f 5-7 they cannot reach 0.3.) |
| M-07 | MoonShots | LPOM, precision on miss for the LSB plane [NON-exact; added Thu ~16:03; from M-02 + M-03; M-06 is kept for the verify-tail screen offered to NewIdeas]: v3 reads the LSB sidecar (720,896 B) on every LSB miss of an expert whose best rank in the check is 0 or 1 (~14 of POM's 81 MB / token on the narrow path). LP1: an LSB miss at best rank 1 is skipped (the use runs at MSB-3 with the suite's group-mean bias, no read, no insert); rank 0 still reads. LP01: rank-0 misses skipped too. Screen (CPU, the M-03 / M-04 replay, narrow path 4 / 3 and 16 / 10, LSB pool C_l 44 strict LFU as v3): SSD bytes / token with POM R2 4 (fixed slots, R-44's build) and with M-04's VD, and f = share of the rank-1 (rank-0) gate energy on accepted rows served without the LSB; predicted KLD = v3 + f_MSB x gap(R2 4) + f_1 x (plan4333333333 - v3) [+ f_0 x (plan3333333333 - plan4333333333)] from M-02's measured plans | PASS: LP1 on top of POM R2 4 gives <= -6 % SSD bytes / token vs POM R2 4 alone on the narrow path with predicted KLD inside 1.10 x ar4 on all four parts. KILL: < -3 %, or over the gate | **PASS on predicted quality 16:04 (MoonShots, scripts/lpom24.py, data/lpom24.txt, 9 s CPU)**: narrow path (4 / 3): LSB bytes / token v3 13.99 MB -> LP1 6.21 (-56 %), rank-1 energy served without the LSB 26.1 %; **POM R2 4 + LP1 = 73.3 MB / token (-9.6 % vs POM R2 4's 81.1)**, predicted KLD 0.00239 / 0.00387 / 0.01838 / 0.01588, inside (code at 94 % of the gate; M-03's measured slope would put it lower). 16 / 10: 70.5 (-9.7 %), inside (93 %). LP01 (rank 0 too): -17 % but 2x over the gate on all four. POM R2 4 + VD + LP1: 65.6 (-19.1 %) but predicted code at 102 % of the gate on the narrow path (99 % at 16 / 10): one of VD or LP1, not both, at today's KLD budget. LP1 saves what VD saves at the same quality, and the engine needs no new kernel: a rank-1 use whose LSB is not resident runs on the 3-bit path ranks 2-9 already use (per-use tier choice), and the miss reads nothing. Narrow-path stack: v3 99.1 -> POM 81.1 -> + LP1 73.3 -> + M-01 ~71. CONFIRMATION next (pre-registered): M-03's harness with a random 20 % of rank >= 4 uses at 2 planes AND a random 26 % of rank-1 uses at MSB-3, one quiet lock (~5 min GPU); PASS = inside the gate on all four parts, KILL = over on >= 2. **CONFIRMED 16:55 (scripts/lpom_harness.py via lpom_runner.sh, data/kl125/screen_lpom.txt, 325 s in one quiet lock 16:49-16:55, 6,624 wrapped MoE calls)**: per-part mean KLD (top-1) 0.00221 (0.994) / 0.00388 (0.988) / 0.01770 (0.943) / 0.01515 (0.958) = INSIDE the gate on all four (87 / 86 / 87 / 74 % of each target; predicted 0.00239 / 0.00387 / 0.01838 / 0.01588). Over M-03's POM-only run (0.00202 / 0.00353 / 0.01639 / 0.01432) LP1 adds +0.00019 / +0.00035 / +0.00131 / +0.00083, as predicted from M-02's plan (+0.00019 / +0.00032 / +0.00113 / +0.00119). So POM R2 4 + LP1 = -9.6 % SSD bytes / token beyond POM on the narrow path (73.3 MB / token vs v3 99.1: -26 %) inside the gate. Caveats as M-03: random uses stand in for real misses; ifeval top-1 0.943 vs ar4 0.947. Engine (BigMoE, with R-44 (b)): a rank-1 use whose LSB is not resident runs on the tier-3 path, and the LSB miss reads nothing (no insert). Left in the budget: VD + LP1 (-19.1 %) was predicted at 102 % of the code gate; measured runs land ~8 % under prediction, so one more harness run (F 0.28 of rank >= 4, F1 0.26) would settle it, if the variable-slot build is wanted. **Routed (NewIdeas 17:22): R-44 (b)'s second level (BigMoE, arms A / P / L).** |
| M-08 | MoonShots | SPLIT45 [exact pricing for R-58's S45-T; 45 GB route; proposed Fri 16:34, waits for NewIdeas' GO; CPU replay <= 10 min, PAUSE-gated]: REF2-T needs every used expert's MSB-3 part (2,048,000 B) but the LSB part (720,896 B) only when its best rank in the check is < 2, so at REF2's RAM (C x 2,768,896 B per layer) its two pools (C_m, C_l) can hold more experts. wall24's checks on the expA traces (4 sets, 800 tokens, tokens 0-399 warm, 400+ measured), 16 rows advancing 8 (S45's mean kept 8.27) and 10; strict pools (no free staging); lru (the ext default) and lfu 0.99; joint MSB + LSB misses count as one read. At the RAM of REF2 C 192 / 224 / 256 / 288 (S45 36.3 GB, S45b 40.5, ~45, ~49 GB): REF2 reads and MB per kept token vs REF2-T per C_l (C_m from equal RAM), the best split, SSD ms per kept token at 10.6 GB/s, RAM-side expert bytes per check (R-58's -15 %). Calibration: S45 measured 51-86 misses per kept token at C 192 (true-token rows give fewer; read ratios). scripts/split45.py -> data/split45.txt | PASS: REF2-T reads per kept token -30 % or better vs REF2 at equal RAM, with a C_l that fits, at >= 2 of the 4 RAM points -> S45-T runs at that split. KILL: under -10 % at every RAM point (S45-T's case rests on expert bytes only). BETWEEN otherwise | **KILL by the letter on reads; bytes -40-44 % (MoonShots 17:38, NewIdeas GO 16:54; scripts/split45.py, data/split45.txt; 77 s CPU)**. Best REF2-T split vs REF2 at equal RAM, reads per kept token (A 8 / A 10, lru / lfu): C 192 -8.3 / -5.6 / -7.5 / -5.2 %; C 224 -9.3 / -5.4 / -9.5 / -5.2 %; C 256 -7.7 / -1.1 / -7.1 / -0.2 %; C 288 +0.8 / +6.5 / +1.3 / +6.8 %: never -10 %. Why: MSB misses fall ~30-40 % (C 224 A 8 lru: 26.2 full reads -> 18.8 MSB) but the LSB pool adds 5-9 small reads per token (best rank < 2 in 16 rows: ~17 distinct experts per layer per check). SSD MB per kept token at the same splits: -40 to -44 % (C 224 A 8 lru: 72.4 -> 40.8 MB at C_m 269 / C_l 96); RAM-side expert bytes per check -19.4 % everywhere (R-58's -15 %). Second reading (not the line): my round-24 burst bench (data/mtlio_bench.txt) has 4 x 0.69 MB = 1 x 2.77 MB and 8 experts 1.79 ms whole or chunked, so bursts of >= 4 reads are bandwidth-bound and 16-row checks' SSD time should follow bytes (b45's 24 GB AR regime, ~1 read per layer, is latency-bound instead); S45-T measures it. For S45-T: C_l must be >= ~96 per layer (C_l 0: ~100 LSB reads per token; 32: +25-145 % reads vs REF2; 64: -0 to +66 %); best splits: RAM of C 224 -> C_m 257-269 / C_l 96-128; C 256 -> 301-312 / 96-128; C 288 -> 333-344 / 128-160. vs today's REF2 C 224: REF2-T at ~45 GB (C_m 301 / C_l 128) reads -33 %, MB -59 %; at ~49 GB (333 / 160) -48 % / -69 % (REF2 C 256 alone: -27 % / -27 %). Calibration: the replay's REF2 C 192 gives 35.9 misses per kept token vs S45's measured 69 (true-token rows: union 67 experts per layer vs ~100 live), so read ratios only |
| M-09 | MoonShots | CONTRIB16 [NEW BET for the bytes wall (Kamen 18:45 FAST INTEGRATION item 4); non-exact, same-quality target; added Fri 19:20]: per-check expert budgets (MoE-Spec, AcceptMoE, UGB) choose the window's kept experts by ROUTER MASS and fail IFEval by 5-7; the router is flat (top-1 ~21 % of the top-10 weight, L1-17) and B1 showed gate signals miss what matters. Bet: choose by each expert's actual output contribution sum ||w E_e(x)|| over the window's rows, estimated at run time by a free static proxy (router mass x the expert's mean output norm). Prior art (web 19:1x): norm-based expert utility exists only as training-time designs (Autonomy-of-Experts, Routing-Free MoE 2604.00801, router-coupling loss ICLR'26); no inference-time verify-window budget chosen by contribution found. Screen (scripts/contrib16.py, shared GPU ~3 min, rough): Fit12's captured MoE inputs (layers 2 / 12 / 24 / 36 / 46), real router + 4-bit experts + gated shared expert (L1-17 / B1's code); windows of 16 consecutive tokens (tokens 1024-2047; static scales from tokens 0-1023); keep B of the window's union by MASS / CONTRIB (oracle) / MSCALE (proxy), renormalized; metric = L1-17's MoE-block output error vs the exact top-10 block; reference = per-row top-8 (passed the Eval suite 5/5) recomputed here; descriptive arms added 19:22, before the run, not in the lines: MASS+mu / MSCALE+mu = dropped experts replaced by w x their mean output. Prediction: CONTRIB needs ~20-35 % fewer experts than MASS at top-8's error; MSCALE gets 20-40 % of that gap (BETWEEN 45 %, PASS 30 %, KILL 25 %) | PASS: at top-8's block error CONTRIB needs >= 25 % fewer experts per 16-token window than MASS AND MSCALE captures >= half of that saving -> a runtime design + an Eval-suite run. KILL: CONTRIB needs < 10 % fewer than MASS at that error (router mass is already the right signal). BETWEEN otherwise | **KILL 20:54 (shared GPU, rough, 19 s; data/contrib16.txt)**: natural window union 76.3; per-row top-8 63.6 at block error 0.0259. Experts per window at that error: MASS 50.0, CONTRIB (oracle) 51.4, MSCALE 51.1: contribution norms pick no better than router mass (greedy by norm ignores the renormalization). **Descriptive arm, the lead**: dropped experts replaced by w x their mean output (from tokens 0-1023, no renormalization) need 38.8 (MASS+mu) / 36.9 (MSCALE+mu) experts at the same error: -22-26 % vs MASS, -49-52 % vs the natural union; at B 40 the error 0.0598 -> 0.0234. In-sample and confounded with 'no renormalization': follow-up M-09b separates both |
| M-09b | MoonShots | MUCOMP [follow-up of M-09's descriptive arm; non-exact, same-quality target; added Fri 20:55]: a dropped expert's contribution w E_e(x) replaced by w mu_e, mu_e = the expert's mean output (2,560 floats, ~5 KB vs 2.77 MB of weights). Prior art (2 web searches 20:5x): skipping without compensation (Lu et al. 2402.14800, MoDES 2511.15690), zero-output experts (ZEDA); mean replacement is classic in CNN channel pruning; not found for MoE experts. Screen (scripts/mucomp16.py, shared GPU ~1 min, rough; M-09's inputs and metric): mu from tokens 0-2047, evaluated OUT of sample on tokens 6144-8191 (and in sample on 1024-2047 for reference); window budgets by router mass with dropped experts (a) renormalized, (b) dropped without renormalization, (c) replaced by w mu; and the AR-consistent per-row form: each row keeps its top-k (k 4-8) renormalized vs top-k + w mu for ranks k+1..10 (a served model defined per token, so spec decoding stays exact vs its own AR). Reference: per-row top-8 renormalized (passed the Eval suite 5/5), its block error and union. Prediction: (c) out of sample keeps >= 2/3 of its in-sample gain; per-row top-6 + mu ~top-8's error at union -20 % vs top-8 | PASS: out of sample, (c) needs <= 0.8x (a)'s experts per window at top-8's error AND per-row top-7 or top-6 + mu has block error <= top-8's -> an Eval-suite candidate (per-row form first: AR-consistent). KILL: (b) within 10 % of (c)'s experts at that error (the gain is skipping the renormalization, not mu), or out of sample (c) needs >= 0.9x (a)'s experts. BETWEEN otherwise || **KILL 21:02 (shared GPU, rough, 31 s; data/mucomp16.txt)**: out of sample (tokens 6144-8191; natural union 73.0, per-row top-8 61.0 at block error 0.0259) experts per window at top-8's error: renorm 49.7, drop 40.3, mu 40.5 (mu/renorm 0.82; drop/mu 0.99 -> KILL): mu adds nothing out of sample; M-09's in-sample edge (37.0 vs drop 39.8) was memorization, and most of the 'lead' was skipping the renormalization. Per-row top-k + mu out of sample: k 7 1.18x top-8's error (union -24.9 %), k 6 1.78x. SIDE FINDING (the real effect): dropping experts WITHOUT renormalizing the kept weights beats renormalizing: window budgets need 40.3 vs 49.7 experts at top-8's error (-19 %), and per-row top-8 + mu (~= no renorm) has 0.71x top-8-renormalized's error (in sample 0.55x). L1-17's top-k and EvalSuite's b12llk8 renormalize, so the top-8 lever (RANKED_45GB #11) may be judged in its worse form; direct per-row no-renorm check = M-09c |
| M-09c | MoonShots | NORENORM [from M-09b's side finding; non-exact, a served-model choice per token (AR-consistent); added Fri 23:43]: per-row top-k WITHOUT renormalizing the kept weights (the dropped experts' contributions just removed; the model renormalizes over its top-10 only) vs top-k renormalized over the k kept (L1-17, EvalSuite's b12llk8). Screen (scripts/norenorm16.py, shared GPU ~1 min, rough; M-09b's inputs and metric, out of sample on tokens 6144-8191 + in sample 1024-2047): block error and window union for k 5-8, both forms. Prediction: top-8 no-renorm ~0.7x top-8-renorm's error; top-7 no-renorm ~1.1-1.2x | PASS: out of sample, top-7 no-renorm block error <= top-8-renorm's (a strictly better lever: union -25 % instead of -16 % at no more error) -> NewIdeas' same-scores batch (RANKED #11) in the no-renorm form. KILL: top-8 no-renorm >= 0.9x top-8-renorm's error (no real gain). BETWEEN otherwise (then: top-8 no-renorm as the better form of #11 at the same union) || **BETWEEN 07:30 (shared GPU, rough, 9 s; data/norenorm16.txt)**: out of sample (in sample within 0.01x), per-row block error vs top-8 renormalized (0.0259): top-8 no-renorm 0.66x at the same union (61.0, -16.4 % vs natural); top-7 no-renorm 1.11x at union 54.9 (-24.9 %; line <= 1.00x); top-6 1.71x; renormalized top-7 2.03x, top-6 3.74x. So the no-renorm form is the better form of the top-k lever: keep the kept experts' top-10-normalized weights and drop the rest without renormalizing (L1-17's top-k and EvalSuite's b12llk8 renormalize). For the 45 GB same-scores batch (RANKED #11): top-8 no-renorm (block error 0.66x of the form that failed the strict gate at 24 GB), and top-7 no-renorm as the -25 % union candidate |
| M-09d | MoonShots | TOPK-KLD [the real-model check of M-09c; non-exact; Kamen's 19:55-20:45 test hour; added Sat 19:58]: the Eval suite's own tf screen (my M-07 harness, read-only machinery), v3 plan 4433333333 for every expert, each row keeps only its top-8 experts WITHOUT renormalizing (their top-10-normalized weights; the dropped contributions just removed). Per-part mean KLD vs the 4-bit reference, next to v3 and the strict gate 1.10 x ar4 (scripts/topk_harness.py 8 norenorm, one quiet lock <= 10 min). Prediction: v3 + top-8 renormalized failed the strict gate at 24 GB; the no-renorm form's block error is 0.66x, so KLD rises ~half as much: inside on code / gsm, borderline on ifeval | PASS: inside the gate on all four parts (the top-8 lever, -16 % experts per check, cleared at KLD level in its better form -> the same-scores batch). KILL: over on >= 2 parts | **KILL 21:18 (shared GPU, C 64, 172 s; data/kl125/screen_topk8_norenorm.txt)**: v3 + per-row top-8 without renormalization, KLD vs the 4-bit reference: code 0.00348 / gsm 0.00633 / ifeval 0.02964 / multi 0.02539 vs the gate 0.00253 / 0.00449 / 0.02041 / 0.02055 = 138 / 141 / 145 / 124 %: over on all four (v3 alone 0.00172 / 0.00275 / 0.01367 / 0.01097; top-1 agreement 0.992 / 0.986 / 0.931 / 0.946). M-09c's 0.66x block error does not bring top-8 under the strict gate on top of v3; the top-8 lever (RANKED #11) stays a task-test (same-scores) question. Earlier attempts 19:59 / 20:07 stopped by a heat pause and the swap guard |
NewIdeas adds a batch of new one-line ideas at every GO wake; the older sections below are the archive and the long records.
**Count since Thu 19:15 (updated Sat 06:12): tested 43 | killed 33 | alive 7** (alive: B12LL exact 5/5 x2; C_m 59 + hc8-lazy + streamed
embed = bitwise trims, fits < 12 GB to ~2K prefill; DFlash2 code/math (= MTP at 27.5 GB); block-16 DFlash2 x1.44 under a budget only; L4b
running; batch-invariant dense kernel (design); UGB-0.5 @12 GB (4/5, IFEval -5.0 at the boundary, scorer re-score pending)).
Tonight's kills: L1-06 joint verify, L1-09 early exit (layer 40 agrees 0.057), L1-10 copy rows (x1.03), trees (+5 %), L2-05 / L2-08, B1.
LINE 1 READING (Sat 06:12): at the largest setting allowed (< 45 GB process -> C_m 192, 27.5 GB) an 8-row exact pass is 144-172 ms
(stall 64-95 + compute ~70); code-350 needs ~14 ms: ~11x. The 125B cannot run in RAM here; bytes per pass is the binding lever.
suite; L4 self-distill running; batch-invariant dense kernel (design); C_m 59 transient cut; UGB strict-bar @wide passes; joint verify).
Formula: visible tok/s = 1/(1-F) x S x A / T_pass. Line 1 needs A / T_pass ~2.5x today's on code; line 2's exact ceiling is physical
(~+15 % headroom: slots + compute), beyond that only routing changes.

> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
Goal (Kamen's): a small, cheap drafter that keeps ~14 of 16 per 16-row check (~98 % per guess) on chat, math, code and agent text.
Drafts never change the output, so everything here is exact-safe; adopting a drafter needs no exactness call. Drafting budget:
<= ~7 ms per block at 45 GB (the 350 plan's check is ~34-40 ms; DFlash2 costs 12.5 ms today, 7.9 ms with R-15). Rules as above
(10-min screens, PASS / KILL first, one new bet in flight), plus: no paid GPUs; free cloud only on an account Kamen creates;
only really small models train on the laptop; downloads go via the orchestrator for Kamen's yes.
Measured ceilings (D-01; Chat1000 REPORT 12, Oct 22: teacher-forced on the 125B's own greedy outputs; kept per 16-row check,
per-token rate in brackets):
| domain | DFlash2 today (L1-20) | near-copy (the 125B, 10 % of experts) | AutoRound int4 twin | float-order floor | Kamen's goal |
|---|---|---|---|---|---|
| code (HumanEval+) | 9.57 (0.924; D-02b) | 13.4 (0.983) | 14.8 | 15.2 | 14 |
| math (GSM8K CoT) | 7.00 (0.871; D-02b) | 12.2 (0.970) | 14.2 | 15.0 | 14 |
| instructions (IFEval answers) | D-02 | 6.6 (0.872) | 10.6 (0.947) | 13.2 (0.977) | 14 |
| chat, 8 languages | D-02 | 8.0 (0.901) | 11.1 (0.952) | 13.7 | 14 |
| agent turns (vs the same runtime) | D-02 | 6.8 (0.886) | - | 11.0 (0.959: two exact runtimes) | 14 |
Reading: the float-order floor is the 125B against itself with another summation order; a drafter that is not the verify's own
arithmetic can't reliably beat it, because those near-ties are decided by rounding. So 14 of 16 is OPEN on code and math (twins
reach 14-15; today's drafter is the gap), and ABOVE the floor on instructions, chat and agent text, whose exact ceiling is
11.0-13.7 with ~0.95 per token (10.6-11.1) the realistic top. Past that only accepting near-ties helps (non-exact, Kamen's call);
Chat1000 REPORT 2 found that rewrites agent actions (a wrong port, a wrong tool), so it is not proposed.
Track lines: a direction stays alive only if its screen moves DFlash2's per-token agreement by >= +1 point on code or math, or by
>= +1 point toward the twin level on chat / agent, at <= ~7 ms of drafting per block. Milestones under the ceilings (Kamen's 14
stays the goal wherever the floor allows it): code >= 13.0 and math >= 12.5 per 16-row check; chat >= 10, agent >= 9; held out.
Why it matters at 24 GB too (R-43, BigMoE b43): there each extra checked row costs ~12 ms of SSD wait, so 16-row checks beat today's MTP passes only at ~11-13 kept of 16 on code (break-even ~8.9-10.5) and ~8+ on math.
| id | who | screen | PASS / KILL (written first) | result |
|---|---|---|---|---|
| D-01 | NewIdeas | CEIL [desk check of existing data; lines and reading written together]: per-domain ceilings from Chat1000 REPORT 12 + Leap L1-18 / L1-20 | PASS: 14 of 16 open on >= 1 everyday domain. KILL the 98 % framing on any domain whose float-order floor is < 14 | **DONE Wed 15:4x**: open on code (floor 15.2) and math (15.0); above the floor on IFEval (13.2), chat (13.7) and agent turns (11.0). Table above |
| D-02 | Leap | DOMAINS: DFlash2 (L1-20 weights, block 16, 1-step replay, the official loop) on Chat1000's tf_gen sets (code / gsm / IFEval / multi) + the 12 agent turns, against the 125B's own tokens: kept per check, per-token rate, rate per depth, and the share of first misses where the 125B is >= 0.99 sure, per domain | Measurement (<= 10 min): sets the per-domain gap (near-copy minus DFlash2) that D-03..D-06 are judged against; SPLIT by Leap 15:44 (agreed by NewIdeas): D-02b = the replay on code (dfl2d_code, 24) + math (dfl2d_math, 60) first: kept per check, per-token rate, rate per depth. The stored features keep the 125B's top-1 token but not its probability, so the >= 0.99-sure share of first misses comes with D-02a. D-02a = dfl2c_extract features + the 125B's top-1 p for IFEval (30) / multi (24) / agent (12 turns), and code + math p (~10-15 min); then D-02b on those three; D-02b DONE 15:48 (Leap, data/d02b_screen.txt): code 666 checks, kept 9.57, per token 0.924, P(15) 0.276, depth 1->15 0.94->0.86; math 2,046 checks, kept 7.00, per token 0.871, P(15) 0.086, depth 0.91->0.79 (near-copy 0.983 / 0.970). D-02a + the other three domains handed to MTP Head (TrainLab) with the role change. **D-02a walk DONE 17:4x (MTP Head, ~/TrainLab/data/d02b_walk_d02a.txt; capture bit-equal to Leap's on math)**: IFEval 30 samples, 2,850 checks: kept 2.45, per token 0.594, depth 1 0.629; multi 24, 3,294 checks: 2.06, 0.515, depth 1 0.505; agent 12 turns, 637 checks: 3.01, 0.672, depth 1 0.622. Near-copy 6.6 / 8.0 / 6.8; MTP Head's prediction was ~5 / ~4.5 / ~5. Next (MTP Head): the miss table, incoai's original weights on the same sets, then D-11. **D-02a RESULT 19:13 (MTP Head, REPORT 'D-02a RESULT', data/d02a_miss.txt, 8080203)**: near-ties (top-2 within 0.1) are only 5-15 % of DFlash2's first misses in every domain, so the exact ceiling is not the limit. First misses at depths 1-3 (sure = the 125B >= 0.99 / tie / mid): code 0.10 / 0.13 / 0.78, math 0.14 / 0.08 / 0.78, IFEval 0.08 / 0.15 / 0.76, multi 0.26 / 0.09 / 0.66, agent 0.33 / 0.08 / 0.59. Deeper misses are mostly sure (lookahead failures): code sure 0.27 / 0.72 / 0.85 at depths 4-7 / 8-11 / 12-15, math 0.33 / 0.53 / 0.71. Base rate over all positions: the 125B is >= 0.99 sure on 0.89 of code tokens, 0.80 of math, 0.34 of IFEval, 0.39 of multi. Route: code depths >= 4 sure 0.61 -> knowledge; math 0.46 -> mixed (D-13 next). Incoai's original weights = L1-20 within 0.07 (D-11 KILL) |
| D-03 | TrainLab | NGRAM [the track's bet in flight; prior art (NewIdeas web pass Wed ~16:0x): arXiv 2601.21204 ("Scaling Embeddings Outperforms Scaling Experts") names its n-gram embedding + a light linear projection as a possible ultra-fast drafter, as work in progress with no result; no published drafter reads such a table, so: proposed, untested]: does the 125B's own 51B n-gram (PLE) table carry next-token information DFlash2 lacks? On L1-18's code data (data/dfl2d_code; + gsm if quick): take the 16 n-gram rows of the correct prefix (mtplx ngram_row_layout hashing) and fit a ridge map from the 2,560-d row vector to the target token's LM-head row on half the positions. Score top-1 / top-4 on the other half (nearest LM-head row over the vocab), overall and on positions where the 125B is >= 0.9 sure (and on DFlash2's first misses if headroom18's files give them). Control: the same probe on the previous token's embedding alone | PASS: n-gram probe top-1 >= 30 % on held-out positions where the 125B is >= 0.9 sure, AND >= +10 points over the control. KILL: < 10 %, or <= control + 3 points | **KILL Wed 16:21 (MTP Head / TrainLab, REPORT 'D-03 NGRAM' 4320e13; data/d03_ngram.txt)**: halves = whole samples (12 / 12, two folds pooled), ridge with CV'd lambda, 6,375 held-out code completion rows. Top-1 on the 5,925 rows where the 125B is >= 0.9 sure: n-gram rows 5.6 % vs token embedding (control) 9.3 % (-3.7 points; top-4 8.9 vs 12.4); both together 9.8 %. gsm (information): 14.3 vs 19.2 %. Hashing checked (checkpoint constants = derived; dequantize = MLX bit for bit). The table's vector is knowledge for layer 1's gated projections, not a linearly readable next-token code. DFlash2's first misses not scored (no per-position list) |
| D-04 | Leap | SD1 [pre-registered Mon 14:20, TABLE_350; GO now, after R-40 INV0]: acceptance-aware loss at block 16 (each depth's loss weighted by R_j, stop-gradient) | its pre-registered lines (+3-8 % expected) | **DEAD: already answered by R-05 (SD1 resized to the rapid protocol, 21 steps from L1-20): W / U = 0.970 (90 % CI 0.954-0.986) at block 32, 0.976 at block 16, so the R_j weighting hurts (Leap, Wed 15:5x). No new loss forms for now: breadth first** |
| D-05 | TrainLab (moved from BigMoE by the roles, Wed 15:5x; BigMoE shares its engine's MTP logs) | MTPHYB [seed 1; fits Kamen's MTP-head focus]: the native MTP head's per-depth agreement at depths 1-3 (teacher-forced or from engine logs, the 125B's own tokens) on code / gsm / IFEval / agent, vs DFlash2's (D-02; L1-20 code 0.94 / 0.92 / 0.90) | PASS: MTP head >= DFlash2 + 3 points per token at depths 1-2 on >= 2 domains (a hybrid, MTP for depths 1-2, DFlash2 after, then prices at ~+2.5 ms per MTP step). KILL: <= DFlash2 + 1 point on every domain | **BETWEEN 19:25** (TrainLab; REPORT 'D-05 ... READ-OUT', data/d05_compare_exact.txt): same 16 prompts (8 code + 8 gsm), same cut (the answer's first end token), MTP chain k = 3 in BigMoE's bitwise exact mode vs DFlash2 L1-20, depths 1 / 2 / 3: code 0.972 / 0.981 / 0.947 vs 0.969 / 0.941 / 0.943, gsm 0.918 / 0.898 / 0.874 vs 0.904 / 0.885 / 0.858 (counts in REPORT). Cost: MTP k = 3 is 4.5 ms per check with the MTP layer resident (8.3 ms streamed, ~1.3 ms per extra step); DFlash2 is 12.5 ms. Hybrid upper bound at 45 GB: +4.1 % / +3.5 % tokens per check for +2.9 % check time, so net <= +1 %, not worth a build. For D98 the gap is depths 4-15 (code 0.86-0.98, gsm 0.68-0.91), not depths 1-3 Follow-up ORACLE-PREFIX (true tokens at depths 1-3, then the block) declined by NewIdeas 19:3x: L1-20's 2-step 'tail 0.9' (its own confident prefix kept, in distribution thanks to partial masking) already moved block 16 9.54 -> 9.58 and block 32 10.23 -> 10.51 (TABLE_350 L1-20), and with depths 1-3 at ~0.97 the own prefix equals the true one in ~91 % of code checks; the next bet is D-09. |
| D-06 | MoonShots | PTR [odd place]: the 125B's own attention as a copy pointer. On agent turns and code, at DFlash2's first misses: is the token after the context position that the QSA heads attend to most (at the last accepted row) the target token? It would catch the near-verbatim copies that suffix matching (Chat1000 REPORT 14, closed) misses, at ~0 cost (the attention is already computed) | PASS: >= 25 % of the first misses on agent text recovered. KILL: < 8 % | [taken: MoonShots 15:50] Run as pre-registered here before the lock: capture on Chat1000 agent125 turns edit00 / bash17 / text23 / edit09 / write14 (the 125B, 0.25 setting, teacher-forced; DENSE causal softmax of the 12 QSA layers recomputed in a hook: the real layers intersect it with the indexer's 2,048-token pick); DFlash2's first misses -> the native MTP chain's first misses (passes simulated as agent_analyze.py) until D-02; pointer = top attention position (max over 24 heads, PRIMARY = max over all 12 layers; sinks 0-3 and the row's last two positions excluded) at the last computed row before drafting; readings A (token after it) and B (the attended token, induction-style), ungated and gated on alignment with the known bonus token; baseline = prompt lookup at the same misses; also block tau of PTR alone and best-of(MTP, PTR), and your >= 0.9 proxy (same row). **KILL 15:56 (MoonShots, scripts/ptr_capture.py + ptr_analyze.py, data/ptr/analyze.txt; capture 295 s, peak 26.3 GB)**: of 199 MTP first misses on the 5 agent turns the pointer recovers 4.5 % (reading B, induction-style) / 2.0 % (A, token after the attended position); gated on alignment 4.0 / 1.0 %; prompt lookup recovers 10.6 % at the same misses and PTR adds only 1.5 % beyond it. Block tau: MTP 3.84, PTR alone 1.71, oracle best-of(MTP, PTR) 4.01 (+4 %, a selector bound). Best single layer post hoc: L43 reading B 8.5 %. Even your same-row proxy (not available when drafting; p >= 0.9, previous token in context, 498 positions) reaches only 24.7 % (B) / 17.7 % (A). Reading: at the draft point the 125B's attention does not sit on a copy source; where copying happens, lookup already finds it (as Chat1000 REPORT 14). Caveat: dense softmax, not the indexer's 2,048-token pick |
| D-08 | NewIdeas | CALC [new to us; NewIdeas Wed 16:2x, the next bet after D-03's KILL]: a symbolic co-drafter. Arithmetic results in CoT math are computed by the 125B and are hard for a small neural drafter, but trivial for a calculator: when the text ends in '<expression> =', propose the computed result's tokens exactly. Generalizes to other deterministic spans (bracket closure, dates, schema keys). Screen (CPU, existing data: Leap's dfl2d_math, 60 GSM8K CoT answers of the 125B): the share X of completion tokens that a calculator determines and that equal the 125B's own tokens | PASS: X >= 5 % (then one replay saving DFlash2's first-miss positions measures how many of its misses these are). KILL: X < 2 % (cannot move math's 0.871 per token by +1 point) | **KILL (MTP Head's walk, ~/TrainLab/data/d08_calc_walk.txt, TrainLab cdff830; logged by NewIdeas 18:48)**: my script took the LAST `gen` tokens as the answer, but `gen` is the prompt length (ids[gen - 3:gen] = '</think>\n\n'), so the answer is ids[gen - off:] (my bug, found by MTP Head; checked). Over the real answers X = 2.28 % (326 of 14,320 completion tokens), and DFlash2 L1-20 already drafts 81.6 % of them (266 hits; 54 first misses = 16.6 %, follow-up line >= 40 %; they are 3.0 % of all 1,822 first misses, median depth 10): under ~1 % more kept tokens at best. The walk reproduces D-02b's 7.00 kept per check exactly. Superseded first read, wrong region: BETWEEN (NewIdeas Wed 16:2x, scripts/d08_calc.py, 1.2 s CPU): X = 3.11 % of 5,764 completion tokens (53 results, 3.4 tokens each) match a calculator exactly; 13 detected expressions where the 125B wrote something else (mostly '$' currency formatting my parser misses, so X is ~3.1-3.6 %). Worth: if DFlash2 misses these at 50-80 %, math's per-token rate goes 0.871 -> ~0.887-0.897, kept per check 7.00 -> ~7.6-8.0 (+9-14 %), exact, for a CPU plug-in. Follow-up at zero GPU cost: MTP Head's planned replay saves DFlash2's first-miss positions on gsm; count those on calculator tokens |
| D-07 | NewIdeas | SELF [seed 5; pricing from Leap's measured costs] | KILL if every self-speculation variant needs > 3x the ~7 ms drafting budget | **KILL by arithmetic (Wed 15:4x)**: an AR step of the near-copy reads the whole dense part (3.44 GB, TABLE_350 A5: ~11.5 ms at M = 1), so 15 steps cost ~170 ms. An 8-layer early exit reads ~0.9 GB a step (~3 ms; ~44 ms a block) and is far weaker than the near-copy. One Jacobi pass of the near-copy at M = 16 costs the check minus 90 % of its expert time (~150 - 53 = ~95 ms) |
| D-09 | TrainLab (MTP HEAD) | DSCALE [the next D98 bet, NewIdeas Wed 19:3x; data scaling is published for drafters (EAGLE-3), untested here; it decides B6 / the OPD seed]: is the far-depth gap a DATA problem? Both drafters are ~0.97 at depths 1-3 on code (D-05); the gap is depths 4-15 (DFlash2 code 0.98 -> 0.86, gsm 0.91 -> 0.68). L1-20 learned from only 22.6K tokens of the 125B's outputs, and Leap's lattice oracle (14.11 per 16-row pass, top-64 recall >= 0.91 at every depth) says the knowledge is close while a small picker (L1-21, +1 %) and affordable trees (L1-22) cannot pick it. Screen: retrain the deployed recipe (L1-20: same init, hyper-parameters, steps per token, partial masking) on 1/4 and 1/2 of its 99 training windows (by prompt; same hold-outs); score tokens per pass at block 16, 1 step, on the 24 held-out HumanEval samples and on dfl2d_math (60 GSM8K answers, D-02b's walk), with L1-20 (all the data) as the third point; fit tokens per pass against log2(data) per domain. Local, inside L1-20's envelope (peak 33.7 GB): start only with >= 40 GB free (not beside the Eval suite's 125B runs), stop if swap grows > 2 GB | PASS: >= +0.4 tokens per doubling on code or gsm (16x the data would give >= ~11 at block 16; then a priced proposal for Kamen: an on-policy data run, the 125B's own outputs or teacher-forced logits on local text, plus a longer local training, or the free cloud tier of B6). KILL: <= +0.15 per doubling on both (more data will not close the gap; the lever is the drafter's architecture and capacity, the MTP-head design question) | **PRELIM BETWEEN (MTP Head, ran Wed 22:11-22:34; ~/TrainLab/HANDOVER.md + data/d02b_walk_d09{q,h}.txt, checked by NewIdeas 02:1x; its formal read-out was due at ~01:43 and was cut off by Kamen's full stop)**: kept per check at block 16: 1/4 of the data (0.257 of the anchors) code 9.37 / math 6.65; 1/2 (0.506) 9.46 / 6.82; full (L1-20) 9.57 / 7.00; least-squares slope per doubling code +0.10, math +0.18 (no PASS; math just over KILL's 0.15). NewIdeas' read: even at +0.18 per doubling, math 7 -> 12 would take ~28 doublings, so more data alone will not close the gap (as Leap's DS / DS2 found); the next D98 bet is the drafter's architecture and capacity (MTP Head's program). **Formal read-out 02:15 (MTP Head) confirms BETWEEN** with the pre-registered fit (code +0.10, math +0.18 per doubling; rounding < 0.01). Near depths 1-5 flat; far depths 11-15 +0.02 per token per doubling (code 0.844 -> 0.889, math 0.771 -> 0.813): only the far gap answers to data, and the flat near depths cap the total, so data alone cannot reach ~98 %. |
| D-10 | MTP Head (account-2) | SOFTKD [published: DistillSpec (KD for drafters), untested here; NewIdeas Thu 15:4x; run if D-02a shows confident misses dominate]: retrain L1-20's recipe with the 125B's top-32 probabilities as soft targets (KL or TVD) instead of hard tokens; same windows, steps and holds; block 16, 1 step, on the 24 HumanEval + dfl2d_math. Needs top-32 p on the training windows (add to D-02a's capture). D-02a goes first: its share of first misses where the 125B is >= 0.99 sure vs near-ties picks the route (knowledge problem vs the exact ceiling) | PASS: >= +0.3 kept per check on code or math. KILL: <= +0.1 on both | **WITHDRAWN (MTP Head Thu 15:5x; NewIdeas' error)**: L1-20's trainer (Leap's dfl2c_train.py; MTP Head's copy sticky/d09_train.py, line 123) already uses soft cross-entropy to the 125B's stored top-32 (forward KL), so D-09's slopes are a soft-target recipe's. I proposed it without reading the trainer. Left: the divergence form (TVD / reverse KL / JSD, DistillSpec), one screen only if D-02a shows confident misses dominate (MTP Head expects <= +0.1 kept). D-02a GO 15:5x (MTP Head's plan): code needs no capture (~/Leap/data/l118_tgt has the 125B's top-32 for all 24 dfl2d_code texts); one capture (sticky/d02a_extract.py, ~145K prefill tokens, ~10 min, one quiet lock) for math p + IFEval (30) / multi (24) / agent (12 turns) features + p; then the D-02b walk and the first-miss table: share with p1 >= 0.99 vs near-ties (p1 - p2 <= 0.1) per domain and depth band. |
| D-11 | MTP Head (account-2) | DOMAINMIX [NewIdeas 17:56; from D-02a; L1-20 = 201 steps from incoai's DFlash2 on 22.5K anchors of HumanEval + GSM8K windows only (TrainLab REPORT 1078, HANDOVER 106)]: on instructions / multi-turn chat / agent turns L1-20 keeps 2.45 / 2.06 / 3.01 per check (depth-1 rate 0.63 / 0.51 / 0.62) vs the near-copy's 6.6 / 8.0 / 6.8; code 9.57, math 7.00. If the gap is domain coverage, a tune with those domains in the data closes much of it, which more code / math data could not (D-09). Screen, after MTP Head's walk of incoai's original weights on the same sets (if the original keeps +1.0 more there, L1-20's tune cost it that; if not, the 125B's own text in those domains is the lever): L1-20's recipe on its he + gsm windows plus the 125B's own answers on instruction, chat and agent prompts outside D-02a's 30 / 24 / 12 (EvalSuite's IFEval and agent rows, more of Chat1000; dedup by prompt), teacher top-32 captured as in D-02a (~7 min per ~146K tokens); walk all five sets | PASS: kept per check +1.0 or better on at least two of IFEval / multi / agent AND code and math within -0.1 of L1-20 -> the weights to Leap for the 45 GB engine (the drafter changes speed only), then more such data. KILL: under +0.3 on all three | **KILLED by prior evidence 18:58 (NewIdeas, on MTP Head's 18:41 note; my error: I proposed it without reading Leap's earlier tunes)**: incoai's ORIGINAL DFlash2 keeps 2.52 / 2.11 / 3.02 on IFEval / multi / agent (L1-20 2.45 / 2.06 / 3.01), so L1-20's code / math tune cost <= 0.07, and broad training did not help those domains; Leap's L3 / L4 agent-text tunes (up to 268K tokens) moved agent tau8 only 2.86 -> 2.91 and 2.85 -> 2.96 (~/Leap/data/dfl2_result_l3_*_agent.txt, l4_*_agent.txt). Not a data-coverage gap: a drafter capacity / design question for MTP Head's design study (the miss table decides how much of it is near-ties). **Narrowed 20:48 by D-14 B: for multilingual text it may be language coverage after all (Bulgarian / Arabic sure-token hits 0.37 / 0.45 vs Spanish 0.94); this KILL holds for agent / IFEval, D-15a tests the languages** |
| D-12 | MTP Head (account-2) | MTPCHAT [measurement for the design study; NewIdeas 19:00; from Leap's L2b on agent (~/Leap/HANDOVER.md:145, data/dfl2_result_l3_ft_agent_hyb.txt)]: on agent turns the native MTP chain (K=7, the 125B's own MTP head, ~/Chat1000/data/agent125 'mtp' arrays) keeps tau 3.54-3.70 vs DFlash2's 2.86-2.96 (Leap's 'x0.82' = DFlash2 / MTP tau); a per-anchor hybrid lost to MTP alone (cross-check 3.37-3.44, oracle +9 %): 'Agentic = MTP alone'. Untested on IFEval / multi (only agent125 stores mtp arrays): capture the MTP head's K=7 predictions on D-02a's 30 / 24 texts (from the 125B's final hidden states, if D-02a's features hold them; else one short capture) and walk both drafters | PASS: MTP tau >= DFlash2's + 0.3 on IFEval or multi -> price a per-domain / online drafter choice for the 45 GB engine (check cost by row count: 8 rows vs 16). KILL: MTP <= DFlash2 + 0.1 on both | **KILLED by desk check 19:15 (MTP Head's arithmetic, accepted by NewIdeas; no K=7 run)**: D-05's MTP chain on IFEval / multi has depths 1-3 at 0.637 / 0.550 / 0.540 and 0.464 / 0.464 / 0.504 (tau3 2.18 / 1.79), the same as DFlash2's 0.629 / 0.537 / 0.538 and 0.505 / 0.498 / 0.519; even at 0.6 per depth over depths 4-7, MTP K=7 is ~2.4 on IFEval, under DFlash2's 2.45. Agent stays as Leap found it (MTP alone 3.54 vs 2.9). Caveat: D-05 had 8 prompts per domain, v3 plan mode |
| D-13 | MTP Head (account-2) | DIVERGE [pre-registered by MTP Head 19:1x, REPORT 'D-13'; the TVD / reverse-KL screen of D-02a's route: code's deep misses are knowledge misses; Leap's R-05 tested only an acceptance-weighted depth loss (W / U 0.970, KILL)]: Leap's R-05 rapid protocol (21 steps from L1-20), three arms: U = forward KL to the 125B's top-32 (control, today's recipe), S = forward KL to sharpened targets softmax(lp / 0.5), V = TVD on the top-32 plus a rest bucket. Walk at block 16 on code + math, ratios with a 90 % bootstrap CI; one 15-min lock | PASS: S / U or V / U >= 1.03 on code or math with CI low > 1.00 -> the full 2-epoch run. KILL: both <= 1.01 on both | **KILL Thu 19:33 (MTP Head, REPORT 'D-13 RESULT', 9c21c4c, data/d13_ratio.txt; pasted by NewIdeas 16:16)**: kept per check U / S / V: code 9.40 / 9.39 / 9.32, math 6.71 / 6.67 / 6.68; ratios to U: S 0.999 (code), 0.994 (math); V 0.991, 0.996; every 90 % CI upper bound <= 1.01. The target's form (sharpened forward KL, TVD) doesn't fix the mid-confidence misses |
| D-14 | MTP Head (account-2) | PROBES [design study, CPU, stored data; MTP Head 20:47, REPORT 'D-14 probes', data/d14_probe.txt] | measurement | **DONE 20:47**: A. depth-1 hit rate of DFlash2 L1-20 on tokens the 125B is >= 0.99 sure of: code 0.996, math 0.993, IFEval 0.899, agent 0.795, multi 0.622 (mid tokens 0.791 / 0.744 / 0.565 / 0.529 / 0.452): outside code and math the drafter misses even tokens the 125B finds obvious, a knowledge / coverage gap. B. multi by language, sure-token depth-1 hits: es 0.936, fr 0.824, de 0.734, ru 0.697, ja 0.683, zh 0.672, ar 0.454, bg 0.365; kept per check es 2.85, fr 2.74, de 2.39, ru 2.11, ja 2.08, zh 2.00, bg 1.60, ar 1.59 (the 125B itself is sure on 29-47 % of tokens in every language), which fits the training-data coverage of incoai's drafter. C. agent: the true 4-gram at a first miss is inside the 2,048-token window 14 %, only beyond it 13 %, nowhere earlier 73 %: a longer window reaches <= ~13 % |
| D-15a | MTP Head (account-2) | LANGPROBE [pre-registered by MTP Head 20:47, REPORT 'D-15a'; from D-14 B]: can a little of the 125B's own text in a language teach the drafter that language? D-02a's 24 multilingual answers, 3 per language: two per language (16) as training windows, the third held out (bg2 de5 fr8 es11 ru14 zh17 ja20 ar23). Rapid tune from L1-20 on L1-20's recipe (forward KL to the top-32, block 32, pmask 0.5, lr 2e-5), 2 epochs (~38 steps). Score: the D-02b walk (block 16, 1 step) on the 8 held-out answers vs L1-20's walk of the same 8, kept per check and the sure-token depth-1 hit rate per language; code + math walked for forgetting against D-13's U (9.40 / 6.71). One quiet lock (15 min) | PASS: held-out kept +0.3 or more AND sure-token depth-1 hits +0.10 or more, code / math within -3 % of U -> D-15 (new multilingual prompts, the 125B's answers, a proper tune; the same test on IFEval). KILL: held-out kept +0.1 or less. Else BETWEEN. MTP Head's prediction: +0.3-0.6 (PASS 45 %) | **KILL 21:00 (MTP Head, REPORT 'D-15a RESULT')**: 8 held-out kept 2.07 -> 2.06; sure-token depth-1 hits 0.655 -> 0.643; bg / ar unmoved; code -1.3 %. The multilingual gap isn't cheap coverage (the native MTP head is as weak there). Literature (MTP Head's sweep, design study 1): multilingual needs a pretrained drafter body (Osprey 2609.09338): a download, so Kamen's call; parked |
| D-16 | MTP Head (account-2) | HCFULL [MTP Head's design study 1 (REPORT 'D98 DESIGN STUDY 1', 01e1eb5); NewIdeas GO 21:1x]: the 125B's native MTP head reads the full pre-mixer widened stream (4 x 2,560 = 10,240 per position), but DFlash2 gets only the plain mean of the 4 streams at 5 taps. Add the final widened stream as an extra DFlash2 input through a zero-initialised projection (L1-20's function at the start). Capture ~1.5 GB (l120 + code / math sets), Leap's rapid protocol, U vs H; two locks. Prior art: richer target taps fix agent text (DLite 2610.23787: 12-14 taps, agentic tau ~1.6-1.8x DFlash's) | PASS: +3 % or better kept on code or math, or +0.3 kept on agent -> the full run. KILL: +1 % or less on all | **KILL (MTP Head, REPORT 'D-16 RESULT', f25aac7)**: paired per-step loss vs U on the same batches, steps 11-21: Hc (hc-lr 5e-6, fan-in-matched) +0.144 (CI +0.111 to +0.175), worse on every step from step 4; H (1e-3) diverged; Hb (1.5e-6) unfinished (the Mac slept), its delta shrinks with the rate but never turns negative. No walks. Caveat: 21 steps under-read a new input; no 2-epoch run, since D-17 / D-19 trace the native head's chat / agent lead to its chain (a switch captures it). Next from MTP Head: D-21 REFINE (check Leap's L1-19 / L1-20 2-step results first: block 16 9.54 -> 9.58, block 32 10.23 -> 10.51) |
| D-17 | MTP Head (account-2) | MTPDEEP [MTP Head's design study 1; NewIdeas GO 21:1x; not D-12's average question: a per-round router's oracle]: the native MTP chain at k = 7 in exact mode on D-02a's IFEval / multi / agent prompts, plus agent125's stored chain; per round, the MTP chain's and DFlash2's kept counts, so the per-round oracle max and a confidence router can be scored. One lock. Prior art: unrolled MTP training keeps chained heads accurate (AngelSpec 2607.25852: cumulative depth-3 0.27 -> 0.52; FastMTP on Qwen3-Next 0.48 -> 0.62 at depth 3; Microsoft 2610.00888 +12-15 points at depth 3 on chat); per-round MTP / block routing (WhiFlash 2606.07710): agentic 4.12 -> 4.90, chat 3.40 -> 5.47 | PASS: the per-round oracle max(MTP, DFlash2) >= +0.5 kept over DFlash2 on chat or agent -> design the router + unrolled MTP training (D-18, priced). KILL: the oracle < +0.2 on all three | **PASS 02:33 (MTP Head, REPORT 'D-17 RESULT', a349caa)**: tokens per pass, DFlash2 -> oracle K15 (the better chain each pass): code 10.44 -> 11.62 (1.11x, CI 1.05-1.17), gsm 6.60 -> 7.45, ifeval 2.24 -> 2.66, multi 2.22 -> 3.18 (+0.96), agent 3.28 -> 4.69 (+1.41). The native MTP chain ALONE beats DFlash2 on chat / agent: 1.32x / 1.30x (CIs above 1.18), same 16-row check (so D-12's desk KILL was wrong there; the measurement wins). The cheap confidence router (the 125B's p on the bonus token) gets none of the code / math oracle. 45 GB estimate (not measured): a text-type switch (MTP K7 on chat / agent, DFlash2 on code / math) ~+12-22 % on chat / agent, 0 on code; the code oracle needs a learned selector or a 2-branch tree. D-18 MTP1 desk KILL (code -2.5 %). D-16b H arm diverged (hc-lr ~200x too high); retry at 5e-6 / 1.5e-6 queued |
| D-19 | MTP Head (account-2), then Leap's live loop | SWITCH [exact: drafts only; 45 GB route; NewIdeas 02:35 from D-17]: how much of D-17's oracle can a real switch get? CPU replay on D-17's per-pass data (both chains' kept per pass, in order per prompt) of online rules that see only the past: (a) a per-request text type from the prompt (code / math vs the rest), (b) a bandit over the last W passes (each pass also scores the unchosen chain against the verified tokens, so its kept is known up to the chosen chain's kept + 1), (c) (a) then (b). Drafting cost per pass priced in (DFlash2 ~13 ms; the MTP K7 chain measured) | PASS: a rule gets >= +0.4 kept per pass on chat or agent (>= ~40 % of the oracle) with code / math >= -0.1 -> the live-loop A/B at C 224 (Leap; tok/s, every token == REF2). KILL: under +0.15 on both chat and agent | **PASS by the text-type rule only (MTP Head, REPORT 'D-19 RESULT', f312ad6)**: type (MTP K15 on ifeval / multi / agent, DFlash2 on code / gsm): multi +0.70, agent +0.97, ifeval +0.19, code / gsm 0. Past-only bandits (W 2 / 4 / 8, partial or full feedback, with or without the type start): chat / agent +0.48 to +0.85 but code -0.10 to -0.60 and gsm -0.27 to -0.56: none meets the code / math floor (they keep trying MTP on code, where an MTP pass costs ~2.7 tokens). So the engine needs the text type up front (an agent / tool turn, the prompt's language), not a learned switch. MTP K7 halves the 15 sequential MTP steps and keeps 94 % of the multi gain but 66 % of agent's |
| D-20 | Leap's live loop + MTP Head | SWITCH45 [exact: drafts only; 45 GB route; NewIdeas 02:49 from D-19]: the text-type switch in the live loop at C 224: DFlash2 on code / math turns, the native MTP chain (K7 and K15) on chat / instruction / agent turns, with the type known up front (a tool / agent turn, or a non-code, non-math prompt); measure tok/s per type vs DFlash2 everywhere, every token == REF2. Runs after Leap's STEP 4c fix (the loop is exact only under ~512 keys until then; the drafter choice doesn't touch exactness) | PASS: chat / agent mean tok/s +10 % or better (drafting cost included), code / math unchanged -> the switch into the 45 GB loop's defaults (exact). KILL: under +4 % on chat / agent | **Pre-check PASS 16:18 (NewIdeas, CPU, scripts/type_rule.py, data/type_rule.txt)**: a keyword rule on the prompt (tool schema -> agent; code markers -> code; \boxed / LaTeX / solve / arithmetic -> math; else chat) picks D-19's drafter for 98.5 % of the Eval suite's 263 decode prompts outside MGSM: HumanEval 60 / 60, MATH 50 / 50, agent 29 / 29, multi 24 / 24, IFEval 96 / 100 (1 -> code, 3 -> math: the cheap direction, they lose only IFEval's +0.19). No code or math prompt goes to the MTP chain (where a pass costs ~2.7 tokens). Open: MGSM (non-English math, 80 / 80 labelled math -> DFlash2) has no D-19 data, and D-14 found DFlash2 weak on non-English text: add MGSM to D-22 / D-20's walks to pick its drafter. Caveat: the rule is per request (the last user message + tools); a conversation can change type between turns. **PASS 02:xx (MTP Head, REPORT 'D-20 SWITCH45 RESULT', 56625a7; Leap's REF2b exact live loop, C 160 (C 224 didn't fit), 256 tokens, trained head b3, arms interleaved per prompt; tokens identical across all arms on all 11 prompts, == REF2 rows where they exist; quiet lock)**: mean tok/s dfl / mtp7 / mtp15: chat (5) 10.9 / 14.9 (+36 %) / 11.8; agent (2) 10.5 / 15.9 (+51 %) / 16.1; code (2) 25.1 / 20.4 (-19 %); math (2) 16.8 / 18.1 (+8 %). The switch (DFlash2 on code, MTP K7 on the rest) beats MTP everywhere: mtp7 keeps about as much per check at an 8-row check that's ~15-25 % cheaper; on code DFlash2's 16-row check pays (15.4 / 16 kept on HumanEval/68). Caveats: C 160, 2-5 prompts per type, one run per arm. **NewIdeas 02:02: the rule becomes DFlash2 on code only, MTP K7 on math / chat / agent; next a confirmation with more prompts at C 224 when memory allows, then into the loop's defaults (exact); in SUMMARY now, to the orchestrator after 07:45** |
| D-22 | MTP Head (account-2) | MTPTTT [exact: drafts only; pre-registered by MTP Head, REPORT 'D-22', c854af8; NewIdeas GO 09:54 with the chat / agent walks added]: train the native MTP head the way it chains (EAGLE-3's training-time test inside its attention: depth j of anchor s attends to the true-pair history u <= s plus its own chain entries, as the MTP cache does at inference). Fixes the 10-22 kill (mtp125n_d4: depth j trained over a fresh cache, lost depths 2-4). Data: L1-20's 99 training windows + D-16a's widened stream (captured); no new 125B generation. Held out: the 24 HumanEval and 60 GSM8K walk samples. Gate first: the forward reproduces MTPLX's real chain position by position on 16 prompts (>= 0.98 at depth 1, >= 0.95 over depths 1-7). D-20 cost note: MTP k = 3 is 4.5 ms per check with the layer resident (+1.3 ms per extra step): K7 ~9.7 ms, K15 ~20 ms, DFlash2 12.5 | PASS: code K15 >= 1.15x the untrained head (7.78) with gsm >= 1.00x, OR agent / chat (multi) +0.3 kept or more over the untrained chain with code / gsm >= 1.00x (the switch's MTP domains); BIG if code reaches DFlash2's level (>= 10.4). KILL: < 1.05x on code and gsm AND < +0.1 on agent / chat. Prediction (code): 1.10-1.20x; PASS 40 %, KILL 30 % | **Interim lead 17:13 (MTP Head, by its own forward; not yet the real runtime)**: after 76 of 198 steps (0.77 epoch, 12 min): held-out code (24) K15 7.54 -> 9.21 tokens per pass (1.22x), K7 6.17 -> 6.54; gsm (60) flat 6.54 -> 6.58. Every 20-step block of online held-out loss lower than the untrained head (-0.10 to -0.17, CIs < 0); KL on the 10 validation windows drops at every depth (depth 4 0.42 -> 0.22, depth 8 1.30 -> 0.38). Held-out clean (no 32-gram shared with training). Gate: the forward reproduces MTPLX's chain (depth 1 0.993, depths 1-7 0.974; deep depths 0.86-0.93). Next: finish the 198 steps, then the real-runtime read-out (installed in MTPLX, D-17's greedy decode, answers token-equal), with the chat / agent / multi walks. **REAL-RUNTIME PASS 20:56 (MTP Head, REPORT 'D-22 REAL-RUNTIME RESULT', 810470b; BIG sent by MTP Head)**: MTPLX with the step-76 head, D-05's 8 held-out code prompts, answers token-identical to the stock run: K15 7.78 -> 9.83 tokens per pass (1.263x, CI 1.185-1.366), K7 6.21 -> 6.87 (1.105x, CI 1.070-1.142); per-depth rate depth 7 0.87 -> 0.94, depth 10 0.79 -> 0.94, depth 15 0.70 -> 0.93 (the decay is gone). gsm: 1 of 8 answers so far (K15 1.125x). Not yet DFlash2's level (9.83 vs 10.44 on the same answers). Next: the remaining gsm decodes, run b2 (1 epoch, cosine lr; 2 epochs over-fit, gsm -3 %), chat windows (IFEval 70 + MGSM 50) then a chat-inclusive run; D-20 tests the trained head on chat / agent too (D20_HEAD=base76). NewIdeas 20:55: keep D-20's / D-02a's measurement prompts disjoint from the IFEval / MGSM training windows. **Complete (MTP Head ~21:0x): gsm (8) K15 6.07 -> 6.36 = 1.048x (CI 1.001-1.109), K7 1.027x; all 16 answers token-identical.** Next (shared runs): held-out chat windows (IFEval 70 + MGSM 50), run b3 (code / math + chat, depths 1-15, 1 epoch, cosine), read-outs on code / gsm / ifeval / multi / mgsm **Run b3 (MTP Head ~22:3x, REPORT 'D-22 run b3', 7500583; depths 1-15, cosine 1 epoch, 73 IFEval windows of the 125B's answers + L1-20's 99; shared GPU, 172 steps; its forward, held-out)**: code (24) K15 10.12 = 1.341x (CI 1.275-1.416), K7 1.085x; gsm (60) K15 7.57 = 1.157x (CI 1.137-1.178), K7 1.067x (step 76: 1.221x / 1.006x). The real runtime read step 76 higher than its forward did, so b3 may reach DFlash2's 10.44 on code; b3's real-runtime decode on D-05's 32 prompts (code / gsm / ifeval / multi) runs. The shipped-line A/B (H50 stack) waits for >= 50 GB available. NewIdeas 23:19: asked whether b3 touches only the MTP layer's dense weights; if so, a cross-line exact drop-in (shipped 45 GB, 24 GB, 12 GB). **b3 REAL-RUNTIME PASS 23:4x (MTP Head, REPORT 'D-22 b3 REAL-RUNTIME RESULT'; MTPLX with b3, D-05's 32 held-out prompts, all answers token-identical)**: code K15 7.78 -> 10.98 (1.411x, CI 1.287-1.565), K7 1.124x, ~0.97 per depth at every depth 1-15, above DFlash2 L1-20's 10.44 on the same 8 answers (same 16-row check); gsm K15 6.07 -> 6.86 (1.131x, CI 1.093-1.180), above DFlash2's 6.60, K7 1.049x; ifeval / multi +2-3 %, n.s. (73 IFEval windows aren't enough for chat). MTP Head's BIG line (code >= 10.4) met; it sends the orchestrator an update. 45 GB reading (estimate): on the 16-row route the K15 chain's ~20 ms of drafting vs DFlash2's 12.5 ms makes code about even at ~150 ms checks; on the shipped MTP line (width <= 7) the K7 gain applies: +12 % code, +5 % gsm tokens per pass (turn53 A/B, needs >= 50 GB). NewIdeas 23:40: D-20 gains an arm 'MTP b3 everywhere'; R-60 uses b3's weights |
| R-60 | BigMoE (24 GB), Breakthrough (12 GB), MTP Head (shipped 45 GB A/B) | MTPDROP [exact: drafts only; FAST INTEGRATION; NewIdeas 23:22 from D-22]: the trained MTP head into every MTP-drafted line. b3 changes only the MTP layer's dense parts (fc_hidden, fc_embedding, attention q / k / v / o, the router moe.gate, the shared expert; 69.1M weights); the 512 routed experts, norms, HC mixers, lm_head / embeddings and the trunk are untouched; the trained router picks different experts from the same 512. Ship in bf16 (~138 MB, +~0.1 GB) or 8-bit g64 (stock 4-bit keeps only 12-33 % of the change). Production drop-in: keep MTPLX's MoE kernels, replace only those weights | per line (shared GPU, rough): tokens identical to the line's default AND kept per check +3 % or better -> one quiet confirmation -> default on NewIdeas' word; KILL under +1 %. Memory: 24 GB peak <= 24, 12 GB <= 11.85 | |
| R-61N | MTP Head (CPU replay) | CHAT-NT1 [NON-exact, Kamen's call; orchestrator's night brief Sun 1 Nov; NT-1 ran only on 24 HumanEval samples at block 32 (56 % of code's first misses at >= 99 % confidence)]: near-tie acceptance on CHAT, our weakest text (2-3 kept per check), never measured. Re-run nt_accept.py + nt_analyze.py on D-02a's IFEval / multi / agent captures (the 125B's top-32): confidence bands of the first misses; forgiven share and tokens per pass at rho 0.5 / 0.25 / 0.1; the drafter top-3 oracle (any of its top-3 inside the accept set). Published: Medusa typical acceptance, Judge Decoding, AutoJudge, FLy | PASS: tokens per pass on chat +30 % or better at rho 0.25 -> a set-trained drafter row (set loss + KL inside the set) and the quality test for Kamen. KILL: under +10 % at rho 0.1 | |
| R-62P | Fit12 (CPU, cap45 replay) | PRUNE-COLD [exact under greedy verify; brief Sun 1 Nov; prior art HiSpec, FASER, EcoSpec (not cache-state aware); our R-54 oracle at 12 GB -9.7 %, S45c truncation +12-17 % live]: in greedy mode dropping rows j.. at any layer keeps the output exact, since the next check recomputes them (batch-invariant kernels). Replay BigMoE's cap45 traces at 45 GB residency: (a) oracle SSD bytes if a check paid only for the kept prefix + 1; (b) 'row j needs a non-resident expert at layer L -> end the check at j-1 and issue that read async': SSD ms saved vs tokens lost, on top of t 0.3 truncation | PASS: (b) saves >= 15 % of SSD ms per kept token beyond t 0.3 -> a live screen (Leap's loop). KILL: (a)'s oracle under 10 % beyond t 0.3 | **01:4x (Fit12, data/r62p/r62p.txt; cap45 at C 224; the replay reproduces r57b's baseline exactly, 141,883 misses at C 160; t 0.3 modelled from Leap's measured surplus)**: demand misses per token B16 56.9, T03 45.0; ORACLE (kept prefix + 1) 39.2 = -12.9 % vs T03 (not killed). The CUT rule meets the letter (-17.6 %) but cuts almost every pass to row 0 (passes per token +645 %, modelled cycle per token +396 %): no live screen. Side numbers: AR-like passes 37 misses per token vs T03's 45, so 16-row checks cost only ~21 % more SSD per token than AR. **NewIdeas 01:14: PARKED: the oracle's 13 % needs a rule that knows which rows will be rejected (drafter confidence x the row's marginal non-resident reads, EcoSpec-style), which needs a capture with per-row drafter probabilities; open it when Leap's loop logs them** |
| R-63S | MoonShots (CPU, after account-1's 04:00 reset) | STREAMS [aggregate tok/s across parallel parts: whether that counts is Kamen's call; brief Sun 1 Nov; published: batching, Skeleton-of-Thought]: overlay the per-layer expert sets of 3 code prompts check by check (cap45 or Leap's live captures) | PASS: union <= ~190 per layer (chance level) vs 3 x 73 = 219, AND for 8 parts of one chat answer <= 11 new experts per extra stream -> priced for Kamen as an aggregate-throughput mode. KILL: unions at 3 x 73 (no sharing) || **BETWEEN 04:02 (MoonShots, CPU 9 s; scripts/streams45.py, data/streams45.txt)**: A. 3 independent prompts overlaid check by check (BigMoE's cap45 live 16-row checks, C 160; one stream 70.7 experts per layer): union / sum 0.755 over all 20 three-prompt combinations (= 165 of 219 in 73-units; with both code prompts 0.76-0.78), exactly the chance level (0.756: popularity overlap only) -> PASS on A. B. 8 parts of one answer (expA true tokens, 16-token windows at the same offset): new experts per extra part chat ~21 (the script's 13.0 is low because the chat trace is shorter than 800 tokens: parts 6-8 partly empty; over the 5 full parts (152.5 - 68.7) / 4 = 21), code 23.5, math 23.6, bulgarian 19.0 -> FAIL on B (line <= 11), though parts of one answer share far more than independent prompts (~45 new per extra stream at chance). Pricing for Kamen's call: 3 independent streams in one pass read expert bytes per token x0.755 and the dense part once (13.2 GB per check -> ~25.4 GB for 3 checks' tokens: about -36 % bytes per token, aggregate tok/s, not one chat's speed); 8 parallel parts of one answer: union ~223 per layer vs 8 x 69 = 552 (x0.40), but only for decomposable answers (L1-32p: too rare to carry 350) Correction (MoonShots Sun 06:1x, found while building M-10): the part A loader kept each check's commit replay (layer 1, kept rows) in place of layer 1's 16-row set. A's verdict used layers 0, 4, ..., 44 and is unchanged; the printed one-stream mean is 71.7 distinct experts per layer, not 70.7. |
| M-10 | MoonShots (CPU, cap45 replay) | SIBLINGS [decision input for RANKED #10, exact trees; Leap 05:3x routed A (REF2c bundle) vs B (exact trees) to NewIdeas. Published: EcoSpec / EVICT / S2-MoE price tree nodes by their marginal expert cost; no per-layer measurement of sibling expert sharing found. Pre-registered Sun 06:12]: do sibling rows (same prefix, different token) share experts, so tree branches are cheap in bytes at 45 GB? BigMoE's cap45 live checks hold exact sibling pairs: the first rejected draft row of check k (row a_k + 1; a_k from the commit replay's row count) and check k+1's row 0 (the target's token at the same position, same prefix). Per layer: sibling overlap vs consecutive accepted rows vs chance (popularity draws); the new experts a sibling row adds to a 16-row check's union vs a late chain row's (row 15 given rows 0-14); the target branch's first m rows (m 1 / 4 / 8) given the drafted check's union | PASS (branches cheap in bytes): a sibling row adds <= 0.5x a late chain row's new experts (mean over layers, both directions) -> Leap / NewIdeas: exact trees cost mostly compute and the exactness machinery, not bytes. KILL: >= 0.85x (a sibling costs about a chain row, so trees pay bytes like longer chains, which S45c says hurt at 45 GB). BETWEEN otherwise | **BETWEEN by the letter, KILL in substance (MoonShots 06:1x, scripts/siblings45.py, data/siblings45.txt; 176 live checks, 170 exact sibling pairs, CPU 16 s)**: a sibling row adds 2.35 (target token into the drafted check) / 2.55 (drafted token into the next check) new experts per layer vs 2.79 for a late chain row (row 15 given rows 0-14) = 0.84x / 0.91x (KILL line 0.85 on both). Siblings share 3.88 of 10 experts (layers 0-7 / 8-23 / 24-47: 3.19 / 4.04 / 4.00) vs 3.38 for consecutive accepted rows and 0.81 by chance. Their MoE inputs don't converge with depth (cosine 0.61 / 0.59 / 0.60 vs 0.62 / 0.64 / 0.70 for consecutive rows). A whole second branch of m rows adds 8.3 (m 4) / 15.8 (m 8) / 30.7 (m 16) new experts per layer to the drafted check's union. So exact trees get no expert-byte discount at 45 GB: each sibling costs about a chain row, and a sibling pays only where its acceptance beats the truncated chain row it displaces (t 0.3). Input for A vs B: B's gain is acceptance only, priced at chain-row bytes plus the exactness machinery |
| M-11 | MoonShots (CPU) | BIASCORR [published: DFQ bias correction (Nagel 2019), KBVQ-MoE (ICLR 2026); a fast-integration screen, not an item-4 bet. Pre-registered Sun 06:39]: REF2-T's MSB-3 tier already removes the per-group mean of its weight error (b3 = bf16(b + s mean(l))). What is left is the systematic error (W4 - W3) mean(x), carried by the MoE input's mean direction, which is strong here (rows at cosine ~0.6, M-10). Correction: gate / up + (W4 - Wt) mean(x) (the layer's calibration mean); down + (W4 - Wt) mean(h_e) (the expert's calibration tokens); ~7.7 KB per expert. Test on CPU: Fit12's x at layers 2 / 12 / 24 / 36 / 46, the real router, experts and shared expert; calibration on tokens 0-4095, scored out of sample on 4096-8191. MoE block error (L1-17) vs all-4-bit for 4433333333 (REF2-T), 4333333333, 4433332222 and 4433333300 (top-8, no renorm), each without and with the correction | PASS: the correction cuts REF2-T's block error by >= 25 % out of sample AND a cheaper corrected plan (4333333333, 4433332222 or top-8) lands at or under uncorrected REF2-T's error -> KLD screen on the harness (GPU, shared). KILL: < 10 % cut on REF2-T and no cheaper corrected plan within 1.2x of uncorrected REF2-T. BETWEEN otherwise | **BETWEEN (MoonShots 06:4x, scripts/biascorr16.py, data/biascorr16.txt; CPU 254 s)**: MoE inputs carry 25-41 % of their energy in the mean (layers 2 / 12 / 24 / 36 / 46: 40.8 / 28.9 / 27.8 / 24.7 / 39.5 %). The correction cuts REF2-T's block error 0.01210 -> 0.00978 (layer mean) / 0.00949 (expert mean, shrunk): -21.6 %, on every layer (-13 to -28 %). Cheaper plans stay over even corrected: 4333333333 1.25x uncorrected REF2-T, 4433332222 1.51x, top-8 no-renorm 2.13x. Alone it is a free quality gain for REF2-T (~7.7 KB per expert), e.g. room for HC8's x1.09. Follow-up M-11b: a low-rank correction along the input's top principal directions (QERA / LQER / MiLo, published) |
| M-11b | MoonShots (CPU) | LOWRANKCORR [published: LQER 2024, QERA 2024, MiLo 2025 (low-rank compensators for 3-bit MoEs); fast integration. Pre-registered Sun 06:47 after M-11]: M-11's mean correction is the rank-1 case. Correct a 3-bit tier use of expert e along the layer's top-r eigenvectors U_r of E[x x^T] (calibration tokens): gate / up + (W4 - W3) U_r U_r^T x. U_r^T x is computed once per token per layer and shared by all 10 experts; per expert (W4 - W3) U_r is 2.56 KB x r. Down: M-11's mean correction, or rank 8 on the expert's own h basis. Same data, split and block metric as M-11. Plans 4433333333 / 4333333333 / 3333333333 (no LSB planes at all) at r 16 / 32 / 64 | PASS: 3333333333 or 4333333333, corrected at r <= 64, at <= 1.00x uncorrected REF2-T's block error out of sample -> KLD screen on the harness (GPU, shared), then the strict gate. 3333333333 would drop every LSB plane: ~-7 % bytes per check and ~28 % more experts per GB of cache. KILL: no cheaper corrected plan <= 1.2x uncorrected REF2-T at r <= 64. BETWEEN otherwise | **BETWEEN (MoonShots 06:5x, scripts/lowrank16.py, data/lowrank16.txt; CPU 321 s)**: the input's top 1 / 4 / 16 / 32 / 64 directions hold 25-41 / 32-49 / 42-58 / 48-63 / 55-69 % of its energy, but the block error falls less. REF2-T goes 1.00 -> 0.78x (mean) / 0.74 (r16) / 0.72 (r32) / 0.70 (r64) / 0.67x (r32 + rank-8 down). Cheaper plans stay over: 4333333333 at best 1.08x (r32d8), 3333333333 at best 1.84x (ranks 0-1 at 4 bits stay load-bearing). Bytes don't move either: at 4333333333 the corrections (~133 KB per 3-bit expert use, ~8.6 MB per layer per check) equal the rank-1 LSB reads they save. A quality knob, not a bytes lever. The cheap form, M-11's mean correction (-22 % block error for ~+0.3 % bytes, a per-expert vector added to the gate / up pre-activations and the down output of 3-bit ranks), could ride in the next reference bundle if someone wants KLD headroom |
| M-12 | MoonShots (CPU) | RARE2 [SSD-miss wall; published pieces: static per-expert bit widths by usage (MC-MoE, MxMoE, AlphaQ). Pre-registered Sun 06:57]: misses land mostly on rarely used experts. Rule (per token, so the loop stays exact vs its own one-token decoding): experts in a static rare set (bottom 20 / 40 % by calibration routing frequency, per layer) use a 2-bit tier (MSB-2, b2 = bf16(b + s mean(q & 3))) at ranks 2-9; everything else as REF2-T. M-11's mean correction pays for it. Step 1 (this row): quality on M-11's data and metric (calibration 0-4095, scored 4096-8191, 5 layers). Step 2 only if step 1 passes: a cap45 replay at C 224 / 288 for the share of SSD miss bytes saved | PASS: rare-40 % with the correction at <= 1.00x uncorrected REF2-T's block error -> step 2. KILL: rare-20 % with the correction > 1.20x. BETWEEN otherwise | **Step 1 BETWEEN (MoonShots 07:0x, scripts/rare2.py, data/rare2.txt; CPU 209 s)**: rare-20 % (102 experts per layer, 4.9 % of rank 2-9 picks go 2-bit) is 1.20x uncorrected REF2-T, 0.97x with M-11's correction. Rare-40 % (12.4 % of picks) is 1.38x, 1.15x corrected. So the correction's headroom pays for rare-20 % only. Step 2 (cap45 replay: share of SSD miss bytes on rare-20 % experts at ranks 2-9) not run per the lines; offered to NewIdeas. Rough upper estimate -10-15 % SSD bytes (an MSB-2 miss is ~1.4 MB vs 2.16). Cost: a split record plane for rare experts, a per-expert tier flag and correction vectors in tier_glu, and a gate run | **Step 2 GO (NewIdeas 07:4x; pre-registered Sun 07:4x, lines NewIdeas')**: cap45 chained replay under the live rule LRU + popularity prior (BigMoE seed45_lru.py LRU+P30) at C 224 / 256 / 278 (K 246 / 287 / ~315 MSB-3 slots). Rare-20 % = the bottom 102 experts per layer by popularity on the OTHER prompts. Per-row ranks recomputed from cap45's x with the real router. Bytes (DESIGN_3bit.md record sizes): a miss = 2.16 MB MSB-3; under RARE2 a rare expert needed only at ranks 2-9 reads its MSB-2 record (2-bit codes + s + b2, 1.54 MB); a later rank 0-1 use of it pays the bit-1 plane (+0.77 MB); rare experts needed at rank 0-1 read MSB-2 + bit-1 (2.30 MB). LSB traffic is the same in both arms and left out. PASS >= 10 % fewer SSD MB per kept token at C 256, KILL < 5 %, BETWEEN otherwise. **Step 2 KILL (MoonShots 07:4x, scripts/rare2ssd.py, data/rare2ssd.txt; CPU 24 s)**: under LRU + 30 x popularity, chained, RARE2 saves 4.3 % of MSB-side SSD MB per kept token at C 256 (3.8 % at C 224, 4.8 % at C 278). Rare-20 % experts take only 5.7 of 32.4 misses per kept token at C 256 (~18 %; the prior already keeps popular ones, and popular experts miss too, M-13); each saves 0.62 MB of 2.16 (upgrades for later rank-0-1 use: 0.3 per kept token). The recomputed router holds 98.16 % of the captured ids. The M-12 line is closed; M-11's correction does not join.
| M-13 | MoonShots (CPU, cap45 replay) | MISSFLOOR [SSD-miss wall; a measurement for BigMoE's cache sizing, no new method. Pre-registered Sun 07:0x]: how low can misses go with a bigger cache alone? cap45 (6 live REF2 prompts, real 16-row checks, the residency at loop start res0 at C 160). Per layer: (a) LRU and decayed-LFU replays at C 160 (must land within 15 % of the logged medians: 970 / 552 / 624 / 870 / 601 / 666 per check), then C 224 / 288 / 384; (b) the floor with an unlimited cache: experts first used in the decode and absent from res0, per check, in each answer's second half (steady state). In misses per kept token and SSD MB per kept token | PASS (cache size can close the SSD side for 350): the steady-state floor <= 14 misses per kept token on the code prompts (~30 MB at 2.16 MB = 350 tok/s at 10.6 GB/s). KILL (fewer bytes per miss needed whatever the RAM): > 28. BETWEEN otherwise | **KILL by the letter, a sizing pointer in substance (MoonShots 07:0x, scripts/missfloor45.py, data/missfloor45.txt; CPU 3 s)**: the decayed-LFU replay lands within 10 % of the logged C 160 medians on 5 of 6 prompts (HumanEval/68 has only 5 checks captured; dropped). From the C 160 start, even an unlimited cache leaves 13-30 first-use experts per kept token in each answer's second half (code 29.8 on HumanEval/91; math 13-23; line 14). But those first uses are mostly popular experts: 42-53 % fall in the top 288 per layer and 68-79 % in the top 384 (popularity counted leave-one-prompt-out). So a popularity-warm C 384 cache would cut them ~4x, code ~30 -> ~7 per kept token, inside the 350 SSD budget. C 288 only halves them. LFU replay from res0, median misses per check, C 224 / 288 / 384: 263-448 / 152-284 / 117-228 (vs C 160 534-804). RAM: S45b's ~104 MB per slot at REF2-T sizes makes C 384 ~50 GB. Short answers (<= 256 tokens) and 6 prompts: a pointer for BigMoE's sizing, not a result |
| M-14 | MoonShots (CPU, cap45 chained replay) | EVICT45 [SSD-miss wall. BigMoE SEED45 (07:2x): at K 266 (C 240) the live LRU leaves 56 % to Belady (43.5 -> 19.2 misses per kept token); popularity priors take 8 %. RANKED #12 still says '<= ~5-8 % (our Belady bound)', a 24 GB number. Published: SRRIP / DRRIP (Jaleel 2010), DIP (Qureshi 2007), Belady-labelled learned eviction (Hawkeye 2016; FlashMoE 2601.17063). Pre-registered Sun 07:38]: how much of the gap do online policies take? Same chained replay as BigMoE's seed45_replay.py (cap45, res0 forced in per prompt, prompts 2-6 counted). Policies per layer: LRU; SRRIP (insert at RRPV 1 / 2 / 3 of 0-3); second-use promotion (new experts inserted at the LRU end); a learned key, a linear fit of log next-use distance on recency, decayed count, popularity, rows that used the expert, used by a kept row, in the prompt's prefill set and layer, trained leave-one-prompt-out on Belady labels | PASS: the best online policy takes >= 30 % of the LRU -> MIN gap at K 266 (<= 36.2 misses per kept token) with no prompt worse than LRU by > 2 % -> BigMoE builds it (residency only, exact). KILL: < 10 % (> 41.1). BETWEEN otherwise | **BETWEEN (MoonShots 07:4x, scripts/evict45.py, data/evict45.txt; CPU 35 s)**: replay matches BigMoE's (MIN 19.2 exactly; LRU 42.9 vs 43.5). At K 266 the learned key, leave-one-prompt-out, gives 39.5 misses per kept token (-7.9 % vs LRU, every prompt better by >= 2.5 %) = 14 % of the LRU -> MIN gap; SRRIP-2 41.4 (6 %). LIP / mid insertion are worse than LRU (new experts are reused soon). Same shares at K 246 / 287. The key weights popularity (strong), recency, and prefill-only experts (go far). Horizon oracles show where Belady's gain is (LRU + protect experts needed within H checks): H 1 15 %, 2 26 %, 3 35 %, 5 49 %, 10 68 %, 20 74 % of the gap. So even a perfect next-check routing predictor gives only ~15 %; the gap is medium-range reuse (5-20 checks) of the answer's working set. Follow-up M-14b: in-answer working-set features (uses so far in this prompt, short-window frequency) and a binned nonlinear key |
| M-14b | MoonShots (CPU, cap45 chained replay) | EVICT45-WS [follow-up of M-14: the gap is medium-range reuse of the answer's working set (horizon oracle H 5 49 %, H 10 68 %). Pre-registered Sun 07:42]: add in-answer features to the key: uses so far in this prompt, uses in the last 10 checks, mean reuse interval in this prompt, checks since the prompt started. Two keys, both trained leave-one-prompt-out on Belady labels from LRU states: the linear key with these features, and a binned table (recency x in-prompt uses x popularity quintile -> mean log next use). Same chained replay, K 246 / 266 / 287 | PASS: >= 30 % of the LRU -> MIN gap at K 266 (<= 35.8 misses per kept token) with no prompt worse than LRU -> BigMoE builds the key (residency only, exact). KILL: no better than M-14's linear key (< 16 % of the gap). BETWEEN otherwise | **BETWEEN (MoonShots 07:4x, scripts/evict45ws.py, data/evict45ws.txt; CPU 51 s)**: at K 266 the linear key with in-answer features gives 38.4 misses per kept token vs LRU 42.9 (-10.5 %, every prompt better by >= 6.5 %) = 19 % of the gap; the binned table 38.9 (17 %); M-14's key 39.5 (14 %). Same shares at K 246 / 287 (19 %). Uses so far in this prompt carry the new signal (coefficient -0.28; more uses -> sooner reuse). An exact, residency-only key: a linear score over a few per-expert counters (recency, decayed count, popularity, in-prompt uses, prefill flags), cheap in the ext. Worth ~-10 % SSD misses on top of a bigger C; the remaining ~80 % of the Belady gap needs 5-20 checks of foresight that no online signal here gives |
| M-15 | MoonShots (CPU, D-02a teacher-forced replay) | EXACT-NT [kept wall on chat. Published core: speculative cascades (Narasimhan et al., ICLR 2025, arXiv 2405.19261): a token-level deferral rule run by speculative execution stays lossless w.r.t. that rule. New here: the block drafter's FRESH first-position token as the deferral token. Pre-registered Sun 09:0x]: RANKED #21's near-tie acceptance made EXACT. The served model's own one-token rule: take T = the drafter's token from its normal block call started at this position (L1-20, cand21's selector at t 0.0, depth 1) when T != the 125B's top-1 and p_125B(T) >= rho x p_top1, else the top-1. T depends only on the prefix (target features up to the position), so the live loop stays bit-exact vs this rule's own AR decode (needs batch-invariant drafter calls). The rule itself is a served-model change (strict gate + Kamen, like REF2-T), not a non-exact loop. Loop cost: T is needed only at rows with p_top1 <= 1/(1+rho) up to the first miss (~1-2 extra drafter calls per chat check). What the exact form loses vs R-61N: a first miss is forgiven only if the block's drafted token == T, and a hit turns into a miss where the block's token is the top-1 but T is another near-tie. Replay: R-61N's chat chain (TrainLab data/r61n/chatnt_b16.jsonl: 6,145 blocks, 54 answers) + a fresh drafter block at every chat position (~13.7K calls, MLX on the CPU, <= 4 threads, PAUSE-gated); forgiven share, rule-made misses, tokens per pass LB / MOD by nt_analyze's rules at rho 0.5 / 0.25 / 0.1. Self-check: the chain's depth-1 tokens == the fresh T at the same positions | PASS: chat MOD >= +15 % at rho 0.25 net of rule-made misses (>= 2/3 of R-61N's non-exact +23 %) -> NewIdeas: RANKED #21 gets an exact-loop form + the strict-gate quality test for Kamen. KILL: < +8 % at rho 0.25. BETWEEN otherwise | **BETWEEN, final (MoonShots Sun 11:4x; ~/MoonShots/scripts/exactnt.py, data/exactnt.txt; as run data/exactnt_v1.txt; stage 1 = 13.7K fresh drafter blocks on the shared GPU, 27 min)**: R-61N reproduced exactly (rho 0.25: 17.7 % forgiven, LB +7.8 % / MOD +22.9 %). EXACT-NT, chat, rho 0.25: 13.1 % forgiven, 0.017 rule-made misses per block, LB +4.5 % / MOD +14.0 % (ifeval +19.6 %, multi +9.3 %); rho 0.1 +22.2 % (R-61N +38.8 %; ifeval +30.4 %); rho 0.5 +6.6 % (+10.1 %). The exact form keeps ~61 % of R-61N's gain (line 2/3). Why: at first misses the block's token == the fresh T at 97.1 % at depth 1 (the 2.9 % gap = my replay's cache-chunking numerics; batch-invariant drafter calls make it 100 %, so these numbers are slightly pessimistic) but only 19.7 % at depth >= 2, where the fresh drafter usually picks something else; the fresh block would have hit the text at 27.3 % of all first misses. Convention: the text token is the greedy one, as in R-61N (it != the capture's top-1 at 2.2 % of positions); with the capture's top-1 (as run) +12.5 %. Meaning: RANKED #21 has an exact-loop form at ~60 % of its gain, as a served-model change (strict gate + Kamen, like REF2-T). What would lift it: a drafter whose deep block tokens agree with its own fresh first tokens (self-consistency across anchors; MTP Head's training lane) |
| M-16 | MoonShots (CPU sizing now; live dual-pass screen tonight, 45 GB route) | RUNAHEAD [SSD wall at 45 GB, exact. Published core: runahead execution (Mutlu et al., HPCA 2003: on a long cache miss the core runs ahead with invalid values only to discover and prefetch later misses, then re-executes exactly). Prior art checked Sun 11:5x: draft-model expert prediction + prefetch (MoE-SpeQ / Specter 2511.14102, SP-MoE 2510.10302), residency-gated NON-exact verify (AcceptMoE 2608.02989), speculative expert execution (2603.19289, non-exact, -14 % TPOT); our one-layer look-ahead R-57 / R-57b KILL (+3.4 %: recall 0.69, lead = one layer). New here: the TARGET itself is the runahead, over the SAME drafted rows, and the exact re-check rides in the next pass. Pre-registered Sun 11:5x]: today the check waits on its SSD misses layer by layer: R-69b (C 160, OOO) cycle 157 ms with demand SSD 99 ms, and per prompt cycle ~ 50 ms + 1.1 x demand SSD, so the misses sit on the critical path. RUNAHEAD: pass k computes [A = the rows pass k-1 accepted provisionally, now all resident -> EXACT verdict, tokens emitted only from here] + [B = the next drafted block, computed with non-resident experts SKIPPED (no wait); B's misses are queued to the SSD as they are found, deep queue, and land during the rest of pass k]. B's provisional verdict picks the bonus token and the next block; pass k+1 re-checks B exactly. Output bit-exact (every emitted row comes from an all-resident exact pass; needs batch-invariant rows, REF2b). Cost: accepted rows are computed twice (~1.2-1.4x compute per pass); a pass whose provisional verdict differs from the exact one wastes its B rows (one bubble). Model: cycle ~ max(compute(A+B) ~65-75 ms, SSD at deep queue ~55-80 ms) vs ~140-157 ms today -> up to ~1.8-2x before bubbles. Decisive unknowns: (U1) recall of the exact pass's misses by B's runahead requests; (U2) provisional == exact verdict (accepted length + bonus); (U3) the next block drafted from runahead features == the exact one; (U4) compute of A+B. Step 1 (CPU, now, scripts/runahead_mass.py, cap45 + LRU + prior replay at C 224 / 256 + real experts): skipped routing mass per row-layer, the share of misses needed at ranks 0-1, and the MoE block error of runahead variants vs REF2-T (skip; skip + renormalize; skip + per-expert mean-output stand-in; hybrid = wait for misses needed at ranks 0-1, skip the rest) on layers 2 / 12 / 24 / 36 / 46, in units of REF2-T's own tier error vs 4-bit (M-11: 0.0121). Step 2 (tonight, 45 GB route, own harness on read-only BigMoE / EvalSuite imports): live dual pass per check (runahead then exact, same cache state) -> U1 / U2 / U3 measured, U4 timed | Step 1 PASS: the best variant's block error <= 2x REF2-T's tier error (the 4433333300 drop-ranks-8-9 level is 2.33x), with >= 70 % of the misses still skipped -> step 2 tonight. KILL: every variant > 5x. BETWEEN otherwise (step 2 at lower priority). Step 2 PASS (-> BIG + a BigMoE build proposal): modelled tok/s >= +40 % at C 224 from measured U1-U4; KILL < +15 % | **Step 1 PASS (MoonShots 12:3x; ~/MoonShots/scripts/runahead_mass.py, data/runahead_mass.txt, 30 min CPU)**: cap45, 171 live 16-row checks (prompts 2-6), LRU + prior at C 224 (K 246): 288 misses per check; skipped routing mass per row-layer mean 4.1 % (P50 0, P90 13 %, P99 45 %; 30 % of row-layers skip something; rank 0 skipped on 2.1 %); 9.6 % of the misses are needed at rank 0-1 by some row. MoE block error vs the REF2-T block, layers 2 / 12 / 24 / 36 / 46 pooled, in units of REF2-T's own tier error vs 4-bit on the same rows (0.0100): skip 0.6x, skip + renormalize 0.8x, mean stand-in 0.6x (no help), HYBRID (wait for the rank 0-1 misses, skip the other 90 %) 0.3x; C 256: skip 0.4x, hybrid 0.2x. Caveat: pooled error is dominated by layers 36 / 46 (largest outputs); at layer 2 skip is ~9x REF2-T's error there (small outputs). Step 2 harness ready (~/MoonShots/scripts/runahead_live.py + runahead_turn.sh): per check a skip runahead pass and a hybrid runahead pass (resident slots via the ext's side-effect-free lookup + tierglu phase 1; rolled back) around the exact pass, then an all-resident exact re-run (must be bitwise == the first exact pass) that commits; self-check tokens == S45-T's REF2-T reference; U1-U4 for both variants; ~5 min at C 224, 45 GB route -> asks for a night slot  **Sensitivity, written before step 2 (MoonShots 12:4x; scripts/runahead_sens.py on MTP Head's D-23 dfl log, 45 GB default, 18 prompts, 1,000 checks; Breakthrough's FIXED-PART constant 30 ms + draft, SSD 0.149 ms per miss)**: the gain hinges on the PURE compute per re-checked row (beta; D-23b's 7.82 ms per row includes that row's SSD misses). beta 2.5 ms: code +70-81 %, math +65-78 %, chat +42-54 %; 3.5: +46-55 / +40-50 / +30-41 %; 5.0: +19-27 / +13-21 / +16-25 %; 7.82 (all compute): -9..-17 / ~0 % (bubble 5-15 %; agent overstated: no context term). Physical guess 3.8-5.6 ms (7.82 minus the row's ~27 misses x 0.149 ms, fully or 56 % on the critical path). Step 2 measures beta directly (the all-resident exact re-run's ms vs rows). Engine refinement if it lives: wait for the misses of the first L1 layers and re-check A rows only from layer L1 (a ragged pass)  Prior-art re-check 12:5x (2 more searches): AHEAD (ASP-DAC 2027, Zhang et al.: self-draft + at most one expert prefetch during drafting, demand loading at verify) and MoE-Spec 2602.16052 (expert budget, non-exact) are not runahead verification; nothing found that runs the target with misses skipped and re-checks exactly next pass  **STEP 2 KILL (MoonShots Mon 01:31; ~/MoonShots/data/runahead/ra_224.txt + ra_224_checks.jsonl; one quiet turn at C 224, 231 s, 481 checks over 6 code / math + 3 chat prompts)**: modelled tok/s vs today skip +0.4 % (hybrid -6.5 %, hybrid8 -13.0 %; SSD at 10 GB/s -4.7 %; A + B within 16 rows -10.4 / -17.4 / -23.3 %); every token == the REF2-T reference, all-resident re-runs bitwise == the first exact pass 481 / 481. Measured: all-resident pass 47.9 ms + 2.64 ms per row (beta as hoped), U2 skip 78-97 % (hybrid 90-100 %), U1 recall 71-84 % (hybrid ~89-93 %), U3 64-93 %, SSD at deep queue 0.157 ms per read. Why: the default at C 224 is SSD-BANDWIDTH-bound, not latency-bound: on code today's ~936 reads per check take ~147 ms even at deep queue (~70 % of a ~200 ms check), so overlap has little left to hide, while each runahead pass adds the fixed 47.9 ms, bubbles (3-22 % of checks) and ~20 % unrequested misses. The C 192 arm was cut by a heat PAUSE. The 45 GB wall is bytes, not latency (agrees with CACHE-SIZE45): the levers are fewer bytes (M-17) and more RAM |
| M-17 | MoonShots (CPU replay, any time; no slot) | LAYSZ45 [exact: residency only. Published core: per-layer cache allocation (AdapMoE-style). Ours: R-27 LAYSZ at 24 GB, -10.3 % bytes per token held-out in the strict replay, but BigMoE's engine A/B b36 gave -1.5 % (noise 1.0; JIT prefetch + JOINT covered it), so R-27 never became the 24 GB config; the mechanism stays in the engine (BIGMOE_TIER_SIZES, read by stream._tier_layer, the tier path the 45 GB default uses). Pre-registered Sun 14:5x]: per-layer (C_m, C_l) for the 45 GB default instead of the uniform split (C 224 = 246 MSB-3 + 112 LSB slots per layer; C 256 = 287 + 112), same total pool bytes (slots 2,162,688 / 770,048 B). Replay (~/MoonShots/scripts/laysz45.py): cap45's six prompts chained (res0 forced per prompt, prompts 2-6 counted), the live rule LRU + 30 x popularity prior (leave-one-prompt-out, as seed45_lru.py), strict pools (trimmed to C after every check); MSB pool needs = the check's union of top-10 experts, LSB pool needs = the union of rank 0-1 experts (ranks from the real router, as runahead_mass.py). Per-layer miss curves on a grid; Lagrangian over all 96 pool sizes at the uniform pool bytes; held-out = calibrate on the other counted prompts, score the left-out one; SSD bytes = MSB misses x 2,162,688 + LSB misses x 770,048 per kept token. Arms: FLOOR (C_m >= 160 = the gate's MAXK, C_l >= 32 = 16 rows x 2 ranks: today's ext needs no staging) = the verdict arm; FREE (C_m >= 64, C_l >= 0) and UNIFORM RE-SPLIT (one C_m / C_l for all layers) reported for reference. Caveat written first: cap45 checks are 16 rows; the default's draft truncation (t 0.3) makes checks shorter. Added 14:5x before any result: a SHORT robustness arm (P30) where each cap45 check keeps its first min(16, kept + d) rows, d drawn from D-23's code + math checks at C 224 (rows - kept: mean 1.8 code / 2.4 math; D-23 rows mean 12.3 code / 9.0 math / 3.1 chat; cap45 has no chat prompt) | PASS: FLOOR held-out SSD MB per kept token <= -8 % vs uniform at both C 224 and C 256 (-> an engine A/B via BIGMOE_TIER_SIZES at C 224). KILL: > -3 % at both. BETWEEN otherwise | **PASS (MoonShots 15:2x; ~/MoonShots/scripts/laysz45.py, data/laysz45.txt, 41 s CPU)**: held-out SSD MB per kept token FLOOR -20.7 % at C 224 (116.9 -> 92.7) and -19.3 % at C 256 (85.0 -> 68.6), every prompt -12.9 .. -27.6 %; the replay reproduces SEED45 exactly (LRU + prior 47.1 / 32.4 base misses per kept token). FREE -20.6 / -19.3 % (the floor costs nothing); RESPLIT (one C_m / C_l for all layers) -3.0 / -0.6 % (the gain is the per-layer shape); plain LRU -20.9 / -21.0 %; SHORT checks (8.0 rows) -19.0 / -18.1 %. Cross-domain (scripts/xdom45.py; Breakthrough's expA traces as 16-row checks, cap45's sizes): bulgarian -32.7 / -25.1 %, chat -22.0 / -14.2 %, code -18.6 / -15.2 %, math -24.8 / -25.6 % (each set's own optimum -22 .. -41 %). Sizes: C 224 C_m 160-408 (median 252), C_l 32-128 (median 72); C 256 C_m 160-448, C_l 32-160; layers 0-2 / 6-7 ~350-450, layers 15-16 / 31-32 at the floor; MSB share of pool bytes 86 -> 91 %. BIGMOE_TIER_SIZES files ~/MoonShots/data/laysz45_224.json / _256.json. R-27 never became the 24 GB config (b36 engine A/B: -1.5 % bytes vs -8.6 % replayed); the mechanism stays in stream._tier_layer. Expected ~+6 % tok/s whole set, ~+10 % code (D-23: SSD wait 29 % of the cycle). Proposed engine A/B (NewIdeas' call): C 224, R57_PRESET=45, uniform vs laysz45_224.json; PASS <= -10 % bytes at tok/s >= +3 %, KILL > -3 % bytes  **C 300 / 320 sizes (NewIdeas 20:4x; MoonShots 22:50, LAYSZ45_C=300,320 LAYSZ45_RUNS=16:P30, data/laysz45_c300_320.txt, 22 s CPU)**: held-out FLOOR (LRU + prior, strict, C_m >= 160, C_l >= 32) C 300 (343 + 112 per layer, 39.75 GB of pools) 56.4 -> 46.2 MB per kept token (-18.1 %; per prompt -10.1 .. -29.4 %), C 320 (369 + 112, 42.45 GB) 47.0 -> 38.0 (-19.3 %; -7.2 .. -32.5 %); sizes C 300 C_m 168-472 / C_l 40-200, C 320 C_m 184-480 / C_l 40-208. md5 (pin): laysz45_224.json 3098a50550e27a1ebb2075e3caf86960, laysz45_256.json ee76e2a7390c18f8d53b539a3ba9700a, laysz45_300.json cd0df64a980d217d75a53a10d600ddce, laysz45_320.json f0a3997c88379d01cdc2b39175c39307  **ENGINE PASS (BigMoE C45 arm L, Mon ~01:3x, quiet, C 224, laysz45_224.json)**: SSD MB per kept token 96.0 -> 79.5 (-17.2 %; replay -20.7 %), steady tok/s +8.3 %, exact; HumanEval/91 -7.1 % (BigMoE's lounge note to MoonShots). **ENGINE PASS at C 256 (BigMoE C50, Mon 07:20, quiet, laysz45_256.json)**: SSD MB per kept token 74.0 -> 62.4 (-15.7 %; replay -19.3 %), steady +5.0 %, exact, 5/6 prompts faster (HumanEval/91 +7.1 %); C 300 / 320 skipped (NO-SWAP 49 < 52 / 55). ADOPTED in R57_PRESET=45 (md5-pinned) |
| M-18 | MoonShots (CPU replay, DAY9 conditions 1-2; no slot) | BAND45 [bytes side of ROUTE_350 Q1 / Q2 (NEW-1); the flips are Breakthrough's map. Pre-registered Mon 10:4x, before any result]: SSD reads and MB per kept token at C 256 + the M-17 per-layer sizes when drafter-sure rows of the check verify with fewer experts. Replay ~/MoonShots/scripts/band45.py: cap45's six prompts chained (real check rows incl. rejected ones), LRU + 30 x popularity (one fixed prior per seed), strict; drafter confidence = real DFlash2 cf vectors of D-23 checks (arm dfl, C 224) transplanted by type and kept (6 seeds), the engine's t 0.3 row cut; row j gated by cf[j] (the token it verifies), the last row of a full check exact. Rules: Q1 top-2 (ranks 0-1 at today's tier, own gate weights) at cf >= 0.9 / 0.95 / 0.99 / 0.999; Q2 bands >= 0.999 top-2, 0.99-0.999 top-4, 0.9-0.99 top-8 with ranks 0-3 at 4 bits, < 0.9 exact. Tok/s: C50's c45l_256 per prompt (63.6 steady = median) with each cycle cut by the saved SSD bytes at 12 GB/s (fully exposed) and, as the upper bound, also by the saved expert-union RAM bytes at the GPU rate M-16's beta implies. Lines vs ROUTE_350's bytes assumption [P] (code SSD x0.55 with Q1; Q2 a further x0.85): Q1 at cf >= 0.99 on code SSD MB per kept token <= x0.65 = PASS (the route's SSD term holds), > x0.85 = KILL (overstated > 3x), BETWEEN otherwise; Q2 <= x0.90 of Q1 0.99 on code = holds. Then the same replay on Breakthrough's capture (realizable gate on its windows) when it lands. | **Mon 10:5x RESULT (commit 4b4d084; data/band45_capture.txt, band45.txt)**: Q1 at cf >= 0.99 on code: SSD MB per kept token x0.60 (-40.3 %) on Breakthrough's capture (12 code windows, 391 checks, realizable DFlash2 gate) -> **PASS** (<= x0.65); x0.68 on cap45 + D-23 transplant (BETWEEN there; real rejected-row routing, thin code). Reads per check 260 -> 186 (base 142 -> 68, rank 0-1 sidecar 118 unchanged), expert union -41 %. Implied steady (C50 63.6): 70.2 SSD-only / 76.2 +GPU (six-prompt median); HE91 62.1 -> 74.7 / 88.9, HE68 84.7 -> 114.5 / 125.8. cf >= 0.999: -28.3 %, 68.1 / 71.3. IFEval -10.5 %, multi -8.4 % (drafter sure on 21 / 18 % of chat rows vs 67 % code). Q2 (DAY9 bands): code x0.63 (-37.3 %) = x1.05 of Q1 0.99 -> the route's 'a further x0.85' FAILS on bytes (the 0.9-0.99 band's ranks 0-3 at 4 bits add sidecar reads, LSB +23 %); with that band at today's tiers x0.57 (-42.6 %). Lead: sure rows at the resident 3-bit part (no LSB read): code x0.46 (-53.8 %), median 72.1 / 79.3 (needs a flip arm). **M-18b LAYSZ-Q1 (pre-registered Mon 11:0x, before the run)**: re-fit M-17's per-layer C_m / C_l split (Lagrangian, FLOOR C_m >= 160, C_l >= 32, C 256 pool bytes) for Q1 0.99's needs; cross-set held-out: fit on cap45 (transplant, seed 0) -> score on the capture, and the reverse; vs the adopted M-17 sizes. PASS = code SSD MB per kept token under Q1 0.99 <= -8 % vs M-17's sizes held-out in both directions; KILL = > -3 % in both; BETWEEN otherwise (scripts/laysz_q1.py). **Flips (Breakthrough 11:0x, ~/FableGod/decision_map/out/DAY9_q1.txt)**: cf >= 0.99 gated code rows 0.09 % (2/2211), IFEval 0.59 %, multi 1.26 %; cf >= 0.999 0.00 %; all code rows 1.34 % / 1.25 % -> Q1 PASS both sides (sent to the orchestrator with these tok/s). The two tier arms (3-bit sure rows; top-8 at today's tiers) need a ~2 h exact build in the map (BigMoE's tier path works at decode widths only); decide after M-18b. **M-18b RESULT (Mon 11:1x, commit ad221f5, data/laysz_q1.txt): KILL** - the per-layer split re-fitted for Q1 0.99's needs is +0.6 % (fit on the capture, scored on cap45) / -2.1 % (fit on cap45, scored on the capture) on code vs M-17's sizes held-out (in-sample -0.9 / -5.3 %); MSB share 87-89 % vs M-17's 90.3 %; Q1-fitted sizes cost EXACT +1-5 %. M-17's sizes stay; after Q1 the rank 0-1 sidecar reads drop only by reading less (the 3-bit sure-row arm, x0.46; the orchestrator's call on Breakthrough's ~2 h build). **Q2 flips (Breakthrough 11:2x, ~/FableGod/decision_map/out/DAY9_read.txt, DAY9_q2_attrib.txt)**: KILL by the line as written (0.9-0.99 top-8 band 1.07 %, 5/467), but that band flips the same 1.07 % when left exact (its own cut net 0 on code); with no byte gain over Q1 0.99 (x1.05) Q2 adds nothing on either side. Mon 15:59: the 3-bit arm is queued at Breakthrough behind E2 + A6 (wait_t3b.sh). |
> (a quote from the project owner, kept private)
| M-18c | MoonShots (CPU replay; no slot) | WIDE-Q1 [pre-registered by NewIdeas Mon 20:48, before any run; carried by band45.py]: does Q1 (top-2 on drafter-sure rows, cf >= 0.99) make 32-row checks pay on code? (D32: dense x1.24 for 32 rows vs 16; NAX6 pays only at wide checks.) Rows 1-16 as M-18; rows 17-32 = the next positions of the 125B's own answer; kept per check from MTP Head's MATCH 32-row replay (a2086f2: HumanEval+ x1.204 at 32 rows, 12.35 kept); rows gated by the drafter's cf where it exists, else MATCH-copied rows treated as sure; same C 256 pools with M-17 sizes, LRU + prior; cost: dense x1.24 (D32), a block-32 draft (MoonShots' estimate), SSD at 12 GB/s fully exposed. Line on code implied steady tok/s, 32 rows + Q1 vs 16 rows + Q1: >= +10 % = PASS (wide checks + NAX6 earn an engine slot, Leap / BigMoE); <= +3 % = KILL; BETWEEN otherwise. Per prompt (HumanEval/68-like prompts sit at the 16-row cap). **RESULT Mon 22:4x (scripts/wide45.py, data/wide45.txt; 4 seeds, guarded run, peak 171 MB): KILL** - 32 rows + Q1 vs 16 rows + Q1, both with the engine's t 0.3 cut, pooled code tok/s: -1.3 % (today's kernels) / -0.6 % (NAX6) (line <= +3 % KILL). Kept 7.98 -> 9.08 per check, rows 10.61 -> 14.30, Q1 SSD MB per kept token 29.0 -> 28.2. Without the cut (always 16 / 32): -16.8 / -14.0 %. Optimistic variant (rows 16-31 all top-2, NewIdeas' 'MATCH-copied rows sure'): +4.3 / +3.8 % (BETWEEN). Per window: only cap-bound prompts gain (HumanEval/136 kept 13.25 -> 21.20, +13 %; HE135 / 103 / 146 +3-4 %); median -0.4 / +0.1 %. EXACT alone: -0.2 / 0.0 %. Replay check: the 16-row cut walk gives 46.5 MB per kept token EXACT / 29.0 Q1 (M-18: 47.0 / 28.1); '16 always' reproduces the real uncut walk on 373 / 391 checks. Width chosen per check is M-21's question. |
| M-18d | MoonShots (CPU replay; no slot) | Q1-24 [pre-registered by NewIdeas Mon 22:00 (its message), before any run; the 24 GB line's new bet]: does Q1 (top-2 experts on drafter-sure verify rows, cf >= 0.99, kept weights not renormalised) cut the 24 GB line's SSD bytes as much as at 45 GB? Replay at the 24 GB default (BigMoE fast24.py / bigmoe_ext.cpp): plan4433333333 (v3), C_m 147 + C_l 46 per layer, decayed LFU 0.99 per gated call, narrow verify passes; arms v3 exact and b16 (bigmoe/budget.py: the 16 experts with the largest summed w^2 per pass). Option A (MoonShots 22:21, NewIdeas did not choose B by 22:55): Breakthrough's 12 code windows, 4-row checks, acceptance coupled to the real DFlash2 walk, the Q1 gate on DFlash2's cf as the stand-in for MTP's (no per-row MTP confidences exist), b16 weights ~1 / (rank + 1) (the capture has no gate weights). scripts/q1_24.py, 4 seeds. Line (NewIdeas'): code SSD MB per kept token, Q1 vs the same replay without Q1: <= x0.70 = PASS (Q1 joins the 24 GB FAST version's same-scores test), > x0.90 = KILL, else BETWEEN. **RESULT Mon 23:0x, option A (scripts/q1_24.py, data/q1_24.txt; 4 seeds, guarded, peak 171 MB)**: v3 exact verify 175.0 -> 115.2 SSD MB per kept token = **x0.658 -> PASS** (line <= x0.70; within 0.05 of it, so option B, MTP Head's per-row MTP probabilities, is requested to confirm); **Option B run pre-registered Tue 11:1x, same line, before running** (MTP Head's export ~/TrainLab/data/mtp_map_code/, 06:57: the native MTP chain's own right / wrong per depth walks the 4-row checks and its top-1 probability gates Q1; scripts/q1_24.py with Q124_GATE=mtp -> data/q1_24_mtp.txt); **Option B RESULT Tue 11:1x: v3 175.5 -> 100.4 = x0.572 -> PASS (confirms A with margin); b16 82.0 -> 86.8 = x1.058 -> KILL**; MTP's 4-row checks keep 3.75 per check (the DFlash2 stand-in 3.52); **re-check Tue 17:0x on MTP Head's aligned v2 export (v1 was one row late): v3 x0.574 PASS, b16 x1.058 KILL, kept 3.74 - unchanged** (data/q1_24_mtp_mtp_map_code_v2.txt); b16 81.8 -> 88.2 = **x1.079 -> KILL** (the budget already caps the union at 16, and Q1 only changes which 16 fill it, toward less cached experts). Kept per 4-row check 3.52 (DFlash2's walk; conservative for MTP). |
| M-20 | MoonShots (SSD + CPU timing, ~2 min, quiet lock; no slot) | SSDWAKE [new-idea ask: Kamen 21:5x via the orchestrator, 'genuinely new' aimed at the 45 GB SSD stack. Not in TRIED (power-state / APST / keepalive absent; GPU idle-gap DVFS was measured: #3 KILL, R-04 BETWEEN). Prior-art check 22:0x: no MoE / LLM-offload work on SSD idle states, no published Apple Silicon wake-latency numbers; PC NVMe pays 2-60 ms after deep idle (entered after 10s-100s of ms), so my prior is low (~25 %) for the route's ~1 ms gaps. Pre-registered Mon 22:0x, before any timing]: does the internal SSD charge a wake-up penalty on the first reads after the short idle gaps the 45 GB route leaves between layers (each layer's demand misses go out as one burst after its router: ~3.6 misses x 9 pieces; the SSD idles ~1 ms per layer while the GPU computes, ~10-15 ms around draft + commit), and does a trickle of 4 KB keepalive reads (every ~200 us, ~20 MB/s) remove it? Screen: scripts/ssdwake.py, F_NOCACHE preads of fresh random experts' MSB parts from records3 (read-only), bursts of 4 x 9 pieces and single 2.16 MB reads after busy-waited gaps 0 / 0.25 / 0.5 / 1 / 2 / 5 / 10 / 20 ms, plain vs keepalive alternating within each of 40 rounds, default QoS. Line: PASS if the burst after a 1 ms gap is >= 15 % and >= 50 us slower than back to back AND the keepalive removes >= 70 % of that penalty; KILL if the 1 ms-gap penalty is < 5 % or < 20 us for bursts and single reads; BETWEEN otherwise. Exact either way (no model change). If PASS: ~48 bursts per check x the penalty is the prize, and the keepalive goes to BigMoE's loader. **RESULT Mon 22:09 (quiet lock; scripts/ssdwake.py, data/ssdwake.txt): BETWEEN by the line** - the 1 ms-gap burst penalty is +64 us (+7.5 %, line 15 %) and the pre-registered 4 KB / 200 us keepalive removes 44 % (line 70 %). The effect is real and starts at short gaps: single 2.16 MB reads +16 % after 0.25 ms, +18 % (+54 us) after 1 ms, +33 % after 20 ms (back to back 296 us; bursts 847 us). Exploratory follow-up (not pre-registered): a busy CPU core with no I/O changes nothing (the SSD path, not host wake-ups); a 4 KB read every 50 us (~80 MB/s) removes all of it (burst -3 us, single -6 us at 1 ms) without slowing back-to-back bursts. Prize: ~48 bursts per check x ~60 us = ~3 ms per check at 45 GB (~+3 % at C50's 96.8 ms cycle, ~9 % at a 34 ms cycle), exact; the engine's real gaps are unverified -> next is a keepalive thread in BigMoE's loader and an engine A/B in a quiet slot. |
| M-21 | MoonShots (CPU replay; no slot) | BWIDTH [EXACT. The orchestrator 22:5x from Fable A's ASSUMPTION_HUNT (mission 31): consecutive checks' accepted lengths correlate 0.60 on code (P(long | long) 0.74 vs 0.25 after a short one; math 0.24, IFEval 0.34). Not in TRIED (CALSTOP / the t 0.3 cut / AdaEDL size by the drafter's confidence, not by the outcome history). Pre-registered Mon 22:3x, before any run]: a branch predictor for the check width. 'last2' = 8 rows after two short checks, 32 after two long ones, else 16; 'sat2' = a 2-bit saturating counter (start 1; long +1, short -1; 0 -> 8, 1-2 -> 16, 3 -> 32). long = the check verified its full width with no cut and kept every row; short = kept <= width / 4. The t 0.3 cut stays on inside the chosen width. Replay (scripts/bwidth45.py on wide45.py's machinery): Breakthrough's 12 code windows, right / wrong per position from the real uncut walk, MTP Head's per-depth rates where unknown (base16 for 8 / 16-row checks, base32 for 32; variant match16 / match32), 4 seeds (one uniform draw per position, shared by all widths); a myopic oracle (the best width per check in hindsight) as the ceiling. Cost (the orchestrator's): ms per check = M0 + 2.0 ms x verified rows + draft (block 8 / 16 / 32: 5.4 / 6.3 / 8.2 ms), M0 so that fixed 16 + cut = C50's 96.8 ms cycle; beside it the SSD-aware model from M-18c's byte replay (C 256 + M-17 sizes, EXACT). Line (the orchestrator's): pooled code tok/s (kept per check-ms) of the better predictor vs fixed 16 + cut, flat cost model: >= +2 % = PASS, < +0.5 % = KILL, BETWEEN otherwise (the replay stands in for Fable's halved napkin, +1-2 %). **RESULT Mon 22:4x (scripts/bwidth45.py, data/bwidth45.txt; 4 seeds, guarded, peak 198 MB): KILL, borderline** - the better predictor (last2) gains +0.49 % on code (flat model; line < +0.5 % KILL), +0.2 % / +0.4 % SSD-aware (today / NAX6); sat2 -0.6 %. With MATCH's rates (variant) last2 +0.88 % (BETWEEN), sat2 +0.7 %. The room exists but the history doesn't find it: the hindsight oracle (best width per check) is +10.5 % flat / +9.0 % SSD-aware and picks 8 rows on 64 % of checks (last2 on 18 %, sat2 43 %), so the gain is in foreseeing short checks, which the acceptance history barely signals. Fixed 32: +1.4 % flat but -1.9 % SSD-aware. Per window (last2): only cap-bound prompts gain (HumanEval/136 +9.9 %, /135 +8.1 %, /103 +4.2 %), the rest -1.6 .. +0.5 %. |
| M-21b | MoonShots (CPU replay; no slot) | STOPCLS [EXACT. The orchestrator 22:5x after M-21's oracle (+9.0 % SSD-aware, mostly short checks). Published form: DISCO's continue-drafting classifier, SpecDec++'s acceptance head. TRIED: CALSTOP (a calibrated per-depth keep model) was KILL in substance; new here are the history features and tuning on the SSD-aware objective. Pre-registered Mon 22:4x, before any run]: a logistic model of P(draft j accepted | drafts 1..j-1 accepted) on logit cf_j, log cumprod(cf), depth, depth^2, the last two checks' kept, and logit cf_j x depth (the walks carry no top-2 margin). Trained on the real uncut walks of half the windows, 2 folds by window, never by row. The check verifies while the predicted survival S_j >= mu, i.e. expected kept per row-cost; mu is tuned on the train fold (SSD-aware). Widths 1-16 (today's block-16 drafter). Beside it: t* = the t-cut threshold re-tuned on the same folds (to separate retuning from learning) and the hindsight oracle (the ceiling). Harness: bwidth45's (coupled walks, per-domain acceptance rates from the capture's own uncut walks where unknown, C 256 + M-17 sizes byte replay, EXACT). Line (the orchestrator's): pooled code SSD-aware tok/s on the test folds vs fixed 16 + t 0.3: >= +2 % PASS, < +1 % KILL, else BETWEEN; chat (IFEval, multi) reported separately. **RESULT Mon 23:0x (scripts/stop45.py, data/stop45.txt; 2 folds by window, 2 seeds, guarded, peak 277 MB): KILL on code** - the learned stop is -1.5 % (today's kernels) / -1.0 % (NAX6) vs t 0.3; the re-tuned t* -0.2 / -0.4 %; the hindsight oracle +12.3 / +9.4 %. Chat: IFEval learned +2.3 % / 0.0 %, t* (0.4) +2.6 / +0.7 %, oracle +21.7 / +16.0 %; multi learned +2.4 / +0.2 %, t* +1.4 / -0.5 %, oracle +17.1 / +12.3 %. Reading: the history and calibration add nothing beyond the threshold; on chat a shorter cut (t ~0.4) gains ~2 % at today's per-row cost and ~0 once NAX6 makes rows cheaper; the oracle room (where the first wrong draft is) is not visible in cf, depth or the history. |
| M-22 | MoonShots (CPU replay; no slot) | TWOSTAGE [EXACT. The orchestrator 23:0x: Fable C's C1-3 (~/FableGod/fablec/INVENT.md). Prior art per Fable C: staged drafts (Spector & Re), proxy two-pass checks (non-exact); R-32 STOPPRED (a layer-12 probe) KILL; here the 125B's own verdict on the prefix decides the tail. Pre-registered Mon 23:0x, before any run]: verify rows 0..s-1 as a normal check + commit and run the tail s..n-1 from the committed state only if all s are accepted; s = 1 + the first depth where cumprod(cf) < P*, inside today's t 0.3 chain of n rows. Kept is unchanged by construction (a rejected prefix means a rejected tail), so only the cost moves. Exact by REF2-T's batch invariance (Fable C). Replay: M-21b's harness (coupled walks, t 0.3, Breakthrough's 36 windows), the byte replay per pass (LRU + 30 x popularity, strict), C 256 and C 320 with the M-17 per-layer sizes. Cost per pass: fixed F + rows x 0.61 ms (dense 0.264 + GDN / attention 0.35) + the pass's SSD MB at 12 GB/s + its expert union at 174 GB/s, draft 6.3 ms once per check; F = 13.73 ms for the first pass (D32), the second pass costs F2: 'today' F2 34 ms with P* 0.4, '17 ms dense' F2 17 with P* 0.6 (Fable C's numbers). Max form: F + max(SSD, union + rows); add form: F + SSD + union + rows. Line (Fable C's): modelled code cycle (tok/s at equal kept) vs the one-pass check, max form: >= +5 % PASS, < +2 % KILL, else BETWEEN; judged per scenario; chat (IFEval, multi) reported separately. **RESULT Mon 23:1x (scripts/twostage45.py, data/twostage45.txt; 2 seeds, guarded, peak 263 MB): KILL on code in every scenario** - max form -2.6 % (today: F2 34 ms, P* 0.4) / -3.3 % (F2 17, P* 0.6) at C 256, -2.5 / -3.0 % at C 320; add form -1.7 .. -2.4 %. Chat: IFEval +0.2-0.4 %, multi +0.4 % (today) / +1.4 % (17 ms). Inside today's t 0.3 chain the residual tail is short (Fable C's own caveat), so a failed prefix saves little, while code prefixes usually hold and pay the second pass's fixed cost (79 / 221 second passes per ~845 code checks). |
| M-23 | MoonShots (CPU replay; runs after MTP Head's DUALDRAFT writes per-row drafts from both drafters, Tue ~14:15; no slot) | FIRSTMISS [EXACT. The orchestrator (Mon ~23:0x) after M-21 / M-21b / M-22: cf- and history-based width / stop rules sit within +-3 % of t 0.3 on code while hindsight shows +9-12 %, so the missing piece is foresight of the first wrong draft. TRIED: no disagreement-as-stop entry; cloud task 2's committee result bounds the signal (when L1-20 misses, the second drafter also misses 61-66 %). QUEUED Mon 23:0x, pre-registered before any run]: does the two drafters' disagreement at a row, plus the row's token class (newline, identifier start, punctuation, other), predict the first miss better than cf? Model: M-21b's logistic acceptance head plus the disagreement and class features, against its cf-only twin as the control; 2 folds by window; the rule verifies while the predicted survival >= mu, tuned on the train fold. Harness: stop45.py's (coupled walks, C 256 + M-17 sizes byte replay, EXACT). Input: DUALDRAFT's per-row drafts on windows aligned with the capture's (or the capture re-walked on DUALDRAFT's windows). Line (the orchestrator's): code SSD-aware tok/s vs t 0.3 on the test folds: >= +3 % PASS, < +1 % KILL, else BETWEEN; chat reported separately. **Spec update (the orchestrator, Mon 23:3x): use Fable B's pre-registered screen, ~/FableGod/fableb/FIRST_MISS.md section 5** - stop45's acceptance head with the content signals (token class, copy state, 3-gram and unigram novelty, line position, last kept) plus the disagreement flag, against its cf-only twin; same folds and line. Key diagnostic: disagreement must mark >= 30 % of the cf >= 0.99 misses while flagging <= 5 % of those rows. Fable B's own read: content signals lift the row AUC 0.80 -> 0.89 on code but width / cut rules cash only ~+1 %. **PARKED Tue 18:5x (the orchestrator): DUALDRAFT's per-row drafts exist (MTP Head, on disk since 09:28) but DUALDRAFT is a KILL: the twin drafters are wrong together (P(x2 wrong | L1-20 wrong) 0.866 code / 0.892 math), so the twin's flag is weak; the screen moves to M-23b with the native MTP head as the second opinion.** |
| M-23b | MTP Head (CPU replay after ~19:40 on its tree3_map data; the orchestrator Tue 18:5x); MoonShots (line, harness: stop45.py / bwidth45.py read-only) | FIRSTMISS-MTP [EXACT; now a wall 1 (SSD) lever: rejected rows carry 52-59 % of a check's misses (Fable B), so a better cut saves SSD reads. The orchestrator Tue 18:5x: park M-23, use the MTP head as the second opinion (MTP Head 18:51: aligned, q = P(MTP wrong | DFlash2 wrong) ~0.6 at depth 0, vs the twin's 0.866). Pre-registered Tue 18:5x, before any run]: Fable B's screen for M-23 (~/FableGod/fableb/FIRST_MISS.md section 5) with the MTP head in the twin's place. Harness: stop45.py's coupled walks on Breakthrough's 36 capture windows, SSD-aware tok/s at C 256 + M-17 (the byte replay on the union per block, EXACT gate); logistic acceptance heads, 2 folds by window, the rule verifies while the predicted survival >= mu (mu tuned on the train fold). Heads: A = cf + cumprod + depth + the last two kept (M-21b's; the control); B = A + Fable B's content signals (first_miss.py: token class, copy state, 3-gram and unigram novelty, log line position); C_d = B + the MTP head at depths 0..d-1: the flag dfl_own != mtp_t3[..., 0], the MTP top-1 probability mtp_p3[..., 0], and its probability of the drafted token (mtp_p3 where dfl_own is in its top-3, else 0), d = 1 / 2 / 3. Input: ~/TrainLab/data/tree3_map/<set>/<window>.npz (aligned to the map = the capture's windows; teacher-forced = exact up to the first miss). Cost: the MTP chain at ~2.2 ms per step (MTP Head), charged d x 2.2 ms per check on the critical path; a reading with the cost hidden is reported apart, not judged. Line (M-23's): code SSD-aware tok/s, C_d (d chosen on the train fold) vs t 0.3 on the test folds: >= +3 % PASS, < +1 % KILL, else BETWEEN; chat (IFEval + multi) the same lines, reported apart; C_d vs B also reported (what the MTP head adds). Diagnostic (M-23's): on cf >= 0.99 rows, the MTP flag at depth 0 must mark >= 30 % of the first misses while flagging <= 5 % of those rows. Prediction: the diagnostic passes on marks (~40-50 %) and sits near the flag line (~4-6 %); code BETWEEN (+1-3 %) with the 2.2 ms per depth as the main risk; chat BETWEEN-PASS. **RESULT (MTP Head's run, Tue 20:31, ~/TrainLab/data/m23b.txt; filled by MoonShots 22:4x): KILL on both lines with the MTP cost charged: code C_d vs t 0.3 -1.12 % (C_d = C1; B vs t 0.3 +0.63 %, C_d vs B -1.74 %), chat pooled -0.89 % (IFEval -1.7 %, multi -0.1 %). With the 2.2 ms per depth hidden: code +1.3 %, IFEval +3.4 %, multi +5.9 % (C3: +6.1 / +8.6 %). Diagnostic (registered, depth 0): code has 0 first misses among its 262 cf >= 0.99 rows (not testable); multi marks 90 % of 10 at 5.4 % flagged (just over the 5 % line); any depth (information): code 4 / 4 marked at 0.6 % flagged, multi 79 % at 3.2 %. Reading: unlike the twin drafter, the MTP head does see the confident first misses, but they are rare and its 2.2 ms per step costs more than the cut saves; it pays only if its first steps run hidden (beside the draft, or on the ANE), and then mostly on chat.** |
| M-25 | MoonShots (CPU replay; no slot) | SIDECAR SIZES [EXACT, residency only. The orchestrator Mon 23:2x: Fable A, Fable C and cloud task 18 find the rank 0-1 LSB sidecar pool missing ~15.8 times per kept token at every C (~12 MB, a third of the bytes at C 320). Note: M-17's adopted sizes already size the sidecar per layer jointly with the base pool (C 256: C_l average 89 vs uniform 112), so both baselines are reported. Pre-registered Mon 23:2x, before any run]: replay (wide45 / band45 machinery: LRU + 30 x popularity, strict, prompts as uncounted prefill) on Breakthrough's capture, code (12 windows) and chat (IFEval + multi, 24), the union per 16-row block (the uncut walk's checks); per-layer miss curves of both pools over size grids. Arms at C 256 and C 320, base pool held at its baseline: (1) per-layer sidecar sizes at the same sidecar RAM (M-17's Lagrangian on the sidecar curves); (2) the sidecar at 1.5x and 2x slots (+2.07 / +4.14 GB at 112 -> 168 / 224); (3) both (per-layer sizes at 1.5x / 2x RAM); control: the same extra RAM allocated jointly over both pools. Allocation on one half of the windows, scored on the other (2 folds by window). Baselines: uniform (C 256: 287 + 112, C 320: 369 + 112) and M-17's per-layer sizes. Line (the orchestrator's): PASS = arm (1) cuts code SSD MB per kept token >= 5 % vs the uniform baseline at equal RAM; KILL < 2 %; BETWEEN otherwise; MB per kept token (total and sidecar) and each arm's pool GB reported, chat separately. **RESULT Mon 23:3x (scripts/sidecars45.py, data/sidecars45.txt; held-out by window, 110 s, peak 335 MB): arm (1) KILL at C 256 (-1.1 %), BETWEEN at C 320 (-2.1 %)** on code (chat -2.2 / -3.5 %). Code MB per kept token, uniform 53.12 / 26.68 (sidecar 8.70 = a third at C 320): sidecar x1.5 / x2 uniform (+2.07 / +4.14 GB) -7.0 / -10.7 % (C 256), -14.0 / -21.2 % (C 320); per-layer sidecar x1.5 / x2 -8.0 / -11.4 %, -15.9 / -22.7 %; M-17's adopted sizes -12.8 / -14.3 %; a joint re-fit of both pools at equal RAM -14.7 / -18.1 %; joint +2.07 GB -27.6 / -32.2 %, +4.14 GB -38.8 / -43.7 %. Reading: the sidecar is not a sizing problem at equal RAM; extra RAM buys 2-4x more when allocated jointly (a base miss costs 2.16 MB vs 0.77), so the sidecar's share of bytes is a symptom, not the place to spend. Caveat: 16-row blocks (uncut), as asked; the t 0.3 cut shrinks the unions. |
| M-25b | MoonShots (CPU replay + size files for BigMoE; no slot) | JOINT SIZES [EXACT, residency only. The orchestrator Mon 23:3x after M-25 (spend RAM jointly over both pools). Pre-registered Mon 23:3x, before any run]: per-layer (C_m, C_l) sizes from M-17's joint Lagrangian (+ greedy fill, FLOOR C_m >= 160, C_l >= 32, P30 pools) at C 256 and C 320, at equal RAM and +2.07 / +4.14 GB. Fits: (i) cap45 only (M-17's data: at equal RAM this reproduces M-17, an identity check; the extra-RAM sizes are new) and (ii) cap45 + the capture's training windows (code + chat, pooled bytes per pooled kept token), scored on the capture's held-out windows (2 folds by window, 16-row blocks, as M-25). Files: data/joint45_<C>_<extra>.json in M-17's BIGMOE_TIER_SIZES format, fitted on cap45 + all capture windows. Line (the orchestrator's): PASS if fit (ii) beats M-17's adopted sizes by >= 2 % code SSD MB per kept token at equal RAM on the held-out windows; KILL < 1 %; BETWEEN otherwise; chat and the extra-RAM gains reported. **RESULT Mon 23:4x (scripts/joint45.py, data/joint45.txt, curves data/joint45_curves.npz; 322 s, guarded, peak 405 MB)**: identity check passes (fit (i) at equal RAM = M-17's sizes on 48 / 48 layers at both C). Fit (ii) vs M-17 at equal RAM, code held-out: **C 256 -1.4 % (BETWEEN), C 320 -2.8 % (PASS)**; IFEval / multi -2.0 to -2.3 % (C 256), -1.4 to -2.0 % (C 320). Extra RAM vs M-17: +2.07 GB -16.1 % (C 256) / -20.0 % (C 320), +4.14 GB -29.3 / -33.8 % (fit (ii); cap45-only fits a point or two less). Files for BigMoE (M-17's format, BIGMOE_TIER_SIZES; fitted on cap45 + all 36 windows): data/joint45_{256,320}_{0,2070,4140}.json (pools 33.93 / 36.00 / 38.07 GB at C 256, 42.45 / 44.51 / 46.58 GB at C 320). |
| M-26 | MoonShots (CPU replay; no slot) | SIDECARDROP [bytes side of the 45 GB shrink path; quality is Breakthrough's t3_all (Tue). The orchestrator Mon 23:4x. Pre-registered Mon 23:4x, before any run]: plan 3333333333 (all ranks 3-bit, no LSB sidecar pool, no LSB reads); the whole pool RAM (uniform C 256: 33.93 GB, C 320: 42.45 GB) goes to the base pools with a base-only per-layer fit (M-17's Lagrangian, FLOOR C_m >= 160, P30) on cap45 + the capture's training windows. Compared with M-17's sizes and the M-25b joint fit (both on plan 4433333333 with the sidecar) at C 256 and C 320, code and chat, SSD MB per kept token, without Q1 and with Q1 (top-2 on rows with DFlash2 cf >= 0.99; MSB only under 3333333333). Harness: M-25b's (capture held-out by window, 2 folds; 16-row blocks of the uncut walk), plus the same on the capture's cut rows (the engine's t 0.3 checks) reported beside it. Files: data/sidedrop45_<C>.json (C_l = 0). Line (the orchestrator's): PASS = the 3333333333 arm <= 33 MB per kept token on code at C 256 (the 45 GB line) with Q1 on the 16-row harness; KILL > 40; BETWEEN otherwise. **RESULT Tue 00:0x (scripts/sidedrop45.py, data/sidedrop45.txt; 19 min, guarded, peak 372 MB): PASS** - SIDECARDROP with Q1 at C 256: 11.78 MB per kept token on code (16-row harness; cut rows 11.85). Code, C 256: no Q1 M-17 46.31 / joint 45.67 / SIDECARDROP 24.46 (-47 %); Q1 27.83 / 27.15 / 11.78 (-58 %). C 320: no Q1 22.85 / 22.22 / 8.13; Q1 14.96 / 14.14 / 4.62. Chat (IFEval / multi) at C 256: no Q1 30.2 -> 17.0 / 18.1; Q1 27.0-27.7 -> 14.5 / 16.3. Why: no LSB reads at all (the sidecar was 23 % of M-17's bytes) and its RAM (3.3-4.1 GB) moves to the base pools (C_m sum 14,168 -> 15,690 at C 256). Implied steady (C50's six prompts, SSD fully exposed, the capture's code ratio on math too): 63.6 -> 75.8 tok/s without Q1, 85.3 with Q1 (Q1 alone 73.5). NOT exact: ranks 0-1 at 3 bits; quality = Breakthrough's t3_all (Tue). Files: data/sidedrop45_{256,320}.json (C_l = 0; plan 3333333333). Quality so far (Breakthrough's t3_all map, Tue lounge, SCOREBOARD 8b3d8e5): code BETWEEN (+0.59 pts over the t4 control; sure rows 0 / 2922 flips), chat +4-5 pts (mid rows flip ~5 % vs 0.4 %); EvalSuite's same-scores (IFEval first) decides. |
| M-27 | MoonShots (CPU measurement + pricing; no slot) | LOSSLESS45 [BREAKTHROUGH PUSH bet (Kamen-approved push, the orchestrator Tue 17:0x): lossless compression of expert bytes, EXACT. TRIED (mostly mine, 10-18..10-27): W24-floor (3-bit copy floor 2.98 of 3.5 b / weight, -15 % MSB), W24-sp (11-bit lossless scales, -10.9 % at 24 GB), R-16 W24-ec (GPU rANS 46 GB/s of codes, 2.72 b; KILL at 24 GB for GPU contention), C1 (net <= +6 % after decode), R-23 (cross-expert context 0.0035 b), LSB planes uniform. New here: the 45 GB line's price (C 256 + M-17 sizes, the sidecar ~23 % of bytes), a per-layer sample over all 48 layers, an explicit shared-base / delta test, and the decode-time balance. Pre-registered Tue 17:0x, before the run]: sample 4 experts x 48 layers of records3 (both parts): per layer the order-0 entropy and Huffman length of the 3-bit codes, the LSB planes' entropy, the scales / biases' 16-bit entropy; shared base + delta = H(codes of expert e | codes of expert e' at the same position) and H(e - e' mod 8) for every pair in a layer vs H0. Formats priced on band45's capture reads at C 256 (M-18: code 142.2 base + 118.2 sidecar reads per check): today; fixed-width (11-bit scales, cheap decode); entropy floor (rANS codes at H0 + 0.016 b, entropy-coded scales); 16 KB-aligned parts. Lines: bytes - PASS if the entropy-coded store saves >= 12 % SSD MB per kept token on code at C 256, KILL < 8 % (the orchestrator's suggestion), BETWEEN otherwise; time (decides whether bytes convert) - the GPU rANS decode of a check's missed codes at the measured 46 GB/s must take less than the SSD time it saves at 12 GB/s, else flagged 'no net gain' (CPU decode reported as a judgment). **RESULT Tue 17:1x (scripts/lossless45.py, data/lossless45.txt; 192 records over all 48 layers, 25 s): bytes PASS (13.1 % code, 12.8 / 13.1 % chat, entropy store), time NO NET GAIN** - per code check 52.0 MB fewer SSD bytes = 4.33 ms at 12 GB/s, but the GPU rANS decode of the missed codes takes 5.20 ms at the measured 46 GB/s (net -0.87 ms). Measured: 3-bit codes H0 2.724 b (per layer 2.700-2.731), Huffman 2.765, rANS ~2.740 of 3; the LSB plane 1.00000 b (incompressible); scales / biases ~8 of 16 b (11-bit fixed width lossless); shared base + delta: none (H0 - H(e | e') -0.0003 b over 288 pairs; deltas cost +0.229 b). Fixed-width 11-bit scales alone: -5.0 %. **EXPLORATORY LEAD (not pre-registered)**: the fixed-width form held packed in RAM as well (bit unpacking inside the dequant kernels, no separate decode) fits ~4.6 % more parts in the same pools: held-out code -17.6 % vs M-17 at C 256 (-24.5 % at C 320), chat -14 to -16 %; exact; implied 63.6 -> ~66.9 tok/s. R-16 folded this pack into rANS (killed for decode contention), so it was never built alone. Needs Leap's kernels (11-bit scale reads) and a converted store (BigMoE; ~63 GB vs the disk floor). **Tue 17:1x, NewIdeas' pipelining question** (decode each landed record in the gate-spin shadow, only the last-landing one exposed, ~2.3 ms per check): on the entropy store alone vs today that would net ~+2 ms per check, but (1) stacked on the 11-bit pack the rANS layer adds only 38.16 -> 34.93 MB per kept token (-24.6 % vs M-17 in total), ~2.2 ms of SSD per check, against ~2.3 ms of exposed decode -> ~0; (2) R-16's live b42 measured a concurrent GPU decode costing the check +9-12 ms per pass at 6 GB/s (24 GB) - the shadow is not free (it contends with the OOO compute of resident experts); at ~2 GB/s here that is a few ms. So the entropy store stays BETWEEN; worth a live gate-pipeline micro-test only after M-27b. |
| M-27b | Leap (kernels) + BigMoE (store, live A/B); MoonShots (sizes, line) | PACKED11 [EXACT. From M-27's exploratory lead; the orchestrator Tue 17:0x: pre-register before any build. Pre-registered Tue 17:0x, before any build or run]: records3's s3 / b3 (MSB parts) and b (LSB parts) stored as 11-bit fixed-width fields (W24-sp: all 221,184 pieces round-trip bitwise) in the SSD store AND held packed in the RAM pools; Leap's dequant kernels (3c-N / NAX6-N) unpack them in place; ONE in-place conversion with the native layout (BigMoE plans; disk floor 100 GB). Parts 2,064,384 / 720,896 B (16 KB-aligned); pool sizes re-fitted at equal RAM (M-25b's joint fit with the packed part sizes; MoonShots supplies the files). Replay prediction (M-27, held-out): code SSD MB per kept token -17.6 % at C 256 (-24.5 % at C 320), chat -14..-16 %; implied C50 steady +5 %. Gate before the A/B: a kernel unit test - packed-scale dequant gives the bf16-scale path's weights bitwise on every piece of a full layer. Live A/B (BigMoE, C50-style: cap45's six prompts at C 256, today's default = records3 + M-17 sizes vs packed store + packed pools + re-fitted sizes, same RAM, separate passes per arm, quiet lock): PASS = every token bitwise identical AND code SSD MB per kept token <= -10 % AND the six-prompt median steady tok/s >= +3 %; KILL = any token differs, or MB per kept token > -5 %, or median steady < +1 %; BETWEEN otherwise; C 320 reported if run. Size files ready (Tue 17:1x): ~/MoonShots/data/packed45_{256,320}.json (C 256: C_m sum 14,752, C_l 4,824 at 33.93 GB; C 320: 17,984 / 7,376 at 42.44 GB; key part_bytes). **Tue 17:1x: Kamen approved the combined conversion (native layout + 11-bit packed scales, one in-place change); M-27b's live A/B = BigMoE's pre-registered before / after A/B, reported against both lines; packed45 files feed the after-arm.** C 224 file for BigMoE's A/B (Tue 17:3x, BigMoE's ask: the night's memory likely limits the A/B to C 224): data/packed45_224.json at laysz45_224.json's pool RAM (29.677 GB; C_m sum 12,976 vs 12,466, C_l 4,008 vs 3,528), md5 11309e04c5f9e901916542aa81047ae0; held-out prediction vs M-17 at C 224: code -15.0 %, chat -14.2 / -14.6 %. |
| M-28 | MoonShots (SSD + CPU timing, ~1 min, quiet lock; by day Wed 15:00-15:30 if BigMoE doesn't need it, else the orchestrator's slot; numbers to BigMoE, the lead) | SSDMICRO [WALL 1 sprint lever 1 (kill the per-call overhead), EXACT: reads only. The orchestrator Tue 18:3x, charter ~/FableGod/push/WALL1_SPRINT.md: find what makes the ~0.28 ms fixed cost per loader call (SSD_WALL2's fit at C 256: SSD busy = 0.283 ms per call + MB / 12.7 GB/s; ~70-77 calls per check = ~20 ms). TRIED: M-20 SSDWAKE (+50-65 us after 0.25-1 ms gaps; a 4 KB / 50 us keepalive removes it), TABLE_350 B1 (QD 64 rates by size; QD 1 16 KB 82 us, 1 MB 217, 2.9 MB 331), REC1 / REC2 (one aligned read per record; unaligned F_NOCACHE destinations slow), 9 pieces vs one whole read (~0.28 vs ~0.40 ms at QD 1, 24 GB era). New here: the per-call cost taken apart at the loader's real call shape (1-3 records per call at C 256) with per-read timestamps (GCD start, read, return), every factor in one interleaved run. Pre-registered Tue 18:5x, before any timing]: harness ~/MoonShots/scripts/ssdmicro.c (C; BigMoE's load_jobs pattern: one page-aligned F_NOCACHE pread per record part, one dispatch_apply on the user-interactive queue; records3 read-only and never parsed, every read on bytes no other read of the run touches and that are not resident in the page cache (mincore)) + ssdmicro_sum.py, runner run_ssdmicro.sh. R0 = the median C 256 call: 2 base records (2 x 2,162,688 B) as 2 whole reads after a 0.8 ms idle gap. Fixed cost = call time - bytes at 12.7 GB/s. Sets (100 rounds; all 81 points interleaved per round in blocks by keepalive setting; medians): S1 lone reads 4 KB-4 MB at QD 1 (first-byte intercept; gaps 0 / 0.8 ms); S2 queue depth: k parallel 256 KB reads, k 1-64; S3 one record in k = 1-32 pieces (2.16 MB-68 KB), 2 records at k 1 / 4 / 8; S4 idle gaps 0-10 ms (R0, R0 in 8 pieces); S5 keepalive spacing (BigMoE's 4 KB pread + usleep(n) loop, n 25-500 us); S6 F_NOCACHE vs the page cache; S7 GCD vs 8 pre-spun threads vs serial; S8 JOINT (base call then sidecar call vs one call; whole and 8 + 3 pieces). Lines: L1 = Fable A's D1 screen (SSD_WALL2 section 5): pieces (1 record whole - in 8 pieces) PASS >= 0.10 ms, KILL < 0.04; joint (two whole calls - one 8 + 3 call) PASS >= 0.20, KILL < 0.10; PASS = both pass, KILL = both kill, else BETWEEN. L2 = the orchestrator's question: R0's fixed cost >= 0.20 ms -> the drive + dispatch path makes most of the 0.28 (keepalive / pieces / pre-spun threads act there); <= 0.10 -> most of it sits outside the I/O path (release pickup, rebuild, call overhead) and BigMoE's per-burst trace must find it; between: split. L3 = lever 1's target (<= 5 ms per check at ~45 calls after JOINT): the best measured 2-record call at gap 0.8 (any dispatch / pieces / keepalive / cache) has fixed cost PASS <= 0.11 ms, KILL > 0.20. Descriptive: first byte, QD curve, best piece size, the cheapest keepalive spacing whose 0.8 ms penalty is <= 10 us, F_NOCACHE delta, GCD wake. My prediction: R0 ~0.50 ms, fixed ~0.15-0.20 (the I/O path makes 55-70 % of the 0.28); pieces -0.05..-0.10; joint -0.15..-0.25. Runtime ~1 min (no new round after 12 min), ~30 GB read; up to 8 spinning cores only around S7's calls. **RESULT Tue 18:59 (run moved up by Kamen's 'small tests now'; quiet lock 18:58:49-18:59:07, Kamen on the Mac, CPU 83-86 % idle; 100 rounds in 13.7 s; data/ssdmicro.txt): L1 KILL: pieces make a lone record SLOWER (+39 us in 8 pieces, +17..+93 at 2-32; 2 records +39 / +49 at 4 / 8); joint in 8 + 3 pieces -76 us (< 100), but joint with whole reads -120 us (1+1), -155 (2+2). L2 split: R0 527.5 us, fixed 187 us = 66 % of the engine's 283 (first byte ~119, idle wake +44, GCD start + return 8, QD-2 sharing ~16); ~96 us per call sits outside the I/O path (last record's rebuild, release pickup). L3 BETWEEN: best call = whole reads + keepalive, fixed 130 us (line 110 / 200); its floor is the first byte. Keepalive: usleep(25-100) = real spacing 145-270 us removes the whole wake; 400+ us leaves half. Page cache +49 us per call; pre-spun threads no gain. Per check at C 256: keepalive -3.4 ms busy, JOINT -4..5 ms; don't build pieces or pre-spun threads.** |
| M-29 | MoonShots (SSD + CPU timing, ~20 s, quiet lock; tonight if the SSD is quiet before the 00:30 runs, else Wed by day; numbers to BigMoE, the lead) | SSDMICRO2 [WALL 1 sprint lever 1, EXACT: reads and copies only. The orchestrator's brief Tue 22:3x: 'reading straight into aligned slot memory (the zero-copy candidate), the copy-tail cost, keepalive spacing variants'. From M-28: of the median call's ~0.28 ms fixed cost, ~0.19 is the I/O path and ~0.10 sits outside it. The tier loader (bigmoe_ext.cpp read_expert, rec_fmt 1 on a record part) reads each missed part into a pooled aligned bounce, then memcpys its 9 (base) / 6 (sidecar) pieces into the pool slots on the job's thread. TRIED: REC1 (preadv straight into the slots) KILL for unaligned F_NOCACHE destinations (records3's s / b pieces are 51,200 B = 12.5 x 4 KB; engine pool strides leave slots 2 KB-aligned). New here: aligned zero-copy (pieces padded to 4 KB inside the same 2,162,688 B part, or one contiguous slot), Metal pool memory as the destination, and a wake-ahead ping instead of a keepalive trickle. Pre-registered Tue 22:3x, before any timing]: harness ~/MoonShots/scripts/ssdmicro2.m (C + Metal; M-28's method: one GCD dispatch_apply per call, F_NOCACHE, fresh records, none resident in the page cache, 0.8 ms busy-waited idle gap, all 50 points interleaved per round in blocks by keepalive setting, 150 rounds, medians) + ssdmicro2_sum.py, runner run_ssdmicro2.sh. Sets: Z zero-copy with 1 and 2 base records per call (Z0 today = bounce + memcpy into engine-stride slots; Z0n no copy; Z0p copy split over 3 projections; Z1 one read into a contiguous 16 KB-aligned slot; Z2 REC1; Z3 4 KB-padded pieces into 16 KB-aligned slots; Z3b the same into 4 KB- but never 16 KB-aligned slots; Z4 16 KB-padded pieces (2,260,992 B); Z0 / Z1 / Z3 / Z3b also into Metal shared buffers); S the sidecar part (today, no copy, contiguous, 4 KB-padded into 16 / 4 KB-aligned slots); K keepalive spacing usleep(n) n = 50 / 100 / 125 / 150 / 175 / 200 with 4 KB reads, n = 200 with 16 / 64 KB reads; P wake-ahead ping: one 4 KB read T = 25 / 50 / 100 / 200 / 400 us before the call, no trickle; C keepalive 50 with today's copy and with Z3b. Lines: Z = the best Metal zero-copy arm (Z1m / Z3m / Z3bm) vs today's Z0m on 2-record calls: PASS = saves >= 50 us per call AND within 15 us of Z0n (no DMA penalty for reading into slot memory); KILL = saves < 20 us; else BETWEEN; with it whether 4 KB-aligned slots match 16 KB-aligned ones (within 15 us: then the pool padding costs ~0.15 GB at 45 GB instead of ~1 GB). P = some T <= 200 us removes >= 70 % of the 0.8 ms wake penalty: PASS (the engine can ping at the router's launch instead of a trickle); no T removes >= 30 %: KILL; else BETWEEN. Descriptive: the copy tail per base / sidecar record against M-28's ~96 us outside the I/O path; the widest keepalive spacing with penalty <= 10 us; the read-size effect. My prediction: copy tail 50-100 us per base record, ~25 per sidecar; Z PASS (60 %), 4 KB alignment enough (50 %), REC1 slow; P PASS at T 100-200 (50 %). Runtime ~20 s, ~1.1 GB of pre-faulted slot memory, < 2 cores. **RESULT Tue 22:33 (quiet lock, 150 rounds in 12.3 s, data/ssdmicro2.txt): Z PASS: zero-copy into 4 KB-aligned Metal slots (s / b pieces padded to 53,248 B inside the same part) saves 173 us per 2-record call (716 -> 543 us), +5 us vs no copy at all; 4 KB = 16 KB alignment (-5 us), so the padding costs ~0.18 GB at C 256, not ~1 GB; Metal = malloc. Copy tail 119-122 us per base record alone, 170-176 with two side by side, 81 split over 3 projections; sidecar 44.5. REC1 +843 us. K: widest keepalive spacing with penalty <= 10 us ~340 us real (usleep 150), half usleep 50's reads. P PASS: one 4 KB ping 200 us before the burst removes 97.5 % of the +48 us wake (T 100: 22 %, T 400: 78 %). C: keepalive + zero-copy 487 vs 703 us (-31 %). Per check after JOINT: zero-copy -5.5..-8 ms SSD busy bound (Fable A budgeted -2.5).** **ADOPTED into the conversion (BigMoE 22:4x): records3's conversion format is now 'native+p11+pad' (MSB s3 / b3 padded to 36,864 B, LSB stored l1-first at 0 / 204,800 / 409,600 then the three b pieces packed, lsb_sz unchanged, the b copy ~10 us); Leap moves the p11 kernels' stride to the pool shape; BigMoE builds the direct path (aligned pieces straight into record-order pools); the A/B night gains a B2 + zero-copy arm.** **Wed 09:01 (batched ticket): the wake-ahead ping at the engine's gaps: after 1.2 / 2.0 ms idle (+50 / +49 us), a 4 KB ping 150 us ahead removes 87 / 94 %, 250 us ahead 91 / 87 %.** |
| M-29b | MoonShots (SSD + CPU timing, ~10 s, quiet lock; tonight after M-31) | ZEROCOPY-24 [WALL 1, the 24 GB line; EXACT. From M-29's zero-copy PASS at 1-2 records per call (the 45 GB line's bursts): at the 24 GB line a loader call carries ~4-15 misses, so many copies run side by side (memory-bandwidth contention could grow the tail) and the reads run at queue depth up to 15 (does reading into slot memory stay penalty-free?). Pre-registered Tue 22:4x, before any timing]: ssdmicro2.m in its 'big' mode (M-29's method; 150 rounds): Z0m (bounce + copy into Metal pool slots, today), Z0pm (the copy split over 3 projections), Z0n (no copy), Z3bm (zero-copy: 4 KB-padded pieces into 4 KB-aligned Metal slots) at 4, 8 and 15 base records per call; summary ssdmicro2b_sum.py appended to data/ssdmicro2.txt. Line: Z3bm saves >= 50 us per call vs Z0m at both 8 and 15 records: PASS; < 20 us at both: KILL; else BETWEEN. My prediction: PASS (the tail stays one or two records' copies, ~120-200 us). **RESULT Tue 22:46 (quiet lock, 150 rounds, data/ssdmicro2.txt): BETWEEN. Zero-copy saves 21.5 / 8.1 / 60.2 us per call at 4 / 8 / 15 records: the exposed copy tail shrinks to 45-78 us (earlier copies overlap later reads) and 9-piece reads into the slots cost 34-59 us of bandwidth at QD 4-15. A 45 GB-line lever (-173 us at 1-2 records), about neutral at 24 GB; an exploratory contiguous arm is queued.** **Repeat 22:47 (with an exploratory contiguous arm; the verdict stays the registered 22:46 run's): zero-copy saves 45.4 / 61.4 / 157.5 us at 4 / 8 / 15 records (would PASS): the copy tail swings 45-78 us (22:46) vs 61-208 us (22:47) at the same queue depths, so at 24 GB zero-copy never loses but its size is unsettled (-8..-158 us per call). The contiguous one-read-per-record slot is worse at high QD (-23 us at 15 records vs -158 for 9 aligned pieces): prefer aligned pieces.** **Wed 09:01 third run (250 rounds, the batched ticket): zero-copy saves 45.6 / 29.2 / 43.0 us at 4 / 8 / 15 records; across the three runs ~30-60 us per call typical (158 once): the 24 GB value is ~-30..-60 us per call.** |
| M-30 | MoonShots (SSD + CPU timing, ~10 s, quiet lock; tonight if the SSD is quiet before the 00:30 runs; numbers to BigMoE and Fable A for lever 2's design) | QUEUEMIX [WALL 1 sprint lever 2 (never let the queue empty), EXACT: reads only. Lever 2 (SSD_WALL2 design 2) issues layer L+1's forecast reads with layer L's burst so the drive never idles and the next demand reads skip the first byte (~119 us) and the idle wake (~48 us); the same forecast reads share the drive's bandwidth with the demand reads beside them. TRIED / specs: no measurement of demand reads issued into a busy drive; no I/O-priority work (setiopolicy_np absent from TRIED, PRIOR_ART, SSD_WALL2, WALL1_D1_SPEC). Pre-registered Tue 22:4x, before any timing]: harness ~/MoonShots/scripts/ssdmicro3.c (+ ssdmicro3_sum.py, run_ssdmicro3.sh; M-28's method: fresh records, none resident in the page cache, F_NOCACHE, 0.8 ms busy-waited idle gap, points interleaved, 150 rounds, medians). Demand = 2 base records by one GCD dispatch_apply (the loader's call). Forecast = F = 1 / 2 / 4 records from F pre-spun threads: M mixed (issued with the demand call; default, IOPOL_UTILITY and IOPOL_THROTTLE), B busy (issued Delta = 100 / 200 / 400 / 800 us before it; default; throttle at 200 / 400), C chained (issued when the demand call returns); A = the demand call alone after the idle gap, A0 back to back. Lines: L-busy = some busy or mixed arm's demand call >= 100 us faster than A: PASS (a busy drive saves the first byte + wake net of the shared bandwidth); every busy and mixed arm slower than A: KILL (forecast reads in flight only cost the demand; lever 2's timing gain then comes only from forecasts that finish before the demand arrives); else BETWEEN. L-priority = a throttle / utility arm's demand call >= 50 us faster than the same arm at default AND its forecasts end <= 1 ms after the demand call: PASS; forecasts > 2 ms late or demand < 20 us faster: KILL; else BETWEEN. Descriptive: the demand cost per forecast record in flight, the chained forecasts' read time (does a read issued into a just-busy drive skip the first byte?). My prediction: L-busy BETWEEN / KILL (60 %: forecasts still transferring cost more than the first byte saves, except at Delta 400-800 where they are nearly done); L-priority KILL (75 %: the kernel's throttle delays reads by milliseconds). Runtime ~10 s, < 6 cores (4 spinning forecast threads only during their points). **RESULT Tue 22:39 (quiet lock, 150 rounds in 6.8 s, data/ssdmicro3.txt): L-busy BETWEEN (best -60 us vs 531 alone after idle, line 100): a busy drive never saves the demand's first byte, only the ~50 us wake; forecasts mixed in cost the demand +165 / +320 / +632 us at F 1 / 2 / 4 (their byte share), issued 100-200 us earlier still +121..+219 (F 2); chained (issued when the demand returns) cost nothing and finish 474 / 796 us later. L-priority BETWEEN: IOPOL_THROTTLE / UTILITY forecast threads shield the demand completely (mixed F 2: 535 vs 851 us) but the kernel holds them 1.6-2.5 ms. For design 2: chain forecasts after the demand burst, never mix them in; throttled threads only for forecasts >= 2 layers ahead.** |
| M-31 | MoonShots (SSD + CPU timing, ~10 s, quiet lock; tonight; numbers to BigMoE) | HEADPING [WALL 1, the floor under lever 1, EXACT: reads only. After JOINT, keepalive / ping and zero-copy, a burst still pays the drive's first byte (~119 us, M-28), and a busy drive does not hide it (M-30). Unusual place: is part of the first byte a cold lookup inside the drive (the FTL's logical-to-physical map, a NAND page not yet sensed) that a tiny read can warm? If so, the loader pings the heads of every forecast expert (4 KB each, no bandwidth wasted on wrong forecasts) and the right ones start faster. TRIED / PRIOR_ART / DEAD: no head ping, FTL or page-register entry. Pre-registered Tue 22:4x, before any timing]: harness ~/MoonShots/scripts/ssdmicro4.c (+ ssdmicro4_sum.py, run_ssdmicro4.sh; fresh records, none resident in the page cache, F_NOCACHE, 0.8 ms busy-waited gap, 150 rounds, medians). Probe: one full base part (2,162,688 B) read in-thread at QD 1 with BigMoE's keepalive running (usleep 50) so the idle wake stays out. H0 no ping; H1 4 KB of the record's head pinged Delta = 100 / 300 / 1,000 / 5,000 us before the read (from the ping's end); H2 the part's last 4 KB; H3 the next record's head; H4 a far record's head; H5 64 KB of the head; G0 / G1 the loader's 2-record GCD call without / with both heads pinged (300 us); H0 / H1 (100, 300) also with the keepalive off. Line: H1 at Delta 100 / 300 / 1,000 cuts the full read by >= 30 us vs H0: PASS; < 10 us at all three: KILL; else BETWEEN. H2-H5 tell the mechanism (map cache if H2 / H3 help like H1; NAND page if only H1 / H5). My prediction: KILL (55 %); the first byte is sensing + controller time with no cache to warm. **RESULT Tue 22:45 (quiet lock, 150 rounds in 4.5 s, data/ssdmicro4.txt): KILL. Keepalive on, a 4 KB head ping moves the later full read (296 us, QD 1) by +0.6..+3.2 us at Delta 100 us - 5 ms; the part's tail, the next record's head, a far head and 64 KB do no better (-2.1..+2.7); the 2-record call +3.2. The first byte has no warmable map or page cache; keepalive off, a ping 100 us ahead saves 42 us = the wake (M-29's ping).** |
| M-32 | MoonShots (CPU replay; Wed, once BigMoE's pre-route capture (pr_224 / pr_160) gives per-layer forecast recall, and only if lever 2 goes ahead; no slot) | FCASTSIZE [WALL 1 lever 2 x lever 3, EXACT: residency only. From my papers sweep (research/sweep_1103/WALL1_SSD.md item 4; Fate arXiv 2502.12224's shallow-favoring cache): with a layer-ahead forecast, a miss the forecast catches is read early and hidden, so pool slots are worth more in the layers the forecast misses. TRIED: 'Concave stall-time objective for slot allocation' (KILL) priced misses, not forecast coverage; M-17 / M-25b fit misses only. Pre-registered Tue 23:0x, before any run]: re-fit M-25b's joint per-layer (C_m, C_l) sizes with each layer's misses weighted by (1 - that layer's forecast recall), at equal RAM to the current joint sizes (C 224 and C 256), held out by window as M-25b; score on held-out windows by the forecast-missed demand MB per kept token, priced in exposed ms per check with M-28 / M-29's burst costs (first byte 119 us + bytes at 12.2 GB/s per burst, the 0.5 ms cover). Line (the sweep's): >= 1.0 ms less exposed per check at equal RAM vs the current joint sizes: PASS; < 0.3 ms: KILL; else BETWEEN. My prediction: BETWEEN (0.3-1.0 ms). **RESULT Wed 09:02 (pulled forward by the orchestrator; 0 s replay, data/fcastsize.txt): KILL. Re-fit to the corrected pre-route's per-layer recall (0.68-0.86, nearly flat), the sizes cut forecast-missed demand by only -0.01..-0.08 ms exposed per check at C 224 / 256 and add 1.9 % to all demand on code. Keep M-25b's joint sizes. Lever 2 by-product: the forecast leaves 13.3 of code's 45.7 MB per kept token unnamed at C 256.** |
| M-33 | MoonShots (CPU replay, minutes; no slot) | COPYROUTE [WALL 1 lever 2 (forecast), EXACT: timing only. From my papers sweep (research/sweep_1103/WALL1_SSD.md, 'a gap'): no prior art for position-level routing replay (ExpertFlow / DoMoE / MoE-Infinity match tokens, semantics or traces); our files have the prompt footprint as a set (COLD-CUT) and copy-span drafting (L1-10, MATCH), not routing replay. Pre-registered Tue 23:0x, before any run]: for an answer row t whose last 8 tokens match an earlier span of the window (prompt or earlier answer) ending at s, forecast row t's experts at every layer with the routing recorded at row s; it exists before the check starts and covers all 48 layers at once. Data: Breakthrough's 36 map windows (~/FableGod/decision_map/out/moonshots, ids [T-1, 48, 10] for prompt and answer rows). Per domain: coverage (matched answer rows / answer rows), recall@10 per layer averaged over layers 4-47, split by match into the prompt vs the earlier answer; baselines on the same rows: the previous row's routing and a random row of the window. Line (adapted from the sweep's, which needs agent captures we lack): code recall >= 0.80 over layers 4-47 AND >= 15 % of code answer rows matched: PASS (then an agent capture with prefill routing decides its size); recall < 0.65 or < 5 % matched: KILL; else BETWEEN. My prediction: coverage 15-30 % on code, < 10 % on chat; recall 0.6-0.8 (routing follows the local context but also the position and the whole prefix). **RESULT Tue 23:02 (1 s, data/copyroute.txt): KILL. Code: 62 % of answer rows matched (nearly all into the prompt) but recall@10 over layers 4-47 is 0.339 (previous row 0.216, random 0.118); by depth 0.88 (L0), 0.59 (L6), ~0.4 (L12-24), ~0.2 (L36-42): early layers route by the token, deep ones by the prefix. Chat 0.6 % matched. Leftover: layers 0-6 as a layer-0 forecast.** |
| M-34 | MoonShots (CPU replay, light, beside FLOOR; no slot); quality screen = Breakthrough's map if NewIdeas routes it | SURE2B [BYTES10, bytes per expert (the orchestrator Tue 23:5x: MoonShots owns bytes / expert); TEN_X_LEDGER lever 6 '2-bit sure-row records x0.67 per sure-row miss [P], unmeasured'. FAST (lossy on sure rows only; Kamen's decision 2 allows cache-state FAST modes). TRIED: records4 2+1+1 (P2 + B3 + B4 planes) KILL / PARKED for 2-bit-heavy rank plans on ALL rows (per-part prefill KL 2.4-3.9x the target; +46-49 % more separate reads); MSB-plane 2-bit on all rows ppl +11.3 % (35B). New here: 2 bits only on drafter-sure rows, laid out so a 2-bit read is a PREFIX of the 3-bit record [P2 codes | scales | B3 plane] (one read either way; no extra calls). Pre-registered Wed 05:5x, before any run]: replay = NewIdeas' ten_x_ledger.py part C, reused read-only (36 map windows, HPD-0 live checks at t 0.3, plain LRU per layer, prompt warm-up). Base = QOM + 3-bit sure rows (sure rows load ranks 0-1, no LSB) at cf 0 and cf 0.5. Variant: a sure row's base miss reads the 2-bit prefix (codes 1,228,800 + bf16 scales 307,200 = 1,540,096 B padded, x0.712 of the 2,162,688 B part); an unsure row needing an expert held at 2 bits reads the B3 plane (622,592 B padded); slots fixed-size (no capacity gain; a byte-budget LRU variant reported as the upper bound). Pools 24 GB (147 + 46) and 12 GB (51 + 32). Line (bytes side): code SSD MB per kept token at 24 GB, cf 0, fixed slots, vs the base: <= x0.80 PASS, > x0.92 KILL, else BETWEEN; IFEval / multi and 12 GB reported. Quality side (not mine; proposed to NewIdeas for Breakthrough's map): the 2-bit form w = s3 * (2 * (q3 >> 1) + 0.5) + b3 on sure rows' ranks 0-1, the map's flip lines as for '3-bit sure rows'. My prediction: x0.80-0.85 on code (upgrade reads eat part of the x0.712), BETWEEN. **RESULT Wed 05:49 (16 s, data/sure2b.txt): KILL by the line (code 24 GB cf 0 x0.921; IFEval x0.98, multi x0.985): at cf 0 the bonus row's 10 ranks + LSB dominate, and 28 % of the 2-bit loads are later upgraded. Exploratory on top of ledger lever 1 (gated bonus row): every read becomes a 2-bit prefix, x0.712 with fixed slots (24 GB: code 18.0, IFEval 13.7, multi 9.9 MB per kept token), x0.63-0.65 with variable-size slots (16.3 / 12.3 / 8.7). Conditional on lever 1's screen and a 2-bit quality screen.** **Engine convention (NewIdeas 05:51: lever 1 as the map tested it, qom_0_live_bq; 05:52): 24 GB code 35.7 -> 29.7 (x0.833 fixed slots; x0.777 variable), IFEval 19.4 -> 13.9 (x0.715), multi 13.9 -> 9.9 (x0.712); 12 GB code x0.813, chat x0.712-0.714. Screen helper scripts/to2bit.py (self-test OK on 2 experts: 3-bit unpack bit-exact to mx.dequantize; 2-bit triple within 0.4 % of the rule; gather_qmm bits=2 within 0.7 % of dense; the 2-bit weights are 31 % relative RMS from the 3-bit ones).** |
| M-35 | MoonShots (CPU replay; the orchestrator's re-derivation rule, NewIdeas' ask 14:41) | SWEEPCHECK [independent re-derivation of THE SWEEP v1's store prices (SWEEP.md rows 1-4: 11-bit / GPTQ-3, TRL-3, TRL-2+3, TRL-2 records under QOM cf 0 + 3-bit-class sure rows + DRAFT17, one LRU pool per layer at equal RAM, no LSB pool); own replay (scripts/sweep_stores_check.py on Breakthrough's capture checks), none of NewIdeas' scripts]: **FOUND Wed 14:4x: 24 GB CONFIRMED (all stores within 1-4 %, within 7 % under any stand-in; prefill share within 3-5 %). 12 GB: code confirmed (within 3 %); the chat cells hinge on the wasted-row stand-in: far rows +6..+17 %, p+97 (NewIdeas') 0..+2 %, true-path rows -18..-34 %. The real wasted rows' routing is unmeasured; a live capture of verify passes with wrong-draft rows would settle the 12 GB chat prices (data/sweep_stores_check.txt).** |
| M-36 | MoonShots (CPU; the orchestrator's re-derivation rule, NewIdeas' ask 19:4x) | SWEEPQ [independent re-derivation of THE SWEEP v1.1's quality column in the engine frame (NewIdeas 63f25fe): my own per-row KL(REF2-T || arm) on the decision map's 7 arms vs baseline.json, measured flips, my own flips ~ sqrt(KL) fit, the frame recomputed (scripts/sweep_quality_check.py); g / keep taken from NewIdeas' trellis_draft17.txt]: **CONFIRMED within +-0.2 flip points on all 6 store rows (today 1.17 / 8.38 / 6.70 vs 1.18 / 8.22 / 6.82); MSB-2 check reproduced (x1.02 / x1.09 / x1.23). FOUND (frame): linear-in-MSE r under-predicts chat KL at large r, so the TRL-2 / mix chat rows may be +0.1..+0.5 points optimistic; MSE cannot see trellis error structure (needs the screen-7 TRL arm); the 171-slot rows rest on an uninterpolated g stretch (~0.1 points on code). data/sweep_quality_check.txt** |
| M-37 | MoonShots (CPU; the orchestrator's re-derivation rule, NewIdeas' ask 20:0x) | CHATROW [independent re-derivation of THE SWEEP's chat row 10a / 10b (NewIdeas ad0bde0): QOM at cf 0.5 (Fable C's 1(c)), 10a / 10b LSB forms, 24 GB on 11-bit parts with the Cl search; own replay (scripts/chat_route_check.py on Breakthrough's capture checks, the live chain's dcf)]: **CONFIRMED (bytes) within 0-3 % with NewIdeas' stand-in, same best Cl; chat +2..+6 % with far stand-ins. Quality [G] additivity attacked: 10b stacks two perturbations on the same sure rows (super-additive risk; MSB-2 measured x1.09 / x1.23 above additive on chat); scaling from other cf cuts under-predicts the 0.5-0.99 rows; a map arm with the 1(c) masks + the store measures it. data/chat_route_check.txt** |
| M-38 | MoonShots (CPU replay, 1 core, share lock; the orchestrator Wed 20:5x) | FORESIGHT [far-ahead foresight screen: how much of the LRU -> OPT miss gap survives a partly-right forecast of the future expert schedule? M-14's chained cap45 replay (BigMoE seed45: 6 prompts, res0 forced per prompt, prompts 2-6 counted; one MSB-3 pool per layer) at K 266 (45 GB EXACT, M-14's C) and K 147 (24 GB, C_m of the 147 / 46 split). Policy FH(H): experts with a forecast use within the next H checks stay (all forecast: the farthest forecast use leaves first), the rest leave by LRU; the forecast never looks past the prompt's own last check (a drafter forecasts its own continuation). (1) perfect forecast H 1/2/4/8/16 + OPT (Belady). (2) noisy at H 8/16: each true future (layer, expert, check) use kept with probability r 0.5/0.7/0.9, drawn once per use; the dropped uses replaced by an equal count of false positives at the same check, drawn from that layer's leave-one-prompt-out popularity (recall = precision = r); 5 seeds. Refs: LRU; M-14's learned key re-fit per K (M-14: -7.9 % misses at K 266). Sensitivities after the lines: per-look noise (fresh draw every check, r 0.7); full-horizon checks only (long-generation proxy: captures are ~29 checks per prompt); M-14's crossing oracle convention. Bytes = misses x 2,162,688 B (MSB-3 parts; LSB outside the replay, as M-14). Lines (the orchestrator's): PASS = r 0.7, H 8, K 266: >= 20 % fewer misses than LRU AND better than the learned key; BETWEEN = 10-20 %; KILL = < 10 % or no better than the learned key. Novelty: published family SP-MoE, MoE-SpeQ, MoE-Infinity, ProMoE (and our own W24-belady / IDEAS:1139 perfect-foresight bounds); candidate new part = the 8-16-check horizon fed by a far-ahead drafter, priced under noise]: **PRE-REGISTERED Wed 21:00, before any run. PASS Wed 21:15: noisy FH at r 0.7, H 8, K 266 = 31.5 misses per kept token, -26.5 % vs LRU 42.9 (learned key -7.8 %, OPT -55.3 %); perfect H 1/2/4/8/16 -8 / -14 / -23 / -34 / -45 % (M-14's hor reproduced); H 8 r 0.5 / 0.9 -19.2 / -31.7 %; H 16 r 0.7 -33.2 %. 24 GB (K 147): r 0.7 H 8 -27.7 % (learned -5.7 %; perfect H 8 = 92 % of the gap). Holds under per-look noise (-22.0 %), recall falling 0.9 -> 0.5 with depth (-24.5 %), uniform false positives (-25.0 %), full-horizon checks (-27.3 %); a popularity-only forecast is +24 % (worse than LRU: the false positives are no hidden prior); 0 recall on LRU's misses still -19 % (the value is knowing which residents return within H). Target for the GPU recall step: per (layer, expert, check) recall 0.7 at precision 0.7 over 8 checks = residents' 'used again within 8 checks' at recall 0.92 / precision 0.72, vs recency's best 0.83 / 0.52 (m 8); r 0.5 falls to BETWEEN. data/foresight45.txt** **FAR follow-up Wed 22:0x (Fable FAR, same frame, ~/FableGod/far/FAR.md): at per-use recall ~ precision ~ 0.7, real forecasters cut misses only 7 % (nearest-row KNN) / 14.6 % (ridge state map), because their errors repeat across a resident's uses while M-38's are independent; the true-path ceiling is recall 0.68 (the rejected rows hold 32 % of the reuse); perfect H 8 cuts stalled layers only 5 % where misses fall 34 %. So the PASS holds for a forecaster at the per-resident 0.92 / 0.72, not for one at per-use 0.7. FAR-WAIT (22:1x): the SSD wait is ~10-16 ms + 0.15-0.17 ms per miss (bandwidth-bound above the floor), so a miss cut maps ~1:1 onto SSD time: -26.5 % = -20 ms (-14.5 %) of a cap45 check, -6..-9 % at the live 100-200 misses.** |
| M-39 | MoonShots (CPU replay, 1 core, share lock; the orchestrator Wed 21:0x) | SPANRELAX [Fable FULL's bet 'relax only inside '#' comments', pre-registered in ~/FableGod/full/FULL.md section 5 (rules A / B, PASS / KILL lines, prediction A 12.1, B 12.3, 0.2 %); RELAX_350's positions (relax_curves.py, read-only) + Mission 25's region scan (comment_loophole.py, read-only) + EvalSuite relax_core.CodeScan / comment_rows as the engine's token view (read-only); scripts/span_relax_check.py]: **KILL by the letter Wed 21:17: 4 of rule A's 70 relaxed accepts (5.7 %; B 5 of 74, 6.8 %) sit where the scan and the token view disagree (line > 2 %), and they are what the line guards against: the 125B opens a comment (argmax ' #'), the scan labels that position 'comment' from its target token, and rule A accepts a code draft (' for', ' if'). Otherwise above the lines: A 12.65 (x1.200, profile) with 0 other deviations outside comments, B 12.78 at 0.14 % code-token deviation. With the engine's causal tagger (relax_core.CodeScan + comment_rows): A 12.69 with 0 code drafts accepted, B 12.78 at 0.11 %; its 5 disagreements are comment continuations (the draft extends a comment line the 125B ends) - a post-hoc variant, re-register before use. The 125B's top-8 covers 12 of the 24 windows (comment share 15.5 % there vs 13.5 % over 24); relaxed accepts on QOM-gated rows: all at cf 0, 63 % at cf >= 0.5, 14 % at cf >= 0.9. Profile kept, Mission 25 halves it for real; length inflation (arXiv 2610.23818) is invisible here. data/span_relax_check.txt** |
| M-40 | MoonShots (CPU census, 1 core, share lock; the orchestrator Wed 21:4x) | R1CENSUS [Fable FULL's round 2 R2.2 (a), pre-registered in ~/FableGod/full/FULL.md '## Round 2': can one answer be written as parallel parts? ast census of recorded answers: parts = top-level defs / classes, edges = Name / Attribute references, the best independent set by body text (signatures, docstrings, imports, prose and driver code sequential), tokens = characters; HPD-0 walk 24 HumanEval+ + 100 MBPP windows, 38,284 Magicoder python items, the d16w agent windows; scripts/r1_census.py]: **KILL Wed 21:46 on every source (line: >= 30 % of answers with >= 2 independent parts holding >= 50 % of their tokens; KILL < 15 %): HumanEval+ 0 % (1.04 parts per answer), MBPP 0 % (1.00), Magicoder 1.2 % (>= 2 independent parts 7.2 %; those hold 34 % of their answer). Readings tried: each method its own part 6.3 % (>= 2 parts 30.6 %); that plus code-only denominators 16.8 %, still under 30. Concurrent text (what could run beside another stream) averages 0.8 % of a Magicoder answer (4.6 % per-method). The d16w agent windows carry no Python (bash, edits, a Swift file). FULL predicted HumanEval+ / MBPP <= 10 %, Magicoder 40-60 %. data/r1_census.txt** |
| M-41 | MoonShots (CPU replay, 1 core, share lock; the orchestrator Wed 21:4x; data: Breakthrough's fresh top-8, 22:23) | SPANRELAX-C [Fable FULL's re-registered comment relax, pre-registered in ~/FableGod/full/FULL.md R2.1: rule C = the causal tagger (EvalSuite relax_core CodeScan + comment_rows, read-only) + the line-end guard ('\n' not in the argmax) + R2 top-3 at >= 0.1 p1 on in_comment rows only; fresh HumanEval+ = the 12 HPD-0 windows outside the map (Breakthrough's rulec_he12.json, the map harness's baseline config), MBPP = 100 windows (rulec_mbpp.json, pending); scripts/span_relax_c.py]: **HumanEval+ BETWEEN Wed 22:24: kept(16) x1.074 = engine 11.32 (PASS >= x1.10, KILL < x1.05; prediction x1.13); 21 relaxed accepts (0.73 % of positions), 0 outside in_comment rows, 0 code-token deviations, 0 % re-scan disagreement; alignment (top-1 == draft vs the walk's own acceptance) 99.79 %. The fresh windows have fewer comment rows (11.3 % vs 15.5 %) and an easier exact profile (10.15 vs 7.78 kept). Context: M-39's 12 seen windows x1.189; all 24 pooled x1.137 (FULL's pooled prediction x1.15); the guard costs 0.004-0.015. MBPP PASS Wed 22:37: kept(16) x1.125 (PASS >= x1.03, KILL < x1.01; prediction x1.04) on 100 windows / 15,351 positions; 294 relaxed accepts (1.92 %), 0 outside in_comment rows, 0 code-token deviations, 0 % re-scan disagreement, alignment 99.04 %; MBPP answers hold 19.9 % comment rows (FULL guessed 5-8 %) and a hard exact profile (5.01 kept); the guard costs 0.010. data/span_relax_c.txt** **ENGINE Fri 01:3x (BigMoE 5b1d00f, C 192, 12 HumanEval, pair means; the flag-off gate == REF2-T passed): KILL by FULL's engine line: seconds per task x1.167 (KILL >= 0.97), tokens x1.000 (no length inflation: the line-end guard held), kept per check +10.4 % (the replay's x1.07-1.14) but tok/s -9.5 %: the kept gain is eaten by per-check host work (the tagger's per-token decodes and an extra mx.eval sync for dlm / rank). The relaxed R1 0.5 arm the same: kept +9.9 % (replay +11 %), tok/s +1.3 %. Noise floor ~+-6 %, a heat freeze 01:00-01:02. Both rules' kept gains are real; the engine form is the loss.** |
| M-42 | MoonShots (CPU replay, share lock, Thu by day outside quiet locks; the orchestrator Thu 07:1x for NewIdeas' SWEEP v1.3) | SPLITCHECK [independent re-derivation of SWEEP rank 5's bytes, drafted 0.3 + bonus gate 0.4 (NewIdeas scripts/split_price.py: 53.9 / 62.4 / 64.0 MB per kept token at 24 GB, chat x0.71 / 0.73 of the cf 0.5 route): my own replay (scripts/split_route_check.py = M-37's, Breakthrough's capture checks, LRU per layer, 10a form, 11-bit parts, best Cl), checks cf 0.5 and cf 0.3 first, V1 / V2 beside]: lines written first: CONFIRMED if the checks and rank 5 are within +-5 % of NewIdeas' with the same p+97 stand-in and the chat ratio within +-0.03; the far stand-in is a sensitivity (M-37: chat +2..+6 %); OFF otherwise (say which cell). **PRE-REGISTERED Thu 07:2x, before any run. CONFIRMED Thu 10:30: with the p+97 stand-in rank 5 = 52.8 / 62.0 / 64.2 MB per kept token [53.9 / 62.4 / 64.0], x0.979 / 0.994 / 1.003, same best Cl (92 / 92 / 69), chat x0.71 / 0.73 of the cf 0.5 route = NewIdeas'; checks cf 0.5 60.3 / 87.2 / 87.4 [61.8 / 87.9 / 87.2] and cf 0.3 49.7 / 49.6 / 52.2 [50.2 / 49.2 / 51.7]; V1 / V2 within 2.5 %. Far stand-ins put chat 3-9 % higher (rank 5 68.0 / 66.2) at the same ratio (x0.73 / 0.74). data/split_route_check.txt** |
| M-43 | MoonShots (CPU, share lock, Thu by day outside quiet locks; the orchestrator Thu 07:1x for NewIdeas' SWEEP v1.3) | KLPERERR [independent re-derivation of 'data-aware codes need a KL screen': the screen-7 trellis store's errors cost ~2-3.5x more KL per unit of expert-output error than truncation (NewIdeas a2c2a52: 3.5x / 1.8x / 2.4x, code noisy). Mine: per-row KL(REF2-T || arm) of my own (M-36's) on the s7 arms; (1) NewIdeas' formula with my KL; (2) the formula with MEASURED output-error ratios: scripts/msb2_judge.py, MSB-3 / MSB-2 / TRL errors on the map's own rows at L7 / 18 / 30 / 41 from the original 4-bit safetensors (records3-conversion-proof), validated by reproducing Fit12's TRL / MSB-3 ratios; (3) the direct per-unit comparison (dKL trl / cal) / (dE trl / cal), router-weighted, no T01 / E29 model]: lines written first: CONFIRMED if (3) puts trellis / truncation above 1.5x on >= 2 domains with its SE excluding 1 and (1) is within its SE of NewIdeas'; NOT CONFIRMED if (3) is within one SE of 1.0 on >= 2 domains (the claim would rest on the assumed 5 units of MSB-2); PARTLY otherwise. **PRE-REGISTERED Thu 07:2x, before any run. CONFIRMED Thu 10:33, slightly smaller: (3) direct, no T01 / E29 model: the trellis store costs 3.18x / 1.65x / 2.34x the KL of MSB-2 truncation per unit of router-weighted expert-output error on the same rows (per-row SE ~+-1.3 / 0.3 / 0.4; code noisy). (1) NewIdeas' formula with my KL reproduces 3.53x / 1.86x / 2.43x [3.5 / 1.8 / 2.4]; my per-row KL terms match theirs within 0-4 %. (2) The measured output errors (data/msb2_judge.json, from the original 4-bit safetensors; my TRL / MSB-3 ratios reproduce Fit12's to 3 decimals): MSB-2 / MSB-3 = 4.98-5.10 (the assumed 5 holds), but the store's ratio pooled by error over the 4 layers is 1.29-1.30 / 1.79-1.82 / 1.72-1.78, above Fit12's simple layer means 1.23 / 1.68 / 1.67 (late layers carry more error); with them the formula gives 3.21x / 1.67x / 2.28x. Caveat: errors pooled over all answer rows (the arms touch gated rows' kept experts only). data/kl_per_error_check.txt** |
| M-44 | MoonShots (patch, CPU-tested) + BigMoE (engine, Sat 00:30 ticket; the orchestrator Fri 01:4x) | RELAXV2 [the host-light v2 of both relaxed rules after Fri's KILL-in-engine (patches/r57_relax/apply_v2.py + relax_span.py, 4e2269b): dlm / rank in the argmax readback (one GPU sync per check instead of two), cached token texts, the tagger only on blocks that can touch a comment, rule C + R1 0.5 allowed together; CPU tests ALL OK: v2's decisions == v1's on all 3,296 recorded checks; host us per check v1 29-37 -> v2 15-20 (both ~0.03 % of a check: the tagger was NOT Fri's cost; the second sync, noise and the heat freeze remain); replay stacking: rule C + R1 0.5 = x1.10 / x1.23 / x1.23 (fresh HumanEval+ / MBPP / map 12), 77-89 % of the sum]. Arms vs qall (same prompts, A-B-A, the flag-off gate == REF2-T first): qallr2 (R1 0.5), qall+SPAN2 (rule C), qall+SPAN2+R1 (both). Line per arm (FULL R2.1's engine line): PASS = median seconds per task <= x0.92 vs qall AND tokens per task <= x1.05; BETWEEN = seconds <= x0.97 (tokens < x1.15); KILL = seconds >= x0.97 or tokens >= x1.15. Diagnostic beside it: kept per check vs the replay, and the post-readback time per check (ce -> c2) vs qall. Prediction: if the second sync was Fri's cost, x0.91-0.93 for one rule, x0.85-0.90 for both; if not, Fri repeats. **PRE-REGISTERED Fri 01:4x, before the run** |
Parked seed, OPENED as R-54 (NewIdeas 20:48) (NewIdeas 19:3x, published, untested here): EXPERT-AWARE DRAFT SELECTION for the 24 GB line, where an extra checked row costs ~12 ms of SSD waiting (~108 reads, R-43). Pick draft rows / tree nodes by acceptance per marginal expert READ (non-resident experts not already loaded for this pass), not per row: EcoSpec (arXiv 2607.12696, marginal expert cost in draft selection, up to 1.62x on Qwen3-235B / DeepSeek-V3.1 / GPT-OSS-120B), S2-MoE (2608.15018, routing-aware tree expansion on edge devices), EVICT (2605.00342, cost-justified tree truncation). Cheapest screen: from real 24 GB verify passes' per-row expert ids, how unevenly marginal reads fall across rows (a heavy tail = room). Owner when opened: BigMoE after R-44, or Fit12 on account-2.
PRIOR ART (Kamen's r/macmini lead, relayed by the orchestrator Wed 23:2x; NewIdeas Thu 15:5x): someone ran the 125B on a 24 GB M4 Pro Mac mini with ISTA-DASLab's GSQ-RCO IQ3_XXS (~76 GB on disk): 11 tok/s on short prompts, 8.5 long, 180 tok/s prefill, '9.5 GB peak runtime' (the post's numbers as relayed; the post itself not re-read). GSQ = Gumbel-Softmax scalar quantization per tensor (arXiv 2604.18556); RCO = Riemannian constrained optimization of the per-tensor type mix under a size budget (2605.00649); plain GGUF for llama.cpp (model card as listed by third parties). Their Flash-Next IQ3_XXS scores -0.55 points on the makers' task average, about T3R's size class (FOR_KAMEN 3, Tue). The 9.5 GB is almost surely llama.cpp's own allocations: the weights are mmapped, clean file-backed pages live in macOS's file cache outside the process footprint, so the whole 24 GB machine is their expert cache (my inference, unverified). Not comparable to our lifetime phys_footprint: our engines pread with F_NOCACHE into counted pools (bigmoe/stream.py, ext). Speed context: our exact 4-bit lines ~29 tok/s at 24 GB and ~16 at 12 GB on an M5 Pro (SSD ~2x an M4 Pro's). Nothing to adopt: non-exact, and mmap only moves memory out of the count.
Parked seeds: on-policy distillation (AdaFlash OPD: +26 % in-domain on code; needs ~1M on-policy tokens from the 125B, ~5-9 h of **Priced by MTP Head Thu 15:5x**: 1M tokens = 44x L1-20's 22.5K anchors = 5.5 doublings; at D-09's log-linear slope (optimistic) code 9.57 -> ~10.1, math 7.00 -> ~8.0, for ~7 h of decode + ~10 h of training (~17 GPU-h, not laptop-sized). A 1/4 version (250K tokens: ~9.9 / ~7.6) costs ~2 h decode + ~2.6 h training. Parked: revisit only if the design study needs data that only the 125B can make.
the Mac at 45 GB: priced after D-02 / D-03); trees (+9 % measured, L1-22; need a tree-aware GDN verify); confidence-gated block
length (R-24 / R-32: needs a drafting loop no engine has); recycling the check's rejected rows (published: Carryover Drafting, arXiv 2609.14717); router picks or more hidden layers as drafter inputs (DFlash2 already
reads 5 target layers); near-tie acceptance (non-exact; the evidence is against it, see the reading).

## BET ORDER for 350 on the 125B in <= 24 GB, all 512 experts (NewIdeas Sun 04:12; supersedes earlier line-1 ranking)
Physics now measured (Leap, TABLE_350): at 24 GB exact routing reads 44-83 experts per token from the SSD whatever the drafter
(misses are drafting-invariant), so the SSD caps it at ~70-170 tok/s; an exact resident 16-row check is ~77-82 ms with every kernel
done (kernel line at its tail); exact tokens per check are stuck at 9.5-10.5. So 350 needs SSD BYTES per token cut ~4x WITHOUT
dropping experts, then compute per check ~2-3x. The assumption every route shares: "each expert use reads and computes the whole
expert at one fixed precision". Bets that break it, in order:
| # | bet | tag | cheapest decisive test | kill line | owner |
|---|---|---|---|---|---|
| B1 [KILLED 05:44, Leap: gate ranking keeps 0.769 of output energy at 30 % neurons, line 0.90; worse per byte than fewer bits; SSD reads <= 16 KB cap ~265k/s, sub-expert reads need >= 128 KB units] | NEURON-SPARSE EXPERT READS: read the gate rows of a cold expert first, keep the intermediate neurons that carry the output, then read only those up-rows / down-columns | published pieces (CATS gate thresholding, TEAL, Deja Vu, LLM in a flash, PowerInfer-2 neuron clusters); a gate-first two-phase SSD read for MoE experts: NEW to check | prior art first; then real expert weights + captured expert inputs (one short capture): fraction of neurons carrying 90 / 95 / 99 % of each expert's output energy, and the bytes that leaves | top 30 % of neurons carry < 90 % of output energy at the median -> kill | Leap [taken Sun 04:16 EET; prior art: FloE (ICML 2025) does the two-phase read for PCIe-offloaded experts (up-first, ~9x fewer bytes, -4-8 %); pre-registered in ~/Leap/TABLE_350.md B1 with an SSD chunk-size microbench beside the energy test] KILLED Sun 05:44 EET: top 30 % keeps 0.769 of the energy under the gate-only ranking (median; oracle 0.926; gate-w = gate, the up projection decides; 90 % needs 48 % of the neurons by gate, 26 % by oracle). MoE-block error at f 0.5 is 2.3x MSB-3's while reading 0.67 of the bytes. SSD: reads of 16 KB or less cap at ~265k/s (4 KB 1.1 GB/s, 64 KB 10.2, 128 KB 12.7, whole expert 13.8); QD1 82-331 us. Even a perfect predictor with in-sample bundles of 16-64 neurons needs 62-68 % of the bundles -> at best 0.69x a whole-expert read, at 90 % energy. Closes the predictor variants too (FloE-style up-first) |
| B2 [TrainLab: screen DONE 05:01, data in ~/TrainLab/data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz; gate = NewIdeas] | CONFIDENCE-GATED PRECISION for cold reads: on passes the drafter is sure of, read cold experts as a 2-bit code; full precision otherwise | new (UGB applied to bytes, not routing) | tf screen with per-position KLD for 2-bit cold experts (Fit12's Q2 codes or MSB-2) + drafter confidence; gate analysis on CPU (ugb_rule.py) | no gate passes the task-result bar with >= 70 % of code passes on 2-bit | TrainLab (paced) |
| B2-r1 | RESULT 05:02 (turn18, pessimistic: EVERY lower rank 2-bit): realizable DRAFT gate passes the strict rules only at ~50 % of code passes (line 70 %); oracle 75 % -> the pessimistic bound fails. Deployed, only COLD reads would be 2-bit (~15-30 % of lower-rank uses at 24 GB): interpolated, ranks 4-9 cold-only ~ifeval 0.018 = likely PASS even ungated | - | turn18b: cold-only arm with real LRU residency at C 128 / 192, + SSD bytes saved | fails the strict rules ungated AND gated at >= 70 % code | TrainLab |
| B2-r2 | RESULT 05:41 (turn18c128, cold-only at C=128, my gate data/ugb_b2c128.txt): GATED = KILLED by the letter (realizable DRAFT gate passes the strict rules at only 51 % (r4) / 63 % (r2) of code passes, line 70 %; oracle 75-94 %). UNGATED cold-only r4 = -22.3 % SSD bytes per token (168.7 -> 131.1 MB), mean KLD near v3 (x0.52-0.70 of ar4), failing only code P99.9 (worst 1-2 positions) and an 8-seq top-1 CI -> a TASK-RESULT candidate (Kamen's fast-mode bar) | - | engine read of 2-bit cold experts (planar MSB-2), then an Eval suite decode verdict at 24 GB; C=192 screen (turn18c192) for the 45 GB waypoint | fails the decode rules | BigMoE / Breakthrough (engine), Eval suite |
| B2-r3 | CORRECTION 06:23 (TrainLab): a 2-bit read is records4's planar P2 part (1,540,096 B): cold-only r4 = -19.2 % bytes (not -22.3 %), r2 -23.7 %; speed ~+5-8 % (natural v3). Gated UGB dead (DEAD.md). 3B screen (turn19) measures the real P2 numerics. Fit12 B3 (in-sample, unverified): trellis 2.06 bits = 1.15x MSB-3 output error; gate/up 2-bit + down 3-bit (2.39 b) = 0.73x | - | - | - | - |
| B3 | Data-aware 2-bit expert codes (GPTQ-2 / trellis) vs MSB-3 output error | published, untested here | Fit12's Q2 (running) | 2-bit output error > 1.5x MSB-3's | Fit12 |
| B4 | Kernel-native, page-aligned expert store + Metal fast resource loading (MTLIOCommandQueue): the ~22 ms per pass of fixed read cost | published APIs, untested here | microbench: per-call fixed cost and GB/s vs today's preadv path | < 30 % less fixed cost per call | Fit12 |
| B5 [DONE 07:02, Leap: mid-loop async_eval every 4 layers in verify = -17.5 ms per resident 16-row check (-11.8 %), bitwise, ~10 lines: EXACT adopt-candidate for BigMoE; a native loop adds only ~3.4 ms more, under the line] | Native decode loop, no Python per check (~7-8 ms per check of dispatch) | engine work | compiled / native loop for the 16-row check; ms per check | < 5 ms saved per check | Leap (after B1)  PASS by the letter Sun 06:55 EET (Leap TABLE_350 B5): eager 16-row check idles the GPU 21 ms per 48 layers (build 0.42 ms/layer; trace busy 31.9 of 37.2 ms per 12 layers), but async_eval every 4 layers (exact, bitwise, ~10 lines of Python) recovers 17.5 ms (-11.8 %); a native loop adds only ~3.4 ms on the resident path (< 5). Adopt-candidate: mid-loop async_eval in verify. Open: the SSD-streamed engine (per-layer routing sync) -> B5s DONE 14:58 (BigMoE, LEDGER B5s): already in place there (BIGMOE_ASYNC_EVERY=1 async-evals every layer; the GPU then waits at each layer's gate, not on Python); every 4 layers = no gain (code 83.4 vs 84.4 ms per check, b27) |
| B6 | A much stronger drafter (data scale / size): tokens per check 9.5 -> 12.6+ | published route | needs the free cloud tier Kamen sets up | - | PARKED until the tier exists |
| B7 | POOL SPLIT / TIER PLAN at 24 GB: Leap found the small LSB-1 sidecar pool causes ~3/4 of frugal's reads at C_m 192; for exact v3, replay reads per token for plan 4433333333 at today's C_m / C_l vs rebalanced splits, vs 4444444444 and 3333333333 (all experts kept) | untested here (added 05:03) | CPU: b12rec29.py-style two-pool LRU replay on the 125B traces at a 24 GB byte budget; bytes per token per plan | best split cuts bytes per token < 20 % at equal plan -> kill the split; plan changes need a quality screen | Leap or Fit12 |
| B7/B9-r1 | RESULT Sun 06:35 (MoonShots, scripts/budget24.py, data/budget24_*.txt; 125B traces, 4 sets, tokens 400-799, budget = today's 24 GB split 147 MSB + 46 LSB = 353 MB/layer): today's split 144.4 MB/token (16-row checks 144.3: drafting-invariant); rebalanced split -0.5 % (already optimal); 2-bit WARM tier for ranks 4-9 (HOBBIT-style, best C_m 129 C_l 32 C_2 32) -18.0 % (16-row -21 %; 25 % of rank 4-9 uses at 2 bits; bigger 2-bit pools are worse) -> KILLED by the 30 % line. References: 3333333333 -25.0 %, 4444444444 +56.8 %. NEW CANDIDATE '3B' = no LSB sidecar (all ranks MSB-3) + cold ranks 4-9 read at 2 bits with a 16-32-slot 2-bit cache: -38.5..-39.8 % (Q2 1.33 MB, no 2-bit cache: -31 %). Gate = quality of ranks 0-1 at 3 bits (never measured here) + cold 2-bit ranks 4-9 (B2-r2 anchor) -> needs a tf KLD screen | - | tf screen of 3B (TrainLab turn18 machinery: plan 3333333333 + cold-only 2-bit ranks 4-9 at real LRU residency) | fails the task-result bar | open |
| B8 [KILLED 10:34, Fit12 b8_fixed: OPT-2 vs LRU C128 -8.2 %, C192 -4.7 %, line 10 %] | FORECAST-GUIDED EVICTION: the Belady bound is -38 % misses vs LRU; evict by forecast reuse (drafted tokens + a cheap router forecast from the same-layer pre-attention state) instead of recency (exact: a cache policy) | published adjacent (SpecMD least-stale, Speculating Experts 2603.19289), untested here (added 05:03) | CPU: replay with the recorded same-layer forecasts (data/samelayer) as the eviction oracle-lite; misses per token vs LRU and vs Belady at C 128 / 192 | < 10 % fewer misses than LRU -> kill | Fit12 |
| B9 | MULTI-PRECISION RESIDENCY at 24 GB (merge with B7): RAM holds hot experts at v3 tiers AND 2-bit copies of warm experts for ranks >= 4, so many SSD misses become 2-bit hits; at 2 bits ~40 % residency fits in 24 GB, about one turn's working set (~184 / 512 experts per layer per 150 tokens) | published (HOBBIT multi-precision cache 2411.01433), untested here (added 05:57); quality anchor = B2 cold-only r4 (mean KLD near v3) | CPU: multi-pool LRU replay at a 24 GB byte budget on the 125B traces: SSD bytes per token and share of rank 4-9 uses served at 2-bit, vs today's 3-bit + sidecar split | < 30 % fewer SSD bytes per token than today -> kill | MoonShots |
| B10 | **TRELLIS-CODED RESIDENCY (genuinely new bet in flight, 06:24)**: experts as ~2.1-bit trellis codes at MSB-3-level error (Fit12 B3 in-sample: TRL-2 2.06 b = 1.15x MSB-3) -> 1.29 MB per expert instead of 2.16 MB + sidecar: ~65 % of all experts resident at 24 GB (today ~25 %), ALL resident at the 45 GB waypoint (~35 GB) | published codes (QTIP bitshift trellis, our D2 encoder); trellis-coded MoE residency on an SSD-streamed laptop: new | (a) Fit12: out-of-sample output error on held-out experts / tokens + encoder time per expert; (b) Leap: Metal trellis-decode expert kernel at S=4 / 16 vs 3c 3-bit; (c) MoonShots: budget24.py replay at 1.29 MB per expert, all ranks, 24 GB | (a) held-out error > 1.5x MSB-3; (b) decode > 2x slower per expert than 3c; (c) < 50 % fewer SSD bytes per token at 24 GB | Fit12 / Leap / MoonShots |
| B10-c | BYTES SIDE PASSED (06:25, MoonShots budget24_b10.txt): 1.29 MB experts at 24 GB -> C 273 per layer (53 % resident): 21.3 MB / token vs 144.4 (-85.2 %), ~16.5 reads / token steady (77 today); 1.35 MB: -82.6 %. 45 GB: ALL experts fit (~39.4 GB total, ~5.6 GB headroom), zero steady-state SSD reads. Open: (a) held-out quality (Fit12), (b) Metal decode cost (Leap), (e) encoder time for 24,576 experts (Fit12; Metal encoder = MoonShots if > ~12 h CPU) | - | - | - | - |
| B10-v2 | SPEC CHANGE 06:58 (TrainLab turn19): 3B KILLED (x2.25-3.97 of v3; ranks 0-1 at 3 bits = x2.1-3.7 alone: the top-2 ranks' 4-bit precision is load-bearing). So B10 v2 = trellis base ~2.1 b for every expert (= MSB-3 for ranks 2-9) + a trellis RESIDUAL (~1.5-2 b) read only for rank 0-1 uses (= 4-bit grid). Tests: Fit12 base vs MSB-3 (kill > 1.5x) and base+residual vs 4-bit (kill > 1.2x) held out; MoonShots replay with base + residual pools (kill < 50 % fewer bytes at 24 GB); Leap decode of both paths (kill > 2x 3c); TrainLab layer-sliced tf screen (kill > 1.5x v3). 2-bit cold-read line STOPPED: faithful B2 / B9 save 12-18 % (+3-8 %); c192r4 x0.98-1.11 of v3 | - | - | - | - |
| B10-v2c | BYTES PASSED (07:02, MoonShots budget24_b10v2.txt): base 1.29 MB + rank 0-1 residual pool (~64 slots best): 24 GB 47.7-57.9 MB / token (-60..-67 %) at residual 0.9-1.2 MB, 46-49 reads / token (77 today); 45 GB -89..-91 %. SSD then supports ~220-270 tok/s at 24 GB (~900 at 45 GB) with full overlap. Speed estimate +21-35 % if decode is free (Leap) and ~1 MB reads hold GB/s (MoonShots microbench). After B10 the binding cost at 24 GB is COMPUTE per check again (~70 ms per exact 16-row check after async) | - | - | - | - |
| B11 | TRELLIS EVERYTHING (after B10): the exact 16-row check's bytes floor is ~10 GB (~34 ms: dense ~3 GB + ~70 distinct experts x 48 x 2.16 MB); with ~2.1-b trellis codes for experts AND dense weights (residual for rank 0-1 uses and sensitive dense parts) it is ~5.5 GB (~18 ms) -> ~23 ms at 80 % efficiency + edges / drafting ~10 ms = ~33 ms per check = ~290 tok/s at 9.5 tokens; with confidence truncation (+8 %) ~315. The road to 350 at 24 GB is B10 + B11 + truncation + a modest tokens-per-check gain | published codes (QTIP), new combination (added 07:02) | after B10(b): held-out error of trellis-coded dense matrices (GDN / attention projections, lm_head) vs their 4-bit, and the decode kernel at dense shapes | dense error > 1.3x 4-bit, or decode > 1.5x the 4-bit qmm at 16 rows | Fit12 (codes) + Leap (kernel), queued behind B10 |
| B10-a | CORRECTION 07:34 (Fit12 B3b): the 1.15x was in-sample OVERFIT. Out of sample (H from half the tokens, N/K 1.6): TRL-2 2.19x MSB-3 (FAILS 1.5x), gate/up-2 + down-3 1.62x, TRL-3 down 0.62x; 1/N fit -> ~1.36x at large calibration (estimate); ideal RD floor at 2.0 b = 0.90x. DECISIVE: large-calibration capture (~32K tokens, N/K ~12), judged on held-out text over all tokens and routed tokens, + v2 residual (target code-vs-W4 <= 0.2 E4 ~ 0.07x MSB-3, conservative) + encode time; ~6 min GPU + ~15 min CPU after quiet mode (09:22). Encode: ~6-7 s / expert on 6 threads -> ~14 h on 18 cores; fan rule caps CPU at 4 cores -> a Metal encoder is REQUIRED if B10 lives (MoonShots on standby). B4 KILLED (MTLIO +23 % fixed cost; aligned direct reads no gain). B8 re-run pending (float bug; full Belady -37..-44 %) | - | - | - | Fit12 / MoonShots |
| B10-v2d | BYTES CORRECTION 09:34 (NewIdeas scripts/b10_resid.py, data/b10r/; same expA traces, reproduces MoonShots' 47.7 exactly under LRU): the engine evicts by DECAYED LFU (BIGMOE_EVICT_LFU=0.99, adopted in both stacks), not LRU, and DFlash2 verifies 16 rows. Under both: B10 v2 at 24 GB (base 1.29 + residual 0.9 MB, Cb 229 Cr 64) = 39.8 MB / token (-16.6 % vs 47.7; per-token LFU 42.5); residual 0.65 MB: 33.5. Today's v3 under the same model: 144.5 -> 116.0. So the SSD cap at 24 GB is ~345 tok/s at 13.8 GB/s ONLY with a cross-layer read pipeline (MoonShots 09:57, data/ssd_b10.txt: ~1 MB reads reach ~13.8 GB/s only with >= 13 in flight continuously; per-layer bursts of 4-8 units get 7.8-10.1 GB/s, a burst costs 0.26-0.34 ms fixed; serial per-layer bursts at 40 MB / token = 48 x (0.3 + 8.3 MB / 14 GB/s) ~ 43 ms per 10-token check = ~230 tok/s), not 220-270 as first estimated: the 24 GB SSD bound stops being the blocker IF the base is ~2.06 b; at the 2.39 b fallback (gate/up-2 + down-3) it is 66.4 MB / token under LRU (~55 est. under LFU + 16 rows) -> ~250. Residual by routing WEIGHT (w >= t, cumulative mass) instead of rank: no better at equal w^2 proxy (rank 0-1 is on the frontier) -> no lever. Caveat: the replay routes rejected rows like the true continuation (as MoonShots' did) | - | MoonShots: re-run with LFU in budget24.py (~30 s CPU) to verify | differs > 5 % from 39.8 -> I recheck | NewIdeas |
| B10-v2v | VERIFIED 16:41 (Fit12, own replay from the engine's ensure / make_room / lfu_victim, ~/Fit12/scripts/fit_v3replay.py, data/b10r/v3replay*.txt): v3 at 24 GB, 16-row / accept-10 LFU 115.9 vs my 116.0; per-token LRU 143.5 vs 144.5; per-token LFU 127.3 vs 125.8; RAM curve 28 / 32 / 36 / 40 / 45 GB = 84.7 / 62.2 / 46.7 / 35.4 / 26.9 vs mine 84.3 / 61.8 / 46.5 / 35.1 / 26.6 (all within +1.2 %). So the '24 GB SSD wall, lifts at ~38-40 GB' numbers stand. One entry was wrong, and it was my bug: 16-row LRU 144.3 vs Fit12's 132.0. My replay() lets a check's misses evict experts the same check still needs; the engine protects them. So LFU's gain at 16 rows is -12 %, not -20 %. LFU numbers unaffected; fix replay() before reusing its LRU arm | - | - | - | NewIdeas |
| B10-12 | 12 GB LINE TOO (09:41, NewIdeas data/b10r/b10_12gb.txt, same replay): at B12's budget (C_m 51 + C_l 20 = 125.7 MB / layer) today's split reads 366.8 MB / token (157.5 MSB + 35.8 LSB reads); B10 v2 (base 1.29 + residual 0.9, best Cb 86 Cr 16) reads 175.9 (-52 %; residual 0.65: 161.9, -56 %); decayed LFU, dead at 12 GB in Fit12's runs, would cut both ~10-15 %. The 12 GB line is SSD-bound (~7.7 GB/s at 21 tok/s code), so a B10 quality pass ~doubles 12 GB speed at B12LL's exact-v3-routing quality: B10 serves BOTH focuses. Gate = B10a + the task-result bar (12 GB rule) | - | TrainLab optional independent re-run | differs > 5 % -> recheck | NewIdeas |
| B10-b | DECODE SPEED (Leap 10:33, TABLE_350 'B10(b) result', data/b10_bench2.txt; layer 24, natural routing, us per pair at S = 4 / 16): 3c-3 6.58 / 5.69, 3c-4 8.94 / 7.77, today 7.16 / 6.14. LUT code (Fit12's 4,096-entry table) base 17.2 / 16.4 = 2.6-2.9x 3c-3: KILL (the per-weight gather caps decode at ~300 G weights / s, device or threadgroup LUT alike); + residual 3.5-3.9x 3c-4: KILL. QTIP 3INST computed code: base 7.59 / 6.88 = 1.15-1.21x 3c-3 PASS; base + residual 14.1 / 13.3 = 1.58-1.71x 3c-4 PASS; spec (residual ranks 0-1) 1.41-1.50x today. Format-free fix: tabulate 3INST into the encoder LUT and re-encode (synthetic NMSE vs LUT: k=2 1.00x Gauss / 1.04x Laplace; k=3 1.01x / 1.20x; needs Fit12's held-out check). After B10a's 2-bit kill this matters for TRL-3 (3.07 b) only: k=3 decode not measured yet (expect ~1.2-1.4x 3c-3 with 3INST; kernel generalisation + one short lock) | - | - | - | - |
| B11-c | CORRECTION 09:41 (NewIdeas): B11's ~2.1 b for DENSE weights cannot meet its own line (dense error <= 1.3x 4-bit): at 2.1 b even an ideal code has ~10x+ the 4-bit MSE (2^-2R). The realistic dense rate is ~3.5 b (TRL-3.5 ~ RTN-4 error) -> 3.44 -> ~2.7 GB per check, ~-8 % check time, not -25 %. Revised 24 GB arithmetic (code, 16-row DFlash2 FT + truncation at ~10.3 rows, 9.38 tokens; 80 % of 280 GB/s; 3.5 ms drafting): B10 v2 ~6.1-6.7 GB per check -> ~280 tok/s; with the non-exact dense diet ~305. 350 then needs kernels at >= 90 % AND drafting overlap, or a non-exact union / top-8 cut. A drafter alone cannot close it at block 16: Leap's oracle (right wherever the target p >= 0.9) is 10.86 tokens | - | - | - | - |
| B10-K | **KILLED 10:34 (Fit12 B10a, ~/Fit12/data/b10/b10a.txt, pre-registered lines)**: fit on 31K tokens of local notes / code, judged on held-out text (the ref model's eval prompts + completions, 4,095 tokens, 36 experts over 9 layers): TRL-2 base 2.49x MSB-3 (all tokens) / 2.59x (routed), line 1.5x; 8K-token fit 3.03x; Fit12's 1/N fit puts unlimited same-source calibration at ~2.35x. gate/up-2 + down-3 (2.40 b) 1.87x / 1.91x. Residual vs W4: k_r=1 2.43 E4, k_r=2 0.66 E4 (line 0.2). Encode 10 s per expert at 4 threads. B8 KILLED the same hour (Fit12 b8_fixed: Belady capped at 2 passes vs LRU, C128 -8.2 %, C192 -4.7 %; line 10 %). Consequence: the 24 GB SSD wall stays (today's v3 under LFU + 16 rows: 116 MB / token -> ~120 tok/s even with a perfect read pipeline); the 350 STACK's S1 is gone. ONE FINAL TEST before closing the 2-bit-codes route (new hypothesis, not a moved line): B10-r, below | - | - | - | - |
| B10-r | **KILLED 17:12 (Fit12 ~/Fit12/data/b10r/b10r.txt, PLAN.md 'B10-r RESULT'; numbers re-read by NewIdeas 19:57)**: calibration 32,768 fit + 8,192 val tokens over 7 domains, 0 of 4,065 32-grams shared with B; d* 0.3 chosen on val (flat 1.775 / 1.750 / 1.743x). TRL-2 at d*: B all 1.854x (line 1.5), B routed 1.957x, U (unseen literary prose) 2.287x (line 1.8) -> all three lines fail: data-aware 2-bit expert codes are CLOSED. Domain shift explained part of it (2.49x -> 1.85x), but held-out text from the same broad mix is still 1.74x (1/N fit ~1.67x unlimited). Side result: TRL-3 with Leap's 3INST table (~3.07 b, decodable) 0.499x MSB-3 on B all (gate 0.445 / up 0.451 / down 0.601), 0.524x routed, 0.620x U, 0.468x val; encode 2.71 s / matrix at 4 threads -> T3R row. Original bet: FINAL 2-bit test (10:34, NewIdeas -> Fit12): is the held-out failure DOMAIN SHIFT? The fit set was 8 technical documents; the judge is eval text. B3b's same-capture half split gave 2.19x at only 4,168 rows; if that source followed a + b/N with a ~1.1, it would be ~1.25x at 31K, against 2.49x cross-domain. Test: calibration >= 30K tokens of a BROAD mix disjoint from the judge (eval-dataset items outside the suite's picks: math, code, instructions, multilingual + Kamen's agentic text, local only) and an H damping sweep chosen on a validation split, never on the judge; judge = the same text B; plus a fragility judge on a domain absent from calibration. Same capture: gate/up TRL-3 held-out (for a 'TRL-3 replaces MSB-3' option: down TRL-3 0.685x at 3.07 b) | GPU ~8 min + CPU ~30 min at 4 cores | TRL-2 > 1.5x MSB-3 on judge B (all or routed), or > 1.8x on the unseen domain -> the 2-bit codes route is CLOSED | Fit12 (STALLED 10:31-16:24: idle, the 10:34 / 13:17 peer sends sat unread; woken via its uds socket 16:24, now running B10-r, then the v3 replay check (a) with its own code; item (b) dropped, BigMoE b30 + S10b cover it) |
| T3R | **KILLED 21:47 on K1 (Fit12 ~/Fit12/data/t3r/k1.txt, k1_side.txt, k2.txt; PLAN.md 'T3R K1 / K2 RESULT')**: at 12 GB (51 / 20) 256.4 -> 214.3 MB / token = -16.4 % (line 20 %); 59 / 20 -17.0 %; 24 GB 115.9 -> 82.0 = -29.2 %. Records: base 1,884,160 B (0.871x MSB), R1 655,360 B (0.851x LSB). Why so little at 12 GB: one 16-row check's union (~100+ / layer) exceeds the 46-75 base slots, so extra slots buy ~nothing and only record size counts. Under B12LL's 4-row LRU protocol -20.6 % (per token -20.8..-21.5 %): within a point of the line before 3INST's 1.15-1.21x decode cost, so not a pass in substance; I held the pre-registered protocol. K2 PASSED (top-2 path TRL-3 + R1 = 0.475 E4 vs W4 on B routed, line 1.0; R1 keeps ~0.31 of the base's error at 1.06 b/w), but vs bf16 it is a trade (top-2 ~1.48 vs 1.0 E4, ranks 2-9 ~2.56 vs ~3.97). Also needs ~62 GB of records beside v3's (disk at the 100 GB floor) and a GPU encoder (~31 s / expert at 2 threads). Reopen only if Kamen wants a non-exact 24 GB fast mode AND disk for a second format (bytes matter at 24 GB: BigMoE's gate wait 52 of 81 ms). Original bet (19:58, NewIdeas -> Fit12, from B10-r's side result): TRL-3 3INST base (~3.07 b) for EVERY expert + a 1-bit trellis residual (k_r=1) for ranks 0-1 in place of v3's LSB sidecar. Hypothesis: better than v3 on both rank classes (ranks 2-9 at 0.5x MSB-3, measured; ranks 0-1 < 1 E4 if R1 lifts TRL-3's ~1.5 E4) AND ~25-30 % fewer SSD bytes / token (reads -13 %, ~16 % more slots in the same RAM). Line 2 (12 GB, SSD-bound): ~+30-40 % if it holds. 24 GB: ~116 -> ~85 MB / token (~165 tok/s cap), a lever but not the 3x that 350 needs. Non-exact opt-in mode; adopting it is Kamen's call. Costs if alive: GPU encoder (8 s / expert on CPU = 55 h), 3INST decode 1.15-1.21x per 3-bit expert (~5-7 ms / check where compute binds) | published (QTIP trellis) + untested (trellis residual for the top ranks, slots in an SSD-streamed MoE) | K1: replay v3 vs T3R at 12 and 24 GB (engine LFU, 16-row checks, exact on-disk sizes, best residual-pool size), minutes on 1 core. K2: k_r=1 residual on TRL-3 for the B10-r 36 experts, held-out B all / routed / U in E4 units, ~20-30 min CPU on <= 2 threads while fanwatch is down | K1: < 20 % fewer bytes / token than v3 at 12 GB -> KILL. K2: > 1.0 E4 on B routed -> no v3-quality claim (report the level) | Fit12. Only after K1 + K2: MoonShots GPU encoder, TrainLab tf screen (v3 / T3R / all-4-bit + an unseen domain), Leap decode cost |
| B11-d | 350 ARITHMETIC WITH B10, CORRECTED 09:45 (NewIdeas; my 09:42 version assumed bandwidth-bound experts and is WITHDRAWN). Leap's measured 16-row check (TABLE_350 'Where an exact 16-row check stands'): experts 62 / layer x 48 at 11.9 / 11.1 / 8.5 us each for 4 / 3 / 2-bit (per-expert cost stops following bytes below 4 bits: ~3-4 us fixed per expert call), dense ~18 ms, GDN 6.2, attention / HC elementwise / routing ~6-9, edges / sync ~9-10, drafting 3.5 = 77-82 ms; async_eval -17.5. So B10 is an SSD lever (24 GB: 144 -> ~40 MB / token; 12 GB: -52 %), NOT a compute lever: at the same union a trellis expert saves <= ~2.6 us of ~11 (and may cost more with decode). After B10, 24 GB code is compute-bound at ~9.4 tokens / ~60 ms = ~155 tok/s with today's kernels. Near-perfect engineering (experts and dense at ~85 % of floor incl. low-bit kernels, HC8 + ~3.5 b dense, GDN / attention fused ~8 ms, native loop, drafting overlapped) ~36-39 ms -> ~245-265 tok/s. 350 (~27 ms per check at 9.4 tokens) needs a further ~x1.35 that no measured exact lever gives: the block-16 drafter oracle is +10-15 %, union / top-k cuts are non-exact (top-8 passed 5/5 at 12 GB; budgets fail IFEval by 5-7). B10 stays necessary (the 24 GB SSD wall and the 12 GB line), not sufficient | - | - | - | NewIdeas |
| W24 | **24 GB SSD WALL HUNT: NOTHING REACHES 2x AT v3 QUALITY (MoonShots Mon 03:00-03:30; numbers re-read by NewIdeas 03:32; ~/MoonShots/data/wall24*.txt, msb_entropy.txt, scale_exp.txt; scripts/wall24.py reproduces NewIdeas' 116.0 MB / token exactly)**. (1) RESIDENCY POLICY IS CAPPED AND ONLINE-EXHAUSTED: at v3's split (C_m 147 / C_l 46, 16-row checks accepting 10) Belady's MIN with bypass (no policy of any kind can read less) is -41.0 % vs the engine's LFU (11 rows / 9 accepted: -43.7 %; tokens 400-599 only: -42.7 %, so no end effect), Belady with the engine's forced admission -34..-38 %; that gap is worth ~+6.8 GB of LFU pool. It needs routing knowledge of FUTURE tokens: perfect next check -10 %, next 2 -17 %, next 5 -29 %. Online, nothing beats LFU: bypass / TinyLFU-style admission +52 % (consecutive checks share experts), reuse-gap predictors +2..+119 %, the router's near misses (ranks 11-20) as hints -0.5..+4 %, one pool shared by all layers +38..+41 % (-11 % only with oracle knowledge of the check's later layers). Close this family unless a predictor of routing >= 10 tokens ahead appears. (2) LOSSLESS CODING FLOOR: real MSB-3 codes carry 2.72 bits (order-1 context adds nothing), bf16 scales / biases ~8 of 16 bits -> floor 2.98 of 3.5 bits per weight = <= -15 % bytes per miss; only the scale / bias part has a cheap exact decoder (row W24-sp). (3) BEST STACKS (replay, tokens 400+, LFU base 117.4): exact = W24-sp 12-bit pack -8.3 % + RAM +1.25 GB (MEM levers n-gram hot cache 256 MB, MLX cache cap 512 MB; caveat M24: those caps did not lower the MLX peak) -17.7 % -> 96.6 MB / token (~147 tok/s SSD cap at 14.2 GB/s) [RESTATED Tue 20:04 by MoonShots, data/w24_exact_stack.txt: the in-kernel pack is dead (W24-sp step 2), so EXACT 24 GB = RAM +1.25 GB to the pools (n-gram 256 MB + MLX cache cap 512 MB; best split C_m 159 / C_l 44, -9.3 %) + W24-ec + 11-bit at load (SSD bytes only) -> 94.4 MB / token = -19.6 % (proportional split 157 / 49: 94.9), SSD cap ~150 tok/s; the M24 caveat on the MLX peak still applies]; non-exact = T3R -29 % (K2 passed, killed on its 12 GB line) x RAM +1.9 GB incl. HC8 -13.7 % x top-8 (S9) ~-17 % -> ~60 MB / token (~2.0x, ~237 tok/s). 350 at 24 GB needs <= 40.6 MB / token at 14.2 GB/s: no combination found. Outside the rules: the internal SSD is at its measured ceiling (14.2-14.6 GB/s, data/ssd_qd2.txt); a second read path (Thunderbolt 5 NVMe enclosures, ~6 GB/s each by vendor figures, unmeasured here, striped with the internal SSD) would lift the wall ~1.4-2x at today's bytes: spending, so Kamen's call via the orchestrator | result | done | - | MoonShots |
| W24-sp | **EXACT SCALE / BIAS PACK AT 24 GB** (the MEM audit's 'scale/bias pack' lever, priced at 24 GB; v3's own 3c kernels make it feasible now). Every bf16 s3 / b3 / b in records3 fits losslessly in 12 bits: sign + 7 mantissa + a 4-bit exponent offset from a per-projection base (48 random experts: exponent span per projection median 2-3, max 5; 11 bits suffice there). MSB part 132 -> 127 x 16 KB, LSB part 47 -> 45: bytes per miss -3.8 / -4.3 % and the same pool bytes hold C_m 152 / C_l 48 instead of 147 / 46 -> -8.3 % SSD bytes / token at 24 GB (11-bit: -10.5 %), outputs bitwise unchanged (the kernel rebuilds each bf16 value with ~3 integer ops per group of 64 weights). Also shrinks the 12 GB and 45 GB pools | exact, format + kernel | (1) CPU, no GPU: pack / unpack every expert of all 48 layers (24,576 x 9 pieces), bitwise round trip, max exponent span; (2) GPU, one layer: 3c (+ combine) reading packed scales vs today, bitwise outputs and kernel time; then a records conversion in place per layer (disk is at its floor) | any projection needs > 12 bits beyond a 1 % escape list, or not bitwise, or 3c slower by > 3 %, or the gain on the engine's own trace < 6 % | step (1) MoonShots DONE Mon 08:22-08:27 (GO NewIdeas 03:32; CPU 1 thread, taskpolicy -b, scale / bias pieces only; ~/MoonShots/data/sb_audit.txt, scripts/sb_audit.py): PASS. All 221,184 pieces (5.66 G bf16) round-trip bitwise at 12 bits with 0 escapes. 11 bits hold for all 48 layers: no value anywhere is zero / subnormal / inf / nan and the widest piece spans 7 exponents, so a plain 3-bit offset fits every piece (with a reserved zero code, 5 pieces in L38 / L47 would escape). 11-bit with the LSB part at 44 blocks: C_m 154 / C_l 49, 104.6 MB / token = -10.9 % (12-bit -8.3 %); step (2) after 08:01 with the 3c owner (Leap's 3c harness), then BigMoE converts records + pools, Breakthrough re-sizes the 12 GB pools; Kamen: the internal SSD's ceiling and a second SSD are on FOR_KAMEN ('A second SSD'). **STEP (2) (MoonShots Tue 19:58, w24sp/tg11.py + w24sp/screen.py, data/w24sp_screen.txt): IN-KERNEL FORM KILLED.** Bitwise PASS: 0 mismatching elements at S = 1 / 4 / 16 on L24 / L38 / L47 (462 calls per layer: plans 4444444444 / 3333333333 put every expert at a 4-bit and a 3-bit rank, plus 4433333333 on real routing, with and without the fused combine; incl. the span-7 pieces); converted files 2,785,280 B per expert (MSB 2,064,384 + LSB 720,896 vs 2,932,736). Timing (layer 24, real routing, A + BC, 15 paired rounds, A/A within 1 %): +21.7 % at S = 1, +42.6 % at 4, +52.9 % at 16 -> KILL (> 5 %): the per-lane rebuild (4 byte loads + ~10 integer ops per scale and per bias, 4 lanes per group) about doubles the issue work of a 3c kernel that is issue-bound, not DRAM-bound. Surviving form: 11 bits as an SSD-only format expanded to bf16 at load (pools and 3c unchanged, no slot gain): -4.8 % at 24 GB alone, -11.4 % stacked with W24-ec (data/w24ec_replay_load.txt) |
Genuinely new bet in flight: B1 (gate-first neuron-sparse SSD reads) and B2 (confidence-gated precision). Updates from the night:
L2-07 KILLED (2 rows fixable, line 4); Fit12's exact stack FOLDED into bt_b12 (+7 % default, +13-15 % with C_m 59 opt-in, shas equal);
new row: SCREENED lm_head for argmax-only verify (MoonShots IDEAS #2: 2-3-bit screen + per-row bounds, exact greedy; kill: exact rows
> 25 % of vocab at the median or screen > 50 % of today's lm_head time at 8 rows) -> owner Leap after B5.

## 350 STACK at 24 GB on code (NewIdeas Sun 09:49; from Leap's measured 16-row check, TABLE_350; ms per check after async_eval ~60)
Base: DFlash2 block 16 FT + confidence truncation = 9.38 tokens at ~10.3 rows (53 experts / layer, top-10). 350 needs ~27 ms.
| # | piece | saves (ms / check) or gain | exact? | status / owner | kill line |
|---|---|---|---|---|---|
| S1 [KILLED: B10a 2.49x, B10-r 1.85x / 2.29x unseen (17:12); route closed] | B10 trellis experts: SSD 144 -> ~40 MB / token at 24 GB (12 GB: -52 %); compute only if the low-bit kernel reaches ~85 % of its floor | SSD wall lifted; ~8 ms if kernel 85 % | format change (quality gate) | B10a / B10(b) / screen queued 10:21 | see B10 rows |
| S2 | native check loop (no Python per op) | ~3.4 | yes | B5 measured; owner open (engine) | < 2 ms |
| S3 | GDN core fusion (6.2 ms vs a 0.97 floor) | ~3-4 | yes | Fit12 has the 35B GDN16 port | < 2 ms |
| S4 | attention / HC elementwise / routing fusion | ~2-3 (estimate) | yes | open | measure first |
| S5 [NOT FREE, TrainLab turn21 Mon 09:43, REPORT.md e4421be: x1.092 of v3 mean KLD (gsm x0.997, ifeval x1.061, code x1.244, multi x1.066 [1.031, 1.102]); line 5 %; stays 12 GB-only] | HC mixers bf16 -> 8-bit on the main track (HC8, 'free KLD' at 12 GB) | ~3 | no (KLD check) | TrainLab strict tf screen (asked 09:50; hc8.py from Breakthrough) | adds > 5 % to v3's mean KLD |
| S6 | dense at ~3.5 b (GDN / attention / lm_head) | ~1.5 | no | after B10 (Fit12 codes) | dense error > 1.3x 4-bit |
| S7 | drafting overlapped with the check's tail (PEARL-style post-verify, published) | ~3 | yes | open | < 2 ms |
| S10 [STEP 2 DONE 14:58, BigMoE, LEDGER S10a / JT1 / OR1: JOINT ADOPTED, exact, +6.7 / +8.0 / +7.2 % tok/s code / prose / edit, -11.5 / -14 / -7 % ms per check (line 5 %): PASS. Ceiling with PERFECT forecasts 3 layers ahead (oracle): +35-44 % code, +33-48 % prose, +19 % edit; the rest is forecast recall] | cross-layer SSD read pipeline at 24 GB: misses of layers L+1..L+3 in flight while L computes (>= 13 reads in flight); without it B10's ~40 MB / token caps at ~230 tok/s (09:57, MoonShots) | SSD 43 -> ~29 ms per check, hidden under compute | yes | step 1 (b27 anatomy, ~/BigMoE/data/b27/anatomy.txt): code check 81 ms = gate wait 52 + GPU 30; SSD busy 93 % of it but at 3.8 reads in flight (~9-10 GB/s of 14); misses 284 / check: 24 % prefetched in time, 37 % known >= 1 layer ahead, 63 % unpredicted; prefetch precision ~48 %. Step 2: each tier layer read its base and sidecar misses as two serial bursts; JOINT (Fit12 B1 v2, ported) starts the sidecar's reads with the base's (b29, 3 clean rounds, shas identical): gate wait -8 / -10 / -17 ms per check, in flight 3.8 -> 4.6. PF_QD 8 / 16, PREFETCH2, SB1 spin back-off: no gain. Oracle (true ids reserved 1 / 3 layers ahead through the same queue, b29): D1 +4.5 / +31 / +19 %, D3 +35 / +33 / +19 %, D3 at QD16 +44 / +48 / +2.5 % (QD16 slows edit's nearest-layer reads); even then a code check is 47 ms vs ~29 GPU (SSD-bound). Trace-side predictability (Fit12 since 13:20) decides how much of that is reachable. b30 15:26 (your pre-registered rule, LEDGER TH1 / FC1): THETA 0.3 neutral; ranked lists 24 + THETA: prose +5.0 %, code -0.4 % -> fails; L+2 forecasts (+ per-distance queue) lose -> NONE adopted. In-engine forecast recall of misses (router_{L+d} on layer L's MoE input, raw, unique list): code 58 / 27 / 15 % at lead 1 / 2 / 3 (27 / 15 / 11), prose 63 / 40 / 26 % (20 / 14 / 10), edit 52 / 17 / ~1 % (50 / 16 / -); precision of issued reads ~40 % at every lead; reads issued and used cover 29 / 34 / 20 % of misses (lead 1), lead 2 adds 1-2 %. Next = a long-lead predictor (proposed to NewIdeas: hidden-state trace + offline recall@16 at lead 2, kill < 55 %) | step 2 saves < 5 % per check |
| S10b | **PARKED BY THE PRIMARY LINE, BY 0.6 POINT (TrainLab Mon ~14:00, REPORT 'S10b on b32' 66663ec, data/s10b_fit_b32.txt); S10c OPENED ON THE SECONDARY (NewIdeas 14:08)**. The plain router reproduces BigMoE's checker exactly. Selected on the validation prompt: M1a (linear + affinity); the MLP tied it. Held-out prompt 4, lead 2, per pass at 4.8 reads per pass-layer: top-k among non-resident (PRIMARY) 44.4 % vs the router's 32.9 % (leave-one-out 46.2 vs 35.2), < 45, so park. The shipped pick rule (top-K over all 512, resident skipped; pre-registered as secondary): 53.3 % vs 36.5 % (LOO 55.6 vs 39.0); the shipped lead-1 policy catches 50.4 % at 5.14 reads on the same passes (48.4 % at 4.8). Shipped rule at top-12: 44.6 % with 2.8 reads at 43 % precision, vs the router's 43.2 % with 6.6 reads at 18 %: the same catch for under half the reads. Lead 3 holds (51.0 vs 34.9, shipped rule). Miss-only training was worse everywhere. **S10c (pre-registered NewIdeas 14:08; model frozen, no refit)**: [S10c-2 = b34 PRE-REGISTERED Tue 19:57 by BigMoE (NewIdeas' rapid-screen design): ONE 10-min lock, runs A1 B1 B2 A2 (prompt 4 of each domain, 400 tok, b33 settings). B = frozen M1a lead-2 lists at L = 2, 8, .., 44 -> T = L+2 replacing T's lead-1 router lists, shipped rule elsewhere; both arms PRED_LINES=3 + PRED_CAP=61 (each list at the shipped 61 slots), same PF_QD=4 queue. PASS: late + demand misses at the 8 targets (MSB + LSB sidecar) per token <= -15 % AND rusage disk bytes per token <= +2 %. KILL: > -5 % or bytes > +5 %. INVALID if any sha differs. scripts/b34_runner.sh, s10c_ab.py, bigmoe/m1a.py; waits for TrainLab's export.] [b34 AMENDED Tue 20:05 before the run, per NewIdeas' note after R-22: runs A1 B12 B16 A2 (M1a top-12 and top-16 per row; BIGMOE_TRACE on in all runs for gate wait + per-layer reads). PASS also if the 8 target layers' SSD bytes per token (ext read trace x part size) drop >= 10 % at tok/s no worse than -1 % vs A1 + A2. KILL unchanged. Reported per arm: tok/s, gate wait and SSD bytes per token.] [B33 CAPTURED 16:16-16:24 (BigMoE; ~/BigMoE/data/b33/b33_note.txt): exact (20 shas = b32), clean; misses = on time 13.7 % / late 21.0 % / queued 11.1 % / demand 54.3 %; NO join to b32 (pass boundaries depend on timing), b33 self-contained with x for all prompts; router on late + demand held-out, lead 2 per pass: k2 25.2 / k4 38.1 / k16 64.4 %; TrainLab scores next.] (1) b33 (BigMoE, ~8 min of GPU): 2-bit labels resident / on time / late / demand on b32's passes, joined to b32's x. Score M1a vs the plain router, both on the shipped rule at <= 4.8 reads per pass-layer, lead 2, held-out prompt 4 + LOO, on LATE + DEMAND misses only. PASS: M1a >= router + 10 points (it is +16.8 on all misses), then (2). KILL: < +5 points, meaning its edge was on-time prefetches; S10b / S10c closed. Between: (2) only if BigMoE prices the integration as small. (2) Engine A/B (BigMoE designs and pre-registers it): M1a's lead-2 picks REPLACE the lead-1 router picks at the same rule and budget. Exact (prefetch moves timing, not math). Line +5 % tok/s on code / prose, KILL <= +2 %. Earlier: **B32 BUDGET SWEEP + LABEL CAVEAT (BigMoE Mon 13:16, ~/BigMoE/data/b32/budget_note.txt); BAR RE-ANCHORED TO READS (NewIdeas 13:46)**: the deployable plain router per row is 63.9 % held-out at lead 2 (k16). Per pass (union over rows), lead 2, held-out: k2 28.4 % at 3.5 reads per pass-layer, k4 41.4 % at 6.8 (5.7 never used), k8 54.3 % at 12.9, k16 66.7 % at 24.5. The shipped lead-1 policy catches 50.4 % of lead-1 misses with 4.8 reads (3.7 unused), and the SSD runs ~93 % busy. Caveat: the 'miss' label includes prefetches that landed on time (~27 % in the Oct-19 runs), so a long-lead predictor gains only on late + demand misses. New bar: the MLP must pay in reads, not slots. Per pass, at the shipped budget (~4.8 reads per pass-layer), lead 2, held-out, it must catch >= 50.4 % of misses: the shipped lead-1 policy's catch, so it could REPLACE the lead-1 reads and issue them a layer earlier (the router alone gets ~34 % there). Below 45 % parks S10b. A pass or between: b33 (2-bit labels resident / on time / late / demand, ~8 min of GPU, BigMoE) scores it on late + demand, then an engine A/B design. TrainLab: report the per-pass MLP at k = 2 / 4 / 8 (recall, reads, precision), as BigMoE asked. Earlier: **ALIVE, BELOW THE A/B LINE (TrainLab Mon 09:43, REPORT.md e4421be)**: MLP (hidden 512) on router_{L+2}; held-out lead-2 recall@16 of misses 55.6 % vs plain router 49.6 % (code 61.1 vs 55.4, math 62.1 vs 47.7, edit 40.4 vs 41.6; leave-one-out 57.2 vs 50.2); lead 1 worse (52.0 vs 58.9). It predicts the routed set far better (76.0 vs 58.8 %) but spends its 16 slots on resident experts. NewIdeas 09:46: GO b32 (BigMoE re-capture with residency snapshots) + (a) the deployable score (top-16 NON-resident predictions; lines 65 / 55 at lead 2) and (b) a miss-only fit, exploratory. **TRACE CAPTURED (BigMoE b31, GPU Mon 08:15-08:22; read by NewIdeas 08:32)**: 439 MB in ~/BigMoE/data/b31/s10b/, 20 prompts x 400 tokens (code / prose / edit / math, 5 each), 4405 verify passes, 8834 rows; miss records match the routed ids on every checked pass. Plain router baseline (router_{L+d} on layer L's MoE input, top-16, recall of the row's misses; first checker pass, 3838 of 4405 passes): lead 1 57.6 %, **lead 2 49.1 %** (code 53.0, prose 52.1, edit 42.7, math 49.4), lead 3 46.4 %. So the predictor must add >= 6 points to clear the 55 % kill line and >= 16 for the 65 % A/B line. Cold per-prompt cache in this harness (18-34 tok/s): its miss rates are not test_spec125's. Final numbers (fixed checker 310cf27, all passes + held-out prompt 4) pending; TrainLab fits after turn21. LONG-LEAD EXPERT PREDICTOR (NewIdeas 15:27, from BigMoE b30: today's router forecaster recalls 58 / 27 / 15 % of code misses at lead 1 / 2 / 3, ~40 % precision; code needs >= 2 layers of lead because its ~0.6 ms of GPU per layer is shorter than a read burst; the oracle at lead 3 is worth +35 % code / +33 % prose, exact). Bet: a learned per-layer predictor of layer L+2's experts from layer L's MoE input (linear on router_{L+2}, then a small MLP; published direction: ProMoE stride predictors, Pre-gated MoE), used ONLY for prefetch, so outputs stay exact. Test: (1) BigMoE engine trace, ~3 min GPU, ~3 GB disk (125 GB free): layer L's MoE input + the true top-10 of L+1..L+3 for ~12K verify rows over code / prose / edit / math; (2) TrainLab fits offline (CPU <= 4 cores) and judges on held-out prompts | recall@16 of the MISSES (non-resident set from the trace) at lead 2 | < 55 % (today's lead-1 recall) -> KILL; >= 65 % -> engine A/B (adopt at >= +5 % code, shas identical) | BigMoE trace (next wake ~18:08) -> TrainLab fit. DISK 23:31: the disk is at 100 GB (BigMoE waits for > 115 GB before its <= 5 GB trace; Kamen must empty the Trash to get ~124). NewIdeas suggested a <= 1 GB trace over a spread subset of ~8 predictor layers, same kill line, deleted after the fit; BigMoE's call under the floor rule. TRACE LAYOUT (BigMoE 03:09, code ~/BigMoE 6809e3e; slim spec taken: capture b31 runs after the silent night, its quiet lock after Fit12's and Breakthrough's (~08:45); CAPTURED 10-26 08:15-08:22 (b31, code 6809e3e, checker 310cf27): 439 MB in ~/BigMoE/data/b31/s10b/, 20 prompts x 400 tokens, 4405 verify passes, 8834 verify rows, 0 trace/ext mismatches on 3838 passes (the other 567, see the join rule, pending the fixed checker); deleted after the verdict): pNN_<domain><i>.npz, one per prompt (code / prose / edit / math = GSM8K test 0-4, 5 each, <= 400 tokens): one entry per decode/verify call of a MoE layer: seq (pass), L (0-47 trunk, 48 = MTP head), rows, row_off -> ids [rows, 10] int16 (routed experts, score descending); x_off (-1 unless L in 2,8,14,20,26,32,38,44) -> x [rows, 2560] uint16 = bf16 bits of layer L's MoE input (the router's input; qwen4_exp has no norm swap, so the engine's forecaster applies router_{L+d} to it as is). A verify pass = one seq covering L 0..47, the same rows in the same order at every layer: take each layer's FIRST call in a seq (after a partly accepted verify, MTPLX's commit re-runs layer 1 over the kept rows under the same seq: fewer rows, all resident, not in miss.bin; b31: 567 such calls, skip them). miss.bin: int32 records [seq, L, n, (e | miss << 16) x n] per served (pass, ext layer): its unique experts, miss = not resident when served (b30 FC1's definition); ext L >= side0 (meta.json) are LSB sidecars: ignore. routers.npz: wT [512, 2560] f16 for every T = L+1..L+3. meta.json: prompts, counts, shas. Join: a row's misses at T = its ids at (seq, T) that are miss-flagged in (seq, T). Held out: prompt 4 of each domain (or leave-one-prompt-out). B32 (NewIdeas GO 09:08, code 3eb8ada; b31 deleted 09:46): same files + res.bin = int32 records [seq, L, nT, (T, w_0..w_15) x nT], one per JIT prefetch decision of base layer L (right before the ext's schedule(), after L's own loads are released): residency of pools T = L+1..L+3 then, bit e % 32 of w_{e // 32} = expert e resident; deployable score = recall of (seq, T)'s misses by the top-16 among experts NOT resident in (seq, L -> T); x kept for all 20 prompts. B32 CAPTURED 10:07-10:13 (482 MB in ~/BigMoE/data/b32/s10b/, 4363 verify passes, 8861 rows; checker clean: 0 unequal, 0 mismatches, every scored step has a snapshot). Plain router: all-512 held-out lead 1/2/3 58.7 / 49.4 / 47.1 %; DEPLOYABLE (top-16 non-resident) held-out 75.3 / 63.9 / 63.1 % (lead 2: code 67.8, prose 66.6, edit 58.9, math 63.8). Read-budget caveat: k = 16 non-resident per row = ~16-32 unique reads per pass-layer vs ~2-3 misses; budget sweep DONE 13:12 (~/BigMoE/data/b32/budget_note.txt): lead 2 per pass-layer k2 29.3 / k4 42.3 / k8 55.3 / k16 67.2 % of the pass's misses for 3.5 / 6.8 / 12.9 / 24.5 reads (77-93 % never routed), vs 2.6 misses; shipped lead-1 policy ~4.8 reads (3.7 unused). LABEL CAVEAT: S10b's miss = not in the table at T's service, and prefetches are promoted only there, so on-time prefetches count as misses (0.0 % of non-resident-at-decision experts were in the table at service; Oct-19 split 27 % on time / 27-29 % late / 43-47 % demand). Option b33: 2-bit labels (on time / late / demand), ~8 min GPU, NewIdeas' call. ~/BigMoE/scripts/s10b_check.py checks the files and prints the plain router baseline's recall@16 of the misses at lead 1 / 2 / 3 (the number to beat). **b34 = S10c step 2, BETWEEN (BigMoE Tue 20:4x, ~/BigMoE/data/b34/b34_note.txt), exact (all shas equal)**: M1a lead-2 top-12 at the 8 traced layers vs shipped: late + demand misses -8.6 % (A1-A2 noise 0.4 %), on-time hits +21 %, those layers' SSD bytes -8.8 % (noise 1.2 %; 15 % fewer prefetch reads); top-16: misses -7.9 %, bytes +1.2 %, so K = 12 wins. Whole process inside noise (8 of 48 layers). Extrapolated to 46 layers: ~-7..-8 % SSD bytes/token, exact, no decode cost. **NewIdeas 20:44: GO for the all-layer build as its confirmation** (exact, cheap: ~7-min capture ~2.2 GB + TrainLab's 46 fits + one 10-min A/B; stacks with R-16 / R-27). Confirmation lines: PASS whole-process SSD bytes/token <= -5 % at tok/s no worse than -1 %; KILL > -2 % |
| = | S1-S7 | ~60 -> ~35 ms = ~270 tok/s | | | |
| S8 | stronger block-16 drafter (oracle 'right where the target is >= 0.9 sure' = 10.86; FT 9.88 from 50K tokens) | +10-15 % | yes | open (training lab, after the B10 screen); L5's +11 % came mostly from training depths 8-15 | projected < +8 % |
| S9 [STRICT RULES FAIL, TrainLab turn21 Mon 09:43: x2.23 of v3 KLD; fails ifeval (KLD x1.53, top-1 -1.7 pt), code (P99.9 x2.82), multi (KLD x1.38); gsm passes. Stays an opt-in judged by the task bar; its 12 GB 5/5 decode pass and FOR_KAMEN item 3 stand. gsm 7 of 15 after the 08:45 thermal sleep + the 09:07 lid hold] | top-8 routing opt-in (passed 5/5 at 12 GB as b12llk8u2) | +~9 % | no (task bar) | TrainLab strict tf screen (asked 09:50); Eval suite task verdict after Thu | fails the task bar |
| S11 [KILLED 23:37, Leap TABLE_350 'S11 result'] | decode once per distinct expert instead of per (row, rank) pair, in today's 3c | Estimate 23:31 ~5 ms per check (model: 3c ALU-bound on the decode). MEASURED 23:37 (layer 24, natural routing): the grouped copy is BITWISE equal to per-pair 3c (all plans, S 1-16) but SLOWER: S 10 3c 0.597 vs grouped gm 2 0.714 ms per layer (gm 1 = machinery only 0.673; gm 4 0.955; gm 8 1.85); per check at 10.3 rows -5.8 ms (3-bit) / -6.5 ms (today's mix), at 16 rows -7.4 / -8.2 ms. Sharing the decode adds time (registers per member cut the occupancy 3c needs to hide its read latency); the decode ALU is not 3c's limit | - | Leap | not bitwise, or < 3 ms per check -> drop: DROPPED |
| D32 [Leap, DONE Mon 10:02: PASS, R = 1.24; TABLE_350 'D32'] | dense part of an exact check at 32 rows vs 16 (NewIdeas 23:41: decides block 32 for 350): 17 dense shapes x counts, 16-row exact paths vs 32-row paths (2x16 / adjacent threadgroups / one simdgroup per 2 row tiles / MLX 32), bitwise per row vs the 16-row path | R = 32-row / 16-row ~1.3-1.4 predicted. RESULT: R = 1.24 (16 rows 17.96 ms, 32 rows 22.19 ms; all 32-row kernel paths bitwise = 2 x 16). Block 32 breaks even on dense at 11.6 tokens per check, on experts at ~14.3 (L1-20: 10.5) | yes (bitwise) | Leap | PASS R <= 1.25; KILL R >= 1.60 |
| = | S1-S9 | ~330-350 (WITH B10; void since 10:36) | | | |
| ! | **10:36: B10 is dead, so the 24 GB SSD wall is back**: v3 under LFU + 16-row checks reads ~116 MB / token -> ~120 tok/s even with a perfect read pipeline (S10); top-8 -> ~145. 350 at 24 GB needs SSD bytes / token cut ~3x and no live lever does it (B10-r is the last 2-bit test). 17:12: B10-r KILLED, so the 2-bit route is closed; the wall numbers were verified independently (B10-v2v). Live byte levers at 24 GB: top-8 (S9), MoonShots' 24 GB hunt (T3R -29 % at 24 GB but KILLED on its 12 GB line 21:47; reopening needs a non-exact 24 GB mode + disk). 23:41 (Leap, S11 + par32 union numbers): at 45 GB a 16-row check's distinct expert bytes are ~5.5 GB (~23 ms at 240 GB/s) and today's 3c runs ABOVE that (29 ms, pair-bound), so 350 (~27 ms / check at 9.4 tokens) needs tokens per byte ~2x. Exact levers left are small: S8 oracle-grade drafter at block 16 -12.5 % expert reads / token; block 32 at the oracle's 15.0 tokens: experts -4 % but dense per token -37 % IF dense stays bytes-bound at 32 rows (asked Leap to time dense 16 vs 32 rows); S9 top-8 -17 %. Multi-stream (L1-32) is resident-only -> parked. Open research question for NewIdeas: a 2x tokens-per-byte lever at 45 GB. SSD cap vs RAM for v3 (replay, LFU, 16-row checks, perfect pipeline; data/b10r/v3_ram_curve.txt, v3_45gb.txt): 24 GB 116 MB / token ~121 tok/s | 28 GB 84 ~166 | 32 GB 62 ~227 | 36 GB 46.5 ~301 | 40 GB 35 ~399 | 45 GB 26.6 ~526. So the SSD stops binding for 350 at ~38-40 GB. At the 45 GB waypoint S2-S11 apply: ~270 with engineering, ~300-330 with S8 / S9 | | | | |
| ! | **UPDATE 22:30 (NewIdeas): CA DONE (Leap 19:10, TABLE_350 'CA'): exact 16-row check 143 / 150 ms (1K / 8K) + 12.5 drafting = ~64 tok/s; experts 58.7, HC 31-34, GDN projections 17-19, QSA 7-19, commit replay 9.7, glue 5.8. Every lever together ~55-60 ms, still ~1.5x over the 34-40 ms that 12-14 tokens need. Outside calibration: Splash (incoai, Kamen's link 22:16) reports 95 tok/s for this model on a 64 GB M5 Pro, resident at 2 bits, chain DFlash2: ~40-65 ms per cycle with custom per-shape Metal kernels, so Leap's all-levers estimate is realistic. Its two loop tricks are R-11 / R-12.** **UPDATE 14:20 (Leap's CA interim + SD; data/price350.txt rerun): drafting is MEASURED at 12.5 ms per block-16 check (forward 7.5 + head top-16 over 248K 4.5 + selector 0.8; 15.5 at block 32), not 3.5. The engine's own resident exact 16-row check (MTPLX kernels, 4-bit, top-10) is ~130-147 ms before drafting, about 2x the best measured kernels (D32 / L1-30c), so the engine today would give L1-20 ~60-67 tok/s at 45 GB. With every part at its floor, 350 needs 15.2 tokens per 16-row check with today's drafting (97 % of a perfect drafter), or 12.4 with an 8-bit drafter + fused head (~4.5 ms, row DR). SD: all exact drafter levers stacked give ~10.7-11.8 at block 16 (+12-24 %), so with floor kernels and fast drafting the exact ceiling is ~300-335 tok/s; NT-1 (non-exact, +10-21 %) is the swing to 350. The 45 GB path to 350: (1) the engine's kernels to Leap's (row KP), then to ~90-100 % of floor; (2) DR; (3) the exact drafter levers (trees +9 %, SD1, selectors); (4) a non-exact acceptance mode, Kamen's call.** **REPRICED FROM MEASURED KERNELS (NewIdeas Mon 13:46; scripts/price350.py, data/price350.txt; Leap asked for D32's 22.2 ms): with today's kernels 350 at 45 GB is out of reach even for a perfect drafter. It needs BOTH a check near its floor AND +26-43 % tokens per 16-row check; block 16 beats block 32 at every kernel level.** Inputs: D32 dense 17.96 / 22.19 ms (16 / 32 rows; floor 12.3); L1-30c per-expert cost at 16 rows ~10 us at ~2.7 b (4-bit 11.9 us read-bound, 3-bit 11.1, 2-bit K-paired 8.5 vs a 5.5 floor), so the experts take ~26 ms of a 16-row check against a 17.5 ms byte floor; attention / GDN core / routing / glue 10-20 ms (unmeasured); drafting ~3.5. Cross-check: BigMoE's b28 anatomy measures a whole exact check at 7.4 rows at 52.8 ms of GPU. Non-exact stack (top-8, ~2.7 b experts, dense diet), one stream: today's kernels 51-69 ms per 16-row check, so 350 needs 18-24 tokens (perfect drafter 15.6; L1-20 gives 139-185 tok/s), and 73-91 ms per 32-row check, needing 25-32 (perfect 30.5). Every part at 90 % of its floor (other 7 ms): 39 ms, so 13.7 tokens per 16-row check (L1-20 x1.43, 97 % of the lattice oracle 14.1; L1-20 alone 244 tok/s) or 17.6 per 32-row check (x1.67). At the floor: 12.0 (x1.26; 278 tok/s with L1-20) / 15.5 (x1.48). Correction to (3) below: the old '17-19 per 32-row check' was priced on a bytes-only check with a lean 15 ms loop (par45's 233 tok/s); the measured check is 1.5-2x off its floor, so the crux has two halves, the check and the drafter, not the drafter alone. Next: (a) measure the unmeasured 10-20 ms (attention, GDN core, routing, glue) of a resident check; (b) the dense diet's real time at 16 rows (D32 timed the exact dense); (c) expert reads (2-bit reads alone at 78 % of floor, L1-30c); (d) SD for +26-43 % at block 16; NT-1 (+10-21 %, non-exact) stacks on (d). **Earlier: 350 at 45 GB: no exact lever doubles tokens per byte; the crux is the drafter (NewIdeas Mon 03:27; answers the 23:41 question)**. (1) Sharing: the union grows as rows^0.6-0.72 (par32: 66.1 -> 100.5 from 16 to 32 rows), so experts per accepted token fall only as rows^-0.3..-0.4; halving them needs ~6-12x the accepted rows per check. Only k streams of one answer beat that (L1-32: 0.59x at 4 streams), and they change the text. (2) Codes sit near their bound. From Fit12's per-matrix table (~/Fit12/data/b10r/b10r.txt): TRL-3 everywhere is 0.50-0.66x MSB-3's error at ~3.07 b (-12 % bytes); gate TRL-2 + up/down TRL-3 (2.73 b) ~0.90x (B) / 1.15-1.20x (U) at -22 %; gate/up TRL-2 + down TRL-3 (2.40 b) 1.31-1.75x. (3) Tokens per check needed for 350, assuming 280 GB/s and ~15 ms of non-byte time per check (today ~25-29): exact v3 reads ~10.9 GB per 16-row check, so it needs ~19 tokens, above the perfect drafter's 15.6 on code; at 32 rows (~14.8 GB) it needs ~24 of a perfect 30.5. With the non-exact stack (top-8 0.84x union, TRL 2.73 b 0.78x, dense diet 2.1 GB): ~14 per 16-row check (90 % of perfect), or ~17-19 per 32-row check (55-62 % of perfect; L1-20 gets 10.2-10.5, its lattice oracle 20.3). Leap's bytes-only pricing (cand21) gives ~17.5 at 32 rows. So 350 at 45 GB needs a drafter that nearly doubles tokens per 32-row check on code, on top of the non-exact stack and a lean loop. (4) The drafter was still learning when L1-20 stopped. Selector agreement on the training run's own validation split (in-distribution) from step 100 to 200: d16 0.59 -> 0.66, d20 0.45 -> 0.52, d24 0.38 -> 0.43, d31 0.33 -> 0.38 (~/Leap/data/l120_ft.state.json), training loss still falling; 99 train windows = 25.2K generated tokens (Leap: TABLE_350's '42K' was wrong), 2 epochs. Cheapest decisive test of the crux: DS (below). If it fails, 350 at 45 GB needs a behavior change (parallel streams of one answer, Kamen's call) or training beyond the local budget. (5) Side effect: a TRL 2.73 b base set is ~41 GB, so with the dense diet every base fits at 45 GB (residuals streamed); that would un-park L1-32 there. It needs ~57 GB of disk for a second format (101 GB free; Kamen's Trash ~124 GB) | | | | |
| DS [NewIdeas 03:27; Leap; DONE Mon 11:26: data-flat (s_fix 1.013, CI 0.990-1.035), compute ~5 % per doubling of steps (c50 1.052, CI 1.025-1.080), s_prop 1.066 -> no new 125B data; DS2 next; TABLE_350 'DS'] | Drafter data slope: is DFlash2's depth decay a data problem? Retrain L1-20's recipe (block 32, partial masking 0.5, same 201 steps, same lr) on a random half of its 109 windows. Score it against the existing L1-20 weights on the 24 held-out HumanEval samples (data/dfl2d_code) at block 32, 1-step and 2-step best | published: drafter scaling (EAGLE-3 'training-time test'; DFlash's own training). Ours: the slope on a near-deterministic MoE target at block 32, where the perfect drafter gets 30.5 | ~28 min of training + ~2 min of scoring; no new 125B generation | slope = tokens(100 %) / tokens(50 %) at block 32. KILL < 1.05: data-flat, so no affordable drafter reaches ~18. GO to data generation (4x on-policy windows, ~3 h of 125B time) >= 1.12. Between: one 2x-data point decides. LEAP'S DESIGN (adopted): half = stratified random half of the 99 train windows (36 code + 14 math, fixed seed), val unchanged, same trainer (ada5b1f) via a symlinked data dir. Arm B = 50 % @ 201 steps gives s_fix = T(100 %) / T(B). If s_fix < 1.12, arm A = 50 % @ 2 epochs (~100 steps): s_prop = T(100 %) / T(A) (double the data, train in proportion) and T(B) / T(A) (the compute slope). KILL only if s_fix < 1.05 AND T(B) / T(A) < 1.05; if compute moves, 100 % @ 402 steps (~56 min) before any new 125B data. Scoring: block 32 1-step and 2-step tail 0.9 (fixed), block 16 1-step; per-sample paired bootstrap over the 24 samples (sample noise only); L1-20 re-scored the same way. Leap predicts s_fix 1.02-1.06, s_prop 1.04-1.08 | Leap (drafter scripts); NewIdeas reads |
| DS2 [Leap, pre-registered Mon 11:30; DONE 12:25: KILL, c100 = 1.005 (CI 0.980-1.035): twice the steps at full data +0.5 %, val plateaus after ~200 steps; DS + DS2: this drafter recipe saturates at ~10.5 tokens per 32-row check; TABLE_350 'DS2'] | Compute slope at 100 % data: L1-20's recipe on all of its windows at 401 steps (L1-20 = 201), scored like DS. c100 = T(401 steps) / T(L1-20) | published: drafter compute scaling. Ours: the slope for this drafter on this target | ~57 min of GPU (weights on a RAM disk), no new 125B data | ALIVE (price a long run) c100 >= 1.12; KILL (training this drafter harder cannot reach ~17-19) c100 < 1.07; between: 801 steps decides. Leap predicts 1.03-1.07 (70 % KILL) | Leap |
| SD [GO NewIdeas Mon 12:30; Leap offered it 12:25; prior-art pass, no GPU. **DONE (Leap Mon ~14:15, TABLE_350 'SD'): no drafter family gives +26-43 % at block 16.** With no depth decay at all (0.93 held at every depth), block 16 gives only 9.81 (+3 %): the gap at block 16 is per-token quality, not depth. 12.0 needs a flat 0.96 per token and 13.7 a flat 0.98; only target-class models reach that. AR drafters, semi-AR and exposure-bias fixes <= ~+3-5 % at block 16, with extra drafting time; higher-order selectors +5 % (xPress), L1-21 +1 %; capacity: DFlash's own 3 / 5 / 8 layers = 0.94 / 1 / 1.06, which answers the shrink test without a GPU; trees +9 % at 16 rows (L1-22), the best exact lever measured. All exact levers stacked: ~+12-24 %, so 10.7-11.8. **SD1 GO (NewIdeas 14:20)**: acceptance-aware loss (2606.11552 claims +21-76 % over a position-uniform trainer; ours is position-uniform). Arms U vs W at block 16, where W weights each depth by its remaining expected length; ~50 min of GPU, no new 125B data. KILL W/U < 1.05, PASS >= 1.10 (Leap predicts 1.03-1.08). Runs after CA run 3] | A structurally different drafter or verify shape for block 32. DS + DS2: the DFlash2 recipe saturates at ~10.5 tokens per 32-row check (2x data +1 %, 2x steps +0.5 %), yet the lattice oracle (drafter's top-64 per position) is 20.3: the right tokens are mostly among the candidates and the first-order selector loses them at depth. Cover: (i) autoregressive drafters and AR tails inside the block (EAGLE-3 training-time test, HASS, GRIFFIN, semi-AR / PARD-style); (ii) selectors stronger than first order over the lattice (what L1-21's picker lacked); (iii) exposure-bias fixes (training on the drafter's own drafts); (iv) capacity: bigger drafters, plus a one-sided shrink test (L1-20's recipe with 5 -> 3 layers: T >= 0.95x of 5 layers kills 'a bigger drafter'); (v) verify shapes at 32 rows (trees after L1-22, priced with D32) | published (each family); untested here | prior art + pricing only. Per family: published accepted length vs a parallel-block baseline, drafting ms per check on this Mac, new 125B data needed or not, the cheapest offline kill test on existing data (dfl2d_code, l118_tgt top-32, the lattice) | Output: a ranked table in TABLE_350 'SD' + one pre-registered test (<= 1 GPU hour; new 125B data only with a GO). Price against NewIdeas' 350-at-45-GB line (13:46, '! REPRICED' row): with the check near its floor, 350 needs 12.0-13.7 tokens per 16-row check (L1-20 9.54, lattice oracle 14.1) or 15.5-17.6 per 32-row check (L1-20 10.5, oracle 20.3); a design's own drafting ms counts against it | Leap |
| NT-1 [NewIdeas; replay Mon 10:04, analysis 13:37: PARKED as a non-exact multiplier, not a 350 lever by itself] | Near-tie (lossy) acceptance at the first miss: accept the drafter's token when the 125B gives it >= rho x its top-1 probability (or it is the 125B's #2 where top-1 p < 0.9) | published: Judge Decoding (ICLR 2025), AutoJudge (NeurIPS 2025: up to 1.5x accepted tokens per cycle at < 1 % GSM8K loss); untested here | replay of L1-20 at block 32 on the 24 held-out HumanEval samples (scripts/nt_accept.py, 10 s of GPU; reproduces Leap's 10.23 exactly) + scripts/nt_analyze.py, data/nt/nt_accept_b32.txt | RESULT: 56 % of first misses sit where the 125B is >= 99 % sure (no near-tie rule can forgive them), 6 % at 0.9-0.99, 26 % at 0.5-0.9, 11 % below 0.5. Forgiven share, tokens per pass LB (+1 token) / MOD (the run continues with the exact replay's survival, later misses forgiven at the same rate; optimistic): rho 0.5 11 %, +1 % / +10 %; rho 0.25 19 %, +2 % / +18 %; rho 0.1 25 %, +2 % / +25 % (5 seeds +25.2-25.3 %); #2 with p1 < 0.9 18 %, +2 % / +16 %. Forgiving only the first miss: rho 0.5 +9.5 %, rho 0.1 +21 %. 21 of the 68 rho-0.5 pairs touch digits, operators, brackets or control words. The pre-registered gate (MOD clears +25 %) is met only by the loosest rule, at the line, on an optimistic estimate: no GPU decode + Eval suite test now. Re-open at block 16 beside a check near its floor (350 needs +26-43 % there) or for a non-exact fast mode (Kamen's call). Side finding: 10 of 598 first misses are numeric near-ties (the drafter matched the 125B's teacher-forced top-1; the official output differs) | NewIdeas |
| DR [GO NewIdeas Mon 14:20; Leap proposed it] | Drafting at the speed of its bytes. DFlash2's 1-step draft costs 12.5 ms per block-16 check (forward 7.5, head with top-16 over 248K 4.5, selector 0.8). An 8-bit drafter (forward ~2-3 ms) plus a fused lm_head + top-k kernel (~1.5 ms) should give ~4.5 ms | published (int8 drafters, fused top-k); untested here | Leap: conversion + one kernel, minutes of GPU | PASS: draft <= 5 ms at block 16 AND tokens per check within 1 % of today's (24 held-out samples, block 16 and 32). KILL: draft > 8 ms or tokens -3 %. The output stays exact either way (the drafter only moves acceptance). Worth ~8 ms on every DFlash2-drafted check (45 GB floor: 43.3 -> 35.3 ms) | Leap |
| KP [NewIdeas Mon 14:20: a pricing question to Leap + BigMoE, no build yet] **[KP PRICED Tue 19:20 by BigMoE, ~/BigMoE/data/kp_pricing.txt: exact levers only on 45 GB = R-09 X5 HC -9.7 + gmr24 experts -3.7 = ~-13.4 ms per 16-row check (~10 %). 24 GB: no exact lever; ANY GPU saving is capped by the SSD idle time per pass (b28: 5.1 / 6.3 / 3.8 ms code / prose / edit = <= 7 / 10 / 2 %), and Leap's kernels do not cover 3c, so no port. 12 GB: no exact lever; dense <= 3.7 ms per pass even ideal, SSD cap applies.]**  **NOTE (NewIdeas ~22:40): reference-change levers priced by Leap so far: NAX experts -22..-26 ms per 16-row check (CA), the batch-invariant dense set -21 (L1-23 / R-13), MTPLX's 3-kernel HC read ~-14 (R-09 F8). Together ~-55..-60 of ~156 ms. Once KP prices the bitwise ports, the rest goes to Kamen as ONE decision: adopt a new exact reference built from the faster kernels (output may differ in the last bits; Eval-suite equality check first).**| Port Leap's measured kernels into the engines. CA interim: the engine's resident 16-row check with MTPLX's kernels is ~130-147 ms, about 2x what D32's dense (qmm16c / mr2) and L1-25 / L1-30's native-layout NAX experts imply. Nothing in BigMoE's LEDGER tracks a port (only B5s). Question: per line (24 GB: 3c ext experts + MTPLX dense at ~3-7 rows; 12 GB: Breakthrough's engine; 45 GB: MTPLX at 16 rows), which of Leap's kernels beat what ships, by how many ms per pass at the shipped row counts, and is each bitwise equal to what ships? | ours | pricing only, from Leap's tables + BigMoE's b28 anatomy | A bitwise-equal kernel is an exact speed lever (BigMoE's / Breakthrough's call). One that changes rounding changes the exact reference itself: Kamen's call, via the orchestrator | Leap + BigMoE |
| CA [NewIdeas Mon 13:46, asked of Leap; Leap pre-registered 14:05 (TABLE_350 'CA': 12-layer slice through MTPLX's forward with the engine's snapshot / accept / capture-commit, one command buffer per part under a Metal trace, 1K + 8K context; prediction: GPU-side other 15-25 ms at 16 rows, + ~20 ms eager glue). **DONE Mon 19:04 (run 3: knockouts, no trace, 1 min of GPU; TABLE_350 'CA', data/ca_diff_report.txt): engine's exact resident 16-row check, 48 layers = 143 ms at 1K / 150 at 8K + drafting 12.5 (~64 tok/s). Experts 58.7, HC reads / writes 31-34 (eager at 16 rows; MTPLX's fused read serves 2-8 rows only), GDN projections 17-19, QSA 7 / 19, commit replay 9.7, eager glue 5.8, GDN recurrence ~4. Other ~45-65 ms, not 20; parts sum to the check within 0-2.4 %. Ranked: HC -23..-25, experts -22..-26 (NAX), matmuls ~-22 (D32), drafting -8, glue -6]** | Check anatomy at 45 GB: the unmeasured part of an exact resident check. Time every part of one exact resident check at 8 / 16 / 32 rows on real weights: attention (QSA layers incl. the indexer), GDN core (Fit12's port), routing (top-k, sort, gather maps), glue (launch gaps, syncs, copies) and drafting, with dense and experts as in D32 / L1-30c. Also time the end-to-end check: the parts must sum to it within 5 % | ours (each part's floor is ours to compute) | ~10-20 min of GPU, one quiet lock; layer subsets scaled to 48 layers | A measurement, no kill line. Prediction (Leap TABLE_350): other 10-20 ms at 16 rows. It sets price350's 'other' and ranks the check levers. If other <= 10 ms, the check is mostly dense + experts (kernel work); if >= 15 ms, glue / GDN core / attention come first | Leap |
Leftover candidates if a piece fails: near-tie acceptance at the 7 % of code tokens the target is < 0.9 sure of (non-exact, ~+10 %);
block 32 (net ~+2 % under this cost model unless the drafter nears the oracle); per-check union budget (fails IFEval by 5-7 today).

## LINE 1: 350 tok/s, ranked by expected gain x chance
| id | idea | tag | cheapest decisive test | kill line | owner |
|---|---|---|---|---|---|
| L1-01 | Batch-invariant dense kernel for 8-16-row passes (exact wide passes without 5-row chunking, which costs 3.5-6.4x on GDN shapes) | published idea (batch-invariant kernels), untested here | microbench the pass's dense shapes at M=1/4/8/16: fixed-tile kernel vs MLX's per-M choice; ms + bit-equality across M | > 1.5x MLX's best at M=8 | MoonShots [taken Sat 02:40; DONE 10:10: prototype multi-row qmv bitwise but 1.6x MLX at M=8 -> KILLED; band map: 4-bit layers exact to 12 rows with no chunking (bands 1 | 2-12 | 13+); NEW L1-01b: per-op 16-row padded qmm for lm_head / in_proj_qkv / q_proj, ~-3.4 ms per 8-row pass, exact under its own numerics (MoonShots 10:15 msg)] |
| L1-02 [DONE Fit12 04:05: 8-row pass 144-172 ms incl. draft at C192 (verify 135-163 = stall 64-95 + compute ~70); DFlash2 32.6 vs MTP 33.0 tok/s; code-350 needs ~11x; T16 not runnable (<= 9 rows at top-10), T8 173 ms oracle; ~/Fit12/data/x1] | REAL 8-row DFlash2 pass time at the best setting that fits < 45 GB (24 GB / C=192) with exact verify: the number code-350 hinges on (needs ~20 ms incl. draft at 6.93 tokens) | untested here | Fit12's fit_dfl2.py prototype adapted to v3 24 GB: ms per pass, misses, tokens per pass on 3 code prompts, 256 tokens | 8-row pass >= 60 ms (code-350 then needs > 3x from kernels) | Fit12 |
| L1-03 [DONE Fit12 04:05; CORRECTED 05:00 (Leap's catch): the 04:05 '~0.8x the replay' came from a harness with no end-of-turn stop (post-turn tokens counted). On identical spans real/replay = 0.93-0.98 -> the replay holds; free-form code (not in the replay set) 0.71-0.79x of HumanEval's tau stands; tok/s = MTP at C192] | DFlash2 end-to-end check (real decode, not replay) at 24 GB on code + GSM: tokens per pass and tok/s vs MTP, corrected counting | check of a claim | same batch as L1-02 | tokens/pass < 0.9x Leap's replay -> the replay overstated | Fit12 |
| L1-04 | Block-16 DFlash (positional extension of the block-8 drafter, fine-tuned on code, local, < 1 h) | published (DFlash uses block 16), untested here | CPU first: extrapolate DFlash2's per-depth acceptance curve (.97 .92 .88 .85 .80 .77 .74) to depth 15 | projected tau16 < 9 -> kill before training | Leap [taken: Leap, Sat 02:35; CPU 02:42: projected tau16 10.2-11.9 on code (flat 0.956 conditional, d2-7) >= 9 -> NOT killed; next = block-16 fine-tune on code, after L4; ~/Leap/QUEUE_L1-04_L1-06.md. Sat 04:39 L5 running: ORIGINAL drafter at block 16 = code tau16 8.88 / GSM8K 6.56 (x1.29 / x1.15); with Fit12 T16/T8 ~1.5-1.65 the fine-tune kill line moved 9.0 -> 10.3 before any result (pass >= 11.3). RESULT 05:47 (~/Leap/DFL2E_RESULT.md): fine-tuned code tau16 9.88 (x1.44 vs block 8; deep depths 0.91 conditional), GSM8K 6.98, block 8 kept (6.76) -> KILLED for exact; under a per-pass expert budget (16 rows ~ 8 rows) it is a free x1.44, 9 % short of the ~10.9 that code-350 needs] |
| L1-05 | L4: DFlash2 self-distilled on the 125B's OWN agentic outputs | published route | queued batch (<= 2 h) | agentic tau8 < 1.10x MTP, same counting | Leap [RESULT Sat 07:48 (~/Leap/DFL2D_RESULT.md): KILLED. Fine-tuned agentic tau8 2.96 = the original's 2.96 (line 4.07 = 1.10 x MTP 3.70); val flat (depth-1 0.80 -> 0.80) on 18.5K tokens / 144 steps; code 6.87 -> 6.73. DFlash2 agentic stays ~3 after L2b / L3 / L4; agentic = MTP + lookup. Data deleted] |
| L1-06 | Joint verify: DFlash2 block + MTP chain as two branches of one pass (accept the longer); oracle +9 % on agentic | untested here | CPU: L2b's per-block acceptance of both drafters + a rows->pass-time model | net (accepted / pass time) < +5 % | Leap [taken: Leap, Sat 02:35; CPU 02:42: KILLED unless 15 rows cost <= 1.04x 8 rows: the expert union grows x1.52 (46 -> 70 per layer, agentic routes), plus GDN chunking; realistic net -3 to -12 %] |
| L1-07 | MMA-tile expert kernel, batch-invariant (one tile height, fixed K order, same kernel for decode) | published kernel style, new here | after L1-01: expert-half microbench at M=8/16 vs 3c | < 1.3x faster than 3c at M=16 | MoonShots |
| L1-08 [taken: TrainLab, 04:13; paced sleep to ~06:10, then turn17 at R=8 + PAUSE-safe resume] | Per-pass shared-expert budget on 8-16-row passes, gated by drafter confidence (UGB): the non-exact pass-time lever | new (UGB) | tf screen at R=8/16 with per-position KLD + drafter prob (TrainLab's turn17 machinery, R=8) | no gate passes the suite's rules with >= 70 % of code passes budgeted | TrainLab |
| L1-09 | Early-exit verify of confident passes (final norm + lm_head at layer 36 / 40) | published (LayerSkip, CALM), untested here | one tf screen saving per-position top-1 agreement of early exits vs the full model, joined with confidence | top-1 agreement < 0.98 at layer 40 on code positions with p >= 0.9 | Leap [taken: Leap, Sat 02:50; screen = scripts/dfl2_exit.py (exit = the model's own final HC mixer + head on layer 36 / 40's stream), runs at the end of the L5 batch. RESULT 06:04 (~/Leap/QUEUE_L1-09.md): KILLED by a wide margin: agreement at layer 40 = 0.057 code / 0.060 GSM8K / 0.033 agentic (0.056 at full p >= 0.9), at layer 36 ~0.002; exits never confident. Control PASSED 06:31: exit after layer 48 = full model (1.000 on all sets); the prediction forms only in the last layers (layer 44 0.32, 46 0.68, 47 0.93 on code); kill stands] |
| L1-10 | Copy-span rows: verify copied spans (tool outputs / file text re-emitted) as cheap prefill-like rows (F lever, 1.2-1.36x on agentic) | published-adjacent (lookup decoding) | CPU on Chat1000's fshare data: tokens in copies >= 16 that lookup drafting does NOT already accept | < 5 % of agentic output left uncaptured | Leap [taken: Leap, Sat 02:44; CPU 02:48: NOT killed, narrowly: lookup (K=5) leaves 5.8-6.5 % of output in copies >= 16 unaccepted (accepts 75-78 % of copy tokens); mostly its width cap. One pass per span saves ~11 % of passes (lookup-only), ~21 % est. with MTP; next = wider lookup drafts on long matches, engine test; ~/Leap/QUEUE_L1-10.md. Stage 2 Sat 04:52 (CPU, ~/Leap/QUEUE_L1-10.md): CLOSED as a 350 lever. Wider lookup does not pay (rows cost ~2.9 ms each in RAM, union^0.72 on SSD); longest-match lookup + widening to 8 drafts = x1.03-1.04 agentic, exact, a small tweak for BigMoE/SpecDecode] |
| L1-11 | Trees over DFlash2 top-16 | published | Leap's ceiling: 7.38-7.76 of 8 by construction | PARKED | - |
| L1-12 | The 350 physics table, 125B + 35B (NewIdeas 08:09, orchestrator 08:20) | analysis | CPU: bytes floor per pass from the weight files x expert union x measured tokens / pass | - | Leap [DONE Sat 08:58, ~/Leap/TABLE_350.md: NewIdeas' read CONFIRMED. 125B bit-exact cannot be resident (71.7 GB) and tops out ~260 at the floor; v3 tiers (~56 GB, > 45 GB rule) ~305 (350 only at the pure floor if code rows share >= 20 % more experts); 350 in practice needs the budget (433 at the floor; >= ~80 % kernel efficiency). 35B fits (20.4 GB), floor ~560, measured 219: 350 needs ~63 % of the floor (today 39 %), AA 18.2 vs 39.8] |
| L1-13 | Profile the 35B's 16-row DFlash block (54.5 ms vs a 21.4 ms bytes floor): dense GDN / attention / router / routed experts / shared / head / drafter / glue | measurement | GPU ~10 min: per-component timing of one 16-row (and 8-row) verify pass, captured layer inputs (SpecDecode roofline.py style), each vs its bytes floor | no component set whose fix (each to >= 80 % of its floor) brings the block to <= 34 ms (= 350 on code at 11.9 tokens / block) -> the 35B code-350 kernel route is dead | Leap [taken: Leap, Sat 08:59. RESULT 09:07 (~/Leap/TABLE_350.md section L1-13; corrects the 09:03 note, which assumed the drafter / loop split): KILLED by its letter, 3 % short. The 16-row pass takes 45.1 ms vs an 18.7 ms floor (42 %); drafter 5.5 ms (floor 3.8), loop overhead 3.9. Time over the floor: GDN mixers 9.7-10 ms (12.2 vs 2.04, grows with rows), routed experts 7.4 (67 % of floor), attention 3.3, shared / router / head 2.6, glue 3.0. Every byte-bound part at 80 % of floor -> 35.1 ms = 340 tok/s (> 34 ms line); + a tighter loop or + L1-14's truncation -> ~365. GDN alone -> 266. A kernel PROGRAM (multi-row GDN first), not one kernel] |
| L1-14 | Confidence-truncated verify on the 35B (verify only the drafts whose running confidence >= t), real decode | published (dynamic draft length: SpecDec++, AdaEDL), untested here | GPU ~5 min, same lock as L1-13: z-lab's loop at block 16 with truncation, t = 0.3 / 0.5 / 0.7, 12 HumanEval+ prompts | < +5 % code tok/s over plain block 16 (218.9) | Leap [taken: Leap, Sat 08:59. RESULT 09:06 (~/Leap/TABLE_350.md section L1-14): NOT killed. Real decode, same run: t=0 221.9 tok/s, t=0.3 240.6 (+8.4 %), 0.5 237.3 (+6.9 %), 0.7 233.7 (+5.3 %); outputs leave plain only at near-ties (batched verify). Exact, cheap (~10 lines in the loop); worth porting to the 125B's DFlash2 path once it verifies > 9 rows] |
| L1-15 | Multi-row GDN kernel for 16-row verify passes (35B: 30 mixers 12.2 ms vs a 2.04 ms floor; the 125B has 36): projections + conv + norms + recurrence + gated norm, batch-invariant (the same kernels for 1-row decode) | published pieces (chunkwise / recurrent GDN verify: fla, FlashInfer MTP, TreeWY, ReplaySSM; batch-invariant kernels), new here | GPU: per-op profile + bitwise S-row vs plain decode, then 30 mixers at 16 rows, ms + bitwise | > 6 ms or not batch-invariant (pass <= 4 ms) | Leap [taken Sat 10:33 (NewIdeas 10:13); DONE 10:51: ALIVE between the lines: 30 mixers at 16 rows 4.56 ms (stock 11.57), spec == plain BITWISE (16 and 8 rows, outputs + states); not yet <= 4 ms. Gap was the projections (MLX 19-21 % of floor at 16 rows), not the recurrence. New qmm16 = 16-row 4-bit NAX matmul, padded for decode (+1.05 ms/token on 30 mixers). 35B code ~219 -> ~254. ~/Leap/GDN16_RESULT.md; 12:41 END TO END: 35B code 222.6 -> 251.3 tok/s (+12.9 %) with the GDN kernels + the rollback replay submitted separately (z-lab left it lazy inside the next pass: -4 ms); + L1-14 truncation 268.4 (+20.6 %). Matmul now qmm16c (fp32-exact dequant, = MLX qmv accuracy); GDN non-matmul ops bitwise stock 0.32.4]
| L1-16 | 125B (Kamen 13:1x: 350 on the 125B only): structure SHARED across a layer's 512 experts (activation-weighted base, shared input / output subspaces) + low-bit per-expert coefficients, to go below 2 bits/param at near-4-bit quality (resident ~26 GB; 16-row pass ~4.9 GB -> ~390 at 80 % kernels) | published at mild ratios (D2-MoE, MoBE -24-30 %, merging); new: sharing + low-bit codes at <= 1.5 bits | CPU/GPU, no training: layers 12 / 24 / 36, weight-space shared energy, then output error on real routed inputs at equal bytes vs MSB-3 and RTN-2 | at <= 1.5 bits no shared form beats RTN-2 g64 output error (pass: <= MSB-3's error) | Leap [taken Sat 13:45; KILLED 15:06: no shared structure in weight space (mean expert 1/512, subspaces near isotropic); on output error at <= 1.5 b no shared form beats RTN-2 over the whole expert (down has no shared output subspace, 0.44-0.65); partial positive: gate / up on the layer's activation-weighted input subspace beat RTN-2 at ~half the bits (r 768 @ 1.35 b: 0.084 / 0.112 vs 0.130 / 0.171) -> a component for Fit12's codes. ~/Leap/TABLE_350.md, data/act16.txt]
| L1-17 | 125B: fewer experts per token (top-8 / 6 / 4 renormalized, or adaptive k holding >= p of the top-10 weight): the 16-row union (73 / layer) drives bytes per pass; stacks with Fit12's codes (~5.4 GB pass -> ~350 zone). Non-exact (Kamen's call) | published (dynamic expert skipping, top-p routing, Ada-K), untested here | GPU ~3 min: Fit12's captured MoE inputs through the real router + 4-bit experts: MoE-block output error vs MSB-3 on all 10, and the 16-token union | top-8 > 3x MSB-3's error (pass: union -25 % at <= 1.5x MSB-3) | Leap [taken Sat 15:10; 15:31 NOT KILLED, pass missed by 1 point: top-8 = 0.78x MSB-3's MoE-block error at union -16 %; adaptive p 0.8 = 1.43x at -24 % (line -25 % at <= 1.5x); top-6 2.98x at -34 %. Router flat (top-8 holds 88 % of the top-10 weight); the shared expert carries 15-56 % of the block output. Next: Eval-suite verdict on top-8 / adaptive (routed 15:3x: Fit12 builds the hooks in its fork, the Eval suite judges). ~/Leap/TABLE_350.md, data/topk17.txt]
| L1-18 | 125B tokens per pass (A3; 350 needs 12.4 / pass at the ~280 tok/s byte stack, x1.26): how much drafting headroom does the 125B leave on code? If DFlash2's misses sit where the target is confident, a stronger drafter (bigger, on-policy distilled, block 32) has room; if at the target's near-ties, exact tokens/pass is capped | published (acceptance-aware training, DistillSpec, EAGLE-3 / 3.1); the headroom measurement for this target is ours | 24 public HumanEval samples: 125B top-32 log-probs (one streamed prefill) + DFlash2 block-16 replay: agreement by target confidence; oracle tau16/32 | oracle tau32 (t 0.9) < 12.4 (pass: >= 16 and DFlash2 tau16 <= 0.7 x oracle tau16) | Leap [taken Sat 15:42; 15:51 BETWEEN THE LINES, strong finding: the 125B is nearly deterministic on code (88.9 % of tokens at top-1 p >= 0.99) and 63 % of DFlash2's first misses are on those; DFlash2's agreement on confident tokens decays with depth (0.57 at depth 10, 0.12 at 15). Perfect drafter: 15.6 / pass at block 16, 30.5 at block 32; oracle (t 0.9) 10.9 / 15.0 (kill 12.4, pass 16). Next: a 2-step refining drafter (re-mask low-confidence positions), no training. ~/Leap/TABLE_350.md] |
| L1-19 | 125B tokens per pass: a 2-step refining DFlash2 drafter (MaskGIT-style: re-mask the step-1 positions the selector is unsure of, predict again; +1 drafter forward, ~3.5 ms / pass), no training | published idea (iterative masked decoding), untested on DFlash2 drafting | GPU ~3 min: teacher-forced replay of the official loop, 24 public HumanEval samples, 4 variants | < +10 % tokens per pass at block 16 (pass >= +15 %: tau16 >= 10.2) | Leap [taken Sat 15:52; KILLED 16:06: every 2-step variant is worse (8.62 / 7.81 / 8.25 vs 8.88; no masks 1.81): DFlash2 was trained on all-masked blocks only, filled tokens are out of distribution; at block 32 it gets 8.56 (depth collapse). Depth / refinement gains need training -> next bet: a depth-robust block-32 retrain with partial masking. data/refine19.txt] |
| L1-20 | 125B tokens per pass: DFlash2 fine-tuned at block 32 on the 125B's own outputs (200 public L5 prompts, ~50K tokens) with random partial masking (half the blocks reveal 10-60 % of positions; no loss there) so 2-step refinement is in-distribution; local training (< 1B) | published parts (DFlash training, MaskGIT / masked diffusion, acceptance-aware training; L5 here +11 % at block 16); new: block 32 + partial masking for a refining DFlash2 | 2 GPU batches (generation + features ~55 min; training <= 50 min + replay): tokens per pass on the 24 held-out HumanEval samples, best of block 16 / 32 x 1- / 2-step | < 10.0 (pass >= 13.0 = 12.4 / real-loop 0.95) | Leap [taken Sat 16:42; 19:13 BETWEEN THE LINES by its letter: best 10.51 / pass (block 32, 2-step tail 0.9; 1-step 10.23; block 16 9.54 vs L5 9.88); 2nd step +3 % only. Priced at the stack (32 rows cost x1.65 the experts), block 32 = block 16: best 311-313 tok/s vs 292 original (+7 %, = DBloom 2608.30427); the 13.0 line ignored row cost. New: the drafter's top-64 lattice holds the 125B's token at >= 0.91 of depths; an oracle picker = 14.11 / 16-row pass (398 tok/s), 20.29 at block 32: the first-order selector is the loss -> L1-21. ~/Leap/TABLE_350.md, data/cand21_l120.txt] |
| L1-21 | 125B tokens per pass: an autoregressive picker over DFlash2's top-64 lattice (conditions on every chosen token + the drafter's hidden state; the shipped selector is first-order). L1-20's oracle: 14.11 / 16-row pass vs the selector's 9.54 | published class (LiLiCorr 2608.20530 +7-19 %, BPD lattice rescoring +5-25 %, Hydra / ReDrafter); untested here; the +48 % headroom is ours | GPU ~30 min: dump lattices on L1-20's windows, train a ~5 M picker (local), held-out HumanEval replay at blocks 16 / 32, priced at the stack | < 10.5 per 16-row pass and < +10 % priced at block 32 (pass >= 13.1 per 16-row pass or >= 350 at the top-8 stack) | Leap [taken Sat 19:17; 20:20 KILLED: picker +1 % (block 16: 9.65 vs selector 9.54; block 32: 10.33 vs 10.23; control incoai 9.00 vs 8.88); priced 312 vs 311 tok/s. = published (Domino / DSpark ~ DFlash2's selector). The lattice holds the token (oracle 14.11) but a small picker cannot tell which one. L1-22 screen (trees, same run): N 15 at 16 rows = 10.43 / check (+9 % at equal rows) but priced 294 < truncated chain 311 -> killed by its line. ~/Leap/TABLE_350.md, data/pick21_*.txt, tree22.txt] |
| L1-23 | 125B: bit-exact 16-row kernels for the dense layers (MoonShots' L1-01b, handed to Leap): one batch-invariant kernel per shape called at 16 rows every pass (qmm16c for 4-bit, MLX's fixed 16-row config where padding-independent). Today an exact 16-row check chunks rows (12+4 / 5s): dense ~57 ms vs ~12 ms floor and ~16 ms at 1 row | published pieces (batch-invariant kernels; NAX MMA), new here for the 125B's dense set | GPU ~5 min: kern23.py microbench (per-check shapes / counts, cold cache), bitwise row-independence | 4-bit set > 25 ms per 16-row check (pass <= 15 ms) | Leap [taken Sat 20:26; 20:58 PASS: 4-bit set 14.0 ms per 16-row check vs 35.2 today chunked (floor 7.1); whole dense 56.8 -> 27.4 ms (floor 11.9): -29 ms per exact 16-row check. MLX's own 16-row call is padding-independent on every shape (exact without a new kernel); qmm16c wins on o_proj 7.51 -> 2.39, in_proj_z 5.65 -> 1.79, in_proj_qkv, lm_head, q_proj. Left: block_inject 4 ms, HC mix down, small shapes, then the routed experts. data/kern23.txt] |
| L1-24 | 125B routed experts in verify checks: read each DISTINCT expert once with MLX's own per-row (qmv) arithmetic -> bit-identical to the engine (no reference change). Screen 21:4x: 16-row check experts 60.7 ms vs 34.1 floor (56 %); all engine expert paths bitwise equal and row-count invariant | published pieces (qmv_mr / tier_glu here; grouped MoE GEMM); new: few-rows dedup with single-row arithmetic | GPU ~3 min: test_gmr24.py on real weights, bitwise + ms | not bitwise or >= 1.10 ms / layer at 16 rows (pass <= 0.90) | Leap [taken Sat 21:51; 22:30 KILLED by its line: bitwise EQUAL at 1/4/8/16 rows (+ one expert in 16 rows) but 1.132 vs sorted 1.209 ms per layer at 16 rows (51 % of floor). Cause: the expert path is ALU-bound on MLX's qmv arithmetic (~7 us per routed pair), not bandwidth-bound -> the fix is a tensor-unit (NAX, qmm16c-style) expert kernel with new rounding: ~-20-25 ms per 16-row check. data/gmr24.txt] |
| L1-25 | 125B routed experts on the tensor units: one 16-row NAX tile per distinct expert (qmm16c code-MMA + segment prologue), batch-invariant by construction, new rounding (one-time reference change) | published pieces (grouped MoE GEMM, batch-invariant kernels), new here for few-rows verify on the M5 | GPU ~3 min: test_qmmg25.py, real weights: invariance, error vs fp32, ms | > 0.95 ms / layer at 16 rows, not invariant, or error > MLX's (pass <= 0.75) | Leap [taken Sat 22:32; 22:38 BETWEEN THE LINES: batch-invariant (bitwise), 2.3x more accurate than MLX vs fp32, 0.776 ms per layer at 16 rows (75 % of floor; pass 0.75) vs MLX sorted 1.242: experts of a 16-row check 59.6 -> 37.2 ms; 8 rows 31.9 -> 23.8. With L1-23 an exact resident 16-row check's dense + experts ~116 -> ~60 ms. data/qmmg25.txt] |
| L1-26 | 125B: fragment-native 4-bit weight layout for the tensor-unit kernels (Kamen's 'mend to fit the GPU' (a)): weights stored in matmul2d fragment order (one 1 KB block per simdgroup per tile-group; two rows' codes paired per word -> two bf16 operands per 3 ops); bit-identical to qmm16c / experts_nax | published pieces (Marlin / FT interleaving, FLUTE restructuring, Ladder layouts), new for M5 matmul2d | GPU ~3 min: test_qmm26.py (dense + real experts) | not bitwise or < +10 % on 16-row experts (pass >= 85 % of floor) | Leap [taken Sat 23:26; 23:31 KILLED by its letter on the experts (bitwise equal; 16 rows 0.709 vs 0.762 ms per layer = +7.5 %, 82 % of floor) but the dense 4-bit shapes gain +16-57 % (77-86 % of floor), bitwise -> adopt for the dense set. First run hit a strided-input copy per call (fixed). CORRECTED 00:42 (fair interleaved A/B at equal configs, data/ab28.txt): dense x1.15-1.22 (76-84 % of floor), experts x1.06 (8 rows) / x1.08 (16 rows, 81 %); the +16-57 % compared a best-of-5 with one config. data/qmm26.txt] |
| L1-27 | 125B: 4-bit weights fed to the tensor units AS 4-bit: matmul2d bfloat x uint4b_format -> float (installed SDK 26.5), the packed MLX codes as a tensor_inline operand, per-group affine epilogue via cooperative-tensor coordinates; no software unpacking (Kamen's 'mend' (a)) | published API (WWDC26 330), new for exact few-row 125B checks | GPU ~3 min: test_qmm27.py (error vs fp32, invariance, ms dense + real experts) | error > MLX's, not invariant, or 16-row experts not faster than the native layout 0.709 ms / layer (pass >= 90 % of floor) | Leap [taken Sat 23:36; 00:42 KILLED: int4 operands work (error = qmm16c's, invariant) but slower everywhere in a fair interleaved A/B: dense x0.68-0.87 vs qmm16c, experts x0.77-0.80 vs experts_nax (55-61 % of floor). data/ab28.txt] |
| L1-28 | 125B: native-layout tensor-unit kernel with two 32-column tiles per simdgroup (x fragments / row sums / K loop shared; bit-identical to qmm16n) | kernel tuning, published style | GPU ~5 min: ab29.py fair A/B + bitwise | < +5 % on experts and dense, or not bitwise (pass >= +10 % or >= 88 % of floor on 16-row experts) | Leap [taken Sun 01:43; 01:44 KILLED: bitwise equal but slower at equal ks (16-row experts x0.94, dense x0.96-0.99); the native layout at 75-85 % of floor is near this design's limit. data/ab29.txt] |
| L1-29 | 125B: bf16 16-row tensor-unit matmul for the hyper-connection mixers (HC mix down 5.32 ms per check vs 2.27 floor) | kernel, published style | GPU ~3 min: test_bmm30.py fair A/B | < +15 % on HC mix down (pass >= 80 % of floor) | Leap [taken Sun 01:46; 01:47 PASS: HC mix down 5.11 -> 2.59 ms per check (88 % of floor, paired x1.97), invariant, error = MLX's; mix up no gain (keep MLX form c); ple key x1.18. Exact 16-row check dense ~18 ms. data/bmm30.txt] |
| L1-30 | 125B: do fewer bits buy time in RAM? Native-layout tensor-unit expert kernel at 4 / 3 / 2 bits (3-bit = v3's MSB-3 as 2+1 bit planes) at 16 rows; the 72-77 ms check assumes 3.5-bit experts at 4-bit's 82 % of floor, the 350 recipe needs ~2-bit | published style (BitBLAS, FLUTE, MLX 2-8-bit qmm), untested on M5 tensor units | GPU ~5 min: test_low30.py fair A/B vs MLX gather_qmm at 4 / 3 / 2 bits, error, invariance | 2-bit < x1.4 vs 4-bit native at 16 rows (pass 2-bit >= x1.6 and 3-bit >= x1.2; ideal x1.80 / x1.29) | Leap [taken Sun 02:55; 03:00 KILLED: 16 rows 2-bit x1.22, 3-bit x1.08 vs 4-bit native (bytes allow x1.80 / x1.29); bitwise-invariant, same error; MLX's own 2-bit is slower than its 4-bit (x0.51). Each expert costs ~10 us of non-byte work at 16 rows -> 4-bit already meets it; 3-bit check experts ~33 ms (not 28), check ~77-82 ms. 350's ~2-bit term needs a kernel at ~5.5 us per expert (today 9.9). data/low30.txt] |
| L1-30b/c | 125B: where the ~10 us per 2-bit expert goes (profile), then read-side fixes: K-paired codes, exact grid, prefetch | kernel tuning | GPU ~10 min: test_low30b.py / test_low30c.py, bitwise vs base, fair A/B | all fixes combined < +15 % at 2-bit 16 rows (pass >= +30 %) | Leap [Sun 03:08 profile: 4-bit memory-bound (reads alone 11.5 of 11.9 us); 2-bit reads alone 7.0 us (78 % of floor) + ~2.8 us non-overlapped multiply / unpack. 03:13 KILLED by letter: all combined +13 % (prefetch hurts, x0.94-0.98); K-pairing alone +15.7 % at 2-bit -> 2-bit x1.38 vs 4-bit (8.5 us per expert, 64 % of floor); 3-bit x0.97, 4-bit x1.00; exact grid +1-5 %. Check experts: 4-bit ~35 / 3-bit ~33 / 2-bit ~25 ms. data/low30b.log, low30c.log] |
| L1-31 | 125B: the GDN core (conv, norms, recurrence, gated norm) per exact 16-row check, as the engine runs S > 6 | published pieces (fla, FlashInfer GDN, MTPLX fused rows), untested here at 16 rows | GPU ~2 min: gdn31.py | stock core < 4 ms per check (pass >= 8) | Leap [taken Sun 03:13 EET; 03:15 BETWEEN: core 4.07 ms (prework 1.51, recurrence T16 2.58, gated norm 0.61; floor 0.97) + capture-commit replay 2.10 = 6.2 ms per check (~8 %); a fused exact program saves ~3-4 ms (~4-5 %). MTPLX's fused conv+norm runs at S 16 (1.09 ms) but is not bitwise. data/gdn31.txt] |
| L1-32 | 125B: intra-request parallelism: one chat decodes k parts of its answer at once (k streams x 16-row checks in one pass); MoE experts are shared across the parts | published (Skeleton-of-Thought, APAR, Hogwild!, PASTA, Multiverse); the MoE expert-union economics here is ours | CPU: par32.py (expert union on the routing traces), par32b.py (24 GB LRU replay) | reads per accepted token at 4 x 16 > 0.75x one chain (pass <= 0.55x) | Leap [Sun 03:17-03:19 EET, CPU. RESIDENT: between, near pass: 4 streams of one answer 0.59x DRAM expert reads per token (top-8 0.60x; unrelated 0.79x), ~2x end to end est. 24 GB: KILLED: 4 streams RAISE SSD misses per token 1.7-1.9x at C 192 (1.05-1.18x at C 288): the pool thrashes on the 2.35x working set. Resident-only lever (45 GB needs <= ~2.65 b/weight all-resident). data/par32*.txt] |
| L1-32p | **PARKED, NO STAGE 1 (NewIdeas 09:01)**: MoonShots' own par45 (data/par45.txt; re-read) puts 350 at >= 67 % of output tokens inside parallel parts (88 % with row-scaled non-byte time), while the published router numbers put real use far below that: ParallelPrompt 10.3 % of real prompts have independent parts, SoT-R sends math / code / writing to normal decoding (MoonShots predicts f(code) < 20 %). So it is not a 350 route on any realistic mix, and stage 1 (~1 h of 35B GPU + a harness) would only confirm a pre-registered kill. On file as a possible chat fast mode (~1.6-2x on the ~10 % of prompts with independent parts, i.e. ~5-10 % on a mix; changes the answer's structure): Kamen's call, not proposed now. Side number worth keeping: the single-stream non-exact 45 GB stack (TRL base + dense diet, lean 15 ms loop) = 233 tok/s at 9.5 tokens per 16-row check, so 350 there needs ~1.5x tokens per check. **125B PARALLEL-ANSWER FAST MODE AT 45 GB (design only; MoonShots Mon 08:31-08:36; follows L1-32; the route if DS kills the drafter; adopting it is Kamen's call, this row only prepares the question)**. (a) SCHEME: only Skeleton-of-Thought with its router (SoT-R) fits a 125B we can't train. The model answers YES / NO to 'independent parts?', writes a 3-8 point skeleton, then k <= 4 streams each expand one point, seeing prompt + skeleton + their own text only. A user's own list of sub-tasks is the special case (ParallelPrompt). Hogwild! (shared KV, RoPE-rotated views) and Group Think (token-interleaved masks) need cross-stream visibility that only attention gives, and 36 of our 48 mixers are GDN (see (c)). APAR / PASTA / Multiverse / APR / ParaThinker need training. (b) PUBLISHED QUALITY: SoT (ICLR 2024, 12 models, GPT-4 judges): not worse than normal in ~60 % of answers, wins ~ losses. Good on generic / knowledge / common-sense / roleplay / counterfactual (1.89-2.33x there, up to 2.39x), poor on math / coding / Fermi / writing (later steps need earlier details). Less coherent, answers 1-2x longer, some open models misread the skeleton / expansion prompts. SoT-R restores math / code / writing by sending them to normal decoding. ParallelPrompt (NeurIPS 2025, 358K WildChat + LMSYS prompts): 10.3 % of real prompts have independent parts; quality kept ~92 % (reading comprehension), 82-85 % (NER, keywords), ~76 % (creative); dependency blindness in ~25 %. Hogwild! (training-free; QwQ-32B, Qwen3 1.7B-235B, DeepSeek-R1): 2-4 workers reach LIMO / OlympiadBench / LiveCodeBench / AIME'25 accuracy in fewer sequential steps, but 1.7B / 4B models get distracted and workers don't always notice duplicated work. Group Think (zero-shot Llama 8B / 70B): toy enumeration, Floyd-Warshall and programming tasks only. Nobody reports IFEval; independent parts can't see global constraints (word / paragraph counts, required endings, formats). (c) ENGINE: k streams are k ordinary sequences forked after the skeleton. Copy the 36 GDN states (48 heads x 128 x 128 fp32 = 113 MB per stream); share the 12 attention layers' prefix KV (24 KB / token). One verify pass of k x 16 rows: per-stream drafts, block-diagonal attention, a stream-batched multi-row GDN kernel (L1-15), the MoE layer on the union of all rows' experts (the saving), per-stream acceptance and GDN rollback. Streams retire as parts end; parts are joined in skeleton order. Each part can stay bit-exact to its own serial decode (batch-invariant kernels), but the answer text changes: an opt-in mode beside the exact modes. THE GDN PROBLEM WITH INTERLEAVED WRITERS: attention sums over cached (k, v), so another writer's tokens can be shown at any position for free (Hogwild! rotates queries, Group Think masks). A GDN state S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T is an ordered product with nothing to re-index. One shared state mixes interleaved writers word by word (a recurrence has no mask). Private states that absorb the others' steps must re-run their own current step after every insertion (k (k - 1) x step-length extra GDN rows per check), using (k, v) computed in the writer's context. Visibility through the 12 attention layers alone is untested anywhere. So: independent streams only. (e) SPEED (~/MoonShots/scripts/par45.py, data/par45.txt). Inputs: L1-32 unions per layer (top-8 54.6 -> 131.5 at 4 x 16; top-10 66.1 -> 155.5); the '! 350' row's 1 x 16 routed bytes (4.9 GB) scaled by the union; dense diet 2.1 GB read once per pass; + 2 x 113 MB GDN state per stream; 280 GB/s; 9.5 tokens per stream per check. 1 x 16: 7.2 GB, 40.8 ms, 233 tok/s. 4 x 16: 14.8 GB, 67.9 ms, 38 tokens = 560 tok/s in lockstep (top-10 568) with the lean 15 ms loop; 420 if non-byte time grows to 37.5 ms at 64 rows (L1-13: the 35B's GDN mixers already take 12.2 ms vs a 2.0 floor at 16 rows). Experts see 3.9 rows each instead of 2.3, so dequant-heavy formats must stay byte-bound at ~4 rows. Per answer (router + 40-token skeleton serial, 4 parts of 0.75-1.25x the mean, streams retire): 1.97x visible tok/s (1.60x with row-scaling); if SoT answers run 1.3x longer, time to answer only 1.54x (1.24x). On a mix, 350 needs 67 % of output tokens inside parallel parts (88 % with row-scaling). With exact v3 weights streamed instead (C 288), L1-32b's 4 streams keep SSD misses per token at 1.04-1.18x, so code stays SSD-capped at ~270 tok/s (24.5 x 2.16 MB per token at 14.2 GB/s). RESIDENCY: TRL 2.73 b base set 41.2 GB + dense diet 2.1 + 4 GDN states 0.45 + 8K tokens of KV 0.2 + drafter + buffers = ~45-46 GB, at the line; the mode adds ~0.6 GB over one stream. TRL's own costs (~57 GB of disk, its decode kernel) belong to the 45 GB stack | design (published scheme; ours: the GDN constraint and the MoE pricing) | Quality first, no engine work: each stream is an ordinary sequence, so the mode's TEXT comes from serial calls (router -> skeleton -> k expansion calls -> join). STAGE 1, 35B tool model (Qwen3.6-35B-A3B 4-bit, same gggA hybrid, ~106 tok/s, greedy): arm A normal, arm B SoT-R. Rows: the Eval suite's decode rows (IFEval strict prompt-level, HumanEval+ pass@1, math) + the prompts behind the expA traces (code / math / chat / bulgarian). Per family: router fire rate, f = share of output tokens inside parallel parts, sequential-step ratio, length ratio, task deltas by the suite's decode rule (delta > -5 points, exact McNemar p >= 0.05). ~1 h of 35B time, share lock, no downloads. STAGE 2 only if stage 1 passes: the 125B at small N (rows where the 35B fired + 20 IFEval rows), same arms, ~1 h on today's engine | KILL if arm B fails the suite's decode rule on IFEval, HumanEval+ or math, OR f < 50 % token-weighted over the 4 trace sets (the mode then caps near ~310 tok/s at the lean loop; 350 needs 67 %). 50-67 %: it stacks with a drafter but can't make 350 alone; NewIdeas decides. Quality passes but f is high only on chat: a chat-only fast mode, Kamen's call, not a 350 route. Pre-registered prediction from SoT-R: f(code) < 20 %, f(math) ~0, f(chat) 40-70 %, so KILL on f for the 350 route | MoonShots (design); stage 1 needs NewIdeas' GO and the Eval suite's 35B harness |
| L1-33 | 125B: screened argmax-only lm_head for verify rows. A 2-3-bit screen of the 4-bit head (248320 x 2560) plus a per-row error bound; read exact 4-bit rows only where the bound can't exclude them. Greedy argmax provably identical, so verify stays exact (logits themselves are not needed except when the repetition guard is armed). Today the head costs 4.3 ms per 8-row check (MLX's exact band) or 2.2 ms as 16-row qmm | published (vLLM PR #60566 FP8 screen; exact MIPS / FEXIPRO), new here: low-bit screen + bound on Apple GPU | CPU first on real 125B hidden states + head (MoonShots IDEAS #2 / data/screen_head.txt did a 2B proxy): fraction of vocab rows needing the exact read per position; then a GPU screen kernel at 8 / 16 rows | exact rows > 25 % of the vocab at the median position, or the screen + exact reads > 50 % of the head's 16-row qmm time (1.1 ms) | open (MoonShots proposed, Sun 04:05) [taken: Leap, Sun 04:12 EET, pre-registered in ~/Leap/TABLE_350.md L1-33; PARKED 04:13: NewIdeas' order puts it after B1 and B5] |

## LINE 2: better < 12 GB, ranked
| id | idea | tag | cheapest decisive test | kill line | owner |
|---|---|---|---|---|---|
| L2-01 | UGB-0.5 (b16 on confident passes, exact elsewhere) as the 12 GB fast mode | new (UGB) | Eval suite core verdict (running ~02:45) + r2 | fails any rule | Eval suite / Fit12 |
| L2-02 [Fit12 04:05: load transient SOLVED, exact (hc8 on lazy bf16: C_m 59 load peak 12.17 -> 10.84 GB); p8k prefill still 12.35 (s_moe 2048: 11.99). DONE 06:35 (B2): the crash was Fit12's harness (trims installed before load_b12's _env); fixed, C_m 59 + both trims = 22.3 / 13.7 tok/s code / prose (+5 % vs U1), shas = U1, peaks 10.48 load / 11.76 at 2.6K / 12.43 at 8K -> fold GO, < 12 GB to ~2.6K-token prompts] | Cut the load / long-prefill transient so C_m 59 fits < 12 GB (+6-7 % exact) | untested here | peak footprint at load, 2.6K and 8K prefill with the transient cut | C_m 59 peak > 11.95 GB | Breakthrough (in progress) |
| L2-03 [Fit12 04:05: bitwise, -0.36 GB every phase. DONE 06:35 (B2): tok/s unchanged (f59emb 22.3 / 13.6 vs 22.5 / 13.8), part of the C_m 59 fold] | Stream the untied 4-bit embed table (358 MB, one row per token) -> +3.5 slots per layer (~+3 % exact) | untested here | build the row reader; shas equal; footprint + tok/s | < +2 % or shas differ | Fit12 |
| L2-04 [Fit12 04:05: at the 8K prefill peak the extra is MLX buffers (IOAccelerator 10 GB + 467 MB owned-unmapped), malloc+VM only ~0.3 GB: no non-MLX slack to reclaim] | Reclaim the ~1.0-1.4 GB non-MLX footprint (vmmap breakdown first) for slots | untested here | vmmap / footprint breakdown at 2.6K and 8K | < 0.4 GB reclaimable | Fit12 |
| L2-05 [Fit12 02:40: ALREADY TESTED: the engine's BIGMOE_SPARE is exactly this ring (promote on use, no victims); Breakthrough SCOREBOARD 455-460 + 609-619: loses in every form at equal RAM. Propose KILLED] | Victim-free staging ring for JIT prefetch at C_m 51 (J1: victims cost ~80 extra misses / pass) | new here | build + A/B on 4 prompts, shas equal | < +5 % | Fit12 |
| L2-06 [DONE 07:34, Breakthrough: 8-bit KV -0.5..-0.7 GB at 64K, +9 % per pass, tf KLD below the noise floor; 64K NOT under 12 GB: the PREFILL PEAK (13.3-14.0 GB vs steady 11.3-11.8) binds; task check queued] | 8-bit KV for long agentic context (64K under 12 GB; today 16-20K) | published (KV quant), untested here | tf KLD at 16K / 32K contexts with 8-bit KV vs bf16 | KLD over the strict targets at 32K | Breakthrough |
| L2-06t | TASKS (13:10, Breakthrough SCOREBOARD): bf16 perfect at 32K and 60K (magic numbers 4/4, variable tracking 5/5 with no distractors named, uuid 1/1). 8-bit KV at 32K: answers right (4/4, 5/5, 1/1) BUT it also named all 3 distractor-chain variables, which bf16 never did. So 8-bit KV is NOT free at 32K; the 60K 8-bit arm was lost to the Mac's idle sleep (re-queued). No adoption before the 60K arm + more distractor items | - | - | - | Breakthrough |
| L2-06t-dx | **2K SELF-TEST FAILED, RUN STOPPED (Breakthrough Mon ~14:00, 1.5 min of GPU); FIX GO (NewIdeas 14:08)**: the cache restore is now bit-exact (fingerprints equal), yet item 0's answers differ: state outside the cache. Likely expert-pool residency: the AR path adds cold ranks 2-9 late (B12LL), so its output depends on what is resident, by design. Fix GO: give every item the post-document pool state. If PFR's re-created pools start empty, empty the pools before every item. If they start warm, snapshot and restore the residency instead. The 2K self-test (prefill, answer, restore + pools, answer: ids equal) must pass before the long runs; item 0 is re-asked last in every run; same lines (a)-(c). Decode tok/s now starts cold per item: information only. **INVALID RUN Mon 09:44-10:01 (Breakthrough SCOREBOARD 'Queue L2-06t-dx'), RE-RUN GO (NewIdeas 12:30)**: the fork was not exact (item 0, re-asked last, answered differently in 3 of 4 runs) and the 48-token cap cut every 60K answer, so no verdict on 8-bit KV. Re-run with Breakthrough's fix: (1) deep-copied snapshot; (2) a 2K snapshot/restore self-test first (prefill, answer, restore, answer: ids must match, else stop and report); (3) cap 160 tokens, same ask; (4) item 0 asked first and re-asked last in every run: a run whose re-ask differs is INVALID; same 12 items per arm, same lines (a)-(c). Priority lower than before: PFR-a makes 64K fit with bf16 KV (memory pending Fit12's a64m), so 8-bit KV now matters for speed (+9 % per pass at 64K) and past ~70K. Original: 8-bit KV distractor check (Breakthrough, proposed Mon ~03:08; GO NewIdeas 03:10, after its copy-drop run, one lock ~15 min): per length (32K, 60K) one document with 6 interleaved 4-step VAR chains (distinct values, unique names), each chain asked as its own item from the same post-document state (forked cache, no earlier Q&A in context), so 12 items per arm; arms bf16 and kv:64 in the shipped env + lane; decode tok/s at 60K reported as information (the +9 % per pass) | published (KV quant); the distractor failure is ours (L2-06t: 8-bit named a whole distractor chain at 32K, bf16 never) | kv8_long.py KV8L_SET=_dx | kv8 passes only if over the 12 items (a) gold names within 2 of bf16's, (b) distractor names <= bf16's + 2, (c) no item names a (near-)whole other chain where bf16 doesn't; else 8-bit KV is not adopted for long context. If Fit12's PFR A/B finds an exact memory-only PFR, 64K fits with bf16 KV and 8-bit KV matters only past ~70K or for speed | Breakthrough |
| L2-07 | B12L's truncations (4/5; faster than B12LL: 33.7 vs 21 code): decode-time run-on stop rule (ThinkingCap card lesson, read only) | new here | CPU on the 17 truncated rows: which ones a stop rule would end correctly | < 4 rows fixable | Breakthrough |
| L2-10 | Cut the long-context PREFILL PEAK (the 64K-under-12-GB blocker: 13.3-14.0 GB vs steady 11.3-11.8) | untested here (added 07:34) | footprint + MLX-active sampler during a 64K prefill to locate the peak (indexer scores at s_attn 512? KV growth copies?), then s_attn / s_moe / one-pass growth | < 1 GB of the peak attributable to anything we can change | Breakthrough (after its task check) |
| L2-10p | PRE-REGISTERED (13:10, Breakthrough SCOREBOARD f8cc5b5). Finding: b12_prefill never enters mtplx's prefill attention phase, so the sparse lane never engages past 32K; each 512-row chunk builds a dense [24, 512, T] score plane (~1.8 GB transient), and freed-not-reclaimed GPU memory adds up to ~2.2 GB (per-chunk KV growth copies). PASS: the N = 65,000 plain-path prefill peak < 12.0 GB (MTP path reported beside), one-pass KV growth sha- and logits-exact vs base, 32K + 60K bf16 tasks unchanged (4/4, 5/5 with 0 distractors, 1/1). KILL: < 1 GB of the peak attributable to what we can change | GPU batch pk64.py, 6 variants ~36 min | as left | Breakthrough |
| L2-10b1 | BATCH 1, NOT A VERDICT (13:45, Breakthrough SCOREBOARD ecf812c): it ran without the shipped BIGMOE_TIER_PF_GROUP=64, so the MoE phase built a ~1.4 GB full-union expert scratch per layer call and the footprint peak stayed ~13.7 GB. The L2-06 memory runs (kv8_tf, kv8_long) lacked it too, so the '64K prefill peak 13.3-14.0 GB' overstates the shipped config. Found anyway: the sparse prefill lane engages (all 756 chunk x layer calls past 32K on metal_selector + flash_kernel), attention-phase MLX peak 13.42 -> 11.43 GB (-2.0), 64K prefill 1.30x faster (306 -> 235 s; MTP 297 -> 232). sha differs from b12_prefill's dense path, BUT the suite's reference path (MTPLX generate_ar) already enters the prefill phase, so the lane may be the REFERENCE behaviour and b12_prefill the deviation: compare lane output with generate_ar's before calling it non-exact. ANSWER 14:07 (Breakthrough, code + a crash): in the suite's config the lane is NOT the reference. generate_ar chunks only with MTPLX_SUSTAINED_PREFILL=1 (set by MTPLX's profiles, not by load_b12 or ~/EvalSuite), so on B12 it prefills the whole prompt as one dense forward ([24, T, T]; a 65K attempt asked Metal for 203 GB). The suite's non-MTP B12 rows at RULER 16,384 build a ~13 GB score plane (Eval suite told). lane_ref now judges against generate_ar one-span (the real reference) and chunked; the shipped s_moe 4096 lane can't be bitwise anyway, so the 32K / 60K task check is the guard. One-pass KV growth exact, no gain. 64K decode uses a dense mask in all 12 QSA layers (MTPLX_QSA_FLASH=1 opt-in, unset). The MTP head costs ~1.7-1.8 GB: the main reason MTP at 64K misses 12 GB. Batch 2 (shipped env, day_pk64b.sh) queued behind BigMoE | - | - | - | Breakthrough |
| L2-10r | **COPY DROP FAILED, L2-10 STOPPED (Mon 08:14, Breakthrough data/pk64/results.txt, g64_lane_sc vs the same-morning control g64_lane_r2; read by NewIdeas 08:20)**: exact (sha / logits 56c4e038e8 / 1ccb5e8c59 = lane) but nothing saved: MLX peak 11.62 GB in both arms, footprint peak 13.40 GB vs the control's 12.80 (line <= 12.39), max MLX active 12.23 vs 12.14, prefill 302 vs 241 s. By the pre-registered rule L2-10 stops; BIGMOE_TIER_PF_SCATTER stays default off. The control itself drifted from Sunday (12.69 -> 12.80 GB, 207-212 -> 241 s), so single-run gaps under ~0.15 GB are not readable on this path. Without the copy drop, lane + 8-bit KV at 64K projects ~11.9-12.0 GB (QSA KV 1.60 -> ~0.80 GB), if L2-06t-dx passes. Earlier: RESULT (14:45, Breakthrough SCOREBOARD 47e090f, shipped env): line NOT met, not killed. 64K plain path: base 13.58 GB (prefill 312 s); lane 12.69 GB (prefill 212 s, 1.47x faster); lane + s_moe 2048 12.32 GB (exact vs lane, prefill 298 s); MLX cache 1 GB worse (13.67); persistent group buffers no gain. MTP path 15.25 -> 14.53 (the head's ~1.8 GB). Remaining peak = the post-MoE phase: MLX active 11.2-11.6 + ~1.0 GB just-freed GPU memory, mostly MoE output copies (~4 x 210 MB per layer at s_moe 4096). lane_ref: no pair bitwise; b12_prefill vs generate_ar already differs by 0.69 max |dlogprob| (first token top-20) with the lane off, and lane / flash-skip / one-span move it 0.5-1.5: the lane is the same size of change as the accepted b12_prefill difference, so the 32K / 60K task check is the guard and adoption is Kamen's call. Flash-skip decode at 64K: ~-7 % median, inside noise. DECISION (NewIdeas 14:45): bank lane 12.69 GB / 1.47x prefill; next the lane task check (32K + 60K, off vs on) + the lost 60K 8-bit arm; then ONE exact lever: drop 2 of the MoE output copies in _prefill_grouped (pass: >= 0.3 GB off the 64K peak, sha identical; else stop). s_moe 1024 and non-exact s_attn 256 skipped. If 8-bit KV passes its 60K + distractor check, lane + copy drop + 8-bit KV projects ~11.7 GB | GPU ~25 min + a small engine change | as stated | Breakthrough |
| L2-10t | LANE TASK CHECK PASSED (15:14, Breakthrough SCOREBOARD 6207af6, shipped env): lane off vs on word-for-word identical at 32K and 60K (mq 4/4, vt 5/5 + 0 distractors, uuid 1/1); 60K prefill 268 -> 210 s (1.28x), 60K peak 13.40 -> 12.70 GB. 8-bit KV at 60K: identical to bf16, peak 12.86 vs 13.84 (first-run env); 8-bit distractors 3 at 32K, 0 at 60K (one item each): weak, needs more distractor items. Lane BANKED; adoption on FOR_KAMEN.md. Next: the MoE output-copy drop (pass >= 0.3 GB off 12.69, sha = lane) | - | - | - | Breakthrough |
| L2-11 PFR | **PORTED, DEFAULT ON (Breakthrough Mon ~14:00, bt_b12 445e301)**: its own A/B on v3l, AR path: p8 = r8 and p64 = r64 on KV, logits and sha32 (= Fit12's v3p refs); prefill 27.5 -> 18.9 s (8.5K), 265 -> 208 s (64K); prefill-phase peak 11.05 -> 9.27 GB, 13.54 -> 11.08 GB. Defaults now B12_ENGINE=v3l (= v3i bitwise) + B12_PFR=1 on prefills > 4096 tokens; opt out B12_PFR=0 / B12_ENGINE=v3i. b12_prefill now asserts s_moe % s_attn == 0. **a64m PASS, MEMORY CLOSED (Fit12 Mon 12:30, PLAN.md 'PFR-a memory RESULT')**: PFR-a at 64K without the KV hash: logits / sha8 / sha32 = r64 (logits file byte-identical); lifetime peak 11.77 GB (line 11.95), decode phase 11.75, prefill phase 9.32. The A/B arms' +0.34 GB was the hash. So a 64K prompt now runs exactly within 12 GB, prefill 261 -> 196 s. PF2 (layer-major prefill) would add <= ~8-10 %, at 64K only: parked (NewIdeas 13:46). Fit12's per-prompt C_m (size the decode pools from n when PFR restores them; short / mid prompts have ~2 GB of decode headroom): GO as a pre-registered test (NewIdeas 13:46). Sizes 2.6K, 8.5K and the largest n that gets C_m > 51; C_m chosen so the lifetime peak stays <= 11.95 GB. MTP path: logits hash + sha32 must equal C_m 51's at every size. AR path: sha reported as information (it depends on residency by design). PASS: all MTP hashes equal, every peak <= 11.95 GB, decode >= +4 % at 8.5K. KILL: any MTP mismatch, a peak > 11.95 GB, or < +2 %. Port after PFR's, Breakthrough's. **PFR-a PASS: EXACT AT 8.5K AND 64K (Fit12 Mon 11:31; PLAN.md 'PFR A/B stage 2 RESULT', 'PFR-a RESULT'; data/pfr/ab_*.txt)**. Stage 1 (8,474 tokens): all arms equal (KV ecd389371d, logits c57fe16fd1, sha32 142abd5d7e); PFR prefill 27.4 -> 18.8 s. Stage 2 (64K): c64 (release + 4096 chunks) = r64 on every field (KV 2227a92414), so release / restore is exact at 64K too; s64 (16250 chunks, no release) = T2's pfr64 exactly and repeats identically, so T2's break was the chunk size alone. Cause: b12_prefill cuts each s_moe chunk into 512-token query chunks from the chunk's own start, so s_moe % 512 != 0 shifts the query grid (KV differs from layer 0, before any MoE). PFR-a (release + 512-aligned chunks of 16384 x3 + 15848): KV, logits, sha8, sha32 = r64, logits files byte-identical; prefill 261.1 -> 196.4 s (-25 %), TTFT -64.5 s, prefill-phase peak 13.70 -> 9.32 GB. No arm shows decode-state dependence (pools re-created, bitwise equal), so no pointer-swap fix is needed. CORRECTION to '64K under 12 GB': with PFR the 64K peak is set by the decode phase, 11.62 GB in the clean T2 runs but 11.97-12.09 GB in the A/B runs (a64 lifetime 12.12), likely the ~3.2 GB float32 KV-hash copies (MLX active 11.15 GB in all); a clean a64m (no hash, line <= 11.95 GB) is pre-registered and queued. **PORT GO (NewIdeas 12:30) to Breakthrough**: Fit12's rule (release for n > 4096; k = ceil(n / 16384), chunk = 512 * ceil(n / (k * 512)), tail > 8 tokens); also an assert (or a query grid on absolute positions) in b12_prefill so a chunk size off the 512 grid cannot silently break exactness again; before PFR becomes the default, Breakthrough's own A/B in its shipped env at 8.5K and 64K must match PFR-off on KV hash, first-step logits and sha32 (bit-exact, so default-on needs no Kamen decision); no '64K under 12 GB' claim until a64m. Earlier: **lanepfr64 (T2 arm 4, finished Mon 08:04 after the silent-night freeze, timings void; Fit12 PLAN.md; read by NewIdeas 08:20)**: lane + PFR is not exact either: sha8 56c4e038e8 = lane64's, but logits 6d30f0597c != 1ccb5e8c59 and sha32 3513442745 != aef9384fda; prefill-phase peak 9.73 GB, lifetime peak 11.96 GB. The A/B (8.5K, r8 / s8 / c8 / p8 + KV hash + repeats) is queued #2 for the GPU (behind TrainLab turn21). **T2 KILL on the sha line, Mon 01:19 (Fit12 PLAN.md 'PFR T2 RESULT', data/pfr/{ref64,pfr64,lane64}.txt; verified by NewIdeas 03:02)**: at 65,000 tokens pfr64 != ref64 in tokens (sha32 bba3d6e0f5 vs dc67a9270b) AND first-step logits (68ddb5344e vs 4dcabb778f); ref64 and lane64 reproduce Breakthrough's pk64 bit for bit, so harness and engine are sound. The other lines would pass: prefill 259.9 -> 208.8 s (-51.1 s, line 25 s), prefill-phase peak 13.56 -> 9.39 GB, lifetime peak 13.72 -> 11.62 GB (< 12.0): the memory result is real, the exactness is not. **T1 downgraded**: it compared token shas only (pfr64's first 8 tokens equal lane64's while its logits differ from both). Cause test pre-registered (Fit12 PLAN.md 'PFR A/B pre-registration', 01:22): N 8,474, r8 ref / s8 s_moe 8474 with pools resident (A: chunk size) / c8 PFR chunks of exactly 4096 (B: release + reset) / p8 PFR cap 16384 (both), equal = logits hash AND sha32; stage 2 at 64K only if 8.5K is all-equal. GO after 08:01 (NewIdeas 03:02) with two adds: (1) a hash of the KV cache at the end of prefill in every arm, which splits prefill math from decode state (s8 ends its prefill with different pool contents, so s8 != r8 alone would not prove A); (2) one repeat of any arm that differs (if it differs from itself, the restore has a race, e.g. a slot read before its load lands; pfr64 decoded 9 % faster than ref64). Nothing ported; Fit12 told Breakthrough to wait. T1 23:38 passed its lines on TOKEN shas (82995cdb4e / 0b0d91cd43 in all 3 arms, data/pfr/{ref51,pfr51,pfr59}.txt): 8,474-token prefill 29.59 -> 21.22 s (286 -> 399 tok/s, +39.4 %; line +25 %), prefill-phase peak 11.62 -> 6.42 GB (line -3.0), lifetime peak 11.62 -> 10.96; C_m 59 after a PFR prefill peaks 11.79 <= 11.95 (shipped C_m 59 prefill: 12.43) so the decode headroom stands; TTFT 29.85 -> 21.68 s; 2.6K +0.33 s (port: n > 4096 only); cold start ~+0.12 s; swap unchanged. T2 at 64K (plain path, + lane if it fits: does 64K fit under 12.0 GB?) GO 23:39. PREFILL WITH THE DECODE POOLS RELEASED (Fit12 ~22:00, pre-registered in ~/Fit12/PLAN.md 'PFR pre-registration' before any build; GO for T1 by NewIdeas 23:31). Line-2 decode anatomy at C_m 51 (~165 ms / pass): SSD 108.7 ms (66 %, at the burst floor), compute gaps 39.2, drafts 8.3, edges 8.7; exact decode levers left are small. But b12_prefill is chunk-major (each 4096-token MoE chunk reads every expert of every layer: ~72 GB / chunk, ~5.4-6.3 s measured per extra chunk-read), and the idle decode pools (6.03 GB) set the 12 GB prefill peak. PFR frees the pools before a long prefill, runs s_moe = min(P, cap), then rebuilds the pools for decode. Claimed exact by construction (s_moe and pool size bitwise): refuted at 64K by T2, cause under test. Predicted: 8,474-token prefill 30.8 -> 18.2-20.1 s (+54..+69 %); 64K lane 212 -> ~148 s; 8.5K prefill-phase peak ~6.3 GB (from 11.51); then C_m 59 at 8.5K (+~5 % decode). Follow-up PF2 (layer-major prefill, experts read once per prompt) | exact; untested here (memory-phase reuse) | T0 CPU audit + logs PASSED (+54..+69 % vs line +25 %; 6.03 GB freed vs line 3). T1 GPU (one quiet lock, ~15 min): 2.6K + 8,474-token prompts, arms ref / PFR C_m 51 / PFR + C_m 59; also TTFT + first-32-token tok/s and swap (NewIdeas adds, information) | any decode sha != ref; 8.5K prefill < +25 %; prefill-phase peak not >= 3.0 GB below ref -> KILL; C_m 59 peak > 11.95 GB drops the headroom part | Fit12 (b12_prefill and the shipped env stay Breakthrough's) |
| L2-08 [Fit12 03:20: KILLED on existing data, ~/NewIdeas/data/samelayer/results.txt + union_forecast.txt: under MTP every verify row waits for its whole top-10, and the same-layer top-10 forecast recalls 0.735 (code) / 0.82 (ifeval+multi) of it, < 0.85; top-16 reaches 0.86 / 0.93 but at +0.6x extra reads (16-token union), in a regime where the SSD is the binding cost (D1k, J1 victims); ceiling even if perfect = the attention half, ~0.45 ms x 48 = <= +15 %] | Overlap compute (43 ms / pass) with loader stalls (107 ms) by issuing the next layer's waited reads from a same-layer forecast | published (pre-gating), untested at 12 GB | CPU: recall@2 of the same-layer pre-attention forecast for ranks 0-1 on traces (data/samelayer) | recall@4 < 0.85 for the waited set | Fit12 |
| L2-09 | Compression / fewer bits / shared expert base | - | Fit12 C1 | KILLED (14 % entropy needs 150-200 G sym/s; re-quant 1.36x error; experts independent) | - |

## Sun 1 Nov 06:1x: night results and routing
- R-56c PASS (EvalSuite 6c53085): REF2-T's live loop on the gate bit-exact at full length (36/36, 15,525 tokens, 14/14 past 512 keys). -> REF2-T + gate is the 45 GB default, REF2b the exact 4-bit mode; BigMoE wires it and sizes C to the 45 GB budget (R-56c peak 33.4 GB).
- R-55 gate ADOPTED in BigMoE (8e2b22e, default MAXK 160); R-52 PF_REUSE ADOPTED (ce25fe6, official 24 GB peak 22.97 GB).
- R-60 / R-60b (BigMoE, 24 GB): D-22 b3 head kept +5.2-7.6 %, tok/s +0.4-0.6 % (SSD-bound per token at C 128): no default.
- D-22 shipped 45 GB line A-B-A (TrainLab 9522477): k7 b3 code +6-15 %, math +3-8 %, chat -8..+2 %; adapt mode ~flat; all shas equal.
- bytes45 (BigMoE 812f327): GB per kept token 1.56 (4-bit r16) -> ~0.85 (REF2-T + truncation); the walls' scoreboard (RANKED header).
- R-61a GDNX ADOPTED into REF2 (Leap 129c86e): exact, -1.9 % cycle. Leap: REF2c bundle first (+ K-60), one regen + same-scores.
- R-64T TREE-PRICE (new, Breakthrough, CPU): best 16-node EAGLE-2-style tree vs 16-chain from Leap's DFlash2 top-K logs; PASS >= +10 % kept at equal bytes per kept token, KILL < +5 %.
- R-63S STREAMS BETWEEN (MoonShots: 3 streams -36 % bytes per token, aggregate only). B-1 KILL on paper; lossless records BETWEEN (~8.5 %, 4-7 cores to decode); R-45V v3q BETWEEN (-16 MB, not adopted).

## Sun 1 Nov 07:5x: morning results and routing
- BigMoE wired the 45 GB default (R57_PRESET=45, 1f7c243). Leap's correction: lifetime peak 38.0 GB at C 224, ~0.13 GB per C unit -> 45 GB = C ~278. Tonight's C45 turn: C 224 / 240 / 256 (+272), the LRU + popularity-prior arm (-9.6 % misses, exact; GO) and the OOO arm.
- R-69b OOO + TIER_COMBINE (Leap / BigMoE): exact on 4 prompts, cycle -11 %, steady +13 % (shared C 160). GO: EvalSuite full-length audit vs results/ref2t; adopt on PASS.
- Leap: K-60-T same headroom in 3c; K-61 bit-identical 3c CLOSED; K-62 NAX4/NAX6 KILL at the 45 GB point (parked, wide checks); R-62b att_single INV PASS (-5 ms); R-66 FATT16 BETWEEN; R-67 KILL; R-68 in_proj -0.5 ms; R-65 census 3,944 kernels per check. REF2c frozen -> EvalSuite regen + same-scores tonight; fused router ships alone.
- MoonShots: M-10 SIBLINGS (sibling rows 0.84-0.91x a chain row's new experts: trees get no byte discount), M-11 / M-11b BETWEEN (quality knob), M-12 RARE2 step 1 BETWEEN (step 2 GO), M-13 MISSFLOOR KILL / sizing pointer (C 384 ~59 GB: not on this Mac), M-14 / M-14b EVICT45 BETWEEN (linear key -10.5 %); prior-art sweep -> RANKED #17 (LK losses, Spec-AUF to MTP Head).
- BIG sent 07:45 to the orchestrator: REF2-T + gate is the 45 GB default (50.9 steady, code 74; R-56c / R-58).
- M-5 FAIL on math (Fit12 07:4x, clean): live 12 GB default cf8bd7840a peaks 12.35 GB on math vs the 11.95 line (agent 11.53 OK); in-row rise, not creep. M-5b bisect GO (live / pinned 677c6558b6 / v3o / HOT_MB 1024 / exact); fix to Breakthrough.
- M-12 RARE2 step 2 KILL (MoonShots 2b9fd03): -4.3 % SSD MB per kept token at C 256 (lines PASS >= 10, KILL < 5); popular experts miss too. M-11 stays out.
- MTP Head 08:3x: R-61N CHAT-NT1 BETWEEN (+23 % per pass at rho 0.25 vs +30 line; non-exact); exact lead: 125B #1 in DFlash2 top-3 at 34 % of chat first misses (-> R-64T); #20 BIGDRAFT not local, paid ~$650-1,050 for net +8-20 % -> FOR_KAMEN proposal; LK losses dropped (T = 0), Spec-AUF in b5; D-23 queued behind EvalSuite's R-56d.
- R-70D DRAFTPF (new, PARKED until tonight's C45 spin logs): SP-MoE-style prefetch of the next check's experts during the drafting window (SSD idle ~5-8 ms per cycle, ~25-40 records = upper bound 13-21 % of a check's misses at C 256), predicted from the draft tokens (per-layer token -> expert tables, leave-one-prompt-out) into reserved slots. Pays only if the default stays SSD-bound with OOO on. Kill line: < 10 % of miss bytes prefetchable correctly within the window (cap45 replay).
- D-23 SWITCHDEF KILL (MTP Head 09:11): switch -2.7 % on the default (C 224, 18 prompts, exact); held BIG dropped. Lead: MTP-chain truncation rule (CPU).
- M-5b (Fit12 11:5x): the 12 GB math peak is v3p (R-53), not fold-fast (b12llff3 on v3o 11.61-11.65 GB); fix (engine back to v3o, exact) with Breakthrough; decision 5 stands.
- M-15 EXACT-NT BETWEEN final (MoonShots): exact-loop near-tie acceptance +14.0 % chat at rho 0.25 (PASS >= 15); served-model change.
- D-23b (MTP Head 11:5x, CPU replay of D-23's logs): the KILL is the verify width; with rows(t 0.3) on the MTP chain + early exit the switch is +12-19 % whole set (estimate). D-23c tonight 23:00 quiet (dfl vs mtp7e on 6 chat + 4 agent; PASS >= +10 %, KILL < +4 %). Night order: BigMoE C45 -> D-23c -> EvalSuite REF2c-T.
- 12:1x: TREES KILL at equal cost (Breakthrough 91b0a56); CALSTOP no gain; S45d withdrawn (t 0.3 optimal: rows 7.82 ms, fixed part 64-179 ms per check). Next for Breakthrough: FIXED-PART split (CPU, D-23 logs). Leap: REF2c frozen 91cd316 -> EvalSuite runner 36 tonight; K-73 exact kernel stack as an arm in BigMoE's C45. EvalSuite's M-15 AR offer declined (BETWEEN, parked).
- PMIN12 (Breakthrough 12:1x, 12 GB, rough shared): cutting the MTP chain at p_min 0.3 / 0.5: chat +12.9 / +18.7 %, all prompts +8.1 % (0.3), peak 11.79 GB; exact-mode token check running; adoption: exact mode on equal tokens, fast mode via EvalSuite decode check then Kamen.
- 12:4x: FIXED-PART (Breakthrough d18fd92): agent start-up 48 % (first check after long prefill 1.6-3.4 s), code SSD 60 %, chat constant 79 %. M-16 RUNAHEAD step 1 PASS (MoonShots) -> step 2 tonight (3rd in the quiet line); RUNAHEAD-BUILD review to Breakthrough. Leap: K-80 tonight (5th), day = chat constant part (drafter stack + fusions). 12 GB: live 8db71cc537 (v3p + F12_R53A=0, ~11.65 GB); PMIN12 PASS fast mode (chat +21 % at 0.5, non-exact) -> EvalSuite decode check; R-53D by day shared.
- 13:0x: M-5d PASS (Fit12): live 12 GB default 8db71cc537 peaks 11.774 GB lifetime on 16 math rows, tokens equal; headroom 0.19 GB (prefill-start +0.88 GB allocation). RUNAHEAD-BUILD KILL as registered (Breakthrough a760b81) unless step 2's beta <= ~3.4; EvalSuite runner 37 = PMIN12 decode check (shared by day); Leap runner 62 = K-80 + k83 stacked every-token check after runner 36.
- 13:2x: b5 (Spec-AUF) vs b4 (MTP Head): +0-3 % at greedy, math only significant; chat depth-1 unchanged -> b5 is the all-round head, no runtime run; D-23c stays on b3 tonight.
- 13:5x: D-24 (MTP Head): MTP trees don't pay on the default (leaf branches -7 .. +0.9 %); K15 + cut beats K7 + cut (code +14.5 %); D-23c v2 = dfl vs mtp15e (b5), one 30-min quiet ticket tonight after C45; asked to run it on the current default (OOO + COMBINE) if the harness allows. Day rule: one SSD-heavy shared run at a time (17 heat pauses 12:03-13:41).
- 14:5x: M-17 LAYSZ45 (new, MoonShots, CPU, exact): R-27's non-uniform per-layer pool sizes (24 GB: -10.3 % bytes per token held-out) on the 45 GB default; cap45 chained, LRU + prior, strict pools, Lagrangian at uniform-C 224 / 256 budgets, LOO. PASS >= -8 % SSD MB per kept token at both; KILL < -3 %. (AdapMoE-style per-layer allocation.)
- 15:xx: SG12 (Fit12, exact, 12 GB): the last 0.19 GB headroom is lost to one prefill scratch on short prompts (< ~130 tokens per MoE call: 215-320 experts x 2.76 MB); grouped fix F12_SG12 (G <= 64, padded to keep the per-pair kernel) bit-equal in a synthetic test; 1a / 1b after EvalSuite r37 (~18:30), then R-53D; value ~0.4-0.6 GB headroom (C_m 60 -> ~64-66, Breakthrough's call).
- 15:3x: M-17 LAYSZ45 PASS in replay (MoonShots): -20.7 % / -19.3 % SSD MB per kept token at C 224 / 256 held-out, exact -> engine A/B as arm L in BigMoE's C45 tonight (laysz45_224.json). CACHE-SIZE45: SSD ceiling 188 tok/s today, 231 with M-17.
> (a quote from the project owner, kept private)
- 19:5x: PMIN12 decode check (EvalSuite r37): scores identical (decision 4 PASS), memory 11.99 GB vs 11.95 line, control (live default) 11.95x on the same HumanEval+ rows -> decision 6 in FOR_KAMEN (recommend yes after Fit12's trim gives >= 0.1 GB headroom). SG12 1a exact but small (-0.07..-0.12 GB); Fit12 HumanEval+ per-phase memory trace GO.
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
- 20:4x: window extended through Monday (Kamen 20:3x). After ~05:45: BigMoE C50 (C 300 / 320 + per-layer arm) -> per-row split -> Leap fixed-constant capture -> DLOG top-ups; MoonShots writes laysz45_300 / 320 (CPU). Posted to the lounge.
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
- 22:0x: Fable mission 9 decision map (~/FableGod/decision_map/, 16 teacher-forced ablations, ~3 h, 45 GB preset): runner Breakthrough; wire 5 ADAPT points Monday by day (baseline >= 99.9 % top-1 vs D-02a), night 1 in Monday night's quiet queue; night 2 (48 runs, ~6 h) later. Lab loop: Fable designs, we run, results back to Fable.
- 22:5x: SG12 1b KILL (-0.07 GB, line 0.10; binding peak moved to late prefill +0.80 GB); R-53D within +-3 % and exact -> no quiet ABBA, live 8db71cc537 stays. Fit12's HumanEval+ trace (late-prefill transient) after 10:35 Mon.
- AT WAKE (Mon): read ~/FableGod/MONDAY_TREE.md (pre-registered predictions for tonight's results) + leads/2026-11-01_architecture_mismatch_fable.md; route: unshipped-ledger items (bmm16 HC -2.5 ms PASS, KP HC / gmr24), W23 prefill transient (~-10 % misses at equal RAM; BigMoE's memory anatomy tonight), W2 mixed-precision HC4 (~1 % hot channels), QSA indexer skip at short context (-1..-2 ms), W18 SSD throttle logging, slab caching (grep TRIED first); process fixes (ABBA + paired CIs, replay-before-verdict, drafter-data harvest under GPU jobs).
- 23:2x: tonight's idle NO-SWAP only 35-36 GB before Chrome closes (02:00): MTP Head's C 224 gates failed (D-23c deferred; DLOG at C 160 queued behind M-16); C45 (gate 42) waits for a post-02:00 grant; M-16 asked for MARGIN 4 (orchestrator); r36 running. BigMoE C50 now waits for ~/TrainLab/data/E15_SLOT_DONE (not pid 58883).
- Mon 01:3x: BigMoE C45 (01:18-01:26, exact in all 8 runs): OOO A-B-A +18.7 % steady (53.9 at C 224); L PASS (-17.2 % SSD MB/kept, +8.3 % steady) -> ADOPTED; K-73 +2.4 % -> ADOPTED; P +1.5 % -> not adopted; C 240 +5.2 %, C 256 +10.0 %; C 272-320 skipped (46 GB free). BIG pending for 07:15 (in SUMMARY). C50 adds L at C 256. Breakthrough's decision map queued #4 (A1 = D-02a's config; top-k masks ranked scores).
- Mon 01:3x: M-16 RUNAHEAD step 2 KILL (best +0.4 % at C 224, exact 481/481): the default is SSD-bandwidth-bound (code ~936 reads ~147 ms of ~200 ms); levers = fewer bytes + more RAM.
- Mon 01:5x: BigMoE C50 prepared (adoptions in R57_PRESET=45: L sizes md5-pinned, K-73 with OOO; arms L256, U256, U300, L300, L320, U320, guards 11:30 / 11:50) but blocked from arming by the peers hook while quiet locks are held / awaited; trigger changed to E15_SLOT_DONE only, so C50 runs ahead of runner 62 (C 160 fits by day). Warm split -> Monday night.
- Mon 04:21 (NewIdeas, woken by peers): E1 + E5 KILL (MTP Head REPORT 40c90a5); HPD-1 queued last by MTP Head, placement OK.
  Decision map Q1 topk2 = MONDAY_TREE outcome (a) on code-sure only (0.21 %); mid rows 26-44 % vs predicted <= 3 / 5-10 %;
  multi-sure 5.07 % at the kill line; routed to Fable (peer send, no wake). TRIED re-merged (M-17 ADOPTED); TRIED_SHORT:
  L, OOO, K-73, C 256 lines now ADOPTED, E1/E5 and Q1 added. My transcript grep loaded EvalSuite's r36 quiet chunk ~3 min
  (told them). pacer SLEEP ~20 h (weekly 42 % vs 16 %+14); the 45 GB BIG goes out by script at 07:16
  (scratchpad big_then_sleep.sh: peer send to the orchestrator's pid, flag .big_45gb_sent, log big_send.log).
- Mon 06:47: D-23c' C 192 PASS +10.6 % exact (chat +14 %, thin; D-23d confirms). Decision map night 1 done: Q1 no-renorm
  (topk2r 2.33 % vs 0.21 %) -> Fable + BigMoE's Q1 spec; middle-layer MoE reuse (12-35, 0.10-0.14 % code-sure) = lead.
  Night 2: told Breakthrough to stage it: one ~30-min ticket first (12-35 at once, lag 2 / 4, edge groups 8-11 and
  36-39), per-layer only where needed, in ~50-min tickets behind C50 / D-23d / runner 62, rest tonight. Asked MTP Head
  whether math -8.6 % in D-23c' is drift (same drafter both arms) -> D-23d ABBA. TRIED re-merged (814 ideas).
- Mon 06:53: D-23c' corrected to BETWEEN (~+8.7 %): arms ran back to back on each prompt, so the second verified on warm
  experts (fewer misses on 17/18, w 1.12-1.21). D-23d now separates the arms (22 prompts apart, first arm alternating,
  23 prompts outside every training set). Process rule for every A/B timing: separate passes or cold starts, report w.
- Mon 07:25: C50 at C 256: L PASS (-15.7 % MB per kept token, +5.0 %), full default 63.6 steady / 56.4 whole run, 41.7 GB,
  exact; C 300/320 skipped (NO-SWAP 49 < 52/55). REF2c-T FAIL (IFEval 85 vs 90, 5/0; EvalSuite d4). Night 2 stage 1:
  12-35 at once 1.03 % code-sure, lag 2/4 2.46/4.76 %, chained 8.11 % (KILL code); edges 8-11 free, 36-39 0.82 %.
  Stage 2a: told Breakthrough to class rows by drafter confidence (UGB died on oracle vs realizable) and add bytes saved.
  BigMoE Monday night: warm per-row split, then P on L; C 300/320 on a night with >= 55 GB NO-SWAP.
- Mon 07:26: Monday-night line routed: BigMoE warm per-row split, then P on L (~1 h); then EvalSuite's REF2c-T vs REF2-T IFEval
  re-test (441 prompts, two pre-registered stages, stop early when decided; ~6 ms per check = ~7 % at stake). BIG update
  (63.6 steady at C 256 + L) sent to the orchestrator; C 300/320 need a Chrome-closed night (Kamen's call).
- Mon 08:40: D-23d CHAT SWITCH PASS (exact; chat +24.5 % time-weighted, 19/19, multi +27.4 %, w 0.992 with separate
  passes); agent stays DFlash2 (G -0.5 %). Decision 7 in FOR_KAMEN (recommend yes); Leap / BigMoE wire on Kamen's word.
  HPD-1 running under a shared lock (verdict ~10:30-11:00). EvalSuite IFX armed (runner 38, GO file from BigMoE;
  ~/EvalSuite is read-only for me, so I do not touch GO).
- Mon 09:25: HPD-1 KILL (H -0.0077 on code uncertain positions, line +0.01; U control reads ~10:00 and decides HPD-2).
  First check after a ~9K prefill: MTP 1.9-2.8 s (7/8 cold runs), DFlash2 0.3-2.5 s; without it MTP +30 % on agent.
  R-80 killed kernel builds as the cause; K-80 (runner 62) is the per-check split, still waiting. Asked MTP Head to split
  D-23e's first-check misses by tier and layer: a cold 3-bit pool after a 4-bit prefill would point at the PARKED
  tier-pool warm-up (exact if the CPU-derived 3-bit records are bit-equal to the stored ones). D-23e ~10:00 at C 224.
- Mon 10:36: Kamen 10:1x 'Yes, 1-3': decision 7 chat switch YES (BigMoE wires; C 256 measurement still open, a night
  slot), Chrome / Spotify / Steam close after 01:00 for C 300/320 (BigMoE arms), MBPP YES (~/TrainLab/data/mbpp).
  Kamen 10:3x: all 9 ROUTE_350 condition tests TODAY (DAY9 file; day rule lifted today, memory gates stay). NAX is no
  longer parked: C1 (NAX16 live at C 256) and C0 (16-row GPU timeline) run in DAY9 (Leap). HPD-1 U: H - U +0.0023
  (the loss is the extra epochs, not the masking). Fit12's HumanEval+ memory trace slips to Tue (Fit12 SLEEP, not
  reassigned: 12 GB work follows 350). pacer: SLEEP to ~Tue 18:3x (weekly 53 % vs 34 %); DAY9 results to files.
- Mon 12:07 (woken by peers, still SLEEP): Tue night = Fable's FLOOR (FLOOR_125B.md sec 5, Breakthrough, ~10 h; Kamen
  'Yes, queue both'); IFX leftovers and other night items -> Wed night (lounge note). DAY9 so far: K-HEAD KILL (MTP Head
  198d2fd), K-80 PASS (first-cycle excess ~130 ms: warm build +84 ms, cold miss burst), K-83 exact +0.7 %, Q1 code SSD
  bytes x0.60 (MoonShots M-18). Read DAY9 results at wake.
- Mon 15:18: DAY9 condition 8 E2 / D2 x2 PASS (MTP Head 1d8ca7f): one doubling of fresh MBPP code from the 125B, code kept
  per check 9.57 -> 9.79 (+0.22 per doubling, line +0.10), math flat; x4 after BigMoE's A6. DUALDRAFT queued Tue by day.
- Mon 17:09: MATCH (session match override) BETWEEN (HumanEval+ x1.038 at 16 rows, x1.204 at 32; agent x1.047; MBPP x0.980;
  a2086f2; next: a precision gate). BigMoE A6 PASS: keeping MLX's buffer cache across requests cuts the first-cycle excess
  465/1131/1681 -> 211/154/130 ms at C 160, exact. KEPT_LEDGER.md started (Fable DRAFTER_WALL sec 3 item 7, orchestrator).
- Mon 19:13: E2 / D2 final PASS by the line (code 9.57 -> 9.79 -> 9.85 over two doublings, slope +0.14; math 7.06; c051d23):
  data is a slow lever. Tonight MTP Head: COPYGATE, EARLYTAP ~22:30, then BigMoE's ticket and D-23x; Tue: DUALDRAFT, DIVERSE2.
- Mon 20:47 (ULTRA week: pacing lifted, RULES.md top): AT-WAKE items routed: bmm16 HC (PASS, new rounding) superseded by
  Fable's bitwise D1 for REF2-T (keep for a REF3 bundle); KP / gmr24 already KILL; W23 prefill transient moot (caches grow to
  ~50 GB instead); W2 HC4 behind D4 (Kamen's call); QSA indexer skip -> Leap (check REF2's path first; TRIED #16); W18 SSD
  throttle logging -> BigMoE; slab caching: no source found, dropped. Leap's queue ordered (RANKED rows 24-29).
- Mon 20:48: proposed M-18c WIDE-Q1 to MoonShots (pre-registered: 32-row checks + Q1 0.99 vs 16-row + Q1 on code, implied
  steady >= +10 % PASS / <= +3 % KILL; inputs D32 dense x1.24, MATCH 32-row kept, band45 replay). Decides wide checks + NAX6.
- Mon 21:18: D1-B exact but -0.73 ms (KILL by 0.07; D1 ~-1.3..-1.8, not -3.4) -> Leap: mission 27 probe, D5, expert kernel,
  then A (orchestrator 21:15). Kamen's COMBINATION (BigMoE integrates every exact lever from Tue): GPU order = BigMoE's day
  screens and the nightly combined measurement first; Wed night FLOOR after it; RELAX smoke Tue day (by ~14:20) or Wed day,
  RELAX night Thu 00:30. Fit12 HT1: the HumanEval+ peak is layer 0's single-path MoE call (+0.89-1.12 GB); exact trim
  (balanced groups <= 64) expected -0.5..-0.65 GB, queued after Fit12's T0 gates.
- Mon 22:19 (main): Fable's NEWIDEAS_SCREEN (cloud task 16): #5 MTP-seeded block drafting KILL (at block starts MTP 0.865 vs
  DFlash2 0.941); #6 lookahead acceptance waits for MTP Head's walk with draft-conditioned rows (after its move to main; its
  chain ends Tue 15:00); #1 partial-sum pre-route -> RANKED 31 (BigMoE's 10-min capture); #2 sketch-on-miss -> FAST 2b
  (Breakthrough: shadows + map arm). M-20 SSDWAKE BETWEEN -> RANKED 30 (BigMoE keepalive A/B). H-1 -> FAST 1b; Leap's exact
  order (c) -> D -> G3 confirmed. Novelty tags on all 215 ADOPTED / PASS levers (~/TRIED/NOVELTY.md); M-18d (Q1 at 24 GB)
  queued at MoonShots after M-18c.
- Mon 22:55: M-18c WIDE-Q1 KILL (32 + Q1 vs 16 + Q1: code -1.3 % / -0.6 % NAX6); M-21 BWIDTH KILL, M-21b STOPCLS KILL on code
  (chat t ~0.4 +2 % today, ~0 with NAX6); layer-wise P-frame map KILL on bytes (x0.91); plain rank-32 SVD shadows KILL
  (activation-aware shadows next, Tue capture; FAST 2b unproven); NAX7-F KILL, G3 rough KILL, M27-C int8 rough PASS
  1.55-1.73x -> Leap: design C (FAST) -> D (Tue after 15:00) -> ANE step -> kernel A; D5 parked. Fable B's B3: seeded argmax
  (FEXIPRO, published) low priority; predict-then-correct EXACT KILL; 32 rows only as copy-driven blocks. M-18d: option A
  tonight, B only if A is within 0.05 of a line.
- Mon 23:00: M-18d Q1-24 option A: v3 175.0 -> 115.2 SSD MB per kept token (x0.658 PASS, within 0.05 -> option B asked
  of MTP Head); b16 x1.079 KILL (the budget already caps the union). FAST_TRACK 24 GB #4: Q1 is FM1's quality-safer
  alternative, not an add-on (b16 stays lower in bytes, 81.8).
- Mon 23:31: Fable B's B5 FIRST_MISS: the first miss is visible before the check (row AUC 0.80 with cf, 0.89 with content
  on code); 49 % of code first misses are comment tokens; copy spans miss 1 row in 1,773; width / cut rules cash only ~+1 % on
  code (19 % of misses at cf >= 0.99, unflagged); chat per-row cut +4.5 % (halved +2.2 %). -> chat t 0.4 arm queued with
  BigMoE (exact, pre-registered >= +1.5 % PASS); comment loophole to be re-priced at RELAX arm B. M-22 TWOSTAGE KILL on code
  (-1.7..-3.3 %). Breakthrough's Tue map list: xcapture first, then two-pass, 3-bit ranks 0-1, Q1-UNION, leftovers.
- Tue 00:09: M-26 SIDECARDROP PASS on bytes (3333333333 + freed RAM: code 46.3 -> 24.5 MB per kept token, with Q1 11.8;
  implied 75.8 / 85.3 steady) -> FAST 2c; quality = Breakthrough's t3_all map, now 2nd on Tue (xcapture stays 1st, then
  t3_all / t4_all, the 3-bit sure-rows arm, domain-adaptive, Q1-UNION). M-25 sidecar re-split KILL / BETWEEN; M-25b joint
  sizes -2.8 % at C 320 (PASS), +2.07 GB -16 / -20 % -> RANKED row 23 + optional C 320 arm to BigMoE. cmb1t (chat t 0.4)
  queued by BigMoE with my line.
- Tue 00:50, BACKLOG (orchestrator 00:5x, from the Lunatic's round 2; no GPU tonight):
  * AGENT-TURN metric for BigMoE / EvalSuite: time a whole long-context agent turn (prefill of ~9-12K tokens + the first
    check + decode of the answer), report turn tok/s (tokens out / turn time) and the prefill share. Near 350 tok/s decode,
    prefill is ~70-80 % of an agent turn and no current number sees it. Prompts: EvalSuite's agent set / D-23d's 4 agent turns.
  * DRAFTER HORIZON for MTP Head (once awake, after its move to main): on rows the 125B is sure of, DFlash2 is 99.6 % right
    at block position 0 but 88.4 % at positions 12-15 (lu_alpha.out): deep-position accuracy is a training target (FAST S2).
  * TRIED: the Lunatic's S1 'parallel same-domain streams' merged onto L1-32 (BETWEEN at 45 GB, KILL at 24 GB, PARKED as
    aggregate); new data: 4 code streams union 179 vs 166 experts, RAM per token x0.47. Aggregate vs 350 is Kamen's call.
- Tue 01:01: AGENT-TURN metric = EvalSuite's AT14 (dec_agent's 14 turns, 9.1-12K prefill, 128-token answers) in the RELAX
  PREREG (control + B, Thu night), read-out only: prefill s, first-check ms, decode s, answer tokens, turn tok/s, prefill
  share, TTFT; then the key list to BigMoE for its engine.
- Tue 01:45: NIGHT RE-ORDER (orchestrator 01:4x): BigMoE's 45 GB turn memory-blocked (NO-SWAP 34.6 GB vs 47-55; ~38 claude
  CLIs ~15 GB + apps; freeing them is Kamen's call). GPU re-ordered under ~30 GB, one at a time: Breakthrough's map now
  (xcapture -> t3_all / t4_all -> 3-bit sure rows -> domain-adaptive -> Q1-UNION; 10:00 rule lifted tonight) -> Leap's
  N7-15 + B6 FATT16-S screens -> Fit12's T3 timing if ready -> IFX via BigMoE's GO touch (after those, or 04:00 latest).
  MTP Head's chain at 06:00. BigMoE's 45 GB arms (c45j_320 included) wait for a night with freed memory.
- Tue 01:50: Fit12's T3 timing queued (runner pid 76612): asks for one quiet lock (~15 min, MEM 12 GB, >= 16 GB no-swap)
  only after Breakthrough's watcher 1 outputs exist (the map's guard would stop the map otherwise); gives way to waiting
  Leap / EvalSuite / BigMoE tickets; no start after 05:05. BigMoE's GO watcher touches IFX's GO 15 min after the small jobs'
  last lock, or at 04:00.
- Tue 02:42: sketch-on-miss (FAST 2b) KILL (whitened shadows 0.92 / 0.89 at rank 32 / 64, bar 0.5). t3_all map: code BETWEEN
  (+0.59 pts, sure rows 0 / 2922), chat +4-5 pts -> proposed a T3ALL arm for EvalSuite's RELAX PREREG (IFEval first; a
  code-only form as the fallback). Fit12 T3 timing: 2-rank pass 1 == 10-rank tokens and logits on 7 / 7; code 56.2 tok/s vs
  18.9 exact 16-row (3x) but under the KILL line 70; pass 1 is 145 of 211 ms. Leap: N7-15 KILL, B6 FATT16-S BETWEEN (1.70x at
  10K keys). IFX starts via BigMoE's GO watcher after Breakthrough's watcher 2 (04:00 at the latest).
- Tue 03:01: T3ALL in EvalSuite's RELAX PREREG rev 2 (plan 3333333333, exact acceptance, IFEval first then HumanEval+ / math;
  decision 4 vs REF2-T; the code-only fallback reads its math + HumanEval+ lines). Thu order: control, A, B, E, T3ALL, F, C, G
  (drop G, C, F if short); ~7 h with replays after BigMoE. Morning: refresh SUMMARY_FOR_KAMEN with the night's results.
- Tue 06:3x-06:4x (main): night routed. Breakthrough 03:19: Q1-UNION cf 0.9 PASS (all-row flips below Q1 0.99 on
  every domain), 3-bit sure rows PASS (free), domain-adaptive PASS (not over cf 0.8). IFX STOP PASS 06:37 -> FOR_KAMEN
  decision 8 (REF2c-T). BigMoE's night never ran (memory). MTP Head's night_1102 went on at 06:00.
  NewIdeas CPU replays (scripts/union_theta.py, union_oea.py; ~18 s, 0.05 GB): union per check code x0.720 (cf 0.99)
  / x0.618 (0.9). CORRECTION 06:4x: I wrote that the orchestrator's 'x0.595' had no source; it has one (MoonShots
  data/band45.txt: Q1 0.9 code SSD MB per kept token 66.3 vs 111.4 on cap45). Mine is the union per map block, a
  different quantity; both are kept, labelled. Prior art: Q1-UNION's union step = OEA (arXiv
  2511.02237, ICML 2026): tag adapted. OEA k8 x0.83 (= top-8's bytes, rows keep 9.0 of 10), stack sure k0 2 + unsure
  k0 8 code x0.527 / chat x0.76, unsure k0 6 code x0.435 / chat x0.60. Masks (scripts/dm_oea_masks.py, self-check
  bit-equal to q1u_cf09) -> Breakthrough wired 4 arms (78ccef8, ~20-25 min, MEM 17 GB; slot: the orchestrator).
  Proposed RELAX rev 3 to EvalSuite: Q1-UNION 0.9 in A / B / C. Orchestrator told (uds 88684).
- Tue 06:4x: orchestrator 06:4x: RELAX rev 3 APPROVED (Q1-UNION 0.9 + copy in A / B / C); the map's OEA arms get a
  memory-gated slot (NO-SWAP >= 25 GB, to 14:30, with Breakthrough's 4 two-pass arms); Wed 00:30 narrowed FLOOR + the
  best passing OEA arm at scale. The union stack's SSD side (scripts/band45_oea.py = MoonShots' band45 + 6 rules, writes
  only data/band45_oea; 54 s, 0.5 GB), all six cap45 prompts, SSD MB per kept token vs EXACT 63.1 (implied tok/s
  SSD-only / + GPU, model): Q1 0.9 -34.9 % (72.2 / 84.3); OEA k8 -18.6 % (67.9 / 72.1); OEA k6 -36.3 % (72.9 / 83.3);
  HYB 0.9 u8 -44.4 % (75.0 / 91.4; code -47.2 %); HYB 0.9 u6 -52.7 % (77.7 / 99.1); HYB 0.99 u8 -35.7 % (72.7 / 84.1);
  HYB 0.9 u8 + 3-bit sure rows -55.8 % (79.6 / 100.2; code -60.2 %). Quality: the map arms. OEA's venue confirmed on
  icml.cc (ICML 2026 poster 62958).
- Tue 06:5x PRE-REGISTERED (before any run) U-32, the union stack at 32 rows (NewIdeas, CPU): do 32-row checks pay on code
  once sure rows cost ~no new experts? Same replay as M-18c (MoonShots' wide45 copied to scripts/wide45_oea.py, writes only
  data/band45_oea: Breakthrough's 12 HumanEval windows, MATCH's 32-row rates, C 256 + M-17, cost model today / NAX6).
  Rules: EXACT, Q1 0.9 (= Q1-UNION's bytes), HYB 0.9 u8, HYB 0.9 u8 3b, HYB 0.9 u8 3b deep sure (rows 16-31 as sure, as
  M-18c's deep sure); arms 16 cut and 32 cut; 2 seeds. Line (M-18c's): pooled code implied steady tok/s, 32 cut vs 16 cut
  under the same rule: >= +10 % PASS, <= +3 % KILL, else BETWEEN. Reported too: each rule's 16 cut vs EXACT 16 cut.
- Tue 06:5x U-32 RESULT = KILL on its pre-registered line (data/band45_oea/wide45_oea.log, 2 seeds, 19 s): 32 cut vs 16 cut
  under the same rule, pooled code, today / NAX6: EXACT -0.4 / -0.2 %, Q1 0.9 -1.0 / -0.5 %, HYB 0.9 u8 -1.3 / -0.8 %,
  HYB 0.9 u8 3b -1.3 / -0.9 %; deep sure +3.8 / +3.3 % (BETWEEN, as M-18c's +4.3 / +3.8). Kept 8.00 -> 9.09 per check for
  10.6 -> 14.3 rows: the union stack doesn't change that 32 rows buy too few kept tokens; only cap-bound prompts gain
  (HumanEval/136 +18 %). 32 rows stays copy-driven only. Side result, the stack at 16 cut on code (wide45's cost model,
  model tok/s today): EXACT 86.3, Q1 0.9 130.2 (+51 %), + 3-bit sure rows 146.6 (+70 %), HYB 0.9 u8 139.9 (+62 %),
  HYB 0.9 u8 3b 158.9 (+84 %); all six cap45 prompts (band45, SSD-only / + GPU): Q1 0.9 72.2 / 84.3, Q1 0.9 3b
  76.4 / 91.8, HYB 0.9 u8 3b 79.6 / 100.2. So 3-bit sure rows (map-tested only at cf 0.99 / 0.999) are worth more than
  OEA's unsure-row cut; proposed to Breakthrough: 2 optional arms, q1u_cf09 and hyb_0.9_u8 with sure rows' ranks 0-1 at 3
  bits.
- Tue 06:5x: Breakthrough 06:54 added both 3-bit combos (hyb_0.9_u8_t3, q1u_cf09_t3: sure rows' ranks 0-1 from records3 at
  3 bits, union members at 4), pre-registered on my line (SCOREBOARD 8c917f0); its mask builder reproduces my 5 arms bit for
  bit (tonight's 78-window FLOOR masks come from it). Load order: oea_k8, oea_k6, hyb_0.9_u8, hyb_0.9_u6, then the 2 combos,
  then its 4 two-pass arms; waiting for memory (NO-SWAP >= 25 GB, window to 14:30). My 24 GB price of the stack
  (scripts/q1_24_oea.py, v3 pools C_m 147 + C_l 46, 4-row MTP checks) waits on fanwatch's heat PAUSE (06:54).
- Tue 06:5x-07:0x: correction: my labels 07:0x-07:4x this morning ran ahead of the clock (real 06:35-06:58); fixed in
  the files (messages already sent keep them). 24 GB price of the stack (scripts/q1_24_oea.py, the M-18d windows and v3
  pools, 2 seeds, 230 s): SSD MB per kept token no Q1 175.0, Q1 0.99 115.2 (x0.658 = M-18d), Q1 0.9 103.4 (x0.591),
  + 3-bit sure rows 93.8 (x0.536), OEA k8 134.3 (x0.767), HYB 0.9 u8 83.5 (x0.477), + 3b 73.9 (x0.422); b16 was 81.8.
  Sent to MoonShots (97727). MTP Head 06:57: D-23x done (426234f): the first check after a 9K prefill reads 2-3x the
  records of later checks at 4-10x the time per read (agent 2.5-3.1 s vs 0.2-0.26) -> pool warm-up (BigMoE mission 20);
  KWALL (b) BETWEEN (depth-1 mid 0.860, top-3 0.982); ANCHORCOV KILL as stated (chat 0.21; 0.78 / 0.80 with first-miss-
  probability anchors). EvalSuite 07:00: RELAX rev 3 in (Q1-UNION 0.9 + copy in A / B / C, G stays Q1 0.999; mask =
  dm_q1u_masks' rule on 200 random checks, posts exactly Q1's union); T0 running since 06:58 (VOID gate passed).
- Tue 07:11: chat SSD price of the stack (band45 capture mode on the map's 12 code / 12 IFEval / 12 multi windows, C 256
  + M-17, my copy; waited for the heat PAUSE to clear at 07:03; 437 s). SSD MB per kept token, code / IFEval / multi:
  Q1 0.9 -50.3 / -19.7 / -16.4 %; Q1 0.9 3b -67.1 / -27.8 / -21.7 %; OEA k8 -21.0 / -17.8 / -20.8 %; OEA k6 -39.9 /
  -33.6 / -38.5 %; HYB 0.9 u8 -56.2 / -32.4 / -32.4 %; HYB 0.9 u6 -61.2 / -43.2 / -45.2 %; HYB 0.9 u8 3b -73.0 / -40.5 /
  -37.7 %. Implied six-prompt median steady tok/s (SSD-only / + GPU, model): HYB 0.9 u8 76.8 / 92.3, + 3b 80.5 / 99.8.
  Chat rows are mostly unsure (cf < 0.9 on 59-66 %), so OEA's unsure-row cut is chat's lever; chat is ~79 % constant
  per check, so its tok/s moves less than code's.
- Tue 09:09: P1-OEA PRE-REGISTRATION (12 GB, FAST; my new bet for the 12 GB line). Pass 1 of the two-pass check reads
  the union of the 16 block rows' top experts. That pass-1 read is the binding resource (PLAN_12GB_TWOPASS: on the per-layer chain,
  'no forecast or residency removes' it). OEA (arXiv 2511.02237) lets every row also use its ranks that are already in that
  union, for zero extra reads. Two designs:
  * P1-K1 = top-1 + union members. CPU (scripts/pass1_bytes.py = Breakthrough's dm_rankcache replay, LRU, C 51, wrong-draft
    stand-ins; its k1 = 2 row reproduces Breakthrough's 0.708): code pass-1 0.708 -> 0.329 GB per check (x0.465; read time
    54 -> 25 ms at 13 GB/s; hit rate 64.1 -> 69.4 %), IFEval 0.279 -> 0.137, multi 0.154 -> 0.077. Experts per row 2.44 vs
    2.00; mean gate mass 0.357 vs top-2's 0.358 (estimate: scripts/pass1_mass.py, Breakthrough's measured mean rank profile),
    but wider (P10 0.212: some row-layers keep only rank 0).
  * P1-K2 = top-2 + union members: today's reads (union identical, checked), 3.98 experts per row, mass 0.529 vs 0.358.
  Masks (36 windows, prompt rows exact, checked: OEA keeps the base ranks; OEA k2 is a superset of the control):
  ~/NewIdeas/data/dm_masks/{top_k2, oea_k2, oea_k1} (scripts/dm_oea_masks.py, new top_k<k> control arm).
  GPU ask (Breakthrough's map, ~4 min each, all-rows teacher-forced runs, PERT rmask=<absolute dir>, renorm False):
  top_k2 (the control: the map's own topk2 also cuts prompt rows, so it is not comparable), oea_k2, oea_k1.
  Read-out: dm_cert_read's CPU columns with each run as the certificate source: certified share, blocks needing pass 2,
  pass-1-alone certified-row flips (Wilson 95 %), all-row flips; code at theta 0.8 decides, 0.9 / 0.95 and chat reported.
  Lines (code, theta 0.8, vs top_k2):
  * P1-K1 PASS: certified flips <= the control's 95 % upper bound AND blocks needing pass 2 <= control + 10 points. KILL:
    certified flips > 2x the control's upper bound, or blocks needing pass 2 > control + 20 points. Else BETWEEN.
    Breakeven model: ~29 ms of pass-1 reads saved vs ~110-120 ms per pass 2 (Fit12's T3: pass 1 145 of 211 ms, pass 2 in
    55-59 % of code blocks) = ~+25 points.
  * P1-K2 PASS: blocks needing pass 2 <= control - 5 points with certified flips <= the control's upper bound (a better
    certificate at today's bytes). Else KILL as a lever.
  * A PASS goes to the mixed runs (two-pass flips through the context, Breakthrough's harness), then Fit12's timing. The
    quality verdict stays EvalSuite's. If P1-K1 passes, the 12 GB code cycle is ~211 -> ~182 ms (+16 %, model).
- Tue 09:11: orchestrator: P1-OEA slot APPROVED, right after the map's 10 arms in Breakthrough's same run (~11:15-11:30),
  before EvalSuite's RELAX smoke; it told Breakthrough. GPU one big job at a time to 15:00 (heat); Kamen OK'd fans to 7000
  (boost on since 09:05). Read-out ready: scripts/p1_cert_read.py (self-test = Breakthrough's topk2 columns exactly).
- Tue 10:16: MAP RESULT, the 6 union-stack arms (Breakthrough's load, 09:34-09:55; read with a private copy of
  dm_oea_read.py, Breakthrough's OEA_read.txt untouched). Lines as pre-registered: per domain all-row flips vs Q1 at
  0.99's level (code 1.34 / IFEval 5.42 / multi 3.69 %), PASS <= 1x, BETWEEN <= 1.5x, KILL above. Rows 3,366 / 3,397 /
  3,553. All-row flips code / IFEval / multi:
  * oea_k8 0.62 / 3.83 / 3.07 % = PASS / PASS / PASS
  * oea_k6 0.71 / 6.06 / 5.38 % = PASS / BETWEEN / BETWEEN
  * hyb_0.9_u8 1.34 / 6.18 / 4.08 % = PASS / BETWEEN / BETWEEN
  * hyb_0.9_u6 1.28 / 8.01 / 5.71 % = PASS / BETWEEN / KILL
  * hyb_0.9_u8_t3 1.19 / 6.68 / 4.39 % = PASS / BETWEEN / BETWEEN
  * q1u_cf09_t3 1.19 / 5.06 / 3.24 % = PASS / PASS / PASS (q1u_cf09 at 4 bits: 0.98 / 4.92 / 3.04)
  Gated code flips (cf >= 0.99): 0-1 of 2,211 in every arm. Paired (McNemar, scripts/union_paired.py), vs Q1 at 0.99:
  q1u_cf09_t3 z -0.78 / -0.88 / -1.33 (no worse anywhere); vs q1u_cf09 +7 / +5 / +7 flips net (z <= 1.4). On code
  every hybrid ties Q1 at 0.99 (u8_t3 z -0.85, u6 -0.32); on IFEval the unsure-row cut costs (u8_t3 +2.83, u6 +5.44).
  Reading:
  (1) q1u_cf09_t3 (Q1-UNION 0.9, sure rows' ranks 0-1 at 3 bits) is the best arm that passes every domain. SSD MB per
      kept token: cap45 -46.3 % (code -53.5 %; 76.4 / 91.8 tok/s SSD-only / + GPU, model) vs Q1-UNION 0.9's -34.9 %
      (72.2 / 84.3); map capture code / IFEval / multi -67.1 / -27.8 / -21.7 % vs -50.3 / -19.7 / -16.4 %; 24 GB
      x0.536 vs x0.591. My pick for Wed's FLOOR at scale (Breakthrough's run (4), 'the best one'). The engine version
      (3 bits only for experts no unsure row in the check uses) has the same bytes and keeps more 4-bit math than
      the map arm, which forced sure rows to 3 bits even where the LSB was loaded.
  (2) Every union arm passes on code, so a code-only stack is open: hyb_0.9_u8_t3 code SSD MB per kept token -60.2 % on
      cap45 and -73.0 % on the capture, 24 GB x0.422. Chat keeps its unsure rows at 10.
  (3) OEA k8 alone (published) passes everywhere and needs no drafter signal: the chat-switch candidate (-18 / -21 %
      chat SSD bytes).
  (4) KILL: hyb_0.9_u6 on multi (5.71 % > 5.54). Not a verdict on code.
- Tue 10:23: P1-OEA RESULT (3 all-rows map runs, 10:07-10:18 in Breakthrough's load; scripts/p1_cert_read.py ->
  data/P1OEA_read.txt; paired: scripts/p1_paired.py). Pre-registered lines (37f8e37): code, theta 0.8, vs p1_top_k2.
  * P1-K1 (top-1 + union members, x0.465 pass-1 reads) = PASS. Certified 89.0 % (control 89.7 %). Blocks needing pass 2
    52 % (205/391) vs 53 % (207/391); paired 7 vs 9 discordant blocks (z -0.50). Certified-row flips 0.23 % (7/2997)
    [0.11-0.48] = the control's 0.23 % (7/3019). So the same certificate on code at x0.465 pass-1 reads. At theta 0.9:
    +2.0 points of pass-2 blocks (z +2.1). Chat is weaker: IFEval / multi +1.4 / +3.1 points of pass-2 blocks at 0.8,
    IFEval certified flips 2.51 vs 1.54 %. Chat runs pass 2 in 81-93 % of blocks anyway.
  * P1-K2 (top-2 + union members, today's reads) = PASS, by 0.1 point on the blocks line. Pass 2 in 48 % (187/391) vs
    53 %; paired 27 vs 7 (z -3.43). Certified flips 0/3068 vs 7/3019. Bigger on chat: multi pass-2 blocks 81 -> 66 %
    (z -14.9), IFEval 92 -> 88 %; certified flips multi 3.66 -> 1.53 %, IFEval 1.54 -> 0.50 %. Code time model: ~-4 ms
    per check (pass 2's 38 ms x 48/53). The GPU cost of ~4.0 vs 2.0 experts per row in pass 1 is unpriced.
  Next for P1-K1, the read lever:
  (a) End-to-end mixed runs, PRE-REGISTERED here before any run. p1c_oea_k1_p08 = P1-K1's mask on rows whose p1 in
      p1_oea_k1 is >= 0.8, the rest exact (Breakthrough's cert gate with my mask as the cut). Control p1c_top_k2_p08 =
      top-2 on rows whose p1 in p1_top_k2 is >= 0.8 (cert_top2_p08 stands in if the slot is short). Masks
      ~/NewIdeas/data/dm_masks/p1c_{oea_k1,top_k2}_p08 (scripts/p1_mixed_masks.py; certified rows = the read-out's
      2997 / 3019 on code). Read-out scripts/p1_mixed_read.py; its self-test reproduces CERT_read's cert_top2_p08
      (code 0.27 % (8/2976), uncertified 12.05 %). Line (code, theta 0.8): two-pass certified-row flips <= 1 % and <=
      the control's Wilson upper bound = PASS; > 5 % or > 2x the bound = KILL; else BETWEEN. Chat reported.
  (b) Fit12 timing: P1-K1 as T3's pass 1, beside t80ck2 (same prompts, one quiet lock). Time model: bandwidth only
      (54 -> 25 ms of reads at 13 GB/s) gives 211 -> ~182 ms per code check (+16 %); if pass 1's 145 ms scales with
      its misses, up to ~-70 ms (+50 %). T3P's wait / gap split (Fit12, 15:07) says which. Proposed line: code tok/s
      >= 1.10x t80ck2 = PASS, <= 1.03x = KILL.
- Tue 10:26: slots (orchestrator 10:2x; Breakthrough 10:25):
  * P1-OEA mixed runs: p1c_oea_k1_p08, then p1c_top_k2_p08, right after EvalSuite's RELAX smoke. Start window
    11:00-11:35, hard stop 11:45; T0 waits for them. Breakthrough checked the masks: code certified rows 2997 / 3019;
    uncertified and prompt rows all kept in 36/36 windows. Read-out: scripts/p1_mixed_read.py.
  * P1-K1 timing (Fit12): inside Fit12's 15:07 quiet lock after T3P, or before 15:00 if MTP Head's chain ends by
    ~14:40; lock <= ~25 min.
  * Kamen: the 45 GB map result (Q1-UNION + 3-bit passes everywhere, -46 % SSD bytes, model 92 tok/s) is in his 10:22
    summary. P1-OEA's PASS goes to him as a promising lead, labelled a model with timing pending.
  * RELAX rev 4 (Q1-UNION + 3-bit sure rows) is EvalSuite's feasibility call; rev 3 stays if rev 4 costs the slot.
  * Breakthrough skips hyb_0.9_u8_t3 at scale tonight (the night is past ~10 h; the deadline would cut it).
- Tue 10:38: LIVE-CUT CAVEAT on every union arm, and two map arms PRE-REGISTERED (before any run).
  The map's union masks (mine and Breakthrough's, bit-equal) put all 16 rows of each walk check in the union: realized
  rows plus wrong-draft stand-ins. The live default ends a check once the product of the drafts' cf falls below 0.3
  (dm_gate's caveat). Emulated on the walks (scripts/dm_oea_masks.py <arm>_live): checks cut before their first miss
  code 8.4 / IFEval 6.8 / multi 4.1 % (doc: 4-8 %); rows per live check code 9.74 (cap45 capture 9.10), IFEval 3.70,
  multi 2.97, vs 15.7 on the map. So live sure rows find fewer union members. Experts kept per realized row-layer
  under Q1-UNION 0.9: code 6.17 -> 6.00, IFEval 8.73 -> 8.24, multi 9.05 -> 8.65 (about 1.3 fewer per sure row on
  chat). OEA k8 barely moves (8.97 / 8.60 / 8.54). The members carry Q1-UNION's chat PASS: plain Q1 0.9 is 8.86 %
  IFEval and 7.60 % multi on the map vs Q1-UNION's 4.92 / 3.04. So the map's union-arm verdicts can be optimistic on
  chat. SSD prices (band45 on live cap45 checks) are not affected; RELAX measures the live engine directly.
  Arms (36 windows; masks ~/NewIdeas/data/dm_masks/<arm>, rmask, renorm False):
  * q1u_0.9_live: Q1-UNION 0.9 on live-cut checks. Unions: code x0.519, IFEval x0.816, multi x0.857 of the live exact
    union.
  * reach_0.9_0.5_live: plus REACH-CUT. An unsure row whose chance of being reached (the product of the drafts' cf
    before it) is < 0.5 also keeps only its top-2 + union members: rows past the first miss are discarded, so their
    experts are pure cost. Adapted from AcceptMoE's offline per-position commitment weights (arXiv 2608.02989), made
    per row and online. Unions x0.450 / 0.655 / 0.715 = -13 / -20 / -17 % vs q1u_0.9_live. CPU screen
    (scripts/reach_screen.py, p1_oea_k2's flips on the band's realized rows as a pessimistic proxy): est. all-row flips
    1.13 / 6.59 / 3.88 % at tau 0.5, 1.04 / 5.45 / 3.49 % at tau 0.4.
  Lines (the union arms', unchanged): per domain all-row flips <= Q1 at 0.99's level (code 1.34 / IFEval 5.42 / multi
  3.69 %) = PASS, <= 1.5x = BETWEEN, above = KILL. q1u_0.9_live failing where q1u_cf09 passed means the map's
  union-arm verdicts are optimistic on that domain, and the FLOOR at scale should use live-cut unions. REACH is worth
  keeping only if it PASSES where q1u_0.9_live passes (paired cost reported via scripts/union_paired.py).
- Tue 10:40: orchestrator 10:4x:
  * The live-cut arms (q1u_0.9_live, reach_0.9_0.5_live) run in today's slot right after the P1 mixed runs, from ~10:45;
    the GPU has been idle since EvalSuite's smoke ended at 10:26.
  * Kamen gets the caveat now.
  * If q1u_0.9_live fails on chat, tell the orchestrator at once: Thursday's RELAX rev 3 runs Q1-UNION at 4 bits on
    chat too.
  * EvalSuite: t3 in RELAX is not small (it needs a per-row 'no LSB' mask in the engine's tier call). Rev 3 stays for
    Thursday; rev 4 is off.
  Note: the live-cut caveat does not touch P1-OEA. Fit12's T3 runs pass 1 on all 16 rows with no running-confidence
  cut (PLAN.md T3), so the map's 16-row unions are the right model there.
- Tue 10:40: Fit12 10:40: P1-K1 wired and pre-registered (Fit12 PLAN.md 'T3P amendment').
  * Arm t80ck1u runs beside t80ck2 inside T3P's quiet lock: 14:3x if MTP Head's chain ends by ~14:40, else 15:07.
  * My line is taken as is. C 47 is the verdict, C 51 a replicate. HumanEval 156 / 42 / 68 / 91, 256 tokens, arm order
    alternating per prompt.
  * Engine: each wide pass-1 call posts only U = the rows' rank-0 ids. Each row's ranks in U are compacted to KU = 4
    columns, all 4-bit, weights p / top-10 mass with no renorm. Calls of <= 2 rows stay top-2. Unit tests pass.
  * Costs it measures that my model left out: tier_glu has no zero-skip, so 4 columns are ~2x K2's pass-1 MoE compute
    (~+10 ms per check, its estimate); and pass 2 may miss more, since rank 1 is no longer warmed.
  * KU 4 vs the map's uncapped mask (my CPU check on the oea_k1 masks, Breakthrough's rank profile): 4.7 % of code
    row-layers keep > 4 experts. The cap drops their lowest union members, ~0.081 of gate mass each. Mean kept mass on
    code 0.3574 -> 0.3536 (-1.1 %), still about top-2's 0.358; IFEval -0.8 %, multi -1.1 %. Likely inside the
    certificate's noise. If the timing passes, one KU-4 mask run on the map (4 min) confirms it before adoption.
- Tue 10:42: P1-REACH rapid screen = KILL (CPU model; scripts/p1_reach_screen.py, pass1_bytes' replay with p1_oea_k1's
  certificate). Pass 1 skips the rank-0 reads of rows unlikely to be reached, with those rows forced uncertified
  (pessimistic). Code pass-1 GB per check 0.327 -> 0.294 / 0.265 / 0.239 at tau 0.2 / 0.4 / 0.6. Blocks needing pass 2
  52.4 -> 53.5 / 56.3 / 62.1 %. Net -1.3 / -0.2 / +4.9 ms per check (pass 2 at ~120 ms per block). Even with no
  certificate loss the ceiling is ~5 ms (~2.5 %). Not worth a GPU run.
- Tue 10:44: Fit12 10:4x:
  * Every per-prompt line and each arm's TOTAL line report pass-2 / exact ms and misses per check beside pass 1's.
  * Two INFO arms added, no line: t80ck2m3 (top-2) and t80ck1um3 (P1-K1). Both use Breakthrough's 3-bit pass-1 lever
    (M-18): pass 1 at 3 bits from the MSB-3 pool, no sidecar post.
  * t80ck1u's line and its balanced order are unchanged: 1st on two prompts and 4th on the other two, as t80ck2.
  * So P1-K1 at 3 bits gets timed too. Its certificate would need its own map run if the timing makes it worth it.
- Tue 10:52: P1-OEA MIXED RESULT (end to end; Breakthrough's map 10:44-10:51; scripts/p1_mixed_read.py ->
  data/P1OEA_mixed.txt). Pre-registered line: code, theta 0.8.
  * P1-K1 = PASS. Two-pass certified-row flips 0.23 % (7/2997) [0.11-0.48] = the control's 0.23 % (7/3019).
  * All code rows 1.52 % vs 1.96 % (control); uncertified rows through the context 11.9 vs 17.0 %; windows with a flip
    9 vs 10 of 12. Breakthrough's cert_top2_p08 (certificate from topk2): 0.27 % (8/2976), all rows 1.63 %.
  * Chat is weaker, as the CPU columns said. IFEval certified flips 2.43 vs 1.40 %, all rows 9.48 vs 8.86 %; multi
    4.22 vs 3.45 %, all rows 7.06 = 7.06 %. T3 runs chat through the cf gate / exact check anyway.
  So on code, P1-K1's certificate equals today's top-2 end to end, at x0.465 pass-1 reads. Speed is Fit12's t80ck1u
  (14:3x or 15:07); a KU-4 mask run confirms the engine's 4-column cap if the timing passes.
- Tue 11:01: LIVE-CUT RESULT (Breakthrough's map 10:51-10:59; scripts/live_read.py -> data/LIVE_read.txt). Lines as
  pre-registered (7a0339d). All-row flips, code / IFEval / multi:
  * q1u_0.9_live: 1.25 / 6.45 / 3.77 % = PASS / BETWEEN / BETWEEN. Paired vs the 16-row q1u_cf09: z +2.18 / +4.30 /
    +2.55, worse everywhere, so the 16-row unions were optimistic. Paired vs Q1 at 0.99 (the line): z -0.51 / +2.47 /
    +0.25. So with live-length checks, Q1-UNION 0.9 is a CODE lever. On chat it is BETWEEN: IFEval significantly over
    the line; multi over by 0.08 points, a statistical tie.
  * reach_0.9_0.5_live: 1.28 / 6.80 / 4.31 % = PASS / BETWEEN / BETWEEN. Paired vs q1u_0.9_live: z +0.21 / +1.03 /
    +1.84. By the pre-registered rule, REACH passes where q1u_0.9_live passes (code): a code-only add-on with the live
    code union -13 % (x0.450 vs x0.519). Its SSD price needs band45 with a reach rule (next, CPU).
  Consequences:
  * RELAX rev 3's Q1-UNION arms (A / B / C) run HumanEval+, math, agent and AT14 only; IFEval / multi run in E / F
    (gate off) and G (cf 0.999). So Thursday is not exposed on chat. Code passes live; math and agent have no map
    windows.
  * A chat FAST mode needs a stricter cutoff. Q1 at 0.99 is the line itself, and the union only adds experts, so
    Q1-UNION 0.99 sits at or below it (capture chat SSD -10.5 / -8.4 %, vs 0.9's -19.7 / -16.4 %).
  * This morning's 'passes everywhere' for q1u_cf09_t3 was on 16-row unions. Live it would sit about at the code line
    and BETWEEN on chat (3 bits added +0.1-0.2 points per domain on 16 rows).
  * FLOOR step C tonight should build q1u masks on live-cut checks; Breakthrough told (11:00), with live_segments().
  * Orchestrator told at once (11:00), with the correction for Kamen's 10:22 line.
- Tue 11:02: Breakthrough 11:01: data/floor1/LIVE_CUT is set, so FLOOR step B builds q1u_cf09 and q1u_cf09_t3 on
  live-cut checks. It copied live_segments(); its live masks equal my q1u_0.9_live on 36/36 windows. oea_k8 stays
  16-row (orchestrator). The t3 double-serve is fixed: the t3 arms take their 3-bit rows from the mask (rows with rank
  0 dropped), so a live cut's exact bonus row is exact in both mask and gate (checked: 36 windows, 5,140 t3 rows).
  REACH on HumanEval-24 is skipped tonight (the night is full).
- Tue 11:05: REACH-CUT SSD price (scripts/band45_oea.py with a position-aware reach rule; BAND45_RULES subset,
  BAND45_TAG _reach -> data/band45_oea/band45_reach.txt and band45_capture_reach.txt; full tables untouched).
  * cap45 live checks, SSD MB per kept token: code 61.3 (-45.0 %) vs Q1 0.9's 66.3 (-40.5 %), i.e. x0.925; all six
    prompts 36.9 vs 41.0 (x0.90), model 74.1 / 90.1 tok/s vs 72.2 / 84.3. With 3-bit sure rows: 29.7 vs 33.9, model
    78.5 / 98.5 vs 76.4 / 91.8. The math prompts' quality is unmeasured: no math map windows.
  * Map capture windows: code 22.7 vs 23.4 MB (x0.97); IFEval / multi x0.97 / x0.98. The expert union shrinks more:
    code 2.77 vs 2.96 GB per check (-6 %), IFEval -13 %, multi -10 %. Deep rows' experts are often already
    cached, so the GPU side gains more than the SSD side. Median model 75.9 / 91.1 tok/s vs 73.9 / 85.6.
  Verdict: a small code-only add-on (code PASS on the map at no measured cost; ~3-8 % fewer code SSD bytes, ~6 %
  smaller union). It ranks below the main levers and goes to FAST_TRACK 2d as an option, not a bet.
- Tue 11:08: PRE-REGISTERED (before any run) TOPK8-BOUND, for the chat switch. OEA k8 is the one chat lever that needs
  no drafter signal: -18 / -21 % chat SSD on the capture, PASS on 16-row unions. Live chat checks are 3-4 rows, and the
  45 GB chat switch drafts with the MTP head in short checks. There OEA k8's members thin out (kept 9.0 -> 8.6 per
  realized row-layer live) toward plain top-8 (8.0). Arm top_k8 = plain top-8 on the realized rows, prompt exact (masks
  ~/NewIdeas/data/dm_masks/top_k8, 36/36 checked; bytes = OEA k8's). It is the worst case for OEA k8 at any check
  length. Line (the union arms'): per domain all-row flips <= Q1 at 0.99's level = PASS, <= 1.5x BETWEEN, above KILL.
  PASS on IFEval and multi => OEA k8 is a safe chat-switch FAST lever at any check length. Slot: the next free map
  slot, not tonight (FLOOR is full); 4 min.
- Tue 14:51: peers since 11:08:
  * Breakthrough 11:08: top_k8 is wired for its next free map slot (not tonight).
  * MoonShots 11:09: M-18d option B PASS. Using MTP Head's export (the native MTP chain's acceptance per depth, and
    its top-1 probability as the Q1 gate), the 24 GB default's code SSD MB per kept token goes 175.5 -> 100.4 =
    x0.572 (line <= x0.70; option A x0.658). MTP keeps 3.75 per 4-row check. It prices Q1 at 0.99, the safe 24 GB form.
  * Breakthrough 11:10: 3-bit pass 1 PASSes at theta 0.8. Certified code flips 4/2970 vs 4-bit 8/2976; pass 2 in 57 vs
    55 % of blocks (SCOREBOARD 73836a3). Stacked with P1-K1: ~0.25 GB of pass-1 reads per check at C 51 (projected;
    the stack is not map-tested). Offered as its next arm.
  * MTP Head: KWALL (a) PASS (mid rows top-3 0.888, union 0.903); DIVERSE2 not finished, resumable.
- Tue 14:51: P1-K1 TIMING (Fit12 14:39-14:46, one quiet lock; ~/Fit12/PLAN.md 'T3P + P1-K1 RESULT').
  * KILL AS REGISTERED: summed tokens / summed decode s, t80ck1u / t80ck2 = x0.849 (C 51 replicate x0.848).
  * Pass 1 per check 145.9 -> 103.9 ms; pass-1 bytes 950 -> 461 MB = x0.485 (my CPU x0.465). The 4 columns add 7.9 ms
    of GPU gap.
  * But kept per check 11.88 -> 10.11, and pass 2 / exact 37.7 -> 84.1 ms per check. Per prompt: /68 x1.31, /42 x1.02,
    /156 x0.96, /91 x0.60.
  * The confound (Fit12, post hoc): on /91, t80ck2 diverged from ex at token 117 and stopped at EOS at 177. ex and
    t80ck1u ran to the 256 cap through a hard stretch (ex kept 4.72 per check there), so t80ck2 skipped 80 hard tokens.
  * On the 630 tokens all arms share (each prompt up to t80ck2's divergence from ex), the arms take the same checks,
    paths and draft cf (kept 14.32 vs 13.91). Time ratios there: t80ck1u x1.100 (C 51 x1.098), exactly my line;
    3-bit top-2 x1.122; 3-bit P1-K1 x1.195 (x1.156). Excluding /91, the registered metric gives x1.07.
  * T3P's split: pass 1 = 57 % SSD wait (bandwidth-bound, ~11 GB/s; 950 MB per pass = 1.29x the replay) + 43 % GPU gap
    (~1.3 ms per layer at 16 rows). Pass 2's <= 4-row exact calls are bandwidth-bound too. Open: the commit phase reads
    49-137 MB from disk per check with 0 loader misses (9-18 ms).
  The verdict stands as registered (KILL). The flaw is in the metric: a FAST arm changes the text, so summed tok/s over
  different texts measures the texts. Hence a new, pre-registered test (below).
- Tue 14:51: PRE-REGISTERED (before any run) P1-K1 TIMING RE-TEST, shared-text metric (Fit12's proposal, my call:
  queue it; the slot is the orchestrator's).
  * Arms: t80ck2 (control), t80ck1u (P1-K1, KU 4), t80ck2m3 and t80ck1um3 (3-bit pass 1).
  * Prompts: 8+ HumanEval (the 4 above + 4 or more new, chosen before the run). 256 tokens, C 47 verdict, C 51
    replicate, balanced arm order, one quiet lock (~10 min).
  * Metric: per prompt, S = the longest prefix on which every compared arm's committed tokens equal ex's. Each arm's
    decode time over S = its checks' times, prorated by tokens for the check that straddles S's end. Ratio = sum over
    prompts of the control's time / the arm's time. The registered summed metric is reported beside.
  * Line (as before): t80ck1u >= 1.10x = PASS, <= 1.03x = KILL, else BETWEEN. t80ck1um3 is judged on the same line
    only if Breakthrough's stacked map arm (P1-K1 KU 4 at 3 bits) clears its certificate. If S covers < 50 % of the
    generated tokens, the run is VOID for the verdict.
- Tue 14:52: AMENDMENT before any run. The P1-K1 timing re-test adopts Fit12's pre-registration as written (Fit12
  PLAN.md 'T3R', screen scr_t3r; my OK given).
  * One C 47 process, quiet, ~10 min, MEM 12 GB.
  * Prompts: T3P's 4 (HumanEval 156 / 42 / 68 / 91) + the next 4 code keys in suite order (16, 6, 79, 19).
  * Arms: ex, t80ck2, t80ck1u, t80ck2m3, t80ck1um3.
  * Metric (scripts/t3_shared.py): shared-text tok/s, cut per prompt at the first token where the two compared arms'
    outputs differ (pairwise vs t80ck2). This replaces my 'every arm equals ex' S, which is stricter and shares less text.
  * Line unchanged: >= 1.10x PASS, <= 1.03x KILL. No verdict if shared text covers < 50 % of tokens.
  * On T3P's data the script gives x1.100 (C 51 x1.098): a coin flip at the line.
  * For t80ck1um3's verdict: Breakthrough's stacked map arm, P1-K1 with Fit12's KU 4 cap at 3 bits. Masks
    ~/NewIdeas/data/dm_masks/oea_k1_ku4: oea_k1 capped to each row's first 4 kept ranks; 4.3 % of masked row-layers
    capped; rank 0 always kept; a subset of oea_k1. Read-out: p1_cert_read.py p1_top_k2 <run>; the line is P1-K1's.
- Tue 14:53: EvalSuite 11:12: RELAX smoke PASS (cb2f778); T0 (the 12 GB same-scores test) is held to Thursday.
  Breakthrough 11:10-11:11: 3-bit pass 1 PASS; plan = two quality-cleared pass-1 levers (3-bit, P1-K1), stacked
  ~0.25 GB [P] (73836a3).
- Tue 14:54: orchestrator 14:5x: both set for tonight. (1) Fit12's T3R 00:30-00:45, first in line, quiet; BigMoE's
  turns follow from ~00:45 (<= ~41 GB peak). (2) At the head of Breakthrough's FLOOR after BigMoE (~03:45): the stacked
  arm p1_oea_k1_ku4_t3 (P1-K1, KU 4, 3 bits; all rows), then top_k8 if room. Exact arm list sent to Breakthrough
  (14:54). Not BIG (agreed); T3P's 57 / 43 split puts Leap's dense work on the 12 GB board.
- Tue 14:56: P2-PREFETCH (12 GB, EXACT lever on top of the two-pass check; CPU screen scripts/p2_prefetch_screen.py).
  Fit12's T3P: pass 1 is 57 % SSD wait + 43 % GPU gap (~1.3 ms per layer), so the SSD idles in every layer's gap. Pass
  2's exact calls are bandwidth-bound (78 ms of wait per pass). Idea: in pass 1's gap at layer l, read layer l's
  experts (ranks 1-9 not in pass 1's union) for rows predicted to fail the certificate, up to B per layer. Pass 2
  then finds them resident. Only timing changes, so tokens stay identical to the arm without prefetch (exact).
  Prior art: draft-phase prefetch of the target's experts (MoE-SpeQ 2511.14102, SP-MoE 2510.10302). Ours: the
  window is pass 1's GPU gap, and rows are picked by drafter certainty. Tag: adapted.
  Replay: Breakthrough's dm_rankcache LRU at C 51 on code, P1-K1 pass 1, p1_oea_k1's certificate at 0.8; B = 6 per
  layer (1.3 ms x 11 GB/s / 2.2 MB). Pass-2 misses per check 156.3 -> 81.0 with cf < 0.99 (oracle 56.3). Pass-1
  misses +12.4 (evictions). Prefetch reads 279, of which 85 used. Model -12.6 ms per check (oracle -20.0; B 3 -8.1;
  reach gating adds nothing). On P1-K1's ~170 ms check that is ~x1.08. Stacked with 3-bit P1-K1 (x1.16-1.20 on shared
  text), the 12 GB code check models at ~x1.25-1.30 vs t80ck2: ~70-73 tok/s, at T3's 70 line.
  Caveats: the replay's single LRU vs Fit12's two pools (C_m 47 / C_l 32, MSB-3 + LSB parts); pass 2's real row set
  (option (a): rows n..j) is larger than 'uncertified rows'; prefetches must finish inside the gap without delaying
  the next layer's demand reads.
  PROPOSED to Fit12 (pre-registration text, its call to wire): arm t80ck1u_pf = t80ck1u + prefetch for rows with
  drafter cf < 0.99, ranks 1-9 not in U, B 6 per layer, issued after U's posts in each wide pass-1 call, lower
  priority than demand reads. Gate: tokens and lsha equal t80ck1u's (exact). Line (on identical text, so no
  confound): tok/s >= 1.05x t80ck1u = PASS, <= 1.01x = KILL. Also report pass-2 misses and ms per check, prefetch
  reads used, and pass-1 misses.
- Tue 14:56: Breakthrough 14:54: both arms head FLOOR night 1 (step S0, after BigMoE): p1_oea_k1_ku4_t3, then top_k8;
  JSONs as decision_map/out/<name>.json. The stack arm's cut rows take their kept ranks from records3 at 3 bits
  (routed_topk K 10 with my rank mask; test_t3 vs a dequantize reference 1.9e-2, ok); uncut rows exact. My lines are
  in its SCOREBOARD ceeda3f. Fit12 14:54: T3R armed for 00:30-00:45; t80ck1um3 stays INFO until the stack arm's
  certificate clears; results by ~00:50.
- Tue 15:02: Fit12 15:0x: P2-PREFETCH wired and pre-registered with my line (Fit12 PLAN.md 'T3R-PF').
  * Slot: a second ~3-min process after T3R tonight, only if T3R ends by 00:38 with the SSD <= 62 C; else the next slot.
  * Mechanism (no ext change): pass 1's base-pool post carries up to 6 extra ids with the loader's late flag (bit 21;
    BIGMOE_DEFER 8 in T3's env). The loader frees the GPU for the waited ids, then reads the late ones; pass 1 never
    uses their slots.
  * Ids: ranks 1-9 of rows with draft cf < 0.99, minus ids already posted, in row then rank order, deduped, base pool
    only (rank-1 LSB parts are not prefetched).
  * Arms: t80ck1upf vs t80ck1u (primary) and t80ck2pf vs t80ck2 (second verdict), each in the same process. Prompts
    156 / 42 / 68 / 91, balanced 2:2.
  * Gate: tokens + lsha equal to the base. Line: >= 1.05x PASS, <= 1.01x KILL.
  * Synthetic tests pass: late ids follow my rule, and output with prefetch on equals off bitwise, for K2 and K1U.
  * The run reports gap overrun per layer, loader late reads, late ids forced to wait (must be 0), pass-2 ms and misses,
    and pass-1 misses.
- Tue 15:09: RANKED Leap's PURGE-1 (orchestrator's ask, 15:0x). Measured (Leap, 15:05): volatile purgeable slabs are
  reclaimed after the file cache and before any compression or swap; toggles cost 0.2-2 us.
  * Value: it lets RANKED #4's C 300 / 320 caches (blocked on a 55 GB free-memory night) run any night as 'C 256
    resident + an elastic volatile tier', and partly by day. MoonShots' replays: C 320 roughly halves code SSD MB per
    kept token vs C 256 (no Q1 46.3 -> 22.9; with Q1 27.8 -> 15.0). SSD wait is ~half of today's 84 ms check, so the
    model is +15-30 % by night when the tier fills to ~+8.5 GB, less by day. Exact (cache only).
  * Prior art: published OS practice. Apple's 'Caching and Purgeable Memory' guide and Metal's MTLPurgeableState on
    buffers / heaps; OS-reclaimable weight caching also exists as mmap'd weights in the page cache (llama.cpp). No
    LLM / MoE expert tier found. Tag: adapted.
  * ACCOUNTING (my measurement, scripts/native/purge_footprint.c): volatile memory is invisible to the RAM-budget
    metric. 256 MB touched: phys_footprint 257 MB; set volatile: 1.1 MB; memory written while volatile is never
    charged, not even to ri_lifetime_max_phys_footprint. So the tier must be reported separately (tier peak beside the
    footprint). For the 24 / 12 GB lines it counts or stays off. For 45 GB it is allowed ('more RAM OK'), reported.
  * By day the tier outlives Kamen's apps' file cache under pressure (file cache goes first), so cap it by day.
  * Rank: top of the exact 45 GB queue with NAX6, as the enabler of #4 (new row 33). Routed to BigMoE + Leap.
- Tue 15:42: peers 15:1x-15:4x:
  * Leap: PURGE-1's footprint blind spot confirmed for MTLBuffers (256 MB shared buffer 259.7 MB non-volatile -> 3.7 MB
    volatile; written-while-volatile never charged; the lifetime peak keeps the non-volatile high-water). TABLE_350
    carries the rule: the tier counts in full for the 24 / 12 GB lines.
  * BigMoE: the volatile expert tier is its job, first on Wed by day (the orchestrator's order; then the no-LSB mask).
    Design: per-layer purgeable MTLBuffer victim slabs behind the LRU pools. Evictions copy in, misses copy out, slabs
    are pinned only during the layer's service, and the tier peak is reported beside the footprint. Armed for Wed
    00:45 at C 224 (<= 41 GB): mission 20 -> combination.
  * MTP Head 15:37: TRIED corrections (an off-by-one reading D-24's MTP rows). KWALL (b) mid 0.855 / tie 0.340 (BETWEEN
    stands); KWALL (a) union 0.796 (PASS stands on top-3). MTPPICK = KILL on paper (exact loop: kept +7 % code, est.
    tok/s x0.93); TOP3-USE desk = KILL. TRIED re-merged.
  * EvalSuite 15:13: T0 (the 12 GB two-pass same-scores test) still 28/289 rows; the rest needs ~3.5-4 h of one-job GPU
    at MEM 14. It asks me (I order the 12 GB queue): Wed by day after FLOOR, or Thu after RELAX. My call: Thu by day
    after RELAX, as planned. 45 GB work goes first, BigMoE's volatile tier is first on Wed by day, and T0's result
    doesn't feed RELAX. Wed by day only if the orchestrator finds an idle one-job gap after FLOOR and BigMoE's tier.
- Tue 15:43: MTP Head 15:42: DEPTH32 = KILL (HYB+G 12.60 / 6.10 per 32-row check vs line 14 / 7.5; MTP-32 alone 12.55
  beats the hybrid). T3's commit-phase disk reads (49-137 MB per check, 0 loader misses): the embed memmap is not it
  (<= 8 rows per pass). My guess for Fit12 (lounge note, low priority): swap-ins of the process's own pages; vm_stat
  swapins / pageins deltas per check would show it.
- Tue 15:46: PRICE of the elastic tier (PURGE-1) on live cap45 checks. scripts/band45_oea.py with BAND45_SIZES =
  MoonShots' per-layer FLOOR sizes laysz45_300 / _320 (39.75 / 42.45 GB of pools vs C 256's 33.93); data/band45_oea/
  band45_c300.txt / _c320.txt. SSD MB per kept token, all six prompts (code / math):
  * EXACT: C 256 63.1 (111.4 / 51.4); C 300 46.3 (92.6 / 35.1) = -26.6 %; C 320 40.7 (86.9 / 29.6) = -35.5 %. Code gains
    less than math (-22 vs -42 % at C 320).
  * Q1 0.9 at C 320: 29.3 (-54 % vs C 256 EXACT); + 3-bit sure rows: 24.1 (-62 %); REACH 0.9 0.5: 27.3.
  * Model, SSD-only (SSD ~42 of 84.2 ms per check, bytes fully exposed): +5.8 GB of tier (C 300) ~+17 % (74 tok/s); +8.5
    GB (C 320) ~+21.5 % (77 tok/s); with Q1-UNION 0.9 at C 320 ~+36 % (87); + 3-bit sure rows ~+45 % (92).
  Caveat: a victim tier behind C 256's pools is not the same policy as per-layer-fitted C 300 / 320 pools, so treat these
  as the target, not the floor. Sent to BigMoE as the tier's priced target.
- Tue 15:47: VICTIM-TIER replay (scripts/victim_tier.py: band45's machinery; C 256's resident pools + a per-layer
  LRU victim tier; a tier hit costs no SSD read and moves the record back; 2 seeds). SSD MB per kept token, all six
  cap45 prompts:
  * EXACT: C 256 62.5; + tier sized to C 300's extra 45.8 (-26.8 %; fitted C 300 -26.4 %); + tier sized to C 320's
    extra 40.2 (-35.7 %; fitted -35.2 %).
  * Q1 0.9: + tier(320) 29.1 (-53.4 %; fitted -53.2 %).
  So BigMoE's planned victim design gets the full benefit of bigger fitted pools.
  * PREDICTION for BigMoE's tier smoke tonight (vt1_224 = C 224 + 64 base / 32 sidecar slots per layer, ~7.8 GB, ~04:00;
    tier assumed never purged): 79.2 -> 52.1 (-34.2 %) vs C 224 alone (C 256 alone: 62.5); with Q1 0.9 50.4 -> 34.9.
    Sent to BigMoE as a lounge note.

- Tue 17:01: PRE-REGISTRATION, Q1-ON-MISS ON LIVE-CUT CHECKS (FAST, 45 GB; before any GPU run). Fable C's C1-2
  (SCOREBOARD 00d485c, Mon 23:0x), superseded then by Q1-UNION because the outputs depend on the cache state. Revived
  because Q1-UNION 0.9 is BETWEEN on chat at live lengths: its union members nearly vanish on 3-4-row chat checks.
  * Arm: rows whose drafter cf >= theta keep ranks 0-1, the check's union members, and every rank 2-9 expert that is
    resident in the base pool after the check's loads. Only the rank 2-9 experts that would MISS are dropped (weight 0,
    no renorm). So the SSD reads equal Q1's at the same theta by construction. GPU work is about exact's union minus
    the dropped misses, so the gain is SSD-only.
  * Masks (CPU, built): scripts/dm_oea_masks.py arm qom_<theta>, suffix _live; data/dm_masks/qom_{0.9,0.5,0}_live
    (+ qom_0.99_live spare).
    - Replay: plain LRU per layer at laysz45_256's C_m (mean 295 of 512), carried across a domain's windows, warmed by
      the prompt rows' top-10. Each check loads its union, then the kept resident experts are touched.
    - Self-check: QOM_C=0 reproduces Breakthrough's q1u_cf09 bit for bit (36/36 windows).
    - The replay is conservative: base-pool misses per check run ~1.05-1.25x band45's LRU + popularity (code 6.4 %
      of loads, chat 2.1-2.3 %), so the masks drop slightly more than the engine would.
  * Kept share of sure rows' rank 2-9 pairs (code / IFEval / multi):
    - qom_0.9_live: 94.5 / 97.7 / 97.9 % (q1u_0.9_live keeps 37 / 43 / 47 %);
    - qom_0.5_live: 93.8 / 97.0 / 97.4 %;
    - qom_0_live (every row with a draft): 93.6 / 96.8 / 97.2 %.
  * Price: SSD MB per kept token on Breakthrough's capture (live-cut checks), reads = Q1's.
    data/band45_oea/band45_capture_qom.txt; vs EXACT 47.0 / 31.1 / 30.7:
    - theta 0.9: -50.3 / -19.7 / -16.4 %;
    - theta 0.5: -60.4 / -36.9 / -33.4 %;
    - theta 0: -64.3 / -58.5 / -62.4 %.
    - Code SSD-only tok/s model (C50's HE68 / HE91, bytes fully exposed): EXACT 84.7 / 62.1; 0.9: 125.4 / 78.7;
      0: 144.9 / 85.0 [P, halve when deciding].
    - Chat checks are less SSD-bound, so chat tok/s gains are smaller than chat byte cuts.
    - After QOM, code reads are mostly the ranks 0-1 LSB sidecar (118 of 142 reads per check at theta 0). The matching
      'LSB-on-miss' (serve a rank 0-1 at 3 bits when its LSB part misses) is the next step and needs a per-(row, layer)
      bit-depth mask in the harness.
  * SLOT (the orchestrator 17:1x): Wed after 15:00, not tonight's FLOOR.
  * Runs (Breakthrough's map, ~4 min each, priority order): qom_0.9_live, qom_0.5_live, qom_0_live. Harness as for the
    union arms: PERT.update(rmask=<absolute dir>, renorm=False).
  * Lines, per domain (the union arms'): all-row flips <= Q1 at 0.99's level (code 1.34 / IFEval 5.42 / multi 3.69 %) =
    PASS; <= 1.5x = BETWEEN; above = KILL.
    - Paired McNemar vs gate_top2_cf099 and vs q1u_0.9_live.
    - Read-out: scripts/qom_read.py -> data/QOM_read.txt.
    - A theta passes only where its domain passes. The lowest passing theta per domain is the FAST candidate.
  * FLAG FOR KAMEN (with the result): outputs depend on the cache state, so two runs with different histories are not
    token-identical. Within one engine process the same history gives the same tokens.
- Tue 17:57: BREAKTHROUGH PUSH, shortlist sent to the orchestrator 17:51 (research/PUSH_SHORTLIST.md, e4b6953): all four
  thinker files priced on our data and ranked. Two checks of my own:
  * Per-call SSD latency share under FAST stacks (scripts/calls_count.py; cap45, C 256, Fable A's fit 0.283 ms per call +
    MB / 12.7 GB/s): EXACT 39 %, U3 51 %, HYB u8 + t3 54 %, QOM theta 0 54 %. JOINT cuts loader calls 70.8 -> 41.9 per
    check. So Fable A's D1 is worth more on top of the FAST stack. QOM-full (theta 0 + LSB-on-miss) = 10.4 ms busy per
    check (9.6 with JOINT) vs EXACT 51.4.
  * Fable B's LSB-adjacency merge (ROW_WALL sec 4) is overstated (scripts/lsb_both.py). Only 11.1 % of LSB misses
    coincide with an MSB miss of the same expert, so it saves 5.0 % of reads (3.9 % under U3), not 25-30 %.
- Tue 17:57: ROUTE-W REPLAY (the Lunatic's Route W, LUNACY2 sec 2, pre-registered there: PASS <= 60 expert groups per kept
  token, KILL > 80 at 8 streams, Q1-UNION, mixed domains). scripts/route_w.py -> data/ROUTEW_read.txt: merged passes of S
  random map windows (36: 12 code, 24 chat), blocks = live-cut checks.
  * Groups per kept token:
    - 1 stream: EXACT 465.6, Q1U 0.9 356.3;
    - 8 streams: EXACT 236.9, Q1U 0.9 162.5, Q1U 0.5 128.9, HYB 0.9 u8 140.3;
    - 12 streams: Q1U 0.9 144.6, Q1U 0.5 117.3.
    - Verdict: KILL at today's drafter (8 streams with Q1-UNION 0.9 = 2x the KILL line).
    - The walk16 shape (all 16 rows, no cut) is worse: 291.4 at 8 streams. The extra rows are mostly rejected.
  * Aggregate tok/s by the Lunatic's own pass model (whole model in RAM, draft hidden, no SSD) [P]:
    - 1 mixed stream: EXACT 78, Q1U 0.9 85;
    - 8 streams: EXACT 302, Q1U 0.9 391, Q1U 0.5 451;
    - 12 streams: EXACT 358, Q1U 0.9 451, Q1U 0.5 515.
  * Reading:
    - Width is worth x4-5 in aggregate even at today's drafter, at ~50 tok/s per stream. ~1000 needs ~3x more kept per
      row (the 'horizon fix', 32-row blocks at agreement ~0.95), which is the same drafter wall as single stream.
    - Chat blocks keep only 1.9-2.3 tokens, and they dominate the groups.
    - Whether an aggregate over parallel sessions counts is Kamen's call (shortlist decision 1).
- Tue 19:15: T3R RESULT (Fit12 19:01-19:08, run early at Kamen's request for small tests now; PLAN.md 'T3R NOW';
  t3r_verdict.py on data/scr/t3rn). Registered shared-text metric: t80ck1u / t80ck2 = x1.065, coverage 88 / 84 %.
  * That is BETWEEN on the line (>= 1.10 PASS, <= 1.03 KILL), but VOID by the runner's standing rule.
    - A heat PAUSE 19:07:51-19:08:31 (SSD 72 C) fell inside the run. 36.9 s landed in one pass-1 call of /19's 3-bit
      arm.
    - /19's verdict pair straddled the cool-down: t80ck1u ran at the SSD's peak (~4.8 GB/s pass-1 reads vs ~7.2),
      t80ck2 after it.
    - Per prompt: x1.27 1.11 0.93 1.02 1.08 1.30 1.18 0.79. Exact arm reproduced bitwise (t80ck2 == T3t tokens + lsha,
      4 / 4).
    - Post hoc without /19 (NOT the registered metric): x1.114.
  * INFO: t80ck1um3 (P1-K1 KU 4 at 3 bits, the stack Breakthrough's map scores tonight) x1.227 on shared text (coverage
    76 / 73 %). T3P's x1.195 (4 prompts) held on 8.
  * T3R-PF did not run: SSD 63 C at r47's end, over its 62 C gate.
  * MY CALL (I order the 12 GB queue): option (a), a clean re-run, widened (the 8-prompt spread x0.79-1.30 is too wide
    for a +-5 % line).
- Tue 19:15: PRE-REGISTRATION, T3R2 (before any run). Same arms (t80ck1u vs t80ck2, plus t80ck1um3 as INFO), same
  shared-text metric, same lines (>= 1.10 PASS, <= 1.03 KILL, no verdict below 50 % coverage).
  * Prompts: T3R's 8 plus 8 more HumanEval prompts that Fit12 picks by a fixed rule written before the run (the next 8
    of the T3 pool by index). The verdict is the POOLED ratio over all 16; the original 8 are reported beside.
  * Runner rules: start only with the SSD <= 55 C and no PAUSE. A PAUSE inside a prompt's verdict pair voids that pair,
    which is re-run once at the end. Report SSD C at each pair.
  * Then T3R-PF as registered (fd081ae), on its own gates.
  * AMENDMENT 19:27 (before any run). Fit12 implemented T3R2 (PLAN.md 'T3R2'): prompts = T3R's 8 + HumanEval positions 6-13
    (161, 17, 56, 141, 101, 63, 114, 77); arm order alternates; PAUSE-void / re-run / exclusion rules dry-tested.
    - Added: a verdict needs >= 12 of the 16 prompts counted (excluded = voided twice); fewer gives no verdict.
    - The pooled ratio is reported either way.
  * Slot: the orchestrator placed it Wed ~15:00, first in line (Fit12 holds it). If Breakthrough's map prefix can't run
    tonight, it goes after Breakthrough's Wed arms, then T0. The Mac should be cool by day.
  * SLOT MOVED (the orchestrator Tue 22:3x, Kamen's Wed research day 13:45-23:30): T3R2 runs Wed 13:05-13:40, after
    FLOOR's 13:00 deadline; every batch test ends before Kamen arrives.
  * HARNESS NOTE (Fit12 22:5x, before any run; timing and metric untouched; dry-tested on 4 cases incl. a cut mid main
    pass and mid re-runs):
    - a paused prompt's files move to void1_* as soon as its PAIR line says VOID, not at its re-run, so a run cut by a cap
      or the hard stop cannot leave it counted;
    - t3r2_verdict.py counts a prompt only if its LAST PAIR line (main or re-run) says ok.
    - Deadline rule: r47 starts only with >= 10 min left to 13:39 (it takes ~7-9 min); p47 (T3R-PF) only with >= 260 s.
    - Leap's stale-slot rule (S7B-PROBE) does not touch the T3 path: passes and calls are synced, one post per pool per
      layer per call; G1 bitwise at C_m 47 (Fit12 PLAN.md 'T3R2').
    My read: both changes only tighten the registered void / 12-of-16 rules. Accepted as registered.

- Tue 22:59: RESULT, Q1-ON-MISS (QOM) on live-cut checks (pre-registered 554dfe0; Breakthrough's prefix, decision map,
  36 windows; scripts/qom_read.py -> data/QOM_read.txt). PASS on all three domains at theta 0.9 and theta 0.5:
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | line: Q1 at 0.99's level (gate_top2_cf099) | 1.34 % | 5.42 % | 3.69 % | |
  | q1u_0.9_live (today's FAST lead in RELAX) | 1.25 % | 6.45 % | 3.77 % | PASS / BETWEEN / BETWEEN |
  | qom_0.9_live | 0.48 % (16) | 2.24 % (76) | 1.44 % (51) | PASS / PASS / PASS |
  | qom_0.5_live | 0.36 % (12) | 2.68 % (91) | 1.63 % (58) | PASS / PASS / PASS |
  * Paired McNemar, qom_0.5_live vs q1u_0.9_live: code 6 / 36 (z -4.63), IFEval 39 / 167 (z -8.92), multi 27 / 103
    (z -6.67); vs gate_top2_cf099 z -5.15 / -6.80 / -6.28. theta 0.9 the same picture (z -3.9..-9.8).
  * For scale: oea_k8 on 16-row checks 0.62 / 3.83 / 3.07 %, cold6_all 0.65 / 5.00 / 4.84 %. QOM has the lowest flips
    of any FAST arm on the map, at Q1's SSD reads.
  * The FAST candidate (the lowest passing theta per domain, as registered): theta 0.5 on all three, until qom_0_live
    runs (Breakthrough 22:59: after BigMoE's night ticket, ~04:40-05:00; the prefix lost it to five quiet windows).
  * Price [P] (capture, live-cut checks, C 256; reads = Q1's): SSD MB per kept token -60.4 / -36.9 / -33.4 %;
    SSD-only model, code HE68 / HE91: EXACT 84.7 / 62.1 -> 138.9 / 83.2 tok/s (halve the gain when deciding).
    The gain is SSD-only: GPU work stays near exact's union (94-98 % of sure rows' rank 2-9 pairs kept).
  * Caveats:
    - The masks' cache is a plain LRU at C 256's per-layer C_m (mean 295 of 512). At C 160 the engine would drop more,
      so the by-day C 160 line needs its own map arm (QOM_C=164).
    - Map flips are teacher-forced per-token flips; EvalSuite's same-scores tests decide FAST adoption.
    - FLAG FOR KAMEN (decision 2): the outputs depend on the cache state, so runs with different histories are not
      token-identical; within one engine process the same history gives the same tokens.
  * Reproducibility (MoonShots' sweep, Tue 23:0x): PSR's logical cache (arXiv 2610.34721) decides against a residency set
    that changes only through routing, so copies, prefetch timing and OS purges change latency, never tokens.
    MoonShots' CPU screen (~30 min; WED_1104_READY, Breakthrough) tests it on QOM's mask builder. It does not remove
    the dependence on the session's own history, which stays decision 2.
  * NEXT: (1) the 3-bit bound below (pre-registered now); (2) engine: U3's 3-bit sure-row path + 'keep resident ranks
    2-9 on sure rows' as a FAST flag (BigMoE), then EvalSuite's same-scores; (3) Kamen's decision 2.
- Tue 23:0x: PRE-REGISTRATION, QOM + 3-BIT SURE ROWS (qom_0.5_live_t3; FAST, 45 GB; before any GPU run). The quality bound
  for QOM-full (LSB-on-miss: a sure row's rank 0-1 served at 3 bits only when its LSB part misses).
  * Arm: the qom_0.5_live masks with ranks 0-1 off on this check's sure rows (cf >= 0.5; never the live bonus row), so the
    harness's ('t3mask',) gate serves those ranks from records3 at 3 bits: PERT.update(rmask=<abs>
    /Users/kamen/NewIdeas/data/dm_masks/qom_0.5_live_t3, gate=("t3mask",), t3=3, t3k=2, renorm=False) (hyb_t3's
    convention). Masks: scripts/dm_oea_masks.py arm suffix _t3; the base-pool replay is unchanged, so ranks 2-9
    equal qom_0.5_live's bit for bit. Build: started 23:00, stopped 23:01 when BigMoE's quiet lock began; it
    runs again after the lock (04:28), ~04:30-04:50, and is checked against qom_0.5_live before any run.
  * Why it bounds QOM-full: the same SSD reads on sure rows (no LSB read for them) and 3-bit on every sure-row rank 0-1,
    not only on the misses. So its flips are never better than QOM-full's.
  * Lines (the same, per domain): all-row flips <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %) = PASS; <= 1.5x = BETWEEN;
    above = KILL. Paired McNemar vs qom_0.5_live (the 3-bit cost) and vs gate_top2_cf099. PASS here = QOM-full
    passes on the map without a new bit-depth harness.
  * Price [P] (capture, C 256; reads = Q1 0.5 3b): SSD MB per kept token -80.7 / -52.6 / -44.9 %; SSD-only model code
    HE68 / HE91 177.0 / 93.9 tok/s (halve). Replay busy per check at theta 0 + 3-bit: 10.4 ms vs EXACT 51.4 (calls_count).
  * Conditional arm: qom_0_live_t3 (masks built) runs only on the domains where qom_0_live passes.
  * Slot: Breakthrough's map, ~5 min per arm; asked of Breakthrough 23:0x: right after qom_0_live (~05:00) once the
    masks are checked, or Wed morning before FLOOR's 13:00 deadline (Breakthrough's call).
> (a quote from the project owner, kept private)
  A FAST mode may depend on the cache state. Conditions: EvalSuite certifies it under >= 2 different prompt orders
  (histories); EXACT stays the default and history-independent; the logical-cache variant (PSR) is welcome later.
  Recorded in FOR_KAMEN / PUSH_SHORTLIST / SUMMARY_FOR_KAMEN; QOM is FAST_TRACK item 0. Order: cf 0 (~05:00) ->
  qom_0.5_live_t3 -> the C 160 arm below -> BigMoE's engine path -> EvalSuite (two prompt orders).
- Tue 23:2x: PRE-REGISTRATION, QOM AT C 160 POOLS (qom_0.5_live_c164; FAST; before any GPU run). The 22:59 masks assume
  C 256's per-layer base pools (laysz45_256 C_m, mean 295 of 512). By day the engine runs at C 160 (uniform C_m 164), where
  more rank 2-9 experts miss, so QOM drops more. This arm measures that cost.
  * Arm: the qom_0.5_live rule with the replay's base pool at 164 slots per layer (dm_oea_masks.py suffix _c164; same
    prefill warm-up, same live-cut checks). Setup as the QOM arms: PERT.update(rmask=<abs>
    /Users/kamen/NewIdeas/data/dm_masks/qom_0.5_live_c164, renorm=False).
  * Lines (the same, per domain): all-row flips <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %) = PASS; <= 1.5x = BETWEEN;
    above = KILL. Paired McNemar vs qom_0.5_live = the cost of the smaller cache. The mask builder's kept share of sure-row
    rank 2-9 pairs is reported beside it (C 256: 93.8 / 97.0 / 97.4 %).
  * Reads: Q1 0.5's at C 160 (the cut itself does not depend on C; the misses it avoids do).
  * My prediction [G]: still PASS on all three, at a kept share ~2-4 points lower and flips +0.1-0.5 points.
  * Masks: built with the t3 sets after BigMoE's quiet lock (~04:30); READY marker as for the t3 arms. Slot: asked of
    Breakthrough (FLOOR night 1's gaps before 12:45), after qom_0.5_live_t3.
- Tue 21:25: PRIOR ART, STRATA (the orchestrator relaying Kamen's r/LocalLLaMA read). Sources: akitaonrails.com
  (2026-10-22, the author's own 5090 runs), besthub.dev "How Strata Runs a 125B Model on a 12GB Gaming GPU", aiweekly,
  discuss.pytorch.kr, insiderllm review. All secondary; the GitHub README (Niko1221/Strata) was not read directly.
  * What it is: an MIT single-model engine on llama.cpp / ggml for Qwen3.8-Flash-Next, OUR 125B (512 x 48 = 24,576
    experts, top-10 + shared, 6B active, a ~29 GB SSD lookup table = the n-gram table).
  * Methods:
    - experts tiered VRAM (most-used; 90-97 % hits on a 5090 with 20-22 GiB) / RAM (all) / SSD (the table);
    - the CPU computes missing experts from RAM while the GPU computes cached ones (Fiddler / kTransformers style);
    - the model's MTP head for speculative decoding (--spec 4, --spec-min-p 0.5; README claims 1.6-1.8x);
    - int8 KV (--kv int8) and the 32K most-read KV positions resident in VRAM;
    - IQ2 / IQ3 / Q2_0 quants; a 'Coder' variant that drops half the experts.
  * Measured by others: 5090 ~150-190 tok/s (whole model in RAM, no SSD streaming, 1.8 TB/s VRAM); RTX 5070 12 GB Q2_0
    94 tok/s (README). The Reddit 150 vs 70-80 on 144 GB of VRAM at 6-bit is mostly its MTP speculation (model fully in
    VRAM).
  * For us: nothing new.
    - Tiered caching, CPU expert compute and MTP speculation are published (one review: 'existing llama.cpp ideas').
      Fiddler-style CPU compute doesn't apply on unified memory (DRAM is shared; WHOLE-CHIP is our analog).
    - KV residency doesn't apply on unified memory, and expert pruning is against the rules.
    - The one partial gap: 8-bit KV as a long-context SPEED lever at 45 GB (FAST). TRIED has it only as a 12 GB
      memory lever (L2-06 KV8 KILL on footprint) and MoonShots' parked napkin (+20-45 % at 100K). At 10K keys
      attention adds ~23.5 ms per check (FIXED-PART), so KV8 is worth pricing for agent turns: a Fable B / Leap item.
- Tue 21:39: QUEUED (the orchestrator 21:3x): F1 FAST-K CERTIFICATION with EvalSuite, after Thursday's RELAX. F1 =
  Leap's fs4 (FATT16) in place of REF2-T's chunk set in the 12 QSA layers. Attention per check ~23.5 -> ~13 ms at 10K
  (B6: 1.70x layer level, INV PASS, more accurate than MLX vs fp64). Ranked FAST_TRACK 1c.
  * Prerequisites (owner Leap -> BigMoE): the engine switch, then a token / INV smoke (16-row and 8-row checks at 2K /
    10K).
  * EvalSuite designs and pre-registers the same-scores test. The request:
    - rows: long-context / agent (AT14's 14 turns at 9.1-12K prefill; long-context items if the suite has them);
    - arms: REF2-T control vs F1;
    - read-out per turn: attention ms per check, turn tok/s, prefill share;
    - line: EvalSuite's standard FAST same-scores rule.
  * KV8 (Leap 21:3x): BETWEEN, not pursued.
  * Tue 21:53 PRE-REGISTERED by EvalSuite (63cd47e, ~/EvalSuite/docs/PREREG_f1_2026-11-03.md): decision 4 on the core 210
    vs ref2t plus RULER 24 at 16K (<= 1 net lost); stage 2 on IFX's 220 only if IFEval alone fails; agent 29 as flags
    (agreement vs floor, copy breaks). The core 210 is in because the switch routes every 16- / 8-row verify call, so
    short answers change too. AT14 timing A1 / F1 / A2 on the live loop, C 256, quiet. ~3.5 h GPU; earliest Fri, after
    the conversion checks. EvalSuite's note: AT14 turns re-prefill the whole context, so turn tok/s moves only ~+0.5 %;
    the decode +2-4 % transfers. The engine switch is Leap's Wed by-day item (its status line).

- Wed 03:27: RESULT, qom_0_live (QOM at cf 0 = every drafted row sure; the live bonus row stays exact; Breakthrough's
  night prefix, landed 03:26). Lines as registered: all-row flips <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %).
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0.9_live | 0.48 % (16) | 2.24 % (76) | 1.44 % (51) | PASS / PASS / PASS |
  | qom_0.5_live | 0.36 % (12) | 2.68 % (91) | 1.63 % (58) | PASS / PASS / PASS |
  | qom_0_live | 0.42 % (14) | 2.53 % (86) | 1.77 % (63) | PASS / PASS / PASS |
  * Paired McNemar, qom_0_live vs q1u_0.9_live: code 8 / 36 (z -4.22), IFEval 46 / 179 (z -8.87), multi 26 / 97
    (z -6.40); vs gate_top2_cf099 z -4.52 / -7.15 / -5.92. cf 0 costs nothing measurable over cf 0.5 on the map.
  * FAST candidate (the lowest passing theta per domain, as registered): theta 0 on all three domains.
  * READ WITH the 03:4x last-row note below: on the map the last row of every live check stays exact (cf -1); the price
    I gave for decision 2 (band45_capture_qom, capture convention) gated that row by its own draft.
- Wed 03:31: QOM MASKS READY for Breakthrough's after_bigmoe.sh (built 03:28-03:31 under a share lock, CPU only):
  qom_0.5_live_t3, qom_0_live_t3 (cf 0 passed, so it runs), qom_0.5_live_c164; 36 windows each.
  * Bit-check (scripts/qom_t3_check.py): ranks 2-9 equal the 4-bit twins bit for bit; ranks 0-1 off only on whole sure
    rows (every layer), never on a row the twin keeps at all 10; ALL OK. Rows at 3 bits: cf 0.5 51.9 / 58.6 / 57.9 %,
    cf 0 53.3 / 70.3 / 72.8 % of the windows' rows (code / IFEval / multi).
  * c164: ranks 0-1 never dropped; differences from qom_0.5_live only on sure rows. Kept share of sure-row rank 2-9 pairs
    80.5 / 89.3 / 90.3 % vs 93.8 / 97.0 / 97.4 % at C 256: -13.3 / -7.7 / -7.1 points. MY PREDICTION (-2-4 points)
    MISSED: the C 160 pool keeps far less. The flips line decides; expect more than the +0.1-0.5 points I predicted.
- Wed 03:4x: BYTES10 LEDGER RAN (research/TEN_X_LEDGER.md; data/ledger/). Three findings, in order of weight.
  * (1) FABLE C'S BYTES10 REPLAY HAD A KEEP-MASK BUG (~/FableGod/fablec/bytes10_replay.py:135: keep[sure, :ks] on a
    [rows, 48, 10] array slices layers, not ranks). Base (MSB) reads were counted on layers 0-9 only. Every level in its
    BYTES10.md / FINDINGS.md was 2-4x low: EXACT at 45 GB code 18.7 -> 60.4; 24 GB 96.5 -> 245.2; FAST start (QOM cf 0
    + t3) 24 GB 32.6 / 46.2 / 45.8 -> 61.7 / 86.1 / 83.1. The fixed copy (scripts/bytes10_replay_fix.py, one line) and
    my independent replay (scripts/ten_x_ledger.py) agree up to the miss-count order and match W1B's 62.4 [K].
    The screen-1 masks and the map's own scripts index correctly: no quality result is touched.
  * (2) 24 GB FAST on paper: QOM cf 0 + t3 61.7 / 86.1 / 83.1 -> + gated bonus row 30.8 / 20.0 / 14.2 -> + top-1 on
    sure rows 16.8 / 11.8 / 7.9 -> + rank-class pools 8.8 / 8.7 / 6.9 -> + 11-bit 7.3 / 7.2 / 5.7 [P] unhalved,
    ~25 / 33 / 32 halved. Screens 1-2 (13:45) decide the two quality-open levers; screen 3 calibrates. EXACT at 24 GB:
    OPT bound 117 / 94 / 84, so no exact path to 10.
  * (3) MY DECISION-2 PRICE GATED THE LAST ROW. band45_capture_qom (the numbers Kamen saw) took the capture's cf for
    every verified row, the bonus row included; the map's qom_*_live arms keep that row exact. QOM as tested, at 45 GB:
    cf 0.5 -57 / -27 / -21 %, cf 0 -59 / -32 / -27 % (ledger part C); the rest of the capture's cut is the gated bonus
    row (screen 1). The ledger's section 4 reconciles the two (scripts/ledger_lastrow.py).
  * Screen-1 note for Breakthrough / Fable C: (a) at cf 0 is unaffected; for (b) / (c) the proxy cf (the next segment's
    first draft, dd[L + 1]) is one draft later than the token the bonus row emits (dd[L]; the gate file's g[p]).
- Wed 03:5x: RESULT, QOM + 3-BIT SURE ROWS (pre-registered Tue 23:0x; Breakthrough's after_bigmoe.sh ran both arms
  03:39-03:54 on the masks READY at 03:31). Lines as registered: all-row flips <= Q1 at 0.99's level (1.34 / 5.42 /
  3.69 %) = PASS, <= 1.5x = BETWEEN.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0.5_live_t3 | 0.65 % (22) | 4.42 % (150) | 3.49 % (124) | PASS / PASS / PASS |
  | qom_0_live_t3 | 0.71 % (24) | 4.45 % (151) | 4.14 % (147) | PASS / PASS / BETWEEN |
  * The 3-bit cost is real (McNemar vs the 4-bit twin): cf 0.5 z +2.50 / +5.00 / +5.98 (+0.29 / +1.74 / +1.86 points);
    cf 0 z +2.50 / +5.25 / +7.05 (+0.29 / +1.92 / +2.37 points). vs gate_top2_cf099: cf 0.5 z -3.35 / -2.32 / -0.53;
    cf 0 z -3.00 / -2.20 / +1.17.
  * As registered, these bound QOM-full (3 bits on a sure row's ranks 0-1 only when its LSB part misses): QOM-full
    passes on the map at cf 0 on code and IFEval and at cf 0.5 on multi (cf 0 is BETWEEN there).
  * FAST candidate (the lowest passing theta per domain): 3-bit sure rows at cf 0 on code and IFEval, cf 0.5 on multi;
    4-bit QOM at cf 0 everywhere.
  * Price as tested (the map's convention, bonus row exact; ledger part C, plain LRU): 45 GB code cf 0 + 3b -78 %,
    IFEval cf 0 + 3b -49 %, multi cf 0.5 + 3b -31 %; 24 GB -74 / -58 / -43 %.
- Wed 03:58: RESULT, QOM AT C 160 POOLS (qom_0.5_live_c164; pre-registered Tue 23:2x; same run).
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0.5_live_c164 | 0.59 % (20) | 3.47 % (118) | 2.65 % (94) | PASS / PASS / PASS |
  * The C 160 cost (McNemar vs qom_0.5_live, C 256's pools): z +1.89 / +2.54 / +4.08 (+0.23 / +0.79 / +1.02 points);
    vs gate_top2_cf099 z -3.90 / -4.74 / -3.21.
  * My prediction: PASS (right); kept share -2-4 points (wrong: -13 / -8 / -7); flips +0.1-0.5 points (right on code,
    low on chat: +0.8-1.0).
  * Meaning: QOM holds on the map at the by-day C 160 line; the smaller pool keeps fewer resident rank 2-9 experts on
    sure rows and costs ~1 point of flips on chat, still inside Q1 at 0.99's level.
  * Harness check (03:59): Breakthrough installed the t3 arms as hyb_t3(<my mask dir>, 0.9); the 0.9 is unused there,
    and dm_common's ('t3mask',) gate takes the 3-bit rows from the masks (rank 0 off in every layer). Both t3 arms ran as
    registered.
- Wed 04:0x: PRE-REGISTRATION, BYTES10 SCREENS (FAST; before any run). The lines are Fable C's (~/FableGod/fablec/
  BYTES10.md section 5, written 00:35, before any run), recorded here with the ledger's corrections; NewIdeas orders the
  map slots. The ledger's prices were bugged in Fable C's file; these lines are about quality, not bytes.
  * Screen 1, THE GATED BONUS ROW. Arms on Fable C's masks (~/FableGod/fablec/masks/<arm>, 36 windows each; setup
    PERT.update(rmask=<abs dir>, renorm=False), i.e. q1_union on that dir):
    (a) qom_0_live_bonus: qom_0_live with every realized bonus row gated (ranks 0-1 + the block's union members at that
        layer; QOM's resident keep is not reproduced on that row, so a PASS is conservative);
    (b) q1u_0.9_live_bonus9: q1u_0.9_live, bonus row gated when the proxy cf >= 0.9;
    (c) qom_0.5_live_bonus5: qom_0.5_live, bonus row gated when the proxy cf >= 0.5.
    Lines, per domain: all-row flips <= the same arm without the gate + 0.3 points AND <= Q1 at 0.99's level (1.34 /
    5.42 / 3.69 %) = PASS; <= 1.5x that level = BETWEEN; above = KILL; paired McNemar vs the ungated arm. Fable C's
    prediction: (a) PASS on code (+0.1-0.3 points), BETWEEN on chat; (b) PASS where q1u_0.9_live passes.
    ORDER: (a) first. It decides the ledger's 24 GB FAST start (QOM cf 0 + 3b: 54.5 / 84.4 / 81.7 with the last row
    exact vs 35.7 / 19.4 / 13.9 gated) and whether decision 2's capture price stands. Then (c), then (b).
    NOTE (NewIdeas): in (b) / (c) the proxy cf is the next segment's first draft, dd[L + 1]: one draft later than the
    token the bonus row emits (dd[L], the cf an engine would gate it by). (a) gates every realized bonus row and is
    unaffected; read (b) / (c) as approximations.
  * Screen 2, TOP-1 ON SURE ROWS at live checks. hyb_0.9_k1_u10_live_t3 (NewIdeas' masks, 04:0x): sure rows (cf >= 0.9)
    keep rank 0 at 3 bits plus the block's union members at 4 bits; unsure rows and the bonus row exact. Setup:
    PERT.update(rmask=<abs>/hyb_0.9_k1_u10_live_t3, gate=("t3mask",), t3=3, t3k=1, renorm=False) (t3k 1: rank 0 only).
    Twin: q1u_0.9_live_t3 (top-2, the same gate; PERT.update(rmask=<abs>/q1u_0.9_live_t3, gate=("t3mask",), t3=3,
    t3k=2, renorm=False)), the live-check form of Fable C's q1u_cf09_t3 (16-row checks would mix check lengths).
    Lines: all-row flips <= Q1 at 0.99's level = PASS; <= 1.5x = BETWEEN; above = KILL; paired McNemar vs the twin.
    Fable C's prediction: BETWEEN on code (+0.3-0.6 points over top-2), KILL on chat.
  * Screen 2b (NewIdeas, conditional): qom_0_k1_live_t3 = QOM at cf 0 with sure rows at top-1 (rank 0 at 3 bits; union
    members and resident ranks 1-9 at 4 bits), t3k=1: the form the ledger's 24 GB path uses. Runs only on the domains
    where screen 2 is not KILL. Lines the same; paired McNemar vs qom_0_live_t3. My prediction [G]: BETWEEN on code,
    KILL on chat (cf 0 puts 2-3x more rows at top-1 than cf 0.9).
  * Screen 3, THE ENGINE VS THE REPLAY (CPU read-out of BigMoE's W1B 24 GB run at 13:45; no extra GPU). From the traced
    arm, split the process SSD bytes into prefill and decode, and compute decode MB per kept token on the six prompts.
    Reference (corrected): code EXACT at 24 GB, LRU, 212.8 (my miss count) / 245.2 (Fable C's fixed count), x 7.91 /
    kept_live. Lines: within 0.8-1.25x = the replay is calibrated at 24 GB and the ledger's prices stand; below 0.8x =
    the engine's LRU + popularity beats plain LRU, so the policy lever is partly cashed (re-price); above 1.25x = the
    engine reads more than its unions (prefetch waste, the K-73 phases): the first byte lever at 24 GB. Fable C's
    prediction: 0.9-1.1x.
- Wed 04:42: RESULT, BYTES10 SCREEN 1 (a): THE GATED BONUS ROW, qom_0_live_bonus (Fable C's masks; Breakthrough ran it
  in a FLOOR gap on the orchestrator's 04:0x go; landed 04:41). Line (pre-registered 04:0x, Fable C's): PASS = flips
  <= qom_0_live + 0.3 points AND <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %); BETWEEN <= 1.5x that level; KILL above.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0_live (ungated) | 0.42 % (14) | 2.53 % (86) | 1.77 % (63) | - |
  | qom_0_live_bonus | 0.65 % (22) | 6.12 % (208) | 6.50 % (231) | PASS / BETWEEN / KILL |
  * The gate's cost (McNemar vs qom_0_live): code 14 / 6 (z +1.79, +0.24 points: inside +0.3); IFEval 158 / 36
    (z +8.76, +3.59 points); multi 193 / 25 (z +11.38, +4.73 points). vs gate_top2_cf099: z -3.29 / +1.48 / +6.09.
  * Fable C's prediction: PASS on code (+0.1-0.3), BETWEEN on chat. Code and IFEval as predicted; multi worse (KILL).
  * WHAT IT TESTED: the bonus row kept ranks 0-1 + the block's union members only. QOM's resident keep (rank 2-9 experts
    already in the pool after the check's loads, 0 extra SSD reads) was not reproduced on that row, as Fable C flagged.
    So this KILLs the union-only form on chat, not the gated bonus row in QOM's own form.
  * Ledger consequence (TEN_X_LEDGER section 4): on code the engine convention's bytes stand (24 GB QOM cf 0 + 3b:
    35.7 instead of 54.5); on chat the map convention stands (84.4 / 81.7) until QOM's own form is tested (next entry).
- Wed 04:4x: PRE-REGISTRATION, SCREEN 1 IN QOM'S FORM (qom_0_live_bq; FAST; before any run). Screen 1 (a) gated the
  bonus row to ranks 0-1 + the block's union members. QOM's own sure rows also keep every rank 2-9 expert already resident
  after the check's loads. That keep costs 0 SSD reads and took Q1-UNION's flips down 62-65 % at cf 0.9 (q1u_0.9_live
  1.25 / 6.45 / 3.77 % -> qom_0.9_live 0.48 / 2.24 / 1.44 %). This arm puts it on the gated row.
  * Arm: qom_0_live with each live segment's last row gated by the draft it verifies (the walk's next draft after the
    cut, as an engine would: DFlash2 drafts the block in one pass, the live cut drops only inputs; exact when no draft is
    left). The row keeps ranks 0-1 + the block's union members + the resident ranks 2-9, like every other sure row.
    Discarded last rows (after a miss) are gated the same way; they change only the union. Masks:
    scripts/dm_oea_masks.py suffix _live_bq. Setup: PERT.update(rmask=<abs>/qom_0_live_bq, renorm=False) (q1_union).
  * Reads: the engine convention's (TEN_X_LEDGER section 4), the same as screen 1 (a)'s. 45 GB QOM cf 0: -64 / -58 /
    -61 % vs EXACT. 24 GB QOM cf 0 + 3b: 35.7 / 19.4 / 13.9 MB per kept token (54.5 / 84.4 / 81.7 with the row exact).
  * Lines (screen 1's): flips <= qom_0_live + 0.3 points AND <= 1.34 / 5.42 / 3.69 % = PASS; <= 1.5x the level =
    BETWEEN; above = KILL. Paired McNemar vs qom_0_live (the gate's cost) and vs qom_0_live_bonus (what the resident keep
    buys on that row).
  * My prediction [G]: the resident keep removes ~60 % of screen 1 (a)'s cost (+0.24 / +3.59 / +4.73 points), giving
    ~0.5 / 3.9 / 3.6 %: code PASS; chat BETWEEN (over the +0.3 condition, inside 1.5x the level).
  * Conditional arm: qom_0_live_bq_t3 = the same with every sure row's ranks 0-1 at 3 bits, the gated last row included
    (harness: gate=("t3mask",), t3=3, t3k=2). The FAST stack's full form. Runs on the domains where qom_0_live_bq is not
    KILL. Paired vs qom_0_live_t3.
  * Slot: asked of the orchestrator after screen 2 (a FLOOR gap before 13:00, else 13:45).
  * Masks built 04:46 (READY): builder self-check first: qom_0_live rebuilt with the edited builder is BIT-IDENTICAL to
    the map-tested masks (36 files). qom_0_live_bq: realized bonus rows gated 105 / 480 / 699 (code / IFEval / multi);
    union per check x0.274 / 0.225 / 0.219 of EXACT's (qom_0_live x0.363 / 0.502 / 0.572); kept share of sure-row rank
    2-9 pairs 92.9 / 90.4 / 89.1 %. qom_0_live_bq_t3 bit-checked against it: ALL OK.
- Wed 04:56: RESULT, BYTES10 SCREEN 2: TOP-1 ON SURE ROWS (hyb_0.9_k1_u10_live_t3 vs its twin q1u_0.9_live_t3;
  Breakthrough ran both after 1 (a) in the same process, 04:37-04:55; my read-out matches Breakthrough's). Line
  (pre-registered 04:0x): flips <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %) = PASS, <= 1.5x = BETWEEN, else KILL.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | q1u_0.9_live_t3 (top-2 twin) | 1.34 % (45) | 6.62 % (225) | 4.33 % (154) | PASS / BETWEEN / BETWEEN |
  | hyb_0.9_k1_u10_live_t3 (top-1) | 1.60 % (54) | 8.36 % (284) | 4.84 % (172) | BETWEEN / KILL / BETWEEN |
  * Top-1's cost over the top-2 twin (McNemar): code 21 / 12 (z +1.57, +0.27 points), IFEval 114 / 55 (z +4.54, +1.74),
    multi 72 / 54 (z +1.60, +0.51). vs gate_top2_cf099: z +1.37 / +6.43 / +3.23.
  * Fable C's prediction: BETWEEN on code (+0.3-0.6 points): hit (+0.27); KILL on chat: IFEval hit, multi BETWEEN.
  * Both arms are union-only on sure rows (no resident keep). QOM's form (resident ranks 1-9 kept, no extra reads) is
    screen 2b, qom_0_k1_live_t3 (built, READY): it qualifies on code and multi (screen 2 not KILL there) and goes to the
    13:45 slot (the orchestrator closed FLOOR night's insertions at 04:47).
  * Ledger: the 24 GB path's lever 2 (top-1 on sure rows, x0.55 / x0.59 / x0.56 of bytes) is BETWEEN on code and multi
    and KILL on IFEval in its union-only form; 2b decides its QOM form.
- Wed 05:06: RESULT, SCREEN 1 IN QOM'S FORM, qom_0_live_bq (pre-registered 04:4x; Breakthrough chained it after
  screen 2 on the orchestrator's 04:47 go). Line (screen 1's): flips <= qom_0_live + 0.3 points AND <= 1.34 / 5.42 /
  3.69 % = PASS; <= 1.5x the level = BETWEEN; above = KILL.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0_live (last row exact) | 0.42 % (14) | 2.53 % (86) | 1.77 % (63) | - |
  | qom_0_live_bonus (union-only gate) | 0.65 % (22) | 6.12 % (208) | 6.50 % (231) | PASS / BETWEEN / KILL |
  | qom_0_live_bq (QOM-form gate) | 0.39 % (13) | 4.09 % (139) | 3.26 % (116) | PASS / BETWEEN / BETWEEN |
  * The gate's cost vs qom_0_live: code 6 / 7 (z -0.28, none); IFEval 96 / 43 (z +4.50, +1.56 points); multi 77 / 24
    (z +5.27, +1.49 points). Chat is over the +0.3 condition but UNDER Q1 at 0.99's level on all three (the line every
    union arm is judged by): vs gate_top2_cf099 z -4.94 / -3.13 / -1.13.
  * What the resident keep buys on the gated row (McNemar vs qom_0_live_bonus): code 5 / 14 (z -2.06), IFEval 77 / 146
    (z -4.62), multi 55 / 170 (z -7.67): it removes 57 % / 68 % of the union-only gate's chat cost and all of code's.
  * My prediction (code PASS, chat BETWEEN ~3.9 / 3.6 %): hit (4.09 / 3.26).
  * Reads: the engine convention's, the same as screen 1 (a)'s. 45 GB QOM cf 0: -64 / -58 / -61 % vs EXACT (as tested
    with the last row exact: -59 / -32 / -27 %). This is the number Kamen saw for decision 2: it now has a map result
    (PASS on code; BETWEEN on chat by screen 1's stricter line, inside Q1's level).
  * qom_0_live_bq_t3 (pre-approved if not KILL): GO sent to Breakthrough 05:07 (already installed). Its reads: 24 GB
    35.7 / 19.4 / 13.9 MB per kept token (vs 54.5 / 84.4 / 81.7 with the last row exact).
  * qom_0_live_bq_t3's line, made explicit before its result (Breakthrough started it 05:06:54; it is the 04:4x
    registration's screen 1 line with the 3-bit twin): PASS = flips <= qom_0_live_t3 + 0.3 points AND <= the level, i.e.
    <= 1.01 / 4.75 / 3.69 %; BETWEEN <= 1.5x the level (2.01 / 8.13 / 5.54 %); KILL above. Paired McNemar vs
    qom_0_live_t3 (0.71 / 4.45 / 4.14 %) and vs qom_0_live_bq. My expectation [G]: code PASS; IFEval BETWEEN (bq + the
    3-bit cost of ~+1.9 points); multi BETWEEN (qom_0_live_t3 was already over 3.69 there).
- Wed 05:13: RESULT, qom_0_live_bq_t3 (the gated last row in QOM's form + 3-bit sure rows; line made explicit 05:07:
  PASS <= 1.01 / 4.75 / 3.69 %, BETWEEN <= 2.01 / 8.13 / 5.54 %). Breakthrough ran it 05:07-05:12 on the pre-approval.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0_live_bq_t3 | 0.77 % (26) | 6.09 % (207) | 5.29 % (188) | PASS / BETWEEN / BETWEEN |
  * vs qom_0_live_t3 (the gate's cost at 3 bits): code 9 / 7 (z +0.50), IFEval 102 / 46 (z +4.60), multi 92 / 51
    (z +3.43). vs qom_0_live_bq (the 3-bit cost): code 17 / 4 (z +2.84), IFEval 130 / 62 (z +4.91), multi 118 / 46
    (z +5.62).
  * My expectation (code PASS, chat BETWEEN): hit. On code the gated last row is free in both forms (z -0.28 at 4 bits,
    +0.50 at 3 bits).
  * FAST candidates per domain by the registered lines (24 GB MB per kept token, my replay, before-block miss count):
    - code: QOM cf 0 + the gated last row + 3-bit (PASS): 35.7 (EXACT 212.8).
    - IFEval: PASS options: QOM cf 0 + 3-bit 84.4. BETWEEN options for EvalSuite: + the gated row at 4 bits 47.3 (under
      Q1's level), + 3-bit 19.4 (over it).
    - multi: PASS options: QOM cf 0 93.7, QOM cf 0.5 + 3-bit 99.5. BETWEEN: + the gated row at 4 bits 34.4 (under Q1's
      level), + 3-bit 13.9 (over it, near the 1.5x edge).
  * FLOOR night 1 resumed (Breakthrough: the last insertion tonight). Screen 2b (QOM-form top-1) at 13:45.
- Wed 05:5x: MoonShots M-34 SURE2B (pre-registered by MoonShots 05:49; ~/MoonShots/data/sure2b.txt, 353c2c8): ledger
  lever 6 (2-bit sure-row records: [P2 codes | bf16 scales | B3 plane], a 2-bit read is a 1,540,096 B prefix of the
  3-bit part = x0.712). Alone at cf 0 (bonus row exact): x0.921 / x0.98 / x0.985 at 24 GB -> KILL by its line. On top
  of the gated last row (every row sure at cf 0): x0.712 fixed slots, x0.63-0.65 byte-budget slots.
  * Its lever-1 base is my 'all' convention (25.2 / 19.3 / 13.9 at 24 GB). The map tested the 'engine' convention
    (qom_0_live_bq: last rows gated only when a next draft exists): 35.7 / 19.4 / 13.9. Chat is the same; on code 136 of
    428 segments keep an exact last row, which the 2-bit factor does not touch, so code's factor sits between x0.712
    and x0.92. Asked MoonShots for the engine-convention code row (scripts/ledger_lastrow.py blocks_conv 'engine').
  * Chat with the gated last row + 3-bit + 2-bit sure rows: IFEval 19.4 -> ~13.8, multi 13.9 -> ~9.9 MB per kept token
    at 24 GB with fixed slots (~12.3 / 8.7 with byte-budget slots) [P]. Multi crosses 10.
- Wed 05:5x: PRE-REGISTRATION, 2-BIT SURE ROWS (qom_0_live_bq_t2; FAST; before any build or run; slot 13:45+ via the
  orchestrator: FLOOR night's insertions are closed). The quality bound for lever 6, as t3 bounded QOM-full.
  * Arm: qom_0_live_bq_t3's masks (ranks 0-1 off on every sure row, the gated last row included). The harness serves
    those ranks at 2 bits reconstructed from the 3-bit codes, w = s3 * (2 * (q3 >> 1) + 0.5) + b3 (MoonShots' formula;
    no new store). This needs a bits = 2 path in dm_t3.routed_topk (Breakthrough's harness; MoonShots has the formula).
    Setup as bq_t3 with t3=2: PERT.update(rmask=<abs>/qom_0_live_bq_t3, gate=("t3mask",), t3=2, t3k=2, renorm=False).
  * Lines (bq_t3's): PASS = flips <= qom_0_live_bq_t3 + 0.3 points AND <= the level (1.34 / 5.42 / 3.69 %); BETWEEN <=
    1.5x the level; KILL above. Paired McNemar vs qom_0_live_bq_t3 (the 2-bit cost) and vs qom_0_live_bq.
  * My prediction [G]: code BETWEEN (2 bits on the sure rows' top-2 experts costs ~+0.5-1.0 points over 3 bits); chat
    KILL (bq_t3 is already at 6.09 / 5.29 %).
  * 05:5x (MoonShots, c5af2d3): M-34 rerun on the engine convention (my blocks_conv 'engine', imported read-only; base
    35.7 / 19.4 / 13.9 reproduced). 24 GB: code 35.7 -> 29.7 (x0.833 fixed / x0.777 byte-budget), IFEval 19.4 -> 13.9
    (x0.715 / x0.639), multi 13.9 -> 9.9 (x0.712 / x0.627); 12 GB code x0.813 / x0.741, chat x0.71 / x0.54-0.55. The
    screen helper ~/MoonShots/scripts/to2bit.py is with Breakthrough (one-line install: dm_t3._proj bits == 2 ->
    to2bit.proj2; self-test: 3-bit unpack bit-exact to mx.dequantize, gather_qmm within 0.7 % of dense; 2-bit weights
    ~31 % relative RMS from 3-bit). qom_0_live_bq_t2 is one install away for the 13:45 list.
- Wed 07:27: RESULT, SCREEN 2b (qom_0_k1_live_t3 vs qom_0_live_t3; pre-registered 04:0x) AND 2-BIT SURE ROWS
  (qom_0_live_bq_t2 vs qom_0_live_bq_t3; pre-registered 05:5x). Breakthrough's morning batch 07:06-07:26, one process,
  36 windows each, 0 row-count warnings. My read-out (scripts/qom_read.py, 07:3x) matches Breakthrough's
  (~/FableGod/decision_map/out/SCREEN_read_*.txt) flip for flip.
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0_live_t3 (2b's twin, top-2 at 3 bits) | 0.71 % (24) | 4.45 % (151) | 4.14 % (147) | PASS / PASS / BETWEEN |
  | qom_0_k1_live_t3 (top-1, QOM form) | 0.59 % (20) | 4.65 % (158) | 3.24 % (115) | PASS / PASS (information) / PASS |
  | qom_0_live_bq_t3 (bq_t2's twin) | 0.77 % (26) | 6.09 % (207) | 5.29 % (188) | PASS / BETWEEN / BETWEEN |
  | qom_0_live_bq_t2 (2-bit sure rows) | 1.49 % (50) | 10.13 % (344) | 9.77 % (347) | BETWEEN / KILL / KILL |
  * 2b vs top-2 at 3 bits (McNemar): code 9 / 13 (z -0.85), IFEval 56 / 49 (z +0.68), multi 39 / 71 (z -3.05); vs
    gate_top2_cf099 z -3.81 / -1.83 / -1.26. IFEval was outside 2b's registered domains (screen 2 KILLed it), so its
    PASS is information only.
  * My prediction (BETWEEN code, KILL chat): wrong. The reading (Breakthrough's, and mine): in QOM's form a RESIDENT
    rank 1 stays at 4 bits, where top-2 at 3 bits serves it at 3. At C 256 pools most of them are resident (the masks
    keep ~93 % of sure-row rank 2-9 pairs as union members or residents), so top-1 drops little and gains precision on
    what it keeps. The same reason ties 2b to the pool: a smaller pool keeps fewer rank 1 experts (07:4x entry).
  * bq_t2 vs bq_t3: code 34 / 10 (z +3.62), IFEval 208 / 71 (z +8.20), multi 228 / 69 (z +9.23); vs bq z +5.40 /
    +11.23 / +12.77. My prediction (code BETWEEN, chat KILL): hit. Lever 6 (MoonShots M-34, 2-bit sure rows) is KILL on
    chat and BETWEEN on code (1.49 % > bq_t3 + 0.3 = 1.07 and > the level 1.34): it leaves the ranked path.
  * Harness (Breakthrough): MoonShots' to2bit.py c5af2d3 pinned in decision_map, dm_t3._proj bits == 2; self-test OK on
    L12 e7 and L40 e300.
- Wed 07:4x: CAVEAT ON THE 24 GB QUALITY CLAIMS, AND PRE-REGISTRATION, SCREEN 4: THE 24 GB POOL (FAST; before any
  build or run).
  * The caveat. Every QOM mask but qom_0.5_live_c164 kept the residents of laysz45_256's base pools (C_m mean 295 of
    512 per layer: the 45 GB default; the build logs say "pool m"). The ledger's and BYTES10's 24 GB MB per kept token
    come from a replay at 147 base slots per layer. So "code FAST stack map-PASS at 35.7 MB per kept token at 24 GB"
    pairs a 45 GB-pool quality with a 24 GB-pool price. At C 164 (cf 0.5, 4 bits) the smaller pool cost +0.23 / +0.79 /
    +1.02 points (03:58); at 147, at cf 0 (every drafted row sure) and with top-1 (rank 1 kept only when resident or a
    union member) it will cost more. Until screen 4 runs, the QOM stacks' 24 GB status is "map-tested at the 45 GB
    pool"; the 45 GB statements stand.
  * Arms (scripts/dm_oea_masks.py, suffix _c147 = a 147-slot base pool per layer, the replay's 24 GB line; built CPU
    only at the next gap between quiet windows; bit-checked against their 4-bit twins with qom_t3_check.py; READY
    marker as before):
    (B) qom_0_k1_live_bq_c147_t3: the <= 10 candidate. QOM cf 0, sure rows at top-1 (rank 0 at 3 bits; union members
        and resident ranks 1-9 at 4 bits), the last row gated in QOM's form, the 24 GB pool. Setup
        PERT.update(rmask=<abs>/qom_0_k1_live_bq_c147_t3, gate=("t3mask",), t3=3, t3k=1, renorm=False).
    (C) qom_0_live_bq_c147_t3: today's code PASS stack (top-2 at 3 bits + the gated last row) at the 24 GB pool. Setup
        as qom_0_live_bq_t3 (t3=3, t3k=2) on its own dir.
    (A, optional, after B and C) qom_0_k1_live_bq_t3: (B) at the 45 GB pool, to separate the pool's cost from the
        gate's.
  * Lines, per domain, all-row flips: PASS <= Q1 at 0.99's level (1.34 / 5.42 / 3.69 %), BETWEEN <= 1.5x (2.01 / 8.13
    / 5.54 %), KILL above. These are the 24 GB verdicts. Paired McNemar: C vs qom_0_live_bq_t3 (the pool's cost at
    top-2), B vs C (top-1's cost at the 24 GB pool), B vs qom_0_k1_live_t3.
  * Predictions [G]: C code PASS (~1.0 %), IFEval BETWEEN (~7 %), multi BETWEEN or KILL (~6 %, at the 5.54 edge).
    B code PASS or BETWEEN (~1.2 %), IFEval BETWEEN (~7.5 %), multi KILL (~6.5 %): top-1 leans on residency more.
  * Bytes at 24 GB [P], my replay (before-block misses; the last row on the engine convention): C 35.7 / 19.4 / 13.9;
    B from scripts/ledger_top1.py (filled in before any GPU run).
  * Slot: the orchestrator's call (two arms, ~20 min of the map on a shared GPU, MEM ~17 GB).
- Wed 07:5x: TOP-1 PRICES AND AN AMENDMENT TO SCREEN 4 (before any GPU run). Masks built 07:50-07:52 under a share lock
  (CPU only), bit-checked ALL OK (data/dm_masks/check_c147.log), READY. Prices: scripts/ledger_top1.py
  (data/ledger/ledger_top1.txt); it reproduces the top-2 rows exactly (24 GB: 54.5 / 84.4 / 81.7 map, 35.7 / 19.4 / 13.9
  engine convention).
  * 2b served a kept rank 1 at 4 bits (t3k=1: ranks >= 1 come from the stock path at the model's precision). An engine
    matching that needs rank 1's LSB part, which misses often in the 46-slot LSB pool at 24 GB. The cheap form serves a
    kept rank 1 from its 3-bit base record, with no LSB read on sure rows, which is what the ledger's top-1 lever
    (x0.55-0.59) assumed. So 2b's PASS does not validate that lever: the form it tested is the dear one.
  | rule (last row gated: engine convention) | 45 GB | 24 GB | 12 GB |
  |---|---|---|---|
  | QOM cf 0 + 3-bit, top-2 (qom_0_live_bq_t3's form) | 7.7 / 4.5 / 4.4 | 35.7 / 19.4 / 13.9 | 83.1 / 72.6 / 52.9 |
  | top-1, kept rank 1 at 4 bits (2b's form: LSB read on a miss) | 12.9 / 9.3 / 7.4 | 32.4 / 21.9 / 15.3 | 58.0 / 39.5 / 28.1 |
  | top-1, kept rank 1 at 3 bits (no LSB on sure rows) | 5.7 / 2.9 / 2.9 | 24.0 / 11.5 / 7.7 | 52.1 / 32.3 / 22.9 |
  * Amendment: (B) is replaced by (B') qom_0_k1_live_bq_c147_t3keep, the cheap form. The harness's existing t3keep gate
    (P1-K1 at 3 bits) does it: rows the mask cuts are served wholly from records3 at 3 bits with their own rank mask.
    Mask: qom_0_k1_live_bq_c147 (the 4-bit twin, rank 0 kept; READY). Setup PERT.update(rmask=<abs>
    /Users/kamen/NewIdeas/data/dm_masks/qom_0_k1_live_bq_c147, gate=("t3keep",), t3=3, t3k=10, t3keep=True,
    renorm=False). On sure rows every kept expert runs at 3 bits, ranks 2-9 too. That is the engine's own precision for
    ranks 2-9, but stricter than the map's other arms, which serve them at 4 bits: a conservative test. Sure rows with
    nothing dropped (20 / 12 / 12 of 3295 / 3408 / 3565) stay at 4 bits, because the gate keys on a drop.
  * B' lines: the level lines above (PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x). Paired: B' vs C and B' vs
    qom_0_k1_live_t3 (both mix top-1's cost with the 3-bit ranks 2-9; the verdict is the line).
  * Prediction [G] B': code BETWEEN (~1.5 %), IFEval KILL (~9 %), multi BETWEEN or KILL (~5.5-6.5 %).
  * If B' passes on a domain, that domain has a 24 GB FAST stack, map-tested at the 24 GB pool, at 24.0 / 11.5 / 7.7 MB
    per kept token (multi under 10) before the pool policy and 11-bit.
  * Order: C first (it checks a claim already reported), then B'. B (2b's form at the 24 GB pool) is dropped: on chat it
    reads more than C.
- Wed 08:3x: RESULT, SCREEN 3: THE ENGINE VS THE REPLAY AT 24 GB (pre-registered 04:0x; scripts/screen3_read.py,
  data/ledger/screen3_read.txt). BigMoE's W1B 24 GB traced arm (data/w1b24/w24t, 07:54-07:56): tier 4433333333 with
  C_m 147 / C_l 46 per layer, the replay's 24 GB line exactly; 6 prompts x 256 tokens (HumanEval/68, /91, 4 math);
  33.2 tok/s untraced, 30.1 traced; lifetime footprint 23.87 GB. Decode = the misses (miss.bin) in each prompt's own
  verify / MTP passes, in the ledger's record sizes.
  | prompt | kept per pass | decode misses, MB per kept (trunk + MTP head + LSB) | engine reads (trace) | process disk_read |
  |---|---|---|---|---|
  | HumanEval/68 | 7.76 | 200.8 + 1.4 + 23.7 = 226.0 | 266.3 | 427.1 |
  | HumanEval/91 | 3.88 | 217.4 + 6.8 + 23.1 = 247.4 | 291.4 | 409.2 |
  | math x 4 | 1.30-3.32 | 195.4-228.4 | 263.5-342.9 | 333.6-438.4 |
  * Against the replay (code, EXACT, 24 GB, LRU): 212.8 (my count) / 245.2 (Fable C's). Unscaled: 1.06 / 1.16 on the
    two code prompts (0.92-1.16 on all six). The registered line scaled the reference by 7.91 / kept per pass; the data
    refutes that scaling: per-kept bytes stay flat from 1.30 to 7.76 kept per pass (195-247 MB), as ledger part A said
    (x0.91 from k 1 to 16). On the one prompt where the scaling is neutral (HumanEval/68, 7.76 kept): 1.04.
  * VERDICT: CALIBRATED (0.8-1.25x). The replay's 24 GB prices stand; plain LRU and the engine's LFU 0.99 + prefetch
    miss about the same (the policy lever is not cashed yet). Fable C's prediction (0.9-1.1x): hit (1.11 on code by my
    count, 0.97 by its own). The scaled "below 0.8" cells (0.17-0.57) are the line's scaling artifact, not a finding.
  * Beside the line: the engine READS 18-26 % more than it misses (55 % on the 5 tok/s math prompt). On HumanEval/68 49 %
    of issued prefetches were used (pf_issued 10,693, used 5,196). That is ~40 MB per kept token on code, a byte lever
    of ~x0.85 at 24 GB, traded against latency hiding (BigMoE's call). The process bytes add ~130-160 MB per token on
    top (prefill and the untraced rest: 35-40 % of the process bytes on 256-token answers).
- Wed 08:48: RESULT, SCREEN 4: THE 24 GB POOL (pre-registered 07:4x, amended 07:5x; Breakthrough ran C then B'
  08:33-08:47 in one process, 36 windows each, 0 row-count warnings; my read-out 08:57 matches flip for flip).
  | arm | code | IFEval | multi | verdict |
  |---|---|---|---|---|
  | qom_0_live_bq_t3 (45 GB pool, 05:13) | 0.77 % (26) | 6.09 % (207) | 5.29 % (188) | PASS / BETWEEN / BETWEEN |
  | C qom_0_live_bq_c147_t3 (24 GB pool) | 1.07 % (36) | 8.07 % (274) | 5.99 % (213) | PASS / BETWEEN / KILL |
  | B' qom_0_k1_live_bq_c147_t3keep | 1.69 % (57) | 11.25 % (382) | 8.61 % (306) | BETWEEN / KILL / KILL |
  * The pool's cost at top-2 (C vs bq_t3): code 20 / 10 (z +1.83, +0.30 points), IFEval 126 / 59 (z +4.93, +1.98),
    multi 75 / 50 (z +2.24, +0.70). B' vs C: z +3.55 / +6.83 / +6.31; B' vs 2b: z +5.08 / +12.48 / +11.39.
  * Predictions: C code PASS / IFEval BETWEEN / multi BETWEEN-or-KILL: hit. B' code BETWEEN / IFEval KILL / multi
    BETWEEN-or-KILL: hit (multi 8.61 % was worse than my ~6.5).
  * Meaning:
    - Code: the FAST stack (QOM cf 0 + 3-bit sure rows + the gated last row) is map-PASS at the 24 GB pool, at 35.7 MB per
      kept token [P, replay calibrated by screen 3]. The 07:4x caveat is answered for code.
    - Chat at 24 GB: that stack is BETWEEN on IFEval (8.07 %, at the 8.13 edge) and KILL on multi. The chat stacks
      tested only at the 45 GB pool stay unverified at 24 GB, and they will lose ~1-2 points the same way.
    - Top-1 in its cheap form is off the 24 GB path (KILL on chat; BETWEEN on code at 24.0 vs C's PASS at 35.7).
    - Under QOM the pool decides quality, not only bytes: that is the next idea (08:5x).
- Wed 08:5x: NEW IDEA, PRE-REGISTRATION: QUALITY POOL (tag: new as far as I know; skip-on-miss with a cache exists in
  HOBBIT / AdapMoE-style offloading, but not a pool chosen for the router mass that skipping rows keep). FAST.
  * Observation (screen 4 + the mask logs): under QOM a sure row keeps a rank 2-9 expert only if it is a union member
    or resident. From the 45 GB pool to 147 slots the kept share of sure-row rank 2-9 pairs falls 92.9 / 90.4 / 89.1 % ->
    75.8 / 79.0 / 80.4 %, and flips rise +0.30 / +1.98 / +0.70 points (C vs qom_0_live_bq_t3). The pool's policy (plain
    LRU in the masks; LFU 0.99 in the engine, which misses about as LRU does: screen 3) serves the loads' hit rate, not
    what sure rows keep. Ranks 2-9 carry ~68 % of the mean top-10 router mass (ten_x_ledger MASS).
  * Idea: at the same 147 slots, pick the eviction / admission policy that maximizes the router mass sure rows keep.
    Candidates: LFU with decay 0.99 / 0.999 (count of loads + resident uses); mass-LFU (score += MASS[rank] for every
    routed (row, rank) pair, loaded or not: a ghost history; admission by score) at decay 0.99 / 0.999; SLRU (80 %
    protected).
  * Cheapest kill test (CPU, ~5-10 min, no GPU): scripts/pool_quality.py, a stats-only twin of dm_oea_masks' QOM pool
    under qom_0_live_bq's rule (cf 0, top-2, the gated last row), per domain: the kept share of sure-row rank 2-9 pairs
    (count and MASS-weighted) and the load misses. Self-check: LRU must reproduce 75.8 / 79.0 / 80.4 % at 147 slots and
    92.9 / 90.4 / 89.1 % at the 45 GB pool.
  * Line: a policy is ALIVE if on IFEval AND multi it closes >= 1/3 of the MASS-weighted gap between LRU at 147 slots
    and LRU at the 45 GB pool, with load misses <= 1.10x LRU's at 147; otherwise KILL. ALIVE -> C's masks with that
    policy (qom_0_live_bq_c147<policy>_t3) and a map slot, paired vs C.
  * Prediction [G]: decayed LFU and mass-LFU +2-4 points on chat (short of 1/3 on IFEval: KILL there), SLRU +1-2.
- Wed 08:58: RESULT, QUALITY POOL (pre-registered 08:5x): KILL. scripts/pool_quality.py ran 57 s under a share lock
  (data/ledger/pool_quality.txt). Self-check exact: LRU's kept shares equal the builder's logs (75.8 / 79.0 / 80.4 % at
  147 slots, 92.9 / 90.4 / 89.1 % at the 45 GB pool).
  | policy at 147 slots | gap closed, MASS-weighted (code / IFEval / multi) | load misses x LRU 147 |
  |---|---|---|
  | slru (80 % protected) | +3 / +8 / +3 % | x0.94 / x0.91 / x0.96 |
  | mlfu_d99 (mass-weighted ghost LFU) | -1 / +9 / -7 % | x0.92 / x1.06 / x1.13 |
  | lfu_d99 / lfu_d999 | -21..-116 % | x1.15-2.93 |
  * No policy closes 1/3 of the gap on IFEval and multi: the lost keep is capacity, not policy. Frequency LFU is worse
    than LRU here: the per-request working set moves, and recency follows it.
  * My prediction (LFU +2-4 points; SLRU +1-2; KILL on IFEval): the verdict hit, the direction for LFU missed.
  * By-product: SLRU reads x0.91-0.96 of LRU's load misses with no keep cost: a small exact lever, priced on QOM's
    gated stack only.
  * What does move the keep is capacity (the 45 GB pool doubles the slots). So the capacity levers (11-bit: ~178 slots in
    the same GB; LOSSLESS45; the volatile tier) count twice under QOM: fewer misses AND more kept experts. Next: price
    11-bit's capacity effect on both (CPU, pre-registered below).
- Wed 08:59: PRE-REGISTRATION, 11-BIT'S CAPACITY UNDER QOM (CPU pricing; no quality claim). 11-bit scales shrink every
  record to x0.824 [K MoonShots]; the ledger prices only that (x0.824 per byte read). In the same 24 GB the pools also
  hold 1 / 0.824 more records: 147 -> 178 base slots, 46 -> 56 LSB slots (12 GB: 51 -> 62, 32 -> 39). Under QOM that
  counts twice: fewer misses, and more rank 2-9 experts kept on sure rows (screen 4: the pool costs +0.30 / +1.98 / +0.70
  points between 147 slots and the 45 GB pool).
  * Run (scripts/capacity_11bit.py): bytes = ledger_top1's replay of C's rule (QOM cf 0 + 3-bit + the gated last row,
    engine convention) at 147 / 46 and 178 / 56, the latter x0.824; keep = pool_quality's LRU kept share at 178 slots.
  * Prediction [G]: bytes x0.824 x ~0.85 = ~x0.70 on code (35.7 -> ~25); kept share +3-4 points on chat (~1/4 of the gap
    to the 45 GB pool).
  * Use: C's price with 11-bit at 24 GB, and whether a map arm at 178 slots (qom_0_live_bq_c178_t3, quality of C after
    11-bit) is worth a slot: ask for it if the kept share closes >= 1/3 of the MASS gap on IFEval.
- Wed 08:59: RESULT, 11-BIT'S CAPACITY UNDER QOM [WRONG, CORRECTED 13:4x: see ERRATUM; 11-bit is x0.89-0.90 at 24 GB] (pre-registered 08:59, before the run; scripts/capacity_11bit.py, data/ledger/capacity_11bit.txt; 15 s; self-check: 24 GB without 11-bit = 35.7 /
  19.4 / 13.9, ledger_top1's engine row).
  | domain | 24 GB C stack | + 11-bit with its capacity (x) | 12 GB | + 11-bit (x) | kept 147 -> 178 slots (gap closed) |
  |---|---|---|---|---|---|
  | code | 35.7 | 22.6 (x0.633) | 83.1 | 61.1 (x0.735) | 76.7 -> 81.2 % (+27 %) |
  | IFEval | 19.4 | 11.6 (x0.595) | 72.6 | 49.6 (x0.683) | 80.0 -> 83.2 % (+29 %) |
  | multi | 13.9 | 8.5 (x0.615) | 52.9 | 35.9 (x0.679) | 81.5 -> 84.1 % (+32 %) |
  * 11-bit is worth x0.60-0.63 at 24 GB under QOM (x0.68-0.74 at 12 GB), not the ledger's x0.824: the same GB holds 21 %
    more records, and at 24 GB the misses are capacity misses. My prediction (~x0.70 on code; +3-4 keep points on chat):
    bytes better than predicted, keep hit.
  * Keep: +27-32 % of the gap to the 45 GB pool, under my 1/3 line on IFEval (29 %), so no map arm asked for. By the
    screen 4 pool cost, linear: code ~0.98 % (PASS), IFEval ~7.5 % (BETWEEN), multi ~5.8 % (KILL edge) [G].
  * Needs: BigMoE sizing the pools by bytes after the 11-bit conversion (the 24 GB tier already sizes by GB: 17.07 GB
    pools). EXACT gets the capacity effect too (not priced here).
- Wed 13:2x: RESULT, T3R2 (pre-registered Tue 19:15, 4dbe21a; amendment 19:27, 1ba42c3; Fit12 13:05-13:12, quiet, 12 GB).
  Re-computed by NewIdeas with Fit12's t3r2_verdict.py on ~/Fit12/data/scr/t3r2/r47.txt: PASS.
  * t80ck1u / t80ck2 x1.148 pooled on the shared text over 16 of 16 prompts (line >= 1.10), coverage 80 / 77 %:
    57.3 vs 49.9 tok/s (2638 tok / 46.01 s vs 2639 / 52.86 s). T3R's original 8 beside it: x1.138 (coverage 88 / 84 %).
  * Full outputs (texts differ after the cut, not the metric): 44.17 vs 40.77 tok/s (x1.083).
  * Per prompt x0.78-1.46; only /91 below 1 (x0.78). The pooled ratio cleared the line on both halves of the prompts.
  * Clean: 16 PAIR lines ok, no PAUSE, no re-runs, SSD 25 -> 62 C; t80ck2 = T3t (4 / 4); K1U engaged on all 16; peak
    lifetime 11.354 GB (limit 11.95).
  * INFO: 3-bit t80ck1um3 x1.268 (x1.254 on the original 8; coverage 76 / 74 %).
  * Next (orchestrator 13:1x): EvalSuite same-scores at 12 GB (FAST adopts on same-scores, the 11-02 two-track rule),
    behind RELAX + T0.
- Wed 13:2x: RESULT, T3R-PF (P2-PREFETCH, exact; my line Tue 14:56, fd081ae; Fit12 p47 after r47). KILL:
  t80ck1upf / t80ck1u x0.901 (line <= 1.01 KILL); second verdict t80ck2pf / t80ck2 x0.927. Gate PF2 8 / 8 PASS (tokens +
  logits equal), forced waits 0, no PAUSE. Re-computed with the same script.
  * My CPU model (-12.6 ms per check) failed in both halves:
    - The late reads were not hidden. Pass 1 +26.2 ms per check at K1U (108.2 -> 134.4; misses 283 -> 409, about the
      late reads) and +22.9 ms at K2 (687 -> 839): each read cost about its full read time.
    - Pass 2 hardly used them: misses per check 338 -> 333 (K1U), 138 -> 134 (K2); ms 87.5 -> 85.3, 41.2 -> 37.3. The
      replay had 156 -> 81.
    - Most prefetches went to checks that never ran pass 2. HumanEval/68 had no pass 2 at all (pass2 0, exact 0) and
      still lost 17 % (t80ck1u 81.50 -> 67.72 tok/s) and 9 % (t80ck2 61.54 -> 55.81). Over the 4 prompts, K1U had pass 2
      on 19 of 89 checks.
  * Lesson: price a gap-read lever by the checks that use it, and measure exposure first. At 12 GB a read in pass 1's
    gap delayed pass 1 by its full read time.
- Wed 13:2x: Fit12's K12 (Leap's stale-slot probe on T3, a sync at every MoE layer) PASS 12 / 12 (prompt, arm) pairs
  bitwise equal (tokens + logits + per-check log), 15,964 synced calls (k12_verdict.py re-run): no stale reads in T3.
- Wed 13:25: PRE-REGISTRATION, POOLSPLIT (BYTES10, FAST stack only; CPU replay; untested re-tune, not new). Under C's
  stack (QOM cf 0 + 3-bit sure rows + the gated last row), only unsure rows read LSB parts. Those are last rows with no
  next draft (DFlash2 drafts the needed token on 68 / 99 / 99 % of segments). Yet the 24 GB split keeps 46 LSB slots
  per layer (1.70 GB). Moving LSB RAM to base slots at equal GB per layer (24 GB: Cl 0 -> Cm 163, +11 %; 12 GB: Cl 0 ->
  Cm 62, +22 %) counts twice under QOM: fewer base misses and more resident rank 2-9 experts kept on sure rows.
  * Run (scripts/pool_split.py): C's replay (ledger_top1's loop, engine convention) at equal bytes per layer for Cl in
    {0, 4, 8, 12, 16, 24, 32, default}, at 24 GB and 12 GB, with and without 11-bit (records x0.824). Keep =
    pool_quality's LRU kept share at each split's Cm (24 GB). Self-check: the default splits must give 35.7 / 19.4 /
    13.9 and 83.1 / 72.6 / 52.9.
  * Prediction [G]: chat's best split is Cl 0-8, code's Cl 8-24 (code has ~32 % unsure last rows). Best-split bytes at
    24 GB x0.90-0.95 (chat) and x0.93-0.97 (code); keep +1 point (~10 % of the gap to the 45 GB pool). 12 GB gains more:
    x0.85-0.90.
  * Line: ALIVE (a FAST-mode pool-split flag for BigMoE; EXACT keeps its split) if the best split cuts bytes >= 5 % on
    all three domains at 24 GB, or holds bytes <= x1.00 while closing >= 10 % of the keep gap. KILL if < 3 % on all three
    and < 5 % of the gap. BETWEEN otherwise.
- Wed 13:28: RESULT, POOLSPLIT [11-bit rows WRONG, CORRECTED 13:4x: see ERRATUM; the non-11-bit rows stand] (pre-registered 13:25; scripts/pool_split.py, data/ledger/pool_split.txt; 106 s, 3
  procs; self-checks: all 12 default cells equal the ledger's, keep at 147 / 178 = capacity_11bit's). ALIVE: the best
  split cuts bytes >= 5 % on all three domains at 24 GB.
  | line | code | IFEval | multi | keep gap closed (24 GB) |
  |---|---|---|---|---|
  | 24 GB default 147 / 46 | 35.7 | 19.4 | 13.9 | 0 % |
  | 24 GB 163 / 0 | 32.8 (x0.917) | 16.5 (x0.849) | 12.0 (x0.861) | +15 / +16 / +17 % |
  | 24 GB 11-bit default 178 / 56 | 22.6 | 11.6 | 8.5 | +27 / +29 / +32 % |
  | 24 GB 11-bit 198 / 0 | 20.5 (x0.904) | 9.4 (x0.808) | 7.1 (x0.833) | +41 / +46 / +50 % |
  | 12 GB 62 / 0 (default 51 / 32) | 75.6 (x0.909) | 60.2 (x0.829) | 43.6 (x0.825) | - |
  | 12 GB 11-bit 75 / 0 (default 62 / 39) | 55.3 (x0.906) | 39.8 (x0.803) | 28.7 (x0.799) | - |
  * Why: chat has no unsure rows left (LSB reads 0.0 MB per kept even with no LSB pool); code reads 2.3-2.7 MB of LSB
    parts at Cl 0 vs 0.8-1.2 at the default, far less than the base slots save. Practical setting: Cl 4-8 (a staging
    margin), within 1 % of Cl 0 (24 GB 11-bit 195 / 8: 20.5 / 9.7 / 7.3).
  * Prediction check: chat's best Cl 0-8 hit; code's Cl 8-24 missed (code's best is also Cl 0). Bytes better than
    predicted (chat x0.85-0.86 vs x0.90-0.95; code x0.917 vs x0.93-0.97); keep +1.4-2.4 points (15-17 % of the gap) vs
    +1 point.
  * 24 GB chat bytes now under 10 MB per kept token [P]: IFEval 9.4, multi 7.1 (11-bit + POOLSPLIT; SLRU would add
    x0.91-0.96). Quality is the open part: the 24 GB pool put C at 8.07 / 5.99 % on chat (IFEval BETWEEN, multi KILL).
    Linear in the keep gap, 198 slots gives ~7.2 % / ~5.6 % [G]: IFEval BETWEEN, multi at the BETWEEN / KILL edge
    (5.54). Code 20.5 (PASS stack at 147 slots; more slots only help).
  * FAST mode only: EXACT serves ranks 0-1 at 4 bits on every row and keeps its own split. Needs BigMoE's FAST-mode
    pool sizing (C_l 4-8, the rest to C_m) with the 11-bit conversion.
  * Next: map screen 5 = C at 198 slots (qom_0_live_bq_c198_t3; masks are CPU, mine), the quality of the 24 GB chat
    stack after 11-bit + POOLSPLIT. Pre-registered below; GPU slot is the orchestrator's call (not by day).
- Wed 13:3x: [WITHDRAWN 13:4x before any run: its 198-slot premise was the 11-bit error; see ERRATUM] PRE-REGISTRATION, MAP SCREEN 5 = C at the 198-slot pool (before any run; masks are CPU, built now).
  * Arm: qom_0_live_bq_c198_t3. C's stack (QOM cf 0 + 3-bit sure rows, t3mask t3k=2, + the gated last row) at a flat
    198-slot LRU base pool per layer. That is the 24 GB FAST pool after the 11-bit conversion and POOLSPLIT (Cl 0); 11-bit
    is lossless, so the pool is the only change vs screen 4's C (147 slots: 1.07 / 8.07 / 5.99 %). Twin for the +0.3
    reading: qom_0_live_bq_t3 (the 45 GB pool, 0.77 / 6.09 / 5.29 %).
  * Lines (screen 4's): PASS <= level 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x (2.01 / 8.13 / 5.54), KILL above; per
    domain.
  * Prediction [G], linear in the keep gap closed (+41 / +46 / +50 %): code ~0.9 % PASS, IFEval ~7.2 % BETWEEN, multi
    ~5.6 % at the BETWEEN / KILL edge. Chat PASS on both: <= 15 %.
  * Use: chat PASS on both -> the 24 GB chat stack is <= 10 MB per kept token in bytes (9.4 / 7.1 [P]) AND map-PASS
    -> EvalSuite candidate. BETWEEN -> chat at 24 GB needs one more quality lever. KILL on multi -> more capacity first.
  * Cost: one map arm (~7 min GPU, shared lock), any slot the orchestrator picks (not by day).
- Wed 13:4x: ERRATUM (my error; found while pricing a smaller FAST store): 11-BIT IS NOT x0.824 PER BYTE.
  * What was wrong. The ledger (TEN_X_LEDGER 03:4x) priced 11-bit as "x0.824 on every byte [K MoonShots]". MoonShots'
    -17.6 % is a NET replay result at the 45 GB C 256 pools that already includes ~4.6 % more parts in the same RAM
    (data/lossless45.txt, EXPLORATORY: joint fit with part sizes 2,064,384 / 720,896 B). The packed store's parts are
    2,064,384 / 720,896 B vs records3's 2,162,688 / 770,048: x0.955 / x0.936 per part. At 08:59 I then added 1 / 0.824 more
    slots on top (147 -> 178): double counting. At 13:28 POOLSPLIT reused x0.824.
  * Corrected (scripts/capacity_11bit.py re-run; 24 GB 147 / 46 -> 154 / 49 and 12 GB 51 / 32 -> 53 / 34 at equal RAM,
    each line's own part sizes; old output kept as data/ledger/capacity_11bit_0859_wrong.txt):
    | domain | 24 GB C stack | + 11-bit (x) | was (08:59) | 12 GB | + 11-bit (x) | keep gap closed at 154 |
    |---|---|---|---|---|---|---|
    | code | 35.7 | 32.1 (x0.898) | 22.6 (x0.633) | 83.1 | 77.7 (x0.934) | +6 % |
    | IFEval | 19.4 | 17.3 (x0.888) | 11.6 (x0.595) | 72.6 | 67.0 (x0.923) | +7 % |
    | multi | 13.9 | 12.4 (x0.894) | 8.5 (x0.615) | 52.9 | 48.7 (x0.922) | +8 % |
  * POOLSPLIT corrected (scripts/pool_split.py re-run; 13:28 output kept as pool_split_1328_wrong11bit.txt):
    | 24 GB line | code | IFEval | multi | keep gap closed |
    |---|---|---|---|---|
    | 11-bit 154 / 49 (default at equal RAM) | 32.1 | 17.3 | 12.4 | +6 / +7 / +8 % |
    | 11-bit 171 / 0 (POOLSPLIT) | 29.3 (x0.914) | 14.5 (x0.839) | 10.6 (x0.852) | +21 / +23 / +25 % |
    | 11-bit 169 / 4 (practical) | 29.5 | 14.8 | 10.8 | +20 / +21 / +23 % |
    | was (13:28): 11-bit 198 / 0 | 20.5 | 9.4 | 7.1 | +41 / +46 / +50 % |
    12 GB 11-bit 65 / 0: 70.1 / 54.6 / 39.4 (default 53 / 34: 77.7 / 67.0 / 48.7). Without 11-bit nothing changes (24 GB 163 / 0
    x0.917 / x0.849 / x0.861; 12 GB 62 / 0 x0.909 / x0.829 / x0.825). Self-checks: the 11-bit defaults equal the corrected
    capacity_11bit columns.
  * What changes:
    - WITHDRAWN: "11-bit x0.60-0.63 at 24 GB" (08:59, sent to the orchestrator and MoonShots 09:01); "code 22.6 / ~21 at
      24 GB"; "chat under 10 MB per kept at 24 GB" (13:28 / 13:32); BUILD_DAY_1104.md build #6's "~22 after the 11-bit
      store".
    - Still true: POOLSPLIT without 11-bit (x0.917 / x0.849 / x0.861 at 24 GB, 163 / 0); 11-bit is still worth more under
      QOM than per byte (x0.89-0.90 at 24 GB vs x0.955), because the extra parts are kept too, just ~5 % of them, not 21 %.
    - The 45 GB EXACT 11-bit cell (60.4 -> 49.8) is MoonShots' own net figure at C 256 (code -17.6 %, IFEval -16.1 %,
      multi -15.8 %): right as a net, wrong as "per byte"; relabelled.
    - Map screen 5 WITHDRAWN before any run: its premise (a 198-slot pool = 24 GB after 11-bit + POOLSPLIT) was the
      error. The real pool is 171 slots, and chat stays above 10 MB there anyway, so the arm cannot change a
      decision. The c198 masks stay on disk as an upper bracket; no GPU asked.
  * Where 24 GB stands now [P]: code 29.3 (map-PASS stack at 147 slots; more slots only help), ~27.5 with SLRU; chat 14.5 /
    10.6 (quality BETWEEN / KILL at 147 slots; ~7.6 / ~5.8 % at 171 by the linear keep model, the same classes). Under 10
    at 24 GB needs ~x0.34 more on code and ~x0.7 / ~x0.95 on chat, plus chat's quality.
  * Lesson: take a lever's factor from its parts (record sizes), never from another line's net replay; a net figure
    already holds that line's capacity effect.
- Wed 13:4x: PRE-REGISTRATION, TRELLIS FAST STORE (BYTES10, code at 24 GB; CPU byte replay only). A new angle on B10
  (trellis-coded residency, killed Sun on an EXACT-quality line: TRL-2 out of sample 1.85x MSB-3's output error vs 1.5x).
  Two things changed since: (1) under QOM capacity counts twice (bytes and keep), and code's C-stack bytes fall steeply
  between ~200 and ~300 slots (24.9 at 198 -> 7.7 at the 45 GB pool's ~295, records3 units); (2) the FAST bar is the
  map's flips, not 1.5x MSB-3. MSB-2 sure rows (lever 6) gave code 1.49 % (BETWEEN, line 1.34). TRL-2 has ~0.4x MSB-2's
  error [G: one bit ~4x]. So code may PASS where MSB-2 did not.
  * Record sizes (derived 13:4x, all four observed part sizes reproduced exactly): W = 3 x 2560 x 640 = 4,915,200
    weights per expert; 16 KB-aligned parts. MSB-3 11-bit 2,064,384 B; TRL-3 (3.07 b) 1,900,544; gate/up TRL-2 + down
    TRL-3 (2.40 b) 1,474,560; TRL-2 (2.06 b) 1,277,952. 24 GB slots at Cl 0 (equal RAM): 171 / 185 / 239 / 276.
  * Run (scripts/trellis_store.py): C's rule (QOM cf 0 + gated last row, engine convention) at those slots; sure rows'
    misses at the record size. Unsure rows (code ~32 % of segments' last rows; chat ~0):
    - U1: served from the trellis pool too (no exact rows: optimistic);
    - U2: read from v3 every use (MSB 2,064,384 + LSB 720,896 for ranks 0-1, no pool: conservative).
    Keep: pool_quality's LRU kept share at each slot count.
  * Line (bytes only; quality untested here): ALIVE if code <= 10 MB per kept token under U2 at TRL-2 (or the 2.40 b mix);
    KILL if code > 12 even under U1 at TRL-2; BETWEEN otherwise.
  * Prediction [G]: TRL-2 code U1 ~9-11, U2 ~11-13; IFEval / multi ~4-6.
  * If ALIVE, it becomes a proposal for Kamen, not a build: Metal TRL-2 encoder (CPU ~45 h), ~30 GB of disk for a second
    store, Leap's decode kernel, then a map screen with encoded experts. Code-quality proxy [G]: ~1.0 %, by interpolating
    t3 / t2 flips at TRL-2's measured error.
- Wed 13:44: RESULT, TRELLIS FAST STORE byte test (pre-registered 13:4x; scripts/trellis_store.py, data/ledger/
  trellis_store.txt; 29 s). BETWEEN by the line: code U1 5.8 <= 12, but U2 43.2 > 10. Cross-checks hold: the MSB-3 11-bit
  rows bracket POOLSPLIT's 171 / 0 cell (code U1 26.8 <= 29.3 <= U2 58.4; chat U1 = POOLSPLIT's 14.4-14.5 / 10.6).
  | store (24 GB, Cl 0) | slots | code U1 / U2 | IFEval U1 / U2 | multi | keep gap closed (code / IFEval / multi) |
  |---|---|---|---|---|---|
  | MSB-3 11-bit (today's FAST + POOLSPLIT) | 171 | 26.8 / 58.4 | 14.4 / 15.0 | 10.6 | +21 / +23 / +25 % |
  | TRL-3 3.07 b (0.50x MSB-3's error [K B10-r]) | 185 | 21.8 / 54.8 | 11.4 / 12.0 | 8.5 | +32 / +36 / +38 % |
  | gate/up TRL-2 + down TRL-3, 2.40 b (1.62x [K B10-a]) | 239 | 10.1 / 46.3 | 5.0 / 5.6 | 4.1 | +67 / +75 / +81 % |
  | TRL-2 2.06 b (1.85x [K B10-r]) | 276 | 5.8 / 43.2 | 3.0 / 3.6 | 2.7 | +87 / +94 / +100 % |
  * Code's U2 is all unsure rows: 39.2 MB per kept token of v3 reads in every store (the last rows with no next draft,
    ~32 % of code segments, each reading 10 MSB + 2 LSB parts per layer uncached). Two ways out, both untested:
    (a) a 17th draft, so every last row is gated (code unsure rows -> ~chat's ~0; then code ~= U1); (b) serve those rows
    from the trellis pool too (no exact rows in FAST).
  * Chat has almost no unsure rows: the 2.40 b mix puts IFEval / multi at 5.0-5.6 / 4.1 and TRL-2 at 3.0-3.6 / 2.7 MB per
    kept token, with the keep gap to the 45 GB pool closed 75-100 % (more experts resident than today's 24 GB).
  * Quality is the open part. Sure rows served at TRL-k instead of MSB-3, interpolating the map's t3 / t2 points by
    weight error [G]: TRL-3 better than today's t3 (chat maybe ~5 / ~4.5 %); 2.40 b / TRL-2 between t3 and t2 (code ~1.0
    % PASS-ish; chat ~7-8 % BETWEEN / KILL). The real test needs encoded experts in the map.
  * Not a build ask: cost if pursued = a TRL encoder for 24,576 experts (CPU ~45 h at 4 threads, or a Metal encoder),
    ~31-36 GB of disk for a second store (TRL-2 31.4 GB, the 2.40 b mix 36.2 GB; disk 147 GB free, floor 100), Leap's decode kernel,
    then a map screen. The orchestrator / Kamen decide. Cheapest next step (CPU, mine): pin the quality proxy by
    measuring MSB-2's weight error in MSB-3 units on real records, so the t3 / t2 interpolation has a measured axis.
  * INFO add-on 13:4x (after the result; does not change the registered BETWEEN): U3 = unsure rows read through a small
    v3 pool (MSB + LSB as one 2,785,280 B unit, LRU, V slots per layer), the rest of the 24 GB RAM to trellis slots; code
    only (chat has ~no unsure rows); V in {8, 16, 32}; scripts/trellis_u3.py.
  * INFO result 13:4x (scripts/trellis_u3.py, data/ledger/trellis_u3.txt): a small v3 pool does not rescue code's unsure
    rows. v3 part 49.5 -> 44.6 / 40.8 / 35.6 MB per kept at V 8 / 16 / 32 (conservative units). Their experts barely
    repeat, so code needs (a) a 17th draft (asked MTP Head 13:46) or (b) the unsure rows served from the trellis pool (U1).
  * Reference points for (b) and the quality proxy, from map arms already run (qom_read's flip metric vs REF2-T, code /
    IFEval / multi):
    - t4_all (every expert of every row at 4 bits): 0.33 / 2.15 / 1.58 % (the metric's floor for any change);
    - t3_all (every expert of every row at 3 bits, no drops): 0.92 / 7.24 / 5.83 %;
    - qom_0_live_bq_t3 (C at the 45 GB pool): 0.77 / 6.09 / 5.29 %;
    - qom_0_live_bq_t2 (2-bit sure rows): 1.49 / 10.13 / 9.77 %.
    Serving the unsure rows at the sure rows' precision too costs little on top (t3_all vs C). Code stays under its 1.34
    line at 3 bits everywhere. A TRL-2 store (error 1.85x MSB-3, ~0.4x MSB-2 [G]) puts code at ~1.1-1.3 % [G]: PASS-edge.
- Wed 13:5x: PRE-REGISTRATION, DRAFT17 (bytes, CPU replay). MTP Head 13:5x: DFlash2 can draft the bonus row: one more
  masked block position, drafts 1-15 unchanged, likely < 1 ms; a ~3 min GPU timing is offered for 14:25. At cf 0 any
  drafted row is sure, so every last row gets gated: blocks_conv's 'all' convention vs 'engine' (today: code 32 % of
  segments have no next draft).
  * Run (scripts/draft17.py): C's rule (QOM cf 0 + 3-bit + gated last row) with conv 'engine' vs 'all' at:
    - 24 GB 147 / 46, 163 / 0, and 11-bit 171 / 0;
    - 12 GB 62 / 0;
    - the trellis TRL-2 / 2.40 b pools (U = none).
    Code only: chat has ~1 % ungated last rows.
  * Line (bytes): DRAFT17 is a lever on its own if 'all' cuts code's bytes >= 5 % at 24 GB 171 / 0. KILL as a byte lever
    if < 2 %.
  * Prediction [G]: code x0.90-0.93 at 171 / 0, since code reads 2.6 MB of LSB parts there plus the unsure rows' top-10.
  * Quality: untested. The bq arms gated 68 % of code's last rows at ~0 cost (0.42 -> 0.39 %); the other 32 % are the
    question. If ALIVE, a map arm qom_0_live_ball_c147_t3 (masks are mine) is the test.
  * Quality evidence already on the map (found 13:5x, before the run): Fable C's screen 1 (a), qom_0_live_bonus, gated
    EVERY realized bonus row (the union-only form, no resident keep): code 0.65 % (PASS), vs 0.42 % ungated and 0.39 %
    for bq (QOM form, 68 % of code's bonus rows). So gating all of code's bonus rows costs <= ~+0.25 points at the 45 GB
    pool, less with the resident keep [G]. At the 24 GB pool, C's 1.07 % -> ~1.15-1.3 % [G]: still under 1.34. The
    builder now takes _ball (dm_oea_masks.py: _bq + a gated last row when no next draft is left).
- Wed 14:04: RESULT, DRAFT17 (pre-registered 13:5x; scripts/draft17.py, data/ledger/draft17.txt; 4 s; self-check: the
  engine column equals ledger_top1 / pool_split / trellis_store). ALIVE, far past the line: x0.665 at 24 GB 11-bit 171 / 0
  (line >= 5 %). My prediction x0.90-0.93 MISSED: code's ungated bonus rows were a third of code's bytes. Each one is an
  exact row reading its full top-10 (+ LSB parts), and ranks 2-9 are rarely resident.
  | code, C's rule | engine (today) | all (a drafted bonus row) | x |
  |---|---|---|---|
  | 24 GB 147 / 46 | 35.7 | 25.2 | x0.706 |
  | 24 GB 163 / 0 (POOLSPLIT) | 32.8 | 21.9 | x0.669 |
  | 24 GB 11-bit 171 / 0 | 29.3 | 19.5 | x0.665 |
  | 12 GB 62 / 0 | 75.6 | 56.1 | x0.742 |
  | 24 GB trellis 2.40 b, 239 slots | 46.3 (U2) | 7.2 | - |
  | 24 GB trellis TRL-2, 276 slots | 43.2 (U2) | 4.0 | - |
  * Cost (MTP Head 14:03, ~/TrainLab/REPORT.md:2870, data/draft17_time.txt; K83 drafter path, L1-20):
    - 16 -> 17 positions: -0.16 / +0.14 ms per pass at context 600 / 1,900, with drafted tokens identical. This needs a
      one-line change to the _rows16 test in Leap's draft_fused.py.
    - As the code stands, +8.6 ms per check (R-78's forms fall back at any row count other than 16).
    - Masking the extra position from rows 0-15 costs the same.
  * Quality: Fable C's qom_0_live_bonus (every bonus row gated, union-only) put code at 0.65 % (PASS) vs 0.42 %
    ungated. The QOM form keeps residents, so it should cost less. Chat already gates ~99 % of last rows, so DRAFT17
    is ~free there. The map arm below measures it at the 24 GB pool.
  * Chat is unchanged by DRAFT17 (its bytes were already gated).
- Wed 14:0x: PRE-REGISTRATION, MAP SCREEN 6 = DRAFT17's quality at the 24 GB pool (before any run).
  * Arm: qom_0_live_ball_c147_t3. It is C (screen 4's qom_0_live_bq_c147_t3: 1.07 / 8.07 / 5.99 %) with every last row
    gated, including the bonus rows that have no next draft today (dm_oea_masks.py _ball). Same 147-slot pool, same
    t3mask t3k=2 serving (tbits_mask, 3 bits, k 2).
  * Lines (screen 4's): PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above. The verdict that matters is code's;
    chat should equal C's within noise (it already gates ~99 % of last rows).
  * Prediction [G]: code ~1.15-1.3 % (PASS); IFEval ~8.1 %, multi ~6.0 % (C's, unchanged).
  * Use: code PASS -> DRAFT17 joins code's 24 GB FAST stack: 35.7 -> 25.2 today, 19.5 with 11-bit + POOLSPLIT [P].
    That needs Leap's one-line _rows16 fix, or it costs 8.6 ms per check.
  * Cost: one map arm (~7 min GPU, shared lock), in a slot the orchestrator picks. Masks are built now (CPU, 1 core, ~2 min).
  * Masks built + bit-checked 14:0x (build_ball_c147.log, check_ball_c147.log: ALL OK) -> READY.
    - Realized bonus rows gated: code 188 vs C's 105, IFEval 481 vs 480, multi 699 vs 699, so only code changes.
    - Code's sure-row rank 2-9 kept share at 147 slots 73.7 % vs C's 75.8 %: the exact bonus rows had been warming the pool
      with their full top-10. A small extra quality cost; screen 6 measures it with everything else.
    - Harness line Breakthrough needs: ABLATIONS["qom_0_live_ball_c147_t3"] = tbits_mask(<masks>, 3, 2), as for
      qom_0_live_bq_c147_t3.
- Wed 14:1x: DRAFT17 ROUTED (orchestrator):
  - Leap lands the _rows16 fix first, before its FAST HC read.
  - Screen 6 takes screen 5's freed slot (Thu ~07:4x beside T0, or at RELAX_DONE).
  - MoonShots re-derives 35.7 -> 25.2 independently on CPU; until it agrees, the claim reaches Kamen marked unchecked.
- Wed 14:1x: TRELLIS STORE, quality-test sizing (CPU count, 30 s). The map's windows touch most experts: code 15,417
  distinct (layer, expert) pairs in top-2 (63 % of 24,576) and 22,645 in top-10 (92 %); IFEval 59 / 92 %, multi 52 / 90 %.
  So any map screen of a trellis store needs nearly the whole encode: B10a's ~6-7 s per expert on 6 threads makes that
  ~44 h, so it needs a Metal encoder, plus calibration Hessians or a data-free form. The cheapest real quality test is a
  layer-subset screen: TRL-2 on 8 of 48 layers (~4,096 experts, ~7-11 h CPU encode), with a harness override per layer
  range (Breakthrough). Its extrapolation to 48 layers is a risk. A proposal for after Kamen's yes, not an ask now.
- Wed 14:1x: PRICES, TRELLIS x DRAFT17 (scripts/trellis_draft17.py, data/ledger/trellis_draft17.txt; 45 s; cross-checks
  equal draft17.txt). Every last row gated (no exact rows), MB per kept token, code / IFEval / multi:
  | line | MSB-3 11-bit | TRL-3 3.07 b | TRL-2 + 3 mix 2.40 b | TRL-2 2.06 b |
  |---|---|---|---|---|
  | 24 GB (slots) | 19.5 / 14.3 / 10.6 (171) | 15.8 / 11.4 / 8.5 (185) | 7.2 / 5.0 / 4.1 (239) | 4.0 / 3.0 / 2.7 (276) |
  | 12 GB (slots) | 52.0 / 54.4 / 39.4 (65) | 44.9 / 45.2 / 32.8 (71) | 28.5 / 25.3 / 18.4 (91) | 21.7 / 18.0 / 13.0 (105) |
  Keep gap closed at 24 GB: TRL-3 +32-38 %, mix +67-81 %, TRL-2 +87-100 %.
- Wed 14:1x: PRE-REGISTRATION (draft; the encoder details are amended in with Fit12 before any run), MAP SCREEN 7 =
  the trellis FAST store's quality (the orchestrator 14:3x, Kamen's direction). A layer subset, because a full-model
  arm needs ~the whole encode (QUEUE 14:1x: 63-92 % of experts touched), ~55 h of CPU.
  * Masks: qom_0_live_ball_c239_t3 (C + DRAFT17 at the 2.40 b mix's 239-slot pool), built + bit-checked, READY 14:1x.
    Code's kept share 86.0 % vs 73.7 % at 147 slots.
  * Proposed form and layers (Fit12 to confirm): the 2.40 b mix (gate / up TRL-2, down TRL-3), S = 8 layers (3, 9, 15,
    21, 27, 33, 39, 45), all 512 experts each.
  * Four arms, the same masks, all rows in checks (prefill rows untouched):
    - 7-ref: every kept expert at MSB-3 (as C).
    - 7-trl: layers S from the TRL mix for every kept expert (any rank); other layers as 7-ref.
    - 7-cal: layers S at MSB-2 (to2bit) instead.
    - 7-cal-all: all 48 layers at MSB-2.
  * Harness need (Breakthrough): a per-layer expert-weight source for kept experts of in-check rows (TRL files /
    records3 at 2 bits), beside tbits_mask.
  * Estimate per domain: est = ref + (trl - ref) x (cal_all - ref) / (cal - ref). If (cal - ref) < 0.3 points on a
    domain, the factor is unreliable there, and I use the linear 48 / 8 = 6 instead (conservative), flagged.
  * Lines on est (screen 4's): PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above. The raw arms are reported
    beside it. Layer sensitivity is very uneven (reuse arms: L36-39 3.51 % code vs L08-11 1.19 %), so the calibration
    arms carry the extrapolation; the verdict is an estimate, not a full-model measurement.
  * Prediction [G]: 7-ref ~0.95 / ~6.8 / ~5.5 %. Mix all-layer est: code ~1.2 % PASS, IFEval ~8 % BETWEEN, multi ~7 %
    KILL.
  * Use: code PASS -> the mix store is a code-FAST candidate at 24 GB (7.2 MB per kept [P]), and the full ~36 GB store +
    encoder go to Kamen. Chat BETWEEN / KILL -> chat stays on the MSB-3 FAST stack (14.3 / 10.6).
  * Cost: four map arms (~28 min GPU, share lock) after Fit12's encode (~9 h at 4 threads for 4,096 experts [G]).
- Wed 14:3x: AMENDMENT to screen 7 (before any run; Fit12 14:2x answers + its catch).
  * Fit12's catch, checked: code has 3,366 map rows (1 flip = 0.03 points), and C's margin to 1.34 is ~9 flips. A
    4-8-layer TRL subset moves code by ~1 flip, so flip counts cannot resolve it.
  * The fix: a continuous read-out. The map stores each row's top-8 ids + log-probs for REF2-T and every arm.
    - scripts/map_kl.py computes mean per-row KL(REF2-T || arm) on the reference's top-8 + a tail bucket.
    - Over ten C-family full-model arms, flips = a + b sqrt(KL) fits with rms 0.07 points on code (max 0.10), 0.37 IFEval,
      0.15 multi (data/ledger/map_kl_fit.txt).
    - KL at the lines: code 0.0082, IFEval 0.0164, multi 0.0114. C at 147 slots: 0.0057 / 0.0386 / 0.0313.
  * S = {7, 18, 30, 41} (the layers B10-r's Hessian capture covers). Form = the 2.40 b mix, data-aware (B10-r's recipe;
    damping gate / up 0.3, down 0.1; in-sample error vs MSB-3 gate 1.33x, up 1.35x, down 0.64x). Files = an MLX 8-bit
    affine view (5.22 MB per expert, 2.67 GB per layer), read like records3 (gather_qmm bits 8).
  * Primary metric: per domain est_KL = KL_ref + (KL_trl - KL_ref) x (KL_cal_all - KL_ref) / (KL_cal - KL_ref), with
    the four arms of 14:1x on qom_0_live_ball_c239_t3. Then est_flips = the fit above. Lines on est_flips: PASS <= 1.34 /
    5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above. The raw flips and the factor (KL_cal_all - KL_ref) / (KL_cal - KL_ref)
    are reported beside them. If KL_cal - KL_ref < 2x its paired standard error on a domain, that domain is "no estimate".
  * Prediction [G] (KL from the arms above: MSB-3's sure-row contribution x the mix's error ratio ~1.1-1.4): code est KL
    ~0.005 -> ~1.0 % PASS; IFEval ~0.031 -> ~7.3 % BETWEEN; multi ~0.030 -> ~6.0 % KILL. 7-ref (C + DRAFT17 at 239
    slots, all MSB-3) is a full-model measurement on its own.
  * Encode time decides the date: background QoS 5.4 h per layer (4 layers ~22 h); normal priority 1.35 h per layer
    (~5.4 h, by ~20:00 if started now) needs the orchestrator's OK by day.
  * Resolution, measured 14:3x (same rule, 45 GB pool vs 147 slots: qom_0_live_bq_t3 vs qom_0_live_bq_c147_t3):
    - paired KL z = 8.1 / 18.5 / 10.9 (code / IFEval / multi) vs paired flips z = 1.8 / 4.9 / 2.2: KL is ~4-5x stronger in z.
    - The harness is deterministic (the capture re-run has 0 flips), so a paired difference has no run noise. Its z is
      ~sqrt(rows) / CV of the per-row shift, independent of the effect's size. A small 4-layer TRL effect stays
      detectable if its per-row shape is like this pair's (CV ~7 on code). (My 14:31 note to the orchestrator said
      "about 10x finer": the measured figure is ~4-5x in z.)
- Wed 14:4x: DRAFT17 INDEPENDENTLY RE-DERIVED (MoonShots 5679af7, its own replay on different capture checks):
  - code at 24 GB 35.6 -> 25.7 (x0.723) vs my 35.7 -> 25.2 (x0.706);
  - 12 GB x0.772 vs my x0.742;
  - chat x0.99-1.00.
  It agrees within ~2-4 %. The findings note (~/FableGod/findings/2026-11-04_1434_draft17.md) carries both. Quality is
  still screen 6's (tonight).
- Wed 14:4x: THE SWEEP, extraction done: research/SWEEP_CANDIDATES.md (33 candidates with sources). Pricing and the
  ranking go to research/SWEEP.md (due Thu morning).
- Wed 14:4x: THE SWEEP rows 1-4 RE-DERIVED (MoonShots M-35, d05b2ac, its own replay on Breakthrough's capture checks):
  - 24 GB confirmed on all four stores (x0.99-1.04); the prefill shares too (120.9 / 44.3 / 17.0 vs 117.9 / 43.1 / 17.0);
    12 GB code confirmed (x0.99-1.03).
  - 12 GB chat cells depend on the stand-in for rows wasted after a wrong draft (true routing not captured):
    - MoonShots' far-row stand-in gives x1.06-1.17 of mine;
    - mine (p + 97) agrees within 2 %;
    - true-path stand-ins give x0.66-0.88;
    - so the spread is -34..+17 %, while 24 GB holds within 7 % under any stand-in.
  - SWEEP.md marks those cells. A live capture of verify passes that routes the wrong-draft rows would pin them.
- Wed 14:5x: AMENDMENT 2 to screen 7 (before any run; Breakthrough's wiring question 14:4x).
  * The map runs D-02a's 4-bit model, so in C's form (tbits_mask 3, 2) the kept ranks 2-9 are served at 4 bits on the
    stock path. The engine's FAST pool holds MSB-3 records and serves them at 3 bits. All C-form arms and the lines
    share that frame, so in-frame comparisons (screen 4, screen 6 vs C) stand. But s7_trl moves ranks 2-9 at S from
    4 bits to TRL, which double-counts MSB-3's own ranks 2-9 term against the engine.
  * Added arm s7_ref3: every kept expert at MSB-3 at all 48 layers (bits 3, t3keep-like), same masks.
  * Reading:
    - est_abs_TRL = KL_ref + (KL_trl - KL_ref) x (KL_cal_all - KL_ref) / (KL_cal - KL_ref);
    - in-frame est = KL_ref + (est_abs_TRL - KL_ref3), mapped to flips by the pre-registered fit, verdict on the lines;
    - est_abs_TRL alone (conservative) is reported beside it, and is the verdict if s7_ref3 does not run;
    - s7_ref3 - s7_ref = the ranks 2-9 MSB-3 term at 239 slots, measured for the first time.
  * My sweep's KL model used t4_all's KL as the ranks 2-9 term. That is wrong: t4_all is the t-path's numerics floor
    (every expert via records3 at 4 bits). SWEEP.md is corrected with a t3_all-based guess until s7_ref3 measures it.
- Wed 14:5x: SCREENS 6 + 7 SCHEDULED (Breakthrough, relaying the orchestrator's batch):
  - Start 20:00, in this order: screen 6, s7_ref, s7_ref3 (~10 min; 3-bit all-kept path = the t3keep math), s7_cal,
    s7_cal_all, Fable C's 1(b) / 1(c), then s7_trl once Fit12's store is whole.
  - No new arm after 23:10, hard stop 23:30; expected end ~21:25.
  - The reading (amendment 2) is copied into Breakthrough's SCOREBOARD addendum (f70c03b). The slot prints map_kl.py
    for C, 6 and all of 7; qom_read.py is mine.
- Wed 16:3x: W4PRICE: QOM's kept ranks 2-9 priced at 4 bits ahead of tonight's s7_ref3 (scripts/w4_price.py,
  data/ledger/w4_price.txt; CPU replay [P], quality [G], UNCHECKED: MoonShots re-derives it if flag 4 is chosen).
  - Ask (the orchestrator, lounge): BigMoE's engine flag for the kept ranks 2-9 is 3 = MSB-3 (today's byte prices) or
    4 = LSB reads (C's measured map form). "s7_ref3 tonight decides; if 3-bit fails, re-price the bytes at 4."
  - Frame: C + DRAFT17 (every row gated), 11-bit parts, equal RAM per layer. Width 4 reads the LSB part (720,896 B) of
    every kept rank 2-9 expert of every row in the block. Those parts go through an LSB LRU of Cl slots, so the MSB
    pool has Cm = (B - Cl x 720,896) // 2,064,384 slots. The 3-bit prefill warms the MSB pool only, so the LSB pool
    starts cold in each window.
  - Check: width 3 at 171 / 0 = 19.5 / 14.3 / 10.6 (= trellis_draft17.txt).
  - 24 GB, MB per kept token (code / IFEval / multi):
    - at today's 171 / 0 split, every kept LSB is re-read each block: 164 / 274 / 273 (x8.4 on code). Flag 4 must
      not run on the 3-bit split.
    - best split Cl 160 / Cm 115: 40.2 / 33.0 / 24.9 (x2.06 / x2.30 / x2.36).
    - at C's measured slots (Cl 64 / Cm 148): 66.3 / 71.5 / 56.8.
  - 12 GB: width 3 at 65 / 0 = 52.0 / 54.4 / 39.4; best width 4 (Cl 48 / Cm 48) = 80.2 / 92.8 / 68.7 (x1.54-1.74).
  - Quality at the split [G] (SWEEP.md's KL model). Width 4 is the map's own C form, so only the pool term moves.
    - Code keep 70.8 % at Cm 115 vs 76.7 % at 147 (g -0.36): code KL 0.0072, flips 1.25 % (PASS, line 1.34).
    - The 12 GB rows extrapolate the pool term far below 147 (g < -1): code 1.41-1.56 % BETWEEN, a guess only.
    - Chat is over its lines in C's form at any pool (the sweep's chat barrier), so code is the verdict.
  - Upper end: the LSB part at Cl 160 is 8.3 of code's 40.2, so a warm LSB pool lowers the total by at most that.
  - So the 3-bit tail is worth ~2x FAST bytes at 24 GB (x1.5-1.7 at 12 GB). If tonight's 3-bit read fails, flag 4
    goes with the Cl 160 / Cm 115 split (code [G] PASS) at ~2x today's bytes.
- Wed 16:5x: DENSE3 MAP ARM PRE-REGISTERED (Leap's DENSE-3 = the orchestrator's FAST dense diet (b); optional, at the
  END of tonight's 20:00 batch if Breakthrough has it in by ~19:30):
  - What: the trunk's dense 4-bit g64 projections (GDN in / out, attention q / k / v / o) at their MSB-3 values, written
    in today's 4-bit format by Leap's sim (~/Leap/scripts/dense3.py: codes & 0xEEEEEEEE, biases bf16(b + s / 2);
    bitwise the real 3-bit values per Leap's check). The router, shared expert, indexer, experts, embed and head are
    untouched. The gain if it passes: -0.34 GB per check, ~-1.2..-1.8 ms once a 3-bit 16-row kernel exists (Leap).
  - Arm: qom_0_live_ball_c147_t3+dense3 (screen 6's FAST code stack + DENSE-3, through ablation()'s "+" stack). Code
    for dm_ablations.py: data/arms/dense3_arm.py (asserts >= 96 modules converted and no load-time fusion
    (BIGMOE_PKD1 / BIGMOE_SCOUT_BUNDLE) that would hide the rewrite; restores in clear()). Twin: screen 6, same batch.
  - INVALID (no verdict) if the arm's stored top-8 ids and log-probs equal screen 6's on every row: the rewrite never
    reached the forward.
  - Read: map_kl.py for the arm. The dense term = KL(arm) - KL(screen 6) per domain, with its paired SE. Code flips
    come from the arm's own count and from the pre-registered fit of its KL.
  - Verdict on code (the FAST code route; chat is over its lines in C's form at any pool, so chat is reported only):
    PASS <= 1.34 %, BETWEEN <= 2.01 %, KILL above. If screen 6 itself is not PASS on code, DENSE-3 gets no
    adoption verdict tonight; its term is recorded for later stacks.
  - Priors:
    - Leap: FAIL (3-bit vs 4-bit outputs differ by 17-18 % relative rms per projection, ~2x the 4-bit grid's noise,
      on weights every token reads).
    - Mine: BETWEEN-KILL. Code's headroom above C + DRAFT17's estimate (0.0063) is ~0.0019 KL to the 0.0082 line.
      The routed experts' ranks 0-1 at MSB-3 alone cost 0.0020.
- Wed 17:0x: DENSE3 ARM INSTALLED (Breakthrough 57813ba; my 8fefaa3 code verbatim). It runs LAST in tonight's 20:00
  batch (after 1(c), or after s7_trl if the store is whole), ~10 min; batch end ~21:35.
  - CPU wiring test on a mock model: 9/9 OK, including the exact rewrite, untouched decoys, the restore and the fusion
    guard.
  - GPU smoke (HumanEval_135, 144 rows, Breakthrough's run): 228 modules, 1,509 MB of 4-bit weights rewritten; MLX
    peak 13.9 vs 12.6 GB; 0/144 rows identical to screen 6 (valid); 0 top-1 flips vs screen 6, mean |dlp1| 0.0069.
    One window is not a read; the verdict waits for the full arm. dense3.py hash ec71be4478bc606f.
- Wed 18:0x: SCREEN 6 RESULT = PASS on code (DRAFT17's quality at the 24 GB pool; arm qom_0_live_ball_c147_t3, landed
  17:58 in the batch that started 17:38; read by the pre-registration, QUEUE 14:0x).
  - Flips (qom_read.py): code 1.07 % (36), IFEval 7.83 % (266), multi 5.99 % (213): PASS / BETWEEN / KILL, C's verdicts.
  - Paired vs C (qom_0_live_bq_c147_t3): code 12 / 12 (z 0.00), IFEval 13 / 21 (z -1.37), multi 0 / 0. Gating code's
    last 83 ungated bonus rows moved no code flip net.
  - KL (map_kl.py), C -> screen 6: code 0.0057 -> 0.0061 (+0.0004; the sweep's [G] term was +0.0006); IFEval 0.0386 and
    multi 0.0313 unchanged.
  - Prediction was code ~1.15-1.3 % (PASS): it came in at C's 1.07 %.
  - Use: DRAFT17 joins code's 24 GB FAST stack. Code bytes are 35.7 -> 25.2 MB per kept token on today's stack, and 19.5
    with 11-bit + POOLSPLIT [P] (MoonShots re-derived x0.723). BigMoE built it (R57_D17, 723ee9b).
  - UNCHECKED until Breakthrough's interim read-out (~19:05, s67_slot.log) re-derives screen 6's verdict vs C.
- Wed 18:4x: S7_REF3 READ = the 3-bit tail PASSES on code, so BigMoE's flag stays 3 (s7_ref 18:21, s7_ref3 18:42;
  map_kl.py + paired flips; UNCHECKED until Breakthrough's interim read-out, ~21:00).
  - s7_ref (C + DRAFT17 at 239 slots, kept ranks 2-9 at 4 bits = the map's C form): 0.74 / 6.48 / 5.35 % flips; KL
    0.0040 / 0.0232 / 0.0236.
  - s7_ref3 (every kept expert at MSB-3 at all 48 layers = the engine's FAST frame): 1.04 / 7.62 / 6.36 %; KL 0.0051 /
    0.0297 / 0.0324.
  - Paired (only-ref / only-ref3): code 7 / 17 (z +2.04), IFEval 60 / 99 (z +3.09), multi 57 / 93 (z +2.94).
  - E29, the ranks 2-9 MSB-3 term at 239 slots (first measurement): +0.0011 / +0.0065 / +0.0088. The sweep's [G] guess
    was 0.0017 / 0.0071 / 0.0078.
  - Verdict for the flag: 3 STANDS on code.
    - Direct: s7_ref3's code 1.04 % at 239 slots, under the 1.34 % line.
    - At the 24 GB MSB-3 pool [G]: today's stack in the map frame (sweep row 0, code KL 0.0058) + E29 = 0.0069 -> 1.23 %
      (PASS; the margin is ~4 flips).
    - Flag 4 (x2.06 bytes, W4PRICE 16:3x) is not needed for code.
  - Chat in the engine frame [G]: IFEval 7.67 -> 8.32 % (BETWEEN -> KILL, line 8.13); multi 6.01 -> 6.88 % (KILL
    either way). Flag 4 would keep IFEval BETWEEN but not multi, so neither width opens a chat route.
  - Next: the sweep's model takes E29 in place of its guess (engine frame: each row adds r x E29). It moves once, with
    screen 7's final read-out (~21:20).
- Wed 19:5x: SCREEN 7 CALIBRATION READ + THE SWEEP IN THE ENGINE FRAME (v1.1; [G] rows UNCHECKED, asked of MoonShots).
  - Breakthrough's interim read-out (19:38, ba53ede) matches my s7_ref3 read. It ran my map_kl.py, so its KL numbers
    are the same code, not a re-derivation; its flip counts come from its own reader.
  - scripts/s7_read.py = amendment 2's reading, one command when s7_trl lands (data/ledger/s7_read.txt):
    - gate OK on all three domains: cal - ref z 5.8 / 9.0 / 8.6 per row (2.4 / 12.6 / 5.5 per window);
    - scale 11.1 / 12.8 / 17.1 (Breakthrough's 11.7 on code is the rounded table's);
    - E29 paired: +0.00114 / +0.00649 / +0.00886, z 3.4 / 9.6 / 13.6 per row, 1.8 / 7.9 / 9.6 per window.
  - Model checks (scripts/sweep_engine.py, data/ledger/sweep_engine.txt):
    - r is an MSE ratio vs the 4-bit weights (Fit12's E4 units: MSB-3 = 3 E4, MSB-2 = 15 E4, so MSB-2 = r 5).
      s7_cal_all measures 0.0180 / 0.1094 / 0.1465; the model says 0.0177 / 0.1004 / 0.1192 (implied r 5.10 / 5.51 /
      6.26). Linear in MSE holds on code; chat runs 8-19 % above it at this size.
    - s7_ref (239 slots) closes 0.89 / 0.86 / 0.92 of the 147 -> 45 GB KL gap; the keep proxy said 0.67 / 0.75 /
      0.81. v1's pool term was pessimistic for big pools; v1.1 interpolates between the three measured anchors.
  - SWEEP v1.1, est. flips % in the engine frame [G] (code / IFEval / multi):
    - today's FAST stack 1.18 / 8.22 / 6.82 (PASS / KILL / KILL);
    - TRL-2 1.30 / 8.52 / 7.96 and the mix 1.24-1.31 / 8.29-8.71 / 7.57-8.02 (code PASS at the edge);
    - TRL-3 0.99 / 7.07 / 5.66, GPTQ-3 1.06 / 7.53 / 5.99 (PASS / BETWEEN / KILL).
    - At 24 GB no store passes IFEval, even at r = 0. Chat needs a different rule: 1(c), QOM cf 0.5 with the bonus
      row gated, sits at 0.48 / 2.88 / 1.91 % on the map (45 GB pool). Its 24 GB bytes are the next piece.
  - tail_bits = 3 (BigMoE told 18:4x; Breakthrough asked 19:38).
- Wed 19:5x: FRAME FIX to the s7_ref3 read (18:4x) and SWEEP v1.1 (19:5x) -> SWEEP v1.2.
  - The FAST stack certifies against REF2-T (EvalSuite PREREG_f1), and REF2-T serves ranks 2-9 at 3 bits (BigMoE's
    preset 4433333333). The FAST pool's MSB-3 kept experts therefore match the reference's own precision. E29 is
    REF2-T's own cost against the 4-bit model, not a FAST cost.
  - The verdict frame is amendment 2's in-frame one: KL = map(c) + (r - 1) x (T01 + E29(c)). Adding r x E29 (v1.1,
    and my 18:4x "engine frame") judges against the 4-bit model; it stays as the conservative column.
  - Corrections, est. flips % [G] (code / IFEval / multi):
    - today's FAST stack at 24 GB: 1.08 / 7.61 / 5.98 (PASS / BETWEEN / KILL), not 1.18-1.23 / 8.22-8.32 KILL /
      6.82-6.88. IFEval stays BETWEEN.
    - TRL-2 1.19 / 7.87 / 7.20; the mix 1.13-1.21 / 7.63-8.09 / 6.79-7.28. The code routes under 10 MB pass with
      room [G]; s7_trl decides the mix.
    - TRL-3 0.86 / 6.32 / 4.61; GPTQ-3 0.94 / 6.85 / 5.03 (PASS / BETWEEN / BETWEEN).
  - Unchanged: flag 3 stands, and it is now the like-for-like choice (flag 4 would move the kept experts away from
    REF2-T toward the 4-bit model, at ~2x bytes). Screen 7's reading was already in this frame (amendment 2).
  - MoonShots re-derived v1.1's arithmetic (413d214: within +-0.2 points; the KL approximations agree to 4 decimals).
    Its flags are in v1.2: TRL-2 / mix chat cells +0.1..+0.5 possible (KL grows faster than linear in r on chat), the
    TRL rows wait for s7_trl, and the 171-slot rows +-0.1 (an uninterpolated stretch).
  - DENSE-3: a Thu leftover (13:10 spare). A measurement beats two predictions, and the trunk-diet row has no number.
- Wed 20:0x: CHAT ROUTE PRICED (scripts/chat_route_price.py, data/ledger/chat_route_price.txt; [P] UNCHECKED, quality [G]).
  - The rule: Fable C's 1(c), QOM cf 0.5 with the bonus row gated at 0.5, at 24 GB on today's 11-bit parts, RAM split
    searched. Check: the same replay gives today's cf 0 stack 19.5 / 14.3 / 10.6 (= trellis_draft17.txt).
    (A first run read 26.2 / 40.4 / 41.4 there: my bug, its check ran at cf 0.5; fixed before any number was used.)
  - 4-bit sure rows (top-2 LSB read): 61.8 / 87.9 / 87.2 MB per kept token (Cl 92 / 69 / 69). Quality [G] in the
    verdict frame 0.65 / 4.34 / 2.85 %: PASS on all three domains. It adds bonus5g (measured) and the 164-slot pool
    term (qom_0.5_live_c164); the best split holds 147 MSB slots, so chat may be a little optimistic.
  - 3-bit sure rows: 44.5 / 73.0 / 75.9 MB (Cl 23); [G] 0.92 / 5.56 / 4.35 % (PASS / BETWEEN / BETWEEN).
  - EXACT (REF2-T reads) at its best split: 190.6 / 176.8 / 155.1. So chat PASS at 24 GB costs ~half of EXACT's bytes,
    ~6-8x today's FAST stack (14.3 / 10.6). A per-request rule (code cf 0, chat cf 0.5) gets PASS on all three.
  - Kill / confirm: a map arm of the rule at its split (qom_0.5_live_bonus5g at ~147 slots); then EvalSuite.
- Wed 20:0x: FRAME CONFIRMED + CREDIT / STRICT VIEWS (SWEEP v1.2).
  - EvalSuite 20:06: FAST same-scores tests compare against REF2-T's rows (4433333333, ranks 2-9 at 3 bits), not
    REF2b; w24t's plan is that one; after a NAX6 PASS Thursday, NAX6's rows (same tiers).
  - MoonShots (cd7e1f0): the frame is sound for today's row (its in-frame 1.05 / 7.73 / 5.87 vs my 1.08 / 7.61 /
    5.98). For independent quantizers, the credit view (r - 1) x E29 is the expected-score view; disagreement with
    REF2-T is the strict view (r + 1) x E29. Both views are in SWEEP v1.2 (mine; MoonShots' within ~0.2).
    - Credit view, code / IFEval / multi: TRL-2 1.19 / 7.87 / 7.20; mix 1.13-1.21 / 7.63-8.09 / 6.79-7.28;
      TRL-3 0.86 / 6.32 / 4.61; GPTQ-3 0.94 / 6.85 / 5.03.
    - Strict view: TRL-2 1.40 / 9.12 / 8.66; mix 1.34-1.41 / 8.89-9.28 / 8.29-8.70; TRL-3 1.11 / 7.74 / 6.54;
      GPTQ-3 1.17 / 8.16 / 6.83.
    - The code routes under 10 MB are PASS (credit) or BETWEEN (strict). s7_trl's pre-registered verdict is the
      credit view; s7_read.py now prints the strict one beside it (information).
- Wed 20:1x: PRE-REGISTERED (before any run; a Thu map-slot candidate): CHAT ROUTE ARM, the kill test for SWEEP row 10a.
  - Arm: qom_0.5_live_bonus5g at 147 MSB slots, 4-bit sure rows (no t3): QOM cf 0.5 masks at pool 147 (dm_oea_masks),
    then Fable C's bonus gate at 0.5 with the gate file's cf (bytes10_bonus_masks.py <arm> 0.5 g). 147 = row 10a's
    IFEval / multi split (Cl 69).
  - Frame: the map's 4-bit sure rows = REF2-T's ranks 0-1, and its 4-bit kept ranks 2-9 stand in for REF2-T's MSB-3, so
    the map's flips are REF2-T's frame here (no E29 term).
  - Lines: PASS <= 1.34 / 5.42 / 3.69 % on every domain = the chat route holds at 24 GB (bytes 61.8 / 87.9 / 87.2);
    BETWEEN <= 1.5x; KILL above. Paired with qom_0.5_live_c164 and qom_0.5_live_bonus5g (information).
  - Prediction [G]: ~0.7 / ~4.6 / ~3.0 % (row 10a's 0.65 / 4.34 / 2.85 at 164 slots, a little worse at 147).
  - Cost: one mask build (CPU, ~10 min) + one map arm (~10 min GPU, share lock), on Breakthrough's slot.
- Wed 20:1x: CHAT ARM MASKS READY, bit-checked (the orchestrator 20:1x: run tonight after s7_trl if ready by ~22:30).
  - Built: qom_0.5_live_c147 (dm_oea_masks.py, uniform 147-slot pool; kept share of sure-row rank 2-9 pairs 77.8 /
    87.0 / 88.4 %), then Fable C's bonus gate at 0.5 (proxy g), imported with its output set to data/dm_masks:
    qom_0.5_live_c147_bonus5g (gated 165 of 1,368 bonus rows, pairs x0.59, as 1(c)).
  - scripts/chat_arm_check.py -> OK (data/dm_masks/check_c147_chat.log):
    - ranks 0-1 equal across pools m / c164 / c147, and kept share orders 95.1 > 84.8 > 82.5 %;
    - the gate only drops;
    - the 165 gated rows equal 1(c)'s masks row for row;
    - 7,774 other rows differ from 1(c) by the smaller pool's drops.
  - Harness line (Breakthrough): as 1(c)'s arm, PERT.update(rmask='/Users/kamen/NewIdeas/data/dm_masks/qom_0.5_live_c147_bonus5g',
    renorm=False); no t3; 36 windows. Read per 20:1x (lines on all three domains; paired vs qom_0.5_live_bonus5g and
    qom_0.5_live_c164 for information; map_kl.py beside).
- Wed 20:1x: CHAT ROUTE BYTES RE-DERIVED (MoonShots M-37, c388167, its own replay on Breakthrough's capture checks):
  - With the same stand-in, every cell is within 0-3 % at the same best Cl: 10a 60.3 / 87.2 / 87.4, 10b 43.0 / 72.3 /
    76.0, EXACT 190.0 / 177.3 / 155.8, and the cf 0 check 19.5 / 14.8 / 10.9.
  - Far-row stand-ins: chat +2..+6 %.
  - Quality caveat (its attack): 10b stacks two perturbations on the same sure rows (drops + 3-bit top-2), so chat may
    be super-additive. 10a (top-2 at 4 bits) carries less. Fable C: 1(c)'s cost partly comes from its union-only
    gated row (no resident keep), so the arm is conservative. Tonight's arm measures 10a directly.
- Wed 22:0x: CF SWEEP, BYTES SIDE: ALIVE (scripts/cf_sweep_price.py, data/ledger/cf_sweep_price.txt; kill line in the
  script's docstring before the run). Idea: the chat PASS route (cf 0.5 + bonus gate) costs ~88 MB per kept token on
  chat at 24 GB; a lower threshold may pass chat for far fewer bytes.
  - Bytes [P] (24 GB, 11-bit parts, 4-bit sure rows, bonus row gated at the same t, best split), code / IFEval / multi:
    cf 0.1 47.5 / 29.6 / 21.8; cf 0.2 47.6 / 33.6 / 30.2; cf 0.3 50.2 / 49.2 / 51.7; cf 0.4 55.6 / 68.9 / 71.5;
    cf 0.5 61.8 / 87.9 / 87.2 (= chat_route_price.txt, the check).
  - Read: cf 0.3 is x0.56 / x0.59 of cf 0.5 on IFEval / multi -> ALIVE (line: <= 0.6x on both). cf 0.2 is x0.38 / x0.35.
  - Quality is the open half. The map's own arms (measured, 45 GB pool, 4-bit sure rows) show the threshold barely
    matters there: cf 0 / 0.5 / 0.9 = 2.53 / 2.68 / 2.24 % IFEval, 1.77 / 1.63 / 1.44 % multi. At small pools it
    does: cf 0 + t3 at 147 slots is 8.07 / 5.99 %, and the costs on chat are the bonus gate at cf 0 (qom_0_live_bq vs
    qom_0_live: +1.6 / +1.5 points), t3 (+1.9 / +2.4) and the pool. [G] for cf 0 + 4-bit + bq at 147 slots
    (bq_c147_t3 - T01): ~0.9 / ~6.9 / ~4.8 %; cf 0.5: 0.65 / 4.34 / 2.85 % (tonight's chat arm measures it).
  - Next: one map arm, qom_0.3_live_c147_bonus3g (masks building), pre-registered below once bit-checked.
- Wed 22:0x: CF 0.3 = CF 0 ON THE LIVE MAP, and the cf 0.3 chat arm PRE-REGISTERED:
  - Masks: qom_0.3_live_c147 equals qom_0_live_c147 bit for bit in 36 / 36 windows (data/dm_masks/check_cf03.log).
    Live checks keep drafts only while the running cf product is >= 0.3, so every drafted row has cf >= 0.3 and any
    threshold <= 0.3 changes nothing for drafted rows. On the live map and the replay, cf 0.1 / 0.2 / 0.3 differ only
    by the bonus-row gate (the bonus row is gated when its next draft's cf >= t). So the bytes drop from cf 0.3 to
    cf 0.1 (IFEval 49.2 -> 29.6 MB) is the bonus row alone.
  - Arm: qom_0.3_live_c147_bonus3g (= cf 0 drafted rows, 4-bit sure rows, 147 MSB slots, the bonus row gated at next
    cf >= 0.3; Fable C's bonus builder, gated 528 of 1,368 bonus rows; drops only; every row the 0.5 gate changed is
    gated here too). Harness line as the chat arm's: PERT.update(rmask='/Users/kamen/NewIdeas/data/dm_masks/
    qom_0.3_live_c147_bonus3g', renorm=False); no t3; 36 windows; ~5 min.
  - Lines as 20:1x, measured flips: PASS <= 1.34 / 5.42 / 3.69 % on every domain = a chat route at 50.2 / 49.2 /
    51.7 MB per kept token holds at 24 GB (x0.56-0.59 of the cf 0.5 route's chat bytes); BETWEEN <= 1.5x; KILL above.
  - [G] range (additive KL; 45 GB qom_0_live + the pool term at 147 slots between cf 0.5's 4-bit c164 term and cf 0's
    t3-form term + the bonus term between bonus5g's and bq's): code 0.66-0.88, IFEval 4.5-6.9, multi 2.9-4.8 %. A real
    coin flip on chat. Paired with tonight's chat arm (qom_0.5_live_c147_bonus5g) and qom_0_live_bq_c147_t3.
  - Slot: after s7_trl and the chat arm if it can start by 22:50 (Breakthrough's slot), else Thursday's first map slot.
- Wed 22:0x: the orchestrator said yes to the cf 0.3 chat arm. It runs last tonight (after the chat arm, s7_trl and the
  Rule C top-8 maps) if it can start by 22:50, else first in Thursday's map slot; Breakthrough has the harness line.
  Read: scripts/chat_arm_read.py qom_0.3_live_c147_bonus3g (it now takes an arm name; retested on 1(c): 0.48 / 2.88 /
  1.91 %).
- Wed 22:1x: CHAT ARM = PASS on every domain (pre-registered 20:1x; first read, independent re-derivation pending).
  - qom_0.5_live_c147_bonus5g (QOM cf 0.5 + the bonus row gated at 0.5, 4-bit sure rows, the 24 GB line's 147 MSB
    slots), landed 22:14, 36 windows. Measured flips: 0.71 / 4.30 / 2.73 % (code / IFEval / multi) vs lines 1.34 /
    5.42 / 3.69 -> PASS x3. Mean KL 0.0025 / 0.0115 / 0.0075.
  - Prediction ~0.7 / ~4.6 / ~3.0 % ([G] 0.65 / 4.34 / 2.85): on target.
  - Paired against 1(c) (the same rule at the 45 GB pool, 0.48 / 2.88 / 1.91 %), the 24 GB pool costs +0.23 / +1.42 /
    +0.82 points (McNemar p 0.15 / 5e-6 / 0.003).
  - So the chat route holds at 24 GB: ~88 MB per kept token on chat [P] (61.8 / 87.9 / 87.2 at the best splits,
    re-derived c388167), about half of EXACT's bytes. SWEEP row 10a and item 2 now carry the measured numbers.
  - Read: scripts/chat_arm_read.py, data/ledger/chat_arm_read.txt.
- Wed 22:2x: SCREEN 7 VERDICT (s7_trl, pre-registered amendment 2): the 2.40 b trellis mix = BETWEEN / KILL / KILL.
  - s7_trl landed 22:20 (36 windows, every expert from Fit12's store, no fallback). Read: scripts/s7_read.py,
    data/ledger/s7_read.txt. Re-derived by Breakthrough's read-out (8a473b0): unrounded KL ref 0.00398 / 0.02317 /
    0.02357, trl 0.00458 / 0.02612 / 0.02776, scale 11.12 / 12.82 / 17.05, the same verdict.
  - trl - ref at the 4 layers: +0.00060 / +0.00296 / +0.00419 (z 2.5 / 4.6 / 5.5 per row; 1.9 / 4.3 / 6.7 per window).
  - 48-layer estimates, flips % (code / IFEval / multi):
    - in-frame (the verdict): 1.45 / 9.54 / 10.44 -> BETWEEN / KILL / KILL;
    - est_abs (conservative): 1.54 / 10.07 / 10.98;
    - strict: 1.63 / 10.57 / 11.48.
  - The MSE model missed. At the 4 layers the mix did 0.48 / 0.44 / 0.58 of MSB-2's damage. In the SWEEP's units that
    is r_eff 2.72 / 2.55 / 3.15, against r 1.62-1.87 from MSE, so 1.7-2.1x / 1.6-1.9x / 2.0-2.5x the predicted
    trl - ref.
    - Caveat: multi's scale (17.1 vs linear 12) comes from MSB-2 at all 48 layers and may overstate a smaller
      perturbation. At a linear 12, multi's in-frame estimate is still ~9.1 % (KILL).
  - Consequences: SWEEP row 2 is KILLED as a 24 GB route; rows 1 / 3 / 4 carry a caveat (their r comes from MSE ratios);
    item 1 rewritten: no measured code PASS route under 10 MB at 24 GB (today's stack stays at ~19.5 MB).
- Wed 22:3x: TRELLIS RESCUE PRICED (the orchestrator's 22:3x ask; scripts/rescue_price.py, data/ledger/rescue_price.txt).
  All quality [G] from the SWEEP model with screen 7's measured r_eff (2.72 / 2.55 / 3.15) in place of MSE ratios.
  Unchecked (MoonShots re-derives before Kamen).
  - A. 24 GB, the mix on a fraction of layers, MSB-3 elsewhere: code keeps its line up to ~60 % of layers (1.31 % at
    12.1 MB on code vs 19.5 today; store ~41 GB). Chat is over its lines at every fraction: today's stack is already
    BETWEEN / KILL, and trellis only adds. The chat PASS route (cf 0.5 rule) needs 4-bit top-2 = today's MSB-3 + LSB
    parts.
  - B. 45 GB all-resident: no selection keeps all three lines.
    - All-MSB-3 (50.7 GB of experts, 75 % resident at ~38 GB) already misses multi: 3.92 % vs 3.69. REF2-T keeps
      ranks 0-1 at 4 bits, so any store without them pays T01.
    - Fitting needs >= ~80 % trellis: 1.19 / 7.65 / 8.09 %. The full mix (34.8 GB of experts, fits): 1.30 / 8.23 /
      8.84 %.
  - C. TRL-3 at 24 GB (row 3, 185 slots): 15.8 / 11.4 / 8.5 MB (x0.81 of today). Quality between 0.86 / 6.32 / 4.61
    (MSE r 0.52) and 1.04 / 7.34 / 5.87 (r ~1.0 if it misses like the mix): code PASS, chat BETWEEN either way.
    At 45 GB: 44.5 GB of experts, 85 % resident.
  - The model runs low on chat: phi 1 gives 9.21 / 9.41 vs the measured in-frame 9.54 / 10.44, so chat cells are if
    anything optimistic.
  - Next (cheapest split of the miss): a down-only arm at screen 7's 4 layers (TRL-3 on down, MSB-3 gate / up; the
    store already holds the files). It needs Breakthrough's TRL view to select matrices.
- Wed 22:4x: POWER CHECK KILLS MY DOWN-ONLY ARM DESIGN before anyone builds it (data/ledger/down_arm_power.txt).
  - s7_trl_down (TRL-3 on down, MSB-3 on gate / up, at screen 7's 4 layers) is feasible on Breakthrough's side (1-1.5 h
    of build, ~7 min of map time).
  - But the expected effect is z ~1-2 per domain, and the SE of TRL-3's effective ratio is ~1.0 with all three
    domains combined. Separating 0.52 (MSE) from ~1.0 (the mix's miss factor) needs ~0.24, i.e. ~17x the rows or
    layers. A 4-layer KL arm cannot decide TRL-3.
  - Better tests, in cost order:
    - (1) Fit12, CPU: its output-weighted judge on code / multilingual inputs, if it has such captures at L7 / 18 /
      30 / 41. If TRL-2's ratio there is ~2.5-3.5 (vs 1.85 on its broad set), the miss is the inputs' distribution,
      and TRL-3's ratio on the same inputs is its real r.
    - (2) Breakthrough, GPU, about the down arm's build: a per-row layer-output probe at S on the map's own hidden
      states. Squared error of the kept experts' summed output under MSB-3 / the mix / down-only TRL-3 vs 4-bit:
      thousands of row-layers, so high power. It separates "the inputs differ" (output-error ratio ~2.5-3.2) from
      "trellis errors cost more KL per unit error" (ratio ~1.6-1.9). In the second case only big KL arms can rank
      trellis codes, and the family is closed here.
    - (3) Stop: TRL-3's best case is ~19 % fewer bytes at 24 GB at about today's quality.
- Wed 22:3x: the orchestrator agreed: no down-only arm. Test (1) goes to Fit12 directly (CPU only, <= 4 GB, share lock,
  before 23:00 or Thursday by day). The GPU probe (2) is built only if (1) says the inputs explain the miss; otherwise
  the trellis family stops here. PRE-REGISTERED reading of (1), before Fit12 runs it:
  - Quantity: Fit12's output-weighted judge (routed tokens preferred) of the screen 7 store vs MSB-3 on captured
    code / multilingual inputs at L7 / 18 / 30 / 41. Whole-expert output error if available, else the mean of the
    three matrices (TRL-2 gate / up, TRL-3 down).
  - Compare with the map's effective ratio for the mix: 2.72 / 2.55 / 3.15 (code / IFEval / multi; its broad-set
    judge said 1.62-1.87).
  - >= ~2.4 on the code or multilingual inputs: the inputs' distribution explains the miss. TRL-3's ratio on those
    inputs is its real r, and the GPU probe (2) is worth building.
  - <= ~2.0: the inputs do not explain it; trellis errors cost more KL per unit error here. The family stops.
  - Between: weak; the orchestrator's call.
- Wed 22:4x: CF 0.3 CHAT ARM = PASS / BETWEEN / BETWEEN (pre-registered 22:0x; Breakthrough's s67 read-out gives the
  same numbers).
  - qom_0.3_live_c147_bonus3g (= cf 0 drafted rows + the bonus row gated at next cf >= 0.3, 4-bit sure rows, 147
    slots), landed 22:42. Flips 0.74 / 5.56 / 3.86 % vs lines 1.34 / 5.42 / 3.69; KL 0.0038 / 0.0201 / 0.0126.
  - [G] range was 0.66-0.88 / 4.5-6.9 / 2.9-4.8: inside it, at its low end. A near miss on chat: +0.14 / +0.17 points
    over the lines.
  - Against the cf 0.5 arm, the cheaper route costs +0.03 / +1.26 / +1.13 points (McNemar p 1 / 0.0008 / 0.0001) and
    saves 0.81 / 0.56 / 0.59x of the chat bytes. So the chat PASS route stays cf 0.5 at ~88 MB.
  - Read: scripts/chat_arm_read.py qom_0.3_live_c147_bonus3g -> data/ledger/chat_arm_read_qom_0.3_live_c147_bonus3g.txt.
- Wed 22:4x: SPLIT-THRESHOLD CHAT ROUTE, BYTES: both pre-registered hybrids DIE ON BYTES (scripts/split_price.py,
  data/ledger/split_price.txt; line in the docstring before the run: <= 0.75x the cf 0.5 route's chat bytes on both).
  - The checks reproduce (cf 0.5 87.9 / 87.2, cf 0.3 49.2 / 51.7).
  - V1 (drafted 0.3 + bonus 0.5): 73.3 / 71.9 MB (x0.83 / 0.82). V2 (drafted 0.5 + bonus 0.3): 66.0 / 69.2 (x0.75 / 0.79).
    Each of the two changes saves only ~17-25 %; the cf 0.3 route's ~42 % needs both.
  - Not pre-registered, priced beside them: drafted 0.3 + bonus 0.4 = 62.4 / 64.0 MB (x0.71 / 0.73). It meets the
    same bytes line. Quality [G], scaling the cf 0.5 -> 0.3 flip gap by bytes share: ~5.2 / ~3.5 % (lines 5.42 /
    3.69), a coin flip slightly in its favour. A candidate map arm for Thursday, low priority (~28 % fewer chat bytes
    than the PASS route if it passes).
  - Side fact: drafted 0.3 with the bonus row never gated costs 88.7 / 82.7 MB, as much as the cf 0.5 route. On chat the
    bonus row carries a large share of the bytes.
- Wed 22:4x: TRELLIS TEST (1) READ: the inputs' distribution does NOT explain the miss, so the trellis family STOPS
  (pre-registered 22:3x, 5fa0f47).
  - Fit12 ran its output-weighted judge on the map's OWN answer rows (Breakthrough's xcapture, the 36 windows, 10,316
    rows) at L7 / 18 / 30 / 41. Whole-expert output error of the store vs MSB-3, router-weighted, ranks 2-9:
    code 1.12-1.31, IFEval 1.58-1.83, multi 1.59-1.75.
  - Same judge on its own held-out parts: code 1.00-1.25, multi 1.18-1.38, IFEval prompts 1.13-1.40, English prose
    1.62-1.92.
  - Per matrix on the map's rows: gate / up (TRL-2) code 1.40-1.72, chat 2.03-2.48; down (TRL-3) 0.61-0.79.
  - Rule: whole-expert ratio <= ~2.0 in all three domains -> STOP. Chat inputs are harder than Fit12's held-out parts
    (whole ~1.7 vs ~1.25); code inputs are not.
  - Residual: the KL damage is 2.1-2.4x the output-error ratio on code and 1.4-2.0x on chat. Part is noise (r_eff's SE
    is ~+-1.0 on code, +-0.5-0.6 on chat). The rest means one of two things:
    - trellis errors cost more KL per unit of output error than truncation (structured vs random errors); or
    - s7_trl and s7_cal differ in which rows or ranks they touch. A harness question for Breakthrough; it changes no
      build decision, only how rows 3-4 (TRL-3, GPTQ-3) are priced.
  - Fit12's files: ~/Fit12/data/trl_judge/judge_map.json, judge_L{7,18,30,41}.json; scripts fit_trl_judge_map.py,
    fit_trl_judge.py.
- Wed 22:4x: PRE-REGISTERED (the orchestrator 22:5x: a low-priority Thursday arm after DENSE-3, only if a share slot is
  free): qom_0.3_live_c147_bonus4g.
  - The arm: cf 0 drafted rows (= qom_0.3_live_c147) + the bonus row gated at next cf >= 0.4. 313 of 1,368 bonus rows
    gated; drops only; every row gated at 0.4 is gated identically at 0.3 (data/dm_masks/check_cf03_b4.log).
  - Harness line: PERT.update(rmask='/Users/kamen/NewIdeas/data/dm_masks/qom_0.3_live_c147_bonus4g', renorm=False);
    no t3; 36 windows; ~5 min.
  - Bytes [P]: 53.9 / 62.4 / 64.0 MB per kept token at 24 GB (x0.87 / 0.71 / 0.73 of the cf 0.5 route).
  - Lines, measured flips: PASS <= 1.34 / 5.42 / 3.69 on every domain = a ~63 MB chat route holds; BETWEEN <= 1.5x;
    KILL above.
  - [G]: code 0.7-0.8, IFEval 4.9-5.5, multi 3.2-3.8 %: the cf 0.5 -> 0.3 flip gap, scaled by bytes share.
  - Paired with the cf 0.3 arm (primary) and the cf 0.5 arm. Read: scripts/chat_arm_read.py qom_0.3_live_c147_bonus4g.
- Wed 22:5x: SAME READS CONFIRMED (Breakthrough, from the run logs). s7_trl and s7_cal made identical reads: 144 calls
  and 368,983 (row, rank) pairs over the 36 windows, equal window by window. Every kept expert of the gated rows at the
  4 layers (ranks 0-1 + the mask's 2-9) comes from the source; prompt rows stay on the 4-bit path; DRAFT17 gates every
  realized check row. So screen 7's gap is not a coverage artifact.
  - Predicted trl - ref from Fit12's whole-expert output ratios (layer means 1.23 / 1.68 / 1.67), scaled by the
    measured MSB-2 pair: D_cal x ((r - 1) T01 + r E29) / (4 T01 + 5 E29) = 0.00017 / 0.00161 / 0.00177.
  - Measured: 0.00060 / 0.00296 / 0.00419 -> 3.5x / 1.8x / 2.4x (code noisy: z 2.5).
  - The same calibration fits MSB-2 itself within 1.0-1.3x.
  - Finding: on identical reads and inputs, the trellis store's errors cost ~2-3.5x more KL per unit of expert-output
    error than truncation's. Mechanism unknown: structured (data-aware) errors aligned with what later layers amplify
    is the leading guess. SWEEP row 4 (GPTQ-3, also data-aware) carries the risk; any data-aware code needs a KL
    screen, not an error ratio.
  - The Thursday arm qom_0.3_live_c147_bonus4g is installed on Breakthrough's side (8 / 8 checks; 313 rows = mine).
- Thu 07:0x: CLOUD3 C3-R15 / R15b MASKS BUILT, BIT-CHECKED, BYTES RE-DERIVED (the orchestrator 07:0x; run in
  Breakthrough's Fri 00:30 map slot after Cloud3's tile arms; hand-off after Fable C's ~09:00 audit).
  - Build: dm_oea_masks.py parse() gains qom_<thr>_u<k>[_reach<tau>] (QOM with unsure rows at top-k + union members +
    resident; optional REACH-CUT, rows keeping residents). Then Fable C's bonus gate at 0.5 (proxy g), exactly as the
    measured route qom_0.5_live_c147_bonus5g. So each arm differs from the measured route by its cut alone. (Cloud3
    wrote "qom_0.5_live_bq_c147's form"; the measured route uses the union-only gate, not dm_oea_masks' _bq.)
    - R15 = qom_0.5_u8_live_c147_bonus5g
    - R15b = qom_0.5_u8_reach0.5_live_c147_bonus5g
  - Bit-checks (scripts/r15_check.py, data/dm_masks/check_r15.log): all OK.
    - u10 rebuilds the route's base bit for bit (36 / 36).
    - u8 drops only ranks 8-9 on the unsure rows. 7,724 sure rows move through the smaller unions and the lighter pool
      (39,956 experts added, 46,610 dropped on unsure rows).
    - The reach cut keeps ranks 0-1 everywhere.
    - Both gates drop only, on the route's own 165 rows.
  - Bytes, re-derived (scripts/r15_price.py; my replay: 24 GB, 11-bit parts, best split; the route reproduces 61.8 /
    87.9 / 87.2):
    - R15: 54.3 / 68.2 / 64.0 MB (x0.88 / 0.78 / 0.73; Cloud3 x0.93 / 0.81 / 0.82);
    - R15b: 49.3 / 57.1 / 54.0 MB (x0.80 / 0.65 / 0.62; Cloud3 x0.86 / 0.64 / 0.64).
    Within ~10 %, mine a little more favourable.
  - PRE-REGISTERED (Cloud3's lines, adopted as filed; measured flips):
    - R15: PASS = <= 1.34 / 5.42 / 3.69 on every domain AND within 1.3x of the route's 0.71 / 4.30 / 2.73. KILL if IFEval
      or multi cross their lines. Prediction (Cloud3): within +0.1 of the route.
    - R15b: PASS = <= 1.34 / 5.42 / 3.69. Prediction (Cloud3): route + 0.03 / 0.35 / 0.54 = ~0.74 / 4.65 / 3.27.
    - Both are paired with the route (primary); R15b also with R15.
  - Harness lines: PERT.update(rmask='/Users/kamen/NewIdeas/data/dm_masks/<arm>', renorm=False); no t3; 36 windows;
    ~5 min each. Read: scripts/chat_arm_read.py <arm>.
- Thu 07:1x: AMENDMENT to the R15 / R15b pre-registration (before any run), per Fable C's audit
  (~/FableGod/fablec/AUDIT_1105.md, item 5 and section 3; the orchestrator asked me to check it first).
  - Order: R15 first, R15b after it.
  - Verdict lines for both:
    - PASS = under 1.34 / 5.42 / 3.69 on every domain AND within +0.5 points of the route's 0.71 / 4.30 / 2.73;
    - KILL = IFEval or multi over its line;
    - under the lines but more than +0.5 above the route = BETWEEN (neither adopted nor killed).
    - Cloud3's filed guard (within 1.3x of the route's) is reported beside it as information.
  - Expectations (Fable C, replacing Cloud3's "+0 to +0.1"):
    - R15: +0.2-0.4 points on chat, PASS ~70 %.
    - R15b: IFEval ~4.95 / multi ~3.6, a ~0.1-point margin on multi, PASS ~50 %.
  - Fable C's bytes read: the raw x0.82 / x0.64 cuts are worth x0.91 / x0.82 of the engine's chat bytes and +2-4 %
    chat tok/s at 24 GB (SSD is ~6 of ~70 ms per chat check).
  - My replay's raw cuts are x0.78 / 0.73 and x0.65 / 0.62.
- Thu 07:3x: SPLIT ARMS BUILT, BIT-CHECKED, PRE-REGISTERED (the orchestrator 07:2x; Fable C's audit F4 / test 1, its
  top-ranked lever).
  - Rule (audit_split_replay.py strict_row): the map's qom_0_live_bq form (QOM cf 0 + the bonus row gated by its next
    draft, 45 GB pool). Drafted rows whose cf >= 0.9 (split_0.9) / >= 0.99 (split_0.99) keep ranks 0-1 + union members
    only (no resident keep); every other row as QOM; loads unchanged.
  - Build: dm_oea_masks.py gains qom_<thr>_split<cf> (strict rows also touch only what they keep, as the audit's replay)
    and a tiles-per-kept-token stat.
    - split_0.9 = qom_0_split0.9_live_bq; split_0.99 = qom_0_split0.99_live_bq.
  - Bit-checks (scripts/split_check.py, data/dm_masks/check_split.log): OK.
    - qom_0_split1.1_live_bq == qom_0_live_bq bit for bit (36 / 36); ranks 0-1 never touched.
    - split_0.99 drops 910,285 experts on 5,691 rows; split_0.9 drops 1,382,387 on 7,339 rows.
    - LRU drift adds ~2,200 elsewhere.
  - Tiles per kept token [P, this builder, 45 GB pool]:
    - code 294.7 (QOM) -> 192.7 (x0.654) / 158.4 (x0.537);
    - IFEval 490 -> 452 / 405 (x0.92 / 0.83); multi 477 -> 449 / 413 (x0.94 / 0.86).
  - Cross-check vs the audit's replay: QOM's code tiles agree (294.7 vs 291), but its split ratios are lower (x0.60 /
    x0.47). Cause: the bonus-row convention. The audit's replay gives a check-final bonus row the next block's first
    draft cf (6 exact code rows, 3,319 rows >= 0.9); the map's _bq form leaves it exact (125 exact rows, 3,033 >= 0.9).
    The g arrays are identical. The _ball (DRAFT17) form would sit between (3,158 >= 0.9); one command if wanted.
  - Lines (the audit's test 1; code decisive, all-row flips):
    - PASS <= 1.34 %; BETWEEN 1.34-2.0 %; KILL > 2.0 %. The lever is killed if both arms are > 2.0 % on code, or if
      BigMoE's R8 group-count log shows QOM's groups <= 0.6x exact's.
    - Priors (audit [G]): split_0.9 1.2-1.5 %, split_0.99 0.9-1.2 %.
    - Chat runs as the control (lines 5.42 / 3.69, information).
  - Twin: qom_0_live_bq (measured 0.39 / 4.09 / 3.26).
  - Harness line: PERT.update(rmask='/Users/kamen/NewIdeas/data/dm_masks/<arm>', renorm=False); no t3; 36 windows.
    Read: scripts/chat_arm_read.py <arm>.
- Thu 07:3x: SPLIT ARMS, DRAFT17 _ball FORM = test 1's PRIMARY split arms (the orchestrator 07:3x: the engine's FAST keep
  QOM-KEEP gives every last row a cf). The bq pair (9b6451e) stays as the gentler bracket, run only if the window has
  time, else Friday.
  - Arms: qom_0_split0.9_live_ball, qom_0_split0.99_live_ball. Twin: qom_0_live_ball (QOM-KEEP + DRAFT17 at the 45 GB
    pool). Never measured on the map, so it is offered as the control arm. It differs from the measured qom_0_live_bq on
    3,575 rows: the no-next-draft bonus rows gated, plus LRU drift.
  - Bit-checks (data/dm_masks/check_split_ball.log): OK.
    - qom_0_split1.1_live_ball == qom_0_live_ball bit for bit (36 / 36); ranks 0-1 never touched.
    - split_0.99 drops 953,764 experts on 5,849 rows; split_0.9 drops 1,429,194 on 7,455 rows; ~2,400 added by LRU drift.
  - Code tiles per kept token, 45 GB pool [P]: QOM-KEEP 291.4 -> 182.9 (x0.628) / 147.9 (x0.508).
    - Side by side: bq x0.654 / 0.537; _ball x0.628 / 0.508; the audit's replay x0.601 / 0.469 (its QOM 291).
    - Chat tiles: IFEval 490 -> 452 / 405 (x0.92 / 0.83); multi 477 -> 449 / 413 (x0.94 / 0.86).
  - Lines unchanged (QUEUE Thu 07:3x; the audit's test 1): code PASS <= 1.34 %, BETWEEN 1.34-2.0 %, KILL > 2.0 % (the
    lever dies if both arms are > 2.0 %, or on R8 <= 0.6x). Chat runs as the control.
- Thu 08:5x: KAMEN'S DECISION on RELAX's ship mode (08:52, verbatim, via the orchestrator): "Ship after the check".
  - What ships, as the FAST track's mode (EXACT stays the default): B on code (Q1-UNION cf >= 0.9 + copy gate, HC8,
    comment rule) + E on chat (R1 0.5). EvalSuite's RESULT 8797f54: ship = BETWEEN (decision 4 PASS; bars fail
    thinly).
  - The check: EvalSuite's long-row drift re-check for B (PREREG a692f0f, ~1.7 h), Friday by day, ending before 15:00.
    PASS -> B on code ships; KILL -> the code part does not ship. E on chat is not affected. Nothing ships before the
    drift verdict.
  - HC8 code-only? As built, HC8 is process-global: Leap's hc8r.hc8_install swaps the trunk GatedResidual mixers to
    8-bit QuantizedLinear in place (keep=False frees the bf16 originals: that is its -0.6 GB), and ref2's hc8_fast read
    serves every row, prefill included (BigMoE scripts/r57_screen.py: R57_HC8 = install + switch together).
    - Code-only would need keep=True (bf16 kept, the -0.6 GB lost) plus a per-request module swap and hc8_fast toggle
      at request boundaries. That is engine code (BigMoE / Leap), not built or tested.
    - My recommendation: HC8 global, and E8 (E + HC8, ~1.5 h) joins Friday's check. One engine state, the RAM saving
      kept, and chat gains HC8's fixed-cost cut too.
    - Fallback if E8 cannot fit before 15:00 or fails: the request-boundary swap, bit-checked so that each domain
      reproduces its tested tokens.
- Thu 08:5x: RELAX SHIP, THE E8 CASE DECIDED (the orchestrator asked for it in writing, so Friday has no open question).
  E8 confirmed: HC8 ships global, and E8 runs Friday right after the drift check, both before 15:00.
  - Drift PASS + E8 PASS -> ship the FAST mode as decided: B on code + E on chat, both with HC8.
  - Drift PASS + E8 FAIL -> FAST ships for CODE ONLY (B with HC8); chat stays on EXACT, the default. Chat's FAST follows
    once it can run E without HC8 (as tested), in one of two ways:
    - (i) if the shipped FAST mode picks a code / chat profile per launch, the chat profile simply does not install HC8
      (R57_HC8 off): no engine work, but E must then be the tested E-without-HC8 build;
    - (ii) if one process serves both domains, BigMoE's request-boundary swap (keep=True + module swap + hc8_fast toggle),
      bit-checked so each domain reproduces its tested tokens.
> (a quote from the project owner, kept private)
    was never gated by it. An E8 fail is an engineering-scope problem (HC8 on chat), not a fail of the tested E, so it
    should not hold back the code part that passed its own check. Chat does not ship anything untested.
    Note: a process with HC8 installed is not exact, so "chat on EXACT" means chat uses the EXACT launch (no HC8), not
    the FAST process.
  - Drift KILL -> the code part does not ship. E on chat ships only if E8 PASSES (HC8 global) or as (i) / (ii) above.
  - Nothing ships before the drift verdict. Kamen can override any of this via the orchestrator.
- Thu 10:3x: MOONSHOTS RE-DERIVED BOTH OPEN SWEEP v1.3 NUMBERS (M-42 / M-43, cd5f64b).
  - Rank 5 bytes (its own M-37 replay with split thresholds): 52.8 / 62.0 / 64.2 MB vs my 53.9 / 62.4 / 64.0; chat
    x0.71 / 0.73, identical. Checks within 0-3 %; far stand-ins put chat 3-9 % higher at the same ratio.
  - KL per unit of output error:
    - My formula with its own per-row KL: 3.53x / 1.86x / 2.43x.
    - Its msb2_judge (W4 / MSB-3 / MSB-2 rebuilt from the original 4-bit safetensors, plus Fit12's store) puts MSB-2 at
      4.98-5.10x MSB-3's output error, so the assumed 5 holds.
    - Pooled by error (not layer means), the store is 1.30 / 1.29 (code, ranks 0-1 / 2-9), 1.82 / 1.79 (IFEval) and
      1.78 / 1.72 (multi).
    - The direct ratio with no T01 / E29 model is 3.18x / 1.65x / 2.34x.
    -> "~1.7-3.2x (re-derived by MoonShots)". Code is noisy (+-1.3); IFEval / multi clear 1 by > 2 SE.
    - Caveat: errors are pooled over all answer rows, while the arms touch gated rows' kept experts (assumed
      proportional).
  - SWEEP v1.3 rank 5 and item 3 updated; the findings note too.
- Thu 18:5x: SPLIT ARMS READ (test 1, code windows run 10:31-10:33; chat control Friday): code PASS on both
  (pre-registered Thu 07:3x). Breakthrough's read (2da53f2) gives the same flips.
  - split_0.99_ball: 0.92 % (prior 0.9-1.2), mean KL 0.0079; vs the twin qom_0_live_ball (0.51 %) McNemar 23 / 9,
    p 0.02 (+0.41 points).
  - split_0.9_ball: 1.31 % (prior 1.2-1.5; the line is 1.34), mean KL 0.0133; vs the twin 37 / 10, p 1e-4 (+0.80).
  - Caveat: split_0.9's mean KL is over code's KL line (0.0082) while its flips pass. Its divergence is tail-heavy, so
    EvalSuite's same-scores test is the real gate.
  - The audit's PASS also needs R8 (BigMoE's group-count log: QOM's groups >= 0.85x exact's); BigMoE has the keepset
    split (Breakthrough 4963da6).
  - Tiles at the 45 GB pool [P]: x0.628 / x0.508 of QOM-KEEP.
  - Reader: chat_arm_read.py now handles code-only arms (+code).
- Thu 19:0x: TRIED housekeeping (the orchestrator 18:5x).
  - The LOSSY350 block that was appended by hand to TRIED.md (17:0x) now lives in a part, ~/TRIED/orchestrator.md (26
    lines). tried_merge.py's SESSIONS and TRIED's README list it, so every merge keeps it; re-merged, 26 / 26 present.
  - Six of those lines had been cut at their first physical line; they were rebuilt from their sources:
    - WALL-THIN / TIEFLOOR / FLIPMODEL joined from ~/FableGod/fablec/WALL_1105.md;
    - AUDIT-SPLIT / B2D given their date and source fields;
    - the TRIED 337 correction reshaped to the README's fields.
  - The 337 correction (Fable A, held out: energy outside the top 1536 directions 0.13-0.19, not 0.01-0.03) stands
    as its own line. Leap's original line is Leap's to amend.
- Thu 19:0x: NEW PRE-REGISTERED ARMS, as their authors wrote them (the orchestrator 18:5x). Lines are quoted from each
  author's file, written before any run. Several are on hold for Kamen's GPU yes (per their status lines).
  - Fable C (~/FableGod/fablec/build_1105/INSTALL.md sections 2 / 6; code arms, Friday).
    - Each arm = qom_0_split0.99_live_ball + one rule; tiles are vs the twin qom_0_live_ball, code / IFEval / multi:
      - K8: the resident keep only at ranks 2-7 (x0.539 / 0.768 / 0.782).
      - K6: only at ranks 2-5 (x0.449 / 0.605 / 0.613).
      - A (r0.7): rows with cf < 0.99 and reach < 0.7 strict.
      - A' (m0.9): rows with cf in [0.9, 0.99) strict in layers 12-35 (x0.572 / 0.876 / 0.903).
      - B: A + rows with a real draft cf >= 0.999 keep no routed expert in layers 8-35.
      - K4: a probe, run last (x0.356 / 0.431 / 0.435).
    - Lines: PASS = all-row code flips <= 1.34 % AND non-tie <= 0.87 %; KILL = over either; BETWEEN = all-row <= 1.34
      with non-tie 0.87-1.0. The code preset = the passing arm with the fewest tiles.
  - FAR never-wait tier 1 (~/FableGod/far/NEVERWAIT_TEST.md section 4).
    - PASS: nw1 (_c208) <= 1.34 % AND D <= +0.40 points -> build the engine form for the code preset.
    - Pool-conditional PASS: BETWEEN at _c208 but clean (D <= +0.15) at the published pool.
    - KILL: nw1 > 2.0 % or D > +0.80 at _c208 AND not clean at the published pool.
    - The stack split0.99 + nw1 PASS <= 1.34 %; nw1r1 is the fallback (PASS <= 1.34 % and D <= +0.25; KILL D > +0.60).
  - Fable A T1 input-sparsity arms (~/FableGod/lossy350a/build_1105/INSTALL.md; DM_XS=1):
    - Arms: xs64 / xs48 / xs40 / xs32, standalone and stacked on qom_0_live_ball.
    - Identity: xs64 has 0 flips.
    - Standalone FITS: non-tie <= 0.66 / 1.03 / 0.72 %.
    - Stacked PASS: all-row <= 1.34 / 5.42 / 3.69 AND non-tie <= 0.87 / 2.22 / 1.63.
    - Lever dead: standalone xs32 non-tie code > 1.03 % or xs48 > 0.21 %; stacked xs32 > +1.2 all-row code points or
      xs48 > +0.25.
  - GOAL INERT (~/FableGod/goal/build_1105/INSTALL.md section 5; engine arms on the 12 HumanEval keys, BigMoE runs them;
    S = seconds per task, T = tokens per task, arm / control):
    - CD16 (comments + docstrings kept wholesale, 16 rows): PASS = median S <= 0.90 AND median T <= 1.05 AND verdicts
      equal on >= 11 / 12 with no net loss above 1; KILL = S >= 0.97, or T >= 1.15, or >= 2 tasks flip to fail. The
      replay predicts S 0.84.
    - C16: PASS S <= 0.93, T <= 1.05; KILL S >= 0.98, T >= 1.15, >= 2 flips.
    - G16 / D16 / the 32-row exact-engine arms / COPYRUN64: as in the file.
  - FULL (~/FableGod/full/build_1105/INSTALL.md sections 3-4).
    - T-B1 comment-free arm: PASS = comment tokens <= 2 % AND kept outside the copy run >= x1.15 AND tok/s >= +8 % AND
      no newly failing task; KILL = kept < x1.05, OR tok/s < +3 %, OR >= 2 new failures. Adoption needs Kamen's yes +
      decision 4.
    - T-A1 copy probe: PASS = confirmed >= 0.90 AND (DIRECT or SLOPE) <= 3.0 ms per copied token at W 32 or <= 1.8 at
      W 64; KILL = confirmed < 0.80, OR every DIRECT / SLOPE at both widths >= 4.5 ms.
  - ZERO FLAT-2 (~/FableGod/zero/LOSSY350_ZERO.md section 4; <= 10 min GPU after the native conversion, every owned lever
    on): PASS flat <= 30 ms (350 live on code at K >= 12); BETWEEN 30-36 (needs K >= 15-18); KILL flat > 36 ms.
  - Fable B FAST-TAX (att_single + gdn_inproj on in the FAST preset; ~/FableGod/fableb/LOSSY350_B.md section 4):
    predicted -2.4 to -2.9 ms per check. Its INSTALL and lines are not written yet (build_1105 has check_fasttax.py /
    fasttax_aba.sh); to be added when Fable B files them.
