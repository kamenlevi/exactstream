# W24: which arm goes to the 24 GB card gate (pre-registered Mon 9 Nov 19:1x, before any W24 data)

Target (Kamen 18:5x via the orchestrator): 120 tok/s WHOLE DECODE on HumanEval, lifetime peak under 24.0 GB, lossy allowed,
same scores on the known benchmarks before it ships (EvalSuite's card gate: IFBench, GPQA Diamond, LiveCodeBench v6, both
orders, vs NAX6-N; ~10 h at 24 GB). Arms and code: scripts/w24.sh, data/r57/snap/w24 (commits f3caa26 .. 4911bca).

**Valid arm:** all three sets ran (12 HumanEval, 12 MBPP, 12 IFEval), FAST PATH on every set (tier_calls = tier_chunk = 0),
lifetime peak <= 24.0 GB on every set, no aborts, no cut rows. Rows with a heat freeze are counted and named in the report.

**Pick:** among valid arms, the one with the highest HumanEval whole decode (pooled). If the best QOM-rule arm (w12, w12pf,
w10, w8: optional uses served only when resident) is within 5 % of the U3 arm (w16u: optional uses never served), the QOM
arm goes: it drops fewer expert uses, so it is the smaller quality risk. If no arm is valid, the gate waits and the next
step is the fix the report names.

**Reported for every arm:** HumanEval and MBPP separately, plus IFEval (chat): whole decode / whole request / first token
median / lifetime peak; kept per check; cycle median; SSD MB per kept token; QOM drop rate (C 256: HE 11.4 %, MBPP 12.2 %);
heat freezes. The gap to 120 = 120 / (HumanEval whole decode) for the picked arm, with what closes it.

**Then:** scripts/w24_card_env.sh <arm> -> data/stack/card_w24_env.txt (+ R57_HARD=1200), committed; NewIdeas and EvalSuite
told; the orchestrator gets the first number by SendMessage.

# W24c: does the 01:25 gate switch from w13t? (pre-registered Mon 9 Nov 23:1x, before any W24c data)

Orders, both applied: the orchestrator 23:0x to BigMoE ("If it beats w13t in a same-run comparison beyond the 4.6 % noise
you measured, commit the new env before 01:25; otherwise w13t stays") and the orchestrator 23:1x via NewIdeas (>= 120 tok/s
HumanEval whole decode under 24 GB, NewIdeas' number check, EvalSuite's dry run of the new env, all by 01:20).
Arms (scripts/w24c.sh; env only on snapshot w24 = 016cb7f): w13c / w13d = w13t as gated (first and last), w14a / w14b =
140 base + 28 sidecar slots, 14 rows, R57_D7B=all (only the 4-bit DFlash2 drafter; the gated w13t also holds the bf16 one,
1.14 GB, which drafts only math / agent-typed prompts). Diagnostics after the decision passes: w13a (w13t + R57_D7B=all),
w13b (141 base slots, 13 rows + R57_D7B=all), MBPP + IFEval for w13c.

**Switch to the 14-row config only if ALL hold** (scripts/w24c_compare.py prints conditions 1-4 with their values):
1. HumanEval whole decode pooled over all 12 prompts of w14a AND w14b (sum n / sum n/tps) >= 120.0.
2. No HumanEval row of w14a / w14b has a non-empty bad field (a PAUSED row makes the claim unclean: no switch tonight).
3. Lifetime peak (max of mem[1], mem[-1], 10^9 bytes) < 24.000 GB in w14a and w14b, and in w14a's MBPP and IFEval sets.
4. The w14 pair beats the controls' mean (w13c, w13d) by more than 4.6 % on the prompts unpaused and uncut in all four.
5. w14a's MBPP and IFEval sets ran: no aborts, no cut rows, 0 measured checks over 300 ms in unpaused runs (the fast path).
6. NewIdeas' number check (its board, 'MON 23:1x GATE-SWITCH RULE') and EvalSuite's dry run of the new env pass by 01:20.
Flag only: whether the two controls agree within 2 % (thermal drift is named in the result otherwise).
If any fails, w13t runs as committed (data/stack/card_w24_env.txt at 8f76fc3) and a 120 config gets its own gate Tue night.
**Switch mechanics:** data/stack/card_w24c_env.txt (the w14a env, committed before the run for EvalSuite's dry run) is
copied over data/stack/card_w24_env.txt and committed by 01:20; EvalSuite's start script reads the committed file at 01:25.

**Amended Mon 23:2x, still before any W24c data (deadline and mechanics only; conditions 1-5 unchanged):** the gate moved
> (a quote from the project owner, kept private)
data/stack/card_w24c_env.txt PASSED 23:2x (8aecd3c, blob 5d2b765e), so condition 6 is NewIdeas' number check alone.
NO COPY (EvalSuite 23:2x): on a pass NewIdeas relays the verdict to EvalSuite by 02:05, and EvalSuite re-arms its
launcher on card_w24c_env.txt itself (a frozen copy of blob 5d2b765e). BOTH env files stay untouched until the gate ends:
any change makes the launcher wait. W24c's start-by moved from 00:50 to 01:15 (NewIdeas 23:12, the queue owner).

**OUTCOME + OVERRIDE (Mon 9 Nov 23:4x, after the W24c data; recorded at the orchestrator's request):** conditions 1-3 PASSED
(HumanEval pooled over w14a + w14b 126.11 >= 120.0; 0 bad rows; peak 23.808 < 24.000 GB; w14a's MBPP 23.691 and IFEval 23.635
GB). Condition 4 FAILED: the 14-row pair was +4.1 % over the same-run w13t controls, short of the > +4.6 % bar. As written, the
rule kept w13t. BigMoE sent the conflict to the orchestrator instead of picking; the orchestrator OVERRODE condition 4 and
decided to switch the 02:10 gate to the 14-row config (data/stack/card_w24c_env.txt), subject to NewIdeas' number check, with
NewIdeas relaying to EvalSuite. The reasons it gave, both measured in this run:
1. The gated w13t is not reliably under the 24 GB line: w13d peaked 24.006 GB (HumanEval/68) with tokens identical to w13c
   (23.916), so its footprint moves ~0.1 GB between identical runs. The 14-row config peaked 23.64-23.81 GB on all three sets.
2. The +4.6 % bar was the noise between two separate runs (W24 vs W24b). Inside this ABBA run the controls agree within 0.4 %
   (120.91 / 121.38) and so do the 14-row repeats (125.84 / 126.38), with tokens identical within each pair.
Published numbers come from the same-run pair, not a best-of: 125.8 / 126.4 tok/s (pooled 126.1), lifetime peak 23.81 GB.
Condition 5 as worded (0 checks over 300 ms on w14a's MBPP and IFEval): MBPP 0 of 314; IFEval 3-4 of 633, all the first or last
check of a prompt (the same-run w13t IFEval diagnostic shows whether that is normal for chat; it did not enter the decision).

# W24d: w13b vs the 14-row config for the 02:10 gate (pre-registered Mon 9 Nov 23:5x, before any W24d data)

Order (the orchestrator 23:5x, "yes, option B, with the 14-row config staying armed as the fallback"). W24c's diagnostic w13b
(141 base + 26 sidecar slots, 13 rows, R57_D7B=all) ran 129.9 tok/s at 23.84 GB in one sample after the ABBA block. W24d
(scripts/w24d.sh) runs a same-run ABBA on HumanEval: w14c, w13e, w13f, w14d. w14c / w14d = the 14-row config (140 + 28 slots,
14 rows, R57_D7B=all; = data/stack/card_w24c_env.txt); w13e / w13f = w13b. The 14-row config stays armed for the gate meanwhile.
**w13b replaces the 14-row config only if ALL hold:**
1. Its HumanEval whole decode pooled over w13e + w13f (all 12 prompts each; sum n / sum n/tps) beats the 14-row config's,
   pooled over w14c + w14d, by more than 1 %.
2. Every w13b run (w13e, w13f) has a lifetime peak (max of mem[1], mem[-1], 10^9 bytes) under 24.000 GB.
3. Zero bad rows: no non-empty bad field and no cut row in any of the four arms.
4. data/stack/card_w24d_env.txt (w13b), EvalSuite's dry run of it and NewIdeas' number check, all by 01:30.
Anything short of that: the 14-row config runs. Both configs' numbers are reported from this same run. If W24d cannot start
by 00:15, option A: no W24d, and the 14-row config runs.

**W24d OUTCOME (Mon 9 Nov 23:56):** w13b WINS. Pooled HumanEval w13b (w13e + w13f) 130.33 vs the 14-row config (w14c + w14d)
125.91 = +3.51 % (> +1 %); w13b peaks 23.819 / 23.781 GB (< 24.000); zero bad or cut rows in all four arms; EvalSuite's dry run
of card_w24d_env.txt passed (blob e06f495e, its amendment 12d). Remaining: NewIdeas' number check by 01:30; its FINAL word to
EvalSuite names the gated file. data/w24/w24d_verdict.txt.

**Gate config set (Tue 10 Nov 01:35; corrected 01:4x on EvalSuite's note):** the final word was the ORCHESTRATOR'S (01:33),
given in NewIdeas' place: NewIdeas was at its account limit until 02:00, so ITS NUMBER CHECK DID NOT RUN. This deviates from
condition 4 of the W24d rule (NewIdeas' check by 01:30) and came after that deadline; EvalSuite records the deviation on the
orchestrator's word in its PREREG amendment 12d (0f47fb7). In the check's place, EvalSuite re-derived the result from the
raw w1[34][cdef]h_100_runs.json and it matches data/w24/w24d_verdict.txt exactly. EvalSuite's launcher re-armed at 01:35:23 on
data/stack/card_w24d_env.txt (blob e06f495e, frozen copy), i.e. w13b; the card gate RUNS since 01:36:43 (first row: lifetime
peak 23.6 GB), MEM 24 GB until ~09:44 (central) or ~11:32 (slow), cutoff 15:00 (EvalSuite results/card_w24_start.log).
(The 01:35 version of this note said "NewIdeas' final word": wrong.)

# W24e: does a smaller prefill chunk keep w13b under 24 GB on long prompts? (pre-registered Tue 10 Nov 02:0x, before any W24e data)

Order (the orchestrator, ~02:0x): (A) is decided: the gated w13b's label states where it stays under 24 GB and the gate's
per-row peaks above that. [Label wording REPLACED Tue 08:4x on the orchestrator's recount, because GPQA and LCB6 are benchmark
prompts too and crossed: the 02:0x wording "under 24 GB on benchmark-length prompts" was too generous. (A) now reads: "under
24 GB on prompts up to about 560 tokens, up to 25.1 GB on longer ones" (the gate, all 40 processes: 22 over 24.0 GB, max
25.112 GB; first crossings 15 LCB6 + 7 GPQA-D, none IFBench; shortest prompt at a first crossing 561 tokens; the longest row
still under the line 592 tokens, 23.954 GB).] (B) = W24e (scripts/w24e.sh), after the gate, with NewIdeas: w13b exactly as gated
(data/stack/card_w24d_env.txt, blob e06f495e, its env line verbatim) plus R57_PFCHUNK = 1024 / 512 / 256 / 1024 again (pc1024,
pc512, pc256, pc1024b), on six long card-gate prompts (726 to 2,801 tokens: gpqad/recnTTKdBzfuoZ7w7, lcb6/abc394_g,
lcb6/abc400_g, lcb6/arc192_a, gpqad/rec6sE2CRtD4drtHg, lcb6/3705), CAP 128, R57_LOGHASH=1, one process per arm.
**A smaller chunk (512 or 256) carries the gate's label only if, on all six prompts, its tokens AND logits hashes equal pc1024's**,
and pc1024b equals pc1024 (determinism). Among the chunks that qualify, the package uses the LARGEST (the cheapest first token)
whose lifetime peak (max of mem[1], mem[-1]) is < 24.000 GB on all six prompts. If none qualifies, Wednesday ships (A).
Reported either way: per prompt and chunk size, the first token (the first-token cost) and the lifetime peak.

**W24e OUTCOME (Tue 10 Nov 08:53-09:00; data 8e515d5, data/w24/w24e_report.txt): NO smaller chunk qualifies, so Wednesday ships
(A): w13b as gated, R57_PFCHUNK = 1024.** Determinism holds (pc1024b == pc1024: tokens 6/6, logits hashes 6/6). pc512: tokens
0/6, hashes 0/6; pc256: tokens 1/6, hashes 0/6 (both FAIL the identity condition). Max lifetime peak over the six prompts
(726-2,801 tokens): 1024 25.025 / 24.961 GB, 512 24.345 GB, 256 23.954 GB (the only one under the line). First token per
prompt: 1024 5.6-14.6 s; 512 8.5-21.6 s (1.4-1.6x); 256 13.0-31.2 s (2.1-2.3x). A strictly-under-24 GB variant (256-row chunks)
changes the tokens, so it would need its own card gate. The card gate itself PASSED 08:41 on all 6 lines (EvalSuite
results/CARD_w24_DONE): IFBench 40.7 / 40.7 % vs 44.0 %, GPQA-D 47.0 / 47.0 % vs 44.0 %, LCB v6 29.8 / 32.8 % vs 32.1 %.
