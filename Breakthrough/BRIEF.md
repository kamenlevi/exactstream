# Breakthrough: Qwen3.8-Flash-Next 125B at Q4 quality in a tiny RAM footprint, and ~1000 tok/s

Approved by Kamen on 2026-10-19 (~21:50). A separate long-term project. It must never slow the main track, whose goal is a stable public release soon.

## Why (Kamen's words)
"my end goal is getting this model - yes this model at q4 uncompromised quality to run on between 6 and 12 MAX gb of ram... it WILL need a new genuine breakthrough... if we crack how llms are ran we will be able to find the weak spot where such numbers become possible."

## Goals (Kamen's)
- **Small:** 6-12 GB of RAM, so any 16 GB Mac runs it without problems.
  - Lower bar: 45-50 tok/s on a Mac like Kamen's (M5 Pro, 64 GB, 2 TB SSD).
  - Target: 50-75 tok/s on the cheapest Mac mini you can buy now, which implies ~100-150 tok/s on Kamen's Mac (his stated lower bar there).
> (a quote from the project owner, kept private)
  - Ideal: 400-600 tok/s on Kamen's Mac.
- **Ultra-light:** at most 4 GB of RAM (runs on an 8 GB Mac), 30-50 tok/s.
- **Speed:** with the model comfortably in RAM on Kamen's Mac, ~1000 tok/s.
  - Kamen (answered 2026-10-19 evening, via the orchestrator): ~1000 tok/s means ONE chat (a single stream). Throughput across many parallel chats matters too, as a second target.
- **Steady** means sustained, e.g. 50-60 tok/s constantly, not peaks.

## Quality bar (Kamen's)
"if they are really similar in intelligence, then there isn't such a problem. i mean if answers are drastically different and the model thinks in a different way entirely for everything, then that might be a problem."
Working definition (orchestrator's, approved in chat): the same scores within noise on a small suite (GSM8K, coding, general knowledge, long prompts), and roughly 90%+ identical final answers against today's Q4. The 3-bit tier plan sits at 94.4% (427 vs 431 of 450 GSM8K).

## Physics framing (orchestrator's estimate; check and refine it first)
- Today a token moves ~4.5 GB of weights: ~3.2 GB non-expert plus ~1.3 GB of experts (10 of 512 per layer x 48 layers x 2.77 MB), shared over ~3.5 tokens per speculative pass.
- In 6-12 GB most expert bytes come from the SSD: 14.6 GB/s on Kamen's Mac, likely 3-5 GB/s on a base Mac mini. Exact streaming tops out at ~20-30 tok/s here and single digits on a base mini, so the small goals need ~5-10x fewer bytes per token from slow storage.
- 1000 tok/s for one conversation: memory bandwidth (~300 GB/s) caps it at ~150-250 tok/s even fully in RAM, so ~4-5x fewer bytes per pass are needed. Many parallel streams are compute-bound instead (~25 TFLOPS), where 1000 tok/s in total is plausible.
- Measured today: a mid-size matvec costs bytes / ~300 GB/s + ~20 us per dependent kernel; a cold expert read has a ~0.34 ms floor; LRU at a 25% cache hits ~73%; experts are independent in raw coordinates (lossless coding caps at ~6-8%).

## Approach
- Research first: ~/.local/share/pacer/RULES.md "Research first". Search the principles and related fields, run a prior-art pass, and give each idea a "Prior art" line.
- Hunt for ideas that change the math (far fewer bytes per token at the quality bar), not engineering polish.
- Score every idea on four numbers: bytes per token (RAM and SSD), real RAM footprint (including OS file cache and Python), sustained tok/s, and quality (suite + agreement).

## Resources and rules
- Folder ~/Breakthrough (git): IDEAS.md, SCOREBOARD.md, research/ for briefs. Start from ~/MoonShots/IDEAS.md, ~/MoonShots/PRIOR_ART.md, ~/.local/share/pacer/briefs/, and MoonShots' handover note (HANDOVER_FROM_MOONSHOTS.md, arriving soon).
- GPU only in gaps and at night: `peer lock quiet --queue`, never ahead of main-track jobs. Usage: about 15-20% of each window; pacer session name `breakthrough` (priority med).
- Never change the main engines (~/BigMoE, ~/SpecDecode) directly. Hand ideas over as patches with a one-command A/B.
- RULES.md applies in full: pacing, locks, memory guard, disk floor 100 GB, no parallel fan-outs over 3, utilization reported in every RESULT.
- Send the orchestrator (session "AI/LLM project recommendations") a RESULT only when final. Posting anything publicly is Kamen's call.

## Open questions (bring them back to the orchestrator)
- Does ~1000 tok/s mean one conversation, or the total across many parallel streams?
- Which exact Mac mini counts as "the cheapest one you can buy now" (chip, RAM, SSD size)? Its SSD speed matters a lot.

## First steps
1. Read the notes and write the bytes-per-token budget for each goal: where every byte goes today, and what each goal requires.
2. A research pass on fundamental limits and on radical ideas that break them, with at most 3 helpers; briefs go in research/.
3. Pick 3 candidate directions, each with a cheap first experiment, and report them as one RESULT.
