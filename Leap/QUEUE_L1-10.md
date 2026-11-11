# QUEUE L1-10, copy-span rows: does prompt lookup already capture the copy lever? Leap, Sat 2026-10-24 02:48 (CPU)
Method and line are NewIdeas' (QUEUE.md): the tokens in copies >= 16 that lookup drafting does NOT already accept;
kill if < 5 % of agentic output.

**How.** scripts/copyspan_lookup.py:
- Kamen's sessions rendered exactly as Chat1000's fshare.py: the 125B's output format and tokenizer, the same session
  sample, the same copy-span finder (verbatim spans >= 16 tokens found earlier in the context).
- Prompt-lookup drafting simulated as spec125's default:
  - the continuation after the most recent earlier occurrence of the longest 4/3/2-token suffix;
  - up to K = 5 drafts per pass, greedy acceptance;
  - a pass emits accepted + 1.
- The MTP cross-check (lookup used only when MTP's first token agrees) modelled as an 85 % pass rate.

**Result (80 sessions, 6.0 M output tokens): NOT KILLED, narrowly.**

| | no cross-check | cross-check 0.85 |
|---|---|---|
| copy spans >= 16, share of output | 25.6 % (fshare: 24.7 %) | 25.6 % |
| copy-span tokens accepted from lookup drafts | 77.5 % | 74.8 % |
| **copy tokens NOT accepted by lookup, share of output** | **5.8 %** | **6.5 %** |
| passes inside copy spans (lookup-only decoder) | 12.1 % of passes | 13.0 % |
| one pass per span instead: passes saved | 10.7 % | 11.6 % |

**What it means.**
- Lookup already finds most copies. What it leaves is mostly its width cap: 5 drafts + 1 per pass, ~4.1 tokens per
  pass inside copies.
  - With perfect lookup, 1/6 of span tokens are still a pass's own token: ~4.3 % of output.
- So the lever is "verify a long copy in one wide pass", not "find copies lookup misses".
- Size, in the real decoder (MTP ~3.70 per pass on the rest, fixed counting):
  - copies cost ~0.063 passes per output token of 0.264 in all;
  - one pass per span saves ~21 % of passes, i.e. up to ~1.27x on agentic.
  - This holds only if a span pass costs like a normal pass. A span is ~36 tokens on average; as plain verify rows
    it costs more, as prefill-like rows less (Chat1000: ~8x cheaper per row at wide passes).
- Simplest real test (engine, not mine to change): let spec125's lookup draft run wider than max_draft = 5 when its
  match is long (its `lookup_width`). Measure agentic tok/s at widths 5 / 16 / 32 on the 12 turns. Owner: BigMoE /
  SpecDecode via the orchestrator, or a copy in ~/Leap.

## Stage 2 (Sat 04:52, CPU): wider lookup drafts. CLOSED as a 350 lever; +3-4 % exact tweak at most
No kill line was pre-registered for this stage; reported as measured. scripts/copyspan_width.py, same 80 sessions
(5.43 M value / text tokens, copies >= 16 = 28.6 %), same lookup rule, plus:
- widths: fixed K = 5 / 8 / 16 / 32, or adaptive (5 drafts, widened to W while the previous pass accepted every lookup
  draft, i.e. inside a copy run);
- lookup "L" = longest match (SAM-decoding style, published): of the recent occurrences of the last 8 tokens, draft
  from the one whose match reaches furthest back; else spec125's most-recent 4/3/2-gram rule.
- Pass cost, two models:
  - SSD-streamed C192: a + b x expert union of the rows (Fit12's 4-row 94 ms and 8-row 149 ms passes);
  - in RAM: 15 ms + 2.9 ms per row (today's exact 3c expert kernel, ~6 us per (row, rank) pair whatever the union;
    Chat1000 REPORT 15:19).

| policy (rows <= 9 run on today's engine) | copy tok/pass | ms per copy token, SSD | in RAM |
|---|---|---|---|
| fixed 5 (spec125 today) | 4.10 | 1.000 | 1.000 |
| adaptive 5->8 | 4.80 | 0.977 | 0.958 |
| fixed 16 / 32 | 7.29 / 9.08 | 1.072 / 1.308 | 1.044 / 1.372 |
| L fixed 5 | 4.46 | 0.928 | 0.925 |
| **L adaptive 5->8** | 5.41 | **0.891** | **0.869** |
| L adaptive 5->16 | 6.70 | 0.878 | 0.862 |
| L fixed 32 | 9.74 | 1.264 | 1.322 |

- Rows cost too much for wide copy passes to pay. In RAM every row costs ~2.9 ms of expert work; on SSD the union grows as
  rows^0.72. Deep lookup drafts are right too rarely (fixed 32: 9.7 tokens for 29 rows).
- What is left: longest-match lookup + run-aware widening to 8 drafts. That is -11 to -13 % time per copy token, and
  copies are ~29 % of agentic time, so **x1.03-1.04 on agentic**, exact, on today's engine (9 rows).
  - A small tweak for BigMoE / SpecDecode's lookup, not a 350 lever.
- Pointer copies (a copied token with no row) are the only form of the F lever that escapes the row cost. Chat1000
  bounds them at F 0.13-0.17, and they are not exact decoding (the model never checks the copy).
