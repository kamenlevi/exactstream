# Long-context runs at 64K and 128K tokens, Wed 11 Nov 2026 05:26-07:01 EET

**Why.** Claude Code sessions should run to 64-128K tokens. These runs measure the cold first token (the prefill),
the decode speed and the lifetime memory peak at 64K and 128K, at each memory setting. The results set the context
window Claude Code is told about and the default memory setting for it.

**Setup.**
- Prompt: the papers (MAIN, KERNELS, RELEASE, EVALUATION, DRAFTER), cut to 64,000 or 128,000 tokens. The 128K prompt
  repeats them from the start of MAIN. Then the request "Summarize the main findings of the document above in five
  bullet points." The prompts stay local, since they are draft texts (sha256 prefixes: 64K d66acef4453fa095, 128K
  b4844f3325306dd3).
- Engine: the agent clone's `run.py --config chat --mem M --max-tokens 64`.
- Prefill: PF-A (`R57_PFLM=1`) is the prefill chat.py and serve.py use at 24 and 32 GB. "Default" is run.py's leaner
  prefill. 45 GB ran as published (no PF-A there).
- Order: by predicted lifetime peak. An arm starts only when NO-SWAP is at least its predicted peak + 2 GB and the Mac
  is otherwise idle (runner: `long_ctx_1111/long_ctx.sh`).
- At 07:01 the runner was stopped by design: the cached 32 GB agent run took the Mac once no remaining arm fitted.
- Machine: MacBook Pro M5 Pro, 64 GB.

| setting, prefill | prompt | first token | prefill rate | whole decode (64 tokens) | lifetime peak (predicted) | swap growth |
|---|---|---|---|---|---|---|
| 24 GB, default | 64K | 472.2 s | 135.5 tok/s | 12.0 tok/s | 32.01 GB (32.2) | none |
| 24 GB, PF-A | 64K | 196.9 s | 325.0 tok/s | 10.8 tok/s | 37.90 GB (37.5) | none |
| 32 GB, PF-A | 64K | 185.8 s | 344.5 tok/s | 14.6 tok/s | 46.30 GB (44.7) | none |
| 45 GB, as published | 64K | 549.7 s | 116.4 tok/s | 18.1 tok/s | 52.61 GB (51.7) | none |
| 24 GB, default | 128K | 1,233.7 s (about 1,142 s without a 91 s heat pause) | 103.8 (about 112) tok/s | 7.6 tok/s | 39.24 GB (39.2) | none |
| 24 GB, PF-A | 128K | 660.6 s (about 580 s without an 81 s heat freeze) | 193.8 (about 221) tok/s | 7.0 tok/s | 51.56 GB (51.6) | none |
| 32 GB, default | 128K | 1,392.3 s | 91.9 tok/s | not measured (see 2) | 47.15 GB (42.7) | none |

Not run, because they did not fit in the free memory (the forecast was raised from each setting's measured 64K
peak): 32 GB PF-A at 128K (61.0 GB predicted) and 45 GB at 128K (59.0 GB predicted).

**Notes.**
1. **Heat pauses.** fanwatch paused the research jobs twice during these arms (`~/Library/Logs/fanwatch.log`).
   - 24 GB default 128K: PAUSE set at 05:59:47 and cleared at 06:01:28. The run's own line says it waited 91.3 s, so
     the clean prefill is about 1,142 s.
   - 24 GB PF-A 128K: PAUSE at 06:45:10, then a SIGSTOP freeze from 06:45:50 to 06:47:11 (81 s), inside the prefill.
     The clean prefill is about 580 s.
   - The "1 gaps" note on the PF-A arms is the engine's 15-s pass-gap counter (`engine/bigmoe/gated.py:130`).
     Layer-major passes over 64K tokens are longer than 15 s, so it fires without any freeze; the other PF-A runs had
     no freeze in fanwatch's log.
2. **The 32 GB 128K arm produced one token.** run.py's research loop at 32 GB has a fixed 600 s job budget
   (`engine/loop/r57_screen_rel1.py:311`, `HARD = 600`: no new decode check after 600 s of running time). This prefill
   alone took 1,392 s, so the decode stopped after the first token. The run summary's "6922.4 tok/s" is that one token
   over about 0 s; it is not a speed. The first token and peak are valid. chat.py (`chat.py:312`) and serve.py
   (`serve.py:157`) set `HARD = 1e12`, so neither stops a long answer. At 24 GB the budget is 1,200 s (chat-24.env
   `R57_HARD`), and the 24 GB 128K arm stayed within it.
3. **PF-A at long prompts.** At 24 GB, PF-A was 2.40x faster than the default prefill at 64K and about 1.97x at 128K
   (heat-corrected; 1.87x raw). It used 5.9 GB more memory at 64K and 12.3 GB more at 128K. The 1.1.0 text ("a
   2.4-3.8x faster first token, 0.3-3.2 GB more memory") was measured on prompts up to 30K.
4. **The 32 GB 128K peak beat its forecast by 4.5 GB** (47.15 GB against 42.7, which was a straight line from the 10K
   and 30K peaks). Its default prefill was also slower than the 24 GB default at 128K (1,392 s against about 1,142 s).
   This was not examined further.
5. **Decode slows down at long context.** These are 64-token answers only:
   - 7.0-7.6 tok/s at 128K and 10.8-12.0 at 64K at 24 GB;
   - 14.6 tok/s at 64K at 32 GB;
   - 18.1 tok/s at 64K at 45 GB.

   The same settings decode 31-130 tok/s on short prompts.

**For Claude Code.**
- Most turns are not cold. With the conversation cache, a turn prefills only its new tokens. In the cached 32 GB agent
  run (07:02), live hits prefilled about 546 new tokens on an 8.7K-token conversation and gave the first token in about
  6 s.
- The whole conversation is prefilled again only on a cache miss, such as resuming a long session after the server
  restarts. That takes about 3 minutes at 64K (24 or 32 GB, PF-A), about 10-11 minutes at 128K (24 GB, PF-A), and about
  9 minutes at 64K at 45 GB.
- Memory for a 128K window (`tools/claude_local.sh` tells Claude Code 131,072 tokens at every setting):
  - **24 GB:** a cold 128K prefill with serve.py's PF-A peaks at 51.6 GB, which fits this 64 GB Mac.
  - **32 GB** (the launcher's default): PF-A peaks at 46.3 GB at 64K, and 128K is forecast at about 61 GB, which would
    swap. The default prefill peaks at 47.2 GB at 128K.
  - **45 GB:** 52.6 GB at 64K; 128K is forecast at 59 GB, not measured.
- Proposal (Kamen's or the orchestrator's pick; launcher and prefill-path settings only, exact modes unchanged):
  - (a) At 32 GB, serve.py uses PF-A only when a cold prefill is at most about 64K new tokens, and the default prefill
    above that. PF-A's tokens equal the default prefill's except on a short last chunk, so the switch keeps or improves
    exactness. It needs a stand-in check plus one real 96K-128K run.
  - (b) Until then, tell Claude Code 65,536 tokens at `--mem 45` and at `--mem 32`, and 131,072 at `--mem 24`.
  - (c) An idea, untested: run PF-A in 32K-token segments, layer-major within each segment. Segment edges at multiples
    of 1,024 add no short chunks. This would hold PF-A's extra memory to about 3 GB at any length. It needs an
    exactness check and a timing check.

**For the 1.1.0 docs** (the orchestrator's call; flagged before the 14:00 go):
- LIMITS.md:22 and INSTALL.md:8 stop at 30K (28.2 GB at 24 GB). These runs add:
  - at 24 GB, 32.0 GB at 64K and 39.2 GB at 128K with the default prefill, and 37.9 / 51.6 GB with chat.py's PF-A;
  - at 32 GB, 46.3 GB at 64K with chat.py's PF-A and 47.2 GB at 128K with the default prefill;
  - at 45 GB, 52.6 GB at 64K.
- INSTALL.md:45's PF-A range (2.4-3.8x, +0.3-3.2 GB) holds up to 30K. At 24 GB it was 2.4x and +5.9 GB at 64K, and
  about 2x and +12.3 GB at 128K.

**Files.** `long_ctx_1111/`: the runner script, its log, arms.tsv, summary.txt and each arm's run.py output. The full
engine logs are in the clone's `out/cli_1111_*.log` (local).
