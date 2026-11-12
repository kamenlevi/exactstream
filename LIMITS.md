# Known limits

Everything was measured on one MacBook Pro with an M5 Pro (18-core CPU, 20-core GPU), 64 GB of memory and a 2 TB SSD,
on macOS 26.5 (25F71).

- MLX is pinned to 0.32.2. Its sorted gather_qmm leaves rows unwritten past 32,767 sorted rows when the row count is
  not a multiple of 64 (MLX issue #4632); the engine pads to 64. 0.32.4 fixes the bug but changes tokens against our
  exact references.
- First token on short prompts at 45 GB takes 2.7-4.6 s (2.26 s on `chat`). Strata (MIT; github.com/Niko1221/Strata),
  which we ran on the same MacBook Pro and prompts, takes 0.5-1.0 s, so up to 256 generated tokens it wins or ties every
  whole-request number; our decode is ahead on code and math, about level on chat, and behind on exact chat.
- The chat prefill path (PF-D) is on for `chat` at 45 GB only. It was measured on chat-length prompts (37-101 tokens)
  and checked exact on 10K and 30K-token prompts, where it shortens the first token by only 3-4 % and adds about 2 GB
  to the lifetime peak.
- At 32 and 24 GB, `chat.py` and `serve.py` prefill long prompts layer-major (PF-A). Its tokens match the default
  prefill's except when a prompt's last 1,024-token chunk is under about 200 tokens: there the arithmetic rounds
  differently and an answer can differ. `R57_PFLM=0` in the shell turns it off (INSTALL.md).
- Lifetime peaks go above the memory line on long prompts. At 45 GB: 45.1-46.4 GB on 7.9K tokens, and `chat` as shipped
  (with PF-D) reaches 47.5 GB on 10K tokens and 48.6 GB on 30K. At 32 GB, `chat` reaches 34.0 GB on 10K tokens and
  35.5 GB on 30K.
- At 32 GB, `exact` was measured on chat (IFEval) only, and `lossy` has no code or math rows of its own.
- At 24 GB, prompts up to about 500 tokens stayed under 24 GB in our runs, with little room: this is a measurement, not a
  bound (a 448-token prompt peaked at 23.954 GB, a 520-token one at 24.01 GB). Longer ones take more: about 25.0 GB at
  3K tokens, measured on a simulated 36 GB Mac (memory locked away to leave the engine 25-27 GB), and on the 64 GB Mac
  with the default prefill 25.7 GB at 7K, 25.8 GB at 10K and 28.2 GB at 30K.
- Longer prompts take more. At 24 GB, 32.0 GB at 64K tokens and 39.2 GB at 128K with the default prefill, and 37.9 and
  51.6 GB with `chat.py`'s layer-major prefill, which is 2.4x faster at 64K and about 2x at 128K. At 32 GB, `chat.py`
  reaches 46.3 GB at 64K; at 45 GB, 52.6 GB.
- P2 (24 GB, prompts of 256 tokens or fewer) is lossy: on 12 IFEval prompts 2 of 12 answers equal the prefill path's.
  Its gate covered the benchmark sets it touches, IFBench and GPQA Diamond's 67 prompts of 256 tokens or fewer (the same
  scores within the gate's rule: -1.33 and -2.99 points, p 0.80 and 0.63); LiveCodeBench v6 has no prompt that short.
  Its speed was measured on 12 IFEval prompts only: first token 1.12 s (median) against 3.11 s without it.
- At 24 GB, on those 12 IFEval prompts, P2 gives the first token in 1.12 s against Strata's 1.46 s on the same MacBook Pro
  and prompts, and the whole request at 24.4 against Strata's 20.3-21.0 tok/s. On the other sets we have not timed P2:
  there, from the 1.1.0 runs, Strata gives the first token 1.9-2.5x faster, while our decode is ahead on code (3.3-5.0x).
- Coding agents were tested at 32 and 45 GB only. EvalSuite's Claude Code agent test passed with the conversation cache
  at 32 GB (two runs) and at 45 GB; without the cache, at 32 GB, it failed its tool-call probe line (27 of 30 against
  28). At 24 GB only a three-task smoke test ran. Codex was only smoke-tested (three tasks at 24 GB).
- The conversation cache is on at 45 and 32 GB and off at 24 GB, where it is unchecked. It holds up to 2.5 GB by default.
  In the agent tests the server's lifetime peak was 36.17 GB at 32 GB and 49.43 GB at 45 GB. Agent prompts at 24 GB
  take it over 24 GB (27.3 GB at 12.6K tokens), and without the cache every turn there prefills its whole prompt (about
  31 s to the first token at 12.6K tokens).
- `serve.py` runs one request at a time. Every turn in `chat.py` prefills the whole conversation.
- The lossy configs are gated on the named benchmarks only. On a multilingual chat set, M1-B (the `chat` configs' lossy
  part) fails our own blunder check, which is information only.
- Tested on one MacBook Pro, with two benchmark prompts per config in the clean-clone test (IFEval); other Macs and
  macOS versions are untested.
