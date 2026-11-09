# Known limits

Everything was measured on one MacBook Pro with an M5 Pro (18-core CPU, 20-core GPU), 64 GB of memory and a 2 TB SSD,
on macOS 26.5 (25F71).

- MLX is pinned to 0.32.2. Its sorted gather_qmm leaves rows unwritten past 32,767 sorted rows when the row count is
  not a multiple of 64 (MLX issue #4632); the engine pads to 64. 0.32.4 fixes the bug but changes tokens against our
  exact references.
- First token on short prompts at 45 GB takes 2.7-4.6 s (2.26 s on `chat`). Strata (MIT; github.com/Niko1221/Strata),
  which we ran on the same MacBook Pro and prompts, takes 0.5-1.0 s, so up to 256 generated tokens it wins or ties every
  whole-request number; our decode is ahead on code and math, about level on chat, and behind on exact chat.
- The chat prefill path (PF-D) is on for `chat` at 45 GB only and was measured on chat-length prompts (37-101 tokens).
- Lifetime peaks go above 45.0 GB on long prompts (45.1-46.4 GB on 7.9K tokens).
- At 32 GB, `exact` was measured on chat (IFEval) only, and `lossy` has no code or math rows of its own.
- The lossy configs are gated on the named benchmarks only. On a multilingual chat set, M1-B (the `chat` configs' lossy
  part) fails our own blunder check, which is information only.
- Tested on one MacBook Pro, with two IFEval prompts per config in the clean-clone test; other Macs and macOS versions are
  untested.
