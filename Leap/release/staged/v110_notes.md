**What is new**

- A 24 GB setting. `--mem 24` runs one lossy setting (with `--config lossy` or `chat`) that scored the same as the exact model on IFBench, GPQA Diamond and LiveCodeBench. It generates about 130 tokens per second on HumanEval, 91 on MBPP, 41 on math and 31 on chat. In our tests it stayed under 24 GB on prompts up to about 500 tokens and used more on longer ones: 28.2 GB at 30K tokens, and more past that (LIMITS.md). It is meant for a Mac with 36 GB or more; we have only tested it on our 64 GB MacBook Pro.
- `chat.py`, a chat in the terminal. It keeps the model loaded between messages and shows each answer's speed, first token and memory. A paste is one message, and `/file` sends a file with your message. At 32 and 24 GB it prefills long prompts layer-major, for a 2.4-3.8x faster first token and 0.3-3.2 GB more memory.
- `serve.py`, a local server for chat apps on your Mac, in the OpenAI and Anthropic API formats, with streaming. It is text only for now: it does not handle tool calls yet, and every request prefills its whole prompt.
- An idle chat or server now uses about 1.5 % of one CPU core instead of a whole core. Speed and results are unchanged.
- The 12 GB and 24 GB paper, "A 125B mixture-of-experts model in 32, 24 and 12 GB of memory on one Mac", linked from the README.

**What was wrong**

- run.py's help text and the papers called the layer-major prefill (PF-A) exact. That holds on the 30K-token prompt we checked, but not when a prompt's last 1,024-token chunk is short: with 115 tokens in the last chunk the results differ from the first check and the answer from its 36th token, while with 300 tokens they are identical. The boundary is about 200 tokens. In v1.0.1 PF-A was off unless you turned it on; in v1.1.0 `chat.py` turns it on at 32 and 24 GB.

**What is correct now**

- INSTALL.md, LIMITS.md, `chat.py`, run.py's help text and the papers say that PF-A gives the default prefill's results except when a prompt runs less than about 200 tokens past a multiple of 1,024, where the arithmetic rounds differently and an answer can differ. `R57_PFLM=0` turns it off in `chat.py`.
- The papers also give the cause of PF-B's difference at 30K tokens and a fix that is tested in a research copy but not yet in the package, and two formatting slips are fixed.

The drafter files stay on the v1.0.0 release.
