**What was wrong**

- The papers, and the help text of run.py's --env option, said the faster prefill form PF-B gives exactly the same results as the default prefill. That had only been checked on prompts up to 7.9K tokens. On a 30K-token prompt PF-B's results differ: its logits differ from the first check and its text from the 40th token. PF-B is a research switch that is off unless you turn it on, so nothing that runs by default changes.
- The papers did not say that very long prompts go over the memory lines, and one paper said long prompts run without the chat prefill path (PF-D). They run with it.

**What is correct now**

- The papers say PF-B is shown exact only up to 7.9K tokens, and that the cause of the difference at 30K is not found yet.
- The other two prefill paths are exact on long prompts too. The layer-major prefill (PF-A) gives bit-identical results to the default prefill on a 30K-token prompt. PF-D, on in `chat` at 45 GB, gives identical results on 10K and 30K-token prompts.
- The papers and LIMITS.md give the memory on long prompts: `chat` at 45 GB reaches 47.5 GB on 10K tokens and 48.6 GB on 30K, and `chat` at 32 GB reaches 34.0 and 35.5 GB.
- run.py's help text no longer calls PF-B exact.

Everything else is the same as v1.0.0. The drafter files stay on the v1.0.0 release.
