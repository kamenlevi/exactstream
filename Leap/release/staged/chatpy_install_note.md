# Staged: INSTALL.md lines for chat.py, used only if Kamen ships chat.py (orchestrator Tue 14:2x)

chat.py (local, uncommitted) uses PF-A, the layer-major prefill, at `--mem 32` and `--mem 24`. Its answers are identical
to the default prefill's, and its first token is 2.4-3.8x faster on long prompts. If chat.py ships, INSTALL.md gains this
line under the 24 GB setting:

  On a 36 GB Mac, for conversations near 30K tokens, start chat with `R57_PFLM=0`
  (`R57_PFLM=0 .venv/bin/python chat.py --mem 24`): it saves 2.4 GB of peak memory, at about 2.7x the first-token time.

Source: Leap data/tue_afternoon_1110.md, section 4. At 29,988 tokens with --mem 24, PF-A peaks at 30.60 GB with a 71.73 s
first token; the default prefill peaks at 28.17 GB with a 194.95 s first token.
