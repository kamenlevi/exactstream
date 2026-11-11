> (a quote from the project owner, kept private)

The local chat.py at `--mem 32` now sets R57_PFLM=1 (PF-A), the layer-major prefill that last night's exact check found
bitwise-equal to the default at 30K tokens (data/prefill_1109.md). `R57_PFLM=0` in the shell turns it off.
configs/*-32.env and run.py are unchanged. chat.py stays local and uncommitted until Kamen decides on shipping it.

Check, Tue 11:08-11:10:
- Two chat.py processes, chat-32, at most 256 tokens: PF-A (its new default) and R57_PFLM=0 (the default prefill).
- Each got a short prompt, then /clear, then a long one: ~24K characters of a paper ending "Summarize the main findings
  of the document above in five bullet points.", so the long prefill takes PF-A's multi-chunk path.
- Share lock 'MEM 36 GB', nothing else running, NO-SWAP 51 GB, swap flat.

| prompt | answers | first token: PF-A / default | prefill rate | lifetime peak: PF-A / default | whole decode |
|---|---|---|---|---|---|
| short (24 tokens, one chunk: PF-A not engaged) | identical, 82 / 82 tokens | 2.07 / 2.07 s | - | 31.23 / 31.23 GB | 39.8 / 39.1 tok/s |
| long (6,980 tokens) | identical, 256 / 256 tokens | **17.12 / 54.06 s (3.16x)** | 408 / 129 tok/s | 34.34 / 33.85 GB (+0.48) | 40.9 / 42.2 tok/s |

- The default run's 129 tok/s prefill is last night's default rate (119-127). The PF-A run's 408 tok/s is PF-A's (437-440
  at 10K / 30K). So PF-A ran in the PF-A process only; the engine's log line does not name the prefill path.
- Both chat-32 peaks on the long prompt are over the 32 GB line, the default too, as found last night (data/chat_session_1109.md).
- Files (data/pfa_chat_1110/): summary.txt, run_pfachat.sh, pfa.jsonl and default.jsonl (chat.py's per-answer records:
  prompt, answer, P, pfs, tps, peak).
