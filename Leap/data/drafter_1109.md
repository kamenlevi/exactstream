# Tonight's 32 GB drafter test: DFlash2 vs the MTP head on one code request (Tue 10 Nov 00:06-00:07)

One realistic code request through chat.py (the unchanged loop) at 32 GB (chat-32). Only the drafter is forced:
DFlash2 (`--drafter dfl`) against the 125B's MTP head (`--drafter mtp`). Each run stops at the 2,048-token cap, one run
each, back to back. The request asks for a CSV-report command-line tool with argparse, logging, dataclasses, a matplotlib
chart and a pytest file (116 prompt tokens with the template; text in bench_code.sh). Asked by the orchestrator at 23:2x on
Kamen's yes. The 45 GB pair stayed out.

Conditions:
- Kamen idle (36 min at the start) and Chrome closed.
- Ran after Breakthrough's q1u_0_live arm (ended 00:01:05) and Fit12's S1p (00:01:48-00:04:14), so nothing else ran.
- Share lock 'MEM 38 GB' from 00:05:26.
- NO-SWAP 41 GB at the start; swap flat at 5,618 MB.

| drafter | whole decode tok/s | kept / check | checks | cycle ms | check + commit ms | draft ms | misses / check | SSD per kept token | first token | lifetime peak | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DFlash2 (dfl, type code) | **102.4** | 6.14 | 334 | 59.5 | 52.3 | 6.3 | 35 | 9.1 MB | 4.03 s | 31.84 GB | 43 s |
| MTP head (mtp, type chat) | **85.3** | 6.02 | 340 | 67.9 | 47.1 | 20.4 | 32 | 9.5 MB | 3.99 s | 31.60 GB | 47 s |

Cycle, check + commit, draft, misses and SSD are per-check medians from the loop's own line. Wall is the whole process,
including load.

## Findings

1. **DFlash2 is 1.20x the MTP head on this code request at 32 GB:** 102.4 against 85.3 tok/s whole decode.
2. **The gap is the draft cost, not acceptance.** Both kept about 6 tokens per check (6.14 against 6.02). The MTP head's
   draft takes 20.4 ms per check against 6.3 ms, so its cycle is 67.9 ms against 59.5. Its check + commit is 5 ms shorter
   (47.1 against 52.3 ms), which wins back part of the gap.
3. **The prompt-copy drafter played no part.** It used 0 of 2,612 drafted positions on this fresh code (dfl run; g2 is
   on for code only). So this request compares the two drafters alone.
4. **Memory stayed under the 32 GB line** for this short prompt: 31.84 and 31.60 GB lifetime peak.
5. **Relaxed acceptance (chat-32 is lossy)** kept 2 (dfl) and 30 (mtp) tokens that exact checking would have rejected.
6. **Against Kamen's chat session** (data/chat_session_1109.md):
   - His fresh Tetris on DFlash2 (D1) ran at 101.8 tok/s with 6.07 kept per check, the same as tonight's 102.4 at 6.14.
   - His code on the MTP head ran at 52.2 (B1, cold pools right after a benchmark) and 76.8 tok/s (C1). Tonight's mtp
     run reached 85.3.
7. **Both answers were plausible Python** (a module docstring, then the tool) and were cut by the cap mid-file. Quality
   was not scored.

## Limits

- One run per drafter on one request.
- The runs were back to back with dfl first. A second run can gain from warm caches, which favours mtp, so the true gap
  may be a little larger.
- Shared lock with nothing else running. Fans on the night profile.

## Files (data/drafter_1109/)

- summary.txt: the bench's own lines.
- run_32_dfl.out and run_32_mtp.out: chat.py's screen output, with the answers.
- chat_1110_000607.log and .jsonl: the engine log and per-answer record for the dfl run.
- chat_1110_000650.log and .jsonl: the same for the mtp run.
- bench_code.sh: the bench.
- runner_1109b.sh: the night runner, with its start conditions.
