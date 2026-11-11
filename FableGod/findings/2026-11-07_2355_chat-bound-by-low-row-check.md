# Chat speed on the 45 GB engine is bound by the low-row check's fixed cost, not by verify width or the SSD

Sat 7 Nov 2026, 23:5x. Measured by BigMoE (N45, 45 GB, C 256, 12 IFEval prompts, quiet). Re-read per check by NewIdeas
(~/NewIdeas/scripts/chat/chat_rows.py, data/firstcheck/chat_rows_n45.txt).

- Today: chat (12 IFEval) decodes at 39.9 tok/s exact and 45.0 tok/s in the lossy config (whole decode).
- Chat checks are already narrow. The chat drafter (an MTP head that stops drafting below 0.3 confidence) sends ~3
  rows per check (median 2-3). 2.5 of them are kept (81 %). So there are few wasted rows, and verifying fewer rows
  cannot help much.
- The check itself is the wall:
  - It costs ~51 ms (exact) / ~40 ms (lossy) even at 1-2 rows and with <= 10 expert misses.
  - The memory floor of a 1-2-row forward is ~15-20 ms: ~2.7 GB of dense weights plus ~1-2 GB of experts at
    ~295 GB/s. The narrow kernels at 1-2 rows are latency-bound, not bandwidth-bound.
- SSD misses add ~4 ms at 1-2 rows, ~10 ms at 3-4 rows and ~22 ms at 5+ rows. They are secondary.
- Drafting costs ~2.6 ms per drafted row (the head runs serially).
- For comparison: math checks verify ~9 rows and keep 6.6; code keeps ~11 of 16.
- What 150 tok/s on chat would need: a ~25 ms cycle with ~3.75 tokens kept per check. That means the low-row check
  near its floor (about 2x faster) AND about 1.5x higher acceptance on chat.
- Status: the chat plan ranks the levers (lossy kernels on chat after an exactness bisect, low-row dense kernels,
  drafter speed, a trained chat head, top-8 experts, a chat expert-skipping rule). By Wednesday, ~55-65 tok/s
  (lossy) is the honest estimate [P].
- Lesson: for low-acceptance workloads, measure the check's cost at its real row count before optimizing verify
  width or SSD traffic.
