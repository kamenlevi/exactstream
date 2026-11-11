# L3 result: DFlash2 fine-tuned on Kamen's agentic sessions, regeneration-free labels (Leap, Fri 2026-10-23 22:05, checks 23:33)
- Pre-registration: DFL2C_PREREG.md (18:46, commits 02664c2, 16e0eee). Bet assigned by NewIdeas at 18:24. The
  orchestrator OK'd the local run at 18:58.
- Runs, both under a quiet lock:
  - batch 1 (data, features, checks), 19:26-19:47;
  - batch 2 (training + eval), 21:46-22:01.
- Raw output: logs/job.out and data/dfl2_result_l3_*.txt.
- All L3 data (features, windows, fine-tuned weights) was deleted at 22:04 and 23:34. It never left this Mac and was
  never committed.

**Verdict: KILL (NewIdeas' line).**
- Held-out agentic tau8 went 2.86 -> 2.91 (+2 %). MTP K=7 gets 3.54 on the same tokens (x0.82).
- Predicted 3.35 (3.0-3.7). The result is below the predicted range.

| | original | fine-tuned |
|---|---|---|
| **held-out agentic tau8 (12 turns, window 2,048)** | 2.86 | **2.91** |
| first-token acceptance | .64 | .63 |
| P(first j right), j = 2..7 | .41 .31 .22 .17 .14 .11 | .43 .32 .24 .19 .15 .12 |
| hybrid vs MTP alone (3.54): cross-check / oracle (original's from L2b, same turns and window) | 3.37 / 3.86 | 3.44 / 3.87 |
| code guard tau8 (HumanEval+) | 6.15 | 6.04 |
| validation (held-out sessions of Kamen's text), on-path top-1 at depth 1 / 4 / 7 | .775 / .659 / .515 | .820 / .844 / .817 |

Per turn, original -> fine-tuned:
- Up: bash21 2.68 -> 3.23 (+21 %), text27 2.33 -> 2.78 (+19 %), bash17 +5 %, bash19 +4 %, edit00 +2 %.
- Flat: edit03, edit06.
- Down: write11 2.91 -> 2.59 (-11 %), edit09 -5 %, text23 / text25 / write14 -3 %.

**What happened.**
- The drafter learned its training distribution. On held-out sessions of the same kind, on-path accuracy rose at
  every depth, most at the deep ones (depth 7: .515 -> .817).
- It did not transfer to the 125B's own agentic text:
  - first-token acceptance is flat;
  - deeper depths gained 1-2 points.
- Most likely reason: the training text is Claude's, and the eval text is the 125B's own output.
  - The labels are the 125B's next token on Claude's contexts. The contexts stay Claude-shaped (wording, tool-call
    layout, code style).
  - Content seems to transfer. Bash turns gain (+4-21 %): the commands, paths and tools in Kamen's projects are the
    same whoever types them.
  - Style does not transfer. Write turns lose (-11 %): the drafter learned the 125B's guesses about Claude's code, not
    the 125B's own code.
- Data size was not what limited it: the validation gains are large.
- They are mostly real learning, not repeated content. Check 23:33, scripts/l3_checks.py:
  - 9.5 %, 9.4 % and 14.2 % of the 3 val windows' 32-token spans also occur in training.
  - In the first two this is ~580 spans each. That matches the identical system + tools preamble every window starts
    with, which holds no anchors. The third window has ~600 more.
  - Bound, if the overlap were assistant text spread evenly: memorising all of it would explain at most ~23 % of the
    depth-7 gain (.142 x (1 - .515) = .07 of .30).
  - An assistant-only split was not measured: the Mac was in others' quiet locks.

**Against the predictions.**
- On-path rate: predicted ~.70, got .748. OK.
- Validation depth 1: predicted ~.60 -> ~.70, got .775 -> .820.
  - The starting point was higher than I assumed.
  - The depth-1 gain was smaller than predicted; the deep-depth gains were much larger.
- Held-out agentic: predicted 3.35 (3.0-3.7), got 2.91. First token: predicted .72 (.68-.78), got .63. **Miss.**
- Code guard: predicted ~5.9 (5.5-6.2), got 6.04. OK; there is no forgetting worth the name.

**Deviations and process.**
- Only 14 % of the 256K tokens are assistant tokens, so there were 34,736 train anchors. The prereg assumed 60-80K.
- Batch 2's first start (20:47) was stopped by the swap guard at step 10. The cause was MLX's buffer cache.
  - Fix: cache capped at 2 GB, plus micro-batches with summed gradients (the same loss and gradient).
  - Peak 22 GB. Two epochs (320 steps) took 13.7 min.
- Checks:
  - forward vs the official path: PASS (argmax .991);
  - the original weights reproduce 2.86 / 6.15: PASS;
  - every fine-tuned tensor reaches the eval's drafter (it loads with strict=False): PASS (23:33).
    - 79 / 79 names and shapes match, and all 79 equal the file after loading.
    - 72 differ from the original; the other 7 are most likely norm weights whose small updates round away in bf16.
    - Mean relative change of the 2-D weights: 1.1 % (max 4.5 %).

**What it means.**
- Agentic DFlash2 from Claude's session text is dead. Agentic stays on MTP (as after L2b). DFlash2 stays on code and
  math, where the fine-tune would cost 2 %.
- The route left for an agentic block drafter is the published one: train on the 125B's own agentic outputs
  (regenerated, as in EAGLE-3 and NeMo practice).
  - The argument for it: in-distribution, these per-depth accuracies imply about +40 % in tau (product of the depth
    accuracies, 3.3 -> 4.6).
  - Applied to the 12 turns, that would be 2.86 -> ~4.0. This is a projection from another distribution, not a
    measurement.
  - Cost: the 125B generating ~36K+ agentic tokens on other sessions' contexts (~40 min of GPU with the prefills),
    the 125B prefill for features (~12 min), and the same 14-min fine-tune. All local.
- For 350: even ~4.0 is far from the ~5.1 tokens per pass that agentic 350 needs (A1 / L2b arithmetic).
