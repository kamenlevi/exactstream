# Cloud Fable 5.1, task 14 (CLOUD_TASK_FASTMODE.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~21:2x
(The full ANSWER.md stays in the cloud session: 923 words, 3 of 8 Python runs, no web.)

- **The code fast mode:** the gate carries it; the comment loophole and HC8 ride along for free; R1 goes last. On this week's exact engine (our guess: 78 by day, 90 at night) the mode lands at:

  | mode | by day | night | deviation (diagnostic) |
  |---|---|---|---|
  | gate 0.999 (halved) | ~91 | ~104 | ~1.6 % |
  | gate 0.99 | ~96 | ~110 | ~1.6 % |
  | + R1 | ~100 | ~114 | ~2.8 % |

- **Two things don't stack:**
  - 32-row checks: the copy override's x1.211 loses to +32 ms of rows and a 1.55x union, 66 vs 78 tok/s.
  - The two-pass loses to the single-pass gate at 45 GB.
- **Chat:**
  - Kept is 2-2.5 because the 125B is unsure on 55-58 % of chat rows. Per-token agreement is ~0.6, so a block ends after ~1.5 rows plus the bonus.
  - Kept is flat in block length (2.50 / 2.47 / 2.43 at 16 / 8 / 6 rows), but cost isn't. **8-row checks on chat are exact, need no EvalSuite slot, and are worth +30 %:** more than any quality-defined chat lever.
  - Best configuration: MTP chain + 8-row blocks + R1, ~23 tok/s by day and ~27 at night.
  - R2 only if IFEval holds: its 7.1 % deviation is above the int4 twin's ~5 %, and there is one IFEval failure on record.
- **The MTP head's top-3:**
  - In exact mode only a tree can use it.
  - A leaf at the first miss gains nothing, because the bonus token already is the target's argmax.
  - A continuing branch needs ~2.4 GB of state per node.
  - In a fast mode, R2 cashes the same information without a tree.
- **Risk ranking:**
  1. R2 on chat
  2. the gate on chat (keep the gate code-only)
  3. gate + R1 on code

  The loophole and 8-row chat blocks are near-zero risk.
- **Test plan:**
  - Wednesday: A (gate alone, the control), B (the mode), C (+ R1).
  - Thursday: D (8-row chat, exact, by day), E (+ R1), F (+ R2), G (+ gate, optional).
  - Drop G, then C, if short. Ship B and E Friday if they pass.

## Orchestrator's check (21:2x)
- **The chat kept-vs-block numbers** (2.50 / 2.47 / 2.43) are cloud's arithmetic on our chat profile; MTP Head / BigMoE should measure them. The "leaf at the first miss gains nothing" argument is correct by the definition of the bonus token. It goes to Fable A as the first thing mission 28 must beat.
- **Routed:**
  - BigMoE: 8-row chat blocks as an exact lever in the combination's chat-switch group; the 32-row variant is predicted to lose (measure anyway).
  - EvalSuite and NewIdeas: cloud's A-G test order, to reconcile with PREREG_relax's arms. The gate stays code-only.
