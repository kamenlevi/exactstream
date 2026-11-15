# Engine spec: relaxed acceptance in comments / docstrings (LEVERS_A_MOONSHOTS.md H / I), for BigMoE

MoonShots, Sun 15 Nov 2026 01:5x. The rule is LOSSY and needs EvalSuite's CODE-ONLY gate before it ships (Kamen's call).
Its purpose is to never pay T16's +1.9 ms per cycle again. BigMoE attributed that cost to a host-side span scan
(BigMoE HANDOVER, SAT 14:37: "InertScan host work likely").

## 1. The span state: where it lives and when it updates
- **Code:** `week98/span_tracker.py`, class `SpanTracker`: a character state machine, 15 small fields (`__slots__`).
  - Test, every HE+ / MBPP answer in the BANK: recall 99.32 % of in-span rows, precision 100 %, 0 false in-span rows
    in code (7,691), ordinary strings (115) and prose (481).
  - Speed: **0.87 us per token** in Python.
- **One tracker per request,** created at the assistant turn's first answer token (prose mode; it follows ``` fences).
  It lives with the request's decode state, next to the conversation-cache entry.
  - The conversation cache restores to a checkpoint, so store the tracker's fields with each checkpoint (a tuple of
    ~15 scalars).
  - Or replay the accepted tokens since the checkpoint (1 us per token).
- **Token text table:** `tokenizer.decode([id])` for every id, built once at start-up (248,077 strings, ~5 MB).
  Partial UTF-8 tokens decode to U+FFFD, which the lexer treats as an ordinary character.
- **Updates:** after each check, `update(t)` for every ACCEPTED token in order, the bonus token included. ~30 us per
  32-token check.
- **Per drafted row:** while the GPU runs the check, the CPU copies the tracker and steps it through the block's drafted
  tokens (~26 us per 30 rows). That gives two flags per drafted row j:
  - `span_before[j]`: in span after the prefix up to row j's token;
  - `span_after[j]`: still in span after row j's own drafted token.
  This is off the critical path.
- **Optional O(1) form for C / Metal:** precompute `T[token_id, state] -> state` over the reachable state set (a few
  dozen states), an int16 table of ~248K x S.

## 2. The acceptance rule on the check's own logits (no extra pass)
At drafted row j, the drafted token d_j, and the 125B's verify logits L_j (already computed by the check):
1. If d_j == argmax L_j: accept (exact path, unchanged).
2. Else, if `relax_k > 0` AND `span_before[j]` AND `span_after[j]` AND rank_j <= relax_k - 1, where
   rank_j = #{v : L_j[v] > L_j[d_j]} (strict, so ties count in the drafter's favour):
   ACCEPT d_j (relaxed), count it, and continue to row j + 1. The verify already conditioned row j + 1 on d_j.
3. Else: reject at j. The bonus token is argmax L_j, as today.

- **Cost:** gather L_j[d_j] (32 values), then one compare-and-count over 32 x 248K logits (~8M compares, ~0.1-0.3 ms on
  the GPU). Run it only for rows that fail step 1 and are in span, or for all 32 rows in one batched call.
- **`span_after` is a must-have guard.** A relaxed token must start AND end inside the span:
  - a drafted token that ends a comment ("\n" plus code) must not be relaxed;
  - nor one that closes a docstring early (a quote run).
  - Otherwise relaxation could inject code or close a docstring. H's estimates did not include this guard, so they are
    a slight overestimate.
- **Recommended extra guard:** don't relax where the 125B's own argmax would close the docstring (its text contains the
  closing quote run). The tracker can't see a closing quote before it arrives.
- **Family switch:** relaxation only on the code family (R57's family switch = code), never on chat or math prompts.
  `relax_k = 0` is today's exact behaviour.

## 3. The counters for EvalSuite's engagement rule
Per request (runs.json), next to `accepted`:
- `relaxed`: tokens accepted by step 2 (each one differs from the 125B's greedy);
- `relaxed_by_rank`: counts for rank 1 .. k-1;
- `span_rows`: drafted rows checked while in span;
- `relaxed_pos`: the answer offsets of the relaxed tokens, for audits and for diffing against the exact output.

EvalSuite's CODE-ONLY gate can then check:
- engagement > 0 on HE+ / MBPP;
- engagement == 0 on chat;
- deviation rate per 100 accepted tokens. H measured 16 % of in-span deep rows deviating on MBPP at k = 5, and 3-5 % on
  HE+. MBPP is the risk set.

## 4. Open before it can count
- HE+'s exact number: R(5) is OPEN at 96.67-97.82. MTP Head's HE+ cells with drafted tokens (06:00) settle it with
  week98/relax.py.
- EvalSuite's CODE-ONLY gate (both orders), with the guards above built in. Adoption is Kamen's call.
