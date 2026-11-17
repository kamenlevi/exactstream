# IDEAS_1117: value prediction of the 125B's state (MoonShots, BREAK98, Tue 17 Nov 07:3x-08:1x)

Lens: an idea from an unusual place. This one comes from CPU value prediction: last-value and context (FCM) predictors
with a confidence gate (Lipasti & Shen 1996; Sazeides & Smith 1997). Labels: [K] measured, [P] derived, [G] estimate.
Raw data: ~/MoonShots/week98/break98/ (svp_screen.py / .txt / .json, svp_oracle.py / .txt / .json). Inputs are
BigMoE's bank captures: 24 heplus24 + 100 mbppbank files, the files dd_tree reads. Per position: wd (the head's true
input state, 10,240 wide), tk (the 125B's top-32 next tokens) and ids.

## Bottom line
- **SESSION VALUE PREDICTION (SVP) = KILL, screened on CPU in 14 s on the real captures [K].**
  - Hit share: a drafted position's last 4-16 tokens recur at an earlier verified position only 9-12 % of the time.
  - At those hits, the retrieved TRUE state makes the 125B's own next-token call 61-63 % of the time. Plain n-gram copying
    of the next token (a killed family) scores the same 61-63 %; on generated rows the two are identical by construction.
- **The general bound is stronger.** Take the hindsight-best earlier state in the session: the max cosine to the true
  state over every verified position <= the anchor, i.e. what a perfect retrieval key could fetch.
  - It sits 0.67-0.76 rad from the true state (cos 0.71-0.78).
  - Only 3-6 % of drafted positions come within 0.40 rad, the bank proxy's ~88-92 % band [K].
  - On the bank proxy's scale (0.34 rad -> 91.9, 0.525 -> 67.1, 0.63 -> 49.6), that extrapolates to ~40 % [P], far
    under the head's own carry.
  - So no retrieval key (n-gram, learned, kNN) can deliver a drafted position's state from the session, and lever 3
    (a cross-corpus kNN of true states) inherits the verdict.
- **The fact behind it [K]:** the 125B's state is a long-context quantity.
  - 16 identical tokens read in the PROMPT vs written in the ANSWER give states with R2 -0.40 to each other, farther
    apart than the file's mean state.
  - At 8-16-token answer-side repeats, R2 is 0.07-0.22.
  - With ASSIM / ERRMEM / MEANPULL's KILL (session correction from verified rows: all 8 arms lower), this closes both
    session-statistics routes: you can neither look the state up nor correct the carry toward it. The information has
    to be computed on the drafted tokens, which is the believed limit's own premise, now measured from this side.
- **One measured part for ZERO / route 2 (not an idea of mine, a price):**
  - The expert one-call is compute-bound (module 2, last night, quiet): its kernels stream ~170 GB/s on distinct bytes
    against a ~290 GB/s peak, for ~23 ms of a check's ~54 ms GPU busy [K: m2_prof_w1, Leap's tonight verdict].
  - So ~120 GB/s of DRAM bandwidth sits idle during the experts, ~2.7 GB per check [P].
  - Another engine (ANE, CPU AMX) could read that during the check without slowing it. What it reads can't depend on
    the drafted tokens (they don't exist yet), so it is a budget for route 2 / ZERO, not a route-1 channel.

## 1. SVP: the mechanism, as proposed
- **Source:** CPU value prediction predicts an instruction's result from the values the same static instruction
  produced before.
  - Last-value: the previous value.
  - Context / FCM: the value that followed the same recent history, used only when a confidence counter says it is
    reliable.
- **Mapping:**
  - "Instruction" = the 125B at a position.
  - "History" = the last n tokens (n = 16 / 12 / 8 / 6 / 4, longest first).
  - "Value table" = the true states the session's checks and prefill already computed, at every verified position.
- **Use:** at a drafted position t (depth d past the anchor), if the last n drafted tokens ended earlier at s <= the
  anchor, the head's stand-in state becomes wd[s] (true, from the 125B's manifold) instead of its own carry.
- **Price [P]:** zero GPU bytes; one hash lookup per drafted node on the CPU (microseconds); a few MB of stored states
  per session.
  - It is the cheapest conceivable channel. That is why it was worth killing properly.
- **Why it could have worked:** code repeats identifiers and boilerplate, and a retrieved state is a REAL 125B state,
  not a regression's blur.

## 2. The screens (CPU, existing dumps; on the true path the drafted tokens are the true tokens)
**2a. n-gram keys (svp_screen.py --pool gen, answer-side matches; prompt rows have no tk in the capture).** At hits:

| set, depth band | hit share | 125B top-1 agree (wd[s] vs wd[t]) | n-gram copy right | cos | R2 |
|---|---|---|---|---|---|
| HE+ d 1 | 0.112 | 0.610 | 0.613 | 0.757 | 0.147 |
| HE+ d 8-15 | 0.110 | 0.611 | 0.614 | 0.757 | 0.148 |
| HE+ d 16-30 | 0.088 | 0.607 | 0.610 | 0.752 | 0.073 |
| MBPP d 1 | 0.123 | 0.629 | 0.629 | 0.778 | 0.201 |
| MBPP d 8-15 | 0.125 | 0.633 | 0.633 | 0.779 | 0.204 |
| MBPP d 16-30 | 0.119 | 0.636 | 0.636 | 0.783 | 0.215 |

- With prompt-side matches allowed (svp_screen_all), HE+ hits rise to 62-65 %, mostly 16-gram: the answer repeats the
  prompt's signature and docstring. But those states have R2 -0.40 to the true one: the same text, a different role.
- Longer keys help the call only a little: 16-gram 0.75-0.77, 4-gram 0.53-0.60. Still, the call equals copying.

**2b. The hindsight bound (svp_oracle.py: max over s <= anchor of cos(wd[s], wd[t]), all verified rows incl. prompt).**

| set, depth band | best cos mean (p10 / p90) | angle mean (rad) | share <= 0.40 | share <= 0.55 | anchor's own state: cos, angle |
|---|---|---|---|---|---|
| HE+ d 1 | 0.775 (0.648 / 0.895) | 0.670 | 0.048 | 0.213 | 0.629, 0.882 |
| HE+ d 8-15 | 0.744 (0.595 / 0.886) | 0.715 | 0.043 | 0.179 | 0.488, 1.054 |
| HE+ d 16-30 | 0.723 (0.575 / 0.870) | 0.747 | 0.030 | 0.134 | 0.466, 1.079 |
| MBPP d 1 | 0.773 (0.647 / 0.903) | 0.671 | 0.062 | 0.228 | 0.649, 0.856 |
| MBPP d 8-15 | 0.732 (0.572 / 0.898) | 0.729 | 0.057 | 0.198 | 0.464, 1.082 |
| MBPP d 16-30 | 0.713 (0.547 / 0.890) | 0.757 | 0.049 | 0.169 | 0.443, 1.107 |

- **Why it can't help, in one line:** the head's own attention over the true prefix is already a learned, soft
  retrieval over exactly this table: every verified position's true state. The bound says the table itself sits
  ~0.7 rad from what a drafted position needs, so no lookup into it, hard or soft, closes the drift.
- **Reading:** retrieval does beat the anchor's own state (0.67-0.76 vs 0.86-1.11 rad). But the head needs ~0.40 rad
  to hold its own accuracy, and only 3-6 % of positions get there even with hindsight. Even a perfect, free selector
  would touch ~5 % of nodes, so the kept gain is bounded near zero [P].
- **Quality risk (had it passed):** a confident wrong retrieval misleads the head more than its own carry, which is why
  CPU value predictors need confidence counters. Moot now.
- **What kills it:** measured above. Hits are rare, retrieved calls equal copying, and the hindsight bound is ~0.7 rad.

## 3. What this changes for the other lanes
- **Retrieval and correction are both closed** (this screen + SB-ASSIM). Proposals that "feed the head a cached / mean
  / corrected true state" need a new mechanism to come back. A kNN over a large external corpus included: if the
  session's own, closest-context states sit at 0.7 rad, a foreign corpus's sit farther.
- **The role effect:** the same text in prompt vs answer gives R2 -0.40. Any distilled proxy (FAR) must model the
  document role and position, not just local tokens. Probably worth a feature: an "answer-side" flag or a position
  embedding relative to the answer start.
- **For ZERO (measured, not mine to design):** ~120 GB/s of DRAM is idle during the ~23 ms of compute-bound expert
  kernels per check, ~2.7 GB [P]. It is only usable for work that doesn't need the drafted tokens. Candidates: the
  NEXT check's expert prefetch at SSD scale, or route 2's extra rows' bytes; not the drafter's per-token channel.

## 4. Screen to rerun or extend (no GPU, nothing for tonight's capture)
- `cd ~/MoonShots/week98/break98 && ~/BigMoE/.venv/bin/python -I -B svp_screen.py --pool gen` (14 s), and
  `svp_oracle.py` (seconds). Both read only BigMoE's bank captures.
- **Capture ask by 18:00: none.** The mechanism is dead on existing data. If FAR's proxy wants the role feature, the
  bank captures already carry `gen` (the answer start).
