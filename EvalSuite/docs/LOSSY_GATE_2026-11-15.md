# LOSSY_GATE: the fastest gate that still decides "same scores" for a lossy check (WEEK98, goal B)

EvalSuite, Sun 15 Nov 01:xx. Goal B needs a 32-row check under ~23-25 ms, which means reading under half the exact bytes
(low-bit weak experts, acceptance-weighted expert sets, union caps). Every such variant changes the output, so it counts
only after a same-scores gate (WEEK98 rules) and Kamen's call. This is that gate, built for many variants a night.

**In one line:** a 15-minute pre-screen kills the bad variants, a 45-minute fast gate (45 GB; 76 min at 24 GB) decides
"same scores" at a stated resolution, and the full card gate (5-7 h) runs only for the one variant that ships.

| Tier | What it answers | Items | 45 GB | 24 GB | Verdict |
|---|---|---|---|---|---|
| 0 pre-screen | Is the variant obviously worse? | replay of ~100 exact rows + 40 decoded items | <= 15 min | <= 15 min | KILL or GO only |
| 1 fast gate | Same scores at -5 pt pooled resolution? | 364 (HumanEval+ 164, MBPP 100, IFEval 100) | ~45 min | ~76 min | PASS / FAIL |
| 2 release gate | The card gate vs NAX6-N | 381 x 2 orders | ~5-7 h | ~7 h | PASS / FAIL for a release |

The times come from our own gates, with process loads and heat included. The P2 arm at 45 GB made 169,696 tokens in 83
min, or 2,045 tokens/min, with the fans up; this morning's heat (53 % frozen) halved that to 1,084. The P2 arm at 24 GB
ran at 1,216 tokens/min. The w24 card gate took 7 h 04 min.

## 1. What a variant must give (test from a config flag, no release build)
A variant is three things:
1. **An env-flag delta on a committed screen snapshot** (BigMoE's data/r57/snap/<snap>/, as P2 was:
   R57_P2=256,R57_P2_ROWS=16,...). The base config's env line plus the variant's flags makes one env file.
   - EvalSuite pins it by full git blob, plus the snapshot's tree hashes, the screen's sha256 and the .so. A dry run
     checks the pins before any GPU time. No package or release build is involved.
2. **An engagement counter per prompt in runs.json,** as P2's `p2` field was: e.g. the rows checked through the lossy
   path. The gate voids any kept run where the variant did not engage. A flag that silently does nothing must never
   "pass".
3. **Its byte and time figures,** which the screen already logs per check (demand SSD, misses, MB per kept token, check
   ms). Breakthrough prices them (LOSSY_CHECK_PLAN.md); the gate reports them next to the verdict and never mixes them
   in.

**Wanted from BigMoE: a REPLAY mode for tier 0.** It would feed stored rows through the variant's check as 32-row
drafts and log, per position, whether the check accepted and the exact top-1 margin.
- That is the "oracle drafter", the same trick as T1 COPYABILITY.
- R57_REFTOK does not do it today: it compares a finished run's tokens with a reference afterwards.
- Without replay, tier 0 falls back to 0b below.

## 2. Tier 0: the 15-minute pre-screen (kill only, never adopts)
**0a, replay (needs the mode above):** about 100 cached base rows (25K tokens) pass through the variant's check as
perfect drafts. That takes ~2-3 min plus a load, since a 32-row check at 25-50 ms covers 25K tokens in about a minute.
- **A, agreement:** the share of positions where the lossy check keeps the exact token. For a lossy check this is the
  per-token output fidelity, and it does not compound the way a free run does.
- **F_hi, high-margin flips:** flips per 1,000 tokens where the exact top-1 margin is >= 2 nats. A lossy check that
  flips near-ties is cheap in quality; one that flips confident tokens is not. Specter's INT4 copy agreed with its
  parent on 96.6 % of tokens.

**0b, the mini-panel (always runs):** 20 HumanEval+ and 20 IFEval items, decoded and scored. That is ~11K tokens: ~7 min
at 45 GB, ~10 min at 24 GB.
- It is paired against the cached base rows.
- From R57_REFTOK, it also logs where each answer first diverges from the base row.

**Kill lines.** Pre-registered once, and calibrated once before the first variant (about 1 h of GPU).
- **Calibration:** run tier 0 on configs whose fate we already know.
  - Passed a gate: exact (A = 100 %), P2-24, P2-45, w13b (vs NAX6-N), m1b.
  - Failed one: 12 GB t80ck1u (vs REF2-T).
- **KILL if any of these holds:**
  - A is below the lowest passing calibrator's A minus 0.5 pt;
  - F_hi is above twice the highest passing calibrator's;
  - the mini-panel shows a net loss of >= 4 of 40 against the base;
  - any output is malformed, empty or looping.
- **Otherwise GO to tier 1.** Tier 0 is tuned so it does not kill variants that would pass: when in doubt, GO.

## 3. Tier 1: the fast gate (decides "same scores")
**The panel:** HumanEval+ 164 (60 + 104), MBPP 100 and IFEval 100, so 364 items and ~92K tokens.
- **Why these:** short answers (202-306 tokens on average) give many items per minute. They cover code, where lossy
  checks are meant to reach 1,000 tok/s, and chat. Exact references for all three exist already (ref2tnax: humaneval 60,
  ifeval 100; ref2tnaxn.r3: humanevalx 104; ref2tnaxn.r2: mbpp 100).
- **Left out because they are slow:** GPQA (1,666 tokens per answer), LCB (1,570) and math (~700). They stay in tier 2.

**The reference:**
- **Against exact:** NAX6-N's stored rows at 45 GB, with no control arm. The 45 GB exact path is deterministic, so
  stored rows are valid.
- **Against the shipping base**, for attribution: the base config's own panel rows, run ONCE per base snapshot and
  cached. They must use the same keys, order and 20-prompt process split, because with QOM, tokens depend on order and
  state; with that fixed, they are deterministic (BigMoE's 45 GB A/B: 12 / 12, logits 721 / 721).
  - After that, each variant costs one arm. This is where the speed comes from: N variants cost N arms, not 2N.
- **At 24 GB:** the 24 GB card runs QOM on code sets, so its cached base rows are the reference for attribution, and
  NAX6-N's rows are the reference for the release question.

**The order:** sets interleaved round-robin, so any prefix of the run is a balanced sample (for the interim look).

**The lines (all per variant, vs the reference above):**
- **Pooled, the deciding line:** a paired delta over all 364 items above -3 points, and an exact McNemar p >= 0.05. A
  gain with p < 0.05 counts as DIFFERS (HIGHER) and passes.
- **Per set, a guard:** decision 4 as at every EvalSuite gate (a delta above -5 points and p >= 0.05) on HumanEval+,
  MBPP and IFEval each, so a variant cannot trade code for chat.
- **Engagement:** the counter is > 0 on every kept run, and every key has one uncut run.
- **PASS** = the pooled line and all three guards pass, with valid engagement.

**Early stop (saves ~20 min on clear losers):** one interim look at 182 items (50 %). The variant FAILs early if the
pooled McNemar z <= -2.8 (a Haybittle-Peto boundary, which leaves the final test's alpha at ~0.05). There is no early
PASS.

**What it can see.** The discordance on this panel is measured, not assumed. The shipped v1' stack (CERT's forward
arm) against exact disagrees on 16 of 364 pairs (4.4 %). More lossy variants may reach the card sets' 12 %.

| True pooled drop | Power to FAIL it, 4.4 % discordant (measured) | Power, 12 % discordant |
|---|---|---|
| -3 pt | ~78 % | ~38 % |
| -5 pt | ~100 % | ~79 % |
| -8 pt | ~100 % | > 99 % |

- **False FAIL** of a truly equal variant: ~3 % at 4.4 % discordance and ~7 % at 12 %.
- **Plainly:** at today's discordance, tier 1 catches a pooled 3-point loss 4 times in 5, and a 5-point loss almost
  always.
- **Checked on real rows:** tools/lossy_d4.py reproduces CERT's published forward lines exactly. The shipped stack's
  pooled line is -2.20 points (p 0.077), so it passes, 0.8 points above the screening line (docs/PREREG_lossygate_2026-11-15.md).

## 4. Tier 2: the release gate (only for what ships)
- **The card gate**, unchanged: IFBench 150, GPQA Diamond 100 and LCB v6 131, both orders, against NAX6-N. Decision 4
  per benchmark and order. It ran for w24, m1b and lcoh.
- **Cost:** about 870K tokens, so ~5-7 h at 45 GB and ~7 h at 24 GB. It adds the long-reasoning sets that tier 1 skips.
- It runs once, for the variant Kamen wants to ship.

## 5. Why ~364 items: how much variants disagree today
| Gate | Set | Discordant / n | Rate |
|---|---|---|---|
| w24 vs NAX6-N | IFBench, GPQA, LCB | 15 / 150, 9 / 100, 7 / 131 | 10 %, 9 %, 5 % |
| m1b vs NAX6-N | IFBench, GPQA | 18-20 / 150, 10-14 / 100 | 12-13 %, 10-14 % |
| P2-24 vs the card | IFBench, GPQA (67) | 16 / 150, 4 / 67 | 11 %, 6 % |
| P2-45 vs chat45 | IFBench, GPQA (67) | 23 / 150, 9 / 67 | 15 %, 13 % |

- **The basis:** with ~12 % of items discordant, McNemar needs ~370 pairs to see a 5-point drop at 80 % power, and only
  ~145 to see an 8-point drop. So the per-set lines (100-164 items each) are guards against large set-specific losses,
  and the pooled line carries the decision.
- **Why not more items:** the panel's time rises linearly. Tier 2 exists for the finer question.

## 6. Throughput per night (8 h)
- **At 45 GB:** about 10 tier-0 screens (2.5 h) plus 4 tier-1 gates (3 h), or one tier-2 gate.
- **The once-per-base costs:** the base panel run (~45 min at 45 GB) and the tier-0 calibration (~1 h), both reused by
  every variant on that base.
- **GPU locks:** share locks only (never quiet tonight), with MEM in the reason, yielding to other MEM jobs.

## 7. What exists and what is to build
**Ready:**
- runner 63 (PREREG_p2gate45: an env-file arm with blob, snapshot-tree and .so pins; P2-style engagement checks; a
  paired verdict against a reference run);
- set_d4.py (decision 4); the exact panel rows; the w24 / P2 / chat45 runs as calibrators.

**EvalSuite builds (~1 h, when the first variant is ready):**
- runner 64, which is runner 63 with:
  - any key panel and the round-robin order;
  - the pooled line and the interim look;
  - a generic engagement field (CARD_ENGAGE=<runs.json field>);
  - the tier-0 mini-panel.
- Plus a pre-registration (PREREG_lossygate) with the calibration step and these lines, written before the first
  variant.

**BigMoE (asked):**
- the REPLAY mode for 0a;
- an engagement counter in each variant;
- one committed snapshot per variant family.

**Breakthrough:** the variant list with its byte and time pricing (LOSSY_CHECK_PLAN.md). The gate sorts variants by
their price, so the cheapest checks get gated first.

## 8. Limits, plainly
- **Resolution:** tier 1 resolves pooled losses of about 5 points. A 2-3 point loss can pass tier 1 and is caught only
  by tier 2's larger sets, or not at all.
- **Scope:** the panel is short-answer code and instruction following. Long reasoning (GPQA, LCB, math) is tested only
  in tier 2.
- **One order per tier-1 arm.** With QOM, tokens move with order, so the base rows must match the variant's order and
  process split exactly.
- **Speed is not gated:** timing claims stay with the screens' own A/Bs under a quiet lock. The gate decides quality
  only.

## 9. The REPLAY mode, spec for BigMoE (tier 0a; the orchestrator puts it on BigMoE's list)
**What it measures:** how often the variant's check keeps the exact token, at every position, given the exact prefix.
This is the variant's per-token fidelity, with no compounding.

**The interface:** `R57_REPLAY=<rows.jsonl>[:margins.json]` together with R57_KEYS, where rows.jsonl holds key ->
prompt ids and reference completion ids.
1. **Prefill** each key's prompt as the variant normally would, with its own prefill path and flags.
2. **Checks:** run checks of the variant's normal width (32 rows = the current token plus 31 drafted rows). The drafted
   rows are the next 31 REFERENCE tokens, not the drafter's. The check path, expert selection, row count and lossy
   flags are exactly those of decoding.
   - **CORRECTED (Tue 2026-11-17 06:4x, information):** "32 rows" was this spec's assumption.
     - On the r57 screens (cc1, rpl1/2, t1c1, g1a, g1b) the normal width is 16 rows: up to 15 drafts plus the current
       token. Every gated env leaves R57_ROWS unset (default 16), and 32 rows exist only under R57_ROWS=32 or
       R57_G32TEST.
     - The REPLAY runs used that width, as rule 2 requires: 10,315 drafted rows over 708 checks (14.6 per check).
     - No tier-0a line or verdict changes.
3. **Teacher forcing:** after each check, commit the REFERENCE block whatever the check accepted, and never the check's
   correction token. Every position is then judged with the exact prefix. Continue to the row's end, or a cap
   (R57_REPLAY_CAP, default 2048).
4. **Log per drafted position:** accepted (the check's pick == the reference token), the check's own top-1 id, and its
   own top-1 minus top-2 logit margin in nats.
5. **The exact margins:** run REPLAY once with the exact config on the same rows. It must accept 100 % (a self-check),
   and its margins are saved to margins.json, which variant runs then read.
6. **Output per key** (runs.json, or a replay.json beside it):
   - n positions and n accepted;
   - flips binned by the exact margin: < 0.5, 0.5-1, 1-2, >= 2 nats;
   - the usual per-check stats (check ms, demand SSD, misses, MB);
   - the variant's engagement counter.
7. **Determinism:** the same keys in the same order. Two runs of the same config must give identical logs.

**The rows:** the first 100 keys of the tier-1 panel (round-robin order), from the cached base run or exact rows: about
25K tokens, which is about 780 checks (~1 min at 25-50 ms per check, plus a load).

**Read by the gate:** A = accepted / n, and F_hi = the flips at margin >= 2 nats per 1,000 positions (section 2).

## 10. The CODE-ONLY scope (the orchestrator 01:2x; Breakthrough: only a code-scoped check reaches ~23-29 ms)
- **Tier 1, code:** HumanEval+ 164 and MBPP 100 (264) pooled, with the same -3 point screening line, per-set guards and
  an interim look at 132.
  - IFEval 100 then runs in its own processes as the switch-off check. The variant's counter must be 0 there, and its
    tokens must be identical to a cached base run's, or pass decision 4 against exact rows.
  - About 35 min at 45 GB for the code block (~61K tokens), plus ~15 min for IFEval.
- **The verdict is labelled "(CODE-ONLY mode)"** and licenses a code-only mode, never a lossy check on chat.
- **Gate code-scoped variants against a cached base run, not exact.** That measures the variant's own cost. The exact
  comparison is information here, and tier 2 asks the release question.
  - **Corrected 01:3x (the orchestrator, from MoonShots' ~/Papers/CERT_CODE_CAVEAT_2026-11-15.md):** the "-2.65 points
    against exact" said here before was CERT's forward order only.
    - The reverse order is +0.76 points (HumanEval+ 0.00, MBPP +2.00).
    - Both orders together are -0.95 points (net -5 of 528 pairs).
    - So "v1' fails the code line against exact" holds for one order only.
- **The order effect (PREREG_lossygate amendment 1c):**
  - Each arm runs in its own processes from the same cold start, in the same order and process split as its cached
    base run. It never runs back-to-back with another arm in one process, because a warm cache changes QOM's tokens and
    favours the second arm.
  - So the order's cache state hits both arms alike and cancels in the pairs. What remains, QOM noise once the tokens
    diverge, is discordance that McNemar counts as noise.
  - Still, one order is one draw: CERT's v1' moved 3.4 points between orders. A variant that PASSes tier 1 in the
    forward order therefore runs the reverse order too, against the base's reverse run.
  - The tier-1 verdict that licenses tier 2 is then read on both orders pooled. A forward FAIL stops the variant there.
- **Engagement (corrected 01:5x):** CARD_ENGAGE=<field>:<key>+<key> sums only the listed counts of a dict field.
  - Breakthrough's counter (LOSSY_CHECK_SPEC.md) takes CARD_ENGAGE=lossy_check:e3+edrop+lmsl_rows. That sum must be
    > 0 on code and 0 on IFEval.
  - Its `checks` and `e4` are nonzero on every prompt, so the whole-dict sum would read chat as engaged and fail the
    switch-off check wrongly.
  - V3's R57_HC4 has no set scope in the spec. As written it would run 4-bit mixers on chat too, so V3 can only get a
    CODE-ONLY verdict if HC4 is scoped to the code sets and `hc4` joins the sum
    (lossy_check:e3+edrop+lmsl_rows+hc4). The IFEval identity check would catch it as well.
- **Not tested here:** the gate switches by benchmark set. A deployed code-only mode needs a request classifier, and its
  misroutes are not tested by this gate.

## 11. Acceptance-rule variants (the orchestrator 01:5x; MoonShots' result H)
- **What they are:** a rule that keeps a drafted token the 125B's own check rejects.
  - Example, R(k): inside a comment or docstring span, a deep miss is accepted when the drafted token's 125B rank is
    <= k.
  - The check itself is unchanged; the output changes because the accept rule does.
- **The gate carries them like check variants:**
  - tiers 0-2;
  - the CODE-ONLY scope with the IFEval switch-off check;
  - the cached base reference;
  - both orders before tier 2.
- **What differs:**
  - **Engagement:** the rule needs its own runs.json field per prompt, counting the tokens it accepted that were not
    the 125B's top-1 (e.g. `"accept_rule": {"accepted": n, "spans": n}`, with CARD_ENGAGE=accept_rule:accepted). The
    count must be > 0 on code and 0 on IFEval.
    - It must not reuse the base's own relaxation counts: lossy-45 already runs a relaxed span accept, logged as rlx /
      spn on every QOM run.
    - It is gated with CARD_ENGAGE_MODE=any (03:2x): > 0 summed over each process's code runs. An accept rule only
      engages where a draft qualifies, so many single code runs honestly read 0.
  - **Runaway answers (amendment 3a, 10:3x; MoonShots' J(CC) on the 2B):** delta 2 failed through capped / looping
    answers and docstring rewrites. So tier 1 now carries a RUNAWAY guard for every variant: paired cap-hit-or-loop
    increases over the pooled sets, FAIL at McNemar p < 0.05.
    - The speed read-out (tools/lossy_speed.py) quotes kept per check on answers uncapped in both arms. On the 2B, part
      of the delta-1 lift was runaway answers.
  - **Pins:** the drafter decides which tokens are offered, so the variant's amendment pins:
    - the drafter checkpoint (sha256);
    - when the rule audits with the 2B, the 2B checkpoint and the audit settings;
    - the span detector's code, through the trees as usual.
  - **The risk sits on MBPP:** 16 % of in-span deep rows take a token that isn't the 125B's top-1 (MoonShots), and T16
    (11-07) flipped an MBPP task.
    - The per-set MBPP guard (decision 4) is the line to watch, since the pooled code line can hide a set's loss.
    - The verdict quotes MBPP's discordant pairs on their own.
  - **Tier 0's REPLAY lines don't transfer:** an accept rule's agreement A is 1 minus its relaxed share by construction.
    So tier 0 for an accept rule is the mini-panel's KILL line only, until an amendment calibrates A / F_hi for accept
    rules.
  - **A rule together with a check variant** is gated as one variant, under its own amendment and against the same
    cached base. Two separate verdicts are never added up.
- **Nothing to build in the gate for this:**
  - the key-subset engagement (runner 64, 01:5x);
  - the reverse order and the pooled verdict (amendment 1c).
  - What the variant brings: its counter field, its set-scoped switch (the R57_QOM_SETS plumbing) and its pins.

