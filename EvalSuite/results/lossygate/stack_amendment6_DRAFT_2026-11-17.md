# DRAFT, not filed: PREREG_lossygate amendment 6, the GATED, MEASURED lossy stack for Saturday (t30b + tree-16 + CC(1) + G1's pick)

Prepared Tue 2026-11-17 09:4x from the orchestrator's decisions of 09:3x and 09:4x (relayed by NewIdeas).
- It is filed into docs/PREREG_lossygate_2026-11-15.md when BigMoE sends the stack's blob (Thu ~midday). That is
  BEFORE the smoke and before any stack data.
- Placeholders are in <angle brackets>. The open points are listed at the end.

## Amendment 6 (Thu 2026-11-19 <HH:Mx>, before any stack data): the lossy stack's gate (tier 1 in both orders vs exact, the card gate (A), the headline timing run)

### Why
- **The orchestrator, 09:3x:** a GATED, MEASURED lossy number by Saturday on a NEW config: t30b + tree-16 + CC(1) +
  G1's pick, on BigMoE's r57 path (plumbed from Wednesday).
  - It supersedes the chain-16 stk / stk0 plan. tools/stk_chain.sh (ede7aa9) was never armed and has no data.
- **The orchestrator, 09:4x:**
  - t30b runs at Q8 + the 32K screen if MoonShots' Wed ~07:30 Q8 read-out holds kept within its pre-registered line.
    Otherwise it runs at bf16 + the 32K screen. It never runs as stored.
  - Sync-free is BigMoE's default if MoonShots' bit test on BigMoE's ops passes by Wed evening. fused2 is out.
  - In lossy mode, tier 1 and the card gate run ON the chosen format, so the gate covers it. This amendment therefore
    pins the format and the screen, alongside the tree flags.

### 1. The pins (from BigMoE's blob; the runner checks them at every start and per process)
- **The env:** <path>, blob sha256 <...>, tag <...>. The snapshot trees, the screen and the .so are <...> (bm_pin).
- **The draft policy** (BigMoE's flags, 10:56 Tue; final unless a test forces a change, and BigMoE says so first):
  - **Code:** R57_DRAFT=t30b, i.e. code drafting by t30b over the shipped MTP layer (unset = DFlash, as today).
    - **The format:** R57_T30B_FMT=<q8h32 | b16h32>.
      - q8h32 is the default: MTP Head's t30b_q8h32.safetensors (b5388cc4...) + Leap's ScreenedHead 32K + session
        extras.
      - b16h32 is the fallback.
      - "stored" is CERTIFY98's count check only, never this stack.
    - **Tree-16:** R57_TREE_NODES=15 (best-first pops; check rows = nodes + 1 = 16) and R57_TREE_K=8 (top-8 children
      pushed per pop, as in the T8 dumps and tree_cc32).
    - **Sync-free:** R57_TREE_SF=<1 | 0> (MoonShots' sync-free frontier; 0 = the host-synced heapq loop).
  - **Chat:** unchanged. R57_DRAFT changes code drafting only.
  - **Acceptance:** CC(1) with amendment 3's flags unchanged: R57_RELAX_DELTA=1,
    R57_RELAX_DELTA_SETS=humaneval,humanevalx,mbpp. On the tree it follows BigMoE's rule, per check:
    - The 125B verifies the current token + the 15 popped nodes in one forward (tree mask), so each node's row is
      conditioned on its own path. The walk starts at the current token's row. At node u, with the 125B's logits z_u:
      1. if a selected child of u carries argmax z_u, it is accepted (exact);
      2. else, with CC on for this set: take the selected child c of u with the largest z_u[c], and accept it if
         z_u[c] >= max z_u - 1 nat (a close call, counted in close_call.accepted);
      3. else the walk stops, and the check emits argmax z_u (the bonus token).
    - Close calls may occur at any depth and any number of times per check, exactly as in the gated chain rule
      (relax_delta.relaxed_k: no per-check cap; amendment 3's d1 = delta 1 nat, not a count). On a chain (k = 1) the
      rule reduces to the gated one.
    - <R57_TREE_CC_MAX: unset (no cap), unless NewIdeas asks for a cap by Wed midday.>
  - **G1's pick:** <the env EvalSuite applies at V2's verdict by 5d (pick5d.py; tools/g1_pick.sh), or none
    ("comments-only")>.
  - **The expert rewrite, tgl16eu** (Breakthrough's flag; IN the stack by the orchestrator's 11:1x decision, if Leap's
    c25 A/B by Wed evening holds: ~-2.5 to -3 ms at 16 rows, no extra expert GB, determinism): R57_XRW=tgl16eu <on |
    unset> (Breakthrough's handoff name, docs/EXPERT_REWRITE_HANDOFF.md; final on BigMoE's confirmation).
    - runs.json carries the per-run delta of the frozen knob's (knob_v1) counter: "xrw": {"mode", "cand" (calls run
      by tgl16eu), "orig" (calls sent to today's kernel), "fallback_not_p11" (calls on a non-p11 store)}.
    - The expert one-call's exact part is bitwise == today.
    - The TIER-4 UPGRADE runs ~28 % of expert pairs on their 4-bit weights instead of MSB-3. Those are 3-bit pairs of
      experts that also have a 4-bit pair in the same call, so it costs no extra bytes.
    - Those pairs' outputs move toward the more accurate tier (the tier-3 / tier-4 gap is ~0.25 rel L2). A pair's tier
      depends on the call's other rows, as today's 1c tiers already do.
    - Its quality is judged inside this gate, with no separate pass (Breakthrough, from the orchestrator 11:1x;
      ~/Breakthrough/docs/EXPERT_REWRITE.md, 2f126cc).
- **Engagement (validity):** CARD_ENGAGE = close_call:accepted@run, tree:sib_accepted@run <+ G1's terms> <+
  xrw:cand@run, if tgl16eu is on>.
  - With tgl16eu on, it is also VOID unless every run has xrw.mode == tgl16eu, and xrw.orig == 0 and
    xrw.fallback_not_p11 == 0 summed over the run. EvalSuite checks this from runs.json before the verdict
    (Breakthrough 11:1x: cand > 0 implies upgrades in practice, ~28 % of pairs per call).
  - runs.json's "tree" block is {"nodes", "k", "checks", "sib_accepted"}. sib_accepted counts accepted nodes that are
    not their parent's head-top-1 child, i.e. what a chain could not have kept.
  - Each term must be > 0 on every kept run. The QOM sets are <...>.
  - Runner 69's parser reads it as is. A term is field[:key+key][@run]; "tree" is a dict, so its "sib_accepted" is
    summed over the run's kept code answers. The DRY on BigMoE's env confirms it.
- **The format's evidence:**
  - MoonShots' Q8 PASS on Tue was the basis of the orchestrator's 09:4x choice (~/MoonShots/week98/draft/
    DRAFT_PLAN_1117.md, STATUS Tue 11:1x). The registered read-out is Wed ~07:30, against its line: <result, file>.
  - Sync-free: MoonShots' bit test on BigMoE's ops, bitwise on 176 + 120 trees: <file>.
- **The scope:** CARD_SCOPE=code. CC and t30b act on the code sets, while IFEval runs r57's unchanged chat path; tier 1
  still reads IFEval by decision 4 against exact.

### 2. The smoke (first: Thursday by day, after BigMoE's key test)
- **What runs:** runner 69's card tier on tools/stk_smoke_keys.txt (gpqad/rec2ZTsqazBZ3LTWZ, ifbench/124, lcb6/3692),
  both orders, CARD_RUNSUF=_smoke.
- **The check:** a function check, not a verdict. It needs rc 0 and a "(tier 2" DONE line that is not VOID or
  INCOMPLETE.
- **The memory rule:** gate = max(43, the smoke's lifetime peak + 0.5 GB, rounded up to a whole GB), and CARD_MEM =
  gate - 4.
  - The smoke reads 3 answers per process; full processes read 20. That is information, and the +2 GB swap guard is
    the backstop.

### 3. Tier 1 (Thursday by day; the AWAY rule may push part into the evening)
- **The comparison:** the stack against the EXACT references, results/ref2tnax, ref2tnaxn.r3 and ref2tnaxn.r2
  (CARD_REF=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; NewIdeas 09:3x). It runs both orders, forward then reverse, 364 keys
  each.
- **The lines:** the registered tier 1, on the two orders pooled (lossy_d4.py's pooled pair "EX+EX"):
  - pooled: delta above -3 pt AND McNemar p >= 0.05 (the screening line);
  - per set (HumanEval+, MBPP, IFEval): decision 4, delta above -5 pt AND p >= 0.05. IFEval uses decision 4 against
    exact, not identity;
  - the runaway guard (amendment 3a);
  - the interim after the forward order's first 182 keys: STOP-FAIL if z <= -2.8.
- **The stops:** a forward FAIL, or an interim STOP-FAIL, ends the chain. There is then no reverse order and no card
  gate.
- **Attribution:** a FAIL cannot be pinned on one lever. That is acceptable for a headline gate, whose question is the
  stack as a whole.
- **Tiers 0 and 0a** are not run. They only kill, the smoke checks function, and tier 1 decides.

### 4. The headline timing run (i) (Thursday night, first in the night's order)
- **What runs:** runner 69, CARD_TIMING=1, with the stack's pinned env on the chosen format.
  - One queued quiet lock (90 min) and no share locks.
  - HumanEval+ 164 + MBPP 100 (the code keys; IFEval dropped), in the panel's forward order, one order. No verdict.
- **The number:** whole decode, n / decode wall (sum n / sum(n / tps)), per set and pooled.
  - The config line goes beside it: format, screen, tree flags, CC(1), G1's pick, sync-free.
  - Tokens per second with prefill are reported beside it, never as the headline.
- **Heat:** a heat pause inside a timed decode (the fanwatch log) taints that answer, and the answer is re-run (twice at
  most).
- **The night's order (NewIdeas):**
  1. this run, ~45 min, quiet;
  2. BigMoE's exact timing (ii): t30b + tree-16, HE+ 164 + MBPP 100, token identity 264 / 264;
  3. CERTIFY98's count check, ~30 min (its own PREREG: t30b as stored, chain-32, exact acceptance).

### 5. The card gate (A) (Friday night)
- **Tier 2:** IFBench 150, GPQA 100 and LCB v6 131, both orders, against NAX6-N (ref2tnaxn.r6).
  - Decision 4 per benchmark and per order.
  - Amendment 1's PENDING / extension rule applies (CARD_EXTEND). The identity sets are <CARD_IDENTSETS, if any>.
- **The scope:** the full (A) scope, with lcb6 in every lever's sets.
- **Memory:** CARD_MEM comes from the smoke (section 2).
- **Typing (information):** IFBench chat 141 / math 6 / code 2 / agent 1; GPQA chat 67 / math 33; LCB code 131.

### 6. The reading (~Sat 07:00)
- **GATED, MEASURED:** tier 1 PASSES (both orders, pooled), the card gate PASSES (every benchmark and order), and the
  timing run gives the number.
- **A step that slips:** if the card gate has not finished, the reading is "measured, card gate pending". If tier 1
  has not finished, it is "measured, not gated".
- **A tier-1 FAIL:** the stack is not adopted, and any number is information only.
  - A bisect is a NEW config: the first, by the orchestrator's 11:1x plan, turns tgl16eu off. It needs its own dated
    amendment before its data, and its own tier 1 and card gate. This gate's runs are not reused as its reference.

### 7. Carried over
- **BREAK98_HOLD:** nothing starts while it exists. A running step yields at once and resumes when it is gone.
- **The guards:**
  - the AWAY rule (away_secs < 900: no start, and a running step yields);
  - HOLD, PAUSE and the lid;
  - one > 16 GB job at a time;
  - the +2 GB swap guard;
  - the yield to another session's run > 5 GB.
- **Validity as registered:** the pins; engagement > 0 on every kept run; QOM; every key has exactly one uncut run.

## Open (to settle before filing)
- **BigMoE:**
  - the blob (Thu midday);
  - the format after MoonShots' Wed ~07:30 read-out;
  - sync-free on or off;
  - any flag change forced by a test.
  - The tree flags, the CC-on-tree rule, the engagement field and chat's path are named (10:56 Tue).
- **NewIdeas:** a per-check cap on close calls (R57_TREE_CC_MAX) or none, by Wed midday.
- **Breakthrough / BigMoE:** tgl16eu's r57 flag and its runs.json engagement field; whether it is in, after Leap's c25
  A/B (Wed evening).
- **EvalSuite:** G1's pick at V2's verdict (5d, by pick5d.py; NewIdeas 09:4x).
- **EvalSuite:**
  - DONE Tue 11:1x (bd19455): runner 69's timing summary reads the tree flags (the shape and the draft flags).
  - DRY passes of runner 69 on the stack's env.
  - A chain for Thursday (smoke -> tier 1 forward -> tier 1 reverse) and the night's follower (the timing run, then the
    card gate on Friday night). Both are built from tools/stk_chain.sh, with the exact references in place of G1's
    base runs.
