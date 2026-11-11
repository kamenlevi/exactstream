# BREAKTHROUGH PUSH

> (a quote from the project owner, kept private)

## Target

- **350 tok/s** on the full 125B (all 512 experts, no pruning) on this MacBook Pro M5 Pro 64 GB.
- First at 45 GB, then 24 GB.
- Both tracks:
  - EXACT = bitwise to its reference.
  - FAST = the same task scores in EvalSuite.
- A revolution means a step change in how a check is done or how the model is fed, measured on this machine. A +10 % tweak doesn't count.

## Where we stand (Tue 17:00)

**Measured**
- [K] 63.6 tok/s exact at 45 GB: REF2-T, C 256 + OOO + K-73 + per-layer sizes.

**Anatomy of one check** [K] (BigMoE c45l_256, median, C 256)
- The cycle is 96.8 ms:
  - draft 12.0 ms, SSD idle;
  - check 84.2 ms: SSD busy 42.0, gate spin 42.6.
- 171 expert misses per check.
- A verified row costs ~7.82 ms.
- Code keeps ~10 tokens per 16-row check. Live chat checks hold 3-4 rows.

**The drafter already proposes the right token** [K]
- DFlash2's top-3 holds the 125B's token 89 % of the time at mid rows (KWALL (a), MTP Head).
- At ~7.8 ms per extra verified row, everything that tries to use this prices out (TOP3-USE, DEPTH32, DIVERSE2, ROUTEPRED):
  - trees: -10 .. +2 %;
  - a learned picker: +0.3 %;
  - deeper drafts: x0.66;
  - ensembles: +0.3-1 %;
  - next-check expert prediction: 0.07 precision.
- The lever is the row cost.

**Passing levers that haven't been stacked yet**
- Q1-UNION 0.9 + 3-bit sure rows: PASS on code at live lengths, BETWEEN on chat.
- OEA k8: PASS in every domain.
- Native layout NAX6-N: +21 %. 3c-N gives an exact fallback, and the conversion is prepared.
- NAX6-H: +9.1 %.
- Volatile expert tier: built, smoke test tonight.
- Warm pools: mission 20, tonight.
- U3 engine mode: tonight.
- P1-K1 + 3-bit pass 1: the 12 GB line.

**Route** [P] (Fable C's ROUTE_50): FAST 350 at 50 GB is 12-20 ms per check short.

## The walls: one thinker each

CPU only. Answers go in ~/FableGod/push/ (Lunatic: ~/FableGod/lunatic/).

| Thinker | Mission | File |
|---|---|---|
| Fable A | THE SSD WALL: make the ~42 ms of SSD waiting per check vanish at 45 GB, then 24 GB | push/SSD_WALL2.md |
| Fable B | THE VERIFY-ROW WALL: make a row 5-10x cheaper for choosing, keep the output exact, cash in the top-3 | push/ROW_WALL.md |
| Fable C | THE 350 STACK (mission C4, updated): price every passing lever together against the 350 budget | fablec/UNION_STACK.md |
| Lunatic | 1000 tok/s: what would have to be true, and the routes others call impossible | lunatic/LUNACY2.md |

## The builders

Each bet is pre-registered before any GPU run, and 10-min screens come first.

- **Breakthrough:** staged verification on the decision map, CPU first. Does a cheap pass pick the right path out of the top-3 tree?
- **MTP Head:** the tree-draft data for it: top-3 per row with probabilities on the map windows, aligned.
- **NewIdeas:**
  - prior art: staged / cascade verification, lossless weight compression, expert-major scheduling;
  - pricing;
  - the push scoreboard in RANKED / FAST_TRACK;
  - keeping every bet pre-registered.
- **MoonShots:** lossless compression of expert bytes, exact. Measure on CPU how compressible records3 is.
- **BigMoE:** tonight's chain (mission 20, U3, combination, look-ahead, volatile tier). Then the measured FAST code stack.
- **Leap:** kernels for whatever wins: tree masks, native conversion, decode kernels.
- **EvalSuite:** Thursday's RELAX, then same-scores tests for the stack.

## Rules

- Both tracks are first-class. Exact means bitwise; FAST is judged by task scores.
- Research comes first: check prior art, tag each idea published / adapted / new, and keep at least one genuinely new bet.
- Mark every number [K] measured, [P] arithmetic, or [G] judgment. Halve modeled gains when deciding.
- Night is for big tests, day for small ones. By day only one GPU job runs at a time.
- Swap guard: +2 GB. Disk floor: ~110 GB.
- Only next-caliber finds go to Kamen unasked.

## Cadence

1. **By ~18:00 today:** the thinkers' first drafts. The orchestrator merges them into a ranked shortlist for Kamen.
2. **Tonight / Wed:** CPU screens.
3. **Wed after 15:00 (one job) and Thu by day:** GPU screens.
4. **A night:** the stacked measurement.

## Results, round 1 (Tue 17:18-17:27; orchestrator's synthesis, numbers from the four files)

**Fable A (push/SSD_WALL2.md).** The SSD's 42 ms per check is two walls:
- ~20 ms is a FIXED cost per loader call (0.28 ms x ~70-77 calls);
- ~21 ms is bytes.

Findings:
- Exposed today: ~24 ms per check.
- 350 needs <= 33 MB per kept token. EXACT at 45 GB by day holds ~40-46, so 350 is closed on bytes. FAST at 45 GB fits with the sidecar drop, and then the GPU floor (~42 ms) binds.
- Night (C 320): EXACT fits at zero margin, and only with designs 1 + 2.
- Designs:
  1. zero-latency bursts (exact I/O scheduling: JOINT, keepalive, parallel >= 256 KB pieces);
  2. the queue never empties (pre-route reads join the open burst);
  3. fewer bytes, exact (packs).
- Tonight's BigMoE chain already measures part of it: the stall trace with and without JOINT, and the keepalive A/B.

**Fable C (fablec/UNION_STACK.md).**
- FAST code at 50.5 GB is ~265 tok/s at targets, ~290-297 with one-call OOO. Decision-grade (halved): ~215-225 expected at 50 GB, ~210-220 at 45 GB. The gap left is the exposed SSD.
- Chat: 56-63 tok/s. 350 on chat would need >= 17 kept per check, which only the row wall could give.
- New cut, COLD-CUT: code bytes x0.35 at 45 GB, deterministic. Its screen is pre-registered.
- EXACT at 50.5 GB:
  - 3c-N on native records: 134-155 tok/s (bitwise to REF2-T);
  - NAX6-N: 157-186 tok/s (a new reference).

**Fable B (push/ROW_WALL.md).**
- A verified row costs 5.5-8.5 ms: 3.0 ms GPU + 2.5-5.5 ms SSD. A choosing-only row can cost 0.35-1.2 ms.
- But the exact pass already cashes in the top-3. At the first miss the bonus token IS the target's argmax.
- Conditional on a miss, top-3 holds the token only 0.69 on code and 0.33 on chat. KWALL's 0.888 counts rows that were already right.
- Exact tree nodes are worth 0.14 tokens each on code, and an exact row's physics floor is ~1.5 ms, so trees stay dead at any reachable row price.
- Where the row cost DOES pay: the EXACT row, 7.82 -> ~2.3-2.5 ms at the floor = +40 % on code.

**Lunatic (lunatic/LUNACY2.md).**
- Single-stream, even with the whole model in RAM, tops out near 315 tok/s EXACT and 350 FAST. The 125B's own uncertain rows (13.5 % on code) cap agreement.
- 1000 tok/s exists only as WIDTH. Route W: one user's parallel agents are a batch sharing expert reads, FAST at 8-12 streams = ~880-1,270 tok/s aggregate. 63 % of Kamen's agent generation last week ran with >= 8 sessions at once.
- Route C (copy verification) failed on his traffic: copies are 6.2 % of output.
- EXACT at 1000 needs the store in RAM (71.8 vs 64 GB). The SSD caps EXACT near 285 tok/s at any width.

**Orchestrator's read.**
- Single-stream 350 is at the edge of physics on this Mac. It's reachable only FAST, on code, with every lever stacked and the SSD hidden.
- The drafter/tree lane is closed.
- The revolution candidate is Route W: many streams sharing one expert read. It's Kamen's call whether it becomes a track.
