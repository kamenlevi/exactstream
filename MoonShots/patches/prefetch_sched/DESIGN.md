# Verify-pass prefetch scheduler for BigMoE's loader: design, bound-first plan

## What the loader does today (bigmoe_ext.cpp, read at 8835881+)
- The routing row of layer L carries the look-ahead predictions (row[65] = n1 | n2<<8, ids at row[66..]): L+1's top-B and L+2's.
- **Spare-slot prefetch** (`prefetch(L+d, …)`, after L's release): one `dispatch_after` block per expert. In-flight reads are
  unbounded, there is no priority versus demand reads, and spares cost residency (BigMoE: 12/12 best; wider loses).
- **Victim-slot prefetch** (`BIGMOE_PF_VICTIM`, +5 % on spec k=3, exact): only while L stalls. Exactness rule: the GPU's fast path
  for layer X reads the residency table published one pass ahead. A victim may only leave a table the GPU cannot be reading,
  which is guaranteed while the GPU is held at L's gate.

## Step 0: measure the ceiling before building anything (cheap, uses the existing oracle replay)
BigMoE's oracle only gave layer L+1's true ids at L's stall: ideal +13 %, realistic −10 %. The ceiling for *any* scheduler is a
**full-pass oracle**: at the first gate of each verify pass, issue every recorded miss of every layer of that pass at once, as one
deep queue. Bound runs only, so use enough spares not to evict anything needed, e.g. `BIGMOE_SPARE` high at a residency lowered by
the same bytes. Same tokens are required.
- If full-pass ≤ +15 %: a scheduler can add at most ~10 % over today's victim prefetch. Build only the cheap parts (A, B).
- If full-pass ≥ +30 %: early knowledge is worth a lot, and the remaining question is *safe* victims outside stalls (C).

## The scheduler, in three flag-gated pieces (cheapest first)
**A. Bounded, demand-first prefetch queue (`BIGMOE_PF_QD=n`).** Replace "one `dispatch_after` per prefetch" with a CPU-side pending
queue served by a fixed number of in-flight reads (n ≈ 4–8). Demand reads (`ensure` → `load_jobs`) never wait in it, and they are
issued immediately at user-interactive QoS. That keeps the NVMe queue from filling with speculative reads ahead of a demand miss.
Pending entries are issued in (distance, rank) order: L+1's top ranks first, then L+2's. Entries whose layer has already been served
are dropped. Invariant to keep: a pending (not yet issued) prefetch must not hold a slot that `ensure` may spin on, so reserve the
slot at *issue* time, not at enqueue time.
**B. Per-layer feedback throttle (`BIGMOE_PF_FDP=1`).** Per layer, an EWMA of accuracy (used / issued) and lateness (prefetch
`ready` after the layer's gate). B_eff(L) = clamp(round(B · accuracy / target), 2, 2B): low-recall layers (early layers; the
125B's look-ahead recall ≈ 0.75) issue fewer speculative reads, high-recall middle layers more. Exact (only *which* reads are issued changes).
**C. Victims outside stalls (`BIGMOE_PF_VICTIM_ANY=1`). Only if step 0 shows ≥ +30 %.** Needs a race-free protocol for "the GPU has not
read layer X's table this pass": the routing-row kernel writes a per-(pass, layer) "reading" marker *before* its table lookup.
The CPU removes the victim from X's table, then checks the marker. If it is still unset, the removal is guaranteed visible
(fresh-line rule) and the slot may be reused; otherwise the victim is put back. This is BigMoE's gate code, so it must be designed
with its owner.

## Hardware A/B plan (simulators were not calibrated: judge on hardware only)
spec125 k=3 at 50 %, same window, 3 runs each, tokens identical:
1. today: victim B=12 (36.3 tok/s)
2. step 0 full-pass oracle (bound)
3. A (QD 4 / 8)
4. A+B
5. (if justified) A+B+C
Report tok/s, misses/stalled layers per pass, prefetch issued/used/late.
