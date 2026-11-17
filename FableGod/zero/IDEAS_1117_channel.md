# IDEAS_1117: the cheapest 125B-state channel on the M5 Pro, from first principles (fable-zero, Tue 17 Nov 07:4x-09:00, desk only)

The question (BREAK98, route 1 per the orchestrator's 07:3x refinement): deliver the 125B's state at every drafted
position for a small fraction of a 125B row's cost, inside a <= ~44 ms cycle at kept 24. My lens: bytes, kernels and
timing on this Mac (~280 GB/s, ~14 TMAC/s bf16 at the check's shapes), and what the check already reads. Sizes are
the real ones (dense 2.13 GB, C6 experts 4.60 GB per 32-row check, lossy expert mean ~2.3 MB). CPU calc:
price_1117_channel.py beside this file; nothing ran on the GPU. [K] measured, [P] priced from measured parts,
[G] guess (halved when it enters a tok/s number). Records read: BREAK98.md, THINK_1116c.md, Lunatic / Fable B / Fable C /
FAR IDEAS_1116c, EARLYDEATH_PREREG.md, treek_union.txt, TRIED (SD-semiAR, DA-0/DA-1, R-32 STOPPRED, L1-09, Rnd17#7,
D-07 SELF, ED-CHAIN#1-3, ANCHORCOV, S12 / S14 taps).

## 1. The byte law of a channel: there are only four shapes, and three are priced out by arithmetic

A "125B-class state at a drafted position" is a computation on that position's prefix. On a DRAM-bound machine its
price is the WEIGHT BYTES the computation reads, divided by the number of positions that share the read. Per block of
31 drafted rows [P]:

| shape | price per drafted row | per block | verdict |
|---|---:|---:|---|
| (a) SEQUENTIAL per drafted token, any model bigger than the on-chip cache: weights re-read per step. The head itself is in this class (177.6 MB bf16 per level at 0.66-0.92 ms = 190-270 GB/s [K]; QAT 95 MB measured, 26 MB target) | one 125B layer's dense alone 44 MB per step; 8 layers dense-only 355 MB per step | 1.4 GB (one layer) / 11 GB (8 layers) / 33 GB (the 2B) | **dead for any 125B slice**: the 31 sequential steps of even ONE layer's dense cost 4.9 ms, with no experts. D-07 SELF and the partial-trunk pricing (0.62 ms per layer per pass) are the same fact. Only a head-sized model can be sequential, and only a cache-resident one is cheap (f <= 0.1 ms per level, unreached). |
| (b) WAVEFRONT / progressive (rows at different layers in one pass; staircase pipelines; "draft 8, refresh, draft 8") | the dense is re-read once per STEP, not once per pass | 2.13 GB x steps | dead: the M5 Pro's one amortiser is "all rows at one layer". Blocks of 8 or 16 are flat in bytes per kept (IDEAS_1116c). |
| (c) BATCHED pass over the drafted rows AFTER drafting (the audit's shape; S12's taps done per block; a partial-depth 125B pass; a distilled refiner) | model bytes / 31 | 2B 1.06 GB = 3.8 ms; 125B layers 1-8 dense-only 0.35 GB = 1.3 ms, +lossy union 0.9 GB = 3.2 ms; layers 1-16 0.71 / 2.4 GB = 2.5 / 8.6 ms; 0.1-0.3 GB refiner 0.4-1.1 ms | **the only live shape for route 1.** It is SERIAL (the draft must exist first), so its bytes AND its ALU add to the cycle, unless the pass IS the check's own first layers (section 2). |
| (d) EXTRA ROWS inside the existing check pass | ALU only: 0.43 ms per row bf16 [P: 119 G MAC / 32 rows + 2.36 G MAC experts at 14 TMAC/s]; zero bytes if routed inside the loaded union; ALU meets the byte bound at ~58 rows | | route 2 (the orchestrator's "~128 own-information rows at the price of 32"): named, not developed here. |

Two consequences that bound every route-1 idea:
- **The bar.** Lunatic's exchange rate (1.8 ms per extra kept at the C6 + int8 bound) is 504 MB of weight reads per
  extra kept token. A channel that buys +1 kept per check may read <= 0.5 GB per block; +3 kept, <= 1.5 GB. The 2B at
  P = 1 (1.06 GB) must buy >= +2.1 kept to pay; measured at P = 2 it buys +1.45 for 2.12 GB: below the bar, as recorded.
- **The room.** At kept 24 and <= 44 ms: check at 85 % of the C6 + int8 bound 29.6 + draft 5.2 + overhead 1.5 = 36.3 ms,
  so a channel gets <= 7.7 ms = 2.2 GB per block if the check really lands at 85 % (Leap's ladder says 32-40 ms, which
  leaves 0-4 ms). Kept 24 itself is x1.48 over tree-32: route 1 alone, from measured parts, buys at most +1.5-3 kept
  (section 2), so **kept 24 is route 1 + route 2 together, not either**; I say so in section 4.

## 2. Idea 1 (best, route 1): THE CHECK AS THE TAP PASS: mid-depth 125B states at every drafted row, free, used to re-draft the dying tail inside the same check
- **Mechanism.** The check already runs layers 1-48 over all 31 drafted rows. At layer L (16, where early death reads
  its lens) it holds the 125B's layer-L state at EVERY drafted position, the mid-depth taps that S12 showed fix drift
  in the sandbox (+3.1-4.5 pt at depths 15-30 when the drafter sees layers 1-12 of the trunk at the drafted rows) and
  that THINK_1116c closed as "taps the 125B can't give" because they were priced sequentially. Here they cost zero
  bytes: the check computes them anyway. Use them once, in the same pass:
  1. at layer L the early-death probe (P-tuned ridge, pre-registered) names the first miss m^ ;
  2. the head, fed the layer-L states of rows <= m^ as taps (its TrainLab tap input, retrained to layer-L taps) plus its
     own carried state, re-drafts rows m^+1..31 (the rows the check was about to drop) during the check's layers L+1..48
     of the confident prefix: the head's ~14 levels are small kernels in the same command stream, ALU hidden under the
     experts' DRAM time, bytes 14 x 26 MB = 0.36 GB at the QAT head;
  3. the re-drafted tail runs layers 1..48 as new rows: layers 1-L re-read the dense for those rows (0.36 GB at L 8,
     0.71 at L 16) with their experts ROUTED INSIDE THE ALREADY-LOADED UNION (zero expert bytes; they are information
     rows until layer 48 verifies them exactly), then layers L+1..48 with the prefix.
  Nothing is lossy in acceptance: every token the check emits is the 125B's own (lossy rules as today). The 125B's
  layer-L TOKEN is never used (L1-09: layer-40 exit agrees with the full model on 5.7 % of confident positions; only
  the STATE is used, as a tap).
- **What it reuses from the check.** All taps (layers 1-L over the drafted rows), the loaded expert union, the KV /
  GDN caches of the prefix. The only new bytes are the dense re-read for the tail's first L layers and the head's
  levels.
- **Price [P].** L 8: 0.72 GB = 2.6 ms per check (dense 0.36 + head 0.36); L 16: 1.07 GB = 3.8 ms. At the bar it must
  buy +1.4 (L 8) / +2.1 (L 16) kept per check.
- **The number it moves (bounded from records).** Two measured parts set the ceiling: SD-semiAR (re-drafting the tail
  after the confident prefix with the head's OWN information) gave +3 % at block 32 for +3.5 ms (KILL); Lunatic's
  re-anchoring at the first fork gave 13.77 -> 14.10 because the fork guess fails ~60 %. The new element is the tap:
  the re-draft sees the 125B's processing of the whole prefix, which is exactly the "re-plan after a branch it didn't
  expect" Lunatic names as the missing information (fork chains hold 34-70 % of the gap). If taps recover S12's
  share of the drift gap on the re-drafted tail (fresh 97.9 vs own 93-96 at depths 3-14), the tail's expected kept
  rises from ~1.5 (own) to ~3-4.5 rows per re-drafted block: **+1.5..+3 kept per check [P on S12 + the fork tables],
  +9-18 %.** At L 8 that is 0.9-1.7 ms per kept (under the bar); at L 16 1.3-2.5 (around it). On its own ~+8-15 %
  tok/s at the frame [P]; it is the drift lever that fits the room, not x1.48.
- **Screen (two <= 10-min CPU tests on tonight's capture, mid_{heplus24,mbppbank}.npz, Wed morning; the LAB frame of
  lab_truestate.py / Lunatic's drift_*.py on the same cells).**
  - T1 TAP-LAB: head frozen; a ridge from the layer-L residual (2560) to the head's tap input, fit on even files,
    scored on odd; at each anchor feed the TRUE final state at depth 0 and layer-L taps at positions 1..d (true-text
    rows, i.e. the prefix case) vs own carried state only. Walk-strict bands 3-6 / 7-14 / 15-30 for L 8 / 16 / 24.
    PASS: >= 40 % of the own -> fresh gap closed at L 16 (bands 3-14 from 93-96 to >= 95.5-97); KILL < 15 %.
  - T2 FORK-TAP: Lunatic's drift_fork frame restricted to chains that passed a fork: own vs tapped-own accuracy at
    depths 2-30 after the fork. PASS: the 10-19 pt post-fork gap halves. This is the one that prices the re-draft.
  - If T1 passes, the re-draft's kept per check is replayed on the 127-node dumps with the tail re-drawn from the
    tapped head (needs the head retrained with taps: TrainLab, 1 day); the GPU build is a check with a mid-pass
    callback (Leap: the early-death hook already splits the pass at layer 16).
- **Quality risk.** None in acceptance (exact rows). The risk is cost: the mid-pass sync (the probe + ~14 head levels
  between layer L and L+1) adds serial time if the head's kernels do not hide under the experts' DRAM phase; today's
  one-call is already 4,022 dispatches, so ~60 more small ones are in the noise of the encode, but the DEPENDENCY
  (layer L+1 of the tail waits for the head) is a real stall of ~14 x f ms = 1.4-3.8 ms at f 0.1-0.27, unless the
  prefix's layers L+1.. run first and the tail's layers 1..48 after (then only the tail's dense re-read is extra).
- **Kills it.** T1 < 15 % (layer-16 states do not carry what S12's layers 1-12 of the 2B carried); or the early-death
  probe itself failing its PASS (then m^ is wrong and the re-draft starts from the wrong row); or f per level staying
  at 0.27 ms with no overlap (then 14 levels are 3.8 ms serial and the idea is SD-semiAR's price again).
- **Tag.** New as a mechanism: TRIED has tail re-drafting without taps (SD-semiAR KILL), taps without a batched source
  (S12 / S14), and layer-L lenses used only to DROP rows (STOPPRED, ED-CHAIN, Rnd17#7). The combination (the check's
  own layer-L states as free taps for a re-draft of the rows it is about to drop) is not in the records.

## 3. Idea 2: the 10 MB-per-row rule, and what it admits for route 1
- From the bar (504 MB per extra kept) and a realistic +1..+3 kept per block, any post-draft pass must read **<= ~0.5-1.5
  GB per block, i.e. <= 16-48 MB per drafted row**, and buy >= its bytes / 0.5 GB in kept. What that admits, with its
  number:
  - the 125B's layers 1-8 DENSE ONLY (attention + GDN + shared expert, no routed experts; 0.35 GB, 1.3 ms, 11.5 MB per
    row): Fable B's partial-width question; it pays at >= +0.7 kept. T1 above answers it at the state level if BigMoE
    adds the dense-only layer-8 state to tonight's capture (section 5);
  - a distilled 0.1-0.3 GB refiner over the draft (IDEAS_1116c idea 2; 0.4-1.1 ms; pays at >= +0.2-0.6 kept): the
    cheapest per row of all, training by FAR / TrainLab, not Saturday;
  - the 2B at P = 1 (3.8 ms; needs >= +2.1 kept, measured +1.45 at P = 2): only if hidden under the check on a 64-deep
    draft (IDEAS_1116c idea 1), where only its 1.06 GB count and its ALU hides;
  - layers 1-16 with experts (2.4 GB, 8.6 ms, needs +4.8 kept): out. A 4B or P = 2 anything: out.
- Rule for NewIdeas' screening: price every route-1 candidate as (bytes per block) / 504 MB = the kept it must buy;
  anything above +3 is not credible against the measured ladder (2B +8.9 %, 4B +14.9 %, true state +28 %).

## 4. The <= 44 ms cycle at kept 24 (what the arithmetic allows, honestly)
| term | ms | source |
|---|---:|---|
| lossy 32-row check, C6 + int8 at 85 % | 29.6 | Leap's modules (ladder v4 says 32-40; at 40 the room is gone) |
| idea 1 (L 8, tail re-draft) | +2.6 | section 2 |
| draft, QAT head 31 levels | 5.2 (f 0.10) / 8.8 (f 0.27) | PLAN900 ladder v4 |
| overhead | 1.5 | |
| **cycle** | **38.9-42.5** | kept needed for 550: 21.4-23.4; for 775: 30.1-32.9 |
- Kept: tree-32 16.26 + idea 1 (+1.5..+3) = 17.8-19.3 [P]. **Route 1 alone reaches ~420-500 tok/s at this cycle.** Kept
  24 needs the rows of route 2 on top (K64 at early-death prices: 20.77 own-information + idea 1's tail re-draft on a
  tree's spine ~ 22-24). 775 is not in this table at any kept a 32-row check can hold with a 29.6 ms check; it needs
  the check nearer its 25 ms bound AND kept 30+, which is K128-class rows plus the drift fixed.

## 5. For tonight's capture (BigMoE, by 18:00): what idea 1's screens need beyond layer 8 / 16 / 24 residuals
1. At the same rows and layers, the head's INPUT space: if the head reads the 10,240-wide pre-mixer stream, capture
   that stream at layers 8 / 16 / 24 too (or confirm a 2560 -> 10,240 ridge is the intended adapter; T1 works either
   way, the first is cleaner).
2. The layer-8 state with the routed experts' contribution REMOVED (dense + attention + shared expert only), one
   layer, one set: the dense-only partial-width question of section 3 at zero extra GPU time beyond one subtraction.
3. Optional, if the capture has the budget: the layer-16 state of the FIRST REJECTED row of each block in the 127-node
   tree dumps (the row the re-draft replaces), so T2 can be read at the exact rows idea 1 acts on instead of the
   true-text proxy.

## 6. One line
On the M5 Pro a 125B-class state per drafted token can only come from a batched pass over the draft (sequential and
wavefront shapes re-read the dense: 1.4 GB per block for ONE layer), priced at 504 MB of reads per extra kept; the one
free batched pass is the check itself, so the best route-1 channel is CHECK-AS-TAP-PASS: at layer 8-16 the check's own
states at every drafted row tap the head, which re-drafts the rows early death would drop, inside the same check (+0.7-1.1
GB, +2.6-3.8 ms; +1.5..+3 kept [P on S12 + fork tables]; screen = TAP-LAB / FORK-TAP on tonight's mid_*.npz, PASS >= 40 %
of the drift gap closed at layer 16). Kept 24 is this plus route 2's rows, not either alone.
