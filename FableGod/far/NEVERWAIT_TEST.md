# NEVERWAIT_TEST: one joint test spec (Fable FAR + Fable ZERO), for Friday night

v2, Thu 5 Nov 2026 17:0x EET, by Fable FAR. v1 (16:4x) was reconciled with ZERO's eight inputs; v2 folds in ZERO's one
review (ten points, each checked against the code) and what the engine's own log says about its pools. ZERO's arm names
are withdrawn for these. For: the orchestrator; Breakthrough (map arms, section 2); BigMoE (speed ceiling, section 3;
engine form, section 6). Background: `~/FableGod/far/LOSSY350_FAR.md`, `~/FableGod/zero/LOSSY350_ZERO.md`. Exposure
counts: `~/FableGod/zero/late_terms.out` (the 2.3 definition), `late_count_pools.out`. Marks: [K] measured (source),
[P] replay or arithmetic on [K], [G] judgement. CPU scripts behind the numbers (logs only, seconds), in
`~/FableGod/far/`: `far_ckfit.py`, `far_exposed.py`, `far_pairs.py`, `far_warmsplit.py`, `far_defer*.py`.

**Built (Thu 19:0x): `~/FableGod/far/build_1105/`, start at its `INSTALL.md`.** The harness hook of 2.5 as an anchored
patch with a CPU self-test, the builder of 2.3, the masks of 2.7 (tier 1, the same pair at 266 slots, tier 2 but `nwh`), an installer, the GPU
self-check and a read-out with 2.6's lines. Two things the build pins down beyond this text: self-check (ii) has its
own masks `_c208_nw0` (lateness 0, no tail rule, so `keep` is the twin's exactly) and (i) is the arm `_c208_nwz` (the
twin's masks through the never-wait path); and the first check's anchor is row a - 1, which a split arm may cut but
which is never late. Section 3 (the speed ceiling) is BigMoE's engine test and is not in the build.

## 0. What is tested

**NEVER-WAIT (= ZERO's LATE-INJECT): a check never waits for an SSD read.** A demand-miss expert's read is issued and the
layer goes on without it; the expert runs when its bytes are in, on the SAVED layer-L input, and its term is written into
the residual streams through layer L's own inject gates, one layer late. Two questions, two tests:
- quality, on the decision map: does moving the demand misses' terms one layer late keep code under the 1.34 % line?
- speed, in the engine, before anything is built: how many ms of today's check are the wait?

**Tag: adapted, RE-OPENED. Not new.** It is KTransformers' Expert Deferral (SOSP '25) and the project's own deferral
family: TRIED 29 (B12LL, ADOPTED at 12 GB), 33 (DH_W4, quality PASS at 24 GB), 76, 183 (LATE RANK 1, KILL), 608 (DHW4,
KILL: "quality passed but no speed"), 631 (I4, KILL), 959, 970. What is open is one point never priced or run: the 45 GB
FAST preset, where the only reads left on the path are the demand misses of ranks 0-1 (1-2 % of the expert picks), with
the late group moved into a LATER layer's expert call instead of a second call inside its own layer.

## 1. Why this time

**1a. Why the speed should show up when DHW4's and LATE RANK 1's did not.**
- There, "half a layer late" meant: the late expert was still computed in its own layer's window, in a second expert
  call ("phase B") that SPUN until the read landed. The read stayed on the critical path; the harness hid 45-50 % of it,
  the v3 engine none ("two phase-B spins + a one-expert call per layer eat the overlap", TRIED 183; DHW4 34.8 -> 34.9
  tok/s, TRIED 608; B8: "late reads do NOT hide, 60 ms vs 44 ms per token", SCOREBOARD).
- Here nothing spins and no call is added: the late group joins the expert call of the first later layer at which its
  bytes are already in. If the read is slow the term is later, never the check.
- The dose is small and the SSD has room: 62 reads per check, 18.8 ms of SSD time inside an 88 ms check [K qallt_208];
  B8 moved 44 reads per TOKEN on a saturated SSD.
- The wait is measurably on today's critical path [K, four designs on BigMoE's logs]: within-run fit 0.77-0.85 of the
  gate service for QOM, 0.68-0.73 for U3C (`far_ckfit.txt`); a knee model finds no free allowance at low service on
  live checks (`far_exposed.txt`); same-token pairs at two cache sizes give 1.08-1.13 in exact mode (542 tag pairs, R2
  0.98-1.00), 0.92-0.99 between today's two exact builds, 0.83-0.90 for U3C (`far_pairs.txt`); the engine's own
  cold-vs-warm samples give 0.83 (quartiles 0.76-0.88; 65 samples, `far_warmsplit.txt`). So removing the wait is worth
  0.7-1.0 of the gate service: QOM all-in 95 -> 112-120 tok/s, U3C 100 -> 128-150 [P]. Section 3 measures it directly.

**1b. Why the quality is not I4's.** I4 (35B) moved EVERY routed expert of every row and layer a full layer late: 100 %
of the MoE mass, KLD 0.400 (0.206 with rank 0 in place). NEVER-WAIT moves the demand misses only. At the high-exposure
pool (2.2) that is 3.55 % of the MoE mass per realized row-layer on code (1.59 % if rank 0 waits): 10 late terms per row
out of 480 (6 of its own top-2, 4 kept members), and 93 % of rows carry at least one, so it is a small effect on every
row, not a large one on a few [P ZERO `late_terms.out`]. At the published pool it is 1.14 % and 3.2 terms. The 125B's own
points in between [K SCOREBOARD]: a cold rank 1 half a layer late (13.9-15.2 late terms per token) cost +0.0056 KLD on
hard text and nothing on code / math; cold ranks 2-9 late instead of dropped took hard-text KLD from 0.1117 to 0.0178.
What none of them measured on the 125B: a late RANK 0, and a FULL layer of lateness. That is what the arm is for.

## 2. (a) The map arms (Breakthrough's driver; code windows decide)

### 2.1 Names (dm_oea_masks.py's grammar; one new suffix)

| short name | builder name | what |
|---|---|---|
| twin (high exposure) | `qom_0_live_ball_c208` | today's FAST code preset on 208 uniform slots (DRAFT17 `_ball` form, as the split arms of Thu 10:36; `_bq` if Friday's driver uses that: an arm and its twin must share it) |
| **nw1** (decisive) | `qom_0_live_ball_c208_nw1` | that twin with every demand miss one layer late (2.3-2.4) |
| nw1r1 (fallback, section 4) | `qom_0_live_ball_c208_nw1r1` | rank-0 demand misses stay in place (waited); the rest one layer late |
| nw1 at the published pool | `qom_0_live_ball_nw1`, twin on record: `qom_0_live_ball` (0.51 % on code) | the low-exposure point: `laysz45_256`'s per-layer sizes, which are also the engine's own table at its C 256 default |
| the stack | `qom_0_split0.99_live_ball_nw1`, twins on record: 0.51 % and split 0.99's 0.92 % | the shared flip budget (2.6) |
| nw2 | `qom_0_live_ball_c208_nw2` | two layers late: the faster-check share (section 5) |
| nw1s | `qom_0_live_ball_c208_nw1s` | late only on realized rows whose draft cf >= 0.9; the other rows' demand misses stay in place |
| nwh | `..._c208_nwh` | half a layer late (B12LL's point): says whether an extra call would buy quality |

Never-wait arms are built WITHOUT `_t3`: a `_t3` mask sets a sure row's `keep[p, l, :2]` False (dm_oea_masks.py
183-185, the harness serves those from records3), so `keep AND late` would silently lose every late top-2. The `_nw*`
suffix must be stripped before the `_t3` strip (line 109) and the `_c<N>` match (line 111).

### 2.2 Pool convention: two pools that bracket the engine, the higher one decisive

- What the engine holds [K `~/BigMoE/data/c224t/qallt_208.txt`, `r57_screen.py` 44-55]: at C 208 the pools are uniform,
  `C_m=226 C_l=112 per layer`, plus a volatile victim tier of 64 / 32 slots; per-layer tables exist only for C 224 /
  256 / 300 / 320 (`laysz45_<C>.json`; `laysz45_256` is the table the published masks were built on). Its 6.6 misses
  per kept token count record reads of two sizes: 86 MB per check over 62 reads is 1.38 MB per read, against 2.16 MB
  for a base record and 0.77 MB for a sidecar one [P]. So the base-pool misses a map replay models are fewer than 6.6.
- What the replay gives on the 36 windows, code, QOM's rule, misses per kept token [P ZERO]: 2.16 at the published pool,
  3.58 at 266 uniform, 5.64 at 224, **6.64 at 208**, 11.7 at 147.
- Convention: **the decisive arm takes its cold set from 208 uniform slots (`_c208`)**, the replay that reproduces the
  engine's whole miss count; it is the upper end of the base-pool exposure, so a PASS there holds for every 45 GB
  setting in use. The published pool is the lower end, run beside it (one extra arm; its twin exists). BigMoE's answer
  to 8.1 says where between the two the engine sits.
- The same replay decides BOTH which tails QOM keeps and which terms are late, so `_c208` needs its own twin and the "+"
  is always read against the arm's own twin.
- The driver prints per domain: demand misses per kept token (expect 6.6-6.7 on code at `_c208`, 2.2 at the published
  pool), late terms per realized row split top-2 / members (expect 6.0 + 3.9 = ~10 on code at `_c208`; 3.2 at the
  published pool; ~5.6 would mean the members were missed), late MoE mass per realized row-layer (3.55 %; 1.14 %), and
  the share of realized rows with a late term (93 %). IFEval 3.8 terms / 1.48 %, multi 2.8 / 1.12 % at `_c208`.

### 2.3 The late set (in the builder's own loop)

At each live segment and layer l the builder loads the check's union U_l into the LRU (lines 168-173:
`for e in U: ... q[4] += e not in c; touch`).
- M_l = the experts of U_l that are NOT in the cache at the moment of their touch (exactly the events `q[4]` counts).
- late[p, l, r] = keep[p, l, r] AND ids[p, l, r] in M_l, for the check's realized rows p (A .. first miss), every rank:
  a row's ranks 0-1, the members it keeps, and an undrafted row's ranks 2-9 follow the same rule (one line after 180).
  Residents that are kept but are not in U_l are never late. Stand-in rows enter U_l as in the twin and are not masked.
- nw1r1: R0_l = the experts some row of the check (realized or stand-in) has at rank 0; late only if in M_l minus R0_l.
- nw1s: as nw1, on realized rows whose draft cf >= 0.9 only.
- The tail, resolved in the mask so the harness has no special case: at L = 47 (and L = 46 under nw2) a late rank-0
  term is put back in place (late False: the engine waits there, at most one stall per check) and any other late term
  is dropped (keep False, late False).
- Output: `keep` (the twin's, but for the tail's drops) plus `late` [rows, 48, 10] bool, rank-sorted like `keep`.

### 2.4 The late rule, exactly

For a row and layer L with late terms, with x_L the MoE input the layer already has, s_r the model's own gate scores
(no renormalisation while a term is absent, as every `rmask` arm) and inj_L the inject gates returned by layer L's
MLP-side read (`mlp_hyper_connection`):
1. In place: block_out_L = sum over kept, not-late ranks of s_r E_r(x_L) + the shared expert, written as today:
   `hidden = _hyper_residual_write(hyper, block_out_L, inj_L)`.
2. Late: delta_L = sum over late ranks of s_r E_r(x_L) (layer L's experts, layer L's input, layer L's scores).
3. nw1: delta_L is written right after layer L+1's own MoE write, through LAYER L's gates:
   `hidden = _hyper_residual_write(hidden, delta_L, inj_L)` on the stream layer L+1 returns. So layer L+1's two reads,
   its mixer, its router and its experts ran without it; layer L+2 (or, for L = 46, the model's final read) sees it.
   nw2: on the stream layer L+2 returns. nwh: inside layer L+1, before its MLP-side read (bt_b12.py's place).
4. Prompt rows and rows outside the walk: untouched.

Why layer L's gates: the write is `hyper + block_out x inject` over four streams, linear, and a read returns the stream
unchanged (`~/BigMoE/ref/qwen4_exp.py` 582-606, 1243-1290), so streams change only through writes. A term written
later through its own gates is the identical vector, late; pushed through layer L+1's gates it would land in the
streams with other weights. The project's existing late path does exactly this (bt_b12.py:
`PENDH[L + 1].append((f, inject))`, then `_hyper_residual_write(hyper, f(block_out), inj)`), so the 12 GB calibration
is for this form. (My LOSSY350_FAR v1 had it joining the next call's output; ZERO's v1 had it right.)

### 2.5 Harness patch (reference semantics; four traps ZERO found in v1's sketch are closed here)

```
per window:  CUR["lm"] = npz["late"]                  # beside CUR["rm"] = npz["keep"]
_pert_scores, ONE read position for both masks (dm_common.py 173-190):
    p0 = CUR["kpos"][li] as _row_rm reads it; _row_rm advances kpos ONCE (never call it twice: the second call
         would read the next rows' flags and trip the misalignment warning at 466-469)
    lt = CUR["lm"][p0:p0 + m, li]; rows past the mask: False
    keep and late are rank-sorted, the scores arrive in argpartition order: gather BOTH through the same rank
    permutation; return s_now = scores * (keep & ~late), s_late = scores * (keep & late)
MoE wrapper, layer L:
    y_now = combine(expert_out, s_now) + shared                       # the block's return value, as today
    STASH[L].append(combine(expert_out, s_late))                      # delta_L, this call's rows, in order
        (>= 32-row chunks: _combine(y_sorted, inv_order, s_late, shared * 0);
         narrower: (y * s_late[..., None]).sum(-2) on the same switch_mlp output)
hc post-hook: only for the layer's MLP-side read, picked by identity (self is layers[li].mlp_hyper_connection);
    both reads are registered as (li, "hc") and the hook sees no kwargs:   INJ[li] = y[2]
post-wrap on DecoderLayer.__call__ (dm_common does not wrap it today), layer M; its return value is the written
stream for every M including 47, so there is no final-read case:
    L = M - 1 (nw1) or M - 2 (nw2)
    if L in STASH: d = concat(STASH.pop(L)); assert d rows == hidden rows
                   hidden = _hyper_residual_write(hidden, d, INJ.pop(L))
    return hidden
end of each forward chunk: STASH and INJ empty (assert). STASH holds exactly the current chunk's rows.
```
An hc_pre hook on the next layer's attention-side read is NOT equivalent (there is no layer 48, and PLE layers add
`self.ple(hidden, ...)` before that read). Self-checks: (i) `late` all False: logits equal the twin's bit for bit;
(ii) lateness 0 (L = M): equal to the twin up to the fp order of two adds; (iii) the kpos warning stays silent.

### 2.6 Lines (code; all-row flips on the 12 code windows, ~3,370 realized rows, McNemar against the arm's own twin)

Let T = the twin's flips and D = arm - T.
- **PASS:** nw1 (`_c208`) <= 1.34 % AND D <= +0.40 points. Build the engine form for the code preset (section 6).
  "Clean" if D <= +0.15 (about 5 rows: inside the noise): then it stacks with either split.
- **Pool-conditional PASS:** nw1 at `_c208` is BETWEEN but at the published pool it is clean (D <= +0.15 on the 0.51 %
  twin): it ships only where the engine's base-pool misses are at most ~3 per kept token (8.1 decides).
- **BETWEEN:** nw1 in (1.34, 2.0] or D in (+0.40, +0.80]: nw1r1 decides (section 4).
- **KILL:** nw1 > 2.0 % or D > +0.80 at `_c208` AND not clean at the published pool.
- **The stack:** `split0.99 + nw1` PASS = <= 1.34 %. If it fails while nw1 passes, the two levers do not ship together
  and the orchestrator picks by tok/s (section 7's prices). On record at the published pool (TRIED 1281): `_ball` twin
  0.51 %, split 0.99 0.92 %, split 0.9 1.31 %; split 0.9 leaves 0.03 points.
- If T itself is over 1.34 % at `_c208`, report that first.
- **Read D by the draft's cf band** (>= 0.9 / < 0.9, Fable C's `audit_k8_bands` join): 93 % of rows carry a late term,
  so D is an all-row effect; the split says where the flips land and whether nw1s is worth its arm before nw1r1.
- Chat (IFEval, multi) is a control, not a gate: there are 1.6-2.2 ms per check to remove and NEVER-WAIT can stay off.
  If run: PASS <= 5.42 / 3.69 with D <= +0.40.

### 2.7 Order and GPU minutes (12 code windows, ~1.7-2 min per arm)

Tier 1 (<= 10 min, five arms): twin `_c208`, nw1, nw1r1, nw1 at the published pool, the stack at the published pool.
Tier 2, if the night has room: nw2, nw1s, the stack at `_c208` (two arms: its twin and the arm), `split0.9 + nw1`,
nwh, then chat for nw1 and its twin (24 windows, ~3.5 min each).

## 3. (b) The speed ceiling, before any engine work (BigMoE, the 00:32 ticket)

Question: by how many ms does today's QOM check shorten when its demand reads are already in? 12 HumanEval, the QOM
all-in arm, the ticket's C; U3C the same way if minutes allow.
- **Method A (preferred): oracle pre-issue.** Pass A as today with `R57_STALL=1` (the read trace). Pass B: the same
  prompts, pass A's demand reads of each check issued before that check starts and LANDED before its first layer
  (lever 2's forecast-read path with an oracle list is the nearest existing code). Tokens should equal pass A's: QOM's
  keep rule already counts the block's own loads as present. If a changed eviction order makes a prompt diverge, read
  the matching prefix.
- **Pass C (report only):** the same oracle list issued AT the check's start, so the reads land while the check
  computes. A-vs-B cannot see what reads cost when they share the memory controller with the GPU (F_RDADVISE's SSD DMA
  slowed the GPU 73 %, TRIED 176); never-wait's reads land during compute. Expected small at ~90-120 MB per check [G].
- **Method B (no new code): `R57_WARMSPLIT`.** It already runs a sampled check cold and then warm and logs both. Usable
  if the sampled cold run is the LIVE check under the FAST arm (same rows, same gating); Wednesday's fastBEhw_160
  samples re-ran something heavier (602 misses cold against 138 in the arm's live checks). Read-out: `far_warmsplit.py`.
- Lines, with S = pass A's median gate service per check (20.6 ms at C 208 by day):
  **PASS: the median check shortens by >= 0.6 x S** (at C 208: <= 76 ms from 88; the fits predict 72, the pairs 68).
  BETWEEN: 0.3-0.6 x S. **KILL: < 0.3 x S** (< 6 ms at C 208): the wait is not where the logs say; drop the bet.
  U3C at C 208: PASS <= 64 ms from 85, KILL > 74. Pass C within 2 ms of pass B, else say by how much.
- This is a ceiling: the engine form adds one write per late layer and the late groups' own compute stays.

## 4. (c) The fallback arm: rank 0 waits (nw1r1)

Rule: an expert some row of the check needs at rank 0 is waited for, as today, and is in place for every row; every
other demand miss (a rank 1, a member, an undrafted row's tail) is late by the nw1 rule.
- Exposure at `_c208` on code [P ZERO]: 22.6 rank-0 and 29.9 rank-1 misses per check; stalled layers 22.8 -> 14.1;
  late terms per realized row 9.98 -> 5.39; late mass 3.55 % -> 1.59 %; 90 % of rows still carry a late term.
- Speed: by the wait law (0.27 ms per stalled layer + 0.155 ms per miss) 7.3 of 14.4 ms stays: half of nw1's gain.
- Quality prior: the only late form measured on the 125B is of this kind (LATE RANK 1: +5.5 % KLD on hard text,
  nothing on code), and TOP1-QOM 2b (sure rows keep rank 0 + members + residents, so a cold rank 1 is not loaded)
  passed x3 at the 45 GB pool (TRIED 121, with its own caveats).
- Lines: PASS nw1r1 <= 1.34 % and D <= +0.25; KILL D > +0.60. It ships only if nw1 does not PASS.

## 5. What a PASS licenses, and what it does not

- nw1 PASS = NEVER-WAIT on the code preset at TODAY's check time. At 1.5 ms per layer 98 % of the late mass lands one
  layer late; at 1.0 ms 89 %; at the 350 pace (0.63 ms) 58 / 27 / 13 % land one / two / three+ layers late [P FAR 1c].
  A faster check needs nw2 (and nw3 if nw2 is close) before the lever is carried into a 350 design.
- It says nothing for chat (control only) or for 24 / 12 GB (twice the dose: its own arm at `_c147`, later).
- A map PASS is a screen; adoption is EvalSuite same-scores on code, as for every FAST arm.

## 6. The engine form to build on a PASS (BigMoE judges the cost)

1. The gate posts a cold demand expert's read and releases at once; the layer's expert call runs without that group.
2. Kept per late layer: the MoE input rows, the group's gate scores, layer L's MLP-side inject gates (16 x 4 values).
3. At the next layer's expert call, the groups whose bytes are in run on their saved input; their sum goes out as its
   own tensor and is written after that layer's MoE write through the saved gates (one extra write per late layer; the
   write must stay separate from the layer's own). Not in yet: the call after.
4. Layer 47: wait for a cold rank 0 (one stall at most), drop the rest. Rows cut by the live rule drop their pending.
5. The engine's own line: no spin anywhere, no second expert call inside a layer. Engine A/B: 12 HumanEval,
   A-B-A-B, PASS median check >= 0.5 x S shorter at kept within 2 %.

## 7. Predictions, registered now

| | prediction | basis |
|---|---|---|
| twin T at `_c208`, code | 0.6-0.9 % | [G]: 0.51 at the published pool; QOM drops more tails at 208 slots |
| nw1 at `_c208`, D on code | +0.1 to +0.4 points; PASS 60 % | [P/G]: LATE RANK 1's KLD per late term, x2.3 for rank 0's energy, x2-3 for a full layer, at 6 late top-2 terms + 4 members per row; ~0 on easy text in the measured runs |
| nw1 at the published pool, D | +0.03 to +0.15; clean 80 % | a third of the dose |
| nw1r1, D on code | +0.05 to +0.2; PASS 80 % | under half the mass, no rank 0 |
| the stack (`split0.99 + nw1`, published pool) | 0.95-1.15 %: PASS 65 % | 0.92 + nw1's increment there |
| the stack at `_c208` | 1.1-1.6 %: BETWEEN more likely than PASS | both increments at the higher dose |
| nw2 - nw1 | +0.1 to +0.3 | [G] |
| chat, nw1 D at `_c208` | IFEval +0.3 to +0.8, multi +0.3 to +0.65 | 3.8 / 2.8 late terms per row, on hard text |
| speed ceiling at C 208 | QOM check 88 -> 68-73 ms (95 -> 112-120 tok/s); U3C 85 -> 53-63 (100 -> 128-150) | section 1a |

## 8. Open points for the builders

1. BigMoE, pool realism: of qallt_208's 62 reads per check, how many are base records, how many sidecar, and at which
   ranks? The bytes say about half and half (1.38 MB per read). It fixes where between 2.2's two pools the engine sits
   (and whether Friday's ticket runs uniform C 208 or a per-layer table: then that table is the arm's pool). If a real
   share of the wait is for sidecar parts of experts whose base record is resident, say so: that half has a cheaper
   lossy answer than lateness (compute from the resident planes, read the rest in the background), worth its own line.
2. BigMoE: which of section 3's methods fits the ticket; whether the expert kernel can return the late groups' sum
   separately (6 item 3), or a second small call is cheaper.
3. Breakthrough: `_ball` or `_bq` on Friday; the builder changes are the `late` line after 180, the tail rule, and
   the suffix strip before line 109.
4. ZERO's review points and where they landed: kpos, DecoderLayer post-wrap, INJ by identity, STASH (2.5); `_t3`,
   suffix order (2.1); exposure numbers with members (1b, 2.2, 4, 7); cf-band split and nw1s (2.6, 2.1); pool realism
   (2.2, 8.1); pass C (3).
